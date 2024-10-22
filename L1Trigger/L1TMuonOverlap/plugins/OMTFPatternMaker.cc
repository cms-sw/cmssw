#include <iostream>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Transition.h"

#include "L1Trigger/L1TMuonOverlap/plugins/OMTFPatternMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfigMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigWriter.h"

#include "SimDataFormats/Track/interface/SimTrack.h"

#include "Math/VectorUtil.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

OMTFPatternMaker::OMTFPatternMaker(const edm::ParameterSet& cfg)
    : theConfig(cfg),
      g4SimTrackSrc(cfg.getParameter<edm::InputTag>("g4SimTrackSrc")),
      esTokenParams_(esConsumes<edm::Transition::BeginRun>()) {
  inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));
  inputTokenSimHit = consumes<edm::SimTrackContainer>(theConfig.getParameter<edm::InputTag>("g4SimTrackSrc"));

  edm::ConsumesCollector consumesColl(consumesCollector());
  myInputMaker = new OMTFinputMaker(consumesColl);

  makeGoldenPatterns = theConfig.getParameter<bool>("makeGoldenPatterns");
  makeConnectionsMaps = theConfig.getParameter<bool>("makeConnectionsMaps");
  mergeXMLFiles = theConfig.getParameter<bool>("mergeXMLFiles");

  myOMTFConfig = nullptr;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFPatternMaker::~OMTFPatternMaker() {
  delete myOMTFConfig;
  delete myOMTFConfigMaker;
  delete myOMTF;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  const L1TMuonOverlapParams* omtfParams = &iSetup.getData(esTokenParams_);

  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  ///Initialise XML writer with default pdf.
  myWriter = new XMLConfigWriter(myOMTFConfig);

  ///For making the patterns use extended pdf width in phi, as pdf are later shifted by the mean value
  ///For low pt muons non shifted pdfs would go out of the default pdf range.
  L1TMuonOverlapParams omtfParamsMutable = *omtfParams;
  std::vector<int> generalParams = *omtfParamsMutable.generalParams();
  nPdfAddrBits = omtfParams->nPdfAddrBits();

  if (!mergeXMLFiles)
    generalParams[L1TMuonOverlapParams::GENERAL_ADDRBITS] = 2 * nPdfAddrBits;
  omtfParamsMutable.setGeneralParams(generalParams);

  myOMTFConfig->configure(&omtfParamsMutable);
  myOMTF->configure(myOMTFConfig, omtfParams);
  myOMTFConfigMaker = new OMTFConfigMaker(myOMTFConfig);

  ///Clear existing GoldenPatterns
  if (!mergeXMLFiles) {
    const std::map<Key, GoldenPattern*>& theGPs = myOMTF->getPatterns();
    for (auto itGP : theGPs)
      itGP.second->reset();
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::endRun(edm::Run const&, edm::EventSetup const&) {}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::beginJob() {
  myOMTFConfig = new OMTFConfiguration();
  myOMTF = new OMTFProcessor();
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::endJob() {
  if (makeGoldenPatterns && !makeConnectionsMaps) {
    myWriter->initialiseXMLDocument("OMTF");
    const std::map<Key, GoldenPattern*>& myGPmap = myOMTF->getPatterns();
    for (auto itGP : myGPmap) {
      if (!itGP.second->hasCounts())
        continue;
      itGP.second->normalise(nPdfAddrBits);
    }

    GoldenPattern dummyGP(Key(0, 0, 0), myOMTFConfig);
    dummyGP.reset();

    ///Put back default value of the pdf width.
    L1TMuonOverlapParams omtfParamsMutable = *myOMTFConfig->getRawParams();
    std::vector<int> generalParams = *omtfParamsMutable.generalParams();
    generalParams[L1TMuonOverlapParams::GENERAL_ADDRBITS] = nPdfAddrBits;
    omtfParamsMutable.setGeneralParams(generalParams);
    myOMTFConfig->configure(&omtfParamsMutable);

    for (auto itGP : myGPmap) {
      ////
      unsigned int iPt = theConfig.getParameter<int>("ptCode") + 1;
      if (iPt > 31)
        iPt = 200 * 2 + 1;
      else
        iPt = RPCConst::ptFromIpt(iPt) * 2.0 +
              1;  //MicroGMT has 0.5 GeV step size, with lower bin edge  (uGMT_pt_code - 1)*step_size
      ////
      if (itGP.first.thePtCode == iPt && itGP.first.theCharge == theConfig.getParameter<int>("charge")) {
        std::cout << *itGP.second << std::endl;
        myWriter->writeGPData(*itGP.second, dummyGP, dummyGP, dummyGP);
      }
    }
    std::string fName = "GPs.xml";
    myWriter->finaliseXMLDocument(fName);
  }

  if (makeConnectionsMaps && !makeGoldenPatterns) {
    myWriter->initialiseXMLDocument("OMTF");
    std::string fName = "Connections.xml";
    unsigned int iProcessor = 0;
    ///Order important: printPhiMap updates global vector in OMTFConfiguration
    myOMTFConfigMaker->printPhiMap(std::cout);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 0);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 1);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 2);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 3);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 4);
    myOMTFConfigMaker->printConnections(std::cout, iProcessor, 5);
    myWriter->writeConnectionsData(myOMTFConfig->getMeasurements4D());
    myWriter->finaliseXMLDocument(fName);
  }

  if (mergeXMLFiles) {
    GoldenPattern* dummy = new GoldenPattern(Key(0, 0, 0), myOMTFConfig);
    dummy->reset();

    std::string fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
    const std::map<Key, GoldenPattern*>& myGPmap = myOMTF->getPatterns();
    for (auto itGP : myGPmap) {
      myWriter->writeGPData(*itGP.second, *dummy, *dummy, *dummy);
    }
    fName = "GPs.xml";
    myWriter->finaliseXMLDocument(fName);
    ///Write GPs merged by 4 above iPt=71, and by 2 below//
    //////////////////////////////////////////////////////
    ///4x merging
    fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
    myOMTF->averagePatterns(-1);
    myOMTF->averagePatterns(1);
    writeMergedGPs();
    fName = "GPs_4x.xml";
    myWriter->finaliseXMLDocument(fName);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::writeMergedGPs() {
  const std::map<Key, GoldenPattern*>& myGPmap = myOMTF->getPatterns();

  GoldenPattern* dummy = new GoldenPattern(Key(0, 0, 0), myOMTFConfig);
  dummy->reset();

  unsigned int iPtMin = 9;
  Key aKey = Key(0, iPtMin, 1);
  while (myGPmap.find(aKey) != myGPmap.end()) {
    GoldenPattern* aGP1 = myGPmap.find(aKey)->second;
    GoldenPattern* aGP2 = dummy;
    GoldenPattern* aGP3 = dummy;
    GoldenPattern* aGP4 = dummy;

    ++aKey.thePtCode;
    while (myGPmap.find(aKey) == myGPmap.end() && aKey.thePtCode <= 401)
      ++aKey.thePtCode;
    if (aKey.thePtCode <= 401 && myGPmap.find(aKey) != myGPmap.end())
      aGP2 = myGPmap.find(aKey)->second;

    if (aKey.thePtCode > 71) {
      ++aKey.thePtCode;
      while (myGPmap.find(aKey) == myGPmap.end() && aKey.thePtCode <= 401)
        ++aKey.thePtCode;
      if (aKey.thePtCode <= 401 && myGPmap.find(aKey) != myGPmap.end())
        aGP3 = myGPmap.find(aKey)->second;

      ++aKey.thePtCode;
      while (myGPmap.find(aKey) == myGPmap.end() && aKey.thePtCode <= 401)
        ++aKey.thePtCode;
      if (aKey.thePtCode <= 401 && myGPmap.find(aKey) != myGPmap.end())
        aGP4 = myGPmap.find(aKey)->second;
    }
    ++aKey.thePtCode;
    while (myGPmap.find(aKey) == myGPmap.end() && aKey.thePtCode <= 401)
      ++aKey.thePtCode;
    myWriter->writeGPData(*aGP1, *aGP2, *aGP3, *aGP4);

    ///Write the opposite charge.
    Key aTmpKey = aGP1->key();
    aTmpKey.theCharge = -1;
    if (myGPmap.find(aTmpKey) != myGPmap.end())
      aGP1 = myGPmap.find(aTmpKey)->second;
    else
      aGP1 = dummy;

    aTmpKey = aGP2->key();
    aTmpKey.theCharge = -1;
    if (myGPmap.find(aTmpKey) != myGPmap.end())
      aGP2 = myGPmap.find(aTmpKey)->second;
    else
      aGP2 = dummy;

    aTmpKey = aGP3->key();
    aTmpKey.theCharge = -1;
    if (myGPmap.find(aTmpKey) != myGPmap.end())
      aGP3 = myGPmap.find(aTmpKey)->second;
    else
      aGP3 = dummy;

    aTmpKey = aGP4->key();
    aTmpKey.theCharge = -1;
    if (myGPmap.find(aTmpKey) != myGPmap.end())
      aGP4 = myGPmap.find(aTmpKey)->second;
    else
      aGP4 = dummy;

    myWriter->writeGPData(*aGP1, *aGP2, *aGP3, *aGP4);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFPatternMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  if (mergeXMLFiles)
    return;

  ///Get the simulated muon parameters
  const SimTrack* aSimMuon = findSimMuon(iEvent);
  if (!aSimMuon) {
    edm::LogError("OMTFPatternMaker") << "No SimMuon found in the event!";
    return;
  }

  myInputMaker->initialize(evSetup, myOMTFConfig);

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;

  ///Filter digis by dropping digis from selected (by cfg.py) subsystems
  if (!theConfig.getParameter<bool>("dropDTPrimitives")) {
    iEvent.getByToken(inputTokenDTPh, dtPhDigis);
    iEvent.getByToken(inputTokenDTTh, dtThDigis);
  }
  if (!theConfig.getParameter<bool>("dropRPCPrimitives"))
    iEvent.getByToken(inputTokenRPC, rpcDigis);
  if (!theConfig.getParameter<bool>("dropCSCPrimitives"))
    iEvent.getByToken(inputTokenCSC, cscDigis);

  //l1t::tftype mtfType = l1t::tftype::bmtf;
  l1t::tftype mtfType = l1t::tftype::omtf_pos;
  //l1t::tftype mtfType = l1t::tftype::emtf_pos;

  ///Loop over all processors, each covering 60 deg in phi
  for (unsigned int iProcessor = 0; iProcessor < 6; ++iProcessor) {
    ///Input data with phi ranges shifted for each processor, so it fits 11 bits range
    OMTFinput myInput = myInputMaker->buildInputForProcessor(
        dtPhDigis.product(), dtThDigis.product(), cscDigis.product(), rpcDigis.product(), iProcessor, mtfType);

    ///Connections maps are made by hand. makeConnetionsMap method
    ///provides tables for checking their consistency.
    if (makeConnectionsMaps)
      myOMTFConfigMaker->makeConnetionsMap(iProcessor, myInput);

    if (makeGoldenPatterns)
      myOMTF->fillCounts(iProcessor, myInput, aSimMuon);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
const SimTrack* OMTFPatternMaker::findSimMuon(const edm::Event& ev, const SimTrack* previous) {
  const SimTrack* result = nullptr;
  edm::Handle<edm::SimTrackContainer> simTks;
  ev.getByToken(inputTokenSimHit, simTks);

  for (std::vector<SimTrack>::const_iterator it = simTks->begin(); it < simTks->end(); it++) {
    const SimTrack& aTrack = *it;
    if (!(aTrack.type() == 13 || aTrack.type() == -13))
      continue;
    if (previous && ROOT::Math::VectorUtil::DeltaR(aTrack.momentum(), previous->momentum()) < 0.07)
      continue;
    if (!result || aTrack.momentum().pt() > result->momentum().pt())
      result = &aTrack;
  }
  return result;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OMTFPatternMaker);
