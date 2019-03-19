
#include "L1Trigger/L1TMuonBayes/plugins/L1TMuonBayesMuCorrelatorTrackProducer.h"

#include <iostream>
#include <strstream>
#include <vector>
#include <fstream>
#include <memory>

#include <TFile.h>
#include <TStyle.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/PdfModuleWithStats.h"
#include "L1Trigger/L1TMuonBayes/interface/MuTimingModuleWithStat.h"


L1TMuonBayesMuCorrelatorTrackProducer::L1TMuonBayesMuCorrelatorTrackProducer(const edm::ParameterSet& cfg)
  :edmParameterSet(cfg), muCorrelatorConfig(std::make_shared<MuCorrelatorConfig>()) {

  produces<l1t::RegionalMuonCandBxCollection >("MuCorr");
  produces<l1t::BayesMuCorrTrackBxCollection >("BayesMuCorrTracks");

  muStubsInputTokens.inputTokenDTPh = consumes<L1MuDTChambPhContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTPh"));
  muStubsInputTokens.inputTokenDTTh = consumes<L1MuDTChambThContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTTh"));
  muStubsInputTokens.inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcCSC"));
  muStubsInputTokens.inputTokenRPC = consumes<RPCDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcRPC"));

  edm::InputTag l1TrackInputTag = cfg.getParameter<edm::InputTag>("L1TrackInputTag");
  ttTrackToken = consumes< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >(l1TrackInputTag);

  inputTokenSimTracks = consumes<edm::SimTrackContainer>(edmParameterSet.getParameter<edm::InputTag>("g4SimTrackSrc")); //TODO remove


  //Range of the BXes for which the emulation is performed,
  if(edmParameterSet.exists("bxRangeMin") ){
    bxRangeMin = edmParameterSet.getParameter<int>("bxRangeMin");
  }
  if(edmParameterSet.exists("bxRangeMax") ){
    bxRangeMax = edmParameterSet.getParameter<int>("bxRangeMax");
  }

  if(edmParameterSet.exists("useStubsFromAdditionalBxs") ){
    useStubsFromAdditionalBxs = edmParameterSet.getParameter<int>("useStubsFromAdditionalBxs");
  }

  for(unsigned int ptBin = 0; ptBin < muCorrelatorConfig->getPtHwBins().size(); ++ptBin) {
    edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<"ptBin "<<setw(2)<<ptBin<<" range Hw: "<<muCorrelatorConfig->ptBinString(ptBin, 0)<<" = "<<muCorrelatorConfig->ptBinString(ptBin, 1)<<std::endl;
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
L1TMuonBayesMuCorrelatorTrackProducer::~L1TMuonBayesMuCorrelatorTrackProducer(){
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesMuCorrelatorTrackProducer::beginJob(){

  //m_Reconstruction.beginJob();

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesMuCorrelatorTrackProducer::endJob(){

  //m_Reconstruction.endJob();

  IPdfModule* pdfModule = muCorrelatorProcessor->getPdfModule();
  PdfModuleWithStats* pdfModuleWithStats = dynamic_cast<PdfModuleWithStats*>(pdfModule);
  if(pdfModuleWithStats) {
    // using TFileService insteed
    /*gStyle->SetOptStat(111111);
    TFile outfile("muCorrPdfs.root", "RECREATE");
    cout<<__FUNCTION__<<": "<<__LINE__<<" creating file "<<outfile.GetName()<<endl;

    outfile.cd();
    //pdfModuleWithStats->write();*/

    if(edmParameterSet.exists("generatePdfs") && edmParameterSet.getParameter<bool>("generatePdfs")) {
      if(edmParameterSet.exists("pdfModuleFileName") ) {//if we read the patterns directly from the xml, we do it only once, at the beginning of the first run, not every run
        pdfModuleFileName = edmParameterSet.getParameter<edm::FileInPath>("pdfModuleFileName").fullPath();
      }
      pdfModuleWithStats->generateCoefficients();
      writePdfs(pdfModule, pdfModuleFileName);
    }
  }


  MuTimingModuleWithStat* muTimingModule = dynamic_cast<MuTimingModuleWithStat*>(muCorrelatorProcessor->getMuTimingModule() );
  if(muTimingModule) {
    if(edmParameterSet.exists("generateTiming") && edmParameterSet.getParameter<bool>("generateTiming")) {
      string muTimingModuleFileName = "muTimingModule.xml";
      if(edmParameterSet.exists("outputTimingFile") ) {//if we read the patterns directly from the xml, we do it only once, at the beginning of the first run, not every run
        muTimingModuleFileName = edmParameterSet.getParameter<std::string>("outputTimingFile");
      }
      muTimingModule->generateCoefficients();
      writeTimingModule(muTimingModule, muTimingModuleFileName);
    }
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesMuCorrelatorTrackProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  /*for(int ptHw = 0; ptHw < 512; ptHw++) {
    cout<<"ptHw "<<setw(3)<<ptHw<<" = "<<setw(5)<<muCorrelatorConfig->hwPtToGev(ptHw)<<" GeV ptBin "<<muCorrelatorConfig->ptHwToPtBin(ptHw)<<endl;
  }*/

  inputMaker = std::make_unique<MuCorrelatorInputMaker>(edmParameterSet, iSetup, muCorrelatorConfig, muStubsInputTokens);
  ttTracksInputMaker = std::make_unique<TTTracksInputMaker>(edmParameterSet);;

  if(!muCorrelatorProcessor) {
    std::string pdfModuleType = "PdfModule"; //GoldenPatternParametrised GoldenPatternWithStat GoldenPattern
    if(edmParameterSet.exists("pdfModuleType") ) {
      pdfModuleType = edmParameterSet.getParameter<std::string>("pdfModuleType");
    }

    PdfModule* pdfModule = nullptr;

    if(pdfModuleType == "PdfModule") {
      pdfModule = new PdfModule(muCorrelatorConfig);
      edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<" creating PdfModule for muCorrelatorProcessor"<<std::endl;
    }
    else if(pdfModuleType == "PdfModuleWithStats") {
      pdfModule = new PdfModuleWithStats(muCorrelatorConfig);
      edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<" creating PdfModuleWithStats for muCorrelatorProcessor"<<std::endl;
    }
    else {
      throw cms::Exception("L1TMuonBayesMuCorrelatorTrackProducer::beginRun: unknown pdfModuleType: " + pdfModuleType);
    }


    if(edmParameterSet.exists("generatePdfs") && edmParameterSet.getParameter<bool>("generatePdfs")) {
      //dont read the pdf if they are going to be generated
    }
    else if(edmParameterSet.exists("pdfModuleFileName") ) {//if we read the patterns directly from the xml, we do it only once, at the beginning of the first run, not every run
      pdfModuleFileName = edmParameterSet.getParameter<edm::FileInPath>("pdfModuleFileName").fullPath();
      edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<" reading the pdfModule from file "<<pdfModuleFileName<<std::endl;
      readPdfs(pdfModule, pdfModuleFileName);
    }

    std::unique_ptr<PdfModule> pdfModuleUniqPtr(pdfModule);

    muCorrelatorProcessor = std::make_unique<MuCorrelatorProcessor>(muCorrelatorConfig, std::move(pdfModuleUniqPtr));


    if(edmParameterSet.exists("generateTiming") && edmParameterSet.getParameter<bool>("generateTiming")) {
      std::unique_ptr<MuTimingModule> muTimingModuleUniqPtr = std::make_unique<MuTimingModuleWithStat>(muCorrelatorConfig.get());
      muCorrelatorProcessor->setMuTimingModule(muTimingModuleUniqPtr);
    }
    else if(edmParameterSet.exists("timingModuleFile") ) {
      string timingModuleFile = edmParameterSet.getParameter<edm::FileInPath>("timingModuleFile").fullPath();
      edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<" reading the MuTimingModule from file "<<timingModuleFile<<std::endl;

      std::unique_ptr<MuTimingModule> muTimingModuleUniqPtr = std::make_unique<MuTimingModule>(muCorrelatorConfig.get());
      readTimingModule(muTimingModuleUniqPtr.get(), timingModuleFile);
      muCorrelatorProcessor->setMuTimingModule(muTimingModuleUniqPtr);
    }

    edm::LogImportant("L1TMuonBayesMuCorrelatorTrackProducer")<<" muCorrelatorProcessor constructed"<<std::endl;

  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesMuCorrelatorTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup){
  inputMaker->loadAndFilterDigis(iEvent);

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> candidates(new l1t::RegionalMuonCandBxCollection);
  candidates->setBXRange(bxRangeMin, bxRangeMax);

  std::unique_ptr<l1t::BayesMuCorrTrackBxCollection> bayesMuCorrTracks(new l1t::BayesMuCorrTrackBxCollection);

  //std::cout<<"\n"<<__FUNCTION__<<":"<<__LINE__<<" iEvent "<<iEvent.id().event()<<" #####################################################################"<<endl;
  for(int bx = bxRangeMin; bx <= bxRangeMax; bx++) {

    auto muonStubsInput = inputMaker->MuCorrelatorInputMaker::buildInputForProcessor(0, l1t::tftype::bmtf, bx, bx + useStubsFromAdditionalBxs);
    //std::cout<<muonStubsInput<<std::endl;

    auto ttTRacks = ttTracksInputMaker->loadTTTracks(iEvent, edmParameterSet, muCorrelatorConfig.get());

    LogTrace("omtfEventPrintout")<<"\n\nEvent "<<iEvent.id().event()<<" muonStubsInput bx "<<bx<<": \n "<<muonStubsInput<<endl;
    for(auto& ttTRack : ttTRacks) {
      LogTrace("omtfEventPrintout")<<*ttTRack<<endl;
    }
    LogTrace("omtfEventPrintout")<<"\n";

    //for(unsigned int iProcessor=0; iProcessor<m_OMTFConfig->nProcessors(); ++iProcessor)
    {
      AlgoTTMuons algoTTMuons = muCorrelatorProcessor->processTracks(muonStubsInput, ttTRacks);
      std::vector<l1t::RegionalMuonCand> procCandidates = muCorrelatorProcessor->getFinalCandidates(0, l1t::bmtf, algoTTMuons); //processor type is just ignored

      //fill outgoing collection
      for (auto & candMuon :  procCandidates) {
        candidates->push_back(bx, candMuon);
      }

      l1t::BayesMuCorrTrackCollection bayesMuCorrTracksInBx = muCorrelatorProcessor->getMuCorrTrackCollection(0, algoTTMuons);
      for (auto & muTrack :  bayesMuCorrTracksInBx) {
        bayesMuCorrTracks->push_back(bx, muTrack);
      }

    }

    //edm::LogInfo("L1TMuonBayesMuCorrelatorTrackProducer") <<"MuCorr:  Number of candidates in BX="<<bx<<": "<<candidates->size(bx) << std::endl;;
  }

  iEvent.put(std::move(candidates), "MuCorr");
  iEvent.put(std::move(bayesMuCorrTracks), "BayesMuCorrTracks");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

void L1TMuonBayesMuCorrelatorTrackProducer::readPdfs(IPdfModule* pdfModule, std::string fileName) {
  // open the archive
  std::ifstream ifs(fileName);
  assert(ifs.good());
  boost::archive::xml_iarchive ia(ifs);

  // write class instance to archive
  PdfModule* pdfModuleImpl = dynamic_cast<PdfModule*>(pdfModule);
  // write class instance to archive
  if(pdfModuleImpl) {
    PdfModule& pdfModuleRef = *pdfModuleImpl;
    ia >> BOOST_SERIALIZATION_NVP(pdfModuleRef);

    LogTrace("omtfEventPrintout")<<__FUNCTION__<<": "<<__LINE__<<" pdfModule->getCoefficients().size() "<<pdfModuleRef.getCoefficients().size()<<endl;
  }
  //else TODO
}

void L1TMuonBayesMuCorrelatorTrackProducer::writePdfs(const IPdfModule* pdfModule, std::string fileName) {
  std::ofstream ofs(fileName);

  boost::archive::xml_oarchive xmlOutArch(ofs);
  //boost::archive::text_oarchive txtOutArch(ofs);

  const PdfModule* pdfModuleImpl = dynamic_cast<const PdfModule*>(pdfModule);
  // write class instance to archive
  if(pdfModuleImpl) {
    cout<<__FUNCTION__<<": "<<__LINE__<<" writing pdf to file "<<fileName<<endl;
    const PdfModule& pdfModuleRef = *pdfModuleImpl;
    xmlOutArch << BOOST_SERIALIZATION_NVP(pdfModuleRef);
    //txtOutArch << (*pdfModuleImpl);
  }
  // archive and stream closed when destructors are called
}


void L1TMuonBayesMuCorrelatorTrackProducer::readTimingModule(MuTimingModule* muTimingModule, std::string fileName) {
  // open the archive
  std::ifstream ifs(fileName);
  assert(ifs.good());
  boost::archive::xml_iarchive ia(ifs);

  // write class instance to archive
  //PdfModule* pdfModuleImpl = dynamic_cast<PdfModule*>(pdfModule);
  // write class instance to archive
  if(muTimingModule) {
    MuTimingModule& muTimingModuleRef = *muTimingModule;
    ia >> BOOST_SERIALIZATION_NVP(muTimingModuleRef);

    LogTrace("omtfEventPrintout")<<__FUNCTION__<<": "<<__LINE__<<" readTimingModule from file "<<fileName<<endl;
  }
  //else TODO
}

void L1TMuonBayesMuCorrelatorTrackProducer::writeTimingModule(const MuTimingModule* muTimingModule, std::string fileName) {
  std::ofstream ofs(fileName);

  boost::archive::xml_oarchive xmlOutArch(ofs);
  //boost::archive::text_oarchive txtOutArch(ofs);

  //const PdfModule* pdfModuleImpl = dynamic_cast<const PdfModule*>(pdfModule);
  // write class instance to archive
  if(muTimingModule) {
    cout<<__FUNCTION__<<": "<<__LINE__<<" writing MuTimingModule to file "<<fileName<<endl;
    const MuTimingModule& muTimingModuleRef = *muTimingModule;
    xmlOutArch << BOOST_SERIALIZATION_NVP(muTimingModuleRef);
    //txtOutArch << (*pdfModuleImpl);
  }
  // archive and stream closed when destructors are called
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonBayesMuCorrelatorTrackProducer);
