/* 
 *  \class EcalLaserAnalyzer
 *
 *  primary author: Julie Malcles - CEA/Saclay
 *  author: Gautier Hamel De Monchenault - CEA/Saclay
 */

#include "TAxis.h"
#include "TH1.h"
#include "TProfile.h"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TMath.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalLaserAnalyzer.h"

#include <sstream>
#include <fstream>
#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TShapeAnalysis.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TAPD.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TAPDPulse.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPN.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPNPulse.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPNCor.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TMem.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/PulseFitWithFunction.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"

//========================================================================
EcalLaserAnalyzer::EcalLaserAnalyzer(const edm::ParameterSet& iConfig)
    //========================================================================
    : iEvent(0),
      eventHeaderCollection_(iConfig.getParameter<std::string>("eventHeaderCollection")),
      eventHeaderProducer_(iConfig.getParameter<std::string>("eventHeaderProducer")),
      digiCollection_(iConfig.getParameter<std::string>("digiCollection")),
      digiProducer_(iConfig.getParameter<std::string>("digiProducer")),
      digiPNCollection_(iConfig.getParameter<std::string>("digiPNCollection")),
      rawDataToken_(consumes<EcalRawDataCollection>(edm::InputTag(eventHeaderProducer_, eventHeaderCollection_))),
      pnDiodeDigiToken_(consumes<EcalPnDiodeDigiCollection>(edm::InputTag(digiProducer_))),
      mappingToken_(esConsumes()),
      // Framework parameters with default values
      _nsamples(iConfig.getUntrackedParameter<unsigned int>("nSamples", 10)),
      _presample(iConfig.getUntrackedParameter<unsigned int>("nPresamples", 2)),
      _firstsample(iConfig.getUntrackedParameter<unsigned int>("firstSample", 1)),
      _lastsample(iConfig.getUntrackedParameter<unsigned int>("lastSample", 2)),
      _nsamplesPN(iConfig.getUntrackedParameter<unsigned int>("nSamplesPN", 50)),
      _presamplePN(iConfig.getUntrackedParameter<unsigned int>("nPresamplesPN", 6)),
      _firstsamplePN(iConfig.getUntrackedParameter<unsigned int>("firstSamplePN", 7)),
      _lastsamplePN(iConfig.getUntrackedParameter<unsigned int>("lastSamplePN", 8)),
      _timingcutlow(iConfig.getUntrackedParameter<unsigned int>("timingCutLow", 2)),
      _timingcuthigh(iConfig.getUntrackedParameter<unsigned int>("timingCutHigh", 9)),
      _timingquallow(iConfig.getUntrackedParameter<unsigned int>("timingQualLow", 3)),
      _timingqualhigh(iConfig.getUntrackedParameter<unsigned int>("timingQualHigh", 8)),
      _ratiomincutlow(iConfig.getUntrackedParameter<double>("ratioMinCutLow", 0.4)),
      _ratiomincuthigh(iConfig.getUntrackedParameter<double>("ratioMinCutHigh", 0.95)),
      _ratiomaxcutlow(iConfig.getUntrackedParameter<double>("ratioMaxCutLow", 0.8)),
      _presamplecut(iConfig.getUntrackedParameter<double>("presampleCut", 5.0)),
      _niter(iConfig.getUntrackedParameter<unsigned int>("nIter", 3)),
      _fitab(iConfig.getUntrackedParameter<bool>("fitAB", false)),
      _alpha(iConfig.getUntrackedParameter<double>("alpha", 1.5076494)),
      _beta(iConfig.getUntrackedParameter<double>("beta", 1.5136036)),
      _nevtmax(iConfig.getUntrackedParameter<unsigned int>("nEvtMax", 200)),
      _noise(iConfig.getUntrackedParameter<double>("noise", 2.0)),
      _chi2cut(iConfig.getUntrackedParameter<double>("chi2cut", 10.0)),
      _ecalPart(iConfig.getUntrackedParameter<std::string>("ecalPart", "EB")),
      _docorpn(iConfig.getUntrackedParameter<bool>("doCorPN", false)),
      _fedid(iConfig.getUntrackedParameter<int>("fedID", -999)),
      _saveallevents(iConfig.getUntrackedParameter<bool>("saveAllEvents", false)),
      _qualpercent(iConfig.getUntrackedParameter<double>("qualPercent", 0.2)),
      _debug(iConfig.getUntrackedParameter<int>("debug", 0)),
      resdir_(iConfig.getUntrackedParameter<std::string>("resDir")),
      pncorfile_(iConfig.getUntrackedParameter<std::string>("pnCorFile")),
      nCrys(NCRYSEB),
      nPNPerMod(NPNPERMOD),
      nMod(NMODEE),
      nSides(NSIDES),
      runType(-1),
      runNum(0),
      fedID(-1),
      dccID(-1),
      side(2),
      lightside(2),
      iZ(1),
      phi(-1),
      eta(-1),
      event(0),
      color(-1),
      pn0(0),
      pn1(0),
      apdAmpl(0),
      apdAmplA(0),
      apdAmplB(0),
      apdTime(0),
      pnAmpl(0),
      pnID(-1),
      moduleID(-1),
      channelIteratorEE(0)

//========================================================================

{
  if (_ecalPart == "EB") {
    ebDigiToken_ = consumes<EBDigiCollection>(edm::InputTag(digiProducer_, digiCollection_));
  } else if (_ecalPart == "EE") {
    eeDigiToken_ = consumes<EEDigiCollection>(edm::InputTag(digiProducer_, digiCollection_));
  }

  // Geometrical constants initialization

  if (_ecalPart == "EB") {
    nCrys = NCRYSEB;
  } else {
    nCrys = NCRYSEE;
  }
  iZ = 1;
  if (_fedid <= 609)
    iZ = -1;
  modules = ME::lmmodFromDcc(_fedid);
  nMod = modules.size();
  nRefChan = NREFCHAN;

  for (unsigned int j = 0; j < nCrys; j++) {
    iEta[j] = -1;
    iPhi[j] = -1;
    iModule[j] = 10;
    iTowerID[j] = -1;
    iChannelID[j] = -1;
    idccID[j] = -1;
    iside[j] = -1;
    wasTimingOK[j] = true;
    wasGainOK[j] = true;
  }

  for (unsigned int j = 0; j < nMod; j++) {
    int ii = modules[j];
    firstChanMod[ii - 1] = 0;
    isFirstChanModFilled[ii - 1] = 0;
  }

  // Quality check flags

  isGainOK = true;
  isTimingOK = true;

  // PN linearity corrector

  pnCorrector = new TPNCor(pncorfile_);

  // Objects dealing with pulses

  APDPulse = new TAPDPulse(_nsamples,
                           _presample,
                           _firstsample,
                           _lastsample,
                           _timingcutlow,
                           _timingcuthigh,
                           _timingquallow,
                           _timingqualhigh,
                           _ratiomincutlow,
                           _ratiomincuthigh,
                           _ratiomaxcutlow);
  PNPulse = new TPNPulse(_nsamplesPN, _presamplePN);

  // Object dealing with MEM numbering

  Mem = new TMem(_fedid);

  // Objects needed for npresample calculation

  Delta01 = new TMom();
  Delta12 = new TMom();
}

//========================================================================
EcalLaserAnalyzer::~EcalLaserAnalyzer() {
  //========================================================================

  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//========================================================================
void EcalLaserAnalyzer::beginJob() {
  //========================================================================

  // Create temporary files and trees to save adc samples
  //======================================================

  ADCfile = resdir_;
  ADCfile += "/APDSamplesLaser.root";

  APDfile = resdir_;
  APDfile += "/APDPNLaserAllEvents.root";

  ADCFile = new TFile(ADCfile.c_str(), "RECREATE");

  for (unsigned int i = 0; i < nCrys; i++) {
    std::stringstream name;
    name << "ADCTree" << i + 1;
    ADCtrees[i] = new TTree(name.str().c_str(), name.str().c_str());

    ADCtrees[i]->Branch("ieta", &eta, "eta/I");
    ADCtrees[i]->Branch("iphi", &phi, "phi/I");
    ADCtrees[i]->Branch("side", &side, "side/I");
    ADCtrees[i]->Branch("dccID", &dccID, "dccID/I");
    ADCtrees[i]->Branch("towerID", &towerID, "towerID/I");
    ADCtrees[i]->Branch("channelID", &channelID, "channelID/I");
    ADCtrees[i]->Branch("event", &event, "event/I");
    ADCtrees[i]->Branch("color", &color, "color/I");
    ADCtrees[i]->Branch("adc", &adc, "adc[10]/D");
    ADCtrees[i]->Branch("pn0", &pn0, "pn0/D");
    ADCtrees[i]->Branch("pn1", &pn1, "pn1/D");

    ADCtrees[i]->SetBranchAddress("ieta", &eta);
    ADCtrees[i]->SetBranchAddress("iphi", &phi);
    ADCtrees[i]->SetBranchAddress("side", &side);
    ADCtrees[i]->SetBranchAddress("dccID", &dccID);
    ADCtrees[i]->SetBranchAddress("towerID", &towerID);
    ADCtrees[i]->SetBranchAddress("channelID", &channelID);
    ADCtrees[i]->SetBranchAddress("event", &event);
    ADCtrees[i]->SetBranchAddress("color", &color);
    ADCtrees[i]->SetBranchAddress("adc", adc);
    ADCtrees[i]->SetBranchAddress("pn0", &pn0);
    ADCtrees[i]->SetBranchAddress("pn1", &pn1);

    nevtAB[i] = 0;
  }

  // Define output results filenames and shape analyzer object (alpha,beta)
  //=====================================================================

  // 1) AlphaBeta files

  doesABTreeExist = true;

  std::stringstream nameabinitfile;
  nameabinitfile << resdir_ << "/ABInit.root";
  alphainitfile = nameabinitfile.str();

  std::stringstream nameabfile;
  nameabfile << resdir_ << "/AB.root";
  alphafile = nameabfile.str();

  FILE* test;
  if (_fitab)
    test = fopen(alphainitfile.c_str(), "r");
  else
    test = fopen(alphafile.c_str(), "r");
  if (test == nullptr) {
    doesABTreeExist = false;
    _fitab = true;
  };
  delete test;

  TFile* fAB = nullptr;
  TTree* ABInit = nullptr;
  if (doesABTreeExist) {
    fAB = new TFile(nameabinitfile.str().c_str());
  }
  if (doesABTreeExist && fAB) {
    ABInit = (TTree*)fAB->Get("ABCol0");
  }

  // 2) Shape analyzer

  if (doesABTreeExist && fAB && ABInit && ABInit->GetEntries() != 0) {
    shapana = new TShapeAnalysis(ABInit, _alpha, _beta, 5.5, 1.0);
    doesABTreeExist = true;
  } else {
    shapana = new TShapeAnalysis(_alpha, _beta, 5.5, 1.0);
    doesABTreeExist = false;
    _fitab = true;
  }
  shapana->set_const(_nsamples, _firstsample, _lastsample, _presample, _nevtmax, _noise, _chi2cut);

  if (doesABTreeExist && fAB)
    fAB->Close();

  //  2) APD file

  std::stringstream nameapdfile;
  nameapdfile << resdir_ << "/APDPN_LASER.root";
  resfile = nameapdfile.str();

  // Laser events counter

  laserEvents = 0;
}

//========================================================================
void EcalLaserAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c) {
  //========================================================================

  ++iEvent;

  // retrieving DCC header
  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const EcalRawDataCollection* DCCHeader = nullptr;
  e.getByToken(rawDataToken_, pDCCHeader);
  if (!pDCCHeader.isValid()) {
    edm::LogError("nodata") << "Error! can't get the product retrieving DCC header" << eventHeaderCollection_.c_str()
                            << " " << eventHeaderProducer_.c_str();
  } else {
    DCCHeader = pDCCHeader.product();
  }

  //retrieving crystal data from Event
  edm::Handle<EBDigiCollection> pEBDigi;
  const EBDigiCollection* EBDigi = nullptr;
  edm::Handle<EEDigiCollection> pEEDigi;
  const EEDigiCollection* EEDigi = nullptr;
  if (_ecalPart == "EB") {
    e.getByToken(ebDigiToken_, pEBDigi);
    if (!pEBDigi.isValid()) {
      edm::LogError("nodata") << "Error! can't get the product retrieving EB crystal data " << digiCollection_.c_str();
    } else {
      EBDigi = pEBDigi.product();
    }
  } else if (_ecalPart == "EE") {
    e.getByToken(eeDigiToken_, pEEDigi);
    if (!pEEDigi.isValid()) {
      edm::LogError("nodata") << "Error! can't get the product retrieving EE crystal data " << digiCollection_.c_str();
    } else {
      EEDigi = pEEDigi.product();
    }
  } else {
    edm::LogError("cfg_error") << " Wrong ecalPart in cfg file ";
    return;
  }

  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection> pPNDigi;
  const EcalPnDiodeDigiCollection* PNDigi = nullptr;
  e.getByToken(pnDiodeDigiToken_, pPNDigi);
  if (!pPNDigi.isValid()) {
    edm::LogError("nodata") << "Error! can't get the product " << digiPNCollection_.c_str();
  } else {
    PNDigi = pPNDigi.product();
  }

  // retrieving electronics mapping
  const auto& TheMapping = c.getData(mappingToken_);

  // ============================
  // Decode DCCHeader Information
  // ============================

  for (EcalRawDataCollection::const_iterator headerItr = DCCHeader->begin(); headerItr != DCCHeader->end();
       ++headerItr) {
    // Get run type and run number

    int fed = headerItr->fedId();
    if (fed != _fedid && _fedid != -999)
      continue;

    runType = headerItr->getRunType();
    runNum = headerItr->getRunNumber();
    event = headerItr->getLV1();

    dccID = headerItr->getDccInTCCCommand();
    fedID = headerItr->fedId();
    lightside = headerItr->getRtHalf();

    // Check fed corresponds to the DCC in TCC

    if (600 + dccID != fedID)
      continue;

    // Cut on runType

    if (runType != EcalDCCHeaderBlock::LASER_STD && runType != EcalDCCHeaderBlock::LASER_GAP &&
        runType != EcalDCCHeaderBlock::LASER_POWER_SCAN && runType != EcalDCCHeaderBlock::LASER_DELAY_SCAN)
      return;

    // Retrieve laser color and event number

    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    color = settings.wavelength;
    if (color < 0)
      return;

    std::vector<int>::iterator iter = find(colors.begin(), colors.end(), color);
    if (iter == colors.end()) {
      colors.push_back(color);
      edm::LogVerbatim("EcalLaserAnalyzer") << " new color found " << color << " " << colors.size();
    }
  }

  // Cut on fedID

  if (fedID != _fedid && _fedid != -999)
    return;

  // Count laser events

  laserEvents++;

  // ======================
  // Decode PN Information
  // ======================

  TPNFit* pnfit = new TPNFit();
  pnfit->init(_nsamplesPN, _firstsamplePN, _lastsamplePN);

  double chi2pn = 0;
  unsigned int samplemax = 0;
  int pnGain = 0;

  std::map<int, std::vector<double> > allPNAmpl;
  std::map<int, std::vector<double> > allPNGain;

  // Loop on PNs digis

  for (EcalPnDiodeDigiCollection::const_iterator pnItr = PNDigi->begin(); pnItr != PNDigi->end(); ++pnItr) {
    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());

    if (_debug == 1)
      edm::LogVerbatim("EcalLaserAnalyzer")
          << "-- debug -- Inside PNDigi - pnID=" << pnDetId.iPnId() << ", dccID=" << pnDetId.iDCCId();

    // Skip MEM DCC without relevant data

    bool isMemRelevant = Mem->isMemRelevant(pnDetId.iDCCId());
    if (!isMemRelevant)
      continue;

    // Loop on PN samples

    for (int samId = 0; samId < (*pnItr).size(); samId++) {
      pn[samId] = (*pnItr).sample(samId).adc();
      pnG[samId] = (*pnItr).sample(samId).gainId();
      if (samId == 0)
        pnGain = pnG[samId];
      if (samId > 0)
        pnGain = int(TMath::Max(pnG[samId], pnGain));
    }

    if (pnGain != 1)
      edm::LogVerbatim("EcalLaserAnalyzer") << "PN gain different from 1";

    // Calculate amplitude from pulse

    PNPulse->setPulse(pn);
    pnNoPed = PNPulse->getAdcWithoutPedestal();
    samplemax = PNPulse->getMaxSample();
    chi2pn = pnfit->doFit(samplemax, &pnNoPed[0]);
    if (chi2pn == 101 || chi2pn == 102 || chi2pn == 103)
      pnAmpl = 0.;
    else
      pnAmpl = pnfit->getAmpl();

    // Apply linearity correction

    double corr = 1.0;
    if (_docorpn)
      corr = pnCorrector->getPNCorrectionFactor(pnAmpl, pnGain);
    pnAmpl *= corr;

    // Fill PN ampl vector

    allPNAmpl[pnDetId.iDCCId()].push_back(pnAmpl);

    if (_debug == 1)
      edm::LogVerbatim("EcalLaserAnalyzer") << "-- debug -- Inside PNDigi - PNampl=" << pnAmpl << ", PNgain=" << pnGain;
  }

  // ===========================
  // Decode EBDigis Information
  // ===========================

  int adcGain = 0;

  if (EBDigi) {
    // Loop on crystals
    //===================

    for (EBDigiCollection::const_iterator digiItr = EBDigi->begin(); digiItr != EBDigi->end(); ++digiItr) {
      // Retrieve geometry
      //===================

      EBDetId id_crystal(digiItr->id());
      EBDataFrame df(*digiItr);
      EcalElectronicsId elecid_crystal = TheMapping.getElectronicsId(id_crystal);

      int etaG = id_crystal.ieta();  // global
      int phiG = id_crystal.iphi();  // global

      std::pair<int, int> LocalCoord = MEEBGeom::localCoord(etaG, phiG);

      int etaL = LocalCoord.first;   // local
      int phiL = LocalCoord.second;  // local

      int strip = elecid_crystal.stripId();
      int xtal = elecid_crystal.xtalId();

      int module = MEEBGeom::lmmod(etaG, phiG);
      int tower = elecid_crystal.towerId();

      int apdRefTT = MEEBGeom::apdRefTower(module);

      std::pair<int, int> pnpair = MEEBGeom::pn(module);
      unsigned int MyPn0 = pnpair.first;
      unsigned int MyPn1 = pnpair.second;

      int lmr = MEEBGeom::lmr(etaG, phiG);
      unsigned int channel = MEEBGeom::electronic_channel(etaL, phiL);
      assert(channel < nCrys);

      setGeomEB(etaG, phiG, module, tower, strip, xtal, apdRefTT, channel, lmr);

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer")
            << "-- debug -- Inside EBDigi - towerID:" << towerID << " channelID:" << channelID << " module:" << module
            << " modules:" << modules.size();

      // APD Pulse
      //===========

      // Loop on adc samples

      for (unsigned int i = 0; i < (*digiItr).size(); ++i) {
        EcalMGPASample samp_crystal(df.sample(i));
        adc[i] = samp_crystal.adc();
        adcG[i] = samp_crystal.gainId();
        adc[i] *= adcG[i];
        if (i == 0)
          adcGain = adcG[i];
        if (i > 0)
          adcGain = TMath::Max(adcG[i], adcGain);
      }

      APDPulse->setPulse(adc);

      // Quality checks
      //================

      if (adcGain != 1)
        nEvtBadGain[channel]++;
      if (!APDPulse->isTimingQualOK())
        nEvtBadTiming[channel]++;
      nEvtTot[channel]++;

      // Associate PN ampl
      //===================

      int mem0 = Mem->Mem(lmr, 0);
      int mem1 = Mem->Mem(lmr, 1);

      if (allPNAmpl[mem0].size() > MyPn0)
        pn0 = allPNAmpl[mem0][MyPn0];
      else
        pn0 = 0;
      if (allPNAmpl[mem1].size() > MyPn1)
        pn1 = allPNAmpl[mem1][MyPn1];
      else
        pn1 = 0;

      // Fill if Pulse is fine
      //=======================

      if (APDPulse->isPulseOK() && lightside == side) {
        ADCtrees[channel]->Fill();

        Delta01->addEntry(APDPulse->getDelta(0, 1));
        Delta12->addEntry(APDPulse->getDelta(1, 2));

        if (nevtAB[channel] < _nevtmax && _fitab) {
          if (doesABTreeExist)
            shapana->putAllVals(channel, adc, eta, phi);
          else
            shapana->putAllVals(channel, adc, eta, phi, dccID, side, towerID, channelID);
          nevtAB[channel]++;
        }
      }
    }

  } else if (EEDigi) {
    // Loop on crystals
    //===================

    for (EEDigiCollection::const_iterator digiItr = EEDigi->begin(); digiItr != EEDigi->end(); ++digiItr) {
      // Retrieve geometry
      //===================

      EEDetId id_crystal(digiItr->id());
      EEDataFrame df(*digiItr);
      EcalElectronicsId elecid_crystal = TheMapping.getElectronicsId(id_crystal);

      int etaG = id_crystal.iy();
      int phiG = id_crystal.ix();

      int iX = (phiG - 1) / 5 + 1;
      int iY = (etaG - 1) / 5 + 1;

      int tower = elecid_crystal.towerId();
      int ch = elecid_crystal.channelId() - 1;

      int module = MEEEGeom::lmmod(iX, iY);
      if (module >= 18 && side == 1)
        module += 2;
      int lmr = MEEEGeom::lmr(iX, iY, iZ);
      int dee = MEEEGeom::dee(lmr);
      int apdRefTT = MEEEGeom::apdRefTower(lmr, module);

      std::pair<int, int> pnpair = MEEEGeom::pn(dee, module);
      unsigned int MyPn0 = pnpair.first;
      unsigned int MyPn1 = pnpair.second;

      int hashedIndex = 100000 * eta + phi;
      if (channelMapEE.count(hashedIndex) == 0) {
        channelMapEE[hashedIndex] = channelIteratorEE;
        channelIteratorEE++;
      }
      unsigned int channel = channelMapEE[hashedIndex];
      assert(channel < nCrys);

      setGeomEE(etaG, phiG, iX, iY, iZ, module, tower, ch, apdRefTT, channel, lmr);

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer")
            << "-- debug -- Inside EEDigi - towerID:" << towerID << " channelID:" << channelID << " module:" << module
            << " modules:" << modules.size();

      // APD Pulse
      //===========

      if ((*digiItr).size() > 10)
        edm::LogVerbatim("EcalLaserAnalyzer") << "SAMPLES SIZE > 10!" << (*digiItr).size();

      // Loop on adc samples

      for (unsigned int i = 0; i < (*digiItr).size(); ++i) {
        EcalMGPASample samp_crystal(df.sample(i));
        adc[i] = samp_crystal.adc();
        adcG[i] = samp_crystal.gainId();
        adc[i] *= adcG[i];

        if (i == 0)
          adcGain = adcG[i];
        if (i > 0)
          adcGain = TMath::Max(adcG[i], adcGain);
      }

      APDPulse->setPulse(adc);

      // Quality checks
      //================

      if (adcGain != 1)
        nEvtBadGain[channel]++;
      if (!APDPulse->isTimingQualOK())
        nEvtBadTiming[channel]++;
      nEvtTot[channel]++;

      // Associate PN ampl
      //===================

      int mem0 = Mem->Mem(lmr, 0);
      int mem1 = Mem->Mem(lmr, 1);

      if (allPNAmpl[mem0].size() > MyPn0)
        pn0 = allPNAmpl[mem0][MyPn0];
      else
        pn0 = 0;
      if (allPNAmpl[mem1].size() > MyPn1)
        pn1 = allPNAmpl[mem1][MyPn1];
      else
        pn1 = 0;

      // Fill if Pulse is fine
      //=======================

      if (APDPulse->isPulseOK() && lightside == side) {
        ADCtrees[channel]->Fill();

        Delta01->addEntry(APDPulse->getDelta(0, 1));
        Delta12->addEntry(APDPulse->getDelta(1, 2));

        if (nevtAB[channel] < _nevtmax && _fitab) {
          if (doesABTreeExist)
            shapana->putAllVals(channel, adc, eta, phi);
          else
            shapana->putAllVals(channel, adc, eta, phi, dccID, side, towerID, channelID);
          nevtAB[channel]++;
        }
      }
    }
  }
}
// analyze

//========================================================================
void EcalLaserAnalyzer::endJob() {
  //========================================================================

  // Adjust channel numbers for EE
  //===============================

  if (_ecalPart == "EE") {
    nCrys = channelMapEE.size();
    shapana->set_nch(nCrys);
  }

  // Set presamples number
  //======================

  double delta01 = Delta01->getMean();
  double delta12 = Delta12->getMean();
  if (delta12 > _presamplecut) {
    _presample = 2;
    if (delta01 > _presamplecut)
      _presample = 1;
  }

  APDPulse->setPresamples(_presample);
  shapana->set_presample(_presample);

  //  Get alpha and beta
  //======================

  if (_fitab) {
    edm::LogVerbatim("EcalLaserAnalyzer") << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
    edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+     Analyzing data: getting (alpha, beta)     +=+";
    TFile* fAB = nullptr;
    TTree* ABInit = nullptr;
    if (doesABTreeExist) {
      fAB = new TFile(alphainitfile.c_str());
    }
    if (doesABTreeExist && fAB) {
      ABInit = (TTree*)fAB->Get("ABCol0");
    }
    shapana->computeShape(alphafile, ABInit);
    edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+    .................................... done  +=+";
    edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
  }

  // Don't do anything if there is no events
  //=========================================

  if (laserEvents == 0) {
    ADCFile->Close();
    std::stringstream del;
    del << "rm " << ADCfile;
    system(del.str().c_str());
    edm::LogVerbatim("EcalLaserAnalyzer") << " No Laser Events ";
    return;
  }

  // Set quality flags for gains and timing
  //=========================================

  double BadGainEvtPercentage = 0.0;
  double BadTimingEvtPercentage = 0.0;

  int nChanBadGain = 0;
  int nChanBadTiming = 0;

  for (unsigned int i = 0; i < nCrys; i++) {
    if (nEvtTot[i] != 0) {
      BadGainEvtPercentage = double(nEvtBadGain[i]) / double(nEvtTot[i]);
      BadTimingEvtPercentage = double(nEvtBadTiming[i]) / double(nEvtTot[i]);
    }
    if (BadGainEvtPercentage > _qualpercent) {
      wasGainOK[i] = false;
      nChanBadGain++;
    }
    if (BadTimingEvtPercentage > _qualpercent) {
      wasTimingOK[i] = false;
      nChanBadTiming++;
    }
  }

  double BadGainChanPercentage = double(nChanBadGain) / double(nCrys);
  double BadTimingChanPercentage = double(nChanBadTiming) / double(nCrys);

  if (BadGainChanPercentage > _qualpercent)
    isGainOK = false;
  if (BadTimingChanPercentage > _qualpercent)
    isTimingOK = false;

  // Analyze adc samples to get amplitudes
  //=======================================

  edm::LogVerbatim("EcalLaserAnalyzer") << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
  edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+     Analyzing laser data: getting APD, PN, APD/PN, PN/PN    +=+";

  if (!isGainOK)
    edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+ ............................ WARNING! APD GAIN WAS NOT 1    +=+";
  if (!isTimingOK)
    edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+ ............................ WARNING! TIMING WAS BAD        +=+";

  APDFile = new TFile(APDfile.c_str(), "RECREATE");

  int ieta, iphi, channelID, towerID, flag;

  for (unsigned int i = 0; i < nCrys; i++) {
    std::stringstream name;
    name << "APDTree" << i + 1;

    APDtrees[i] = new TTree(name.str().c_str(), name.str().c_str());

    //List of branches

    APDtrees[i]->Branch("event", &event, "event/I");
    APDtrees[i]->Branch("color", &color, "color/I");
    APDtrees[i]->Branch("iphi", &iphi, "iphi/I");
    APDtrees[i]->Branch("ieta", &ieta, "ieta/I");
    APDtrees[i]->Branch("side", &side, "side/I");
    APDtrees[i]->Branch("dccID", &dccID, "dccID/I");
    APDtrees[i]->Branch("towerID", &towerID, "towerID/I");
    APDtrees[i]->Branch("channelID", &channelID, "channelID/I");
    APDtrees[i]->Branch("apdAmpl", &apdAmpl, "apdAmpl/D");
    APDtrees[i]->Branch("apdTime", &apdTime, "apdTime/D");
    if (_saveallevents)
      APDtrees[i]->Branch("adc", &adc, "adc[10]/D");
    APDtrees[i]->Branch("flag", &flag, "flag/I");
    APDtrees[i]->Branch("flagAB", &flagAB, "flagAB/I");
    APDtrees[i]->Branch("pn0", &pn0, "pn0/D");
    APDtrees[i]->Branch("pn1", &pn1, "pn1/D");

    APDtrees[i]->SetBranchAddress("event", &event);
    APDtrees[i]->SetBranchAddress("color", &color);
    APDtrees[i]->SetBranchAddress("iphi", &iphi);
    APDtrees[i]->SetBranchAddress("ieta", &ieta);
    APDtrees[i]->SetBranchAddress("side", &side);
    APDtrees[i]->SetBranchAddress("dccID", &dccID);
    APDtrees[i]->SetBranchAddress("towerID", &towerID);
    APDtrees[i]->SetBranchAddress("channelID", &channelID);
    APDtrees[i]->SetBranchAddress("apdAmpl", &apdAmpl);
    APDtrees[i]->SetBranchAddress("apdTime", &apdTime);
    if (_saveallevents)
      APDtrees[i]->SetBranchAddress("adc", adc);
    APDtrees[i]->SetBranchAddress("flag", &flag);
    APDtrees[i]->SetBranchAddress("flagAB", &flagAB);
    APDtrees[i]->SetBranchAddress("pn0", &pn0);
    APDtrees[i]->SetBranchAddress("pn1", &pn1);
  }

  for (unsigned int iref = 0; iref < nRefChan; iref++) {
    for (unsigned int imod = 0; imod < nMod; imod++) {
      int jmod = modules[imod];

      std::stringstream nameref;
      nameref << "refAPDTree" << imod << "_" << iref;

      RefAPDtrees[iref][jmod] = new TTree(nameref.str().c_str(), nameref.str().c_str());

      RefAPDtrees[iref][jmod]->Branch("eventref", &eventref, "eventref/I");
      RefAPDtrees[iref][jmod]->Branch("colorref", &colorref, "colorref/I");
      if (iref == 0)
        RefAPDtrees[iref][jmod]->Branch("apdAmplA", &apdAmplA, "apdAmplA/D");
      if (iref == 1)
        RefAPDtrees[iref][jmod]->Branch("apdAmplB", &apdAmplB, "apdAmplB/D");

      RefAPDtrees[iref][jmod]->SetBranchAddress("eventref", &eventref);
      RefAPDtrees[iref][jmod]->SetBranchAddress("colorref", &colorref);
      if (iref == 0)
        RefAPDtrees[iref][jmod]->SetBranchAddress("apdAmplA", &apdAmplA);
      if (iref == 1)
        RefAPDtrees[iref][jmod]->SetBranchAddress("apdAmplB", &apdAmplB);
    }
  }

  assert(colors.size() <= nColor);
  unsigned int nCol = colors.size();

  // Declare PN stuff
  //===================

  for (unsigned int iM = 0; iM < nMod; iM++) {
    unsigned int iMod = modules[iM] - 1;

    for (unsigned int ich = 0; ich < nPNPerMod; ich++) {
      for (unsigned int icol = 0; icol < nCol; icol++) {
        PNFirstAnal[iMod][ich][icol] = new TPN(ich);
        PNAnal[iMod][ich][icol] = new TPN(ich);
      }
    }
  }

  // Declare function for APD ampl fit
  //===================================

  PulseFitWithFunction* pslsfit = new PulseFitWithFunction();
  double chi2;

  for (unsigned int iCry = 0; iCry < nCrys; iCry++) {
    for (unsigned int icol = 0; icol < nCol; icol++) {
      // Declare APD stuff
      //===================

      APDFirstAnal[iCry][icol] = new TAPD();
      IsThereDataADC[iCry][icol] = 1;
      std::stringstream cut;
      cut << "color==" << colors[icol];
      if (ADCtrees[iCry]->GetEntries(cut.str().c_str()) < 10)
        IsThereDataADC[iCry][icol] = 0;
    }

    unsigned int iMod = iModule[iCry] - 1;
    double alpha, beta;

    // Loop on events
    //================

    for (Long64_t jentry = 0; jentry < ADCtrees[iCry]->GetEntriesFast(); jentry++) {
      ADCtrees[iCry]->GetEntry(jentry);

      // Get back color

      unsigned int iCol = 0;
      for (unsigned int i = 0; i < nCol; i++) {
        if (color == colors[i]) {
          iCol = i;
          i = colors.size();
        }
      }

      // Retreive alpha and beta

      std::vector<double> abvals = shapana->getVals(iCry);
      alpha = abvals[0];
      beta = abvals[1];
      flagAB = int(abvals[4]);
      iphi = iPhi[iCry];
      ieta = iEta[iCry];
      towerID = iTowerID[iCry];
      channelID = iChannelID[iCry];

      // Amplitude calculation

      APDPulse->setPulse(adc);
      adcNoPed = APDPulse->getAdcWithoutPedestal();

      apdAmpl = 0;
      apdAmplA = 0;
      apdAmplB = 0;
      apdTime = 0;

      if (APDPulse->isPulseOK()) {
        pslsfit->init(_nsamples, _firstsample, _lastsample, _niter, alpha, beta);
        chi2 = pslsfit->doFit(&adcNoPed[0]);

        if (chi2 < 0. || chi2 == 102 || chi2 == 101) {
          apdAmpl = 0;
          apdTime = 0;
          flag = 0;
        } else {
          apdAmpl = pslsfit->getAmpl();
          apdTime = pslsfit->getTime();
          flag = 1;
        }
      } else {
        apdAmpl = 0;
        apdTime = 0;
        flag = 0;
      }

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer") << "-- debug test -- apdAmpl=" << apdAmpl << ", apdTime=" << apdTime;
      double pnmean;
      if (pn0 < 10 && pn1 > 10) {
        pnmean = pn1;
      } else if (pn1 < 10 && pn0 > 10) {
        pnmean = pn0;
      } else
        pnmean = 0.5 * (pn0 + pn1);

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer") << "-- debug test -- pn0=" << pn0 << ", pn1=" << pn1;

      // Fill PN stuff
      //===============

      if (firstChanMod[iMod] == iCry && IsThereDataADC[iCry][iCol] == 1) {
        for (unsigned int ichan = 0; ichan < nPNPerMod; ichan++) {
          PNFirstAnal[iMod][ichan][iCol]->addEntry(pnmean, pn0, pn1);
        }
      }

      // Fill APD stuff
      //================

      if (APDPulse->isPulseOK()) {
        APDFirstAnal[iCry][iCol]->addEntry(apdAmpl, pnmean, pn0, pn1, apdTime);
        APDtrees[iCry]->Fill();

        // Fill reference trees
        //=====================

        if (apdRefMap[0][iMod + 1] == iCry || apdRefMap[1][iMod + 1] == iCry) {
          apdAmplA = 0.0;
          apdAmplB = 0.0;
          eventref = event;
          colorref = color;

          for (unsigned int ir = 0; ir < nRefChan; ir++) {
            if (apdRefMap[ir][iMod + 1] == iCry) {
              if (ir == 0)
                apdAmplA = apdAmpl;
              else if (ir == 1)
                apdAmplB = apdAmpl;
              RefAPDtrees[ir][iMod + 1]->Fill();
            }
          }
        }
      }
    }
  }

  delete pslsfit;

  ADCFile->Close();

  // Remove temporary file
  //=======================
  std::stringstream del;
  del << "rm " << ADCfile;
  system(del.str().c_str());

  // Create output trees
  //=====================

  resFile = new TFile(resfile.c_str(), "RECREATE");

  for (unsigned int iColor = 0; iColor < nCol; iColor++) {
    std::stringstream nametree;
    nametree << "APDCol" << colors[iColor];
    std::stringstream nametree2;
    nametree2 << "PNCol" << colors[iColor];

    restrees[iColor] = new TTree(nametree.str().c_str(), nametree.str().c_str());
    respntrees[iColor] = new TTree(nametree2.str().c_str(), nametree2.str().c_str());

    restrees[iColor]->Branch("iphi", &iphi, "iphi/I");
    restrees[iColor]->Branch("ieta", &ieta, "ieta/I");
    restrees[iColor]->Branch("side", &side, "side/I");
    restrees[iColor]->Branch("dccID", &dccID, "dccID/I");
    restrees[iColor]->Branch("moduleID", &moduleID, "moduleID/I");
    restrees[iColor]->Branch("towerID", &towerID, "towerID/I");
    restrees[iColor]->Branch("channelID", &channelID, "channelID/I");
    restrees[iColor]->Branch("APD", &APD, "APD[6]/D");
    restrees[iColor]->Branch("Time", &Time, "Time[6]/D");
    restrees[iColor]->Branch("APDoPN", &APDoPN, "APDoPN[6]/D");
    restrees[iColor]->Branch("APDoPNA", &APDoPNA, "APDoPNA[6]/D");
    restrees[iColor]->Branch("APDoPNB", &APDoPNB, "APDoPNB[6]/D");
    restrees[iColor]->Branch("APDoAPDA", &APDoAPDA, "APDoAPDA[6]/D");
    restrees[iColor]->Branch("APDoAPDB", &APDoAPDB, "APDoAPDB[6]/D");
    restrees[iColor]->Branch("flag", &flag, "flag/I");

    respntrees[iColor]->Branch("side", &side, "side/I");
    respntrees[iColor]->Branch("moduleID", &moduleID, "moduleID/I");
    respntrees[iColor]->Branch("pnID", &pnID, "pnID/I");
    respntrees[iColor]->Branch("PN", &PN, "PN[6]/D");
    respntrees[iColor]->Branch("PNoPN", &PNoPN, "PNoPN[6]/D");
    respntrees[iColor]->Branch("PNoPNA", &PNoPNA, "PNoPNA[6]/D");
    respntrees[iColor]->Branch("PNoPNB", &PNoPNB, "PNoPNB[6]/D");

    restrees[iColor]->SetBranchAddress("iphi", &iphi);
    restrees[iColor]->SetBranchAddress("ieta", &ieta);
    restrees[iColor]->SetBranchAddress("side", &side);
    restrees[iColor]->SetBranchAddress("dccID", &dccID);
    restrees[iColor]->SetBranchAddress("moduleID", &moduleID);
    restrees[iColor]->SetBranchAddress("towerID", &towerID);
    restrees[iColor]->SetBranchAddress("channelID", &channelID);
    restrees[iColor]->SetBranchAddress("APD", APD);
    restrees[iColor]->SetBranchAddress("Time", Time);
    restrees[iColor]->SetBranchAddress("APDoPN", APDoPN);
    restrees[iColor]->SetBranchAddress("APDoPNA", APDoPNA);
    restrees[iColor]->SetBranchAddress("APDoPNB", APDoPNB);
    restrees[iColor]->SetBranchAddress("APDoAPDA", APDoAPDA);
    restrees[iColor]->SetBranchAddress("APDoAPDB", APDoAPDB);
    restrees[iColor]->SetBranchAddress("flag", &flag);

    respntrees[iColor]->SetBranchAddress("side", &side);
    respntrees[iColor]->SetBranchAddress("moduleID", &moduleID);
    respntrees[iColor]->SetBranchAddress("pnID", &pnID);
    respntrees[iColor]->SetBranchAddress("PN", PN);
    respntrees[iColor]->SetBranchAddress("PNoPN", PNoPN);
    respntrees[iColor]->SetBranchAddress("PNoPNA", PNoPNA);
    respntrees[iColor]->SetBranchAddress("PNoPNB", PNoPNB);
  }

  // Set Cuts for PN stuff
  //=======================

  for (unsigned int iM = 0; iM < nMod; iM++) {
    unsigned int iMod = modules[iM] - 1;

    for (unsigned int ich = 0; ich < nPNPerMod; ich++) {
      for (unsigned int icol = 0; icol < nCol; icol++) {
        PNAnal[iMod][ich][icol]->setPNCut(PNFirstAnal[iMod][ich][icol]->getPN().at(0),
                                          PNFirstAnal[iMod][ich][icol]->getPN().at(1));
      }
    }
  }

  // Build ref trees indexes
  //========================
  for (unsigned int imod = 0; imod < nMod; imod++) {
    int jmod = modules[imod];
    if (RefAPDtrees[0][jmod]->GetEntries() != 0 && RefAPDtrees[1][jmod]->GetEntries() != 0) {
      RefAPDtrees[0][jmod]->BuildIndex("eventref");
      RefAPDtrees[1][jmod]->BuildIndex("eventref");
    }
  }

  // Final loop on crystals
  //=======================

  for (unsigned int iCry = 0; iCry < nCrys; iCry++) {
    unsigned int iMod = iModule[iCry] - 1;

    // Set cuts on APD stuff
    //=======================

    for (unsigned int iCol = 0; iCol < nCol; iCol++) {
      std::vector<double> lowcut;
      std::vector<double> highcut;
      double cutMin;
      double cutMax;

      cutMin = APDFirstAnal[iCry][iCol]->getAPD().at(0) - 2.0 * APDFirstAnal[iCry][iCol]->getAPD().at(1);
      if (cutMin < 0)
        cutMin = 0;
      cutMax = APDFirstAnal[iCry][iCol]->getAPD().at(0) + 2.0 * APDFirstAnal[iCry][iCol]->getAPD().at(1);

      lowcut.push_back(cutMin);
      highcut.push_back(cutMax);

      cutMin = APDFirstAnal[iCry][iCol]->getTime().at(0) - 2.0 * APDFirstAnal[iCry][iCol]->getTime().at(1);
      cutMax = APDFirstAnal[iCry][iCol]->getTime().at(0) + 2.0 * APDFirstAnal[iCry][iCol]->getTime().at(1);
      lowcut.push_back(cutMin);
      highcut.push_back(cutMax);

      APDAnal[iCry][iCol] = new TAPD();
      APDAnal[iCry][iCol]->setAPDCut(APDFirstAnal[iCry][iCol]->getAPD().at(0),
                                     APDFirstAnal[iCry][iCol]->getAPD().at(1));
      APDAnal[iCry][iCol]->setAPDoPNCut(APDFirstAnal[iCry][iCol]->getAPDoPN().at(0),
                                        APDFirstAnal[iCry][iCol]->getAPDoPN().at(1));
      APDAnal[iCry][iCol]->setAPDoPN0Cut(APDFirstAnal[iCry][iCol]->getAPDoPN0().at(0),
                                         APDFirstAnal[iCry][iCol]->getAPDoPN0().at(1));
      APDAnal[iCry][iCol]->setAPDoPN1Cut(APDFirstAnal[iCry][iCol]->getAPDoPN1().at(0),
                                         APDFirstAnal[iCry][iCol]->getAPDoPN1().at(1));
      APDAnal[iCry][iCol]->setTimeCut(APDFirstAnal[iCry][iCol]->getTime().at(0),
                                      APDFirstAnal[iCry][iCol]->getTime().at(1));
      APDAnal[iCry][iCol]->set2DAPDoAPD0Cut(lowcut, highcut);
      APDAnal[iCry][iCol]->set2DAPDoAPD1Cut(lowcut, highcut);
    }

    // Final loop on events
    //=======================

    for (Long64_t jentry = 0; jentry < APDtrees[iCry]->GetEntriesFast(); jentry++) {
      APDtrees[iCry]->GetEntry(jentry);

      double pnmean;
      if (pn0 < 10 && pn1 > 10) {
        pnmean = pn1;
      } else if (pn1 < 10 && pn0 > 10) {
        pnmean = pn0;
      } else
        pnmean = 0.5 * (pn0 + pn1);

      // Get back color
      //================

      unsigned int iCol = 0;
      for (unsigned int i = 0; i < nCol; i++) {
        if (color == colors[i]) {
          iCol = i;
          i = colors.size();
        }
      }

      // Fill PN stuff
      //===============

      if (firstChanMod[iMod] == iCry && IsThereDataADC[iCry][iCol] == 1) {
        for (unsigned int ichan = 0; ichan < nPNPerMod; ichan++) {
          PNAnal[iMod][ichan][iCol]->addEntry(pnmean, pn0, pn1);
        }
      }

      // Get ref amplitudes
      //===================

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer") << "-- debug test -- Last Loop event:" << event << " apdAmpl:" << apdAmpl;
      apdAmplA = 0.0;
      apdAmplB = 0.0;

      for (unsigned int iRef = 0; iRef < nRefChan; iRef++) {
        RefAPDtrees[iRef][iMod + 1]->GetEntryWithIndex(event);
      }

      if (_debug == 1)
        edm::LogVerbatim("EcalLaserAnalyzer")
            << "-- debug test -- Last Loop apdAmplA:" << apdAmplA << " apdAmplB:" << apdAmplB << ", event:" << event
            << ", eventref:" << eventref;

      // Fill APD stuff
      //===============

      APDAnal[iCry][iCol]->addEntry(apdAmpl, pnmean, pn0, pn1, apdTime, apdAmplA, apdAmplB);
    }

    moduleID = iMod + 1;

    if (moduleID >= 20)
      moduleID -= 2;  // Trick to fix endcap specificity

    // Get final results for APD
    //===========================

    for (unsigned int iColor = 0; iColor < nCol; iColor++) {
      std::vector<double> apdvec = APDAnal[iCry][iColor]->getAPD();
      std::vector<double> apdpnvec = APDAnal[iCry][iColor]->getAPDoPN();
      std::vector<double> apdpn0vec = APDAnal[iCry][iColor]->getAPDoPN0();
      std::vector<double> apdpn1vec = APDAnal[iCry][iColor]->getAPDoPN1();
      std::vector<double> timevec = APDAnal[iCry][iColor]->getTime();
      std::vector<double> apdapd0vec = APDAnal[iCry][iColor]->getAPDoAPD0();
      std::vector<double> apdapd1vec = APDAnal[iCry][iColor]->getAPDoAPD1();

      for (unsigned int i = 0; i < apdvec.size(); i++) {
        APD[i] = apdvec.at(i);
        APDoPN[i] = apdpnvec.at(i);
        APDoPNA[i] = apdpn0vec.at(i);
        APDoPNB[i] = apdpn1vec.at(i);
        APDoAPDA[i] = apdapd0vec.at(i);
        APDoAPDB[i] = apdapd1vec.at(i);
        Time[i] = timevec.at(i);
      }

      // Fill APD results trees
      //========================

      iphi = iPhi[iCry];
      ieta = iEta[iCry];
      dccID = idccID[iCry];
      side = iside[iCry];
      towerID = iTowerID[iCry];
      channelID = iChannelID[iCry];

      if (!wasGainOK[iCry] || !wasTimingOK[iCry] || IsThereDataADC[iCry][iColor] == 0) {
        flag = 0;
      } else
        flag = 1;

      restrees[iColor]->Fill();
    }
  }

  // Get final results for PN
  //==========================

  for (unsigned int iM = 0; iM < nMod; iM++) {
    unsigned int iMod = modules[iM] - 1;

    side = iside[firstChanMod[iMod]];

    for (unsigned int ch = 0; ch < nPNPerMod; ch++) {
      pnID = ch;
      moduleID = iMod + 1;

      if (moduleID >= 20)
        moduleID -= 2;  // Trick to fix endcap specificity

      for (unsigned int iColor = 0; iColor < nCol; iColor++) {
        std::vector<double> pnvec = PNAnal[iMod][ch][iColor]->getPN();
        std::vector<double> pnopnvec = PNAnal[iMod][ch][iColor]->getPNoPN();
        std::vector<double> pnopn0vec = PNAnal[iMod][ch][iColor]->getPNoPN0();
        std::vector<double> pnopn1vec = PNAnal[iMod][ch][iColor]->getPNoPN1();

        for (unsigned int i = 0; i < pnvec.size(); i++) {
          PN[i] = pnvec.at(i);
          PNoPN[i] = pnopnvec.at(i);
          PNoPNA[i] = pnopn0vec.at(i);
          PNoPNB[i] = pnopn1vec.at(i);
        }

        // Fill PN results trees
        //========================

        respntrees[iColor]->Fill();
      }
    }
  }

  // Remove temporary files
  //========================
  if (!_saveallevents) {
    APDFile->Close();
    std::stringstream del2;
    del2 << "rm " << APDfile;
    system(del2.str().c_str());

  } else {
    APDFile->cd();
    APDtrees[0]->Write();

    APDFile->Close();
    resFile->cd();
  }

  // Save results
  //===============

  for (unsigned int i = 0; i < nCol; i++) {
    restrees[i]->Write();
    respntrees[i]->Write();
  }

  resFile->Close();

  edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+    .................................................. done  +=+";
  edm::LogVerbatim("EcalLaserAnalyzer") << "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
}

void EcalLaserAnalyzer::setGeomEB(
    int etaG, int phiG, int module, int tower, int strip, int xtal, int apdRefTT, int channel, int lmr) {
  side = MEEBGeom::side(etaG, phiG);

  assert(module >= *min_element(modules.begin(), modules.end()) &&
         module <= *max_element(modules.begin(), modules.end()));

  eta = etaG;
  phi = phiG;
  channelID = 5 * (strip - 1) + xtal - 1;
  towerID = tower;

  std::vector<int> apdRefChan = ME::apdRefChannels(module, lmr);
  for (unsigned int iref = 0; iref < nRefChan; iref++) {
    if (channelID == apdRefChan[iref] && towerID == apdRefTT && apdRefMap[iref].count(module) == 0) {
      apdRefMap[iref][module] = channel;
    }
  }

  if (isFirstChanModFilled[module - 1] == 0) {
    firstChanMod[module - 1] = channel;
    isFirstChanModFilled[module - 1] = 1;
  }

  iEta[channel] = eta;
  iPhi[channel] = phi;
  iModule[channel] = module;
  iTowerID[channel] = towerID;
  iChannelID[channel] = channelID;
  idccID[channel] = dccID;
  iside[channel] = side;
}

void EcalLaserAnalyzer::setGeomEE(
    int etaG, int phiG, int iX, int iY, int iZ, int module, int tower, int ch, int apdRefTT, int channel, int lmr) {
  side = MEEEGeom::side(iX, iY, iZ);

  assert(module >= *min_element(modules.begin(), modules.end()) &&
         module <= *max_element(modules.begin(), modules.end()));

  eta = etaG;
  phi = phiG;
  channelID = ch;
  towerID = tower;

  std::vector<int> apdRefChan = ME::apdRefChannels(module, lmr);
  for (unsigned int iref = 0; iref < nRefChan; iref++) {
    if (channelID == apdRefChan[iref] && towerID == apdRefTT && apdRefMap[iref].count(module) == 0) {
      apdRefMap[iref][module] = channel;
    }
  }

  if (isFirstChanModFilled[module - 1] == 0) {
    firstChanMod[module - 1] = channel;
    isFirstChanModFilled[module - 1] = 1;
  }

  iEta[channel] = eta;
  iPhi[channel] = phi;
  iModule[channel] = module;
  iTowerID[channel] = towerID;
  iChannelID[channel] = channelID;
  idccID[channel] = dccID;
  iside[channel] = side;
}

DEFINE_FWK_MODULE(EcalLaserAnalyzer);
