/* 
 *
 *  \class EcalTestPulseAnalyzer
 *
 *  primary author: Julie Malcles - CEA/Saclay
 *  author: Gautier Hamel De Monchenault - CEA/Saclay
 */
#include "TFile.h"
#include "TTree.h"

#include "EcalTestPulseAnalyzer.h"

#include <sstream>
#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPNFit.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TSFit.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"

using namespace std;

//========================================================================
EcalTestPulseAnalyzer::EcalTestPulseAnalyzer(const edm::ParameterSet& iConfig)
    //========================================================================
    : iEvent(0),
      eventHeaderCollection_(iConfig.getParameter<std::string>("eventHeaderCollection")),
      eventHeaderProducer_(iConfig.getParameter<std::string>("eventHeaderProducer")),
      digiCollection_(iConfig.getParameter<std::string>("digiCollection")),
      digiProducer_(iConfig.getParameter<std::string>("digiProducer")),
      digiPNCollection_(iConfig.getParameter<std::string>("digiPNCollection")),
      rawDataToken_(consumes<EcalRawDataCollection>(edm::InputTag(eventHeaderProducer_, eventHeaderCollection_))),
      pnDiodeDigiToken_(consumes<EcalPnDiodeDigiCollection>(edm::InputTag(digiProducer_, digiPNCollection_))),
      mappingToken_(esConsumes()),
      // framework parameters with default values
      _nsamples(iConfig.getUntrackedParameter<unsigned int>("nSamples", 10)),
      _presample(iConfig.getUntrackedParameter<unsigned int>("nPresamples", 3)),
      _firstsample(iConfig.getUntrackedParameter<unsigned int>("firstSample", 1)),
      _lastsample(iConfig.getUntrackedParameter<unsigned int>("lastSample", 2)),
      _samplemin(iConfig.getUntrackedParameter<unsigned int>("sampleMin", 3)),
      _samplemax(iConfig.getUntrackedParameter<unsigned int>("sampleMax", 9)),
      _nsamplesPN(iConfig.getUntrackedParameter<unsigned int>("nSamplesPN", 50)),
      _presamplePN(iConfig.getUntrackedParameter<unsigned int>("nPresamplesPN", 6)),
      _firstsamplePN(iConfig.getUntrackedParameter<unsigned int>("firstSamplePN", 7)),
      _lastsamplePN(iConfig.getUntrackedParameter<unsigned int>("lastSamplePN", 8)),
      _niter(iConfig.getUntrackedParameter<unsigned int>("nIter", 3)),
      _chi2max(iConfig.getUntrackedParameter<double>("chi2Max", 10.0)),
      _timeofmax(iConfig.getUntrackedParameter<double>("timeOfMax", 4.5)),
      _ecalPart(iConfig.getUntrackedParameter<std::string>("ecalPart", "EB")),
      _fedid(iConfig.getUntrackedParameter<int>("fedID", -999)),
      resdir_(iConfig.getUntrackedParameter<std::string>("resDir")),
      nCrys(NCRYSEB),
      nMod(NMODEB),
      nGainPN(NGAINPN),
      nGainAPD(NGAINAPD),
      towerID(-1),
      channelID(-1),
      runType(-1),
      runNum(0),
      fedID(-1),
      dccID(-1),
      side(-1),
      iZ(1),
      phi(-1),
      eta(-1),
      event(0),
      apdAmpl(0),
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

  // Define geometrical constants

  if (_ecalPart == "EB") {
    nCrys = NCRYSEB;
  } else {
    nCrys = NCRYSEE;
  }

  iZ = 1;
  if (_fedid <= 609)
    iZ = -1;

  dccMEM = ME::memFromDcc(_fedid);
  modules = ME::lmmodFromDcc(_fedid);
  nMod = modules.size();
}

//========================================================================
EcalTestPulseAnalyzer::~EcalTestPulseAnalyzer() {
  //========================================================================

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//========================================================================
void EcalTestPulseAnalyzer::beginJob() {
  //========================================================================

  // Define temporary file

  rootfile = resdir_;
  rootfile += "/TmpTreeTestPulseAnalyzer.root";

  outFile = new TFile(rootfile.c_str(), "RECREATE");

  for (unsigned int i = 0; i < nCrys; i++) {
    std::stringstream name;
    name << "Tree" << i;

    trees[i] = new TTree(name.str().c_str(), name.str().c_str());

    //List of branches

    trees[i]->Branch("iphi", &phi, "phi/I");
    trees[i]->Branch("ieta", &eta, "eta/I");
    trees[i]->Branch("side", &side, "side/I");
    trees[i]->Branch("dccID", &dccID, "dccID/I");
    trees[i]->Branch("towerID", &towerID, "towerID/I");
    trees[i]->Branch("channelID", &channelID, "channelID/I");
    trees[i]->Branch("event", &event, "event/I");
    trees[i]->Branch("apdGain", &apdGain, "apdGain/I");
    trees[i]->Branch("pnGain", &pnGain, "pnGain/I");
    trees[i]->Branch("apdAmpl", &apdAmpl, "apdAmpl/D");
    trees[i]->Branch("pnAmpl0", &pnAmpl0, "pnAmpl0/D");
    trees[i]->Branch("pnAmpl1", &pnAmpl1, "pnAmpl1/D");

    trees[i]->SetBranchAddress("ieta", &eta);
    trees[i]->SetBranchAddress("iphi", &phi);
    trees[i]->SetBranchAddress("side", &side);
    trees[i]->SetBranchAddress("dccID", &dccID);
    trees[i]->SetBranchAddress("towerID", &towerID);
    trees[i]->SetBranchAddress("channelID", &channelID);
    trees[i]->SetBranchAddress("event", &event);
    trees[i]->SetBranchAddress("apdGain", &apdGain);
    trees[i]->SetBranchAddress("pnGain", &pnGain);
    trees[i]->SetBranchAddress("apdAmpl", &apdAmpl);
    trees[i]->SetBranchAddress("pnAmpl0", &pnAmpl0);
    trees[i]->SetBranchAddress("pnAmpl1", &pnAmpl1);
  }

  // Initializations

  for (unsigned int j = 0; j < nCrys; j++) {
    iEta[j] = -1;
    iPhi[j] = -1;
    iModule[j] = 10;
    iTowerID[j] = -1;
    iChannelID[j] = -1;
    idccID[j] = -1;
    iside[j] = -1;
  }

  for (unsigned int j = 0; j < nMod; j++) {
    firstChanMod[j] = 0;
    isFirstChanModFilled[j] = 0;
  }

  // Define output results file name

  std::stringstream namefile;
  namefile << resdir_ << "/APDPN_TESTPULSE.root";
  resfile = namefile.str();

  // TP events counter
  TPEvents = 0;
}

//========================================================================
void EcalTestPulseAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c) {
  //========================================================================

  ++iEvent;

  // Retrieve DCC header
  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const EcalRawDataCollection* DCCHeader = nullptr;
  e.getByToken(rawDataToken_, pDCCHeader);
  if (!pDCCHeader.isValid()) {
    edm::LogError("nodata") << "Error! can't get the product  retrieving DCC header" << eventHeaderCollection_.c_str();
  } else {
    DCCHeader = pDCCHeader.product();
  }

  // retrieving crystal data from Event
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
  } else {
    e.getByToken(eeDigiToken_, pEEDigi);
    if (!pEEDigi.isValid()) {
      edm::LogError("nodata") << "Error! can't get the product retrieving EE crystal data " << digiCollection_.c_str();
    } else {
      EEDigi = pEEDigi.product();
    }
  }

  // Retrieve crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection> pPNDigi;
  const EcalPnDiodeDigiCollection* PNDigi = nullptr;
  e.getByToken(pnDiodeDigiToken_, pPNDigi);
  if (!pPNDigi.isValid()) {
    edm::LogError("nodata") << "Error! can't get the product " << digiCollection_.c_str();
  } else {
    PNDigi = pPNDigi.product();
  }

  // retrieving electronics mapping
  const auto& TheMapping = c.getData(mappingToken_);
  ;

  // ====================================
  // Decode Basic DCCHeader Information
  // ====================================

  for (EcalRawDataCollection::const_iterator headerItr = DCCHeader->begin(); headerItr != DCCHeader->end();
       ++headerItr) {
    int fed = headerItr->fedId();

    if (fed != _fedid && _fedid != -999)
      continue;

    runType = headerItr->getRunType();
    runNum = headerItr->getRunNumber();
    event = headerItr->getLV1();

    dccID = headerItr->getDccInTCCCommand();
    fedID = headerItr->fedId();

    if (600 + dccID != fedID)
      continue;

    // Cut on runType

    if (runType != EcalDCCHeaderBlock::TESTPULSE_MGPA && runType != EcalDCCHeaderBlock::TESTPULSE_GAP &&
        runType != EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM)
      return;
  }

  // Cut on fedID

  if (fedID != _fedid && _fedid != -999)
    return;

  // Count TP events
  TPEvents++;

  // ======================
  // Decode PN Information
  // ======================

  TPNFit* pnfit = new TPNFit();
  pnfit->init(_nsamplesPN, _firstsamplePN, _lastsamplePN);

  double chi2pn = 0;
  double ypnrange[50];
  double dsum = 0.;
  double dsum1 = 0.;
  double bl = 0.;
  double val_max = 0.;
  int samplemax = 0;
  unsigned int k;
  int pnG[50];
  int pngain = 0;

  std::map<int, std::vector<double> > allPNAmpl;
  std::map<int, std::vector<int> > allPNGain;

  for (EcalPnDiodeDigiCollection::const_iterator pnItr = PNDigi->begin(); pnItr != PNDigi->end();
       ++pnItr) {  // Loop on PNs

    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());

    bool isMemRelevant = false;
    for (unsigned int imem = 0; imem < dccMEM.size(); imem++) {
      if (pnDetId.iDCCId() == dccMEM[imem]) {
        isMemRelevant = true;
      }
    }

    // skip mem dcc without relevant data
    if (!isMemRelevant)
      continue;

    for (int samId = 0; samId < (*pnItr).size(); samId++) {  // Loop on PN samples
      pn[samId] = (*pnItr).sample(samId).adc();
      pnG[samId] = (*pnItr).sample(samId).gainId();

      if (pnG[samId] != 1)
        edm::LogVerbatim("EcalTestPulseAnalyzer") << "PN gain different from 1 for sample " << samId;
      if (samId == 0)
        pngain = pnG[samId];
      if (samId > 0)
        pngain = TMath::Max(pnG[samId], pngain);
    }

    for (dsum = 0., k = 0; k < _presamplePN; k++) {
      dsum += pn[k];
    }
    bl = dsum / ((double)_presamplePN);

    for (val_max = 0., k = 0; k < _nsamplesPN; k++) {
      ypnrange[k] = pn[k] - bl;

      if (ypnrange[k] > val_max) {
        val_max = ypnrange[k];
        samplemax = k;
      }
    }

    chi2pn = pnfit->doFit(samplemax, &ypnrange[0]);

    if (chi2pn == 101 || chi2pn == 102 || chi2pn == 103)
      pnAmpl = 0.;
    else
      pnAmpl = pnfit->getAmpl();

    allPNAmpl[pnDetId.iDCCId()].push_back(pnAmpl);
    allPNGain[pnDetId.iDCCId()].push_back(pngain);
  }

  // ===========================
  // Decode EBDigis Information
  // ===========================

  TSFit* pstpfit = new TSFit(_nsamples, 650);
  pstpfit->set_params(
      _nsamples, _niter, _presample, _samplemin, _samplemax, _timeofmax, _chi2max, _firstsample, _lastsample);
  pstpfit->init_errmat(10.);

  double chi2 = 0;
  double yrange[10];
  int adcgain = 0;
  int adcG[10];

  if (EBDigi) {
    for (EBDigiCollection::const_iterator digiItr = EBDigi->begin(); digiItr != EBDigi->end();
         ++digiItr) {  // Loop on EB crystals

      EBDetId id_crystal(digiItr->id());
      EBDataFrame df(*digiItr);

      int etaG = id_crystal.ieta();  // global
      int phiG = id_crystal.iphi();  // global

      int etaL;  // local
      int phiL;  // local
      std::pair<int, int> LocalCoord = MEEBGeom::localCoord(etaG, phiG);

      etaL = LocalCoord.first;
      phiL = LocalCoord.second;

      eta = etaG;
      phi = phiG;

      side = MEEBGeom::side(etaG, phiG);

      EcalElectronicsId elecid_crystal = TheMapping.getElectronicsId(id_crystal);

      towerID = elecid_crystal.towerId();
      int strip = elecid_crystal.stripId();
      int xtal = elecid_crystal.xtalId();
      channelID = 5 * (strip - 1) + xtal - 1;  // FIXME

      int module = MEEBGeom::lmmod(etaG, phiG);
      int iMod = module - 1;

      assert(module >= *min_element(modules.begin(), modules.end()) &&
             module <= *max_element(modules.begin(), modules.end()));

      std::pair<int, int> pnpair = MEEBGeom::pn(module);
      unsigned int MyPn0 = pnpair.first;
      unsigned int MyPn1 = pnpair.second;

      unsigned int channel = MEEBGeom::electronic_channel(etaL, phiL);

      if (isFirstChanModFilled[iMod] == 0) {
        firstChanMod[iMod] = channel;
        isFirstChanModFilled[iMod] = 1;
      }

      iEta[channel] = eta;
      iPhi[channel] = phi;
      iModule[channel] = module;
      iTowerID[channel] = towerID;
      iChannelID[channel] = channelID;
      idccID[channel] = dccID;
      iside[channel] = side;

      // get adc samples
      //====================

      for (unsigned int i = 0; i < (*digiItr).size(); ++i) {
        EcalMGPASample samp_crystal(df.sample(i));
        adc[i] = samp_crystal.adc();
        adcG[i] = samp_crystal.gainId();

        if (i == 0)
          adcgain = adcG[i];
        if (i > 0)
          adcgain = TMath::Max(adcG[i], adcgain);
      }
      // Remove pedestal
      //====================
      for (dsum = 0., dsum1 = 0., k = 0; k < _presample; k++) {
        dsum += adc[k];
        if (k < _presample - 1)
          dsum1 += adc[k];
      }

      bl = dsum / ((double)_presample);

      for (val_max = 0., k = 0; k < _nsamples; k++) {
        yrange[k] = adc[k] - bl;
        if (yrange[k] > val_max) {
          val_max = yrange[k];
        }
      }

      apdGain = adcgain;

      if (allPNAmpl[dccMEM[0]].size() > MyPn0)
        pnAmpl0 = allPNAmpl[dccMEM[0]][MyPn0];
      else
        pnAmpl0 = 0;
      if (allPNAmpl[dccMEM[0]].size() > MyPn1)
        pnAmpl1 = allPNAmpl[dccMEM[0]][MyPn1];
      else
        pnAmpl1 = 0;

      if (allPNGain[dccMEM[0]].size() > MyPn0)
        pnGain = allPNGain[dccMEM[0]][MyPn0];
      else
        pnGain = 0;

      // Perform the fit on apd samples
      //================================

      chi2 = pstpfit->fit_third_degree_polynomial(&yrange[0], ret_data);

      //Retrieve APD amplitude from fit
      //================================

      if (val_max > 100000. || chi2 < 0. || chi2 == 102) {
        apdAmpl = 0;
        apdTime = 0;

      } else {
        apdAmpl = ret_data[0];
        apdTime = ret_data[1];
      }

      trees[channel]->Fill();
    }

  } else {
    for (EEDigiCollection::const_iterator digiItr = EEDigi->begin(); digiItr != EEDigi->end();
         ++digiItr) {  // Loop on EE crystals

      EEDetId id_crystal(digiItr->id());
      EEDataFrame df(*digiItr);

      phi = id_crystal.ix();
      eta = id_crystal.iy();

      int iX = (phi - 1) / 5 + 1;
      int iY = (eta - 1) / 5 + 1;

      side = MEEEGeom::side(iX, iY, iZ);

      // Recover the TT id and the electronic crystal numbering from EcalElectronicsMapping

      EcalElectronicsId elecid_crystal = TheMapping.getElectronicsId(id_crystal);

      towerID = elecid_crystal.towerId();
      channelID = elecid_crystal.channelId() - 1;

      int module = MEEEGeom::lmmod(iX, iY);
      if (module >= 18 && side == 1)
        module += 2;  // Trick to fix endcap specificity
      int iMod = module - 1;

      assert(module >= *min_element(modules.begin(), modules.end()) &&
             module <= *max_element(modules.begin(), modules.end()));

      std::pair<int, int> pnpair = MEEEGeom::pn(module, _fedid);

      unsigned int MyPn0 = pnpair.first;
      unsigned int MyPn1 = pnpair.second;

      int hashedIndex = 100000 * eta + phi;

      if (channelMapEE.count(hashedIndex) == 0) {
        channelMapEE[hashedIndex] = channelIteratorEE;
        channelIteratorEE++;
      }

      unsigned int channel = channelMapEE[hashedIndex];

      if (isFirstChanModFilled[iMod] == 0) {
        firstChanMod[iMod] = channel;
        isFirstChanModFilled[iMod] = 1;
      }

      iEta[channel] = eta;
      iPhi[channel] = phi;
      iModule[channel] = module;
      iTowerID[channel] = towerID;
      iChannelID[channel] = channelID;
      idccID[channel] = dccID;
      iside[channel] = side;

      assert(channel < nCrys);

      // Get adc samples
      //====================

      for (unsigned int i = 0; i < (*digiItr).size(); ++i) {
        EcalMGPASample samp_crystal(df.sample(i));
        adc[i] = samp_crystal.adc();
        adcG[i] = samp_crystal.gainId();

        if (i == 0)
          adcgain = adcG[i];
        if (i > 0)
          adcgain = TMath::Max(adcG[i], adcgain);
      }

      // Remove pedestal
      //====================
      for (dsum = 0., dsum1 = 0., k = 0; k < _presample; k++) {
        dsum += adc[k];
        if (k < _presample - 1)
          dsum1 += adc[k];
      }

      bl = dsum / ((double)_presample);

      for (val_max = 0., k = 0; k < _nsamples; k++) {
        yrange[k] = adc[k] - bl;
        if (yrange[k] > val_max) {
          val_max = yrange[k];
        }
      }
      apdGain = adcgain;

      int dccMEMIndex = 0;
      if (side == 1)
        dccMEMIndex += 2;  // Trick to fix endcap specificity

      if (allPNAmpl[dccMEM[dccMEMIndex]].size() > MyPn0)
        pnAmpl0 = allPNAmpl[dccMEM[dccMEMIndex]][MyPn0];
      else
        pnAmpl0 = 0;
      if (allPNAmpl[dccMEM[dccMEMIndex + 1]].size() > MyPn1)
        pnAmpl1 = allPNAmpl[dccMEM[dccMEMIndex + 1]][MyPn1];
      else
        pnAmpl1 = 0;

      if (allPNGain[dccMEM[dccMEMIndex]].size() > MyPn0)
        pnGain = allPNGain[dccMEM[dccMEMIndex]][MyPn0];
      else
        pnGain = 0;

      // Perform the fit on apd samples
      //=================================

      chi2 = pstpfit->fit_third_degree_polynomial(&yrange[0], ret_data);

      //Retrieve APD amplitude from fit
      //=================================

      if (val_max > 100000. || chi2 < 0. || chi2 == 102) {
        apdAmpl = 0;
        apdTime = 0;

      } else {
        apdAmpl = ret_data[0];
        apdTime = ret_data[1];
      }

      trees[channel]->Fill();
    }
  }

}  // end of analyze

//========================================================================
void EcalTestPulseAnalyzer::endJob() {
  //========================================================================

  // Don't do anything if there is no events
  if (TPEvents == 0) {
    outFile->Close();

    // Remove temporary file

    std::stringstream del;
    del << "rm " << rootfile;
    system(del.str().c_str());

    edm::LogVerbatim("EcalTestPulseAnalyzer") << " No TP Events ";
    return;
  }

  edm::LogVerbatim("EcalTestPulseAnalyzer") << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
  edm::LogVerbatim("EcalTestPulseAnalyzer") << "\t+=+     Analyzing test pulse data: getting APD, PN  +=+";

  // Create output ntuples:

  //edm::LogVerbatim("EcalTestPulseAnalyzer")<< "TP Test Name File "<< resfile.c_str();

  resFile = new TFile(resfile.c_str(), "RECREATE");

  restrees = new TTree("TPAPD", "TPAPD");
  respntrees = new TTree("TPPN", "TPPN");

  restrees->Branch("iphi", &iphi, "iphi/I");
  restrees->Branch("ieta", &ieta, "ieta/I");
  restrees->Branch("dccID", &dccID, "dccID/I");
  restrees->Branch("side", &side, "side/I");
  restrees->Branch("towerID", &towerID, "towerID/I");
  restrees->Branch("channelID", &channelID, "channelID/I");
  restrees->Branch("moduleID", &moduleID, "moduleID/I");
  restrees->Branch("flag", &flag, "flag/I");
  restrees->Branch("gain", &gain, "gain/I");
  restrees->Branch("APD", &APD, "APD[6]/D");

  respntrees->Branch("pnID", &pnID, "pnID/I");
  respntrees->Branch("moduleID", &moduleID, "moduleID/I");
  respntrees->Branch("gain", &gain, "gain/I");
  respntrees->Branch("PN", &PN, "PN[6]/D");

  restrees->SetBranchAddress("iphi", &iphi);
  restrees->SetBranchAddress("ieta", &ieta);
  restrees->SetBranchAddress("dccID", &dccID);
  restrees->SetBranchAddress("side", &side);
  restrees->SetBranchAddress("towerID", &towerID);
  restrees->SetBranchAddress("channelID", &channelID);
  restrees->SetBranchAddress("moduleID", &moduleID);
  restrees->SetBranchAddress("flag", &flag);
  restrees->SetBranchAddress("gain", &gain);
  restrees->SetBranchAddress("APD", APD);

  respntrees->SetBranchAddress("pnID", &pnID);
  respntrees->SetBranchAddress("moduleID", &moduleID);
  respntrees->SetBranchAddress("gain", &gain);
  respntrees->SetBranchAddress("PN", PN);

  TMom* APDAnal[1700][10];
  TMom* PNAnal[9][2][10];

  for (unsigned int iMod = 0; iMod < nMod; iMod++) {
    for (unsigned int ich = 0; ich < 2; ich++) {
      for (unsigned int ig = 0; ig < nGainPN; ig++) {
        PNAnal[iMod][ich][ig] = new TMom();
      }
    }
  }

  for (unsigned int iCry = 0; iCry < nCrys; iCry++) {  // Loop on data trees (ie on cristals)

    for (unsigned int iG = 0; iG < nGainAPD; iG++) {
      APDAnal[iCry][iG] = new TMom();
    }

    // Define submodule and channel number inside the submodule (as Patrice)

    unsigned int iMod = iModule[iCry] - 1;

    moduleID = iMod + 1;
    if (moduleID >= 20)
      moduleID -= 2;  // Trick to fix endcap specificity

    for (Long64_t jentry = 0; jentry < trees[iCry]->GetEntriesFast(); jentry++) {
      // PN Means and RMS

      if (firstChanMod[iMod] == iCry) {
        PNAnal[iMod][0][pnGain]->addEntry(pnAmpl0);
        PNAnal[iMod][1][pnGain]->addEntry(pnAmpl1);
      }

      // APD means and RMS

      APDAnal[iCry][apdGain]->addEntry(apdAmpl);
    }

    if (trees[iCry]->GetEntries() < 10) {
      flag = -1;
      for (int j = 0; j < 6; j++) {
        APD[j] = 0.0;
      }
    } else
      flag = 1;

    iphi = iPhi[iCry];
    ieta = iEta[iCry];
    dccID = idccID[iCry];
    side = iside[iCry];
    towerID = iTowerID[iCry];
    channelID = iChannelID[iCry];

    for (unsigned int ig = 0; ig < nGainAPD; ig++) {
      APD[0] = APDAnal[iCry][ig]->getMean();
      APD[1] = APDAnal[iCry][ig]->getRMS();
      APD[2] = APDAnal[iCry][ig]->getM3();
      APD[3] = APDAnal[iCry][ig]->getNevt();
      APD[4] = APDAnal[iCry][ig]->getMin();
      APD[5] = APDAnal[iCry][ig]->getMax();
      gain = ig;

      // Fill APD tree

      restrees->Fill();
    }
  }

  // Get final results for PN and PN/PN

  for (unsigned int ig = 0; ig < nGainPN; ig++) {
    for (unsigned int iMod = 0; iMod < nMod; iMod++) {
      for (int ch = 0; ch < 2; ch++) {
        pnID = ch;
        moduleID = iMod;
        if (moduleID >= 20)
          moduleID -= 2;  // Trick to fix endcap specificity

        PN[0] = PNAnal[iMod][ch][ig]->getMean();
        PN[1] = PNAnal[iMod][ch][ig]->getRMS();
        PN[2] = PNAnal[iMod][ch][ig]->getM3();
        PN[3] = PNAnal[iMod][ch][ig]->getNevt();
        PN[4] = PNAnal[iMod][ch][ig]->getMin();
        PN[5] = PNAnal[iMod][ch][ig]->getMax();
        gain = ig;

        // Fill PN tree
        respntrees->Fill();
      }
    }
  }

  outFile->Close();

  // Remove temporary file

  std::stringstream del;
  del << "rm " << rootfile;
  system(del.str().c_str());

  // Save final results

  restrees->Write();
  respntrees->Write();
  resFile->Close();

  edm::LogVerbatim("EcalTestPulseAnalyzer") << "\t+=+    ...................................... done  +=+";
  edm::LogVerbatim("EcalTestPulseAnalyzer") << "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
}

DEFINE_FWK_MODULE(EcalTestPulseAnalyzer);
