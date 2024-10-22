// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecAlgos
// Class:      MahiDebugger
//
/**\class MahiDebugger MahiDebugger.cc RecoLocalCalo/HcalRecAlgos/plugins/MahiDebugger.cc

 Description: Tool to extract and store debugging information from the HBHE Reconstruction algorithm Mahi

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jay Mathew Lawhorn
//         Created:  Sat, 10 Feb 2018 10:02:38 GMT
//
//

// system include files
#include <utility>
#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>

#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"

//#include "RecoLocalCalo/HcalRecAlgos/test/SimpleHBHEPhase1AlgoDebug.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCorrectionFunctions.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/MahiFit.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/parseHBHEPhase1AlgoDescription.h"

//
// class declaration
//

class MahiDebugger : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MahiDebugger(const edm::ParameterSet&);
  ~MahiDebugger() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  // Special HB- correction
  float hbminusCorrectionFactor(const HcalDetId& cell, int runnum, float energy, bool isRealData) const;
  void endJob() override;

  // ----------member data ---------------------------

  // Python-configurables
  bool dynamicPed_;
  float ts4Thresh_;
  float chiSqSwitch_;

  bool applyTimeSlew_;
  HcalTimeSlew::BiasSetting slewFlavor_;
  double tsDelay1GeV_ = 0;

  bool calculateArrivalTime_;
  int timeAlgo_;
  float thEnergeticPulses_;
  float meanTime_;
  float timeSigmaHPD_;
  float timeSigmaSiPM_;

  std::vector<int> activeBXs_;

  int nMaxItersMin_;
  int nMaxItersNNLS_;

  float deltaChiSqThresh_;
  float nnlsThresh_;

  unsigned int bxSizeConf_;
  int bxOffsetConf_;

  //for pulse shapes
  HcalPulseShapes theHcalPulseShapes_;
  int cntsetPulseShape_;
  std::unique_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
  std::unique_ptr<ROOT::Math::Functor> pfunctor_;

  std::unique_ptr<MahiFit> mahi_;

  edm::EDGetTokenT<HBHEChannelInfoCollection> token_ChannelInfo_;
  const edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> tokDelay_;

  const HcalTimeSlew* hcalTimeSlewDelay;

  edm::Service<TFileService> FileService;
  TTree* outTree;

  int run;
  int evt;
  int ls;

  int nBxTrain;

  int ieta;
  int iphi;
  int depth;

  int nSamples;
  int soi;

  bool use8;

  float inTimeConst;
  float inDarkCurrent;
  float inPedAvg;
  float inGain;

  float inNoiseADC[10];
  float inNoiseDC[10];
  float inNoisePhoto[10];
  float inPedestal[10];

  float totalUCNoise[10];

  float mahiEnergy;  //SOI charge
  float chiSq;
  float arrivalTime;

  float ootEnergy[7];  //OOT charge
  float pedEnergy;     //pedestal charge

  float count[10];        //TS value 0-9
  float inputTS[10];      //input TS samples
  int inputTDC[10];       //input TS samples
  float itPulse[10];      //SOI pulse shape
  float ootPulse[7][10];  //OOT pulse shape
};

MahiDebugger::MahiDebugger(const edm::ParameterSet& iConfig)
    : dynamicPed_(iConfig.getParameter<bool>("dynamicPed")),
      ts4Thresh_(iConfig.getParameter<double>("ts4Thresh")),
      chiSqSwitch_(iConfig.getParameter<double>("chiSqSwitch")),
      applyTimeSlew_(iConfig.getParameter<bool>("applyTimeSlew")),
      calculateArrivalTime_(iConfig.getParameter<bool>("calculateArrivalTime")),
      timeAlgo_(iConfig.getParameter<int>("timeAlgo")),
      thEnergeticPulses_(iConfig.getParameter<double>("thEnergeticPulses")),
      meanTime_(iConfig.getParameter<double>("meanTime")),
      timeSigmaHPD_(iConfig.getParameter<double>("timeSigmaHPD")),
      timeSigmaSiPM_(iConfig.getParameter<double>("timeSigmaSiPM")),
      activeBXs_(iConfig.getParameter<std::vector<int>>("activeBXs")),
      nMaxItersMin_(iConfig.getParameter<int>("nMaxItersMin")),
      nMaxItersNNLS_(iConfig.getParameter<int>("nMaxItersNNLS")),
      deltaChiSqThresh_(iConfig.getParameter<double>("deltaChiSqThresh")),
      nnlsThresh_(iConfig.getParameter<double>("nnlsThresh")),
      tokDelay_(esConsumes<HcalTimeSlew, HcalTimeSlewRecord>(edm::ESInputTag("", "HBHE"))) {
  usesResource("TFileService");

  mahi_ = std::make_unique<MahiFit>();

  mahi_->setParameters(dynamicPed_,
                       ts4Thresh_,
                       chiSqSwitch_,
                       applyTimeSlew_,
                       HcalTimeSlew::Medium,
                       calculateArrivalTime_,
                       timeAlgo_,
                       thEnergeticPulses_,
                       meanTime_,
                       timeSigmaHPD_,
                       timeSigmaSiPM_,
                       activeBXs_,
                       nMaxItersMin_,
                       nMaxItersNNLS_,
                       deltaChiSqThresh_,
                       nnlsThresh_);

  token_ChannelInfo_ = consumes<HBHEChannelInfoCollection>(iConfig.getParameter<edm::InputTag>("recoLabel"));
}

MahiDebugger::~MahiDebugger() {}

//
// member functions
//

// ------------ method called for each event  ------------
void MahiDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  hcalTimeSlewDelay = &iSetup.getData(tokDelay_);

  run = iEvent.id().run();
  evt = iEvent.id().event();
  ls = iEvent.id().luminosityBlock();

  edm::EventBase const& eventbase = iEvent;
  nBxTrain = int(eventbase.bunchCrossing());

  Handle<HBHEChannelInfoCollection> hChannelInfo;
  iEvent.getByToken(token_ChannelInfo_, hChannelInfo);

  for (HBHEChannelInfoCollection::const_iterator iter = hChannelInfo->begin(); iter != hChannelInfo->end(); iter++) {
    const HBHEChannelInfo& hci(*iter);
    const HcalDetId detid = hci.id();

    ieta = detid.ieta();
    iphi = detid.iphi();
    depth = detid.depth();

    const bool isRealData = true;

    const MahiFit* mahi = mahi_.get();
    mahi_->setPulseShapeTemplate(
        hci.recoShape(), theHcalPulseShapes_, hci.hasTimeInfo(), hcalTimeSlewDelay, hci.nSamples(), hci.tsGain(0));
    MahiDebugInfo mdi;
    // initialize energies so that the values in the previous iteration are not stored
    mdi.mahiEnergy = 0;
    for (unsigned int ioot = 0; ioot < 7; ioot++)
      mdi.ootEnergy[ioot] = 0;
    mahi->phase1Debug(hci, mdi);

    nSamples = mdi.nSamples;
    soi = mdi.soi;

    inTimeConst = mdi.inTimeConst;
    inDarkCurrent = mdi.inDarkCurrent;
    inPedAvg = mdi.inPedAvg;
    inGain = mdi.inGain;

    use8 = mdi.use3;
    chiSq = mdi.chiSq;
    arrivalTime = mdi.arrivalTime;
    mahiEnergy = mdi.mahiEnergy;
    mahiEnergy *= hbminusCorrectionFactor(detid, run, mahiEnergy, isRealData);
    for (unsigned int ioot = 0; ioot < 7; ioot++) {
      ootEnergy[ioot] = mdi.ootEnergy[ioot];
      ootEnergy[ioot] *= hbminusCorrectionFactor(detid, run, ootEnergy[ioot], isRealData);
    }
    pedEnergy = mdi.pedEnergy;
    pedEnergy *= hbminusCorrectionFactor(detid, run, pedEnergy, isRealData);

    for (int i = 0; i < nSamples; i++) {
      count[i] = mdi.count[i];
      inputTS[i] = mdi.inputTS[i];
      inputTDC[i] = mdi.inputTDC[i];
      itPulse[i] = mdi.itPulse[i];
      for (unsigned int ioot = 0; ioot < 7; ioot++)
        ootPulse[ioot][i] = mdi.ootPulse[ioot][i];

      inNoiseADC[i] = mdi.inNoiseADC[i];
      inNoiseDC[i] = mdi.inNoiseDC[i];
      inNoisePhoto[i] = mdi.inNoisePhoto[i];
      inPedestal[i] = mdi.inPedestal[i];
      totalUCNoise[i] = mdi.totalUCNoise[i];
    }
    if (nSamples == 8) {
      count[8] = 8;
      count[9] = 9;
    }

    outTree->Fill();
  }
}

float MahiDebugger::hbminusCorrectionFactor(const HcalDetId& cell,
                                            int runnum,
                                            const float energy,
                                            const bool isRealData) const {
  float corr = 1.f;
  if (isRealData && runnum > 0)
    if (cell.subdet() == HcalBarrel) {
      const int ieta = cell.ieta();
      const int iphi = cell.iphi();
      corr = hbminus_special_ecorr(ieta, iphi, energy, runnum);
    }
  return corr;
}

// ------------ method called once each job just before starting event loop  ------------
void MahiDebugger::beginJob() {
  outTree = FileService->make<TTree>("HcalTree", "HcalTree");

  outTree->Branch("run", &run, "run/I");
  outTree->Branch("evt", &evt, "evt/I");
  outTree->Branch("ls", &ls, "ls/I");
  outTree->Branch("nBxTrain", &nBxTrain, "nBxTrain/I");

  outTree->Branch("ieta", &ieta, "ieta/I");
  outTree->Branch("iphi", &iphi, "iphi/I");
  outTree->Branch("depth", &depth, "depth/I");
  outTree->Branch("nSamples", &nSamples, "nSamples/I");
  outTree->Branch("soi", &soi, "soi/I");

  outTree->Branch("inTimeConst", &inTimeConst, "inTimeConst/F");
  outTree->Branch("inDarkCurrent", &inDarkCurrent, "inDarkCurrent/F");
  outTree->Branch("inPedAvg", &inPedAvg, "inPedAvg/F");
  outTree->Branch("inGain", &inGain, "inGain/F");

  outTree->Branch("use8", &use8, "use8/B");
  outTree->Branch("mahiEnergy", &mahiEnergy, "mahiEnergy/F");
  outTree->Branch("chiSq", &chiSq, "chiSq/F");
  outTree->Branch("arrivalTime", &arrivalTime, "arrivalTime/F");
  outTree->Branch("ootEnergy", &ootEnergy, "ootEnergy[7]/F");
  outTree->Branch("pedEnergy", &pedEnergy, "pedEnergy/F");
  outTree->Branch("count", &count, "count[10]/F");
  outTree->Branch("inputTS", &inputTS, "inputTS[10]/F");
  outTree->Branch("inputTDC", &inputTDC, "inputTDC[10]/I");
  outTree->Branch("itPulse", &itPulse, "itPulse[10]/F");
  outTree->Branch("ootPulse", &ootPulse, "ootPulse[7][10]/F");

  outTree->Branch("inNoiseADC", &inNoiseADC, "inNoiseADC[10]/F");
  outTree->Branch("inNoiseDC", &inNoiseDC, "inNoiseDC[10]/F");
  outTree->Branch("inNoisePhoto", &inNoisePhoto, "inNoisePhoto[10]/F");
  outTree->Branch("inPedestal", &inPedestal, "inPedestal[10]/F");
  outTree->Branch("totalUCNoise", &totalUCNoise, "totalUCNoise[10]/F");
}

// ------------ method called once each job just after ending the event loop  ------------
void MahiDebugger::endJob() {}

#define add_param_set(name)          \
  edm::ParameterSetDescription name; \
  name.setAllowAnything();           \
  desc.add<edm::ParameterSetDescription>(#name, name)

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MahiDebugger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recoLabel");
  desc.add<bool>("dynamicPed");
  desc.add<bool>("calculateArrivalTime");
  desc.add<int>("timeAlgo");
  desc.add<double>("thEnergeticPulse");
  desc.add<double>("ts4Thresh");
  desc.add<double>("chiSqSwitch");
  desc.add<bool>("applyTimeSlew");
  desc.add<double>("meanTime");
  desc.add<double>("timeSigmaHPD");
  desc.add<double>("timeSigmaSiPM");
  desc.add<std::vector<int>>("activeBXs");
  desc.add<int>("nMaxItersMin");
  desc.add<int>("nMaxItersNNLS");
  desc.add<double>("deltaChiSqThresh");
  desc.add<double>("nnlsThresh");

  //desc.add<std::string>("algoConfigClass");
  //add_param_set(algorithm);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MahiDebugger);
