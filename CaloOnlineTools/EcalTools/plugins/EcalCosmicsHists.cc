// -*- C++ -*-
//
// Package:   EcalCosmicsHists
// Class:     EcalCosmicsHists
//
/**\class EcalCosmicsHists EcalCosmicsHists.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

#include "CaloOnlineTools/EcalTools/plugins/EcalCosmicsHists.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include <vector>
#include "TLine.h"

using namespace cms;
using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalCosmicsHists::EcalCosmicsHists(const edm::ParameterSet& iConfig)
    : ecalRawDataColl_(iConfig.getParameter<edm::InputTag>("ecalRawDataColl")),
      ecalRecHitCollectionEB_(iConfig.getParameter<edm::InputTag>("ecalRecHitCollectionEB")),
      ecalRecHitCollectionEE_(iConfig.getParameter<edm::InputTag>("ecalRecHitCollectionEE")),
      barrelClusterCollection_(iConfig.getParameter<edm::InputTag>("barrelClusterCollection")),
      endcapClusterCollection_(iConfig.getParameter<edm::InputTag>("endcapClusterCollection")),
      l1GTReadoutRecTag_(iConfig.getUntrackedParameter<std::string>("L1GlobalReadoutRecord", "gtDigis")),
      l1GMTReadoutRecTag_(iConfig.getUntrackedParameter<std::string>("L1GlobalMuonReadoutRecord", "gtDigis")),
      runNum_(-1),
      histRangeMax_(iConfig.getUntrackedParameter<double>("histogramMaxRange", 1.8)),
      histRangeMin_(iConfig.getUntrackedParameter<double>("histogramMinRange", 0.0)),
      minTimingAmpEB_(iConfig.getUntrackedParameter<double>("MinTimingAmpEB", 0.100)),
      minTimingAmpEE_(iConfig.getUntrackedParameter<double>("MinTimingAmpEE", 0.100)),
      minRecHitAmpEB_(iConfig.getUntrackedParameter<double>("MinRecHitAmpEB", 0.027)),
      minRecHitAmpEE_(iConfig.getUntrackedParameter<double>("MinRecHitAmpEE", 0.18)),
      minHighEnergy_(iConfig.getUntrackedParameter<double>("MinHighEnergy", 2.0)),
      fileName_(iConfig.getUntrackedParameter<std::string>("fileName", std::string("ecalCosmicHists"))),
      runInFileName_(iConfig.getUntrackedParameter<bool>("runInFileName", true)),
      startTime_(iConfig.getUntrackedParameter<double>("TimeStampStart", 1215107133.)),
      runTimeLength_(iConfig.getUntrackedParameter<double>("TimeStampLength", 3.)),
      numTimingBins_(iConfig.getUntrackedParameter<int>("TimeStampBins", 1800)) {
  naiveEvtNum_ = 0;
  cosmicCounter_ = 0;
  cosmicCounterEB_ = 0;
  cosmicCounterEEM_ = 0;
  cosmicCounterEEP_ = 0;
  cosmicCounterTopBottom_ = 0;

  // TrackAssociator parameters
  edm::ParameterSet trkParameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  trackParameters_.loadParameters(trkParameters, iC);
  trackAssociator_.useDefaultPropagator();

  string title1 = "Seed Energy for All Feds; Seed Energy (GeV)";
  string name1 = "SeedEnergyAllFEDs";
  int numBins = 200;  //(int)round(histRangeMax_-histRangeMin_)+1;
  allFedsHist_ = new TH1F(name1.c_str(), title1.c_str(), numBins, histRangeMin_, histRangeMax_);

  fedMap_ = new EcalFedMap();
}

EcalCosmicsHists::~EcalCosmicsHists() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void EcalCosmicsHists::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  bool hasEndcapClusters = true;
  int ievt = iEvent.id().event();

  edm::Handle<reco::SuperClusterCollection> bscHandle;
  edm::Handle<reco::SuperClusterCollection> escHandle;

  naiveEvtNum_++;

  //LogDebug("EcalCosmicsHists")<< "  My Event: " << naiveEvtNum_ << " " << iEvent.id().run() << " " << iEvent.id().event() << " " << iEvent.time().value();
  //LogDebug("EcalCosmicsHists")<< "Timestamp: " << iEvent.id().run() << " " << iEvent.id().event() << " " << iEvent.time().value();

  // check DB payload
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  iSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  if (naiveEvtNum_ <= 1) {
    LogWarning("EcalCosmicsHists") << "Global EB ADC->GeV scale: " << agc->getEBValue() << " GeV/ADC count";
    LogWarning("EcalCosmicsHists") << "Global EE ADC->GeV scale: " << agc->getEEValue() << " GeV/ADC count";
  }
  //float adcEBconst = agc->getEBValue();
  //float adcEEconst = agc->getEEValue();

  //

  //===================TIMESTAMP INFORMTION=================================
  // Code added to get the time in seconds
  unsigned int timeStampLow = (0xFFFFFFFF & iEvent.time().value());
  unsigned int timeStampHigh = (iEvent.time().value() >> 32);
  double rawEventTime = (double)(timeStampHigh) + ((double)(timeStampLow) / 1000000.);
  double eventTime = rawEventTime - startTime_;  //Notice the subtraction of the "starttime"
  //std::cout << "Event Time " << eventTime << " High " <<timeStampHigh<< " low"<<timeStampLow <<" value " <<iEvent.time().value() << std::endl;
  //========================================================================

  iEvent.getByLabel(barrelClusterCollection_, bscHandle);
  if (!(bscHandle.isValid())) {
    LogWarning("EcalCosmicsHists") << barrelClusterCollection_ << " not available";
    return;
  }
  LogDebug("EcalCosmicsHists") << "event " << ievt;

  iEvent.getByLabel(endcapClusterCollection_, escHandle);
  if (!(escHandle.isValid())) {
    LogWarning("EcalCosmicsHists") << endcapClusterCollection_ << " not available";
    hasEndcapClusters = false;
    //return;
  }

  Handle<EcalRecHitCollection> hits;
  iEvent.getByLabel(ecalRecHitCollectionEB_, hits);
  if (!(hits.isValid())) {
    LogWarning("EcalCosmicsHists") << ecalRecHitCollectionEB_ << " not available";
    return;
  }
  Handle<EcalRecHitCollection> hitsEE;
  iEvent.getByLabel(ecalRecHitCollectionEE_, hitsEE);
  if (!(hitsEE.isValid())) {
    LogWarning("EcalCosmicsHists") << ecalRecHitCollectionEE_ << " not available";
    return;
  }

  Handle<EcalRawDataCollection> DCCHeaders;
  iEvent.getByLabel(ecalRawDataColl_, DCCHeaders);
  if (!DCCHeaders.isValid())
    LogWarning("EcalCosmicsHists") << "DCC headers not available";

  //make the bx histos right here
  //TODO: Right now we are filling histos for errors...
  int orbit = -100;
  int bx = -100;
  int runType = -100;

  for (EcalRawDataCollection::const_iterator headerItr = DCCHeaders->begin(); headerItr != DCCHeaders->end();
       ++headerItr) {
    headerItr->getEventSettings();
    int myorbit = headerItr->getOrbit();
    int mybx = headerItr->getBX();
    int myRunType = headerItr->getRunType();
    int FEDid = headerItr->fedId();
    TH2F* dccRuntypeHist = FEDsAndDCCRuntypeVsBxHists_[FEDid];
    if (dccRuntypeHist == nullptr) {
      initHists(FEDid);
      dccRuntypeHist = FEDsAndDCCRuntypeVsBxHists_[FEDid];
    }
    dccRuntypeHist->Fill(mybx, myRunType);

    if (bx == -100) {
      bx = mybx;
    } else if (bx != mybx) {
      LogWarning("EcalCosmicsHists") << "This header has a conflicting bx OTHERS were " << bx << " here " << mybx;
      dccBXErrorByFEDHist_->Fill(headerItr->fedId());
      if (bx != -100) {
        dccErrorVsBxHist_->Fill(bx, 0);
      }
    }

    if (runType == -100) {
      runType = myRunType;
    } else if (runType != myRunType) {
      LogWarning("EcalCosmicsHists") << "This header has a conflicting runType OTHERS were " << bx << " here " << mybx;
      dccRuntypeErrorByFEDHist_->Fill(headerItr->fedId());
      if (bx != -100)
        dccErrorVsBxHist_->Fill(bx, 2);
    }

    if (orbit == -100) {
      orbit = myorbit;
    } else if (orbit != myorbit) {
      LogWarning("EcalCosmicsHists") << "This header has a conflicting orbit; OTHERS were " << orbit << " here "
                                     << myorbit;
      dccOrbitErrorByFEDHist_->Fill(headerItr->fedId());
      if (bx != -100)
        dccErrorVsBxHist_->Fill(bx, 1);
    }
  }
  dccEventVsBxHist_->Fill(bx, runType);
  dccRuntypeHist_->Fill(runType);

  std::vector<bool> l1Triggers = determineTriggers(iEvent, iSetup);
  bool isEcalL1 = l1Triggers[4];
  bool isHCALL1 = l1Triggers[3];
  bool isRPCL1 = l1Triggers[2];
  bool isCSCL1 = l1Triggers[1];
  bool isDTL1 = l1Triggers[0];

  if (runNum_ == -1) {
    runNum_ = iEvent.id().run();
  }

  int numberOfCosmics = 0;
  int numberOfCosmicsEB = 0;
  int numberOfCosmicsEEP = 0;
  int numberOfCosmicsEEM = 0;
  int numberOfCosmicsTop = 0;
  int numberOfCosmicsBottom = 0;
  int numberOfHighEClusters = 0;
  //int eventnum = iEvent.id().event();
  std::vector<EBDetId> seeds;

  //++++++++++++++++++BEGIN LOOP OVER EB SUPERCLUSTERS+++++++++++++++++++++++++//

  const reco::SuperClusterCollection* clusterCollection_p = bscHandle.product();
  for (reco::SuperClusterCollection::const_iterator clus = clusterCollection_p->begin();
       clus != clusterCollection_p->end();
       ++clus) {
    double energy = clus->energy();
    double phi = clus->phi();
    double eta = clus->eta();
    double time = -1000.0;
    double ampli = 0.;
    double secondMin = 0.;
    double secondTime = -1000.;
    int numXtalsinCluster = 0;

    EBDetId maxDet;
    EBDetId secDet;

    numberofBCinSC_->Fill(clus->clustersSize());
    numberofBCinSCphi_->Fill(phi, clus->clustersSize());

    for (reco::CaloCluster_iterator bclus = (clus->clustersBegin()); bclus != (clus->clustersEnd()); ++bclus) {
      double cphi = (*bclus)->phi();
      double ceta = (*bclus)->eta();
      TrueBCOccupancy_->Fill(cphi, ceta);
      TrueBCOccupancyCoarse_->Fill(cphi, ceta);
    }

    std::vector<std::pair<DetId, float> > clusterDetIds = clus->hitsAndFractions();  //get these from the cluster
    for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin();
         detitr != clusterDetIds.end();
         ++detitr) {
      //Here I use the "find" on a digi collection... I have been warned...
      if ((*detitr).first.det() != DetId::Ecal) {
        std::cout << " det is " << (*detitr).first.det() << std::endl;
        continue;
      }
      if ((*detitr).first.subdetId() != EcalBarrel) {
        std::cout << " subdet is " << (*detitr).first.subdetId() << std::endl;
        continue;
      }
      EcalRecHitCollection::const_iterator thishit = hits->find((*detitr).first);
      if (thishit == hits->end())
        continue;
      //The checking above should no longer be needed...... as only those in the cluster would already have rechits..

      EcalRecHit myhit = (*thishit);

      double thisamp = myhit.energy();
      if (thisamp > minRecHitAmpEB_) {
        numXtalsinCluster++;
      }
      if (thisamp > secondMin) {
        secondMin = thisamp;
        secondTime = myhit.time();
        secDet = (EBDetId)(*detitr).first;
      }
      if (secondMin > ampli) {
        std::swap(ampli, secondMin);
        std::swap(time, secondTime);
        std::swap(maxDet, secDet);
      }
    }

    double fullnumXtalsinCluster = clusterDetIds.size();

    float E2 = ampli + secondMin;
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId((EBDetId)maxDet);
    int FEDid = 600 + elecId.dccId();

    numberOfCosmics++;
    numberOfCosmicsEB++;

    //Set some more values
    seeds.push_back(maxDet);
    int ieta = maxDet.ieta();
    int iphi = maxDet.iphi();
    int ietaSM = maxDet.ietaSM();
    int iphiSM = maxDet.iphiSM();

    int LM = ecalElectronicsMap_->getLMNumber(maxDet);  //FIX ME

    // top and bottom clusters
    if (iphi > 0 && iphi < 180) {
      numberOfCosmicsTop++;
    } else {
      numberOfCosmicsBottom++;
    }

    // fill the proper hist
    TH1F* uRecHist = FEDsAndHists_[FEDid];
    TH1F* E2uRecHist = FEDsAndE2Hists_[FEDid];
    TH1F* energyuRecHist = FEDsAndenergyHists_[FEDid];
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    TH1F* freqHist = FEDsAndFrequencyHists_[FEDid];
    TH1F* iphiProfileHist = FEDsAndiPhiProfileHists_[FEDid];
    TH1F* ietaProfileHist = FEDsAndiEtaProfileHists_[FEDid];
    TH2F* timingHistVsFreq = FEDsAndTimingVsFreqHists_[FEDid];
    TH2F* timingHistVsAmp = FEDsAndTimingVsAmpHists_[FEDid];
    TH2F* E2vsE1uRecHist = FEDsAndE2vsE1Hists_[FEDid];
    TH2F* energyvsE1uRecHist = FEDsAndenergyvsE1Hists_[FEDid];
    TH2F* occupHist = FEDsAndOccupancyHists_[FEDid];
    TH2F* timingHistVsPhi = FEDsAndTimingVsPhiHists_[FEDid];
    TH2F* timingHistVsModule = FEDsAndTimingVsModuleHists_[FEDid];

    if (uRecHist == nullptr) {
      initHists(FEDid);
      uRecHist = FEDsAndHists_[FEDid];
      E2uRecHist = FEDsAndE2Hists_[FEDid];
      energyuRecHist = FEDsAndenergyHists_[FEDid];
      timingHist = FEDsAndTimingHists_[FEDid];
      freqHist = FEDsAndFrequencyHists_[FEDid];
      timingHistVsFreq = FEDsAndTimingVsFreqHists_[FEDid];
      timingHistVsAmp = FEDsAndTimingVsAmpHists_[FEDid];
      iphiProfileHist = FEDsAndiPhiProfileHists_[FEDid];
      ietaProfileHist = FEDsAndiEtaProfileHists_[FEDid];
      E2vsE1uRecHist = FEDsAndE2vsE1Hists_[FEDid];
      energyvsE1uRecHist = FEDsAndenergyvsE1Hists_[FEDid];
      occupHist = FEDsAndOccupancyHists_[FEDid];
      timingHistVsPhi = FEDsAndTimingVsPhiHists_[FEDid];
      timingHistVsModule = FEDsAndTimingVsModuleHists_[FEDid];
    }

    uRecHist->Fill(ampli);
    E2uRecHist->Fill(E2);
    E2vsE1uRecHist->Fill(ampli, E2);
    energyuRecHist->Fill(energy);
    energyvsE1uRecHist->Fill(ampli, energy);
    allFedsHist_->Fill(ampli);
    allFedsE2Hist_->Fill(E2);
    allFedsenergyHist_->Fill(energy);
    allFedsenergyHighHist_->Fill(energy);
    allFedsE2vsE1Hist_->Fill(ampli, E2);
    allFedsenergyvsE1Hist_->Fill(ampli, energy);
    freqHist->Fill(naiveEvtNum_);
    iphiProfileHist->Fill(iphi);
    ietaProfileHist->Fill(ieta);

    allFedsFrequencyHist_->Fill(naiveEvtNum_);
    allFedsiPhiProfileHist_->Fill(iphi);
    allFedsiEtaProfileHist_->Fill(ieta);
    allOccupancy_->Fill(iphi, ieta);
    TrueOccupancy_->Fill(phi, eta);
    allOccupancyCoarse_->Fill(iphi, ieta);
    TrueOccupancyCoarse_->Fill(phi, eta);
    allFedsNumXtalsInClusterHist_->Fill(numXtalsinCluster);
    NumXtalsInClusterHist_->Fill(fullnumXtalsinCluster);
    numxtalsVsEnergy_->Fill(energy, numXtalsinCluster);
    numxtalsVsHighEnergy_->Fill(energy, numXtalsinCluster);

    //Fill the hists for the time stamp information
    allFedsFreqTimeVsPhiHist_->Fill(iphi, eventTime);
    allFedsFreqTimeVsPhiTTHist_->Fill(iphi, eventTime);
    allFedsFreqTimeVsEtaHist_->Fill(eventTime, ieta);
    allFedsFreqTimeVsEtaTTHist_->Fill(eventTime, ieta);
    //end time stamp hists

    occupHist->Fill(ietaSM, iphiSM);
    if (fullnumXtalsinCluster == 1) {
      allOccupancySingleXtal_->Fill(iphi, ieta);
      energySingleXtalHist_->Fill(energy);
    }

    // Exclusive trigger plots

    if (isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
      allOccupancyExclusiveECAL_->Fill(iphi, ieta);
      allOccupancyCoarseExclusiveECAL_->Fill(iphi, ieta);
      if (ampli > minTimingAmpEB_) {
        allFedsTimingHistECAL_->Fill(time);
        allFedsTimingPhiEtaHistECAL_->Fill(iphi, ieta, time);
        allFedsTimingTTHistECAL_->Fill(iphi, ieta, time);
        allFedsTimingLMHistECAL_->Fill(LM, time);
      }
      triggerExclusiveHist_->Fill(0);
    }

    if (!isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && isHCALL1) {
      allOccupancyExclusiveHCAL_->Fill(iphi, ieta);
      allOccupancyCoarseExclusiveHCAL_->Fill(iphi, ieta);
      if (ampli > minTimingAmpEB_) {
        allFedsTimingHistHCAL_->Fill(time);
        allFedsTimingPhiEtaHistHCAL_->Fill(iphi, ieta, time);
        allFedsTimingTTHistHCAL_->Fill(iphi, ieta, time);
        allFedsTimingLMHistHCAL_->Fill(LM, time);
      }
      triggerExclusiveHist_->Fill(1);
    }

    if (!isEcalL1 && isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
      allOccupancyExclusiveDT_->Fill(iphi, ieta);
      allOccupancyCoarseExclusiveDT_->Fill(iphi, ieta);
      if (ampli > minTimingAmpEB_) {
        allFedsTimingHistDT_->Fill(time);
        allFedsTimingPhiEtaHistDT_->Fill(iphi, ieta, time);
        allFedsTimingTTHistDT_->Fill(iphi, ieta, time);
        allFedsTimingLMHistDT_->Fill(LM, time);
      }
      triggerExclusiveHist_->Fill(2);
    }

    if (!isEcalL1 && !isDTL1 && isRPCL1 && !isCSCL1 && !isHCALL1) {
      allOccupancyExclusiveRPC_->Fill(iphi, ieta);
      allOccupancyCoarseExclusiveRPC_->Fill(iphi, ieta);
      if (ampli > minTimingAmpEB_) {
        allFedsTimingHistRPC_->Fill(time);
        allFedsTimingPhiEtaHistRPC_->Fill(iphi, ieta, time);
        allFedsTimingTTHistRPC_->Fill(iphi, ieta, time);
        allFedsTimingLMHistRPC_->Fill(LM, time);
      }
      triggerExclusiveHist_->Fill(3);
    }

    if (!isEcalL1 && !isDTL1 && !isRPCL1 && isCSCL1 && !isHCALL1) {
      allOccupancyExclusiveCSC_->Fill(iphi, ieta);
      allOccupancyCoarseExclusiveCSC_->Fill(iphi, ieta);
      if (ampli > minTimingAmpEB_) {
        allFedsTimingHistCSC_->Fill(time);
        allFedsTimingPhiEtaHistCSC_->Fill(iphi, ieta, time);
        allFedsTimingTTHistCSC_->Fill(iphi, ieta, time);
        allFedsTimingLMHistCSC_->Fill(LM, time);
      }
      triggerExclusiveHist_->Fill(4);
    }

    // Inclusive trigger plots

    if (isEcalL1) {
      triggerHist_->Fill(0);
      allOccupancyECAL_->Fill(iphi, ieta);
      allOccupancyCoarseECAL_->Fill(iphi, ieta);
    }
    if (isHCALL1) {
      triggerHist_->Fill(1);
      allOccupancyHCAL_->Fill(iphi, ieta);
      allOccupancyCoarseHCAL_->Fill(iphi, ieta);
    }
    if (isDTL1) {
      triggerHist_->Fill(2);
      allOccupancyDT_->Fill(iphi, ieta);
      allOccupancyCoarseDT_->Fill(iphi, ieta);
    }
    if (isRPCL1) {
      triggerHist_->Fill(3);
      allOccupancyRPC_->Fill(iphi, ieta);
      allOccupancyCoarseRPC_->Fill(iphi, ieta);
    }
    if (isCSCL1) {
      triggerHist_->Fill(4);
      allOccupancyCSC_->Fill(iphi, ieta);
      allOccupancyCoarseCSC_->Fill(iphi, ieta);
    }

    // Fill histo for Ecal+muon coincidence
    if (isEcalL1 && (isCSCL1 || isRPCL1 || isDTL1) && !isHCALL1)
      allFedsTimingHistEcalMuon_->Fill(time);

    if (ampli > minTimingAmpEB_) {
      timingHist->Fill(time);
      timingHistVsFreq->Fill(time, naiveEvtNum_);
      timingHistVsAmp->Fill(time, ampli);
      allFedsTimingHist_->Fill(time);
      allFedsTimingVsAmpHist_->Fill(time, ampli);
      allFedsTimingVsFreqHist_->Fill(time, naiveEvtNum_);
      timingHistVsPhi->Fill(time, iphiSM);
      timingHistVsModule->Fill(time, ietaSM);
      allFedsTimingPhiHist_->Fill(iphi, time);
      allFedsTimingPhiEtaHist_->Fill(iphi, ieta, time);
      allFedsTimingTTHist_->Fill(iphi, ieta, time);
      allFedsTimingLMHist_->Fill(LM, time);
      if (FEDid >= 610 && FEDid <= 627)
        allFedsTimingPhiEbmHist_->Fill(iphi, time);
      if (FEDid >= 628 && FEDid <= 645)
        allFedsTimingPhiEbpHist_->Fill(iphi, time);

      if (FEDid >= 610 && FEDid <= 627)
        allFedsTimingEbmHist_->Fill(time);
      if (FEDid >= 628 && FEDid <= 645)
        allFedsTimingEbpHist_->Fill(time);
      if (FEDid >= 613 && FEDid <= 616)
        allFedsTimingEbmTopHist_->Fill(time);
      if (FEDid >= 631 && FEDid <= 634)
        allFedsTimingEbpTopHist_->Fill(time);
      if (FEDid >= 622 && FEDid <= 625)
        allFedsTimingEbmBottomHist_->Fill(time);
      if (FEDid >= 640 && FEDid <= 643)
        allFedsTimingEbpBottomHist_->Fill(time);
    }

    // *** High Energy Clusters Analysis ** //

    if (energy > minHighEnergy_) {
      LogInfo("EcalCosmicsHists") << "High energy event " << iEvent.id().run() << " : " << iEvent.id().event() << " "
                                  << naiveEvtNum_ << " : " << energy << " " << numXtalsinCluster << " : " << iphi << " "
                                  << ieta << " : " << isEcalL1 << isHCALL1 << isDTL1 << isRPCL1 << isCSCL1;

      numberOfHighEClusters++;
      allOccupancyHighEnergy_->Fill(iphi, ieta);
      allOccupancyHighEnergyCoarse_->Fill(iphi, ieta);
      allFedsOccupancyHighEnergyHist_->Fill(iphi, ieta, energy);
      allFedsenergyOnlyHighHist_->Fill(energy);

      HighEnergy_2GeV_occuCoarse->Fill(iphi, ieta);
      HighEnergy_2GeV_occu3D->Fill(iphi, ieta, energy);

      HighEnergy_NumXtal->Fill(fullnumXtalsinCluster);
      HighEnergy_NumXtalFedId->Fill(FEDid, fullnumXtalsinCluster);
      HighEnergy_NumXtaliphi->Fill(iphi, fullnumXtalsinCluster);
      HighEnergy_energy3D->Fill(iphi, ieta, energy);
      HighEnergy_energyNumXtal->Fill(fullnumXtalsinCluster, energy);

      if (energy > 100.0) {
        LogInfo("EcalCosmicsHists") << "Very high energy event " << iEvent.id().run() << " : " << iEvent.id().event()
                                    << " " << naiveEvtNum_ << " : " << energy << " " << numXtalsinCluster << " : "
                                    << iphi << " " << ieta << " : " << isEcalL1 << isHCALL1 << isDTL1 << isRPCL1
                                    << isCSCL1;

        HighEnergy_100GeV_occuCoarse->Fill(iphi, ieta);
        HighEnergy_100GeV_occu3D->Fill(iphi, ieta, energy);
      }
    }

    // *** end of High Energy Clusters analysis *** //

  }  //++++++++++++++++++END LOOP OVER EB SUPERCLUSTERS+++++++++++++++++++++++//

  //+++++++++++++++++++LOOP OVER ENDCAP EE CLUSTERS++++++++++++++++++++//

  if (hasEndcapClusters) {
    clusterCollection_p = escHandle.product();
    for (reco::SuperClusterCollection::const_iterator clus = clusterCollection_p->begin();
         clus != clusterCollection_p->end();
         ++clus) {
      double energy = clus->energy();
      //double phi    = clus->phi();
      //double eta    = clus->eta();
      double time = -1000.0;
      double ampli = 0.;
      double secondMin = 0.;
      double secondTime = -1000.;
      int clusSize = clus->clustersSize();
      int numXtalsinCluster = 0;

      EEDetId maxDet;
      EEDetId secDet;
      //LogInfo("EcalCosmicsHists") << "Here is what we initialized the maxDet to: " << maxDet;

      //      for (reco::basicCluster_iterator bclus = (clus->clustersBegin()); bclus != (clus->clustersEnd()); ++bclus) {
      // 	//double cphi = (*bclus)->phi();
      // 	//double ceta = (*bclus)->eta();
      //         //TODO: extend histos to EE
      //         //TrueBCOccupancy_->Fill(cphi,ceta);
      //         //TrueBCOccupancyCoarse_->Fill(cphi,ceta);
      //       }

      std::vector<std::pair<DetId, float> > clusterDetIds = clus->hitsAndFractions();  //get these from the cluster
      for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin();
           detitr != clusterDetIds.end();
           ++detitr) {
        //LogInfo("EcalCosmicsHists") << " Here is the DetId inside the cluster: " << (EEDetId)(*detitr);
        //Here I use the "find" on a digi collection... I have been warned...

        if ((*detitr).first.det() != DetId::Ecal) {
          LogError("EcalCosmicsHists") << " det is " << (*detitr).first.det();
          continue;
        }
        if ((*detitr).first.subdetId() != EcalEndcap) {
          LogError("EcalCosmicsHists") << " subdet is " << (*detitr).first.subdetId();
          continue;
        }

        EcalRecHitCollection::const_iterator thishit = hitsEE->find((*detitr).first);

        if (thishit == hitsEE->end()) {
          LogInfo("EcalCosmicsHists") << " WARNING: EEDetId not found in the RecHit collection!";
          continue;
        }
        // The checking above should no longer be needed......
        // as only those in the cluster would already have rechits..

        EcalRecHit myhit = (*thishit);

        //LogInfo("EcalCosmicsHists") << " Found hit for DetId: " << (EEDetId)(*detitr);
        double thisamp = myhit.energy();
        if (thisamp > minRecHitAmpEE_) {
          numXtalsinCluster++;
        }
        if (thisamp > secondMin) {
          secondMin = thisamp;
          secondTime = myhit.time();
          secDet = (EEDetId)(*detitr).first;
        }
        if (secondMin > ampli) {
          std::swap(ampli, secondMin);
          std::swap(time, secondTime);
          std::swap(maxDet, secDet);
        }

        //LogInfo("EcalCosmicsHists") << "maxDetId is now: " << (EEDetId)(maxDet);
      }

      double fullnumXtalsinCluster = clusterDetIds.size();

      float E2 = ampli + secondMin;

      ecalElectronicsMap_->getElectronicsId((EEDetId)maxDet);
      //int FEDid = 600+elecId.dccId();

      //Set some more values
      //TODO: need to fix the seeds vector to be DetId or have another one for EE
      //seeds.push_back(maxDet);

      int ix = maxDet.ix();
      int iy = maxDet.iy();
      int iz = maxDet.zside();

      //      LogWarning("EcalCosmicsHists") << "EE cluster (x,y,z) : ( "
      //				     << ix << " , " << iy << " , " << iz
      //				     << " ) " << std::endl;

      if (!EEDetId::validDetId(ix, iy, iz)) {
        LogWarning("EcalCosmicsHists") << "INVALID EE DetId !!!" << endl;
        return;
      }

      numberOfCosmics++;
      if (iz < 0) {
        numberOfCosmicsEEM++;
      } else {
        numberOfCosmicsEEP++;
      }

      //int LM = ecalElectronicsMap_->getLMNumber(maxDet) ;//FIX ME

      //TODO: extend histos to EE
      //TH1F* uRecHist = FEDsAndHists_[FEDid];
      //TH1F* E2uRecHist = FEDsAndE2Hists_[FEDid];
      //TH1F* energyuRecHist = FEDsAndenergyHists_[FEDid];
      //TH1F* timingHist = FEDsAndTimingHists_[FEDid];
      //TH1F* freqHist = FEDsAndFrequencyHists_[FEDid];
      //TH1F* iphiProfileHist = FEDsAndiPhiProfileHists_[FEDid];
      //TH1F* ietaProfileHist = FEDsAndiEtaProfileHists_[FEDid];
      //TH2F* timingHistVsFreq = FEDsAndTimingVsFreqHists_[FEDid];
      //TH2F* timingHistVsAmp = FEDsAndTimingVsAmpHists_[FEDid];
      //TH2F* E2vsE1uRecHist = FEDsAndE2vsE1Hists_[FEDid];
      //TH2F* energyvsE1uRecHist = FEDsAndenergyvsE1Hists_[FEDid];
      //TH1F* numXtalInClusterHist = FEDsAndNumXtalsInClusterHists_[FEDid];
      //TH2F* occupHist = FEDsAndOccupancyHists_[FEDid];
      //TH2F* timingHistVsPhi = FEDsAndTimingVsPhiHists_[FEDid];
      //TH2F* timingHistVsModule = FEDsAndTimingVsModuleHists_[FEDid];
      //if(uRecHist==0)
      //{
      //  initHists(FEDid);
      //  uRecHist = FEDsAndHists_[FEDid];
      //  E2uRecHist = FEDsAndE2Hists_[FEDid];
      //  energyuRecHist = FEDsAndenergyHists_[FEDid];
      //  timingHist = FEDsAndTimingHists_[FEDid];
      //  freqHist = FEDsAndFrequencyHists_[FEDid];
      //  timingHistVsFreq = FEDsAndTimingVsFreqHists_[FEDid];
      //  timingHistVsAmp = FEDsAndTimingVsAmpHists_[FEDid];
      //  iphiProfileHist = FEDsAndiPhiProfileHists_[FEDid];
      //  ietaProfileHist = FEDsAndiEtaProfileHists_[FEDid];
      //  E2vsE1uRecHist = FEDsAndE2vsE1Hists_[FEDid];
      //  energyvsE1uRecHist = FEDsAndenergyvsE1Hists_[FEDid];
      //  numXtalInClusterHist = FEDsAndNumXtalsInClusterHists_[FEDid];
      //  occupHist = FEDsAndOccupancyHists_[FEDid];
      //  timingHistVsPhi = FEDsAndTimingVsPhiHists_[FEDid];
      //  timingHistVsModule = FEDsAndTimingVsModuleHists_[FEDid];
      //}
      //uRecHist->Fill(ampli);
      //E2uRecHist->Fill(E2);
      //E2vsE1uRecHist->Fill(ampli,E2);
      //energyuRecHist->Fill(energy);
      //energyvsE1uRecHist->Fill(ampli,energy);
      //allFedsHist_->Fill(ampli);

      if (iz < 0) {
        EEM_FedsSeedEnergyHist_->Fill(ampli);
        EEM_FedsenergyHist_->Fill(energy);
        EEM_FedsenergyHighHist_->Fill(energy);
        EEM_FedsE2Hist_->Fill(E2);
        EEM_FedsE2vsE1Hist_->Fill(ampli, E2);
        EEM_FedsenergyvsE1Hist_->Fill(ampli, energy);
        EEM_AllOccupancyCoarse_->Fill(ix - 0.5, iy - 0.5);
        EEM_AllOccupancy_->Fill(ix - 0.5, iy - 0.5);

        EEM_FedsNumXtalsInClusterHist_->Fill(numXtalsinCluster);
        EEM_NumXtalsInClusterHist_->Fill(fullnumXtalsinCluster);
        EEM_numxtalsVsEnergy_->Fill(energy, numXtalsinCluster);
        EEM_numxtalsVsHighEnergy_->Fill(energy, numXtalsinCluster);
        EEM_numberofBCinSC_->Fill(clusSize);

        if (fullnumXtalsinCluster == 1) {
          EEM_OccupancySingleXtal_->Fill(ix - 0.5, iy - 0.5);
          EEM_energySingleXtalHist_->Fill(energy);
        }

        if (ampli > minTimingAmpEE_) {
          EEM_FedsTimingHist_->Fill(time);
          EEM_FedsTimingVsAmpHist_->Fill(time, ampli);
          EEM_FedsTimingTTHist_->Fill(ix - 0.5, iy - 0.5, time);
        }

        // Exclusive trigger plots

        if (isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEM_OccupancyExclusiveECAL_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseExclusiveECAL_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEM_FedsTimingHistECAL_->Fill(time);
            EEM_FedsTimingTTHistECAL_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEM_triggerExclusiveHist_->Fill(0);
        }

        if (!isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && isHCALL1) {
          EEM_OccupancyExclusiveHCAL_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseExclusiveHCAL_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEM_FedsTimingHistHCAL_->Fill(time);
            EEM_FedsTimingTTHistHCAL_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEM_triggerExclusiveHist_->Fill(1);
        }

        if (!isEcalL1 && isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEM_OccupancyExclusiveDT_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseExclusiveDT_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEM_FedsTimingHistDT_->Fill(time);
            EEM_FedsTimingTTHistDT_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEM_triggerExclusiveHist_->Fill(2);
        }

        if (!isEcalL1 && !isDTL1 && isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEM_OccupancyExclusiveRPC_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseExclusiveRPC_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEM_FedsTimingHistRPC_->Fill(time);
            EEM_FedsTimingTTHistRPC_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEM_triggerExclusiveHist_->Fill(3);
        }

        if (!isEcalL1 && !isDTL1 && !isRPCL1 && isCSCL1 && !isHCALL1) {
          EEM_OccupancyExclusiveCSC_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseExclusiveCSC_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEM_FedsTimingHistCSC_->Fill(time);
            EEM_FedsTimingTTHistCSC_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEM_triggerExclusiveHist_->Fill(4);
        }

        // Inclusive trigger plots

        if (isEcalL1) {
          EEM_triggerHist_->Fill(0);
          EEM_OccupancyECAL_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseECAL_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isHCALL1) {
          EEM_triggerHist_->Fill(1);
          EEM_OccupancyHCAL_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseHCAL_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isDTL1) {
          EEM_triggerHist_->Fill(2);
          EEM_OccupancyDT_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseDT_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isRPCL1) {
          EEM_triggerHist_->Fill(3);
          EEM_OccupancyRPC_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseRPC_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isCSCL1) {
          EEM_triggerHist_->Fill(4);
          EEM_OccupancyCSC_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyCoarseCSC_->Fill(ix - 0.5, iy - 0.5);
        }

      } else {
        EEP_FedsSeedEnergyHist_->Fill(ampli);
        EEP_FedsenergyHist_->Fill(energy);
        EEP_FedsenergyHighHist_->Fill(energy);
        EEP_FedsE2Hist_->Fill(E2);
        EEP_FedsE2vsE1Hist_->Fill(ampli, E2);
        EEP_FedsenergyvsE1Hist_->Fill(ampli, energy);
        EEP_AllOccupancyCoarse_->Fill(ix - 0.5, iy - 0.5);
        EEP_AllOccupancy_->Fill(ix - 0.5, iy - 0.5);

        EEP_FedsNumXtalsInClusterHist_->Fill(numXtalsinCluster);
        EEP_NumXtalsInClusterHist_->Fill(fullnumXtalsinCluster);
        EEP_numxtalsVsEnergy_->Fill(energy, numXtalsinCluster);
        EEP_numxtalsVsHighEnergy_->Fill(energy, numXtalsinCluster);
        EEP_numberofBCinSC_->Fill(clusSize);

        if (fullnumXtalsinCluster == 1) {
          EEP_OccupancySingleXtal_->Fill(ix - 0.5, iy - 0.5);
          EEP_energySingleXtalHist_->Fill(energy);
        }

        if (ampli > minTimingAmpEE_) {
          EEP_FedsTimingHist_->Fill(time);
          EEP_FedsTimingVsAmpHist_->Fill(time, ampli);
          EEP_FedsTimingTTHist_->Fill(ix - 0.5, iy - 0.5, time);
        }

        // Exclusive trigger plots

        if (isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEP_OccupancyExclusiveECAL_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseExclusiveECAL_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEP_FedsTimingHistECAL_->Fill(time);
            EEP_FedsTimingTTHistECAL_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEP_triggerExclusiveHist_->Fill(0);
        }

        if (!isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && isHCALL1) {
          EEP_OccupancyExclusiveHCAL_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseExclusiveHCAL_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEP_FedsTimingHistHCAL_->Fill(time);
            EEP_FedsTimingTTHistHCAL_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEP_triggerExclusiveHist_->Fill(1);
        }

        if (!isEcalL1 && isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEP_OccupancyExclusiveDT_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseExclusiveDT_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEP_FedsTimingHistDT_->Fill(time);
            EEP_FedsTimingTTHistDT_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEP_triggerExclusiveHist_->Fill(2);
        }

        if (!isEcalL1 && !isDTL1 && isRPCL1 && !isCSCL1 && !isHCALL1) {
          EEP_OccupancyExclusiveRPC_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseExclusiveRPC_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEP_FedsTimingHistRPC_->Fill(time);
            EEP_FedsTimingTTHistRPC_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEP_triggerExclusiveHist_->Fill(3);
        }

        if (!isEcalL1 && !isDTL1 && !isRPCL1 && isCSCL1 && !isHCALL1) {
          EEP_OccupancyExclusiveCSC_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseExclusiveCSC_->Fill(ix - 0.5, iy - 0.5);
          if (ampli > minTimingAmpEE_) {
            EEP_FedsTimingHistCSC_->Fill(time);
            EEP_FedsTimingTTHistCSC_->Fill(ix - 0.5, iy - 0.5, time);
          }
          EEP_triggerExclusiveHist_->Fill(4);
        }

        // Inclusive trigger plots

        if (isEcalL1) {
          EEP_triggerHist_->Fill(0);
          EEP_OccupancyECAL_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseECAL_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isHCALL1) {
          EEP_triggerHist_->Fill(1);
          EEP_OccupancyHCAL_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseHCAL_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isDTL1) {
          EEP_triggerHist_->Fill(2);
          EEP_OccupancyDT_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseDT_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isRPCL1) {
          EEP_triggerHist_->Fill(3);
          EEP_OccupancyRPC_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseRPC_->Fill(ix - 0.5, iy - 0.5);
        }
        if (isCSCL1) {
          EEP_triggerHist_->Fill(4);
          EEP_OccupancyCSC_->Fill(ix - 0.5, iy - 0.5);
          EEP_OccupancyCoarseCSC_->Fill(ix - 0.5, iy - 0.5);
        }
      }

      // *** High Energy Clusters Analysis ** //

      if (energy > minHighEnergy_) {
        LogInfo("EcalCosmicsHists") << "High energy event in EE " << iEvent.id().run() << " : " << iEvent.id().event()
                                    << " " << naiveEvtNum_ << " : " << energy << " " << numXtalsinCluster << " : " << ix
                                    << " " << iy << " : " << isEcalL1 << isHCALL1 << isDTL1 << isRPCL1 << isCSCL1;

        //  numberOfHighEClusters++;
        //  allOccupancyHighEnergy_->Fill(iphi, ieta);
        //  allOccupancyHighEnergyCoarse_->Fill(iphi, ieta);
        //  allFedsOccupancyHighEnergyHist_->Fill(iphi,ieta,energy);
        if (iz < 0) {
          EEM_FedsenergyOnlyHighHist_->Fill(energy);
          EEM_OccupancyHighEnergy_->Fill(ix - 0.5, iy - 0.5);
          EEM_OccupancyHighEnergyCoarse_->Fill(ix - 0.5, iy - 0.5);
        } else {
          EEP_FedsenergyOnlyHighHist_->Fill(energy);
        }
        //  HighEnergy_2GeV_occuCoarse->Fill(iphi,ieta);
        //  HighEnergy_2GeV_occu3D->Fill(iphi,ieta,energy);

        //  HighEnergy_NumXtal->Fill(fullnumXtalsinCluster);
        //  HighEnergy_NumXtalFedId->Fill(FEDid,fullnumXtalsinCluster);
        //  HighEnergy_NumXtaliphi->Fill(iphi,fullnumXtalsinCluster);
        //  HighEnergy_energy3D->Fill(iphi,ieta,energy);
        //  HighEnergy_energyNumXtal->Fill(fullnumXtalsinCluster,energy);

        if (energy > 100.0) {
          LogInfo("EcalCosmicsHists") << "Very high energy event in EE " << iEvent.id().run() << " : "
                                      << iEvent.id().event() << " " << naiveEvtNum_ << " : " << energy << " "
                                      << numXtalsinCluster << " : " << ix << " " << iy << " : " << isEcalL1 << isHCALL1
                                      << isDTL1 << isRPCL1 << isCSCL1;
          //    HighEnergy_100GeV_occuCoarse->Fill(iphi,ieta);
          //    HighEnergy_100GeV_occu3D->Fill(iphi,ieta,energy);
        }
      }

      // *** end of High Energy Clusters analysis *** //

    }  //++++++++++++++++++END LOOP OVER EE SUPERCLUSTERS+++++++++++++++++++++++++++++++++++
  }

  HighEnergy_numClusHighEn->Fill(numberOfHighEClusters);
  HighEnergy_ratioClusters->Fill((double(numberOfHighEClusters)) / (double(numberOfCosmics)));

  numberofCosmicsHist_->Fill(numberOfCosmics);
  EEP_numberofCosmicsHist_->Fill(numberOfCosmicsEEP);
  EEM_numberofCosmicsHist_->Fill(numberOfCosmicsEEM);
  numberofCosmicsHistEB_->Fill(numberOfCosmicsEB);

  if (numberOfCosmics > 0) {
    cosmicCounter_++;
    numberofGoodEvtFreq_->Fill(naiveEvtNum_);
    allFedsFreqTimeHist_->Fill(eventTime);
    //This line will work in 21X!!
    //std::cout << " Orbit " <<   iEvent.orbitNumber() << " BX " << iEvent.bunchCrossing()<< std::endl;
    //std::cout << " BX " << iEvent.experimentType() << std::endl;
  }

  if (numberOfCosmicsEB > 0)
    cosmicCounterEB_++;
  if (numberOfCosmicsEEP > 0)
    cosmicCounterEEP_++;
  if (numberOfCosmicsEEM > 0)
    cosmicCounterEEM_++;

  if (numberOfCosmicsTop && numberOfCosmicsBottom) {
    cosmicCounterTopBottom_++;
    numberofCosmicsTopBottomHist_->Fill(numberOfCosmicsTop + numberOfCosmicsBottom);
  }

  // *** TrackAssociator *** //

  // get reco tracks
  edm::Handle<reco::TrackCollection> recoTracks;
  iEvent.getByLabel("cosmicMuons", recoTracks);

  if (recoTracks.isValid()) {
    //    LogWarning("EcalCosmicsHists") << "... Valid TrackAssociator recoTracks !!! " << recoTracks.product()->size();
    std::map<int, std::vector<DetId> > trackDetIdMap;
    int tracks = 0;
    for (reco::TrackCollection::const_iterator recoTrack = recoTracks->begin(); recoTrack != recoTracks->end();
         ++recoTrack) {
      if (fabs(recoTrack->d0()) > 70 || fabs(recoTrack->dz()) > 70)
        continue;
      if (recoTrack->numberOfValidHits() < 20)
        continue;

      //if (recoTrack->pt() < 2) continue; // skip low Pt tracks

      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *recoTrack, trackParameters_);

      //       edm::LogVerbatim("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " <<
      //    recoTrack->pt() << " , " << recoTrack->eta() << " , " << recoTrack->phi() ;
      //       edm::LogVerbatim("TrackAssociator") << "Ecal energy in crossed crystals based on RecHits: " <<
      //    info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      //       edm::LogVerbatim("TrackAssociator") << "Ecal energy in 3x3 crystals based on RecHits: " <<
      //    info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
      //       edm::LogVerbatim("TrackAssociator") << "Hcal energy in crossed towers based on RecHits: " <<
      //    info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
      //       edm::LogVerbatim("TrackAssociator") << "Hcal energy in 3x3 towers based on RecHits: " <<
      //    info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
      //       edm::LogVerbatim("TrackAssociator") << "Number of muon segment matches: " << info.numberOfSegments();

      //       std::cout << "\n-------------------------------------------------------\n Track (pt,eta,phi): " <<
      //    recoTrack->pt() << " , " << recoTrack->eta() << " , " << recoTrack->phi() << std::endl;
      //       std::cout << "Ecal energy in crossed crystals based on RecHits: " <<
      //    info.crossedEnergy(TrackDetMatchInfo::EcalRecHits) << std::endl;
      //       std::cout << "Ecal energy in 3x3 crystals based on RecHits: " <<
      //    info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1) << std::endl;
      //       std::cout << "Hcal energy in crossed towers based on RecHits: " <<
      //    info.crossedEnergy(TrackDetMatchInfo::HcalRecHits) << std::endl;
      //       std::cout << "Hcal energy in 3x3 towers based on RecHits: " <<
      //    info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1) << std::endl;

      for (unsigned int i = 0; i < info.crossedEcalIds.size(); i++) {
        // only checks for barrel
        if (info.crossedEcalIds[i].det() == DetId::Ecal && info.crossedEcalIds[i].subdetId() == 1) {
          EBDetId ebDetId(info.crossedEcalIds[i]);
          trackAssoc_muonsEcal_->Fill(ebDetId.iphi(), ebDetId.ieta());
          //std::cout << "Crossed iphi: " << ebDetId.iphi()
          //    << " ieta: " << ebDetId.ieta() << " : nCross " << info.crossedEcalIds.size() << std::endl;

          EcalRecHitCollection::const_iterator thishit = hits->find(ebDetId);
          if (thishit == hits->end())
            continue;

          //EcalRecHit myhit = (*thishit);
          //double thisamp = myhit.energy();
          //std::cout << " Crossed energy: " << thisamp << " : nCross " << info.crossedEcalIds.size() << std::endl;
        }
      }

      //edm::LogVerbatim("TrackAssociator") << " crossedEcalIds size: " << info.crossedEcalIds.size()
      //				  << " crossedEcalRecHits size: " << info.crossedEcalRecHits.size();
      numberofCrossedEcalIdsHist_->Fill(info.crossedEcalIds.size());
      tracks++;
      if (!info.crossedEcalIds.empty())
        trackDetIdMap.insert(std::pair<int, std::vector<DetId> >(tracks, info.crossedEcalIds));
    }

    // Now to match recoTracks with cosmic clusters

    int numAssocTracks = 0;
    int numAssocClusters = 0;
    edm::LogVerbatim("TrackAssociator") << "Matching cosmic clusters to tracks...";
    int numSeeds = seeds.size();
    int numTracks = trackDetIdMap.size();
    while (!seeds.empty() && !trackDetIdMap.empty()) {
      double bestDr = 1000;
      double bestDPhi = 1000;
      double bestDEta = 1000;
      double bestEtaTrack = 1000;
      double bestEtaSeed = 1000;
      double bestPhiTrack = 1000;
      double bestPhiSeed = 1000;
      EBDetId bestTrackDet;
      EBDetId bestSeed;
      int bestTrack = -1;
      std::map<EBDetId, EBDetId> trackDetIdToSeedMap;

      //edm::LogVerbatim("TrackAssociator") << "NumTracks:" << trackDetIdMap.size() << " numClusters:" << seeds.size();

      for (std::vector<EBDetId>::const_iterator seedItr = seeds.begin(); seedItr != seeds.end(); ++seedItr) {
        for (std::map<int, std::vector<DetId> >::const_iterator mapItr = trackDetIdMap.begin();
             mapItr != trackDetIdMap.end();
             ++mapItr) {
          for (unsigned int i = 0; i < mapItr->second.size(); i++) {
            // only checks for barrel
            if (mapItr->second[i].det() == DetId::Ecal && mapItr->second[i].subdetId() == 1) {
              EBDetId ebDet = (mapItr->second[i]);
              double seedEta = seedItr->ieta();
              double deta = ebDet.ieta() - seedEta;
              if (seedEta * ebDet.ieta() < 0)
                deta > 0 ? (deta = deta - 1.) : (deta = deta + 1.);
              double dR;
              double dphi = ebDet.iphi() - seedItr->iphi();
              if (abs(dphi) > 180)
                dphi > 0 ? (dphi = 360 - dphi) : (dphi = -360 - dphi);
              dR = sqrt(deta * deta + dphi * dphi);
              if (dR < bestDr) {
                bestDr = dR;
                bestDPhi = dphi;
                bestDEta = deta;
                bestTrackDet = mapItr->second[i];
                bestTrack = mapItr->first;
                bestSeed = (*seedItr);
                bestEtaTrack = ebDet.ieta();
                bestEtaSeed = seedEta;
                bestPhiTrack = ebDet.iphi();
                bestPhiSeed = seedItr->iphi();
              }
            }
          }
        }
      }
      if (bestDr < 1000) {
        //edm::LogVerbatim("TrackAssociator") << "Best deltaR from matched DetId's to cluster:" << bestDr;
        deltaRHist_->Fill(bestDr);
        deltaPhiHist_->Fill(bestDPhi);
        deltaEtaHist_->Fill(bestDEta);
        deltaEtaDeltaPhiHist_->Fill(bestDEta, bestDPhi);
        seedTrackEtaHist_->Fill(bestEtaSeed, bestEtaTrack);
        seedTrackPhiHist_->Fill(bestPhiSeed, bestPhiTrack);
        seeds.erase(find(seeds.begin(), seeds.end(), bestSeed));
        trackDetIdMap.erase(trackDetIdMap.find(bestTrack));
        trackDetIdToSeedMap[bestTrackDet] = bestSeed;
        numAssocTracks++;
        numAssocClusters++;

        // for high energy analysis
        if (bestDPhi < 1.5 && bestDEta < 1.8) {
          //edm::LogVerbatim("TrackAssociator") << "Best seed ieta and iphi: ("
          //					<< bestSeed.ieta() << ", " << bestSeed.iphi() << ") ";
          //check if the bestSeed is a High Energy one
          EcalRecHitCollection::const_iterator Ecalhit = hits->find(bestSeed);
          if (Ecalhit == hits->end()) {
            continue;
          }
          double EcalhitEnergy = Ecalhit->energy();
          //edm::LogVerbatim("TrackAssociator") << "Best seed energy: " << EcalhitEnergy ;
          HighEnergy_bestSeed->Fill(EcalhitEnergy);
          HighEnergy_bestSeedOccupancy->Fill(bestSeed.iphi(), bestSeed.ieta());
        }
      } else {
        edm::LogVerbatim("TrackAssociator") << "could not match cluster seed to track." << bestDr;
        break;  // no match found
      }
    }
    if (numSeeds > 0 && numTracks > 0) {
      ratioAssocClustersHist_->AddBinContent(1, numAssocClusters);
      ratioAssocClustersHist_->AddBinContent(2, numSeeds);
    }
    if (numTracks > 0) {
      ratioAssocTracksHist_->AddBinContent(1, numAssocTracks);
      ratioAssocTracksHist_->AddBinContent(2, numTracks);
      numberofCosmicsWTrackHist_->Fill(numSeeds);
    }
  } else {
    //    LogWarning("EcalCosmicsHists") << "!!! No TrackAssociator recoTracks !!!";
  }

  // Study on Tracks for High Energy events
  edm::Handle<reco::TrackCollection> recoTracksBarrel;
  iEvent.getByLabel("cosmicMuonsBarrelOnly", recoTracksBarrel);
  if (!recoTracksBarrel.isValid()) {
    //edm::LogWarning("EcalCosmicsHists") << "RecoTracksBarrel not valid!! " ;
  } else {
    //edm::LogWarning("EcalCosmicsHists") << "Number of barrel reco tracks found in the event: " << recoTracksBarrel->size() ;
    HighEnergy_numRecoTrackBarrel->Fill(recoTracksBarrel->size());
    for (reco::TrackCollection::const_iterator recoTrack = recoTracksBarrel->begin();
         recoTrack != recoTracksBarrel->end();
         ++recoTrack) {
      //edm::LogWarning("EcalCosmicsHists") << "Track (pt,eta,phi): " << recoTrack->pt() << " , " << recoTrack->eta() << " , " << recoTrack->phi() ;
      //edm::LogWarning("EcalCosmicsHists") << "Track innermost hit: " << recoTrack->innerPosition().phi() ;
    }

    for (reco::SuperClusterCollection::const_iterator clus = clusterCollection_p->begin();
         clus != clusterCollection_p->end();
         ++clus) {
      double energy = clus->energy();
      double phi = clus->phi();
      double eta = clus->eta();

      if (recoTracksBarrel->empty())
        HighEnergy_0tracks_occu3D->Fill(phi, eta, energy);
      if (recoTracksBarrel->size() == 1)
        HighEnergy_1tracks_occu3D->Fill(phi, eta, energy);
      if (recoTracksBarrel->size() == 2)
        HighEnergy_2tracks_occu3D->Fill(phi, eta, energy);

      std::vector<std::pair<DetId, float> > clusterDetIds = clus->hitsAndFractions();  //get these from the cluster
      for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterDetIds.begin();
           detitr != clusterDetIds.end();
           ++detitr) {
        if ((*detitr).first.det() != DetId::Ecal) {
          continue;
        }
        if ((*detitr).first.subdetId() != EcalBarrel) {
          continue;
        }
        EcalRecHitCollection::const_iterator thishit = hits->find((*detitr).first);
        if (thishit == hits->end()) {
          continue;
        }

        double rechitenergy = thishit->energy();
        int ieta = ((EBDetId)(*detitr).first).ieta();
        int iphi = ((EBDetId)(*detitr).first).iphi();
        if (rechitenergy > minRecHitAmpEB_) {
          if (recoTracksBarrel->empty())
            HighEnergy_0tracks_occu3DXtal->Fill(iphi, ieta, rechitenergy);
          if (recoTracksBarrel->size() == 1)
            HighEnergy_1tracks_occu3DXtal->Fill(iphi, ieta, rechitenergy);
          if (recoTracksBarrel->size() == 2)
            HighEnergy_2tracks_occu3DXtal->Fill(iphi, ieta, rechitenergy);

          if (rechitenergy > 10)
            edm::LogWarning("EcalCosmicsHists") << "!!!!! Crystal with energy " << rechitenergy << " at (ieta,iphi) ("
                                                << ieta << " ," << iphi << "); Id: " << (*detitr).first.det();
        }
      }
    }

    // look at angle between 2 recoTracks
    if (recoTracksBarrel->size() == 2) {
      reco::TrackCollection::const_iterator Track1 = recoTracksBarrel->begin();
      reco::TrackCollection::const_iterator Track2 = (recoTracksBarrel->begin()) + 1;
      float angle = (acos(sin(Track1->theta()) * cos(Track1->phi()) * sin(Track2->theta()) * cos(Track2->phi()) +
                          sin(Track1->theta()) * sin(Track1->phi()) * sin(Track2->theta()) * sin(Track2->phi()) +
                          cos(Track1->theta()) * cos(Track2->theta()))) /
                    0.017453292519943;
      //edm::LogWarning("EcalCosmicsHists") << "Tracks angle: " << angle;
      HighEnergy_TracksAngle->Fill(angle);
      if ((Track1->innerPosition().phi()) > 0 && (Track2->innerPosition().phi()) < 0) {
        //edm::LogWarning("EcalCosmicsHists") << "Top-bottom tracks";
        HighEnergy_TracksAngleTopBottom->Fill(angle);
      }
    }
  }

  // *** end of TrackAssociator code *** //

  // *** HCAL RecHits code *** //

  //hcal rechits
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel("hbhereco", hbhe);

  edm::Handle<HFRecHitCollection> hfrh;
  iEvent.getByLabel("hfreco", hfrh);

  edm::Handle<HORecHitCollection> horh;
  iEvent.getByLabel("horeco", horh);

  if (hbhe.isValid()) {
    //    LogInfo("EcalCosmicHists") << "event " << ievt << " HBHE RecHits collection size " << hbhe->size();
    const HBHERecHitCollection hbheHit = *(hbhe.product());
    for (HBHERecHitCollection::const_iterator hhit = hbheHit.begin(); hhit != hbheHit.end(); hhit++) {
      //      if (hhit->energy() > 0.6){
      hcalEnergy_HBHE_->Fill(hhit->energy());
      //      }
    }
  } else {
    //    LogWarning("EcalCosmicHists") << " HBHE RecHits **NOT** VALID!! " << endl;
  }

  if (hfrh.isValid()) {
    //    LogInfo("EcalCosmicHists") << "event " << ievt << " HF RecHits collection size " << hfrh->size();
    const HFRecHitCollection hfHit = *(hfrh.product());
    for (HFRecHitCollection::const_iterator hhit = hfHit.begin(); hhit != hfHit.end(); hhit++) {
      hcalEnergy_HF_->Fill(hhit->energy());
    }
  } else {
    //    LogWarning("EcalCosmicHists") << " HF RecHits **NOT** VALID!! " << endl;
  }

  if (horh.isValid()) {
    //    LogInfo("EcalCosmicHists") << "event " << ievt << " HO RecHits collection size " << horh->size();
    const HORecHitCollection hoHit = *(horh.product());
    for (HORecHitCollection::const_iterator hhit = hoHit.begin(); hhit != hoHit.end(); hhit++) {
      //     if (hhit->energy() > 0.6){
      hcalEnergy_HO_->Fill(hhit->energy());
      //     }
    }
  } else {
    //    LogWarning("EcalCosmicHists") << " HO RecHits **NOT** VALID!! " << endl;
  }

  // *** end of HCAL code *** //
}

std::vector<bool> EcalCosmicsHists::determineTriggers(const edm::Event& iEvent, const edm::EventSetup& eventSetup) {
  std::vector<bool> l1Triggers;  //DT,CSC,RPC,HCAL,ECAL
                                 //0 , 1 , 2 , 3  , 4
  l1Triggers.reserve(5);

        for (int i = 0; i < 5; i++)
    l1Triggers.push_back(false);

  // get the GMTReadoutCollection
  Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  iEvent.getByLabel(l1GMTReadoutRecTag_, gmtrc_handle);
  L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
  if (!(gmtrc_handle.isValid())) {
    LogWarning("EcalCosmicsHists") << "l1MuGMTReadoutCollection"
                                   << " not available";
    return l1Triggers;
  }
  // get hold of L1GlobalReadoutRecord
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  iEvent.getByLabel(l1GTReadoutRecTag_, L1GTRR);

  //Ecal
  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  eventSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);
  const L1GtTriggerMenu* menu = menuRcd.product();
  edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  iEvent.getByLabel(edm::InputTag("gtDigis"), gtRecord);
  // Get dWord after masking disabled bits
  const DecisionWord dWord = gtRecord->decisionWord();

  bool l1SingleEG1 = menu->gtAlgorithmResult("L1_SingleEG1", dWord);
  bool l1SingleEG5 = menu->gtAlgorithmResult("L1_SingleEG5", dWord);
  bool l1SingleEG8 = menu->gtAlgorithmResult("L1_SingleEG8", dWord);
  bool l1SingleEG10 = menu->gtAlgorithmResult("L1_SingleEG10", dWord);
  bool l1SingleEG12 = menu->gtAlgorithmResult("L1_SingleEG12", dWord);
  bool l1SingleEG15 = menu->gtAlgorithmResult("L1_SingleEG15", dWord);
  bool l1SingleEG20 = menu->gtAlgorithmResult("L1_SingleEG20", dWord);
  bool l1SingleEG25 = menu->gtAlgorithmResult("L1_SingleEG25", dWord);
  bool l1DoubleNoIsoEGBTBtight = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_tight", dWord);
  bool l1DoubleNoIsoEGBTBloose = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_loose ", dWord);
  bool l1DoubleNoIsoEGTopBottom = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottom", dWord);
  bool l1DoubleNoIsoEGTopBottomCen = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen", dWord);
  bool l1DoubleNoIsoEGTopBottomCen2 = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen2", dWord);
  bool l1DoubleNoIsoEGTopBottomCenVert = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCenVert", dWord);

  l1Triggers[4] = l1SingleEG1 || l1SingleEG5 || l1SingleEG8 || l1SingleEG10 || l1SingleEG12 || l1SingleEG15 ||
                  l1SingleEG20 || l1SingleEG25 || l1DoubleNoIsoEGBTBtight || l1DoubleNoIsoEGBTBloose ||
                  l1DoubleNoIsoEGTopBottom || l1DoubleNoIsoEGTopBottomCen || l1DoubleNoIsoEGTopBottomCen2 ||
                  l1DoubleNoIsoEGTopBottomCenVert;

  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
  for (igmtrr = gmt_records.begin(); igmtrr != gmt_records.end(); igmtrr++) {
    std::vector<L1MuRegionalCand>::const_iterator iter1;
    std::vector<L1MuRegionalCand> rmc;

    //DT triggers
    int idt = 0;
    rmc = igmtrr->getDTBXCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        idt++;
      }
    }
    //if(idt>0) std::cout << "Found " << idt << " valid DT candidates in bx wrt. L1A = "
    //  << igmtrr->getBxInEvent() << std::endl;
    if (igmtrr->getBxInEvent() == 0 && idt > 0)
      l1Triggers[0] = true;

    //RPC triggers
    int irpcb = 0;
    rmc = igmtrr->getBrlRPCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        irpcb++;
      }
    }
    //if(irpcb>0) std::cout << "Found " << irpcb << " valid RPC candidates in bx wrt. L1A = "
    //  << igmtrr->getBxInEvent() << std::endl;
    if (igmtrr->getBxInEvent() == 0 && irpcb > 0)
      l1Triggers[2] = true;

    //CSC Triggers
    int icsc = 0;
    rmc = igmtrr->getCSCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        icsc++;
      }
    }
    //if(icsc>0) std::cout << "Found " << icsc << " valid CSC candidates in bx wrt. L1A = "
    //  << igmtrr->getBxInEvent() << std::endl;
    if (igmtrr->getBxInEvent() == 0 && icsc > 0)
      l1Triggers[1] = true;
  }

  L1GlobalTriggerReadoutRecord const* gtrr = L1GTRR.product();

  for (int ibx = -1; ibx <= 1; ibx++) {
    bool hcal_top = false;
    bool hcal_bot = false;
    const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
    std::vector<int> valid_phi;
    if ((psb.aData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(4) >> 10) & 0x1f);
    }
    if ((psb.bData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(4) >> 10) & 0x1f);
    }
    if ((psb.aData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(5) >> 10) & 0x1f);
    }
    if ((psb.bData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(5) >> 10) & 0x1f);
    }
    std::vector<int>::const_iterator iphi;
    for (iphi = valid_phi.begin(); iphi != valid_phi.end(); iphi++) {
      //std::cout << "Found HCAL mip with phi=" << *iphi << " in bx wrt. L1A = " << ibx << std::endl;
      if (*iphi < 9)
        hcal_top = true;
      if (*iphi > 8)
        hcal_bot = true;
    }
    if (ibx == 0 && hcal_top && hcal_bot)
      l1Triggers[3] = true;
  }

  edm::LogInfo("EcalCosmicsHists") << "**** Trigger SourceSource ****";
  if (l1Triggers[0])
    edm::LogInfo("EcalCosmicsHists") << "DT";
  if (l1Triggers[2])
    edm::LogInfo("EcalCosmicsHists") << "RPC";
  if (l1Triggers[1])
    edm::LogInfo("EcalCosmicsHists") << "CSC";
  if (l1Triggers[3])
    edm::LogInfo("EcalCosmicsHists") << "HCAL";
  if (l1Triggers[4])
    edm::LogInfo("EcalCosmicsHists") << "ECAL";
  edm::LogInfo("EcalCosmicsHists") << "************************";

  return l1Triggers;
}

// insert the hist map into the map keyed by FED number
void EcalCosmicsHists::initHists(int FED) {
  using namespace std;

  string FEDid = intToString(FED);
  string title1 = "Energy of Seed Crystal ";
  title1.append(fedMap_->getSliceFromFed(FED));
  title1.append(";Seed Energy (GeV);Number of Cosmics");
  string name1 = "SeedEnergyFED";
  name1.append(intToString(FED));
  int numBins = 200;  //(int)round(histRangeMax_-histRangeMin_)+1;
  TH1F* hist = new TH1F(name1.c_str(), title1.c_str(), numBins, histRangeMin_, histRangeMax_);
  FEDsAndHists_[FED] = hist;
  FEDsAndHists_[FED]->SetDirectory(nullptr);

  TH1F* E2hist = new TH1F(Form("E2_FED_%d", FED), Form("E2_FED_%d", FED), numBins, histRangeMin_, histRangeMax_);
  FEDsAndE2Hists_[FED] = E2hist;
  FEDsAndE2Hists_[FED]->SetDirectory(nullptr);

  TH1F* energyhist =
      new TH1F(Form("Energy_FED_%d", FED), Form("Energy_FED_%d", FED), numBins, histRangeMin_, histRangeMax_);
  FEDsAndenergyHists_[FED] = energyhist;
  FEDsAndenergyHists_[FED]->SetDirectory(nullptr);

  TH2F* E2vsE1hist = new TH2F(Form("E2vsE1_FED_%d", FED),
                              Form("E2vsE1_FED_%d", FED),
                              numBins,
                              histRangeMin_,
                              histRangeMax_,
                              numBins,
                              histRangeMin_,
                              histRangeMax_);
  FEDsAndE2vsE1Hists_[FED] = E2vsE1hist;
  FEDsAndE2vsE1Hists_[FED]->SetDirectory(nullptr);

  TH2F* energyvsE1hist = new TH2F(Form("EnergyvsE1_FED_%d", FED),
                                  Form("EnergyvsE1_FED_%d", FED),
                                  numBins,
                                  histRangeMin_,
                                  histRangeMax_,
                                  numBins,
                                  histRangeMin_,
                                  histRangeMax_);
  FEDsAndenergyvsE1Hists_[FED] = energyvsE1hist;
  FEDsAndenergyvsE1Hists_[FED]->SetDirectory(nullptr);

  title1 = "Time for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  title1.append(";Relative Time (1 clock = 25ns);Events");
  name1 = "TimeFED";
  name1.append(intToString(FED));
  TH1F* timingHist = new TH1F(name1.c_str(), title1.c_str(), 78, -7, 7);
  FEDsAndTimingHists_[FED] = timingHist;
  FEDsAndTimingHists_[FED]->SetDirectory(nullptr);

  TH1F* freqHist =
      new TH1F(Form("Frequency_FED_%d", FED), Form("Frequency for FED %d;Event Number", FED), 100, 0., 100000);
  FEDsAndFrequencyHists_[FED] = freqHist;
  FEDsAndFrequencyHists_[FED]->SetDirectory(nullptr);

  TH1F* iphiProfileHist =
      new TH1F(Form("iPhi_Profile_FED_%d", FED), Form("iPhi Profile for FED %d", FED), 360, 1., 361);
  FEDsAndiPhiProfileHists_[FED] = iphiProfileHist;
  FEDsAndiPhiProfileHists_[FED]->SetDirectory(nullptr);

  TH1F* ietaProfileHist =
      new TH1F(Form("iEta_Profile_FED_%d", FED), Form("iEta Profile for FED %d", FED), 172, -86, 86);
  FEDsAndiEtaProfileHists_[FED] = ietaProfileHist;
  FEDsAndiEtaProfileHists_[FED]->SetDirectory(nullptr);

  TH2F* timingHistVsFreq =
      new TH2F(Form("timeVsFreqFED_%d", FED), Form("time Vs Freq FED %d", FED), 78, -7, 7, 100, 0., 100000);
  FEDsAndTimingVsFreqHists_[FED] = timingHistVsFreq;
  FEDsAndTimingVsFreqHists_[FED]->SetDirectory(nullptr);

  TH2F* timingHistVsAmp = new TH2F(
      Form("timeVsAmpFED_%d", FED), Form("time Vs Amp FED %d", FED), 78, -7, 7, numBins, histRangeMin_, histRangeMax_);
  FEDsAndTimingVsAmpHists_[FED] = timingHistVsAmp;
  FEDsAndTimingVsAmpHists_[FED]->SetDirectory(nullptr);

  TH1F* numXtalInClusterHist = new TH1F(Form("NumXtalsInCluster_FED_%d", FED),
                                        Form("Num active Xtals In Cluster for FED %d;Num Active Xtals", FED),
                                        25,
                                        0,
                                        25);
  FEDsAndNumXtalsInClusterHists_[FED] = numXtalInClusterHist;
  FEDsAndNumXtalsInClusterHists_[FED]->SetDirectory(nullptr);

  TH2F* OccupHist = new TH2F(Form("occupFED_%d", FED), Form("Occupancy FED %d;i#eta;i#phi", FED), 85, 1, 86, 20, 1, 21);
  FEDsAndOccupancyHists_[FED] = OccupHist;
  FEDsAndOccupancyHists_[FED]->SetDirectory(nullptr);

  TH2F* timingHistVsPhi = new TH2F(Form("timeVsPhiFED_%d", FED),
                                   Form("time Vs Phi FED %d;Relative Time (1 clock = 25ns);i#phi", FED),
                                   78,
                                   -7,
                                   7,
                                   20,
                                   1,
                                   21);
  FEDsAndTimingVsPhiHists_[FED] = timingHistVsPhi;
  FEDsAndTimingVsPhiHists_[FED]->SetDirectory(nullptr);

  TH2F* timingHistVsModule = new TH2F(Form("timeVsModuleFED_%d", FED),
                                      Form("time Vs Module FED %d;Relative Time (1 clock = 25ns);i#eta", FED),
                                      78,
                                      -7,
                                      7,
                                      4,
                                      1,
                                      86);
  FEDsAndTimingVsModuleHists_[FED] = timingHistVsModule;
  FEDsAndTimingVsModuleHists_[FED]->SetDirectory(nullptr);

  TH2F* dccRuntypeVsBxFED =
      new TH2F(Form("DCCRuntypeVsBxFED_%d", FED), Form("DCC Runtype vs. BX FED %d", FED), 3600, 0, 3600, 24, 0, 24);
  FEDsAndDCCRuntypeVsBxHists_[FED] = dccRuntypeVsBxFED;
  FEDsAndDCCRuntypeVsBxHists_[FED]->SetDirectory(nullptr);
}

// ------------ method called once each job just before starting event loop  ------------
void EcalCosmicsHists::beginRun(edm::Run const&, edm::EventSetup const& eventSetup) {
  edm::ESHandle<EcalElectronicsMapping> handle;
  eventSetup.get<EcalMappingRcd>().get(handle);
  ecalElectronicsMap_ = handle.product();

  //Here I will init some of the specific histograms
  int numBins = 200;  //(int)round(histRangeMax_-histRangeMin_)+1;

  //=============Special Bins for TT and Modules borders=============================
  double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0,
                          1,   6,   11,  16,  21,  26,  31,  36,  41,  46,  51,  56,  61,  66,  71,  76,  81, 86};
  double modEtaBins[10] = {-85, -65, -45, -25, 0, 1, 26, 46, 66, 86};
  double ttPhiBins[73];
  double modPhiBins[19];
  double timingBins[79];
  double highEBins[11];
  for (int i = 0; i < 79; ++i) {
    timingBins[i] = -7. + double(i) * 14. / 78.;
    if (i < 73) {
      ttPhiBins[i] = 1 + 5 * i;
      if (i < 19) {
        modPhiBins[i] = 1 + 20 * i;
        if (i < 11) {
          highEBins[i] = 10. + double(i) * 20.;
        }
      }
    }
  }
  //=============END Special Bins for TT and Modules borders===========================

  //====================Frequency for Event timing information=========================
  double timingEndInSeconds = runTimeLength_ * 3600.;
  double timingBinWidth = timingEndInSeconds / double(numTimingBins_);
  //====================END Frequency for Event timing information=====================

  allFedsenergyHist_ =
      new TH1F("energy_AllClusters", "energy_AllClusters;Cluster Energy (GeV)", numBins, histRangeMin_, histRangeMax_);
  allFedsenergyHighHist_ =
      new TH1F("energyHigh_AllClusters", "energyHigh_AllClusters;Cluster Energy (GeV)", numBins, histRangeMin_, 200.0);
  allFedsenergyOnlyHighHist_ = new TH1F("energyHigh_HighEnergyClusters",
                                        "energy of High Energy Clusters;Cluster Energy (GeV)",
                                        numBins,
                                        histRangeMin_,
                                        200.0);
  allFedsE2Hist_ = new TH1F(
      "E2_AllClusters", "E2_AllClusters;Seed+highest neighbor energy (GeV)", numBins, histRangeMin_, histRangeMax_);
  allFedsE2vsE1Hist_ = new TH2F("E2vsE1_AllClusters",
                                "E2vsE1_AllClusters;Seed Energy (GeV);Seed+highest neighbor energy (GeV)",
                                numBins,
                                histRangeMin_,
                                histRangeMax_,
                                numBins,
                                histRangeMin_,
                                histRangeMax_);
  allFedsenergyvsE1Hist_ = new TH2F("energyvsE1_AllClusters",
                                    "energyvsE1_AllClusters;Seed Energy (GeV);Energy(GeV)",
                                    numBins,
                                    histRangeMin_,
                                    histRangeMax_,
                                    numBins,
                                    histRangeMin_,
                                    histRangeMax_);
  allFedsTimingHist_ = new TH1F("timeForAllFeds", "timeForAllFeds;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingVsFreqHist_ = new TH2F("timeVsFreqAllEvent",
                                      "time Vs Freq All events;Relative Time (1 clock = 25ns);Event Number",
                                      78,
                                      -7,
                                      7,
                                      2000,
                                      0.,
                                      200000);
  allFedsTimingVsAmpHist_ = new TH2F("timeVsAmpAllEvents",
                                     "time Vs Amp All Events;Relative Time (1 clock = 25ns);Amplitude (GeV)",
                                     78,
                                     -7,
                                     7,
                                     numBins,
                                     histRangeMin_,
                                     histRangeMax_);
  allFedsFrequencyHist_ = new TH1F("FrequencyAllEvent", "Frequency for All events;Event Number", 2000, 0., 200000);

  //--------Special Hists for times stamp info----------------------------------------------
  allFedsFreqTimeHist_ = new TH1F("FrequencyAllEventsInTime",
                                  Form("Time of Cosmic Events; Time (s);Passing Event rate/%5g s", timingBinWidth),
                                  numTimingBins_,
                                  0.,
                                  timingEndInSeconds);
  allFedsFreqTimeVsPhiHist_ =
      new TH2F("FrequencyAllEventsInTimeVsPhi",
               Form("Time of Cosmic Events vs iPhi; iPhi;Time (s)/%5g s", timingBinWidth * 360.),
               360,
               1.,
               361.,
               numTimingBins_ / 360,
               0.,
               timingEndInSeconds);
  allFedsFreqTimeVsPhiTTHist_ =
      new TH2F("FrequencyAllEventsInTimeVsTTPhi",
               Form("Time of Cosmic Events vs iPhi (TT bins); iPhi;Time (s)/%5g s", timingBinWidth * 72.),
               72,
               1.,
               361.,
               numTimingBins_ / 72,
               0.,
               timingEndInSeconds);
  allFedsFreqTimeVsEtaHist_ =
      new TH2F("FrequencyAllEventsInTimeVsEta",
               Form("Time of Cosmic Events vs iEta; Time (s)/%5g s; iEta", timingBinWidth * 172.),
               numTimingBins_ / 172,
               0.,
               timingEndInSeconds,
               172,
               -86.,
               86.);
  allFedsFreqTimeVsEtaTTHist_ =
      new TH2F("FrequencyAllEventsInTimeVsTTEta",
               Form("Time of Cosmic Events vs Eta (TT bins);Time (s)/%5g s; iEta", timingBinWidth * 35.),
               numTimingBins_ / 35,
               0.,
               timingEndInSeconds,
               35,
               ttEtaBins);
  //--------END Special Hists for times stamp info------------------------------------------

  allFedsiPhiProfileHist_ = new TH1F("iPhiProfileAllEvents", "iPhi Profile all events;i#phi", 360, 1., 361.);
  allFedsiEtaProfileHist_ = new TH1F("iEtaProfileAllEvents", "iEta Profile all events;i#eta", 172, -86, 86);

  allOccupancy_ = new TH2F("OccupancyAllEvents", "Occupancy all events;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  TrueOccupancy_ =
      new TH2F("TrueOccupancyAllEvents", "True Occupancy all events;#phi;#eta", 360, -3.14159, 3.14159, 172, -1.5, 1.5);
  allOccupancyCoarse_ =
      new TH2F("OccupancyAllEventsCoarse", "Occupancy all events Coarse;i#phi;i#eta", 360 / 5, 1, 361., 35, ttEtaBins);
  TrueOccupancyCoarse_ = new TH2F("TrueOccupancyAllEventsCoarse",
                                  "True Occupancy all events Coarse;#phi;#eta",
                                  360 / 5,
                                  -3.14159,
                                  3.14159,
                                  34,
                                  -1.5,
                                  1.5);

  // single xtal cluster occupancy
  allOccupancySingleXtal_ =
      new TH2F("OccupancySingleXtal", "Occupancy single xtal clusters;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  energySingleXtalHist_ = new TH1F(
      "energy_SingleXtalClusters", "Energy single xtal clusters;Cluster Energy (GeV)", numBins, histRangeMin_, 200.0);

  allFedsTimingPhiHist_ = new TH2F("timePhiAllFEDs",
                                   "time vs Phi for all FEDs (TT binning);i#phi;Relative Time (1 clock = 25ns)",
                                   72,
                                   1,
                                   361,
                                   78,
                                   -7,
                                   7);
  allFedsTimingPhiEbpHist_ = new TH2F("timePhiEBP",
                                      "time vs Phi for FEDs in EB+ (TT binning) ;i#phi;Relative Time (1 clock = 25ns)",
                                      72,
                                      1,
                                      361,
                                      78,
                                      -7,
                                      7);
  allFedsTimingPhiEbmHist_ = new TH2F("timePhiEBM",
                                      "time vs Phi for FEDs in EB- (TT binning);i#phi;Relative Time (1 clock = 25ns)",
                                      72,
                                      1,
                                      361,
                                      78,
                                      -7,
                                      7);
  allFedsTimingPhiEtaHist_ =
      new TH3F("timePhiEtaAllFEDs",
               "(Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHist_ =
      new TH3F("timeTTAllFEDs",
               "(Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHist_ = new TH2F(
      "timeLMAllFEDs", "(LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)", 92, 1, 92, 78, -7, 7);

  allFedsTimingEbpHist_ = new TH1F("timeEBP", "time for FEDs in EB+;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingEbmHist_ = new TH1F("timeEBM", "time for FEDs in EB-;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingEbpTopHist_ =
      new TH1F("timeEBPTop", "time for FEDs in EB+ Top;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingEbmTopHist_ =
      new TH1F("timeEBMTop", "time for FEDs in EB- Top;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingEbpBottomHist_ =
      new TH1F("timeEBPBottom", "time for FEDs in EB+ Bottom;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingEbmBottomHist_ =
      new TH1F("timeEBMBottom", "time for FEDs in EB- Bottom;Relative Time (1 clock = 25ns)", 78, -7, 7);

  numberofCosmicsHist_ =
      new TH1F("numberofCosmicsPerEvent", "Number of cosmics per event;Number of Cosmics", 30, 0, 30);
  numberofCosmicsHistEB_ =
      new TH1F("numberofCosmicsPerEvent_EB", "Number of cosmics per event EB;Number of Cosmics", 30, 0, 30);

  numberofCosmicsWTrackHist_ =
      new TH1F("numberofCosmicsWTrackPerEvent", "Number of cosmics with track per event", 30, 0, 30);
  numberofCosmicsTopBottomHist_ = new TH1F(
      "numberofCosmicsTopBottomPerEvent", "Number of top bottom cosmics per event;Number of Cosmics", 30, 0, 30);
  numberofGoodEvtFreq_ = new TH1F("frequencyOfGoodEvents",
                                  "Number of events with cosmic vs Event;Event Number;Number of Good Events/100 Events",
                                  2000,
                                  0,
                                  200000);

  numberofCrossedEcalIdsHist_ = new TH1F("numberofCrossedEcalCosmicsPerEvent",
                                         "Number of crossed ECAL cosmics per event;Number of Crossed Cosmics",
                                         10,
                                         0,
                                         10);

  allOccupancyExclusiveECAL_ = new TH2F("OccupancyAllEvents_ExclusiveECAL",
                                        "Occupancy all events Exclusive ECAL ;i#phi;i#eta",
                                        360,
                                        1.,
                                        361.,
                                        172,
                                        -86,
                                        86);
  allOccupancyCoarseExclusiveECAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveECAL",
                                              "Occupancy all events Coarse Exclusive ECAL;i#phi;i#eta",
                                              360 / 5,
                                              ttPhiBins,
                                              35,
                                              ttEtaBins);
  allOccupancyECAL_ =
      new TH2F("OccupancyAllEvents_ECAL", "Occupancy all events ECAL;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseECAL_ = new TH2F("OccupancyAllEventsCoarse_ECAL",
                                     "Occupancy all events Coarse ECAL;i#phi;i#eta",
                                     360 / 5,
                                     ttPhiBins,
                                     35,
                                     ttEtaBins);
  allFedsTimingHistECAL_ =
      new TH1F("timeForAllFeds_ECAL", "timeForAllFeds ECAL;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingPhiEtaHistECAL_ =
      new TH3F("timePhiEtaAllFEDs_ECAL",
               "ECAL (Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHistECAL_ =
      new TH3F("timeTTAllFEDs_ECAL",
               "ECAL (Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHistECAL_ = new TH2F("timeLMAllFEDs_ECAL",
                                      "ECAL (LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)",
                                      92,
                                      1,
                                      92,
                                      78,
                                      -7,
                                      7);

  allOccupancyExclusiveDT_ = new TH2F(
      "OccupancyAllEvents_ExclusiveDT", "Occupancy all events Exclusive DT;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseExclusiveDT_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveDT",
                                            "Occupancy all events Coarse Exclusive DT;i#phi;i#eta",
                                            360 / 5,
                                            1,
                                            361.,
                                            35,
                                            ttEtaBins);
  allOccupancyDT_ =
      new TH2F("OccupancyAllEvents_DT", "Occupancy all events DT;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseDT_ = new TH2F(
      "OccupancyAllEventsCoarse_DT", "Occupancy all events Coarse DT;i#phi;i#eta", 360 / 5, 1, 361., 35, ttEtaBins);
  allFedsTimingHistDT_ = new TH1F("timeForAllFeds_DT", "timeForAllFeds DT;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingPhiEtaHistDT_ =
      new TH3F("timePhiEtaAllFEDs_DT",
               "DT (Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHistDT_ =
      new TH3F("timeTTAllFEDs_DT",
               "DT (Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHistDT_ = new TH2F("timeLMAllFEDs_DT",
                                    "DT (LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)",
                                    92,
                                    1,
                                    92,
                                    78,
                                    -7,
                                    7);

  allOccupancyExclusiveRPC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveRPC", "Occupancy all events Exclusive RPC;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseExclusiveRPC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveRPC",
                                             "Occupancy all events Coarse Exclusive RPC;i#phi;i#eta",
                                             360 / 5,
                                             1,
                                             361.,
                                             35,
                                             ttEtaBins);
  allOccupancyRPC_ =
      new TH2F("OccupancyAllEvents_RPC", "Occupancy all events RPC;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseRPC_ = new TH2F(
      "OccupancyAllEventsCoarse_RPC", "Occupancy all events Coarse RPC;i#phi;i#eta", 360 / 5, 1, 361., 35, ttEtaBins);
  allFedsTimingHistRPC_ =
      new TH1F("timeForAllFeds_RPC", "timeForAllFeds RPC;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingPhiEtaHistRPC_ =
      new TH3F("timePhiEtaAllFEDs_RPC",
               "RPC (Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHistRPC_ =
      new TH3F("timeTTAllFEDs_RPC",
               "RPC (Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHistRPC_ = new TH2F("timeLMAllFEDs_RPC",
                                     "RPC (LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)",
                                     92,
                                     1,
                                     92,
                                     78,
                                     -7,
                                     7);

  allOccupancyExclusiveCSC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveCSC", "Occupancy all events Exclusive CSC;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseExclusiveCSC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveCSC",
                                             "Occupancy all events Coarse Exclusive CSC;i#phi;i#eta",
                                             360 / 5,
                                             1,
                                             361.,
                                             35,
                                             ttEtaBins);
  allOccupancyCSC_ =
      new TH2F("OccupancyAllEvents_CSC", "Occupancy all events CSC;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseCSC_ = new TH2F(
      "OccupancyAllEventsCoarse_CSC", "Occupancy all events Coarse CSC;i#phi;i#eta", 360 / 5, 1, 361., 35, ttEtaBins);
  allFedsTimingHistCSC_ =
      new TH1F("timeForAllFeds_CSC", "timeForAllFeds CSC;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingPhiEtaHistCSC_ =
      new TH3F("timePhiEtaAllFEDs_CSC",
               "CSC (Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHistCSC_ =
      new TH3F("timeTTAllFEDs_CSC",
               "CSC (Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHistCSC_ = new TH2F("timeLMAllFEDs_CSC",
                                     "CSC (LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)",
                                     92,
                                     1,
                                     62,
                                     78,
                                     -7,
                                     7);

  allOccupancyExclusiveHCAL_ = new TH2F("OccupancyAllEvents_ExclusiveHCAL",
                                        "Occupancy all events Exclusive HCAL;i#phi;i#eta",
                                        360,
                                        1.,
                                        361.,
                                        172,
                                        -86,
                                        86);
  allOccupancyCoarseExclusiveHCAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveHCAL",
                                              "Occupancy all events Coarse Exclusive HCAL;i#phi;i#eta",
                                              360 / 5,
                                              1,
                                              361.,
                                              35,
                                              ttEtaBins);
  allOccupancyHCAL_ =
      new TH2F("OccupancyAllEvents_HCAL", "Occupancy all events HCAL;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyCoarseHCAL_ = new TH2F(
      "OccupancyAllEventsCoarse_HCAL", "Occupancy all events Coarse HCAL;i#phi;i#eta", 360 / 5, 1, 361., 35, ttEtaBins);
  allFedsTimingHistHCAL_ =
      new TH1F("timeForAllFeds_HCAL", "timeForAllFeds HCAL;Relative Time (1 clock = 25ns)", 78, -7, 7);
  allFedsTimingPhiEtaHistHCAL_ =
      new TH3F("timePhiEtaAllFEDs_HCAL",
               "HCAL (Phi,Eta,time) for all FEDs (SM,M binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               18,
               modPhiBins,
               9,
               modEtaBins,
               78,
               timingBins);
  allFedsTimingTTHistHCAL_ =
      new TH3F("timeTTAllFEDs_HCAL",
               "HCAL (Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               78,
               timingBins);
  allFedsTimingLMHistHCAL_ = new TH2F("timeLMAllFEDs_HCAL",
                                      "HCAL (LM,time) for all FEDs (SM,LM binning);LM;Relative Time (1 clock = 25ns)",
                                      92,
                                      1,
                                      92,
                                      78,
                                      -7,
                                      7);

  TrueBCOccupancy_ = new TH2F(
      "BCTrueOccupancyAllEvents", "True SB Occupancy all events;#phi;#eta", 360, -3.14159, 3.14159, 172, -1.5, 1.5);
  TrueBCOccupancyCoarse_ = new TH2F("BCTrueOccupancyAllEventsCoarse",
                                    "True BC Occupancy all events Coarse;#phi;#eta",
                                    360 / 5,
                                    -3.14159,
                                    3.14159,
                                    34,
                                    -1.5,
                                    1.5);

  numberofBCinSC_ =
      new TH1F("numberofBCinSC", "Number of Basic Clusters in Super Cluster;Num Basic Clusters", 20, 0, 20);  //SC
  numberofBCinSCphi_ = new TH2F("numberofBCinSCphi",
                                "Number of Basic Clusters in Super Cluster;phi;Num Basic Clusters",
                                360 / 5,
                                -3.14159,
                                3.14159,
                                20,
                                0,
                                20);  //SC

  allFedsTimingHistEcalMuon_ =
      new TH1F("timeForAllFeds_EcalMuon", "timeForAllFeds Ecal+Muon;Relative Time (1 clock = 25ns)", 78, -7, 7);

  triggerHist_ = new TH1F("triggerHist", "Trigger Number", 5, 0, 5);
  triggerHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  triggerHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  triggerHist_->GetXaxis()->SetBinLabel(3, "DT");
  triggerHist_->GetXaxis()->SetBinLabel(4, "RPC");
  triggerHist_->GetXaxis()->SetBinLabel(5, "CSC");

  triggerExclusiveHist_ = new TH1F("triggerExclusiveHist", "Trigger Number (Mutually Exclusive)", 5, 0, 5);
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(3, "DT");
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(4, "RPC");
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(5, "CSC");

  runNumberHist_ = new TH1F("runNumberHist", "Run Number", 1, 0, 1);

  deltaRHist_ = new TH1F("deltaRHist", "deltaR", 500, -0.5, 499.5);
  deltaEtaHist_ = new TH1F("deltaIEtaHist", "deltaIEta", 170, -85.5, 84.5);
  deltaPhiHist_ = new TH1F("deltaIPhiHist", "deltaIPhi", 720, -360.5, 359.5);
  ratioAssocTracksHist_ = new TH1F("ratioAssocTracks", "num assoc. tracks/tracks through Ecal", 11, 0, 1.1);
  ratioAssocClustersHist_ = new TH1F("ratioAssocClusters", "num assoc. clusters/total clusters", 11, 0, 1.1);
  trackAssoc_muonsEcal_ = new TH2F(
      "trackAssoc_muonsEcal", "Map of muon hits in Ecal", 360, 1., 361., 172, -86, 86);  //360, 0 , 360, 170,-85 ,85);
  deltaEtaDeltaPhiHist_ =
      new TH2F("deltaEtaDeltaPhi", "Delta ieta vs. delta iphi", 170, -85.5, 84.5, 720, -360.5, 359.5);
  seedTrackEtaHist_ = new TH2F("seedTrackEta", "track ieta vs. seed ieta", 170, -85.5, 84.5, 170, -85.5, 84.5);
  seedTrackPhiHist_ = new TH2F("seedTrackPhi", "track iphi vs. seed iphi", 720, -360.5, 359.5, 720, -360.5, 359.5);

  dccEventVsBxHist_ = new TH2F("dccEventVsBx", "DCC Runtype vs. bunch crossing", 3600, 0, 3600, 24, 0, 24);
  dccBXErrorByFEDHist_ = new TH1F("dccBXErrorByFED", "Incorrect BX number by FED", 54, 601, 655);
  dccOrbitErrorByFEDHist_ = new TH1F("dccOrbitErrorByFED", "Incorrect orbit number by FED", 54, 601, 655);
  dccRuntypeErrorByFEDHist_ = new TH1F("dccRuntypeErrorByFED", "Incorrect DCC Runtype by FED", 54, 601, 655);
  dccRuntypeHist_ = new TH1F("dccRuntype", "DCC Runtype frequency", 24, 0, 24);
  dccErrorVsBxHist_ = new TH2F("dccErrorVsBX", "DCC Errors vs. BX", 3600, 0, 3600, 3, 0, 3);

  hcalEnergy_HBHE_ = new TH1F("hcalEnergy_HBHE", "RecHit Energy HBHE", 440, -10, 100);
  hcalEnergy_HF_ = new TH1F("hcalEnergy_HF", "RecHit Energy HF", 440, -10, 100);
  hcalEnergy_HO_ = new TH1F("hcalEnergy_HO", "RecHit Energy HO", 440, -10, 100);
  hcalHEHBecalEB_ =
      new TH2F("hcalHEHBecalEB", "HCAL HBHE RecHit energy vs ECAL EB energy", numBins, histRangeMin_, 300.0, 40, -5, 5);

  NumXtalsInClusterHist_ = new TH1F("NumXtalsInClusterAllHist", "Number of Xtals in Cluster;NumXtals", 150, 0, 150);
  numxtalsVsEnergy_ = new TH2F("NumXtalsVsEnergy",
                               "Number of Xtals in Cluster vs Energy;Energy (GeV);Number of Xtals in Cluster",
                               numBins,
                               histRangeMin_,
                               histRangeMax_,
                               150,
                               0,
                               150);
  numxtalsVsHighEnergy_ = new TH2F("NumXtalsVsHighEnergy",
                                   "Number of Xtals in Cluster vs Energy;Energy (GeV);Number of Xtals in Cluster",
                                   numBins,
                                   histRangeMin_,
                                   200.,
                                   150,
                                   0,
                                   150);

  // high energy analysis
  allOccupancyHighEnergy_ =
      new TH2F("OccupancyHighEnergyEvents", "Occupancy high energy events;i#phi;i#eta", 360, 1., 361., 172, -86, 86);
  allOccupancyHighEnergyCoarse_ = new TH2F("OccupancyHighEnergyEventsCoarse",
                                           "Occupancy high energy events Coarse;i#phi;i#eta",
                                           72,
                                           ttPhiBins,
                                           35,
                                           ttEtaBins);
  allFedsOccupancyHighEnergyHist_ = new TH3F("OccupancyHighEnergyEvents3D",
                                             "(Phi,Eta,energy) for all high energy events;i#phi;i#eta;energy (GeV)",
                                             18,
                                             modPhiBins,
                                             9,
                                             modEtaBins,
                                             10,
                                             highEBins);
  allFedsNumXtalsInClusterHist_ =
      new TH1F("NumActiveXtalsInClusterAllHist", "Number of active Xtals in Cluster;NumXtals", 100, 0, 100);

  HighEnergy_NumXtal = new TH1F("HighEnergy_NumXtal", "Num crystals in high E clusters;num crystals", 150, 0, 150);
  HighEnergy_NumXtalFedId = new TH2F(
      "HighEnergy_NumXtalFedId", "Num crystals in cluster vs FedId;FedId;num crystals", 36, 610., 645., 150, 0, 150);
  HighEnergy_NumXtaliphi = new TH2F(
      "HighEnergy_NumXtaliphi", "Num crystals in cluster vs iphi;i#phi;num crystals", 360, 1., 361., 150, 0, 150);
  HighEnergy_energy3D = new TH3F("HighEnergy_energy3D",
                                 "(Phi,Eta,energy) for all high energy events;i#phi;i#eta;energy (GeV)",
                                 72,
                                 ttPhiBins,
                                 35,
                                 ttEtaBins,
                                 10,
                                 highEBins);

  HighEnergy_energyNumXtal = new TH2F("HighEnergy_energyNumXtal",
                                      "Energy in cluster vs Num crystals in cluster;num crystals;energy",
                                      150,
                                      0,
                                      150,
                                      200,
                                      0.,
                                      200.);

  HighEnergy_bestSeed = new TH1F("HighEnergy_bestSeedEnergy", "BestSeed Energy from TrackAss", 200, 0., 200.);
  HighEnergy_bestSeedOccupancy = new TH2F(
      "HighEnergy_bestSeedOccupancy", "Occupancy HighEn events from TrackAss;i#phi;i#eta", 360, 1., 361., 172, -86, 86);

  HighEnergy_numClusHighEn = new TH1F("HighEnergy_numClusHighEn", "Num High Energy Clusters", 7, 0, 7);
  HighEnergy_ratioClusters =
      new TH1F("HighEnergy_ratioClusters", "Num High Energy Clusters/Num tot Clusters", 100, 0., 1.1);

  HighEnergy_numRecoTrackBarrel = new TH1F("HighEnergy_numRecoTracksBarrel", "Num BarrelRecoTracks", 10, 0, 10);
  HighEnergy_TracksAngle = new TH1F("HighEnergy_TracksAngle", "Angle between tracks", 720, 0., 180.);
  HighEnergy_TracksAngleTopBottom =
      new TH1F("HighEnergy_TopBottomTracksAngle", "Angle between top-bottom tracks", 720, 0., 180.);

  HighEnergy_2GeV_occuCoarse = new TH2F("HighEnergy_occu2GeV_Coarse",
                                        "Occupancy high energy events with more than 2 GeV;i#phi;i#eta",
                                        72,
                                        ttPhiBins,
                                        35,
                                        ttEtaBins);
  HighEnergy_2GeV_occu3D = new TH3F("HighEnergy_2GeV_energy3D",
                                    "(iphi,ieta,energy) for all high energy events w > 10 GeV;i#phi;i#eta;energy (GeV)",
                                    72,
                                    ttPhiBins,
                                    35,
                                    ttEtaBins,
                                    10,
                                    highEBins);
  HighEnergy_100GeV_occuCoarse = new TH2F("HighEnergy_occu100GeV_Coarse",
                                          "Occupancy high energy events with more than 100 GeV;i#phi;i#eta",
                                          72,
                                          ttPhiBins,
                                          35,
                                          ttEtaBins);
  HighEnergy_100GeV_occu3D =
      new TH3F("HighEnergy_100GeV_energy3D",
               "(iphi,ieta,energy) for all high energy events more than 100 GeV;i#phi;i#eta;energy (GeV)",
               72,
               ttPhiBins,
               35,
               ttEtaBins,
               10,
               highEBins);
  HighEnergy_0tracks_occu3D = new TH3F("HighEnergy_0Tracks_energy3D",
                                       "(iphi,ieta,energy) for all events with 0 tracks;i#phi;i#eta;energy (GeV)",
                                       72,
                                       ttPhiBins,
                                       35,
                                       ttEtaBins,
                                       10,
                                       highEBins);
  HighEnergy_1tracks_occu3D = new TH3F("HighEnergy_1Tracks_energy3D",
                                       "(iphi,ieta,energy) for all events with 1 tracks;i#phi;i#eta;energy (GeV)",
                                       72,
                                       ttPhiBins,
                                       35,
                                       ttEtaBins,
                                       10,
                                       highEBins);
  HighEnergy_2tracks_occu3D = new TH3F("HighEnergy_2Tracks_energy3D",
                                       "(iphi,ieta,energy) for all events with 2 tracks;i#phi;i#eta;energy (GeV)",
                                       72,
                                       ttPhiBins,
                                       35,
                                       ttEtaBins,
                                       10,
                                       highEBins);
  HighEnergy_0tracks_occu3DXtal = new TH3F("HighEnergy_0Tracks_energy3DXtal",
                                           "(iphi,ieta,energy) for all events with 0 tracks;i#phi;i#eta;energy (GeV)",
                                           360,
                                           1.,
                                           361.,
                                           172,
                                           -86,
                                           86,
                                           200,
                                           0.,
                                           200.);
  HighEnergy_1tracks_occu3DXtal = new TH3F("HighEnergy_1Tracks_energy3DXtal",
                                           "(iphi,ieta,energy) for all events with 1 tracks;i#phi;i#eta;energy (GeV)",
                                           360,
                                           1.,
                                           361.,
                                           172,
                                           -86,
                                           86,
                                           200,
                                           0.,
                                           200.);
  HighEnergy_2tracks_occu3DXtal = new TH3F("HighEnergy_2Tracks_energy3DXtal",
                                           "(iphi,ieta,energy) for all events with 2 tracks;i#phi;i#eta;energy (GeV)",
                                           360,
                                           1.,
                                           361.,
                                           172,
                                           -86,
                                           86,
                                           200,
                                           0.,
                                           200.);

  //EE histograms

  // EE-
  EEM_FedsSeedEnergyHist_ =
      new TH1F("SeedEnergyAllFEDs", "Seed Energy for EEM Feds; Seed Energy (GeV)", 200, histRangeMin_, 10.0);

  EEM_AllOccupancyCoarse_ =
      new TH2F("OccupancyAllEventsCoarse", "Occupancy all events Coarse EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_AllOccupancy_ = new TH2F("OccupancyAllEvents", "Occupancy all events EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_FedsenergyHist_ =
      new TH1F("energy_AllClusters", "energy_AllClusters_EEM;Cluster Energy (GeV)", numBins, histRangeMin_, 10.0);
  EEM_FedsenergyHighHist_ = new TH1F(
      "energyHigh_AllClusters", "energyHigh_AllClusters in EEM;Cluster Energy (GeV)", numBins, histRangeMin_, 200.0);
  EEM_FedsenergyOnlyHighHist_ = new TH1F("energyHigh_HighEnergyClusters",
                                         "energy of High Energy Clusters in EEM;Cluster Energy (GeV)",
                                         numBins,
                                         histRangeMin_,
                                         200.0);
  EEM_FedsE2Hist_ =
      new TH1F("E2_AllClusters", "E2_AllClusters_EEM;Seed+highest neighbor energy (GeV)", numBins, histRangeMin_, 10.0);
  EEM_FedsE2vsE1Hist_ = new TH2F("E2vsE1_AllClusters",
                                 "E2vsE1_AllClusters_EEM;Seed Energy (GeV);Seed+highest neighbor energy (GeV)",
                                 numBins,
                                 histRangeMin_,
                                 10.0,
                                 numBins,
                                 histRangeMin_,
                                 10.0);
  EEM_FedsenergyvsE1Hist_ = new TH2F("energyvsE1_AllClusters",
                                     "energyvsE1_AllClusters_EEM;Seed Energy (GeV);Energy(GeV)",
                                     numBins,
                                     histRangeMin_,
                                     10.0,
                                     numBins,
                                     histRangeMin_,
                                     10.0);
  EEM_FedsTimingHist_ = new TH1F("timeForAllFeds", "timeForAllFeds_EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_numberofCosmicsHist_ =
      new TH1F("numberofCosmicsPerEvent", "Number of cosmics per event EEM;Number of Cosmics", 30, 0, 30);

  EEM_FedsTimingVsAmpHist_ = new TH2F("timeVsAmpAllEvents",
                                      "time Vs Amp All Events EEM;Relative Time (1 clock = 25ns);Amplitude (GeV)",
                                      78,
                                      -7,
                                      7,
                                      numBins,
                                      histRangeMin_,
                                      10.0);
  EEM_FedsTimingTTHist_ = new TH3F("timeTTAllFEDs",
                                   "(ix,iy,time) for all FEDs (SM,TT binning) EEM;ix;iy;Relative Time (1 clock = 25ns)",
                                   20,
                                   0,
                                   100,
                                   20,
                                   0,
                                   100,
                                   78,
                                   -7,
                                   7);

  EEM_OccupancySingleXtal_ =
      new TH2F("OccupancySingleXtal", "Occupancy single xtal clusters EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_energySingleXtalHist_ = new TH1F("energy_SingleXtalClusters",
                                       "Energy single xtal clusters EEM;Cluster Energy (GeV)",
                                       numBins,
                                       histRangeMin_,
                                       200.0);

  EEM_OccupancyExclusiveECAL_ = new TH2F(
      "OccupancyAllEvents_ExclusiveECAL", "Occupancy all events Exclusive ECAL  EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseExclusiveECAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveECAL",
                                               "Occupancy all events Coarse Exclusive ECAL EEM;ix;iy",
                                               20,
                                               0,
                                               100,
                                               20,
                                               0,
                                               100);
  EEM_OccupancyECAL_ =
      new TH2F("OccupancyAllEvents_ECAL", "Occupancy all events ECAL EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseECAL_ =
      new TH2F("OccupancyAllEventsCoarse_ECAL", "Occupancy all events Coarse ECAL EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_FedsTimingHistECAL_ =
      new TH1F("timeForAllFeds_ECAL", "timeForAllFeds ECAL EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_FedsTimingTTHistECAL_ =
      new TH3F("timeTTAllFEDs_ECAL",
               "(ix,iy,time) for all FEDs (SM,TT binning) ECAL EEM;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEM_OccupancyExclusiveDT_ = new TH2F(
      "OccupancyAllEvents_ExclusiveDT", "Occupancy all events Exclusive DT EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseExclusiveDT_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveDT",
                                             "Occupancy all events Coarse Exclusive DT EEM;ix;iy",
                                             20,
                                             0,
                                             100,
                                             20,
                                             0,
                                             100);
  EEM_OccupancyDT_ = new TH2F("OccupancyAllEvents_DT", "Occupancy all events DT EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseDT_ =
      new TH2F("OccupancyAllEventsCoarse_DT", "Occupancy all events Coarse DT EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_FedsTimingHistDT_ =
      new TH1F("timeForAllFeds_DT", "timeForAllFeds DT EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_FedsTimingTTHistDT_ =
      new TH3F("timeTTAllFEDs_DT",
               "(ix,iy,time) for all FEDs (SM,TT binning) DT EEM;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEM_OccupancyExclusiveRPC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveRPC", "Occupancy all events Exclusive RPC EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseExclusiveRPC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveRPC",
                                              "Occupancy all events Coarse Exclusive RPC EEM;ix;iy",
                                              20,
                                              0,
                                              100,
                                              20,
                                              0,
                                              100);
  EEM_OccupancyRPC_ =
      new TH2F("OccupancyAllEvents_RPC", "Occupancy all events RPC EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseRPC_ =
      new TH2F("OccupancyAllEventsCoarse_RPC", "Occupancy all events Coarse RPC EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_FedsTimingHistRPC_ =
      new TH1F("timeForAllFeds_RPC", "timeForAllFeds RPC EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_FedsTimingTTHistRPC_ =
      new TH3F("timeTTAllFEDs_RPC",
               "(ix,iy,time) for all FEDs (SM,TT binning) RPC EEM;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEM_OccupancyExclusiveCSC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveCSC", "Occupancy all events Exclusive CSC EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseExclusiveCSC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveCSC",
                                              "Occupancy all events Coarse Exclusive CSC EEM;ix;iy",
                                              20,
                                              0,
                                              100,
                                              20,
                                              0,
                                              100);
  EEM_OccupancyCSC_ =
      new TH2F("OccupancyAllEvents_CSC", "Occupancy all events CSC EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseCSC_ =
      new TH2F("OccupancyAllEventsCoarse_CSC", "Occupancy all events Coarse CSC EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_FedsTimingHistCSC_ =
      new TH1F("timeForAllFeds_CSC", "timeForAllFeds CSC EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_FedsTimingTTHistCSC_ =
      new TH3F("timeTTAllFEDs_CSC",
               "(ix,iy,time) for all FEDs (SM,TT binning) CSC EEM;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEM_OccupancyExclusiveHCAL_ = new TH2F(
      "OccupancyAllEvents_ExclusiveHCAL", "Occupancy all events Exclusive HCAL EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseExclusiveHCAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveHCAL",
                                               "Occupancy all events Coarse Exclusive HCAL EEM;ix;iy",
                                               20,
                                               0,
                                               100,
                                               20,
                                               0,
                                               100);
  EEM_OccupancyHCAL_ =
      new TH2F("OccupancyAllEvents_HCAL", "Occupancy all events HCAL EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyCoarseHCAL_ =
      new TH2F("OccupancyAllEventsCoarse_HCAL", "Occupancy all events Coarse HCAL EEM;ix;iy", 20, 0, 100, 20, 0, 100);
  EEM_FedsTimingHistHCAL_ =
      new TH1F("timeForAllFeds_HCAL", "timeForAllFeds HCAL EEM;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEM_FedsTimingTTHistHCAL_ =
      new TH3F("timeTTAllFEDs_HCAL",
               "(ix,iy,time) for all FEDs (SM,TT binning) HCAL EEM;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEM_numberofBCinSC_ =
      new TH1F("numberofBCinSC", "Number of Basic Clusters in Super Cluster EEM;Num Basic Clusters", 20, 0, 20);

  EEM_triggerHist_ = new TH1F("triggerHist", "Trigger Number EEM", 5, 0, 5);
  EEM_triggerHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  EEM_triggerHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  EEM_triggerHist_->GetXaxis()->SetBinLabel(3, "DT");
  EEM_triggerHist_->GetXaxis()->SetBinLabel(4, "RPC");
  EEM_triggerHist_->GetXaxis()->SetBinLabel(5, "CSC");

  EEM_triggerExclusiveHist_ = new TH1F("triggerExclusiveHist", "Trigger Number (Mutually Exclusive) EEM", 5, 0, 5);
  triggerExclusiveHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  EEM_triggerExclusiveHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  EEM_triggerExclusiveHist_->GetXaxis()->SetBinLabel(3, "DT");
  EEM_triggerExclusiveHist_->GetXaxis()->SetBinLabel(4, "RPC");
  EEM_triggerExclusiveHist_->GetXaxis()->SetBinLabel(5, "CSC");

  EEM_NumXtalsInClusterHist_ =
      new TH1F("NumXtalsInClusterAllHist", "Number of Xtals in Cluster EEM;NumXtals", 150, 0, 150);
  EEM_numxtalsVsEnergy_ = new TH2F("NumXtalsVsEnergy",
                                   "Number of Xtals in Cluster vs Energy EEM;Energy (GeV);Number of Xtals in Cluster",
                                   numBins,
                                   histRangeMin_,
                                   10.0,
                                   150,
                                   0,
                                   150);
  EEM_numxtalsVsHighEnergy_ =
      new TH2F("NumXtalsVsHighEnergy",
               "Number of Xtals in Cluster vs Energy EEM;Energy (GeV);Number of Xtals in Cluster",
               numBins,
               histRangeMin_,
               200.,
               150,
               0,
               150);

  EEM_OccupancyHighEnergy_ =
      new TH2F("OccupancyHighEnergyEvents", "Occupancy high energy events EEM;ix;iy", 100, 0, 100, 100, 0, 100);
  EEM_OccupancyHighEnergyCoarse_ = new TH2F(
      "OccupancyHighEnergyEventsCoarse", "Occupancy high energy events Coarse EEM;ix;iy", 20, 0, 100, 20, 0, 100);

  EEM_FedsNumXtalsInClusterHist_ =
      new TH1F("NumActiveXtalsInClusterAllHist", "Number of active Xtals in Cluster EEM;NumXtals", 100, 0, 100);

  // EE+
  EEP_FedsSeedEnergyHist_ =
      new TH1F("SeedEnergyAllFEDs", "Seed Energy for EEP Feds; Seed Energy (GeV)", 200, histRangeMin_, 10.0);

  EEP_AllOccupancyCoarse_ =
      new TH2F("OccupancyAllEventsCoarse", "Occupancy all events Coarse EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_AllOccupancy_ = new TH2F("OccupancyAllEvents", "Occupancy all events EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_FedsenergyHist_ =
      new TH1F("energy_AllClusters", "energy_AllClusters_EEP;Cluster Energy (GeV)", numBins, histRangeMin_, 10.0);
  EEP_FedsenergyHighHist_ = new TH1F(
      "energyHigh_AllClusters", "energyHigh_AllClusters in EEP;Cluster Energy (GeV)", numBins, histRangeMin_, 200.0);
  EEP_FedsenergyOnlyHighHist_ = new TH1F("energyHigh_HighEnergyClusters",
                                         "energy of High Energy Clusters in EEP;Cluster Energy (GeV)",
                                         numBins,
                                         histRangeMin_,
                                         200.0);
  EEP_FedsE2Hist_ =
      new TH1F("E2_AllClusters", "E2_AllClusters_EEP;Seed+highest neighbor energy (GeV)", numBins, histRangeMin_, 10.0);
  EEP_FedsE2vsE1Hist_ = new TH2F("E2vsE1_AllClusters",
                                 "E2vsE1_AllClusters_EEP;Seed Energy (GeV);Seed+highest neighbor energy (GeV)",
                                 numBins,
                                 histRangeMin_,
                                 10.0,
                                 numBins,
                                 histRangeMin_,
                                 10.0);
  EEP_FedsenergyvsE1Hist_ = new TH2F("energyvsE1_AllClusters",
                                     "energyvsE1_AllClusters_EEP;Seed Energy (GeV);Energy(GeV)",
                                     numBins,
                                     histRangeMin_,
                                     10.0,
                                     numBins,
                                     histRangeMin_,
                                     10.0);
  EEP_FedsTimingHist_ = new TH1F("timeForAllFeds", "timeForAllFeds_EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_numberofCosmicsHist_ =
      new TH1F("numberofCosmicsPerEvent", "Number of cosmics per event EEP;Number of Cosmics", 30, 0, 30);

  EEP_FedsTimingVsAmpHist_ = new TH2F("timeVsAmpAllEvents",
                                      "time Vs Amp All Events EEP;Relative Time (1 clock = 25ns);Amplitude (GeV)",
                                      78,
                                      -7,
                                      7,
                                      numBins,
                                      histRangeMin_,
                                      10.0);
  EEP_FedsTimingTTHist_ = new TH3F("timeTTAllFEDs",
                                   "(ix,iy,time) for all FEDs (SM,TT binning) EEP;ix;iy;Relative Time (1 clock = 25ns)",
                                   20,
                                   0,
                                   100,
                                   20,
                                   0,
                                   100,
                                   78,
                                   -7,
                                   7);

  EEP_OccupancySingleXtal_ =
      new TH2F("OccupancySingleXtal", "Occupancy single xtal clusters EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_energySingleXtalHist_ = new TH1F("energy_SingleXtalClusters",
                                       "Energy single xtal clusters EEP;Cluster Energy (GeV)",
                                       numBins,
                                       histRangeMin_,
                                       200.0);

  EEP_OccupancyExclusiveECAL_ = new TH2F(
      "OccupancyAllEvents_ExclusiveECAL", "Occupancy all events Exclusive ECAL  EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseExclusiveECAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveECAL",
                                               "Occupancy all events Coarse Exclusive ECAL EEP;ix;iy",
                                               20,
                                               0,
                                               100,
                                               20,
                                               0,
                                               100);
  EEP_OccupancyECAL_ =
      new TH2F("OccupancyAllEvents_ECAL", "Occupancy all events ECAL EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseECAL_ =
      new TH2F("OccupancyAllEventsCoarse_ECAL", "Occupancy all events Coarse ECAL EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_FedsTimingHistECAL_ =
      new TH1F("timeForAllFeds_ECAL", "timeForAllFeds ECAL EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_FedsTimingTTHistECAL_ =
      new TH3F("timeTTAllFEDs_ECAL",
               "(ix,iy,time) for all FEDs (SM,TT binning) ECAL EEP;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEP_OccupancyExclusiveDT_ = new TH2F(
      "OccupancyAllEvents_ExclusiveDT", "Occupancy all events Exclusive DT EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseExclusiveDT_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveDT",
                                             "Occupancy all events Coarse Exclusive DT EEP;ix;iy",
                                             20,
                                             0,
                                             100,
                                             20,
                                             0,
                                             100);
  EEP_OccupancyDT_ = new TH2F("OccupancyAllEvents_DT", "Occupancy all events DT EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseDT_ =
      new TH2F("OccupancyAllEventsCoarse_DT", "Occupancy all events Coarse DT EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_FedsTimingHistDT_ =
      new TH1F("timeForAllFeds_DT", "timeForAllFeds DT EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_FedsTimingTTHistDT_ =
      new TH3F("timeTTAllFEDs_DT",
               "(ix,iy,time) for all FEDs (SM,TT binning) DT EEP;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEP_OccupancyExclusiveRPC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveRPC", "Occupancy all events Exclusive RPC EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseExclusiveRPC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveRPC",
                                              "Occupancy all events Coarse Exclusive RPC EEP;ix;iy",
                                              20,
                                              0,
                                              100,
                                              20,
                                              0,
                                              100);
  EEP_OccupancyRPC_ =
      new TH2F("OccupancyAllEvents_RPC", "Occupancy all events RPC EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseRPC_ =
      new TH2F("OccupancyAllEventsCoarse_RPC", "Occupancy all events Coarse RPC EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_FedsTimingHistRPC_ =
      new TH1F("timeForAllFeds_RPC", "timeForAllFeds RPC EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_FedsTimingTTHistRPC_ =
      new TH3F("timeTTAllFEDs_RPC",
               "(ix,iy,time) for all FEDs (SM,TT binning) RPC EEP;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEP_OccupancyExclusiveCSC_ = new TH2F(
      "OccupancyAllEvents_ExclusiveCSC", "Occupancy all events Exclusive CSC EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseExclusiveCSC_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveCSC",
                                              "Occupancy all events Coarse Exclusive CSC EEP;ix;iy",
                                              20,
                                              0,
                                              100,
                                              20,
                                              0,
                                              100);
  EEP_OccupancyCSC_ =
      new TH2F("OccupancyAllEvents_CSC", "Occupancy all events CSC EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseCSC_ =
      new TH2F("OccupancyAllEventsCoarse_CSC", "Occupancy all events Coarse CSC EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_FedsTimingHistCSC_ =
      new TH1F("timeForAllFeds_CSC", "timeForAllFeds CSC EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_FedsTimingTTHistCSC_ =
      new TH3F("timeTTAllFEDs_CSC",
               "(ix,iy,time) for all FEDs (SM,TT binning) CSC EEP;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEP_OccupancyExclusiveHCAL_ = new TH2F(
      "OccupancyAllEvents_ExclusiveHCAL", "Occupancy all events Exclusive HCAL EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseExclusiveHCAL_ = new TH2F("OccupancyAllEventsCoarse_ExclusiveHCAL",
                                               "Occupancy all events Coarse Exclusive HCAL EEP;ix;iy",
                                               20,
                                               0,
                                               100,
                                               20,
                                               0,
                                               100);
  EEP_OccupancyHCAL_ =
      new TH2F("OccupancyAllEvents_HCAL", "Occupancy all events HCAL EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyCoarseHCAL_ =
      new TH2F("OccupancyAllEventsCoarse_HCAL", "Occupancy all events Coarse HCAL EEP;ix;iy", 20, 0, 100, 20, 0, 100);
  EEP_FedsTimingHistHCAL_ =
      new TH1F("timeForAllFeds_HCAL", "timeForAllFeds HCAL EEP;Relative Time (1 clock = 25ns)", 78, -7, 7);
  EEP_FedsTimingTTHistHCAL_ =
      new TH3F("timeTTAllFEDs_HCAL",
               "(ix,iy,time) for all FEDs (SM,TT binning) HCAL EEP;ix;iy;Relative Time (1 clock = 25ns)",
               20,
               0,
               100,
               20,
               0,
               100,
               78,
               -7,
               7);

  EEP_numberofBCinSC_ =
      new TH1F("numberofBCinSC", "Number of Basic Clusters in Super Cluster EEP;Num Basic Clusters", 20, 0, 20);

  EEP_triggerHist_ = new TH1F("triggerHist", "Trigger Number EEP", 5, 0, 5);
  EEP_triggerHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  EEP_triggerHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  EEP_triggerHist_->GetXaxis()->SetBinLabel(3, "DT");
  EEP_triggerHist_->GetXaxis()->SetBinLabel(4, "RPC");
  EEP_triggerHist_->GetXaxis()->SetBinLabel(5, "CSC");

  EEP_triggerExclusiveHist_ = new TH1F("triggerExclusiveHist", "Trigger Number (Mutually Exclusive) EEP", 5, 0, 5);
  EEP_triggerExclusiveHist_->GetXaxis()->SetBinLabel(1, "ECAL");
  EEP_triggerExclusiveHist_->GetXaxis()->SetBinLabel(2, "HCAL");
  EEP_triggerExclusiveHist_->GetXaxis()->SetBinLabel(3, "DT");
  EEP_triggerExclusiveHist_->GetXaxis()->SetBinLabel(4, "RPC");
  EEP_triggerExclusiveHist_->GetXaxis()->SetBinLabel(5, "CSC");

  EEP_NumXtalsInClusterHist_ =
      new TH1F("NumXtalsInClusterAllHist", "Number of Xtals in Cluster EEP;NumXtals", 150, 0, 150);
  EEP_numxtalsVsEnergy_ = new TH2F("NumXtalsVsEnergy",
                                   "Number of Xtals in Cluster vs Energy EEP;Energy (GeV);Number of Xtals in Cluster",
                                   numBins,
                                   histRangeMin_,
                                   10.0,
                                   150,
                                   0,
                                   150);
  EEP_numxtalsVsHighEnergy_ =
      new TH2F("NumXtalsVsHighEnergy",
               "Number of Xtals in Cluster vs Energy EEP;Energy (GeV);Number of Xtals in Cluster",
               numBins,
               histRangeMin_,
               200.,
               150,
               0,
               150);

  EEP_OccupancyHighEnergy_ =
      new TH2F("OccupancyHighEnergyEvents", "Occupancy high energy events EEP;ix;iy", 100, 0, 100, 100, 0, 100);
  EEP_OccupancyHighEnergyCoarse_ = new TH2F(
      "OccupancyHighEnergyEventsCoarse", "Occupancy high energy events Coarse EEP;ix;iy", 20, 0, 100, 20, 0, 100);

  EEP_FedsNumXtalsInClusterHist_ =
      new TH1F("NumActiveXtalsInClusterAllHist", "Number of active Xtals in Cluster EEP;NumXtals", 100, 0, 100);
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalCosmicsHists::endJob() {
  using namespace std;
  if (runInFileName_) {
    fileName_ += "-" + intToString(runNum_) + ".graph.root";
  } else {
    fileName_ += ".root";
  }

  TFile root_file_(fileName_.c_str(), "RECREATE");

  for (map<int, TH1F*>::const_iterator itr = FEDsAndHists_.begin(); itr != FEDsAndHists_.end(); ++itr) {
    string dir = fedMap_->getSliceFromFed(itr->first);
    TDirectory* FEDdir = gDirectory->mkdir(dir.c_str());
    FEDdir->cd();

    TH1F* hist = itr->second;
    if (hist != nullptr)
      hist->Write();
    else {
      cerr << "EcalCosmicsHists: Error: This shouldn't happen!" << endl;
    }
    // Write out timing hist
    hist = FEDsAndTimingHists_[itr->first];
    if (hist != nullptr)
      hist->Write();
    else {
      cerr << "EcalCosmicsHists: Error: This shouldn't happen!" << endl;
    }

    hist = FEDsAndFrequencyHists_[itr->first];
    hist->Write();

    hist = FEDsAndiPhiProfileHists_[itr->first];
    hist->Write();

    hist = FEDsAndiEtaProfileHists_[itr->first];
    hist->Write();

    hist = FEDsAndE2Hists_[itr->first];
    hist->Write();

    hist = FEDsAndenergyHists_[itr->first];
    hist->Write();

    hist = FEDsAndNumXtalsInClusterHists_[itr->first];
    hist->Write();

    TH2F* hist2 = FEDsAndTimingVsAmpHists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndTimingVsFreqHists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndE2vsE1Hists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndenergyvsE1Hists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndOccupancyHists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndTimingVsPhiHists_[itr->first];
    hist2->Write();

    hist2 = FEDsAndTimingVsModuleHists_[itr->first];
    hist2->Write();

    //2d hist
    map<int, TH2F*>::const_iterator itr2d;
    itr2d = FEDsAndDCCRuntypeVsBxHists_.find(itr->first);
    if (itr2d != FEDsAndDCCRuntypeVsBxHists_.end()) {
      TH2F* hist2 = itr2d->second;
      hist2->GetYaxis()->SetBinLabel(1, "COSMIC");
      hist2->GetYaxis()->SetBinLabel(2, "BEAMH4");
      hist2->GetYaxis()->SetBinLabel(3, "BEAMH2");
      hist2->GetYaxis()->SetBinLabel(4, "MTCC");
      hist2->GetYaxis()->SetBinLabel(5, "LASER_STD");
      hist2->GetYaxis()->SetBinLabel(6, "LASER_POWER_SCAN");
      hist2->GetYaxis()->SetBinLabel(7, "LASER_DELAY_SCAN");
      hist2->GetYaxis()->SetBinLabel(8, "TESTPULSE_SCAN_MEM");
      hist2->GetYaxis()->SetBinLabel(9, "TESTPULSE_MGPA");
      hist2->GetYaxis()->SetBinLabel(10, "PEDESTAL_STD");
      hist2->GetYaxis()->SetBinLabel(11, "PEDESTAL_OFFSET_SCAN");
      hist2->GetYaxis()->SetBinLabel(12, "PEDESTAL_25NS_SCAN");
      hist2->GetYaxis()->SetBinLabel(13, "LED_STD");
      hist2->GetYaxis()->SetBinLabel(14, "PHYSICS_GLOBAL");
      hist2->GetYaxis()->SetBinLabel(15, "COSMICS_GLOBAL");
      hist2->GetYaxis()->SetBinLabel(16, "HALO_GLOBAL");
      hist2->GetYaxis()->SetBinLabel(17, "LASER_GAP");
      hist2->GetYaxis()->SetBinLabel(18, "TESTPULSE_GAP");
      hist2->GetYaxis()->SetBinLabel(19, "PEDESTAL_GAP");
      hist2->GetYaxis()->SetBinLabel(20, "LED_GAP");
      hist2->GetYaxis()->SetBinLabel(21, "PHYSICS_LOCAL");
      hist2->GetYaxis()->SetBinLabel(22, "COSMICS_LOCAL");
      hist2->GetYaxis()->SetBinLabel(23, "HALO_LOCAL");
      hist2->GetYaxis()->SetBinLabel(24, "CALIB_LOCAL");
      hist2->Write();
    }

    root_file_.cd();
  }
  allFedsHist_->Write();
  allFedsE2Hist_->Write();
  allFedsenergyHist_->Write();
  allFedsenergyHighHist_->Write();
  allFedsenergyOnlyHighHist_->Write();
  allFedsE2vsE1Hist_->Write();
  allFedsenergyvsE1Hist_->Write();
  allFedsTimingHist_->Write();
  allFedsTimingVsAmpHist_->Write();
  allFedsFrequencyHist_->Write();
  allFedsTimingVsFreqHist_->Write();
  allFedsiEtaProfileHist_->Write();
  allFedsiPhiProfileHist_->Write();
  allOccupancy_->Write();
  TrueOccupancy_->Write();
  allOccupancyCoarse_->Write();
  TrueOccupancyCoarse_->Write();
  allOccupancyHighEnergy_->Write();
  allOccupancyHighEnergyCoarse_->Write();
  allOccupancySingleXtal_->Write();
  energySingleXtalHist_->Write();
  allFedsNumXtalsInClusterHist_->Write();
  allFedsTimingPhiHist_->Write();
  allFedsTimingPhiEbpHist_->Write();
  allFedsTimingPhiEbmHist_->Write();
  allFedsTimingEbpHist_->Write();
  allFedsTimingEbmHist_->Write();
  allFedsTimingEbpTopHist_->Write();
  allFedsTimingEbmTopHist_->Write();
  allFedsTimingEbpBottomHist_->Write();
  allFedsTimingEbmBottomHist_->Write();
  allFedsTimingPhiEtaHist_->Write();
  allFedsTimingTTHist_->Write();
  allFedsTimingLMHist_->Write();
  allFedsOccupancyHighEnergyHist_->Write();

  numberofBCinSC_->Write();         //SC
  numberofBCinSCphi_->Write();      //SC
  TrueBCOccupancyCoarse_->Write();  //BC
  TrueBCOccupancy_->Write();        //BC

  numxtalsVsEnergy_->Write();
  numxtalsVsHighEnergy_->Write();

  allOccupancyExclusiveECAL_->Write();
  allOccupancyCoarseExclusiveECAL_->Write();
  allOccupancyECAL_->Write();
  allOccupancyCoarseECAL_->Write();
  allFedsTimingPhiEtaHistECAL_->Write();
  allFedsTimingHistECAL_->Write();
  allFedsTimingTTHistECAL_->Write();
  allFedsTimingLMHistECAL_->Write();

  allOccupancyExclusiveHCAL_->Write();
  allOccupancyCoarseExclusiveHCAL_->Write();
  allOccupancyHCAL_->Write();
  allOccupancyCoarseHCAL_->Write();
  allFedsTimingPhiEtaHistHCAL_->Write();
  allFedsTimingHistHCAL_->Write();
  allFedsTimingTTHistHCAL_->Write();
  allFedsTimingLMHistHCAL_->Write();

  allOccupancyExclusiveDT_->Write();
  allOccupancyCoarseExclusiveDT_->Write();
  allOccupancyDT_->Write();
  allOccupancyCoarseDT_->Write();
  allFedsTimingPhiEtaHistDT_->Write();
  allFedsTimingHistDT_->Write();
  allFedsTimingTTHistDT_->Write();
  allFedsTimingLMHistDT_->Write();

  allOccupancyExclusiveRPC_->Write();
  allOccupancyCoarseExclusiveRPC_->Write();
  allOccupancyRPC_->Write();
  allOccupancyCoarseRPC_->Write();
  allFedsTimingPhiEtaHistRPC_->Write();
  allFedsTimingHistRPC_->Write();
  allFedsTimingTTHistRPC_->Write();
  allFedsTimingLMHistRPC_->Write();

  allOccupancyExclusiveCSC_->Write();
  allOccupancyCoarseExclusiveCSC_->Write();
  allOccupancyCSC_->Write();
  allOccupancyCoarseCSC_->Write();
  allFedsTimingPhiEtaHistCSC_->Write();
  allFedsTimingHistCSC_->Write();
  allFedsTimingTTHistCSC_->Write();
  allFedsTimingLMHistCSC_->Write();

  allFedsTimingHistEcalMuon_->Write();

  //EE
  TDirectory* EEMinusDir = gDirectory->mkdir("EEMinus");
  EEMinusDir->cd();
  EEM_FedsSeedEnergyHist_->Write();
  EEM_AllOccupancyCoarse_->Write();
  EEM_AllOccupancy_->Write();
  EEM_FedsenergyHist_->Write();
  EEM_FedsenergyHighHist_->Write();
  EEM_FedsenergyOnlyHighHist_->Write();
  EEM_FedsE2Hist_->Write();
  EEM_FedsE2vsE1Hist_->Write();
  EEM_FedsenergyvsE1Hist_->Write();
  EEM_FedsTimingHist_->Write();
  EEM_numberofCosmicsHist_->Write();
  EEM_FedsTimingVsAmpHist_->Write();
  EEM_FedsTimingTTHist_->Write();
  EEM_OccupancySingleXtal_->Write();
  EEM_energySingleXtalHist_->Write();
  EEM_OccupancyExclusiveECAL_->Write();
  EEM_OccupancyCoarseExclusiveECAL_->Write();
  EEM_OccupancyECAL_->Write();
  EEM_OccupancyCoarseECAL_->Write();
  EEM_FedsTimingHistECAL_->Write();
  EEM_FedsTimingTTHistECAL_->Write();
  EEM_OccupancyExclusiveDT_->Write();
  EEM_OccupancyCoarseExclusiveDT_->Write();
  EEM_OccupancyDT_->Write();
  EEM_OccupancyCoarseDT_->Write();
  EEM_FedsTimingHistDT_->Write();
  EEM_FedsTimingTTHistDT_->Write();
  EEM_OccupancyExclusiveRPC_->Write();
  EEM_OccupancyCoarseExclusiveRPC_->Write();
  EEM_OccupancyRPC_->Write();
  EEM_OccupancyCoarseRPC_->Write();
  EEM_FedsTimingHistRPC_->Write();
  EEM_FedsTimingTTHistRPC_->Write();
  EEM_OccupancyExclusiveCSC_->Write();
  EEM_OccupancyCoarseExclusiveCSC_->Write();
  EEM_OccupancyCSC_->Write();
  EEM_OccupancyCoarseCSC_->Write();
  EEM_FedsTimingHistCSC_->Write();
  EEM_FedsTimingTTHistCSC_->Write();
  EEM_OccupancyExclusiveHCAL_->Write();
  EEM_OccupancyCoarseExclusiveHCAL_->Write();
  EEM_OccupancyHCAL_->Write();
  EEM_OccupancyCoarseHCAL_->Write();
  EEM_FedsTimingHistHCAL_->Write();
  EEM_FedsTimingTTHistHCAL_->Write();
  EEM_numberofBCinSC_->Write();
  EEM_triggerHist_->Write();
  EEM_triggerExclusiveHist_->Write();
  EEM_NumXtalsInClusterHist_->Write();
  EEM_numxtalsVsEnergy_->Write();
  EEM_numxtalsVsHighEnergy_->Write();
  EEM_OccupancyHighEnergy_->Write();
  EEM_OccupancyHighEnergyCoarse_->Write();
  EEM_FedsNumXtalsInClusterHist_->Write();
  root_file_.cd();

  TDirectory* EEPlusDir = gDirectory->mkdir("EEPlus");
  EEPlusDir->cd();
  EEP_FedsSeedEnergyHist_->Write();
  EEP_AllOccupancyCoarse_->Write();
  EEP_AllOccupancy_->Write();
  EEP_FedsenergyHist_->Write();
  EEP_FedsenergyHighHist_->Write();
  EEP_FedsenergyOnlyHighHist_->Write();
  EEP_FedsE2Hist_->Write();
  EEP_FedsE2vsE1Hist_->Write();
  EEP_FedsenergyvsE1Hist_->Write();
  EEP_FedsTimingHist_->Write();
  EEP_numberofCosmicsHist_->Write();
  EEP_FedsTimingVsAmpHist_->Write();
  EEP_FedsTimingTTHist_->Write();
  EEP_OccupancySingleXtal_->Write();
  EEP_energySingleXtalHist_->Write();
  EEP_OccupancyExclusiveECAL_->Write();
  EEP_OccupancyCoarseExclusiveECAL_->Write();
  EEP_OccupancyECAL_->Write();
  EEP_OccupancyCoarseECAL_->Write();
  EEP_FedsTimingHistECAL_->Write();
  EEP_FedsTimingTTHistECAL_->Write();
  EEP_OccupancyExclusiveDT_->Write();
  EEP_OccupancyCoarseExclusiveDT_->Write();
  EEP_OccupancyDT_->Write();
  EEP_OccupancyCoarseDT_->Write();
  EEP_FedsTimingHistDT_->Write();
  EEP_FedsTimingTTHistDT_->Write();
  EEP_OccupancyExclusiveRPC_->Write();
  EEP_OccupancyCoarseExclusiveRPC_->Write();
  EEP_OccupancyRPC_->Write();
  EEP_OccupancyCoarseRPC_->Write();
  EEP_FedsTimingHistRPC_->Write();
  EEP_FedsTimingTTHistRPC_->Write();
  EEP_OccupancyExclusiveCSC_->Write();
  EEP_OccupancyCoarseExclusiveCSC_->Write();
  EEP_OccupancyCSC_->Write();
  EEP_OccupancyCoarseCSC_->Write();
  EEP_FedsTimingHistCSC_->Write();
  EEP_FedsTimingTTHistCSC_->Write();
  EEP_OccupancyExclusiveHCAL_->Write();
  EEP_OccupancyCoarseExclusiveHCAL_->Write();
  EEP_OccupancyHCAL_->Write();
  EEP_OccupancyCoarseHCAL_->Write();
  EEP_FedsTimingHistHCAL_->Write();
  EEP_FedsTimingTTHistHCAL_->Write();
  EEP_numberofBCinSC_->Write();
  EEP_triggerHist_->Write();
  EEP_triggerExclusiveHist_->Write();
  EEP_NumXtalsInClusterHist_->Write();
  EEP_numxtalsVsEnergy_->Write();
  EEP_numxtalsVsHighEnergy_->Write();
  EEP_OccupancyHighEnergy_->Write();
  EEP_OccupancyHighEnergyCoarse_->Write();
  EEP_FedsNumXtalsInClusterHist_->Write();
  root_file_.cd();

  triggerHist_->Write();
  triggerExclusiveHist_->Write();

  NumXtalsInClusterHist_->Write();

  numberofCosmicsHist_->Write();
  numberofCosmicsHistEB_->Write();

  numberofCosmicsWTrackHist_->Write();
  numberofCosmicsTopBottomHist_->Write();
  numberofGoodEvtFreq_->Write();
  numberofCrossedEcalIdsHist_->Write();

  runNumberHist_->SetBinContent(1, runNum_);
  runNumberHist_->Write();

  deltaRHist_->Write();
  deltaEtaHist_->Write();
  deltaPhiHist_->Write();
  ratioAssocClustersHist_->Write();
  ratioAssocTracksHist_->Write();
  deltaEtaDeltaPhiHist_->Write();
  seedTrackPhiHist_->Write();
  seedTrackEtaHist_->Write();
  dccEventVsBxHist_->Write();
  dccOrbitErrorByFEDHist_->Write();
  dccBXErrorByFEDHist_->Write();
  dccRuntypeErrorByFEDHist_->Write();
  dccErrorVsBxHist_->Write();
  dccRuntypeHist_->Write();

  trackAssoc_muonsEcal_->Write();

  hcalEnergy_HBHE_->Write();
  hcalEnergy_HF_->Write();
  hcalEnergy_HO_->Write();
  hcalHEHBecalEB_->Write();

  TDirectory* highEnergyDir = gDirectory->mkdir("HighEnergy");
  highEnergyDir->cd();
  HighEnergy_NumXtal->Write();
  HighEnergy_NumXtalFedId->Write();
  HighEnergy_NumXtaliphi->Write();
  HighEnergy_energy3D->Write();
  HighEnergy_energyNumXtal->Write();
  HighEnergy_bestSeed->Write();
  HighEnergy_bestSeedOccupancy->Write();
  HighEnergy_numClusHighEn->Write();
  HighEnergy_ratioClusters->Write();
  HighEnergy_numRecoTrackBarrel->Write();
  HighEnergy_TracksAngle->Write();
  HighEnergy_TracksAngleTopBottom->Write();
  HighEnergy_2GeV_occuCoarse->Write();
  HighEnergy_2GeV_occu3D->Write();
  HighEnergy_100GeV_occuCoarse->Write();
  HighEnergy_100GeV_occu3D->Write();
  HighEnergy_0tracks_occu3D->Write();
  HighEnergy_1tracks_occu3D->Write();
  HighEnergy_2tracks_occu3D->Write();
  HighEnergy_0tracks_occu3DXtal->Write();
  HighEnergy_1tracks_occu3DXtal->Write();
  HighEnergy_2tracks_occu3DXtal->Write();

  root_file_.cd();

  TDirectory* TimeStampdir = gDirectory->mkdir("EventTiming");
  TimeStampdir->cd();
  allFedsFreqTimeHist_->Write();
  allFedsFreqTimeVsPhiHist_->Write();
  allFedsFreqTimeVsPhiTTHist_->Write();
  allFedsFreqTimeVsEtaHist_->Write();
  allFedsFreqTimeVsEtaTTHist_->Write();

  root_file_.cd();

  root_file_.Close();

  LogWarning("EcalCosmicsHists") << "---> Number of cosmic events: " << cosmicCounter_ << " in " << naiveEvtNum_
                                 << " events.";
  LogWarning("EcalCosmicsHists") << "---> Number of EB cosmic events: " << cosmicCounterEB_ << " in " << naiveEvtNum_
                                 << " events.";
  LogWarning("EcalCosmicsHists") << "---> Number of EE- cosmic events: " << cosmicCounterEEM_ << " in " << naiveEvtNum_
                                 << " events.";
  LogWarning("EcalCosmicsHists") << "---> Number of EE+ cosmic events: " << cosmicCounterEEP_ << " in " << naiveEvtNum_
                                 << " events.";

  //  LogWarning("EcalCosmicsHists") << "---> Number of top+bottom cosmic events: " << cosmicCounterTopBottom_ << " in " << cosmicCounter_ << " cosmics in " << naiveEvtNum_ << " events.";
}

std::string EcalCosmicsHists::intToString(int num) {
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return (myStream.str());  //returns the string form of the stringstream object
}
