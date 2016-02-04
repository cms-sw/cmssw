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
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalCosmicsHists.h,v 1.4 2010/01/04 15:07:39 ferriff Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TGraph.h"
#include "TNtuple.h"


// *** for TrackAssociation
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
// ***

//
// class declaration
//

class EcalCosmicsHists : public edm::EDAnalyzer {
   public:
      explicit EcalCosmicsHists(const edm::ParameterSet&);
      ~EcalCosmicsHists();


   private:
      virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;
      virtual void analyze(edm::Event const &, edm::EventSetup const &);
      virtual void endJob() ;
      std::string intToString(int num);
      void initHists(int);
      std::vector<bool> determineTriggers(const edm::Event&, const edm::EventSetup& eventSetup);

  // ----------member data ---------------------------
  
  edm::InputTag ecalRawDataColl_;
  edm::InputTag ecalRecHitCollectionEB_;
  edm::InputTag ecalRecHitCollectionEE_;
  edm::InputTag barrelClusterCollection_;
  edm::InputTag endcapClusterCollection_;
  edm::InputTag l1GTReadoutRecTag_;
  edm::InputTag l1GMTReadoutRecTag_;
  
  int runNum_;
  double histRangeMax_, histRangeMin_;
  double minTimingAmpEB_;
  double minTimingAmpEE_;
  double minRecHitAmpEB_;
  double minRecHitAmpEE_;
  double minHighEnergy_;
  
  double *ttEtaBins;
  double *modEtaBins;
  std::string fileName_;
  bool runInFileName_;

  double startTime_, runTimeLength_;
  int numTimingBins_; 
  
  std::map<int,TH1F*> FEDsAndHists_;
  std::map<int,TH1F*> FEDsAndE2Hists_;
  std::map<int,TH1F*> FEDsAndenergyHists_;
  std::map<int,TH1F*> FEDsAndTimingHists_;
  std::map<int,TH1F*> FEDsAndFrequencyHists_;
  std::map<int,TH1F*> FEDsAndiPhiProfileHists_;
  std::map<int,TH1F*> FEDsAndNumXtalsInClusterHists_;
  std::map<int,TH1F*> FEDsAndiEtaProfileHists_;
  std::map<int,TH2F*> FEDsAndTimingVsFreqHists_;
  std::map<int,TH2F*> FEDsAndTimingVsAmpHists_;
  std::map<int,TH2F*> FEDsAndE2vsE1Hists_;
  std::map<int,TH2F*> FEDsAndenergyvsE1Hists_;
  std::map<int,TH2F*> FEDsAndOccupancyHists_;  
  std::map<int,TH2F*> FEDsAndTimingVsPhiHists_;  
  std::map<int,TH2F*> FEDsAndTimingVsEtaHists_;  
  std::map<int,TH2F*> FEDsAndTimingVsModuleHists_;  
  std::map<int,TH2F*> FEDsAndDCCRuntypeVsBxHists_;

  TH1F* allFedsHist_;
  TH1F* allFedsE2Hist_;
  TH1F* allFedsenergyHist_;
  TH1F* allFedsenergyHighHist_;
  TH1F* allFedsenergyOnlyHighHist_;
  TH1F* allFedsTimingHist_;
  TH1F* allFedsFrequencyHist_;
  TH1F* allFedsiPhiProfileHist_;
  TH1F* allFedsiEtaProfileHist_;
  TH1F* allFedsNumXtalsInClusterHist_;
  TH1F* NumXtalsInClusterHist_;
  TH1F* numberofCosmicsHist_;
  TH1F* numberofCosmicsWTrackHist_;
  TH1F* numberofGoodEvtFreq_;

  TH1F* numberofCosmicsHistEB_;

  //TH1F* numberofSCCosmicsHist_;//SC Num cosmics
  TH1F* numberofBCinSC_;//SC Num cosmics
  TH2F* numberofBCinSCphi_;//SC

  TH2F* TrueBCOccupancy_;
  TH2F* TrueBCOccupancyCoarse_;

  TH2F* numxtalsVsEnergy_;
  TH2F* numxtalsVsHighEnergy_;

  TH2F* allFedsE2vsE1Hist_;
  TH2F* allFedsenergyvsE1Hist_;
  TH2F* allOccupancy_; //New file to do eta-phi occupancy
  TH2F* TrueOccupancy_; //New file to do eta-phi occupancy
  TH2F* allOccupancyCoarse_; //New file to do eta-phi occupancy
  TH2F* TrueOccupancyCoarse_; //New file to do eta-phi occupancy

  // single xtal clusters
  TH2F* allOccupancySingleXtal_;
  TH1F* energySingleXtalHist_;

  TH2F* allFedsTimingVsFreqHist_;
  TH2F* allFedsTimingVsAmpHist_;
  TH2F* allFedsTimingPhiHist_;
  TH2F* allFedsTimingEtaHist_;
  TH2F* allFedsTimingPhiEbpHist_;
  TH2F* allFedsTimingPhiEbmHist_;
  TH3F* allFedsTimingPhiEtaHist_;
  TH3F* allFedsTimingTTHist_;
  TH2F* allFedsTimingLMHist_;

  TH1F* allFedsTimingEbpHist_;
  TH1F* allFedsTimingEbmHist_;
  TH1F* allFedsTimingEbpTopHist_;
  TH1F* allFedsTimingEbmTopHist_;
  TH1F* allFedsTimingEbpBottomHist_;
  TH1F* allFedsTimingEbmBottomHist_;

  TH2F* allOccupancyECAL_; 
  TH2F* allOccupancyCoarseECAL_; 
  TH2F* allOccupancyExclusiveECAL_; 
  TH2F* allOccupancyCoarseExclusiveECAL_; 
  TH1F* allFedsTimingHistECAL_;
  TH3F* allFedsTimingPhiEtaHistECAL_;
  TH3F* allFedsTimingTTHistECAL_;
  TH2F* allFedsTimingLMHistECAL_;

  TH2F* allOccupancyDT_; 
  TH2F* allOccupancyCoarseDT_; 
  TH2F* allOccupancyExclusiveDT_; 
  TH2F* allOccupancyCoarseExclusiveDT_; 
  TH1F* allFedsTimingHistDT_;
  TH3F* allFedsTimingPhiEtaHistDT_;
  TH3F* allFedsTimingTTHistDT_;
  TH2F* allFedsTimingLMHistDT_;

  TH2F* allOccupancyRPC_; 
  TH2F* allOccupancyCoarseRPC_; 
  TH2F* allOccupancyExclusiveRPC_; 
  TH2F* allOccupancyCoarseExclusiveRPC_; 
  TH1F* allFedsTimingHistRPC_;
  TH3F* allFedsTimingPhiEtaHistRPC_;
  TH3F* allFedsTimingTTHistRPC_;
  TH2F* allFedsTimingLMHistRPC_;

  TH2F* allOccupancyCSC_; 
  TH2F* allOccupancyCoarseCSC_; 
  TH2F* allOccupancyExclusiveCSC_; 
  TH2F* allOccupancyCoarseExclusiveCSC_; 
  TH1F* allFedsTimingHistCSC_;
  TH3F* allFedsTimingPhiEtaHistCSC_;
  TH3F* allFedsTimingTTHistCSC_;
  TH2F* allFedsTimingLMHistCSC_;

  TH2F* allOccupancyHCAL_; 
  TH2F* allOccupancyCoarseHCAL_; 
  TH2F* allOccupancyExclusiveHCAL_; 
  TH2F* allOccupancyCoarseExclusiveHCAL_; 
  TH1F* allFedsTimingHistHCAL_;
  TH3F* allFedsTimingPhiEtaHistHCAL_;
  TH3F* allFedsTimingTTHistHCAL_;
  TH2F* allFedsTimingLMHistHCAL_;

  TH1F* allFedsTimingHistEcalMuon_;
  
  TH1F* triggerHist_;
  TH1F* triggerExclusiveHist_;

  TH2F* allOccupancyHighEnergy_; 
  TH2F* allOccupancyHighEnergyCoarse_; 
  TH3F* allFedsOccupancyHighEnergyHist_;

  TH1F* runNumberHist_;
  TH1F* deltaRHist_;
  TH1F* deltaEtaHist_;
  TH1F* deltaPhiHist_;
  TH1F* ratioAssocTracksHist_;
  TH1F* ratioAssocClustersHist_;
  TH2F* deltaEtaDeltaPhiHist_;
  TH2F* seedTrackPhiHist_;
  TH2F* seedTrackEtaHist_;

  // DCC Event type (runtype) vs. bx
  TH2F* dccEventVsBxHist_;
  TH1F* dccBXErrorByFEDHist_;
  TH1F* dccOrbitErrorByFEDHist_;
  TH1F* dccRuntypeErrorByFEDHist_;
  TH1F* dccRuntypeHist_;
  TH2F* dccErrorVsBxHist_;
  
  // track association
  TH2F* trackAssoc_muonsEcal_;
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters trackParameters_;
  TH1F* numberofCrossedEcalIdsHist_;

  // top & bottom
  TH1F* numberofCosmicsTopBottomHist_;
  int cosmicCounterTopBottom_;

  // hcal energy
  TH1F * hcalEnergy_HBHE_;
  TH1F * hcalEnergy_HF_;
  TH1F * hcalEnergy_HO_;
  TH2F * hcalHEHBecalEB_;

  // high energy analysis (inherited from serena)
  TH1F* HighEnergy_NumXtal;
  TH2F* HighEnergy_NumXtalFedId;
  TH2F* HighEnergy_NumXtaliphi;
  TH3F* HighEnergy_energy3D;
  TH2F* HighEnergy_energyNumXtal;

  TH1F* HighEnergy_numClusHighEn;
  TH1F* HighEnergy_ratioClusters;

  TH1F* HighEnergy_bestSeed;
  TH2F* HighEnergy_bestSeedOccupancy;

  TH2F* HighEnergy_2GeV_occuCoarse;
  TH3F* HighEnergy_2GeV_occu3D;
  TH2F* HighEnergy_100GeV_occuCoarse;
  TH3F* HighEnergy_100GeV_occu3D;

  TH1F* HighEnergy_numRecoTrackBarrel;
  TH1F* HighEnergy_TracksAngle;
  TH1F* HighEnergy_TracksAngleTopBottom;
  TH3F* HighEnergy_0tracks_occu3D;
  TH3F* HighEnergy_1tracks_occu3D;
  TH3F* HighEnergy_2tracks_occu3D;

  TH3F* HighEnergy_0tracks_occu3DXtal;
  TH3F* HighEnergy_1tracks_occu3DXtal;
  TH3F* HighEnergy_2tracks_occu3DXtal;
  
  //for timestamp information
  TH1F* allFedsFreqTimeHist_;
  TH2F* allFedsFreqTimeVsPhiHist_;
  TH2F* allFedsFreqTimeVsPhiTTHist_;
  TH2F* allFedsFreqTimeVsEtaHist_;
  TH2F* allFedsFreqTimeVsEtaTTHist_;
  
  // Plots for EE
  TH2F* EEP_AllOccupancyCoarse_;
  TH2F* EEP_AllOccupancy_;
  TH1F* EEP_FedsenergyHist_;
  TH1F* EEP_FedsenergyHighHist_;
  TH1F* EEP_FedsenergyOnlyHighHist_;
  TH1F* EEP_FedsE2Hist_;
  TH2F* EEP_FedsE2vsE1Hist_;
  TH2F* EEP_FedsenergyvsE1Hist_;
  TH1F* EEP_FedsSeedEnergyHist_;

  TH1F* EEP_FedsNumXtalsInClusterHist_;
  TH1F* EEP_NumXtalsInClusterHist_;
  TH2F* EEP_numxtalsVsEnergy_;
  TH2F* EEP_numxtalsVsHighEnergy_;
  TH1F* EEP_numberofBCinSC_;

  TH1F* EEP_numberofCosmicsHist_;

  TH2F* EEP_OccupancySingleXtal_;
  TH1F* EEP_energySingleXtalHist_;

  TH1F* EEP_FedsTimingHist_;        
  TH2F* EEP_FedsTimingVsAmpHist_;
  TH3F* EEP_FedsTimingTTHist_;

  TH2F* EEP_OccupancyECAL_; 
  TH2F* EEP_OccupancyCoarseECAL_; 
  TH2F* EEP_OccupancyExclusiveECAL_; 
  TH2F* EEP_OccupancyCoarseExclusiveECAL_; 
  TH1F* EEP_FedsTimingHistECAL_;
  TH3F* EEP_FedsTimingTTHistECAL_;

  TH2F* EEP_OccupancyDT_; 
  TH2F* EEP_OccupancyCoarseDT_; 
  TH2F* EEP_OccupancyExclusiveDT_; 
  TH2F* EEP_OccupancyCoarseExclusiveDT_; 
  TH1F* EEP_FedsTimingHistDT_;
  TH3F* EEP_FedsTimingTTHistDT_;

  TH2F* EEP_OccupancyRPC_; 
  TH2F* EEP_OccupancyCoarseRPC_; 
  TH2F* EEP_OccupancyExclusiveRPC_; 
  TH2F* EEP_OccupancyCoarseExclusiveRPC_; 
  TH1F* EEP_FedsTimingHistRPC_;
  TH3F* EEP_FedsTimingTTHistRPC_;

  TH2F* EEP_OccupancyCSC_; 
  TH2F* EEP_OccupancyCoarseCSC_; 
  TH2F* EEP_OccupancyExclusiveCSC_; 
  TH2F* EEP_OccupancyCoarseExclusiveCSC_; 
  TH1F* EEP_FedsTimingHistCSC_;
  TH3F* EEP_FedsTimingTTHistCSC_;

  TH2F* EEP_OccupancyHCAL_; 
  TH2F* EEP_OccupancyCoarseHCAL_; 
  TH2F* EEP_OccupancyExclusiveHCAL_; 
  TH2F* EEP_OccupancyCoarseExclusiveHCAL_; 
  TH1F* EEP_FedsTimingHistHCAL_;
  TH3F* EEP_FedsTimingTTHistHCAL_;

  TH1F* EEP_triggerHist_;
  TH1F* EEP_triggerExclusiveHist_;

  TH2F* EEP_OccupancyHighEnergy_; 
  TH2F* EEP_OccupancyHighEnergyCoarse_; 

  // EE-
  TH2F* EEM_AllOccupancyCoarse_;
  TH2F* EEM_AllOccupancy_;
  TH1F* EEM_FedsenergyHist_;
  TH1F* EEM_FedsenergyHighHist_;
  TH1F* EEM_FedsenergyOnlyHighHist_;
  TH1F* EEM_FedsE2Hist_;
  TH2F* EEM_FedsE2vsE1Hist_;
  TH2F* EEM_FedsenergyvsE1Hist_;
  TH1F* EEM_FedsSeedEnergyHist_;

  TH1F* EEM_FedsNumXtalsInClusterHist_;
  TH1F* EEM_NumXtalsInClusterHist_;
  TH2F* EEM_numxtalsVsEnergy_;
  TH2F* EEM_numxtalsVsHighEnergy_;
  TH1F* EEM_numberofBCinSC_;

  TH1F* EEM_numberofCosmicsHist_;

  TH2F* EEM_OccupancySingleXtal_;
  TH1F* EEM_energySingleXtalHist_;

  TH1F* EEM_FedsTimingHist_;        
  TH2F* EEM_FedsTimingVsAmpHist_;
  TH3F* EEM_FedsTimingTTHist_;

  TH2F* EEM_OccupancyECAL_; 
  TH2F* EEM_OccupancyCoarseECAL_; 
  TH2F* EEM_OccupancyExclusiveECAL_; 
  TH2F* EEM_OccupancyCoarseExclusiveECAL_; 
  TH1F* EEM_FedsTimingHistECAL_;
  TH3F* EEM_FedsTimingTTHistECAL_;

  TH2F* EEM_OccupancyDT_; 
  TH2F* EEM_OccupancyCoarseDT_; 
  TH2F* EEM_OccupancyExclusiveDT_; 
  TH2F* EEM_OccupancyCoarseExclusiveDT_; 
  TH1F* EEM_FedsTimingHistDT_;
  TH3F* EEM_FedsTimingTTHistDT_;

  TH2F* EEM_OccupancyRPC_; 
  TH2F* EEM_OccupancyCoarseRPC_; 
  TH2F* EEM_OccupancyExclusiveRPC_; 
  TH2F* EEM_OccupancyCoarseExclusiveRPC_; 
  TH1F* EEM_FedsTimingHistRPC_;
  TH3F* EEM_FedsTimingTTHistRPC_;

  TH2F* EEM_OccupancyCSC_; 
  TH2F* EEM_OccupancyCoarseCSC_; 
  TH2F* EEM_OccupancyExclusiveCSC_; 
  TH2F* EEM_OccupancyCoarseExclusiveCSC_; 
  TH1F* EEM_FedsTimingHistCSC_;
  TH3F* EEM_FedsTimingTTHistCSC_;

  TH2F* EEM_OccupancyHCAL_; 
  TH2F* EEM_OccupancyCoarseHCAL_; 
  TH2F* EEM_OccupancyExclusiveHCAL_; 
  TH2F* EEM_OccupancyCoarseExclusiveHCAL_; 
  TH1F* EEM_FedsTimingHistHCAL_;
  TH3F* EEM_FedsTimingTTHistHCAL_;
  
  TH1F* EEM_triggerHist_;
  TH1F* EEM_triggerExclusiveHist_;

  TH2F* EEM_OccupancyHighEnergy_; 
  TH2F* EEM_OccupancyHighEnergyCoarse_; 

  EcalFedMap* fedMap_;

  TFile* file;
  
  int naiveEvtNum_; 
  int cosmicCounter_;
  int cosmicCounterEB_;
  int cosmicCounterEEP_;
  int cosmicCounterEEM_;

  std::vector<int> l1Accepts_;
  std::vector<std::string> l1Names_;

  const EcalElectronicsMapping* ecalElectronicsMap_;

};
