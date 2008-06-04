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
// $Id: EcalCosmicsHists.h,v 1.12 2008/06/02 15:56:56 torimoto Exp $
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
// ***

//
// class declaration
//

class EcalCosmicsHists : public edm::EDAnalyzer {
   public:
      explicit EcalCosmicsHists(const edm::ParameterSet&);
      ~EcalCosmicsHists();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string intToString(int num);
      void initHists(int);

  // ----------member data ---------------------------
  
  edm::InputTag ecalRecHitCollection_;
  edm::InputTag barrelClusterCollection_;
  edm::InputTag endcapClusterCollection_;
  edm::InputTag l1GTReadoutRecTag_;
  
  
  int runNum_;
  double histRangeMax_, histRangeMin_;
  double minTimingAmp_;
  
  double *ttEtaBins;
  double *modEtaBins;
  std::string fileName_;
  
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

  TH1F* allFedsHist_;
  TH1F* allFedsE2Hist_;
  TH1F* allFedsenergyHist_;
  TH1F* allFedsenergyHighHist_;
  TH1F* allFedsTimingHist_;
  TH1F* allFedsFrequencyHist_;
  TH1F* allFedsiPhiProfileHist_;
  TH1F* allFedsiEtaProfileHist_;
  TH1F* allFedsNumXtalsInClusterHist_;
  TH1F* numberofCosmicsHist_;
  TH1F* numberofCosmicsWTrackHist_;
  TH1F* numberofGoodEvtFreq_;

  TH2F* allFedsE2vsE1Hist_;
  TH2F* allFedsenergyvsE1Hist_;
  TH2F* allOccupancy_; //New file to do eta-phi occupancy
  TH2F* TrueOccupancy_; //New file to do eta-phi occupancy
  TH2F* allOccupancyCoarse_; //New file to do eta-phi occupancy
  TH2F* TrueOccupancyCoarse_; //New file to do eta-phi occupancy

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

  // track association
  TH2F* trackAssoc_muonsEcal_;
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters trackParameters_;
  TH1F* numberofCrossedEcalIdsHist_;

  // top & bottom
  TH1F* numberofCosmicsTopBottomHist_;
  int cosmicCounterTopBottom_;

  EcalFedMap* fedMap_;

  TFile* file;
  
  int naiveEvtNum_; 
  int cosmicCounter_;

  std::vector<int> l1Accepts_;
  std::vector<std::string> l1Names_;

};
