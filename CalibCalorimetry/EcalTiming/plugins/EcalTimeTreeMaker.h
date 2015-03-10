#ifndef EcalTimeTreeMaker_h
#define EcalTimeTreeMaker_h

// -*- C++ -*-
//
// Package:   EcalTimeTreeMaker
// Class:     EcalTimeTreeMaker
//
/**\class EcalTimeTreeMaker EcalTimeTreeMaker.h

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Skeleton Derived from an example by:  F. DE GUIO C. DOGLIONI P. MERIDIANI
// Authors:                              Seth Cooper, Giovanni Franzoni (UMN)
//         Created:  Mo Jul 14 5:46:22 CEST 2008
// $Id: EcalTimeTreeMaker.h,v 1.5 2011/09/10 11:03:44 franzoni Exp $
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

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TGraph.h"
#include "TTree.h"

#include <vector>

// *** for TrackAssociation
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
// ***

//for track length
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
//
#include "CalibCalorimetry/EcalTiming/interface/EcalTimeTreeContent.h"

// containers for vertices
#include <DataFormats/VertexReco/interface/VertexFwd.h>


class EcalTimeTreeMaker : public edm::EDAnalyzer 
{
   public:

      explicit EcalTimeTreeMaker (const edm::ParameterSet&) ;
      ~EcalTimeTreeMaker () ;

   protected:

      virtual void beginJob () {} ;
      virtual void analyze (const edm::Event&, const edm::EventSetup&) ;
      virtual void endJob () ;
      virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;

  private:

      std::string intToString (int num) ;
      void initHists (int) ;

      //! dump Cluster information
      //! has to run after dumpMUinfo, to have the XtalMap already filled
      void dumpBarrelClusterInfo(const edm::Event& iEvent,
				 const CaloGeometry * theGeometry,
				 const CaloTopology * theCaloTopology,
				 const EcalRecHitCollection* theBarrelEcalRecHits,
				 const reco::BasicClusterCollection* theBarrelBasicClusters,
				 const reco::SuperClusterCollection* theBarrelSuperClusters,
				 EcalClusterLazyTools* lazyTools,
				 const std::map<int,float> & XtalMap,
				 const std::map<int,float> & XtalMapCurved,
				 EcalTimeTreeContent & myTreeVariables_) ;

      void dumpEndcapClusterInfo(const edm::Event& iEvent,
				 const CaloGeometry * theGeometry,
				 const CaloTopology * theCaloTopology,
				 const EcalRecHitCollection* theEndcapEcalRecHits,
				 const reco::BasicClusterCollection* theEndcapBasicClusters,
				 const reco::SuperClusterCollection* theEndcapSuperClusters,
				 EcalClusterLazyTools* lazyTools,
				 const std::map<int,float> & XtalMap,
				 const std::map<int,float> & XtalMapCurved,
				 EcalTimeTreeContent & myTreeVariables_) ;
      
      void dumpVertexInfo(const reco::VertexCollection* recVtxs, EcalTimeTreeContent & myTreeVariables_);
 
      //! dump trigger information
      void dump3Ginfo (const edm::Event& iEvent, const edm::EventSetup& eventSetup,
                       EcalTimeTreeContent & myTreeVariables_) ;
        //! collect trigger information to be dumped
      std::vector<bool> determineTriggers (const edm::Event& iEvent, 
                                           const edm::EventSetup& eventSetup, int Bx=0) ;



    // ----------member data ---------------------------

   protected:

      edm::InputTag barrelEcalRecHitCollection_ ;
      edm::InputTag endcapEcalRecHitCollection_ ;
      edm::InputTag barrelBasicClusterCollection_ ;
      edm::InputTag endcapBasicClusterCollection_ ;
      edm::InputTag barrelSuperClusterCollection_ ;
      edm::InputTag endcapSuperClusterCollection_ ;
      edm::InputTag muonCollection_ ;
      edm::InputTag vertexCollection_ ;
      edm::InputTag l1GMTReadoutRecTag_ ;
      edm::InputTag gtRecordCollectionTag_ ;
      int runNum_ ;
      double minEtEB_;
      double minEtEE_;
      std::string fileName_ ;
      int  naiveId_ ; 

      TrackDetectorAssociator trackAssociator_ ;
      TrackAssociatorParameters trackParameters_ ;

      EcalTimeTreeContent myTreeVariables_ ;

      TFile* file_ ;
      TTree* tree_ ;
      
      std::vector<int> l1Accepts_ ;
      std::vector<std::string> l1Names_ ;

      double hbTreshold_;
      edm::ESHandle<EcalIntercalibConstants> ical;
      edm::ESHandle<EcalADCToGeVConstant> agc;
      edm::ESHandle<EcalLaserDbService> laser;

} ;

#endif
