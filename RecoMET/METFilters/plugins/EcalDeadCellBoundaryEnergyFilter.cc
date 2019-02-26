// -*- C++ -*-
//
// Package:    EcalDeadCellBoundaryEnergyFilter
// Class:      EcalDeadCellBoundaryEnergyFilter
//
/**\class EcalDeadCellBoundaryEnergyFilter EcalDeadCellBoundaryEnergyFilter.cc PhysicsTools/EcalDeadCellBoundaryEnergyFilter/src/EcalDeadCellBoundaryEnergyFilter.cc

 Description: <one line class summary>
 Event filtering for anomalous ECAL events where the energy measured by ECAL is significantly biased due to energy depositions
 in passive or problematic detector regions. The filter will handle ECAL flags and will compute the boundary energy in the channels
 surrounding the problematic regions such as dead channels and gaps.

 // Filter Algos :
 // a)  "TuningMode" keep all events and save event info in a ROOT TTree for tuning/algo development
 // b)  "FilterMode" returns false for all events passing the AnomalousEcalVariables.isDeadEcalCluster() function (--->rejects events affected by energy deposits in Dead Cells)

 Implementation:
 <Notes on implementation>
 */
//
// Original Author:  Konstantinos Theofilatos, Ulla Gebbert and Christian Sander
//         Created:  Sat Nov 14 18:43:21 CET 2009
//
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoMET/METFilters/interface/EcalBoundaryInfoCalculator.h"
#include "DataFormats/METReco/interface/AnomalousECALVariables.h"

class EcalDeadCellBoundaryEnergyFilter: public edm::global::EDFilter<> {
   public:
      explicit EcalDeadCellBoundaryEnergyFilter(const edm::ParameterSet&);
      ~EcalDeadCellBoundaryEnergyFilter() override;

   private:
      void beginJob() override;
      bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      void endJob() override;

      // ----------member data ---------------------------
      const int kMAX;

      edm::EDGetTokenT<EcalRecHitCollection> EBRecHitsToken_;
      edm::EDGetTokenT<EcalRecHitCollection> EERecHitsToken_;

      const std::string FilterAlgo_;
      const bool taggingMode_;

      const bool skimGap_;
      const bool skimDead_;

      const double cutBoundEnergyGapEE, cutBoundEnergyGapEB, cutBoundEnergyDeadCellsEB, cutBoundEnergyDeadCellsEE;

      EcalBoundaryInfoCalculator<EBDetId> ebBoundaryCalc;
      EcalBoundaryInfoCalculator<EEDetId> eeBoundaryCalc;

      double maxBoundaryEnergy_;

      const bool limitFilterToEB_, limitFilterToEE_;
      const std::vector<int> limitDeadCellToChannelStatusEB_;
      const std::vector<int> limitDeadCellToChannelStatusEE_;

      const bool enableGap_;
      const bool debug_;

};

//
// static data member definitions
//

//
// constructors and destructor
//
EcalDeadCellBoundaryEnergyFilter::EcalDeadCellBoundaryEnergyFilter(const edm::ParameterSet& iConfig)
   : kMAX (50)
   //now do what ever initialization is needed
   , EBRecHitsToken_ (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag> ("recHitsEB")))
   , EERecHitsToken_ (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag> ("recHitsEE")))

   , FilterAlgo_ (iConfig.getUntrackedParameter<std::string> ("FilterAlgo", "TuningMode"))
   , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
   , skimGap_ (iConfig.getUntrackedParameter<bool> ("skimGap", false))
   , skimDead_ (iConfig.getUntrackedParameter<bool> ("skimDead", false))
   , cutBoundEnergyGapEE (iConfig.getUntrackedParameter<double> ("cutBoundEnergyGapEE"))
   , cutBoundEnergyGapEB (iConfig.getUntrackedParameter<double> ("cutBoundEnergyGapEB"))
   , cutBoundEnergyDeadCellsEB (iConfig.getUntrackedParameter<double> ("cutBoundEnergyDeadCellsEB"))
   , cutBoundEnergyDeadCellsEE (iConfig.getUntrackedParameter<double> ("cutBoundEnergyDeadCellsEE"))

   , limitFilterToEB_ (iConfig.getUntrackedParameter<bool> ("limitFilterToEB", false))
   , limitFilterToEE_ (iConfig.getUntrackedParameter<bool> ("limitFilterToEE", false))
   , limitDeadCellToChannelStatusEB_ (iConfig.getParameter<std::vector<int> > ("limitDeadCellToChannelStatusEB"))
   , limitDeadCellToChannelStatusEE_ (iConfig.getParameter<std::vector<int> > ("limitDeadCellToChannelStatusEE"))

   , enableGap_ (iConfig.getUntrackedParameter<bool> ("enableGap", false))
   , debug_ (iConfig.getParameter<bool>("debug"))
{

   maxBoundaryEnergy_ = cutBoundEnergyDeadCellsEB;

   if (skimGap_ && debug_ ) edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "Skim Gap!";
   if (skimDead_ && debug_ ) edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "Skim Dead!";

   if( debug_ ) {
      edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "Constructor EcalAnomalousEvent";
      ebBoundaryCalc.setDebugMode();
      eeBoundaryCalc.setDebugMode();
   }

   produces<bool>();

   produces<AnomalousECALVariables> ("anomalousECALVariables");
}

EcalDeadCellBoundaryEnergyFilter::~EcalDeadCellBoundaryEnergyFilter() {
}

// ------------ method called on each new Event  ------------
bool EcalDeadCellBoundaryEnergyFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
   using namespace edm;

   //int eventno = (int) iEvent.eventAuxiliary().event();

   std::vector<BoundaryInformation> v_enNeighboursGap_EB;
   std::vector<BoundaryInformation> v_enNeighboursGap_EE;
   v_enNeighboursGap_EB.reserve(50);
   v_enNeighboursGap_EE.reserve(50);

   std::vector<BoundaryInformation> v_boundaryInfoDeadCells_EB;
   std::vector<BoundaryInformation> v_boundaryInfoDeadCells_EE;
   v_boundaryInfoDeadCells_EB.reserve(50);
   v_boundaryInfoDeadCells_EE.reserve(50);

   // Get the Ecal RecHits
   Handle<EcalRecHitCollection> EBRecHits;
   iEvent.getByToken(EBRecHitsToken_, EBRecHits);
   Handle<EcalRecHitCollection> EERecHits;
   iEvent.getByToken(EERecHitsToken_, EERecHits);

   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord> ().get(theCaloTopology);

   edm::ESHandle<EcalChannelStatus> ecalStatus;
   iSetup.get<EcalChannelStatusRcd> ().get(ecalStatus);

   edm::ESHandle<CaloGeometry> geometry;
   iSetup.get<CaloGeometryRecord> ().get(geometry);

//   int DeadChannelsCounterEB = 0;
//   int DeadChannelsCounterEE = 0;

   int i_EBDead = 0;
   int i_EEDead = 0;
   int i_EBGap = 0;
   int i_EEGap = 0;

   std::vector<DetId> sameFlagDetIds;

   bool pass = true;

   if (!limitFilterToEE_) {

      if (debug_)
         edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "process EB";

      for (EcalRecHitCollection::const_iterator hit = EBRecHits->begin(); hit != EBRecHits->end(); ++hit) {

         bool detIdAlreadyChecked = false;
         DetId currDetId = (DetId) hit->id();
         //add limitation to channel stati
         EcalChannelStatus::const_iterator chit = ecalStatus->find(currDetId);
         int status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
         if (status != 0)
            continue;
         bool passChannelLimitation = false;

         // check if hit has a dead neighbour
         std::vector<int> deadNeighbourStati;
         ebBoundaryCalc.checkRecHitHasDeadNeighbour(*hit, ecalStatus, deadNeighbourStati);

         for (int cs = 0; cs < (int) limitDeadCellToChannelStatusEB_.size(); ++cs) {
            int channelAllowed = limitDeadCellToChannelStatusEB_[cs];

            for (std::vector<int>::iterator sit = deadNeighbourStati.begin(); sit != deadNeighbourStati.end(); ++sit) {
               if (channelAllowed == *sit || (channelAllowed < 0 && std::abs(channelAllowed) <= *sit)) {
                  passChannelLimitation = true;
                  break;
               }
            }
         }

         for (std::vector<DetId>::iterator it = sameFlagDetIds.begin(); it != sameFlagDetIds.end(); it++) {
            if (currDetId == *it)
               detIdAlreadyChecked = true;
         }

         // RecHit is at EB boundary and should be processed
         if (!detIdAlreadyChecked && deadNeighbourStati.empty() && ebBoundaryCalc.checkRecHitHasInvalidNeighbour(
               *hit, ecalStatus)) {

            BoundaryInformation gapinfo = ebBoundaryCalc.gapRecHits(
                  (const edm::Handle<EcalRecHitCollection>&) EBRecHits, (const EcalRecHit *) &(*hit), theCaloTopology,
                  ecalStatus, geometry);

            // get rechits along gap cluster
            for (std::vector<DetId>::iterator it = gapinfo.detIds.begin(); it != gapinfo.detIds.end(); it++) {
               sameFlagDetIds.push_back(*it);
            }

            if (gapinfo.boundaryEnergy > cutBoundEnergyGapEB) {

               i_EBGap++;
               v_enNeighboursGap_EB.push_back(gapinfo);

               if (debug_)
                  edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "EB: gap cluster energy: " << gapinfo.boundaryEnergy << " deadCells: "
                        << gapinfo.detIds.size();
            }
         }

         // RecHit is member of boundary and should be processed
         if (!detIdAlreadyChecked && (passChannelLimitation || (limitDeadCellToChannelStatusEB_.empty()
               && !deadNeighbourStati.empty()))) {

            BoundaryInformation boundinfo = ebBoundaryCalc.boundaryRecHits(
                  (const edm::Handle<EcalRecHitCollection>&) EBRecHits, (const EcalRecHit *) &(*hit), theCaloTopology,
                  ecalStatus, geometry);

            // get boundary of !kDead rechits arround the dead cluster
            for (std::vector<DetId>::iterator it = boundinfo.detIds.begin(); it != boundinfo.detIds.end(); it++) {
               sameFlagDetIds.push_back(*it);
            }

            if (boundinfo.boundaryEnergy > cutBoundEnergyDeadCellsEB) {

               i_EBDead++;
               v_boundaryInfoDeadCells_EB.push_back(boundinfo);

               if (debug_)
                  edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "EB: boundary Energy dead RecHit: " << boundinfo.boundaryEnergy << " ET: "
                        << boundinfo.boundaryET << " deadCells: " << boundinfo.detIds.size();
            }

         }
      }

   } //// End of EB

   sameFlagDetIds.clear();

   if (!limitFilterToEB_) {

      if (debug_)
         edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "process EE";

      for (EcalRecHitCollection::const_iterator hit = EERecHits->begin(); hit != EERecHits->end(); ++hit) {

         bool detIdAlreadyChecked = false;
         DetId currDetId = (DetId) hit->id();
         //add limitation to channel stati
         EcalChannelStatus::const_iterator chit = ecalStatus->find(currDetId);
         int status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
         if (status != 0)
            continue;
         bool passChannelLimitation = false;

         // check if hit has a dead neighbour
         std::vector<int> deadNeighbourStati;
         eeBoundaryCalc.checkRecHitHasDeadNeighbour(*hit, ecalStatus, deadNeighbourStati);

         for (int cs = 0; cs < (int) limitDeadCellToChannelStatusEE_.size(); ++cs) {
            int channelAllowed = limitDeadCellToChannelStatusEE_[cs];

            for (std::vector<int>::iterator sit = deadNeighbourStati.begin(); sit != deadNeighbourStati.end(); ++sit) {
               if (channelAllowed == *sit || (channelAllowed < 0 && std::abs(channelAllowed) <= *sit)) {
                  passChannelLimitation = true;
                  break;
               }
            }
         }

         for (std::vector<DetId>::iterator it = sameFlagDetIds.begin(); it != sameFlagDetIds.end(); it++) {
            if (currDetId == *it)
               detIdAlreadyChecked = true;
         }

         // RecHit is at EE boundary and should be processed
         const CaloSubdetectorGeometry* subGeom = geometry->getSubdetectorGeometry(currDetId);
         auto cellGeom = subGeom->getGeometry(currDetId);
         double eta = cellGeom->getPosition().eta();

         if (!detIdAlreadyChecked && deadNeighbourStati.empty() && eeBoundaryCalc.checkRecHitHasInvalidNeighbour(
               *hit, ecalStatus) && std::abs(eta) < 1.6) {

            BoundaryInformation gapinfo = eeBoundaryCalc.gapRecHits(
                  (const edm::Handle<EcalRecHitCollection>&) EERecHits, (const EcalRecHit *) &(*hit), theCaloTopology,
                  ecalStatus, geometry);

            // get rechits along gap cluster
            for (std::vector<DetId>::iterator it = gapinfo.detIds.begin(); it != gapinfo.detIds.end(); it++) {
               sameFlagDetIds.push_back(*it);
            }

            if (gapinfo.boundaryEnergy > cutBoundEnergyGapEE) {

               i_EEGap++;
               v_enNeighboursGap_EE.push_back(gapinfo);

               if (debug_)
                  edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "EE: gap cluster energy: " << gapinfo.boundaryEnergy << " deadCells: "
                        << gapinfo.detIds.size();
            }
         }

         // RecHit is member of boundary and should be processed
         if (!detIdAlreadyChecked && (passChannelLimitation || (limitDeadCellToChannelStatusEE_.empty()
               && !deadNeighbourStati.empty()))) {

            BoundaryInformation boundinfo = eeBoundaryCalc.boundaryRecHits(
                  (const edm::Handle<EcalRecHitCollection>&) EERecHits, (const EcalRecHit *) &(*hit), theCaloTopology,
                  ecalStatus, geometry);

            // get boundary of !kDead rechits arround the dead cluster
            for (std::vector<DetId>::iterator it = boundinfo.detIds.begin(); it != boundinfo.detIds.end(); it++) {
               sameFlagDetIds.push_back(*it);
            }

            if (boundinfo.boundaryEnergy > cutBoundEnergyDeadCellsEE) {

               i_EEDead++;
               v_boundaryInfoDeadCells_EE.push_back(boundinfo);

               if (debug_)
                  edm::LogInfo("EcalDeadCellBoundaryEnergyFilter") << "EE: boundary Energy dead RecHit: " << boundinfo.boundaryEnergy << " ET: "
                        << boundinfo.boundaryET << " deadCells: " << boundinfo.detIds.size();

            }

         }
      }

   } //// End of EE

   sameFlagDetIds.clear();

   auto pAnomalousECALVariables = std::make_unique<AnomalousECALVariables>(v_enNeighboursGap_EB,
            v_enNeighboursGap_EE, v_boundaryInfoDeadCells_EB, v_boundaryInfoDeadCells_EE);


   bool isGap = pAnomalousECALVariables->isGapEcalCluster(cutBoundEnergyGapEB, cutBoundEnergyGapEE);
   bool isBoundary = pAnomalousECALVariables->isDeadEcalCluster(maxBoundaryEnergy_, limitDeadCellToChannelStatusEB_,
            limitDeadCellToChannelStatusEE_);
   pass = (!isBoundary && ((!isGap && enableGap_) || !enableGap_));

   iEvent.put(std::move(pAnomalousECALVariables), "anomalousECALVariables");

   iEvent.put(std::make_unique<bool>(pass));

   if( taggingMode_ ){
      if (skimDead_ && (i_EBDead >= 1 || i_EEDead >= 1)) {
         return true;
      } else if (skimGap_ && (i_EBGap >= 1 || i_EEGap >= 1)) {
         return true;
      } else if (!skimDead_ && !skimGap_)
         return true;
      else {
         return false;
      }
   }
   else return pass;

/*
   if (FilterAlgo_ == "TuningMode") {
      std::unique_ptr<AnomalousECALVariables> pAnomalousECALVariables(new AnomalousECALVariables(v_enNeighboursGap_EB,
            v_enNeighboursGap_EE, v_boundaryInfoDeadCells_EB, v_boundaryInfoDeadCells_EE));
      iEvent.put(std::move(pAnomalousECALVariables), "anomalousECALVariables");

      if (skimDead_ && (i_EBDead >= 1 || i_EEDead >= 1)) {
         return true;
      } else if (skimGap_ && (i_EBGap >= 1 || i_EEGap >= 1)) {
         return true;
      } else if (!skimDead_ && !skimGap_)
         return true;
      else {
         return false;
      }
   }

   if (FilterAlgo_ == "FilterMode") {
      std::unique_ptr<AnomalousECALVariables> pAnomalousECALVariables(new AnomalousECALVariables(v_enNeighboursGap_EB,
            v_enNeighboursGap_EE, v_boundaryInfoDeadCells_EB, v_boundaryInfoDeadCells_EE));

      bool isGap = pAnomalousECALVariables->isGapEcalCluster(cutBoundEnergyGapEB, cutBoundEnergyGapEE);
      bool isBoundary = pAnomalousECALVariables->isDeadEcalCluster(maxBoundaryEnergy_, limitDeadCellToChannelStatusEB_,
            limitDeadCellToChannelStatusEE_);

      bool result = (!isBoundary && ((!isGap && enableGap_) || !enableGap_));
      if (!result) {
      }
      return result;
   }
*/

//   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void EcalDeadCellBoundaryEnergyFilter::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalDeadCellBoundaryEnergyFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE( EcalDeadCellBoundaryEnergyFilter);
