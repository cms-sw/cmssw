// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1EGammaCrystalsProducer
//
/**\class L1EGammaCrystalsProducer L1EGammaCrystalsProducer.cc SLHCUpgradeSimulations/L1CaloTrigger/plugin/L1EGammaCrystalsProducer.cc

Description: Produces crystal clusters using crystal-level information

Implementation:
[Notes on implementation]
*/
//
// Original Author: Nick Smith, Alexander Savin
// Created: Tue Apr 22 2014
// $Id$
//
//


#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iostream>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

// ECAL TPs
#include "SimCalorimetry/EcalEBTrigPrimProducers/plugins/EcalEBTrigPrimProducer.h"
#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"

// HCAL TPs
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

// Adding boost to read json files for tower mapping
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include <string.h>

class L1EGCrystalClusterProducer : public edm::EDProducer {
   public:
      explicit L1EGCrystalClusterProducer(const edm::ParameterSet&);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      bool cluster_passes_base_cuts(const l1slhc::L1EGCrystalCluster& cluster) const;
      bool cluster_passes_photonWP80(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5, float &e2x2) const;
      bool cluster_passes_electronWP98(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const;

      double EtminForStore;
      double EcalTpEtMin;
      double EtMinForSeedHit;
      bool debug;
      bool useRecHits;
      bool doBremClustering;
      edm::EDGetTokenT<EcalRecHitCollection> ecalRecHitEBToken_;
      edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> ecalTPEBToken_;
      edm::EDGetTokenT<HBHERecHitCollection> hcalRecHitToken_;
      edm::EDGetTokenT< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hcalTPToken_;

      edm::ESHandle<CaloGeometry> caloGeometry_;
      const CaloSubdetectorGeometry * ebGeometry;
      //const CaloSubdetectorGeometry * eeGeometry; // unused a.t.m.
      const CaloSubdetectorGeometry * hbGeometry;
      //const CaloSubdetectorGeometry * heGeometry; // unused a.t.m.
      edm::ESHandle<HcalTopology> hbTopology;
      const HcalTopology * hcTopology_;

      boost::property_tree::ptree towerMap;
      bool useTowerMap;
      std::string towerMapName;

      class SimpleCaloHit
      {
         public:
            EBDetId id;
            GlobalVector position; // As opposed to GlobalPoint, so we can add them (for weighted average)
            float energy=0.;
            bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
            bool isEndcapHit=false; // If using endcap, we won't be using integer crystal indices
            
         // tool functions
            inline float pt() const{return (position.mag2()>0) ? energy*sin(position.theta()) : 0.;};
            inline float deta(SimpleCaloHit& other) const{return position.eta() - other.position.eta();};
            int dieta(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               // int indices do not contain zero
               // Logic from EBDetId::distanceEta() without the abs()
               if (id.ieta() * other.id.ieta() > 0)
                  return id.ieta()-other.id.ieta();
               return id.ieta()-other.id.ieta()-1;
            };
            inline float dphi(SimpleCaloHit& other) const{return reco::deltaPhi(static_cast<float>(position.phi()), static_cast<float>(other.position.phi()));};
            int diphi(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               // Logic from EBDetId::distancePhi() without the abs()
               int PI = 180;
               int result = id.iphi() - other.id.iphi();
               while (result > PI) result -= 2*PI;
               while (result <= -PI) result += 2*PI;
               return result;
            };
            float distanceTo(SimpleCaloHit& other) const
            {
               // Treat position as a point, measure 3D distance
               // This is used for endcap hits, where we don't have a rectangular mapping
               return (position-other.position).mag();
            };
            bool operator==(SimpleCaloHit& other) const
            {
               if ( id == other.id &&
                    position == other.position &&
                    energy == other.energy &&
                    isEndcapHit == other.isEndcapHit
                  ) return true;
                  
               return false;
            };
      };
};

L1EGCrystalClusterProducer::L1EGCrystalClusterProducer(const edm::ParameterSet& iConfig) :
   EtminForStore(iConfig.getParameter<double>("EtminForStore")),
   EcalTpEtMin(iConfig.getUntrackedParameter<double>("EcalTpEtMin", 0.5)), // Default to 500 MeV
   EtMinForSeedHit(iConfig.getUntrackedParameter<double>("EtMinForSeedHit", 1.0)), // Default to 1 GeV
   debug(iConfig.getUntrackedParameter<bool>("debug", false)),
   useRecHits(iConfig.getParameter<bool>("useRecHits")),
   doBremClustering(iConfig.getUntrackedParameter<bool>("doBremClustering", true)),
   ecalRecHitEBToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitEB"))),
   ecalTPEBToken_(consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPEB"))),
   hcalRecHitToken_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hcalRecHit"))),
   hcalTPToken_(consumes< edm::SortedCollection<HcalTriggerPrimitiveDigi> >(iConfig.getParameter<edm::InputTag>("hcalTP"))),
   useTowerMap(iConfig.getUntrackedParameter<bool>("useTowerMap", false)),
   towerMapName(iConfig.getUntrackedParameter<std::string>("towerMapName", "defaultMap.json"))

{
   produces<l1slhc::L1EGCrystalClusterCollection>("L1EGXtalClusterNoCuts");
   produces<l1slhc::L1EGCrystalClusterCollection>("L1EGXtalClusterWithCuts");
   produces<l1extra::L1EmParticleCollection>("L1EGCollectionWithCuts");
   
   // Get tower mapping
   if (useTowerMap) {
      std::cout << "Using tower mapping for ECAL regions.  Map name: " << towerMapName << std::endl;
      std::string base = std::getenv("CMSSW_BASE");
      std::string fpath = "/src/L1Trigger/L1CaloTrigger/data/";
      std::string file = base+fpath+towerMapName;
      read_json(file, towerMap);
   }
}

void L1EGCrystalClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   // Get calo geometry info split by subdetector
   iSetup.get<CaloGeometryRecord>().get(caloGeometry_);
   ebGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
   hbGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
   iSetup.get<HcalRecNumberingRecord>().get(hbTopology);
   hcTopology_ = hbTopology.product();
   HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);
   
   std::vector<SimpleCaloHit> ecalhits;
   std::vector<SimpleCaloHit> hcalhits;
   
   // Retrieve the ecal barrel hits
   // Use ECAL TPs unless otherwise specified
   if (!useRecHits) {
      edm::Handle<EcalEBTrigPrimDigiCollection> pcalohits;
      iEvent.getByToken(ecalTPEBToken_,pcalohits);
      int totNumHits = 0;
      for(auto& hit : *pcalohits.product())
      {
         // Have to comment out kOutOfTime and kLiSpikeFlag because we're testing basic TPs
         //if(hit.energy() > 0.2 && !hit.checkFlag(EcalRecHit::kOutOfTime) && !hit.checkFlag(EcalRecHit::kL1SpikeFlag))
         if(hit.encodedEt() > 0) // hit.encodedEt() returns an int corresponding to 2x the crystal Et
         {

            float et = hit.encodedEt()/8.; // Et is 10 bit, by keeping the ADC saturation Et at 120 GeV it means that you have to divide by 8
            if (et < EcalTpEtMin) continue; // keep the 500 MeV ET Cut

            auto cell = ebGeometry->getGeometry(hit.id());
            SimpleCaloHit ehit;
            ehit.id = hit.id();
            // So, apparently there are (at least) two competing basic vector classes being tossed around in
            // cmssw, the calorimeter geometry package likes to use "DataFormats/GeometryVector/interface/GlobalPoint.h"
            // while "DataFormats/Math/interface/Point3D.h" also contains a competing definition of GlobalPoint. Oh well...
            ehit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
            ehit.energy = et / sin(ehit.position.theta());
            if ( debug ) std::cout << " -- ECAL TP encoded ET: " << hit.encodedEt() << std::endl;
            //std::cout << " -- ECAL TP Et: " << ehit.energy << std::endl;
            //std::cout << totNumHits << " -- ehit iPhi: " << ehit.id.iphi() << " -- tp iPhi: " << hit.id().iphi() << std::endl;
            //std::cout << " -- iEta: " << ehit.id.ieta() << std::endl;
            ecalhits.push_back(ehit);
            totNumHits++;
         }
      }
   } // Done loading ECAL TPs

   if (useRecHits) {
      std::cout << "ECAL Rec Hits is not supported by L1T" << std::endl;
      //// using RecHits (https://cmssdt.cern.ch/SDT/doxygen/CMSSW_6_1_2_SLHC6/doc/html/d8/dc9/classEcalRecHit.html)
      //edm::Handle<EcalRecHitCollection> pcalohits;
      //iEvent.getByToken(ecalRecHitEBToken_,pcalohits);
      //float hitEt;
      //float hitEnergy;
      //for(auto& hit : *pcalohits.product())
      //{
      //   if(hit.energy() > 0.2 && !hit.checkFlag(EcalRecHit::kOutOfTime) && !hit.checkFlag(EcalRecHit::kL1SpikeFlag))
      //   {
      //      auto cell = geometryHelper.getEcalBarrelGeometry()->getGeometry(hit.id());
      //      SimpleCaloHit ehit;
      //      ehit.id = hit.id();
      //      ehit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
      //      hitEnergy = hit.energy();
      //      hitEt = hitEnergy * sin(ehit.position.theta());
      //      if (hitEt > 0.5) { // Add this extra requirement to mimic ECAL TPs 500 MeV ET Min
      //         ehit.energy = hit.energy();
      //         ecalhits.push_back(ehit);
      //      }
      //   }
      //}
   } // Done loading Rec Hits

   if (!useRecHits) {

      //std::cout << "Incorporating Hcal TPs under development at the moment" << std::endl;
      edm::Handle< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hbhecoll;
      iEvent.getByToken(hcalTPToken_,hbhecoll);
      for (auto& hit : *hbhecoll.product())
      {

         // SOI_compressedEt() Compressed ET, integer representing increments of 500 MeV
         // Cut requires 500 MeV TP
         if ( hit.SOI_compressedEt() == 0 ) continue; // SOI_compressedEt() Compressed ET for the "Sample of Interest"
         // Need to use proper decompression here https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.cc#L97-L114

         //std::cout << "  -- HCAL TP: " << hit.SOI_compressedEt() << std::endl;
         //auto cell = geometryHelper.getHcalGeometry()->getGeometry(hit.id());
         //std::cout << "Hit ID: " << hit.id() << std::endl;

         // Check if the detId is in the current geometry setup
         // so that L1EG doesn't crash
         if (!(hcTopology_->validHT(hit.id()))) {
           std::cout << " -- Hcal hit DetID not present in HCAL Geom: " << hit.id() << std::endl;
           continue;
         }

         // Get the detId associated with the HCAL TP
         // if no detIds associated, skip
         std::vector<HcalDetId> hcId = theTrigTowerGeometry.detIds(hit.id());
         if (hcId.size() == 0) {
           std::cout << "Cannot find any HCalDetId corresponding to " << hit.id() << std::endl;
           continue;
         }

         // Skip HCAL TPs which don't have HB detIds
         if (hcId[0].subdetId() > 1) continue;

         // Find the average position of all HB detIds
         GlobalVector hcal_tp_position = GlobalVector(0., 0., 0.);
         for (auto &hcId_i : hcId) {
           if (hcId_i.subdetId() > 1) continue;
           auto cell = hbGeometry->getGeometry(hcId_i);
           if (cell == 0) continue;
           GlobalVector tmpVector = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
           if ( debug ) std::cout << " ---- " << hcId_i << "  subD: " << hcId_i.subdetId() << " : (eta,phi,z), (" << tmpVector.eta() << ", " << tmpVector.phi() << ", " << tmpVector.z() << ")" << std::endl;
           hcal_tp_position = tmpVector;
           break;
         }

         SimpleCaloHit hhit;
         hhit.id = hit.id();
         hhit.position = hcal_tp_position;
         float et = hit.SOI_compressedEt() / 2.;
         hhit.energy = et / sin(hhit.position.theta());
         hcalhits.push_back(hhit);

         if ( debug ) std::cout << "HCAL TP Position (x,y,z): " << hcal_tp_position << ", TP ET : " << hhit.energy << std::endl;
      }
   }

   if (useRecHits) {
      std::cout << "HCAL Rec Hits is not supported by L1T" << std::endl;
      //// Retrive HCAL hits - using RecHits at the moment
      //edm::Handle<HBHERecHitCollection> hbhecoll;
      ////iEvent.getByLabel("hbheprereco", hbhecoll);
      ////iEvent.getByLabel("hbheUpgradeReco", hbhecoll);
      ////iEvent.getByLabel("hltHbhereco", hbhecoll);
      //iEvent.getByToken(hcalRecHitToken_,hbhecoll);
      ////edm::Handle< edm::SortedCollection<HcalTriggerPrimitiveDigi> > hbhecoll;
      ////iEvent.getByToken(hcalTPToken_,hbhecoll);
      //for (auto& hit : *hbhecoll.product())
      //{
      //   //std::cout << "  -- HCAL TP: " << hit.SOI_compressedEt() << std::endl;
      //   //if ( hit.SOI_compressedEt() > 0.1 ) // SOI_compressedEt() Compressed ET for the "Sample of Interest"
      //   if ( hit.energy() > 0.1 ) // SOI_compressedEt() Compressed ET for the "Sample of Interest"
      //     // Need to use proper decompression here https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.cc#L97-L114
      //   {
      //      auto cell = geometryHelper.getHcalGeometry()->getGeometry(hit.id());
      //      SimpleCaloHit hhit;
      //      hhit.id = hit.id();
      //      hhit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
      //      //hhit.energy = hit.SOI_compressedEt();
      //      hhit.energy = hit.energy();
      //      hcalhits.push_back(hhit);
      //   }
      //}
   }

   // Cluster containters
   std::unique_ptr<l1slhc::L1EGCrystalClusterCollection> L1EGXtalClusterNoCuts (new l1slhc::L1EGCrystalClusterCollection );
   std::unique_ptr<l1slhc::L1EGCrystalClusterCollection> L1EGXtalClusterWithCuts( new l1slhc::L1EGCrystalClusterCollection );
   std::unique_ptr<l1extra::L1EmParticleCollection> L1EGCollectionWithCuts( new l1extra::L1EmParticleCollection );
   
   // Clustering algorithm
   while(true)
   {
      // Find highest pt hit (that's not already used)
      SimpleCaloHit centerhit;
      for(const auto& hit : ecalhits)
      {
         if ( !hit.stale && hit.pt() > centerhit.pt() )
         {
            centerhit = hit;
         }
      }
      // If we are less than 1GeV (configurable with EtMinForSeedHit) 
      // or out of hits (i.e. when centerhit is default constructed) we stop
      if ( centerhit.pt() < EtMinForSeedHit ) break;
      if ( debug ) std::cout << "-------------------------------------" << std::endl;
      if ( debug ) std::cout << "New cluster: center crystal pt = " << centerhit.pt() << std::endl;

      // Experimental parameters, don't want to bother with hardcoding them in data format
      std::map<std::string, float> params;
      
      // Find the energy-weighted average position,
      // calculate isolation parameter,
      // calculate pileup-corrected pt,
      // and quantify likelihood of a brem
      GlobalVector weightedPosition;
      GlobalVector ECalPileUpVector;
      float totalEnergy = 0.;
      float ECalIsolation = 0.;
      float ECalPileUpEnergy = 0.;
      float upperSideLobePt = 0.;
      float lowerSideLobePt = 0.;
      float e2x2_1 = 0.;
      float e2x2_2 = 0.;
      float e2x2_3 = 0.;
      float e2x2_4 = 0.;
      float e2x2 = 0.;
      float e2x5_1 = 0.;
      float e2x5_2 = 0.;
      float e2x5 = 0.;
      float e5x5 = 0.;
      float e3x5 = 0.;
      bool electronWP98;
      bool photonWP80;
      std::vector<float> crystalPt;
      std::map<int, float> phiStrip;
      //std::cout << " -- iPhi: " << ehit.id.iphi() << std::endl;
      //std::cout << " -- iEta: " << ehit.id.ieta() << std::endl;
      //std::cout << "2nd JSON mapping for tower iEta -5, iPhi 7: " << towerMap.get<std::string>(key) << std::endl;
      std::string towerKey;
      std::string centerHitTowerRegion;
      if (useTowerMap) {
         towerKey = "("+std::to_string(centerhit.id.ieta())+
            ", "+std::to_string(centerhit.id.iphi())+")";
         centerHitTowerRegion = towerMap.get<std::string>(towerKey);
      }

      for(auto& hit : ecalhits)
      {

         // If using tower regions, check that we match with the centerHit
         if (useTowerMap) {
            towerKey = "("+std::to_string(hit.id.ieta())+
                  ", "+std::to_string(hit.id.iphi())+")";
            if (centerHitTowerRegion != towerMap.get<std::string>(towerKey)) {
                if (debug) std::cout << "Skipping hit. CenterHitRegion: " << centerHitTowerRegion <<
                    "    currentHitRegion: " << towerMap.get<std::string>(towerKey) << std::endl;
                continue;
            }
         }

         if ( !hit.stale &&
               ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
                || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )) // endcap crystals are 30mm on a side, 3.5*sqrt(2) cm radius should enclose 3x3
         {
            weightedPosition += hit.position*hit.energy;
            totalEnergy += hit.energy;
            hit.stale = true;
            crystalPt.push_back(hit.pt());
            if ( debug && hit == centerhit )
               std::cout << "\x1B[32m"; // green hilight
            if ( debug && hit.isEndcapHit ) std::cout <<
               "\tCrystal pt=" << hit.pt() <<
               ", eta=" << hit.position.eta() <<
               ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
            else if ( debug ) std::cout <<
               "\tCrystal (" << hit.dieta(centerhit) << "," << hit.diphi(centerhit) <<
               ") pt=" << hit.pt() <<
               ", eta=" << hit.position.eta() <<
               ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
         }

         if ( abs(hit.dieta(centerhit)) == 0 && abs(hit.diphi(centerhit)) <= 7 )
         {
            phiStrip[hit.diphi(centerhit)] = hit.pt();
         }

         // Build 5x5
         if ( abs(hit.dieta(centerhit)) < 3 && abs(hit.diphi(centerhit)) < 3 )
         {
            e5x5 += hit.energy;
         }

         // Build 3x5
         if ( abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3 )
         {
            e3x5 += hit.energy;
         }

         // Build E2x5
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && abs(hit.diphi(centerhit)) < 3 )
         {
            e2x5_1 += hit.energy;
         }
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && abs(hit.diphi(centerhit)) < 3 )
         {
            e2x5_2 += hit.energy;
         }
         e2x5 = TMath::Max( e2x5_1, e2x5_2 );

         // Build 2x2, highest energy 2x2 square containing the seed
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == 1) )
         {
            e2x2_1 += hit.energy;
         }
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == 1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == -1) )
         {
            e2x2_2 += hit.energy;
         }
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == 1) )
         {
            e2x2_3 += hit.energy;
         }
         if ( (hit.dieta(centerhit) == 0 || hit.dieta(centerhit) == -1) && (hit.diphi(centerhit) == 0 || hit.diphi(centerhit) == -1) )
         {
            e2x2_4 += hit.energy;
         }
         e2x2 = TMath::Max( e2x2_1, e2x2_2 );
         e2x2 = TMath::Max( e2x2, e2x2_3 );
         e2x2 = TMath::Max( e2x2, e2x2_4 );
         params["E2x2"] = e2x2;
         params["E2x5"] = e2x5;
         params["E3x5"] = e3x5;
         params["E5x5"] = e5x5;

         // Isolation and pileup must not use hits used in the cluster
         // As for the endcap hits, well, as far as this algorithm is concerned, caveat emptor...
         if ( !(!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
              && !(centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )
         {
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 14 && abs(hit.diphi(centerhit)) < 14)
                 || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 42. ))
            {
               ECalIsolation += hit.pt();
               if ( hit.pt() > 1. )
                  params["nIsoCrystals1"]++;
            }
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) >= 3 && hit.diphi(centerhit) < 8)
                 || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit) >= 0.0173*3 && hit.dphi(centerhit) < 0.0173*8 ))
            {
               upperSideLobePt += hit.pt();
            }
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) > -8 && hit.diphi(centerhit) <= -3)
                 || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit)*-1 >= 0.0173*3 && hit.dphi(centerhit)*-1 < 0.0173*8 ))
            {
               lowerSideLobePt += hit.pt();
            }
            if ( hit.pt() < 5. &&
                 ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 7 && abs(hit.diphi(centerhit)) < 57 )
                  || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 50.) ))
            {
               ECalPileUpEnergy += hit.energy;
               ECalPileUpVector += hit.position;
            }
         }
      }
      params["uncorrectedE"] = totalEnergy;
      params["uncorrectedPt"] = totalEnergy*sin(weightedPosition.theta());

      // phi strip params
      // lambda returns size of contiguous strip, one-hole strip
      auto countStrip = [&phiStrip](float threshold) -> std::pair<float, float>
      {
         int nContiguous = 1;
         int nOneHole = 1;
         bool firstHole = false;
         for(int i=1; i<=7; ++i)
         {
            if ( phiStrip[i] > threshold && !firstHole )
            {
               nContiguous++;
               nOneHole++;
            }
            else if ( phiStrip[i] > threshold )
               nOneHole++;
            else if ( !firstHole )
               firstHole = true;
            else
               break;
         }
         firstHole = false;
         for(int i=-1; i>=-7; --i)
         {
            if ( phiStrip[i] > threshold && !firstHole )
            {
               nContiguous++;
               nOneHole++;
            }
            else if ( phiStrip[i] > threshold )
               nOneHole++;
            else if ( !firstHole )
               firstHole = true;
            else
               break;
         }
         return std::make_pair<float, float>(nContiguous, nOneHole);
      };
      auto zeropair = countStrip(0.);
      params["phiStripContiguous0"] = zeropair.first;
      params["phiStripOneHole0"] = zeropair.second;
      auto threepair = countStrip(0.03*totalEnergy);
      params["phiStripContiguous3p"] = threepair.first;
      params["phiStripOneHole3p"] = threepair.second;

      // Check if brem clustering is desired
      if (doBremClustering)
      {
         // Check if sidelobes should be included in sum
         if ( upperSideLobePt/params["uncorrectedPt"] > 0.1 )
         {
            for(auto& hit : ecalhits)
            {
               if ( !hit.stale &&
                    (  (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) >= 3 && hit.diphi(centerhit) < 8)
                      || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit) >= 0.0173*3 && hit.dphi(centerhit) < 0.0173*8 )
                    ) )
               {
                  weightedPosition += hit.position*hit.energy;
                  totalEnergy += hit.energy;
                  hit.stale = true;
                  crystalPt.push_back(hit.pt());
               }
            }
         }
         if ( lowerSideLobePt/params["uncorrectedPt"] > 0.1 )
         {
            for(auto& hit : ecalhits)
            {
               if ( !hit.stale &&
                    (  (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) > -8 && hit.diphi(centerhit) <= -3)
                      || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit)*-1 >= 0.0173*3 && hit.dphi(centerhit)*-1 < 0.0173*8 )
                    ) )
               {
                  weightedPosition += hit.position*hit.energy;
                  totalEnergy += hit.energy;
                  hit.stale = true;
                  crystalPt.push_back(hit.pt());
               }
            }
         }
      } // Brem section finished

      // no need to rescale weightedPosition if we only use theta
      float correctedTotalPt = totalEnergy*sin(weightedPosition.theta());
      params["avgIsoCrystalE"] = (params["nIsoCrystals1"] > 0.) ? ECalIsolation/params["nIsoCrystals1"] : 0.;
      params["upperSideLobePt"] = upperSideLobePt;
      params["lowerSideLobePt"] = lowerSideLobePt;
      ECalIsolation /= params["uncorrectedPt"];
      float totalPtPUcorr = params["uncorrectedPt"] - ECalPileUpEnergy*sin(ECalPileUpVector.theta())/19.;
      float bremStrength = params["uncorrectedPt"] / correctedTotalPt;

      if ( debug ) std::cout << "Weighted position eta = " << weightedPosition.eta() << ", phi = " << weightedPosition.phi() << std::endl;
      if ( debug ) std::cout << "Uncorrected Total energy = " << params["uncorrectedE"] << ", total pt = " << params["uncorrectedPt"] << std::endl;
      if ( debug ) std::cout << "Total energy = " << totalEnergy << ", total pt = " << correctedTotalPt << std::endl;
      if ( debug ) std::cout << "Isolation: " << ECalIsolation << std::endl;

      // Calculate H/E if we have are using RecHits
      // else fill with -1. so no one gets confused
      // and thinks 0. is H/E
      float hcalEnergy = 0.;
      float hovere;
      if (hcalhits.size() > 0) {
        for(const auto& hit : hcalhits)
        {
           if ( fabs(hit.deta(centerhit)) < 0.15 && fabs(hit.dphi(centerhit)) < 0.15 )
           {
              hcalEnergy += hit.energy;
           }
        }
        hovere = hcalEnergy/params["uncorrectedE"];
      }
      else hovere = -1.0;

      if ( debug ) std::cout << "H/E: " << hovere << std::endl;



      // Check if cluster passes electron or photon WPs
      float cluster_eta = weightedPosition.eta();
      electronWP98 = cluster_passes_electronWP98( correctedTotalPt, cluster_eta, ECalIsolation, e2x5, e5x5);
      photonWP80 = cluster_passes_photonWP80( correctedTotalPt, cluster_eta, ECalIsolation, e2x5, e5x5, e2x2);

      
      // Form a l1slhc::L1EGCrystalCluster
      reco::Candidate::PolarLorentzVector p4(correctedTotalPt, weightedPosition.eta(), weightedPosition.phi(), 0.);
      l1slhc::L1EGCrystalCluster cluster(p4, hovere, ECalIsolation, centerhit.id, totalPtPUcorr, bremStrength,
            e2x2, e2x5, e3x5, e5x5, electronWP98, photonWP80);
      // Save pt array
      cluster.SetCrystalPtInfo(crystalPt);
      params["crystalCount"] = crystalPt.size();
      cluster.SetExperimentalParams(params);
      L1EGXtalClusterNoCuts->push_back(cluster);


      // Save clusters with some cuts
      if ( cluster_passes_base_cuts(cluster) )
      {
         // Optional min. Et cut
         if ( cluster.pt() >= EtminForStore ) {
            L1EGXtalClusterWithCuts->push_back(cluster);
            L1EGCollectionWithCuts->push_back(l1extra::L1EmParticle(p4, edm::Ref<L1GctEmCandCollection>(), 0));
         }
      }
   }

   iEvent.put(std::move(L1EGXtalClusterNoCuts),"L1EGXtalClusterNoCuts");
   iEvent.put(std::move(L1EGXtalClusterWithCuts), "L1EGXtalClusterWithCuts" );
   iEvent.put(std::move(L1EGCollectionWithCuts), "L1EGCollectionWithCuts" );
}


bool
L1EGCrystalClusterProducer::cluster_passes_photonWP80(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5, float &e2x2) const {
   // These cuts have been optimized based on 92X
   // This cut reaches an 80% efficiency for photons
   // for offline pt > 30 GeV

   if ( fabs(cluster_eta) < 1.479 )
   {
      if ( !( 0.94 + 0.052 * TMath::Exp( -0.044 * cluster_pt ) < (e2x5 / e5x5)) ) return false;
      if ( !(( 0.85 + -0.0080 * cluster_pt ) > iso ) ) return false;
      if ( ( e2x2 / e2x5) < 0.95 ) return false;

      // Passes cuts
      return true;
   }
   return false; // out of eta range
}


bool
L1EGCrystalClusterProducer::cluster_passes_electronWP98(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const {
   // Replica of below just passed arguments differently
   if ( fabs(cluster_eta) < 1.479 )
   {
      if ( !( 0.94 + 0.052 * TMath::Exp( -0.044 * cluster_pt ) < (e2x5 / e5x5)) ) return false;
      if ( !(( 0.85 + -0.0080 * cluster_pt ) > iso ) ) return false;

      // Passes cuts
      return true;
   }
   return false; // out of eta range
}


bool
L1EGCrystalClusterProducer::cluster_passes_base_cuts(const l1slhc::L1EGCrystalCluster& cluster) const {
   //return true;
   
   // Currently this producer is optimized based on cluster isolation and shower shape
   // The following cut is based off of what was shown in the Phase-2 meeting
   // 23 May 2017 from CMSSW 92X
   // Optimization based on min ECAL TP ET = 500 MeV for inclusion
   // Only the barrel is considered
   if ( fabs(cluster.eta()) < 1.479 )
   {
      //std::cout << "Starting passing check" << std::endl;
      float cluster_pt = cluster.pt();
      float clusterE2x5 = cluster.GetExperimentalParam("E2x5");
      float clusterE5x5 = cluster.GetExperimentalParam("E5x5");
      float cluster_iso = cluster.isolation();
     
      if ( !( 0.94 + 0.052 * TMath::Exp( -0.044 * cluster_pt ) < (clusterE2x5 / clusterE5x5)) )
         return false;
      if ( !(( 0.85 + -0.0080 * cluster_pt ) > cluster_iso ) )
         return false;
      return true; // cluster passes all cuts
   }
   return false; // out of eta range
}

DEFINE_FWK_MODULE(L1EGCrystalClusterProducer);
