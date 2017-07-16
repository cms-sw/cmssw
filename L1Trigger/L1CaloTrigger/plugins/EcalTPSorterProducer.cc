// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: EcalTPSorterProducer
//
/**\class EcalTPSorterProducer EcalTPSorterProducer.cc SLHCUpgradeSimulations/L1CaloTrigger/plugin/EcalTPSorterProducer.cc

Description: Produces a collection of ECAL TPs which have been sorted and slimmed based on their geometrical region in the detector 

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Sat July 15 2017
// $Id$
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>

// ECAL TPs
#include "SimCalorimetry/EcalEBTrigPrimProducers/plugins/EcalEBTrigPrimProducer.h"
#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"

// Adding boost to read json files for tower mapping
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include <string.h>

class EcalTPSorterProducer : public edm::EDProducer {
   public:
      explicit EcalTPSorterProducer(const edm::ParameterSet&);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> ecalTPEBToken_;
      double tpsToKeep;
      double nRegions;
      bool debug;

      boost::property_tree::ptree towerMap;
      std::string towerMapName;
};

EcalTPSorterProducer::EcalTPSorterProducer(const edm::ParameterSet& iConfig) :
   ecalTPEBToken_(consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPEB"))),
   tpsToKeep(iConfig.getUntrackedParameter<double>("tpsToKeep", 20)),
   nRegions(iConfig.getUntrackedParameter<double>("nRegions", 24)),
   debug(iConfig.getUntrackedParameter<bool>("debug", false)),
   towerMapName(iConfig.getUntrackedParameter<std::string>("towerMapName", "defaultMap.json"))

{
   produces<EcalEBTrigPrimDigiCollection>("EcalTPsTopPerRegion");
   
   // Get tower mapping
   std::cout << "Using tower mapping for ECAL regions.  Map name: " << towerMapName << std::endl;
   std::string base = std::getenv("CMSSW_BASE");
   std::string fpath = "/src/L1Trigger/L1CaloTrigger/data/";
   std::string file = base+fpath+towerMapName;
   read_json(file, towerMap);
}

void EcalTPSorterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   // For the sorted and slimmed ECAL TPs
   std::unique_ptr<EcalEBTrigPrimDigiCollection> EcalTPsTopPerRegion( new EcalEBTrigPrimDigiCollection );
   
   // Map to store region as defined in the json file, and all hits
   std::map<int, std::vector<EcalEBTriggerPrimitiveDigi>> ecalHitMap;
   // initialize the vectors
   for (int i=1; i <= nRegions; ++i)
   {
      ecalHitMap[i] = std::vector<EcalEBTriggerPrimitiveDigi>();
      //std::cout << "region: " << i << " : " << ecalHitMap[i].size() << std::endl;
   }
   
   // Retrieve the ecal barrel hits
   // Use ECAL TPs unless otherwise specified
   edm::Handle<EcalEBTrigPrimDigiCollection> pcalohits;
   iEvent.getByToken(ecalTPEBToken_,pcalohits);

   std::string towerKey;
   std::string hitRegion;
   for(auto& hit : *pcalohits.product())
   {
      if(hit.encodedEt() > 0) // hit.encodedEt() returns an int corresponding to 2x the crystal Et
      {

         //auto cell = ebGeometry->getGeometry(hit.id());
         //SimpleCaloHit ehit;
         //ehit.id = hit.id();

         //std::cout << " -- iPhi: " << ehit.id.iphi() << std::endl;
         //std::cout << " -- iEta: " << ehit.id.ieta() << std::endl;
         //std::cout << "2nd JSON mapping for tower iEta -5, iPhi 7: " << towerMap.get<std::string>(key) << std::endl;
         towerKey = "("+std::to_string(hit.id().ieta())+
            ", "+std::to_string(hit.id().iphi())+")";
         hitRegion = towerMap.get<std::string>(towerKey);

         ecalHitMap[ std::stoi(hitRegion) ].push_back( hit );
      }
   }

   for (auto& pair : ecalHitMap )
   {
      if (debug) std::cout << "Region: " << pair.first << " length: " << pair.second.size() << std::endl;
      if (debug && pair.first == 5)
      {
         for (auto& hit : pair.second)
         {
            std::cout << " -- pre-sort pt: " << hit.encodedEt() << std::endl;
         }
      }
      // Sort each region
      std::sort(begin(pair.second), end(pair.second), [](const EcalEBTriggerPrimitiveDigi& a, const EcalEBTriggerPrimitiveDigi& b){return a.encodedEt() > b.encodedEt();});
      if (debug && pair.first == 5)
      {
         for (auto& hit : pair.second)
         {
            std::cout << " -- post-sort pt: " << hit.encodedEt() << std::endl;
         }
      }
      // Save top X
      std::vector<EcalEBTriggerPrimitiveDigi> ecalHitsSorted;
      int cnt = 0;
      for (auto& hit : pair.second)
      {
         ++cnt;
         if (cnt > tpsToKeep) continue;
         EcalTPsTopPerRegion->push_back(hit);
         
      }
   }

   iEvent.put(std::move(EcalTPsTopPerRegion),"EcalTPsTopPerRegion");
}


DEFINE_FWK_MODULE(EcalTPSorterProducer);
