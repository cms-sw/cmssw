// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1TowerCalibrator
//
/**\class L1TowerCalibrator L1TowerCalibrator.cc

Description: 
Take the calibrated unclustered ECAL energy and total HCAL
energy associated with the L1CaloTower collection output
from L1EGammaCrystalsEmulatorProducer: l1CaloTowerCollection, "L1CaloTowerCollection"

as well as HGCal Tower level inputs:
BXVector<l1t::HGCalTower> "hgcalTriggerPrimitiveDigiProducer" "tower" "HLT"     

and HCAL HF inputs from:
edm::SortedCollection<HcalTriggerPrimitiveDigi,edm::StrictWeakOrdering<HcalTriggerPrimitiveDigi> > "simHcalTriggerPrimitiveDigis" "" "HLT"     


Implement PU-based calibrations which scale down the ET
in the towers based on mapping nTowers with ECAL(HCAL) ET <= defined PU threshold.
This value has been shown to be similar between TTbar, QCD, and minBias samples.
This allows a prediction of nvtx. Which can be mapped to the total minBias
energy in an eta slice of the detector.  Subtract the total estimated minBias energy
per eta slice divided by nTowers in that eta slice from each trigger tower in
that eta slice.

This is all ECAL / HCAL specific or EM vs. Hadronic for HGCal.

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Thu Nov 15 2018
// $Id$
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <iostream>

#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "TGraph.h"


class L1TowerCalibrator : public edm::EDProducer {
    public:
        explicit L1TowerCalibrator(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);

        double HcalTpEtMin;
        double EcalTpEtMin;
        double HGCalHadTpEtMin;
        double HGCalEmTpEtMin;
        double HFTpEtMin;
        double puThreshold;
        double puThresholdEcal;
        double puThresholdHcal;
        double puThresholdL1eg;
        double puThresholdHGCalEMMin;
        double puThresholdHGCalEMMax;
        double puThresholdHGCalHadMin;
        double puThresholdHGCalHadMax;
        double puThresholdHFMin;
        double puThresholdHFMax;
        bool debug;

        edm::EDGetTokenT< L1CaloTowerCollection > l1TowerToken_;
        edm::Handle< L1CaloTowerCollection > l1CaloTowerHandle;

        edm::EDGetTokenT<l1t::HGCalTowerBxCollection> hgcalTowersToken_;
        edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;
        l1t::HGCalTowerBxCollection hgcalTowers;

        edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalToken_;
        edm::Handle<HcalTrigPrimDigiCollection> hcalTowerHandle;
        edm::ESHandle<CaloTPGTranscoder> decoder_;

};

L1TowerCalibrator::L1TowerCalibrator(const edm::ParameterSet& iConfig) :
    HcalTpEtMin(iConfig.getParameter<double>("HcalTpEtMin")), // Should default to 0 MeV
    EcalTpEtMin(iConfig.getParameter<double>("EcalTpEtMin")), // Should default to 0 MeV
    HGCalHadTpEtMin(iConfig.getParameter<double>("HGCalHadTpEtMin")), // Should default to 0 MeV
    HGCalEmTpEtMin(iConfig.getParameter<double>("HGCalEmTpEtMin")), // Should default to 0 MeV
    HFTpEtMin(iConfig.getParameter<double>("HFTpEtMin")), // Should default to 0 MeV
    puThreshold(iConfig.getParameter<double>("puThreshold")), // Should default to 5.0 GeV
    puThresholdEcal(iConfig.getParameter<double>("puThresholdEcal")), // Should default to 5.0 GeV
    puThresholdHcal(iConfig.getParameter<double>("puThresholdHcal")), // Should default to 5.0 GeV
    puThresholdL1eg(iConfig.getParameter<double>("puThresholdL1eg")), // Should default to 5.0 GeV
    puThresholdHGCalEMMin(iConfig.getParameter<double>("puThresholdHGCalEMMin")), // Should default to 5.0 GeV
    puThresholdHGCalEMMax(iConfig.getParameter<double>("puThresholdHGCalEMMax")), // Should default to 5.0 GeV
    puThresholdHGCalHadMin(iConfig.getParameter<double>("puThresholdHGCalHadMin")), // Should default to 5.0 GeV
    puThresholdHGCalHadMax(iConfig.getParameter<double>("puThresholdHGCalHadMax")), // Should default to 5.0 GeV
    puThresholdHFMin(iConfig.getParameter<double>("puThresholdHFMin")), // Should default to 5.0 GeV
    puThresholdHFMax(iConfig.getParameter<double>("puThresholdHFMax")), // Should default to 5.0 GeV
    debug(iConfig.getParameter<bool>("debug")),
    l1TowerToken_(consumes< L1CaloTowerCollection >(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
    hgcalTowersToken_(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("L1HgcalTowersInputTag"))),
    hcalToken_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis")))
{
    produces< L1CaloTowerCollection >("L1CaloTowerCalibratedCollection");
}

void L1TowerCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{




    // Calibrated output collection
    std::unique_ptr< L1CaloTowerCollection > L1CaloTowerCalibratedCollection(new L1CaloTowerCollection);





    // Load the ECAL+HCAL tower sums coming from L1EGammaCrystalsEmulatorProducer.cc
    iEvent.getByToken(l1TowerToken_,l1CaloTowerHandle);

    // HGCal info
    iEvent.getByToken(hgcalTowersToken_,hgcalTowersHandle);
    hgcalTowers = (*hgcalTowersHandle.product());

    // HF Tower info
    iEvent.getByToken(hcalToken_,hcalTowerHandle);
    
    // Load all towers into L1CaloTowers
    std::vector< L1CaloTower > l1CaloTowers;

    // Barrel ECAL (unclustered) and HCAL
    for (auto& hit : *l1CaloTowerHandle.product())
    {

        L1CaloTower l1Hit;
        l1Hit.ecal_tower_et  = hit.ecal_tower_et;
        l1Hit.hcal_tower_et  = hit.hcal_tower_et;
        // Add min ET thresholds for tower ET
        if (l1Hit.ecal_tower_et < EcalTpEtMin) l1Hit.ecal_tower_et = 0.0;
        if (l1Hit.hcal_tower_et < HcalTpEtMin) l1Hit.hcal_tower_et = 0.0;
        l1Hit.tower_iEta  = hit.tower_iEta;
        l1Hit.tower_iPhi  = hit.tower_iPhi;
        l1Hit.tower_eta  = hit.tower_eta;
        l1Hit.tower_phi  = hit.tower_phi;

        // FIXME There is an error in the L1EGammaCrystalsEmulatorProducer.cc which is
        // returning towers with minimal ECAL energy, and no HCAL energy with these
        // iEta/iPhi coordinates and eta = -88.653152 and phi = -99.000000.
        // Skip these for the time being until the upstream code has been debugged
        if ((int)l1Hit.tower_iEta == -1016 && (int)l1Hit.tower_iPhi == -962) continue;


        l1CaloTowers.push_back( l1Hit );
        if (debug) printf("Barrel tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et_sum %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et);
    }

    // Loop over HGCalTowers and create L1CaloTowers for them and add to collection
    // This iterator is taken from the PF P2 group
    // https://github.com/p2l1pfp/cmssw/blob/170808db68038d53794bc65fdc962f8fc337a24d/L1Trigger/Phase2L1ParticleFlow/plugins/L1TPFCaloProducer.cc#L278-L289
    for (auto it = hgcalTowers.begin(0), ed = hgcalTowers.end(0); it != ed; ++it)
    {
        // skip lowest ET towers
        if (it->etEm() < HGCalEmTpEtMin && it->etHad() < HGCalHadTpEtMin) continue;

        L1CaloTower l1Hit;
        // Set energies normally, but need to zero if below threshold
        if (it->etEm() < HGCalEmTpEtMin) l1Hit.ecal_tower_et  = 0.;
        else l1Hit.ecal_tower_et  = it->etEm();

        if (it->etHad() < HGCalHadTpEtMin) l1Hit.hcal_tower_et  = 0.;
        else l1Hit.hcal_tower_et  = it->etHad();

        l1Hit.tower_eta  = it->eta();
        l1Hit.tower_phi  = it->phi();
        l1Hit.tower_iEta  = -98; // -98 mean HGCal
        l1Hit.tower_iPhi  = -98; 
        l1CaloTowers.push_back( l1Hit );
        if (debug) printf("HGCal tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et_sum %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et);
    }


    // Loop over Hcal HF tower inputs and create L1CaloTowers and add to
    // l1CaloTowers collection
    iSetup.get<CaloTPGRecord>().get(decoder_);
    for (const auto & hit : *hcalTowerHandle.product()) {
        HcalTrigTowerDetId id = hit.id();
        double et = decoder_->hcaletValue(hit.id(), hit.t0());
        if (et < HFTpEtMin) continue;
        // Only doing HF so skip outside range
        if ( abs(id.ieta()) < l1t::CaloTools::kHFBegin ) continue;
        if ( abs(id.ieta()) > l1t::CaloTools::kHFEnd ) continue;

        L1CaloTower l1Hit;
        l1Hit.ecal_tower_et  = 0.;
        l1Hit.hcal_tower_et  = et;
        l1Hit.tower_eta  = l1t::CaloTools::towerEta(id.ieta());
        l1Hit.tower_phi  = l1t::CaloTools::towerPhi(id.ieta(), id.iphi());
        l1Hit.tower_iEta  = id.ieta();
        l1Hit.tower_iPhi  = id.iphi();
        l1CaloTowers.push_back( l1Hit );

        if (debug) printf("HCAL HF tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et_sum %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et);
    }



    // N Tower totals
    // For mapping to estimated nvtx in event
    int i_ecal_hits_leq_threshold = 0;
    int i_hgcalEM_hits_leq_threshold = 0;
    int i_hcal_hits_leq_threshold = 0;
    int i_hgcalHad_hits_leq_threshold = 0;
    int i_hf_hits_leq_threshold = 0;


    // Loop over the collection containing all hits
    // and calculate the number of hits falling into the 
    // "less than or equal" nTowers variable which maps to
    // estimated number of vertices
    for (auto &l1CaloTower : l1CaloTowers)
    {


        // Barrel ECAL
        if(l1CaloTower.ecal_tower_et > 0. && l1CaloTower.tower_iEta != -98) 
        {
            if(l1CaloTower.ecal_tower_et <= puThresholdEcal) 
            {
                i_ecal_hits_leq_threshold++;
            }
        }


        // HGCal EM
        if(l1CaloTower.ecal_tower_et > 0. && l1CaloTower.tower_iEta == -98) 
        {
            if(l1CaloTower.ecal_tower_et <= puThresholdHGCalEMMax && l1CaloTower.ecal_tower_et >= puThresholdHGCalEMMin) 
            {
                i_hgcalEM_hits_leq_threshold++;
            }
        }


        // Barrel HCAL
        if(l1CaloTower.hcal_tower_et > 0. && l1CaloTower.tower_iEta != -98 && abs(l1CaloTower.tower_eta) < 2.0) // abs(eta) < 2 just keeps us out of HF 
        {
            if(l1CaloTower.hcal_tower_et <= puThresholdHcal)
            {
                i_hcal_hits_leq_threshold++;
            }
        }


        // HGCal Had
        if(l1CaloTower.hcal_tower_et > 0. && l1CaloTower.tower_iEta == -98) 
        {
            if(l1CaloTower.hcal_tower_et <= puThresholdHGCalHadMax && l1CaloTower.hcal_tower_et >= puThresholdHGCalHadMin) 
            {
                i_hgcalHad_hits_leq_threshold++;
            }
        }

        // HF
        if(l1CaloTower.hcal_tower_et > 0. && l1CaloTower.tower_iEta != -98 && abs(l1CaloTower.tower_eta) > 2.0) // abs(eta) > 2 keeps us out of barrel HF 
        {
            if(l1CaloTower.hcal_tower_et <= puThresholdHFMax && l1CaloTower.hcal_tower_et >= puThresholdHFMin)
            {
                i_hf_hits_leq_threshold++;
            }
        }
    }



    // For each subdetector, map to the estimated number of PU vertices



    // Use estimated number of PU vertices to subtract off PU contributions
    // to each and every hit. In cases where the energy would go negative,
    // limit this to zero.
    // Add the hit to the output collection: L1CaloTowerCalibratedCollection




    //// Loop over them a second time to make a new calibrated collection
    //for (auto& hit : *l1CaloTowerHandle.product())
    //{
    //    L1CaloTower l1Hit;
    //    l1Hit.ecal_tower_et  = hit.ecal_tower_et;
    //    l1Hit.hcal_tower_et  = hit.hcal_tower_et;
    //    l1Hit.tower_iEta  = hit.tower_iEta;
    //    l1Hit.tower_iPhi  = hit.tower_iPhi;
    //    l1Hit.tower_eta  = hit.tower_eta;
    //    l1Hit.tower_phi  = hit.tower_phi;
    //    L1CaloTowerCalibratedCollection.push_back( l1Hit );
    //    if (debug) printf(" - Output calibrated tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et_sum %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et);
    //}

    iEvent.put(std::move(L1CaloTowerCalibratedCollection),"L1CaloTowerCalibratedCollection");
}



DEFINE_FWK_MODULE(L1TowerCalibrator);
