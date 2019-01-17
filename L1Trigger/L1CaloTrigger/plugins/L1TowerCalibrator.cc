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

Implement PU-based calibrations which scale down the ET
in the towers based on mapping nTowers with ECAL(HCAL) ET <= defined PU threshold.
This value has been shown to be similar between TTbar, QCD, and minBias samples.
This allows a prediction of nvtx. Which can be mapped to the total minBias
energy in an eta slice of the detector.  Subtract that total energy over
the total nTowers in that eta slice.  This is all ECAL / HCAL specific.

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
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <iostream>

#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TGraph.h"


class L1TowerCalibrator : public edm::EDProducer {
    public:
        explicit L1TowerCalibrator(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);

        double puThresholdEcal;
        double puThresholdHcal;
        bool debug;

        edm::EDGetTokenT< L1CaloTowerCollection > l1TowerToken_;
        edm::Handle< L1CaloTowerCollection > l1CaloTowerHandle;

};

L1TowerCalibrator::L1TowerCalibrator(const edm::ParameterSet& iConfig) :
    puThresholdEcal(iConfig.getParameter<double>("puThresholdEcal")), // Should default to 5.0 GeV
    puThresholdHcal(iConfig.getParameter<double>("puThresholdHcal")), // Should default to 5.0 GeV
    debug(iConfig.getParameter<bool>("debug")),
    l1TowerToken_(consumes< L1CaloTowerCollection >(iConfig.getParameter<edm::InputTag>("l1CaloTowers")))
{
    produces< L1CaloTowerCollection >("L1CaloTowerCalibratedCollection");
}

void L1TowerCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


    // Calibrated output collection
    std::unique_ptr< L1CaloTowerCollection > L1CaloTowerCalibratedCollection(new L1CaloTowerCollection);


    // N Tower totals
    // For mapping to estimated nvtx in event
    int i_ecal_hits_leq_threshold = 0;
    int i_hcal_hits_leq_threshold = 0;
    // For calculating nTotal per eta slice for calibrating
    int i_ecal_hits_er1to3 = 0;
    int i_ecal_hits_er4to6 = 0;
    int i_ecal_hits_er7to9 = 0;
    int i_ecal_hits_er10to12 = 0;
    int i_ecal_hits_er13to15 = 0;
    int i_ecal_hits_er16to18 = 0;
    int i_hcal_hits_er1to3 = 0;
    int i_hcal_hits_er4to6 = 0;
    int i_hcal_hits_er7to9 = 0;
    int i_hcal_hits_er10to12 = 0;
    int i_hcal_hits_er13to15 = 0;
    int i_hcal_hits_er16to18 = 0;


    // Load the ECAL+HCAL tower sums coming from L1EGammaCrystalsEmulatorProducer.cc
    iEvent.getByToken(l1TowerToken_,l1CaloTowerHandle);
    // Loop over them once to calculate nTotals
    for (auto& hit : *l1CaloTowerHandle.product())
    {

        if (debug) printf("Input tower iEta %i iPhi %i ecal_et %f hcal_et %f\n", hit.tower_iEta, hit.tower_iPhi, hit.ecal_tower_et, hit.hcal_tower_et);

        // Update n_totals and ET sums
        if(hit.ecal_tower_et > 0.) 
        {
            if(hit.ecal_tower_et <= puThresholdEcal) 
            {
                i_ecal_hits_leq_threshold++;
            }
            
            // Sums by eta
            if( abs(hit.tower_iEta) <= 3 )
            { 
                i_ecal_hits_er1to3++;
            }
            else if( abs(hit.tower_iEta) <= 6 )
            { 
                i_ecal_hits_er4to6++;
            }
            else if( abs(hit.tower_iEta) <= 9 )
            { 
                i_ecal_hits_er7to9++;
            }
            else if( abs(hit.tower_iEta) <= 12 )
            { 
                i_ecal_hits_er10to12++;
            }
            else if( abs(hit.tower_iEta) <= 15 )
            { 
                i_ecal_hits_er13to15++;
            }
            else // ( abs(hit.tower_iEta) <= 18 )
            { 
                i_ecal_hits_er16to18++;
            }
        }

        if(hit.hcal_tower_et > 0.) 
        {
            if(hit.hcal_tower_et <= puThresholdHcal)
            {
                i_hcal_hits_leq_threshold++;
            }
            
            // Sums by eta
            if( abs(hit.tower_iEta) <= 3 )
            { 
                i_hcal_hits_er1to3++;
            }
            else if( abs(hit.tower_iEta) <= 6 )
            { 
                i_hcal_hits_er4to6++;
            }
            else if( abs(hit.tower_iEta) <= 9 )
            { 
                i_hcal_hits_er7to9++;
            }
            else if( abs(hit.tower_iEta) <= 12 )
            { 
                i_hcal_hits_er10to12++;
            }
            else if( abs(hit.tower_iEta) <= 15 )
            { 
                i_hcal_hits_er13to15++;
            }
            else // ( abs(hit.tower_iEta) <= 18 )
            { 
                i_hcal_hits_er16to18++;
            }
        }
    } // end initial loop counting nTotals

    //// Loop over them a second time to make a new calibrated collection
    //for (auto& hit : *l1CaloTowerHandle.product())
    //{
    //    SimpleCaloHit l1Hit;
    //    l1Hit.ecal_tower_et  = hit.ecal_tower_et;
    //    l1Hit.hcal_tower_et  = hit.hcal_tower_et;
    //    l1Hit.total_tower_et  = l1Hit.ecal_tower_et + l1Hit.hcal_tower_et;
    //    l1Hit.tower_iEta  = hit.tower_iEta;
    //    l1Hit.tower_iPhi  = hit.tower_iPhi;
    //    l1Hit.tower_eta  = hit.tower_eta;
    //    l1Hit.tower_phi  = hit.tower_phi;
    //    L1CaloTowerCalibratedCollection.push_back( l1Hit );
    //    if (debug) printf("Tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et_sum %f total_et %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et, l1Hit.total_tower_et);
    //}

    iEvent.put(std::move(L1CaloTowerCalibratedCollection),"L1CaloTowerCalibratedCollection");
}



DEFINE_FWK_MODULE(L1TowerCalibrator);
