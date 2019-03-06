// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1CaloJetProducer
//
/**\class L1CaloJetProducer L1CaloJetProducer.cc

Description: 
Beginning with HCAL TPs, create HCAL jet, then
take L1EG crystal clusters from L1EGammaCrystalsProducer.cc
and clusters them within fixed number of trigger towers

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Tue Aug 29 2018
// $Id$
//
//


#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloJet.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

// For pT calibrations
#include "TF1.h"

// Run2/PhaseI output formats
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"


class L1CaloJetProducer : public edm::EDProducer {
    public:
        explicit L1CaloJetProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);
        //bool cluster_passes_base_cuts(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const;
        int ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const;
        int tower_diPhi( int &iPhi_1, int &iPhi_2 ) const;
        int tower_diEta( int &iEta_1, int &iEta_2 ) const;
        float get_deltaR( reco::Candidate::PolarLorentzVector &p4_1,
                reco::Candidate::PolarLorentzVector &p4_2) const;
        float get_hcal_calibration( float &jet_pt, float &ecal_pt,
                float &ecal_L1EG_jet_pt, float &jet_eta ) const;
        float apply_barrel_HGCal_boundary_calibration( float &jet_pt, float &hcal_pt, float &ecal_pt,
                float &ecal_L1EG_jet_pt, int &seed_iEta ) const;

        //double EtminForStore;
        double HcalTpEtMin;
        double EcalTpEtMin;
        double HGCalHadTpEtMin;
        double HGCalEmTpEtMin;
        double HFTpEtMin;
        double EtMinForSeedHit;
        double EtMinForCollection;

        // For fetching calibrations
        std::vector< double > jetPtBins;
        std::vector< double > emFractionBinsBarrel;
        std::vector< double > absEtaBinsBarrel;
        std::vector< double > jetCalibrationsBarrel;
        std::vector< double > emFractionBinsHGCal;
        std::vector< double > absEtaBinsHGCal;
        std::vector< double > jetCalibrationsHGCal;
        std::vector< double > emFractionBinsHF;
        std::vector< double > absEtaBinsHF;
        std::vector< double > jetCalibrationsHF;

        // For storing calibrations
        std::vector< std::vector< std::vector< double >>> calibrationsBarrel;
        std::vector< std::vector< std::vector< double >>> calibrationsHGCal;
        std::vector< std::vector< std::vector< double >>> calibrationsHF;

        bool debug;
        edm::EDGetTokenT< L1CaloTowerCollection > l1TowerToken_;
        edm::Handle< L1CaloTowerCollection > l1CaloTowerHandle;

        edm::EDGetTokenT<l1slhc::L1EGCrystalClusterCollection> crystalClustersToken_;
        edm::Handle<l1slhc::L1EGCrystalClusterCollection> crystalClustersHandle;
        l1slhc::L1EGCrystalClusterCollection crystalClusters;

        //edm::EDGetTokenT<l1t::HGCalTowerBxCollection> hgcalTowersToken_;
        //edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;
        //l1t::HGCalTowerBxCollection hgcalTowers;

        //edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalToken_;
        //edm::Handle<HcalTrigPrimDigiCollection> hcalTowerHandle;
        //edm::ESHandle<CaloTPGTranscoder> decoder_;

        // Fit function to scale L1EG Pt to align with electron gen pT
        //TF1 ptAdjustFunc = TF1("ptAdjustFunc", "([0] + [1]*TMath::Exp(-[2]*x)) * ([3] + [4]*TMath::Exp(-[5]*x))");


        class l1CaloJetObj
        {
            public:
                bool barrelSeeded = true; // default to barrel seeded
                reco::Candidate::PolarLorentzVector jetCluster;
                reco::Candidate::PolarLorentzVector hcalJetCluster;
                reco::Candidate::PolarLorentzVector ecalJetCluster;
                reco::Candidate::PolarLorentzVector seedTower;
                reco::Candidate::PolarLorentzVector leadingL1EG;
                reco::Candidate::PolarLorentzVector l1EGjet;
                float jetClusterET = 0.;
                float hcalJetClusterET = 0.;
                float ecalJetClusterET = 0.;
                float seedTowerET = 0.;
                float leadingL1EGET = 0.;
                float l1EGjetET = 0.;

                // Matrices to map energy per included tower in ET
                //float total_map[9][9]; // 9x9 array
                //float ecal_map[9][9]; // 9x9 array
                //float hcal_map[9][9]; // 9x9 array
                //float l1eg_map[9][9]; // 9x9 array

                int seed_iEta = -99;
                int seed_iPhi = -99;

                float hcal_seed = 0.;
                float hcal_3x3 = 0.;
                float hcal_5x5 = 0.;
                float hcal_7x7 = 0.;
                float hcal_2x2_1 = 0.;
                float hcal_2x2_2 = 0.;
                float hcal_2x2_3 = 0.;
                float hcal_2x2_4 = 0.;
                float hcal_nHits = 0.;

                float ecal_seed = 0.;
                float ecal_3x3 = 0.;
                float ecal_5x5 = 0.;
                float ecal_7x7 = 0.;
                float ecal_2x2_1 = 0.;
                float ecal_2x2_2 = 0.;
                float ecal_2x2_3 = 0.;
                float ecal_2x2_4 = 0.;
                float ecal_nHits = 0.;

                float total_seed = 0.;
                float total_3x3 = 0.;
                float total_5x5 = 0.;
                float total_7x7 = 0.;
                float total_2x2_1 = 0.;
                float total_2x2_2 = 0.;
                float total_2x2_3 = 0.;
                float total_2x2_4 = 0.;
                float total_nHits = 0.;

                void Init()
                {
                    SetJetClusterP4( 0., 0., 0., 0. );
                    SetHcalJetClusterP4( 0., 0., 0., 0. );
                    SetEcalJetClusterP4( 0., 0., 0., 0. );
                    SetSeedP4( 0., 0., 0., 0. );
                    SetLeadingL1EGP4( 0., 0., 0., 0. );
                    SetL1EGJetP4( 0., 0., 0., 0. );
                }

                void SetJetClusterP4( double pt, double eta, double phi, double mass )
                {
                    this->jetCluster.SetPt( pt );
                    this->jetCluster.SetEta( eta );
                    this->jetCluster.SetPhi( phi );
                    this->jetCluster.SetM( mass );
                }
                void SetHcalJetClusterP4( double pt, double eta, double phi, double mass )
                {
                    this->hcalJetCluster.SetPt( pt );
                    this->hcalJetCluster.SetEta( eta );
                    this->hcalJetCluster.SetPhi( phi );
                    this->hcalJetCluster.SetM( mass );
                }
                void SetEcalJetClusterP4( double pt, double eta, double phi, double mass )
                {
                    this->ecalJetCluster.SetPt( pt );
                    this->ecalJetCluster.SetEta( eta );
                    this->ecalJetCluster.SetPhi( phi );
                    this->ecalJetCluster.SetM( mass );
                }
                void SetSeedP4( double pt, double eta, double phi, double mass )
                {
                    this->seedTower.SetPt( pt );
                    this->seedTower.SetEta( eta );
                    this->seedTower.SetPhi( phi );
                    this->seedTower.SetM( mass );
                }
                void SetLeadingL1EGP4( double pt, double eta, double phi, double mass )
                {
                    this->leadingL1EG.SetPt( pt );
                    this->leadingL1EG.SetEta( eta );
                    this->leadingL1EG.SetPhi( phi );
                    this->leadingL1EG.SetM( mass );
                }
                void SetL1EGJetP4( double pt, double eta, double phi, double mass )
                {
                    this->l1EGjet.SetPt( pt );
                    this->l1EGjet.SetEta( eta );
                    this->l1EGjet.SetPhi( phi );
                    this->l1EGjet.SetM( mass );
                }
        };
                




        class simpleL1obj
        {
            public:
                bool stale = false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
                bool associated_with_tower = false; // L1EGs become associated with a tower to find highest ET total for seeding jets
                bool passesStandaloneSS = false; // Store whether any of the portions of a WP are passed
                bool passesStandaloneIso = false; // Store whether any of the portions of a WP are passed
                bool passesTrkMatchSS = false; // Store whether any of the portions of a WP are passed
                bool passesTrkMatchIso = false; // Store whether any of the portions of a WP are passed
                reco::Candidate::PolarLorentzVector p4;

                void SetP4( double pt, double eta, double phi, double mass )
                {
                    this->p4.SetPt( pt );
                    this->p4.SetEta( eta );
                    this->p4.SetPhi( phi );
                    this->p4.SetM( mass );
                }
                inline float pt() const{return p4.pt();};
                inline float eta() const{return p4.eta();};
                inline float phi() const{return p4.phi();};
                inline float M() const{return p4.M();};
                inline reco::Candidate::PolarLorentzVector GetP4() const{return p4;};
        };
                


        class SimpleCaloHit
        {
            public:
                int tower_iEta = -99;
                int tower_iPhi = -99;
                float tower_eta = -99;
                float tower_phi = -99;
                float ecal_tower_et=0.;
                float hcal_tower_et=0.;
                float total_tower_et=0.;
                float total_tower_plus_L1EGs_et=0.;
                bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
                bool isBarrel=true; // Defaults to a barrel hit
        };
};

L1CaloJetProducer::L1CaloJetProducer(const edm::ParameterSet& iConfig) :
    //EtminForStore(iConfig.getParameter<double>("EtminForStore")),
    HcalTpEtMin(iConfig.getParameter<double>("HcalTpEtMin")), // Should default to 0 MeV
    EcalTpEtMin(iConfig.getParameter<double>("EcalTpEtMin")), // Should default to 0 MeV
    HGCalHadTpEtMin(iConfig.getParameter<double>("HGCalHadTpEtMin")), // Should default to 0 MeV
    HGCalEmTpEtMin(iConfig.getParameter<double>("HGCalEmTpEtMin")), // Should default to 0 MeV
    HFTpEtMin(iConfig.getParameter<double>("HFTpEtMin")), // Should default to 0 MeV
    EtMinForSeedHit(iConfig.getParameter<double>("EtMinForSeedHit")), // Should default to 2.5 GeV
    EtMinForCollection(iConfig.getParameter<double>("EtMinForCollection")), // Testing 10 GeV
    jetPtBins(iConfig.getParameter<std::vector<double>>("jetPtBins")),
    emFractionBinsBarrel(iConfig.getParameter<std::vector<double>>("emFractionBinsBarrel")),
    absEtaBinsBarrel(iConfig.getParameter<std::vector<double>>("absEtaBinsBarrel")),
    jetCalibrationsBarrel(iConfig.getParameter<std::vector<double>>("jetCalibrationsBarrel")),
    emFractionBinsHGCal(iConfig.getParameter<std::vector<double>>("emFractionBinsHGCal")),
    absEtaBinsHGCal(iConfig.getParameter<std::vector<double>>("absEtaBinsHGCal")),
    jetCalibrationsHGCal(iConfig.getParameter<std::vector<double>>("jetCalibrationsHGCal")),
    emFractionBinsHF(iConfig.getParameter<std::vector<double>>("emFractionBinsHF")),
    absEtaBinsHF(iConfig.getParameter<std::vector<double>>("absEtaBinsHF")),
    jetCalibrationsHF(iConfig.getParameter<std::vector<double>>("jetCalibrationsHF")),
    debug(iConfig.getParameter<bool>("debug")),
    l1TowerToken_(consumes< L1CaloTowerCollection >(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
    crystalClustersToken_(consumes<l1slhc::L1EGCrystalClusterCollection>(iConfig.getParameter<edm::InputTag>("L1CrystalClustersInputTag")))
    //hgcalTowersToken_(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("L1HgcalTowersInputTag"))),
    //hcalToken_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis")))

{
    if (debug) printf("L1CaloJetProducer setup\n");
    produces<l1slhc::L1CaloJetsCollection>("L1CaloJetsNoCuts");
    //produces<l1slhc::L1CaloJetsCollection>("L1CaloJetsWithCuts");
    //produces<l1extra::L1JetParticleCollection>("L1CaloClusterCollectionWithCuts");
    produces< BXVector<l1t::Jet> >("L1CaloJetCollectionBXV");


    //// Fit parameters measured on 11 Aug 2018, using 500 MeV threshold for ECAL TPs
    //// working in CMSSW 10_1_7
    //// Adjustments to be applied to reco cluster pt
    //// L1EG cut working points are still a function of non-calibrated pT
    //// First order corrections
    //ptAdjustFunc.SetParameter( 0, 1.06 );
    //ptAdjustFunc.SetParameter( 1, 0.273 );
    //ptAdjustFunc.SetParameter( 2, 0.0411 );
    //// Residuals
    //ptAdjustFunc.SetParameter( 3, 1.00 );
    //ptAdjustFunc.SetParameter( 4, 0.567 );
    //ptAdjustFunc.SetParameter( 5, 0.288 );
    if(debug) printf("\nHcalTpEtMin = %f\nEcalTpEtMin = %f\n", HcalTpEtMin, EcalTpEtMin);
    //for( unsigned int i = 0; i < emFractionBins.size(); i++)
    //{
    //    printf("\n  emFrac: %f", emFractionBins.at(i));
    //}
    //for( unsigned int i = 0; i < absEtaBins.size(); i++)
    //{
    //    printf("\n  absEta: %f", absEtaBins.at(i));
    //}
    //for( unsigned int i = 0; i < jetPtBins.size(); i++)
    //{
    //    printf("\n  jetPt: %f", jetPtBins.at(i));
    //}

    // Fill the calibration 3D vector
    // Dimension 1 is EM Fraction bin
    // Dimension 2 is AbsEta bin
    // Dimension 3 is jet pT bin which is filled with the actual callibration value
    // size()-1 b/c the inputs have lower and upper bounds
    // Do Barrel, then HGCal, then HF
    int index = 0;
    //calibrations[em_frac][abs_eta].push_back( jetCalibrationsBarrel.at(index) );
    for( unsigned int abs_eta = 0; abs_eta < absEtaBinsBarrel.size()-1; abs_eta++)
    {
        std::vector< std::vector< double >> em_bins;
        for( unsigned int em_frac = 0; em_frac < emFractionBinsBarrel.size()-1; em_frac++)
        {
            std::vector< double > pt_bin_calibs;
            for( unsigned int pt = 0; pt < jetPtBins.size()-1; pt++)
            {
                //printf("\n em_frac %d abs_eta %d pt %d", em_frac, abs_eta, pt);
                //printf("\n - em_frac %f abs_eta %f pt %f = %f\n", emFractionBinsBarrel.at(em_frac), absEtaBinsBarrel.at(abs_eta), jetPtBins.at(pt), jetCalibrationsBarrel.at(index));
                pt_bin_calibs.push_back( jetCalibrationsBarrel.at(index) );
                index++;
            }
            em_bins.push_back( pt_bin_calibs );
        }
        calibrationsBarrel.push_back( em_bins );
    }
    if(debug) printf("\nLoading Barrel calibrations: Loaded %i values vs. size() of input calibration file: %i", index, int(jetCalibrationsBarrel.size()));

    index = 0;
    //calibrations[em_frac][abs_eta].push_back( jetCalibrationsHGCal.at(index) );
    for( unsigned int abs_eta = 0; abs_eta < absEtaBinsHGCal.size()-1; abs_eta++)
    {
        std::vector< std::vector< double >> em_bins;
        for( unsigned int em_frac = 0; em_frac < emFractionBinsHGCal.size()-1; em_frac++)
        {
            std::vector< double > pt_bin_calibs;
            for( unsigned int pt = 0; pt < jetPtBins.size()-1; pt++)
            {
                //printf("\n em_frac %d abs_eta %d pt %d", em_frac, abs_eta, pt);
                //printf("\n - em_frac %f abs_eta %f pt %f = %f\n", emFractionBinsHGCal.at(em_frac), absEtaBinsHGCal.at(abs_eta), jetPtBins.at(pt), jetCalibrationsHGCal.at(index));
                pt_bin_calibs.push_back( jetCalibrationsHGCal.at(index) );
                index++;
            }
            em_bins.push_back( pt_bin_calibs );
        }
        calibrationsHGCal.push_back( em_bins );
    }
    if(debug) printf("\nLoading HGCal calibrations: Loaded %i values vs. size() of input calibration file: %i", index, int(jetCalibrationsHGCal.size()));

    index = 0;
    //calibrations[em_frac][abs_eta].push_back( jetCalibrationsHF.at(index) );
    for( unsigned int abs_eta = 0; abs_eta < absEtaBinsHF.size()-1; abs_eta++)
    {
        std::vector< std::vector< double >> em_bins;
        for( unsigned int em_frac = 0; em_frac < emFractionBinsHF.size()-1; em_frac++)
        {
            std::vector< double > pt_bin_calibs;
            for( unsigned int pt = 0; pt < jetPtBins.size()-1; pt++)
            {
                //printf("\n em_frac %d abs_eta %d pt %d", em_frac, abs_eta, pt);
                //printf("\n - em_frac %f abs_eta %f pt %f = %f\n", emFractionBinsHF.at(em_frac), absEtaBinsHF.at(abs_eta), jetPtBins.at(pt), jetCalibrationsHF.at(index));
                pt_bin_calibs.push_back( jetCalibrationsHF.at(index) );
                index++;
            }
            em_bins.push_back( pt_bin_calibs );
        }
        calibrationsHF.push_back( em_bins );
    }
    if(debug) printf("\nLoading HF calibrations: Loaded %i values vs. size() of input calibration file: %i", index, int(jetCalibrationsHF.size()));

    //index = 0;
    //printf("Barrel Calibrations:\n");
    //for( unsigned int i = 0; i < calibrationsBarrel.size(); i++)
    //{
    //    for( unsigned int j = 0; j < calibrationsBarrel[i].size(); j++)
    //    {
    //        for( unsigned int k = 0; k < calibrationsBarrel[i][j].size(); k++)
    //        {
    //            printf("i %i j %i k %i index %i: Input Value %f   Loaded Value %f   Input/Loaded %f\n", 
    //                int(i), int(j), int(k), int(index), jetCalibrationsBarrel.at(index),
    //                calibrationsBarrel[ i ][ j ][ k ], jetCalibrationsBarrel.at(index) / calibrationsBarrel[ i ][ j ][ k ] );
    //            index++;
    //        }
    //    }
    //}
    //index = 0;
    //printf("HGCal Calibrations:\n");
    //for( unsigned int i = 0; i < calibrationsHGCal.size(); i++)
    //{
    //    for( unsigned int j = 0; j < calibrationsHGCal[i].size(); j++)
    //    {
    //        for( unsigned int k = 0; k < calibrationsHGCal[i][j].size(); k++)
    //        {
    //            printf("i %i j %i k %i index %i: Input Value %f   Loaded Value %f   Input/Loaded %f\n", 
    //                int(i), int(j), int(k), int(index), jetCalibrationsHGCal.at(index),
    //                calibrationsHGCal[ i ][ j ][ k ], jetCalibrationsHGCal.at(index) / calibrationsHGCal[ i ][ j ][ k ] );
    //            index++;
    //        }
    //    }
    //}
    //index = 0;
    //printf("HF Calibrations:\n");
    //for( unsigned int i = 0; i < calibrationsHF.size(); i++)
    //{
    //    for( unsigned int j = 0; j < calibrationsHF[i].size(); j++)
    //    {
    //        for( unsigned int k = 0; k < calibrationsHF[i][j].size(); k++)
    //        {
    //            printf("i %i j %i k %i index %i: Input Value %f   Loaded Value %f   Input/Loaded %f\n", 
    //                int(i), int(j), int(k), int(index), jetCalibrationsHF.at(index),
    //                calibrationsHF[ i ][ j ][ k ], jetCalibrationsHF.at(index) / calibrationsHF[ i ][ j ][ k ] );
    //            index++;
    //        }
    //    }
    //}

    if (debug) printf("\nL1CaloJetProducer end\n");
}

void L1CaloJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    //printf("begin L1CaloJetProducer\n");

    // Output collections
    std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloJetsNoCuts (new l1slhc::L1CaloJetsCollection );
    //std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloJetsWithCuts( new l1slhc::L1CaloJetsCollection );
    //std::unique_ptr<l1extra::L1JetParticleCollection> L1CaloClusterCollectionWithCuts( new l1extra::L1JetParticleCollection );
    std::unique_ptr<BXVector<l1t::Jet>> L1CaloJetCollectionBXV(new l1t::JetBxCollection);




    // L1EGs
    iEvent.getByToken(crystalClustersToken_,crystalClustersHandle);
    crystalClusters = (*crystalClustersHandle.product());

    //// HGCal info
    //iEvent.getByToken(hgcalTowersToken_,hgcalTowersHandle);
    //hgcalTowers = (*hgcalTowersHandle.product());

    //// HF Tower info
    //iEvent.getByToken(hcalToken_,hcalTowerHandle);
    
    // Load the ECAL+HCAL tower sums coming from L1EGammaCrystalsEmulatorProducer.cc
    std::vector< SimpleCaloHit > l1CaloTowers;
    
    iEvent.getByToken(l1TowerToken_,l1CaloTowerHandle);
    for (auto& hit : *l1CaloTowerHandle.product())
    {

        SimpleCaloHit l1Hit;
        l1Hit.ecal_tower_et  = hit.ecal_tower_et;
        l1Hit.hcal_tower_et  = hit.hcal_tower_et;
        // Add min ET thresholds for tower ET
        if (l1Hit.ecal_tower_et < EcalTpEtMin) l1Hit.ecal_tower_et = 0.0;
        if (l1Hit.hcal_tower_et < HcalTpEtMin) l1Hit.hcal_tower_et = 0.0;
        l1Hit.total_tower_et  = l1Hit.ecal_tower_et + l1Hit.hcal_tower_et;
        l1Hit.tower_iEta  = hit.tower_iEta;
        l1Hit.tower_iPhi  = hit.tower_iPhi;

        if ( abs(l1Hit.tower_iEta ) <= 18 ) l1Hit.isBarrel = true;
        else l1Hit.isBarrel = false;

        // FIXME There is an error in the L1EGammaCrystalsEmulatorProducer.cc which is
        // returning towers with minimal ECAL energy, and no HCAL energy with these
        // iEta/iPhi coordinates and eta = -88.653152 and phi = -99.000000.
        // Skip these for the time being until the upstream code has been debugged
        if ((int)l1Hit.tower_iEta == -1016 && (int)l1Hit.tower_iPhi == -962) continue;

        l1Hit.tower_eta  = hit.tower_eta;
        l1Hit.tower_phi  = hit.tower_phi;
        l1CaloTowers.push_back( l1Hit );
        if (debug) printf("Tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et %f total_et %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et, l1Hit.total_tower_et);
    }

    //// Loop over HGCalTowers and create SimpleCaloHits for them and add to collection
    //// This iterator is taken from the PF P2 group
    //// https://github.com/p2l1pfp/cmssw/blob/170808db68038d53794bc65fdc962f8fc337a24d/L1Trigger/Phase2L1ParticleFlow/plugins/L1TPFCaloProducer.cc#L278-L289
    //for (auto it = hgcalTowers.begin(0), ed = hgcalTowers.end(0); it != ed; ++it)
    //{
    //    // skip lowest ET towers
    //    if (it->etEm() < HGCalEmTpEtMin && it->etHad() < HGCalHadTpEtMin) continue;

    //    SimpleCaloHit l1Hit;
    //    l1Hit.isBarrel = false;
    //    l1Hit.ecal_tower_et  = it->etEm();
    //    l1Hit.hcal_tower_et  = it->etHad();
    //    l1Hit.total_tower_et  = l1Hit.ecal_tower_et + l1Hit.hcal_tower_et;
    //    l1Hit.tower_eta  = it->eta();
    //    l1Hit.tower_phi  = it->phi();
    //    l1CaloTowers.push_back( l1Hit );
    //    if (debug) printf("Tower isBarrel %d eta %f phi %f ecal_et %f hcal_et %f total_et %f\n", l1Hit.isBarrel, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et, l1Hit.total_tower_et);
    //}


    //// Loop over Hcal HF tower inputs and create SimpleCaloHits and add to
    //// l1CaloTowers collection
    //iSetup.get<CaloTPGRecord>().get(decoder_);
    //for (const auto & hit : *hcalTowerHandle.product()) {
    //    HcalTrigTowerDetId id = hit.id();
    //    double et = decoder_->hcaletValue(hit.id(), hit.t0());
    //    if (et < HFTpEtMin) continue;
    //    // Only doing HF so skip outside range
    //    if ( abs(id.ieta()) < l1t::CaloTools::kHFBegin ) continue;
    //    if ( abs(id.ieta()) > l1t::CaloTools::kHFEnd ) continue;

    //    SimpleCaloHit l1Hit;
    //    l1Hit.isBarrel = false;
    //    l1Hit.ecal_tower_et  = 0.;
    //    l1Hit.hcal_tower_et  = et;
    //    l1Hit.total_tower_et  = l1Hit.ecal_tower_et + l1Hit.hcal_tower_et;
    //    l1Hit.tower_eta  = l1t::CaloTools::towerEta(id.ieta());
    //    l1Hit.tower_phi  = l1t::CaloTools::towerPhi(id.ieta(), id.iphi());
    //    l1Hit.tower_iEta  = id.ieta();
    //    l1Hit.tower_iPhi  = id.iphi();
    //    l1CaloTowers.push_back( l1Hit );

    //    if (debug) printf("Hcal HF Tower isBarrel %d eta %f phi %f ecal_et %f hcal_et %f total_et %f\n", l1Hit.isBarrel, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et, l1Hit.total_tower_et);
    //}


    // Make simple L1objects from the L1EG input collection with marker for 'stale'
    // FIXME could later add quality criteria here to help differentiate likely
    // photons/electrons vs. pions. This could be helpful for L1CaloJets
    std::vector< simpleL1obj > crystalClustersVect;
    for (auto EGammaCand : crystalClusters)
    {
        simpleL1obj l1egObj;
        l1egObj.SetP4(EGammaCand.pt(), EGammaCand.eta(), EGammaCand.phi(), 0.);
        l1egObj.passesStandaloneSS = EGammaCand.GetExperimentalParam("standaloneWP_showerShape");
        l1egObj.passesStandaloneIso = EGammaCand.GetExperimentalParam("standaloneWP_isolation");
        l1egObj.passesTrkMatchSS = EGammaCand.GetExperimentalParam("trkMatchWP_showerShape");
        l1egObj.passesTrkMatchIso = EGammaCand.GetExperimentalParam("trkMatchWP_isolation");
        crystalClustersVect.push_back( l1egObj );
        if (debug) printf("L1EG added from emulator: eta %f phi %f pt %f\n", l1egObj.eta(), l1egObj.phi(), l1egObj.pt());
    }

    // Sorting is unnecessary as we're matching to already built HCAL Jets
    // but it is interesting to know highest pt L1EG, so sort either way
    // Sort clusters so we can always pick highest pt cluster to begin with in our jet clustering
    std::sort(begin(crystalClustersVect), end(crystalClustersVect), [](const simpleL1obj& a,
            simpleL1obj& b){return a.pt() > b.pt();});

    // Match the L1EGs to their associated tower to calculate a TOTAL energy associated
    // with a tower: "total_tower_plus_L1EGs_et".  This can be attributed to multiple
    // L1EGs. Once an L1EG is associated with a tower, mark them as such so they are not
    // double counted for some reason, use "associated_with_tower".
    // This associate will be semi-crude, with barrel geometry, a tower is
    // 0.087 wide, associate them if they are within dEta/dPhi 0.0435.
    float total_et = 0.0;
    int total_nTowers = 0;
    for (auto &l1CaloTower : l1CaloTowers)
    {

        l1CaloTower.total_tower_plus_L1EGs_et = l1CaloTower.total_tower_et; // Set to total before finding associated L1EGs

        int j = 0;
        for (auto &l1eg : crystalClustersVect)
        {

            if (l1eg.associated_with_tower) continue;

            // Could be done very cleanly with iEta/iPhi if we had this from the L1EGs...
            float d_eta = l1CaloTower.tower_eta - l1eg.eta();
            float d_phi = reco::deltaPhi( l1CaloTower.tower_phi, l1eg.phi() );

            if ( fabs( d_eta ) > 0.0435 || fabs( d_phi ) > 0.0435 ) continue;

            j++;
            l1CaloTower.total_tower_plus_L1EGs_et += l1eg.pt();
            if (debug) printf(" - %i L1EG associated with tower: dEta %f dPhi %f L1EG pT %f\n", j, d_eta, d_phi, l1eg.pt());

            l1eg.associated_with_tower = true;

        }

        total_et += l1CaloTower.total_tower_plus_L1EGs_et;
        if (l1CaloTower.total_tower_plus_L1EGs_et > 0) total_nTowers++;
    }

    //for (auto &l1eg : crystalClustersVect)
    //{
    //    if (l1eg.associated_with_tower) continue;
    //    printf(" --- L1EG Not Associated: pt %f eta %f phi %f\n", l1eg.pt(), l1eg.eta(), l1eg.phi());
    //}



    // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const SimpleCaloHit& a,
            SimpleCaloHit& b){return a.total_tower_plus_L1EGs_et > b.total_tower_plus_L1EGs_et;});



    /**************************************************************************
 * Begin with making CaloJets in 9x9 grid based on all energy not included in L1EG Objs.
 * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
 * We plan to further study this choice and possibly move towards a more circular shape
 * Create jetCluster within 9x9 of highest ET seed tower.
 * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
 ******************************************************************************/

    // Experimental parameters, don't want to bother with hardcoding them in data format
    std::map<std::string, float> params;

    std::vector< l1CaloJetObj > l1CaloJetObjs;

    // Count the number of unused HCAL TPs so we can stop while loop after done.
    // Clustering can also stop once there are no seed hits >= EtMinForSeedHit
    int n_towers = l1CaloTowers.size();
    int n_stale = 0;
    bool caloJetClusteringFinished = false;
    while (!caloJetClusteringFinished && n_towers != n_stale)
    {

        l1CaloJetObj caloJetObj;
        caloJetObj.Init();

        // First find highest ET ECAL+HCAL+L1EGs tower and use to seed the 9x9 Jet
        int cnt = 0;
        for (auto &l1CaloTower : l1CaloTowers)
        {

            cnt++;
            if (l1CaloTower.stale) continue; // skip l1CaloTowers which are already used

            if (caloJetObj.jetClusterET == 0.0) // this is the first l1CaloTower to seed the jet
            {
                // Check if the leading unused tower has ET < min for seeding a jet.
                // If so, stop jet clustering
                if (l1CaloTower.total_tower_plus_L1EGs_et < EtMinForSeedHit)
                {
                    caloJetClusteringFinished = true;
                    continue;
                }
                l1CaloTower.stale = true;
                n_stale++;

                // Set seed location needed for delta iEta/iPhi, eta/phi comparisons later
                if (l1CaloTower.isBarrel) caloJetObj.barrelSeeded = true;
                else caloJetObj.barrelSeeded = false;

                // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
                reco::Candidate::PolarLorentzVector hcalP4( l1CaloTower.hcal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector ecalP4( l1CaloTower.ecal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector totalP4( l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);

                // Map center at 4,4
                //caloJetObj.total_map[4][4] = l1CaloTower.total_tower_et; // 9x9 array
                //caloJetObj.ecal_map[4][4]  = l1CaloTower.ecal_tower_et; // 9x9 array
                //caloJetObj.hcal_map[4][4]  = l1CaloTower.hcal_tower_et; // 9x9 array
                //caloJetObj.l1eg_map[4][4]  = l1CaloTower.total_tower_plus_L1EGs_et - l1CaloTower.total_tower_et; // 9x9 array

                if (hcalP4.pt() > 0)
                {
                    caloJetObj.hcal_nHits++;
                    caloJetObj.hcalJetCluster += hcalP4;
                    caloJetObj.hcalJetClusterET += l1CaloTower.hcal_tower_et;
                }
                if (ecalP4.pt() > 0) 
                {
                    caloJetObj.ecal_nHits++;
                    caloJetObj.ecalJetCluster += ecalP4;
                    caloJetObj.ecalJetClusterET += l1CaloTower.ecal_tower_et;
                }
                if (totalP4.pt() > 0) 
                {
                    caloJetObj.total_nHits++;
                    caloJetObj.jetCluster += totalP4;
                    caloJetObj.jetClusterET += l1CaloTower.total_tower_et;
                    caloJetObj.seedTower += totalP4;
                    caloJetObj.seedTowerET += l1CaloTower.total_tower_et;
                }


                caloJetObj.seed_iEta = l1CaloTower.tower_iEta;
                caloJetObj.seed_iPhi = l1CaloTower.tower_iPhi;


                if (debug) printf(" -- hit %i, seeding input     p4 pt %f eta %f phi %f\n", cnt, l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi);
                if (debug) printf(" -- hit %i, seeding input2    p4 pt %f eta %f phi %f\n", cnt, totalP4.pt(), totalP4.eta(), totalP4.phi());
                if (debug) printf(" -- hit %i, seeding reslt tot p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetClusterET, caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());

                // Need to add the seed energy to the dR rings
                caloJetObj.hcal_seed += hcalP4.pt();
                caloJetObj.hcal_3x3 += hcalP4.pt();
                caloJetObj.hcal_5x5 += hcalP4.pt();
                caloJetObj.hcal_7x7 += hcalP4.pt();
                caloJetObj.ecal_seed += ecalP4.pt();
                caloJetObj.ecal_3x3 += ecalP4.pt();
                caloJetObj.ecal_5x5 += ecalP4.pt();
                caloJetObj.ecal_7x7 += ecalP4.pt();
                caloJetObj.total_seed += totalP4.pt();
                caloJetObj.total_3x3 += totalP4.pt();
                caloJetObj.total_5x5 += totalP4.pt();
                caloJetObj.total_7x7 += totalP4.pt();

                // Some discrimination vars, 2x2s including central seed
                caloJetObj.hcal_2x2_1 += hcalP4.pt();
                caloJetObj.hcal_2x2_2 += hcalP4.pt();
                caloJetObj.hcal_2x2_3 += hcalP4.pt();
                caloJetObj.hcal_2x2_4 += hcalP4.pt();
                caloJetObj.ecal_2x2_1 += ecalP4.pt();
                caloJetObj.ecal_2x2_2 += ecalP4.pt();
                caloJetObj.ecal_2x2_3 += ecalP4.pt();
                caloJetObj.ecal_2x2_4 += ecalP4.pt();
                caloJetObj.total_2x2_1 += totalP4.pt();
                caloJetObj.total_2x2_2 += totalP4.pt();
                caloJetObj.total_2x2_3 += totalP4.pt();
                caloJetObj.total_2x2_4 += totalP4.pt();
                continue;
            }

            // Unused l1CaloTowers which are not the initial seed
            // Depending on seed and tower locations calculate iEta/iPhi or eta/phi comparisons
            int hit_iPhi = 99;
            int d_iEta = 99;
            int d_iPhi = 99;
            float d_eta = 99;
            float d_phi = 99;
            if ( caloJetObj.barrelSeeded && l1CaloTower.isBarrel ) // use iEta/iPhi comparisons 
            {
                hit_iPhi = l1CaloTower.tower_iPhi;
                d_iEta = tower_diEta( caloJetObj.seed_iEta, l1CaloTower.tower_iEta );
                d_iPhi = tower_diPhi( caloJetObj.seed_iPhi, hit_iPhi );
            }
            else // either seed or tower are in HGCal, use eta/phi
            {
                d_eta = caloJetObj.seedTower.eta() - l1CaloTower.tower_eta;
                d_phi = reco::deltaPhi( caloJetObj.seedTower.phi(), l1CaloTower.tower_phi );
            }

            // 7x7 HCAL Trigger Towers
            // If seeded in barrel and hit is barrel then we can compare iEta/iPhi, else need to use eta/phi
            // in HGCal / transition region
            if ( (caloJetObj.barrelSeeded && l1CaloTower.isBarrel && abs( d_iEta ) <= 3 && abs( d_iPhi ) <= 3) ||
                    ( fabs( d_eta ) < 0.3 && fabs( d_phi ) < 0.3 ) )
            {

                // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
                reco::Candidate::PolarLorentzVector hcalP4( l1CaloTower.hcal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector ecalP4( l1CaloTower.ecal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector totalP4( l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);

                //caloJetObj.total_map[4+d_iEta][4+d_iPhi] = l1CaloTower.total_tower_et; // 9x9 array
                //caloJetObj.ecal_map[4+d_iEta][4+d_iPhi]  = l1CaloTower.ecal_tower_et; // 9x9 array
                //caloJetObj.hcal_map[4+d_iEta][4+d_iPhi]  = l1CaloTower.hcal_tower_et; // 9x9 array
                //caloJetObj.l1eg_map[4+d_iEta][4+d_iPhi]  = l1CaloTower.total_tower_plus_L1EGs_et - l1CaloTower.total_tower_et; // 9x9 array

                if (hcalP4.pt() > 0)
                {
                    caloJetObj.hcal_nHits++;
                    caloJetObj.hcalJetCluster += hcalP4;
                    caloJetObj.hcalJetClusterET += l1CaloTower.hcal_tower_et;
                }
                if (ecalP4.pt() > 0) 
                {
                    caloJetObj.ecal_nHits++;
                    caloJetObj.ecalJetCluster += ecalP4;
                    caloJetObj.ecalJetClusterET += l1CaloTower.ecal_tower_et;
                }
                if (totalP4.pt() > 0) 
                {
                    caloJetObj.total_nHits++;
                    caloJetObj.jetCluster += totalP4;
                    caloJetObj.jetClusterET += l1CaloTower.total_tower_et;
                }


                if (debug) printf(" ---- hit %i input     p4 pt %f eta %f phi %f\n", cnt, totalP4.pt(), totalP4.eta(), totalP4.phi());
                if (debug) printf(" ---- hit %i resulting p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetClusterET, caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());

                l1CaloTower.stale = true;
                n_stale++;

                if ( abs( d_iEta ) <= 1    && abs( d_iPhi ) <= 1)
                {
                    caloJetObj.hcal_seed += hcalP4.pt();
                    caloJetObj.ecal_seed += ecalP4.pt();
                    caloJetObj.total_seed += totalP4.pt();
                }
                if ( abs( d_iEta ) <= 2    && abs( d_iPhi ) <= 2)
                {
                    caloJetObj.hcal_3x3 += hcalP4.pt();
                    caloJetObj.ecal_3x3 += ecalP4.pt();
                    caloJetObj.total_3x3 += totalP4.pt();
                }
                if ( abs( d_iEta ) <= 3    && abs( d_iPhi ) <= 3)
                {
                    caloJetObj.hcal_5x5 += hcalP4.pt();
                    caloJetObj.ecal_5x5 += ecalP4.pt();
                    caloJetObj.total_5x5 += totalP4.pt();
                }
                if ( abs( d_iEta ) <= 4    && abs( d_iPhi ) <= 4)
                {
                    caloJetObj.hcal_7x7 += hcalP4.pt();
                    caloJetObj.ecal_7x7 += ecalP4.pt();
                    caloJetObj.total_7x7 += totalP4.pt();
                }

                // Some discrimination vars, 2x2s including central seed
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )
                {
                    caloJetObj.hcal_2x2_1 += hcalP4.pt();
                    caloJetObj.ecal_2x2_1 += ecalP4.pt();
                    caloJetObj.total_2x2_1 += totalP4.pt();
                }
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )
                {
                    caloJetObj.hcal_2x2_2 += hcalP4.pt();
                    caloJetObj.ecal_2x2_2 += ecalP4.pt();
                    caloJetObj.total_2x2_2 += totalP4.pt();
                }
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )
                {
                    caloJetObj.hcal_2x2_3 += hcalP4.pt();
                    caloJetObj.ecal_2x2_3 += ecalP4.pt();
                    caloJetObj.total_2x2_3 += totalP4.pt();
                }
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )
                {
                    caloJetObj.hcal_2x2_4 += hcalP4.pt();
                    caloJetObj.ecal_2x2_4 += ecalP4.pt();
                    caloJetObj.total_2x2_4 += totalP4.pt();
                }
            }
        }

        if (caloJetObj.jetClusterET > 0.0)
        {
            l1CaloJetObjs.push_back( caloJetObj );
        }

    } // end while loop of HCAL TP clustering
        



    // Sort JetClusters so we can begin with the highest pt for next step of jet clustering
    std::sort(begin(l1CaloJetObjs), end(l1CaloJetObjs), [](const l1CaloJetObj& a,
            const l1CaloJetObj& b){return a.jetClusterET > b.jetClusterET;});



    /**************************************************************************
 * Progress to adding L1EGs built from ECAL TPs  9x9 grid.
 * Recall, for 9x9 trigger towers gives diameter 0.78
 ******************************************************************************/


    // Cluster together the L1EGs around existing HCAL Jet
    // Cluster within dEta/dPhi 0.4 which is very close to 0.39 = 9x9/2
    //std::cout << " - Input L1EGs: " << crystalClustersVect.size() << std::endl;
    for (auto &caloJetObj : l1CaloJetObjs)
    {

        // For tracking ecal energy spread
        float ecal_dR0p1_leading = 0.;
        float ecal_dR0p05 = 0.;
        float ecal_dR0p075 = 0.;
        float ecal_dR0p1 = 0.;
        float ecal_dR0p125 = 0.;
        float ecal_dR0p15 = 0.;
        float ecal_dR0p2 = 0.;
        float ecal_dR0p3 = 0.;
        float ecal_dR0p4 = 0.;
        float ecal_nL1EGs = 0.;
        float ecal_nL1EGs_standaloneSS = 0.;
        float ecal_nL1EGs_standaloneIso = 0.;
        float ecal_nL1EGs_trkMatchSS = 0.;
        float ecal_nL1EGs_trkMatchIso = 0.;


        // We are pT ordered so we will always begin with the highest pT L1EG
        int cnt = 0;
        for (auto &l1eg : crystalClustersVect)
        {

            cnt++;
            if (l1eg.stale) continue; // skip L1EGs which are already used

            // skip L1EGs outside the dEta/dPhi 0.4 range
            // cluster w.r.t. HCAL seed so the position doesn't change for every L1EG
            float d_eta = caloJetObj.seedTower.eta() - l1eg.eta();
            float d_phi = reco::deltaPhi( caloJetObj.seedTower.phi(), l1eg.phi() );
            float d_eta_to_leading = -99;
            float d_phi_to_leading = -99;
            if ( fabs( d_eta ) > 0.3 || fabs( d_phi ) > 0.3 ) continue; // 7x7

            if (caloJetObj.leadingL1EGET == 0.0) // this is the first L1EG to seed the L1EG ecal jet
            {
                caloJetObj.leadingL1EG += l1eg.GetP4();
                caloJetObj.leadingL1EGET += l1eg.pt();
                caloJetObj.l1EGjet += l1eg.GetP4();
                caloJetObj.l1EGjetET += l1eg.pt();
                caloJetObj.jetCluster += l1eg.GetP4();
                caloJetObj.jetClusterET += l1eg.pt();
                if (debug) printf(" -- L1EG %i, seeding input     p4 pt %f eta %f phi %f\n", cnt, l1eg.pt(), l1eg.eta(), l1eg.phi());
                if (debug) printf(" -- L1EG %i, seeding resulting p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetClusterET, caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());
                if (debug) printf(" -- L1EG %i, ecal l1eg result  p4 pt %f eta %f phi %f\n", cnt, caloJetObj.leadingL1EGET, caloJetObj.leadingL1EG.eta(), caloJetObj.leadingL1EG.phi());
                if (debug) printf(" -- L1EG %i, ecal l1eg result  p4 pt %f eta %f phi %f\n", cnt, caloJetObj.l1EGjetET, caloJetObj.l1EGjet.eta(), caloJetObj.l1EGjet.phi());
                d_eta_to_leading = 0.;
                d_phi_to_leading = 0.;
            }
            else // subsequent L1EGs
            {
                if (debug) printf(" -- L1EG %i, seeding input     p4 pt %f eta %f phi %f\n", cnt, l1eg.pt(), l1eg.eta(), l1eg.phi());
                if (debug) printf(" -- L1EG %i, seeding resulting p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetClusterET, caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());
                if (debug) printf(" -- L1EG %i, ecal l1eg result  p4 pt %f eta %f phi %f\n", cnt, caloJetObj.l1EGjetET, caloJetObj.l1EGjet.eta(), caloJetObj.l1EGjet.phi());
                caloJetObj.l1EGjet += l1eg.GetP4();
                caloJetObj.l1EGjetET += l1eg.pt();
                caloJetObj.jetCluster += l1eg.GetP4();
                caloJetObj.jetClusterET += l1eg.pt();
                d_eta_to_leading = caloJetObj.leadingL1EG.eta() - l1eg.eta();
                d_phi_to_leading = reco::deltaPhi( caloJetObj.leadingL1EG.phi(), l1eg.GetP4().phi() );
            }

            // For all including the seed and subsequent L1EGs
            ecal_nL1EGs++;
            if (l1eg.passesStandaloneSS ) ecal_nL1EGs_standaloneSS++;
            if (l1eg.passesStandaloneIso ) ecal_nL1EGs_standaloneIso++;
            if (l1eg.passesTrkMatchSS ) ecal_nL1EGs_trkMatchSS++;
            if (l1eg.passesTrkMatchIso ) ecal_nL1EGs_trkMatchIso++;
            l1eg.stale = true;


            // Make energy sums in rings, 1 type is centered on highest pT L1EG
            if ( fabs( d_eta_to_leading ) < 0.1   && fabs( d_phi_to_leading ) < 0.1  )  ecal_dR0p1_leading   += l1eg.GetP4().pt();
            // Other type is centered on the HCAL jet center
            if ( fabs( d_eta ) < 0.05  && fabs( d_phi ) < 0.05 )  ecal_dR0p05  += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.075 && fabs( d_phi ) < 0.075)  ecal_dR0p075 += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.1   && fabs( d_phi ) < 0.1  )  ecal_dR0p1   += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.125 && fabs( d_phi ) < 0.125)  ecal_dR0p125 += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.15  && fabs( d_phi ) < 0.15 )  ecal_dR0p15  += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.2   && fabs( d_phi ) < 0.2  )  ecal_dR0p2   += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.3   && fabs( d_phi ) < 0.3  )  ecal_dR0p3   += l1eg.GetP4().pt();
            if ( fabs( d_eta ) < 0.4   && fabs( d_phi ) < 0.4  )  ecal_dR0p4   += l1eg.GetP4().pt();
        }



        // Do barrel to HGCal transition calibrations for situation when
        // only a portion of the jet can be clustered.
        // This is temporary until there is a HGCal method to stitch them.
        // FIXME commented this out to test initial performance w/ HGCal
        //params["transition_calibration"] = apply_barrel_HGCal_boundary_calibration(
        //    caloJetObj.jetClusterET,
        //    caloJetObj.hcalJetClusterET,
        //    caloJetObj.ecalJetClusterET,
        //    caloJetObj.l1EGjetET,
        //    caloJetObj.seed_iEta );



        params["seed_pt"] = caloJetObj.seedTowerET;
        params["seed_eta"] = caloJetObj.seedTower.eta();
        params["seed_phi"] = caloJetObj.seedTower.phi();
        params["seed_iEta"] = caloJetObj.seed_iEta;
        params["seed_iPhi"] = caloJetObj.seed_iPhi;
        params["seed_energy"] = caloJetObj.seedTower.energy();

        params["hcal_pt"] = caloJetObj.hcalJetClusterET;
        params["hcal_3x3"] = caloJetObj.hcal_3x3;
        params["hcal_5x5"] = caloJetObj.hcal_5x5;
        params["hcal_7x7"] = caloJetObj.hcal_7x7;
        params["hcal_2x2"] = std::max( caloJetObj.hcal_2x2_1, std::max( caloJetObj.hcal_2x2_2, std::max( caloJetObj.hcal_2x2_3, caloJetObj.hcal_2x2_4 )));
        params["hcal_nHits"] = caloJetObj.hcal_nHits;

        params["ecal_seed"] = caloJetObj.ecal_seed;
        params["ecal_3x3"] = caloJetObj.ecal_3x3;
        params["ecal_5x5"] = caloJetObj.ecal_5x5;
        params["ecal_7x7"] = caloJetObj.ecal_7x7;
        params["ecal_2x2"] = std::max( caloJetObj.ecal_2x2_1, std::max( caloJetObj.ecal_2x2_2, std::max( caloJetObj.ecal_2x2_3, caloJetObj.ecal_2x2_4 )));
        params["ecal_nHits"] = caloJetObj.ecal_nHits;

        params["total_et"] = total_et;
        params["total_seed"] = caloJetObj.total_seed;
        params["total_3x3"] = caloJetObj.total_3x3;
        params["total_5x5"] = caloJetObj.total_5x5;
        params["total_7x7"] = caloJetObj.total_7x7;
        params["total_2x2"] = std::max( caloJetObj.total_2x2_1, std::max( caloJetObj.total_2x2_2, std::max( caloJetObj.total_2x2_3, caloJetObj.total_2x2_4 )));
        params["total_nHits"] = caloJetObj.total_nHits;
        params["total_nTowers"] = total_nTowers;


        //// return -9 for energy and dR values for ecalJet as defaults
        float hovere = -9;
        if (caloJetObj.ecalJetClusterET > 0.0)
        {
            hovere = caloJetObj.hcalJetClusterET / ( caloJetObj.ecalJetClusterET + caloJetObj.l1EGjetET );
        }
        //params["deltaR_ecal_vs_jet"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.jetCluster );
        //params["deltaR_L1EGjet_vs_jet"] = get_deltaR( caloJetObj.l1EGjet, caloJetObj.jetCluster );
        //params["deltaR_ecal_vs_hcal"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.hcalJetCluster );
        //params["deltaR_ecal_vs_seed_tower"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.seedTower );
        //params["deltaR_ecal_lead_vs_ecal"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.leadingL1EG );
        //params["deltaR_ecal_lead_vs_jet"] = get_deltaR( caloJetObj.jetCluster, caloJetObj.leadingL1EG );
        //params["deltaR_hcal_vs_jet"] = get_deltaR( caloJetObj.hcalJetCluster, caloJetObj.jetCluster );
        //params["deltaR_hcal_vs_seed_tower"] = get_deltaR( caloJetObj.hcalJetCluster, caloJetObj.seedTower );
        //params["deltaR_ecal_vs_seed"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.seedTower );


        params["ecal_leading_pt"] =     caloJetObj.leadingL1EGET;
        params["ecal_leading_eta"] =    caloJetObj.leadingL1EG.eta();
        params["ecal_leading_phi"] =    caloJetObj.leadingL1EG.phi();
        params["ecal_leading_energy"] = caloJetObj.leadingL1EG.energy();
        params["ecal_L1EG_jet_pt"] =     caloJetObj.l1EGjetET;
        //params["ecal_L1EG_jet_eta"] =    caloJetObj.l1EGjet.eta();
        //params["ecal_L1EG_jet_phi"] =    caloJetObj.l1EGjet.phi();
        //params["ecal_L1EG_jet_energy"] = caloJetObj.l1EGjet.energy();
        params["ecal_dR0p1_leading"] =  ecal_dR0p1_leading;
        params["ecal_dR0p05"] =         ecal_dR0p05;
        params["ecal_dR0p075"] =        ecal_dR0p075;
        params["ecal_dR0p1"] =          ecal_dR0p1;
        params["ecal_dR0p125"] =        ecal_dR0p125;
        params["ecal_dR0p15"] =         ecal_dR0p15;
        params["ecal_dR0p2"] =          ecal_dR0p2;
        params["ecal_dR0p3"] =          ecal_dR0p3;
        params["ecal_dR0p4"] =          ecal_dR0p4;
        params["ecal_nL1EGs"] =         ecal_nL1EGs;
        params["ecal_nL1EGs_standaloneSS"] =  ecal_nL1EGs_standaloneSS;
        params["ecal_nL1EGs_standaloneIso"] =  ecal_nL1EGs_standaloneIso;
        params["ecal_nL1EGs_trkMatchSS"] =    ecal_nL1EGs_trkMatchSS;
        params["ecal_nL1EGs_trkMatchIso"] =    ecal_nL1EGs_trkMatchIso;

        params["ecal_pt"] = caloJetObj.ecalJetClusterET;

        params["jet_pt"] = caloJetObj.jetClusterET;
        params["jet_eta"] = caloJetObj.jetCluster.eta();
        params["jet_phi"] = caloJetObj.jetCluster.phi();
        params["jet_mass"] = caloJetObj.jetCluster.mass();
        params["jet_energy"] = caloJetObj.jetCluster.energy();

        // Calibrations
        params["hcal_calibration"] = get_hcal_calibration( 
            params["jet_pt"],
            params["ecal_pt"],
            params["ecal_L1EG_jet_pt"],
            params["jet_eta"] );
        params["hcal_pt_calibration"] = params["hcal_pt"] * params["hcal_calibration"];
        params["jet_pt_calibration"] = params["hcal_pt_calibration"] +
            params["ecal_pt"] + params["ecal_L1EG_jet_pt"];

        float calibratedPt = -1;
        float ECalIsolation = -1; // Need to loop over 7x7 crystals of unclustered energy
        float totalPtPUcorr = -1;
        l1slhc::L1CaloJet caloJet(caloJetObj.jetCluster, calibratedPt, hovere, ECalIsolation, totalPtPUcorr);
        caloJet.SetExperimentalParams(params);
        //for (int i = 0; i < 9; i++)
        //{
        //    for (int j = 0; j < 9; j++)
        //    {
        //        caloJet.total_map[i][j] = caloJetObj.total_map[i][j];
        //        caloJet.ecal_map[i][j] = caloJetObj.ecal_map[i][j];
        //        caloJet.hcal_map[i][j] = caloJetObj.hcal_map[i][j];
        //        caloJet.l1eg_map[i][j] = caloJetObj.l1eg_map[i][j];
        //    }
        //}

        // Only store jets passing ET threshold and within Barrel
        if (params["jet_pt_calibration"] >= EtMinForCollection)
        {
            L1CaloJetsNoCuts->push_back( caloJet );
            //L1CaloJetsWithCuts->push_back( caloJet );
            reco::Candidate::PolarLorentzVector jet_p4 = reco::Candidate::PolarLorentzVector( 
                    params["jet_pt_calibration"], caloJet.p4().eta(), caloJet.p4().phi(), caloJet.p4().M() );
            L1CaloJetCollectionBXV->push_back( 0, l1t::Jet( jet_p4 ) );

            if (debug) printf("Made a Jet, eta %f phi %f pt %f calibrated pt %f\n", caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi(), caloJetObj.jetClusterET, params["jet_pt_calibration"] );
        }


    } // end jetClusters loop


    iEvent.put(std::move(L1CaloJetsNoCuts),"L1CaloJetsNoCuts");
    //iEvent.put(std::move(L1CaloJetsWithCuts), "L1CaloJetsWithCuts" );
    //iEvent.put(std::move(L1CaloClusterCollectionWithCuts), "L1CaloClusterCollectionWithCuts" );
    iEvent.put(std::move(L1CaloJetCollectionBXV),"L1CaloJetCollectionBXV");

    //printf("end L1CaloJetProducer\n");
}










int
L1CaloJetProducer::ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const
{
    // We shouldn't compare integer indices in endcap, the map is not linear
    // Logic from EBDetId::distancePhi() without the abs()
    int PI = 180;
    int result = iPhi_1 - iPhi_2;
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
}


int
L1CaloJetProducer::tower_diPhi( int &iPhi_1, int &iPhi_2 ) const
{
    // 360 Crystals in full, 72 towers, half way is 36
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
}


// Added b/c of the iEta jump from +1 to -1 across the barrel mid point
int
L1CaloJetProducer::tower_diEta( int &iEta_1, int &iEta_2 ) const
{
    // On same side of barrel
    if (iEta_1 * iEta_2 > 0) return iEta_1 - iEta_2;
    else return iEta_1 - iEta_2 - 1;
}


float
L1CaloJetProducer::get_deltaR( reco::Candidate::PolarLorentzVector &p4_1,
        reco::Candidate::PolarLorentzVector &p4_2) const
{
    // Check that pt is > 0 for both or else reco::deltaR returns bogus values
    if (p4_1.pt() > 0 && p4_2.pt() > 0) 
    {
        return reco::deltaR( p4_1, p4_2 );
    }
    else return -1;
}


// Apply calibrations to HCAL energy based on EM Fraction, Jet Eta, Jet pT
float
L1CaloJetProducer::get_hcal_calibration( float &jet_pt, float &ecal_pt,
        float &ecal_L1EG_jet_pt, float &jet_eta ) const
{

    float em_frac = (ecal_L1EG_jet_pt + ecal_pt) / jet_pt;
    float abs_eta = fabs( jet_eta );
    float tmp_jet_pt = jet_pt;
    if (tmp_jet_pt > 499) tmp_jet_pt = 499;

    // Different indices sizes in different calo regions.
    // Barrel...
    size_t em_index = 0;
    size_t eta_index = 0;
    size_t pt_index = 0;
    float calib = 1.0;
    if (abs_eta <= 1.5)
    {
        // Start loop checking 2nd value
        for( unsigned int i = 1; i < emFractionBinsBarrel.size(); i++)
        {
            if(em_frac <= emFractionBinsBarrel.at(i)) break;
            em_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < absEtaBinsBarrel.size(); i++)
        {
            if(abs_eta <= absEtaBinsBarrel.at(i)) break;
            eta_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < jetPtBins.size(); i++)
        {
            if(tmp_jet_pt <= jetPtBins.at(i)) break;
            pt_index++;
        }
        //printf("Barrel calib emId %i etaId %i jetPtId %i\n",int(em_index),int(eta_index),int(pt_index));
        calib = calibrationsBarrel[ eta_index ][ em_index ][ pt_index ];
    } // end Barrel
    else if (abs_eta <= 3.0) // HGCal
    {
        // Start loop checking 2nd value
        for( unsigned int i = 1; i < emFractionBinsHGCal.size(); i++)
        {
            if(em_frac <= emFractionBinsHGCal.at(i)) break;
            em_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < absEtaBinsHGCal.size(); i++)
        {
            if(abs_eta <= absEtaBinsHGCal.at(i)) break;
            eta_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < jetPtBins.size(); i++)
        {
            if(tmp_jet_pt <= jetPtBins.at(i)) break;
            pt_index++;
        }
        //printf("HGCal calib emId %i etaId %i jetPtId %i\n",int(em_index),int(eta_index),int(pt_index));
        calib = calibrationsHGCal[ eta_index ][ em_index ][ pt_index ];
    } // end HGCal
    else // HF
    {
        // Start loop checking 2nd value
        for( unsigned int i = 1; i < emFractionBinsHF.size(); i++)
        {
            if(em_frac <= emFractionBinsHF.at(i)) break;
            em_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < absEtaBinsHF.size(); i++)
        {
            if(abs_eta <= absEtaBinsHF.at(i)) break;
            eta_index++;
        }

        // Start loop checking 2nd value
        for( unsigned int i = 1; i < jetPtBins.size(); i++)
        {
            if(tmp_jet_pt <= jetPtBins.at(i)) break;
            pt_index++;
        }
        //printf("HF calib emId %i etaId %i jetPtId %i\n",int(em_index),int(eta_index),int(pt_index));
        calib = calibrationsHF[ eta_index ][ em_index ][ pt_index ];
    } // end HF

    //printf(" - jet pt %f index %i\n", jet_pt, int(pt_index));
    //printf(" --- calibration: %f\n", calibrations[ em_index ][ eta_index ][ pt_index ] );

    if(calib > 5 && debug)
    {
        printf(" - l1eg %f, ecal %f, jet %f, em frac %f index %i\n", ecal_L1EG_jet_pt, ecal_pt, jet_pt, em_frac, int(em_index));
        printf(" - eta %f, abs eta %f index %i\n", jet_eta, abs_eta, int(eta_index));
        printf(" - jet pt %f tmp_jet_pt %f index %i\n", jet_pt, tmp_jet_pt, int(pt_index));
        printf(" --- calibration: %f\n\n", calib );
    }
    return calib;
}


// Apply calibrations to all energies except seed tower if jet is close to the 
// barrel / HGCal transition boundary
float
L1CaloJetProducer::apply_barrel_HGCal_boundary_calibration( float &jet_pt, float &hcal_pt, float &ecal_pt,
        float &ecal_L1EG_jet_pt, int &seed_iEta ) const
{

    int abs_iEta = abs( seed_iEta );
    // If full 7x7 is in barrel, return 1.0
    if(abs_iEta < 15) return 1.0;

    // Return values are based on 7x7 jet = 49 TTs normally
    // and are w.r.t. the jet area in the barrel including TT 17 
    float calib = 1.0;
    if(abs_iEta == 15) 
    {
        calib = 49./42.;
    }
    else if(abs_iEta == 16) 
    {
        calib = 49./35.;
    }
    else if(abs_iEta == 17) 
    {
        calib = 49./28.;
    }
    jet_pt = jet_pt * calib;
    hcal_pt = hcal_pt * calib;
    ecal_pt = ecal_pt * calib;
    ecal_L1EG_jet_pt = ecal_L1EG_jet_pt * calib;
    return calib;
}


DEFINE_FWK_MODULE(L1CaloJetProducer);
