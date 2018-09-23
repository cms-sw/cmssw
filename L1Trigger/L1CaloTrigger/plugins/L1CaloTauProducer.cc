// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1CaloTauProducer
//
/**\class L1CaloTauProducer L1CaloTauProducer.cc

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

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iostream>

#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloJet.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"
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

// For pT calibrations
#include "TF1.h"

// Run2/PhaseI output formats
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"


class L1CaloTauProducer : public edm::EDProducer {
    public:
        explicit L1CaloTauProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);
        //bool cluster_passes_base_cuts(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const;
        int ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const;
        int tower_diPhi( int &iPhi_1, int &iPhi_2 ) const;
        int tower_diEta( int &iEta_1, int &iEta_2 ) const;
        float get_deltaR( reco::Candidate::PolarLorentzVector &p4_1,
                reco::Candidate::PolarLorentzVector &p4_2) const;

        //double EtminForStore;
        //double EcalTpEtMin;
        double EtMinForSeedHit;
        bool debug;
        edm::EDGetTokenT< L1CaloTowerCollection > l1TowerToken_;
        edm::Handle< L1CaloTowerCollection > l1CaloTowerHandle;

        edm::EDGetTokenT<l1slhc::L1EGCrystalClusterCollection> crystalClustersToken_;
        edm::Handle<l1slhc::L1EGCrystalClusterCollection> crystalClustersHandle;
        l1slhc::L1EGCrystalClusterCollection crystalClusters;

        edm::ESHandle<CaloGeometry> caloGeometry_;
        const CaloSubdetectorGeometry * hbGeometry;
        edm::ESHandle<HcalTopology> hbTopology;
        const HcalTopology * hcTopology_;

        // Fit function to scale L1EG Pt to align with electron gen pT
        TF1 ptAdjustFunc = TF1("ptAdjustFunc", "([0] + [1]*TMath::Exp(-[2]*x)) * ([3] + [4]*TMath::Exp(-[5]*x))");


        class l1CaloJetObj
        {
            public:
                reco::Candidate::PolarLorentzVector jetCluster;
                reco::Candidate::PolarLorentzVector hcalJetCluster;
                reco::Candidate::PolarLorentzVector ecalJetCluster;
                reco::Candidate::PolarLorentzVector seedTower;
                reco::Candidate::PolarLorentzVector leadingL1EG;
                reco::Candidate::PolarLorentzVector l1EGjet;
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
                bool passesStandaloneWP = false; // Store whether any of the WPs are passed
                bool passesTrkMatchWP = false; // Store whether any of the WPs are passed
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
        };
};

L1CaloTauProducer::L1CaloTauProducer(const edm::ParameterSet& iConfig) :
    //EtminForStore(iConfig.getParameter<double>("EtminForStore")),
    //EcalTpEtMin(iConfig.getUntrackedParameter<double>("EcalTpEtMin", 0.5)), // Default to 500 MeV
    EtMinForSeedHit(iConfig.getUntrackedParameter<double>("EtMinForSeedHit", 2.5)), // Default to 2.5 GeV
    debug(iConfig.getUntrackedParameter<bool>("debug", false)),
    l1TowerToken_(consumes< L1CaloTowerCollection >(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
    crystalClustersToken_(consumes<l1slhc::L1EGCrystalClusterCollection>(iConfig.getParameter<edm::InputTag>("L1CrystalClustersInputTag")))

{
    produces<l1slhc::L1CaloJetsCollection>("L1CaloTausNoCuts");
    produces<l1slhc::L1CaloJetsCollection>("L1CaloTausWithCuts");
    //produces<l1extra::L1JetParticleCollection>("L1CaloClusterCollectionWithCuts");
    produces< BXVector<l1t::Tau> >("L1CaloClusterCollectionBXVWithCuts");


    // Fit parameters measured on 11 Aug 2018, using 500 MeV threshold for ECAL TPs
    // working in CMSSW 10_1_7
    // Adjustments to be applied to reco cluster pt
    // L1EG cut working points are still a function of non-calibrated pT
    // First order corrections
    ptAdjustFunc.SetParameter( 0, 1.06 );
    ptAdjustFunc.SetParameter( 1, 0.273 );
    ptAdjustFunc.SetParameter( 2, 0.0411 );
    // Residuals
    ptAdjustFunc.SetParameter( 3, 1.00 );
    ptAdjustFunc.SetParameter( 4, 0.567 );
    ptAdjustFunc.SetParameter( 5, 0.288 );
}

void L1CaloTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    // Output collections
    std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloTausNoCuts (new l1slhc::L1CaloJetsCollection );
    std::unique_ptr<l1slhc::L1CaloJetsCollection> L1CaloTausWithCuts( new l1slhc::L1CaloJetsCollection );
    //std::unique_ptr<l1extra::L1JetParticleCollection> L1CaloClusterCollectionWithCuts( new l1extra::L1JetParticleCollection );
    std::unique_ptr<BXVector<l1t::Tau>> L1CaloClusterCollectionBXVWithCuts(new l1t::TauBxCollection);



    // Get calo geometry info split by subdetector
    iSetup.get<CaloGeometryRecord>().get(caloGeometry_);
    hbGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    iSetup.get<HcalRecNumberingRecord>().get(hbTopology);
    hcTopology_ = hbTopology.product();
    HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);
    iEvent.getByToken(crystalClustersToken_,crystalClustersHandle);
    crystalClusters = (*crystalClustersHandle.product());


    
    // Load the ECAL+HCAL tower sums coming from L1EGammaCrystalsEmulatorProducer.cc
    std::vector< SimpleCaloHit > l1CaloTowers;
    
    iEvent.getByToken(l1TowerToken_,l1CaloTowerHandle);
    for (auto& hit : *l1CaloTowerHandle.product())
    {

        SimpleCaloHit l1Hit;
        l1Hit.ecal_tower_et  = hit.ecal_tower_et;
        l1Hit.hcal_tower_et  = hit.hcal_tower_et;
        l1Hit.total_tower_et  = hit.ecal_tower_et + hit.hcal_tower_et;
        l1Hit.tower_iEta  = hit.tower_iEta;
        l1Hit.tower_iPhi  = hit.tower_iPhi;
        l1Hit.tower_eta  = hit.tower_eta;
        l1Hit.tower_phi  = hit.tower_phi;
        l1CaloTowers.push_back( l1Hit );
        if (debug) printf("Tower iEta %i iPhi %i eta %f phi %f ecal_et %f hcal_et %f total_et %f\n", (int)l1Hit.tower_iEta, (int)l1Hit.tower_iPhi, l1Hit.tower_eta, l1Hit.tower_phi, l1Hit.ecal_tower_et, l1Hit.hcal_tower_et, l1Hit.total_tower_et);
    }


    // Make simple L1objects from the L1EG input collection with marker for 'stale'
    // FIXME could later add quality criteria here to help differentiate likely
    // photons/electrons vs. pions. This could be helpful for L1CaloTaus
    std::vector< simpleL1obj > crystalClustersVect;
    for (auto EGammaCand : crystalClusters)
    {
        simpleL1obj l1egObj;
        l1egObj.SetP4(EGammaCand.pt(), EGammaCand.eta(), EGammaCand.phi(), 0.);
        l1egObj.passesStandaloneWP = EGammaCand.standaloneWP();
        l1egObj.passesTrkMatchWP = EGammaCand.looseL1TkMatchWP();
        crystalClustersVect.push_back( l1egObj );
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
    for (auto &l1CaloTower : l1CaloTowers)
    {

        l1CaloTower.total_tower_plus_L1EGs_et = l1CaloTower.total_tower_et;

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
    bool caloJetClusteringFinished = false;
    while (!caloJetClusteringFinished)
    {

        l1CaloJetObj caloJetObj;
        caloJetObj.SetJetClusterP4( 0., 0., 0., 0. );
        caloJetObj.SetHcalJetClusterP4( 0., 0., 0., 0. );
        caloJetObj.SetEcalJetClusterP4( 0., 0., 0., 0. );
        caloJetObj.SetSeedP4( 0., 0., 0., 0. );
        caloJetObj.SetLeadingL1EGP4( 0., 0., 0., 0. );
        caloJetObj.SetL1EGJetP4( 0., 0., 0., 0. );

        // First find highest ET ECAL+HCAL+L1EGs tower and use to seed the 9x9 Jet
        int cnt = 0;
        for (auto &l1CaloTower : l1CaloTowers)
        {

            if (l1CaloTower.stale) continue; // skip l1CaloTowers which are already used

            cnt++;
            if (caloJetObj.jetCluster.pt() == 0.0) // this is the first l1CaloTower to seed the jet
            {
                // Check if the leading unused tower has ET < min for seeding a jet.
                // If so, stop jet clustering
                if (l1CaloTower.total_tower_plus_L1EGs_et < EtMinForSeedHit)
                {
                    caloJetClusteringFinished = true;
                    continue;
                }
                l1CaloTower.stale = true;

                // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
                reco::Candidate::PolarLorentzVector hcalP4( l1CaloTower.hcal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector ecalP4( l1CaloTower.ecal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector totalP4( l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);

                if (hcalP4.energy() > 0) caloJetObj.hcal_nHits++;
                if (ecalP4.energy() > 0) caloJetObj.ecal_nHits++;
                if (totalP4.energy() > 0) caloJetObj.total_nHits++;

                caloJetObj.hcalJetCluster += hcalP4;
                caloJetObj.ecalJetCluster += ecalP4;
                caloJetObj.jetCluster += totalP4;
                caloJetObj.seedTower += totalP4;

                caloJetObj.seed_iEta = l1CaloTower.tower_iEta;
                caloJetObj.seed_iPhi = l1CaloTower.tower_iPhi;


                if (debug) printf(" -- hit %i, seeding input     p4 pt %f eta %f phi %f\n", cnt, l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi);
                if (debug) printf(" -- hit %i, seeding input2    p4 pt %f eta %f phi %f\n", cnt, totalP4.pt(), totalP4.eta(), totalP4.phi());
                if (debug) printf(" -- hit %i, seeding resulting p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetCluster.pt(), caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());
                //std::cout << " ----- input p4 " << l1eg.pt() << " : " << l1eg.eta() << " : " << l1eg.phi() << " : " << l1eg.M() << std::endl;
                //std::cout << " ----- jet seed p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std::endl;

                // Need to add the seed energy to the dR rings
                caloJetObj.hcal_seed += hcalP4.energy();
                caloJetObj.hcal_3x3 += hcalP4.energy();
                caloJetObj.hcal_5x5 += hcalP4.energy();
                caloJetObj.hcal_7x7 += hcalP4.energy();
                caloJetObj.ecal_seed += ecalP4.energy();
                caloJetObj.ecal_3x3 += ecalP4.energy();
                caloJetObj.ecal_5x5 += ecalP4.energy();
                caloJetObj.ecal_7x7 += ecalP4.energy();
                caloJetObj.total_seed += totalP4.energy();
                caloJetObj.total_3x3 += totalP4.energy();
                caloJetObj.total_5x5 += totalP4.energy();
                caloJetObj.total_7x7 += totalP4.energy();

                // Some discrimination vars, 2x2s including central seed
                caloJetObj.hcal_2x2_1 += hcalP4.energy();
                caloJetObj.hcal_2x2_2 += hcalP4.energy();
                caloJetObj.hcal_2x2_3 += hcalP4.energy();
                caloJetObj.hcal_2x2_4 += hcalP4.energy();
                caloJetObj.ecal_2x2_1 += ecalP4.energy();
                caloJetObj.ecal_2x2_2 += ecalP4.energy();
                caloJetObj.ecal_2x2_3 += ecalP4.energy();
                caloJetObj.ecal_2x2_4 += ecalP4.energy();
                caloJetObj.total_2x2_1 += totalP4.energy();
                caloJetObj.total_2x2_2 += totalP4.energy();
                caloJetObj.total_2x2_3 += totalP4.energy();
                caloJetObj.total_2x2_4 += totalP4.energy();
                continue;
            }

            // Unused l1CaloTowers which are not the initial seed
            int hit_iPhi = l1CaloTower.tower_iPhi;
            int d_iEta = tower_diEta( caloJetObj.seed_iEta, l1CaloTower.tower_iEta );
            int d_iPhi = tower_diPhi( caloJetObj.seed_iPhi, hit_iPhi );
            if ( abs( d_iEta ) <= 4 && abs( d_iPhi ) <= 4 ) // 9x9 HCAL Trigger Towers
            {
                //std::cout << " ------- input jet p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std::en     dl;

                // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
                reco::Candidate::PolarLorentzVector hcalP4( l1CaloTower.hcal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector ecalP4( l1CaloTower.ecal_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);
                reco::Candidate::PolarLorentzVector totalP4( l1CaloTower.total_tower_et, l1CaloTower.tower_eta, l1CaloTower.tower_phi, 0.);

                if (hcalP4.energy() > 0) caloJetObj.hcal_nHits++;
                if (ecalP4.energy() > 0) caloJetObj.ecal_nHits++;
                if (totalP4.energy() > 0) caloJetObj.total_nHits++;

                caloJetObj.hcalJetCluster += hcalP4;
                caloJetObj.ecalJetCluster += ecalP4;
                caloJetObj.jetCluster += totalP4;

                if (debug) printf(" ---- hit %i input     p4 pt %f eta %f phi %f\n", cnt, totalP4.pt(), totalP4.eta(), totalP4.phi());
                if (debug) printf(" ---- hit %i resulting p4 pt %f eta %f phi %f\n", cnt, caloJetObj.jetCluster.pt(), caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi());

                //std::cout << " ------- resulting jet p4 " << hcalJet.pt() << " : " << hcalJet.eta() << " : " << hcalJet.phi() << " : " << hcalJet.M()<< std     ::endl;
                l1CaloTower.stale = true;

                if ( abs( d_iEta ) <= 1    && abs( d_iPhi ) <= 1)
                {
                    caloJetObj.hcal_seed += hcalP4.energy();
                    caloJetObj.ecal_seed += ecalP4.energy();
                    caloJetObj.total_seed += totalP4.energy();
                }
                if ( abs( d_iEta ) <= 2    && abs( d_iPhi ) <= 2)
                {
                    caloJetObj.hcal_3x3 += hcalP4.energy();
                    caloJetObj.ecal_3x3 += ecalP4.energy();
                    caloJetObj.total_3x3 += totalP4.energy();
                }
                if ( abs( d_iEta ) <= 3    && abs( d_iPhi ) <= 3)
                {
                    caloJetObj.hcal_5x5 += hcalP4.energy();
                    caloJetObj.ecal_5x5 += ecalP4.energy();
                    caloJetObj.total_5x5 += totalP4.energy();
                }
                if ( abs( d_iEta ) <= 4    && abs( d_iPhi ) <= 4)
                {
                    caloJetObj.hcal_7x7 += hcalP4.energy();
                    caloJetObj.ecal_7x7 += ecalP4.energy();
                    caloJetObj.total_7x7 += totalP4.energy();
                }

                // Some discrimination vars, 2x2s including central seed
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )
                {
                    caloJetObj.hcal_2x2_1 += hcalP4.energy();
                    caloJetObj.ecal_2x2_1 += ecalP4.energy();
                    caloJetObj.total_2x2_1 += totalP4.energy();
                }
                if ( ( d_iEta == 0 || d_iEta == 1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )
                {
                    caloJetObj.hcal_2x2_2 += hcalP4.energy();
                    caloJetObj.ecal_2x2_2 += ecalP4.energy();
                    caloJetObj.total_2x2_2 += totalP4.energy();
                }
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == 1 ) )
                {
                    caloJetObj.hcal_2x2_3 += hcalP4.energy();
                    caloJetObj.ecal_2x2_3 += ecalP4.energy();
                    caloJetObj.total_2x2_3 += totalP4.energy();
                }
                if ( ( d_iEta == 0 || d_iEta == -1 )  &&  ( d_iPhi == 0 || d_iPhi == -1 ) )
                {
                    caloJetObj.hcal_2x2_4 += hcalP4.energy();
                    caloJetObj.ecal_2x2_4 += ecalP4.energy();
                    caloJetObj.total_2x2_4 += totalP4.energy();
                }
            }
        }

        if (caloJetObj.jetCluster.pt() > 0.0)
        {
            l1CaloJetObjs.push_back( caloJetObj );
        }

    } // end while loop of HCAL TP clustering
        



    // Sort JetClusters so we can begin with the highest pt for next step of jet clustering
    std::sort(begin(l1CaloJetObjs), end(l1CaloJetObjs), [](const l1CaloJetObj& a,
            const l1CaloJetObj& b){return a.jetCluster.pt() > b.jetCluster.pt();});







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
        float ecal_nL1EGs_standalone = 0.;
        float ecal_nL1EGs_trkMatch = 0.;


        // We are pT ordered so we will always begin with the highest pT L1EG
        for (auto &l1eg : crystalClustersVect)
        {

            if (l1eg.stale) continue; // skip L1EGs which are already used

            // skip L1EGs outside the dEta/dPhi 0.4 range
            // cluster w.r.t. HCAL seed so the position doesn't change for every L1EG
            float d_eta = caloJetObj.seedTower.eta() - l1eg.eta();
            float d_phi = reco::deltaPhi( caloJetObj.seedTower.phi(), l1eg.phi() );
            float d_eta_to_leading = -99;
            float d_phi_to_leading = -99;
            if ( fabs( d_eta ) > 0.4 || fabs( d_phi ) > 0.4 ) continue;

            if (caloJetObj.leadingL1EG.pt() == 0.0) // this is the first L1EG to seed the L1EG ecal jet
            {
                caloJetObj.leadingL1EG += l1eg.GetP4();
                caloJetObj.l1EGjet += l1eg.GetP4();
                caloJetObj.jetCluster += l1eg.GetP4();
                d_eta_to_leading = 0.;
                d_phi_to_leading = 0.;
                //std::cout << " ----- input cal jet p4 " << jetCand.pt() << " : " << jetCand.eta() << " : " << jetCand.phi() << std::endl;
                //std::cout << " ----- ecal jet seed p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() <<  std::endl;
            }
            else // subsequent L1EGs
            {
                caloJetObj.l1EGjet += l1eg.GetP4();
                caloJetObj.jetCluster += l1eg.GetP4();
                d_eta_to_leading = caloJetObj.leadingL1EG.eta() - l1eg.eta();
                d_phi_to_leading = reco::deltaPhi( caloJetObj.leadingL1EG.phi(), l1eg.GetP4().phi() );
            }

            // For all including the seed and subsequent L1EGs
            ecal_nL1EGs++;
            if (l1eg.passesStandaloneWP ) ecal_nL1EGs_standalone++;
            if (l1eg.passesTrkMatchWP ) ecal_nL1EGs_trkMatch++;
            l1eg.stale = true;

            // Unused L1EGs which are not the initial ecal jet seed
            //std::cout << " ------- input jet p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() << " : " << ecalJet.M()<< std::endl;
            //std::cout << " ------- resulting jet p4 " << ecalJet.pt() << " : " << ecalJet.eta() << " : " << ecalJet.phi() << " : " << ecalJet.M()<< std::endl;

            // Make energy sums in rings, 1 type is centered on highest pT L1EG
            if ( fabs( d_eta_to_leading ) < 0.1   && fabs( d_phi_to_leading ) < 0.1  )  ecal_dR0p1_leading   += l1eg.GetP4().energy();
            // Other type is centered on the HCAL jet center
            if ( fabs( d_eta ) < 0.05  && fabs( d_phi ) < 0.05 )  ecal_dR0p05  += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.075 && fabs( d_phi ) < 0.075)  ecal_dR0p075 += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.1   && fabs( d_phi ) < 0.1  )  ecal_dR0p1   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.125 && fabs( d_phi ) < 0.125)  ecal_dR0p125 += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.15  && fabs( d_phi ) < 0.15 )  ecal_dR0p15  += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.2   && fabs( d_phi ) < 0.2  )  ecal_dR0p2   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.3   && fabs( d_phi ) < 0.3  )  ecal_dR0p3   += l1eg.GetP4().energy();
            if ( fabs( d_eta ) < 0.4   && fabs( d_phi ) < 0.4  )  ecal_dR0p4   += l1eg.GetP4().energy();
        }

        params["hcal_pt"] = caloJetObj.hcalJetCluster.pt();
        params["hcal_eta"] = caloJetObj.hcalJetCluster.eta();
        params["hcal_phi"] = caloJetObj.hcalJetCluster.phi();
        params["hcal_mass"] = caloJetObj.hcalJetCluster.mass();
        params["hcal_energy"] = caloJetObj.hcalJetCluster.energy();

        params["hcal_seed_pt"] = caloJetObj.seedTower.pt();
        params["hcal_seed_eta"] = caloJetObj.seedTower.eta();
        params["hcal_seed_phi"] = caloJetObj.seedTower.phi();
        params["seed_iEta"] = caloJetObj.seed_iEta;
        params["seed_iPhi"] = caloJetObj.seed_iPhi;
        params["hcal_seed_energy"] = caloJetObj.seedTower.energy();
        params["hcal_seed"] = caloJetObj.hcal_seed;
        params["hcal_3x3"] = caloJetObj.hcal_3x3;
        params["hcal_5x5"] = caloJetObj.hcal_5x5;
        params["hcal_7x7"] = caloJetObj.hcal_7x7;
        params["hcal_2x2_1"] = caloJetObj.hcal_2x2_1;
        params["hcal_2x2_2"] = caloJetObj.hcal_2x2_2;
        params["hcal_2x2_3"] = caloJetObj.hcal_2x2_3;
        params["hcal_2x2_4"] = caloJetObj.hcal_2x2_4;
        params["hcal_nHits"] = caloJetObj.hcal_nHits;
        params["ecal_seed"] = caloJetObj.ecal_seed;
        params["ecal_3x3"] = caloJetObj.ecal_3x3;
        params["ecal_5x5"] = caloJetObj.ecal_5x5;
        params["ecal_7x7"] = caloJetObj.ecal_7x7;
        params["ecal_2x2_1"] = caloJetObj.ecal_2x2_1;
        params["ecal_2x2_2"] = caloJetObj.ecal_2x2_2;
        params["ecal_2x2_3"] = caloJetObj.ecal_2x2_3;
        params["ecal_2x2_4"] = caloJetObj.ecal_2x2_4;
        params["ecal_nHits"] = caloJetObj.ecal_nHits;
        params["total_seed"] = caloJetObj.total_seed;
        params["total_3x3"] = caloJetObj.total_3x3;
        params["total_5x5"] = caloJetObj.total_5x5;
        params["total_7x7"] = caloJetObj.total_7x7;
        params["total_2x2_1"] = caloJetObj.total_2x2_1;
        params["total_2x2_2"] = caloJetObj.total_2x2_2;
        params["total_2x2_3"] = caloJetObj.total_2x2_3;
        params["total_2x2_4"] = caloJetObj.total_2x2_4;
        params["total_nHits"] = caloJetObj.total_nHits;


        // return -9 for energy and dR values for ecalJet as defaults
        float hovere = -9;
        if (caloJetObj.ecalJetCluster.pt() > 0.0)
        {
            hovere = caloJetObj.hcalJetCluster.energy() / caloJetObj.ecalJetCluster.energy();
        }
        params["deltaR_ecal_vs_jet"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.jetCluster );
        params["deltaR_L1EGjet_vs_jet"] = get_deltaR( caloJetObj.l1EGjet, caloJetObj.jetCluster );
        params["deltaR_ecal_vs_hcal"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.hcalJetCluster );
        params["deltaR_ecal_vs_seed_tower"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.seedTower );
        params["deltaR_ecal_lead_vs_ecal"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.leadingL1EG );
        params["deltaR_ecal_lead_vs_jet"] = get_deltaR( caloJetObj.jetCluster, caloJetObj.leadingL1EG );
        params["deltaR_hcal_vs_jet"] = get_deltaR( caloJetObj.hcalJetCluster, caloJetObj.jetCluster );
        params["deltaR_hcal_vs_seed_tower"] = get_deltaR( caloJetObj.hcalJetCluster, caloJetObj.seedTower );
        params["deltaR_ecal_vs_hcal_seed"] = get_deltaR( caloJetObj.ecalJetCluster, caloJetObj.seedTower );


        params["ecal_leading_pt"] =     caloJetObj.leadingL1EG.pt();
        params["ecal_leading_eta"] =    caloJetObj.leadingL1EG.eta();
        params["ecal_leading_phi"] =    caloJetObj.leadingL1EG.phi();
        params["ecal_leading_energy"] = caloJetObj.leadingL1EG.energy();
        params["ecal_L1EG_jet_pt"] =     caloJetObj.l1EGjet.pt();
        params["ecal_L1EG_jet_eta"] =    caloJetObj.l1EGjet.eta();
        params["ecal_L1EG_jet_phi"] =    caloJetObj.l1EGjet.phi();
        params["ecal_L1EG_jet_energy"] = caloJetObj.l1EGjet.energy();
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
        params["ecal_nL1EGs_standalone"] =  ecal_nL1EGs_standalone;
        params["ecal_nL1EGs_trkMatch"] =    ecal_nL1EGs_trkMatch;

        params["ecal_pt"] = caloJetObj.ecalJetCluster.pt();
        params["ecal_eta"] = caloJetObj.ecalJetCluster.eta();
        params["ecal_phi"] = caloJetObj.ecalJetCluster.phi();
        params["ecal_mass"] = caloJetObj.ecalJetCluster.mass();
        params["ecal_energy"] = caloJetObj.ecalJetCluster.energy();

        params["jet_pt"] = caloJetObj.jetCluster.pt();
        params["jet_eta"] = caloJetObj.jetCluster.eta();
        params["jet_phi"] = caloJetObj.jetCluster.phi();
        params["jet_mass"] = caloJetObj.jetCluster.mass();
        params["jet_energy"] = caloJetObj.jetCluster.energy();

        float calibratedPt = -1;
        float ECalIsolation = -1; // Need to loop over 7x7 crystals of unclustered energy
        float totalPtPUcorr = -1;
        l1slhc::L1CaloJet caloJet(caloJetObj.jetCluster, calibratedPt, hovere, ECalIsolation, totalPtPUcorr);
        caloJet.SetExperimentalParams(params);

        L1CaloTausNoCuts->push_back( caloJet );
        // Same for the moment...
        L1CaloTausWithCuts->push_back( caloJet );

        if (debug) printf("Made a Jet, eta %f phi %f pt %f\n", caloJetObj.jetCluster.eta(), caloJetObj.jetCluster.phi(), caloJetObj.jetCluster.pt());


    } // end jetClusters loop


    iEvent.put(std::move(L1CaloTausNoCuts),"L1CaloTausNoCuts");
    iEvent.put(std::move(L1CaloTausWithCuts), "L1CaloTausWithCuts" );
    //iEvent.put(std::move(L1CaloClusterCollectionWithCuts), "L1CaloClusterCollectionWithCuts" );
    iEvent.put(std::move(L1CaloClusterCollectionBXVWithCuts),"L1CaloClusterCollectionBXVWithCuts");
}















int
L1CaloTauProducer::ecalXtal_diPhi( int &iPhi_1, int &iPhi_2 ) const
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
L1CaloTauProducer::tower_diPhi( int &iPhi_1, int &iPhi_2 ) const
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
L1CaloTauProducer::tower_diEta( int &iEta_1, int &iEta_2 ) const
{
    // On same side of barrel
    if (iEta_1 * iEta_2 > 0) return iEta_1 - iEta_2;
    else return iEta_1 - iEta_2 - 1;
}


float
L1CaloTauProducer::get_deltaR( reco::Candidate::PolarLorentzVector &p4_1,
        reco::Candidate::PolarLorentzVector &p4_2) const
{
    // Check that pt is > 0 for both or else reco::deltaR returns bogus values
    if (p4_1.pt() == 0) return -9;
    if (p4_2.pt() == 0) return -9;
    return reco::deltaR( p4_1, p4_2 );
}


DEFINE_FWK_MODULE(L1CaloTauProducer);
