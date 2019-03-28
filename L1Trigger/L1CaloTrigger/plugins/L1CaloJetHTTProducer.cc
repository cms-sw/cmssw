// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1CaloJetHTTProducer
//
/**\class L1CaloJetHTTProducer L1CaloJetHTTProducer.cc

Description: 
Use the L1CaloJetProducer collections to calculate
HTT energy sum for CaloJets

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Fri Mar 22 2019
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

// Run2/PhaseI output formats
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
// GenJets if needed
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"


class L1CaloJetHTTProducer : public edm::EDProducer {
    public:
        explicit L1CaloJetHTTProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);

        double EtaMax;
        double PtMin;

        edm::EDGetTokenT<BXVector<l1t::Jet>> bxvCaloJetsToken_;
        edm::Handle<BXVector<l1t::Jet>> bxvCaloJetsHandle;        

        // Gen jet collections are only loaded and used if requested
        // (use_gen_jets == true)
        edm::EDGetTokenT<std::vector<reco::GenJet>> genJetsToken_;
        edm::Handle<std::vector<reco::GenJet>> genJetsHandle;

        bool debug;

        bool use_gen_jets;

};

L1CaloJetHTTProducer::L1CaloJetHTTProducer(const edm::ParameterSet& iConfig) :
    EtaMax(iConfig.getParameter<double>("EtaMax")),
    PtMin(iConfig.getParameter<double>("PtMin")),
    bxvCaloJetsToken_(consumes<BXVector<l1t::Jet>>(iConfig.getParameter<edm::InputTag>("BXVCaloJetsInputTag"))),
    genJetsToken_(consumes<std::vector<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("genJets"))),
    debug(iConfig.getParameter<bool>("debug")),
    use_gen_jets(iConfig.getParameter<bool>("use_gen_jets"))

{


    produces< float >("CaloJetHTT");


}

void L1CaloJetHTTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


    // Output collections
    std::unique_ptr< float > CaloJetHTT(new float);

    *CaloJetHTT = 0.;


    // CaloJet HTT for L1 collections
    if (!use_gen_jets)
    {
        iEvent.getByToken(bxvCaloJetsToken_,bxvCaloJetsHandle);


        if (bxvCaloJetsHandle.isValid())
        {
            for(const auto& caloJet : *bxvCaloJetsHandle.product())
            {
                if (caloJet.pt() < PtMin) continue;
                if ( fabs(caloJet.eta()) > EtaMax) continue;
                *CaloJetHTT += float(caloJet.pt());
            }
        }

        if (debug)
        {
            printf("BXV L1CaloJetCollection JetHTT = %f   for PtMin %f   and EtaMax %f\n", *CaloJetHTT, PtMin, EtaMax);
        }
    }

    // CaloJet HTT for gen jets
    if (use_gen_jets)
    {
        iEvent.getByToken(genJetsToken_,genJetsHandle);


        if (genJetsHandle.isValid())
        {
            for(const auto& genJet : *genJetsHandle.product())
            {
                if (genJet.pt() < PtMin) continue;
                if ( fabs(genJet.eta()) > EtaMax) continue;
                *CaloJetHTT += float(genJet.pt());
            }
        }

        if (debug)
        {
            printf("Gen Jets HTT = %f   for PtMin %f   and EtaMax %f\n", *CaloJetHTT, PtMin, EtaMax);
        }
    }


    iEvent.put(std::move(CaloJetHTT),"CaloJetHTT");

}



DEFINE_FWK_MODULE(L1CaloJetHTTProducer);
