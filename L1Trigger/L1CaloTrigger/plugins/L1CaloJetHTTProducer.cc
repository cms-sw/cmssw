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


class L1CaloJetHTTProducer : public edm::EDProducer {
    public:
        explicit L1CaloJetHTTProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);

        double EtaMax;
        double PtMin;

        edm::EDGetTokenT<BXVector<l1t::Jet>> bxvCaloJetsToken_;
        edm::Handle<BXVector<l1t::Jet>> bxvCaloJetsHandle;        


        bool debug;

        // Add CaloJets FIXME XXX
};

L1CaloJetHTTProducer::L1CaloJetHTTProducer(const edm::ParameterSet& iConfig) :
    EtaMax(iConfig.getParameter<double>("EtaMax")),
    PtMin(iConfig.getParameter<double>("PtMin")),
    bxvCaloJetsToken_(consumes<BXVector<l1t::Jet>>(iConfig.getParameter<edm::InputTag>("BXVCaloJetsInputTag"))),
    debug(iConfig.getParameter<bool>("debug"))

{


    produces< float >("CaloJetHTT");


}

void L1CaloJetHTTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    //printf("begin L1CaloJetHTTProducer\n");


    // Output collections
    std::unique_ptr< float > CaloJetHTT(new float);



    iEvent.getByToken(bxvCaloJetsToken_,bxvCaloJetsHandle);


    *CaloJetHTT = 0.;
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


    iEvent.put(std::move(CaloJetHTT),"CaloJetHTT");

    //printf("end L1CaloJetHTTProducer\n");
}



DEFINE_FWK_MODULE(L1CaloJetHTTProducer);
