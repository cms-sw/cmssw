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
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloJet.h"


// Run2/PhaseI output formats
#include "DataFormats/L1Trigger/interface/Jet.h"


class L1CaloJetHTTProducer : public edm::EDProducer {
    public:
        explicit L1CaloJetHTTProducer(const edm::ParameterSet&);

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);

        double EtaMax;
        double PtMin;

        edm::EDGetTokenT<l1slhc::L1CaloJetsCollection> caloJetsToken_;
        edm::Handle<l1slhc::L1CaloJetsCollection> caloJetsHandle; 

        edm::EDGetTokenT<BXVector<l1t::Jet>> bxvCaloJetsToken_;
        edm::Handle<BXVector<l1t::Jet>> bxvCaloJetsHandle;        


        bool debug;

        // Add CaloJets FIXME XXX
};

L1CaloJetHTTProducer::L1CaloJetHTTProducer(const edm::ParameterSet& iConfig) :
    EtaMax(iConfig.getParameter<double>("EtaMax")),
    PtMin(iConfig.getParameter<double>("PtMin")),
    caloJetsToken_(consumes<l1slhc::L1CaloJetsCollection>(iConfig.getParameter<edm::InputTag>("L1CaloJetsInputTag"))),
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



    iEvent.getByToken(caloJetsToken_,caloJetsHandle);
    iEvent.getByToken(bxvCaloJetsToken_,bxvCaloJetsHandle);



    float tmp_CaloJetHTT = 0.;
    if (caloJetsHandle.isValid())
    {
        for(const auto& caloJet : *caloJetsHandle.product())
        {
            if (caloJet.GetExperimentalParam("jet_pt_calibration") < PtMin) continue;
            if ( fabs(caloJet.GetExperimentalParam("jet_eta")) > EtaMax) continue;
            tmp_CaloJetHTT += float(caloJet.GetExperimentalParam("jet_pt_calibration"));
        }
    }



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
        if (*CaloJetHTT != tmp_CaloJetHTT)
        {
            printf("BXV Method: %f    CaloJetCustom Method: %f    BXV/Cust: %f\n", *CaloJetHTT, tmp_CaloJetHTT, *CaloJetHTT/tmp_CaloJetHTT);
        }
    }


    iEvent.put(std::move(CaloJetHTT),"CaloJetHTT");

    //printf("end L1CaloJetHTTProducer\n");
}



DEFINE_FWK_MODULE(L1CaloJetHTTProducer);
