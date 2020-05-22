// -*- C++ -*-
//
// Package:    RecoHI/HiJetAlgos/plugins/HiFJRhoFlowModulationProducer
// Class:      HiFJRhoFlowModulationProducer

#include "RecoHI/HiJetAlgos/plugins/HiFJRhoFlowModulationProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

#include "TF1.h"
#include "TH1.h"
#include "TMath.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

HiFJRhoFlowModulationProducer::HiFJRhoFlowModulationProducer(const edm::ParameterSet& iConfig)
  : doEvtPlane_(iConfig.getParameter<bool>("doEvtPlane")),
    doFreePlaneFit_(iConfig.getParameter<bool>("doFreePlaneFit")),
    doJettyExclusion_(iConfig.getParameter<bool>("doJettyExclusion")),
    evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel")),
    jetTag_(consumes<reco::JetView>(iConfig.getParameter<edm::InputTag>("jetTag"))),
    pfCandsToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandSource"))),
    evtPlaneToken_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))) {
  produces<std::vector<double>>("rhoFlowFitParams");
}

HiFJRhoFlowModulationProducer::~HiFJRhoFlowModulationProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
void HiFJRhoFlowModulationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::PFCandidateCollection> pfCands;
  iEvent.getByToken(pfCandsToken_, pfCands);

  edm::Handle<reco::EvtPlaneCollection> evtPlanes;
  if (doEvtPlane_) {
    iEvent.getByToken(evtPlaneToken_, evtPlanes);
    if (evtPlanes.isValid()) {
      for (unsigned int i = 0; i < evtPlanes->size(); ++i) {
        hiEvtPlane[i] = (*evtPlanes)[i].angle(evtPlaneLevel_);
      }
    }
  }

  edm::Handle<reco::JetView> jets;
  if (doJettyExclusion_)
    iEvent.getByToken(jetTag_, jets);

  int nFill = 0;

  double eventPlane2 = -100;
  double eventPlane3 = -100;

  constexpr int nParamVals = 9;
  auto rhoFlowFitParamsOut = std::make_unique<std::vector<double>>(nParamVals, 1e-6);

  rhoFlowFitParamsOut->at(0) = 0;
  rhoFlowFitParamsOut->at(1) = 0;
  rhoFlowFitParamsOut->at(2) = 0;
  rhoFlowFitParamsOut->at(3) = 0;
  rhoFlowFitParamsOut->at(4) = 0;

  double eventPlane2Cos = 0;
  double eventPlane2Sin = 0;

  double eventPlane3Cos = 0;
  double eventPlane3Sin = 0;

  for (auto const& pfCandidate : *pfCands) {
    if (pfCandidate.eta() < -1.0)
      continue;
    if (pfCandidate.eta() > 1.0)
      continue;
    if (pfCandidate.pt() < .3)
      continue;
    if (pfCandidate.pt() > 3.)
      continue;
    if (pfCandidate.particleId() != 1)
      continue;

    if (doJettyExclusion_) {
      bool isGood = true;
      for (auto const& jet : *jets) {
        if (deltaR2(jet, pfCandidate) < .16) {
          isGood = false;
          break;
        }
      }
      if (!isGood)
        continue;
    }

    nFill++;

    if (!doEvtPlane_) {
      eventPlane2Cos += std::cos(2 * pfCandidate.phi());
      eventPlane2Sin += std::sin(2 * pfCandidate.phi());

      eventPlane3Cos += std::cos(3 * pfCandidate.phi());
      eventPlane3Sin += std::sin(3 * pfCandidate.phi());
    }
  }

  if (!doEvtPlane_) {
    eventPlane2 = std::atan2(eventPlane2Sin, eventPlane2Cos) / 2.;
    eventPlane3 = std::atan2(eventPlane3Sin, eventPlane3Cos) / 3.;
  } else {
    eventPlane2 = hiEvtPlane[8];
    eventPlane3 = hiEvtPlane[15];
  }

  if (nFill >= 100 && eventPlane2 > -99) {
    const int nPhiBins = std::max(10, nFill / 30);

    std::string name = "phiTestIEta4_" + std::to_string(iEvent.id().event()) + "_h";
    std::string nameFlat = "phiTestIEta4_Flat_" + std::to_string(iEvent.id().event()) + "_h";

    TH1F* phi_h = new TH1F(name.data(), "", nPhiBins, -TMath::Pi(), TMath::Pi());

    for (auto const& pfCandidate : *pfCands) {
      if (pfCandidate.eta() < -1.0)
        continue;
      if (pfCandidate.eta() > 1.0)
        continue;
      if (pfCandidate.pt() < .3)
        continue;
      if (pfCandidate.pt() > 3.)
        continue;
      if (pfCandidate.particleId() != 1)
        continue;

      if (doJettyExclusion_) {
        bool isGood = true;
        for (auto const& jet : *jets) {
          if (deltaR2(jet, pfCandidate) < .16) {
            isGood = false;
            break;
          }
        }
        if (!isGood)
          continue;
      }

      phi_h->Fill(pfCandidate.phi());
    }

    std::string flowFitForm = "[0]*(1.+2.*([1]*TMath::Cos(2.*(x-[2]))+[3]*TMath::Cos(3.*(x-[4]))))";

    TF1* flowFit_p = new TF1("flowFit", flowFitForm.c_str(), -TMath::Pi(), TMath::Pi());
    flowFit_p->SetParameter(0, 10);
    flowFit_p->SetParameter(1, 0);
    flowFit_p->SetParameter(2, eventPlane2);
    flowFit_p->SetParameter(3, 0);
    flowFit_p->SetParameter(4, eventPlane3);

    if (!doFreePlaneFit_) {
      flowFit_p->FixParameter(2, eventPlane2);
      flowFit_p->FixParameter(4, eventPlane3);
    }

    TF1* lineFit_p = new TF1("lineFit", "[0]", -TMath::Pi(), TMath::Pi());
    lineFit_p->SetParameter(0, 10);

    phi_h->Fit(flowFit_p, "Q", "", -TMath::Pi(), TMath::Pi());
    phi_h->Fit(lineFit_p, "Q", "", -TMath::Pi(), TMath::Pi());

    rhoFlowFitParamsOut->at(0) = flowFit_p->GetParameter(0);
    rhoFlowFitParamsOut->at(1) = flowFit_p->GetParameter(1);
    rhoFlowFitParamsOut->at(2) = flowFit_p->GetParameter(2);
    rhoFlowFitParamsOut->at(3) = flowFit_p->GetParameter(3);
    rhoFlowFitParamsOut->at(4) = flowFit_p->GetParameter(4);

    rhoFlowFitParamsOut->at(5) = flowFit_p->GetChisquare();
    rhoFlowFitParamsOut->at(6) = flowFit_p->GetNDF();

    rhoFlowFitParamsOut->at(7) = lineFit_p->GetChisquare();
    rhoFlowFitParamsOut->at(8) = lineFit_p->GetNDF();

    delete phi_h;

    delete flowFit_p;
    delete lineFit_p;
  }

  iEvent.put(std::move(rhoFlowFitParamsOut), "rhoFlowFitParams");
}

// ------------ method called once each job just before starting event loop  ------------
void HiFJRhoFlowModulationProducer::beginJob() {
  if (doEvtPlane_) {
    constexpr int kMaxEvtPlanes = 1000;
    hiEvtPlane = new float[kMaxEvtPlanes];
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void HiFJRhoFlowModulationProducer::endJob() {
  if (doEvtPlane_)
    delete hiEvtPlane;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJRhoFlowModulationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HiFJRhoFlowModulationProducer);
