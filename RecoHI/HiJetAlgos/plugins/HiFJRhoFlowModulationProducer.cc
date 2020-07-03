// -*- C++ -*-
//
// Package:    RecoHI/HiJetAlgos/plugins/HiFJRhoFlowModulationProducer
// Class:      HiFJRhoFlowModulationProducer

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
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TF1.h"
#include "TH1.h"
#include "TMath.h"
#include "TMinuitMinimizer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
  double lineFunction(double* x, double* par) { return par[0]; }
  double flowFunction(double* x, double* par) {
    return par[0] * (1. + 2. * (par[1] * TMath::Cos(2. * (x[0] - par[2])) + par[3] * TMath::Cos(3. * (x[0] - par[4]))));
  }
};  // namespace

class HiFJRhoFlowModulationProducer : public edm::stream::EDProducer<> {
public:
  explicit HiFJRhoFlowModulationProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const bool doEvtPlane_;
  const bool doFreePlaneFit_;
  const bool doJettyExclusion_;
  const int evtPlaneLevel_;
  edm::EDGetTokenT<reco::JetView> jetTag_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandsToken_;
  const edm::EDGetTokenT<reco::EvtPlaneCollection> evtPlaneToken_;
  static constexpr int kMaxEvtPlane = 1000;
  std::array<float, kMaxEvtPlane> hiEvtPlane_;
  std::unique_ptr<TF1> lineFit_p;
  std::unique_ptr<TF1> flowFit_p;
  int nPhiBins;
};
HiFJRhoFlowModulationProducer::HiFJRhoFlowModulationProducer(const edm::ParameterSet& iConfig)
    : doEvtPlane_(iConfig.getParameter<bool>("doEvtPlane")),
      doFreePlaneFit_(iConfig.getParameter<bool>("doFreePlaneFit")),
      doJettyExclusion_(iConfig.getParameter<bool>("doJettyExclusion")),
      evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel")),
      pfCandsToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandSource"))),
      evtPlaneToken_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))) {
  if (doJettyExclusion_)
    jetTag_ = consumes<reco::JetView>(iConfig.getParameter<edm::InputTag>("jetTag"));
  produces<std::vector<double>>("rhoFlowFitParams");
  TMinuitMinimizer::UseStaticMinuit(false);
  lineFit_p = std::unique_ptr<TF1>(new TF1("lineFit", lineFunction, -TMath::Pi(), TMath::Pi()));
  flowFit_p = std::unique_ptr<TF1>(new TF1("flowFit", flowFunction, -TMath::Pi(), TMath::Pi()));
}

// ------------ method called to produce the data  ------------
void HiFJRhoFlowModulationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& pfCands = iEvent.get(pfCandsToken_);

  if (doEvtPlane_) {
    auto const& evtPlanes = iEvent.get(evtPlaneToken_);
    assert(evtPlanes.size() < kMaxEvtPlane);
    std::transform(evtPlanes.begin(), evtPlanes.end(), hiEvtPlane_.begin(), [this](auto const& ePlane) -> float {
      return ePlane.angle(evtPlaneLevel_);
    });
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

  std::vector<bool> pfcuts;
  for (auto const& pfCandidate : pfCands) {
    if (pfCandidate.particleId() != 1) {
      pfcuts.push_back(false);
      continue;
    }
    if (pfCandidate.eta() < -1.0) {
      pfcuts.push_back(false);
      continue;
    }
    if (pfCandidate.eta() > 1.0) {
      pfcuts.push_back(false);
      continue;
    }
    if (pfCandidate.pt() < .3) {
      pfcuts.push_back(false);
      continue;
    }
    if (pfCandidate.pt() > 3.) {
      pfcuts.push_back(false);
      continue;
    }

    if (doJettyExclusion_) {
      bool isGood = true;
      for (auto const& jet : *jets) {
        if (deltaR2(jet, pfCandidate) < .16) {
          isGood = false;
          break;
        }
      }
      if (!isGood) {
        pfcuts.push_back(false);
        continue;
      }
    }

    nFill++;
    pfcuts.push_back(true);

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
    eventPlane2 = hiEvtPlane_[8];
    eventPlane3 = hiEvtPlane_[15];
  }
  int pfcuts_count = 0;
  if (nFill >= 100 && eventPlane2 > -99) {
    nPhiBins = std::max(10, nFill / 30);

    std::string name = "phiTestIEta4_" + std::to_string(iEvent.id().event()) + "_h";
    std::string nameFlat = "phiTestIEta4_Flat_" + std::to_string(iEvent.id().event()) + "_h";
    TH1F* phi_h = new TH1F(name.data(), "", nPhiBins, -TMath::Pi(), TMath::Pi());
    for (auto const& pfCandidate : pfCands) {
      if (pfcuts.at(pfcuts_count))
        phi_h->Fill(pfCandidate.phi());
      pfcuts_count++;
    }
    flowFit_p->SetParameter(0, 10);
    flowFit_p->SetParameter(1, 0);
    flowFit_p->SetParameter(2, eventPlane2);
    flowFit_p->SetParameter(3, 0);
    flowFit_p->SetParameter(4, eventPlane3);
    if (!doFreePlaneFit_) {
      flowFit_p->FixParameter(2, eventPlane2);
      flowFit_p->FixParameter(4, eventPlane3);
    }

    lineFit_p->SetParameter(0, 10);

    phi_h->Fit(flowFit_p.get(), "Q SERIAL", "", -TMath::Pi(), TMath::Pi());
    phi_h->Fit(lineFit_p.get(), "Q SERIAL", "", -TMath::Pi(), TMath::Pi());
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
    pfcuts.clear();
  }

  iEvent.put(std::move(rhoFlowFitParamsOut), "rhoFlowFitParams");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJRhoFlowModulationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("doEvtPlane", false);
  desc.add<edm::InputTag>("EvtPlane", edm::InputTag("hiEvtPlane"));
  desc.add<bool>("doJettyExclusion", false);
  desc.add<bool>("doFreePlaneFit", false);
  desc.add<bool>("doFlatTest", false);
  desc.add<edm::InputTag>("jetTag", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("pfCandSource", edm::InputTag("particleFlow"));
  desc.add<int>("evtPlaneLevel", 0);
  descriptions.add("hiFJRhoFlowModulationProducer", desc);
}

DEFINE_FWK_MODULE(HiFJRhoFlowModulationProducer);
