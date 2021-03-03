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
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

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
    return par[0] * (1. + 2. * (par[1] * std::cos(2. * (x[0] - par[2])) + par[3] * std::cos(3. * (x[0] - par[4]))));
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
  const edm::EDGetTokenT<reco::JetView> jetToken_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandsToken_;
  const edm::EDGetTokenT<reco::EvtPlaneCollection> evtPlaneToken_;
  std::unique_ptr<TF1> lineFit_p_;
  std::unique_ptr<TF1> flowFit_p_;
};
HiFJRhoFlowModulationProducer::HiFJRhoFlowModulationProducer(const edm::ParameterSet& iConfig)
    : doEvtPlane_(iConfig.getParameter<bool>("doEvtPlane")),
      doFreePlaneFit_(iConfig.getParameter<bool>("doFreePlaneFit")),
      doJettyExclusion_(iConfig.getParameter<bool>("doJettyExclusion")),
      evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel")),
      jetToken_(doJettyExclusion_ ? consumes<reco::JetView>(iConfig.getParameter<edm::InputTag>("jetTag"))
                                  : edm::EDGetTokenT<reco::JetView>()),
      pfCandsToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandSource"))),
      evtPlaneToken_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))) {
  produces<std::vector<double>>("rhoFlowFitParams");
  TMinuitMinimizer::UseStaticMinuit(false);
  lineFit_p_ = std::make_unique<TF1>("lineFit", lineFunction, -TMath::Pi(), TMath::Pi());
  flowFit_p_ = std::make_unique<TF1>("flowFit", flowFunction, -TMath::Pi(), TMath::Pi());
}

// ------------ method called to produce the data  ------------
void HiFJRhoFlowModulationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int nPhiBins = 10;
  auto const& pfCands = iEvent.get(pfCandsToken_);
  static constexpr int kMaxEvtPlane = 29;
  std::array<float, kMaxEvtPlane> hiEvtPlane;

  if (doEvtPlane_) {
    auto const& evtPlanes = iEvent.get(evtPlaneToken_);
    assert(evtPlanes.size() == hi::NumEPNames);
    std::transform(evtPlanes.begin(), evtPlanes.end(), hiEvtPlane.begin(), [this](auto const& ePlane) -> float {
      return ePlane.angle(evtPlaneLevel_);
    });
  }

  edm::Handle<reco::JetView> jets;
  if (doJettyExclusion_)
    iEvent.getByToken(jetToken_, jets);

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

  std::vector<bool> pfcuts(pfCands.size(), false);
  int iCand = -1;
  for (auto const& pfCandidate : pfCands) {
    iCand++;
    if (pfCandidate.particleId() != 1 || std::abs(pfCandidate.eta()) > 1.0 || pfCandidate.pt() < .3 ||
        pfCandidate.pt() > 3.) {
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
        continue;
      }
    }

    nFill++;
    pfcuts[iCand] = true;

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
    eventPlane2 = hiEvtPlane[hi::HF2];
    eventPlane3 = hiEvtPlane[hi::HF3];
  }
  int pfcuts_count = 0;
  if (nFill >= 100 && eventPlane2 > -99) {
    nPhiBins = std::max(10, nFill / 30);

    std::string name = "phiTestIEta4_" + std::to_string(iEvent.id().event()) + "_h";
    std::string nameFlat = "phiTestIEta4_Flat_" + std::to_string(iEvent.id().event()) + "_h";
    std::unique_ptr<TH1F> phi_h = std::make_unique<TH1F>(name.data(), "", nPhiBins, -TMath::Pi(), TMath::Pi());
    phi_h->SetDirectory(nullptr);
    for (auto const& pfCandidate : pfCands) {
      if (pfcuts.at(pfcuts_count))
        phi_h->Fill(pfCandidate.phi());
      pfcuts_count++;
    }
    flowFit_p_->SetParameter(0, 10);
    flowFit_p_->SetParameter(1, 0);
    flowFit_p_->SetParameter(2, eventPlane2);
    flowFit_p_->SetParameter(3, 0);
    flowFit_p_->SetParameter(4, eventPlane3);
    if (!doFreePlaneFit_) {
      flowFit_p_->FixParameter(2, eventPlane2);
      flowFit_p_->FixParameter(4, eventPlane3);
    }

    lineFit_p_->SetParameter(0, 10);

    phi_h->Fit(flowFit_p_.get(), "Q SERIAL", "", -TMath::Pi(), TMath::Pi());
    phi_h->Fit(lineFit_p_.get(), "Q SERIAL", "", -TMath::Pi(), TMath::Pi());
    rhoFlowFitParamsOut->at(0) = flowFit_p_->GetParameter(0);
    rhoFlowFitParamsOut->at(1) = flowFit_p_->GetParameter(1);
    rhoFlowFitParamsOut->at(2) = flowFit_p_->GetParameter(2);
    rhoFlowFitParamsOut->at(3) = flowFit_p_->GetParameter(3);
    rhoFlowFitParamsOut->at(4) = flowFit_p_->GetParameter(4);

    rhoFlowFitParamsOut->at(5) = flowFit_p_->GetChisquare();
    rhoFlowFitParamsOut->at(6) = flowFit_p_->GetNDF();

    rhoFlowFitParamsOut->at(7) = lineFit_p_->GetChisquare();
    rhoFlowFitParamsOut->at(8) = lineFit_p_->GetNDF();

    phi_h.reset();
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
