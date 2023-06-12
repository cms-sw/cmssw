// -*- C++ -*-
//
// Package:    RecoHI/HiJetAlgos/plugins/HiFJRhoFlowModulationProducer
// Class:      HiFJRhoFlowModulationProducer

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

#include "TF1.h"
#include "TF2.h"
#include "TH1.h"
#include "TMath.h"
#include "Math/ProbFuncMathCore.h"
#include "TMinuitMinimizer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

  double flowFunction(double* x, double* par) {
    unsigned int nFlow = par[0];          // Number of fitted flow components is defined in the first parameter
    unsigned int firstFittedVn = par[1];  // The first fitted flow component is defined in the second parameter

    // Add each component separately to the total fit value
    double flowModulation = par[2];
    for (unsigned int iFlow = 0; iFlow < nFlow; iFlow++) {
      flowModulation +=
          par[2] * 2.0 * par[2 * iFlow + 3] * std::cos((iFlow + firstFittedVn) * (x[0] - par[2 * iFlow + 4]));
    }
    return flowModulation;
  }

  /*
   * Weighing function for particle flow candidates in case jetty regions are exluded from the flow fit
   *   x[0] = DeltaPhi between jet axis and particle flow candidate in range [0,2Pi]
   *   x[1] = Absolute value of the jet eta
   *   par[0] = Eta cut applied for the particle flow candidates
   *   par[1] = Exclusion radius around the jet axis
   *   
   *   return: The weight to be used with this particle flow candidate is (1 + number provided by this function)
   */
  double weightFunction(double* x, double* par) {
    // If the particle flow candidate is farther than the exclusion radius from the jet axis, no need for weighting due to area where no particle flow candidates can be found
    if (x[0] > par[1])
      return 0;

    // If the particle flow candidate is closer than 0.4 in phi from the jet axis, then there is part of phi acceptance from which no particle flow candidates can be found. Calculate the half of the size of that strip in eta
    double exclusionArea = TMath::Sqrt(par[1] * par[1] - x[0] * x[0]);

    // Particle flow candidates are only considered until eta = par[0]. Check that the exclusion area does not go beyond that

    // First, check cases where the jet is found outside of the particle flow candidate acceptance in eta
    if (x[1] > par[0]) {
      // Check if the whole exclusion area is outside of the acceptance. If this is the case, we should not add anything to the weight for this particle flow candidate.
      if (x[1] - exclusionArea > par[0])
        return 0;

      // Now we know that part of the exclusion area will be inside of the acceptance. Check how big this is.
      exclusionArea = par[0] - (x[1] - exclusionArea);

    } else {
      // In the next case, the jet is found inside of the particle flow candidate acceptance. In this case, we need to check if some of the exclusion area goes outside of the acceptance. If is does, we need to exclude that from the exclusion area
      if (x[1] + exclusionArea > par[0]) {
        exclusionArea += par[0] - x[1];
      } else {
        // If not, the the exclusion area is two times half of the nominal exclusion area
        exclusionArea *= 2;
      }
    }

    // Normalize the exclusion area to the total acceptance. This number should be added to the total weight for the particle flow candidate
    return (2 * par[0]) / (2 * par[0] - exclusionArea) - 1;
  }
};  // namespace

class HiFJRhoFlowModulationProducer : public edm::stream::EDProducer<> {
public:
  explicit HiFJRhoFlowModulationProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  const int minPfCandidatesPerEvent_;
  const bool doEvtPlane_;
  const bool doFreePlaneFit_;
  const bool doJettyExclusion_;
  const double exclusionRadius_;
  const int evtPlaneLevel_;
  const double pfCandidateEtaCut_;
  const double pfCandidateMinPtCut_;
  const double pfCandidateMaxPtCut_;
  int firstFittedVn_;
  int lastFittedVn_;
  const edm::EDGetTokenT<reco::JetView> jetToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidateToken_;
  const edm::EDGetTokenT<reco::EvtPlaneCollection> evtPlaneToken_;
  std::unique_ptr<TF1> flowFit_p_;
  std::unique_ptr<TF2> pfWeightFunction_;
  reco::PFCandidate converter_;
};
HiFJRhoFlowModulationProducer::HiFJRhoFlowModulationProducer(const edm::ParameterSet& iConfig)
    : minPfCandidatesPerEvent_(iConfig.getParameter<int>("minPfCandidatesPerEvent")),
      doEvtPlane_(iConfig.getParameter<bool>("doEvtPlane")),
      doFreePlaneFit_(iConfig.getParameter<bool>("doFreePlaneFit")),
      doJettyExclusion_(iConfig.getParameter<bool>("doJettyExclusion")),
      exclusionRadius_(iConfig.getParameter<double>("exclusionRadius")),
      evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel")),
      pfCandidateEtaCut_(iConfig.getParameter<double>("pfCandidateEtaCut")),
      pfCandidateMinPtCut_(iConfig.getParameter<double>("pfCandidateMinPtCut")),
      pfCandidateMaxPtCut_(iConfig.getParameter<double>("pfCandidateMaxPtCut")),
      firstFittedVn_(iConfig.getParameter<int>("firstFittedVn")),
      lastFittedVn_(iConfig.getParameter<int>("lastFittedVn")),
      jetToken_(doJettyExclusion_ ? consumes<reco::JetView>(iConfig.getParameter<edm::InputTag>("jetTag"))
                                  : edm::EDGetTokenT<reco::JetView>()),
      pfCandidateToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandSource"))),
      evtPlaneToken_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))) {
  produces<std::vector<double>>("rhoFlowFitParams");

  // Converter to get PF candidate ID from packed candidates
  converter_ = reco::PFCandidate();

  // Sanity check for the fitted flow component input
  if (firstFittedVn_ < 1)
    firstFittedVn_ = 2;
  if (lastFittedVn_ < 1)
    lastFittedVn_ = 2;
  if (firstFittedVn_ > lastFittedVn_) {
    int flipper = lastFittedVn_;
    lastFittedVn_ = firstFittedVn_;
    firstFittedVn_ = flipper;
  }

  // Calculate the number of flow components
  const int nFlow = lastFittedVn_ - firstFittedVn_ + 1;

  // Define all fit and weight functions
  TMinuitMinimizer::UseStaticMinuit(false);
  flowFit_p_ = std::make_unique<TF1>("flowFit", flowFunction, -TMath::Pi(), TMath::Pi(), nFlow * 2 + 3);
  flowFit_p_->FixParameter(0, nFlow);           // The first parameter defines the number of fitted flow components
  flowFit_p_->FixParameter(1, firstFittedVn_);  // The second parameter defines the first fitted flow component
  pfWeightFunction_ = std::make_unique<TF2>("weightFunction", weightFunction, 0, TMath::Pi(), -5, 5, 2);
  pfWeightFunction_->SetParameter(0, pfCandidateEtaCut_);  // Set the allowed eta range for particle flow candidates
  pfWeightFunction_->SetParameter(1, exclusionRadius_);    // Set the exclusion radius around the jets
}

// ------------ method called to produce the data  ------------
void HiFJRhoFlowModulationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the particle flow candidate collection
  auto const& pfCands = iEvent.get(pfCandidateToken_);

  // If we are reading the event plane information for the forest, find the event planes
  std::array<float, hi::NumEPNames> hiEvtPlane;
  if (doEvtPlane_) {
    auto const& evtPlanes = iEvent.get(evtPlaneToken_);
    assert(evtPlanes.size() == hi::NumEPNames);
    std::transform(evtPlanes.begin(), evtPlanes.end(), hiEvtPlane.begin(), [this](auto const& ePlane) -> float {
      return ePlane.angle(evtPlaneLevel_);
    });
  }

  // If jetty regions are excluded from the flow fits, read the jet collection
  edm::Handle<reco::JetView> jets;
  if (doJettyExclusion_) {
    iEvent.getByToken(jetToken_, jets);
  }

  int nFill = 0;

  // Initialize arrays for event planes
  // nFlow: Number of flow components in the fit to particle flow condidate distribution
  const int nFlow = lastFittedVn_ - firstFittedVn_ + 1;
  double eventPlane[nFlow];
  for (int iFlow = 0; iFlow < nFlow; iFlow++)
    eventPlane[iFlow] = -100;

  // Initialize the output vector with flow fit components
  // v_n and Psi_n for each fitted flow component, plus overall normalization factor, chi^2 and ndf for flow fit, and the first fitted flow component
  const int nParamVals = nFlow * 2 + 4;
  auto rhoFlowFitParamsOut = std::make_unique<std::vector<double>>(nParamVals, 1e-6);

  // Set the parameters related to flow fit to zero
  for (int iParameter = 0; iParameter < nFlow * 2 + 1; iParameter++) {
    rhoFlowFitParamsOut->at(iParameter) = 0;
  }
  // Set the first fitted flow component to the last index of the output vector
  rhoFlowFitParamsOut->at(nFlow * 2 + 3) = firstFittedVn_;

  // Initialize arrays for manual event plane calculation
  double eventPlaneCos[nFlow];
  double eventPlaneSin[nFlow];
  for (int iFlow = 0; iFlow < nFlow; iFlow++) {
    eventPlaneCos[iFlow] = 0;
    eventPlaneSin[iFlow] = 0;
  }

  // Define helper variables for looping over particle flow candidates
  std::vector<bool> pfcuts(pfCands.size(), false);
  int iCand = -1;
  double thisPfCandidateWeight = 0;
  double deltaPhiJetPf = 0;
  std::vector<double> pfCandidateWeight(pfCands.size(), 1);

  // Loop over the particle flow candidates
  for (auto const& pfCandidate : pfCands) {
    iCand++;  // Update the particle flow candidate index

    // Find the PF candidate ID from the packed candidates collection
    auto particleFlowCandidateID = converter_.translatePdgIdToType(pfCandidate.pdgId());

    // This cut selects particle flow candidates that are charged hadrons. The ID numbers are:
    // 0 = undefined
    // 1 = charged hadron
    // 2 = electron
    // 3 = muon
    // 4 = photon
    // 5 = neutral hadron
    // 6 = HF tower identified as a hadron
    // 7 = HF tower identified as an EM particle
    if (particleFlowCandidateID != 1)
      continue;

    // Kinematic cuts for particle flow candidate pT and eta
    if (std::abs(pfCandidate.eta()) > pfCandidateEtaCut_)
      continue;
    if (pfCandidate.pt() < pfCandidateMinPtCut_)
      continue;
    if (pfCandidate.pt() > pfCandidateMaxPtCut_)
      continue;

    nFill++;  // Use same nFill criterion with and without jetty region subtraction

    // If the jetty regions are excluded from the fit, find which particle flow candidates are close to jets
    thisPfCandidateWeight = 1;
    if (doJettyExclusion_) {
      bool isGood = true;
      for (auto const& jet : *jets) {
        if (deltaR2(jet, pfCandidate) < exclusionRadius_) {
          isGood = false;
        } else {
          // If the particle flow candidates are not excluded from the fit, check if there are any jets in the same phi-slice as the particle flow candidate. If this is the case, add a weight for the particle flow candidate taking into account the lost acceptance for underlaying event particle flow candidates.
          deltaPhiJetPf = TMath::Abs(pfCandidate.phi() - jet.phi());
          if (deltaPhiJetPf > TMath::Pi())
            deltaPhiJetPf = TMath::Pi() * 2 - deltaPhiJetPf;
          // Weight currently does not take into account overlapping jets
          thisPfCandidateWeight += pfWeightFunction_->Eval(deltaPhiJetPf, TMath::Abs(jet.eta()));
        }
      }
      // Do not use this particle flow candidate in the manual event plane calculation
      if (!isGood) {
        continue;
      }
    }

    // Update the weight for this particle flow candidate
    pfCandidateWeight[iCand] = thisPfCandidateWeight;

    // This particle flow candidate passes all the cuts
    pfcuts[iCand] = true;

    // If the event plane is calculated manually, add this particle flow candidate to the calculation
    if (!doEvtPlane_) {
      for (int iFlow = 0; iFlow < nFlow; iFlow++) {
        eventPlaneCos[iFlow] += std::cos((iFlow + firstFittedVn_) * pfCandidate.phi()) * thisPfCandidateWeight;
        eventPlaneSin[iFlow] += std::sin((iFlow + firstFittedVn_) * pfCandidate.phi()) * thisPfCandidateWeight;
      }
    }
  }

  // Determine the event plane angle
  if (!doEvtPlane_) {
    // Manual calculation for the event plane angle using the particle flow candidates
    for (int iFlow = 0; iFlow < nFlow; iFlow++) {
      eventPlane[iFlow] = std::atan2(eventPlaneSin[iFlow], eventPlaneCos[iFlow]) / (iFlow + firstFittedVn_);
    }
  } else {
    // Read the event plane angle determined from the HF calorimeters from the HiForest
    // Only v2 and v3 are available in the HiForest. This option should not be used if other flow components are fitted
    int halfWay = nFlow / 2;
    for (int iFlow = 0; iFlow < halfWay; iFlow++) {
      eventPlane[iFlow] = hiEvtPlane[hi::HF2];
    }
    for (int iFlow = halfWay; iFlow < nFlow; iFlow++) {
      eventPlane[iFlow] = hiEvtPlane[hi::HF3];
    }
  }

  // Do the flow fit provided that there are enough particle flow candidates in the event and that the event planes have been determined successfully
  int pfcuts_count = 0;
  int nPhiBins = 10;
  if (nFill >= minPfCandidatesPerEvent_ && eventPlane[0] > -99) {
    // Create a particle flow candidate phi-histogram
    nPhiBins = std::max(10, nFill / 30);
    std::string name = "phiTestIEta4_" + std::to_string(iEvent.id().event()) + "_h";
    std::unique_ptr<TH1F> phi_h = std::make_unique<TH1F>(name.data(), "", nPhiBins, -TMath::Pi(), TMath::Pi());
    phi_h->SetDirectory(nullptr);
    for (auto const& pfCandidate : pfCands) {
      if (pfcuts.at(pfcuts_count)) {  // Only use particle flow candidates that pass all cuts
        phi_h->Fill(pfCandidate.phi(), pfCandidateWeight.at(pfcuts_count));
      }
      pfcuts_count++;
    }

    // Set initial values for the fit parameters
    flowFit_p_->SetParameter(2, 10);
    for (int iFlow = 0; iFlow < nFlow; iFlow++) {
      flowFit_p_->SetParameter(3 + 2 * iFlow, 0);
      flowFit_p_->SetParameter(4 + 2 * iFlow, eventPlane[iFlow]);
      // If we are not allowing the event plane angles to be free parameters in the fit, fix them
      if (!doFreePlaneFit_) {
        flowFit_p_->FixParameter(4 + 2 * iFlow, eventPlane[iFlow]);
      }
    }

    // Do the Fourier fit
    phi_h->Fit(flowFit_p_.get(), "Q SERIAL", "", -TMath::Pi(), TMath::Pi());

    // Put the fit parameters to the output vector
    for (int iParameter = 0; iParameter < nFlow * 2 + 1; iParameter++) {
      rhoFlowFitParamsOut->at(iParameter) = flowFit_p_->GetParameter(iParameter + 2);
    }

    // Also add chi2 and ndf information to the output vector
    rhoFlowFitParamsOut->at(nFlow * 2 + 1) = flowFit_p_->GetChisquare();
    rhoFlowFitParamsOut->at(nFlow * 2 + 2) = flowFit_p_->GetNDF();

    phi_h.reset();
    pfcuts.clear();
  }

  // Define the produced output
  iEvent.put(std::move(rhoFlowFitParamsOut), "rhoFlowFitParams");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void HiFJRhoFlowModulationProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void HiFJRhoFlowModulationProducer::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJRhoFlowModulationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("minPfCandidatesPerEvent", 100);
  desc.add<bool>("doEvtPlane", false);
  desc.add<edm::InputTag>("EvtPlane", edm::InputTag("hiEvtPlane"));
  desc.add<bool>("doJettyExclusion", false);
  desc.add<double>("exclusionRadius", 0.4);
  desc.add<bool>("doFreePlaneFit", false);
  desc.add<edm::InputTag>("jetTag", edm::InputTag("ak4PFJetsForFlow"));
  desc.add<edm::InputTag>("pfCandSource", edm::InputTag("packedPFCandidates"));
  desc.add<int>("evtPlaneLevel", 0);
  desc.add<double>("pfCandidateEtaCut", 1.0);
  desc.add<double>("pfCandidateMinPtCut", 0.3);
  desc.add<double>("pfCandidateMaxPtCut", 3.0);
  desc.add<int>("firstFittedVn", 2);
  desc.add<int>("lastFittedVn", 3);
  descriptions.add("hiFJRhoFlowModulationProducer", desc);
}

DEFINE_FWK_MODULE(HiFJRhoFlowModulationProducer);
