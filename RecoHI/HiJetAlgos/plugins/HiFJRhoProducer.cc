// Original Author:  Marta Verweij
//         Created:  Thu, 16 Jul 2015 10:57:12 GMT

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"

// class declaration
class HiFJRhoProducer : public edm::global::EDProducer<> {
public:
  explicit HiFJRhoProducer(const edm::ParameterSet&);
  ~HiFJRhoProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  double calcMedian(std::vector<double>& v) const;
  double calcMd(reco::Jet const& jet) const;
  bool isPackedCandidate(reco::Candidate const* candidate) const;

  // members
  const edm::EDGetTokenT<edm::View<reco::Jet>> jetsToken_;  // input kt jet source
  const unsigned int nExcl_;                                // number of leading jets to exclude
  const unsigned int nExcl2_;                               // number of leading jets to exclude in 2nd eta region
  const double etaMaxExcl_;                                 // max eta for jets to exclude
  const double ptMinExcl_;                                  // min pt for excluded jets
  const double etaMaxExcl2_;                                // max eta for jets to exclude in 2nd eta region
  const double ptMinExcl2_;                                 // min pt for excluded jets in 2nd eta region
  const std::vector<double> etaRanges_;                     // eta boundaries for rho calculation regions
  mutable std::once_flag checkJetCand_;  // check if using packed candidates and cache the information
  mutable bool usingPackedCand_ = false;
};

using namespace reco;

// constructor
HiFJRhoProducer::HiFJRhoProducer(const edm::ParameterSet& iConfig)
    : jetsToken_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jetSource"))),
      nExcl_(iConfig.getParameter<int>("nExcl")),
      nExcl2_(iConfig.getParameter<int>("nExcl2")),
      etaMaxExcl_(iConfig.getParameter<double>("etaMaxExcl")),
      ptMinExcl_(iConfig.getParameter<double>("ptMinExcl")),
      etaMaxExcl2_(iConfig.getParameter<double>("etaMaxExcl2")),
      ptMinExcl2_(iConfig.getParameter<double>("ptMinExcl2")),
      etaRanges_(iConfig.getParameter<std::vector<double>>("etaRanges")) {
  // register your products
  produces<std::vector<double>>("mapEtaEdges");
  produces<std::vector<double>>("mapToRho");
  produces<std::vector<double>>("mapToRhoM");
  produces<std::vector<double>>("ptJets");
  produces<std::vector<double>>("areaJets");
  produces<std::vector<double>>("etaJets");
}

// method called for each event to produce the data
void HiFJRhoProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  // get the inputs jets
  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jetsToken_, jets);

  int neta = static_cast<int>(etaRanges_.size());
  auto mapEtaRangesOut = std::make_unique<std::vector<double>>(neta, -999.);

  for (int ieta = 0; ieta < neta; ieta++) {
    mapEtaRangesOut->at(ieta) = etaRanges_[ieta];
  }
  auto mapToRhoOut = std::make_unique<std::vector<double>>(neta - 1, 1e-6);
  auto mapToRhoMOut = std::make_unique<std::vector<double>>(neta - 1, 1e-6);

  int njets = static_cast<int>(jets->size());

  auto ptJetsOut = std::make_unique<std::vector<double>>();
  ptJetsOut->reserve(njets);
  auto areaJetsOut = std::make_unique<std::vector<double>>();
  areaJetsOut->reserve(njets);
  auto etaJetsOut = std::make_unique<std::vector<double>>();
  etaJetsOut->reserve(njets);

  std::vector<double> rhoVec;
  rhoVec.reserve(njets);
  std::vector<double> rhomVec;
  rhomVec.reserve(njets);

  int nacc = 0;
  unsigned int njetsEx = 0;
  unsigned int njetsEx2 = 0;
  for (auto const& jet : *jets) {
    if (njetsEx < nExcl_ and fabs(jet.eta()) < etaMaxExcl_ and jet.pt() > ptMinExcl_) {
      ++njetsEx;
      continue;
    }
    if (njetsEx2 < nExcl2_ and fabs(jet.eta()) < etaMaxExcl2_ and fabs(jet.eta()) > etaMaxExcl_ and
        jet.pt() > ptMinExcl2_) {
      ++njetsEx2;
      continue;
    }
    float pt = jet.pt();
    float area = jet.jetArea();
    float eta = jet.eta();

    if (eta < mapEtaRangesOut->at(0) || eta > mapEtaRangesOut->at(neta - 1))
      continue;
    if (area > 0.) {
      rhoVec.push_back(pt / area);
      rhomVec.push_back(calcMd(jet) / area);
      ptJetsOut->push_back(pt);
      areaJetsOut->push_back(area);
      etaJetsOut->push_back(eta);
      ++nacc;
    }
  }

  // calculate rho and rhom in eta ranges
  const double radius = 0.2;  // distance kt clusters needs to be from edge
  for (int ieta = 0; ieta < (neta - 1); ++ieta) {
    std::vector<double> rhoVecCur;
    rhoVecCur.reserve(nacc);
    std::vector<double> rhomVecCur;
    rhomVecCur.reserve(nacc);

    double etaMin = mapEtaRangesOut->at(ieta) + radius;
    double etaMax = mapEtaRangesOut->at(ieta + 1) - radius;

    for (int i = 0; i < nacc; ++i) {
      if ((*etaJetsOut)[i] >= etaMin and (*etaJetsOut)[i] < etaMax) {
        rhoVecCur.push_back(rhoVec[i]);
        rhomVecCur.push_back(rhomVec[i]);
      }  // eta selection
    }  // accepted jet loop

    if (not rhoVecCur.empty()) {
      mapToRhoOut->at(ieta) = calcMedian(rhoVecCur);
      mapToRhoMOut->at(ieta) = calcMedian(rhomVecCur);
    }
  }  // eta ranges

  iEvent.put(std::move(mapEtaRangesOut), "mapEtaEdges");
  iEvent.put(std::move(mapToRhoOut), "mapToRho");
  iEvent.put(std::move(mapToRhoMOut), "mapToRhoM");
  iEvent.put(std::move(ptJetsOut), "ptJets");
  iEvent.put(std::move(areaJetsOut), "areaJets");
  iEvent.put(std::move(etaJetsOut), "etaJets");
}

double HiFJRhoProducer::calcMd(reco::Jet const& jet) const {
  // compute md as defined in http://arxiv.org/pdf/1211.2811.pdf

  // loop over the jet constituents
  double sum = 0.;
  for (auto const* daughter : jet.getJetConstituentsQuick()) {
    if (isPackedCandidate(daughter)) {  // packed candidate situation
      auto part = static_cast<pat::PackedCandidate const*>(daughter);
      sum += sqrt(part->mass() * part->mass() + part->pt() * part->pt()) - part->pt();
    } else {
      auto part = static_cast<reco::PFCandidate const*>(daughter);
      sum += sqrt(part->mass() * part->mass() + part->pt() * part->pt()) - part->pt();
    }
  }

  return sum;
}

bool HiFJRhoProducer::isPackedCandidate(const reco::Candidate* candidate) const {
  // check if using packed candidates on the first call and cache the information
  std::call_once(checkJetCand_, [&]() {
    if (typeid(pat::PackedCandidate) == typeid(*candidate))
      usingPackedCand_ = true;
    else if (typeid(reco::PFCandidate) == typeid(*candidate))
      usingPackedCand_ = false;
    else
      throw cms::Exception("WrongJetCollection", "Jet constituents are not particle flow candidates");
  });
  return usingPackedCand_;
}

void HiFJRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetSource", edm::InputTag("kt4PFJets"));
  desc.add<int>("nExcl", 2);
  desc.add<double>("etaMaxExcl", 2.);
  desc.add<double>("ptMinExcl", 20.);
  desc.add<int>("nExcl2", 2);
  desc.add<double>("etaMaxExcl2", 2.);
  desc.add<double>("ptMinExcl2", 20.);
  desc.add<std::vector<double>>("etaRanges", {});
  descriptions.add("hiFJRhoProducer", desc);
}

//--------- method to calculate median ------------------
double HiFJRhoProducer::calcMedian(std::vector<double>& v) const {
  // post-condition: After returning, the elements in v may be reordered and the resulting order is implementation defined.
  // works for even and odd collections
  if (v.empty()) {
    return 0.0;
  }
  auto n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  auto med = v[n];
  if (!(v.size() & 1)) {  // if the set size is even
    auto max_it = std::max_element(v.begin(), v.begin() + n);
    med = (*max_it + med) / 2.0;
  }
  return med;
}

// define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HiFJRhoProducer);
