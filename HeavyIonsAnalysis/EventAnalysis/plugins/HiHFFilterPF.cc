// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

class HiHFFilterPF : public edm::one::EDFilter<> {
public:
  explicit HiHFFilterPF(const edm::ParameterSet&);
  ~HiHFFilterPF() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidateTag_;
  const double threshold_;
  const int minnumtowers_;
  int numMinHFTowersP, numMinHFTowersM;

};

using namespace edm;
using namespace std;

HiHFFilterPF::HiHFFilterPF(const edm::ParameterSet& iConfig) :
  pfCandidateTag_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidateSrc"))),
  threshold_(iConfig.getParameter<double>("threshold")),
  minnumtowers_(iConfig.getParameter<int>("minnumtowers")) { }

HiHFFilterPF::~HiHFFilterPF() {}

bool HiHFFilterPF::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;

  numMinHFTowersP = 0;
  numMinHFTowersM = 0;

  const auto& pfCandidates = iEvent.get(pfCandidateTag_);

  for (const auto& pfcand : pfCandidates) {

    if (pfcand.pdgId() != 1 && pfcand.pdgId() != 2) continue;

    const auto eta = pfcand.eta();
    const auto abseta = std::abs(eta);
    if (abseta > 6 || abseta < 3)
      continue;

    if (pfcand.energy() >= threshold_) {
      if (eta > 0) { numMinHFTowersP++; }
      else { numMinHFTowersM++; }
    }
    
  } // for (const auto& pfcand : *pfCandidates)
  if (numMinHFTowersP >= minnumtowers_ && numMinHFTowersM >= minnumtowers_)
    accepted = true;

  return accepted;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiHFFilterPF);
