#include <vector>

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"

using namespace std;
using namespace fastjet;

class LHEJetFilter : public edm::global::EDFilter<> {
public:
  explicit LHEJetFilter(const edm::ParameterSet&);
  ~LHEJetFilter() override {}

private:
  bool filter(edm::StreamID strid, edm::Event& evt, const edm::EventSetup& params) const override;

  edm::EDGetTokenT<LHEEventProduct> tokenLHEEvent_;
  double jetPtMin_;
  JetDefinition jetdef_;
};

LHEJetFilter::LHEJetFilter(const edm::ParameterSet& params)
    : tokenLHEEvent_(consumes<LHEEventProduct>(params.getParameter<edm::InputTag>("src"))),
      jetPtMin_(params.getParameter<double>("jetPtMin")),
      jetdef_(antikt_algorithm, params.getParameter<double>("jetR")) {}

bool LHEJetFilter::filter(edm::StreamID strid, edm::Event& evt, const edm::EventSetup& params) const {
  edm::Handle<LHEEventProduct> lheinfo;
  evt.getByToken(tokenLHEEvent_, lheinfo);

  if (!lheinfo.isValid()) {
    return true;
  }

  vector<PseudoJet> jetconsts;
  jetconsts.reserve(10);
  const lhef::HEPEUP& hepeup = lheinfo->hepeup();
  for (size_t p = 0; p < hepeup.IDUP.size(); ++p) {
    if (hepeup.ISTUP[p] == 1) {
      const lhef::HEPEUP::FiveVector& mom = hepeup.PUP[p];
      jetconsts.emplace_back(mom[0], mom[1], mom[2], mom[3]);
    }
  }

  ClusterSequence cs(jetconsts, jetdef_);
  vector<PseudoJet> jets = cs.inclusive_jets(jetPtMin_);

  return !jets.empty();
}

// Define module as a plug-in
DEFINE_FWK_MODULE(LHEJetFilter);
