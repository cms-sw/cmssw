/*
 * RecoTauPiZeroTrivialPlugin
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Given an input PFJet, produces collection of trivial 'un-merged' PiZero
 * RecoTauPiZeros.  Each PiZero is composed of only one photon from
 * the jet.
 *
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include <boost/foreach.hpp>

namespace reco { namespace tau {

class RecoTauPiZeroTrivialPlugin : public RecoTauPiZeroBuilderPlugin {
  public:
    explicit RecoTauPiZeroTrivialPlugin(const edm::ParameterSet& pset);
    ~RecoTauPiZeroTrivialPlugin() {}
    return_type operator()(const reco::PFJet& jet) const;
  private:
    RecoTauQualityCuts qcuts_;
};

RecoTauPiZeroTrivialPlugin::RecoTauPiZeroTrivialPlugin(
    const edm::ParameterSet& pset):RecoTauPiZeroBuilderPlugin(pset),
    qcuts_(pset.getParameterSet(
          "qualityCuts").getParameterSet("signalQualityCuts")) {}

RecoTauPiZeroBuilderPlugin::return_type RecoTauPiZeroTrivialPlugin::operator()(
    const reco::PFJet& jet) const {
  std::vector<PFCandidatePtr> pfGammaCands = qcuts_.filterCandRefs(pfGammas(jet));
  PiZeroVector output;
  output.reserve(pfGammaCands.size());

  BOOST_FOREACH(const PFCandidatePtr& gamma, pfGammaCands) {
    std::auto_ptr<RecoTauPiZero> piZero(new RecoTauPiZero(
            0, (*gamma).p4(), (*gamma).vertex(), 22, 1000, true,
            RecoTauPiZero::kTrivial));
    piZero->addDaughter(gamma);
    output.push_back(piZero);
  }
  return output.release();
}

}} // end reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroTrivialPlugin,
    "RecoTauPiZeroTrivialPlugin");
