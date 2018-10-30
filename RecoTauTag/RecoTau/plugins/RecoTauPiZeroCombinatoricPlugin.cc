/*
 * RecoTauPiZeroCombinatoricPlugin
 *
 * Author: Evan K. Friis, UC Davis
 *
 * Build PiZero candidates out of all possible sets of <choose> gammas that are
 * contained in the input PFJet.  Optionally, the pi zero candidates are
 * filtered by a min and max selection on their invariant mass.
 *
 */

#include <algorithm>

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"

#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

namespace reco { namespace tau {

class RecoTauPiZeroCombinatoricPlugin : public RecoTauPiZeroBuilderPlugin {
  public:
  explicit RecoTauPiZeroCombinatoricPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    ~RecoTauPiZeroCombinatoricPlugin() override {}
    // Return type is unique_ptr<PiZeroVector>
    return_type operator()(const reco::PFJet& jet) const override;

  private:
    RecoTauQualityCuts qcuts_;
    double minMass_;
    double maxMass_;
    unsigned int maxInputGammas_;
    unsigned int choose_;
    AddFourMomenta p4Builder_;
};

RecoTauPiZeroCombinatoricPlugin::RecoTauPiZeroCombinatoricPlugin(
    const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauPiZeroBuilderPlugin(pset,std::move(iC)),
    qcuts_(pset.getParameterSet(
          "qualityCuts").getParameterSet("signalQualityCuts")) {
  minMass_ = pset.getParameter<double>("minMass");
  maxMass_ = pset.getParameter<double>("maxMass");
  maxInputGammas_ = pset.getParameter<unsigned int>("maxInputGammas");
  choose_ = pset.getParameter<unsigned int>("choose");
}

RecoTauPiZeroCombinatoricPlugin::return_type
RecoTauPiZeroCombinatoricPlugin::operator()(
    const reco::PFJet& jet) const {
  // Get list of gamma candidates
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  typedef PFCandPtrs::const_iterator PFCandIter;
  PiZeroVector output;

  PFCandPtrs pfGammaCands = qcuts_.filterCandRefs(pfGammas(jet));
  // Check if we have anything to do...
  if (pfGammaCands.size() < choose_)
    return output.release();

  // Define the valid range of gammas to use
  PFCandIter start_iter = pfGammaCands.begin();
  PFCandIter end_iter = pfGammaCands.end();

  // Only take the desired number of piZeros
  end_iter = takeNElements(start_iter, end_iter, maxInputGammas_);

  // Build the combinatoric generator
  typedef CombinatoricGenerator<PFCandPtrs> ComboGenerator;
  ComboGenerator generator(start_iter, end_iter, choose_);

  // Find all possible combinations
  for (ComboGenerator::iterator combo = generator.begin();
      combo != generator.end(); ++combo) {
    const Candidate::LorentzVector totalP4;
    std::unique_ptr<RecoTauPiZero> piZero(
        new RecoTauPiZero(0, totalP4, Candidate::Point(0, 0, 0),
                          111, 10001, true, RecoTauPiZero::kCombinatoric));
    // Add our daughters from this combination
    for (ComboGenerator::combo_iterator candidate = combo->combo_begin();
        candidate != combo->combo_end();  ++candidate) {
      piZero->addDaughter(*candidate);
    }
    p4Builder_.set(*piZero);

    if (piZero->daughterPtr(0).isNonnull())
      piZero->setVertex(piZero->daughterPtr(0)->vertex());

    if ((maxMass_ < 0 || piZero->mass() < maxMass_) &&
        piZero->mass() > minMass_)
      output.push_back(piZero.get());
  }
  return output.release();
}

}}  // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroCombinatoricPlugin,
    "RecoTauPiZeroCombinatoricPlugin");
