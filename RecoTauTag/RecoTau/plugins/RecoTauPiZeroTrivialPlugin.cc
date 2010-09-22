#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

/* 
 * RecoTauPiZeroTrivialPlugin
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Given an input PFJet, produces collection of trivial 'un-merged' PiZero
 * RecoTauPiZeros.  Each PiZero is composed of only one photon from 
 * the jet.
 *
 * $Id $
 *
 */

namespace reco {
  namespace tau {

    class RecoTauPiZeroTrivialPlugin : public RecoTauPiZeroBuilderPlugin
    {
      public:
        explicit RecoTauPiZeroTrivialPlugin(const edm::ParameterSet& pset);
        ~RecoTauPiZeroTrivialPlugin() {}
        std::vector<RecoTauPiZero> operator()(const reco::PFJet& jet) const;
    };

    RecoTauPiZeroTrivialPlugin::RecoTauPiZeroTrivialPlugin(const edm::ParameterSet& pset):RecoTauPiZeroBuilderPlugin(pset){}

    std::vector<RecoTauPiZero> RecoTauPiZeroTrivialPlugin::operator()(const reco::PFJet& jet) const
    {
      typedef std::vector<reco::PFCandidatePtr> PFCandPtrs; 
      typedef PFCandPtrs::iterator PFCandIter;
      std::vector<RecoTauPiZero> output;

      PFCandPtrs pfGammaCands = tau::pfGammas(jet);

      for(PFCandIter gamma = pfGammaCands.begin(); gamma != pfGammaCands.end(); ++gamma)
      {
        RecoTauPiZero piZero(0, (**gamma).p4(), (**gamma).vertex(), 22, 1000, true, name());
        piZero.addDaughter(*gamma);
        output.push_back(piZero);
      }
      return output;
    }

  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory, reco::tau::RecoTauPiZeroTrivialPlugin, "RecoTauPiZeroTrivialPlugin");
