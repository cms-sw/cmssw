#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectRefSelector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

using namespace edm;
using namespace reco;
using namespace std;

namespace reco {
  namespace modules {

    class MCMatchCandRefSelector {
    public:
      explicit MCMatchCandRefSelector(const EDGetTokenT<GenParticleMatch>& srcToken) : srcToken_(srcToken) {}
      void newEvent(const Event& evt, const EventSetup&);
      bool operator()(const CandidateBaseRef&) const;

    private:
      EDGetTokenT<GenParticleMatch> srcToken_;
      const GenParticleMatch* match_ = nullptr;
    };

    void MCMatchCandRefSelector::newEvent(const Event& evt, const EventSetup&) {
      Handle<GenParticleMatch> match;
      evt.getByToken(srcToken_, match);
      match_ = match.product();
    }

    bool MCMatchCandRefSelector::operator()(const CandidateBaseRef& c) const {
      GenParticleRef m = (*match_)[c];
      return m.isNonnull();
    }

    template <>
    struct ParameterAdapter<MCMatchCandRefSelector> {
      static MCMatchCandRefSelector make(const ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return MCMatchCandRefSelector(iC.consumes<GenParticleMatch>(cfg.getParameter<InputTag>("match")));
      }
    };

  }  // namespace modules
}  // namespace reco

EVENTSETUP_STD_INIT(MCMatchCandRefSelector);

typedef SingleObjectRefSelector<Candidate, reco::modules::MCMatchCandRefSelector> MCMatchCandRefSelector;

DEFINE_FWK_MODULE(MCMatchCandRefSelector);
