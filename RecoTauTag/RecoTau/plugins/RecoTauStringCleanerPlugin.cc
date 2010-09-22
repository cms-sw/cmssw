#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h" 
#include "CommonTools/Utils/interface/StringObjectFunction.h" 

namespace reco {
  namespace tau {

    class RecoTauStringCleanerPlugin : public RecoTauCleanerPlugin 
    {
      public:
        explicit RecoTauStringCleanerPlugin(const edm::ParameterSet&);
        ~RecoTauStringCleanerPlugin() {}
        double operator()(const PFTauRef&) const;
      private:
        const StringCutObjectSelector<PFTau> selector_;
        const StringObjectFunction<PFTau> function_;
        double failResult_;
    };

    RecoTauStringCleanerPlugin::RecoTauStringCleanerPlugin(const edm::ParameterSet& pset):
      RecoTauCleanerPlugin(pset),
      selector_(pset.getParameter<std::string>("selection")),
      function_(pset.getParameter<std::string>("selectionPassFunction")),
      failResult_(pset.getParameter<double>("selectionFailValue")) {}

    double RecoTauStringCleanerPlugin::operator()(const PFTauRef& cand) const
    {
      if(selector_(*cand))
        return function_(*cand);
      else
        return failResult_;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, reco::tau::RecoTauStringCleanerPlugin, "RecoTauStringCleanerPlugin");

