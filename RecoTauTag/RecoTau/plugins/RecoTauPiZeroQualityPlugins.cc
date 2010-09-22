#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h" 
#include "CommonTools/Utils/interface/StringObjectFunction.h" 

namespace reco {
  namespace tau {

    class RecoTauPiZeroStringQuality : public RecoTauPiZeroQualityPlugin
    {
      public:
        explicit RecoTauPiZeroStringQuality(const edm::ParameterSet&);
        ~RecoTauPiZeroStringQuality() {}
        double operator()(const RecoTauPiZero&) const;
      private:
        const StringCutObjectSelector<RecoTauPiZero> selector_;
        const StringObjectFunction<RecoTauPiZero> function_;
        double failResult_;
    };

    RecoTauPiZeroStringQuality::RecoTauPiZeroStringQuality(const edm::ParameterSet& pset):
      RecoTauPiZeroQualityPlugin(pset),
      selector_(pset.getParameter<std::string>("selection")),
      function_(pset.getParameter<std::string>("selectionPassFunction")),
      failResult_(pset.getParameter<double>("selectionFailValue")) {}

    double RecoTauPiZeroStringQuality::operator()(const RecoTauPiZero& cand) const
    {
      //std::cout << "PiZero: " << cand.mass() << std::endl;
      if(selector_(cand))
      {
        //std::cout << "Passed selection" << std::endl;
        return function_(cand);
      }
      else
      {
        //std::cout << "failed selection" << std::endl;
        return failResult_;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroQualityPluginFactory, reco::tau::RecoTauPiZeroStringQuality, "RecoTauPiZeroStringQuality");

