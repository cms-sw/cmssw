/*
 * PFRecoTauChargedHadronStringQuality
 *
 * Author: Christian Veelken, LLR
 *
 * RecoTau quality plugin that returns the value given by the supplied string
 * expression.  A string cut can additionally be applied.  ChargedHadrons
 * that fail the cut will be associated with a default value.
 *
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h" 
#include "CommonTools/Utils/interface/StringObjectFunction.h" 

namespace reco { namespace tau {

class PFRecoTauChargedHadronStringQuality : public PFRecoTauChargedHadronQualityPlugin
{
 public:
  explicit PFRecoTauChargedHadronStringQuality(const edm::ParameterSet&);
  ~PFRecoTauChargedHadronStringQuality() {}
  double operator()(const PFRecoTauChargedHadron&) const;
 private:
  const StringCutObjectSelector<PFRecoTauChargedHadron> selector_;
  const StringObjectFunction<PFRecoTauChargedHadron> function_;
  double failResult_;
};

PFRecoTauChargedHadronStringQuality::PFRecoTauChargedHadronStringQuality(const edm::ParameterSet& pset)
  : PFRecoTauChargedHadronQualityPlugin(pset),
    selector_(pset.getParameter<std::string>("selection")),
    function_(pset.getParameter<std::string>("selectionPassFunction")),
    failResult_(pset.getParameter<double>("selectionFailValue")) 
{}

double PFRecoTauChargedHadronStringQuality::operator()(const PFRecoTauChargedHadron& cand) const
{
  if ( selector_(cand) ) return function_(cand);
  else return failResult_;
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronQualityPluginFactory, reco::tau::PFRecoTauChargedHadronStringQuality, "PFRecoTauChargedHadronStringQuality");
