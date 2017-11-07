/*
 * =====================================================================================
 *       Filename:  RecoTauStringCleanerPlugin.cc
 *
 *    Description:  Rank taus by a string function.  There are three arguments,
 *                  a binary [selection] string, an expression to return if that
 *                  selection passes, and value to return if the selection
 *                  fails.
 *        Created:  11/11/2010 11:09:52
 *
 *         Author:  Evan K. Friis (UC Davis), evan.klose.friis@cern.ch
 * =====================================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/TauReco/interface/PFBaseTauFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

namespace reco { namespace tau {

template<class TauType>
class RecoTauGenericStringCleanerPlugin : public RecoTauCleanerPlugin<TauType>
{
 public:
  explicit RecoTauGenericStringCleanerPlugin(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
  ~RecoTauGenericStringCleanerPlugin() override {}
  double operator()(const edm::Ref<std::vector<TauType> >& tau) const override;

 private:
  const StringCutObjectSelector<TauType> selector_;
  const StringObjectFunction<TauType> function_;
  double failResult_;
};

template<class TauType>
RecoTauGenericStringCleanerPlugin<TauType>::RecoTauGenericStringCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC)
  : RecoTauCleanerPlugin<TauType>(pset,std::move(iC)),
    selector_(pset.getParameter<std::string>("selection")),
    function_(pset.getParameter<std::string>("selectionPassFunction")),
    failResult_(pset.getParameter<double>("selectionFailValue")) 
{}

template<class TauType>
double RecoTauGenericStringCleanerPlugin<TauType>::operator()(const edm::Ref<std::vector<TauType> >& cand) const 
{
  if ( selector_(*cand) ) return function_(*cand);
  else return failResult_;
}

template class RecoTauGenericStringCleanerPlugin<reco::PFTau>;
typedef RecoTauGenericStringCleanerPlugin<reco::PFTau> RecoTauStringCleanerPlugin;
template class RecoTauGenericStringCleanerPlugin<reco::PFBaseTau>;
typedef RecoTauGenericStringCleanerPlugin<reco::PFBaseTau> RecoBaseTauStringCleanerPlugin;


}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, reco::tau::RecoTauStringCleanerPlugin, "RecoTauStringCleanerPlugin");
DEFINE_EDM_PLUGIN(RecoBaseTauCleanerPluginFactory, reco::tau::RecoBaseTauStringCleanerPlugin, "RecoBaseTauStringCleanerPlugin");
