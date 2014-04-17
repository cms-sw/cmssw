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

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

namespace reco { namespace tau {

class RecoTauStringCleanerPlugin : public RecoTauCleanerPlugin 
{
 public:
  explicit RecoTauStringCleanerPlugin(const edm::ParameterSet&);
  ~RecoTauStringCleanerPlugin() {}
  double operator()(const PFTauRef& tau) const;

 private:
  const StringCutObjectSelector<PFTau> selector_;
  const StringObjectFunction<PFTau> function_;
  double failResult_;
};

RecoTauStringCleanerPlugin::RecoTauStringCleanerPlugin(const edm::ParameterSet& pset)
  : RecoTauCleanerPlugin(pset),
    selector_(pset.getParameter<std::string>("selection")),
    function_(pset.getParameter<std::string>("selectionPassFunction")),
    failResult_(pset.getParameter<double>("selectionFailValue")) 
{}

double RecoTauStringCleanerPlugin::operator()(const PFTauRef& cand) const 
{
  if ( selector_(*cand) ) return function_(*cand);
  else return failResult_;
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, reco::tau::RecoTauStringCleanerPlugin, "RecoTauStringCleanerPlugin");
