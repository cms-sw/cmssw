/*
 * RecoTauDiscriminantInvariantWidth
 *
 * Compute a (hopefully) p_T independent quantity related to the
 * opening angle.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantFunctions.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <TFormula.h>

namespace reco::tau {

class RecoTauDiscriminantInvariantWidth : public RecoTauDiscriminantPlugin {
  public:
    explicit RecoTauDiscriminantInvariantWidth(
        const edm::ParameterSet& pset);
    std::vector<double> operator()(const reco::PFTauRef& tau) const override;
  private:
    typedef StringObjectFunction<PFTau> TauFunc;
    typedef boost::shared_ptr<TauFunc> TauFuncPtr;
    typedef std::pair<TauFuncPtr, TauFuncPtr> MeanAndWidthFuncs;

    std::map<reco::PFTau::hadronicDecayMode, MeanAndWidthFuncs> transforms_;
    MeanAndWidthFuncs defaultTransform_;
};

RecoTauDiscriminantInvariantWidth::RecoTauDiscriminantInvariantWidth(
    const edm::ParameterSet& pset):RecoTauDiscriminantPlugin(pset) {
  typedef std::vector<edm::ParameterSet> VPSet;
  // Add each of the transformations
  for(auto const& dm : pset.getParameter<VPSet>("decayModes")) {
    uint32_t nCharged = dm.getParameter<uint32_t>("nCharged");
    uint32_t nPiZeros = dm.getParameter<uint32_t>("nPiZeros");
    MeanAndWidthFuncs functions;
    functions.first.reset(new TauFunc(dm.getParameter<std::string>("mean")));
    functions.second.reset(new TauFunc(dm.getParameter<std::string>("rms")));
    transforms_[translateDecayMode(nCharged, nPiZeros)] = functions;
  }
  defaultTransform_.first.reset(
      new TauFunc(pset.getParameter<std::string>("defaultMean")));
  defaultTransform_.second.reset(
      new TauFunc(pset.getParameter<std::string>("defaultRMS")));
}

std::vector<double> RecoTauDiscriminantInvariantWidth::operator()(
    const reco::PFTauRef& tau) const {
  double weightedDeltaR = disc::OpeningDeltaR(*tau);

  std::map<reco::PFTau::hadronicDecayMode, MeanAndWidthFuncs>::const_iterator
    transform = transforms_.find(tau->decayMode());

  const TauFunc* meanFunc = defaultTransform_.first.get();
  const TauFunc* rmsFunc = defaultTransform_.second.get();

  if (transform != transforms_.end()) {
    meanFunc = transform->second.first.get();
    rmsFunc = transform->second.second.get();
  }

  double mean = (*meanFunc)(*tau);
  double rms = (*rmsFunc)(*tau);

  double result = (rms > 0) ? (weightedDeltaR - mean)/rms : -1.;

  return std::vector<double>(1, result);
}

} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory,
    reco::tau::RecoTauDiscriminantInvariantWidth,
    "RecoTauDiscriminantInvariantWidth");
