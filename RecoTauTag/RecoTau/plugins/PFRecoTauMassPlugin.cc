/*
 * =============================================================================
 *       Filename:  PFRecoTauMassPlugin.cc
 *
 *    Description:  Set mass of taus reconstructed in 1prong0pi0 decay mode
 *                  to charged pion mass
 *
 *        Created:  27/10/2015 10:30:00
 *
 *         Authors:  Christian Veelken (Tallinn)
 *
 * =============================================================================
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"

#include <TMath.h>

namespace reco { namespace tau {

template<typename TauType>
class PFRecoTauGenericMassPlugin : public RecoTauModifierPlugin<TauType>
{
 public:

  explicit PFRecoTauGenericMassPlugin(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
  ~PFRecoTauGenericMassPlugin() override;
  void operator()(TauType&) const override;
  void beginEvent() override;
  void endEvent() override;

 private:
  
  int verbosity_;
};

template<typename TauType>
PFRecoTauGenericMassPlugin<TauType>::PFRecoTauGenericMassPlugin(const edm::ParameterSet& cfg, edm::ConsumesCollector &&iC)
    : RecoTauModifierPlugin<TauType>(cfg, std::move(iC))
{
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

template<typename TauType>
PFRecoTauGenericMassPlugin<TauType>::~PFRecoTauGenericMassPlugin()
{}

template<typename TauType>
void PFRecoTauGenericMassPlugin<TauType>::beginEvent()
{}

template<typename TauType>
void PFRecoTauGenericMassPlugin<TauType>::operator()(TauType& tau) const
{
  if ( verbosity_ ) {
    std::cout << "<PFRecoTauGenericMassPlugin::operator()>:" << std::endl;
    std::cout << "tau: Pt = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi() << ", mass = " << tau.mass() << " (decayMode = " << tau.decayMode() << ")" << std::endl;
  }

  if ( tau.decayMode() == TauType::kOneProng0PiZero ) {
    double tauEn = tau.energy();
    const double chargedPionMass = 0.13957; // GeV
    if ( tauEn < chargedPionMass ) tauEn = chargedPionMass;
    double tauP_modified = TMath::Sqrt(tauEn*tauEn - chargedPionMass*chargedPionMass);
    double tauPx_modified = TMath::Cos(tau.phi())*TMath::Sin(tau.theta())*tauP_modified;
    double tauPy_modified = TMath::Sin(tau.phi())*TMath::Sin(tau.theta())*tauP_modified;
    double tauPz_modified = TMath::Cos(tau.theta())*tauP_modified;
    reco::Candidate::LorentzVector tauP4_modified(tauPx_modified, tauPy_modified, tauPz_modified, tauEn);
    if ( verbosity_ ) {
      std::cout << "--> setting tauP4: Pt = " << tauP4_modified.pt() << ", eta = " << tauP4_modified.eta() << ", phi = " << tauP4_modified.phi() << ", mass = " << tauP4_modified.mass() << std::endl;
    }
    tau.setP4(tauP4_modified);
  }
}

template<typename TauType>
void PFRecoTauGenericMassPlugin<TauType>::endEvent()
{}

template class PFRecoTauGenericMassPlugin<reco::PFTau>;
typedef PFRecoTauGenericMassPlugin<reco::PFTau> PFRecoTauMassPlugin;

template class PFRecoTauGenericMassPlugin<reco::PFBaseTau>;
typedef PFRecoTauGenericMassPlugin<reco::PFBaseTau> PFRecoBaseTauMassPlugin;

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory, reco::tau::PFRecoTauMassPlugin, "PFRecoTauMassPlugin");
DEFINE_EDM_PLUGIN(RecoBaseTauModifierPluginFactory, reco::tau::PFRecoBaseTauMassPlugin, "PFRecoBaseTauMassPlugin");
