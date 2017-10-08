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

#include <TMath.h>

namespace reco { namespace tau {

class PFRecoTauMassPlugin : public RecoTauModifierPlugin
{
 public:

  explicit PFRecoTauMassPlugin(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
  ~PFRecoTauMassPlugin() override;
  void operator()(PFTau&) const override;
  void beginEvent() override;
  void endEvent() override;

 private:
  
  int verbosity_;
};

  PFRecoTauMassPlugin::PFRecoTauMassPlugin(const edm::ParameterSet& cfg, edm::ConsumesCollector &&iC)
    : RecoTauModifierPlugin(cfg, std::move(iC))
{
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

PFRecoTauMassPlugin::~PFRecoTauMassPlugin()
{}

void PFRecoTauMassPlugin::beginEvent()
{}

void PFRecoTauMassPlugin::operator()(PFTau& tau) const
{
  if ( verbosity_ ) {
    std::cout << "<PFRecoTauMassPlugin::operator()>:" << std::endl;
    std::cout << "tau: Pt = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi() << ", mass = " << tau.mass() << " (decayMode = " << tau.decayMode() << ")" << std::endl;
  }

  if ( tau.decayMode() == reco::PFTau::kOneProng0PiZero ) {
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

void PFRecoTauMassPlugin::endEvent()
{}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory, reco::tau::PFRecoTauMassPlugin, "PFRecoTauMassPlugin");
