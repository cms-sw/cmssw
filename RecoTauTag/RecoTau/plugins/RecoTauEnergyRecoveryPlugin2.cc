/*
 * =============================================================================
 *       Filename:  RecoTauEnergyRecoveryPlugin2.cc
 *
 *    Description:  Set tau energy to sum of **all** PFCandidates
 *                 (regardless of PFCandidate type and whether 
 *                  PFCandidate passes or fails quality cuts)
 * 
 *           NOTE:  use for testing only,
 *                  this code has **not** yet been commissioned !!
 *
 *        Created:  02/02/2013 17:09:00
 *
 *         Authors:  Christian Veelken (LLR)
 *
 * =============================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>

namespace reco { namespace tau {

class RecoTauEnergyRecoveryPlugin2 : public RecoTauModifierPlugin
{
 public:

  explicit RecoTauEnergyRecoveryPlugin2(const edm::ParameterSet&);
  virtual ~RecoTauEnergyRecoveryPlugin2();
  void operator()(PFTau&) const;
  virtual void beginEvent();

 private:

  double dRcone_;
};

RecoTauEnergyRecoveryPlugin2::RecoTauEnergyRecoveryPlugin2(const edm::ParameterSet& cfg)
  : RecoTauModifierPlugin(cfg),
    dRcone_(cfg.getParameter<double>("dRcone"))
{}

RecoTauEnergyRecoveryPlugin2::~RecoTauEnergyRecoveryPlugin2()
{}

void RecoTauEnergyRecoveryPlugin2::beginEvent()
{}

void RecoTauEnergyRecoveryPlugin2::operator()(PFTau& tau) const
{
  reco::Candidate::LorentzVector tauAltP4(0.,0.,0.,0.);
  
  std::vector<reco::PFCandidatePtr> pfJetConstituents = tau.jetRef()->getPFConstituents();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator pfJetConstituent = pfJetConstituents.begin();
	pfJetConstituent != pfJetConstituents.end(); ++pfJetConstituent ) {
    double dR = deltaR((*pfJetConstituent)->p4(), tau.p4());
    if ( dR < dRcone_ ) tauAltP4 += (*pfJetConstituent)->p4();
  }

  tau.setalternatLorentzVect(tauAltP4);
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauEnergyRecoveryPlugin2,
    "RecoTauEnergyRecoveryPlugin2");
