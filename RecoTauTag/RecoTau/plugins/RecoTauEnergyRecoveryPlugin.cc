/*
 * =============================================================================
 *       Filename:  RecoTauEnergyRecoveryPlugin.cc
 *
 *    Description:  Recovery energy of visible decay products
 *                  lost due to photon conversions/nuclear interactions
 *                  in tracker material.
 *        Created:  09/02/2011 10:28:00
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

#include <algorithm>

namespace reco { namespace tau {

class RecoTauEnergyRecoveryPlugin : public RecoTauModifierPlugin
{
 public:

  explicit RecoTauEnergyRecoveryPlugin(const edm::ParameterSet&);
  virtual ~RecoTauEnergyRecoveryPlugin();
  void operator()(PFTau&) const;
  virtual void beginEvent();

 private:

  edm::InputTag srcVertices_;
  const reco::Vertex* theEventVertex_;

  RecoTauQualityCuts* qcuts_;

  unsigned corrLevel_;

  double lev1PhiWindow_;
  double lev1EtaWindow_;
};

RecoTauEnergyRecoveryPlugin::RecoTauEnergyRecoveryPlugin(const edm::ParameterSet& cfg)
  : RecoTauModifierPlugin(cfg),
    theEventVertex_(0),
    corrLevel_(cfg.getParameter<unsigned>("corrLevel")),
    lev1PhiWindow_(cfg.getParameter<double>("lev1PhiWindow")),
    lev1EtaWindow_(cfg.getParameter<double>("lev1EtaWindow"))
{
  edm::ParameterSet cfgQualityCuts = cfg.getParameter<edm::ParameterSet>("qualityCuts");
  srcVertices_ = cfgQualityCuts.getParameter<edm::InputTag>("primaryVertexSrc");
  qcuts_ = new RecoTauQualityCuts(cfgQualityCuts.getParameter<edm::ParameterSet>("signalQualityCuts"));
}

RecoTauEnergyRecoveryPlugin::~RecoTauEnergyRecoveryPlugin()
{
  delete qcuts_;
}

void RecoTauEnergyRecoveryPlugin::beginEvent()
{
  edm::Handle<reco::VertexCollection> vertices;
  evt()->getByLabel(srcVertices_, vertices);

  if ( vertices->size() >= 1 ) {
    qcuts_->setPV(reco::VertexRef(vertices, 0));
    theEventVertex_ = &vertices->at(0);
  } else {
    theEventVertex_ = 0;
  }
}

const reco::TrackBaseRef getTrack(const reco::PFCandidate& cand)
{
  if      ( cand.trackRef().isNonnull()    ) return reco::TrackBaseRef(cand.trackRef());
  else if ( cand.gsfTrackRef().isNonnull() ) return reco::TrackBaseRef(cand.gsfTrackRef());
  else return reco::TrackBaseRef();
}

bool isTauSignalPFCandidate(const reco::PFTau& tau, const reco::PFCandidatePtr& pfJetConstituent)
{
  bool retVal = false;

  const reco::PFCandidateRefVector& signalPFCandidates = tau.signalPFCands();
  for ( reco::PFCandidateRefVector::const_iterator signalPFCandidate = signalPFCandidates.begin();
	signalPFCandidate != signalPFCandidates.end(); ++signalPFCandidate ) {
    if ( pfJetConstituent.key() == signalPFCandidate->key() ) retVal = true;
  }

  return retVal;
}

double square(double x)
{
  return x*x;
}

void RecoTauEnergyRecoveryPlugin::operator()(PFTau& tau) const
{
  double tauEnergyCorr = 0.;

  if ( corrLevel_ & 1 && theEventVertex_ ) {

    bool needsCorrLevel1 = false;

    std::vector<reco::PFCandidatePtr> pfJetConstituents = tau.jetRef()->getPFConstituents();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator pfJetConstituent = pfJetConstituents.begin();
	  pfJetConstituent != pfJetConstituents.end(); ++pfJetConstituent ) {
      const reco::TrackBaseRef track = getTrack(**pfJetConstituent);
      if ( track.isNonnull() ) {
	double dIP = fabs(track->dxy(theEventVertex_->position()));
	double dZ = fabs(track->dz(theEventVertex_->position()));
      if ( track->pt() > 2.0 && fabs(tau.eta() - (*pfJetConstituent)->eta()) < lev1EtaWindow_ &&
	   !isTauSignalPFCandidate(tau, *pfJetConstituent) && (dZ < 0.2 || dIP > 0.10) ) needsCorrLevel1 = true;
      }
    }

    if ( needsCorrLevel1 ) {
      std::vector<reco::PFCandidatePtr> pfJetConstituents = tau.jetRef()->getPFConstituents();
      for ( std::vector<reco::PFCandidatePtr>::const_iterator pfJetConstituent = pfJetConstituents.begin();
	    pfJetConstituent != pfJetConstituents.end(); ++pfJetConstituent ) {
	if ( fabs(tau.eta() - (*pfJetConstituent)->eta()) < lev1EtaWindow_ &&
	     fabs(tau.phi() - (*pfJetConstituent)->phi()) < lev1PhiWindow_ ) {
	  if ( (*pfJetConstituent)->particleId() == reco::PFCandidate::h0 ) {
	    tauEnergyCorr += (*pfJetConstituent)->energy();
	  } else {
	    if ( !isTauSignalPFCandidate(tau, *pfJetConstituent) ) {
	      double caloEn = (*pfJetConstituent)->ecalEnergy() + (*pfJetConstituent)->hcalEnergy();
	      tauEnergyCorr += caloEn;
	    }
	  }
	}
      }
    }
  }

  if ( corrLevel_ & 2 && theEventVertex_ ) {

    double leadTrackMom = 0.;
    double leadTrackMomErr = 0.;
    double jetCaloEn = 0.;

    std::vector<reco::PFCandidatePtr> pfJetConstituents = tau.jetRef()->getPFConstituents();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator pfJetConstituent = pfJetConstituents.begin();
	  pfJetConstituent != pfJetConstituents.end(); ++pfJetConstituent ) {
      const reco::TrackBaseRef track = getTrack(**pfJetConstituent);
      if ( track.isNonnull() ) {
	double trackPt = track->pt();
	double trackPtErr = track->ptError();
	if ( qcuts_->filter(**pfJetConstituent) &&
	     trackPtErr < (0.20*trackPt) && track->normalizedChi2() < 5.0 && track->hitPattern().numberOfValidPixelHits() >= 1 &&
	     (trackPt - 3.*trackPtErr) > (*pfJetConstituent)->pt() && trackPt < (3.*tau.jetRef()->pt()) ) {
	  if ( track->p() > leadTrackMom ) {
	    leadTrackMom = track->p();
	    leadTrackMomErr = leadTrackMom*(trackPtErr/trackPt);
	  }
	}
      }

      double caloEn = (*pfJetConstituent)->ecalEnergy() + (*pfJetConstituent)->hcalEnergy();
      jetCaloEn += caloEn;
    }

    if ( leadTrackMom > tau.p() ) {
      const double chargedPionMass = 0.13957; // GeV
      double leadTrackEn = sqrt(square(leadTrackMom) + square(chargedPionMass));
      double jetCaloEnErr = 1.00*sqrt(std::max(jetCaloEn, leadTrackEn));
      double combEn = ((1./square(jetCaloEnErr))*jetCaloEn + (1./square(leadTrackMomErr))*leadTrackEn)/
                      ((1./square(jetCaloEnErr)) + (1./square(leadTrackMomErr)));
      tauEnergyCorr = std::max(tauEnergyCorr, combEn - tau.energy());
    }
  }

  if ( tau.energy() > 0. ) {
    double tauEnergy_corrected = tau.energy() + tauEnergyCorr;
    double scaleFactor = tauEnergy_corrected/tau.energy();
    double tauPx_corrected = scaleFactor*tau.px();
    double tauPy_corrected = scaleFactor*tau.py();
    double tauPz_corrected = scaleFactor*tau.pz();
    tau.setalternatLorentzVect(reco::Candidate::LorentzVector(tauPx_corrected, tauPy_corrected, tauPz_corrected, tauEnergy_corrected));
  }
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauEnergyRecoveryPlugin,
    "RecoTauEnergyRecoveryPlugin");
