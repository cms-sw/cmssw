#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <class T> T sqr( T t) {return t*t;}

#include <iostream>

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : theoPtMin(1/cfg.getParameter<double>("ptMin") ),
    theNSigmaInvPtTolerance( cfg.getParameter<double>("nSigmaInvPtTolerance")),
    theTIPMax( cfg.getParameter<double>("tipMax") ),
    theNSigmaTipMaxTolerance( cfg.getParameter<double>("nSigmaTipMaxTolerance")),
    theChi2Max( cfg.getParameter<double>("chi2") )
{ }

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics( const edm::ParameterSet& cfg)
  : theoPtMin(1/cfg.getParameter<double>("ptMin") ),
    theNSigmaInvPtTolerance( cfg.getParameter<double>("nSigmaInvPtTolerance")),
    theTIPMax( cfg.getParameter<double>("tipMax") ),
    theNSigmaTipMaxTolerance( cfg.getParameter<double>("nSigmaTipMaxTolerance")),
    theChi2Max( cfg.getParameter<double>("chi2") )
{ }

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics(double ptmin, double tipmax, double chi2max)
  : theoPtMin(1/ptmin), theNSigmaInvPtTolerance(0.),
    theTIPMax(tipmax), theNSigmaTipMaxTolerance(0.),
    theChi2Max(chi2max)
{ } 

PixelTrackFilterByKinematics::~PixelTrackFilterByKinematics()
{ }

void PixelTrackFilterByKinematics::update(const edm::Event&, const edm::EventSetup&) {}

bool PixelTrackFilterByKinematics::operator()(const reco::Track* track, const PixelTrackFilter::Hits & hits) const
{ return (*this)(track); }

bool PixelTrackFilterByKinematics::operator()(const reco::Track* ptrack) const
{
  if (!ptrack) return false;
  auto const & track = *ptrack;
  if (track.chi2() > theChi2Max) return false;
  if ( (std::abs(track.d0())-theTIPMax) > theNSigmaTipMaxTolerance*track.d0Error()) return false;
  
  float pt_v = float(track.pt());
  float opt_v = 1.f/pt_v;
  float	pz_v = track.pz();
  float	p_v = float(track.p());
  float op_v = 1.f/p_v;
  float cosTheta = pz_v*op_v;
  float osinTheta = p_v*opt_v;
  float errLambda2 =  track.covariance(reco::TrackBase::i_lambda, reco::TrackBase::i_lambda);
  float errInvP2 =    track.covariance(reco::TrackBase::i_qoverp, reco::TrackBase::i_qoverp);
  float covIPtTheta = track.covariance(reco::TrackBase::i_qoverp, reco::TrackBase::i_lambda);
  float errInvPt2 = (   errInvP2
                      + sqr(cosTheta*opt_v)*errLambda2
                      + 2.f*(cosTheta*opt_v)*covIPtTheta
                     )*sqr(osinTheta);

  return (opt_v - theoPtMin) < theNSigmaInvPtTolerance*std::sqrt(errInvPt2);
}
