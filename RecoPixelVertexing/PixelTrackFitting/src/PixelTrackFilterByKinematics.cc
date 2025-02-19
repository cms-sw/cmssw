#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <class T> T sqr( T t) {return t*t;}

#include <iostream>

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics( const edm::ParameterSet& cfg, const edm::EventSetup &es)
  : thePtMin( cfg.getParameter<double>("ptMin") ),
    theNSigmaInvPtTolerance( cfg.getParameter<double>("nSigmaInvPtTolerance")),
    theTIPMax( cfg.getParameter<double>("tipMax") ),
    theNSigmaTipMaxTolerance( cfg.getParameter<double>("nSigmaTipMaxTolerance")),
    theChi2Max( cfg.getParameter<double>("chi2") )
{ }

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics( const edm::ParameterSet& cfg)
  : thePtMin( cfg.getParameter<double>("ptMin") ),
    theNSigmaInvPtTolerance( cfg.getParameter<double>("nSigmaInvPtTolerance")),
    theTIPMax( cfg.getParameter<double>("tipMax") ),
    theNSigmaTipMaxTolerance( cfg.getParameter<double>("nSigmaTipMaxTolerance")),
    theChi2Max( cfg.getParameter<double>("chi2") )
{ }

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics(double ptmin, double tipmax, double chi2max)
  : thePtMin(ptmin), theNSigmaInvPtTolerance(0.),
    theTIPMax(tipmax), theNSigmaTipMaxTolerance(0.),
    theChi2Max(chi2max)
{ } 

PixelTrackFilterByKinematics::~PixelTrackFilterByKinematics()
{ }

bool PixelTrackFilterByKinematics::operator()(const reco::Track* track, const PixelTrackFilter::Hits & hits) const
{ return (*this)(track); }

bool PixelTrackFilterByKinematics::operator()(const reco::Track* track) const
{
  if (!track) return false;
  if (track->chi2() > theChi2Max) return false;
  if ( (fabs(track->d0())-theTIPMax)/track->d0Error() > theNSigmaTipMaxTolerance) return false;
  
  float theta = track->theta();
  float cosTheta = cos(theta);
  float sinTheta = sin(theta);
  float errLambda2 = sqr( track->lambdaError() );

  float pt_v = track->pt();
  float errInvP2 = sqr(track->qoverpError());
  float covIPtTheta = track->covariance(reco::TrackBase::i_qoverp, reco::TrackBase::i_lambda);
  float errInvPt2 = (   errInvP2
                      + sqr(cosTheta/pt_v)*errLambda2
                      + 2*(cosTheta/pt_v)*covIPtTheta
                     ) / sqr(sinTheta);
  if ( (1/pt_v - 1/thePtMin)/sqrt(errInvPt2) > theNSigmaInvPtTolerance ) return false;

  return true;
}
