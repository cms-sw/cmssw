#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <iostream>

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics( const edm::ParameterSet& cfg)
  : thePtMin( cfg.getParameter<double>("ptMin") ),
    theTIPMax( cfg.getParameter<double>("tipMax") ),
    theChi2Max( cfg.getParameter<double>("chi2") )
{ }

PixelTrackFilterByKinematics::PixelTrackFilterByKinematics(float ptmin, float tipmax, float chi2max)
  : thePtMin(ptmin), theTIPMax(tipmax),theChi2Max(chi2max)
{ } 

PixelTrackFilterByKinematics::~PixelTrackFilterByKinematics()
{ }

bool PixelTrackFilterByKinematics::operator()(const reco::Track* track) const
{
  return ( (track) &&   
              (track->pt() > thePtMin)
           && (track->d0() < theTIPMax)
           && (track->chi2() < theChi2Max) );
}
