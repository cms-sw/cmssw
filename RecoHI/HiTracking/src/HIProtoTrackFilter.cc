#include "RecoHI/HiTracking/interface/HIProtoTrackFilter.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

using namespace std;
using namespace edm;

/*****************************************************************************/
HIProtoTrackFilter::HIProtoTrackFilter(const reco::BeamSpot *beamSpot, double tipMax, double chi2Max, double ptMin):
  theTIPMax(tipMax),
  theChi2Max(chi2Max),
  thePtMin(ptMin),
  theBeamSpot(beamSpot)
{
}

/*****************************************************************************/
HIProtoTrackFilter::~HIProtoTrackFilter()
{ }

/*****************************************************************************/
bool HIProtoTrackFilter::operator() (const reco::Track* track,const PixelTrackFilterBase::Hits & recHits) const
{

  if (!track) return false; 

  if (track->chi2() > theChi2Max || track->pt() < thePtMin) return false;
  
  math::XYZPoint vtxPoint(0.0,0.0,0.0);
  
  if(theBeamSpot)
    vtxPoint = theBeamSpot->position();
  
  double d0=0.0;
  d0 = -1.*track->dxy(vtxPoint);
  
  if (theTIPMax>0 && fabs(d0)>theTIPMax) return false;
  
  return true;
}
