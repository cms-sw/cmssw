#include "RecoHI/HiTracking/interface/HIPixelTrackFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace std;
using namespace edm;

/*****************************************************************************/
HIPixelTrackFilter::HIPixelTrackFilter(const SiPixelClusterShapeCache *cache, double ptMin, double ptMax, const edm::EventSetup& es,
                                       const reco::VertexCollection *vertices,
                                       double tipMax, double tipMaxTolerance,
                                       double lipMax, double lipMaxTolerance,
                                       double chi2max,
                                       bool useClusterShape):
  ClusterShapeTrackFilter(cache, ptMin, ptMax, es),
  theVertices(vertices),
  theTIPMax(tipMax),
  theNSigmaTipMaxTolerance(tipMaxTolerance),
  theLIPMax(lipMax),
  theNSigmaLipMaxTolerance(lipMaxTolerance),
  theChi2Max(chi2max),
  thePtMin(ptMin),
  useClusterShape(useClusterShape)
{
}

/*****************************************************************************/
HIPixelTrackFilter::~HIPixelTrackFilter()
{ }

/*****************************************************************************/
bool HIPixelTrackFilter::operator() (const reco::Track* track,const PixelTrackFilterBase::Hits & recHits) const
{

  if (!track) return false; 
  if (track->chi2() > theChi2Max || track->pt() < thePtMin) return false; 
  
  
  math::XYZPoint vtxPoint(0.0,0.0,0.0);
  double vzErr =0.0, vxErr=0.0, vyErr=0.0;
  
  if(theVertices->size()>0) {
    vtxPoint=theVertices->begin()->position();
    vzErr=theVertices->begin()->zError();
    vxErr=theVertices->begin()->xError();
    vyErr=theVertices->begin()->yError();
  } else {
    // THINK OF SOMETHING TO DO IF THERE IS NO VERTEX
  }
  
  double d0=0.0, dz=0.0, d0sigma=0.0, dzsigma=0.0;
  d0 = -1.*track->dxy(vtxPoint);
  dz = track->dz(vtxPoint);
  d0sigma = sqrt(track->d0Error()*track->d0Error()+vxErr*vyErr);
  dzsigma = sqrt(track->dzError()*track->dzError()+vzErr*vzErr);
  
  if (theTIPMax>0 && fabs(d0)>theTIPMax) return false;
  if (theNSigmaTipMaxTolerance>0 && (fabs(d0)/d0sigma)>theNSigmaTipMaxTolerance) return false;
  if (theLIPMax>0 && fabs(dz)>theLIPMax) return false;
  if (theNSigmaLipMaxTolerance>0 && (fabs(dz)/dzsigma)>theNSigmaLipMaxTolerance) return false;
  
  bool ok = true;
  if(useClusterShape) ok = ClusterShapeTrackFilter::operator() (track,recHits);
  
  return ok;
}
