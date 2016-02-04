#include "RecoHI/HiTracking/interface/HIPixelTrackFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

using namespace std;
using namespace edm;

/*****************************************************************************/
HIPixelTrackFilter::HIPixelTrackFilter (const edm::ParameterSet& ps, const edm::EventSetup& es) :
ClusterShapeTrackFilter(ps,es),
theTIPMax( ps.getParameter<double>("tipMax") ),
theNSigmaTipMaxTolerance( ps.getParameter<double>("nSigmaTipMaxTolerance")),
theLIPMax( ps.getParameter<double>("lipMax") ),
theNSigmaLipMaxTolerance( ps.getParameter<double>("nSigmaLipMaxTolerance")),
theChi2Max( ps.getParameter<double>("chi2") ),
thePtMin( ps.getParameter<double>("ptMin") ),
useClusterShape( ps.getParameter<bool>("useClusterShape") ),
theVertexCollection( ps.getParameter<edm::InputTag>("VertexCollection")),
theVertices(0)
{ 
}

/*****************************************************************************/
HIPixelTrackFilter::~HIPixelTrackFilter()
{ }

/*****************************************************************************/
bool HIPixelTrackFilter::operator() (const reco::Track* track,const PixelTrackFilter::Hits & recHits) const
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

/*****************************************************************************/
void HIPixelTrackFilter::update(edm::Event& ev)
{
  
  // Get reco vertices
  edm::Handle<reco::VertexCollection> vc;
  ev.getByLabel(theVertexCollection, vc);
  theVertices = vc.product();
  
  if(theVertices->size()>0) {
    LogInfo("HeavyIonVertexing") 
      << "[HIPixelTrackFilter] Pixel track selection based on best vertex"
      << "\n   vz = " << theVertices->begin()->z()  
      << "\n   vz sigma = " << theVertices->begin()->zError();
  } else {
    LogError("HeavyIonVertexing") // this can be made a warning when operator() is fixed
      << "No vertex found in collection '" << theVertexCollection << "'";
  }

  return;
  
}
