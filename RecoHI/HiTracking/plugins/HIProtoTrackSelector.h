#ifndef HIProtoTrackSelection_h
#define HIProtoTrackSelection_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iostream>

/**
 Selector to select prototracks that pass certain kinematic cuts based on fast vertex
**/

class HIProtoTrackSelector
{
  
 public:
  // input collection type
  typedef reco::TrackCollection collection;
  
  // output collection type
  typedef std::vector<const reco::Track *> container;
  
  // iterator over result collection type.
  typedef container::const_iterator const_iterator;
  
  // constructor from parameter set configurability
  HIProtoTrackSelector(const edm::ParameterSet & iConfig) : 
    vertexCollection_(iConfig.getParameter<edm::InputTag>("VertexCollection")),
    beamSpotLabel_(iConfig.getParameter<edm::InputTag>("beamSpotLabel")),
    ptMin_(iConfig.getParameter<double>("ptMin")), 	
    nSigmaZ_(iConfig.getParameter<double>("nSigmaZ")),
    minZCut_(iConfig.getParameter<double>("minZCut")),
    maxD0Significance_(iConfig.getParameter<double>("maxD0Significance"))
    {};
  
  // select object from a collection and possibly event content
  void select( edm::Handle<reco::TrackCollection>& TCH, const edm::Event & iEvent, const edm::EventSetup & iSetup)
    {
      selected_.clear();
      
      const collection & c = *(TCH.product());
      
      // Get fast reco vertex 
      edm::Handle<reco::VertexCollection> vc;
      iEvent.getByLabel(vertexCollection_, vc);
      const reco::VertexCollection * vertices = vc.product();
      
      math::XYZPoint vtxPoint(0.0,0.0,0.0);
      double vzErr =0.0;
      
      if(vertices->size()>0) {
	vtxPoint=vertices->begin()->position();
	vzErr=vertices->begin()->zError();
	edm::LogInfo("HeavyIonVertexing") << "Select prototracks compatible with median vertex"
				     << "\n   vz = " << vtxPoint.Z()  
				     << "\n   " << nSigmaZ_ << " vz sigmas = " << vzErr*nSigmaZ_
				     << "\n   cut at = " << std::max(vzErr*nSigmaZ_,minZCut_);
      } 
      // Supress this warning, since events w/ no vertex are expected 
      //else {
	//edm::LogError("HeavyIonVertexing") << "No vertex found in collection '" << vertexCollection_ << "'";
      //}
      
      // Get beamspot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotHandle;
      iEvent.getByLabel(beamSpotLabel_, beamSpotHandle);
      
      math::XYZPoint bsPoint(0.0,0.0,0.0);
      double bsWidth = 0.0;
      
      if ( beamSpotHandle.isValid() ) {
	beamSpot = *beamSpotHandle;
	bsPoint = beamSpot.position();
	bsWidth = sqrt(beamSpot.BeamWidthX()*beamSpot.BeamWidthY());
	edm::LogInfo("HeavyIonVertexing") << "Select prototracks compatible with beamspot"
				     << "\n   (x,y,z) = (" << bsPoint.X() << "," << bsPoint.Y() << "," << bsPoint.Z() << ")"  
				     << "\n   width = " << bsWidth
				     << "\n   cut at d0/d0sigma = " << maxD0Significance_;
      } else {
	edm::LogError("HeavyIonVertexing") << "No beam spot available from '" << beamSpotLabel_ << "\n";
      }
      
      
      // Do selection
      int nSelected=0;
      int nRejected=0;
      double d0=0.0; 
      double d0sigma=0.0;
      for (reco::TrackCollection::const_iterator trk = c.begin(); trk != c.end(); ++ trk)
	{
	  
	  d0 = -1.*trk->dxy(bsPoint);
	  d0sigma = sqrt(trk->d0Error()*trk->d0Error() + bsWidth*bsWidth);
	  if ( trk->pt() > ptMin_ // keep only tracks above ptMin
	       && fabs(d0/d0sigma) < maxD0Significance_ // keep only tracks with D0 significance less than cut
	       && fabs(trk->dz(vtxPoint)) <  std::max(vzErr*nSigmaZ_,minZCut_) // within minZCut, nSigmaZ of fast vertex
	       ) 
	  {
	    nSelected++;
	    selected_.push_back( & * trk );
	  } 
	  else 
	    nRejected++;
	}
      
      edm::LogInfo("HeavyIonVertexing") << "selected " << nSelected << " prototracks out of " << nRejected+nSelected << "\n";
      
    }
  
  // iterators over selected objects: collection begin
  const_iterator begin() const { return selected_.begin(); }
  
  // iterators over selected objects: collection end
  const_iterator end() const { return selected_.end(); }
  
  // true if no object has been selected
  size_t size() const { return selected_.size(); }
  
  
 private:
  container selected_;		
  edm::InputTag vertexCollection_; 
  edm::InputTag beamSpotLabel_;
  double ptMin_; 
  double nSigmaZ_;
  double minZCut_; 
  double maxD0Significance_;
  
};


#endif
