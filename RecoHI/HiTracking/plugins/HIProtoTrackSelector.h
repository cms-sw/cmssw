#ifndef HIProtoTrackSelection_h
#define HIProtoTrackSelection_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iostream>
using namespace std;
using namespace edm;


/**
 Selector to select prototracks that pass certain kinematic cuts based on fast vertex
 
 Inspired by CommonTools.RecoAlgos.TrackingParticleSelector.h 
 and SimTracker.TrackHistory.BTrackingParticleSelector.h
 and UserCode.CmsHi.TrackAnalysis.HitPixelLayersTPSelector.h
 and https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideGenericSelectors#Generic_Object_Selector_and_User
 **/

class HIProtoTrackSelector
	{
		
	public:
		// input collection type
		typedef reco::TrackCollection collection;
		
		// output collection type
		typedef vector<const reco::Track *> container;
		
		// iterator over result collection type.
		typedef container::const_iterator const_iterator;
		
		// constructor from parameter set configurability
		HIProtoTrackSelector(const edm::ParameterSet & iConfig) : 
		vertexCollection_(iConfig.getParameter<string>("VertexCollection")),
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
			
			if(vertices->size()>0) {
				vtx_=vertices->begin()->position();
				vzErr_=vertices->begin()->zError();
				LogInfo("HeavyIonVertexing") << "Select prototracks compatible with median vertex"
				<< "\n   vz = " << vtx_.Z()  
				<< "\n   " << nSigmaZ_ << " vz sigmas = " << vzErr_*nSigmaZ_
				<< "\n   cut at = " << max(vzErr_*nSigmaZ_,minZCut_);
			} else {
				LogError("HeavyIonVertexing") << "No vertex found in collection '" << vertexCollection_ << "'";
			}
			
			int nSelected=0;
			int nRejected=0;
			for (reco::TrackCollection::const_iterator trk = c.begin(); trk != c.end(); ++ trk)
			{
				
				if ( trk->pt() > ptMin_ && // keep only tracks above ptMin
					 fabs(trk->d0()/trk->d0Error()) < maxD0Significance_ && // keep only tracks with D0 significance less than cut
					 fabs(trk->dz(vtx_)) <  max(vzErr_*nSigmaZ_,minZCut_) // keep all tracks within minZCut or within nSigmaZ zErrors of fast vertex
					) 
				{
					//LogTrace("HeavyIonVertexing") << "SELECTED: dz=" << trk->dz(vtx_) << "\t d0/d0err=" << trk->d0()/trk->d0Error() << "\t pt=" << trk->pt();
					nSelected++;
					selected_.push_back( & * trk );
				} else {
					//LogTrace("HeavyIonVertexing") << "\t REJECTED: dz=" << trk->dz(vtx_) << "\t d0/d0err=" << trk->d0()/trk->d0Error() << "\t pt=" << trk->pt();
					nRejected++;
				}
				
			}
			
			LogInfo("HeavyIonVertexing") << "selected " << nSelected << " prototracks out of " << nRejected+nSelected << endl;
			
		}
		
		// iterators over selected objects: collection begin
		const_iterator begin() const { return selected_.begin(); }
		
		// iterators over selected objects: collection end
		const_iterator end() const { return selected_.end(); }
		
		// true if no object has been selected
		size_t size() const { return selected_.size(); }
		
		
private:
		container selected_;		
		string vertexCollection_; 
		double ptMin_; 
		double nSigmaZ_;
		double minZCut_; 
		math::XYZPoint vtx_;
		double vzErr_;
		double maxD0Significance_;
		
	};


#endif
