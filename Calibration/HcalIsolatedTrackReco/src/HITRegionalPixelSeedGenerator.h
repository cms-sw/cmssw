#ifndef HITRegionalPixelSeedGenerator_h
#define HITRegionalPixelSeedGenerator_h

//
// Class:           HITRegionalPixelSeedGenerator


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class HITRegionalPixelSeedGenerator : public TrackingRegionProducer {
 public:
  
  explicit HITRegionalPixelSeedGenerator(const edm::ParameterSet& conf_)
  {
    edm::LogInfo ("HITRegionalPixelSeedGenerator")<<"Enter the HITRegionalPixelSeedGenerator";
    
    edm::ParameterSet regionPSet = conf_.getParameter<edm::ParameterSet>("RegionPSet");
    
    ptmin=regionPSet.getParameter<double>("ptMin");
    originradius=regionPSet.getParameter<double>("originRadius");
    halflength=regionPSet.getParameter<double>("originHalfLength");
    vertexSrc=regionPSet.getParameter<std::string>("vertexSrc");
    etaCenter_=regionPSet.getParameter<double>("etaCenter");
    phiCenter_=regionPSet.getParameter<double>("phiCenter");
    deltaTrackEta = regionPSet.getParameter<double>("deltaEtaTrackRegion");
    deltaTrackPhi = regionPSet.getParameter<double>("deltaPhiTrackRegion");
    deltaL1JetEta = regionPSet.getParameter<double>("deltaEtaL1JetRegion");
    deltaL1JetPhi = regionPSet.getParameter<double>("deltaPhiL1JetRegion");
    tracksrc_ = regionPSet.getParameter<edm::InputTag>("trackSrc");
    isoTracksrc_ = regionPSet.getParameter<edm::InputTag>("isoTrackSrc");
    l1jetsrc_ = regionPSet.getParameter<edm::InputTag>("l1tjetSrc");
    usejets_ = regionPSet.getParameter<bool>("useL1Jets");
    usetracks_ = regionPSet.getParameter<bool>("useTracks");
    useIsoTracks_ = regionPSet.getParameter<bool>("useIsoTracks");
    fixedReg_ = regionPSet.getParameter<bool>("fixedReg");
  }
  
  virtual ~HITRegionalPixelSeedGenerator() {}
  
  
  virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const 
    {
      std::vector<TrackingRegion* > result;
      float originz =0.;
      
      double deltaZVertex =  halflength;
      
      
      
      if (usetracks_)
	{
	  edm::Handle<reco::TrackCollection> tracks;
	  e.getByLabel(tracksrc_, tracks);
	  
	  edm::Handle<reco::VertexCollection> vertices;
	  e.getByLabel(vertexSrc,vertices);
	  const reco::VertexCollection vertCollection = *(vertices.product());
	  reco::VertexCollection::const_iterator ci = vertCollection.begin();
	  
	  if(vertCollection.size() > 0) 
	    {
	      originz = ci->z();
	    }
	  else
	    {
	      deltaZVertex = 15.;
	    }
      
	  GlobalVector globalVector(0,0,1);
	  if(tracks->size() == 0) return result;
	  
	  reco::TrackCollection::const_iterator itr = tracks->begin();
	  for(;itr != tracks->end();itr++)
	    {
	      
	      GlobalVector ptrVec((itr)->px(),(itr)->py(),(itr)->pz());
	      globalVector = ptrVec;
	      
	      
	      RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
												   GlobalPoint(0,0,originz), 
												   ptmin,
												   originradius,
												   deltaZVertex,
												   deltaTrackEta,
												   deltaTrackPhi);
	      result.push_back(etaphiRegion);
	      
	    }
	}
      
      if (useIsoTracks_)
        {
          edm::Handle<trigger::TriggerFilterObjectWithRefs> isotracks;
          e.getByLabel(isoTracksrc_, isotracks);

	  std::vector< edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
	
	  isotracks->getObjects(trigger::TriggerTrack, isoPixTrackRefs);
	  
          edm::Handle<reco::VertexCollection> vertices;
          e.getByLabel(vertexSrc,vertices);
          const reco::VertexCollection vertCollection = *(vertices.product());
          reco::VertexCollection::const_iterator ci = vertCollection.begin();

          if(vertCollection.size() > 0) 
	    {
	      originz = ci->z();
	    }
	  else
	    {
	      deltaZVertex = 15.;
	    }
	  
          GlobalVector globalVector(0,0,1);
          if(isoPixTrackRefs.size() == 0) return result;
	  
          for(uint32_t p=0; p<isoPixTrackRefs.size(); p++)
            {
              GlobalVector ptrVec((isoPixTrackRefs[p]->track())->px(),(isoPixTrackRefs[p]->track())->py(),(isoPixTrackRefs[p]->track())->pz());
              globalVector = ptrVec;
	      
              RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
                                                                                                   GlobalPoint(0,0,originz),
                                                                                                   ptmin,
                                                                                                   originradius,
                                                                                                   deltaZVertex,
                                                                                                   deltaTrackEta,
                                                                                                   deltaTrackPhi);
              result.push_back(etaphiRegion);
	    }
	}
      
      if (usejets_)
	{
	  edm::Handle<l1extra::L1JetParticleCollection> jets;
	  e.getByLabel(l1jetsrc_, jets);
	  
          edm::Handle<reco::VertexCollection> vertices;
          e.getByLabel(vertexSrc,vertices);
          const reco::VertexCollection vertCollection = *(vertices.product());
          reco::VertexCollection::const_iterator ci = vertCollection.begin();
          if(vertCollection.size() > 0) 
	    {
	      originz = ci->z();
	    }
	  else
	    {
	      deltaZVertex = 15.;
	    }

	  GlobalVector globalVector(0,0,1);
	  if(jets->size() == 0) return result;
	  
	  for (l1extra::L1JetParticleCollection::const_iterator iJet = jets->begin(); iJet != jets->end(); iJet++) 
	    {
	      GlobalVector jetVector(iJet->p4().x(), iJet->p4().y(), iJet->p4().z());
	      GlobalPoint  vertex(0, 0, originz);
	      
	      RectangularEtaPhiTrackingRegion* etaphiRegion = new RectangularEtaPhiTrackingRegion( jetVector,
												   vertex,
												   ptmin,
												   originradius,
												   deltaZVertex,
												   deltaL1JetEta,
												   deltaL1JetPhi );
	      result.push_back(etaphiRegion);
	    }
	}
      if (fixedReg_)
	{
	  GlobalVector fixedVector(cos(phiCenter_)*sin(2*atan(exp(-etaCenter_))), sin(phiCenter_)*sin(2*atan(exp(-etaCenter_))), cos(2*atan(exp(-etaCenter_))));
	  GlobalPoint  vertex(0, 0, originz);
	  
	  edm::Handle<reco::VertexCollection> vertices;
	  e.getByLabel(vertexSrc,vertices);
	  const reco::VertexCollection vertCollection = *(vertices.product());
	  reco::VertexCollection::const_iterator ci = vertCollection.begin();
	  if(vertCollection.size() > 0) 
	    {
	      originz = ci->z();
	    }
	  else
	    {
	      deltaZVertex = 15.;
	    }
	  
	  RectangularEtaPhiTrackingRegion* etaphiRegion = new RectangularEtaPhiTrackingRegion( fixedVector,
											       vertex,
											       ptmin,
											       originradius,
											       deltaZVertex,
											       deltaL1JetEta,
											       deltaL1JetPhi );
	  result.push_back(etaphiRegion);
	}
      
      return result;
    }
  
  
 private:
  edm::ParameterSet conf_;
  
  float ptmin;
  float originradius;
  float halflength;
  double etaCenter_;
  double phiCenter_;
  float deltaTrackEta;
  float deltaTrackPhi;
  float deltaL1JetEta;
  float deltaL1JetPhi;
  edm::InputTag tracksrc_;
  edm::InputTag isoTracksrc_;
  std::string vertexSrc;
  edm::InputTag l1jetsrc_;
  bool usejets_;
  bool usetracks_;
  bool fixedReg_;
  bool useIsoTracks_;

};

#endif



