#ifndef TauRegionalPixelSeedGenerator_h
#define TauRegionalPixelSeedGenerator_h

//
// Class:           TauRegionalPixelSeedGenerator


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;
class TauRegionalPixelSeedGenerator : public TrackingRegionProducer {
  public:
    
    explicit TauRegionalPixelSeedGenerator(const edm::ParameterSet& conf_){
      edm::LogInfo ("TauRegionalPixelSeedGenerator")<<"Enter the TauRegionalPixelSeedGenerator";

      edm::ParameterSet regionFactoryPSet = conf_.getParameter<edm::ParameterSet>("RegionFactoryPSet");
      edm::ParameterSet regionPSet = regionFactoryPSet.getParameter<edm::ParameterSet>("RegionPSet");

      ptmin=regionPSet.getParameter<double>("ptMin");
      originradius=regionPSet.getParameter<double>("originRadius");
      halflength=regionPSet.getParameter<double>("originHalfLength");
      vertexSrc=regionPSet.getParameter<std::string>("vertexSrc");
      deltaEta = regionPSet.getParameter<double>("deltaEtaRegion");
      deltaPhi = regionPSet.getParameter<double>("deltaPhiRegion");
      jetSrc = regionPSet.getParameter<edm::InputTag>("JetSrc");
    }
  
    virtual ~TauRegionalPixelSeedGenerator() {}
    

    virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const {
      std::vector<TrackingRegion* > result;
      float originz =0.;
	
      double deltaZVertex =  halflength;
      // get Inputs
      edm::Handle<reco::VertexCollection> vertices;
      e.getByLabel(vertexSrc,vertices);
      const reco::VertexCollection vertCollection = *(vertices.product());
      reco::VertexCollection::const_iterator ci = vertCollection.begin();
      if(vertCollection.size() > 0) {
	originz = ci->z();
      }else{
	deltaZVertex = 15.;
      }
      
      //Get the jet direction
      edm::Handle<CaloJetCollection> jets;
      e.getByLabel(jetSrc, jets);
      
      GlobalVector globalVector(0,0,1);
      if(jets->size() == 0) return result;
      
      CaloJetCollection::const_iterator iJet = jets->begin();
      for(;iJet != jets->end();iJet++)
	{
	
	  GlobalVector jetVector((iJet)->p4().x(),(iJet)->p4().y(),(iJet)->p4().z());
	  globalVector = jetVector;
	  
	
	  RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
											       GlobalPoint(0,0,originz), 
											       ptmin,
											       originradius,
											       deltaZVertex,
											       deltaEta,
											       deltaPhi);
	  result.push_back(etaphiRegion);
	  
	}
      
      return result;
    }

  
 private:
  edm::ParameterSet conf_;

  float ptmin;
  float originradius;
  float halflength;
  float deltaEta;
  float deltaPhi;
  edm::InputTag jetSrc;
  std::string vertexSrc;
};

#endif
