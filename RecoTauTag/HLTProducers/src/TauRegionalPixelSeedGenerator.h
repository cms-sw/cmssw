#ifndef TauRegionalPixelSeedGenerator_h
#define TauRegionalPixelSeedGenerator_h

//
// Class:           TauRegionalPixelSeedGenerator


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"


class TauRegionalPixelSeedGenerator : public TrackingRegionProducer {
  public:
    
    explicit TauRegionalPixelSeedGenerator(const edm::ParameterSet& conf_,
	edm::ConsumesCollector && iC){
      edm::LogInfo ("TauRegionalPixelSeedGenerator") << "Enter the TauRegionalPixelSeedGenerator";

      edm::ParameterSet regionPSet = conf_.getParameter<edm::ParameterSet>("RegionPSet");

      m_ptMin        = regionPSet.getParameter<double>("ptMin");
      m_originRadius = regionPSet.getParameter<double>("originRadius");
      m_halfLength   = regionPSet.getParameter<double>("originHalfLength");
      m_deltaEta     = regionPSet.getParameter<double>("deltaEtaRegion");
      m_deltaPhi     = regionPSet.getParameter<double>("deltaPhiRegion");
      token_jet      = iC.consumes<reco::CandidateView>(regionPSet.getParameter<edm::InputTag>("JetSrc"));
      token_vertex   = iC.consumes<reco::VertexCollection>(regionPSet.getParameter<edm::InputTag>("vertexSrc"));
      if (regionPSet.exists("searchOpt")){
	m_searchOpt    = regionPSet.getParameter<bool>("searchOpt");
      }
      else{
	m_searchOpt = false;
      }
      m_measurementTracker ="";
      m_howToUseMeasurementTracker=0;
      if (regionPSet.exists("measurementTrackerName")){
	m_measurementTracker = regionPSet.getParameter<std::string>("measurementTrackerName");
	if (regionPSet.exists("howToUseMeasurementTracker")){
	  m_howToUseMeasurementTracker = regionPSet.getParameter<double>("howToUseMeasurementTracker");
	}
      }
    }
  
    virtual ~TauRegionalPixelSeedGenerator() {}
    

    virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const {
      std::vector<TrackingRegion* > result;

      //      double originZ;
      double deltaZVertex, deltaRho;
        GlobalPoint vertex;
      // get the primary vertex
      edm::Handle<reco::VertexCollection> h_vertices;
      e.getByToken(token_vertex, h_vertices);
      const reco::VertexCollection & vertices = * h_vertices;
      if (not vertices.empty()) {
//        originZ      = vertices.front().z();
	GlobalPoint myTmp(vertices.at(0).position().x(),vertices.at(0).position().y(), vertices.at(0).position().z());
          vertex = myTmp;
          deltaZVertex = m_halfLength;
          deltaRho = m_originRadius;
      } else {
  //      originZ      =  0.;
          GlobalPoint myTmp(0.,0.,0.);
          vertex = myTmp;
          deltaRho = 1.;
         deltaZVertex = 15.;
      }
      
      // get the jet direction
      edm::Handle<edm::View<reco::Candidate> > h_jets;
      e.getByToken(token_jet, h_jets);
      
      for (unsigned int iJet =0; iJet < h_jets->size(); ++iJet)
	{
	  const reco::Candidate & myJet = (*h_jets)[iJet];
          GlobalVector jetVector(myJet.momentum().x(),myJet.momentum().y(),myJet.momentum().z());
//          GlobalPoint  vertex(0, 0, originZ);
          RectangularEtaPhiTrackingRegion* etaphiRegion = new RectangularEtaPhiTrackingRegion( jetVector,
                                                                                               vertex,
                                                                                               m_ptMin,
                                                                                               deltaRho,
                                                                                               deltaZVertex,
                                                                                               m_deltaEta,
                                                                                               m_deltaPhi,
											       m_howToUseMeasurementTracker,
											       true,
											       m_measurementTracker,
											       m_searchOpt);
          result.push_back(etaphiRegion);
      }

      return result;
    }
  
 private:
  edm::ParameterSet conf_;

  float m_ptMin;
  float m_originRadius;
  float m_halfLength;
  float m_deltaEta;
  float m_deltaPhi;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::CandidateView> token_jet; 
  std::string m_measurementTracker;
  double m_howToUseMeasurementTracker;
  bool m_searchOpt;
};

#endif
