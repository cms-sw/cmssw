#ifndef TauRegionalPixelSeedGenerator_h
#define TauRegionalPixelSeedGenerator_h

//
// Class:           TauRegionalPixelSeedGenerator


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
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

using namespace reco;
class TauRegionalPixelSeedGenerator : public TrackingRegionProducer {
  public:
    
    explicit TauRegionalPixelSeedGenerator(const edm::ParameterSet& conf_){
      edm::LogInfo ("TauRegionalPixelSeedGenerator") << "Enter the TauRegionalPixelSeedGenerator";

      edm::ParameterSet regionPSet = conf_.getParameter<edm::ParameterSet>("RegionPSet");

      m_ptMin        = regionPSet.getParameter<double>("ptMin");
      m_originRadius = regionPSet.getParameter<double>("originRadius");
      m_halfLength   = regionPSet.getParameter<double>("originHalfLength");
      m_deltaEta     = regionPSet.getParameter<double>("deltaEtaRegion");
      m_deltaPhi     = regionPSet.getParameter<double>("deltaPhiRegion");
      m_jetSrc       = regionPSet.getParameter<edm::InputTag>("JetSrc");
      m_vertexSrc    = regionPSet.getParameter<edm::InputTag>("vertexSrc");
    }
  
    virtual ~TauRegionalPixelSeedGenerator() {}
    

    virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const {
      std::vector<TrackingRegion* > result;

      double originZ;
      double deltaZVertex;
      
      // get the primary vertex
      edm::Handle<reco::VertexCollection> h_vertices;
      e.getByLabel(m_vertexSrc, h_vertices);
      const reco::VertexCollection & vertices = * h_vertices;
      if (not vertices.empty()) {
        originZ      = vertices.front().z();
        deltaZVertex = m_halfLength;
      } else {
        originZ      =  0.;
        deltaZVertex = 15.;
      }
      
      // get the jet direction
      edm::Handle<reco::CaloJetCollection> h_jets;
      e.getByLabel(m_jetSrc, h_jets);
      const reco::CaloJetCollection & jets = * h_jets;
      
      for (reco::CaloJetCollection::const_iterator iJet = jets.begin(); iJet != jets.end(); ++iJet)
      {
          GlobalVector jetVector(iJet->p4().x(), iJet->p4().y(), iJet->p4().z());
          GlobalPoint  vertex(0, 0, originZ);
          RectangularEtaPhiTrackingRegion* etaphiRegion = new RectangularEtaPhiTrackingRegion( jetVector,
                                                                                               vertex,
                                                                                               m_ptMin,
                                                                                               m_originRadius,
                                                                                               deltaZVertex,
                                                                                               m_deltaEta,
                                                                                               m_deltaPhi );
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
  edm::InputTag m_jetSrc;
  edm::InputTag m_vertexSrc;
};

#endif
