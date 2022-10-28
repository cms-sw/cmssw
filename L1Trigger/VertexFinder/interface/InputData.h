#ifndef __L1Trigger_VertexFinder_InputData_h__
#define __L1Trigger_VertexFinder_InputData_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
#include "L1Trigger/VertexFinder/interface/Vertex.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <vector>

namespace l1tVertexFinder {

  typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;
  typedef std::map<TrackingParticlePtr, edm::RefToBase<TrackingParticle>> TPPtrToRefMap;

  class AnalysisSettings;
  class Stub;

  //=== Unpacks stub & tracking particle (truth) data into user-friendlier format in Stub & TP classes.
  //=== Also makes B-field available to Settings class.

  class InputData {
  public:
    /// Constructor and destructor
    InputData();
    InputData(const edm::Event& iEvent,
              const edm::EventSetup& iSetup,
              const AnalysisSettings& settings,
              const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken,
              const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticlesToken,
              const edm::EDGetTokenT<edm::View<TrackingParticle>> tpToken,
              const edm::EDGetTokenT<edm::ValueMap<l1tVertexFinder::TP>> tpValueMapToken,
              const edm::EDGetTokenT<DetSetVec> stubToken,
              const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken,
              const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken);
    ~InputData();

    // Sort Tracking Particles by vertex z position
    struct SortVertexByPt {
      inline bool operator()(const Vertex vertex0, const Vertex vertex1) { return (vertex0.pT() > vertex1.pT()); }
    };

    // Sort Tracking Particles by vertex z position
    struct SortVertexByZ0 {
      inline bool operator()(const Vertex vertex0, const Vertex vertex1) { return (vertex0.z0() < vertex1.z0()); }
    };

    /// Get the TrackingParticle to TP translation map
    const TPPtrToRefMap& getTPPtrToRefMap() const { return tpPtrToRefMap_; }
    /// Get primary vertex information (vertex from HepMCProduct)
    const Vertex& getHepMCVertex() const { return hepMCVertex_; }
    /// Get primary vertex information (vertex from gen particles)
    const Vertex& getGenVertex() const { return genVertex_; }
    /// Get primary vertex information (vertex from tracking particles)
    const Vertex& getPrimaryVertex() const { return vertex_; }
    /// Get pile-up vertices information
    const std::vector<Vertex>& getPileUpVertices() const { return vertices_; }
    /// Get reconstructable pile-up vertices information
    const std::vector<Vertex>& getRecoPileUpVertices() const { return recoVertices_; }

    const float genPt() const { return genPt_; }
    const float genPt_PU() const { return genPt_PU_; }

  private:
    Vertex hepMCVertex_;                // primary vertex from the edm::HepMCProduct
    Vertex genVertex_;                  // primary vertex formed from the gen particles collection
    Vertex vertex_;                     // primary vertex formed from the tracking particles
    std::vector<Vertex> vertices_;      // pile-up vertices
    std::vector<Vertex> recoVertices_;  // reconstructable pile-up vertices
    std::vector<Stub>
        vAllStubs_;  // all stubs, even those that would fail any tightened front-end readout electronic cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure
                     // the efficiency of these cuts).
    TPPtrToRefMap tpPtrToRefMap_;
    float genPt_;
    float genPt_PU_;
  };

}  // end namespace l1tVertexFinder

#endif
