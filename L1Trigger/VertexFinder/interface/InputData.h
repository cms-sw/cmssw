#ifndef __L1Trigger_VertexFinder_InputData_h__
#define __L1Trigger_VertexFinder_InputData_h__

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
#include "L1Trigger/VertexFinder/interface/Vertex.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <vector>

namespace l1tVertexFinder {

  typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;
  typedef std::map<TrackingParticlePtr, const TP*> TPTranslationMap;

  class AnalysisSettings;
  class Stub;

  //=== Unpacks stub & tracking particle (truth) data into user-friendlier format in Stub & TP classes.
  //=== Also makes B-field available to Settings class.

  class InputData {
  public:
    InputData(const edm::Event& iEvent,
              const edm::EventSetup& iSetup,
              const AnalysisSettings& settings,
              const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken,
              const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticlesToken,
              edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_,
              edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_,
              const edm::EDGetTokenT<TrackingParticleCollection> tpToken,
              const edm::EDGetTokenT<DetSetVec> stubToken,
              const edm::EDGetTokenT<TTStubAssMap> stubTruthToken,
              const edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken);

    // Sort Tracking Particles by vertex z position
    struct SortVertexByZ0 {
      inline bool operator()(const Vertex vertex0, const Vertex vertex1) { return (vertex0.z0() < vertex1.z0()); }
    };

    /// Get tracking particles
    const std::vector<TP>& getTPs() const { return vTPs_; }
    /// Get the TrackingParticle to TP translation map
    const TPTranslationMap& getTPTranslationMap() const { return translateTP_; }
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

    const std::map<DetId, DetId>& getStubGeoDetIdMap() const { return stubGeoDetIdMap_; }

    const float genPt() const { return genPt_; }
    const float genPt_PU() const { return genPt_PU_; }

  private:
    std::vector<TP> vTPs_;              // tracking particles
    Vertex hepMCVertex_;                // primary vertex from the edm::HepMCProduct
    Vertex genVertex_;                  // primary vertex formed from the gen particles collection
    Vertex vertex_;                     // primary vertex formed from the tracking particles
    std::vector<Vertex> vertices_;      // pile-up vertices
    std::vector<Vertex> recoVertices_;  // reconstructable pile-up vertices
    std::vector<Stub>
        vAllStubs_;  // all stubs, even those that would fail any tightened front-end readout electronic cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure
                     // the efficiency of these cuts).
    std::map<DetId, DetId> stubGeoDetIdMap_;
    TPTranslationMap translateTP_;
    float genPt_;
    float genPt_PU_;
  };

}  // end namespace l1tVertexFinder

#endif
