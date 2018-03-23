#ifndef __L1Trigger_VertexFinder_InputData_h__
#define __L1Trigger_VertexFinder_InputData_h__


#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
#include "L1Trigger/VertexFinder/interface/Vertex.h"



namespace l1tVertexFinder {

class Settings;
class Stub;

//=== Unpacks stub & tracking particle (truth) data into user-friendlier format in Stub & TP classes.
//=== Also makes B-field available to Settings class.

class InputData {

public:
  
  InputData(const edm::Event& iEvent, const edm::EventSetup& iSetup, Settings* settings,
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag,
  const edm::EDGetTokenT<DetSetVec> stubInputTag,
  const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag,
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag
   );

  // Sort Tracking Particles by vertex z position
  struct SortVertexByZ0{
    inline bool operator() (const Vertex vertex0, const Vertex vertex1){
      return(vertex0.z0() < vertex1.z0());
    }
  };

  // Get tracking particles
  const std::vector<TP>&     getTPs()      const {return vTPs_;}
  /// Get Primary vertex information
  const Vertex&              getPrimaryVertex()       const {return vertex_;}
  /// Get PileUp Vertices information
  const std::vector<Vertex>& getPileUpVertices()      const {return vertices_;}
  /// Get reconstructable pile-up vertices information
  const std::vector<Vertex>& getRecoPileUpVertices()      const {return recoVertices_;}

  const std::map<DetId, DetId>& getStubGeoDetIdMap() const {return stubGeoDetIdMap_;}

  /// Generated MET
  const float                GenMET()          const { return genMET_;}
  const float                GenMET_PU()          const { return genMET_PU_;}

  const float                GenPt()          const { return genPt_;}
  const float                GenPt_PU()          const { return genPt_PU_;}

private:
  // const edm::EDGetTokenT<TrackingParticleCollection> inputTag;


//  // Can optionally be used to sort stubs by bend.
//  struct SortStubsInBend {
//     inline bool operator() (const Stub* stub1, const Stub* stub2) {
//        return(fabs(stub1->bend()) < fabs(stub2->bend()));
//     }
//  };

private:

  std::vector<TP> vTPs_; // tracking particles
  Vertex vertex_;
  std::vector<Vertex> vertices_;
  std::vector<Vertex> recoVertices_;

  //--- of minor importance ...

  std::vector<Stub> vAllStubs_; // all stubs, even those that would fail any tightened front-end readout electronic cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure the efficiency of these cuts).
  std::map<DetId, DetId> stubGeoDetIdMap_;
  float genMET_;
  float genMET_PU_;
  float genPt_;
  float genPt_PU_;
};

} // end namespace l1tVertexFinder

#endif

