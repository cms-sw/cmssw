#ifndef VERTEXFINDER_CLASSES_H
#define VERTEXFINDER_CLASSES_H

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "L1Trigger/VertexFinder/interface/Vertex.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
#include "L1Trigger/VertexFinder/interface/InputData.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <map>
#include <vector>

namespace {
  struct vfdictionary {
    l1tVertexFinder::InputData id;
    l1tVertexFinder::Vertex vtx;
    l1tVertexFinder::TP tp;
    l1tVertexFinder::Stub s;
    std::map<DetId, DetId> mdid;

    edm::Ptr<l1tVertexFinder::Stub> ptrstub;
    edm::Ptr<l1tVertexFinder::TP> ptrtp;

    edm::RefToBase<TrackingParticle> rtbtp;

    std::map<edm::Ptr<TrackingParticle>, edm::RefToBase<TrackingParticle>> mptrtprtbtp;

    edm::ValueMap<edm::Ptr<l1tVertexFinder::Stub>> vmptrstub;
    edm::ValueMap<edm::Ptr<l1tVertexFinder::TP>> vmptrtp;
    edm::ValueMap<l1tVertexFinder::Stub> vmstub;
    edm::ValueMap<l1tVertexFinder::TP> vmpt;

    std::vector<l1tVertexFinder::Stub> vs;
    std::vector<l1tVertexFinder::TP> vtp;
    std::vector<l1tVertexFinder::Vertex> vvtx;
    std::vector<edm::Ptr<l1tVertexFinder::Stub>> vptrstub;
    std::vector<edm::Ptr<l1tVertexFinder::TP>> vptrtp;

    edm::Wrapper<l1tVertexFinder::Vertex> wvtx;
    edm::Wrapper<l1tVertexFinder::TP> wtp;
    edm::Wrapper<l1tVertexFinder::Stub> ws;
    edm::Wrapper<edm::ValueMap<edm::Ptr<l1tVertexFinder::Stub>>> wvmptrstub;
    edm::Wrapper<edm::ValueMap<edm::Ptr<l1tVertexFinder::TP>>> wvmprttp;
    edm::Wrapper<edm::ValueMap<l1tVertexFinder::Stub>> wvmstub;
    edm::Wrapper<edm::ValueMap<l1tVertexFinder::TP>> wvmtp;
    edm::Wrapper<std::map<DetId, DetId>> wmdid;
    edm::Wrapper<std::vector<l1tVertexFinder::Stub>> wvs;
    edm::Wrapper<std::vector<l1tVertexFinder::TP>> wvtp;
    edm::Wrapper<std::vector<l1tVertexFinder::Vertex>> wvvtx;
    edm::Wrapper<l1tVertexFinder::InputData> wid;
  };
}  // namespace

#endif