// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 07/2024

#ifndef RecoHGCal_TICL_TracksterInferenceAlgo_H__
#define RecoHGCal_TICL_TracksterInferenceAlgo_H__

#include <vector>
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceAlgoBase(const edm::ParameterSet& conf)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~TracksterInferenceAlgoBase() {}

    virtual void inputData(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& tracksters) = 0;
    virtual void runInference(std::vector<Trackster>& tracksters) = 0;
    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
