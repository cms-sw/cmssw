#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "getBestVertex.h"

//from lwtnn
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "lwtnn/LightweightNeuralNetwork.hh"

namespace {
  struct lwtnn {
    lwtnn(const edm::ParameterSet& cfg):
      lwtnnLabel_(cfg.getParameter<std::string>("lwtnnLabel"))
    {}

    static const char *name() { return "TrackLwtnnClassifier"; }

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<std::string>("lwtnnLabel", "trackSelectionLwtnn");
    }

    void beginStream() {}
    void initEvent(const edm::EventSetup& es) {
      edm::ESHandle<lwt::LightweightNeuralNetwork> lwtnnHandle;
      es.get<TrackingComponentsRecord>().get(lwtnnLabel_, lwtnnHandle);
      neuralNetwork_ = lwtnnHandle.product();
    }

    float operator()(reco::Track const & trk,
                     reco::BeamSpot const & beamSpot,
                     reco::VertexCollection const & vertices) const {

      Point bestVertex = getBestVertex(trk,vertices);

      inputs_["trk_pt"] = trk.pt();
      inputs_["trk_eta"] = trk.eta();
      inputs_["trk_lambda"] = trk.lambda();
      inputs_["trk_dxy"] = trk.dxy(beamSpot.position()); // is the training with abs() or not?
      inputs_["trk_dz"] = trk.dz(beamSpot.position()); // is the training with abs() or not?
      inputs_["trk_dxyClosestPV"] = trk.dxy(bestVertex); // is the training with abs() or not?
      inputs_["trk_dzClosestPV"] = trk.dz(bestVertex); // is the training with abs() or not?
      inputs_["trk_ptErr"] = trk.ptError();
      inputs_["trk_etaErr"] = trk.etaError();
      inputs_["trk_lambdaErr"] = trk.lambdaError();
      inputs_["trk_dxyErr"] = trk.dxyError();
      inputs_["trk_dzErr"] = trk.dzError();
      inputs_["trk_nChi2"] = trk.normalizedChi2();
      inputs_["trk_ndof"] = trk.ndof();
      inputs_["trk_nInvalid"] = trk.hitPattern().numberOfLostHits(reco::HitPattern::TRACK_HITS);
      inputs_["trk_nPixel"] = trk.hitPattern().numberOfValidPixelHits();
      inputs_["trk_nStrip"] = trk.hitPattern().numberOfValidStripHits();
      inputs_["trk_nPixelLay"] = trk.hitPattern().pixelLayersWithMeasurement();
      inputs_["trk_nStripLay"] = trk.hitPattern().stripLayersWithMeasurement();
      inputs_["trk_n3DLay"] = (trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo()+trk.hitPattern().pixelLayersWithMeasurement());
      inputs_["trk_nLostLay"] = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
      inputs_["trk_algo"] = trk.algo(); // eventually move to originalAlgo

      auto out = neuralNetwork_->compute(inputs_);
      // there should only one output
      if(out.size() != 1) throw cms::Exception("LogicError") << "Expecting exactly one output from NN, got " << out.size();


      float output = 2.0*out.begin()->second-1.0;
      return output;
    }


    std::string lwtnnLabel_;
    const lwt::LightweightNeuralNetwork *neuralNetwork_;
    // inputs_ is mutable in order to avoid constructing
    // map<string,double> for each track while keeping the operator()
    // interface const. Thread safety is in practice achieved by
    // TrackMVAClassifierBase (and inheriting classes) being a stream
    // module.
    mutable lwt::ValueMap inputs_; //typedef of map<string, double>
  };

  using TrackLwtnnClassifier = TrackMVAClassifier<lwtnn>;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackLwtnnClassifier);
