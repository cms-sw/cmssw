#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "getBestVertex.h"

//from lwtnn
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "lwtnn/LightweightNeuralNetwork.hh"

//Used for manually accepting problematic short/detached tracks to general collection
std::map<int, float> generalCollThresholds = 
{
{4,0.0}, 			//initialStep
{5,0.2},			//lowPtTripletStep
{6,-0.6}, 			//pixelPairStep
{7,0.0},			//detachedTripletStep
{8,-0.8},			//mixedTripletStep
{9,-0.6},			//pixelLessStep
{10,-0.4},			//tobTecStep
{11,0.6},			//jetCoreRegionalStep
{22,0.8},			//highPtTripletStep
{23,0.2},			//lowPtQuadStep
{24,-0.6},			//detachedQuadStep
};

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
                     reco::VertexCollection const & vertices,
                     lwt::ValueMap & inputs) const {
      // lwt::ValueMap is typedef for std::map<std::string, double>
      //
      // It is cached per event to avoid constructing the map for each
      // track while keeping the operator() interface const.

      Point bestVertex = getBestVertex(trk,vertices);

      inputs["trk_pt"] = trk.pt();
      inputs["trk_eta"] = trk.eta();
      inputs["trk_lambda"] = trk.lambda();
      inputs["trk_dxy"] = trk.dxy(beamSpot.position()); // Training done without taking absolute value
      inputs["trk_dz"] = trk.dz(beamSpot.position()); // Training done without taking absolute value
      inputs["trk_dxyClosestPV"] = trk.dxy(bestVertex); // Training done without taking absolute value
      inputs["trk_dzClosestPVNorm"] = std::max(-0.2, std::min(trk.dz(bestVertex), 0.2)); // Training done without taking absolute value
      inputs["trk_ptErr"] = trk.ptError();
      inputs["trk_etaErr"] = trk.etaError();
      inputs["trk_lambdaErr"] = trk.lambdaError();
      inputs["trk_dxyErr"] = trk.dxyError();
      inputs["trk_dzErr"] = trk.dzError();
      inputs["trk_nChi2"] = trk.normalizedChi2();
      inputs["trk_ndof"] = trk.ndof();
      inputs["trk_nInvalid"] = trk.hitPattern().numberOfLostHits(reco::HitPattern::TRACK_HITS);
      inputs["trk_nPixel"] = trk.hitPattern().numberOfValidPixelHits();
      inputs["trk_nStrip"] = trk.hitPattern().numberOfValidStripHits();
      inputs["trk_nPixelLay"] = trk.hitPattern().pixelLayersWithMeasurement();
      inputs["trk_nStripLay"] = trk.hitPattern().stripLayersWithMeasurement();
      inputs["trk_n3DLay"] = (trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo()+trk.hitPattern().pixelLayersWithMeasurement());
      inputs["trk_nLostLay"] = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
      inputs["trk_algo"] = trk.algo(); // eventually move to originalAlgo

      auto out = neuralNetwork_->compute(inputs);
      // there should only one output
      if(out.size() != 1) throw cms::Exception("LogicError") << "Expecting exactly one output from NN, got " << out.size();

      float output = 2.0*out.begin()->second-1.0;

      //Special clauses for special tracks
      //T1qqqq
      if((std::abs(inputs["trk_dxy"])>=0.1) && (inputs["trk_etaErr"]<0.003) && (inputs["trk_dxyErr"]<0.03) &&(inputs["trk_ndof"]>3)){
        //Set value to just above the threshold
        if(generalCollThresholds[trk.algo()]){
                float thres_ = generalCollThresholds[trk.algo()]+0.01;
                return std::max(thres_,output);
        }
      }

      //T5qqqqLL
      if((inputs["trk_pt"]>100.0)&&(inputs["trk_nChi2"]<4.0)&&(inputs["trk_etaErr"]<0.001)){
        //Set value to just above the threshold
        if(generalCollThresholds[trk.algo()]){
                float thres_ = generalCollThresholds[trk.algo()]+0.01;
                return std::max(thres_,output);
        }
      }


      return output;
    }


    std::string lwtnnLabel_;
    const lwt::LightweightNeuralNetwork *neuralNetwork_;
  };

  using TrackLwtnnClassifier = TrackMVAClassifier<lwtnn, lwt::ValueMap>;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackLwtnnClassifier);
