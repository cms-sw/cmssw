#include "PhysicsTools/PatAlgos/interface/MuonMvaIDEstimator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "FWCore/Utilities/interface/StreamID.h"
//#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

using namespace pat;
using namespace cms::Ort;

MuonMvaIDEstimator::MuonMvaIDEstimator(const edm::FileInPath& weightsfile){
  randomForest_ = std::make_unique<ONNXRuntime>(weightsfile.fullPath());
  std::cout << randomForest_.get() << std::endl;
}


MuonMvaIDEstimator::~MuonMvaIDEstimator() {}
 
 void MuonMvaIDEstimator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
   // pfDeepBoostedJetTags
   edm::ParameterSetDescription desc;
   desc.add<edm::FileInPath>("mvaIDTrainingFile",
                             edm::FileInPath("RecoMuon/MuonIdentification/data/mvaID.onnx"));
   desc.add<std::vector<std::string>>("flav_names",
                                      std::vector<std::string>{
                                          "probBAD",
                                          "probGOOD",
                                      });
 
   descriptions.addWithDefaultLabel(desc);
 }
 
 std::unique_ptr<cms::Ort::ONNXRuntime> MuonMvaIDEstimator::initializeGlobalCache(const edm::ParameterSet &iConfig) {
   return std::make_unique<cms::Ort::ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("mvaIDTrainingFile").fullPath());
 }
 void MuonMvaIDEstimator::globalEndJob(const cms::Ort::ONNXRuntime *cache) {}
 const reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration;
 std::vector<float> MuonMvaIDEstimator::computeMVAID(const pat::Muon& muon) const {
   float local_chi2  = muon.combinedQuality().chi2LocalPosition;
   float kink  = muon.combinedQuality().trkKink;  
     
   float segment_comp =  muon.segmentCompatibility( arbitrationType);  
   float n_MatchedStations = muon.numberOfMatchedStations();   
   float pt = muon.pt();
   float eta = muon.eta();
   float global_muon = muon.isGlobalMuon(); 
   float Valid_pixel;
   float tracker_layers;
   float validFraction; 
   if (muon.innerTrack().isNonnull()){
       Valid_pixel  = muon.innerTrack()->hitPattern().numberOfValidPixelHits();
       tracker_layers  = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement();
       validFraction   = muon.innerTrack()->validFraction();
       }
   else{
       Valid_pixel = -99.;
       tracker_layers = -99.0;
       validFraction   = -99.0;
       } 
   float norm_chi2; 
   float n_Valid_hits;
   if (muon.globalTrack().isNonnull()){
       norm_chi2  = muon.globalTrack()->normalizedChi2();
       n_Valid_hits = muon.globalTrack()->hitPattern().numberOfValidMuonHits(); 
       }
   else{
       norm_chi2  = muon.innerTrack()->normalizedChi2();
       n_Valid_hits = muon.innerTrack()->hitPattern().numberOfValidMuonHits();
       }
   std::vector<std::string> input_names_ {"float_input"};
   std::vector<float> vars = {global_muon,validFraction,norm_chi2,local_chi2,kink,segment_comp,n_Valid_hits,n_MatchedStations,Valid_pixel,tracker_layers,pt,eta};
   std::vector<std::string> flav_names_{"probBAD","probGOOD"};
   //for (long unsigned int i=0; i < vars.size(); i++){
   //  input_values_.emplace_back(vars[i]);
   //}
   cms::Ort::FloatArrays input_values_;
   //cms::Ort::FloatArrays outputs;
   input_values_.emplace_back(vars);
   std::vector<float> outputs;  // init as all zeros
   //std::cout << Form("%d -- %d",input_values_[10], input_values_[11]) << std::endl;
   std::cout << randomForest_.get() << std::endl;
   outputs = randomForest_->run(input_names_, input_values_, {}, {"probabilities"})[0];
   assert(outputs.size() == flav_names_.size());
   return outputs;
}
