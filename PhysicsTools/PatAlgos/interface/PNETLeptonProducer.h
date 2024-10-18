#ifndef PhysicsTools_PatAlgos_PNETLeptonProducer
#define PhysicsTools_PatAlgos_PNETLeptonProducer

// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      PNETLeptonProducer
//
/**\class PNETLeptonProducer PNETLeptonProducer.cc PhysicsTools/PatAlgos/plugins/PNETLeptonProducer.cc


*/
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"  // this is flexible enough for our purposes
//#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h" // this will need to be moved to the dedicated producer
#include "PhysicsTools/PatAlgos/interface/LeptonTagInfoCollectionProducer.h"

#include <string>
//
// class declaration
//

template <typename T>
class PNETLeptonProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  PNETLeptonProducer(const edm::ParameterSet&, const cms::Ort::ONNXRuntime*);
  ~PNETLeptonProducer() override {}

  /* void setValue(const std::string var, float val) { */
  /*   if (positions_.find(var) != positions_.end()) */
  /*     values_[positions_[var]] = val; */
  /* } */

  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet& cfg);
  static void globalEndJob(const cms::Ort::ONNXRuntime* cache);

  static edm::ParameterSetDescription getDescription();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override{};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {};

  ///to be implemented in derived classes, filling values for additional variables
  virtual void readAdditionalCollections(edm::Event&, const edm::EventSetup&) {}
  virtual void fillAdditionalVariables(const T&) {}

  edm::EDGetTokenT<pat::LeptonTagInfoCollection<T>> src_;
  edm::EDGetTokenT<std::vector<T>> leps_;
  std::vector<std::string> flav_names_;
  std::string name_;
  std::vector<std::string> input_names_;            // names of each input group - the ordering is important!
  std::vector<std::vector<int64_t>> input_shapes_;  // shapes of each input group (-1 for dynamic axis)
  std::vector<unsigned> input_sizes_;               // total length of each input vector
  std::unordered_map<std::string, btagbtvdeep::PreprocessParams>
      prep_info_map_;  // preprocessing info for each input group

  cms::Ort::FloatArrays data_;
  bool debug_ = false;

  void make_inputs(const pat::LeptonTagInfo<T>&);
};

#endif
