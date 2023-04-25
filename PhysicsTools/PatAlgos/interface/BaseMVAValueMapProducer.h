#ifndef PhysicsTools_PatAlgos_BaseMVAValueMapProducer
#define PhysicsTools_PatAlgos_BaseMVAValueMapProducer

// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      BaseMVAValueMapProducer
//
/**\class BaseMVAValueMapProducer BaseMVAValueMapProducer.cc PhysicsTools/PatAlgos/plugins/BaseMVAValueMapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andre Rizzi
//         Created:  Mon, 07 Sep 2017 09:18:03 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "TMVA/Factory.h"
#include "TMVA/Reader.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <string>
//
// class declaration
//

class BaseMVACache {
public:
  BaseMVACache(const std::string& model_path, const std::string& backend, const bool disableONNXGraphOpt) {
    if (backend == "TF") {
      graph_.reset(tensorflow::loadGraphDef(model_path));
      tf_session_ = tensorflow::createSession(graph_.get());
    } else if (backend == "ONNX") {
      if (disableONNXGraphOpt) {
        Ort::SessionOptions sess_opts;
        sess_opts = cms::Ort::ONNXRuntime::defaultSessionOptions();
        sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        ort_ = std::make_unique<cms::Ort::ONNXRuntime>(model_path, &sess_opts);
      } else {
        ort_ = std::make_unique<cms::Ort::ONNXRuntime>(model_path);
      }
    }
  }
  ~BaseMVACache() { tensorflow::closeSession(tf_session_); }

  tensorflow::Session* getTFSession() const { return tf_session_; }
  const cms::Ort::ONNXRuntime& getONNXSession() const { return *ort_; }

private:
  std::shared_ptr<tensorflow::GraphDef> graph_;
  tensorflow::Session* tf_session_ = nullptr;
  std::unique_ptr<cms::Ort::ONNXRuntime> ort_;
};

template <typename T>
class BaseMVAValueMapProducer : public edm::stream::EDProducer<edm::GlobalCache<BaseMVACache>> {
public:
  explicit BaseMVAValueMapProducer(const edm::ParameterSet& iConfig, const BaseMVACache* cache)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        variablesOrder_(iConfig.getParameter<std::vector<std::string>>("variablesOrder")),
        name_(iConfig.getParameter<std::string>("name")),
        backend_(iConfig.getParameter<std::string>("backend")),
        weightfilename_(iConfig.getParameter<edm::FileInPath>("weightFile").fullPath()),
        isClassifier_(iConfig.getParameter<bool>("isClassifier")),
        tmva_(backend_ == "TMVA"),
        tf_(backend_ == "TF"),
        onnx_(backend_ == "ONNX"),
        batch_eval_(iConfig.getParameter<bool>("batch_eval")) {
    if (!(tmva_ || tf_ || onnx_)) {
      throw cms::Exception("ConfigError") << "Only 'TF', 'ONNX' and 'TMVA' backends are supported\n";
    }

    if (tmva_)
      reader_ = new TMVA::Reader();
    edm::ParameterSet const& varsPSet = iConfig.getParameter<edm::ParameterSet>("variables");
    for (const std::string& vname : varsPSet.getParameterNamesForType<std::string>()) {
      funcs_.emplace_back(
          std::pair<std::string, StringObjectFunction<T, true>>(vname, varsPSet.getParameter<std::string>(vname)));
    }

    values_.resize(variablesOrder_.size());
    size_t i = 0;
    for (const auto& v : variablesOrder_) {
      positions_[v] = i;
      if (tmva_)
        reader_->AddVariable(v, (&values_.front()) + i);
      i++;
    }
    //      reader_.BookMVA(name_,iConfig.getParameter<edm::FileInPath>("weightFile").fullPath() );
    if (tmva_) {
      reco::details::loadTMVAWeights(reader_, name_, weightfilename_);
    }
    if (tf_ || onnx_) {
      inputTensorName_ = iConfig.getParameter<std::string>("inputTensorName");
      outputTensorName_ = iConfig.getParameter<std::string>("outputTensorName");
      output_names_ = iConfig.getParameter<std::vector<std::string>>("outputNames");
      for (const auto& s : iConfig.getParameter<std::vector<std::string>>("outputFormulas")) {
        output_formulas_.push_back(StringObjectFunction<std::vector<float>>(s));
      }
    }

    if (tmva_)
      produces<edm::ValueMap<float>>();
    else {
      for (const auto& n : output_names_) {
        produces<edm::ValueMap<float>>(n);
      }
    }
  }
  ~BaseMVAValueMapProducer() override {}

  void setValue(const std::string var, float val) {
    if (positions_.find(var) != positions_.end())
      values_[positions_[var]] = val;
  }

  static std::unique_ptr<BaseMVACache> initializeGlobalCache(const edm::ParameterSet& cfg);
  static void globalEndJob(const BaseMVACache* cache);

  static edm::ParameterSetDescription getDescription();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override{};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override{};

  ///to be implemented in derived classes, filling values for additional variables
  virtual void readAdditionalCollections(edm::Event&, const edm::EventSetup&) {}
  virtual void fillAdditionalVariables(const T&) {}

  edm::EDGetTokenT<edm::View<T>> src_;
  std::map<std::string, size_t> positions_;
  std::vector<std::pair<std::string, StringObjectFunction<T, true>>> funcs_;
  std::vector<std::string> variablesOrder_;
  std::vector<float> values_;
  TMVA::Reader* reader_;

  std::string name_;
  std::string backend_;
  std::string weightfilename_;
  bool isClassifier_;
  bool tmva_;
  bool tf_;
  bool onnx_;
  bool batch_eval_;
  std::string inputTensorName_;
  std::string outputTensorName_;
  std::vector<std::string> output_names_;
  std::vector<StringObjectFunction<std::vector<float>>> output_formulas_;
};

template <typename T>
void BaseMVAValueMapProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);
  readAdditionalCollections(iEvent, iSetup);
  std::vector<std::vector<float>> mvaOut((tmva_) ? 1 : output_names_.size());
  for (auto& v : mvaOut)
    v.reserve(src->size());

  if (batch_eval_) {
    if (!src->empty()) {
      std::vector<float> data;
      data.reserve(src->size() * positions_.size());
      for (auto const& o : *src) {
        for (auto const& p : funcs_) {
          setValue(p.first, p.second(o));
        }
        fillAdditionalVariables(o);
        data.insert(data.end(), values_.begin(), values_.end());
      }

      std::vector<float> outputs;
      if (tf_) {
        tensorflow::TensorShape input_size{(long long int)src->size(), (long long int)positions_.size()};
        tensorflow::NamedTensorList input_tensors;
        input_tensors.resize(1);
        input_tensors[0] =
            tensorflow::NamedTensor(inputTensorName_, tensorflow::Tensor(tensorflow::DT_FLOAT, input_size));
        for (unsigned i = 0; i < data.size(); ++i) {
          input_tensors[0].second.flat<float>()(i) = data[i];
        }
        std::vector<tensorflow::Tensor> output_tensors;
        tensorflow::run(globalCache()->getTFSession(), input_tensors, {outputTensorName_}, &output_tensors);
        for (unsigned i = 0; i < output_tensors.at(0).NumElements(); ++i) {
          outputs.push_back(output_tensors.at(0).flat<float>()(i));
        }
      } else if (onnx_) {
        cms::Ort::FloatArrays inputs{data};
        outputs =
            globalCache()->getONNXSession().run({inputTensorName_}, inputs, {}, {outputTensorName_}, src->size())[0];
      }

      const unsigned outdim = outputs.size() / src->size();
      for (unsigned i = 0; i < src->size(); ++i) {
        std::vector<float> tmpOut(outputs.begin() + i * outdim, outputs.begin() + (i + 1) * outdim);
        for (size_t k = 0; k < output_names_.size(); k++) {
          mvaOut[k].push_back(output_formulas_[k](tmpOut));
        }
      }
    }
  } else {
    for (auto const& o : *src) {
      for (auto const& p : funcs_) {
        setValue(p.first, p.second(o));
      }
      fillAdditionalVariables(o);
      if (tmva_) {
        mvaOut[0].push_back(isClassifier_ ? reader_->EvaluateMVA(name_) : reader_->EvaluateRegression(name_)[0]);
      } else {
        std::vector<float> tmpOut;
        if (tf_) {
          //currently support only one input sensor to reuse the TMVA like config
          tensorflow::TensorShape input_size{1, (long long int)positions_.size()};
          tensorflow::NamedTensorList input_tensors;
          input_tensors.resize(1);
          input_tensors[0] =
              tensorflow::NamedTensor(inputTensorName_, tensorflow::Tensor(tensorflow::DT_FLOAT, input_size));
          for (size_t j = 0; j < values_.size(); j++) {
            input_tensors[0].second.matrix<float>()(0, j) = values_[j];
          }
          std::vector<tensorflow::Tensor> outputs;
          tensorflow::run(globalCache()->getTFSession(), input_tensors, {outputTensorName_}, &outputs);
          for (int k = 0; k < outputs.at(0).matrix<float>().dimension(1); k++)
            tmpOut.push_back(outputs.at(0).matrix<float>()(0, k));
        } else if (onnx_) {
          cms::Ort::FloatArrays inputs{values_};
          tmpOut = globalCache()->getONNXSession().run({inputTensorName_}, inputs, {}, {outputTensorName_})[0];
        }
        for (size_t k = 0; k < output_names_.size(); k++)
          mvaOut[k].push_back(output_formulas_[k](tmpOut));
      }
    }
  }

  size_t k = 0;
  for (auto& m : mvaOut) {
    std::unique_ptr<edm::ValueMap<float>> mvaV(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*mvaV);
    filler.insert(src, m.begin(), m.end());
    filler.fill();
    iEvent.put(std::move(mvaV), (tmva_) ? "" : output_names_[k]);
    k++;
  }
}

template <typename T>
std::unique_ptr<BaseMVACache> BaseMVAValueMapProducer<T>::initializeGlobalCache(const edm::ParameterSet& cfg) {
  return std::make_unique<BaseMVACache>(cfg.getParameter<edm::FileInPath>("weightFile").fullPath(),
                                        cfg.getParameter<std::string>("backend"),
                                        cfg.getParameter<bool>("disableONNXGraphOpt"));
}

template <typename T>
void BaseMVAValueMapProducer<T>::globalEndJob(const BaseMVACache* cache) {}

template <typename T>
edm::ParameterSetDescription BaseMVAValueMapProducer<T>::getDescription() {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<std::vector<std::string>>("variablesOrder")->setComment("ordered list of MVA input variable names");
  desc.add<std::string>("name")->setComment("output score variable name");
  desc.add<bool>("isClassifier")->setComment("is a classifier discriminator");
  edm::ParameterSetDescription variables;
  variables.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("variables", variables)->setComment("list of input variable definitions");
  desc.add<edm::FileInPath>("weightFile")->setComment("xml weight file");
  desc.add<std::string>("backend", "TMVA")->setComment("TMVA, TF or ONNX");
  desc.add<std::string>("inputTensorName", "")->setComment("Name of tensorflow input tensor in the model");
  desc.add<std::string>("outputTensorName", "")->setComment("Name of tensorflow output tensor in the model");
  desc.add<std::vector<std::string>>("outputNames", std::vector<std::string>())
      ->setComment("Names of the output values to be used in the output valuemap");
  desc.add<std::vector<std::string>>("outputFormulas", std::vector<std::string>())
      ->setComment("Formulas to be used to post process the output");
  desc.add<bool>("batch_eval", false)->setComment("Run inference in batch instead of per-object");
  desc.add<bool>("disableONNXGraphOpt", false)->setComment("Disable ONNX runtime graph optimization");

  return desc;
}

template <typename T>
void BaseMVAValueMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc = getDescription();
  std::string modname;
  if (typeid(T) == typeid(pat::Jet))
    modname += "Jet";
  else if (typeid(T) == typeid(pat::Muon))
    modname += "Muon";
  else if (typeid(T) == typeid(pat::Electron))
    modname += "Ele";
  modname += "BaseMVAValueMapProducer";
  descriptions.add(modname, desc);
}

#endif
