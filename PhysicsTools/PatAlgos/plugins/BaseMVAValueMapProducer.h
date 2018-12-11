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

#include <string>
//
// class declaration
//


template <typename T>
class BaseMVAValueMapProducer : public edm::stream::EDProducer<> {
  public:
  explicit BaseMVAValueMapProducer(const edm::ParameterSet &iConfig):
    src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
    variablesOrder_(iConfig.getParameter<std::vector<std::string>>("variablesOrder")),
    name_(iConfig.getParameter<std::string>("name")),
    backend_(iConfig.getParameter<std::string>("backend")),
    weightfilename_(iConfig.getParameter<edm::FileInPath>("weightFile").fullPath()),
    isClassifier_(iConfig.getParameter<bool>("isClassifier")),
    tmva_(backend_=="TMVA"),tf_(backend_=="TF")
    {
      if(tmva_) reader_=new TMVA::Reader();
      edm::ParameterSet const & varsPSet = iConfig.getParameter<edm::ParameterSet>("variables");
      for (const std::string & vname : varsPSet.getParameterNamesForType<std::string>()) {
	  funcs_.emplace_back(std::pair<std::string,StringObjectFunction<T,true>>(vname,varsPSet.getParameter<std::string>(vname)));
      }

      values_.resize(variablesOrder_.size());
      size_t i=0;
      for(const auto & v : variablesOrder_){
        positions_[v]=i;
        if(tmva_) reader_->AddVariable(v,(&values_.front())+i);
	    i++;
      }
//      reader_.BookMVA(name_,iConfig.getParameter<edm::FileInPath>("weightFile").fullPath() );
    if(tmva_)
    {
        reco::details::loadTMVAWeights(reader_, name_, weightfilename_);
    }else if(tf_)    {
        tensorflow::setLogging("3");
        graph_=tensorflow::loadGraphDef(weightfilename_);
        inputTensorName_=iConfig.getParameter<std::string>("inputTensorName");
        outputTensorName_=iConfig.getParameter<std::string>("outputTensorName");
        output_names_=iConfig.getParameter<std::vector<std::string>>("outputNames");
        for(const auto & s : iConfig.getParameter<std::vector<std::string>>("outputFormulas")) { output_formulas_.push_back(StringObjectFunction<std::vector<float>>(s));} 
        size_t nThreads = iConfig.getParameter<unsigned int>("nThreads");
        std::string singleThreadPool = iConfig.getParameter<std::string>("singleThreadPool");
        tensorflow::SessionOptions sessionOptions;
        tensorflow::setThreading(sessionOptions, nThreads, singleThreadPool);
        session_ = tensorflow::createSession(graph_, sessionOptions);

    } else  {
          throw cms::Exception("ConfigError") << "Only 'TF' and 'TMVA' backends are supported\n";
    }
   if(tmva_) produces<edm::ValueMap<float>>();
   else {
        for(const auto & n : output_names_){
            produces<edm::ValueMap<float>>(n);
        }
   }

  }
  ~BaseMVAValueMapProducer() override {}

  void setValue(const std::string var,float val) {
      if(positions_.find(var)!=positions_.end())
          values_[positions_[var]]=val;
  }
  
  static edm::ParameterSetDescription getDescription();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  void beginStream(edm::StreamID) override {};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {};

  ///to be implemented in derived classes, filling values for additional variables
  virtual void readAdditionalCollections(edm::Event&, const edm::EventSetup&)  {}
  virtual void fillAdditionalVariables(const T&)  {}


  edm::EDGetTokenT<edm::View<T>> src_;
  std::map<std::string,size_t> positions_;
  std::vector<std::pair<std::string,StringObjectFunction<T,true>>> funcs_;
  std::vector<std::string> variablesOrder_;
  std::vector<float> values_;
  TMVA::Reader * reader_;
  tensorflow::GraphDef* graph_;
  tensorflow::Session* session_;

  std::string name_;
  std::string backend_;
  std::string weightfilename_;
  bool isClassifier_;
  bool tmva_;
  bool tf_;
  std::string inputTensorName_;
  std::string outputTensorName_;
  std::vector<std::string> output_names_;
  std::vector<StringObjectFunction<std::vector<float>>> output_formulas_;
  
};

template <typename T>
void
BaseMVAValueMapProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);
  readAdditionalCollections(iEvent,iSetup);
  std::vector<std::vector<float>> mvaOut((tmva_)?1:output_names_.size());
  for( auto & v : mvaOut) v.reserve(src->size());

  for(auto const & o: *src) {
	for(auto const & p : funcs_ ){
        setValue(p.first,p.second(o));
	}
        fillAdditionalVariables(o);
    if(tmva_){
        mvaOut[0].push_back(isClassifier_ ? reader_->EvaluateMVA(name_) : reader_->EvaluateRegression(name_)[0]);
    }
    if(tf_){
        //currently support only one input sensor to reuse the TMVA like config 
        tensorflow::TensorShape input_size   {1,(long long int)positions_.size()} ;
        tensorflow::NamedTensorList input_tensors;
        input_tensors.resize(1); 
        input_tensors[0] = tensorflow::NamedTensor(inputTensorName_, tensorflow::Tensor(tensorflow::DT_FLOAT, input_size));
        for(size_t j =0; j < values_.size();j++) {
           input_tensors[0].second.matrix<float>()(0,j) = values_[j];
        }
       std::vector<tensorflow::Tensor> outputs;
       std::vector<std::string> names; names.push_back(outputTensorName_);
       tensorflow::run(session_, input_tensors, names, &outputs);
       std::vector<float> tmpOut;
       for(int k=0;k<outputs.at(0).matrix<float>().dimension(1);k++) tmpOut.push_back(outputs.at(0).matrix<float>()(0, k));
       for(size_t k=0;k<output_names_.size();k++) mvaOut[k].push_back(output_formulas_[k](tmpOut));

    }
    

  }
  size_t k=0;
  for( auto & m : mvaOut) { 
      std::unique_ptr<edm::ValueMap<float>> mvaV(new edm::ValueMap<float>());
      edm::ValueMap<float>::Filler filler(*mvaV);
      filler.insert(src,m.begin(),m.end());
      filler.fill();
      iEvent.put(std::move(mvaV),(tmva_)?"":output_names_[k]);
      k++;
  }

}

template <typename T>
edm::ParameterSetDescription
BaseMVAValueMapProducer<T>::getDescription(){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  desc.add<std::vector<std::string>>("variablesOrder")->setComment("ordered list of MVA input variable names");
  desc.add<std::string>("name")->setComment("output score variable name");
  desc.add<bool>("isClassifier")->setComment("is a classifier discriminator");
  edm::ParameterSetDescription variables;
  variables.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("variables", variables)->setComment("list of input variable definitions");
  desc.add<edm::FileInPath>("weightFile")->setComment("xml weight file");
  desc.add<std::string>("backend","TMVA")->setComment("TMVA or TF");
  desc.add<std::string>("inputTensorName","")->setComment("Name of tensorflow input tensor in the model");
  desc.add<std::string>("outputTensorName","")->setComment("Name of tensorflow output tensor in the model");
  desc.add<std::vector<std::string>>("outputNames",std::vector<std::string>())->setComment("Names of the output values to be used in the output valuemap");
  desc.add<std::vector<std::string>>("outputFormulas",std::vector<std::string>())->setComment("Formulas to be used to post process the output");
  desc.add<unsigned int>("nThreads",1)->setComment("number of threads");
  desc.add<std::string>("singleThreadPool", "no_threads");


  return desc;
}

template <typename T>
void
BaseMVAValueMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc = getDescription();
  std::string modname;
  if (typeid(T) == typeid(pat::Jet)) modname+="Jet";
  else if (typeid(T) == typeid(pat::Muon)) modname+="Muon";
  else if (typeid(T) == typeid(pat::Electron)) modname+="Ele";
  modname+="BaseMVAValueMapProducer";
  descriptions.add(modname,desc);
}



#endif
