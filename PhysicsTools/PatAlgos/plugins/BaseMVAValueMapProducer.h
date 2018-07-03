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
#include "CommonTools/Utils/interface/TMVAZipReader.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

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
    isClassifier_(iConfig.getParameter<bool>("isClassifier"))  
  {
      edm::ParameterSet const & varsPSet = iConfig.getParameter<edm::ParameterSet>("variables");
      for (const std::string & vname : varsPSet.getParameterNamesForType<std::string>()) {
	  funcs_.emplace_back(std::pair<std::string,StringObjectFunction<T,true>>(vname,varsPSet.getParameter<std::string>(vname)));
      }

      values_.resize(variablesOrder_.size());
      size_t i=0;
      for(const auto & v : variablesOrder_){
	positions_[v]=i;
	reader_.AddVariable(v,(&values_.front())+i);
	i++;
      }
//      reader_.BookMVA(name_,iConfig.getParameter<edm::FileInPath>("weightFile").fullPath() );
      reco::details::loadTMVAWeights(&reader_, name_, iConfig.getParameter<edm::FileInPath>("weightFile").fullPath());
      produces<edm::ValueMap<float>>();

  }
  ~BaseMVAValueMapProducer() override {}

  void setValue(const std::string var,float val) {
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
  TMVA::Reader reader_;
  std::string name_;
  bool isClassifier_;

};

template <typename T>
void
BaseMVAValueMapProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);
  readAdditionalCollections(iEvent,iSetup);
  
  std::vector<float> mvaOut;
  mvaOut.reserve(src->size());
  for(auto const & o: *src) {
	for(auto const & p : funcs_ ){
		values_[positions_[p.first]]=p.second(o);
	}
        fillAdditionalVariables(o);
	mvaOut.push_back(isClassifier_ ? reader_.EvaluateMVA(name_) : reader_.EvaluateRegression(name_)[0]);
  }
  std::unique_ptr<edm::ValueMap<float>> mvaV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*mvaV);
  filler.insert(src,mvaOut.begin(),mvaOut.end());
  filler.fill();
  iEvent.put(std::move(mvaV));

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
