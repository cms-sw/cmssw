#include <cstdio>
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"

#include "GeneratorInterface/Core/interface/GenWeightHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include <boost/algorithm/string.hpp>

class GenWeightProductProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer> {
public:
  explicit GenWeightProductProducer(const edm::ParameterSet& iConfig);
  ~GenWeightProductProducer() override;

private:
  std::vector<std::string> weightNames_;
  gen::GenWeightHelper weightHelper_;
  edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventToken_;
  const edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoHeadTag_;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;
};

//
// constructors and destructor
//
GenWeightProductProducer::GenWeightProductProducer(const edm::ParameterSet& iConfig)
    : genLumiInfoToken_(consumes<GenLumiInfoHeader, edm::InLumi>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      genEventToken_(consumes<GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      genLumiInfoHeadTag_(
          mayConsume<GenLumiInfoHeader, edm::InLumi>(iConfig.getParameter<edm::InputTag>("genLumiInfoHeader"))) {
  weightHelper_.setDebug(iConfig.getUntrackedParameter<bool>("debug", false));
  produces<GenWeightProduct>();
  produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>();
  weightHelper_.setGuessPSWeightIdx(iConfig.getUntrackedParameter<bool>("guessPSWeightIdx", false));
}

GenWeightProductProducer::~GenWeightProductProducer() {}

// ------------ method called to produce the data  ------------
void GenWeightProductProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(genEventToken_, genEventInfo);

  float centralWeight = !genEventInfo->weights().empty() ? genEventInfo->weights().at(0) : 1.;
  auto weightProduct = weightHelper_.weightProduct(genEventInfo->weights(), centralWeight);
  iEvent.put(std::move(weightProduct));
}

void GenWeightProductProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& iLumi, edm::EventSetup const& iSetup) {
  edm::Handle<GenLumiInfoHeader> genLumiInfoHead;
  iLumi.getByToken(genLumiInfoHeadTag_, genLumiInfoHead);
  if (genLumiInfoHead.isValid()) {
    std::string label = genLumiInfoHead->configDescription();
    boost::replace_all(label, "-", "_");
    weightHelper_.setModel(label);
  }

  edm::Handle<GenLumiInfoHeader> genLumiInfoHandle;
  iLumi.getByToken(genLumiInfoToken_, genLumiInfoHandle);

  weightNames_ = genLumiInfoHandle->weightNames();
  weightHelper_.parseWeightGroupsFromNames(weightNames_);

  auto weightInfoProduct = std::make_unique<GenWeightInfoProduct>();
  if (weightHelper_.weightGroups().empty())
    weightHelper_.addUnassociatedGroup();

  for (auto& weightGroup : weightHelper_.weightGroups()) {
    weightInfoProduct->addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo>(weightGroup.clone()));
  }
  iLumi.put(std::move(weightInfoProduct));
}

DEFINE_FWK_MODULE(GenWeightProductProducer);
