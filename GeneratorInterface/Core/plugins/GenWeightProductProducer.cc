#include <cstdio>
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
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
#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"

#include "GeneratorInterface/Core/interface/GenWeightHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"
#include <boost/algorithm/string.hpp>

struct GenWeightInfoProdData {
  bool makeNewProduct;
  GenWeightInfoProduct product;
};

class GenWeightProductProducer : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer,
                                                                edm::LuminosityBlockCache<GenWeightInfoProdData>> {
public:
  explicit GenWeightProductProducer(const edm::ParameterSet& iConfig);
  ~GenWeightProductProducer() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) const override;
  std::shared_ptr<GenWeightInfoProdData> globalBeginLuminosityBlock(const edm::LuminosityBlock& lb,
                                                                    edm::EventSetup const& c) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock& lb, edm::EventSetup const& c) const override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  gen::GenWeightHelper weightHelper_;
  const edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoToken_;
  const edm::EDGetTokenT<GenEventInfoProduct> genEventToken_;
  std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> weightInfoTokens_;
  const bool debug_;
  edm::EDPutTokenT<GenWeightInfoProduct> groupPutToken_;
  GenWeightInfoProduct weightsInfo_;
  bool allowUnassociated_;
};

GenWeightProductProducer::GenWeightProductProducer(const edm::ParameterSet& iConfig)
    : genLumiInfoToken_(consumes<GenLumiInfoHeader, edm::InLumi>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      genEventToken_(consumes<GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      weightInfoTokens_(edm::vector_transform(
          iConfig.getParameter<std::vector<std::string>>("weightProductLabels"),
          [this](const std::string& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)),
      groupPutToken_(produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>()),
      allowUnassociated_(iConfig.getUntrackedParameter<bool>("allowUnassociatedWeights", false)) {
  weightHelper_.setDebug(debug_);
  produces<GenWeightProduct>();
  produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>();
  weightHelper_.setGuessPSWeightIdx(iConfig.getUntrackedParameter<bool>("guessPSWeightIdx", false));
}

GenWeightProductProducer::~GenWeightProductProducer() {}

void GenWeightProductProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // In case there is already a product in the file when this is scheduled
  // (leave the list of weightproducts empty if you always want to produce a new product)
  const auto& productInfo = *luminosityBlockCache(iEvent.getLuminosityBlock().index());
  if (!productInfo.makeNewProduct)
    return;

  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(genEventToken_, genEventInfo);

  float centralWeight = !genEventInfo->weights().empty() ? genEventInfo->weights().at(0) : 1.;
  auto weightProduct = weightHelper_.weightProduct(productInfo.product, genEventInfo->weights(), centralWeight);
  iEvent.put(std::move(weightProduct));
}

std::shared_ptr<GenWeightInfoProdData> GenWeightProductProducer::globalBeginLuminosityBlock(
    const edm::LuminosityBlock& iLumi, edm::EventSetup const& iSetup) const {
  GenWeightInfoProdData productInfo;
  productInfo.makeNewProduct = true;

  edm::Handle<GenWeightInfoProduct> weightInfoHandle;
  for (auto& token : weightInfoTokens_) {
    iLumi.getByToken(token, weightInfoHandle);
    if (weightInfoHandle.isValid()) {
      productInfo.makeNewProduct = false;
      break;
    }
  }

  edm::Handle<GenLumiInfoHeader> genLumiInfoHandle;
  iLumi.getByToken(genLumiInfoToken_, genLumiInfoHandle);

  if (genLumiInfoHandle.isValid()) {
    auto weightGroups = weightHelper_.parseWeightGroupsFromNames(genLumiInfoHandle->weightNames(), allowUnassociated_);
    productInfo.product = GenWeightInfoProduct(weightGroups);
  }
  return std::make_shared<GenWeightInfoProdData>(productInfo);
}

void GenWeightProductProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                                                 edm::EventSetup const& iSetup) const {
  edm::Handle<GenWeightInfoProduct> weightInfoHandle;

  const auto& productInfo = *luminosityBlockCache(iLumi.index());
  if (!productInfo.makeNewProduct)
    return;

  auto weightInfoProduct = std::make_unique<GenWeightInfoProduct>(productInfo.product);
  iLumi.emplace(groupPutToken_, std::move(weightInfoProduct));
}

void GenWeightProductProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genInfo", edm::InputTag{"generator"})
      ->setComment("tag(s) for the GenLumiInfoHeader and GenEventInfoProduct");
  desc.add<std::vector<std::string>>("weightProductLabels", std::vector<std::string>{{""}})
      ->setComment(
          "tag(s) to look for existing GenWeightProduct/GenWeightInfoProducts. "
          "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new "
          "products regardless.");
  desc.addUntracked<bool>("debug", false)->setComment("Output debug info");
  desc.addUntracked<bool>("guessPSWeightIdx", false)
      ->setComment("If not possible to parse text, guess the parton shower weight indices");
  desc.addUntracked<bool>("allowUnassociatedWeights", false)
      ->setComment(
          "Handle weights found in the event that aren't advertised in the weight header (otherwise throw exception)");
  descriptions.add("genWeights", desc);
}

DEFINE_FWK_MODULE(GenWeightProductProducer);
