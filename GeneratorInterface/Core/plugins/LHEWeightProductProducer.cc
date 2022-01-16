#include <cstdio>
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

struct GenWeightInfoProdData {
  bool makeNewProduct;
  GenWeightInfoProduct product;
};

class LHEWeightProductProducer
    : public edm::global::EDProducer<edm::RunCache<GenWeightInfoProdData>, edm::BeginRunProducer> {
public:
  explicit LHEWeightProductProducer(const edm::ParameterSet& iConfig);
  ~LHEWeightProductProducer() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void globalBeginRunProduce(edm::Run& run, edm::EventSetup const& es) const override;
  void globalEndRunProduce(edm::Run& run, edm::EventSetup const& es) const {};
  std::shared_ptr<GenWeightInfoProdData> globalBeginRun(edm::Run const& run, edm::EventSetup const& es) const override;
  void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  gen::LHEWeightHelper weightHelper_;
  const std::vector<std::string> lheLabels_;
  std::vector<edm::EDGetTokenT<LHEEventProduct>> lheEventTokens_;
  std::vector<edm::EDGetTokenT<LHERunInfoProduct>> lheRunInfoTokens_;
  std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> weightInfoTokens_;
  edm::EDPutTokenT<GenWeightInfoProduct> groupPutToken_;
  bool allowUnassociated_;
};

LHEWeightProductProducer::LHEWeightProductProducer(const edm::ParameterSet& iConfig)
    : lheLabels_(iConfig.getParameter<std::vector<std::string>>("lheSourceLabels")),
      lheEventTokens_(edm::vector_transform(
          lheLabels_, [this](const std::string& tag) { return mayConsume<LHEEventProduct>(tag); })),
      lheRunInfoTokens_(edm::vector_transform(
          lheLabels_, [this](const std::string& tag) { return mayConsume<LHERunInfoProduct, edm::InRun>(tag); })),
      weightInfoTokens_(edm::vector_transform(
          iConfig.getParameter<std::vector<edm::InputTag>>("weightProductLabels"),
          [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InRun>(tag); })),
      groupPutToken_(produces<GenWeightInfoProduct, edm::Transition::BeginRun>()),
      allowUnassociated_(iConfig.getUntrackedParameter<bool>("allowUnassociatedWeights", false)) {
  produces<GenWeightProduct>();
  produces<LHEEventProduct>();
  weightHelper_.setFailIfInvalidXML(iConfig.getUntrackedParameter<bool>("failIfInvalidXML", false));
  weightHelper_.setDebug(iConfig.getUntrackedParameter<bool>("debug", false));
  weightHelper_.setGuessPSWeightIdx(iConfig.getUntrackedParameter<bool>("guessPSWeightIdx", false));
}

LHEWeightProductProducer::~LHEWeightProductProducer() {}

void LHEWeightProductProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& productInfo = *runCache(iEvent.getRun().index());

  if (!productInfo.makeNewProduct)
    return;

  edm::Handle<LHEEventProduct> lheEventInfo;
  for (auto& token : lheEventTokens_) {
    iEvent.getByToken(token, lheEventInfo);
    if (lheEventInfo.isValid()) {
      break;
    }
  }

  if (!lheEventInfo.isValid())
    return;

  auto weightProduct =
      weightHelper_.weightProduct(productInfo.product, lheEventInfo->weights(), lheEventInfo->originalXWGTUP());
  iEvent.put(std::move(weightProduct));

  auto newLheEventInfo = std::make_unique<LHEEventProduct>(*lheEventInfo);
  newLheEventInfo->clearWeights();
  if (!lheEventInfo->weights().empty())
    newLheEventInfo->addWeight(lheEventInfo->weights()[0]);
  iEvent.put(std::move(newLheEventInfo));
}

std::shared_ptr<GenWeightInfoProdData> LHEWeightProductProducer::globalBeginRun(edm::Run const& run,
                                                                                edm::EventSetup const& es) const {
  bool hasWeightProduct = false;
  edm::Handle<GenWeightInfoProduct> weightInfoHandle;
  for (auto& token : weightInfoTokens_) {
    run.getByToken(token, weightInfoHandle);
    if (weightInfoHandle.isValid()) {
      hasWeightProduct = true;
      break;
    }
  }
  GenWeightInfoProdData productInfo;
  productInfo.makeNewProduct = !hasWeightProduct;
  if (hasWeightProduct)
    return std::make_shared<GenWeightInfoProdData>(productInfo);

  edm::Handle<LHERunInfoProduct> lheRunInfoHandle;
  for (auto& label : lheLabels_) {
    run.getByLabel(label, lheRunInfoHandle);
    if (lheRunInfoHandle.isValid()) {
      break;
    }
  }
  if (!lheRunInfoHandle.isValid())
    return std::make_shared<GenWeightInfoProdData>(productInfo);

  typedef std::vector<LHERunInfoProduct::Header>::const_iterator header_cit;
  LHERunInfoProduct::Header headerWeightInfo;
  for (header_cit iter = lheRunInfoHandle->headers_begin(); iter != lheRunInfoHandle->headers_end(); iter++) {
    if (iter->tag() == "initrwgt") {
      headerWeightInfo = *iter;
      break;
    }
  }

  gen::WeightGroupInfoContainer groups;
  try {
    groups = weightHelper_.parseWeights(headerWeightInfo.lines(), allowUnassociated_);
  } catch (cms::Exception& e) {
    std::string error = e.what();
    error +=
        "\n   NOTE: if you want to attempt to process this sample anyway, set failIfInvalidXML = False "
        "in the configuration file (current value is ";
    error += weightHelper_.failIfInvalidXML() ? "True" : "False";
    error +=
        ")\n.If you set this flag and the error persists, the issue "
        " is fatal and must be solved at the LHE/gridpack level.";
    throw cms::Exception("LHEWeightProductProducer") << error;
  }
  // Need copy for the run cache
  productInfo.product = GenWeightInfoProduct(groups);
  return std::make_shared<GenWeightInfoProdData>(productInfo);
}

void LHEWeightProductProducer::globalBeginRunProduce(edm::Run& run, const edm::EventSetup& es) const {
  const auto& productInfo = *runCache(run.index());
  auto prod = std::make_unique<GenWeightInfoProduct>(productInfo.product);
  run.emplace(groupPutToken_, std::move(prod));
}

void LHEWeightProductProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("lheSourceLabels", std::vector<std::string>{{"externalLHEProducer"}, {"source"}})
      ->setComment(
          "tag(s) to look for LHERunInfoProduct/LHEEventProduct"
          "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new "
          "products regardless.");
  desc.add<std::vector<edm::InputTag>>("weightProductLabels", std::vector<edm::InputTag>{{""}})
      ->setComment(
          "tag(s) to look for existing GenWeightProduct/GenWeightInfoProducts. "
          "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new "
          "products regardless.");
  desc.addUntracked<bool>("debug", false)->setComment("Output debug info");
  desc.addUntracked<bool>("failIfInvalidXML", true)
      ->setComment("Throw exception if XML header is invalid (rather than trying to recover and parse anyway)");
  desc.addUntracked<bool>("allowUnassociatedWeights", false)
      ->setComment(
          "Handle weights found in the event that aren't advertised in the weight header (otherwise throw exception)");
  descriptions.add("lheWeights", desc);
}

DEFINE_FWK_MODULE(LHEWeightProductProducer);
