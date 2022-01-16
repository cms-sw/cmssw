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
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"

class LHEWeightProductProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer, 
    edm::RunCache<gen::WeightGroupInfoContainer>> {
public:
  explicit LHEWeightProductProducer(const edm::ParameterSet& iConfig);
  ~LHEWeightProductProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& es) override;
  std::shared_ptr<gen::WeightGroupInfoContainer> globalBeginRun(edm::Run const& run, edm::EventSetup const& es) const;
  void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  gen::LHEWeightHelper weightHelper_;
  std::vector<std::string> lheLabels_;
  std::vector<edm::EDGetTokenT<LHEEventProduct>> lheEventTokens_;
  std::vector<edm::EDGetTokenT<LHERunInfoProduct>> lheRunInfoTokens_;
  std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> weightInfoTokens_;
  bool foundWeightProduct_ = false;
  bool hasLhe_ = true;
  edm::EDPutTokenT<GenWeightInfoProduct> groupPutToken_;
  GenWeightInfoProduct weightsInfo_;
};

// TODO: Accept a vector of strings (source, externalLHEProducer) exit if neither are found
LHEWeightProductProducer::LHEWeightProductProducer(const edm::ParameterSet& iConfig)
    : lheLabels_(iConfig.getParameter<std::vector<std::string>>("lheSourceLabels")),
      lheEventTokens_(edm::vector_transform(
          lheLabels_, [this](const std::string& tag) { return mayConsume<LHEEventProduct>(tag); })),
      lheRunInfoTokens_(edm::vector_transform(
          lheLabels_, [this](const std::string& tag) { return mayConsume<LHERunInfoProduct, edm::InRun>(tag); })),
      weightInfoTokens_(edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("weightProductLabels"), 
          [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
      groupPutToken_(produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>()) {
  produces<GenWeightProduct>();
  weightHelper_.setFailIfInvalidXML(iConfig.getUntrackedParameter<bool>("failIfInvalidXML", false));
  weightHelper_.setfillEmptyIfWeightFails(iConfig.getUntrackedParameter<bool>("fillEmptyIfWeightFails", false));
  weightHelper_.setDebug(iConfig.getUntrackedParameter<bool>("debug", false));
  weightHelper_.setGuessPSWeightIdx(iConfig.getUntrackedParameter<bool>("guessPSWeightIdx", false));
}

LHEWeightProductProducer::~LHEWeightProductProducer() {}

// ------------ method called to produce the data  ------------
void LHEWeightProductProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (foundWeightProduct_ || !hasLhe_)
    return;

  edm::Handle<LHEEventProduct> lheEventInfo;
  for (auto& token : lheEventTokens_) {
    iEvent.getByToken(token, lheEventInfo);
    if (lheEventInfo.isValid()) {
      break;
    }
  }

  auto weightProduct = weightHelper_.weightProduct(weightsInfo_, lheEventInfo->weights(), lheEventInfo->originalXWGTUP());
  iEvent.put(std::move(weightProduct));
}

std::shared_ptr<gen::WeightGroupInfoContainer> LHEWeightProductProducer::globalBeginRun(edm::Run const& run, edm::EventSetup const& es) const {
  edm::Handle<LHERunInfoProduct> lheRunInfoHandle;
  for (auto& label : lheLabels_) {
    run.getByLabel(label, lheRunInfoHandle);
    if (lheRunInfoHandle.isValid()) {
      break;
    }
  }
  if (!lheRunInfoHandle.isValid())
    return {};

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
    // TODO: Maybe make unassociated group optional
    groups = weightHelper_.parseWeights(headerWeightInfo.lines(), true);
  } catch (cms::Exception& e) {
    std::string error = e.what();
    error +=
        "\n   NOTE: if you want to attempt to process this sample anyway, set failIfInvalidXML = False "
        "in the configuration file\n. If you set this flag and the error persists, the issue "
        " is fatal and must be solved at the LHE/gridpack level.";
    throw cms::Exception("LHEWeightProductProducer") << error;
  }
  return std::make_shared<gen::WeightGroupInfoContainer>(std::move(groups));
}

void LHEWeightProductProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& es) {
  edm::Handle<GenWeightInfoProduct> weightInfoHandle;

  for (auto& token : weightInfoTokens_) {
    lumi.getByToken(token, weightInfoHandle);
    if (weightInfoHandle.isValid()) {
      foundWeightProduct_ = true;
      return;
    }
  }

  if (!hasLhe_)
    return;

  auto weightInfoProduct = std::make_unique<GenWeightInfoProduct>(*runCache(lumi.getRun().index()));
  weightsInfo_ = *weightInfoProduct;

  lumi.emplace(groupPutToken_, std::move(weightInfoProduct));
}

void LHEWeightProductProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("lheSourceLabels", std::vector<std::string>{{"externalLHEProducer"}, {"source"}})
      ->setComment("tag(s) to look for LHERunInfoProduct/LHEEventProduct"
        "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new products regardless.");
  desc.add<std::vector<edm::InputTag>>("weightProductLabels", std::vector<edm::InputTag>{{""}})
      ->setComment("tag(s) to look for existing GenWeightProduct/GenWeightInfoProducts. "
        "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new products regardless.");
  desc.addUntracked<bool>("debug", false)->setComment("Output debug info");
  desc.addUntracked<bool>("failIfInvalidXML", true)->setComment("Throw exception if XML header is invalid (rather than trying to recover and parse anyway)");
  desc.addUntracked<bool>("fillEmptyIfWeightFails", false)->setComment("Produce an empty product if parsing of header fails");
  descriptions.add("lheWeights", desc);
}


DEFINE_FWK_MODULE(LHEWeightProductProducer);
