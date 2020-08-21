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

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"

class LHEWeightProductProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer, edm::one::WatchRuns> {
public:
  explicit LHEWeightProductProducer(const edm::ParameterSet& iConfig);
  ~LHEWeightProductProducer() override;

private:
  gen::LHEWeightHelper weightHelper_;
  std::vector<std::string> lheLabels_;
  std::vector<edm::EDGetTokenT<LHEEventProduct>> lheEventTokens_;
  std::vector<edm::EDGetTokenT<LHERunInfoProduct>> lheRunInfoTokens_;
  std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> lheWeightInfoTokens_;
  bool foundWeightProduct_ = false;
  bool hasLhe_ = false;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& es) override;
  void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
  void endRun(edm::Run const& run, edm::EventSetup const& es) override;
};

// TODO: Accept a vector of strings (source, externalLHEProducer) exit if neither are found
LHEWeightProductProducer::LHEWeightProductProducer(const edm::ParameterSet& iConfig)
    : lheLabels_(iConfig.getParameter<std::vector<std::string>>("lheSourceLabels")),
      lheEventTokens_(edm::vector_transform(lheLabels_,
            [this](const std::string& tag) { return mayConsume<LHEEventProduct>(tag); })),
      lheRunInfoTokens_(edm::vector_transform(lheLabels_,
            [this](const std::string& tag) { return mayConsume<LHERunInfoProduct, edm::InRun>(tag); })),
      lheWeightInfoTokens_(edm::vector_transform(lheLabels_,
            [this](const std::string& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })) {
  produces<GenWeightProduct>();
  produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>();
  weightHelper_.setFailIfInvalidXML(iConfig.getUntrackedParameter<bool>("failIfInvalidXML", false));
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

  auto weightProduct = weightHelper_.weightProduct(lheEventInfo->weights(), lheEventInfo->originalXWGTUP());
  iEvent.put(std::move(weightProduct));
}

void LHEWeightProductProducer::beginRun(edm::Run const& run, edm::EventSetup const& es) {
  edm::Handle<LHERunInfoProduct> lheRunInfoHandle;
  for (auto& label : lheLabels_) {
    run.getByLabel(label, lheRunInfoHandle);
    if (lheRunInfoHandle.isValid()) {
        hasLhe_ = true;
        break;
    }
  }
  if (!hasLhe_)
      return;


  typedef std::vector<LHERunInfoProduct::Header>::const_iterator header_cit;
  LHERunInfoProduct::Header headerWeightInfo;
  for (header_cit iter = lheRunInfoHandle->headers_begin(); iter != lheRunInfoHandle->headers_end(); iter++) {
    if (iter->tag() == "initrwgt") {
      headerWeightInfo = *iter;
      break;
    }
  }

  weightHelper_.setHeaderLines(headerWeightInfo.lines());
}

void LHEWeightProductProducer::endRun(edm::Run const& run, edm::EventSetup const& es) {}

void LHEWeightProductProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& es) {
  edm::Handle<GenWeightInfoProduct> lheWeightInfoHandle;

  for (auto& token : lheWeightInfoTokens_) {
    lumi.getByToken(token, lheWeightInfoHandle);
    if (lheWeightInfoHandle.isValid()) {
      foundWeightProduct_ = true;
      return;
    }
  }

  if (!hasLhe_)
      return;

  weightHelper_.parseWeights();
  if (weightHelper_.weightGroups().size() == 0)
      weightHelper_.addUnassociatedGroup();

  auto weightInfoProduct = std::make_unique<GenWeightInfoProduct>();
  for (auto& weightGroup : weightHelper_.weightGroups()) {
    weightInfoProduct->addWeightGroupInfo(std::make_unique<gen::WeightGroupInfo>(*weightGroup.clone()));
  }
  lumi.put(std::move(weightInfoProduct));
}

DEFINE_FWK_MODULE(LHEWeightProductProducer);
