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
#include "FWCore/Utilities/interface/transform.h"
#include <boost/algorithm/string.hpp>

class GenWeightProductProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer> {
public:
  explicit GenWeightProductProducer(const edm::ParameterSet& iConfig);
  ~GenWeightProductProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
  gen::GenWeightHelper weightHelper_;
  const edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoToken_;
  const edm::EDGetTokenT<GenEventInfoProduct> genEventToken_;
  std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> weightInfoTokens_;
  const bool debug_;
  bool foundWeightProduct_ = false;
  edm::EDPutTokenT<GenWeightInfoProduct> groupToken_;
};

//
// constructors and destructor
//
GenWeightProductProducer::GenWeightProductProducer(const edm::ParameterSet& iConfig)
    : genLumiInfoToken_(consumes<GenLumiInfoHeader, edm::InLumi>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      genEventToken_(consumes<GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genInfo"))),
      weightInfoTokens_(edm::vector_transform(iConfig.getParameter<std::vector<std::string>>("weightProductLabels"), 
          [this](const std::string& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)),
      groupToken_(produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>()) {
  weightHelper_.setDebug(debug_);
  produces<GenWeightProduct>();
  produces<GenWeightInfoProduct, edm::Transition::BeginLuminosityBlock>();
  weightHelper_.setGuessPSWeightIdx(iConfig.getUntrackedParameter<bool>("guessPSWeightIdx", false));
  weightHelper_.setfillEmptyIfWeightFails(iConfig.getUntrackedParameter<bool>("fillEmptyIfWeightFails", false));
}

GenWeightProductProducer::~GenWeightProductProducer() {}

// ------------ method called to produce the data  ------------
void GenWeightProductProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // In case there is already a product in the file when this is scheduled
  // (leave the list of weightproducts empty if you always want to produce a new product)
  if (foundWeightProduct_)
    return;

  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(genEventToken_, genEventInfo);

  float centralWeight = !genEventInfo->weights().empty() ? genEventInfo->weights().at(0) : 1.;
  auto weightProduct = weightHelper_.weightProduct(genEventInfo->weights(), centralWeight);
  iEvent.put(std::move(weightProduct));
}

void GenWeightProductProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& iLumi, edm::EventSetup const& iSetup) {
  edm::Handle<GenWeightInfoProduct> weightInfoHandle;

  for (auto& token : weightInfoTokens_) {
    iLumi.getByToken(token, weightInfoHandle);
    if (weightInfoHandle.isValid()) {
      foundWeightProduct_ = true;
      return;
    }
  }

  edm::Handle<GenLumiInfoHeader> genLumiInfoHandle;
  iLumi.getByToken(genLumiInfoToken_, genLumiInfoHandle);

  auto weightInfoProduct = std::make_unique<GenWeightInfoProduct>();
  if (genLumiInfoHandle.isValid()) {
    std::string label = genLumiInfoHandle->configDescription();
    boost::replace_all(label, "-", "_");
    weightHelper_.setModel(label);
    weightHelper_.parseWeightGroupsFromNames(genLumiInfoHandle->weightNames());
    // Always add an unassociated group, which generally will not be filled
    weightHelper_.addUnassociatedGroup();

    // Need to have separate copies of the groups in the helper class and in the product,
    // because the helper can still modify the data
    for (auto& weightGroup : weightHelper_.weightGroups()) {
      weightInfoProduct->addWeightGroupInfo(std::unique_ptr<gen::WeightGroupInfo>(weightGroup->clone()));
    }
  } else if (weightHelper_.fillEmptyIfWeightFails() && debug_) {
    std::cerr << "genLumiInfoHeader not found, but fillEmptyIfWeightFails is True. Will produce empty product!" << std::endl;
  } else {
    throw cms::Exception("GenWeightProductProducer")
        << "genLumiInfoHeader not found, code is exiting." << std::endl
        << "If this is expect and want to continue, set fillEmptyIfWeightFails to True";
  }
  iLumi.emplace(groupToken_, std::move(weightInfoProduct));
}

void GenWeightProductProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genInfo", edm::InputTag{"generator"})
      ->setComment("tag(s) for the GenLumiInfoHeader and GenEventInfoProduct");
  desc.add<std::vector<std::string>>("weightProductLabels", std::vector<std::string>{{""}})
      ->setComment("tag(s) to look for existing GenWeightProduct/GenWeightInfoProducts. "
        "If they are found, a new one won't be created. Leave this argument empty if you want to recreate new products regardless.");
  desc.addUntracked<bool>("debug", false)->setComment("Output debug info");
  desc.addUntracked<bool>("guessPSWeightIdx", false)->setComment("If not possible to parse text, guess the parton shower weight indices");
  desc.addUntracked<bool>("fillEmptyIfWeightFails", false)->setComment("Produce an empty product if parsing of header fails");
  descriptions.add("genWeights", desc);
}

DEFINE_FWK_MODULE(GenWeightProductProducer);
