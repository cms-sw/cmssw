#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"
#include "FWCore/Utilities/interface/transform.h"
#include "PhysicsTools/NanoAOD/interface/GenWeightCounters.h"
#include <optional>
#include <iostream>
#include <tinyxml2.h>

namespace {
  typedef std::vector<gen::SharedWeightGroupData> WeightGroupDataContainer;
  typedef std::array<std::vector<gen::SharedWeightGroupData>, 2> WeightGroupsToStore;
}  // namespace
using CounterMap = genCounter::CounterMap;
using Counter = genCounter::Counter;

class LHEWeightsTableProducer : public edm::global::EDProducer<edm::LuminosityBlockCache<WeightGroupsToStore>,
                                                               edm::StreamCache<CounterMap>,
                                                               edm::RunSummaryCache<CounterMap>,
                                                               edm::EndRunProducer> {
public:
  LHEWeightsTableProducer(edm::ParameterSet const& params);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  //func changed//sroychow
  void addWeightGroupToTable(std::map<gen::WeightType, std::vector<double>>& lheWeightTables,
                             std::map<gen::WeightType, std::vector<int>>& weightVecsizes,
                             std::map<gen::WeightType, std::string>& weightlabels,
                             const char* typeName,
                             const WeightGroupDataContainer& weightInfos,
                             WeightsContainer& allWeights,
                             Counter& counter,
                             double genWeight) const;
  WeightGroupDataContainer weightDataPerType(edm::Handle<GenWeightInfoProduct>& weightsHandle,
                                             gen::WeightType weightType,
                                             int& maxStore) const;

  std::vector<double> orderedScaleWeights(const std::vector<double>& scaleWeights,
                                          const gen::ScaleWeightGroupInfo& scaleGroup) const;

  std::vector<double> preferredPSweights(const std::vector<double>& psWeights,
                                         const gen::PartonShowerWeightGroupInfo& pswV) const;

  //Lumiblock
  std::shared_ptr<WeightGroupsToStore> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                  edm::EventSetup const&) const override {
    // Set equal to the max number of groups
    // subtrack 1 for each weight group you find
    bool foundLheWeights = false;
    edm::Handle<GenWeightInfoProduct> lheWeightInfoHandle;
    for (auto& token : lheWeightInfoTokens_) {
      iLumi.getByToken(token, lheWeightInfoHandle);
      if (lheWeightInfoHandle.isValid()) {
        foundLheWeights = true;
        break;
      }
    }
    edm::Handle<GenWeightInfoProduct> genWeightInfoHandle;
    iLumi.getByToken(genWeightInfoToken_, genWeightInfoHandle);

    std::unordered_map<gen::WeightType, int> storePerType;
    for (size_t i = 0; i < weightgroups_.size(); i++)
      storePerType[weightgroups_.at(i)] = maxGroupsPerType_.at(i);

    WeightGroupsToStore weightsToStore;
    for (auto weightType : gen::allWeightTypes) {
      if (foundLheWeights) {
        auto lheWeights = weightDataPerType(lheWeightInfoHandle, weightType, storePerType[weightType]);
        for (auto& w : lheWeights)
          weightsToStore.at(inLHE).push_back({w.index, std::move(w.group)});
      }
      auto genWeights = weightDataPerType(genWeightInfoHandle, weightType, storePerType[weightType]);
      for (auto& w : genWeights)
        weightsToStore.at(inGen).push_back({w.index, std::move(w.group)});
    }
    return std::make_shared<WeightGroupsToStore>(weightsToStore);
  }

  // nothing to do here
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {}
  // create an empty counter
  std::unique_ptr<CounterMap> beginStream(edm::StreamID) const override { return std::make_unique<CounterMap>(); }
  // inizialize to zero at begin run
  void streamBeginRun(edm::StreamID id, edm::Run const&, edm::EventSetup const&) const override {
    streamCache(id)->clear();
  }

  void streamBeginLuminosityBlock(edm::StreamID id,
                                  edm::LuminosityBlock const& lumiBlock,
                                  edm::EventSetup const& eventSetup) const override {
    auto counterMap = streamCache(id);
    edm::Handle<GenLumiInfoHeader> genLumiInfoHead;
    lumiBlock.getByToken(genLumiInfoHeadTag_, genLumiInfoHead);
    if (!genLumiInfoHead.isValid())
      edm::LogWarning("LHETablesProducer")
          << "No GenLumiInfoHeader product found, will not fill generator model string.\n";
    counterMap->setLabel(genLumiInfoHead.isValid() ? genLumiInfoHead->configDescription() : "");
    std::string label = genLumiInfoHead.isValid() ? counterMap->getLabel() : "NULL";
  }
  // create an empty counter
  std::shared_ptr<CounterMap> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
    return std::make_shared<CounterMap>();
  }
  // add this stream to the summary
  void streamEndRunSummary(edm::StreamID id,
                           edm::Run const&,
                           edm::EventSetup const&,
                           CounterMap* runCounterMap) const override;
  // nothing to do per se
  void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, CounterMap* runCounterMap) const override {}
  // write the total to the run
  void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const& es, CounterMap const* runCounterMap) const override;
  // nothing to do here
  //void globalEndRun(edm::Run const& iRun, edm::EventSetup const& es, CounterMap* runCounterMap) const override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  const std::vector<edm::EDGetTokenT<GenWeightProduct>> lheWeightTokens_;
  const std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> lheWeightInfoTokens_;
  const edm::EDGetTokenT<GenWeightProduct> genWeightToken_;
  const edm::EDGetTokenT<GenWeightInfoProduct> genWeightInfoToken_;
  const edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;
  const edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoHeadTag_;
  const std::vector<gen::WeightType> weightgroups_;
  const std::vector<int> maxGroupsPerType_;
  const std::vector<int> pdfIds_;
  const std::unordered_map<gen::WeightType, std::string> weightTypeNames_ = {
      {gen::WeightType::kScaleWeights, "LHEScaleWeight"},
      {gen::WeightType::kPdfWeights, "LHEPdfWeight"},
      {gen::WeightType::kMEParamWeights, "MEParamWeight"},
      {gen::WeightType::kPartonShowerWeights, "PSWeight"},
      {gen::WeightType::kUnknownWeights, "UnknownWeight"},
  };
  //std::unordered_map<std::string, int> weightGroupIndices_;
  int lheWeightPrecision_;
  bool storeAllPSweights_;

  enum { inLHE, inGen };
};
//put back if needed; till now not used
LHEWeightsTableProducer::LHEWeightsTableProducer(edm::ParameterSet const& params)
    : lheWeightTokens_(
          edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
                                [this](const edm::InputTag& tag) { return mayConsume<GenWeightProduct>(tag); })),
      lheWeightInfoTokens_(edm::vector_transform(
          params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
          [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
      genWeightToken_(consumes<GenWeightProduct>(params.getParameter<edm::InputTag>("genWeights"))),
      genWeightInfoToken_(
          consumes<GenWeightInfoProduct, edm::InLumi>(params.getParameter<edm::InputTag>("genWeights"))),
      genEventInfoToken_(consumes<GenEventInfoProduct>(params.getParameter<edm::InputTag>("genEvent"))),
      genLumiInfoHeadTag_(
          mayConsume<GenLumiInfoHeader, edm::InLumi>(params.getParameter<edm::InputTag>("genLumiInfoHeader"))),
      weightgroups_(edm::vector_transform(params.getParameter<std::vector<std::string>>("weightgroups"),
                                          [](auto& c) { return gen::WeightType(c.at(0)); })),
      maxGroupsPerType_(params.getParameter<std::vector<int>>("maxGroupsPerType")),
      pdfIds_(params.getUntrackedParameter<std::vector<int>>("pdfIds", {})),
      lheWeightPrecision_(params.getParameter<int32_t>("lheWeightPrecision")),
      storeAllPSweights_(params.getParameter<bool>("storeAllPSweights")) {
  if (weightgroups_.size() != maxGroupsPerType_.size())
    throw std::invalid_argument("Inputs 'weightgroups' and 'weightgroupNums' must have equal size");
  for (auto& wg : weightTypeNames_) {
    produces<nanoaod::FlatTable>(wg.second);
    produces<nanoaod::FlatTable>(wg.second + "sizes");
  }
  produces<nanoaod::FlatTable>("GENWeight");
  produces<nanoaod::MergeableCounterTable, edm::Transition::EndRun>();
  produces<std::string>("genModel");
}

void LHEWeightsTableProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  //access counter for weight sums
  Counter& counter = *streamCache(id)->get();
  edm::Handle<GenWeightProduct> lheWeightHandle;
  bool foundLheWeights = false;
  for (auto& token : lheWeightTokens_) {
    iEvent.getByToken(token, lheWeightHandle);
    if (lheWeightHandle.isValid()) {
      foundLheWeights = true;
      break;
    }
  }
  //Taken from genweight producer //Added sroychow
  // generator information (always available)
  auto const& genInfo = iEvent.get(genEventInfoToken_);
  const double genWeight = genInfo.weight();
  // table for gen info, always available
  auto outGeninfo = std::make_unique<nanoaod::FlatTable>(1, "genWeight", true);
  outGeninfo->setDoc("generator weight");
  outGeninfo->addColumnValue<float>("", genInfo.weight(), "generator weight");
  iEvent.put(std::move(outGeninfo), "GENWeight");
  //this will take care of sum of genWeights
  counter.incGenOnly(genWeight);

  std::string& model_label = streamCache(id)->getLabel();
  auto outM = std::make_unique<std::string>((!model_label.empty()) ? std::string("GenModel_") + model_label : "");
  iEvent.put(std::move(outM), "genModel");

  WeightsContainer lheWeights;
  if (foundLheWeights) {
    const GenWeightProduct* lheWeightProduct = lheWeightHandle.product();
    lheWeights = lheWeightProduct->weights();
  }

  edm::Handle<GenWeightProduct> genWeightHandle;
  iEvent.getByToken(genWeightToken_, genWeightHandle);
  const GenWeightProduct* genWeightProduct = genWeightHandle.product();
  WeightsContainer genWeights = genWeightProduct->weights();

  auto const& weightInfos = *luminosityBlockCache(iEvent.getLuminosityBlock().index());

  //create a container with dummy weight vector
  std::map<gen::WeightType, std::vector<double>> lheWeightTables;
  std::map<gen::WeightType, std::vector<int>> weightVecsizes;
  std::map<gen::WeightType, std::string> weightlabels;
  for (auto& wg : weightTypeNames_) {
    lheWeightTables.insert(std::make_pair(wg.first, std::vector<double>()));
    weightVecsizes.insert(std::make_pair(wg.first, std::vector<int>()));
    weightlabels.insert(std::make_pair(wg.first, ""));
  }
  if (foundLheWeights) {
    addWeightGroupToTable(
        lheWeightTables, weightVecsizes, weightlabels, "LHE", weightInfos.at(inLHE), lheWeights, counter, genWeight);
  }

  addWeightGroupToTable(
      lheWeightTables, weightVecsizes, weightlabels, "Gen", weightInfos.at(inGen), genWeights, counter, genWeight);

  for (auto& wg : weightTypeNames_) {
    std::string wname = wg.second;
    auto& weightVec = lheWeightTables[wg.first];
    counter.incLHE(genWeight, weightVec, wname);
    auto outTable = std::make_unique<nanoaod::FlatTable>(weightVec.size(), wname, false);
    outTable->addColumn<float>("", weightVec, weightlabels[wg.first], lheWeightPrecision_);

    //now add the vector containing the sizes of alt sets
    auto outTableSizes =
        std::make_unique<nanoaod::FlatTable>(weightVecsizes[wg.first].size(), wname + "_AltSetSizes", false);
    outTableSizes->addColumn<float>(
        "", weightVecsizes[wg.first], "Sizes of weight arrays for weight type:" + wname, lheWeightPrecision_);
    iEvent.put(std::move(outTable), wname);
    iEvent.put(std::move(outTableSizes), wname + "sizes");
  }
}

/*

*/
void LHEWeightsTableProducer::addWeightGroupToTable(std::map<gen::WeightType, std::vector<double>>& lheWeightTables,
                                                    std::map<gen::WeightType, std::vector<int>>& weightVecsizes,
                                                    std::map<gen::WeightType, std::string>& weightlabels,
                                                    const char* typeName,
                                                    const WeightGroupDataContainer& weightInfos,
                                                    WeightsContainer& allWeights,
                                                    Counter& counter,
                                                    double genWeight) const {
  std::unordered_map<gen::WeightType, int> typeCount = {};
  for (auto& type : gen::allWeightTypes)
    typeCount[type] = 0;

  for (const auto& groupInfo : weightInfos) {
    //std::string entryName = typeName;
    gen::WeightType weightType = groupInfo.group->weightType();
    std::string name = weightTypeNames_.at(weightType);
    std::string label = "[" + std::to_string(typeCount[weightType]) + "] " + groupInfo.group->description();
    label.append("[");
    label.append(std::to_string(lheWeightTables[weightType].size()));  //to append the start index of this set
    label.append("]; ");
    auto& weights = allWeights.at(groupInfo.index);
    if (weightType == gen::WeightType::kScaleWeights) {
      if (groupInfo.group->isWellFormed()) {
        const auto scaleGroup = *static_cast<const gen::ScaleWeightGroupInfo*>(groupInfo.group.get());
        weights = orderedScaleWeights(weights, scaleGroup);
        label.append(
            "[1] is mur=0.5 muf=1; [2] is mur=0.5 muf=2; [3] is mur=1 muf=0.5 ;"
            " [4] is mur=1 muf=1; [5] is mur=1 muf=2; [6] is mur=2 muf=0.5;"
            " [7] is mur=2 muf=1 ; [8] is mur=2 muf=2)");
      } else {
        size_t nstore = std::min<size_t>(gen::ScaleWeightGroupInfo::MIN_SCALE_VARIATIONS, weights.size());
        weights = std::vector(weights.begin(), weights.begin() + nstore);
        label.append("WARNING: Unexpected format found. Contains first " + std::to_string(nstore) +
                     " elements of weights vector, unordered");
      }
    } else if (!storeAllPSweights_ && weightType == gen::WeightType::kPartonShowerWeights) { // && groupInfo.group->isWellFormed()
      const auto psGroup = *static_cast<const gen::PartonShowerWeightGroupInfo*>(groupInfo.group.get());
      weights = preferredPSweights(weights, psGroup);
      label.append(
          "PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is "
          "ISR=1 FSR=2");
    }
    //else
    //  label.append(groupInfo.group->description());
    lheWeightTables[weightType].insert(lheWeightTables[weightType].end(), weights.begin(), weights.end());
    weightVecsizes[weightType].emplace_back(weights.size());

    if (weightlabels[weightType].empty())
      weightlabels[weightType].append("[idx in AltSetSizes array] Name [start idx in weight array];\n");

    weightlabels[weightType].append(label);
    typeCount[weightType]++;
  }
}

WeightGroupDataContainer LHEWeightsTableProducer::weightDataPerType(edm::Handle<GenWeightInfoProduct>& weightsHandle,
                                                                    gen::WeightType weightType,
                                                                    int& maxStore) const {
  std::vector<gen::WeightGroupData> allgroups;
  if (weightType == gen::WeightType::kPdfWeights && !pdfIds_.empty()) {
    allgroups = weightsHandle->pdfGroupsWithIndicesByLHAIDs(pdfIds_);
  } else
    allgroups = weightsHandle->weightGroupsAndIndicesByType(weightType);

  int toStore = maxStore;
  if (maxStore < 0 || static_cast<int>(allgroups.size()) <= maxStore) {
    // Modify size in case one type of weight is present in multiple products
    maxStore -= allgroups.size();
    toStore = allgroups.size();
  }

  WeightGroupDataContainer out;
  for (int i = 0; i < toStore; i++) {
    auto& group = allgroups.at(i);
    gen::SharedWeightGroupData temp = {group.index, std::move(group.group)};
    out.push_back(temp);
  }
  return out;
}

std::vector<double> LHEWeightsTableProducer::orderedScaleWeights(const std::vector<double>& scaleWeights,
                                                                 const gen::ScaleWeightGroupInfo& scaleGroup) const {
  std::vector<double> weights;
  weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF1Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF2Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR1muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.centralIndex()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR1muF2Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF1Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF2Index()));

  return weights;
}

std::vector<double> LHEWeightsTableProducer::preferredPSweights(const std::vector<double>& psWeights,
                                                                const gen::PartonShowerWeightGroupInfo& pswV) const {
  std::vector<double> psTosave;

  double baseline = psWeights.at(pswV.weightIndexFromLabel("Baseline"));
  psTosave.emplace_back(psWeights.at(pswV.variationIndex(true, true, gen::PSVarType::def)) / baseline);
  psTosave.emplace_back(psWeights.at(pswV.variationIndex(false, true, gen::PSVarType::def)) / baseline);
  psTosave.emplace_back(psWeights.at(pswV.variationIndex(true, false, gen::PSVarType::def)) / baseline);
  psTosave.emplace_back(psWeights.at(pswV.variationIndex(false, false, gen::PSVarType::def)) / baseline);
  return psTosave;
}

void LHEWeightsTableProducer::streamEndRunSummary(edm::StreamID id,
                                                  edm::Run const&,
                                                  edm::EventSetup const&,
                                                  CounterMap* runCounterMap) const {
  //this takes care for mergeing all the weight sums
  runCounterMap->mergeSumMap(*streamCache(id));
}

void LHEWeightsTableProducer::globalEndRunProduce(edm::Run& iRun,
                                                  edm::EventSetup const&,
                                                  CounterMap const* runCounterMap) const {
  auto out = std::make_unique<nanoaod::MergeableCounterTable>();

  for (auto x : runCounterMap->countermap) {
    auto& runCounter = x.second;
    std::string label = std::string("_") + x.first;
    std::string doclabel = (!x.first.empty()) ? (std::string(", for model label ") + x.first) : "";

    out->addInt("genEventCount" + label, "event count" + doclabel, runCounter.num_);
    out->addFloat("genEventSumw" + label, "sum of gen weights" + doclabel, runCounter.sumw_);
    out->addFloat("genEventSumw2" + label, "sum of gen (weight^2)" + doclabel, runCounter.sumw2_);

    double norm = runCounter.sumw_ ? 1.0 / runCounter.sumw_ : 1;
    //Sum from map
    for (auto& sumw : runCounter.weightSumMap_) {
      //Normalize with genEventSumw
      for (auto& val : sumw.second)
        val *= norm;
      out->addVFloat(sumw.first + "Sumw" + label,
                     "Sum of genEventWeight *" + sumw.first + "[i]/genEventSumw" + doclabel,
                     sumw.second);
    }
  }
  iRun.put(std::move(out));
}
void LHEWeightsTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("lheWeights");
  desc.add<std::vector<edm::InputTag>>("lheInfo", std::vector<edm::InputTag>{{"externalLHEProducer"}, {"source"}})
      ->setComment("tag(s) for the LHE information (LHEEventProduct and LHERunInfoProduct)");
  //desc.add<std::vector<edm::InputTag>>("genWeights");
  desc.add<edm::InputTag>("genWeights");
  desc.add<edm::InputTag>("genEvent", edm::InputTag("generator"))
      ->setComment("tag for the GenEventInfoProduct, to get the main weight");
  desc.add<edm::InputTag>("genLumiInfoHeader", edm::InputTag("generator"))
      ->setComment("tag for the GenLumiInfoProduct, to get the model string");
  desc.add<std::vector<std::string>>("weightgroups");
  desc.add<std::vector<int>>("maxGroupsPerType");
  desc.addOptionalUntracked<std::vector<int>>("pdfIds");
  desc.add<int32_t>("lheWeightPrecision", -1)->setComment("Number of bits in the mantissa for LHE weights");
  desc.add<bool>("storeAllPSweights", false)->setComment("True:stores all 45 PS weights; False:saves preferred 4");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHEWeightsTableProducer);
