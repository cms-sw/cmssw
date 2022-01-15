#include <algorithm>
#include <iostream>

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "PhysicsTools/NanoAOD/interface/GenWeightCounters.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"

namespace {
  typedef std::vector<gen::WeightGroupData> WeightGroupDataContainer;
  typedef std::array<WeightGroupDataContainer, 2> WeightGroupsToStore;
}  // namespace
using CounterMap = genCounter::CounterMap;
using Counter = genCounter::Counter;

class GenWeightsTableProducer : public edm::global::EDProducer<edm::LuminosityBlockCache<WeightGroupsToStore>,
                                                               edm::StreamCache<CounterMap>,
                                                               edm::RunSummaryCache<CounterMap>,
                                                               edm::EndRunProducer> {
public:
  GenWeightsTableProducer(edm::ParameterSet const& params);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  void fillTableIgnoringGroups(std::vector<nanoaod::FlatTable>& weightTablevec,
                               const WeightGroupDataContainer& weightInfos,
                               WeightsContainer& allWeights,
                               size_t maxStore,
                               std::string tablename) const;
  void addWeightGroupToTable(std::vector<nanoaod::FlatTable>& weightTablevec,
                             const WeightGroupDataContainer& weightInfos,
                             WeightsContainer& allWeights) const;
  // Need to either pass the handle or a pointer to avoid a copy and conversion
  // to the base class
  WeightGroupDataContainer weightDataPerType(edm::Handle<GenWeightInfoProduct>& weightsInfoHandle,
                                             gen::WeightType weightType,
                                             size_t maxStore) const;

  WeightGroupsToStore groupsToStore(bool foundLheWeights,
                                    edm::Handle<GenWeightInfoProduct>& genWeightInfoHandle,
                                    edm::Handle<GenWeightInfoProduct>& lheWeightInfoHandle) const;

  std::pair<std::string, std::vector<double>> orderedScaleWeights(const std::vector<double>& scaleWeights,
                                                                  const gen::ScaleWeightGroupInfo& scaleGroup) const;

  std::pair<std::string, std::vector<double>> preferredPSweights(const std::vector<double>& psWeights,
                                                                 const gen::PartonShowerWeightGroupInfo& pswV) const;

  // Lumiblock
  std::shared_ptr<WeightGroupsToStore> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                  edm::EventSetup const&) const override {
    // Set equal to the max number of groups
    // subtrack 1 for each weight group you find
    bool foundLheWeights = false;
    edm::Handle<GenWeightInfoProduct> lheWeightInfoHandle;
    for (auto& token : lheWeightInfoTokens_) {
      iLumi.getRun().getByToken(token, lheWeightInfoHandle);
      if (lheWeightInfoHandle.isValid()) {
        foundLheWeights = true;
        break;
      }
    }

    edm::Handle<GenWeightInfoProduct> genWeightInfoHandle;
    for (auto& token : genWeightInfoTokens_) {
      iLumi.getByToken(token, genWeightInfoHandle);
      if (genWeightInfoHandle.isValid()) {
        break;
      }
    }
    auto tostore = groupsToStore(foundLheWeights, genWeightInfoHandle, lheWeightInfoHandle);
    return std::make_shared<WeightGroupsToStore>(tostore);
  }

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
      edm::LogWarning("LHETablesProducer") << "No GenLumiInfoHeader product found, will not fill generator "
                                              "model string.\n";
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
  // void globalEndRun(edm::Run const& iRun, edm::EventSetup const& es,
  // CounterMap* runCounterMap) const override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  const std::vector<edm::EDGetTokenT<GenWeightProduct>> lheWeightTokens_;
  const std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> lheWeightInfoTokens_;
  const std::vector<edm::EDGetTokenT<GenWeightProduct>> genWeightTokens_;
  const std::vector<edm::EDGetTokenT<GenWeightInfoProduct>> genWeightInfoTokens_;
  const edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;
  const edm::EDGetTokenT<GenLumiInfoHeader> genLumiInfoHeadTag_;
  const std::vector<gen::WeightType> weightgroups_;
  const std::vector<std::string> outputnames_;
  const std::vector<int> maxGroupsPerType_;
  const std::vector<int> pdfIds_;
  int lheWeightPrecision_;
  std::vector<gen::WeightType> unknownOnlyIfEmpty_;
  bool keepAllPSWeights_;
  bool ignoreLheGroups_;
  bool ignoreGenGroups_;
  int nStoreUngroupedLhe_;
  int nStoreUngroupedGen_;

  enum { inLHE, inGen };
};
GenWeightsTableProducer::GenWeightsTableProducer(edm::ParameterSet const& params)
    : lheWeightTokens_(
          edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
                                [this](const edm::InputTag& tag) { return mayConsume<GenWeightProduct>(tag); })),
      lheWeightInfoTokens_(edm::vector_transform(
          params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
          [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InRun>(tag); })),
      genWeightTokens_(
          edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("genWeights"),
                                [this](const edm::InputTag& tag) { return mayConsume<GenWeightProduct>(tag); })),
      genWeightInfoTokens_(edm::vector_transform(
          params.getParameter<std::vector<edm::InputTag>>("genWeights"),
          [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
      genEventInfoToken_(consumes<GenEventInfoProduct>(params.getParameter<edm::InputTag>("genEvent"))),
      genLumiInfoHeadTag_(
          mayConsume<GenLumiInfoHeader, edm::InLumi>(params.getParameter<edm::InputTag>("genLumiInfoHeader"))),
      weightgroups_(edm::vector_transform(params.getParameter<std::vector<std::string>>("weightgroups"),
                                          [](auto& c) { return gen::WeightType(c.at(0)); })),
      outputnames_(params.getParameter<std::vector<std::string>>("outputNames")),
      maxGroupsPerType_(params.getParameter<std::vector<int>>("maxGroupsPerType")),
      pdfIds_(params.getUntrackedParameter<std::vector<int>>("pdfIds", {})),
      lheWeightPrecision_(params.getParameter<int32_t>("lheWeightPrecision")),
      unknownOnlyIfEmpty_(edm::vector_transform(params.getParameter<std::vector<std::string>>("unknownOnlyIfEmpty"),
                                                [](auto& c) { return gen::WeightType(c.at(0)); })),
      keepAllPSWeights_(params.getParameter<bool>("keepAllPSWeights")),
      ignoreLheGroups_(params.getUntrackedParameter<bool>("ignoreLheGroups", false)),
      ignoreGenGroups_(params.getUntrackedParameter<bool>("ignoreGenGroups", false)),
      nStoreUngroupedLhe_(params.getUntrackedParameter<int>("nStoreUngroupedLhe", 10)),
      nStoreUngroupedGen_(params.getUntrackedParameter<int>("nStoreUngroupedGen", 10)) {
  if (weightgroups_.size() != maxGroupsPerType_.size() || weightgroups_.size() != outputnames_.size())
    throw std::invalid_argument(
        "Inputs 'weightgroups', 'maxGroupsPerType', and 'outputNames' must "
        "have equal size"
        "! Found " +
        std::to_string(weightgroups_.size()) + "; " + std::to_string(maxGroupsPerType_.size()) + "; " +
        std::to_string(outputnames_.size()));

  produces<nanoaod::FlatTable>("GENWeight");
  produces<nanoaod::MergeableCounterTable, edm::Transition::EndRun>();
  produces<std::string>("genModel");
  produces<std::vector<nanoaod::FlatTable>>("LHEWeightTableVec");
}

void GenWeightsTableProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // access counter for weight sums
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

  auto const& genInfo = iEvent.get(genEventInfoToken_);
  const double genWeight = genInfo.weight();
  // table for gen info, always available
  auto outGeninfo = std::make_unique<nanoaod::FlatTable>(1, "genWeight", true);
  outGeninfo->setDoc("generator weight");
  outGeninfo->addColumnValue<float>("", genInfo.weight(), "generator weight");
  iEvent.put(std::move(outGeninfo), "GENWeight");
  // this will take care of sum of genWeights
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
  for (auto& token : genWeightTokens_) {
    iEvent.getByToken(token, genWeightHandle);
    if (genWeightHandle.isValid()) {
      break;
    }
  }
  const GenWeightProduct* genWeightProduct = genWeightHandle.product();
  WeightsContainer genWeights = genWeightProduct->weights();

  auto const& weightInfos = *luminosityBlockCache(iEvent.getLuminosityBlock().index());

  // create a container with dummy weight vector
  auto weightTablevec = std::make_unique<std::vector<nanoaod::FlatTable>>();
  if (foundLheWeights) {
    if (ignoreLheGroups_) {
      fillTableIgnoringGroups(*weightTablevec, weightInfos.at(inLHE), lheWeights, nStoreUngroupedLhe_, "LHEWeight");
    } else
      addWeightGroupToTable(*weightTablevec, weightInfos.at(inLHE), lheWeights);
  }

  if (ignoreGenGroups_)
    fillTableIgnoringGroups(*weightTablevec, weightInfos.at(inGen), genWeights, nStoreUngroupedGen_, "GenWeight");
  else
    addWeightGroupToTable(*weightTablevec, weightInfos.at(inGen), genWeights);

  iEvent.put(std::move(weightTablevec), "LHEWeightTableVec");
}

// Sequentially add the weights, up to maxStore
// Note that the order of the weights in the WeightsVector matches the order of
// weightgroups. In very rare cases, this could be modified from the order in
// the LHE file. If this happens, write a warning message in the table info
void GenWeightsTableProducer::fillTableIgnoringGroups(std::vector<nanoaod::FlatTable>& weightTablevec,
                                                      const WeightGroupDataContainer& weightInfos,
                                                      WeightsContainer& allWeights,
                                                      size_t maxStore,
                                                      std::string tablename) const {
  std::vector<double> weights(maxStore);
  size_t groupIdx = 0;
  size_t offset = 0;
  std::string tableInfo = "First ";
  tableInfo.append(std::to_string(maxStore));
  tableInfo.append(" weights; ");
  std::string warnings = "";
  bool foundUnassociated = false;
  for (size_t i = 0; i < maxStore; i++) {
    if (groupIdx >= allWeights.size())
      throw cms::Exception("GenWeightsTableProducer")
          << "Requested " + std::to_string(maxStore) + " weights, which is more than there are in the file";
    size_t entry = i - offset;
    auto& weightsForGroup = allWeights.at(groupIdx);
    weights.at(i) = weightsForGroup.at(entry);

    if (weightInfos.size() <= groupIdx)
      throw cms::Exception("GenWeightsTableProducer")
          << "Unable to match weight to one of " << weightInfos.size() << " WeightGroups";
    auto matchingGroup = weightInfos.at(groupIdx).group;
    if (entry == 0) {
      size_t maxRange = std::min(offset + weightsForGroup.size() - 1, maxStore);
      tableInfo.append("[");
      tableInfo.append(std::to_string(offset));
      tableInfo.append("]-[");
      tableInfo.append(std::to_string(maxRange));
      tableInfo.append("] ");
      tableInfo.append(matchingGroup->name());
      tableInfo.append("; ");
    }

    // Check if the order corresponds to the LHE file order
    try {
      auto matchingInfo = matchingGroup->weightMetaInfo(entry);

      if (matchingInfo.globalIndex != i) {
        warnings.append("Index ");
        warnings.append(std::to_string(i));
        warnings.append(
            " does not match order in the LHE file or gen product (where it is "
            "entry ");
        warnings.append(std::to_string(matchingInfo.globalIndex));
        warnings.append(")");
      }
    } catch (cms::Exception& e) {
      if (!foundUnassociated)
        warnings.append(
            "Could not associate some weights to a group. Cannot verify"
            " that the order is maintained wrt the LHE file or gen product");
      foundUnassociated = true;
    }
    if (entry == weightsForGroup.size() - 1) {
      groupIdx += 1;
      offset += weightsForGroup.size();
    }
  }
  if (!warnings.empty())
    tableInfo.append("WARNING: " + warnings);

  weightTablevec.emplace_back(weights.size(), tablename, false);
  weightTablevec.back().addColumn<float>("", weights, tableInfo, lheWeightPrecision_);
}

void GenWeightsTableProducer::addWeightGroupToTable(std::vector<nanoaod::FlatTable>& weightTablevec,
                                                    const WeightGroupDataContainer& weightInfos,
                                                    WeightsContainer& allWeights) const {
  std::unordered_map<gen::WeightType, int> typeCount = {};
  for (auto& type : gen::allWeightTypes)
    typeCount[type] = 0;

  std::unordered_map<gen::WeightType, std::string> weightTypeNames_;
  for (size_t i = 0; i < weightgroups_.size(); i++) {
    weightTypeNames_[weightgroups_[i]] = outputnames_[i];
  }

  for (const auto& groupInfo : weightInfos) {
    gen::WeightType weightType = groupInfo.group->weightType();
    std::string entryName = weightTypeNames_.at(weightType);
    std::string label = groupInfo.group->description();
    auto weights = allWeights.at(groupInfo.index);
    if (weightType == gen::WeightType::kScaleWeights) {
      const auto& scaleGroup = *static_cast<const gen::ScaleWeightGroupInfo*>(groupInfo.group);
      auto weightsAndLabel = orderedScaleWeights(weights, scaleGroup);
      label.append(weightsAndLabel.first);
      weights = weightsAndLabel.second;
    } else if (weightType == gen::WeightType::kPartonShowerWeights) {
      const auto& psGroup = *static_cast<const gen::PartonShowerWeightGroupInfo*>(groupInfo.group);
      if (!keepAllPSWeights_) {
        auto weightsAndLabel = preferredPSweights(weights, psGroup);
        label.append(weightsAndLabel.first);
        weights = weightsAndLabel.second;
      } else if (psGroup.isWellFormed()) {
        double baseline = weights[psGroup.weightIndexFromLabel("Baseline")];
        for (size_t i = 0; i < weights.size(); i++)
          weights[i] = weights[i] / baseline;
        label = "PS weights (w_var / w_nominal)";
      } else
        label.append(
            "WARNING: Did not properly parse weight information. Verify order "
            "manually.");
    } else if (!groupInfo.group->isWellFormed())
      label.append(
          "WARNING: Did not properly parse weight information. Verify order "
          "manually.");

    if (typeCount[weightType] > 0) {
      entryName.append("AltSet");
      entryName.append(std::to_string(typeCount[weightType]));
    }
    weightTablevec.emplace_back(weights.size(), entryName, false);
    weightTablevec.back().addColumn<float>("", weights, label, lheWeightPrecision_);

    typeCount[weightType]++;
  }
}

WeightGroupsToStore GenWeightsTableProducer::groupsToStore(
    bool foundLheWeights,
    edm::Handle<GenWeightInfoProduct>& genWeightInfoHandle,
    edm::Handle<GenWeightInfoProduct>& lheWeightInfoHandle) const {
  std::unordered_map<gen::WeightType, int> storePerType;
  for (size_t i = 0; i < weightgroups_.size(); i++)
    storePerType[weightgroups_.at(i)] = maxGroupsPerType_.at(i);

  WeightGroupsToStore weightsToStore;
  // The order LHE then GEN is useful for the unknownOnlyIfEmpy check
  bool storeUnknown = unknownOnlyIfEmpty_.empty();
  auto groupsToSearch = unknownOnlyIfEmpty_;
  for (auto genOrLhe : {inLHE, inGen}) {
    bool isLHE = genOrLhe == inLHE;
    if (isLHE && !foundLheWeights)
      continue;
    auto& hand = isLHE ? lheWeightInfoHandle : genWeightInfoHandle;
    bool ignoreGroups = isLHE ? ignoreLheGroups_ : ignoreGenGroups_;
    auto& toStorePerType = weightsToStore[genOrLhe];
    if (ignoreGroups) {
      toStorePerType = hand->allWeightGroupsInfoWithIndices();
    } else {
      for (auto& typeAndCount : storePerType) {
        if (typeAndCount.first == gen::WeightType::kUnknownWeights && !storeUnknown)
          continue;
        // Since the count isn't updated, the counts are effectively independent
        // between LHE and GEN
        auto groupsPerType = weightDataPerType(hand, typeAndCount.first, typeAndCount.second);
        // Only store unknown if at least one specified groups is empty
        if (!storeUnknown && !groupsToSearch.empty()) {
          auto it = std::find(std::begin(groupsToSearch), std::end(groupsToSearch), typeAndCount.first);
          if (it != std::end(groupsToSearch)) {
            if (groupsPerType.empty())
              storeUnknown = true;
            // Remove from array to avoid repeating the check on GEN. NOTE, if
            // parton shower weights are included as one of the ones to
            // consider, this can cause unknown LHE weights to be stored, given
            // the order of the loops
            else
              groupsToSearch.erase(it);
          }
        }
        toStorePerType.insert(std::end(toStorePerType), std::begin(groupsPerType), std::end(groupsPerType));
      }
    }
  }
  return weightsToStore;
}

WeightGroupDataContainer GenWeightsTableProducer::weightDataPerType(
    edm::Handle<GenWeightInfoProduct>& weightsInfoHandle, gen::WeightType weightType, size_t maxStore) const {
  WeightGroupDataContainer allgroups;
  if (weightType == gen::WeightType::kPdfWeights && !pdfIds_.empty()) {
    allgroups = weightsInfoHandle->pdfGroupsWithIndicesByLHAIDs(pdfIds_);
    if (allgroups.size() > maxStore)
      allgroups.resize(maxStore);
  } else
    allgroups = weightsInfoHandle->weightGroupsAndIndicesByType(weightType, maxStore);

  return allgroups;
}

std::pair<std::string, std::vector<double>> GenWeightsTableProducer::orderedScaleWeights(
    const std::vector<double>& scaleWeights, const gen::ScaleWeightGroupInfo& scaleGroup) const {
  std::vector<double> weights;
  std::string labels = "LHE scale variation weights (w_var / w_nominal); ";
  if (scaleGroup.isWellFormed()) {
    weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF05Index()));
    labels += "[0] is muR=0.5 muF=0.5; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF1Index()));
    labels += "[1] is muR=0.5 muF=1; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR05muF2Index()));
    labels += "[2] is muR=0.5 muF=2; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR1muF05Index()));
    labels += "[3] is muR=1 muF=0.5; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.centralIndex()));
    labels += "[4] is muR=1 muF=1; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR1muF2Index()));
    labels += "[5] is muR=1 muF=2; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF05Index()));
    labels += "[6] is muR=2 muF=0.5; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF1Index()));
    labels += "[7] is muR=2 muF=1; ";
    weights.emplace_back(scaleWeights.at(scaleGroup.muR2muF2Index()));
    labels += "[8] is muR=2 muF=2";
  } else {
    weights = scaleWeights;
    size_t nstore = std::min<size_t>(gen::ScaleWeightGroupInfo::MIN_SCALE_VARIATIONS, weights.size());
    weights = std::vector<double>(begin(weights), std::begin(weights) + nstore);
    labels.append("WARNING: Unexpected format found. Contains first " + std::to_string(nstore) +
                  " elements of weights vector, unordered");
  }

  return std::make_pair(labels, weights);
}

std::pair<std::string, std::vector<double>> GenWeightsTableProducer::preferredPSweights(
    const std::vector<double>& psWeights, const gen::PartonShowerWeightGroupInfo& pswV) const {
  std::vector<double> psTosave;

  std::string labels = "PS weights (w_var / w_nominal); ";
  if (pswV.isWellFormed()) {
    double baseline = psWeights.at(pswV.weightIndexFromLabel("Baseline"));
    psTosave.emplace_back(psWeights.at(pswV.variationIndex(true, false, gen::PSVarType::def)) / baseline);
    labels += "[0] is ISR=2 FSR=1; ";
    psTosave.emplace_back(psWeights.at(pswV.variationIndex(false, false, gen::PSVarType::def)) / baseline);
    labels += "[1] is ISR=1 FSR=2; ";
    psTosave.emplace_back(psWeights.at(pswV.isrCombinedDownIndex(gen::PSVarType::def)) / baseline);
    labels += "[2] is ISR=0.5 FSR=1; ";
    psTosave.emplace_back(psWeights.at(pswV.fsrCombinedDownIndex(gen::PSVarType::def)) / baseline);
    labels += "[3] is ISR=1 FSR=0.5; ";
  }
  return std::make_pair(labels, psTosave);
}

void GenWeightsTableProducer::streamEndRunSummary(edm::StreamID id,
                                                  edm::Run const&,
                                                  edm::EventSetup const&,
                                                  CounterMap* runCounterMap) const {
  // this takes care for mergeing all the weight sums
  runCounterMap->mergeSumMap(*streamCache(id));
}

void GenWeightsTableProducer::globalEndRunProduce(edm::Run& iRun,
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
    // Sum from map
    for (auto& sumw : runCounter.weightSumMap_) {
      // Normalize with genEventSumw
      for (auto& val : sumw.second)
        val *= norm;
      out->addVFloat(sumw.first + "Sumw" + label,
                     "Sum of genEventWeight *" + sumw.first + "[i]/genEventSumw" + doclabel,
                     sumw.second);
    }
  }
  iRun.put(std::move(out));
}
void GenWeightsTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("lheWeights");
  desc.add<std::vector<edm::InputTag>>("genWeights", std::vector<edm::InputTag>{{"genWeights"}});
  desc.add<edm::InputTag>("genEvent", edm::InputTag("generator"))
      ->setComment("tag for the GenEventInfoProduct, to get the main weight");
  desc.add<edm::InputTag>("genLumiInfoHeader", edm::InputTag("generator"))
      ->setComment("tag for the GenLumiInfoProduct, to get the model string");
  desc.add<std::vector<std::string>>("weightgroups");
  desc.add<std::vector<std::string>>("outputNames");
  desc.add<std::vector<int>>("maxGroupsPerType");
  desc.addOptionalUntracked<std::vector<int>>("pdfIds");
  desc.add<int32_t>("lheWeightPrecision", -1)->setComment("Number of bits in the mantissa for LHE weights");
  desc.add<std::vector<std::string>>("unknownOnlyIfEmpty")
      ->setComment(
          "Only store weights in an Unknown WeightGroup if one of the "
          "specified groups is empty");
  desc.add<bool>("keepAllPSWeights", false)
      ->setComment("True: stores all PS weights (usually 45); False: saves preferred 4");
  desc.addUntracked<bool>("ignoreLheGroups", false)
      ->setComment(
          "Ignore LHE groups and store the first n weights, regardless of "
          "type");
  desc.addUntracked<bool>("ignoreGenGroups", false)
      ->setComment(
          "Ignore Gen groups and store the first n weights, regardless of "
          "type");
  desc.addUntracked<int>("nStoreUngroupedLhe", 10)
      ->setComment(
          "Store the first n LHE weights (only relevant if ignoreLheGroups is "
          "true)");
  desc.addUntracked<int>("nStoreUngroupedGen", 10)
      ->setComment(
          "Store the first n Gen weights (only relevant if ignoreGenGroups is "
          "true)");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenWeightsTableProducer);
