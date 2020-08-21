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
  typedef std::vector<gen::WeightGroupData> WeightGroupDataContainer;
  typedef std::array<std::vector<gen::WeightGroupData>, 2> WeightGroupsToStore;
}  // namespace
using CounterMap = genCounter::CounterMap;
using Counter = genCounter::Counter;

class LHEWeightsTableProducer : 
public edm::global::EDProducer<edm::LuminosityBlockCache<WeightGroupsToStore>, 
			       edm::StreamCache<CounterMap>, 
			       edm::RunSummaryCache<CounterMap>,
			       edm::EndRunProducer> {
public:
  LHEWeightsTableProducer(edm::ParameterSet const& params);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  //func changed//sroychow
  void addWeightGroupToTable(std::unique_ptr<std::vector<nanoaod::FlatTable>>& lheWeightTables,
			     const char* typeName,
			     const WeightGroupDataContainer& weightInfos,
			     WeightsContainer& allWeights, Counter& counter, 
			     double genWeight) const;

  WeightGroupDataContainer weightDataPerType(edm::Handle<GenWeightInfoProduct>& weightsHandle,
                                             gen::WeightType weightType,
                                             int& maxStore) const;

  std::vector<double> orderedScaleWeights(const std::vector<double>& scaleWeights,
                                          const gen::ScaleWeightGroupInfo* scaleGroup) const;

  std::vector<double> getPreferredPSweights(const std::vector<double>& psWeights, const gen::PartonShowerWeightGroupInfo* pswV) const;

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
 	weightsToStore.at(inLHE).insert(weightsToStore.at(inLHE).end(), lheWeights.begin(), lheWeights.end());
      }
      auto genWeights = weightDataPerType(genWeightInfoHandle, weightType, storePerType[weightType]);
      weightsToStore.at(inGen).insert(weightsToStore.at(inGen).end(), genWeights.begin(), genWeights.end());
    }
    return std::make_shared<WeightGroupsToStore>(weightsToStore);
  }
  
  // nothing to do here
  virtual void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {}
  // create an empty counter
   std::unique_ptr<CounterMap> beginStream(edm::StreamID) const override { return std::make_unique<CounterMap>(); }
  // inizialize to zero at begin run
  void streamBeginRun(edm::StreamID id, 
		      edm::Run const&, edm::EventSetup const&) const override { streamCache(id)->clear(); }
  
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
    //std::cout << "StreamBeginLuminosityBlock:" << id.value() << "\nPrinting counter map keys" << std::endl;
    //for(auto& cm : counterMap->countermap)
    //  std::cout << cm.first << std::endl;
    std::string label = genLumiInfoHead.isValid() ? counterMap->getLabel() : "NULL";
    //std::cout << "StreamBeginLuminosityBlock:" << id.value() << "\nCounterMapLabel:" << label << std::endl;
    
  }
  // create an empty counter
  std::shared_ptr<CounterMap> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
    return std::make_shared<CounterMap>();
  }
  // add this stream to the summary
  void streamEndRunSummary(edm::StreamID id, edm::Run const&, 
			   edm::EventSetup const&, CounterMap* runCounterMap) const override;
  // nothing to do per se
  void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, CounterMap* runCounterMap) const override {}
  // write the total to the run
  void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const& es, CounterMap const* runCounterMap) const override;
  // nothing to do here
  //void globalEndRun(edm::Run const& iRun, edm::EventSetup const& es, CounterMap* runCounterMap) const override {}
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
protected:
  //const std::vector<edm::EDGetTokenT<LHEEventProduct>> lheTokens_;
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
    {gen::WeightType::kScaleWeights, "Scale"},
    {gen::WeightType::kPdfWeights, "Pdf"},
    {gen::WeightType::kMEParamWeights, "MEParam"},
    {gen::WeightType::kPartonShowerWeights, "PartonShower"},
    {gen::WeightType::kUnknownWeights, "Unknown"},
  };
  //std::unordered_map<std::string, int> weightGroupIndices_;
  int lheWeightPrecision_;
  bool storeAllPSweights_;  

  enum { inLHE, inGen };
};
//put back if needed; till now not used
LHEWeightsTableProducer::LHEWeightsTableProducer(edm::ParameterSet const& params)
  :   lheWeightTokens_(
		       edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
					     [this](const edm::InputTag& tag) { return mayConsume<GenWeightProduct>(tag); })),
      lheWeightInfoTokens_(edm::vector_transform(
						 params.getParameter<std::vector<edm::InputTag>>("lheWeights"),
						 [this](const edm::InputTag& tag) { return mayConsume<GenWeightInfoProduct, edm::InLumi>(tag); })),
  genWeightToken_(consumes<GenWeightProduct>(params.getParameter<edm::InputTag>("genWeights"))),
  genWeightInfoToken_(consumes<GenWeightInfoProduct, edm::InLumi>(params.getParameter<edm::InputTag>("genWeights"))),
  genEventInfoToken_(consumes<GenEventInfoProduct>(params.getParameter<edm::InputTag>("genEvent"))),
  genLumiInfoHeadTag_(mayConsume<GenLumiInfoHeader, edm::InLumi>(params.getParameter<edm::InputTag>("genLumiInfoHeader"))),
  weightgroups_(edm::vector_transform(params.getParameter<std::vector<std::string>>("weightgroups"),
				       [](auto& c) { return gen::WeightType(c.at(0)); })),
   maxGroupsPerType_(params.getParameter<std::vector<int>>("maxGroupsPerType")),
   pdfIds_(params.getUntrackedParameter<std::vector<int>>("pdfIds", {})),
  lheWeightPrecision_(params.getParameter<int32_t>("lheWeightPrecision")),
  storeAllPSweights_(params.getParameter<bool>("storeAllPSweights"))
{
  if (weightgroups_.size() != maxGroupsPerType_.size())
    throw std::invalid_argument("Inputs 'weightgroups' and 'weightgroupNums' must have equal size");
  produces<std::vector<nanoaod::FlatTable>>();
  produces<nanoaod::FlatTable>();
  produces<nanoaod::MergeableCounterTable, edm::Transition::EndRun>();
  produces<std::string>();
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
  auto outGeninfo = std::make_unique<nanoaod::FlatTable>(1, "genWeightNEW", true);
  outGeninfo->setDoc("generator weight");
  outGeninfo->addColumnValue<float>("", genInfo.weight(), "generator weight", nanoaod::FlatTable::FloatColumn);
  iEvent.put(std::move(outGeninfo));
  //this will take care of sum of genWeights
  counter.incGenOnly(genWeight);
  
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
  
  auto lheWeightTables = std::make_unique<std::vector<nanoaod::FlatTable>>();
  if (foundLheWeights) {
    addWeightGroupToTable(lheWeightTables, "LHE", weightInfos.at(inLHE), lheWeights, counter, genWeight);
  }

  addWeightGroupToTable(lheWeightTables, "Gen", weightInfos.at(inGen), genWeights, counter, genWeight);
  
  iEvent.put(std::move(lheWeightTables));
  /*
  std::string& model_label = streamCache(id)->getLabel();
  auto outM = std::make_unique<std::string>((!model_label.empty()) ? std::string("GenModel_") + model_label : "");
  iEvent.put(std::move(outM), "genModel");
  */
  //std::cout << "From Event:\n"; 
  //for(auto& cm : counter.weightSumMap_)
  //  std::cout << "WeightName:" << cm.first 
  //      << "\t size:" << cm.second.size() 
  //	      << "\t First value:" << cm.second.at(0) 
  //	      << std::endl;
}

void LHEWeightsTableProducer::addWeightGroupToTable(std::unique_ptr<std::vector<nanoaod::FlatTable>>& lheWeightTables, 
						    const char* typeName,
						    const WeightGroupDataContainer& weightInfos,
						    WeightsContainer& allWeights, Counter& counter, 
						    double genWeight)  const {
  std::unordered_map<gen::WeightType, int> typeCount = {};
  for (auto& type : gen::allWeightTypes)
    typeCount[type] = 0; 
  
  for (const auto& groupInfo : weightInfos) {
    std::string entryName = typeName;
    gen::WeightType weightType = groupInfo.group->weightType();
    std::string name = weightTypeNames_.at(weightType);
    std::string label = groupInfo.group->name();
    auto& weights = allWeights.at(groupInfo.index);
    label.append("; ");
    if (false && weightType == gen::WeightType::kScaleWeights && groupInfo.group->isWellFormed() &&
 	groupInfo.group->nIdsContained() < 10) {
      weights = orderedScaleWeights(weights, dynamic_cast<const gen::ScaleWeightGroupInfo*>(groupInfo.group));
      label.append(
 		   "[1] is mur=0.5 muf=1; [2] is mur=0.5 muf=2; [3] is mur=1 muf=0.5 ;"
 		   " [4] is mur=1 muf=1; [5] is mur=1 muf=2; [6] is mur=2 muf=0.5;"
 		   " [7] is mur=2 muf=1 ; [8] is mur=2 muf=2)");
    } else if (!storeAllPSweights_ && weightType == gen::WeightType::kPartonShowerWeights && groupInfo.group->isWellFormed()) {
      const gen::PartonShowerWeightGroupInfo* psgInfo = dynamic_cast<const gen::PartonShowerWeightGroupInfo*>(groupInfo.group);
      //std::cout << "PSWeights size :" << weights.size() 
      //	<< "\tWtnames size:" << psgInfo->getWeightNames().size() 
      //	<< std::endl;
      //std::cout << "WeightMetaInfo size:" << psgInfo->idsContained().size() << std::endl;
      //for(unsigned int i = 0 ; i < psgInfo->idsContained().size(); i++)
      //  std::cout << "Index :" << i << "\t" << psgInfo->idsContained().at(i).label << std::endl;
      std::cout << "PS weights \n Index: 1\t" << psgInfo->idsContained().at(0).label 
		<< "\t value:" << weights.at(0) << std::endl;  
      std::cout << "PS weights \n Index: 1\t" << psgInfo->idsContained().at(1).label
		<< "\t value:" << weights.at(1) << std::endl;
      for (unsigned int i = 6; i < 10; i++) 
	std::cout << "Index :" << i 
		  << "\t weight name:" << psgInfo->idsContained().at(i).label 
		  << "\t value:" << weights.at(i)
		  << std::endl;
      weights = getPreferredPSweights(weights, dynamic_cast<const gen::PartonShowerWeightGroupInfo*>(groupInfo.group));   
      label.append("PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2");
    } else
      label.append(groupInfo.group->description());
    
    entryName.append(name);
    entryName.append("Weight");
    if (typeCount[weightType] > 0) {
      entryName.append("AltSet");
      entryName.append(std::to_string(typeCount[weightType]));
    }
    counter.incLHE(genWeight, weights, entryName);
    lheWeightTables->emplace_back(weights.size(), entryName, false);
    lheWeightTables->back().addColumn<float>("", weights, label, nanoaod::FlatTable::FloatColumn, lheWeightPrecision_);
    typeCount[weightType]++;
    
    //std::cout << "Weight type read: " << name << std::endl; 
  }
}

WeightGroupDataContainer LHEWeightsTableProducer::weightDataPerType(edm::Handle<GenWeightInfoProduct>& weightsHandle,
								    gen::WeightType weightType,
								    int& maxStore) const {
  WeightGroupDataContainer group;
  if (weightType == gen::WeightType::kPdfWeights && pdfIds_.size() > 0) {
    group = weightsHandle->pdfGroupsWithIndicesByLHAIDs(pdfIds_);
  } else
    group = weightsHandle->weightGroupsAndIndicesByType(weightType);
  
  if (maxStore < 0 || static_cast<int>(group.size()) <= maxStore) {
    // Modify size in case one type of weight is present in multiple products
    maxStore -= group.size();
    return group;
  }
  return std::vector(group.begin(), group.begin() + maxStore);
}

std::vector<double> LHEWeightsTableProducer::orderedScaleWeights(const std::vector<double>& scaleWeights,
								 const gen::ScaleWeightGroupInfo* scaleGroup) const {
  std::vector<double> weights;
  weights.emplace_back(scaleWeights.at(scaleGroup->muR05muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR05muF1Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR05muF2Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR1muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->centralIndex()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR1muF2Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR2muF05Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR2muF1Index()));
  weights.emplace_back(scaleWeights.at(scaleGroup->muR2muF2Index()));
  
  return weights;
}

std::vector<double> LHEWeightsTableProducer::getPreferredPSweights(const std::vector<double>& psWeights, 
								   const gen::PartonShowerWeightGroupInfo* pswV) const {
  std::vector<double> psTosave;
  
  double baseline = psWeights.at(pswV->weightIndexFromLabel("Baseline"));//at 1
  psTosave.emplace_back( psWeights.at(pswV->weightIndexFromLabel("isrDefHi"))/baseline ); // at 6
  psTosave.emplace_back( psWeights.at(pswV->weightIndexFromLabel("fsrDefHi"))/baseline ); // at 7
  psTosave.emplace_back( psWeights.at(pswV->weightIndexFromLabel("isrDefLo"))/baseline ); // at 8
  psTosave.emplace_back( psWeights.at(pswV->weightIndexFromLabel("fsrDefLo"))/baseline ); // at 9
  return psTosave;
}

void LHEWeightsTableProducer::streamEndRunSummary(edm::StreamID id,
			 edm::Run const&,
			 edm::EventSetup const&,
			 CounterMap* runCounterMap) const  {
  std::cout << "<<<<<From StreamEndRunSummary StreamID:" << id.value() << ">>>>>>\n";
  std::cout << "Map label:" << (*streamCache(id)).active_label << std::endl;
  Counter& counter = *streamCache(id)->get();
  for(auto& cm : counter.weightSumMap_)
    std::cout << "WeightName:" << cm.first 
  	      << "\t size:" << cm.second.size() 
  	      << "\t First value:" << cm.second.at(0) 
  	      << std::endl;
  std::cout << "<<<<<End StreamEndRunSummary>>>>>>\n";
  //this takes care for mergeing all the weight sums
  runCounterMap->mergeSumMap(*streamCache(id));
}

void LHEWeightsTableProducer::globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, CounterMap const* runCounterMap) const {
  auto out = std::make_unique<nanoaod::MergeableCounterTable>();
  
  for (auto x : runCounterMap->countermap) {
    auto& runCounter = x.second;
    std::string label = std::string("_") + x.first;
    std::string doclabel = (!x.first.empty()) ? (std::string(", for model label ") + x.first) : "";
    
    out->addInt("genEventCountNEW" + label, "event count" + doclabel, runCounter.num_);
    out->addFloat("genEventSumwNEW" + label, "sum of gen weights" + doclabel, runCounter.sumw_);
    out->addFloat("genEventSumw2NEW" + label, "sum of gen (weight^2)" + doclabel, runCounter.sumw2_);
    
    double norm = runCounter.sumw_ ? 1.0 / runCounter.sumw_ : 1;
    //Sum from map
    std::cout << "SUM map size:" << runCounter.weightSumMap_.size() << std::endl;
    for(auto& sumw : runCounter.weightSumMap_) {
      std::cout << "Adding wsum for:" 
                << sumw.first << "\t Size:" 
		<< sumw.second.size()
		<< "\t First value:" << sumw.second.at(0)
		<< std::endl;

      //Normalize with genEventSumw
      for(auto& val : sumw.second) val *= norm;
      out->addVFloat(sumw.first + "Sumw" + label, 
		     "Sum of genEventWeight *" + sumw.first + "[i]/genEventSumw" + doclabel,
		     sumw.second) ;
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
  desc.add<edm::InputTag>("genEvent", edm::InputTag("generator"))->setComment("tag for the GenEventInfoProduct, to get the main weight");
  desc.add<edm::InputTag>("genLumiInfoHeader", edm::InputTag("generator"))->setComment("tag for the GenLumiInfoProduct, to get the model string");
  desc.add<std::vector<std::string>>("weightgroups");
  desc.add<std::vector<int>>("maxGroupsPerType");
  desc.addOptionalUntracked<std::vector<int>>("pdfIds");
  desc.add<int32_t>("lheWeightPrecision", -1)->setComment("Number of bits in the mantissa for LHE weights");
  desc.add<bool>("storeAllPSweights", -1)->setComment("True:stores all 45 PS weights; False:saves preferred 4");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHEWeightsTableProducer);
