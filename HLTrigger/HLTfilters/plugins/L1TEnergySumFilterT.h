#ifndef HLTrigger_HLTfilters_L1TEnergySumFilterT_h
#define HLTrigger_HLTfilters_L1TEnergySumFilterT_h

#include <vector>
#include <iterator>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
class L1TEnergySumFilterT : public HLTFilter {
public:
  explicit L1TEnergySumFilterT(const edm::ParameterSet&);
  ~L1TEnergySumFilterT() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 edm::EventSetup const&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag const l1tSumTag_;
  edm::EDGetTokenT<std::vector<T>> const l1tSumToken_;

  trigger::TriggerObjectType const l1tSumType_;
  double const minPt_;

  edm::ParameterSet scalings_;
  std::vector<double> theScalings_;

  static trigger::TriggerObjectType typeOfL1TSum(std::string const&);

  double offlineEnergySum(double const Et) const;
};

template <typename T>
L1TEnergySumFilterT<T>::L1TEnergySumFilterT(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1tSumTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      l1tSumToken_(consumes<std::vector<T>>(l1tSumTag_)),
      l1tSumType_(typeOfL1TSum(iConfig.getParameter<std::string>("TypeOfSum"))),
      minPt_(iConfig.getParameter<double>("MinPt")) {
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  theScalings_ = scalings_.getParameter<std::vector<double>>("theScalings");
}

template <typename T>
L1TEnergySumFilterT<T>::~L1TEnergySumFilterT() = default;

template <typename T>
trigger::TriggerObjectType L1TEnergySumFilterT<T>::typeOfL1TSum(std::string const& typeOfSum) {
  trigger::TriggerObjectType sumEnum;

  if (typeOfSum == "MET")
    sumEnum = trigger::TriggerObjectType::TriggerL1PFMET;
  else if (typeOfSum == "ETT")
    sumEnum = trigger::TriggerObjectType::TriggerL1PFETT;
  else if (typeOfSum == "HT")
    sumEnum = trigger::TriggerObjectType::TriggerL1PFHT;
  else if (typeOfSum == "MHT")
    sumEnum = trigger::TriggerObjectType::TriggerL1PFMHT;
  else {
    throw cms::Exception("ConfigurationError")
        << "Wrong type of energy sum: \"" << typeOfSum << "\" (valid choices are: MET, ETT, HT, MHT)";
  }

  return sumEnum;
}

template <typename T>
void L1TEnergySumFilterT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("L1PFEnergySums"));

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double>>("theScalings", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  desc.add<std::string>("TypeOfSum", "HT");
  desc.add<double>("MinPt", -1.);
  descriptions.add(defaultModuleLabel<L1TEnergySumFilterT<T>>(), desc);
}

template <typename T>
bool L1TEnergySumFilterT<T>::hltFilter(edm::Event& iEvent,
                                       edm::EventSetup const& iSetup,
                                       trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1tSumTag_);
  }

  auto const& l1tSums = iEvent.getHandle(l1tSumToken_);

  int nSum(0);
  for (auto iSum = l1tSums->begin(); iSum != l1tSums->end(); ++iSum) {
    double offlinePt = 0.0;
    // pt or sumEt?
    if (l1tSumType_ == trigger::TriggerObjectType::TriggerL1PFMET or
        l1tSumType_ == trigger::TriggerObjectType::TriggerL1PFMHT) {
      offlinePt = offlineEnergySum(iSum->pt());
    } else if (l1tSumType_ == trigger::TriggerObjectType::TriggerL1PFETT or
               l1tSumType_ == trigger::TriggerObjectType::TriggerL1PFHT) {
      offlinePt = offlineEnergySum(iSum->sumEt());
    }

    if (offlinePt >= minPt_) {
      ++nSum;
      edm::Ref<std::vector<T>> ref(l1tSums, std::distance(l1tSums->begin(), iSum));
      filterproduct.addObject(l1tSumType_, ref);
    }
  }

  // return final filter decision
  return nSum > 0;
}

template <typename T>
double L1TEnergySumFilterT<T>::offlineEnergySum(double const Et) const {
  return (theScalings_.at(0) + Et * theScalings_.at(1) + Et * Et * theScalings_.at(2));
}

#endif  // HLTrigger_HLTfilters_L1TEnergySumFilterT_h
