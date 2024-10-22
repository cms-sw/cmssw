#ifndef HLTrigger_HLTfilters_L1TJetFilterT_h
#define HLTrigger_HLTfilters_L1TJetFilterT_h

#include <vector>
#include <cmath>
#include <iterator>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <class T>
class L1TJetFilterT : public HLTFilter {
public:
  explicit L1TJetFilterT(const edm::ParameterSet&);
  ~L1TJetFilterT() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag const l1tJetTag_;
  edm::EDGetTokenT<std::vector<T>> const l1tJetToken_;

  double const minPt_;
  double const minEta_;
  double const maxEta_;
  int const minN_;

  edm::ParameterSet scalings_;           // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;   // barrel scalings
  std::vector<double> overlapScalings_;  // overlap scalings
  std::vector<double> endcapScalings_;   // endcap scalings

  double offlineJetPt(double const pt, double const eta) const;
};

template <class T>
L1TJetFilterT<T>::L1TJetFilterT(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      l1tJetTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      l1tJetToken_(consumes<std::vector<T>>(l1tJetTag_)),
      minPt_(iConfig.getParameter<double>("MinPt")),
      minEta_(iConfig.getParameter<double>("MinEta")),
      maxEta_(iConfig.getParameter<double>("MaxEta")),
      minN_(iConfig.getParameter<int>("MinN")) {
  scalings_ = iConfig.getParameter<edm::ParameterSet>("Scalings");
  barrelScalings_ = scalings_.getParameter<std::vector<double>>("barrel");
  overlapScalings_ = scalings_.getParameter<std::vector<double>>("overlap");
  endcapScalings_ = scalings_.getParameter<std::vector<double>>("endcap");
}

template <class T>
L1TJetFilterT<T>::~L1TJetFilterT() = default;

template <class T>
void L1TJetFilterT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("ak4PFL1PuppiCorrected"));
  desc.add<double>("MinPt", -1.0);
  desc.add<double>("MinEta", -5.0);
  desc.add<double>("MaxEta", 5.0);
  desc.add<int>("MinN", 1);

  edm::ParameterSetDescription descScalings;
  descScalings.add<std::vector<double>>("barrel", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double>>("overlap", {0.0, 1.0, 0.0});
  descScalings.add<std::vector<double>>("endcap", {0.0, 1.0, 0.0});
  desc.add<edm::ParameterSetDescription>("Scalings", descScalings);

  descriptions.add(defaultModuleLabel<L1TJetFilterT<T>>(), desc);
}

template <class T>
bool L1TJetFilterT<T>::hltFilter(edm::Event& iEvent,
                                 const edm::EventSetup& iSetup,
                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1tJetTag_);
  }

  auto const& l1tJets = iEvent.getHandle(l1tJetToken_);

  int nJet(0);
  for (auto iJet = l1tJets->begin(); iJet != l1tJets->end(); ++iJet) {
    if (offlineJetPt(iJet->pt(), iJet->eta()) >= minPt_ && iJet->eta() <= maxEta_ && iJet->eta() >= minEta_) {
      ++nJet;
      edm::Ref<std::vector<T>> ref(l1tJets, std::distance(l1tJets->begin(), iJet));
      filterproduct.addObject(trigger::TriggerObjectType::TriggerL1PFJet, ref);
    }
  }

  // return with final filter decision
  return nJet >= minN_;
}

template <class T>
double L1TJetFilterT<T>::offlineJetPt(double const pt, double const eta) const {
  if (std::abs(eta) < 1.5)
    return (barrelScalings_.at(0) + pt * barrelScalings_.at(1) + pt * pt * barrelScalings_.at(2));
  else if (std::abs(eta) < 2.4)
    return (overlapScalings_.at(0) + pt * overlapScalings_.at(1) + pt * pt * overlapScalings_.at(2));
  else
    return (endcapScalings_.at(0) + pt * endcapScalings_.at(1) + pt * pt * endcapScalings_.at(2));

  /*
  uint const scIdx = std::abs(eta) < 1.5 ? 0 : (std::abs(eta) < 2.4 ? 1 : 2);

  if (scIdx >= scalingConstants.m_constants.size()) {
    throw cms::Exception("Input") << "out-of-range index for L1TObjScalingConstants vector (size="
                                  << scalingConstants.m_constants.size() << ") [jet: pt=" << pt << ", eta=" << eta
                                  << "]";
  }

  return scalingConstants.m_constants.at(scIdx).m_constant + pt * scalingConstants.m_constants.at(scIdx).m_linear +
         pt * pt * scalingConstants.m_constants.at(scIdx).m_quadratic;
  */
}

#endif  // HLTrigger_HLTfilters_L1TJetFilterT_h
