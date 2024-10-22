// -*- C++ -*-
//
// Package:    HLTHcalSimpleRecHitFilter
// Class:      HLTHcalSimpleRecHitFilter
//
/**\class HLTHcalSimpleRecHitFilter HLTHcalSimpleRecHitFilter.cc Work/HLTHcalSimpleRecHitFilter/src/HLTHcalSimpleRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Wed Sep 19 16:21:29 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HLTHcalSimpleRecHitFilter : public HLTFilter {
public:
  explicit HLTHcalSimpleRecHitFilter(const edm::ParameterSet&);
  ~HLTHcalSimpleRecHitFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<HFRecHitCollection> HcalRecHitsToken_;
  edm::InputTag HcalRecHitCollection_;
  double threshold_;
  int minNHitsNeg_;
  int minNHitsPos_;
  bool doCoincidence_;
  std::vector<unsigned int> maskedList_;
};

//
// constructors and destructor
//
HLTHcalSimpleRecHitFilter::HLTHcalSimpleRecHitFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  //now do what ever initialization is needed
  threshold_ = iConfig.getParameter<double>("threshold");
  minNHitsNeg_ = iConfig.getParameter<int>("minNHitsNeg");
  minNHitsPos_ = iConfig.getParameter<int>("minNHitsPos");
  doCoincidence_ = iConfig.getParameter<bool>("doCoincidence");
  if (iConfig.existsAs<std::vector<unsigned int> >("maskedChannels"))
    maskedList_ =
        iConfig.getParameter<std::vector<unsigned int> >("maskedChannels");  //this is using the raw DetId index
  else
    //worry about possible user menus with the old interface
    if (iConfig.existsAs<std::vector<int> >("maskedChannels")) {
      std::vector<int> tVec = iConfig.getParameter<std::vector<int> >("maskedChannels");
      if (!tVec.empty()) {
        edm::LogError("cfg error") << "masked list of channels missing from HLT menu. Migration from vector of ints to "
                                      "vector of uints needed for this release";
        cms::Exception("Invalid/obsolete masked list of channels");
      }
    }
  HcalRecHitCollection_ = iConfig.getParameter<edm::InputTag>("HFRecHitCollection");
  HcalRecHitsToken_ = consumes<HFRecHitCollection>(HcalRecHitCollection_);
}

HLTHcalSimpleRecHitFilter::~HLTHcalSimpleRecHitFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HLTHcalSimpleRecHitFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("HFRecHitCollection", edm::InputTag("hltHfreco"));
  desc.add<double>("threshold", 3.0);
  desc.add<int>("minNHitsNeg", 1);
  desc.add<int>("minNHitsPos", 1);
  desc.add<bool>("doCoincidence", true);
  std::vector<unsigned int> temp;
  desc.add<std::vector<unsigned int> >("maskedChannels", temp)->setComment(" # now by raw detid, not hashed id");
  descriptions.add("hltHcalSimpleRecHitFilter", desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTHcalSimpleRecHitFilter::hltFilter(edm::Event& iEvent,
                                          const edm::EventSetup& iSetup,
                                          trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // using namespace edm;

  // getting very basic uncalRH
  edm::Handle<HFRecHitCollection> crudeHits;
  try {
    iEvent.getByToken(HcalRecHitsToken_, crudeHits);
  } catch (std::exception& ex) {
    edm::LogWarning("HLTHcalSimpleRecHitFilter") << HcalRecHitCollection_ << " not available";
  }

  bool accept = false;

  int nHitsNeg = 0, nHitsPos = 0;
  for (auto hit : *crudeHits) {
    // masking noisy channels
    if (std::find(maskedList_.begin(), maskedList_.end(), hit.id().rawId()) != maskedList_.end())
      continue;

    // only count tower above threshold
    if (hit.energy() < threshold_)
      continue;

    // count
    if (hit.id().zside() < 0)
      ++nHitsNeg;
    else
      ++nHitsPos;
  }

  // Logic
  if (!doCoincidence_)
    accept = (nHitsNeg >= minNHitsNeg_) || (nHitsPos >= minNHitsPos_);
  else
    accept = (nHitsNeg >= minNHitsNeg_) && (nHitsPos >= minNHitsPos_);
  //  edm::LogInfo("HcalFilter")  << "at evet: " << iEvent.id().event()
  //    << " and run: " << iEvent.id().run()
  //    << " Total HF hits: " << crudeHits->size() << " Above Threshold - nNeg: " << nHitsNeg << " nPos " << nHitsPos
  //    << " doCoinc: " << doCoincidence_ << " accept: " << accept << std::endl;

  // result
  return accept;
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTHcalSimpleRecHitFilter);
