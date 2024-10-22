// -*- C++ -*-
//
// Package:    HcalHitSelection
// Class:      HcalHitSelection
//
/**\class HcalHitSelection HcalHitSelection.cc RecoLocalCalo/HcalHitSelection/src/HcalHitSelection.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Thu Nov  4 22:17:56 CET 2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

//
// class declaration
//

class HcalHitSelection : public edm::stream::EDProducer<> {
public:
  explicit HcalHitSelection(const edm::ParameterSet&);
  ~HcalHitSelection() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::InputTag hbheTag, hoTag, hfTag;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  std::vector<edm::EDGetTokenT<DetIdCollection> > toks_did_;
  int hoSeverityLevel;
  std::vector<edm::InputTag> interestingDetIdCollections;
  const HcalTopology* theHcalTopology_;

  //hcal severity ES
  const HcalChannelQuality* theHcalChStatus;
  const HcalSeverityLevelComputer* theHcalSevLvlComputer;
  std::set<DetId> toBeKept;
  template <typename CollectionType>
  void skim(const edm::Handle<CollectionType>& input, CollectionType& output, int severityThreshold = 0) const;

  // ES tokens
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> qualToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> sevToken_;
};

template <class CollectionType>
void HcalHitSelection::skim(const edm::Handle<CollectionType>& input,
                            CollectionType& output,
                            int severityThreshold) const {
  output.reserve(input->size());
  typename CollectionType::const_iterator begin = input->begin();
  typename CollectionType::const_iterator end = input->end();
  typename CollectionType::const_iterator hit = begin;

  for (; hit != end; ++hit) {
    //    edm::LogError("HcalHitSelection")<<"the hit pointer is"<<&(*hit);
    HcalDetId id = hit->detid();
    if (theHcalTopology_->getMergePositionFlag() && id.subdet() == HcalEndcap) {
      id = theHcalTopology_->idFront(id);
    }
    const uint32_t& recHitFlag = hit->flags();
    //    edm::LogError("HcalHitSelection")<<"the hit id and flag are "<<id.rawId()<<" "<<recHitFlag;

    const uint32_t& dbStatusFlag = theHcalChStatus->getValues(id)->getValue();
    int severityLevel = theHcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);
    //anything that is not "good" goes in
    if (severityLevel > severityThreshold) {
      output.push_back(*hit);
    } else {
      //chek on the detid list
      if (toBeKept.find(id) != toBeKept.end())
        output.push_back(*hit);
    }
  }
}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalHitSelection::HcalHitSelection(const edm::ParameterSet& iConfig)
    : hbheTag(iConfig.getParameter<edm::InputTag>("hbheTag")),
      hoTag(iConfig.getParameter<edm::InputTag>("hoTag")),
      hfTag(iConfig.getParameter<edm::InputTag>("hfTag")),
      theHcalTopology_(nullptr),
      theHcalChStatus(nullptr),
      theHcalSevLvlComputer(nullptr) {
  // register for data access
  tok_hbhe_ = consumes<HBHERecHitCollection>(hbheTag);
  tok_hf_ = consumes<HFRecHitCollection>(hfTag);
  tok_ho_ = consumes<HORecHitCollection>(hoTag);

  interestingDetIdCollections = iConfig.getParameter<std::vector<edm::InputTag> >("interestingDetIds");

  const unsigned nLabels = interestingDetIdCollections.size();
  for (unsigned i = 0; i != nLabels; i++)
    toks_did_.push_back(consumes<DetIdCollection>(interestingDetIdCollections[i]));

  hoSeverityLevel = iConfig.getParameter<int>("hoSeverityLevel");

  produces<HBHERecHitCollection>(hbheTag.label());
  produces<HFRecHitCollection>(hfTag.label());
  produces<HORecHitCollection>(hoTag.label());

  // ES tokens
  htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  qualToken_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  sevToken_ = esConsumes<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd>();
}

HcalHitSelection::~HcalHitSelection() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HcalHitSelection::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  theHcalChStatus = &iSetup.getData(qualToken_);
  theHcalSevLvlComputer = &iSetup.getData(sevToken_);
  theHcalTopology_ = &iSetup.getData(htopoToken_);

  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<HFRecHitCollection> hf;
  edm::Handle<HORecHitCollection> ho;

  iEvent.getByToken(tok_hbhe_, hbhe);
  iEvent.getByToken(tok_hf_, hf);
  iEvent.getByToken(tok_ho_, ho);

  toBeKept.clear();
  edm::Handle<DetIdCollection> detId;
  for (unsigned int t = 0; t < toks_did_.size(); ++t) {
    iEvent.getByToken(toks_did_[t], detId);
    if (!detId.isValid()) {
      edm::LogError("MissingInput") << "the collection of interesting detIds:" << interestingDetIdCollections[t]
                                    << " is not found.";
      continue;
    }
    toBeKept.insert(detId->begin(), detId->end());
  }

  auto hbhe_out = std::make_unique<HBHERecHitCollection>();
  skim(hbhe, *hbhe_out);
  iEvent.put(std::move(hbhe_out), hbheTag.label());

  auto hf_out = std::make_unique<HFRecHitCollection>();
  skim(hf, *hf_out);
  iEvent.put(std::move(hf_out), hfTag.label());

  auto ho_out = std::make_unique<HORecHitCollection>();
  skim(ho, *ho_out, hoSeverityLevel);
  iEvent.put(std::move(ho_out), hoTag.label());
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHitSelection);
