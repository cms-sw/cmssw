#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

#include <algorithm>

template <typename T>
class L1TStage2ObjectComparison : public edm::stream::EDProducer<> {

 public:

  L1TStage2ObjectComparison(const edm::ParameterSet& ps);
  ~L1TStage2ObjectComparison() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:  

  edm::EDGetTokenT<BXVector<T>> token1_;
  edm::EDGetTokenT<BXVector<T>> token2_;
  const bool checkBxRange_;
  const bool checkCollSizePerBx_;
  const bool checkObject_;

};

template <typename T>
L1TStage2ObjectComparison<T>::L1TStage2ObjectComparison(const edm::ParameterSet& ps)
    : token1_(consumes<BXVector<T>>(ps.getParameter<edm::InputTag>("collection1"))),
      token2_(consumes<BXVector<T>>(ps.getParameter<edm::InputTag>("collection2"))),
      checkBxRange_(ps.getParameter<bool>("checkBxRange")),
      checkCollSizePerBx_(ps.getParameter<bool>("checkCollSizePerBx")),
      checkObject_(ps.getParameter<bool>("checkObject"))
{
  if (checkBxRange_ || checkCollSizePerBx_) {
    produces<l1t::ObjectRefBxCollection<T>>("collection1ExcessObjects");
    produces<l1t::ObjectRefBxCollection<T>>("collection2ExcessObjects");
  }
  if (checkObject_) {
    produces<l1t::ObjectRefPairBxCollection<T>>("objectMatches");
    produces<l1t::ObjectRefPairBxCollection<T>>("objectMismatches");
  }
}

template <typename T>
void L1TStage2ObjectComparison<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("collection1", edm::InputTag("collection1"))->setComment("L1T object collection 1");
  desc.add<edm::InputTag>("collection2", edm::InputTag("collection2"))->setComment("L1T object collection 2");
  desc.add<bool>("checkBxRange", true)->setComment("Check if BX ranges match");
  desc.add<bool>("checkCollSizePerBx", true)->setComment("Check if collection sizes within one BX match");
  desc.add<bool>("checkObject", true)->setComment("Check if objects match");
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
void L1TStage2ObjectComparison<T>::produce(edm::Event& e, const edm::EventSetup& c)
{
  auto excessObjRefsColl1 = std::make_unique<l1t::ObjectRefBxCollection<T>>();
  auto excessObjRefsColl2 = std::make_unique<l1t::ObjectRefBxCollection<T>>();
  auto matchRefPairs = std::make_unique<l1t::ObjectRefPairBxCollection<T>>();
  auto mismatchRefPairs = std::make_unique<l1t::ObjectRefPairBxCollection<T>>();

  edm::Handle<BXVector<T>> bxColl1;
  edm::Handle<BXVector<T>> bxColl2;
  e.getByToken(token1_, bxColl1);
  e.getByToken(token2_, bxColl2);

  // Set the BX ranges like the input collection BX ranges
  excessObjRefsColl1->setBXRange(bxColl1->getFirstBX(), bxColl1->getLastBX());
  excessObjRefsColl2->setBXRange(bxColl2->getFirstBX(), bxColl2->getLastBX());
  // Set the BX range to the intersection of the two input collection BX ranges
  matchRefPairs->setBXRange(std::max(bxColl1->getFirstBX(), bxColl2->getFirstBX()), std::min(bxColl1->getLastBX(), bxColl2->getLastBX()));
  mismatchRefPairs->setBXRange(std::max(bxColl1->getFirstBX(), bxColl2->getFirstBX()), std::min(bxColl1->getLastBX(), bxColl2->getLastBX()));

  if (checkBxRange_) {
    // Store references to objects in BX that do not exist in the other collection
    typename BXVector<T>::const_iterator it;
    // BX range of collection 1 > collection 2
    for (auto iBx = bxColl1->getFirstBX(); iBx < bxColl2->getFirstBX(); ++iBx) {
      for (it = bxColl1->begin(iBx); it != bxColl1->end(iBx); ++it) {
        edm::Ref<BXVector<T>> ref{bxColl1, bxColl1->key(it)};
        excessObjRefsColl1->push_back(iBx, ref);
      }
    }
    for (auto iBx = bxColl1->getLastBX(); iBx > bxColl2->getLastBX(); --iBx) {
      for (it = bxColl1->begin(iBx); it != bxColl1->end(iBx); ++it) {
        edm::Ref<BXVector<T>> ref{bxColl1, bxColl1->key(it)};
        excessObjRefsColl1->push_back(iBx, ref);
      }
    }
    // BX range of collection 2 > collection 1
    for (auto iBx = bxColl2->getFirstBX(); iBx < bxColl1->getFirstBX(); ++iBx) {
      for (it = bxColl2->begin(iBx); it != bxColl2->end(iBx); ++it) {
        edm::Ref<BXVector<T>> ref{bxColl2, bxColl2->key(it)};
        excessObjRefsColl2->push_back(iBx, ref);
      }
    }
    for (auto iBx = bxColl2->getLastBX(); iBx > bxColl1->getLastBX(); --iBx) {
      for (it = bxColl2->begin(iBx); it != bxColl2->end(iBx); ++it) {
        edm::Ref<BXVector<T>> ref{bxColl2, bxColl2->key(it)};
        excessObjRefsColl2->push_back(iBx, ref);
      }
    }
  }

  // Loop over all BX that exist in both collections
  for (int iBx = matchRefPairs->getFirstBX(); iBx <= matchRefPairs->getLastBX(); ++iBx) {
    auto it1 = bxColl1->begin(iBx);
    auto it2 = bxColl2->begin(iBx);
    while (it1 != bxColl1->end(iBx) && it2 != bxColl2->end(iBx)) {
      if (checkObject_) {
        // Store reference pairs for matching and mismatching objects
        edm::Ref<BXVector<T>> ref1{bxColl1, bxColl1->key(it1)};
        edm::Ref<BXVector<T>> ref2{bxColl2, bxColl2->key(it2)};
        if (*it1 == *it2) {
          matchRefPairs->push_back(iBx, std::make_pair(ref1, ref2));
        } else {
          mismatchRefPairs->push_back(iBx, std::make_pair(ref1, ref2));
        }
      }
      ++it1;
      ++it2;
    }
    if (checkCollSizePerBx_) {
      // Store references to excess objects if there are more objects in one collection (per BX)
      while (it1 != bxColl1->end(iBx)) {
        edm::Ref<BXVector<T>> ref{bxColl1, bxColl1->key(it1)};
        excessObjRefsColl1->push_back(iBx, ref);
        ++it1;
      }
      while (it2 != bxColl2->end(iBx)) {
        edm::Ref<BXVector<T>> ref{bxColl2, bxColl2->key(it2)};
        excessObjRefsColl2->push_back(iBx, ref);
        ++it2;
      }
    }
  }

  // Put data in the event
  if (checkBxRange_ || checkCollSizePerBx_) {
    e.put(std::move(excessObjRefsColl1), "collection1ExcessObjects");
    e.put(std::move(excessObjRefsColl2), "collection2ExcessObjects");
  }
  if (checkObject_) {
    e.put(std::move(matchRefPairs), "objectMatches");
    e.put(std::move(mismatchRefPairs), "objectMismatches");
  }
}

//define plugins for different L1T objects
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
typedef L1TStage2ObjectComparison<GlobalAlgBlk> L1TStage2GlobalAlgBlkComparison;
typedef L1TStage2ObjectComparison<l1t::EGamma> L1TStage2EGammaComparison;
typedef L1TStage2ObjectComparison<l1t::Tau> L1TStage2TauComparison;
typedef L1TStage2ObjectComparison<l1t::Jet> L1TStage2JetComparison;
typedef L1TStage2ObjectComparison<l1t::EtSum> L1TStage2EtSumComparison;
typedef L1TStage2ObjectComparison<l1t::Muon> L1TStage2MuonComparison;
typedef L1TStage2ObjectComparison<l1t::RegionalMuonCand> L1TStage2RegionalMuonCandComparison;
DEFINE_FWK_MODULE(L1TStage2GlobalAlgBlkComparison);
DEFINE_FWK_MODULE(L1TStage2EGammaComparison);
DEFINE_FWK_MODULE(L1TStage2TauComparison);
DEFINE_FWK_MODULE(L1TStage2JetComparison);
DEFINE_FWK_MODULE(L1TStage2EtSumComparison);
DEFINE_FWK_MODULE(L1TStage2MuonComparison);
DEFINE_FWK_MODULE(L1TStage2RegionalMuonCandComparison);
