// -*- C++ -*-
//
// Package:    L1Trigger/L1TCommon
// Class:      L1TComparisonResultFilter
// 
/**\class L1TComparisonResultFilter L1TComparisonResultFilter.cc L1Trigger/L1TCommon/plugins/L1TComparisonResultFilter.cc

 Description: Filters on result of L1T object comparison. Events where the collections match are passing.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis
//         Created:  Fri, 19 Jan 2018 14:08:35 GMT
//
//


// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

//
// class declaration
//

template <typename T>
class L1TComparisonResultFilter : public edm::stream::EDFilter<> {
   public:
      explicit L1TComparisonResultFilter(const edm::ParameterSet&);
      ~L1TComparisonResultFilter() override = default;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void beginStream(edm::StreamID) override { };
      bool filter(edm::Event&, const edm::EventSetup&) override;
      void endStream() override { };

      // ----------member data ---------------------------
      edm::InputTag inputTag_;

      edm::EDGetTokenT<l1t::ObjectRefPairBxCollection<T>> pairCollToken_;
      edm::EDGetTokenT<l1t::ObjectRefBxCollection<T>> coll1Token_;
      edm::EDGetTokenT<l1t::ObjectRefBxCollection<T>> coll2Token_;

      // Maximal mismatch sizes
      // Only analyse the corresponding property if max is not negative
      const int maxBxRangeDiff_;
      const int maxExcess_;
      const int maxSize_;

      // Invert the final result
      const bool invert_;
};

//
// constructors and destructor
//
template <typename T>
L1TComparisonResultFilter<T>::L1TComparisonResultFilter(const edm::ParameterSet& iConfig)
 : inputTag_(iConfig.getParameter<edm::InputTag>("objComparisonColl")),
   maxBxRangeDiff_(iConfig.getParameter<int>("maxBxRangeDiff")),
   maxExcess_(iConfig.getParameter<int>("maxExcess")),
   maxSize_(iConfig.getParameter<int>("maxSize")),
   invert_(iConfig.getParameter<bool>("invert"))
{
   if (maxBxRangeDiff_ > -1 || maxExcess_ > -1) {
      // Take all input tags needed from the same module
      edm::InputTag coll1Tag(inputTag_.label(), "collection1ExcessObjects");
      edm::InputTag coll2Tag(inputTag_.label(), "collection2ExcessObjects");
      coll1Token_ = consumes<l1t::ObjectRefBxCollection<T>>(coll1Tag);
      coll2Token_ = consumes<l1t::ObjectRefBxCollection<T>>(coll2Tag);
   }
   if (maxSize_ > -1) {
      pairCollToken_ = consumes<l1t::ObjectRefPairBxCollection<T>>(inputTag_);
   }
}


//
// member functions
//

// ------------ method called on each new Event  ------------
template <typename T>
bool
L1TComparisonResultFilter<T>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   bool pass = true;

   if (maxBxRangeDiff_ > -1 || maxExcess_ > -1) {
      edm::Handle<l1t::ObjectRefBxCollection<T>> bxColl1;
      edm::Handle<l1t::ObjectRefBxCollection<T>> bxColl2;
      iEvent.getByToken(coll1Token_, bxColl1);
      iEvent.getByToken(coll2Token_, bxColl2);

      // Check if BX ranges match
      if (maxBxRangeDiff_ > -1) {
         const auto bxRange1 = bxColl1->getLastBX() - bxColl1->getFirstBX() + 1;
         const auto bxRange2 = bxColl2->getLastBX() - bxColl2->getFirstBX() + 1;
         pass &= (std::abs(bxRange1 - bxRange2) <= maxBxRangeDiff_);
      }

      // If the BX range check passed check if number of objects per BX matches
      if (pass && maxExcess_ > -1) {
         const auto firstCommonBx = std::max(bxColl1->getFirstBX(), bxColl2->getFirstBX());
         const auto lastCommonBx = std::min(bxColl1->getLastBX(), bxColl2->getLastBX());
         for (auto iBx = firstCommonBx; iBx <= lastCommonBx; ++iBx) {
            pass &= (bxColl1->size(iBx) <= static_cast<unsigned>(maxExcess_));
            pass &= (bxColl2->size(iBx) <= static_cast<unsigned>(maxExcess_));
            if (!pass) break;
         }
      }
   }

   // If the previous checks passed check if there are mismatching objects
   if (pass && maxSize_ > -1) {
      edm::Handle<l1t::ObjectRefPairBxCollection<T>> bxPairColl;
      iEvent.getByToken(pairCollToken_, bxPairColl);

      pass &= (bxPairColl->size() <= static_cast<unsigned>(maxSize_));
   }

   if (invert_) {
      return !pass;
   }
   return pass;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void
L1TComparisonResultFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("objComparisonColl", edm::InputTag("objComparisonColl"))->setComment("Object comparison collection");
  desc.add<int>("maxBxRangeDiff", -1)->setComment("Maximal BX range difference");
  desc.add<int>("maxExcess", -1)->setComment("Maximal allowed excess objects per BX");
  desc.add<int>("maxSize", -1)->setComment("Maximal allowed mismatches");
  desc.add<bool>("invert", false)->setComment("Invert final result");
  descriptions.addWithDefaultLabel(desc);
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
typedef L1TComparisonResultFilter<GlobalAlgBlk> L1TGlobalAlgBlkComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::EGamma> L1TEGammaComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::Tau> L1TTauComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::Jet> L1TJetComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::EtSum> L1TEtSumComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::Muon> L1TMuonComparisonResultFilter;
typedef L1TComparisonResultFilter<l1t::RegionalMuonCand> L1TRegionalMuonCandComparisonResultFilter;
DEFINE_FWK_MODULE(L1TGlobalAlgBlkComparisonResultFilter);
DEFINE_FWK_MODULE(L1TEGammaComparisonResultFilter);
DEFINE_FWK_MODULE(L1TTauComparisonResultFilter);
DEFINE_FWK_MODULE(L1TJetComparisonResultFilter);
DEFINE_FWK_MODULE(L1TEtSumComparisonResultFilter);
DEFINE_FWK_MODULE(L1TMuonComparisonResultFilter);
DEFINE_FWK_MODULE(L1TRegionalMuonCandComparisonResultFilter);
