#ifndef CommonTools_RecoAlgos_RangeMapOwnVectorToVectorConverter_h
#define CommonTools_RecoAlgos_RangeMapOwnVectorToVectorConverter_h
// -*- C++ -*-
//
// Package:     CommonTools/RecoAlgos
// Class  :     RangeMapOwnVectorToVectorConverter
//
/**\class RangeMapOwnVectorToVectorConverter RangeMapOwnVectorToVectorConverter.h "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 25 Sep 2023 14:10:44 GMT
//

// system include files
#include <vector>

// user include files
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

// forward declarations

namespace reco {
  template <typename ID, typename T>
  class RangeMapOwnVectorToVectorConverter : public edm::global::EDProducer<> {
  public:
    RangeMapOwnVectorToVectorConverter(edm::ParameterSet const& iPSet)
        : get_(consumes(iPSet.getParameter<edm::InputTag>("get"))), put_(produces()) {}

    RangeMapOwnVectorToVectorConverter(const RangeMapOwnVectorToVectorConverter&) = delete;  // stop default
    const RangeMapOwnVectorToVectorConverter& operator=(const RangeMapOwnVectorToVectorConverter&) =
        delete;  // stop default

    // ---------- const member functions ---------------------
    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const final {
      auto ownRange = iEvent.get(get_);

      edm::RangeMap<ID, std::vector<T>> vecRange;
      for (auto ids = ownRange.id_begin(); ids != ownRange.id_end(); ++ids) {
        auto range = ownRange.get(*ids);
        vecRange.put(*ids, range.first, range.second);
      }
      iEvent.emplace(put_, std::move(vecRange));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
      edm::ParameterSetDescription pset;
      pset.add<edm::InputTag>("get");

      iConfig.addDefault(pset);
    }

  private:
    // ---------- member data --------------------------------
    edm::EDGetTokenT<edm::RangeMap<ID, edm::OwnVector<T>>> get_;
    edm::EDPutTokenT<edm::RangeMap<ID, std::vector<T>>> put_;
  };
}  // namespace reco
#endif
