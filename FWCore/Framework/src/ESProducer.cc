// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducer
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Apr 16 10:19:37 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"

//
// constants, enums and typedefs
//
namespace edm {
  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  ESProducer::ESProducer() : consumesInfos_{} {}

  // ESProducer::ESProducer(const ESProducer& rhs)
  // {
  //    // do actual copying here;
  // }

  ESProducer::~ESProducer() noexcept(false) {}

  //
  // assignment operators
  //
  // const ESProducer& ESProducer::operator=(const ESProducer& rhs)
  // {
  //   //An exception safe implementation is
  //   ESProducer temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  void ESProducer::updateLookup(eventsetup::ESRecordsToProxyIndices const& iProxyToIndices) {
    itemsToGetFromRecords_.reserve(consumesInfos_.size());
    recordsUsedDuringGet_.reserve(consumesInfos_.size());

    for (auto& info : consumesInfos_) {
      auto& items = itemsToGetFromRecords_.emplace_back();
      items.reserve(info->size());
      auto& records = recordsUsedDuringGet_.emplace_back();
      records.reserve(info->size());
      for (auto& proxyInfo : *info) {
        auto index = iProxyToIndices.indexInRecord(std::get<0>(proxyInfo), std::get<1>(proxyInfo));
        if (index != eventsetup::ESRecordsToProxyIndices::missingProxyIndex()) {
          if (not std::get<2>(proxyInfo).empty()) {
            auto component = iProxyToIndices.component(std::get<0>(proxyInfo), std::get<1>(proxyInfo));
            if (nullptr == component) {
              index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
            } else {
              if (component->label_.empty()) {
                if (component->type_ != std::get<2>(proxyInfo)) {
                  index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
                }
              } else if (component->label_ != std::get<2>(proxyInfo)) {
                index = eventsetup::ESRecordsToProxyIndices::missingProxyIndex();
              }
            }
          }
        }
        items.push_back(index);
        if (index != eventsetup::ESRecordsToProxyIndices::missingProxyIndex()) {
          records.push_back(iProxyToIndices.recordIndexFor(std::get<0>(proxyInfo)));
        }
      }
    }
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}  // namespace edm
