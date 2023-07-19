#ifndef Framework_ESProducerLooper_h
#define Framework_ESProducerLooper_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducerLooper
//
/**\class ESProducerLooper ESProducerLooper.h FWCore/Framework/interface/ESProducerLooper.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Jul 17 09:03:32 EDT 2006
//

// system include files
#include <memory>
#include <set>
#include <string>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations
namespace edm {
  class ESProducerLooper : public ESProducer, public EventSetupRecordIntervalFinder, public EDLooper {
  public:
    ESProducerLooper();
    ESProducerLooper(const ESProducerLooper&) = delete;                   // stop default
    const ESProducerLooper& operator=(const ESProducerLooper&) = delete;  // stop default

    //virtual ~ESProducerLooper();

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    std::set<eventsetup::EventSetupRecordKey> modifyingRecords() const override;
    // ---------- member functions ---------------------------

  protected:
    void setIntervalFor(const eventsetup::EventSetupRecordKey& iKey,
                        const IOVSyncValue& iTime,
                        ValidityInterval& oInterval) override;

    //use this to 'snoop' on what records are being used by the Producer
    void registerFactoryWithKey(const eventsetup::EventSetupRecordKey& iRecord,
                                std::unique_ptr<eventsetup::ESProductResolverFactoryBase> iFactory,
                                const std::string& iLabel = std::string()) override;

  private:
    // ---------- member data --------------------------------
  };
}  // namespace edm

#endif
