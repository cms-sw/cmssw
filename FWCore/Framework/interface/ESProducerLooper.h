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
// $Id: ESProducerLooper.h,v 1.1 2006/07/23 01:24:33 valya Exp $
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
  class ESProducerLooper : public ESProducer, public EventSetupRecordIntervalFinder, public EDLooper
{

   public:
      ESProducerLooper();
      //virtual ~ESProducerLooper();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      virtual std::set<eventsetup::EventSetupRecordKey> modifyingRecords() const;
      // ---------- member functions ---------------------------

   protected:
      void setIntervalFor(const eventsetup::EventSetupRecordKey& iKey,
                          const IOVSyncValue& iTime, 
                          ValidityInterval& oInterval);
        
      //use this to 'snoop' on what records are being used by the Producer
      virtual void registerFactoryWithKey(const eventsetup::EventSetupRecordKey& iRecord ,
                                          std::auto_ptr<eventsetup::ProxyFactoryBase>& iFactory,
                                          const std::string& iLabel= std::string() );
private:
      ESProducerLooper(const ESProducerLooper&); // stop default

      const ESProducerLooper& operator=(const ESProducerLooper&); // stop default

      // ---------- member data --------------------------------

};
}

#endif
