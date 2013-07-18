// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducerLooper
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Jul 17 09:34:30 EDT 2006
// $Id: ESProducerLooper.cc,v 1.1 2006/07/23 01:24:34 valya Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProducerLooper.h"

using namespace edm;
using namespace edm::eventsetup;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ESProducerLooper::ESProducerLooper()
{
}

// ESProducerLooper::ESProducerLooper(const ESProducerLooper& rhs)
// {
//    // do actual copying here;
// }
/*
ESProducerLooper::~ESProducerLooper()
{
}
*/
//
// assignment operators
//
// const ESProducerLooper& ESProducerLooper::operator=(const ESProducerLooper& rhs)
// {
//   //An exception safe implementation is
//   ESProducerLooper temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
ESProducerLooper::setIntervalFor(const EventSetupRecordKey&,
                                 const IOVSyncValue&, 
                                 ValidityInterval& oInterval)
{
  //since non of the dependent records are valid, I will create one that is valid
  // at the beginning of time BUT must also be checked every request
  //oInterval = ValidityInterval(IOVSyncValue::beginOfTime(),
  //                             IOVSyncValue::invalidIOVSyncValue());
 //   }
  //} else {
    //Give one valid for all time
    oInterval = ValidityInterval(IOVSyncValue::beginOfTime(),
                                 IOVSyncValue::endOfTime());
  //}
}

//use this to 'snoop' on what records are being used by the Producer
void 
ESProducerLooper::registerFactoryWithKey(const eventsetup::EventSetupRecordKey& iRecord ,
                                         std::auto_ptr<eventsetup::ProxyFactoryBase>& iFactory,
                                         const std::string& iLabel )
{
  findingRecordWithKey(iRecord);
  ESProxyFactoryProducer::registerFactoryWithKey(iRecord, iFactory,iLabel);
}

//
// const member functions
//
std::set<eventsetup::EventSetupRecordKey>
ESProducerLooper::modifyingRecords() const {
  return findingForRecords();
}
//
// static member functions
//
