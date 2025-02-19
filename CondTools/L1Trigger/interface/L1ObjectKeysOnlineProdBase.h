#ifndef CondTools_L1Trigger_L1ObjectKeysOnlineProdBase_h
#define CondTools_L1Trigger_L1ObjectKeysOnlineProdBase_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ObjectKeysOnlineProdBase
// 
/**\class L1ObjectKeysOnlineProdBase L1ObjectKeysOnlineProdBase.h CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h

 Description: Abstract base class for producers that navigate OMDS to get
 object keys for a given subsystem key.  Each base class should be configured
 (via ParameterSet) with a different subsystem label.

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Fri Aug 22 19:47:58 CEST 2008
// $Id: L1ObjectKeysOnlineProdBase.h,v 1.1 2008/09/19 19:22:58 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

// forward declarations

class L1ObjectKeysOnlineProdBase : public edm::ESProducer {
   public:
      L1ObjectKeysOnlineProdBase(const edm::ParameterSet&);
      ~L1ObjectKeysOnlineProdBase();

      typedef boost::shared_ptr<L1TriggerKey> ReturnType;

      ReturnType produce(const L1TriggerKeyRcd&);

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) = 0 ;
   private:
      // ----------member data ---------------------------
 protected:
      l1t::OMDSReader m_omdsReader ;
};

#endif
