#ifndef CondTools_L1TriggerExt_L1ObjectKeysOnlineProdBaseExt_h
#define CondTools_L1TriggerExt_L1ObjectKeysOnlineProdBaseExt_h

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

// forward declarations

class L1ObjectKeysOnlineProdBaseExt : public edm::ESProducer {
   public:
      L1ObjectKeysOnlineProdBaseExt(const edm::ParameterSet&);
      ~L1ObjectKeysOnlineProdBaseExt();

      typedef boost::shared_ptr<L1TriggerKeyExt> ReturnType;

      ReturnType produce(const L1TriggerKeyExtRcd&);

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) = 0 ;
   private:
      // ----------member data ---------------------------
 protected:
      l1t::OMDSReader m_omdsReader ;
};

#endif
