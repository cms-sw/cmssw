#ifndef CondTools_L1Trigger_L1TriggerKeyDummyProd_h
#define CondTools_L1Trigger_L1TriggerKeyDummyProd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TriggerKeyDummyProd
// 
/**\class L1TriggerKeyDummyProd L1TriggerKeyDummyProd.h CondTools/L1Trigger/interface/L1TriggerKeyDummyProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Mar  1 01:12:16 CET 2008
// $Id: L1TriggerKeyDummyProd.h,v 1.1 2008/03/03 21:52:18 wsun Exp $
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

// forward declarations

class L1TriggerKeyDummyProd : public edm::ESProducer {
   public:
      L1TriggerKeyDummyProd(const edm::ParameterSet&);
      ~L1TriggerKeyDummyProd();

      typedef boost::shared_ptr<L1TriggerKey> ReturnType;

      ReturnType produce(const L1TriggerKeyRcd&);
   private:
      // ----------member data ---------------------------
      L1TriggerKey m_key ;
};


#endif
