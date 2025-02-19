#ifndef CondTools_L1Trigger_L1TriggerKeyListDummyProd_h
#define CondTools_L1Trigger_L1TriggerKeyListDummyProd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TriggerKeyListDummyProd
// 
/**\class L1TriggerKeyListDummyProd L1TriggerKeyListDummyProd.h CondTools/L1Trigger/interface/L1TriggerKeyListDummyProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Mar  1 05:06:43 CET 2008
// $Id: L1TriggerKeyListDummyProd.h,v 1.1 2008/03/03 21:52:18 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

class L1TriggerKeyListDummyProd : public edm::ESProducer {
   public:
      L1TriggerKeyListDummyProd(const edm::ParameterSet&);
      ~L1TriggerKeyListDummyProd();

      typedef boost::shared_ptr<L1TriggerKeyList> ReturnType;

      ReturnType produce(const L1TriggerKeyListRcd&);
   private:
      // ----------member data ---------------------------
};

#endif
