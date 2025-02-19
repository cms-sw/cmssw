#ifndef CondTools_L1Trigger_L1TriggerKeyOnlineProd_h
#define CondTools_L1Trigger_L1TriggerKeyOnlineProd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TriggerKeyOnlineProd
// 
/**\class L1TriggerKeyOnlineProd L1TriggerKeyOnlineProd.h CondTools/L1Trigger/interface/L1TriggerKeyOnlineProd.h

 Description: Get L1TriggerKey objects from all subsystems and collate.

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Mar  2 03:04:19 CET 2008
// $Id: L1TriggerKeyOnlineProd.h,v 1.3 2008/09/19 19:31:13 wsun Exp $
//

// system include files
#include <memory>
#include <vector>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

class L1TriggerKeyOnlineProd : public edm::ESProducer {
   public:
      L1TriggerKeyOnlineProd(const edm::ParameterSet&);
      ~L1TriggerKeyOnlineProd();

      typedef boost::shared_ptr<L1TriggerKey> ReturnType;

      ReturnType produce(const L1TriggerKeyRcd&);
   private:
      // ----------member data ---------------------------
      std::vector< std::string > m_subsystemLabels ;
};

#endif
