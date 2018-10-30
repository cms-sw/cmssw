#ifndef CondTools_L1Trigger_L1SubsystemKeysOnlineProd_h
#define CondTools_L1Trigger_L1SubsystemKeysOnlineProd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1SubsystemKeysOnlineProd
// 
/**\class L1SubsystemKeysOnlineProd L1SubsystemKeysOnlineProd.h CondTools/L1Trigger/interface/L1SubsystemKeysOnlineProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Thu Aug 21 19:30:30 CEST 2008
// $Id: L1SubsystemKeysOnlineProd.h,v 1.1 2008/09/19 19:22:59 wsun Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

// forward declarations

class L1SubsystemKeysOnlineProd : public edm::ESProducer {
   public:
      L1SubsystemKeysOnlineProd(const edm::ParameterSet&);
      ~L1SubsystemKeysOnlineProd() override;

      typedef std::unique_ptr<L1TriggerKey> ReturnType;

      ReturnType produce(const L1TriggerKeyRcd&);
   private:
      // ----------member data ---------------------------
      std::string m_tscKey ;
      l1t::OMDSReader m_omdsReader ;
      bool m_forceGeneration ;
};

#endif
