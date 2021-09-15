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
// $Id: L1TriggerKeyOnlineProd.h,v 1.2 2008/09/12 04:50:59 wsun Exp $
//

// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations
class L1TriggerKey;
class L1TriggerKeyRcd;

class L1TriggerKeyOnlineProd : public edm::ESProducer {
public:
  L1TriggerKeyOnlineProd(const edm::ParameterSet&);
  ~L1TriggerKeyOnlineProd() override;

  typedef std::unique_ptr<L1TriggerKey> ReturnType;

  ReturnType produce(const L1TriggerKeyRcd&);

private:
  // ----------member data ---------------------------
  std::vector<std::string> m_subsystemLabels;
  edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd> l1TriggerKeyToken_;
  std::vector<edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd>> l1TriggerKeyTokenVec_;
};

#endif
