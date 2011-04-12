#ifndef DTTPGConfigProducers_DTConfigDBProducer_h
#define DTTPGConfigProducers_DTConfigDBProducer_h

// -*- C++ -*-
//
// Package:     DTTPGConfigProducers
// Class:       DTConfigDBProducer
// 
/**\class  DTConfigDBProducer  DTConfigDBProducer.h L1TriggerConfig/DTTPGConfigProducers/interface/DTConfigDBProducer.h

 Description: A Producer for the DT config, data retrieved from DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sara Vanini
//         Created:  September 2008
//
//


// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

// @@@ remove headers
//#include "CondFormats/DTObjects/interface/DTConfigList.h"
////#include "CondTools/DT/interface/DTConfigHandler.h"
////#include "CondTools/DT/interface/DTDBSession.h"
//#include "CondFormats/DTObjects/interface/DTConfig1Handler.h"
//#include "CondFormats/DTObjects/interface/DTDB1Session.h" 

//
// class declaration
//

//class DTConfigDBProducer : public edm::ESProducer , public edm::EventSetupRecordIntervalFinder{
class DTConfigDBProducer : public edm::ESProducer{

public:
  DTConfigDBProducer(const edm::ParameterSet&);
  ~DTConfigDBProducer();
  
  std::auto_ptr<DTConfigManager> produce(const DTConfigManagerRcd&);
  
  int readDTCCBConfig(const DTConfigManagerRcd& iRecord);
  void configFromCfg();
  
private:
  std::string mapEntryName(const DTChamberId & chambid) const;

  // ----------member data ---------------------------
  edm::ParameterSet m_ps;
  DTConfigManager* m_manager;
  
  // debug flags
  bool m_debugDB; 
  int m_debugBti;
  int m_debugTraco;
  bool m_debugTSP;
  bool m_debugTST;
  bool m_debugTU;
  bool m_debugSC;
  bool m_debugLUTs;  

  // general DB requests
  bool m_TracoLutsFromDB;
  bool m_UseBtiAcceptParam;

// @@@ remove
//  std::string contact;
//  std::string auth_path;
//  std::string catalog;
//  std::string token;
//  bool local;
  bool cfgConfig;

// @@@ remove
//  DTDB1Session* session;
//  const DTConfigList* rs;
//  DTConfig1Handler* ri;
  
  bool flagDBBti, flagDBTraco, flagDBTSS, flagDBTSM, flagDBLUTS;  
};

#endif
