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

#include "CondTools/DT/interface/DTKeyedConfigCache.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

//
// class declaration
//

class DTConfigDBProducer : public edm::ESProducer{

 public :

  //! Constructor
  DTConfigDBProducer(const edm::ParameterSet&);

  //! Destructor
  ~DTConfigDBProducer();
  
  //! ES produce method
  std::auto_ptr<DTConfigManager> produce(const DTConfigManagerRcd&);
  
 private :

  //! Read DTTPG pedestal configuration
  void readDBPedestalsConfig(const DTConfigManagerRcd& iRecord, DTConfigManager & dttpgConfig);
  
  //! Read CCB string configuration
  int readDTCCBConfig(const DTConfigManagerRcd& iRecord, DTConfigManager & dttpgConfig);

  //! SV for debugging purpose ONLY
  void configFromCfg(DTConfigManager & dttpgConfig);

  //! SV for debugging purpose ONLY
  DTConfigPedestals buildTrivialPedestals();

  //! 110629 SV function for CCB configuration check
  int checkDTCCBConfig(DTConfigManager & dttpgConfig);

  std::string mapEntryName(const DTChamberId & chambid) const;

  // ----------member data ---------------------------
  edm::ParameterSet m_ps;
  
  // debug flags
  bool m_debugDB; 
  int  m_debugBti;
  int  m_debugTraco;
  bool m_debugTSP;
  bool m_debugTST;
  bool m_debugTU;
  bool m_debugSC;
  bool m_debugLUTs;
  bool m_debugPed;

  // general DB requests
  bool m_UseT0;

  bool cfgConfig;

  bool flagDBBti, flagDBTraco, flagDBTSS, flagDBTSM, flagDBLUTS;  

  DTKeyedConfigCache cfgCache;
};

#endif
