#ifndef DTTPGConfigProducers_DTConfigTrivialProducer_h
#define DTTPGConfigProducers_DTConfigTrivialProducer_h

// -*- C++ -*-
//
// Package:     DTTPGConfigProducers
// Class:       DTConfigTrivialProducer
//
/**\class  DTConfigTrivialProducer  DTConfigTrivialProducer.h
 L1TriggerConfig/DTTPGConfigProducers/interface/DTConfigTrivialProducer.h

 Description: A Producer for the DT config available via EventSetup

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sara Vanini
//         Created:  March 2007
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

//
// class declaration
//

class DTConfigTrivialProducer : public edm::ESProducer {
public:
  //! Constructor
  DTConfigTrivialProducer(const edm::ParameterSet &);

  //! destructor
  ~DTConfigTrivialProducer() override;

  //! ES produce method
  std::unique_ptr<DTConfigManager> produce(const DTConfigManagerRcd &);

private:
  //! Build Config Manager
  void buildManager();

  //! Build Config Pedestals
  DTConfigPedestals buildTrivialPedestals();

  std::string mapEntryName(const DTChamberId &chambid) const;

  bool m_debug;
  edm::ParameterSet m_ps;
  DTConfigManager *m_manager;
  DTTPGParameters *m_tpgParams;
};

#endif
