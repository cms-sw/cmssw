#ifndef L1GtConfigProducers_L1GtTriggerMenuXmlProducer_h
#define L1GtConfigProducers_L1GtTriggerMenuXmlProducer_h

/**
 * \class L1GtTriggerMenuXmlProducer
 * 
 * 
 * Description: ESProducer for the L1 Trigger Menu from an XML file .  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMenuXmlProducer : public edm::ESProducer {
public:
  /// constructor
  L1GtTriggerMenuXmlProducer(const edm::ParameterSet&);

  /// destructor
  ~L1GtTriggerMenuXmlProducer() override;

  /// public methods

  /// L1 GT parameters
  std::unique_ptr<L1GtTriggerMenu> produceGtTriggerMenu(const L1GtTriggerMenuRcd&);

private:
  /// XML file for Global Trigger menu (def.xml)
  std::string m_defXmlFile;

  /// XML file for Global Trigger VME configuration (vme.xml)
  std::string m_vmeXmlFile;
};

#endif
