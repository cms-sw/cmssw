#ifndef L1TGlobal_TriggerMenuXmlProducer_h
#define L1TGlobal_TriggerMenuXmlProducer_h

/**
 * \class TriggerMenuXmlProducer
 * 
 * 
 * Description: ESProducer for the L1 Trigger Menu from an XML file .  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TGlobalTriggerMenuRcd.h"

// forward declarations

namespace l1t {

// class declaration
class TriggerMenuXmlProducer : public edm::ESProducer
{

public:

    /// constructor
    TriggerMenuXmlProducer(const edm::ParameterSet&);

    /// destructor
    ~TriggerMenuXmlProducer();


    /// public methods

    /// L1 GT parameters
    boost::shared_ptr<TriggerMenu> produceGtTriggerMenu(
        const L1TGlobalTriggerMenuRcd&);

private:

    /// XML file for Global Trigger menu (def.xml)
    std::string m_defXmlFile;

    /// XML file for Global Trigger VME configuration (vme.xml)
    std::string m_vmeXmlFile;


};

}
#endif
