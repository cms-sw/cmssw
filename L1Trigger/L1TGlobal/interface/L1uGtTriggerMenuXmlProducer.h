#ifndef L1TGlobal_L1uGtTriggerMenuXmlProducer_h
#define L1TGlobal_L1uGtTriggerMenuXmlProducer_h

/**
 * \class L1uGtTriggerMenuXmlProducer
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

#include "CondFormats/L1TObjects/interface/L1uGtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

namespace l1t {

// class declaration
class L1uGtTriggerMenuXmlProducer : public edm::ESProducer
{

public:

    /// constructor
    L1uGtTriggerMenuXmlProducer(const edm::ParameterSet&);

    /// destructor
    ~L1uGtTriggerMenuXmlProducer();


    /// public methods

    /// L1 GT parameters
    boost::shared_ptr<L1uGtTriggerMenu> produceGtTriggerMenu(
        const L1GtTriggerMenuRcd&);

private:

    /// XML file for Global Trigger menu (def.xml)
    std::string m_defXmlFile;

    /// XML file for Global Trigger VME configuration (vme.xml)
    std::string m_vmeXmlFile;


};

}
#endif
