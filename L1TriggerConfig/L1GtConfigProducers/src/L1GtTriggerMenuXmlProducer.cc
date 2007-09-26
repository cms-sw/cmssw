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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlProducer.h"

// system include files
#include <memory>

#include "boost/shared_ptr.hpp"


// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

// constructor(s)
L1GtTriggerMenuXmlProducer::L1GtTriggerMenuXmlProducer(
    const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtTriggerMenuXmlProducer::produceGtTriggerMenu);


    // now do what ever other initialization is needed


    // directory in /data/Luminosity for the trigger menu
    std::string menuDir = parSet.getParameter<std::string>("TriggerMenuLuminosity");


    // def.xml file
    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");

    edm::FileInPath f1("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/" +
                       menuDir + "/" + defXmlFileName);

    m_defXmlFile = f1.fullPath();


    // vme.xml file
    std::string vmeXmlFileName = parSet.getParameter<std::string>("VmeXmlFile");

    if (vmeXmlFileName != "") {
        edm::FileInPath f2("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/" +
                           menuDir + "/" + vmeXmlFileName);

        m_vmeXmlFile = f2.fullPath();

    }

    edm::LogInfo("L1GtConfigProducers")
    << "\n\nL1 Trigger Menu: "
    << "\n\n  def.xml file: \n    " << m_defXmlFile
    << "\n\n  vme.xml file: \n    " << m_vmeXmlFile
    << "\n\n"
    << std::endl;

}

// destructor
L1GtTriggerMenuXmlProducer::~L1GtTriggerMenuXmlProducer()
{

    // empty

}


// member functions

// method called to produce the data
boost::shared_ptr<L1GtTriggerMenu> L1GtTriggerMenuXmlProducer::produceGtTriggerMenu(
    const L1GtTriggerMenuRcd& iRecord)
{

    boost::shared_ptr<L1GtTriggerMenu> pL1GtTriggerMenu =
        boost::shared_ptr<L1GtTriggerMenu>( new L1GtTriggerMenu() );


    //    // set the number of physics trigger algorithms
    //    pL1GtTriggerMenu->setGtNumberPhysTriggers(m_numberPhysTriggers);


    return pL1GtTriggerMenu ;
}

