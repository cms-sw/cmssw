/**
 * \class L1GlobalTriggerSetup
 * 
 * 
 * 
 * Description: see header file 
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

// system include files
#include <iostream>
#include <string>

// user include files
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

// constructor
L1GlobalTriggerSetup::L1GlobalTriggerSetup(
    L1GlobalTrigger& gt, 
    const edm::ParameterSet& iConfig) 
    : m_GT(gt)
    {
    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ << std::endl; 
    
    m_pSet = &iConfig;

}

// destructor
L1GlobalTriggerSetup::~L1GlobalTriggerSetup() {
    if (m_gtConfig) { delete m_gtConfig; }
    m_gtConfig = 0;
}

//
void L1GlobalTriggerSetup::setTriggerMenu(std::string& menuDir) {
    
    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ << std::endl;

    if ( menuDir == "" ) {
        menuDir = m_pSet->getParameter<std::string>("triggerMenu");    
    }

    std::string defXmlFileName = m_pSet->getParameter<std::string>("xmlConfig");
    edm::FileInPath f1("L1Trigger/GlobalTrigger/data/" + menuDir + "/" + defXmlFileName);        

    std::string defXmlFile = f1.fullPath();
    
    if (f1.isLocal()) {
        LogDebug("L1GlobalTriggerSetup") 
            << "FileInPath: XML configuration file: \n  " << defXmlFile 
            << std::endl;        
	} else {
        // FileInPath throw exception if the file is not found;
        // here the exception is thrown only if the file was deleted in the meantime
        throw cms::Exception("FileInPathError")
            << "\nError: XML configuration file not found. Requested file:\n  "
            << defXmlFile << std::endl;
    }
                        
    std::string vmeXmlFile;
    
    std::string vmeXmlFileName = m_pSet->getParameter<std::string>("xmlVME");
    if (vmeXmlFileName != "") {
        edm::FileInPath f2("L1Trigger/GlobalTrigger/data/" + menuDir + "/" + vmeXmlFileName);        

        vmeXmlFile = f2.fullPath();

        if (f2.isLocal()) {
            LogDebug("L1GlobalTriggerSetup") 
                << "FileInPath: XML File for VME-bus preamble: \n  " << vmeXmlFile 
                << std::endl;        
        } else {
            // FileInPath throw exception if the file is not found;
            // here the exception is thrown only if the file was deleted in the meantime
            throw cms::Exception("FileInPathError")
                << "\nError: XML File for VME-bus preamble not found. Requested file:\n  "
                << vmeXmlFile << std::endl;
        }
    } 

    // get the new trigger configuration
    if(!m_gtConfig) m_gtConfig = new L1GlobalTriggerConfig(&m_GT, defXmlFile, vmeXmlFile);

    // set the input mask: bit 0 GCT, bit 1 GMT
    // one converts from vector as there is no "bitset" parameter type
    std::vector<unsigned int> inMaskV = m_pSet->getParameter<std::vector<unsigned int> >("inputMask");
    
    std::bitset<L1GlobalTriggerConfig::NumberInputModules> inMask;
    for (unsigned int i = 0; i < L1GlobalTriggerConfig::NumberInputModules; ++i) {
		if ( inMaskV[i] ) inMask.set(i);
	}
    
    m_gtConfig->setInputMask(inMask);

    if ( m_gtConfig->getInputMask()[0] ) {

        edm::LogVerbatim("L1GlobalTriggerSetup") 
            << "\n**** Calorimeter input disabled! \n     inputMask[0] = " 
            << m_gtConfig->getInputMask()[0] 
            << "     All candidates empty." << "\n**** \n"
            << std::endl;
    } 
    
    if ( m_gtConfig->getInputMask()[1] ) {

        edm::LogVerbatim("L1GlobalTriggerSetup") 
            << "\n**** Global Muon input disabled! \n  inputMask[1] = " 
            << m_gtConfig->getInputMask()[1]
            << "     All candidates empty." << "\n**** \n"
            << std::endl;
    }     
    
    // TODO FIXME add triggerMask stuff
            
}

// static class members
const edm::ParameterSet* L1GlobalTriggerSetup::m_pSet = 0;
L1GlobalTriggerConfig* L1GlobalTriggerSetup::m_gtConfig = 0;
