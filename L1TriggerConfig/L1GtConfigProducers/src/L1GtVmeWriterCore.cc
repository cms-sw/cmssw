/**
 * \class L1GtVmeWriterCore
 * 
 * 
 * Description: core class to write the VME xml file.  
 *
 * Implementation:
 *    Core class to write the VME xml file: L1GtVmeWriter is an EDM wrapper for this class.
 *    L1GtVmeWriterCore can also be used in L1 Trigger Supervisor framework,  with another 
 *    wrapper - it must be therefore EDM-framework free.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVmeWriterCore.h"

// system include files
#include <iostream>

// user include files
//   base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"


// constructor
L1GtVmeWriterCore::L1GtVmeWriterCore(const std::string& outputDir,
        const std::string& vmeXmlFile) :
    m_outputDir(outputDir), m_vmeXmlFile(vmeXmlFile)
{

    // empty now

}

// destructor
L1GtVmeWriterCore::~L1GtVmeWriterCore()
{
    
    // empty now
    
}

// loop over events
void L1GtVmeWriterCore::writeVME()
{

    // just for test
    std::cout << std::endl;

    std::cout << "<" << m_xmlTagVme << ">" << std::endl;
    std::cout << std::endl;
    std::cout << "</" << m_xmlTagVme << ">" << std::endl;

}

// static class members


