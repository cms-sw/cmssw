/**
 * \class L1GtVmeWriter
 * 
 * 
 * Description: write the VME xml file.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVmeWriter.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVmeWriterCore.h"

// constructor
L1GtVmeWriter::L1GtVmeWriter(const edm::ParameterSet& parSet)
{
    
    // empty now - read input parameters
    
}

// destructor
L1GtVmeWriter::~L1GtVmeWriter()
{
    
    // empty now
    
}

// loop over events
void L1GtVmeWriter::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup)
{

   L1GtVmeWriterCore vmeWriter(m_outputDir, m_vmeXmlFile);
   vmeWriter.writeVME();
   
}

// static class members


