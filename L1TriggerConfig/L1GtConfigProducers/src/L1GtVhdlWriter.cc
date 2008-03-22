/**
 * \class L1GtVhdlWriter
 * 
 * 
 * Description: write VHDL templates for the L1 GT.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriter.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// forward declarations

// constructor(s)
L1GtVhdlWriter::L1GtVhdlWriter(const edm::ParameterSet& parSet)
{

    // directory in /data for the VHDL templates
    std::string vhdlDir = parSet.getParameter<std::string>("VhdlTemplatesDir");

//    // def.xml file
//    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");
//
//    edm::FileInPath f1("L1TriggerConfig/L1GtConfigProducers/data/" +
//                       vhdlDir + "/" + defXmlFileName);
//
//    m_defXmlFile = f1.fullPath();



    edm::LogInfo("L1GtConfigProducers")
    << "\n\nL1 GT VHDL directory: "
    << vhdlDir 
    << "\n\n"
    << std::endl;

}

// destructor
L1GtVhdlWriter::~L1GtVhdlWriter()
{
    // empty
}

// loop over events
void L1GtVhdlWriter::analyze(
    const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    edm::ESHandle< L1GtTriggerMenu > l1GtMenu ;
    evSetup.get< L1GtTriggerMenuRcd >().get( l1GtMenu ) ;

    std::vector<ConditionMap> conditionMap = l1GtMenu->gtConditionMap();
    AlgorithmMap algorithmMap = l1GtMenu->gtAlgorithmMap();

    // print with various level of verbosities

    int printVerbosity = 0;
    l1GtMenu->print(std::cout, printVerbosity);


}
