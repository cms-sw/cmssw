/**
 * \class L1GtVhdlWriter
 *
 *
 * Description: write VHDL templates for the L1 GT.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriter.h"

// system include files
#include <filesystem>
#include <iostream>
#include <sys/stat.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVmeWriterCore.h"

// constructor(s)
L1GtVhdlWriter::L1GtVhdlWriter(const edm::ParameterSet& parSet) {
  // directory in /data for the VHDL templates
  vhdlDir_ = parSet.getParameter<std::string>("VhdlTemplatesDir");
  outputDir_ = parSet.getParameter<std::string>("OutputDir");
  menuToken_ = esConsumes();

  if (vhdlDir_[vhdlDir_.length() - 1] != '/')
    vhdlDir_ += "/";
  if (outputDir_[outputDir_.length() - 1] != '/')
    outputDir_ += "/";

  //    // def.xml file
  //    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");
  //
  //    edm::FileInPath f1("L1TriggerConfig/L1GtConfigProducers/data/" +
  //                       vhdlDir + "/" + defXmlFileName);
  //
  //    m_defXmlFile = f1.fullPath();

  edm::LogInfo("L1GtConfigProducers") << "\n\nL1 GT VHDL directory: " << vhdlDir_ << "\n\n" << std::endl;
}

// loop over events
void L1GtVhdlWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1GtTriggerMenu> l1GtMenu = evSetup.getHandle(menuToken_);

  std::vector<ConditionMap> const& conditionMap = l1GtMenu->gtConditionMap();
  AlgorithmMap const& algorithmMap = l1GtMenu->gtAlgorithmMap();

  // print with various level of verbosities
  int printVerbosity = 0;
  l1GtMenu->print(std::cout, printVerbosity);

  //---------------------Here the VHDL files will be created---------------------------------------

  // information that will be delivered by the parser in future
  std::map<std::string, std::string> headerParameters;
  std::vector<std::string> channelVector;

  headerParameters["vhdl_path"] = "/vhdllibrarypath";
  headerParameters["designer_date"] = "20.05.1986";
  headerParameters["designer_name"] = "Philipp Wagner";
  headerParameters["version"] = "2.0";
  headerParameters["designer_comments"] = "produced in CMSSW";
  headerParameters["gtl_setup_name"] = "L1Menu2007NovGR";

  channelVector.reserve(10);
  channelVector.push_back("-- ca1: ieg");
  channelVector.push_back("-- ca2: eg");
  channelVector.push_back("-- ca3: jet");
  channelVector.push_back("-- ca4: fwdjet");
  channelVector.push_back("-- ca5: tau");
  channelVector.push_back("-- ca6: esums");
  channelVector.push_back("-- ca7: jet_cnts");
  channelVector.push_back("-- ca8: free");
  channelVector.push_back("-- ca9: free");
  channelVector.push_back("-- ca10: free");

  // check, weather output directory exists and create it on the fly if not
  if (std::filesystem::is_directory(outputDir_)) {
    std::cout << std::endl << "Ok - Output directory exists!" << std::endl;
  } else {
    if (!mkdir(outputDir_.c_str(), 0666))
      std::cout << std::endl << "Directory: " << outputDir_ << " has been created!" << std::endl;
    else
      std::cout << std::endl << "Error while creating directory: " << outputDir_ << " !" << std::endl;
  }

  // prepare a core with common header
  L1GtVhdlWriterCore vhdlWriter(vhdlDir_, outputDir_, true);
  vhdlWriter.buildCommonHeader(headerParameters, channelVector);
  // write the firmware
  if (vhdlWriter.makeFirmware(conditionMap, algorithmMap)) {
    std::cout << std::endl
              << std::endl
              << "***********************   I'm ready ;-) **************************" << std::endl
              << std::endl
              << "You can find the firmware in dircetory: " << outputDir_ << std::endl
              << std::endl
              << "******************************************************************" << std::endl;
  }

  // Create the VME - XML
  std::string vmeFile = "vme.xml";

  L1GtVmeWriterCore vmeWriter(outputDir_, vmeFile);
  vmeWriter.writeVME(conditionMap, vhdlWriter.getCond2IntMap(), vhdlWriter.retrunCommonHeader());
}
