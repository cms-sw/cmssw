#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLElementaryMaterial.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"

int main(int argc, char* argv[]) {
  // Copied from example stand-alone program in Message Logger July 18, 2007
  std::string const kProgramName = argv[0];
  int rc = 0;

  try {
    // A.  Instantiate a plug-in manager first.
    edm::AssertHandler ah;

    // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     In particular, the job hangs as soon as the output buffer fills up.
    //     That's because, without the message service, there is no mechanism for
    //     emptying the buffers.
    std::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence =
        std::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->makePresence("SingleThreadMSPresence").release());

    // C.  Manufacture a configuration and establish it.
    std::string config =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('TEST')\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(5)\n"
        ")\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.JobReportService = cms.Service('JobReportService')\n"
        "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
        // "process.MessageLogger = cms.Service('MessageLogger')\n"
        "process.m1 = cms.EDProducer('IntProducer',\n"
        "    ivalue = cms.int32(11)\n"
        ")\n"
        "process.out = cms.OutputModule('PoolOutputModule',\n"
        "    fileName = cms.untracked.string('testStandalone.root')\n"
        ")\n"
        "process.p = cms.Path(process.m1)\n"
        "process.e = cms.EndPath(process.out)\n";

    // D.  Create the services.
    std::unique_ptr<edm::ParameterSet> params;
    edm::makeParameterSets(config, params);
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout << "Create DDLElementaryMaterial m" << std::endl;
    DDCompactView cpv;
    DDLElementRegistry locreg;
    DDLElementaryMaterial m(&locreg);
    //  <ElementaryMaterial name="Carbon" density="2.265*g/cm3" symbol=" " atomicWeight="12.011*g/mole" atomicNumber="6"/>

    std::cout << "Initialize names" << std::endl;
    std::vector<std::string> names;
    names.emplace_back("name");
    names.emplace_back("density");
    names.emplace_back("atomicWeight");
    names.emplace_back("atomicNumber");

    std::cout << "Initialize values" << std::endl;
    std::vector<std::string> values;
    values.emplace_back("Carbon");
    values.emplace_back("2.265*g/cm3");
    values.emplace_back("12.011*g/mole");
    values.emplace_back("6");

    std::cout << "Initialize element name and namespace" << std::endl;
    std::string element = "ElementaryMaterial";
    std::string nmspc = "test";

    std::cout << "Load Attributes " << std::endl;
    m.loadAttributes(element, names, values, nmspc, cpv);

    std::cout << "Process Element " << std::endl;
    m.processElement(element, nmspc, cpv);
  }
  //  Deal with any exceptions that may have been thrown.
  catch (cms::Exception& e) {
    std::cout << "cms::Exception caught in " << kProgramName << "\n" << e.explainSelf();
    rc = 1;
  } catch (std::exception& e) {
    std::cout << "Standard library exception caught in " << kProgramName << "\n" << e.what();
    rc = 1;
  } catch (...) {
    std::cout << "Unknown exception caught in " << kProgramName;
    rc = 2;
  }
  std::cout << "Done!!!" << std::endl;
  return rc;
}
