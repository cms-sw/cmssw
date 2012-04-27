#include <string>
#include <vector>
#include <iostream>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DetectorDescription/Parser/src/DDLElementaryMaterial.h"
#include "DetectorDescription/Base/interface/DDException.h"

int main(int argc, char *argv[])
{
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
    boost::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
								 makePresence("MessageServicePresence").release());

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
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(config));


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
    names.push_back("name");
    names.push_back("density");
    names.push_back("atomicWeight");
    names.push_back("atomicNumber");

    std::cout << "Initialize values" << std::endl;
    std::vector<std::string> values;
    values.push_back("Carbon");
    values.push_back("2.265*g/cm3");
    values.push_back("12.011*g/mole");
    values.push_back("6");

    std::cout << "Initialize element name and namespace" << std::endl;
    std::string element = "ElementaryMaterial";
    std::string nmspc = "test";

    std::cout << "Load Attributes " << std::endl;
    m.loadAttributes(element, names, values, nmspc, cpv);

    std::cout << "Process Element " << std::endl;
    m.processElement(element, nmspc, cpv);
  }
  catch (DDException& e)
  {
    std::cerr << "DDD-PROBLEM:" << std::endl 
	      << e << std::endl;
  }  
  //  Deal with any exceptions that may have been thrown.
  catch (cms::Exception& e) {
    std::cout << "cms::Exception caught in "
	      << kProgramName
	      << "\n"
	      << e.explainSelf();
    rc = 1;
  }
  catch (std::exception& e) {
    std::cout << "Standard library exception caught in "
	      << kProgramName
	      << "\n"
	      << e.what();
    rc = 1;
  }
  catch (...) {
    std::cout << "Unknown exception caught in "
	      << kProgramName;
    rc = 2;
  }
  std::cout << "Done!!!" << std::endl;
  return rc;
}
