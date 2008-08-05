#include <string>
#include <vector>
#include <iostream>

//CMSSW main includes
#include <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
//end CMSSW main includes

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
      "process x = {"
      "service = MessageLogger {"
      "untracked vstring destinations = {'infos.mlog','warnings.mlog'}"
      "untracked PSet infos = {"
      "untracked string threshold = 'INFO'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "untracked PSet FwkJob = {untracked int32 limit = 0}"
      "}"
      "untracked PSet warnings = {"
      "untracked string threshold = 'WARNING'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "}"
      "untracked vstring fwkJobReports = {'FrameworkJobReport.xml'}"
      "untracked vstring categories = {'FwkJob'}"
      "untracked PSet FrameworkJobReport.xml = {"
      "untracked PSet default = {untracked int32 limit = 0}"
      "untracked PSet FwkJob = {untracked int32 limit = 10000000}"
      "}"
      "}"
      "service = JobReportService{}"
      "service = SiteLocalConfigService{}"
      "}";


    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ParameterSet>          params_;
    edm::makeParameterSets(config, params_, pServiceSets);

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout << "Create DDLElementaryMaterial m" << std::endl;
    DDLElementaryMaterial m;
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
    m.loadAttributes(element, names, values, nmspc);

    std::cout << "Process Element " << std::endl;
    m.processElement(element, nmspc);
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
