#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVector.h"
//#include "DetectorDescription/Core/interface/DDD.h"
//#include "FWCore/PluginManager/interface/PluginManager.h"
//#include "FWCore/PluginManager/interface/standard.h"
// DDD Interface in CARF
//#include "CARF/DDDInterface/interface/GeometryConfiguration.h"

// Error Detection
//#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

//#include "DetectorDescription/Core/interface/graph_path.h"
//typedef GraphPath<DDLogicalPart,DDPosData*> GPathType;

// The DDD user-code after XML-parsing is located
// in DetectorDescription/Core/src/tutorial.cc
// Please have a look to all the commentary therein.

// BLOCK copy from cmsRun.cpp
#include <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
//#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"
// BLOCK END copy from cmsRun.cpp

using namespace std;
namespace DD { } using namespace DD;

int main(int argc, char *argv[])
{
  //   static TimerProxy timer_("main()");
  //   TimeMe t(timer_,false);
  std::string const kProgramName = argv[0];
  int rc = 0;
  // BLOCK copy from cmsRun.cpp
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
    edm::makeParameterSets(config, params_);

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

// try {
//     edmplugin::PluginManager::configure(edmplugin::standard::config());
//   } catch(cms::Exception& e) {
//     std::cerr << e.what() << std::endl;
//     return 1;
//   }
  
//   // Load the message service plug-in
//   boost::shared_ptr<edm::Presence> theMessageServicePresence;
//   try {
//     theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
// 								 makePresence("MessageServicePresence").release());
//   } catch(cms::Exception& e) {
//     std::cerr << e.what() << std::endl;
//     return 1;
//   }
//   // BLOCK END copy from cmsRun.cpp

//   try { // DDD Prototype can throw DDException defined in DetectorDescription/Core/interface/DDException.h
  
    // Initialize a DDL Schema aware parser for DDL-documents
    // (DDL ... Detector Description Language)
    cout << "initialize DDL parser" << endl;
    DDCompactView cpv;
    DDLParser myP(cpv);// = DDLParser::instance();

    //     cout << "about to set configuration" << endl;
    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  
    //myP->SetConfig("configuration.xml");

    cout << "about to start parsing" << endl;
    string configfile("configuration.xml");
    if (argc==2) {
      configfile = argv[1];
    }
    //    GeometryConfiguration documentProvider("configuration.xml");
    //    FIPConfiguration fp;
    FIPConfiguration fp(cpv);
    fp.readConfig(configfile);
    int parserResult = myP.parse(fp);
    cout << "done parsing" << std::endl;
    cout.flush();
    if (parserResult != 0) {
      cout << " problem encountered during parsing. exiting ... " << endl;
      exit(1);
    }
    cout << " parsing completed" << endl;
  
    cout << endl << endl << "Start checking!" << endl << endl;
 
    DDErrorDetection ed(cpv);
    //ed.scan();
    ed.report( cpv, std::cout);//cout);

    DDConstant::createConstantsFromEvaluator();  // DDConstants are not being created by anyone... it confuses me!
    DDConstant::iterator<DDConstant> cit(DDConstant::begin()), ced(DDConstant::end());
    for(; cit != ced; ++cit) {
      cout << *cit << endl;
    }

    DDVector::iterator<DDVector> vit;
    DDVector::iterator<DDVector> ved(DDVector::end());
    if ( vit == ved ) std::cout << "No DDVectors found." << std::endl;
    for (; vit != ved; ++vit) {
      if (vit->isDefined().second) {
	std::cout << vit->toString() << std::endl;
	const std::vector<double>& tv = *vit;
	std::cout << "size: " << tv.size() << std::endl;
	for (size_t i=0; i < tv.size(); ++i) {
	  std::cout << tv[i] << "\t";
	}
	std::cout << std::endl;
      }
    }  

//     Vector::iterator<Vector> vit(Vector::begin()), ved(Vector::end());
//     for(; vit != ved; ++vit) {
//       cout << *vit << endl;
//     }


    //   TimingReport* tr = TimingReport::current();
    //   tr->dump(cout);    
    return 0;
  
  }
  catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
    {
      cerr << "DDD-PROBLEM:" << endl 
	   << e << endl;
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

  return rc;
  
}
