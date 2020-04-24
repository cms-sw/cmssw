/*----------------------------------------------------------------------

   This is a generic main that can be used with any plugin and a 
   PSet script. It shows the minimum machinery necessary for a
   "standalone" program to issue MessageLogger messages.
   N. B. In this context, standalone means a job where the user
   has provided the main program instead of supplying a module
   for cmsRun to call on.

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------

std::string indirectWarn( int /*num*/ )
{
//  std::cout << "  Returning the string Emit Warning level message " << num << std::endl; 
    return std::string("\t\tEmit Warning level message "); 
}


std::string indirectInfo( int /*num*/ )
{
//  std::cout << "  Returning the string Emit Info level message " << num << std::endl; 
    return std::string("\t\tEmit Info level message "); 
}

void DoMyStuff( )
{
// Issue several types of logger messages.  This function could
// be substantially more complex. This example is about as simple
// as can be.

  double d = 3.14159265357989;
  edm::LogWarning("cat_A")   << "Test of std::setprecision(p):"
  			     << " Pi with precision 12 is " 
  			     << std::setprecision(12) << d;

  for( int i=0; i<25; ++i) {
//  edm::LogInfo("cat_B")    << "\t\tEmit Info level message " << i+1;
    edm::LogInfo("cat_B")    << indirectInfo(i+1) << i+1;
//  edm::LogWarning("cat_C") << "\t\tEmit Warning level message " << i+1;
    edm::LogWarning("cat_C") << indirectWarn(i+1) << i+1;
  }
}  

int main(int, char* argv[]) {

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
    theMessageServicePresence = std::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
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

// D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(config));

// E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

//  Generate a bunch of messages.
    DoMyStuff( );
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
