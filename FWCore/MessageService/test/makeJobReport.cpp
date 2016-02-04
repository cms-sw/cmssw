
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/JobReport.h"



void work()
{

  /*
  // We must initialize the plug-in manager first
  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch(cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    return;
  }
  
  // Load the message service plug-in
  boost::shared_ptr<edm::Presence> theMessageServicePresence;
  try {
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->makePresence("MessageServicePresence").release());
  } catch(cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    return;
  }

  

  //
  // Make JobReport Service up front
  // 
  std::string jobReportFile = "FrameworkJobReport.xml";
  std::auto_ptr<edm::JobReport> jobRep(new edm::JobReport());  
  edm::ServiceToken jobReportToken = 
    edm::ServiceRegistry::createContaining(jobRep);
  
  std::string * jr_name_p = new std::string("FJR.xml");
  edm::MessageLoggerQ::MLqJOB( jr_name_p );

  std::string * jm_p = new std::string("");
  edm::MessageLoggerQ::MLqMOD( jm_p );

  edm::ParameterSet * params_p = new edm::ParameterSet();
  edm::MessageLoggerQ::MLqCFG(params_p);
   */
  
  
  std::cout << "Testing JobReport" << std::endl;
  std::ostringstream ost;
  {
  std::auto_ptr<edm::JobReport> theReport(new edm::JobReport(&ost) ); 
  
  

  std::vector<std::string> inputBranches;
  for (int i = 0; i < 10; i++){
    inputBranches.push_back("Some_Input_Branch");
  }
  
 std::size_t inpFile = theReport->inputFileOpened("InputPFN",
						  "InputLFN",
						  "InputCatalog",
						  "InputType",
						  "InputSource",
						  "InputLabel",
						  "InputGUID",
						  inputBranches);
  
 
  std::vector<std::string> outputBranches;
  for (int i=0; i < 10; i++){
    outputBranches.push_back("Some_Output_Branch_Probably_From_HLT");

  }

  
  std::size_t outFile = theReport->outputFileOpened("OutputPFN",
						    "OutputLFN",
						    "OutputCatalog",
						    "OutputModule",
						    "OutputModuleName",
						    "OutputGUID",
						    "DataType",
						    "OutputBranchesHash",
						    outputBranches);

  
  for (int i=0; i < 1000; i++){
    theReport->eventReadFromFile(inpFile, 1000001, i);
    theReport->eventWrittenToFile(outFile, 1000001, i);
  }

  theReport->inputFileClosed(inpFile);
  theReport->outputFileClosed(outFile);
  
  //edm::LogInfo("FwkJob") << "Is anybody out there?";
  }
  std::cout << ost.str()<<std::endl;

}

int main()
{
  int rc = -1;
  try {
      work();
      
      rc = 0;
  }
  
  catch ( ... ) {
      std::cerr << "Unknown exception caught\n";
      rc = 2;
  }
  return rc;
}
