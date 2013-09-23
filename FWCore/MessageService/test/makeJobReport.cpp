
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/InputType.h"


void work()
{

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
    theReport->eventReadFromFile(edm::InputType::Primary, inpFile);
    theReport->eventWrittenToFile(outFile, 1000001, i);
  }

  theReport->inputFileClosed(edm::InputType::Primary, inpFile);
  theReport->outputFileClosed(outFile);
  
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
