
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/InputType.h"

int work() {
  std::ostringstream ost;
  {
    auto theReport = std::make_unique<edm::JobReport>(&ost);

    std::vector<std::string> inputBranches;
    for (int i = 0; i < 10; i++) {
      inputBranches.push_back("Some_Input_Branch");
    }

    std::size_t inpFile = theReport->inputFileOpened("InputPFN?with&cgi=params",
                                                     "InputLFN",
                                                     "InputCatalog",
                                                     "InputType",
                                                     "InputSource",
                                                     "InputLabel",
                                                     "InputGUID",
                                                     inputBranches);

    std::vector<std::string> outputBranches;
    for (int i = 0; i < 10; i++) {
      outputBranches.push_back("Some_Output_Branch_Probably_From_HLT");
    }

    std::size_t outFile = theReport->outputFileOpened("OutputPFN?with&cgi=params",
                                                      "OutputLFN",
                                                      "OutputCatalog",
                                                      "OutputModule",
                                                      "OutputModuleName",
                                                      "OutputGUID",
                                                      "DataType",
                                                      "OutputBranchesHash",
                                                      outputBranches);

    for (int i = 0; i < 1000; i++) {
      theReport->eventReadFromFile(edm::InputType::Primary, inpFile);
      theReport->eventWrittenToFile(outFile, 1000001, i);
    }

    theReport->inputFileClosed(edm::InputType::Primary, inpFile);
    theReport->outputFileClosed(outFile);
  }
  std::string report = ost.str();
  std::cout << report << std::endl;

  return 0;
}

int main() {
  int rc = -1;
  try {
    rc = work();
  } catch (...) {
    std::cerr << "Unknown exception caught\n";
    rc = 2;
  }
  return rc;
}
