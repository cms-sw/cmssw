/*----------------------------------------------------------------------

Test of the statemachine classes.

----------------------------------------------------------------------*/

#include "FWCore/Framework/test/MockEventProcessor.h"

#include <boost/program_options.hpp>

#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) try {
  std::cout << "Running test in statemachine_t.cc\n";

  // Handle the command line arguments
  std::string inputFile;
  std::string outputFile;
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "inputFile,i", boost::program_options::value<std::string>(&inputFile)->default_value(""))(
      "outputFile,o",
      boost::program_options::value<std::string>(&outputFile)->default_value("statemachine_test_output.txt"));
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  // Get some fake data from an input file.
  // The fake data has the format of a series pairs of items.
  // The first is a letter to indicate the data type
  // r for run, l for lumi, e for event, f for file, s for stop
  // The second item is the run number or luminosity block number
  // for the run and lumi cases.  For the other cases the number
  // is not not used.  This series of fake data items is terminated
  // by a period (blank space and newlines are ignored).
  // Use the trivial default in the next line if no input file
  // has been specified
  std::string mockData = "s 1";
  if (inputFile != "") {
    std::ifstream input;
    input.open(inputFile.c_str());
    if (input.fail()) {
      std::cerr << "Error, Unable to open mock input file named " << inputFile << "\n";
      return 1;
    }
    std::getline(input, mockData, '.');
  }

  std::ofstream output(outputFile.c_str());

  std::vector<bool> fileModes;
  fileModes.push_back(true);
  fileModes.push_back(false);

  for (auto mode : fileModes) {
    output << "\nMachine parameters:  ";
    if (mode)
      output << "mode = NOMERGE";
    else
      output << "mode = FULLMERGE";

    output << "\n";

    edm::MockEventProcessor mockEventProcessor(mockData, output, mode);
    try {
      mockEventProcessor.runToCompletion();
    } catch (edm::MockEventProcessor::TestException const& e) {
      output << "caught test exception\n";
    }
  }
  return 0;
} catch (std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
