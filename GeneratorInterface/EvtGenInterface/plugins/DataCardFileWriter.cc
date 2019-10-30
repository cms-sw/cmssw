#include "GeneratorInterface/EvtGenInterface/interface/DataCardFileWriter.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace gen;

DataCardFileWriter::DataCardFileWriter(const edm::ParameterSet& pset) {
  std::string FileName = pset.getParameter<std::string>("FileName");
  std::string Base = std::getenv("CMSSW_BASE");
  Base += "/src/";
  std::cout << "Writting file:" << Base + FileName << std::endl;
  std::ofstream outputFile(Base + FileName);
  std::vector<std::string> FileContent = pset.getParameter<std::vector<std::string> >("FileContent");
  for (unsigned int i = 0; i < FileContent.size(); i++) {
    outputFile << FileContent.at(i) << std::endl;
  }
  outputFile.close();
  std::cout << "File:" << Base + FileName << " Complete." << std::endl;
}

DEFINE_FWK_MODULE(DataCardFileWriter);
