#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include <iostream>
#include <fstream>
#include <sstream>

typedef produceTrivialCalibrationLut::lutPtrVector lutPtrVector;

int main(int argc, char** argv) {
  const std::string filename = (argc == 1 ? "" : std::string(argv[1]));
  if (filename.empty()) {
    std::cout << "No filename argument supplied - exiting" << std::endl;
    return 0;
  }

  bool allOk = true;

  produceTrivialCalibrationLut* lutProducer = new produceTrivialCalibrationLut();
  lutPtrVector lutVector = lutProducer->produce();

  for (lutPtrVector::const_iterator lut = lutVector.begin(); lut != lutVector.end(); lut++) {
    std::stringstream ss;
    std::string nextFile;
    ss << filename << (*lut)->etaBin() << ".txt";
    ss >> nextFile;
    std::ofstream outFile(nextFile.c_str(), std::ios::out | std::ios::trunc);
    if (!outFile.is_open()) {
      std::cout << "Failed to open output file " << nextFile << "\n";
      allOk = false;
    } else {
      outFile << **lut;
      outFile.close();
    }
  }

  return (allOk ? 0 : -1);
}
