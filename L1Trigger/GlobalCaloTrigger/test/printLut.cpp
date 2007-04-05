#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
  char* filename=argv[1];
  if (filename=="") {
    std::cout << "Please enter a non-blank filename\n";
    return -1;
  }

  std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    std::cout << "Failed to open output file " << filename << "\n";
    return -1;
  }

  produceTrivialCalibrationLut* lutProducer=new produceTrivialCalibrationLut();
  L1GctJetEtCalibrationLut* lut=lutProducer->produce();

  outFile << *lut;
  outFile.close();

  return 0;
}

