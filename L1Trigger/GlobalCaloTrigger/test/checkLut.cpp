#include "L1Trigger/GlobalCaloTrigger/test/L1GctLutFromFile.h"
#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include <iostream>
#include <fstream>
#include <sstream>

typedef produceTrivialCalibrationLut::lutPtrVector lutPtrVector;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "No filename argument supplied - exiting" << std::endl;
    return 0;
  }

  char* filename = argv[1];

  bool allOk = true;

  produceTrivialCalibrationLut* lutProducer = new produceTrivialCalibrationLut();
  lutPtrVector lutVector1 = lutProducer->produce();

  for (lutPtrVector::const_iterator lut1 = lutVector1.begin(); lut1 != lutVector1.end(); lut1++) {
    std::stringstream ss;
    std::string nextFile;
    ss << filename << (*lut1)->etaBin() << ".txt";
    ss >> nextFile;

    static const int NAdd = JET_ET_CAL_LUT_ADD_BITS;
    static const int NDat = JET_ET_CAL_LUT_DAT_BITS;

    std::ifstream inFile(nextFile.c_str(), std::ios::in);
    if (!inFile.is_open()) {
      std::cout << "Failed to open input file " << nextFile << std::endl;
      allOk = false;
    } else {
      L1GctLutFromFile<NAdd, NDat>* lut2 = L1GctLutFromFile<NAdd, NDat>::setupLut(nextFile);

      std::cout << "Eta bin " << (*lut1)->etaBin() << std::endl;
      if (**lut1 == *lut2) {
        std::cout << "Look-up table match check ok\n";
      } else {
        std::cout << "Look-up table match check failed\n";
      }
      if (*lut2 != **lut1) {
        std::cout << "Look-up tables are not equal\n";
      } else {
        std::cout << "Look-up tables are equal\n";
      }
    }
  }

  return (allOk ? 0 : -1);
}
