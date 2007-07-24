#include "L1Trigger/GlobalCaloTrigger/test/L1GctLutFromFile.h"
#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
  if (argc<2) {
    std::cout << "No filename argument supplied - exiting" << std::endl;
    return -1;
  }

  char* filename=argv[1];
  std::ifstream inFile(filename, std::ios::in);
  if (!inFile.is_open()) {
    std::cout << "Failed to open input file " << filename << std::endl;
    return -1;
  }

  static const int NAdd=JET_ET_CAL_LUT_ADD_BITS;
  static const int NDat=JET_ET_CAL_LUT_DAT_BITS;

  std::string fn=filename;
  L1GctLutFromFile<NAdd,NDat>* lut2=L1GctLutFromFile<NAdd,NDat>::setupLut(fn);

  produceTrivialCalibrationLut* lutProducer=new produceTrivialCalibrationLut();
  L1GctJetEtCalibrationLut* lut1=lutProducer->produce();

  if (*lut1 == *lut2) { std::cout << "Look-up table match check ok\n"; } else
                      { std::cout << "Look-up table match check failed\n"; }
  if (*lut2 != *lut1) { std::cout << "Look-up tables are not equal\n"; } else
                      { std::cout << "Look-up tables are equal\n"; }

  return 0;
}

