#include "DataFormats/HGCalDigi/interface/HGCalRawDataEmulatorInfo.h"
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <random>

int main(int argc, char** argv) {
  std::cout << "Basic test of MC truth for the HGCalRawDataEmulatorInfo classes" << std::endl;

  // http://www.cplusplus.com/reference/random/linear_congruential_engine/
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 myrand(seed1);

  // do the trials: time/performance test and exploit randomisation to check
  HGCalSlinkEmulatorInfo slink_info;
  for (unsigned long u = 0; u < 128; u++) {
    for (size_t cb = 0; cb < 10; ++cb) {
      std::vector<bool> enabledCh;
      for (size_t ch = 0; ch < 37; ++ch)
        enabledCh.emplace_back(myrand());
      bool obit = myrand() % 2;
      bool bbit = myrand() % 2;
      bool ebit = myrand() % 2;
      bool tbit = myrand() % 2;
      bool hbit = myrand() % 2;
      bool sbit = myrand() % 2;
      HGCalECONDEmulatorInfo econd_info(obit, bbit, ebit, tbit, hbit, sbit);
      econd_info.addERxChannelsEnable(enabledCh);
      assert(econd_info.bitO() == obit);
      assert(econd_info.bitB() == bbit);
      assert(econd_info.bitE() == ebit);
      assert(econd_info.bitT() == tbit);
      assert(econd_info.bitH() == hbit);
      assert(econd_info.bitS() == sbit);
      slink_info.captureBlockEmulatedInfo(cb).addECONDEmulatedInfo(u, econd_info);
    }
  }

  slink_info.clear();
  std::cout << "\t ...... OK" << std::endl;

  return 0;
}
