#include "DataFormats/HGCalDigi/interface/HGCalDigiCollections.h"
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <random>

// run for instance with:
//
//                         time HGCROCSampleTest  10000000000 => 8 sec
//                         time HGCROCSampleTest  10000000000    y   [to require randomisation] =>
//
// for a measureble amount of time taken

//wrap the procedure to assert TOT read is within quantization error
//a truncation of two bits leads to an uncertainty of +/- 4 ADC counts
bool totOK(uint16_t tot_orig, uint16_t tot_read) {
  int delta(tot_read - tot_orig);
  return (delta >= -4 && delta <= 4);
}

int main(int argc, char** argv) {
  std::cout << "Basic performance tests for HGCROCChannelDataFrame (pseudo-random seed set according to local time)\n"
            << std::endl;
  std::cout << "num parameters entered: " << argc << std::endl;

  // first command line argument is the number of trials
  unsigned long int repetitions = 100;
  if (argc > 1)
    repetitions = std::stoul(argv[1], nullptr, 0);
  std::cout << "\t + repetitions [int]: " << repetitions << std::endl;
  // second command line argument (whatever it is) will activate
  //                  the random choice of values for all inputs
  bool generateRandomValues = (argc > 2 ? true : false);
  std::cout << "\t + generateRandomValues [true/false]: " << generateRandomValues << "\n" << std::endl;

  // init static values
  uint16_t adc(125), adcm1(23), tot(10), toa(8);
  bool tc(false), tp(false), charMode(false);

  // http://www.cplusplus.com/reference/random/linear_congruential_engine/
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 myrand(seed1);

  // do the trials: time/performance test and exploit randomisation to check
  unsigned long int u = 0;
  for (; u < repetitions; u++) {
    // randomise all inputs, if chosen at the command line
    if (generateRandomValues) {
      adc = myrand() % 1024;
      adcm1 = myrand() % 1024;
      tot = myrand() % 2048;
      toa = myrand() % 1024;
      tc = myrand() % 2;
      tp = myrand() % 2;
      charMode = myrand() % 2;
    }

    HGCROCChannelDataFrameSpec aSample;
    aSample.fill(charMode, tc, tp, adcm1, adc, tot, toa);

    bool tc_read = aSample.tc();
    bool tp_read = aSample.tp();
    uint16_t adc_read = aSample.adc(charMode);
    uint16_t adcm1_read = aSample.adcm1(charMode);
    uint16_t tot_read = aSample.tot(charMode);
    uint16_t toa_read = aSample.toa();
    assert(tc == tc_read);
    assert(tp == tp_read);

    //uncomment for a verbose output
    //std::cout << "Tc=" << tc << " Tp=" << tp << " adcm1=" << adcm1 << " adc=" << adc << " tot=" << tot << " toa=" << toa << " char mode=" << charMode << std::endl;
    //aSample.print(std::cout);
    //std::cout << "Tc'=" << tc_read << " Tp'=" << tp_read << " adcm1'=" << adcm1_read << " adc'=" << adc_read << " tot'=" << tot_read << " toa'=" << toa_read << std::endl;

    if (charMode) {
      assert(adcm1_read == 0);
      assert(adc == adc_read);
      assert(totOK(tot, tot_read));
      assert(toa == toa_read);
    } else {
      assert(adcm1 == adcm1_read);
      if (tc) {
        assert(adc_read == 0);
        assert(totOK(tot, tot_read));
      } else {
        assert(tot_read == 0);
        assert(adc == adc_read);
      }
    }
  }

  std::cout << "\nDone " << repetitions << "\t" << u << std::endl;

  return 0;
}
