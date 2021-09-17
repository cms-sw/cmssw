#include "DataFormats/HGCDigi/interface/HGCSample.h"
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <random>

// run for instance with:
//
//                         time HGCSampleTest  10000000000
//                         time HGCSampleTest  1000000000 => 14 sec ; 11 sec after restructuring set()
//                         time HGCSampleTest  10000000000    y   [to require randomisation]
//
// for a measureble amount of time taken

int main(int argc, char** argv) {
  std::cout << "Basic performance tests for HGCSample (pseudo-random seed set according to local time)\n" << std::endl;
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
  uint32_t adc = 124;
  uint32_t gain = 15;
  const uint32_t thrADC = 900;
  bool thr = adc > thrADC;
  uint32_t toa = 0;
  bool mode = false;
  bool toaFired = true;

  // http://www.cplusplus.com/reference/random/linear_congruential_engine/
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 myrand(seed1);

  // do the trials: time/performance test and exploit randomisation to check
  unsigned long int u = 0;
  for (; u < repetitions; u++) {
    // randomise all inputs, if chosen at the command line
    if (generateRandomValues) {
      adc = myrand() % 4096;
      toa = myrand() % 1024;
      gain = myrand() % 16;
      mode = myrand() % 2;
      thr = myrand() % 2;
      toaFired = myrand() % 2;
    }

    HGCSample aSample;
    // writing on an empty container first
    aSample.set(thr, mode, gain, toa, adc);
    aSample.setToAValid(toaFired);
    // check the values that went in also come out
    assert(thr == aSample.threshold());
    assert(mode == aSample.mode());
    assert(toa == aSample.toa());
    assert(gain == aSample.gain());
    assert(adc == aSample.data());
    assert(thr == aSample.threshold());
    assert(toaFired == aSample.getToAValid());

    // std::cout << aSample.raw() << "\t" << thr << "\t" << mode << "\t" << toaFired << "\t" << gain << "\t" << toa << "\t"
    //          << adc << std::endl;

    HGCSample ASample;
    ASample.setThreshold(thr);
    ASample.setMode(mode);
    ASample.setGain(gain);
    ASample.setToA(toa);
    ASample.setData(adc);
    ASample.setToAValid(toaFired);
    // check that using the individual setters yields the same result as the generic set
    assert(ASample.threshold() == aSample.threshold());
    assert(ASample.mode() == aSample.mode());
    assert(ASample.gain() == aSample.gain());
    assert(ASample.toa() == aSample.toa());
    assert(ASample.data() == aSample.data());
    assert(ASample.getToAValid() == aSample.getToAValid());

    HGCSample bSample;
    bSample.setThreshold(thr);
    bSample.setMode(mode);
    bSample.setGain(gain + 100);
    bSample.setToA(toa + 100);
    bSample.setData(adc + 100);
    bSample.setToAValid(toaFired);

    // cover the case where we write on a container with numbers already set
    bSample.setThreshold(thr);
    bSample.setMode(mode);
    bSample.setGain(gain);
    bSample.setToA(toa);
    bSample.setData(adc);
    bSample.setToAValid(toaFired);
    assert(thr == aSample.threshold() && thr == bSample.threshold());
    assert(mode == aSample.mode() && mode == bSample.mode());
    assert(gain == aSample.gain() && gain == bSample.gain());
    assert(toa == aSample.toa() && toa == bSample.toa());
    assert(adc == aSample.data() && adc == bSample.data());
    assert(toaFired == aSample.getToAValid() && toaFired == bSample.getToAValid());
  }

  std::cout << "\nDone " << repetitions << "\t" << u << std::endl;

  return 0;
}
