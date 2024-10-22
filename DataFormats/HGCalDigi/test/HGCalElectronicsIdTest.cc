#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <random>

// run for instance with:
//
//                         time HGCalElectronicsId 10000000000 => 8 sec
// for a measureble amount of time taken
// acceptas an additional argument for verbosity level

int main(int argc, char** argv) {
  std::cout << "Basic check of HGCalElectronicsId class" << std::endl;

  // first command line argument is the number of trials
  unsigned long int repetitions = 100;
  if (argc > 1)
    repetitions = std::stoul(argv[1], nullptr, 0);
  std::cout << "\t + repetitions [int]: " << repetitions << std::endl;

  unsigned long int verbosity = 0;
  if (argc > 2)
    verbosity = std::stoul(argv[2], nullptr, 0);

  // init static values
  bool zside(false);
  uint16_t localfedid(0);
  uint8_t captureblock(0), econdidx(0), econderx(0), halfrocch(0);

  // http://www.cplusplus.com/reference/random/linear_congruential_engine/
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 myrand(seed1);

  // do the trials: time/performance test and exploit randomisation to check
  unsigned long int u = 0;
  for (; u < repetitions; u++) {
    zside = (bool)myrand() % 2;
    localfedid = myrand() % 576;
    captureblock = myrand() % 10;
    econdidx = myrand() % 12;
    econderx = myrand() % 12;
    halfrocch = myrand() % 39;
    bool cmflag = ((halfrocch == 37) || (halfrocch == 38));

    HGCalElectronicsId eid(zside, localfedid, captureblock, econdidx, econderx, halfrocch);
    assert(zside == eid.zSide());
    assert(cmflag == eid.isCM());
    assert(localfedid == eid.localFEDId());
    assert(captureblock == eid.captureBlock());
    assert(econdidx == eid.econdIdx());
    assert(econderx == eid.econdeRx());
    assert(halfrocch == eid.halfrocChannel());

    if (verbosity > 0)
      eid.print(std::cout);
  }

  std::cout << "\nDone " << repetitions << "\t" << u << std::endl;

  return 0;
}
