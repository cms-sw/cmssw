#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"

#include <algorithm>
#include <iostream>
#include <iterator>

struct DetIdAndApvs
{
  uint32_t detId;
  std::vector<uint16_t> apvs;
};

void test( const std::vector<DetIdAndApvs> & detIdAndApvs,
           const std::vector<int> & latencyIndexes, std::vector<uint16_t> & latencies,
           const std::vector<int> & modeIndexes, std::vector<uint16_t> & modes,
           SiStripLatency & latency )
{
  int i = 0;
  int flip = 0;
  int modeFlip = 0;
  std::vector<DetIdAndApvs>::const_iterator detIdAndApv = detIdAndApvs.begin();
  for( ; detIdAndApv != detIdAndApvs.end(); ++detIdAndApv ) {
    std::vector<uint16_t>::const_iterator apv = detIdAndApv->apvs.begin();
    for( ; apv != detIdAndApv->apvs.end(); ++apv, ++i ) {

      // std::cout << "detId = " << detIdAndApv->detId << ", apv = " << *apv << ", detIdAndApv = " << compactValue << std::endl;

      if( find(latencyIndexes.begin(), latencyIndexes.end(), i) != latencyIndexes.end() ){
        if( flip == 0 ) {
          flip = 1;
        }
        else {
          flip = 0;
        }
      }


      if( find(modeIndexes.begin(), modeIndexes.end(), i) != modeIndexes.end() ){
        if( modeFlip == 10 ) {
          modeFlip = 0;
        }
        else {
          modeFlip = 10;
        }
      }


      // std::cout << "For i = " << i << " flip = " << flip << std::endl;
      latency.put(detIdAndApv->detId, *apv, 14+flip, 37+modeFlip);

//       std::cout << "latency stored is = " << latency.latency(detIdAndApv->detId, *apv) << std::endl;
      latencies.push_back(latency.latency(detIdAndApv->detId, *apv));
      modes.push_back(latency.mode(detIdAndApv->detId, *apv));
//       std::cout << std::endl;
    }
  }
  // Finished filling, now compress the ranges
  std::vector<SiStripLatency::Latency> latenciesBeforeCompression = latency.allLatencyAndModes();
  std::cout << "Ranges before compression = " << latenciesBeforeCompression.size() << std::endl;
  latency.compress();
  std::vector<SiStripLatency::Latency> latenciesAfterCompression = latency.allLatencyAndModes();
  std::cout << "Ranges after compression = " << latenciesAfterCompression.size() << std::endl;
}

void check( const std::vector<uint16_t> & latencies, const std::vector<uint16_t> & modes, const std::vector<DetIdAndApvs> & detIdAndApvs, SiStripLatency & latency )
{
  if( latencies.size() != modes.size() ) {
    std::cout << "Error: different size for latencies = " << latencies.size() << " and modes = " << modes.size() << std::endl;
    exit(1);
  }
  std::vector<DetIdAndApvs>::const_iterator detIdAndApv = detIdAndApvs.begin();
  std::vector<uint16_t>::const_iterator it = latencies.begin();
  std::vector<uint16_t>::const_iterator modeIt = modes.begin();
  detIdAndApv = detIdAndApvs.begin();
  int latencyErrorCount = 0;
  int modeErrorCount = 0;
  for( ; detIdAndApv != detIdAndApvs.end(); ++detIdAndApv ) {
    std::vector<uint16_t>::const_iterator apv = detIdAndApv->apvs.begin();
    for( ; apv != detIdAndApv->apvs.end(); ++apv, ++it, ++modeIt ) {
      uint32_t detId = detIdAndApv->detId;
      uint32_t detIdAndApvValue = (detId<<2)|(*apv);
      std::cout << "detId = " << detIdAndApv->detId << ", apv = " << *apv << ", detIdAndApv = " << detIdAndApvValue << std::endl;
      std::cout << "latency passed = " << *it << ", latency saved = " << latency.latency(detIdAndApv->detId, *apv) << std::endl;
      std::cout << "mode passed = " << *modeIt << ", mode saved = " << latency.mode(detIdAndApv->detId, *apv) << std::endl;
      if( *it != latency.latency(detIdAndApv->detId, *apv) ) {
        std::cout << "ERROR: the latency values are different" << std::endl;
        ++latencyErrorCount;
      }
      if( *modeIt != latency.mode(detIdAndApv->detId, *apv) ) {
        std::cout << "ERROR: the mode values are different" << std::endl;
        ++modeErrorCount;
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Single latency value = " << latency.singleLatency() << std::endl;
  std::cout << "Single mode value = " << latency.singleMode() << std::endl;

  std::ostream_iterator<uint16_t> output( std::cout, ", " );
  // Print all latencies
  std::vector<uint16_t> allLatenciesVector;
  latency.allLatencies(allLatenciesVector);
  std::cout << "All latencies in the Tracker = " << allLatenciesVector.size() << ", and are:" << std::endl;
  copy( allLatenciesVector.begin(), allLatenciesVector.end(), output );
  std::cout << std::endl;
  // Print all modes
  std::vector<uint16_t> allModesVector;
  latency.allModes(allModesVector);
  std::cout << "All modes in the Tracker = " << allModesVector.size() << ", and are:" << std::endl;
  copy( allModesVector.begin(), allModesVector.end(), output );
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "Latency errors = " << latencyErrorCount << std::endl;
  std::cout << "Mode errors = " << modeErrorCount << std::endl;
  std::cout << std::endl;
  std::cout << "############################" << std::endl;
  std::cout << std::endl;
}

int main()
{
  std::vector<DetIdAndApvs> detIdAndApvs;
  DetIdAndApvs element1;
  element1.detId = 100000;
  element1.apvs.push_back(0);
  element1.apvs.push_back(1);
  element1.apvs.push_back(2);
  element1.apvs.push_back(3);
  detIdAndApvs.push_back(element1);

  DetIdAndApvs element2;
  element2.detId = 100001;
  element2.apvs.push_back(0);
  element2.apvs.push_back(1);
  element2.apvs.push_back(2);
  element2.apvs.push_back(3);
  element2.apvs.push_back(4);
  element2.apvs.push_back(5);
  detIdAndApvs.push_back(element2);

  DetIdAndApvs element3;
  element3.detId = 9998;
  element3.apvs.push_back(0);
  element3.apvs.push_back(1);
  element3.apvs.push_back(2);
  element3.apvs.push_back(3);
  element3.apvs.push_back(4);
  element3.apvs.push_back(5);
  detIdAndApvs.push_back(element3);

  DetIdAndApvs element4;
  element4.detId = 9999;
  element4.apvs.push_back(0);
  element4.apvs.push_back(1);
  detIdAndApvs.push_back(element4);

  DetIdAndApvs element5;
  element5.detId = 100002;
  element5.apvs.push_back(0);
  element5.apvs.push_back(1);
  element5.apvs.push_back(2);
  element5.apvs.push_back(3);
  element5.apvs.push_back(4);
  element5.apvs.push_back(5);
  detIdAndApvs.push_back(element5);

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Testing the SiStripLatency object" << std::endl;
  std::cout << "---------------------------------" << std::endl << std::endl;

  std::cout << "Testing the empty case" << std::endl;
  std::cout << "----------------------" << std::endl;
  // Testing with all the same values. Expected final size of internal ranges and latencies = 1
  std::vector<int> latencyIndexes;
  std::vector<uint16_t> latencies;
  std::vector<int> modeIndexes;
  std::vector<uint16_t> modes;
  SiStripLatency latency1;
  test(detIdAndApvs, latencyIndexes, latencies, modeIndexes, modes, latency1);
  std::cout << std::endl;
  std::cout << "Filling complete, starting check" << std::endl;
  std::cout << std::endl;
  check(latencies, modes, detIdAndApvs, latency1);

  std::cout << std::endl;
  std::cout << "Testing a case with several ranges" << std::endl;
  std::cout << "----------------------------------" << std::endl;
  SiStripLatency latency2;
  latencyIndexes.push_back(3);
  latencyIndexes.push_back(5);
  latencyIndexes.push_back(10);
  latencyIndexes.push_back(11);
  latencies.clear();
  modeIndexes.push_back(4);
  modes.clear();
  test(detIdAndApvs, latencyIndexes, latencies, modeIndexes, modes, latency2);
  std::cout << std::endl;
  std::cout << "Filling complete, starting check" << std::endl;
  std::cout << std::endl;
  check(latencies, modes, detIdAndApvs, latency2);

  // Checking the method to retrieve all the unique (latencies,modes) pairs
  // Create a latency object with three combinations of latency and mode: (14, 37), (15, 37) and (15, 47)
  std::cout << "Checking the method to retrieve all the unique combinations of latency and mode" << std::endl;
  SiStripLatency latency3;
  latency3.put(1, 0, 14, 37);
  latency3.put(2, 0, 14, 37);
  latency3.put(3, 0, 15, 37);
  latency3.put(4, 0, 15, 47);
  std::cout << "Stored three combinations of latency and mode: (14, 37), (15, 37) and (15, 47)" << std::endl;

  std::vector<SiStripLatency::Latency> uniqueLatenciesAndModes(latency3.allUniqueLatencyAndModes());
  std::vector<SiStripLatency::Latency>::const_iterator it = uniqueLatenciesAndModes.begin();
  std::cout << "Reading back what is returned by the allUniqueLatencyAndModes method" << std::endl;
  for( ; it != uniqueLatenciesAndModes.end(); ++it ) {
    std::cout << "latency = " << int(it->latency) << ", mode = " << int(it->mode) << std::endl;
  }
  if( uniqueLatenciesAndModes.size() == 3 ) {
    std::cout << "Test passed" << std::endl;
  }
  else {
    std::cout << "ERROR: test not passed" << std::endl;
  }







  // Checking the case with different modes, but same Read-out mode.
  std::cout << "Checking the case with different modes, but same Read-out mode." << std::endl;
  SiStripLatency latency4;
  latency4.put(1, 0, 14, 37);
  latency4.put(2, 0, 14, 36);
  latency4.put(3, 0, 14, 37);
  latency4.put(4, 0, 14, 36);
  std::cout << "Stored two combinations of latency and mode: (14, 37), (14, 36)" << std::endl;
  std::cout << "The Read-out mode is the same and is deconvolution" << std::endl;

  if( latency4.singleReadOutMode() == 0 ) {
    std::cout << "Test passed" << std::endl;
  }
  else {
    std::cout << "ERROR: test not passed" << std::endl;
  }

  SiStripLatency latency5;
  latency5.put(1, 0, 14, 47);
  latency5.put(2, 0, 14, 46);
  latency5.put(3, 0, 14, 47);
  latency5.put(4, 0, 14, 46);
  std::cout << "Stored two combinations of latency and mode: (14, 47), (14, 46)" << std::endl;
  std::cout << "The Read-out mode is the same and is peak" << std::endl;

  if( latency5.singleReadOutMode() == 1 ) {
    std::cout << "Test passed" << std::endl;
  }
  else {
    std::cout << "ERROR: test not passed" << std::endl;
  }

  SiStripLatency latency6;
  latency6.put(1, 0, 14, 47);
  latency6.put(2, 0, 14, 46);
  latency6.put(3, 0, 14, 37);
  latency6.put(4, 0, 14, 36);
  std::cout << "Stored four combinations of latency and mode: (14, 47), (14, 46), (14, 37), (14, 36)" << std::endl;
  std::cout << "The Read-out mode is mixed" << std::endl;

  if( latency6.singleReadOutMode() == -1 ) {
    std::cout << "Test passed" << std::endl;
  }
  else {
    std::cout << "ERROR: test not passed" << std::endl;
  }

  return 0;
}
