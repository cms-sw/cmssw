#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"

#include <algorithm>
#include <iostream>
#include <iterator>

using namespace std;

struct DetIdAndApvs
{
  uint32_t detId;
  vector<uint16_t> apvs;
};

void test( const vector<DetIdAndApvs> & detIdAndApvs,
           const vector<int> & latencyIndexes, vector<uint16_t> & latencies,
           const vector<int> & modeIndexes, vector<uint16_t> & modes,
           SiStripLatency & latency )
{
  int i = 0;
  int flip = 0;
  int modeFlip = 0;
  vector<DetIdAndApvs>::const_iterator detIdAndApv = detIdAndApvs.begin();
  for( ; detIdAndApv != detIdAndApvs.end(); ++detIdAndApv ) {
    vector<uint16_t>::const_iterator apv = detIdAndApv->apvs.begin();
    for( ; apv != detIdAndApv->apvs.end(); ++apv, ++i ) {

      // cout << "detId = " << detIdAndApv->detId << ", apv = " << *apv << ", detIdAndApv = " << compactValue << endl;

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


      // cout << "For i = " << i << " flip = " << flip << endl;
      latency.put(detIdAndApv->detId, *apv, 14+flip, 37+modeFlip);

//       cout << "latency stored is = " << latency.latency(detIdAndApv->detId, *apv) << endl;
      latencies.push_back(latency.latency(detIdAndApv->detId, *apv));
      modes.push_back(latency.mode(detIdAndApv->detId, *apv));
//       cout << endl;
    }
  }
  // Finished filling, now compress the ranges
  vector<SiStripLatency::Latency> latenciesBeforeCompression = latency.allLatencyAndModes();
  cout << "Ranges before compression = " << latenciesBeforeCompression.size() << endl;
  latency.compress();
  vector<SiStripLatency::Latency> latenciesAfterCompression = latency.allLatencyAndModes();
  cout << "Ranges after compression = " << latenciesAfterCompression.size() << endl;
}

void check( const vector<uint16_t> & latencies, const vector<uint16_t> & modes, const vector<DetIdAndApvs> & detIdAndApvs, SiStripLatency & latency )
{
  if( latencies.size() != modes.size() ) {
    cout << "Error: different size for latencies = " << latencies.size() << " and modes = " << modes.size() << endl;
    exit(1);
  }
  vector<DetIdAndApvs>::const_iterator detIdAndApv = detIdAndApvs.begin();
  vector<uint16_t>::const_iterator it = latencies.begin();
  vector<uint16_t>::const_iterator modeIt = modes.begin();
  detIdAndApv = detIdAndApvs.begin();
  int latencyErrorCount = 0;
  int modeErrorCount = 0;
  for( ; detIdAndApv != detIdAndApvs.end(); ++detIdAndApv ) {
    vector<uint16_t>::const_iterator apv = detIdAndApv->apvs.begin();
    for( ; apv != detIdAndApv->apvs.end(); ++apv, ++it, ++modeIt ) {
      uint32_t detId = detIdAndApv->detId;
      uint32_t detIdAndApvValue = (detId<<2)|(*apv);
      cout << "detId = " << detIdAndApv->detId << ", apv = " << *apv << ", detIdAndApv = " << detIdAndApvValue << endl;
      cout << "latency passed = " << *it << ", latency saved = " << latency.latency(detIdAndApv->detId, *apv) << endl;
      cout << "mode passed = " << *modeIt << ", mode saved = " << latency.mode(detIdAndApv->detId, *apv) << endl;
      if( *it != latency.latency(detIdAndApv->detId, *apv) ) {
        cout << "ERROR: the latency values are different" << endl;
        ++latencyErrorCount;
      }
      if( *modeIt != latency.mode(detIdAndApv->detId, *apv) ) {
        cout << "ERROR: the mode values are different" << endl;
        ++modeErrorCount;
      }
    }
  }
  cout << endl;
  cout << "Single latency value = " << latency.singleLatency() << endl;
  cout << "Single mode value = " << latency.singleMode() << endl;

  ostream_iterator<uint16_t> output( cout, ", " );
  // Print all latencies
  vector<uint16_t> allLatenciesVector;
  latency.allLatencies(allLatenciesVector);
  cout << "All latencies in the Tracker = " << allLatenciesVector.size() << ", and are:" << endl;
  copy( allLatenciesVector.begin(), allLatenciesVector.end(), output );
  cout << endl;
  // Print all modes
  vector<uint16_t> allModesVector;
  latency.allModes(allModesVector);
  cout << "All modes in the Tracker = " << allModesVector.size() << ", and are:" << endl;
  copy( allModesVector.begin(), allModesVector.end(), output );
  cout << endl;

  cout << endl;
  cout << "Latency errors = " << latencyErrorCount << endl;
  cout << "Mode errors = " << modeErrorCount << endl;
  cout << endl;
  cout << "############################" << endl;
  cout << endl;
}

int main()
{
  vector<DetIdAndApvs> detIdAndApvs;
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

  cout << "---------------------------------" << endl;
  cout << "Testing the SiStripLatency object" << endl;
  cout << "---------------------------------" << endl << endl;

  cout << "Testing the empty case" << endl;
  cout << "----------------------" << endl;
  // Testing with all the same values. Expected final size of internal ranges and latencies = 1
  vector<int> latencyIndexes;
  vector<uint16_t> latencies;
  vector<int> modeIndexes;
  vector<uint16_t> modes;
  SiStripLatency latency1;
  test(detIdAndApvs, latencyIndexes, latencies, modeIndexes, modes, latency1);
  cout << endl;
  cout << "Filling complete, starting check" << endl;
  cout << endl;
  check(latencies, modes, detIdAndApvs, latency1);

  cout << endl;
  cout << "Testing a case with several ranges" << endl;
  cout << "----------------------------------" << endl;
  SiStripLatency latency2;
  latencyIndexes.push_back(3);
  latencyIndexes.push_back(5);
  latencyIndexes.push_back(10);
  latencyIndexes.push_back(11);
  latencies.clear();
  modeIndexes.push_back(4);
  modes.clear();
  test(detIdAndApvs, latencyIndexes, latencies, modeIndexes, modes, latency2);
  cout << endl;
  cout << "Filling complete, starting check" << endl;
  cout << endl;
  check(latencies, modes, detIdAndApvs, latency2);

  // Checking the method to retrieve all the unique (latencies,modes) pairs
  // Create a latency object with three combinations of latency and mode: (14, 37), (15, 37) and (15, 47)
  cout << "Checking the method to retrieve all the unique combinations of latency and mode" << endl;
  SiStripLatency latency3;
  latency3.put(1, 0, 14, 37);
  latency3.put(2, 0, 14, 37);
  latency3.put(3, 0, 15, 37);
  latency3.put(4, 0, 15, 47);
  cout << "Stored three combinations of latency and mode: (14, 37), (15, 37) and (15, 47)" << endl;

  vector<SiStripLatency::Latency> uniqueLatenciesAndModes(latency3.allUniqueLatencyAndModes());
  vector<SiStripLatency::Latency>::const_iterator it = uniqueLatenciesAndModes.begin();
  cout << "Reading back what is returned by the allUniqueLatencyAndModes method" << endl;
  for( ; it != uniqueLatenciesAndModes.end(); ++it ) {
    cout << "latency = " << int(it->latency) << ", mode = " << int(it->mode) << endl;
  }
  if( uniqueLatenciesAndModes.size() == 3 ) {
    cout << "Test passed" << endl;
  }
  else {
    cout << "ERROR: test not passed" << endl;
  }

  return 0;
}
