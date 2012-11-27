#ifndef DQMOffline_L1Trigger_L1TLSBlock_h
#define DQMOffline_L1Trigger_L1TLSBlock_h

/*
 * \class L1TLSBlock
 *
 *
 * Description: offline DQM class for LS blocking
 * 
 * Implementation:
 *   <TODO: enter implementation details>
 *
 * \author: Pietro Vischia - LIP Lisbon pietro.vischia@gmail.com
 *
 * Changelog:
 *    2012/10/23 12:01:01: Creation, infrastructure and blocking by statistics
 *
 * Todo:
 *  - Activate the "method" string usage instead of base one
 *  - improve the switch method (instead of calling functions one should define a template - it is probably an uberpain, but would be more neat)
 *  - in doBlocking, substitute the switch default case with a cms::Exception
 *  - Cleanup includes
 *  - Add other blocking methods
 *  - 
 *
 * $Date: 2012/10/23 11:01:01 $
 * $Revision: 0.0 $
 *
 */

// System include files
//#include <memory>
//#include <unistd.h>

// User include files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/LuminosityBlock.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
//#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQMServices/Core/interface/MonitorElement.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"
//
//#include <TString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

// Forward declarations



// Helper
template<class T1, class T2, class Pred = std::less<T1> >
struct sort_pair_first {
  bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
    Pred p;
    return p(left.first, right.first);
  }
};

// Class declaration
class L1TLSBlock {
  
 public: 
  // typedefs
  typedef std::vector<std::pair<int,double> > LumiTestDoubleList;
  typedef std::vector<std::pair<int,double> > LumiTestIntList;
  typedef std::pair<int,int> LumiRange;
  typedef std::vector<LumiRange> LumiRangeList;
  
  enum BLOCKBY{STATISTICS, N_BLOCKINGBY};
 public:
  // Constructor
  L1TLSBlock();   
  // Destructor
  virtual ~L1TLSBlock();                     
  LumiRangeList doBlocking(LumiTestDoubleList, double, BLOCKBY);
  LumiRangeList doBlocking(LumiTestIntList, int,    BLOCKBY);
  
  // Private Methods
 private:
  void initializeIO(bool);
  void blockByStatistics();
  void orderTestDoubleList();
  void orderTestIntList();
  
  double computeErrorFromRange(LumiRange&);
  
  // Variables
 private:
  LumiTestIntList inputIntList_;
  LumiTestDoubleList inputDoubleList_;
  LumiRangeList outputList_;
  double thresholdD_;
  int thresholdI_;
 
};

#endif
