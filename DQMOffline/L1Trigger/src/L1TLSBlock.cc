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
 *    2012/10/23 12:01:01: Class
 *
 * Todo: see header file
 *
 * $Date: 2012/10/23 12:01:01 $
 * $Revision: 0.0 $
 *
 */

// 

// This class header
#include "DQMOffline/L1Trigger/interface/L1TLSBlock.h"

// System include files
// --

//// User include files
//#include "DQMServices/Core/interface/DQMStore.h"
//
//#include "DataFormats/Scalers/interface/LumiScalers.h"
//#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
//#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
//
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
//
//#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event
//
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
//#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
//#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
//#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
/////
//// Luminosity Information
////#include "DataFormats/Luminosity/interface/LumiDetails.h"
////#include "DataFormats/Luminosity/interface/LumiSummary.h"
//
//// L1TMonitor includes
/////#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
//#include "DQMOffline/L1Trigger/interface/L1TMenuHelper.h"
//
//#include "TList.h"

using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

L1TLSBlock::L1TLSBlock(){

  // Initialize LumiRangeList object. This will be redone at every doBlocking call. Perhaps rethink this
  initializeIO(false);
  
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

L1TLSBlock::~L1TLSBlock(){}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

L1TLSBlock::LumiRangeList L1TLSBlock::doBlocking(LumiTestDoubleList inputList, double threshold, BLOCKBY blockingMethod)
{
  inputDoubleList_ = inputList;
  thresholdD_ = threshold;
  
  initializeIO(true);
  
  orderTestDoubleList();
  
  switch(blockingMethod){
  case STATISTICS:
    blockByStatistics();
    break;
  default:
    cout << "[L1TLSBlock::doBlocking()]: Blocking method does not exist or is not implemented" << endl;
  }
  
  
  return outputList_;
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

L1TLSBlock::LumiRangeList L1TLSBlock::doBlocking(LumiTestIntList inputList, int threshold, BLOCKBY blockingMethod)
{
  inputIntList_ = inputList;
  thresholdI_ = threshold;
  
  initializeIO(true);
  
  orderTestIntList();
  
  switch(blockingMethod){
  case STATISTICS:
    cout << "[L1TLSBlock::doBlocking()]: Blocking by statistics require doubles as inputs for test variable and threshold" << endl;
    break;
  default:
    cout << "[L1TLSBlock::doBlocking()]: Blocking method does not exist or is not implemented" << endl;
  }
  
  return outputList_;
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

void L1TLSBlock::initializeIO(bool outputOnly){
  if(!outputOnly){
    inputIntList_.clear();
    inputDoubleList_.clear();
  }
  outputList_.clear();
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

void L1TLSBlock::orderTestDoubleList(){
  std::sort(inputDoubleList_.begin(), inputDoubleList_.end(), sort_pair_first<int, double>());
  // std::sort(v.begin(), v.end(), sort_pair_second<int, std::greater<int> >());
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

void L1TLSBlock::orderTestIntList(){
  std::sort(inputIntList_.begin(), inputIntList_.end(), sort_pair_first<int, int>());
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

void L1TLSBlock::blockByStatistics(){
  LumiRange currentRange;
  double currentError(0);
  bool resetFlag(true);

  for(LumiTestDoubleList::iterator i=inputDoubleList_.begin(); i!=inputDoubleList_.end(); i++){
    if(resetFlag){
      currentRange = std::make_pair(i->first,i->first);
      resetFlag = false;
    }
    else
      currentRange = std::make_pair(currentRange.first,i->first);
    currentError = computeErrorFromRange(currentRange);
    if(currentError < thresholdD_){
      outputList_.push_back(currentRange);
      resetFlag = true;
    }
    
  }
}

double L1TLSBlock::computeErrorFromRange(LumiRange& lumiRange){
  std::vector<double> errorList;
  errorList.clear();

  for(size_t i=0; i < inputDoubleList_.size(); i++){
    if(inputDoubleList_[i].first>lumiRange.first && inputDoubleList_[i].first<lumiRange.second)
      errorList.push_back(inputDoubleList_[i].second);
  }
  
  double error(-1);
  for(size_t i=0; i<errorList.size(); i++)
    error += 1 / (errorList[i] * errorList[i] );
  return error;

}
