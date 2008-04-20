#ifndef ESUNPACKERCT_H
#define ESUNPACKERCT_H

#include <iostream>
#include <vector>
#include <bitset>
#include <sstream>
#include <map>

#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESRawDataCollections.h"
#include "TBDataFormats/ESTBRawData/interface/ESLocalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;
using namespace edm;

class ESDigiToRawCT;

class ESUnpackerCT {
  
  public :

  typedef unsigned char Word8;
  typedef unsigned short Word16;
  typedef unsigned int Word32;
  typedef long long Word64;  

  ESUnpackerCT(const ParameterSet& ps);
  ~ESUnpackerCT();
  
  void interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis);
  void word2DCCHeader(const vector<Word64>& word);
  void word2CTS(const vector<Word64> & word);
  void word2Crepe(const vector<Word64> & word);
  void word2TLS(const vector<Word64> & word);
  void word2digi(int kchip, const vector<Word16> & word, ESLocalRawDataCollection & kchips, ESDigiCollection & digis);

  void setRunNumber(int i) {run_number_ = i;};
  void setOrbitNumber(int i) {orbit_number_ = i;};
  void setBX(int i) {bx_ = i;};
  void setLV1(int i) {lv1_ = i;};
  void setTriggerType(int i) {trgType_ = i;};

  private :    

  const ParameterSet pset_;

  bool debug_;

  vector<Word64> DCCHeader;

  int fedId_;
  int run_number_;
  int orbit_number_;
  int bx_;
  int lv1_;
  int trgType_;
  int evtLen_;
  int DCCErr_;
  int runNum_;
  int runType_;
  int compFlag_;
  int orbit_;
  int vminor_;
  int vmajor_;
  int optoRX0_;
  int optoRX1_;
  int optoRX2_;
  int FEch_[36];
  int packetLen_;
  int bc_;
  int ev_;
  int BMMeasurements_;
  int beginOfSpillSec_;
  int beginOfSpilliMilliSec_;
  int endOfSpillSec_;
  int endOfSpilliMilliSec_;
  int beginOfSpillLV1_;
  int endOfSpillLV1_;
  int timestamp_sec_;
  int timestamp_usec_;
  int spillNum_;
  int evtInSpill_;
  int camacErr_;
  int vmeErr_;
  int exRunNum_;
  int ADCchStatus_[12];
  int ADCch_[12];
  int TDCStatus_[8];
  int TDC_[8];

  string print(const Word64 & word) const;
  string print(const Word16 & word) const;
  string print(const Word8 & word) const;

};

#endif
