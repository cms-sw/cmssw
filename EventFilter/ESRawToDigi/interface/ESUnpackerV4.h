#ifndef ESUNPACKERV4_H
#define ESUNPCAKERV4_H

#include <iostream>
#include <vector>
#include <bitset>
#include <sstream>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESRawDataCollections.h"
#include "TBDataFormats/ESTBRawData/interface/ESLocalRawDataCollections.h"

using namespace std;
using namespace edm;

class ESDigiToRaw;

class ESUnpackerV4 {
  
  public :
      
  typedef unsigned int Word32;
  typedef long long Word64;  

  ESUnpackerV4(const ParameterSet& ps);
  ~ESUnpackerV4();

  void interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis);
  void word2digi(int kchip, int kPACE[4], const Word64 & word, ESDigiCollection & digis);

  void setRunNumber(int i) {run_number_ = i;};
  void setOrbitNumber(int i) {orbit_number_ = i;};
  void setBX(int i) {bx_ = i;};
  void setLV1(int i) {lv1_ = i;};
  void setTriggerType(int i) {trgtype_ = i;};

  private :    

  const ParameterSet pset_;
    
  int fedId_;
  int run_number_;
  int orbit_number_;
  int bx_;
  int lv1_;
  int dac_;
  int gain_;
  int precision_;
  int runtype_;
  int seqtype_;
  int trgtype_;
  int optoRX0_;
  int optoRX1_;
  int optoRX2_;
  int FEch_[36];

  bool debug_;
  FileInPath lookup_;

  string print(const Word64 & word) const;

  protected :

  Word64 m1, m2, m4, m5, m6, m8, m12, m16, m32;

   int zside_[4288][4], pl_[4288][4], x_[4288][4], y_[4288][4]; 

};

#endif
