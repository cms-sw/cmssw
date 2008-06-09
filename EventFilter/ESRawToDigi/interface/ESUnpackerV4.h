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

using namespace std;
using namespace edm;

class ESDigiToRaw;

class ESUnpackerV4 {
  
  public :
      
  typedef unsigned int Word32;
  typedef long long Word64;  

  ESUnpackerV4(const ParameterSet& ps);
  ~ESUnpackerV4();

  void interpretRawData(int fedId, const FEDRawData & rawData, ESDigiCollection & digis);
  void word2digi(int kchip, const Word64 & word, ESDigiCollection & digis);

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
  int trgtype_;

  bool debug_;
  FileInPath lookup_;

  string print(const Word64 & word) const;

  protected :

  Word64 m2, m4, m5, m8, m16, m32;

  int zside_[1511][4], pl_[1511][4], x_[1511][4], y_[1511][4]; 

};

#endif
