#ifndef EventFilter_ESRawToDigi_ESUnpacker_h
#define EventFilter_ESRawToDigi_ESUnpacker_h

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
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"




class ESDigiToRaw;

class ESUnpacker {
  
  public :
      
  typedef unsigned int Word32;
  typedef unsigned long long Word64;

  ESUnpacker(const edm::ParameterSet& ps);
  ~ESUnpacker();

  void interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis);
  void word2digi(int kchip, int kPACE[4], const Word64 & word, ESDigiCollection & digis);

  void setRunNumber(int i) {run_number_ = i;};
  void setOrbitNumber(int i) {orbit_number_ = i;};
  void setBX(int i) {bx_ = i;};
  void setLV1(int i) {lv1_ = i;};
  void setTriggerType(int i) {trgtype_ = i;};

  private :    

  const edm::ParameterSet pset_;
    
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
  int vminor_;
  int vmajor_;
  int optoRX0_;
  int optoRX1_;
  int optoRX2_;
  int FEch_[36];

  bool debug_;
  edm::FileInPath lookup_;

  std::string print(const Word64 & word) const;

  protected :

  Word64 m1, m2, m4, m5, m6, m8, m12, m16, m32;

   int zside_[4288][4], pl_[4288][4], x_[4288][4], y_[4288][4]; 

};

#endif
