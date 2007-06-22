#ifndef ESUNPACKERTB_H
#define ESUNPCAKERTB_H

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

class ESDigiToRawTB;

class ESUnpackerTB {
  
  public :

  typedef unsigned char Word8;
  typedef unsigned short Word16;
  typedef unsigned int Word32;
  typedef long long Word64;  

  ESUnpackerTB(const ParameterSet& ps);
  ~ESUnpackerTB();
  
  void interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis);
  void word2DCCHeader(const vector<Word64>& word);
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

  string print(const Word64 & word) const;
  string print(const Word16 & word) const;
  string print(const Word8 & word) const;

  protected :

  static const int bDHEAD, bDH, bDEL, bDERR, bDRUN, bDRUNTYPE, bDTRGTYPE, bDCOMFLAG, bDORBIT;
  static const int bDVMINOR, bDVMAJOR, bDCH, bDOPTO;  
  static const int sDHEAD, sDH, sDEL, sDERR, sDRUN, sDRUNTYPE, sDTRGTYPE, sDCOMFLAG, sDORBIT;
  static const int sDVMINOR, sDVMAJOR, sDCH, sDOPTO;  
  static const int bKEC, bKFLAG2, bKBC, bKFLAG1, bKET, bKCRC, bKCE, bKID, bFIBER, bKHEAD1, bKHEAD2;
  static const int sKEC, sKFLAG2, sKBC, sKFLAG1, sKET, sKCRC, sKCE, sKID, sFIBER, sKHEAD1, sKHEAD2;
  static const int bHEAD,  bE1, bE0, bSTRIP, bPACE, bADC2, bADC1, bADC0;
  static const int sHEAD,  sE1, sE0, sSTRIP, sPACE, sADC2, sADC1, sADC0;


};

#endif
