#ifndef ESDATAFORMATTERV4_H
#define ESDATAFORMATTERV4_H

#include <iostream>
#include <vector>
#include <bitset>
#include <sstream>
#include <map>
#include <fstream>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

class ESDigiToRaw;

class ESDataFormatterV4 : public ESDataFormatter {
public:
  typedef ESDataFormatter::DetDigis DetDigis;
  typedef ESDataFormatter::Digis Digis;

  typedef ESDataFormatter::Word8 Word8;
  typedef ESDataFormatter::Word16 Word16;
  typedef ESDataFormatter::Word32 Word32;
  typedef ESDataFormatter::Word64 Word64;

  ESDataFormatterV4(const edm::ParameterSet& ps);
  ~ESDataFormatterV4() override;

  void DigiToRaw(int fedId, Digis& digis, FEDRawData& fedRawData, Meta_Data const& meta_data) const override;

private:
  edm::FileInPath lookup_;
  int fedId_[2][2][40][40];
  int kchipId_[2][2][40][40];
  int paceId_[2][2][40][40];
  int bundleId_[2][2][40][40];
  int fiberId_[2][2][40][40];
  int optoId_[2][2][40][40];

  bool fedIdOptoRx_[56][3];
  bool fedIdOptoRxFiber_[56][3][12];

protected:
  static const int bDHEAD, bDH, bDEL, bDERR, bDRUN, bDRUNTYPE, bDTRGTYPE, bDCOMFLAG, bDORBIT;
  static const int bDVMINOR, bDVMAJOR, bDCH, bDOPTO;
  static const int sDHEAD, sDH, sDEL, sDERR, sDRUN, sDRUNTYPE, sDTRGTYPE, sDCOMFLAG, sDORBIT;
  static const int sDVMINOR, sDVMAJOR, sDCH, sDOPTO;

  static const int bKEC, bKFLAG2, bKBC, bKFLAG1, bKET, bKCRC, bKCE, bKID, bFIBER, bKHEAD1, bKHEAD2;
  static const int sKEC, sKFLAG2, sKBC, sKFLAG1, sKET, sKCRC, sKCE, sKID, sFIBER, sKHEAD1, sKHEAD2;
  static const int bKHEAD;
  static const int sKHEAD;
  static const int bHEAD, bE1, bE0, bSTRIP, bPACE, bADC2, bADC1, bADC0;
  static const int sHEAD, sE1, sE0, sSTRIP, sPACE, sADC2, sADC1, sADC0;

  static const int bOEMUTTCEC, bOEMUTTCBC, bOEMUKEC, bOHEAD;
  static const int sOEMUTTCEC, sOEMUTTCBC, sOEMUKEC, sOHEAD;
};

#endif
