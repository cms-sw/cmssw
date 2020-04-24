#ifndef ESDATAFORMATTERV1_1_H
#define ESDATAFORMATTERV1_1_H

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

#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

class ESDigiToRaw;

class ESDataFormatterV1_1 : public ESDataFormatter {
  
  public :

  typedef  ESDataFormatter::DetDigis DetDigis;
  typedef  ESDataFormatter::Digis Digis;

  typedef ESDataFormatter::Word8  Word8;
  typedef ESDataFormatter::Word16 Word16;
  typedef ESDataFormatter::Word32 Word32;
  typedef ESDataFormatter::Word64 Word64;

  ESDataFormatterV1_1(const edm::ParameterSet& ps);
  ~ESDataFormatterV1_1() override;

  void DigiToRaw(int fedId, Digis & digis, FEDRawData& fedRawData, const Meta_Data & meta_data) const override;


  private :    
    


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
