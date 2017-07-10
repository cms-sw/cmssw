#ifndef EventFilter_L1TRawToDigi_Omtf_CscDataWord64_H
#define EventFilter_L1TRawToDigi_Omtf_CscDataWord64_H

#include<iostream>
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

namespace omtf {

class CscDataWord64 {
public:
  CscDataWord64(const Word64 & data) : rawData(data) {}
  CscDataWord64() : rawData(Word64(DataWord64::csc)<<60) {}

  unsigned int type() const { return type_;}
  unsigned int bxNum() const { return bxNum_; }
  unsigned int hitNum() const { return hitNum_; }
  unsigned int wireGroup() const { return keyWG_; }
  unsigned int quality() const { return quality_; }
  unsigned int clctPattern() const { return clctPattern_; }
  unsigned int cscID() const { return cscID_; }
  unsigned int halfStrip() const { return halfStrip_; }
  unsigned int linkNum() const { return linkNum_;}
  unsigned int station() const { return station_; }
  unsigned int bend() const { return lr_; }
  unsigned int valid() const { return vp_; }

  friend class OmtfPacker;
  friend class CscPacker;
  friend std::ostream & operator<< (std::ostream &out, const CscDataWord64 &o);

private:
  union {
    uint64_t rawData;
    struct {               // watch out - bit fields are  is implementation-defined
      uint64_t dummy7_      : 3; //2:0   unused, oryginalnie TBIN Num
      uint64_t vp_          : 1; //3:3   VP
      uint64_t station_     : 3; //6:4   Station
      uint64_t af_          : 1; //7:7   AF
      uint64_t dummy6_      : 4; //11:8  unused, oryginalnie EPC
      uint64_t sm_          : 1; //12:12 unused, oryginalnie SM
      uint64_t se_          : 1; //13:13 SE
      uint64_t dummy5_      : 1; //14:14 unused, oryginalnie AFEF
      uint64_t dummy4_      : 2; //16:15 unused, oryginalnie ME BXN [11:0]
      uint64_t nit_         : 1; //17:17 NIT
      uint64_t cik_         : 1; //18:18 CIK
      uint64_t dummy3_      : 1; //19:19 unused, oryginalnie AFFF
      uint64_t linkNum_     : 6; //25:20 numer linku CSC
      uint64_t halfStrip_   : 8; //33:26 CLCT key half-strip [7:0]
      uint64_t cscID_       : 4; //37:34 CSC ID [3:0]
      uint64_t lr_          : 1; //38:38 L/R
      uint64_t dummy2_      : 1; //39:39 unused, oryginalnie BXE
      uint64_t dummy1_      : 1; //40:40 unused, oryginalnie BC0
      uint64_t clctPattern_ : 4; //44:41 CLCT pattern [3:0]              4b
      uint64_t quality_     : 4; //48:45 Quality [3:0]
      uint64_t keyWG_       : 7; //55:49 Key wire group [6:0]
      uint64_t hitNum_      : 1; //56:56 int in chamber 0 or 1
      uint64_t bxNum_       : 3; //59:57 SBXN
      uint64_t type_        : 4; //63:60 CSC identifier 0xC
    };
  };
};

} //namespace Omtf
#endif

