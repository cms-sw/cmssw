#ifndef ESDATAFORMATTER_H
#define ESDATAFORMATTER_H

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

class ESDataFormatter {
  
  public :
    
  typedef std::vector<ESDataFrame> DetDigis;
  typedef std::map<int, DetDigis> Digis;

  typedef uint8_t  Word8;
  typedef uint16_t Word16;
  typedef uint32_t Word32;
  typedef uint64_t Word64;

  ESDataFormatter(const edm::ParameterSet& ps) : 
    pset_(ps), run_number_(0), orbit_number_(0), bx_(0), lv1_(0), trgtype_(0),
    kchip_bc_(0), kchip_ec_(0) { 
    debug_ = pset_.getUntrackedParameter<bool>("debugMode", false);
    printInHex_ = pset_.getUntrackedParameter<bool>("printInHex", false);
  };
  virtual ~ESDataFormatter() {};

  virtual void DigiToRaw(int fedId, Digis & digis, FEDRawData& fedRawData) = 0;

  virtual void setRunNumber(int i) {run_number_ = i;};
  virtual void setOrbitNumber(int i) {orbit_number_ = i;};
  virtual void setBX(int i) {bx_ = i;};
  virtual void setLV1(int i) {lv1_ = i;};
  virtual void setTriggerType(int i) {trgtype_ = i;};
  virtual void setKchipBC(int i) {kchip_bc_ = i;};
  virtual void setKchipEC(int i) {kchip_ec_ = i;};

  protected :    
    
  const edm::ParameterSet pset_;

  int run_number_;
  int orbit_number_;
  int bx_;
  int lv1_;
  int trgtype_;
  int kchip_bc_; 
  int kchip_ec_;

  bool debug_;
  bool printInHex_; 

  int formatMajor_; 
  int formatMinor_;

  std::string print(const Word64 & word) const;
  std::string print(const Word16 & word) const;

};

#endif
