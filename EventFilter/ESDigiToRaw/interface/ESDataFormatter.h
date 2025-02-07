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
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ESDataFormatter {
public:
  struct Meta_Data {
    edm::RunNumber_t run_number = 0;
    edm::EventNumber_t orbit_number = 0;
    unsigned int bx = 0;
    edm::EventNumber_t lv1 = 0;
    unsigned int kchip_bc = 0;
    unsigned int kchip_ec = 0;
    Meta_Data() = default;
    Meta_Data(edm::RunNumber_t r,
              edm::EventNumber_t o,
              unsigned int b,
              edm::EventNumber_t l,
              unsigned int k_bc,
              unsigned int k_ec)
        : run_number(r), orbit_number(o), bx(b), lv1(l), kchip_bc(k_bc), kchip_ec(k_ec) {}
  };

  typedef std::vector<ESDataFrame> DetDigis;
  typedef std::map<int, DetDigis> Digis;

  typedef uint8_t Word8;
  typedef uint16_t Word16;
  typedef uint32_t Word32;
  typedef uint64_t Word64;

  ESDataFormatter(const edm::ParameterSet& ps)
      : pset_(ps),
        trgtype_(0),
        debug_(pset_.getUntrackedParameter<bool>("debugMode", false)),
        printInHex_(pset_.getUntrackedParameter<bool>("printInHex", false)) {}
  virtual ~ESDataFormatter() {}

  virtual void DigiToRaw(int fedId, Digis& digis, FEDRawData& fedRawData, const Meta_Data& meta_data) const = 0;

protected:
  const edm::ParameterSet pset_;
  const int trgtype_;
  const bool debug_;
  const bool printInHex_;

  int formatMajor_;
  int formatMinor_;

  std::string print(const Word64& word) const;
  std::string print(const Word16& word) const;
};

#endif
