// Authors: Yulun Miao, Northwestern University

#ifndef CondFormats_HGCalObjects_HGCalTriggerConfiguration_h
#define CondFormats_HGCalObjects_HGCalTriggerConfiguration_h
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include <map>
#include <vector>
#include <iomanip>

// @short configuration for ECON-T module
struct HGCalECONTConfig {
  uint8_t density;  //lsb at the input TC from ROC
  uint8_t dropLSB;  //lsb at the output during the packing
  uint8_t
      select;  //0 = Threshold Sum (TS), 1 = Super Trigger Cell (STC), 2 = Best Choice (BC), 3 = Repeater, 4=Autoencoder (AE).
  uint8_t stcType;       //0 = STC4B(5E+4M), 1 = STC16(5E+4M), 2 = CTC4A(4E+3M), 3 = STC4A(4E+3M), 4 = CTC4B(5E+3M)
  uint8_t eportTxNumen;  //number of elinks
  uint8_t sumType;       //0: MS summed over 48-N TCs, 1: MS summed over 48 TCs
  std::vector<uint16_t> calv;  //12-bit calibration for 48 TCs
  std::vector<uint8_t> tcMux;  //multiplexer between HGCROC and TC to ECONT
  std::vector<uint32_t> offset;
  COND_SERIALIZABLE;
};

// @short configuration for TDAQ block
struct HGCalTDAQConfig {
  uint32_t tdaqBlockHeaderMarker;  // begin of event marker/identifier for
  // nBX should be dynamically read from the TDAQ header
  std::vector<HGCalECONTConfig> econts;  // if size==0, then this TDAQ block is skipped
  COND_SERIALIZABLE;
};

// @short configuration for FED
struct HGCalTriggerFedConfig {
  std::vector<HGCalTDAQConfig> tdaqs;
  std::vector<int32_t> econtSwapOffset;
  std::map<uint8_t, std::vector<uint8_t>> elinksMap;
  // TODO: if we want to add information checking for S-Link header/trailer
  COND_SERIALIZABLE;
};

/**
 *  @short Main HGCal configuration with a tree structure of vectors of
 *         HGCalTriggerFedConfig/HGCalTriggerFedConfig/HGCalTDAQConfig structs as follows:
 *         config.feds[dense_fed_idx].tdaq[dense_tdaq_idx].econt[dense_econt_idx]
 **/
class HGCalTriggerConfiguration {
public:
  std::vector<HGCalTriggerFedConfig> feds;

private:
  COND_SERIALIZABLE;
};

inline std::ostream& operator<<(std::ostream& os, const HGCalTriggerConfiguration& config) {
  std::size_t nfed = config.feds.size();
  os << "HGCalConfiguration( nfed=" << std::dec << nfed << std::endl;
  for (std::size_t ifed = 0; ifed < nfed; ifed++) {
    HGCalTriggerFedConfig fedConfig = config.feds[ifed];
    os << "fed[" << std::dec << ifed << "] with " << std::dec << fedConfig.tdaqs.size() << " TDAQ blocks" << std::endl
       << "SwapOffset=[";
    for (std::size_t iecont = 0; iecont < fedConfig.econtSwapOffset.size(); iecont++) {
      os << fedConfig.econtSwapOffset[iecont] << ", ";
    }
    os << "]," << std::endl;
    for (std::size_t itdaq = 0; itdaq < fedConfig.tdaqs.size(); itdaq++) {
      HGCalTDAQConfig tdaqConfig = fedConfig.tdaqs[itdaq];
      os << "fed[" << std::dec << ifed << "].tdaq[" << itdaq << "], headerMarker = 0x" << std::hex << std::setfill('0')
         << std::setw(8) << tdaqConfig.tdaqBlockHeaderMarker << std::endl;
      if (tdaqConfig.econts.size() == 0) {
        os << "with no active ECON-Ts, skipped" << std::endl;
        continue;
      }
      os << " with " << std::dec << tdaqConfig.econts.size() << " active ECON-Ts" << std::endl;
      for (unsigned int iecont = 0; iecont < tdaqConfig.econts.size(); iecont++) {
        HGCalECONTConfig econtConfig = tdaqConfig.econts[iecont];
        os << "fed[" << std::dec << ifed << "].tdaq[" << itdaq << "].econt[" << iecont << "], density = " << std::dec
           << (int)econtConfig.density << ", dropLSB = " << std::dec << (int)econtConfig.dropLSB
           << ", select = " << std::dec << (int)econtConfig.select << ", stcType = " << std::dec
           << (int)econtConfig.stcType << ", eportTxNumen = " << std::dec << (int)econtConfig.eportTxNumen
           << ", sumType = " << std::dec << (int)econtConfig.sumType << "," << std::endl
           << "calv = [";
        for (unsigned int i = 0; i < econtConfig.calv.size(); i++) {
          os << "(0x" << std::hex << std::setfill('0') << std::setw(3) << econtConfig.calv[i] << ", " << std::dec
             << (float)econtConfig.calv[i] / 2048 << "), ";
        }
        os << "]," << std::endl << "mux = [";
        for (unsigned int i = 0; i < econtConfig.tcMux.size(); i++) {
          os << std::dec << (int)econtConfig.tcMux[i] << ", ";
        }
        os << "]," << std::endl << "offset = [";
        for (unsigned int i = 0; i < econtConfig.offset.size(); i++) {
          os << std::dec << (int)econtConfig.offset[i] << ", ";
        }
        os << std::endl;
      }
    }
  }
  return os;
}

#endif
