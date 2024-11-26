// Authors: Izaak Neutelings (May 2024)
// Sources: https://docs.google.com/spreadsheets/d/13G7sOjssqw4B5AtOcQV3g0W01oZUOMM6Hm_DduxBEPU
#ifndef CondFormats_HGCalObjects_HGCalConfiguraton_h
#define CondFormats_HGCalObjects_HGCalConfiguraton_h
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include <map>
#include <vector>

// @short configuration for ECON eRX (one half of HGROC)
struct HGCalROCConfig {
  uint32_t charMode;  // characterization mode; determines data fields in ROC dataframe
  uint8_t gain;       // pre-amp gain used (1: 80 fC, 2: 160 fC, 4: 320 fC)
  //uint32_t clockPhase;     // fine adjustment of the phase within the 40 MHz
  //uint32_t L1AcceptOffset; // coarse adjustment to get the peak in the right place
  //uint32_t injChannels;    // injected channels for injection scan: 2b word to identify if connected or not+info no capacitor chosen
  //uint32_t injCharge;      // injected charge for injection scan: convert it to a float in units of fC offline (DAC setting?)
  COND_SERIALIZABLE;
};

// @short configuration for ECON-D module
struct HGCalECONDConfig {
  //std::string typecode;
  uint32_t headerMarker;  // begin of event marker/identifier for ECON-D
  uint32_t passThrough;   //pass through mode (this is just as check as it'll be in the ECON-D header anyway)
  std::vector<HGCalROCConfig> rocs;
  COND_SERIALIZABLE;
};

// @short configuration for FED
struct HGCalFedConfig {
  bool mismatchPassthroughMode;  // ignore ECON-D packet mismatches
  uint32_t cbHeaderMarker;       // begin of event marker/identifier for capture block
  uint32_t slinkHeaderMarker;    // begin of event marker/identifier for S-link
  //uint32_t delay; // delay
  std::vector<HGCalECONDConfig> econds;
  COND_SERIALIZABLE;
};

/**
 *  @short Main HGCal configuration with a tree structure of vectors of
 *         HGCalFedConfig/HGCalECONDConfig/HGCalROCConfig structs as follows:
 %         config.feds[dense_fed_idx].econds[dense_econd_idx].rocs[dense_eRx_idx]
 **/
class HGCalConfiguration {
public:
  std::vector<HGCalFedConfig> feds;

private:
  COND_SERIALIZABLE;
};

inline std::ostream& operator<<(std::ostream& os, const HGCalConfiguration& config) {
  uint32_t nfed = config.feds.size();
  uint32_t ntotmod = 0;
  uint32_t ntotroc = 0;
  for (auto const& fed : config.feds) {
    ntotmod += fed.econds.size();  // number of ECON-D modules for this FED
    for (auto const& mod : fed.econds) {
      ntotroc += mod.rocs.size();  // number of eRx half-ROCs for this ECON-D module
    }
  }
  os << "HGCalConfiguration(nfed=" << nfed << ",ntotmod=" << ntotmod << ",ntotroc=" << ntotroc << ")";
  return os;
}

#endif
