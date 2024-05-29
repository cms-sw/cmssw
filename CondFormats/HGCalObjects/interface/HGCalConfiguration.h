// Authors: Izaak Neutelings (May 2024)
// Sources: https://docs.google.com/spreadsheets/d/13G7sOjssqw4B5AtOcQV3g0W01oZUOMM6Hm_DduxBEPU
#ifndef CondFormats_HGCalObjects_HGCalCondSerializableConfig_h
#define CondFormats_HGCalObjects_HGCalCondSerializableConfig_h

//#include <string>
#include <map>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
//#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

/**
 *  @short global HGCal configurations needed for raw data unpacking
 */
struct HGCalROCConfig_t { // configuration for ECON eRX (one half of HGROC)
  uint32_t charMode;       // characterization mode; determines data fields in ROC dataframe
  //uint32_t clockPhase;     // fine adjustment of the phase within the 40 MHz
  //uint32_t L1AcceptOffset; // coarse adjustment to get the peak in the right place
  //uint32_t injChannels;    // injected channels for injection scan: 2b word to identify if connected or not+info no capacitor chosen
  //uint32_t injCharge;      // injected charge for injection scan: convert it to a float in units of fC offline (DAC setting?)
  COND_SERIALIZABLE;
};
struct HGCalECONDConfig_t { // configuration for ECON-D
  uint32_t headerMarker; // begin of event marker/identifier for ECON-D
  std::vector<HGCalROCConfig_t> rocs;
  COND_SERIALIZABLE;
};
struct HGCalFedConfig_t { // configuration for FED
  bool mismatchPassthroughMode; // ignore ECON-D packet mismatches
  uint32_t cbHeaderMarker;      // begin of event marker/identifier for capture block
  uint32_t slinkHeaderMarker;   // begin of event marker/identifier for S-link
  //uint32_t delay; // delay
  std::vector<HGCalECONDConfig_t> econds;
  COND_SERIALIZABLE;
};

/**
 *  @short Main HGCal configuration
 */
//config.fed[0].econds[0].rocs[dense idx]
class HGCalConfiguration {
  public:
    //HGCalMappingModuleIndexer map;
    //std::vector<HGCalROCConfig_t> rocs;
    ////std::vector<HGCalECONDConfig_t> econds;
    std::vector<HGCalFedConfig_t> feds;
    //friend std::ostream& operator<<(std::ostream&, const HGCalCondSerializableConfig&);
  private:
    COND_SERIALIZABLE;
};

#endif