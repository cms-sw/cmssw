#ifndef CaloOnlineTools_HcalOnlineDb_EMap_h
#define CaloOnlineTools_HcalOnlineDb_EMap_h

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include <cstdint>
#include <string>
#include <vector>

class EMap {
public:
  EMap(const HcalElectronicsMap* map);

  struct EMapRow {
    uint32_t rawId = 0;
    int crate = 0;
    int slot = 0;
    int dcc = 0;
    int spigot = 0;
    int fiber = 0;
    int fiberchan = 0;
    int ieta = 0;
    int iphi = 0;
    int idepth = 0;
    std::string topbottom = "";
    std::string subdet = "";
    int zdc_zside = 0;
    int zdc_channel = 0;
    std::string zdc_section = "UNKNOWN";
  };

  const std::vector<EMapRow>& get_map() const noexcept { return map; };

  std::string getSubdetectorString(const HcalSubdetector& _det);
  std::string getZDCSectionString(const HcalZDCDetId::Section& _section);

protected:
  std::vector<EMapRow> map;
};

#endif
