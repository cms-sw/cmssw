#include "CondFormats/CSCObjects/src/headers.h"

namespace CondFormats_CSCObjects {
  struct dictionary {
    std::vector<CSCPedestals::Item> pedcontainer1;
    std::map<int, std::vector<CSCPedestals::Item> > pedmap;
    std::vector<CSCDBPedestals::Item> pedcontainer2;

    std::vector<CSCGains::Item> gcontainer1;
    std::map<int, std::vector<CSCGains::Item> > gmap;
    std::vector<CSCDBGains::Item> gcontainer2;

    std::vector<CSCNoiseMatrix::Item> mcontainer1;
    std::map<int, std::vector<CSCNoiseMatrix::Item> > mmap;
    std::vector<CSCDBNoiseMatrix::Item> mcontainer2;

    std::vector<CSCcrosstalk::Item> ccontainer1;
    std::map<int, std::vector<CSCcrosstalk::Item> > cmap;
    std::vector<CSCDBCrosstalk::Item> ccontainer2;

    std::vector<CSCBadStrips::BadChamber> bschmcontainer;
    std::vector<CSCBadStrips::BadChannel> bschncontainer;

    std::vector<CSCBadWires::BadChamber> bwchmcontainer;
    std::vector<CSCBadWires::BadChannel> bwchncontainer;

    std::vector<CSCDBChipSpeedCorrection::Item> chipCorrcontainer1;
    std::map<int, std::vector<CSCDBChipSpeedCorrection::Item> > chipCorrmap;
    std::vector<CSCDBChipSpeedCorrection::Item> chipCorrcontainer2;

    std::vector<CSCDBGasGainCorrection::Item> gasGainCorrcontainer1;
    std::map<int, std::vector<CSCDBGasGainCorrection::Item> > gasGainCorrmap;
    std::vector<CSCDBGasGainCorrection::Item> gasGainCorrcontainer2;

    std::map<int, CSCMapItem::MapItem> chmap;
    std::pair<const int, CSCMapItem::MapItem> chmapvalue;

    std::vector<CSCMapItem::MapItem> chvector;

    std::vector<cscdqm::DCSAddressType> CSCDQM_DCSAddressType_V;
    std::vector<cscdqm::DCSData> CSCDQM_DCSData_V;
    std::vector<cscdqm::TempMeasType> CSCDQM_TempMeasType_V;
    std::vector<cscdqm::HVVMeasType> CSCDQM_HVVMeasType_V;
    std::vector<cscdqm::LVVMeasType> CSCDQM_LVVMeasType_V;
    std::vector<cscdqm::LVIMeasType> CSCDQM_LVIMeasType_V;
  };
}  // namespace CondFormats_CSCObjects
