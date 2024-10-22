#ifndef PixelDACNames_h
#define PixelDACNames_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelDACNames.h
*   \brief A dummy class with ALL public variables 
*
*   A longer explanation will be placed here later
*/

#include <string>

namespace pos {

  const std::string k_DACName_Vdd = "Vdd";
  const std::string k_DACName_Vana = "Vana";
  const std::string k_DACName_Vsf = "Vsf";
  const std::string k_DACName_Vcomp = "Vcomp";
  const std::string k_DACName_Vleak = "Vleak";
  const std::string k_DACName_VrgPr = "VrgPr";
  const std::string k_DACName_VwllPr = "VwllPr";
  const std::string k_DACName_VrgSh = "VrgSh";
  const std::string k_DACName_VwllSh = "VwllSh";
  const std::string k_DACName_VHldDel = "VHldDel";
  const std::string k_DACName_Vtrim = "Vtrim";
  const std::string k_DACName_VcThr = "VcThr";
  const std::string k_DACName_VIbias_bus = "VIbias_bus";
  const std::string k_DACName_VIbias_sf = "VIbias_sf";
  const std::string k_DACName_VOffsetOp = "VOffsetOp";
  const std::string k_DACName_VbiasOp = "VbiasOp";
  const std::string k_DACName_VOffsetRO = "VOffsetRO";
  const std::string k_DACName_VIon = "VIon";
  const std::string k_DACName_VIbias_PH = "VIbias_PH";
  const std::string k_DACName_VIbias_DAC = "VIbias_DAC";
  const std::string k_DACName_VIbias_roc = "VIbias_roc";
  const std::string k_DACName_VIColOr = "VIColOr";
  const std::string k_DACName_Vnpix = "Vnpix";
  const std::string k_DACName_VsumCol = "VsumCol";
  const std::string k_DACName_Vcal = "Vcal";
  const std::string k_DACName_CalDel = "CalDel";
  const std::string k_DACName_TempRange = "TempRange";
  const std::string k_DACName_WBC = "WBC";
  const std::string k_DACName_ChipContReg = "ChipContReg";

  const unsigned int k_DACAddress_Vdd = 1;
  const unsigned int k_DACAddress_Vana = 2;
  const unsigned int k_DACAddress_Vsf = 3;
  const unsigned int k_DACAddress_Vcomp = 4;
  const unsigned int k_DACAddress_Vleak = 5;
  const unsigned int k_DACAddress_VrgPr = 6;
  const unsigned int k_DACAddress_VwllPr = 7;
  const unsigned int k_DACAddress_VrgSh = 8;
  const unsigned int k_DACAddress_VwllSh = 9;
  const unsigned int k_DACAddress_VHldDel = 10;
  const unsigned int k_DACAddress_Vtrim = 11;
  const unsigned int k_DACAddress_VcThr = 12;
  const unsigned int k_DACAddress_VIbias_bus = 13;
  const unsigned int k_DACAddress_VIbias_sf = 14;
  const unsigned int k_DACAddress_VOffsetOp = 15;
  const unsigned int k_DACAddress_VbiasOp = 16;
  const unsigned int k_DACAddress_VOffsetRO = 17;
  const unsigned int k_DACAddress_VIon = 18;
  const unsigned int k_DACAddress_VIbias_PH = 19;
  const unsigned int k_DACAddress_VIbias_DAC = 20;
  const unsigned int k_DACAddress_VIbias_roc = 21;
  const unsigned int k_DACAddress_VIColOr = 22;
  const unsigned int k_DACAddress_Vnpix = 23;
  const unsigned int k_DACAddress_VsumCol = 24;
  const unsigned int k_DACAddress_Vcal = 25;
  const unsigned int k_DACAddress_CalDel = 26;
  const unsigned int k_DACAddress_TempRange = 27;
  const unsigned int k_DACAddress_WBC = 254;
  const unsigned int k_DACAddress_ChipContReg = 253;
}  // namespace pos

#endif
