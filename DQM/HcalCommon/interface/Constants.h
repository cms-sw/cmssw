#ifndef Constants_h
#define Constants_h

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

#include <vector>

namespace hcaldqm {
  namespace constants {
    /*
 *	Detector Subsystem Status States
 */
    double const GOOD = 1.0;
    double const PROBLEMATIC = 0.9;
    double const BAD = 0.5;
    double const VERY_LOW = 0.5;
    double const VERY_LOW_XXX = 0;
    double const NOT_APPLICABLE = -1;
    double const DELTA = 0.005;

    /*
 *	Electronics Constants
 */
    //	FED2Crate array and CRATE2FED array
    //	use conversion functions in Utilities.h
    //	For fast look up
    //	This is for uTCA Crates/FEDs only - no other way...
    std::map<unsigned int, unsigned int> const crate2fed_map = {
        {24, 1100},
        {20, 1102},
        {21, 1104},
        {25, 1106},
        {31, 1108},
        {35, 1110},
        {37, 1112},
        {34, 1114},
        {30, 1116},
        {22, 1118},
        {29, 1120},
        {32, 1122},
        //{38, 1134},
        {3, 724},
        {7, 726},
        {6, 728},
        {13, 730},

        // New HO crates as of 2020
        {23, 1124},
        {27, 1126},
        {26, 1128},
        {38, 1130},
    };

    std::map<unsigned int, unsigned int> const fed2crate_map = {
        {724, 3},   {725, 3},   {726, 7},   {727, 7},   {728, 6},   {729, 6},   {730, 13},  {731, 13},  {1100, 24},
        {1101, 24}, {1102, 20}, {1103, 20}, {1104, 21}, {1105, 21}, {1106, 25}, {1107, 25}, {1108, 31}, {1109, 31},
        {1110, 35}, {1111, 35}, {1112, 37}, {1113, 37}, {1114, 34}, {1115, 34}, {1116, 30}, {1117, 30}, {1118, 22},
        {1119, 22}, {1120, 29}, {1121, 29}, {1122, 32}, {1123, 32}, {1124, 23}, {1125, 23}, {1126, 27}, {1127, 27},
        {1128, 26}, {1129, 26}, {1130, 38}, {1131, 38}, {1134, 38}, {1135, 38},
    };

    std::vector<unsigned int> const fedList = {724,  725,  726,  727,  728,  729,  730,  731,  1100, 1101, 1102,
                                               1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113,
                                               1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
                                               1125, 1126, 1127, 1128, 1129, 1130, 1131, 1134, 1135};
    std::vector<unsigned int> const fedListuTCA = {
        1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116,
        1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1134, 1135};
    std::vector<unsigned int> const fedListVME = {724, 725, 726, 727, 728, 729, 730, 731};
    std::vector<unsigned int> const crateList = {3,  6,  7,  13, 20, 21, 22, 23, 24, 25,
                                                 26, 27, 29, 30, 31, 32, 34, 35, 37, 38};
    std::vector<unsigned int> const crateListuTCA = {20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 37, 38};
    std::vector<unsigned int> const crateListVME = {3, 6, 7, 13};
    std::vector<unsigned int> const crateListHF = {22, 29, 32};
    std::vector<unsigned int> const crateListHO = {23, 26, 27, 38};

    int const FED_uTCA_MAX_REAL = 50;

    //	FEDs use the first 50 uTCA FED numbers only everywhere
    int const FED_VME_MIN = FEDNumbering::MINHCALFEDID;
    int const FED_VME_MAX = FEDNumbering::MAXHCALFEDID;
    int const FED_VME_DELTA = 1;
    int const FED_VME_NUM = FED_VME_MAX - FED_VME_MIN + 1;

    int const FED_uTCA_MIN = FEDNumbering::MINHCALuTCAFEDID;
    //		int const FED_uTCA_MAX = FEDNumbering::MAXHCALuTCAFEDID;
    int const FED_uTCA_MAX = FED_uTCA_MIN + FED_uTCA_MAX_REAL - 1;
    int const FED_uTCA_NUM = FED_uTCA_MAX - FED_uTCA_MIN + 1;
    int const FED_uTCA_DELTA = 1;
    int const FED_TOTAL_NUM = FED_VME_NUM + FED_uTCA_NUM;

    //	Crates
    int const CRATE_VME_MIN = 0;
    int const CRATE_VME_MAX = 18;
    int const CRATE_VME_DELTA = 1;
    int const CRATE_VME_NUM = CRATE_VME_MAX - CRATE_VME_MIN + 1;

    int const CRATE_uTCA_MIN = 20;
    int const CRATE_uTCA_MAX = 37;
    int const CRATE_uTCA_DELTA = 1;
    int const CRATE_uTCA_NUM = CRATE_uTCA_MAX - CRATE_uTCA_MIN + 1;
    int const CRATE_TOTAL_NUM = CRATE_VME_NUM + CRATE_uTCA_NUM;

    //	Slots
    int const SLOT_uTCA_MIN = 1;
    int const SLOT_uTCA_MAX = 12;
    int const SLOT_uTCA_DELTA = 1;
    int const SLOT_uTCA_NUM = SLOT_uTCA_MAX - SLOT_uTCA_MIN + 1;

    int const SLOT_VME_MIN1 = 2;
    int const SLOT_VME_MAX1 = 7;
    int const SLOT_VME_MIN2 = 13;
    int const SLOT_VME_MAX2 = 18;
    int const SLOT_VME_NUM1 = SLOT_VME_MAX1 - SLOT_VME_MIN1 + 1;
    int const SLOT_VME_NUM2 = SLOT_VME_MAX2 - SLOT_VME_MIN2 + 1;
    int const SLOT_VME_NUM = SLOT_VME_NUM1 + SLOT_VME_NUM2;

    int const SPIGOT_MIN = 0;
    int const SPIGOT_MAX = 11;
    int const SPIGOT_NUM = SPIGOT_MAX - SPIGOT_MIN + 1;

    //	Fibers
    int const FIBER_VME_MIN = 1;
    int const FIBER_VME_MAX = 8;
    int const FIBER_VME_NUM = FIBER_VME_MAX - FIBER_VME_MIN + 1;
    int const FIBER_uTCA_MIN1 = 0;
    int const FIBER_uTCA_MAX1 = 11;
    int const FIBER_uTCA_MIN2 = 12;
    int const FIBER_uTCA_MAX2 = 23;
    int const FIBER_uTCA_NUM = FIBER_uTCA_MAX1 - FIBER_uTCA_MIN1 + 1 + FIBER_uTCA_MAX2 - FIBER_uTCA_MIN2 + 1;

    int const FIBERCH_MIN = 0;
    int const FIBERCH_MAX = 5;
    int const FIBERCH_NUM = FIBERCH_MAX - FIBERCH_MIN + 1;

    //	TP SLBs, Fibers
    int const SLB_MIN = 1;
    int const SLB_MAX = 6;
    int const SLB_NUM = SLB_MAX - SLB_MIN + 1;

    int const TPFIBER_MIN = 0;
    int const TPFIBER_MAX = 5;
    int const TPFIBER_NUM = TPFIBER_MAX - TPFIBER_MIN + 1;

    int const SLBCH_MIN = 0;
    int const SLBCH_MAX = 3;
    int const SLBCH_NUM = SLBCH_MAX - SLBCH_MIN + 1;

    int const TPFIBERCH_MIN = 0;
    int const TPFIBERCH_MAX = 7;
    int const TPFIBERCH_NUM = TPFIBERCH_MAX - TPFIBERCH_MIN + 1;

    /*
 *	Detector Constants
 */

    //	Hcal Subdetector
    int const HB = 1;
    int const HE = 2;
    int const HO = 3;
    int const HF = 4;
    int const SUBDET_NUM = 4;
    int const TPSUBDET_NUM = 2;
    std::string const SUBDET_NAME[SUBDET_NUM] = {"HB", "HE", "HO", "HF"};
    std::string const SUBDETPM_NAME[2 * SUBDET_NUM] = {"HBM", "HBP", "HEM", "HEP", "HOM", "HOP", "HFM", "HFP"};
    std::string const SUBSYSTEM = "Hcal";
    std::string const TPSUBDET_NAME[TPSUBDET_NUM] = {"HBHE", "HF"};
    std::string const TPSUBDETPM_NAME[2 * TPSUBDET_NUM] = {"HBHEM", "HBHEP", "HFM", "HFP"};
    std::string const TPSUBDETFW_NAME[2 * TPSUBDET_NUM] = {"Pure HB", "Overlap HBHE", "Pure HE", "HF"};

    //	iphis
    int const IPHI_MIN = 1;
    int const IPHI_MAX = 72;
    int const IPHI_NUM = 72;
    int const IPHI_NUM_HF = 36;
    int const IPHI_NUM_TPHF = 18;
    int const IPHI_DELTA = 1;
    int const IPHI_DELTA_HF = 2;
    int const IPHI_DELTA_TPHF = 4;

    //	ietas
    int const IETA_MIN = 1;
    int const IETA_DELTA = 1;
    int const IETA_MAX = 41;
    int const IETA_NUM = 2 * (IETA_MAX - IETA_MIN + 1) + 1;
    int const IETA_MIN_HB = 1;
    int const IETA_MAX_HB = 16;
    int const IETA_MIN_HE = 16;
    int const IETA_MAX_HE = 29;
    int const IETA_MIN_HO = 1;
    int const IETA_MAX_HO = 15;
    int const IETA_MIN_HF = 29;
    int const IETA_MAX_HF = 41;

    int const IETA_MAX_TPHBHE = 28;
    int const IETA_MAX_TPHF = 32;

    //	Depth
    int const DEPTH_MIN = 1;
    int const DEPTH_DELTA = 1;
    int const DEPTH_MAX = 4;
    int const DEPTH_NUM = 4;

    //	Caps
    int const CAPS_NUM = 4;

    //	Number of FG Bits
    int const NUM_FGBITS = 6;

    /*
 *	Number of Channels Constants
 */
    int const CHS_NUM[SUBDET_NUM] = {2592, 2592, 2192, 1728};  // HO ??!
    int const TPCHS_NUM[TPSUBDET_NUM] = {2 * 28 * 72, 144};

    /*
 *	Number of Time Samples
 */
    int const TS_NUM[SUBDET_NUM] = {10, 10, 10, 4};

    /*
 *	Value Constants
 */
    double const AXIS_ENERGY_MIN = -10.;
    double const AXIS_ENERGY_MAX = 200;
    int const AXIS_ENERGY_NBINS = 400;
    double const AXIS_TIME_MIN = -50.;
    double const AXIS_TIME_MAX = 50;
    int const AXIS_TIME_NBINS = 200;
    int const AXIS_ADC_NBINS_PED = 100;
    double const AXIS_ADC_MAX_PED = 5;
    int const AXIS_ADC_NBINS = 128;
    double const AXIS_ADC_MIN = 0;
    double const AXIS_ADC_MAX = 128;
    int const AXIS_NOMFC_NBINS_3000 = 300;
    double const AXIS_NOMFC_MAX_3000 = 3000.;
    int const AXIS_NOMFC_NBINS = 300;
    double const AXIS_NOMFC_MIN = 0;
    double const AXIS_NOMFC_MAX = 3000.;
    int const AXIS_TIMETS_NBINS = 10;
    double const AXIS_TIMETS_MIN = 0;
    double const AXIS_TIMETS_MAX = 10;

    int const CALIBEVENTS_MIN = 100;
    int const GARBAGE_VALUE = -1000;
    int const FIBEROFFSET_INVALID = -1000;

    int const RAW_EMPTY = 16;
    int const UTCA_DATAFLAVOR = 0x5;

    double const adc2fC[256] = {
        1.58,   4.73,   7.88,   11.0,   14.2,   17.3,   20.5,   23.6,   26.8,   29.9,   33.1,   36.2,   39.4,
        42.5,   45.7,   48.8,   53.6,   60.1,   66.6,   73.0,   79.5,   86.0,   92.5,   98.9,   105,    112,
        118,    125,    131,    138,    144,    151,    157,    164,    170,    177,    186,    199,    212,
        225,    238,    251,    264,    277,    289,    302,    315,    328,    341,    354,    367,    380,
        393,    406,    418,    431,    444,    464,    490,    516,    542,    568,    594,    620,    569,
        594,    619,    645,    670,    695,    720,    745,    771,    796,    821,    846,    871,    897,
        922,    947,    960,    1010,   1060,   1120,   1170,   1220,   1270,   1320,   1370,   1430,   1480,
        1530,   1580,   1630,   1690,   1740,   1790,   1840,   1890,   1940,   2020,   2120,   2230,   2330,
        2430,   2540,   2640,   2740,   2850,   2950,   3050,   3150,   3260,   3360,   3460,   3570,   3670,
        3770,   3880,   3980,   4080,   4240,   4450,   4650,   4860,   5070,   5280,   5490,

        5080,   5280,   5480,   5680,   5880,   6080,   6280,   6480,   6680,   6890,   7090,   7290,   7490,
        7690,   7890,   8090,   8400,   8810,   9220,   9630,   10000,  10400,  10900,  11300,  11700,  12100,
        12500,  12900,  13300,  13700,  14100,  14500,  15000,  15400,  15800,  16200,  16800,  17600,  18400,
        19300,  20100,  20900,  21700,  22500,  23400,  24200,  25000,  25800,  26600,  27500,  28300,  29100,
        29900,  30700,  31600,  32400,  33200,  34400,  36100,  37700,  39400,  41000,  42700,  44300,  41100,
        42700,  44300,  45900,  47600,  49200,  50800,  52500,  54100,  55700,  57400,  59000,  60600,  62200,
        63900,  65500,  68000,  71300,  74700,  78000,  81400,  84700,  88000,  91400,  94700,  98100,  101000,
        105000, 108000, 111000, 115000, 118000, 121000, 125000, 128000, 131000, 137000, 145000, 152000, 160000,
        168000, 176000, 183000, 191000, 199000, 206000, 214000, 222000, 230000, 237000, 245000, 253000, 261000,
        268000, 276000, 284000, 291000, 302000, 316000, 329000, 343000, 356000, 370000, 384000};

    /*
 *	TObject Related. The first 3 bits are set by the Axis Class
 *	0 - log X axis
 *	1 - log Y axis
 *	2 - log Z axis
 */
    int const BIT_OFFSET = 19;
    int const BIT_AXIS_XLOG = 0;
    int const BIT_AXIS_YLOG = 1;
    int const BIT_AXIS_ZLOG = 2;
    int const BIT_AXIS_LS = 3;
    int const BIT_AXIS_FLAG = 4;

    /*
 *	Orbit Gap Operations enum
 */
    uint8_t const EVENTTYPE_PEDESTAL = 1;
    uint8_t const EVENTTYPE_LASER = 14;
    uint8_t const EVENTTYPE_LED = 15;
    enum OrbitGapType {
      tNull = 0,
      tPedestal = 1,
      tLED = 2,
      tHFRaddam = 3,
      tHBHEHPD = 4,
      tHO = 5,
      tHF = 6,
      tZDC = 7,
      tHEPMega = 8,
      tHEMMega = 9,
      tHBPMega = 10,
      tHBMMega = 11,
      tSomething = 12,
      tCRF = 13,
      tCalib = 14,
      tSafe = 15,
      tUnknown = 16,
      nOrbitGapType = 17
    };
  }  // namespace constants
}  // namespace hcaldqm

#endif
