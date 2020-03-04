// -*- C++ -*-
//
// Package:    CMTRawAnalyzer
#ifndef CMTRawAnalyzer_h
#define CMTRawAnalyzer_h
#include <fstream>
#include <iostream>
#include <cmath>
#include <iosfwd>
#include <bitset>
#include <memory>
using namespace std;
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"

#define NUMADCS 256
// very preliminary,  NEEDS UPDATING
double adc2fC_QIE10[NUMADCS] = {
    // - - - - - - - range 0 - - - - - - - -
    //subrange0
    1.58,
    4.73,
    7.88,
    11.0,
    14.2,
    17.3,
    20.5,
    23.6,
    26.8,
    29.9,
    33.1,
    36.2,
    39.4,
    42.5,
    45.7,
    48.8,
    //subrange1
    53.6,
    60.1,
    66.6,
    73.0,
    79.5,
    86.0,
    92.5,
    98.9,
    105,
    112,
    118,
    125,
    131,
    138,
    144,
    151,
    //subrange2
    157,
    164,
    170,
    177,
    186,
    199,
    212,
    225,
    238,
    251,
    264,
    277,
    289,
    302,
    315,
    328,
    //subrange3
    341,
    354,
    367,
    380,
    393,
    406,
    418,
    431,
    444,
    464,
    490,
    516,
    542,
    568,
    594,
    620,

    // - - - - - - - range 1 - - - - - - - -
    //subrange0
    569,
    594,
    619,
    645,
    670,
    695,
    720,
    745,
    771,
    796,
    821,
    846,
    871,
    897,
    922,
    947,
    //subrange1
    960,
    1010,
    1060,
    1120,
    1170,
    1220,
    1270,
    1320,
    1370,
    1430,
    1480,
    1530,
    1580,
    1630,
    1690,
    1740,
    //subrange2
    1790,
    1840,
    1890,
    1940,
    2020,
    2120,
    2230,
    2330,
    2430,
    2540,
    2640,
    2740,
    2850,
    2950,
    3050,
    3150,
    //subrange3
    3260,
    3360,
    3460,
    3570,
    3670,
    3770,
    3880,
    3980,
    4080,
    4240,
    4450,
    4650,
    4860,
    5070,
    5280,
    5490,

    // - - - - - - - range 2 - - - - - - - -
    //subrange0
    5080,
    5280,
    5480,
    5680,
    5880,
    6080,
    6280,
    6480,
    6680,
    6890,
    7090,
    7290,
    7490,
    7690,
    7890,
    8090,
    //subrange1
    8400,
    8810,
    9220,
    9630,
    10000,
    10400,
    10900,
    11300,
    11700,
    12100,
    12500,
    12900,
    13300,
    13700,
    14100,
    14500,
    //subrange2
    15000,
    15400,
    15800,
    16200,
    16800,
    17600,
    18400,
    19300,
    20100,
    20900,
    21700,
    22500,
    23400,
    24200,
    25000,
    25800,
    //subrange3
    26600,
    27500,
    28300,
    29100,
    29900,
    30700,
    31600,
    32400,
    33200,
    34400,
    36100,
    37700,
    39400,
    41000,
    42700,
    44300,

    // - - - - - - - range 3 - - - - - - - - -
    //subrange0
    41100,
    42700,
    44300,
    45900,
    47600,
    49200,
    50800,
    52500,
    54100,
    55700,
    57400,
    59000,
    60600,
    62200,
    63900,
    65500,
    //subrange1
    68000,
    71300,
    74700,
    78000,
    81400,
    84700,
    88000,
    91400,
    94700,
    98100,
    101000,
    105000,
    108000,
    111000,
    115000,
    118000,
    //subrange2
    121000,
    125000,
    128000,
    131000,
    137000,
    145000,
    152000,
    160000,
    168000,
    176000,
    183000,
    191000,
    199000,
    206000,
    214000,
    222000,
    //subrange3
    230000,
    237000,
    245000,
    253000,
    261000,
    268000,
    276000,
    284000,
    291000,
    302000,
    316000,
    329000,
    343000,
    356000,
    370000,
    384000

};

//shunt1
double const adc2fC_QIE11_shunt1[NUMADCS] = {
    1.89,     5.07,     8.25,     11.43,    14.61,    17.78,    20.96,    24.14,    27.32,    30.50,    33.68,
    36.86,    40.04,    43.22,    46.40,    49.58,    54.35,    60.71,    67.07,    73.43,    79.79,    86.15,
    92.51,    98.87,    105.2,    111.6,    117.9,    124.3,    130.7,    137.0,    143.4,    149.7,    156.1,
    162.5,    168.8,    175.2,    184.7,    197.4,    210.2,    222.9,    235.6,    248.3,    261.0,    273.7,
    286.5,    299.2,    311.9,    324.6,    337.3,    350.1,    362.8,    375.5,    388.2,    400.9,    413.6,
    426.4,    439.1,    458.2,    483.6,    509.0,    534.5,    559.9,    585.3,    610.8,    558.9,    584.2,
    609.5,    634.7,    660.0,    685.3,    710.6,    735.9,    761.2,    786.5,    811.8,    837.1,    862.4,
    887.7,    913.0,    938.3,    976.2,    1026.8,   1077.4,   1128.0,   1178.6,   1229.1,   1279.7,   1330.3,
    1380.9,   1431.5,   1482.1,   1532.7,   1583.3,   1633.8,   1684.4,   1735.0,   1785.6,   1836.2,   1886.8,
    1937.4,   2013.2,   2114.4,   2215.6,   2316.8,   2417.9,   2519.1,   2620.3,   2721.5,   2822.6,   2923.8,
    3025.0,   3126.2,   3227.3,   3328.5,   3429.7,   3530.9,   3632.0,   3733.2,   3834.4,   3935.5,   4036.7,
    4188.5,   4390.8,   4593.2,   4795.5,   4997.9,   5200.2,   5402.6,   5057.5,   5262.3,   5467.1,   5671.8,
    5876.6,   6081.4,   6286.2,   6491.0,   6695.8,   6900.6,   7105.3,   7310.1,   7514.9,   7719.7,   7924.5,
    8129.3,   8436.4,   8846.0,   9255.6,   9665.1,   10074.7,  10484.3,  10893.9,  11303.4,  11713.0,  12122.6,
    12532.1,  12941.7,  13351.3,  13760.8,  14170.4,  14580.0,  14989.5,  15399.1,  15808.7,  16218.2,  16832.6,
    17651.7,  18470.9,  19290.0,  20109.2,  20928.3,  21747.4,  22566.6,  23385.7,  24204.8,  25024.0,  25843.1,
    26662.3,  27481.4,  28300.5,  29119.7,  29938.8,  30757.9,  31577.1,  32396.2,  33215.4,  34444.1,  36082.3,
    37720.6,  39358.9,  40997.2,  42635.4,  44273.7,  40908.7,  42542.6,  44176.5,  45810.4,  47444.3,  49078.3,
    50712.2,  52346.1,  53980.0,  55613.9,  57247.8,  58881.8,  60515.7,  62149.6,  63783.5,  65417.4,  67868.3,
    71136.1,  74404.0,  77671.8,  80939.7,  84207.5,  87475.3,  90743.2,  94011.0,  97278.8,  100546.7, 103814.5,
    107082.3, 110350.2, 113618.0, 116885.8, 120153.7, 123421.5, 126689.3, 129957.2, 134858.9, 141394.6, 147930.3,
    154465.9, 161001.6, 167537.3, 174072.9, 180608.6, 187144.3, 193679.9, 200215.6, 206751.3, 213287.0, 219822.6,
    226358.3, 232894.0, 239429.6, 245965.3, 252501.0, 259036.6, 265572.3, 275375.8, 288447.2, 301518.5, 314589.8,
    327661.2, 340732.5, 353803.8};
//shunt6
double const adc2fC_QIE11_shunt6[NUMADCS] = {
    9.56,      28.24,     46.91,     65.59,     84.27,     102.9,     121.6,     140.3,     159.0,     177.7,
    196.3,     215.0,     233.7,     252.4,     271.0,     289.7,     317.7,     355.1,     392.4,     429.8,
    467.1,     504.5,     541.9,     579.2,     616.6,     653.9,     691.3,     728.6,     766.0,     803.3,
    840.7,     878.0,     915.4,     952.8,     990.1,     1027.5,    1083.5,    1158.2,    1232.9,    1307.6,
    1382.3,    1457.0,    1531.7,    1606.4,    1681.2,    1755.9,    1830.6,    1905.3,    1980.0,    2054.7,
    2129.4,    2204.1,    2278.8,    2353.5,    2428.2,    2502.9,    2577.7,    2689.7,    2839.1,    2988.6,
    3138.0,    3287.4,    3436.8,    3586.2,    3263.0,    3411.3,    3559.6,    3707.9,    3856.2,    4004.5,
    4152.9,    4301.2,    4449.5,    4597.8,    4746.1,    4894.4,    5042.7,    5191.0,    5339.4,    5487.7,
    5710.1,    6006.8,    6303.4,    6600.0,    6896.6,    7193.3,    7489.9,    7786.5,    8083.1,    8379.8,
    8676.4,    8973.0,    9269.6,    9566.3,    9862.9,    10159.5,   10456.2,   10752.8,   11049.4,   11346.0,
    11791.0,   12384.2,   12977.5,   13570.7,   14164.0,   14757.2,   15350.5,   15943.7,   16537.0,   17130.2,
    17723.5,   18316.7,   18910.0,   19503.2,   20096.5,   20689.7,   21283.0,   21876.2,   22469.5,   23062.8,
    23656.0,   24545.9,   25732.4,   26918.9,   28105.4,   29291.9,   30478.4,   31664.9,   29399.4,   30590.1,
    31780.9,   32971.7,   34162.4,   35353.2,   36544.0,   37734.7,   38925.5,   40116.3,   41307.0,   42497.8,
    43688.5,   44879.3,   46070.1,   47260.8,   49047.0,   51428.5,   53810.1,   56191.6,   58573.1,   60954.6,
    63336.2,   65717.7,   68099.2,   70480.8,   72862.3,   75243.8,   77625.4,   80006.9,   82388.4,   84769.9,
    87151.5,   89533.0,   91914.5,   94296.1,   97868.4,   102631.4,  107394.5,  112157.5,  116920.6,  121683.7,
    126446.7,  131209.8,  135972.8,  140735.9,  145499.0,  150262.0,  155025.1,  159788.2,  164551.2,  169314.3,
    174077.3,  178840.4,  183603.5,  188366.5,  193129.6,  200274.2,  209800.3,  219326.4,  228852.5,  238378.7,
    247904.8,  257430.9,  237822.7,  247326.7,  256830.7,  266334.8,  275838.8,  285342.9,  294846.9,  304351.0,
    313855.0,  323359.1,  332863.1,  342367.1,  351871.2,  361375.2,  370879.3,  380383.3,  394639.4,  413647.5,
    432655.6,  451663.6,  470671.7,  489679.8,  508687.9,  527696.0,  546704.1,  565712.2,  584720.3,  603728.3,
    622736.4,  641744.5,  660752.6,  679760.7,  698768.8,  717776.9,  736785.0,  755793.0,  784305.2,  822321.3,
    860337.5,  898353.7,  936369.9,  974386.0,  1012402.2, 1050418.4, 1088434.6, 1126450.7, 1164466.9, 1202483.1,
    1240499.3, 1278515.4, 1316531.6, 1354547.8, 1392564.0, 1430580.1, 1468596.3, 1506612.5, 1544628.7, 1601652.9,
    1677685.3, 1753717.6, 1829750.0, 1905782.3, 1981814.7, 2057847.0};
// for HPD:
static const float adc2fC[128] = {
    -0.5,   0.5,    1.5,    2.5,    3.5,    4.5,    5.5,    6.5,    7.5,    8.5,    9.5,    10.5,   11.5,
    12.5,   13.5,   15.,    17.,    19.,    21.,    23.,    25.,    27.,    29.5,   32.5,   35.5,   38.5,
    42.,    46.,    50.,    54.5,   59.5,   64.5,   59.5,   64.5,   69.5,   74.5,   79.5,   84.5,   89.5,
    94.5,   99.5,   104.5,  109.5,  114.5,  119.5,  124.5,  129.5,  137.,   147.,   157.,   167.,   177.,
    187.,   197.,   209.5,  224.5,  239.5,  254.5,  272.,   292.,   312.,   334.5,  359.5,  384.5,  359.5,
    384.5,  409.5,  434.5,  459.5,  484.5,  509.5,  534.5,  559.5,  584.5,  609.5,  634.5,  659.5,  684.5,
    709.5,  747.,   797.,   847.,   897.,   947.,   997.,   1047.,  1109.5, 1184.5, 1259.5, 1334.5, 1422.,
    1522.,  1622.,  1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5, 2234.5, 2359.5, 2484.5, 2609.5, 2734.5,
    2859.5, 2984.5, 3109.5, 3234.5, 3359.5, 3484.5, 3609.5, 3797.,  4047.,  4297.,  4547.,  4797.,  5047.,
    5297.,  5609.5, 5984.5, 6359.5, 6734.5, 7172.,  7672.,  8172.,  8734.5, 9359.5, 9984.5};
const int nsub = 4;
const int ndepth = 7;
const int neta = 82;
const int nphi = 72;
float bphi = 72.;
const int zneta = 22;
const int znphi = 18;
float zbphi = 18.;
const int npfit = 220;
float anpfit = 220.;  // for SiPM:

class CMTRawAnalyzer : public edm::EDAnalyzer {
public:
  explicit CMTRawAnalyzer(const edm::ParameterSet&);
  ~CMTRawAnalyzer() override;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void beginRun(const edm::Run& r, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& r, const edm::EventSetup& iSetup) override;
  virtual void fillMAP();

private:
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_qie10_;
  ////////////////////////////////////
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1, double phi2);
  ////////////////////////////////////
  std::string fOutputFileName;
  std::string MAPOutputFileName;
  edm::InputTag inputTag_;
  edm::ESHandle<CaloGeometry> geometry;
  edm::ESHandle<HcalDbService> conditions;
  const HcalQIEShape* shape;
  edm::ESHandle<HcalTopology> topo_;
  const HcalTopology* topo;
  /////////////////////////////////////////////
  int verbosity;
  int MAPcreation;
  /////////////////////////////////////////////
  bool recordNtuples_;
  int maxNeventsInNtuple_;
  bool recordHistoes_;
  bool studyRunDependenceHist_;
  bool studyCapIDErrorsHist_;
  bool studyRMSshapeHist_;
  bool studyRatioShapeHist_;
  bool studyTSmaxShapeHist_;
  bool studyTSmeanShapeHist_;
  bool studyDiffAmplHist_;
  bool studyCalibCellsHist_;
  bool studyADCAmplHist_;
  bool studyPedestalsHist_;
  bool studyPedestalCorrelations_;
  bool usePedestalSubtraction_;
  bool useADCmassive_;
  bool useADCfC_;
  bool useADCcounts_;
  bool usecontinuousnumbering_;
  /////////////////////////////////////////////
  double ratioHBMin_;
  double ratioHBMax_;
  double ratioHEMin_;
  double ratioHEMax_;
  double ratioHFMin_;
  double ratioHFMax_;
  double ratioHOMin_;
  double ratioHOMax_;
  /////////////////////////////////////////////
  int flagfitshunt1pedorledlowintensity_;
  int flagLaserRaddam_;
  int flagIterativeMethodCalibrationGroup_;
  int flagtoaskrunsorls_;
  int flagtodefinebadchannel_;
  int howmanybinsonplots_;
  int splashesUpperLimit_;
  int flagabortgaprejected_;
  int bcnrejectedlow_;
  int bcnrejectedhigh_;
  int flagestimatornormalization_;
  int flagcpuoptimization_;
  int flagupgradeqie1011_;
  int flagsipmcorrection_;
  int flaguseshunt_;
  int lsdep_cut1_peak_HBdepth1_;
  int lsdep_cut1_peak_HBdepth2_;
  int lsdep_cut1_peak_HEdepth1_;
  int lsdep_cut1_peak_HEdepth2_;
  int lsdep_cut1_peak_HEdepth3_;
  int lsdep_cut1_peak_HFdepth1_;
  int lsdep_cut1_peak_HFdepth2_;
  int lsdep_cut1_peak_HOdepth4_;
  //
  int lsdep_cut3_max_HBdepth1_;
  int lsdep_cut3_max_HBdepth2_;
  int lsdep_cut3_max_HEdepth1_;
  int lsdep_cut3_max_HEdepth2_;
  int lsdep_cut3_max_HEdepth3_;
  int lsdep_cut3_max_HFdepth1_;
  int lsdep_cut3_max_HFdepth2_;
  int lsdep_cut3_max_HOdepth4_;
  double lsdep_estimator1_HBdepth1_;
  double lsdep_estimator1_HBdepth2_;
  double lsdep_estimator1_HEdepth1_;
  double lsdep_estimator1_HEdepth2_;
  double lsdep_estimator1_HEdepth3_;
  double lsdep_estimator1_HFdepth1_;
  double lsdep_estimator1_HFdepth2_;
  double lsdep_estimator1_HOdepth4_;
  // HE upgrade:
  double lsdep_estimator1_HEdepth4_;
  double lsdep_estimator1_HEdepth5_;
  double lsdep_estimator1_HEdepth6_;
  double lsdep_estimator1_HEdepth7_;
  // HF upgrade:
  double lsdep_estimator1_HFdepth3_;
  double lsdep_estimator1_HFdepth4_;
  // HB upgrade:
  double lsdep_estimator1_HBdepth3_;
  double lsdep_estimator1_HBdepth4_;

  double lsdep_estimator2_HBdepth1_;
  double lsdep_estimator2_HBdepth2_;
  double lsdep_estimator2_HEdepth1_;
  double lsdep_estimator2_HEdepth2_;
  double lsdep_estimator2_HEdepth3_;
  double lsdep_estimator2_HFdepth1_;
  double lsdep_estimator2_HFdepth2_;
  double lsdep_estimator2_HOdepth4_;

  double lsdep_estimator3_HBdepth1_;
  double lsdep_estimator3_HBdepth2_;
  double lsdep_estimator3_HEdepth1_;
  double lsdep_estimator3_HEdepth2_;
  double lsdep_estimator3_HEdepth3_;
  double lsdep_estimator3_HFdepth1_;
  double lsdep_estimator3_HFdepth2_;
  double lsdep_estimator3_HOdepth4_;

  double lsdep_estimator4_HBdepth1_;
  double lsdep_estimator4_HBdepth2_;
  double lsdep_estimator4_HEdepth1_;
  double lsdep_estimator4_HEdepth2_;
  double lsdep_estimator4_HEdepth3_;
  double lsdep_estimator4_HFdepth1_;
  double lsdep_estimator4_HFdepth2_;
  double lsdep_estimator4_HOdepth4_;

  double lsdep_estimator5_HBdepth1_;
  double lsdep_estimator5_HBdepth2_;
  double lsdep_estimator5_HEdepth1_;
  double lsdep_estimator5_HEdepth2_;
  double lsdep_estimator5_HEdepth3_;
  double lsdep_estimator5_HFdepth1_;
  double lsdep_estimator5_HFdepth2_;
  double lsdep_estimator5_HOdepth4_;
  double forallestimators_amplitude_bigger_;
  /////////////////////////////////////////////
  double rmsHBMin_;
  double rmsHBMax_;
  double rmsHEMin_;
  double rmsHEMax_;
  double rmsHFMin_;
  double rmsHFMax_;
  double rmsHOMin_;
  double rmsHOMax_;
  /////////////////////////////////////////////
  double TSpeakHBMin_;
  double TSpeakHBMax_;
  double TSpeakHEMin_;
  double TSpeakHEMax_;
  double TSpeakHFMin_;
  double TSpeakHFMax_;
  double TSpeakHOMin_;
  double TSpeakHOMax_;
  double TSmeanHBMin_;
  double TSmeanHBMax_;
  double TSmeanHEMin_;
  double TSmeanHEMax_;
  double TSmeanHFMin_;
  double TSmeanHFMax_;
  double TSmeanHOMin_;
  double TSmeanHOMax_;
  int lsmin_;
  int lsmax_;
  /////////////////////////////////////////////
  double ADCAmplHBMin_;
  double ADCAmplHEMin_;
  double ADCAmplHOMin_;
  double ADCAmplHFMin_;
  double ADCAmplHBMax_;
  double ADCAmplHEMax_;
  double ADCAmplHOMax_;
  double ADCAmplHFMax_;
  double calibrADCHBMin_;
  double calibrADCHEMin_;
  double calibrADCHOMin_;
  double calibrADCHFMin_;
  double calibrADCHBMax_;
  double calibrADCHEMax_;
  double calibrADCHOMax_;
  double calibrADCHFMax_;
  double calibrRatioHBMin_;
  double calibrRatioHEMin_;
  double calibrRatioHOMin_;
  double calibrRatioHFMin_;
  double calibrRatioHBMax_;
  double calibrRatioHEMax_;
  double calibrRatioHOMax_;
  double calibrRatioHFMax_;
  double calibrTSmaxHBMin_;
  double calibrTSmaxHEMin_;
  double calibrTSmaxHOMin_;
  double calibrTSmaxHFMin_;
  double calibrTSmaxHBMax_;
  double calibrTSmaxHEMax_;
  double calibrTSmaxHOMax_;
  double calibrTSmaxHFMax_;
  double calibrTSmeanHBMin_;
  double calibrTSmeanHEMin_;
  double calibrTSmeanHOMin_;
  double calibrTSmeanHFMin_;
  double calibrTSmeanHBMax_;
  double calibrTSmeanHEMax_;
  double calibrTSmeanHOMax_;
  double calibrTSmeanHFMax_;
  double calibrWidthHBMin_;
  double calibrWidthHEMin_;
  double calibrWidthHOMin_;
  double calibrWidthHFMin_;
  double calibrWidthHBMax_;
  double calibrWidthHEMax_;
  double calibrWidthHOMax_;
  double calibrWidthHFMax_;
  /////////////////////////////////////////////
  int nevent;
  int nevent50;
  int nnnnnn;
  int nnnnnnhbhe;
  int nnnnnnhbheqie11;
  int counterhf;
  int counterhfqie10;
  int counterho;
  int nnnnnn1;
  int nnnnnn2;
  int nnnnnn3;
  int nnnnnn4;
  int nnnnnn5;
  int nnnnnn6;
  double pedestalwHBMax_;
  double pedestalwHEMax_;
  double pedestalwHFMax_;
  double pedestalwHOMax_;
  double pedestalHBMax_;
  double pedestalHEMax_;
  double pedestalHFMax_;
  double pedestalHOMax_;
  /////////////////////////////////////////////
  TH1F* h_bcnvsamplitude_HB;
  TH1F* h_bcnvsamplitude0_HB;
  TH1F* h_bcnvsamplitude_HE;
  TH1F* h_bcnvsamplitude0_HE;
  TH1F* h_bcnvsamplitude_HF;
  TH1F* h_bcnvsamplitude0_HF;
  TH1F* h_bcnvsamplitude_HO;
  TH1F* h_bcnvsamplitude0_HO;
  TH1F* h_orbitNumvsamplitude_HB;
  TH1F* h_orbitNumvsamplitude0_HB;
  TH1F* h_orbitNumvsamplitude_HE;
  TH1F* h_orbitNumvsamplitude0_HE;
  TH1F* h_orbitNumvsamplitude_HF;
  TH1F* h_orbitNumvsamplitude0_HF;
  TH1F* h_orbitNumvsamplitude_HO;
  TH1F* h_orbitNumvsamplitude0_HO;
  TH2F* h_2DsumADCAmplEtaPhiLs0;
  TH2F* h_2DsumADCAmplEtaPhiLs1;
  TH2F* h_2DsumADCAmplEtaPhiLs2;
  TH2F* h_2DsumADCAmplEtaPhiLs3;
  TH2F* h_2DsumADCAmplEtaPhiLs00;
  TH2F* h_2DsumADCAmplEtaPhiLs10;
  TH2F* h_2DsumADCAmplEtaPhiLs20;
  TH2F* h_2DsumADCAmplEtaPhiLs30;
  TH1F* h_sumADCAmplEtaPhiLs;
  TH1F* h_sumADCAmplEtaPhiLs_bbbc;
  TH1F* h_sumADCAmplEtaPhiLs_bbb1;
  TH1F* h_sumADCAmplEtaPhiLs_lscounterM1;
  TH1F* h_sumADCAmplEtaPhiLs_ietaphi;
  TH1F* h_sumADCAmplEtaPhiLs_lscounterM1orbitNum;
  TH1F* h_sumADCAmplEtaPhiLs_orbitNum;
  TH2F* h_mapAiForLS_bad_HB;
  TH2F* h_map0AiForLS_bad_HB;
  TH2F* h_mapAiForLS_good_HB;
  TH2F* h_map0AiForLS_good_HB;
  TH2F* h_mapAiForLS_bad_HE;
  TH2F* h_map0AiForLS_bad_HE;
  TH2F* h_mapAiForLS_good_HE;
  TH2F* h_map0AiForLS_good_HE;
  TH2F* h_mapAiForLS_bad_HO;
  TH2F* h_map0AiForLS_bad_HO;
  TH2F* h_mapAiForLS_good_HO;
  TH2F* h_map0AiForLS_good_HO;
  TH2F* h_mapAiForLS_bad_HF;
  TH2F* h_map0AiForLS_bad_HF;
  TH2F* h_mapAiForLS_good_HF;
  TH2F* h_map0AiForLS_good_HF;
  TH1F* h_numberofhitsHBtest;
  TH1F* h_numberofhitsHEtest;
  TH1F* h_numberofhitsHFtest;
  TH1F* h_numberofhitsHOtest;
  TH1F* h_AmplitudeHBtest;
  TH1F* h_AmplitudeHBtest1;
  TH1F* h_AmplitudeHBtest6;
  TH1F* h_AmplitudeHEtest;
  TH1F* h_AmplitudeHEtest1;
  TH1F* h_AmplitudeHEtest6;
  TH1F* h_AmplitudeHFtest;
  TH1F* h_AmplitudeHOtest;
  TH1F* h_totalAmplitudeHB;
  TH1F* h_totalAmplitudeHE;
  TH1F* h_totalAmplitudeHF;
  TH1F* h_totalAmplitudeHO;
  TH1F* h_totalAmplitudeHBperEvent;
  TH1F* h_totalAmplitudeHEperEvent;
  TH1F* h_totalAmplitudeHFperEvent;
  TH1F* h_totalAmplitudeHOperEvent;
  TH1F* h_amplitudeaveragedbydepthes_HE;
  TH1F* h_ndepthesperamplitudebins_HE;
  TH1F* h_errorGeneral;
  TH1F* h_error1;
  TH1F* h_error2;
  TH1F* h_error3;
  TH1F* h_amplError;
  TH1F* h_amplFine;
  TH1F* h_errorGeneral_HB;
  TH1F* h_error1_HB;
  TH1F* h_error2_HB;
  TH1F* h_error3_HB;
  TH1F* h_error4_HB;
  TH1F* h_error5_HB;
  TH1F* h_error6_HB;
  TH1F* h_error7_HB;
  TH1F* h_amplError_HB;
  TH1F* h_amplFine_HB;
  TH2F* h_mapDepth1Error_HB;
  TH2F* h_mapDepth2Error_HB;
  TH2F* h_mapDepth3Error_HB;
  TH2F* h_mapDepth4Error_HB;
  TH1F* h_fiber0_HB;
  TH1F* h_fiber1_HB;
  TH1F* h_fiber2_HB;
  TH1F* h_repetedcapid_HB;
  TH1F* h_errorGeneral_HE;
  TH1F* h_error1_HE;
  TH1F* h_error2_HE;
  TH1F* h_error3_HE;
  TH1F* h_error4_HE;
  TH1F* h_error5_HE;
  TH1F* h_error6_HE;
  TH1F* h_error7_HE;
  TH1F* h_amplError_HE;
  TH1F* h_amplFine_HE;
  TH2F* h_mapDepth1Error_HE;
  TH2F* h_mapDepth2Error_HE;
  TH2F* h_mapDepth3Error_HE;
  TH2F* h_mapDepth4Error_HE;
  TH2F* h_mapDepth5Error_HE;
  TH2F* h_mapDepth6Error_HE;
  TH2F* h_mapDepth7Error_HE;
  TH1F* h_fiber0_HE;
  TH1F* h_fiber1_HE;
  TH1F* h_fiber2_HE;
  TH1F* h_repetedcapid_HE;
  TH1F* h_errorGeneral_HF;
  TH1F* h_error1_HF;
  TH1F* h_error2_HF;
  TH1F* h_error3_HF;
  TH1F* h_error4_HF;
  TH1F* h_error5_HF;
  TH1F* h_error6_HF;
  TH1F* h_error7_HF;
  TH1F* h_amplError_HF;
  TH1F* h_amplFine_HF;
  TH2F* h_mapDepth1Error_HF;
  TH2F* h_mapDepth2Error_HF;
  TH2F* h_mapDepth3Error_HF;
  TH2F* h_mapDepth4Error_HF;
  TH1F* h_fiber0_HF;
  TH1F* h_fiber1_HF;
  TH1F* h_fiber2_HF;
  TH1F* h_repetedcapid_HF;
  TH1F* h_errorGeneral_HO;
  TH1F* h_error1_HO;
  TH1F* h_error2_HO;
  TH1F* h_error3_HO;
  TH1F* h_error4_HO;
  TH1F* h_error5_HO;
  TH1F* h_error6_HO;
  TH1F* h_error7_HO;
  TH1F* h_amplError_HO;
  TH1F* h_amplFine_HO;
  TH2F* h_mapDepth4Error_HO;
  TH1F* h_fiber0_HO;
  TH1F* h_fiber1_HO;
  TH1F* h_fiber2_HO;
  TH1F* h_repetedcapid_HO;
  /////////////////////////////////////////////
  TH2F* h_mapDepth1ADCAmpl225Copy_HB;
  TH2F* h_mapDepth2ADCAmpl225Copy_HB;
  TH2F* h_mapDepth3ADCAmpl225Copy_HB;
  TH2F* h_mapDepth4ADCAmpl225Copy_HB;
  TH2F* h_mapDepth1ADCAmpl225Copy_HE;
  TH2F* h_mapDepth2ADCAmpl225Copy_HE;
  TH2F* h_mapDepth3ADCAmpl225Copy_HE;
  TH2F* h_mapDepth4ADCAmpl225Copy_HE;
  TH2F* h_mapDepth5ADCAmpl225Copy_HE;
  TH2F* h_mapDepth6ADCAmpl225Copy_HE;
  TH2F* h_mapDepth7ADCAmpl225Copy_HE;
  TH2F* h_mapDepth1ADCAmpl225Copy_HF;
  TH2F* h_mapDepth2ADCAmpl225Copy_HF;
  TH2F* h_mapDepth3ADCAmpl225Copy_HF;
  TH2F* h_mapDepth4ADCAmpl225Copy_HF;
  TH2F* h_mapDepth4ADCAmpl225Copy_HO;
  TH1F* h_ADCAmpl345Zoom_HE;
  TH1F* h_ADCAmpl345Zoom1_HE;
  TH1F* h_ADCAmpl345_HE;
  TH1F* h_ADCAmpl345Zoom_HB;
  TH1F* h_ADCAmpl345Zoom1_HB;
  TH1F* h_ADCAmpl345_HB;
  TH1F* h_ADCAmpl_HBCapIdNoError;
  TH1F* h_ADCAmpl345_HBCapIdNoError;
  TH1F* h_ADCAmpl_HBCapIdError;
  TH1F* h_ADCAmpl345_HBCapIdError;
  TH1F* h_ADCAmplZoom_HB;
  TH1F* h_ADCAmplZoom1_HB;
  TH1F* h_ADCAmpl_HB;
  TH1F* h_AmplitudeHBrest;
  TH1F* h_AmplitudeHBrest1;
  TH1F* h_AmplitudeHBrest6;
  TH2F* h_mapDepth1ADCAmpl_HB;
  TH2F* h_mapDepth2ADCAmpl_HB;
  TH2F* h_mapDepth1ADCAmpl225_HB;
  TH2F* h_mapDepth2ADCAmpl225_HB;
  TH2F* h_mapDepth3ADCAmpl_HB;
  TH2F* h_mapDepth4ADCAmpl_HB;
  TH2F* h_mapDepth3ADCAmpl225_HB;
  TH2F* h_mapDepth4ADCAmpl225_HB;
  TH1F* h_TSmeanA_HB;
  TH2F* h_mapDepth1TSmeanA_HB;
  TH2F* h_mapDepth2TSmeanA_HB;
  TH2F* h_mapDepth1TSmeanA225_HB;
  TH2F* h_mapDepth2TSmeanA225_HB;
  TH2F* h_mapDepth3TSmeanA_HB;
  TH2F* h_mapDepth4TSmeanA_HB;
  TH2F* h_mapDepth3TSmeanA225_HB;
  TH2F* h_mapDepth4TSmeanA225_HB;
  TH1F* h_TSmaxA_HB;
  TH2F* h_mapDepth1TSmaxA_HB;
  TH2F* h_mapDepth2TSmaxA_HB;
  TH2F* h_mapDepth1TSmaxA225_HB;
  TH2F* h_mapDepth2TSmaxA225_HB;
  TH2F* h_mapDepth3TSmaxA_HB;
  TH2F* h_mapDepth4TSmaxA_HB;
  TH2F* h_mapDepth3TSmaxA225_HB;
  TH2F* h_mapDepth4TSmaxA225_HB;
  TH1F* h_Amplitude_HB;
  TH2F* h_mapDepth1Amplitude_HB;
  TH2F* h_mapDepth2Amplitude_HB;
  TH2F* h_mapDepth1Amplitude225_HB;
  TH2F* h_mapDepth2Amplitude225_HB;
  TH2F* h_mapDepth3Amplitude_HB;
  TH2F* h_mapDepth4Amplitude_HB;
  TH2F* h_mapDepth3Amplitude225_HB;
  TH2F* h_mapDepth4Amplitude225_HB;
  TH1F* h_Ampl_HB;
  TH2F* h_mapDepth1Ampl047_HB;
  TH2F* h_mapDepth2Ampl047_HB;
  TH2F* h_mapDepth1Ampl_HB;
  TH2F* h_mapDepth2Ampl_HB;
  TH2F* h_mapDepth1AmplE34_HB;
  TH2F* h_mapDepth2AmplE34_HB;
  TH2F* h_mapDepth1_HB;
  TH2F* h_mapDepth2_HB;
  TH2F* h_mapDepth3Ampl047_HB;
  TH2F* h_mapDepth4Ampl047_HB;
  TH2F* h_mapDepth3Ampl_HB;
  TH2F* h_mapDepth4Ampl_HB;
  TH2F* h_mapDepth3AmplE34_HB;
  TH2F* h_mapDepth4AmplE34_HB;
  TH2F* h_mapDepth3_HB;
  TH2F* h_mapDepth4_HB;
  TH2F* h_mapDepth1ADCAmpl12_HB;
  TH2F* h_mapDepth2ADCAmpl12_HB;
  TH2F* h_mapDepth3ADCAmpl12_HB;
  TH2F* h_mapDepth4ADCAmpl12_HB;
  TH2F* h_mapDepth1ADCAmpl12_HE;
  TH2F* h_mapDepth2ADCAmpl12_HE;
  TH2F* h_mapDepth3ADCAmpl12_HE;
  TH2F* h_mapDepth4ADCAmpl12_HE;
  TH2F* h_mapDepth5ADCAmpl12_HE;
  TH2F* h_mapDepth6ADCAmpl12_HE;
  TH2F* h_mapDepth7ADCAmpl12_HE;
  TH2F* h_mapDepth1ADCAmpl12SiPM_HE;
  TH2F* h_mapDepth2ADCAmpl12SiPM_HE;
  TH2F* h_mapDepth3ADCAmpl12SiPM_HE;
  TH2F* h_mapDepth1ADCAmpl12_HF;
  TH2F* h_mapDepth2ADCAmpl12_HF;
  TH2F* h_mapDepth3ADCAmpl12_HF;
  TH2F* h_mapDepth4ADCAmpl12_HF;
  TH2F* h_mapDepth4ADCAmpl12_HO;
  TH2F* h_mapDepth1linADCAmpl12_HE;
  TH2F* h_mapDepth2linADCAmpl12_HE;
  TH2F* h_mapDepth3linADCAmpl12_HE;
  /////////////////////////////////////////////
  TH1F* h_ADCAmpl_HF;
  TH1F* h_ADCAmplrest1_HF;
  TH1F* h_ADCAmplrest6_HF;
  TH1F* h_ADCAmplZoom1_HF;
  TH2F* h_mapDepth1ADCAmpl_HF;
  TH2F* h_mapDepth2ADCAmpl_HF;
  TH2F* h_mapDepth1ADCAmpl225_HF;
  TH2F* h_mapDepth2ADCAmpl225_HF;
  TH2F* h_mapDepth3ADCAmpl_HF;
  TH2F* h_mapDepth4ADCAmpl_HF;
  TH2F* h_mapDepth3ADCAmpl225_HF;
  TH2F* h_mapDepth4ADCAmpl225_HF;
  TH1F* h_TSmeanA_HF;
  TH2F* h_mapDepth1TSmeanA_HF;
  TH2F* h_mapDepth2TSmeanA_HF;
  TH2F* h_mapDepth1TSmeanA225_HF;
  TH2F* h_mapDepth2TSmeanA225_HF;
  TH2F* h_mapDepth3TSmeanA_HF;
  TH2F* h_mapDepth4TSmeanA_HF;
  TH2F* h_mapDepth3TSmeanA225_HF;
  TH2F* h_mapDepth4TSmeanA225_HF;
  TH1F* h_TSmaxA_HF;
  TH2F* h_mapDepth1TSmaxA_HF;
  TH2F* h_mapDepth2TSmaxA_HF;
  TH2F* h_mapDepth1TSmaxA225_HF;
  TH2F* h_mapDepth2TSmaxA225_HF;
  TH2F* h_mapDepth3TSmaxA_HF;
  TH2F* h_mapDepth4TSmaxA_HF;
  TH2F* h_mapDepth3TSmaxA225_HF;
  TH2F* h_mapDepth4TSmaxA225_HF;
  TH1F* h_Amplitude_HF;
  TH2F* h_mapDepth1Amplitude_HF;
  TH2F* h_mapDepth2Amplitude_HF;
  TH2F* h_mapDepth1Amplitude225_HF;
  TH2F* h_mapDepth2Amplitude225_HF;
  TH2F* h_mapDepth3Amplitude_HF;
  TH2F* h_mapDepth4Amplitude_HF;
  TH2F* h_mapDepth3Amplitude225_HF;
  TH2F* h_mapDepth4Amplitude225_HF;
  TH1F* h_Ampl_HF;
  TH2F* h_mapDepth1Ampl047_HF;
  TH2F* h_mapDepth2Ampl047_HF;
  TH2F* h_mapDepth1Ampl_HF;
  TH2F* h_mapDepth2Ampl_HF;
  TH2F* h_mapDepth1AmplE34_HF;
  TH2F* h_mapDepth2AmplE34_HF;
  TH2F* h_mapDepth1_HF;
  TH2F* h_mapDepth2_HF;
  TH2F* h_mapDepth3Ampl047_HF;
  TH2F* h_mapDepth4Ampl047_HF;
  TH2F* h_mapDepth3Ampl_HF;
  TH2F* h_mapDepth4Ampl_HF;
  TH2F* h_mapDepth3AmplE34_HF;
  TH2F* h_mapDepth4AmplE34_HF;
  TH2F* h_mapDepth3_HF;
  TH2F* h_mapDepth4_HF;
  /////////////////////////////////////////////
  TH1F* h_ADCAmpl_HO;
  TH1F* h_ADCAmplrest1_HO;
  TH1F* h_ADCAmplrest6_HO;
  TH1F* h_ADCAmplZoom1_HO;
  TH1F* h_ADCAmpl_HO_copy;
  TH2F* h_mapDepth4ADCAmpl_HO;
  TH2F* h_mapDepth4ADCAmpl225_HO;
  TH1F* h_TSmeanA_HO;
  TH2F* h_mapDepth4TSmeanA_HO;
  TH2F* h_mapDepth4TSmeanA225_HO;
  TH1F* h_TSmaxA_HO;
  TH2F* h_mapDepth4TSmaxA_HO;
  TH2F* h_mapDepth4TSmaxA225_HO;
  TH1F* h_Amplitude_HO;
  TH2F* h_mapDepth4Amplitude_HO;
  TH2F* h_mapDepth4Amplitude225_HO;
  TH1F* h_Ampl_HO;
  TH2F* h_mapDepth4Ampl047_HO;
  TH2F* h_mapDepth4Ampl_HO;
  TH2F* h_mapDepth4AmplE34_HO;
  TH2F* h_mapDepth4_HO;
  /////////////////////////////////////////////
  TH1F* h_nbadchannels_depth1_HB;
  TH1F* h_runnbadchannels_depth1_HB;
  TH1F* h_runnbadchannelsC_depth1_HB;
  TH1F* h_runbadrate_depth1_HB;
  TH1F* h_runbadrateC_depth1_HB;
  TH1F* h_runbadrate0_depth1_HB;
  TH1F* h_nbadchannels_depth2_HB;
  TH1F* h_runnbadchannels_depth2_HB;
  TH1F* h_runnbadchannelsC_depth2_HB;
  TH1F* h_runbadrate_depth2_HB;
  TH1F* h_runbadrateC_depth2_HB;
  TH1F* h_runbadrate0_depth2_HB;
  TH1F* h_nbadchannels_depth1_HE;
  TH1F* h_runnbadchannels_depth1_HE;
  TH1F* h_runnbadchannelsC_depth1_HE;
  TH1F* h_runbadrate_depth1_HE;
  TH1F* h_runbadrateC_depth1_HE;
  TH1F* h_runbadrate0_depth1_HE;
  TH1F* h_nbadchannels_depth2_HE;
  TH1F* h_runnbadchannels_depth2_HE;
  TH1F* h_runnbadchannelsC_depth2_HE;
  TH1F* h_runbadrate_depth2_HE;
  TH1F* h_runbadrateC_depth2_HE;
  TH1F* h_runbadrate0_depth2_HE;
  TH1F* h_nbadchannels_depth3_HE;
  TH1F* h_runnbadchannels_depth3_HE;
  TH1F* h_runnbadchannelsC_depth3_HE;
  TH1F* h_runbadrate_depth3_HE;
  TH1F* h_runbadrateC_depth3_HE;
  TH1F* h_runbadrate0_depth3_HE;
  TH1F* h_nbadchannels_depth4_HO;
  TH1F* h_runnbadchannels_depth4_HO;
  TH1F* h_runnbadchannelsC_depth4_HO;
  TH1F* h_runbadrate_depth4_HO;
  TH1F* h_runbadrateC_depth4_HO;
  TH1F* h_runbadrate0_depth4_HO;
  TH1F* h_nbadchannels_depth1_HF;
  TH1F* h_runnbadchannels_depth1_HF;
  TH1F* h_runnbadchannelsC_depth1_HF;
  TH1F* h_runbadrate_depth1_HF;
  TH1F* h_runbadrateC_depth1_HF;
  TH1F* h_runbadrate0_depth1_HF;
  TH1F* h_nbadchannels_depth2_HF;
  TH1F* h_runnbadchannels_depth2_HF;
  TH1F* h_runnbadchannelsC_depth2_HF;
  TH1F* h_runbadrate_depth2_HF;
  TH1F* h_runbadrateC_depth2_HF;
  TH1F* h_runbadrate0_depth2_HF;
  TH1F* h_bcnnbadchannels_depth1_HB;
  TH1F* h_bcnnbadchannels_depth2_HB;
  TH1F* h_bcnnbadchannels_depth1_HE;
  TH1F* h_bcnnbadchannels_depth2_HE;
  TH1F* h_bcnnbadchannels_depth3_HE;
  TH1F* h_bcnnbadchannels_depth4_HO;
  TH1F* h_bcnnbadchannels_depth1_HF;
  TH1F* h_bcnnbadchannels_depth2_HF;
  TH1F* h_bcnbadrate0_depth1_HB;
  TH1F* h_bcnbadrate0_depth2_HB;
  TH1F* h_bcnbadrate0_depth1_HE;
  TH1F* h_bcnbadrate0_depth2_HE;
  TH1F* h_bcnbadrate0_depth3_HE;
  TH1F* h_bcnbadrate0_depth4_HO;
  TH1F* h_bcnbadrate0_depth1_HF;
  TH1F* h_bcnbadrate0_depth2_HF;
  TH1F* h_Amplitude_forCapIdErrors_HB1;
  TH1F* h_Amplitude_forCapIdErrors_HB2;
  TH1F* h_Amplitude_forCapIdErrors_HE1;
  TH1F* h_Amplitude_forCapIdErrors_HE2;
  TH1F* h_Amplitude_forCapIdErrors_HE3;
  TH1F* h_Amplitude_forCapIdErrors_HF1;
  TH1F* h_Amplitude_forCapIdErrors_HF2;
  TH1F* h_Amplitude_forCapIdErrors_HO4;
  TH1F* h_Amplitude_notCapIdErrors_HB1;
  TH1F* h_Amplitude_notCapIdErrors_HB2;
  TH1F* h_Amplitude_notCapIdErrors_HE1;
  TH1F* h_Amplitude_notCapIdErrors_HE2;
  TH1F* h_Amplitude_notCapIdErrors_HE3;
  TH1F* h_Amplitude_notCapIdErrors_HF1;
  TH1F* h_Amplitude_notCapIdErrors_HF2;
  TH1F* h_Amplitude_notCapIdErrors_HO4;
  /////////////////////////////////////////////
  TH1F* h_corrforxaMAIN_HE;
  TH1F* h_corrforxaMAIN0_HE;
  TH1F* h_corrforxaADDI_HE;
  TH1F* h_corrforxaADDI0_HE;
  TH1F* h_ADCAmpl_HE;
  TH1F* h_ADCAmplrest_HE;
  TH1F* h_ADCAmplrest1_HE;
  TH1F* h_ADCAmplrest6_HE;
  TH1F* h_ADCAmplZoom1_HE;
  TH2F* h_mapDepth1ADCAmpl_HE;
  TH2F* h_mapDepth2ADCAmpl_HE;
  TH2F* h_mapDepth3ADCAmpl_HE;
  TH2F* h_mapDepth4ADCAmpl_HE;
  TH2F* h_mapDepth5ADCAmpl_HE;
  TH2F* h_mapDepth6ADCAmpl_HE;
  TH2F* h_mapDepth7ADCAmpl_HE;
  TH2F* h_mapADCAmplfirstpeak_HE;
  TH2F* h_mapADCAmplfirstpeak0_HE;
  TH2F* h_mapADCAmplsecondpeak_HE;
  TH2F* h_mapADCAmplsecondpeak0_HE;
  TH2F* h_mapADCAmpl11firstpeak_HE;
  TH2F* h_mapADCAmpl11firstpeak0_HE;
  TH2F* h_mapADCAmpl11secondpeak_HE;
  TH2F* h_mapADCAmpl11secondpeak0_HE;
  TH2F* h_mapADCAmpl12firstpeak_HE;
  TH2F* h_mapADCAmpl12firstpeak0_HE;
  TH2F* h_mapADCAmpl12secondpeak_HE;
  TH2F* h_mapADCAmpl12secondpeak0_HE;
  TH1F* h_gsmdifferencefit1_HE;
  TH1F* h_gsmdifferencefit2_HE;
  TH1F* h_gsmdifferencefit3_HE;
  TH1F* h_gsmdifferencefit4_HE;
  TH1F* h_gsmdifferencefit5_HE;
  TH1F* h_gsmdifferencefit6_HE;
  TH2F* h_mapDepth1ADCAmplSiPM_HE;
  TH2F* h_mapDepth2ADCAmplSiPM_HE;
  TH2F* h_mapDepth3ADCAmplSiPM_HE;
  TH2F* h_mapDepth1ADCAmpl225_HE;
  TH2F* h_mapDepth2ADCAmpl225_HE;
  TH2F* h_mapDepth3ADCAmpl225_HE;
  TH2F* h_mapDepth4ADCAmpl225_HE;
  TH2F* h_mapDepth5ADCAmpl225_HE;
  TH2F* h_mapDepth6ADCAmpl225_HE;
  TH2F* h_mapDepth7ADCAmpl225_HE;
  TH1F* h_TSmeanA_HE;
  TH2F* h_mapDepth1TSmeanA_HE;
  TH2F* h_mapDepth2TSmeanA_HE;
  TH2F* h_mapDepth3TSmeanA_HE;
  TH2F* h_mapDepth4TSmeanA_HE;
  TH2F* h_mapDepth5TSmeanA_HE;
  TH2F* h_mapDepth6TSmeanA_HE;
  TH2F* h_mapDepth7TSmeanA_HE;
  TH2F* h_mapDepth1TSmeanA225_HE;
  TH2F* h_mapDepth2TSmeanA225_HE;
  TH2F* h_mapDepth3TSmeanA225_HE;
  TH2F* h_mapDepth4TSmeanA225_HE;
  TH2F* h_mapDepth5TSmeanA225_HE;
  TH2F* h_mapDepth6TSmeanA225_HE;
  TH2F* h_mapDepth7TSmeanA225_HE;
  TH1F* h_TSmaxA_HE;
  TH2F* h_mapDepth1TSmaxA_HE;
  TH2F* h_mapDepth2TSmaxA_HE;
  TH2F* h_mapDepth3TSmaxA_HE;
  TH2F* h_mapDepth4TSmaxA_HE;
  TH2F* h_mapDepth5TSmaxA_HE;
  TH2F* h_mapDepth6TSmaxA_HE;
  TH2F* h_mapDepth7TSmaxA_HE;
  TH2F* h_mapDepth1TSmaxA225_HE;
  TH2F* h_mapDepth2TSmaxA225_HE;
  TH2F* h_mapDepth3TSmaxA225_HE;
  TH2F* h_mapDepth4TSmaxA225_HE;
  TH2F* h_mapDepth5TSmaxA225_HE;
  TH2F* h_mapDepth6TSmaxA225_HE;
  TH2F* h_mapDepth7TSmaxA225_HE;
  TH1F* h_Amplitude_HE;
  TH2F* h_mapDepth1Amplitude_HE;
  TH2F* h_mapDepth2Amplitude_HE;
  TH2F* h_mapDepth3Amplitude_HE;
  TH2F* h_mapDepth4Amplitude_HE;
  TH2F* h_mapDepth5Amplitude_HE;
  TH2F* h_mapDepth6Amplitude_HE;
  TH2F* h_mapDepth7Amplitude_HE;
  TH2F* h_mapDepth1Amplitude225_HE;
  TH2F* h_mapDepth2Amplitude225_HE;
  TH2F* h_mapDepth3Amplitude225_HE;
  TH2F* h_mapDepth4Amplitude225_HE;
  TH2F* h_mapDepth5Amplitude225_HE;
  TH2F* h_mapDepth6Amplitude225_HE;
  TH2F* h_mapDepth7Amplitude225_HE;
  TH1F* h_Ampl_HE;
  TH2F* h_mapDepth1Ampl047_HE;
  TH2F* h_mapDepth2Ampl047_HE;
  TH2F* h_mapDepth3Ampl047_HE;
  TH2F* h_mapDepth4Ampl047_HE;
  TH2F* h_mapDepth5Ampl047_HE;
  TH2F* h_mapDepth6Ampl047_HE;
  TH2F* h_mapDepth7Ampl047_HE;
  TH2F* h_mapDepth1Ampl_HE;
  TH2F* h_mapDepth2Ampl_HE;
  TH2F* h_mapDepth3Ampl_HE;
  TH2F* h_mapDepth4Ampl_HE;
  TH2F* h_mapDepth5Ampl_HE;
  TH2F* h_mapDepth6Ampl_HE;
  TH2F* h_mapDepth7Ampl_HE;
  TH2F* h_mapDepth1AmplE34_HE;
  TH2F* h_mapDepth2AmplE34_HE;
  TH2F* h_mapDepth3AmplE34_HE;
  TH2F* h_mapDepth4AmplE34_HE;
  TH2F* h_mapDepth5AmplE34_HE;
  TH2F* h_mapDepth6AmplE34_HE;
  TH2F* h_mapDepth7AmplE34_HE;
  TH2F* h_mapDepth1_HE;
  TH2F* h_mapDepth2_HE;
  TH2F* h_mapDepth3_HE;
  TH2F* h_mapDepth4_HE;
  TH2F* h_mapDepth5_HE;
  TH2F* h_mapDepth6_HE;
  TH2F* h_mapDepth7_HE;
  TH2F* h_FullSignal3D_HB;
  TH2F* h_FullSignal3D0_HB;
  TH2F* h_FullSignal3D_HE;
  TH2F* h_FullSignal3D0_HE;
  TH2F* h_FullSignal3D_HO;
  TH2F* h_FullSignal3D0_HO;
  TH2F* h_FullSignal3D_HF;
  TH2F* h_FullSignal3D0_HF;
  TH2F* h_mapCapCalib047_HB;
  TH2F* h_mapCapCalib047_HE;
  TH2F* h_mapCapCalib047_HO;
  TH2F* h_mapCapCalib047_HF;
  TH1F* h_ADCCalib_HB;
  TH1F* h_ADCCalib1_HB;
  TH2F* h_mapADCCalib047_HB;
  TH2F* h_mapADCCalib_HB;
  TH1F* h_RatioCalib_HB;
  TH2F* h_mapRatioCalib047_HB;
  TH2F* h_mapRatioCalib_HB;
  TH1F* h_TSmaxCalib_HB;
  TH2F* h_mapTSmaxCalib047_HB;
  TH2F* h_mapTSmaxCalib_HB;
  TH1F* h_TSmeanCalib_HB;
  TH2F* h_mapTSmeanCalib047_HB;
  TH2F* h_mapTSmeanCalib_HB;
  TH1F* h_WidthCalib_HB;
  TH2F* h_mapWidthCalib047_HB;
  TH2F* h_mapWidthCalib_HB;
  TH2F* h_map_HB;
  TH1F* h_ADCCalib_HE;
  TH1F* h_ADCCalib1_HE;
  TH2F* h_mapADCCalib047_HE;
  TH2F* h_mapADCCalib_HE;
  TH1F* h_RatioCalib_HE;
  TH2F* h_mapRatioCalib047_HE;
  TH2F* h_mapRatioCalib_HE;
  TH1F* h_TSmaxCalib_HE;
  TH2F* h_mapTSmaxCalib047_HE;
  TH2F* h_mapTSmaxCalib_HE;
  TH1F* h_TSmeanCalib_HE;
  TH2F* h_mapTSmeanCalib047_HE;
  TH2F* h_mapTSmeanCalib_HE;
  TH1F* h_WidthCalib_HE;
  TH2F* h_mapWidthCalib047_HE;
  TH2F* h_mapWidthCalib_HE;
  TH2F* h_map_HE;
  TH1F* h_ADCCalib_HO;
  TH1F* h_ADCCalib1_HO;
  TH2F* h_mapADCCalib047_HO;
  TH2F* h_mapADCCalib_HO;
  TH1F* h_RatioCalib_HO;
  TH2F* h_mapRatioCalib047_HO;
  TH2F* h_mapRatioCalib_HO;
  TH1F* h_TSmaxCalib_HO;
  TH2F* h_mapTSmaxCalib047_HO;
  TH2F* h_mapTSmaxCalib_HO;
  TH1F* h_TSmeanCalib_HO;
  TH2F* h_mapTSmeanCalib047_HO;
  TH2F* h_mapTSmeanCalib_HO;
  TH1F* h_WidthCalib_HO;
  TH2F* h_mapWidthCalib047_HO;
  TH2F* h_mapWidthCalib_HO;
  TH2F* h_map_HO;
  TH1F* h_ADCCalib_HF;
  TH1F* h_ADCCalib1_HF;
  TH2F* h_mapADCCalib047_HF;
  TH2F* h_mapADCCalib_HF;
  TH1F* h_RatioCalib_HF;
  TH2F* h_mapRatioCalib047_HF;
  TH2F* h_mapRatioCalib_HF;
  TH1F* h_TSmaxCalib_HF;
  TH2F* h_mapTSmaxCalib047_HF;
  TH2F* h_mapTSmaxCalib_HF;
  TH1F* h_TSmeanCalib_HF;
  TH2F* h_mapTSmeanCalib047_HF;
  TH2F* h_mapTSmeanCalib_HF;
  TH1F* h_WidthCalib_HF;
  TH2F* h_mapWidthCalib047_HF;
  TH2F* h_mapWidthCalib_HF;
  TH2F* h_map_HF;
  TH1F* h_nls_per_run;
  TH1F* h_nls_per_run10;
  TH1F* h_nevents_per_LS;
  TH1F* h_nevents_per_LSzoom;
  TH1F* h_nevents_per_eachLS;
  TH1F* h_nevents_per_eachRealLS;
  TH1F* h_lsnumber_per_eachLS;
  TH1F* h_sumPedestalLS1;
  TH2F* h_2DsumPedestalLS1;
  TH1F* h_sumPedestalperLS1;
  TH2F* h_2D0sumPedestalLS1;
  TH1F* h_sum0PedestalperLS1;
  TH1F* h_sumPedestalLS2;
  TH2F* h_2DsumPedestalLS2;
  TH1F* h_sumPedestalperLS2;
  TH2F* h_2D0sumPedestalLS2;
  TH1F* h_sum0PedestalperLS2;
  TH1F* h_sumPedestalLS3;
  TH2F* h_2DsumPedestalLS3;
  TH1F* h_sumPedestalperLS3;
  TH2F* h_2D0sumPedestalLS3;
  TH1F* h_sum0PedestalperLS3;
  TH1F* h_sumPedestalLS4;
  TH2F* h_2DsumPedestalLS4;
  TH1F* h_sumPedestalperLS4;
  TH2F* h_2D0sumPedestalLS4;
  TH1F* h_sum0PedestalperLS4;
  TH1F* h_sumPedestalLS5;
  TH2F* h_2DsumPedestalLS5;
  TH1F* h_sumPedestalperLS5;
  TH2F* h_2D0sumPedestalLS5;
  TH1F* h_sum0PedestalperLS5;
  TH1F* h_sumPedestalLS6;
  TH2F* h_2DsumPedestalLS6;
  TH1F* h_sumPedestalperLS6;
  TH2F* h_2D0sumPedestalLS6;
  TH1F* h_sum0PedestalperLS6;
  TH1F* h_sumPedestalLS7;
  TH2F* h_2DsumPedestalLS7;
  TH1F* h_sumPedestalperLS7;
  TH2F* h_2D0sumPedestalLS7;
  TH1F* h_sum0PedestalperLS7;
  TH1F* h_sumPedestalLS8;
  TH2F* h_2DsumPedestalLS8;
  TH1F* h_sumPedestalperLS8;
  TH2F* h_2D0sumPedestalLS8;
  TH1F* h_sum0PedestalperLS8;
  TH2F* h_2DsumADCAmplLSdepth4HEu;
  TH2F* h_2D0sumADCAmplLSdepth4HEu;
  TH2F* h_2DsumADCAmplLSdepth5HEu;
  TH2F* h_2D0sumADCAmplLSdepth5HEu;
  TH2F* h_2DsumADCAmplLSdepth6HEu;
  TH2F* h_2D0sumADCAmplLSdepth6HEu;
  TH2F* h_2DsumADCAmplLSdepth7HEu;
  TH2F* h_2D0sumADCAmplLSdepth7HEu;
  TH2F* h_2DsumADCAmplLSdepth3HFu;
  TH2F* h_2D0sumADCAmplLSdepth3HFu;
  TH2F* h_2DsumADCAmplLSdepth4HFu;
  TH2F* h_2D0sumADCAmplLSdepth4HFu;
  TH2F* h_2DsumADCAmplLSdepth3HBu;
  TH2F* h_2D0sumADCAmplLSdepth3HBu;
  TH2F* h_2DsumADCAmplLSdepth4HBu;
  TH2F* h_2D0sumADCAmplLSdepth4HBu;
  TH1F* h_sumADCAmplLS1copy1;
  TH1F* h_sumADCAmplLS1copy2;
  TH1F* h_sumADCAmplLS1copy3;
  TH1F* h_sumADCAmplLS1copy4;
  TH1F* h_sumADCAmplLS1copy5;
  TH1F* h_sumADCAmplLS1;
  TH2F* h_2DsumADCAmplLS1;
  TH2F* h_2DsumADCAmplLS1_LSselected;
  TH1F* h_sumADCAmplperLS1;
  TH1F* h_sumCutADCAmplperLS1;
  TH2F* h_2D0sumADCAmplLS1;
  TH1F* h_sum0ADCAmplperLS1;
  TH1F* h_sumADCAmplLS2;
  TH2F* h_2DsumADCAmplLS2;
  TH2F* h_2DsumADCAmplLS2_LSselected;
  TH1F* h_sumADCAmplperLS2;
  TH1F* h_sumCutADCAmplperLS2;
  TH2F* h_2D0sumADCAmplLS2;
  TH1F* h_sum0ADCAmplperLS2;
  TH1F* h_sumADCAmplLS3;
  TH2F* h_2DsumADCAmplLS3;
  TH2F* h_2DsumADCAmplLS3_LSselected;
  TH1F* h_sumADCAmplperLS3;
  TH1F* h_sumCutADCAmplperLS3;
  TH2F* h_2D0sumADCAmplLS3;
  TH1F* h_sum0ADCAmplperLS3;
  TH1F* h_sumADCAmplLS4;
  TH2F* h_2DsumADCAmplLS4;
  TH2F* h_2DsumADCAmplLS4_LSselected;
  TH1F* h_sumADCAmplperLS4;
  TH1F* h_sumCutADCAmplperLS4;
  TH2F* h_2D0sumADCAmplLS4;
  TH1F* h_sum0ADCAmplperLS4;
  TH1F* h_sumADCAmplLS5;
  TH2F* h_2DsumADCAmplLS5;
  TH2F* h_2DsumADCAmplLS5_LSselected;
  TH1F* h_sumADCAmplperLS5;
  TH1F* h_sumCutADCAmplperLS5;
  TH2F* h_2D0sumADCAmplLS5;
  TH1F* h_sum0ADCAmplperLS5;
  TH1F* h_sumADCAmplLS6;
  TH2F* h_2DsumADCAmplLS6;
  TH2F* h_2DsumADCAmplLS6_LSselected;
  TH2F* h_2D0sumADCAmplLS6;
  TH1F* h_sumADCAmplperLS6;
  TH1F* h_sumCutADCAmplperLS6;
  TH1F* h_sum0ADCAmplperLS6;
  TH1F* h_sumADCAmplperLSdepth4HEu;
  TH1F* h_sumADCAmplperLSdepth5HEu;
  TH1F* h_sumADCAmplperLSdepth6HEu;
  TH1F* h_sumADCAmplperLSdepth7HEu;
  TH1F* h_sumCutADCAmplperLSdepth4HEu;
  TH1F* h_sumCutADCAmplperLSdepth5HEu;
  TH1F* h_sumCutADCAmplperLSdepth6HEu;
  TH1F* h_sumCutADCAmplperLSdepth7HEu;
  TH1F* h_sum0ADCAmplperLSdepth4HEu;
  TH1F* h_sum0ADCAmplperLSdepth5HEu;
  TH1F* h_sum0ADCAmplperLSdepth6HEu;
  TH1F* h_sum0ADCAmplperLSdepth7HEu;
  TH1F* h_sumADCAmplperLSdepth3HBu;
  TH1F* h_sumADCAmplperLSdepth4HBu;
  TH1F* h_sumCutADCAmplperLSdepth3HBu;
  TH1F* h_sumCutADCAmplperLSdepth4HBu;
  TH1F* h_sum0ADCAmplperLSdepth3HBu;
  TH1F* h_sum0ADCAmplperLSdepth4HBu;
  TH1F* h_sumADCAmplperLS6u;
  TH1F* h_sumCutADCAmplperLS6u;
  TH1F* h_sum0ADCAmplperLS6u;
  TH1F* h_sumADCAmplperLS1_P1;
  TH1F* h_sum0ADCAmplperLS1_P1;
  TH1F* h_sumADCAmplperLS1_P2;
  TH1F* h_sum0ADCAmplperLS1_P2;
  TH1F* h_sumADCAmplperLS1_M1;
  TH1F* h_sum0ADCAmplperLS1_M1;
  TH1F* h_sumADCAmplperLS1_M2;
  TH1F* h_sum0ADCAmplperLS1_M2;
  TH1F* h_sumADCAmplperLS3_P1;
  TH1F* h_sum0ADCAmplperLS3_P1;
  TH1F* h_sumADCAmplperLS3_P2;
  TH1F* h_sum0ADCAmplperLS3_P2;
  TH1F* h_sumADCAmplperLS3_M1;
  TH1F* h_sum0ADCAmplperLS3_M1;
  TH1F* h_sumADCAmplperLS3_M2;
  TH1F* h_sum0ADCAmplperLS3_M2;
  TH1F* h_sumADCAmplperLS6_P1;
  TH1F* h_sum0ADCAmplperLS6_P1;
  TH1F* h_sumADCAmplperLS6_P2;
  TH1F* h_sum0ADCAmplperLS6_P2;
  TH1F* h_sumADCAmplperLS6_M1;
  TH1F* h_sum0ADCAmplperLS6_M1;
  TH1F* h_sumADCAmplperLS6_M2;
  TH1F* h_sum0ADCAmplperLS6_M2;
  TH1F* h_sumADCAmplperLS8_P1;
  TH1F* h_sum0ADCAmplperLS8_P1;
  TH1F* h_sumADCAmplperLS8_P2;
  TH1F* h_sum0ADCAmplperLS8_P2;
  TH1F* h_sumADCAmplperLS8_M1;
  TH1F* h_sum0ADCAmplperLS8_M1;
  TH1F* h_sumADCAmplperLS8_M2;
  TH1F* h_sum0ADCAmplperLS8_M2;
  TH1F* h_sumADCAmplLS7;
  TH2F* h_2DsumADCAmplLS7;
  TH2F* h_2DsumADCAmplLS7_LSselected;
  TH2F* h_2D0sumADCAmplLS7;
  TH1F* h_sumADCAmplperLS7;
  TH1F* h_sumCutADCAmplperLS7;
  TH1F* h_sum0ADCAmplperLS7;
  TH1F* h_sumADCAmplperLS7u;
  TH1F* h_sumCutADCAmplperLS7u;
  TH1F* h_sum0ADCAmplperLS7u;
  TH1F* h_sumADCAmplLS8;
  TH2F* h_2DsumADCAmplLS8;
  TH2F* h_2DsumADCAmplLS8_LSselected;
  TH1F* h_sumADCAmplperLS8;
  TH1F* h_sumCutADCAmplperLS8;
  TH2F* h_2D0sumADCAmplLS8;
  TH1F* h_sum0ADCAmplperLS8;
  TH1F* h_sumTSmeanALS1;
  TH2F* h_2DsumTSmeanALS1;
  TH1F* h_sumTSmeanAperLS1;
  TH1F* h_sumTSmeanAperLS1_LSselected;
  TH1F* h_sumCutTSmeanAperLS1;
  TH2F* h_2D0sumTSmeanALS1;
  TH1F* h_sum0TSmeanAperLS1;
  TH1F* h_sumTSmeanALS2;
  TH2F* h_2DsumTSmeanALS2;
  TH1F* h_sumTSmeanAperLS2;
  TH1F* h_sumCutTSmeanAperLS2;
  TH2F* h_2D0sumTSmeanALS2;
  TH1F* h_sum0TSmeanAperLS2;
  TH1F* h_sumTSmeanALS3;
  TH2F* h_2DsumTSmeanALS3;
  TH1F* h_sumTSmeanAperLS3;
  TH1F* h_sumCutTSmeanAperLS3;
  TH2F* h_2D0sumTSmeanALS3;
  TH1F* h_sum0TSmeanAperLS3;
  TH1F* h_sumTSmeanALS4;
  TH2F* h_2DsumTSmeanALS4;
  TH1F* h_sumTSmeanAperLS4;
  TH1F* h_sumCutTSmeanAperLS4;
  TH2F* h_2D0sumTSmeanALS4;
  TH1F* h_sum0TSmeanAperLS4;
  TH1F* h_sumTSmeanALS5;
  TH2F* h_2DsumTSmeanALS5;
  TH1F* h_sumTSmeanAperLS5;
  TH1F* h_sumCutTSmeanAperLS5;
  TH2F* h_2D0sumTSmeanALS5;
  TH1F* h_sum0TSmeanAperLS5;
  TH1F* h_sumTSmeanALS6;
  TH2F* h_2DsumTSmeanALS6;
  TH1F* h_sumTSmeanAperLS6;
  TH1F* h_sumCutTSmeanAperLS6;
  TH2F* h_2D0sumTSmeanALS6;
  TH1F* h_sum0TSmeanAperLS6;
  TH1F* h_sumTSmeanALS7;
  TH2F* h_2DsumTSmeanALS7;
  TH1F* h_sumTSmeanAperLS7;
  TH1F* h_sumCutTSmeanAperLS7;
  TH2F* h_2D0sumTSmeanALS7;
  TH1F* h_sum0TSmeanAperLS7;
  TH1F* h_sumTSmeanALS8;
  TH2F* h_2DsumTSmeanALS8;
  TH1F* h_sumTSmeanAperLS8;
  TH1F* h_sumCutTSmeanAperLS8;
  TH2F* h_2D0sumTSmeanALS8;
  TH1F* h_sum0TSmeanAperLS8;
  TH1F* h_sumTSmaxALS1;
  TH2F* h_2DsumTSmaxALS1;
  TH1F* h_sumTSmaxAperLS1;
  TH1F* h_sumTSmaxAperLS1_LSselected;
  TH1F* h_sumCutTSmaxAperLS1;
  TH2F* h_2D0sumTSmaxALS1;
  TH1F* h_sum0TSmaxAperLS1;
  TH1F* h_sumTSmaxALS2;
  TH2F* h_2DsumTSmaxALS2;
  TH1F* h_sumTSmaxAperLS2;
  TH1F* h_sumCutTSmaxAperLS2;
  TH2F* h_2D0sumTSmaxALS2;
  TH1F* h_sum0TSmaxAperLS2;
  TH1F* h_sumTSmaxALS3;
  TH2F* h_2DsumTSmaxALS3;
  TH1F* h_sumTSmaxAperLS3;
  TH1F* h_sumCutTSmaxAperLS3;
  TH2F* h_2D0sumTSmaxALS3;
  TH1F* h_sum0TSmaxAperLS3;
  TH1F* h_sumTSmaxALS4;
  TH2F* h_2DsumTSmaxALS4;
  TH1F* h_sumTSmaxAperLS4;
  TH1F* h_sumCutTSmaxAperLS4;
  TH2F* h_2D0sumTSmaxALS4;
  TH1F* h_sum0TSmaxAperLS4;
  TH1F* h_sumTSmaxALS5;
  TH2F* h_2DsumTSmaxALS5;
  TH1F* h_sumTSmaxAperLS5;
  TH1F* h_sumCutTSmaxAperLS5;
  TH2F* h_2D0sumTSmaxALS5;
  TH1F* h_sum0TSmaxAperLS5;
  TH1F* h_sumTSmaxALS6;
  TH2F* h_2DsumTSmaxALS6;
  TH1F* h_sumTSmaxAperLS6;
  TH1F* h_sumCutTSmaxAperLS6;
  TH2F* h_2D0sumTSmaxALS6;
  TH1F* h_sum0TSmaxAperLS6;
  TH1F* h_sumTSmaxALS7;
  TH2F* h_2DsumTSmaxALS7;
  TH1F* h_sumTSmaxAperLS7;
  TH1F* h_sumCutTSmaxAperLS7;
  TH2F* h_2D0sumTSmaxALS7;
  TH1F* h_sum0TSmaxAperLS7;
  TH1F* h_sumTSmaxALS8;
  TH2F* h_2DsumTSmaxALS8;
  TH1F* h_sumTSmaxAperLS8;
  TH1F* h_sumCutTSmaxAperLS8;
  TH2F* h_2D0sumTSmaxALS8;
  TH1F* h_sum0TSmaxAperLS8;
  TH1F* h_sumAmplitudeLS1;
  TH2F* h_2DsumAmplitudeLS1;
  TH1F* h_sumAmplitudeperLS1;
  TH1F* h_sumAmplitudeperLS1_LSselected;
  TH1F* h_sumCutAmplitudeperLS1;
  TH2F* h_2D0sumAmplitudeLS1;
  TH1F* h_sum0AmplitudeperLS1;
  TH1F* h_sumAmplitudeLS2;
  TH2F* h_2DsumAmplitudeLS2;
  TH1F* h_sumAmplitudeperLS2;
  TH1F* h_sumCutAmplitudeperLS2;
  TH2F* h_2D0sumAmplitudeLS2;
  TH1F* h_sum0AmplitudeperLS2;
  TH1F* h_sumAmplitudeLS3;
  TH2F* h_2DsumAmplitudeLS3;
  TH1F* h_sumAmplitudeperLS3;
  TH1F* h_sumCutAmplitudeperLS3;
  TH2F* h_2D0sumAmplitudeLS3;
  TH1F* h_sum0AmplitudeperLS3;
  TH1F* h_sumAmplitudeLS4;
  TH2F* h_2DsumAmplitudeLS4;
  TH1F* h_sumAmplitudeperLS4;
  TH1F* h_sumCutAmplitudeperLS4;
  TH2F* h_2D0sumAmplitudeLS4;
  TH1F* h_sum0AmplitudeperLS4;
  TH1F* h_sumAmplitudeLS5;
  TH2F* h_2DsumAmplitudeLS5;
  TH1F* h_sumAmplitudeperLS5;
  TH1F* h_sumCutAmplitudeperLS5;
  TH2F* h_2D0sumAmplitudeLS5;
  TH1F* h_sum0AmplitudeperLS5;
  TH1F* h_sumAmplitudeLS6;
  TH2F* h_2DsumAmplitudeLS6;
  TH1F* h_sumAmplitudeperLS6;
  TH1F* h_sumCutAmplitudeperLS6;
  TH2F* h_2D0sumAmplitudeLS6;
  TH1F* h_sum0AmplitudeperLS6;
  TH1F* h_sumAmplitudeLS7;
  TH2F* h_2DsumAmplitudeLS7;
  TH1F* h_sumAmplitudeperLS7;
  TH1F* h_sumCutAmplitudeperLS7;
  TH2F* h_2D0sumAmplitudeLS7;
  TH1F* h_sum0AmplitudeperLS7;
  TH1F* h_sumAmplitudeLS8;
  TH2F* h_2DsumAmplitudeLS8;
  TH1F* h_sumAmplitudeperLS8;
  TH1F* h_sumCutAmplitudeperLS8;
  TH2F* h_2D0sumAmplitudeLS8;
  TH1F* h_sum0AmplitudeperLS8;
  TH1F* h_sumAmplLS1;
  TH2F* h_2DsumAmplLS1;
  TH1F* h_sumAmplperLS1;
  TH1F* h_sumAmplperLS1_LSselected;
  TH1F* h_sumCutAmplperLS1;
  TH2F* h_2D0sumAmplLS1;
  TH1F* h_sum0AmplperLS1;
  TH1F* h_sumAmplLS2;
  TH2F* h_2DsumAmplLS2;
  TH1F* h_sumAmplperLS2;
  TH1F* h_sumCutAmplperLS2;
  TH2F* h_2D0sumAmplLS2;
  TH1F* h_sum0AmplperLS2;
  TH1F* h_sumAmplLS3;
  TH2F* h_2DsumAmplLS3;
  TH1F* h_sumAmplperLS3;
  TH1F* h_sumCutAmplperLS3;
  TH2F* h_2D0sumAmplLS3;
  TH1F* h_sum0AmplperLS3;
  TH1F* h_sumAmplLS4;
  TH2F* h_2DsumAmplLS4;
  TH1F* h_sumAmplperLS4;
  TH1F* h_sumCutAmplperLS4;
  TH2F* h_2D0sumAmplLS4;
  TH1F* h_sum0AmplperLS4;
  TH1F* h_sumAmplLS5;
  TH2F* h_2DsumAmplLS5;
  TH1F* h_sumAmplperLS5;
  TH1F* h_sumCutAmplperLS5;
  TH2F* h_2D0sumAmplLS5;
  TH1F* h_sum0AmplperLS5;
  TH1F* h_sumAmplLS6;
  TH2F* h_2DsumAmplLS6;
  TH1F* h_sumAmplperLS6;
  TH1F* h_sumCutAmplperLS6;
  TH2F* h_2D0sumAmplLS6;
  TH1F* h_sum0AmplperLS6;
  TH1F* h_RatioOccupancy_HBP;
  TH1F* h_RatioOccupancy_HBM;
  TH1F* h_RatioOccupancy_HEP;
  TH1F* h_RatioOccupancy_HEM;
  TH1F* h_RatioOccupancy_HOP;
  TH1F* h_RatioOccupancy_HOM;
  TH1F* h_RatioOccupancy_HFP;
  TH1F* h_RatioOccupancy_HFM;
  TH1F* h_sumAmplLS7;
  TH2F* h_2DsumAmplLS7;
  TH1F* h_sumAmplperLS7;
  TH1F* h_sumCutAmplperLS7;
  TH2F* h_2D0sumAmplLS7;
  TH1F* h_sum0AmplperLS7;
  TH1F* h_sumAmplLS8;
  TH2F* h_2DsumAmplLS8;
  TH1F* h_sumAmplperLS8;
  TH1F* h_sumCutAmplperLS8;
  TH2F* h_2D0sumAmplLS8;
  TH1F* h_sum0AmplperLS8;
  TH1F* h_pedestal0_HB;
  TH1F* h_pedestal1_HB;
  TH1F* h_pedestal2_HB;
  TH1F* h_pedestal3_HB;
  TH1F* h_pedestalaver4_HB;
  TH1F* h_pedestalaver9_HB;
  TH1F* h_pedestalw0_HB;
  TH1F* h_pedestalw1_HB;
  TH1F* h_pedestalw2_HB;
  TH1F* h_pedestalw3_HB;
  TH1F* h_pedestalwaver4_HB;
  TH1F* h_pedestalwaver9_HB;
  TH1F* h_pedestal0_HE;
  TH1F* h_pedestal1_HE;
  TH1F* h_pedestal2_HE;
  TH1F* h_pedestal3_HE;
  TH1F* h_pedestalaver4_HE;
  TH1F* h_pedestalaver9_HE;
  TH1F* h_pedestalw0_HE;
  TH1F* h_pedestalw1_HE;
  TH1F* h_pedestalw2_HE;
  TH1F* h_pedestalw3_HE;
  TH1F* h_pedestalwaver4_HE;
  TH1F* h_pedestalwaver9_HE;
  TH1F* h_pedestal0_HF;
  TH1F* h_pedestal1_HF;
  TH1F* h_pedestal2_HF;
  TH1F* h_pedestal3_HF;
  TH1F* h_pedestalaver4_HF;
  TH1F* h_pedestalaver9_HF;
  TH1F* h_pedestalw0_HF;
  TH1F* h_pedestalw1_HF;
  TH1F* h_pedestalw2_HF;
  TH1F* h_pedestalw3_HF;
  TH1F* h_pedestalwaver4_HF;
  TH1F* h_pedestalwaver9_HF;
  TH1F* h_pedestal0_HO;
  TH1F* h_pedestal1_HO;
  TH1F* h_pedestal2_HO;
  TH1F* h_pedestal3_HO;
  TH1F* h_pedestalaver4_HO;
  TH1F* h_pedestalaver9_HO;
  TH1F* h_pedestalw0_HO;
  TH1F* h_pedestalw1_HO;
  TH1F* h_pedestalw2_HO;
  TH1F* h_pedestalw3_HO;
  TH1F* h_pedestalwaver4_HO;
  TH1F* h_pedestalwaver9_HO;
  TH2F* h_mapDepth1pedestalw_HB;
  TH2F* h_mapDepth2pedestalw_HB;
  TH2F* h_mapDepth3pedestalw_HB;
  TH2F* h_mapDepth4pedestalw_HB;
  TH2F* h_mapDepth1pedestalw_HE;
  TH2F* h_mapDepth2pedestalw_HE;
  TH2F* h_mapDepth3pedestalw_HE;
  TH2F* h_mapDepth4pedestalw_HE;
  TH2F* h_mapDepth5pedestalw_HE;
  TH2F* h_mapDepth6pedestalw_HE;
  TH2F* h_mapDepth7pedestalw_HE;
  TH2F* h_mapDepth1pedestalw_HF;
  TH2F* h_mapDepth2pedestalw_HF;
  TH2F* h_mapDepth3pedestalw_HF;
  TH2F* h_mapDepth4pedestalw_HF;
  TH2F* h_mapDepth4pedestalw_HO;
  TH2F* h_mapDepth1pedestal_HB;
  TH2F* h_mapDepth2pedestal_HB;
  TH2F* h_mapDepth3pedestal_HB;
  TH2F* h_mapDepth4pedestal_HB;
  TH2F* h_mapDepth1pedestal_HE;
  TH2F* h_mapDepth2pedestal_HE;
  TH2F* h_mapDepth3pedestal_HE;
  TH2F* h_mapDepth4pedestal_HE;
  TH2F* h_mapDepth5pedestal_HE;
  TH2F* h_mapDepth6pedestal_HE;
  TH2F* h_mapDepth7pedestal_HE;
  TH2F* h_mapDepth1pedestal_HF;
  TH2F* h_mapDepth2pedestal_HF;
  TH2F* h_mapDepth3pedestal_HF;
  TH2F* h_mapDepth4pedestal_HF;
  TH2F* h_mapDepth4pedestal_HO;
  TH1F* h_pedestal00_HB;
  TH1F* h_gain_HB;
  TH1F* h_respcorr_HB;
  TH1F* h_timecorr_HB;
  TH1F* h_lutcorr_HB;
  TH1F* h_difpedestal0_HB;
  TH1F* h_difpedestal1_HB;
  TH1F* h_difpedestal2_HB;
  TH1F* h_difpedestal3_HB;
  TH1F* h_pedestal00_HE;
  TH1F* h_gain_HE;
  TH1F* h_respcorr_HE;
  TH1F* h_timecorr_HE;
  TH1F* h_lutcorr_HE;
  TH1F* h_pedestal00_HF;
  TH1F* h_gain_HF;
  TH1F* h_respcorr_HF;
  TH1F* h_timecorr_HF;
  TH1F* h_lutcorr_HF;
  TH1F* h_pedestal00_HO;
  TH1F* h_gain_HO;
  TH1F* h_respcorr_HO;
  TH1F* h_timecorr_HO;
  TH1F* h_lutcorr_HO;
  TH2F* h2_pedvsampl_HB;
  TH2F* h2_pedwvsampl_HB;
  TH1F* h_pedvsampl_HB;
  TH1F* h_pedwvsampl_HB;
  TH1F* h_pedvsampl0_HB;
  TH1F* h_pedwvsampl0_HB;
  TH2F* h2_amplvsped_HB;
  TH2F* h2_amplvspedw_HB;
  TH1F* h_amplvsped_HB;
  TH1F* h_amplvspedw_HB;
  TH1F* h_amplvsped0_HB;
  TH2F* h2_pedvsampl_HE;
  TH2F* h2_pedwvsampl_HE;
  TH1F* h_pedvsampl_HE;
  TH1F* h_pedwvsampl_HE;
  TH1F* h_pedvsampl0_HE;
  TH1F* h_pedwvsampl0_HE;
  TH2F* h2_pedvsampl_HF;
  TH2F* h2_pedwvsampl_HF;
  TH1F* h_pedvsampl_HF;
  TH1F* h_pedwvsampl_HF;
  TH1F* h_pedvsampl0_HF;
  TH1F* h_pedwvsampl0_HF;
  TH2F* h2_pedvsampl_HO;
  TH2F* h2_pedwvsampl_HO;
  TH1F* h_pedvsampl_HO;
  TH1F* h_pedwvsampl_HO;
  TH1F* h_pedvsampl0_HO;
  TH1F* h_pedwvsampl0_HO;
  TH1F* h_shape_Ahigh_HB0;
  TH1F* h_shape0_Ahigh_HB0;
  TH1F* h_shape_Alow_HB0;
  TH1F* h_shape0_Alow_HB0;
  TH1F* h_shape_Ahigh_HB1;
  TH1F* h_shape0_Ahigh_HB1;
  TH1F* h_shape_Alow_HB1;
  TH1F* h_shape0_Alow_HB1;
  TH1F* h_shape_Ahigh_HB2;
  TH1F* h_shape0_Ahigh_HB2;
  TH1F* h_shape_Alow_HB2;
  TH1F* h_shape0_Alow_HB2;
  TH1F* h_shape_Ahigh_HB3;
  TH1F* h_shape0_Ahigh_HB3;
  TH1F* h_shape_Alow_HB3;
  TH1F* h_shape0_Alow_HB3;
  TH1F* h_shape_bad_channels_HB;
  TH1F* h_shape0_bad_channels_HB;
  TH1F* h_shape_good_channels_HB;
  TH1F* h_shape0_good_channels_HB;
  TH1F* h_shape_bad_channels_HE;
  TH1F* h_shape0_bad_channels_HE;
  TH1F* h_shape_good_channels_HE;
  TH1F* h_shape0_good_channels_HE;
  TH1F* h_shape_bad_channels_HF;
  TH1F* h_shape0_bad_channels_HF;
  TH1F* h_shape_good_channels_HF;
  TH1F* h_shape0_good_channels_HF;
  TH1F* h_shape_bad_channels_HO;
  TH1F* h_shape0_bad_channels_HO;
  TH1F* h_shape_good_channels_HO;
  TH1F* h_shape0_good_channels_HO;
  TH1F* h_sumamplitude_depth1_HB;
  TH1F* h_sumamplitude_depth2_HB;
  TH1F* h_sumamplitude_depth1_HE;
  TH1F* h_sumamplitude_depth2_HE;
  TH1F* h_sumamplitude_depth3_HE;
  TH1F* h_sumamplitude_depth1_HF;
  TH1F* h_sumamplitude_depth2_HF;
  TH1F* h_sumamplitude_depth4_HO;
  TH1F* h_sumamplitude_depth1_HB0;
  TH1F* h_sumamplitude_depth2_HB0;
  TH1F* h_sumamplitude_depth1_HE0;
  TH1F* h_sumamplitude_depth2_HE0;
  TH1F* h_sumamplitude_depth3_HE0;
  TH1F* h_sumamplitude_depth1_HF0;
  TH1F* h_sumamplitude_depth2_HF0;
  TH1F* h_sumamplitude_depth4_HO0;
  TH1F* h_sumamplitude_depth1_HB1;
  TH1F* h_sumamplitude_depth2_HB1;
  TH1F* h_sumamplitude_depth1_HE1;
  TH1F* h_sumamplitude_depth2_HE1;
  TH1F* h_sumamplitude_depth3_HE1;
  TH1F* h_sumamplitude_depth1_HF1;
  TH1F* h_sumamplitude_depth2_HF1;
  TH1F* h_sumamplitude_depth4_HO1;
  TH2F* h_mapDepth1Ped0_HB;
  TH2F* h_mapDepth1Ped1_HB;
  TH2F* h_mapDepth1Ped2_HB;
  TH2F* h_mapDepth1Ped3_HB;
  TH2F* h_mapDepth1Pedw0_HB;
  TH2F* h_mapDepth1Pedw1_HB;
  TH2F* h_mapDepth1Pedw2_HB;
  TH2F* h_mapDepth1Pedw3_HB;
  TH2F* h_mapDepth2Ped0_HB;
  TH2F* h_mapDepth2Ped1_HB;
  TH2F* h_mapDepth2Ped2_HB;
  TH2F* h_mapDepth2Ped3_HB;
  TH2F* h_mapDepth2Pedw0_HB;
  TH2F* h_mapDepth2Pedw1_HB;
  TH2F* h_mapDepth2Pedw2_HB;
  TH2F* h_mapDepth2Pedw3_HB;
  TH2F* h_mapDepth1Ped0_HE;
  TH2F* h_mapDepth1Ped1_HE;
  TH2F* h_mapDepth1Ped2_HE;
  TH2F* h_mapDepth1Ped3_HE;
  TH2F* h_mapDepth1Pedw0_HE;
  TH2F* h_mapDepth1Pedw1_HE;
  TH2F* h_mapDepth1Pedw2_HE;
  TH2F* h_mapDepth1Pedw3_HE;
  TH2F* h_mapDepth2Ped0_HE;
  TH2F* h_mapDepth2Ped1_HE;
  TH2F* h_mapDepth2Ped2_HE;
  TH2F* h_mapDepth2Ped3_HE;
  TH2F* h_mapDepth2Pedw0_HE;
  TH2F* h_mapDepth2Pedw1_HE;
  TH2F* h_mapDepth2Pedw2_HE;
  TH2F* h_mapDepth2Pedw3_HE;
  TH2F* h_mapDepth3Ped0_HE;
  TH2F* h_mapDepth3Ped1_HE;
  TH2F* h_mapDepth3Ped2_HE;
  TH2F* h_mapDepth3Ped3_HE;
  TH2F* h_mapDepth3Pedw0_HE;
  TH2F* h_mapDepth3Pedw1_HE;
  TH2F* h_mapDepth3Pedw2_HE;
  TH2F* h_mapDepth3Pedw3_HE;
  TH2F* h_mapDepth1Ped0_HF;
  TH2F* h_mapDepth1Ped1_HF;
  TH2F* h_mapDepth1Ped2_HF;
  TH2F* h_mapDepth1Ped3_HF;
  TH2F* h_mapDepth1Pedw0_HF;
  TH2F* h_mapDepth1Pedw1_HF;
  TH2F* h_mapDepth1Pedw2_HF;
  TH2F* h_mapDepth1Pedw3_HF;
  TH2F* h_mapDepth2Ped0_HF;
  TH2F* h_mapDepth2Ped1_HF;
  TH2F* h_mapDepth2Ped2_HF;
  TH2F* h_mapDepth2Ped3_HF;
  TH2F* h_mapDepth2Pedw0_HF;
  TH2F* h_mapDepth2Pedw1_HF;
  TH2F* h_mapDepth2Pedw2_HF;
  TH2F* h_mapDepth2Pedw3_HF;
  TH2F* h_mapDepth4Ped0_HO;
  TH2F* h_mapDepth4Ped1_HO;
  TH2F* h_mapDepth4Ped2_HO;
  TH2F* h_mapDepth4Ped3_HO;
  TH2F* h_mapDepth4Pedw0_HO;
  TH2F* h_mapDepth4Pedw1_HO;
  TH2F* h_mapDepth4Pedw2_HO;
  TH2F* h_mapDepth4Pedw3_HO;
  TH2F* h_mapGetRMSOverNormalizedSignal_HB;
  TH2F* h_mapGetRMSOverNormalizedSignal0_HB;
  TH2F* h_mapGetRMSOverNormalizedSignal_HE;
  TH2F* h_mapGetRMSOverNormalizedSignal0_HE;
  TH2F* h_mapGetRMSOverNormalizedSignal_HF;
  TH2F* h_mapGetRMSOverNormalizedSignal0_HF;
  TH2F* h_mapGetRMSOverNormalizedSignal_HO;
  TH2F* h_mapGetRMSOverNormalizedSignal0_HO;
  TH2F* h_2D0sumErrorBLS1;
  TH2F* h_2DsumErrorBLS1;
  TH1F* h_sumErrorBLS1;
  TH1F* h_sumErrorBperLS1;
  TH1F* h_sum0ErrorBperLS1;
  TH2F* h_2D0sumErrorBLS2;
  TH2F* h_2DsumErrorBLS2;
  TH1F* h_sumErrorBLS2;
  TH1F* h_sumErrorBperLS2;
  TH1F* h_sum0ErrorBperLS2;
  TH2F* h_2D0sumErrorBLS3;
  TH2F* h_2DsumErrorBLS3;
  TH1F* h_sumErrorBLS3;
  TH1F* h_sumErrorBperLS3;
  TH1F* h_sum0ErrorBperLS3;
  TH2F* h_2D0sumErrorBLS4;
  TH2F* h_2DsumErrorBLS4;
  TH1F* h_sumErrorBLS4;
  TH1F* h_sumErrorBperLS4;
  TH1F* h_sum0ErrorBperLS4;
  TH2F* h_2D0sumErrorBLS5;
  TH2F* h_2DsumErrorBLS5;
  TH1F* h_sumErrorBLS5;
  TH1F* h_sumErrorBperLS5;
  TH1F* h_sum0ErrorBperLS5;
  TH2F* h_2D0sumErrorBLS6;
  TH2F* h_2DsumErrorBLS6;
  TH1F* h_sumErrorBLS6;
  TH1F* h_sumErrorBperLS6;
  TH1F* h_sum0ErrorBperLS6;
  TH2F* h_2D0sumErrorBLS7;
  TH2F* h_2DsumErrorBLS7;
  TH1F* h_sumErrorBLS7;
  TH1F* h_sumErrorBperLS7;
  TH1F* h_sum0ErrorBperLS7;
  TH2F* h_2D0sumErrorBLS8;
  TH2F* h_2DsumErrorBLS8;
  TH1F* h_sumErrorBLS8;
  TH1F* h_sumErrorBperLS8;
  TH1F* h_sum0ErrorBperLS8;
  TH1F* h_averSIGNALoccupancy_HB;
  TH1F* h_averSIGNALoccupancy_HE;
  TH1F* h_averSIGNALoccupancy_HF;
  TH1F* h_averSIGNALoccupancy_HO;
  TH1F* h_averSIGNALsumamplitude_HB;
  TH1F* h_averSIGNALsumamplitude_HE;
  TH1F* h_averSIGNALsumamplitude_HF;
  TH1F* h_averSIGNALsumamplitude_HO;
  TH1F* h_averNOSIGNALoccupancy_HB;
  TH1F* h_averNOSIGNALoccupancy_HE;
  TH1F* h_averNOSIGNALoccupancy_HF;
  TH1F* h_averNOSIGNALoccupancy_HO;
  TH1F* h_averNOSIGNALsumamplitude_HB;
  TH1F* h_averNOSIGNALsumamplitude_HE;
  TH1F* h_averNOSIGNALsumamplitude_HF;
  TH1F* h_averNOSIGNALsumamplitude_HO;
  TH1F* h_maxxSUMAmpl_HB;
  TH1F* h_maxxSUMAmpl_HE;
  TH1F* h_maxxSUMAmpl_HF;
  TH1F* h_maxxSUMAmpl_HO;
  TH1F* h_maxxOCCUP_HB;
  TH1F* h_maxxOCCUP_HE;
  TH1F* h_maxxOCCUP_HF;
  TH1F* h_maxxOCCUP_HO;
  TH1F* h_sumamplitudechannel_HB;
  TH1F* h_sumamplitudechannel_HE;
  TH1F* h_sumamplitudechannel_HF;
  TH1F* h_sumamplitudechannel_HO;
  TH1F* h_eventamplitude_HB;
  TH1F* h_eventamplitude_HE;
  TH1F* h_eventamplitude_HF;
  TH1F* h_eventamplitude_HO;
  TH1F* h_eventoccupancy_HB;
  TH1F* h_eventoccupancy_HE;
  TH1F* h_eventoccupancy_HF;
  TH1F* h_eventoccupancy_HO;
  TH2F* h_2DAtaildepth1_HB;
  TH2F* h_2D0Ataildepth1_HB;
  TH2F* h_2DAtaildepth2_HB;
  TH2F* h_2D0Ataildepth2_HB;
  TH2F* h_mapenophinorm_HE1;
  TH2F* h_mapenophinorm_HE2;
  TH2F* h_mapenophinorm_HE3;
  TH2F* h_mapenophinorm_HE4;
  TH2F* h_mapenophinorm_HE5;
  TH2F* h_mapenophinorm_HE6;
  TH2F* h_mapenophinorm_HE7;
  TH2F* h_mapenophinorm2_HE1;
  TH2F* h_mapenophinorm2_HE2;
  TH2F* h_mapenophinorm2_HE3;
  TH2F* h_mapenophinorm2_HE4;
  TH2F* h_mapenophinorm2_HE5;
  TH2F* h_mapenophinorm2_HE6;
  TH2F* h_mapenophinorm2_HE7;
  TH2F* h_maprphinorm_HE1;
  TH2F* h_maprphinorm_HE2;
  TH2F* h_maprphinorm_HE3;
  TH2F* h_maprphinorm_HE4;
  TH2F* h_maprphinorm_HE5;
  TH2F* h_maprphinorm_HE6;
  TH2F* h_maprphinorm_HE7;
  TH2F* h_maprphinorm2_HE1;
  TH2F* h_maprphinorm2_HE2;
  TH2F* h_maprphinorm2_HE3;
  TH2F* h_maprphinorm2_HE4;
  TH2F* h_maprphinorm2_HE5;
  TH2F* h_maprphinorm2_HE6;
  TH2F* h_maprphinorm2_HE7;
  TH2F* h_maprphinorm0_HE1;
  TH2F* h_maprphinorm0_HE2;
  TH2F* h_maprphinorm0_HE3;
  TH2F* h_maprphinorm0_HE4;
  TH2F* h_maprphinorm0_HE5;
  TH2F* h_maprphinorm0_HE6;
  TH2F* h_maprphinorm0_HE7;
  TH2F* h_mapDepth1RADDAM_HE;
  TH2F* h_mapDepth2RADDAM_HE;
  TH2F* h_mapDepth3RADDAM_HE;
  TH1F* h_sigLayer1RADDAM_HE;
  TH1F* h_sigLayer2RADDAM_HE;
  TH1F* h_sigLayer1RADDAM0_HE;
  TH1F* h_sigLayer2RADDAM0_HE;
  TH1F* h_mapDepth3RADDAM16_HE;
  TH2F* h_mapDepth1RADDAM0_HE;
  TH2F* h_mapDepth2RADDAM0_HE;
  TH2F* h_mapDepth3RADDAM0_HE;
  TH1F* h_AamplitudewithPedSubtr_RADDAM_HE;
  TH1F* h_AamplitudewithPedSubtr_RADDAM_HEzoom0;
  TH1F* h_AamplitudewithPedSubtr_RADDAM_HEzoom1;
  TH1F* h_A_Depth1RADDAM_HE;
  TH1F* h_A_Depth2RADDAM_HE;
  TH1F* h_A_Depth3RADDAM_HE;
  TH1F* h_sumphiEta16Depth3RADDAM_HED2;
  TH1F* h_Eta16Depth3RADDAM_HED2;
  TH1F* h_NphiForEta16Depth3RADDAM_HED2;
  TH1F* h_sumphiEta16Depth3RADDAM_HED2P;
  TH1F* h_Eta16Depth3RADDAM_HED2P;
  TH1F* h_NphiForEta16Depth3RADDAM_HED2P;
  TH1F* h_sumphiEta16Depth3RADDAM_HED2ALL;
  TH1F* h_Eta16Depth3RADDAM_HED2ALL;
  TH1F* h_NphiForEta16Depth3RADDAM_HED2ALL;
  TH1F* h_sigLayer1RADDAM5_HE;
  TH1F* h_sigLayer2RADDAM5_HE;
  TH1F* h_sigLayer1RADDAM6_HE;
  TH1F* h_sigLayer2RADDAM6_HE;
  TH1F* h_sigLayer1RADDAM5_HED2;
  TH1F* h_sigLayer1RADDAM6_HED2;
  TH1F* h_sigLayer2RADDAM5_HED2;
  TH1F* h_sigLayer2RADDAM6_HED2;
  int calibcapiderror[ndepth][neta][nphi];
  float calibt[ndepth][neta][nphi];
  double caliba[ndepth][neta][nphi];
  double calibw[ndepth][neta][nphi];
  double calib0[ndepth][neta][nphi];
  double signal[ndepth][neta][nphi];
  double calib3[ndepth][neta][nphi];
  double signal3[ndepth][neta][nphi];
  double calib2[ndepth][neta][nphi];
  int badchannels[nsub][ndepth][neta][nphi];  // for upgrade
  double sumEstimator0[nsub][ndepth][neta][nphi];
  double sumEstimator1[nsub][ndepth][neta][nphi];
  double sumEstimator2[nsub][ndepth][neta][nphi];
  double sumEstimator3[nsub][ndepth][neta][nphi];
  double sumEstimator4[nsub][ndepth][neta][nphi];
  double sumEstimator5[nsub][ndepth][neta][nphi];
  double sumEstimator6[nsub][ndepth][neta][nphi];
  double sum0Estimator[nsub][ndepth][neta][nphi];
  double amplitudechannel[nsub][ndepth][neta][nphi];
  double tocamplchannel[nsub][ndepth][neta][nphi];
  double maprphinorm[nsub][ndepth][neta][nphi];
  float TS_data[100];
  float TS_cal[100];
  double mapRADDAM_HE[ndepth][neta][nphi];
  int mapRADDAM0_HE[ndepth][neta][nphi];
  double mapRADDAM_HED2[ndepth][neta];
  int mapRADDAM_HED20[ndepth][neta];
  float binanpfit = anpfit / npfit;
  long int gsmdepth1sipm[npfit][neta][nphi][ndepth];
  ///////////////////////////////////////////// end massives
  long int Nevent;
  int Run;
  int run0;
  int runcounter;
  int eventcounter;
  long int orbitNum;
  int bcn;
  int lumi;
  int ls0;
  int lscounter;
  int lscounterM1;
  int lscounter10;
  int nevcounter;
  int lscounterrun;
  int lscounterrun10;
  int nevcounter0;
  int nevcounter00;
  double averSIGNALoccupancy_HB;
  double averSIGNALoccupancy_HE;
  double averSIGNALoccupancy_HF;
  double averSIGNALoccupancy_HO;
  double averSIGNALsumamplitude_HB;
  double averSIGNALsumamplitude_HE;
  double averSIGNALsumamplitude_HF;
  double averSIGNALsumamplitude_HO;
  double averNOSIGNALoccupancy_HB;
  double averNOSIGNALoccupancy_HE;
  double averNOSIGNALoccupancy_HF;
  double averNOSIGNALoccupancy_HO;
  double averNOSIGNALsumamplitude_HB;
  double averNOSIGNALsumamplitude_HE;
  double averNOSIGNALsumamplitude_HF;
  double averNOSIGNALsumamplitude_HO;
  double maxxSUM1;
  double maxxSUM2;
  double maxxSUM3;
  double maxxSUM4;
  double maxxOCCUP1;
  double maxxOCCUP2;
  double maxxOCCUP3;
  double maxxOCCUP4;
  TTree* myTree;
  TFile* hOutputFile;
  ofstream MAPfile;
  /////////////////////////////////////////
  // Get RBX number from 1-35 for Calibration box
  int getRBX(int& i, int& j, int& k);
  void fillDigiErrors(HBHEDigiCollection::const_iterator& digiItr);
  void fillDigiErrorsHF(HFDigiCollection::const_iterator& digiItr);
  void fillDigiErrorsHO(HODigiCollection::const_iterator& digiItr);
  // upgrade:
  void fillDigiErrorsHFQIE10(QIE10DataFrame qie10df);
  void fillDigiErrorsQIE11(QIE11DataFrame qie11df);
  void fillDigiAmplitude(HBHEDigiCollection::const_iterator& digiItr);
  void fillDigiAmplitudeHF(HFDigiCollection::const_iterator& digiItr);
  void fillDigiAmplitudeHO(HODigiCollection::const_iterator& digiItr);
  // upgrade:
  void fillDigiAmplitudeHFQIE10(QIE10DataFrame qie10df);
  void fillDigiAmplitudeQIE11(QIE11DataFrame qie11df);
  int local_event;
  int eta, phi, depth, nTS, cap_num;
  int numOfTS;
  int numOfLaserEv;
  int testmetka;
  float alsmin;
  float blsmax;
  int nlsminmax;
};

/////////////////////////////// -------------------------------------------------------------------

void CMTRawAnalyzer::endJob() {
  if (verbosity > 0)
    std::cout << "==============    endJob  ===================================" << std::endl;

  std::cout << " --------------------------------------- " << std::endl;
  std::cout << " for Run = " << run0 << " with runcounter = " << runcounter << " #ev = " << eventcounter << std::endl;
  std::cout << " #LS =  " << lscounterrun << " #LS10 =  " << lscounterrun10 << " Last LS =  " << ls0 << std::endl;
  std::cout << " --------------------------------------------- " << std::endl;
  h_nls_per_run->Fill(float(lscounterrun));
  h_nls_per_run10->Fill(float(lscounterrun10));

  hOutputFile->SetCompressionLevel(2);

  hOutputFile->Write();
  hOutputFile->cd();

  if (recordNtuples_)
    myTree->Write();

  ///////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "===== full number of events =  " << nevent << std::endl;
  std::cout << "===== possible max number of events in ntuple(each 50th recorded) =  " << nevent50 << std::endl;
  std::cout << "===== but limited by maxNeventsInNtuple  =  " << maxNeventsInNtuple_ << std::endl;
  std::cout << "===== full number of events*HBHEdigis and qie11 =  " << nnnnnnhbhe << "   and  " << nnnnnnhbheqie11
            << std::endl;
  std::cout << "===== full number of events*HBHEdigis =  " << nnnnnn << std::endl;
  std::cout << "===== full number of events*HFdigis and qie10 =  " << counterhf << "   and  " << counterhfqie10
            << std::endl;
  std::cout << "===== full number of events*HOdigis =  " << counterho << std::endl;

  std::cout << "===== Start writing user histograms =====" << std::endl;
  //////////////////////////////////////////////////////////////////////   scaling of some histoes:
  ///////////////////////////////////////////->Write();
  h_bcnvsamplitude_HB->Write();
  h_bcnvsamplitude0_HB->Write();
  h_bcnvsamplitude_HE->Write();
  h_bcnvsamplitude0_HE->Write();
  h_bcnvsamplitude_HF->Write();
  h_bcnvsamplitude0_HF->Write();
  h_bcnvsamplitude_HO->Write();
  h_bcnvsamplitude0_HO->Write();

  h_orbitNumvsamplitude_HB->Write();
  h_orbitNumvsamplitude0_HB->Write();
  h_orbitNumvsamplitude_HE->Write();
  h_orbitNumvsamplitude0_HE->Write();
  h_orbitNumvsamplitude_HF->Write();
  h_orbitNumvsamplitude0_HF->Write();
  h_orbitNumvsamplitude_HO->Write();
  h_orbitNumvsamplitude0_HO->Write();

  h_2DsumADCAmplEtaPhiLs0->Write();
  h_2DsumADCAmplEtaPhiLs1->Write();
  h_2DsumADCAmplEtaPhiLs2->Write();
  h_2DsumADCAmplEtaPhiLs3->Write();

  h_2DsumADCAmplEtaPhiLs00->Write();
  h_2DsumADCAmplEtaPhiLs10->Write();
  h_2DsumADCAmplEtaPhiLs20->Write();
  h_2DsumADCAmplEtaPhiLs30->Write();

  h_sumADCAmplEtaPhiLs->Write();
  h_sumADCAmplEtaPhiLs_bbbc->Write();
  h_sumADCAmplEtaPhiLs_bbb1->Write();
  h_sumADCAmplEtaPhiLs_lscounterM1->Write();
  h_sumADCAmplEtaPhiLs_ietaphi->Write();
  h_sumADCAmplEtaPhiLs_lscounterM1orbitNum->Write();
  h_sumADCAmplEtaPhiLs_orbitNum->Write();

  if (recordHistoes_) {
    h_errorGeneral->Write();
    h_error1->Write();
    h_error2->Write();
    h_error3->Write();
    h_amplError->Write();
    h_amplFine->Write();

    h_errorGeneral_HB->Write();
    h_error1_HB->Write();
    h_error2_HB->Write();
    h_error3_HB->Write();
    h_error4_HB->Write();
    h_error5_HB->Write();
    h_error6_HB->Write();
    h_error7_HB->Write();
    h_amplError_HB->Write();
    h_amplFine_HB->Write();
    h_mapDepth1Error_HB->Write();
    h_mapDepth2Error_HB->Write();
    h_mapDepth3Error_HB->Write();
    h_mapDepth4Error_HB->Write();
    h_fiber0_HB->Write();
    h_fiber1_HB->Write();
    h_fiber2_HB->Write();
    h_repetedcapid_HB->Write();

    h_errorGeneral_HE->Write();
    h_error1_HE->Write();
    h_error2_HE->Write();
    h_error3_HE->Write();
    h_error4_HE->Write();
    h_error5_HE->Write();
    h_error6_HE->Write();
    h_error7_HE->Write();
    h_amplError_HE->Write();
    h_amplFine_HE->Write();
    h_mapDepth1Error_HE->Write();
    h_mapDepth2Error_HE->Write();
    h_mapDepth3Error_HE->Write();
    h_mapDepth4Error_HE->Write();
    h_mapDepth5Error_HE->Write();
    h_mapDepth6Error_HE->Write();
    h_mapDepth7Error_HE->Write();
    h_fiber0_HE->Write();
    h_fiber1_HE->Write();
    h_fiber2_HE->Write();
    h_repetedcapid_HE->Write();

    h_errorGeneral_HF->Write();
    h_error1_HF->Write();
    h_error2_HF->Write();
    h_error3_HF->Write();
    h_error4_HF->Write();
    h_error5_HF->Write();
    h_error6_HF->Write();
    h_error7_HF->Write();
    h_amplError_HF->Write();
    h_amplFine_HF->Write();
    h_mapDepth1Error_HF->Write();
    h_mapDepth2Error_HF->Write();
    h_mapDepth3Error_HF->Write();
    h_mapDepth4Error_HF->Write();
    h_fiber0_HF->Write();
    h_fiber1_HF->Write();
    h_fiber2_HF->Write();
    h_repetedcapid_HF->Write();

    h_errorGeneral_HO->Write();
    h_error1_HO->Write();
    h_error2_HO->Write();
    h_error3_HO->Write();
    h_error4_HO->Write();
    h_error5_HO->Write();
    h_error6_HO->Write();
    h_error7_HO->Write();
    h_amplError_HO->Write();
    h_amplFine_HO->Write();
    h_mapDepth4Error_HO->Write();
    h_fiber0_HO->Write();
    h_fiber1_HO->Write();
    h_fiber2_HO->Write();
    h_repetedcapid_HO->Write();

    ///////////////////////
    h_numberofhitsHBtest->Write();
    h_numberofhitsHEtest->Write();
    h_numberofhitsHFtest->Write();
    h_numberofhitsHOtest->Write();
    h_AmplitudeHBtest->Write();
    h_AmplitudeHBtest1->Write();
    h_AmplitudeHBtest6->Write();
    h_AmplitudeHEtest->Write();
    h_AmplitudeHEtest1->Write();
    h_AmplitudeHEtest6->Write();
    h_AmplitudeHFtest->Write();
    h_AmplitudeHOtest->Write();
    h_totalAmplitudeHB->Write();
    h_totalAmplitudeHE->Write();
    h_totalAmplitudeHF->Write();
    h_totalAmplitudeHO->Write();
    h_totalAmplitudeHBperEvent->Write();
    h_totalAmplitudeHEperEvent->Write();
    h_totalAmplitudeHFperEvent->Write();
    h_totalAmplitudeHOperEvent->Write();

    h_ADCAmpl345Zoom_HB->Write();
    h_ADCAmpl345Zoom1_HB->Write();
    h_ADCAmpl345_HB->Write();

    h_ADCAmpl_HBCapIdNoError->Write();
    h_ADCAmpl345_HBCapIdNoError->Write();
    h_ADCAmpl_HBCapIdError->Write();
    h_ADCAmpl345_HBCapIdError->Write();

    h_ADCAmplZoom_HB->Write();
    h_ADCAmplZoom1_HB->Write();
    h_ADCAmpl_HB->Write();

    h_AmplitudeHBrest->Write();
    h_AmplitudeHBrest1->Write();
    h_AmplitudeHBrest6->Write();

    h_mapDepth1ADCAmpl225_HB->Write();
    h_mapDepth2ADCAmpl225_HB->Write();
    h_mapDepth1ADCAmpl_HB->Write();
    h_mapDepth2ADCAmpl_HB->Write();
    h_mapDepth1ADCAmpl12_HB->Write();
    h_mapDepth2ADCAmpl12_HB->Write();

    h_mapDepth3ADCAmpl225_HB->Write();
    h_mapDepth4ADCAmpl225_HB->Write();
    h_mapDepth3ADCAmpl_HB->Write();
    h_mapDepth4ADCAmpl_HB->Write();
    h_mapDepth3ADCAmpl12_HB->Write();
    h_mapDepth4ADCAmpl12_HB->Write();

    h_TSmeanA_HB->Write();
    h_mapDepth1TSmeanA225_HB->Write();
    h_mapDepth2TSmeanA225_HB->Write();
    h_mapDepth1TSmeanA_HB->Write();
    h_mapDepth2TSmeanA_HB->Write();
    h_mapDepth3TSmeanA225_HB->Write();
    h_mapDepth4TSmeanA225_HB->Write();
    h_mapDepth3TSmeanA_HB->Write();
    h_mapDepth4TSmeanA_HB->Write();

    h_TSmaxA_HB->Write();
    h_mapDepth1TSmaxA225_HB->Write();
    h_mapDepth2TSmaxA225_HB->Write();
    h_mapDepth1TSmaxA_HB->Write();
    h_mapDepth2TSmaxA_HB->Write();
    h_mapDepth3TSmaxA225_HB->Write();
    h_mapDepth4TSmaxA225_HB->Write();
    h_mapDepth3TSmaxA_HB->Write();
    h_mapDepth4TSmaxA_HB->Write();

    h_Amplitude_HB->Write();
    h_mapDepth1Amplitude225_HB->Write();
    h_mapDepth2Amplitude225_HB->Write();
    h_mapDepth1Amplitude_HB->Write();
    h_mapDepth2Amplitude_HB->Write();
    h_mapDepth3Amplitude225_HB->Write();
    h_mapDepth4Amplitude225_HB->Write();
    h_mapDepth3Amplitude_HB->Write();
    h_mapDepth4Amplitude_HB->Write();

    h_Ampl_HB->Write();
    h_mapDepth1Ampl047_HB->Write();
    h_mapDepth2Ampl047_HB->Write();
    h_mapDepth1Ampl_HB->Write();
    h_mapDepth2Ampl_HB->Write();
    h_mapDepth1AmplE34_HB->Write();
    h_mapDepth2AmplE34_HB->Write();
    h_mapDepth1_HB->Write();
    h_mapDepth2_HB->Write();
    h_mapDepth3Ampl047_HB->Write();
    h_mapDepth4Ampl047_HB->Write();
    h_mapDepth3Ampl_HB->Write();
    h_mapDepth4Ampl_HB->Write();
    h_mapDepth3AmplE34_HB->Write();
    h_mapDepth4AmplE34_HB->Write();
    h_mapDepth3_HB->Write();
    h_mapDepth4_HB->Write();

    h_mapDepth1ADCAmpl225Copy_HB->Write();
    h_mapDepth2ADCAmpl225Copy_HB->Write();
    h_mapDepth3ADCAmpl225Copy_HB->Write();
    h_mapDepth4ADCAmpl225Copy_HB->Write();
    h_mapDepth1ADCAmpl225Copy_HE->Write();
    h_mapDepth2ADCAmpl225Copy_HE->Write();
    h_mapDepth3ADCAmpl225Copy_HE->Write();
    h_mapDepth4ADCAmpl225Copy_HE->Write();
    h_mapDepth5ADCAmpl225Copy_HE->Write();
    h_mapDepth6ADCAmpl225Copy_HE->Write();
    h_mapDepth7ADCAmpl225Copy_HE->Write();
    h_mapDepth1ADCAmpl225Copy_HF->Write();
    h_mapDepth2ADCAmpl225Copy_HF->Write();
    h_mapDepth3ADCAmpl225Copy_HF->Write();
    h_mapDepth4ADCAmpl225Copy_HF->Write();
    h_mapDepth4ADCAmpl225Copy_HO->Write();
    ///////////////////////
    h_ADCAmpl_HF->Write();
    h_ADCAmplrest1_HF->Write();
    h_ADCAmplrest6_HF->Write();

    h_ADCAmplZoom1_HF->Write();
    h_mapDepth1ADCAmpl225_HF->Write();
    h_mapDepth2ADCAmpl225_HF->Write();
    h_mapDepth1ADCAmpl_HF->Write();
    h_mapDepth2ADCAmpl_HF->Write();
    h_mapDepth1ADCAmpl12_HF->Write();
    h_mapDepth2ADCAmpl12_HF->Write();
    h_mapDepth3ADCAmpl225_HF->Write();
    h_mapDepth4ADCAmpl225_HF->Write();
    h_mapDepth3ADCAmpl_HF->Write();
    h_mapDepth4ADCAmpl_HF->Write();
    h_mapDepth3ADCAmpl12_HF->Write();
    h_mapDepth4ADCAmpl12_HF->Write();

    h_TSmeanA_HF->Write();
    h_mapDepth1TSmeanA225_HF->Write();
    h_mapDepth2TSmeanA225_HF->Write();
    h_mapDepth1TSmeanA_HF->Write();
    h_mapDepth2TSmeanA_HF->Write();
    h_mapDepth3TSmeanA225_HF->Write();
    h_mapDepth4TSmeanA225_HF->Write();
    h_mapDepth3TSmeanA_HF->Write();
    h_mapDepth4TSmeanA_HF->Write();

    h_TSmaxA_HF->Write();
    h_mapDepth1TSmaxA225_HF->Write();
    h_mapDepth2TSmaxA225_HF->Write();
    h_mapDepth1TSmaxA_HF->Write();
    h_mapDepth2TSmaxA_HF->Write();
    h_mapDepth3TSmaxA225_HF->Write();
    h_mapDepth4TSmaxA225_HF->Write();
    h_mapDepth3TSmaxA_HF->Write();
    h_mapDepth4TSmaxA_HF->Write();

    h_Amplitude_HF->Write();
    h_mapDepth1Amplitude225_HF->Write();
    h_mapDepth2Amplitude225_HF->Write();
    h_mapDepth1Amplitude_HF->Write();
    h_mapDepth2Amplitude_HF->Write();
    h_mapDepth3Amplitude225_HF->Write();
    h_mapDepth4Amplitude225_HF->Write();
    h_mapDepth3Amplitude_HF->Write();
    h_mapDepth4Amplitude_HF->Write();

    h_Ampl_HF->Write();
    h_mapDepth1Ampl047_HF->Write();
    h_mapDepth2Ampl047_HF->Write();
    h_mapDepth3Ampl047_HF->Write();
    h_mapDepth4Ampl047_HF->Write();

    h_mapDepth1Ampl_HF->Write();
    h_mapDepth2Ampl_HF->Write();
    h_mapDepth1AmplE34_HF->Write();
    h_mapDepth2AmplE34_HF->Write();
    h_mapDepth1_HF->Write();
    h_mapDepth2_HF->Write();
    h_mapDepth3Ampl_HF->Write();
    h_mapDepth4Ampl_HF->Write();
    h_mapDepth3AmplE34_HF->Write();
    h_mapDepth4AmplE34_HF->Write();
    h_mapDepth3_HF->Write();
    h_mapDepth4_HF->Write();

    ///////////////////////
    h_ADCAmpl_HO->Write();
    h_ADCAmplrest1_HO->Write();
    h_ADCAmplrest6_HO->Write();

    h_ADCAmplZoom1_HO->Write();
    h_ADCAmpl_HO_copy->Write();
    h_mapDepth4ADCAmpl225_HO->Write();
    h_mapDepth4ADCAmpl_HO->Write();
    h_mapDepth4ADCAmpl12_HO->Write();

    h_TSmeanA_HO->Write();
    h_mapDepth4TSmeanA225_HO->Write();
    h_mapDepth4TSmeanA_HO->Write();

    h_TSmaxA_HO->Write();
    h_mapDepth4TSmaxA225_HO->Write();
    h_mapDepth4TSmaxA_HO->Write();

    h_Amplitude_HO->Write();
    h_mapDepth4Amplitude225_HO->Write();
    h_mapDepth4Amplitude_HO->Write();
    h_Ampl_HO->Write();
    h_mapDepth4Ampl047_HO->Write();
    h_mapDepth4Ampl_HO->Write();
    h_mapDepth4AmplE34_HO->Write();
    h_mapDepth4_HO->Write();

    //////////////////////////////////////////

    h_ADCAmpl345Zoom_HE->Write();
    h_ADCAmpl345Zoom1_HE->Write();
    h_ADCAmpl345_HE->Write();
    h_ADCAmpl_HE->Write();
    h_ADCAmplrest_HE->Write();
    h_ADCAmplrest1_HE->Write();
    h_ADCAmplrest6_HE->Write();

    h_ADCAmplZoom1_HE->Write();

    h_corrforxaMAIN_HE->Write();
    h_corrforxaMAIN0_HE->Write();
    h_corrforxaADDI_HE->Write();
    h_corrforxaADDI0_HE->Write();

    h_mapDepth1ADCAmpl225_HE->Write();
    h_mapDepth2ADCAmpl225_HE->Write();
    h_mapDepth3ADCAmpl225_HE->Write();
    h_mapDepth4ADCAmpl225_HE->Write();
    h_mapDepth5ADCAmpl225_HE->Write();
    h_mapDepth6ADCAmpl225_HE->Write();
    h_mapDepth7ADCAmpl225_HE->Write();

    h_mapADCAmplfirstpeak_HE->Write();
    h_mapADCAmplfirstpeak0_HE->Write();
    h_mapADCAmplsecondpeak_HE->Write();
    h_mapADCAmplsecondpeak0_HE->Write();
    h_mapADCAmpl11firstpeak_HE->Write();
    h_mapADCAmpl11firstpeak0_HE->Write();
    h_mapADCAmpl11secondpeak_HE->Write();
    h_mapADCAmpl11secondpeak0_HE->Write();
    h_mapADCAmpl12firstpeak_HE->Write();
    h_mapADCAmpl12firstpeak0_HE->Write();
    h_mapADCAmpl12secondpeak_HE->Write();
    h_mapADCAmpl12secondpeak0_HE->Write();

    h_gsmdifferencefit1_HE->Write();
    h_gsmdifferencefit2_HE->Write();
    h_gsmdifferencefit3_HE->Write();
    h_gsmdifferencefit4_HE->Write();
    h_gsmdifferencefit5_HE->Write();
    h_gsmdifferencefit6_HE->Write();

    h_mapDepth1ADCAmpl_HE->Write();
    h_mapDepth2ADCAmpl_HE->Write();
    h_mapDepth3ADCAmpl_HE->Write();
    h_mapDepth4ADCAmpl_HE->Write();
    h_mapDepth5ADCAmpl_HE->Write();
    h_mapDepth6ADCAmpl_HE->Write();
    h_mapDepth7ADCAmpl_HE->Write();
    h_mapDepth1ADCAmpl12_HE->Write();
    h_mapDepth2ADCAmpl12_HE->Write();
    h_mapDepth3ADCAmpl12_HE->Write();
    h_mapDepth4ADCAmpl12_HE->Write();
    h_mapDepth5ADCAmpl12_HE->Write();
    h_mapDepth6ADCAmpl12_HE->Write();
    h_mapDepth7ADCAmpl12_HE->Write();

    h_mapDepth1ADCAmplSiPM_HE->Write();
    h_mapDepth2ADCAmplSiPM_HE->Write();
    h_mapDepth3ADCAmplSiPM_HE->Write();
    h_mapDepth1ADCAmpl12SiPM_HE->Write();
    h_mapDepth2ADCAmpl12SiPM_HE->Write();
    h_mapDepth3ADCAmpl12SiPM_HE->Write();

    h_mapDepth1linADCAmpl12_HE->Write();
    h_mapDepth2linADCAmpl12_HE->Write();
    h_mapDepth3linADCAmpl12_HE->Write();

    h_TSmeanA_HE->Write();
    h_mapDepth1TSmeanA225_HE->Write();
    h_mapDepth2TSmeanA225_HE->Write();
    h_mapDepth3TSmeanA225_HE->Write();
    h_mapDepth4TSmeanA225_HE->Write();
    h_mapDepth5TSmeanA225_HE->Write();
    h_mapDepth6TSmeanA225_HE->Write();
    h_mapDepth7TSmeanA225_HE->Write();
    h_mapDepth1TSmeanA_HE->Write();
    h_mapDepth2TSmeanA_HE->Write();
    h_mapDepth3TSmeanA_HE->Write();
    h_mapDepth4TSmeanA_HE->Write();
    h_mapDepth5TSmeanA_HE->Write();
    h_mapDepth6TSmeanA_HE->Write();
    h_mapDepth7TSmeanA_HE->Write();

    h_TSmaxA_HE->Write();
    h_mapDepth1TSmaxA225_HE->Write();
    h_mapDepth2TSmaxA225_HE->Write();
    h_mapDepth3TSmaxA225_HE->Write();
    h_mapDepth4TSmaxA225_HE->Write();
    h_mapDepth5TSmaxA225_HE->Write();
    h_mapDepth6TSmaxA225_HE->Write();
    h_mapDepth7TSmaxA225_HE->Write();
    h_mapDepth1TSmaxA_HE->Write();
    h_mapDepth2TSmaxA_HE->Write();
    h_mapDepth3TSmaxA_HE->Write();
    h_mapDepth4TSmaxA_HE->Write();
    h_mapDepth5TSmaxA_HE->Write();
    h_mapDepth6TSmaxA_HE->Write();
    h_mapDepth7TSmaxA_HE->Write();

    h_Amplitude_HE->Write();
    h_mapDepth1Amplitude225_HE->Write();
    h_mapDepth2Amplitude225_HE->Write();
    h_mapDepth3Amplitude225_HE->Write();
    h_mapDepth4Amplitude225_HE->Write();
    h_mapDepth5Amplitude225_HE->Write();
    h_mapDepth6Amplitude225_HE->Write();
    h_mapDepth7Amplitude225_HE->Write();
    h_mapDepth1Amplitude_HE->Write();
    h_mapDepth2Amplitude_HE->Write();
    h_mapDepth3Amplitude_HE->Write();
    h_mapDepth4Amplitude_HE->Write();
    h_mapDepth5Amplitude_HE->Write();
    h_mapDepth6Amplitude_HE->Write();
    h_mapDepth7Amplitude_HE->Write();

    h_Ampl_HE->Write();
    h_mapDepth1Ampl047_HE->Write();
    h_mapDepth2Ampl047_HE->Write();
    h_mapDepth3Ampl047_HE->Write();
    h_mapDepth4Ampl047_HE->Write();
    h_mapDepth5Ampl047_HE->Write();
    h_mapDepth6Ampl047_HE->Write();
    h_mapDepth7Ampl047_HE->Write();
    h_mapDepth1Ampl_HE->Write();
    h_mapDepth2Ampl_HE->Write();
    h_mapDepth3Ampl_HE->Write();
    h_mapDepth4Ampl_HE->Write();
    h_mapDepth5Ampl_HE->Write();
    h_mapDepth6Ampl_HE->Write();
    h_mapDepth7Ampl_HE->Write();
    h_mapDepth1AmplE34_HE->Write();
    h_mapDepth2AmplE34_HE->Write();
    h_mapDepth3AmplE34_HE->Write();
    h_mapDepth4AmplE34_HE->Write();
    h_mapDepth5AmplE34_HE->Write();
    h_mapDepth6AmplE34_HE->Write();
    h_mapDepth7AmplE34_HE->Write();
    h_mapDepth1_HE->Write();
    h_mapDepth2_HE->Write();
    h_mapDepth3_HE->Write();
    h_mapDepth4_HE->Write();
    h_mapDepth5_HE->Write();
    h_mapDepth6_HE->Write();
    h_mapDepth7_HE->Write();

    ///////////////////////

    h_FullSignal3D_HB->Write();
    h_FullSignal3D0_HB->Write();
    h_FullSignal3D_HE->Write();
    h_FullSignal3D0_HE->Write();
    h_FullSignal3D_HO->Write();
    h_FullSignal3D0_HO->Write();
    h_FullSignal3D_HF->Write();
    h_FullSignal3D0_HF->Write();

    h_nbadchannels_depth1_HB->Write();
    h_runnbadchannels_depth1_HB->Write();
    h_runnbadchannelsC_depth1_HB->Write();
    h_runbadrate_depth1_HB->Write();
    h_runbadrateC_depth1_HB->Write();
    h_runbadrate0_depth1_HB->Write();

    h_nbadchannels_depth2_HB->Write();
    h_runnbadchannels_depth2_HB->Write();
    h_runnbadchannelsC_depth2_HB->Write();
    h_runbadrate_depth2_HB->Write();
    h_runbadrateC_depth2_HB->Write();
    h_runbadrate0_depth2_HB->Write();

    h_nbadchannels_depth1_HE->Write();
    h_runnbadchannels_depth1_HE->Write();
    h_runnbadchannelsC_depth1_HE->Write();
    h_runbadrate_depth1_HE->Write();
    h_runbadrateC_depth1_HE->Write();
    h_runbadrate0_depth1_HE->Write();

    h_nbadchannels_depth2_HE->Write();
    h_runnbadchannels_depth2_HE->Write();
    h_runnbadchannelsC_depth2_HE->Write();
    h_runbadrate_depth2_HE->Write();
    h_runbadrateC_depth2_HE->Write();
    h_runbadrate0_depth2_HE->Write();

    h_nbadchannels_depth3_HE->Write();
    h_runnbadchannels_depth3_HE->Write();
    h_runnbadchannelsC_depth3_HE->Write();
    h_runbadrate_depth3_HE->Write();
    h_runbadrateC_depth3_HE->Write();
    h_runbadrate0_depth3_HE->Write();

    h_nbadchannels_depth1_HF->Write();
    h_runnbadchannels_depth1_HF->Write();
    h_runnbadchannelsC_depth1_HF->Write();
    h_runbadrate_depth1_HF->Write();
    h_runbadrateC_depth1_HF->Write();
    h_runbadrate0_depth1_HF->Write();

    h_nbadchannels_depth2_HF->Write();
    h_runnbadchannels_depth2_HF->Write();
    h_runnbadchannelsC_depth2_HF->Write();
    h_runbadrate_depth2_HF->Write();
    h_runbadrateC_depth2_HF->Write();
    h_runbadrate0_depth2_HF->Write();

    h_nbadchannels_depth4_HO->Write();
    h_runnbadchannels_depth4_HO->Write();
    h_runnbadchannelsC_depth4_HO->Write();
    h_runbadrate_depth4_HO->Write();
    h_runbadrateC_depth4_HO->Write();
    h_runbadrate0_depth4_HO->Write();

    ///////////////////////
    h_mapCapCalib047_HB->Write();
    h_mapCapCalib047_HE->Write();
    h_mapCapCalib047_HO->Write();
    h_mapCapCalib047_HF->Write();

    h_ADCCalib_HB->Write();
    h_ADCCalib1_HB->Write();
    h_mapADCCalib047_HB->Write();
    h_mapADCCalib_HB->Write();
    h_RatioCalib_HB->Write();
    h_mapRatioCalib047_HB->Write();
    h_mapRatioCalib_HB->Write();
    h_TSmaxCalib_HB->Write();
    h_mapTSmaxCalib047_HB->Write();
    h_mapTSmaxCalib_HB->Write();
    h_TSmeanCalib_HB->Write();
    h_mapTSmeanCalib047_HB->Write();
    h_mapTSmeanCalib_HB->Write();
    h_WidthCalib_HB->Write();
    h_mapWidthCalib047_HB->Write();
    h_mapWidthCalib_HB->Write();
    h_map_HB->Write();
    h_ADCCalib_HE->Write();
    h_ADCCalib1_HE->Write();
    h_mapADCCalib047_HE->Write();
    h_mapADCCalib_HE->Write();
    h_RatioCalib_HE->Write();
    h_mapRatioCalib047_HE->Write();
    h_mapRatioCalib_HE->Write();
    h_TSmaxCalib_HE->Write();
    h_mapTSmaxCalib047_HE->Write();
    h_mapTSmaxCalib_HE->Write();
    h_TSmeanCalib_HE->Write();
    h_mapTSmeanCalib047_HE->Write();
    h_mapTSmeanCalib_HE->Write();
    h_WidthCalib_HE->Write();
    h_mapWidthCalib047_HE->Write();
    h_mapWidthCalib_HE->Write();
    h_map_HE->Write();
    h_ADCCalib_HO->Write();
    h_ADCCalib1_HO->Write();
    h_mapADCCalib047_HO->Write();
    h_mapADCCalib_HO->Write();
    h_RatioCalib_HO->Write();
    h_mapRatioCalib047_HO->Write();
    h_mapRatioCalib_HO->Write();
    h_TSmaxCalib_HO->Write();
    h_mapTSmaxCalib047_HO->Write();
    h_mapTSmaxCalib_HO->Write();
    h_TSmeanCalib_HO->Write();
    h_mapTSmeanCalib047_HO->Write();
    h_mapTSmeanCalib_HO->Write();
    h_WidthCalib_HO->Write();
    h_mapWidthCalib047_HO->Write();
    h_mapWidthCalib_HO->Write();
    h_map_HO->Write();
    h_ADCCalib_HF->Write();
    h_ADCCalib1_HF->Write();
    h_mapADCCalib047_HF->Write();
    h_mapADCCalib_HF->Write();
    h_RatioCalib_HF->Write();
    h_mapRatioCalib047_HF->Write();
    h_mapRatioCalib_HF->Write();
    h_TSmaxCalib_HF->Write();
    h_mapTSmaxCalib047_HF->Write();
    h_mapTSmaxCalib_HF->Write();
    h_TSmeanCalib_HF->Write();
    h_mapTSmeanCalib047_HF->Write();
    h_mapTSmeanCalib_HF->Write();
    h_WidthCalib_HF->Write();
    h_mapWidthCalib047_HF->Write();
    h_mapWidthCalib_HF->Write();
    h_map_HF->Write();

    h_nls_per_run->Write();
    h_nls_per_run10->Write();
    h_nevents_per_LS->Write();
    h_nevents_per_LSzoom->Write();
    h_nevents_per_eachRealLS->Write();
    h_nevents_per_eachLS->Write();
    h_lsnumber_per_eachLS->Write();

    // for estimator0:
    h_sumPedestalLS1->Write();
    h_2DsumPedestalLS1->Write();
    h_sumPedestalperLS1->Write();
    h_2D0sumPedestalLS1->Write();
    h_sum0PedestalperLS1->Write();

    h_sumPedestalLS2->Write();
    h_2DsumPedestalLS2->Write();
    h_sumPedestalperLS2->Write();
    h_2D0sumPedestalLS2->Write();
    h_sum0PedestalperLS2->Write();

    h_sumPedestalLS3->Write();
    h_2DsumPedestalLS3->Write();
    h_sumPedestalperLS3->Write();
    h_2D0sumPedestalLS3->Write();
    h_sum0PedestalperLS3->Write();

    h_sumPedestalLS4->Write();
    h_2DsumPedestalLS4->Write();
    h_sumPedestalperLS4->Write();
    h_2D0sumPedestalLS4->Write();
    h_sum0PedestalperLS4->Write();

    h_sumPedestalLS5->Write();
    h_2DsumPedestalLS5->Write();
    h_sumPedestalperLS5->Write();
    h_2D0sumPedestalLS5->Write();
    h_sum0PedestalperLS5->Write();

    h_sumPedestalLS6->Write();
    h_2DsumPedestalLS6->Write();
    h_sumPedestalperLS6->Write();
    h_2D0sumPedestalLS6->Write();
    h_sum0PedestalperLS6->Write();

    h_sumPedestalLS7->Write();
    h_2DsumPedestalLS7->Write();
    h_sumPedestalperLS7->Write();
    h_2D0sumPedestalLS7->Write();
    h_sum0PedestalperLS7->Write();

    h_sumPedestalLS8->Write();
    h_2DsumPedestalLS8->Write();
    h_sumPedestalperLS8->Write();
    h_2D0sumPedestalLS8->Write();
    h_sum0PedestalperLS8->Write();

    // for estimator1:

    // for HE gain stability vs LS:
    h_2DsumADCAmplLSdepth4HEu->Write();
    h_2D0sumADCAmplLSdepth4HEu->Write();
    h_2DsumADCAmplLSdepth5HEu->Write();
    h_2D0sumADCAmplLSdepth5HEu->Write();
    h_2DsumADCAmplLSdepth6HEu->Write();
    h_2D0sumADCAmplLSdepth6HEu->Write();
    h_2DsumADCAmplLSdepth7HEu->Write();
    h_2D0sumADCAmplLSdepth7HEu->Write();
    h_2DsumADCAmplLSdepth3HFu->Write();
    h_2D0sumADCAmplLSdepth3HFu->Write();
    h_2DsumADCAmplLSdepth4HFu->Write();
    h_2D0sumADCAmplLSdepth4HFu->Write();
    // for HB gain stability vs LS:
    h_2DsumADCAmplLSdepth3HBu->Write();
    h_2D0sumADCAmplLSdepth3HBu->Write();
    h_2DsumADCAmplLSdepth4HBu->Write();
    h_2D0sumADCAmplLSdepth4HBu->Write();

    h_sumADCAmplLS1copy1->Write();
    h_sumADCAmplLS1copy2->Write();
    h_sumADCAmplLS1copy3->Write();
    h_sumADCAmplLS1copy4->Write();
    h_sumADCAmplLS1copy5->Write();
    h_sumADCAmplLS1->Write();
    h_2DsumADCAmplLS1->Write();
    h_2DsumADCAmplLS1_LSselected->Write();
    h_sumADCAmplperLS1->Write();
    h_sumCutADCAmplperLS1->Write();
    h_2D0sumADCAmplLS1->Write();
    h_sum0ADCAmplperLS1->Write();

    h_sumADCAmplLS2->Write();
    h_2DsumADCAmplLS2->Write();
    h_2DsumADCAmplLS2_LSselected->Write();
    h_sumADCAmplperLS2->Write();
    h_sumCutADCAmplperLS2->Write();
    h_2D0sumADCAmplLS2->Write();
    h_sum0ADCAmplperLS2->Write();

    h_sumADCAmplLS3->Write();
    h_2DsumADCAmplLS3->Write();
    h_2DsumADCAmplLS3_LSselected->Write();
    h_sumADCAmplperLS3->Write();
    h_sumCutADCAmplperLS3->Write();
    h_2D0sumADCAmplLS3->Write();
    h_sum0ADCAmplperLS3->Write();

    h_sumADCAmplLS4->Write();
    h_2DsumADCAmplLS4->Write();
    h_2DsumADCAmplLS4_LSselected->Write();
    h_sumADCAmplperLS4->Write();
    h_sumCutADCAmplperLS4->Write();
    h_2D0sumADCAmplLS4->Write();
    h_sum0ADCAmplperLS4->Write();

    h_sumADCAmplLS5->Write();
    h_2DsumADCAmplLS5->Write();
    h_2DsumADCAmplLS5_LSselected->Write();
    h_sumADCAmplperLS5->Write();
    h_sumCutADCAmplperLS5->Write();
    h_2D0sumADCAmplLS5->Write();
    h_sum0ADCAmplperLS5->Write();

    h_sumADCAmplLS6->Write();
    h_2DsumADCAmplLS6->Write();
    h_2DsumADCAmplLS6_LSselected->Write();
    h_2D0sumADCAmplLS6->Write();
    h_sumADCAmplperLS6->Write();
    h_sumCutADCAmplperLS6->Write();
    h_sum0ADCAmplperLS6->Write();
    h_sumADCAmplperLS6u->Write();
    h_sumCutADCAmplperLS6u->Write();
    h_sum0ADCAmplperLS6u->Write();

    h_sumADCAmplperLSdepth4HEu->Write();
    h_sumADCAmplperLSdepth5HEu->Write();
    h_sumADCAmplperLSdepth6HEu->Write();
    h_sumADCAmplperLSdepth7HEu->Write();
    h_sumCutADCAmplperLSdepth4HEu->Write();
    h_sumCutADCAmplperLSdepth5HEu->Write();
    h_sumCutADCAmplperLSdepth6HEu->Write();
    h_sumCutADCAmplperLSdepth7HEu->Write();
    h_sum0ADCAmplperLSdepth4HEu->Write();
    h_sum0ADCAmplperLSdepth5HEu->Write();
    h_sum0ADCAmplperLSdepth6HEu->Write();
    h_sum0ADCAmplperLSdepth7HEu->Write();

    h_sumADCAmplperLSdepth3HBu->Write();
    h_sumADCAmplperLSdepth4HBu->Write();
    h_sumCutADCAmplperLSdepth3HBu->Write();
    h_sumCutADCAmplperLSdepth4HBu->Write();
    h_sum0ADCAmplperLSdepth3HBu->Write();
    h_sum0ADCAmplperLSdepth4HBu->Write();

    h_sumADCAmplperLS1_P1->Write();
    h_sum0ADCAmplperLS1_P1->Write();
    h_sumADCAmplperLS1_P2->Write();
    h_sum0ADCAmplperLS1_P2->Write();
    h_sumADCAmplperLS1_M1->Write();
    h_sum0ADCAmplperLS1_M1->Write();
    h_sumADCAmplperLS1_M2->Write();
    h_sum0ADCAmplperLS1_M2->Write();

    h_sumADCAmplperLS3_P1->Write();
    h_sum0ADCAmplperLS3_P1->Write();
    h_sumADCAmplperLS3_P2->Write();
    h_sum0ADCAmplperLS3_P2->Write();
    h_sumADCAmplperLS3_M1->Write();
    h_sum0ADCAmplperLS3_M1->Write();
    h_sumADCAmplperLS3_M2->Write();
    h_sum0ADCAmplperLS3_M2->Write();

    h_sumADCAmplperLS6_P1->Write();
    h_sum0ADCAmplperLS6_P1->Write();
    h_sumADCAmplperLS6_P2->Write();
    h_sum0ADCAmplperLS6_P2->Write();
    h_sumADCAmplperLS6_M1->Write();
    h_sum0ADCAmplperLS6_M1->Write();
    h_sumADCAmplperLS6_M2->Write();
    h_sum0ADCAmplperLS6_M2->Write();

    h_sumADCAmplperLS8_P1->Write();
    h_sum0ADCAmplperLS8_P1->Write();
    h_sumADCAmplperLS8_P2->Write();
    h_sum0ADCAmplperLS8_P2->Write();
    h_sumADCAmplperLS8_M1->Write();
    h_sum0ADCAmplperLS8_M1->Write();
    h_sumADCAmplperLS8_M2->Write();
    h_sum0ADCAmplperLS8_M2->Write();

    h_sumADCAmplLS7->Write();
    h_2DsumADCAmplLS7->Write();
    h_2DsumADCAmplLS7_LSselected->Write();
    h_2D0sumADCAmplLS7->Write();
    h_sumADCAmplperLS7->Write();
    h_sumCutADCAmplperLS7->Write();
    h_sum0ADCAmplperLS7->Write();
    h_sumADCAmplperLS7u->Write();
    h_sumCutADCAmplperLS7u->Write();
    h_sum0ADCAmplperLS7u->Write();

    h_sumADCAmplLS8->Write();
    h_2DsumADCAmplLS8->Write();
    h_2DsumADCAmplLS8_LSselected->Write();
    h_sumADCAmplperLS8->Write();
    h_sumCutADCAmplperLS8->Write();
    h_2D0sumADCAmplLS8->Write();
    h_sum0ADCAmplperLS8->Write();

    // for estimator2:
    h_sumTSmeanALS1->Write();
    h_2DsumTSmeanALS1->Write();
    h_sumTSmeanAperLS1->Write();
    h_sumTSmeanAperLS1_LSselected->Write();
    h_sumCutTSmeanAperLS1->Write();
    h_2D0sumTSmeanALS1->Write();
    h_sum0TSmeanAperLS1->Write();

    h_sumTSmeanALS2->Write();
    h_2DsumTSmeanALS2->Write();
    h_sumTSmeanAperLS2->Write();
    h_sumCutTSmeanAperLS2->Write();
    h_2D0sumTSmeanALS2->Write();
    h_sum0TSmeanAperLS2->Write();

    h_sumTSmeanALS3->Write();
    h_2DsumTSmeanALS3->Write();
    h_sumTSmeanAperLS3->Write();
    h_sumCutTSmeanAperLS3->Write();
    h_2D0sumTSmeanALS3->Write();
    h_sum0TSmeanAperLS3->Write();

    h_sumTSmeanALS4->Write();
    h_2DsumTSmeanALS4->Write();
    h_sumTSmeanAperLS4->Write();
    h_sumCutTSmeanAperLS4->Write();
    h_2D0sumTSmeanALS4->Write();
    h_sum0TSmeanAperLS4->Write();

    h_sumTSmeanALS5->Write();
    h_2DsumTSmeanALS5->Write();
    h_sumTSmeanAperLS5->Write();
    h_sumCutTSmeanAperLS5->Write();
    h_2D0sumTSmeanALS5->Write();
    h_sum0TSmeanAperLS5->Write();

    h_sumTSmeanALS6->Write();
    h_2DsumTSmeanALS6->Write();
    h_sumTSmeanAperLS6->Write();
    h_sumCutTSmeanAperLS6->Write();
    h_2D0sumTSmeanALS6->Write();
    h_sum0TSmeanAperLS6->Write();

    h_sumTSmeanALS7->Write();
    h_2DsumTSmeanALS7->Write();
    h_sumTSmeanAperLS7->Write();
    h_sumCutTSmeanAperLS7->Write();
    h_2D0sumTSmeanALS7->Write();
    h_sum0TSmeanAperLS7->Write();

    h_sumTSmeanALS8->Write();
    h_2DsumTSmeanALS8->Write();
    h_sumTSmeanAperLS8->Write();
    h_sumCutTSmeanAperLS8->Write();
    h_2D0sumTSmeanALS8->Write();
    h_sum0TSmeanAperLS8->Write();

    // for estimator3:
    h_sumTSmaxALS1->Write();
    h_2DsumTSmaxALS1->Write();
    h_sumTSmaxAperLS1->Write();
    h_sumTSmaxAperLS1_LSselected->Write();
    h_sumCutTSmaxAperLS1->Write();
    h_2D0sumTSmaxALS1->Write();
    h_sum0TSmaxAperLS1->Write();

    h_sumTSmaxALS2->Write();
    h_2DsumTSmaxALS2->Write();
    h_sumTSmaxAperLS2->Write();
    h_sumCutTSmaxAperLS2->Write();
    h_2D0sumTSmaxALS2->Write();
    h_sum0TSmaxAperLS2->Write();

    h_sumTSmaxALS3->Write();
    h_2DsumTSmaxALS3->Write();
    h_sumTSmaxAperLS3->Write();
    h_sumCutTSmaxAperLS3->Write();
    h_2D0sumTSmaxALS3->Write();
    h_sum0TSmaxAperLS3->Write();

    h_sumTSmaxALS4->Write();
    h_2DsumTSmaxALS4->Write();
    h_sumTSmaxAperLS4->Write();
    h_sumCutTSmaxAperLS4->Write();
    h_2D0sumTSmaxALS4->Write();
    h_sum0TSmaxAperLS4->Write();

    h_sumTSmaxALS5->Write();
    h_2DsumTSmaxALS5->Write();
    h_sumTSmaxAperLS5->Write();
    h_sumCutTSmaxAperLS5->Write();
    h_2D0sumTSmaxALS5->Write();
    h_sum0TSmaxAperLS5->Write();

    h_sumTSmaxALS6->Write();
    h_2DsumTSmaxALS6->Write();
    h_sumTSmaxAperLS6->Write();
    h_sumCutTSmaxAperLS6->Write();
    h_2D0sumTSmaxALS6->Write();
    h_sum0TSmaxAperLS6->Write();

    h_sumTSmaxALS7->Write();
    h_2DsumTSmaxALS7->Write();
    h_sumTSmaxAperLS7->Write();
    h_sumCutTSmaxAperLS7->Write();
    h_2D0sumTSmaxALS7->Write();
    h_sum0TSmaxAperLS7->Write();

    h_sumTSmaxALS8->Write();
    h_2DsumTSmaxALS8->Write();
    h_sumTSmaxAperLS8->Write();
    h_sumCutTSmaxAperLS8->Write();
    h_2D0sumTSmaxALS8->Write();
    h_sum0TSmaxAperLS8->Write();

    // for estimator4:
    h_sumAmplitudeLS1->Write();
    h_2DsumAmplitudeLS1->Write();
    h_sumAmplitudeperLS1->Write();
    h_sumAmplitudeperLS1_LSselected->Write();
    h_sumCutAmplitudeperLS1->Write();
    h_2D0sumAmplitudeLS1->Write();
    h_sum0AmplitudeperLS1->Write();

    h_sumAmplitudeLS2->Write();
    h_2DsumAmplitudeLS2->Write();
    h_sumAmplitudeperLS2->Write();
    h_sumCutAmplitudeperLS2->Write();
    h_2D0sumAmplitudeLS2->Write();
    h_sum0AmplitudeperLS2->Write();

    h_sumAmplitudeLS3->Write();
    h_2DsumAmplitudeLS3->Write();
    h_sumAmplitudeperLS3->Write();
    h_sumCutAmplitudeperLS3->Write();
    h_2D0sumAmplitudeLS3->Write();
    h_sum0AmplitudeperLS3->Write();

    h_sumAmplitudeLS4->Write();
    h_2DsumAmplitudeLS4->Write();
    h_sumAmplitudeperLS4->Write();
    h_sumCutAmplitudeperLS4->Write();
    h_2D0sumAmplitudeLS4->Write();
    h_sum0AmplitudeperLS4->Write();

    h_sumAmplitudeLS5->Write();
    h_2DsumAmplitudeLS5->Write();
    h_sumAmplitudeperLS5->Write();
    h_sumCutAmplitudeperLS5->Write();
    h_2D0sumAmplitudeLS5->Write();
    h_sum0AmplitudeperLS5->Write();

    h_sumAmplitudeLS6->Write();
    h_2DsumAmplitudeLS6->Write();
    h_sumAmplitudeperLS6->Write();
    h_sumCutAmplitudeperLS6->Write();
    h_2D0sumAmplitudeLS6->Write();
    h_sum0AmplitudeperLS6->Write();

    h_sumAmplitudeLS7->Write();
    h_2DsumAmplitudeLS7->Write();
    h_sumAmplitudeperLS7->Write();
    h_sumCutAmplitudeperLS7->Write();
    h_2D0sumAmplitudeLS7->Write();
    h_sum0AmplitudeperLS7->Write();

    h_sumAmplitudeLS8->Write();
    h_2DsumAmplitudeLS8->Write();
    h_sumAmplitudeperLS8->Write();
    h_sumCutAmplitudeperLS8->Write();
    h_2D0sumAmplitudeLS8->Write();
    h_sum0AmplitudeperLS8->Write();

    // for estimator6:
    h_sumErrorBLS1->Write();
    h_sumErrorBperLS1->Write();
    h_sum0ErrorBperLS1->Write();
    h_2D0sumErrorBLS1->Write();
    h_2DsumErrorBLS1->Write();
    h_sumErrorBLS2->Write();
    h_sumErrorBperLS2->Write();
    h_sum0ErrorBperLS2->Write();
    h_2D0sumErrorBLS2->Write();
    h_2DsumErrorBLS2->Write();

    h_sumErrorBLS3->Write();
    h_sumErrorBperLS3->Write();
    h_sum0ErrorBperLS3->Write();
    h_2D0sumErrorBLS3->Write();
    h_2DsumErrorBLS3->Write();
    h_sumErrorBLS4->Write();
    h_sumErrorBperLS4->Write();
    h_sum0ErrorBperLS4->Write();
    h_2D0sumErrorBLS4->Write();
    h_2DsumErrorBLS4->Write();
    h_sumErrorBLS5->Write();
    h_sumErrorBperLS5->Write();
    h_sum0ErrorBperLS5->Write();
    h_2D0sumErrorBLS5->Write();
    h_2DsumErrorBLS5->Write();

    h_sumErrorBLS6->Write();
    h_sumErrorBperLS6->Write();
    h_sum0ErrorBperLS6->Write();
    h_2D0sumErrorBLS6->Write();
    h_2DsumErrorBLS6->Write();
    h_sumErrorBLS7->Write();
    h_sumErrorBperLS7->Write();
    h_sum0ErrorBperLS7->Write();
    h_2D0sumErrorBLS7->Write();
    h_2DsumErrorBLS7->Write();

    h_sumErrorBLS8->Write();
    h_sumErrorBperLS8->Write();
    h_sum0ErrorBperLS8->Write();
    h_2D0sumErrorBLS8->Write();
    h_2DsumErrorBLS8->Write();

    // for estimator5:
    h_sumAmplLS1->Write();
    h_2DsumAmplLS1->Write();
    h_sumAmplperLS1->Write();
    h_sumAmplperLS1_LSselected->Write();
    h_sumCutAmplperLS1->Write();
    h_2D0sumAmplLS1->Write();
    h_sum0AmplperLS1->Write();

    h_sumAmplLS2->Write();
    h_2DsumAmplLS2->Write();
    h_sumAmplperLS2->Write();
    h_sumCutAmplperLS2->Write();
    h_2D0sumAmplLS2->Write();
    h_sum0AmplperLS2->Write();

    h_sumAmplLS3->Write();
    h_2DsumAmplLS3->Write();
    h_sumAmplperLS3->Write();
    h_sumCutAmplperLS3->Write();
    h_2D0sumAmplLS3->Write();
    h_sum0AmplperLS3->Write();

    h_sumAmplLS4->Write();
    h_2DsumAmplLS4->Write();
    h_sumAmplperLS4->Write();
    h_sumCutAmplperLS4->Write();
    h_2D0sumAmplLS4->Write();
    h_sum0AmplperLS4->Write();

    h_sumAmplLS5->Write();
    h_2DsumAmplLS5->Write();
    h_sumAmplperLS5->Write();
    h_sumCutAmplperLS5->Write();
    h_2D0sumAmplLS5->Write();
    h_sum0AmplperLS5->Write();

    h_sumAmplLS6->Write();
    h_2DsumAmplLS6->Write();
    h_sumAmplperLS6->Write();
    h_sumCutAmplperLS6->Write();
    h_2D0sumAmplLS6->Write();
    h_sum0AmplperLS6->Write();

    h_RatioOccupancy_HBP->Write();
    h_RatioOccupancy_HBM->Write();
    h_RatioOccupancy_HEP->Write();
    h_RatioOccupancy_HEM->Write();
    h_RatioOccupancy_HOP->Write();
    h_RatioOccupancy_HOM->Write();
    h_RatioOccupancy_HFP->Write();
    h_RatioOccupancy_HFM->Write();

    h_sumAmplLS7->Write();
    h_2DsumAmplLS7->Write();
    h_sumAmplperLS7->Write();
    h_sumCutAmplperLS7->Write();
    h_2D0sumAmplLS7->Write();
    h_sum0AmplperLS7->Write();

    h_sumAmplLS8->Write();
    h_2DsumAmplLS8->Write();
    h_sumAmplperLS8->Write();
    h_sumCutAmplperLS8->Write();
    h_2D0sumAmplLS8->Write();
    h_sum0AmplperLS8->Write();

    h_pedestal0_HB->Write();
    h_pedestal1_HB->Write();
    h_pedestal2_HB->Write();
    h_pedestal3_HB->Write();
    h_pedestalaver4_HB->Write();
    h_pedestalaver9_HB->Write();
    h_pedestalw0_HB->Write();
    h_pedestalw1_HB->Write();
    h_pedestalw2_HB->Write();
    h_pedestalw3_HB->Write();
    h_pedestalwaver4_HB->Write();
    h_pedestalwaver9_HB->Write();

    h_pedestal0_HE->Write();
    h_pedestal1_HE->Write();
    h_pedestal2_HE->Write();
    h_pedestal3_HE->Write();
    h_pedestalaver4_HE->Write();
    h_pedestalaver9_HE->Write();
    h_pedestalw0_HE->Write();
    h_pedestalw1_HE->Write();
    h_pedestalw2_HE->Write();
    h_pedestalw3_HE->Write();
    h_pedestalwaver4_HE->Write();
    h_pedestalwaver9_HE->Write();

    h_pedestal0_HF->Write();
    h_pedestal1_HF->Write();
    h_pedestal2_HF->Write();
    h_pedestal3_HF->Write();
    h_pedestalaver4_HF->Write();
    h_pedestalaver9_HF->Write();
    h_pedestalw0_HF->Write();
    h_pedestalw1_HF->Write();
    h_pedestalw2_HF->Write();
    h_pedestalw3_HF->Write();
    h_pedestalwaver4_HF->Write();
    h_pedestalwaver9_HF->Write();

    h_pedestal0_HO->Write();
    h_pedestal1_HO->Write();
    h_pedestal2_HO->Write();
    h_pedestal3_HO->Write();
    h_pedestalaver4_HO->Write();
    h_pedestalaver9_HO->Write();
    h_pedestalw0_HO->Write();
    h_pedestalw1_HO->Write();
    h_pedestalw2_HO->Write();
    h_pedestalw3_HO->Write();
    h_pedestalwaver4_HO->Write();
    h_pedestalwaver9_HO->Write();

    h_mapDepth1pedestalw_HB->Write();
    h_mapDepth2pedestalw_HB->Write();
    h_mapDepth3pedestalw_HB->Write();
    h_mapDepth4pedestalw_HB->Write();
    h_mapDepth1pedestalw_HE->Write();
    h_mapDepth2pedestalw_HE->Write();
    h_mapDepth3pedestalw_HE->Write();
    h_mapDepth4pedestalw_HE->Write();
    h_mapDepth5pedestalw_HE->Write();
    h_mapDepth6pedestalw_HE->Write();
    h_mapDepth7pedestalw_HE->Write();
    h_mapDepth1pedestalw_HF->Write();
    h_mapDepth2pedestalw_HF->Write();
    h_mapDepth3pedestalw_HF->Write();
    h_mapDepth4pedestalw_HF->Write();
    h_mapDepth4pedestalw_HO->Write();

    h_mapDepth1pedestal_HB->Write();
    h_mapDepth2pedestal_HB->Write();
    h_mapDepth3pedestal_HB->Write();
    h_mapDepth4pedestal_HB->Write();
    h_mapDepth1pedestal_HE->Write();
    h_mapDepth2pedestal_HE->Write();
    h_mapDepth3pedestal_HE->Write();
    h_mapDepth4pedestal_HE->Write();
    h_mapDepth5pedestal_HE->Write();
    h_mapDepth6pedestal_HE->Write();
    h_mapDepth7pedestal_HE->Write();
    h_mapDepth1pedestal_HF->Write();
    h_mapDepth2pedestal_HF->Write();
    h_mapDepth3pedestal_HF->Write();
    h_mapDepth4pedestal_HF->Write();
    h_mapDepth4pedestal_HO->Write();

    h_pedestal00_HB->Write();
    h_gain_HB->Write();
    h_respcorr_HB->Write();
    h_timecorr_HB->Write();
    h_lutcorr_HB->Write();
    h_difpedestal0_HB->Write();
    h_difpedestal1_HB->Write();
    h_difpedestal2_HB->Write();
    h_difpedestal3_HB->Write();

    h_pedestal00_HE->Write();
    h_gain_HE->Write();
    h_respcorr_HE->Write();
    h_timecorr_HE->Write();
    h_lutcorr_HE->Write();

    h_pedestal00_HF->Write();
    h_gain_HF->Write();
    h_respcorr_HF->Write();
    h_timecorr_HF->Write();
    h_lutcorr_HF->Write();

    h_pedestal00_HO->Write();
    h_gain_HO->Write();
    h_respcorr_HO->Write();
    h_timecorr_HO->Write();
    h_lutcorr_HO->Write();

    h2_pedvsampl_HB->Write();
    h2_pedwvsampl_HB->Write();
    h_pedvsampl_HB->Write();
    h_pedwvsampl_HB->Write();
    h_pedvsampl0_HB->Write();
    h_pedwvsampl0_HB->Write();
    h2_amplvsped_HB->Write();
    h2_amplvspedw_HB->Write();
    h_amplvsped_HB->Write();
    h_amplvspedw_HB->Write();
    h_amplvsped0_HB->Write();

    h2_pedvsampl_HE->Write();
    h2_pedwvsampl_HE->Write();
    h_pedvsampl_HE->Write();
    h_pedwvsampl_HE->Write();
    h_pedvsampl0_HE->Write();
    h_pedwvsampl0_HE->Write();
    h2_pedvsampl_HF->Write();
    h2_pedwvsampl_HF->Write();
    h_pedvsampl_HF->Write();
    h_pedwvsampl_HF->Write();
    h_pedvsampl0_HF->Write();
    h_pedwvsampl0_HF->Write();
    h2_pedvsampl_HO->Write();
    h2_pedwvsampl_HO->Write();
    h_pedvsampl_HO->Write();
    h_pedwvsampl_HO->Write();
    h_pedvsampl0_HO->Write();
    h_pedwvsampl0_HO->Write();

    h_mapDepth1Ped0_HB->Write();
    h_mapDepth1Ped1_HB->Write();
    h_mapDepth1Ped2_HB->Write();
    h_mapDepth1Ped3_HB->Write();
    h_mapDepth1Pedw0_HB->Write();
    h_mapDepth1Pedw1_HB->Write();
    h_mapDepth1Pedw2_HB->Write();
    h_mapDepth1Pedw3_HB->Write();
    h_mapDepth2Ped0_HB->Write();
    h_mapDepth2Ped1_HB->Write();
    h_mapDepth2Ped2_HB->Write();
    h_mapDepth2Ped3_HB->Write();
    h_mapDepth2Pedw0_HB->Write();
    h_mapDepth2Pedw1_HB->Write();
    h_mapDepth2Pedw2_HB->Write();
    h_mapDepth2Pedw3_HB->Write();

    h_mapDepth1Ped0_HE->Write();
    h_mapDepth1Ped1_HE->Write();
    h_mapDepth1Ped2_HE->Write();
    h_mapDepth1Ped3_HE->Write();
    h_mapDepth1Pedw0_HE->Write();
    h_mapDepth1Pedw1_HE->Write();
    h_mapDepth1Pedw2_HE->Write();
    h_mapDepth1Pedw3_HE->Write();
    h_mapDepth2Ped0_HE->Write();
    h_mapDepth2Ped1_HE->Write();
    h_mapDepth2Ped2_HE->Write();
    h_mapDepth2Ped3_HE->Write();
    h_mapDepth2Pedw0_HE->Write();
    h_mapDepth2Pedw1_HE->Write();
    h_mapDepth2Pedw2_HE->Write();
    h_mapDepth2Pedw3_HE->Write();
    h_mapDepth3Ped0_HE->Write();
    h_mapDepth3Ped1_HE->Write();
    h_mapDepth3Ped2_HE->Write();
    h_mapDepth3Ped3_HE->Write();
    h_mapDepth3Pedw0_HE->Write();
    h_mapDepth3Pedw1_HE->Write();
    h_mapDepth3Pedw2_HE->Write();
    h_mapDepth3Pedw3_HE->Write();

    h_mapDepth1Ped0_HF->Write();
    h_mapDepth1Ped1_HF->Write();
    h_mapDepth1Ped2_HF->Write();
    h_mapDepth1Ped3_HF->Write();
    h_mapDepth1Pedw0_HF->Write();
    h_mapDepth1Pedw1_HF->Write();
    h_mapDepth1Pedw2_HF->Write();
    h_mapDepth1Pedw3_HF->Write();
    h_mapDepth2Ped0_HF->Write();
    h_mapDepth2Ped1_HF->Write();
    h_mapDepth2Ped2_HF->Write();
    h_mapDepth2Ped3_HF->Write();
    h_mapDepth2Pedw0_HF->Write();
    h_mapDepth2Pedw1_HF->Write();
    h_mapDepth2Pedw2_HF->Write();
    h_mapDepth2Pedw3_HF->Write();

    h_mapDepth4Ped0_HO->Write();
    h_mapDepth4Ped1_HO->Write();
    h_mapDepth4Ped2_HO->Write();
    h_mapDepth4Ped3_HO->Write();
    h_mapDepth4Pedw0_HO->Write();
    h_mapDepth4Pedw1_HO->Write();
    h_mapDepth4Pedw2_HO->Write();
    h_mapDepth4Pedw3_HO->Write();

    h_mapGetRMSOverNormalizedSignal_HB->Write();
    h_mapGetRMSOverNormalizedSignal0_HB->Write();
    h_mapGetRMSOverNormalizedSignal_HE->Write();
    h_mapGetRMSOverNormalizedSignal0_HE->Write();
    h_mapGetRMSOverNormalizedSignal_HF->Write();
    h_mapGetRMSOverNormalizedSignal0_HF->Write();
    h_mapGetRMSOverNormalizedSignal_HO->Write();
    h_mapGetRMSOverNormalizedSignal0_HO->Write();

    h_shape_Ahigh_HB0->Write();
    h_shape0_Ahigh_HB0->Write();
    h_shape_Alow_HB0->Write();
    h_shape0_Alow_HB0->Write();
    h_shape_Ahigh_HB1->Write();
    h_shape0_Ahigh_HB1->Write();
    h_shape_Alow_HB1->Write();
    h_shape0_Alow_HB1->Write();
    h_shape_Ahigh_HB2->Write();
    h_shape0_Ahigh_HB2->Write();
    h_shape_Alow_HB2->Write();
    h_shape0_Alow_HB2->Write();
    h_shape_Ahigh_HB3->Write();
    h_shape0_Ahigh_HB3->Write();
    h_shape_Alow_HB3->Write();
    h_shape0_Alow_HB3->Write();

    h_shape_bad_channels_HB->Write();
    h_shape0_bad_channels_HB->Write();
    h_shape_good_channels_HB->Write();
    h_shape0_good_channels_HB->Write();
    h_shape_bad_channels_HE->Write();
    h_shape0_bad_channels_HE->Write();
    h_shape_good_channels_HE->Write();
    h_shape0_good_channels_HE->Write();
    h_shape_bad_channels_HF->Write();
    h_shape0_bad_channels_HF->Write();
    h_shape_good_channels_HF->Write();
    h_shape0_good_channels_HF->Write();
    h_shape_bad_channels_HO->Write();
    h_shape0_bad_channels_HO->Write();
    h_shape_good_channels_HO->Write();
    h_shape0_good_channels_HO->Write();

    h_sumamplitude_depth1_HB->Write();
    h_sumamplitude_depth2_HB->Write();
    h_sumamplitude_depth1_HE->Write();
    h_sumamplitude_depth2_HE->Write();
    h_sumamplitude_depth3_HE->Write();
    h_sumamplitude_depth1_HF->Write();
    h_sumamplitude_depth2_HF->Write();
    h_sumamplitude_depth4_HO->Write();

    h_sumamplitude_depth1_HB0->Write();
    h_sumamplitude_depth2_HB0->Write();
    h_sumamplitude_depth1_HE0->Write();
    h_sumamplitude_depth2_HE0->Write();
    h_sumamplitude_depth3_HE0->Write();
    h_sumamplitude_depth1_HF0->Write();
    h_sumamplitude_depth2_HF0->Write();
    h_sumamplitude_depth4_HO0->Write();

    h_sumamplitude_depth1_HB1->Write();
    h_sumamplitude_depth2_HB1->Write();
    h_sumamplitude_depth1_HE1->Write();
    h_sumamplitude_depth2_HE1->Write();
    h_sumamplitude_depth3_HE1->Write();
    h_sumamplitude_depth1_HF1->Write();
    h_sumamplitude_depth2_HF1->Write();
    h_sumamplitude_depth4_HO1->Write();
    h_bcnnbadchannels_depth1_HB->Write();
    h_bcnnbadchannels_depth2_HB->Write();
    h_bcnnbadchannels_depth1_HE->Write();
    h_bcnnbadchannels_depth2_HE->Write();
    h_bcnnbadchannels_depth3_HE->Write();
    h_bcnnbadchannels_depth4_HO->Write();
    h_bcnnbadchannels_depth1_HF->Write();
    h_bcnnbadchannels_depth2_HF->Write();
    h_bcnbadrate0_depth1_HB->Write();
    h_bcnbadrate0_depth2_HB->Write();
    h_bcnbadrate0_depth1_HE->Write();
    h_bcnbadrate0_depth2_HE->Write();
    h_bcnbadrate0_depth3_HE->Write();
    h_bcnbadrate0_depth4_HO->Write();
    h_bcnbadrate0_depth1_HF->Write();
    h_bcnbadrate0_depth2_HF->Write();

    h_Amplitude_forCapIdErrors_HB1->Write();
    h_Amplitude_forCapIdErrors_HB2->Write();
    h_Amplitude_forCapIdErrors_HE1->Write();
    h_Amplitude_forCapIdErrors_HE2->Write();
    h_Amplitude_forCapIdErrors_HE3->Write();
    h_Amplitude_forCapIdErrors_HF1->Write();
    h_Amplitude_forCapIdErrors_HF2->Write();
    h_Amplitude_forCapIdErrors_HO4->Write();

    h_Amplitude_notCapIdErrors_HB1->Write();
    h_Amplitude_notCapIdErrors_HB2->Write();
    h_Amplitude_notCapIdErrors_HE1->Write();
    h_Amplitude_notCapIdErrors_HE2->Write();
    h_Amplitude_notCapIdErrors_HE3->Write();
    h_Amplitude_notCapIdErrors_HF1->Write();
    h_Amplitude_notCapIdErrors_HF2->Write();
    h_Amplitude_notCapIdErrors_HO4->Write();

    h_averSIGNALoccupancy_HB->Write();
    h_averSIGNALoccupancy_HE->Write();
    h_averSIGNALoccupancy_HF->Write();
    h_averSIGNALoccupancy_HO->Write();

    h_averSIGNALsumamplitude_HB->Write();
    h_averSIGNALsumamplitude_HE->Write();
    h_averSIGNALsumamplitude_HF->Write();
    h_averSIGNALsumamplitude_HO->Write();

    h_averNOSIGNALoccupancy_HB->Write();
    h_averNOSIGNALoccupancy_HE->Write();
    h_averNOSIGNALoccupancy_HF->Write();
    h_averNOSIGNALoccupancy_HO->Write();

    h_averNOSIGNALsumamplitude_HB->Write();
    h_averNOSIGNALsumamplitude_HE->Write();
    h_averNOSIGNALsumamplitude_HF->Write();
    h_averNOSIGNALsumamplitude_HO->Write();

    h_maxxSUMAmpl_HB->Write();
    h_maxxSUMAmpl_HE->Write();
    h_maxxSUMAmpl_HF->Write();
    h_maxxSUMAmpl_HO->Write();

    h_maxxOCCUP_HB->Write();
    h_maxxOCCUP_HE->Write();
    h_maxxOCCUP_HF->Write();
    h_maxxOCCUP_HO->Write();

    h_sumamplitudechannel_HB->Write();
    h_sumamplitudechannel_HE->Write();
    h_sumamplitudechannel_HF->Write();
    h_sumamplitudechannel_HO->Write();

    h_eventamplitude_HB->Write();
    h_eventamplitude_HE->Write();
    h_eventamplitude_HF->Write();
    h_eventamplitude_HO->Write();

    h_eventoccupancy_HB->Write();
    h_eventoccupancy_HE->Write();
    h_eventoccupancy_HF->Write();
    h_eventoccupancy_HO->Write();

    h_2DAtaildepth1_HB->Write();
    h_2D0Ataildepth1_HB->Write();
    h_2DAtaildepth2_HB->Write();
    h_2D0Ataildepth2_HB->Write();

    h_mapenophinorm_HE1->Write();
    h_mapenophinorm_HE2->Write();
    h_mapenophinorm_HE3->Write();
    h_mapenophinorm_HE4->Write();
    h_mapenophinorm_HE5->Write();
    h_mapenophinorm_HE6->Write();
    h_mapenophinorm_HE7->Write();
    h_mapenophinorm2_HE1->Write();
    h_mapenophinorm2_HE2->Write();
    h_mapenophinorm2_HE3->Write();
    h_mapenophinorm2_HE4->Write();
    h_mapenophinorm2_HE5->Write();
    h_mapenophinorm2_HE6->Write();
    h_mapenophinorm2_HE7->Write();

    h_maprphinorm_HE1->Write();
    h_maprphinorm_HE2->Write();
    h_maprphinorm_HE3->Write();
    h_maprphinorm_HE4->Write();
    h_maprphinorm_HE5->Write();
    h_maprphinorm_HE6->Write();
    h_maprphinorm_HE7->Write();
    h_maprphinorm2_HE1->Write();
    h_maprphinorm2_HE2->Write();
    h_maprphinorm2_HE3->Write();
    h_maprphinorm2_HE4->Write();
    h_maprphinorm2_HE5->Write();
    h_maprphinorm2_HE6->Write();
    h_maprphinorm2_HE7->Write();

    h_maprphinorm0_HE1->Write();
    h_maprphinorm0_HE2->Write();
    h_maprphinorm0_HE3->Write();
    h_maprphinorm0_HE4->Write();
    h_maprphinorm0_HE5->Write();
    h_maprphinorm0_HE6->Write();
    h_maprphinorm0_HE7->Write();

    // RADDAM:
    h_mapDepth1RADDAM_HE->Write();
    h_mapDepth2RADDAM_HE->Write();
    h_mapDepth3RADDAM_HE->Write();
    h_mapDepth1RADDAM0_HE->Write();
    h_mapDepth2RADDAM0_HE->Write();
    h_mapDepth3RADDAM0_HE->Write();
    h_AamplitudewithPedSubtr_RADDAM_HE->Write();
    h_AamplitudewithPedSubtr_RADDAM_HEzoom0->Write();
    h_AamplitudewithPedSubtr_RADDAM_HEzoom1->Write();
    h_A_Depth1RADDAM_HE->Write();
    h_A_Depth2RADDAM_HE->Write();
    h_A_Depth3RADDAM_HE->Write();
    h_sumphiEta16Depth3RADDAM_HED2->Write();
    h_Eta16Depth3RADDAM_HED2->Write();
    h_NphiForEta16Depth3RADDAM_HED2->Write();
    h_sumphiEta16Depth3RADDAM_HED2P->Write();
    h_Eta16Depth3RADDAM_HED2P->Write();
    h_NphiForEta16Depth3RADDAM_HED2P->Write();
    h_sumphiEta16Depth3RADDAM_HED2ALL->Write();
    h_Eta16Depth3RADDAM_HED2ALL->Write();
    h_NphiForEta16Depth3RADDAM_HED2ALL->Write();
    h_sigLayer1RADDAM_HE->Write();
    h_sigLayer2RADDAM_HE->Write();
    h_sigLayer1RADDAM0_HE->Write();
    h_sigLayer2RADDAM0_HE->Write();
    h_sigLayer1RADDAM5_HE->Write();
    h_sigLayer2RADDAM5_HE->Write();
    h_sigLayer1RADDAM6_HE->Write();
    h_sigLayer2RADDAM6_HE->Write();
    h_sigLayer1RADDAM5_HED2->Write();
    h_sigLayer1RADDAM6_HED2->Write();
    h_sigLayer2RADDAM5_HED2->Write();
    h_sigLayer2RADDAM6_HED2->Write();
    h_mapDepth3RADDAM16_HE->Write();

    h_amplitudeaveragedbydepthes_HE->Write();
    h_ndepthesperamplitudebins_HE->Write();
    ///////////////////////
  }  //if
  ///////////////////////
  hOutputFile->Close();
  std::cout << "===== Finish writing user histograms and ntuple =====" << std::endl;
  ///////////////////////
}

/////////////////////////////////////////////////////////////////////////
//  SERVICE FUNCTIONS --------------------------------------------------------

double CMTRawAnalyzer::dR(double eta1, double phi1, double eta2, double phi2) {
  double PI = 3.1415926535898;
  double deltaphi = phi1 - phi2;
  if (phi2 > phi1) {
    deltaphi = phi2 - phi1;
  }
  if (deltaphi > PI) {
    deltaphi = 2. * PI - deltaphi;
  }
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
  return tmp;
}

double CMTRawAnalyzer::phi12(double phi1, double en1, double phi2, double en2) {
  // weighted mean value of phi1 and phi2

  double tmp;
  double PI = 3.1415926535898;
  double a1 = phi1;
  double a2 = phi2;

  if (a1 > 0.5 * PI && a2 < 0.)
    a2 += 2 * PI;
  if (a2 > 0.5 * PI && a1 < 0.)
    a1 += 2 * PI;
  tmp = (a1 * en1 + a2 * en2) / (en1 + en2);
  if (tmp > PI)
    tmp -= 2. * PI;

  return tmp;
}

double CMTRawAnalyzer::dPhiWsign(double phi1, double phi2) {
  // clockwise      phi2 w.r.t phi1 means "+" phi distance
  // anti-clockwise phi2 w.r.t phi1 means "-" phi distance

  double PI = 3.1415926535898;
  double a1 = phi1;
  double a2 = phi2;
  double tmp = a2 - a1;
  if (a1 * a2 < 0.) {
    if (a1 > 0.5 * PI)
      tmp += 2. * PI;
    if (a2 > 0.5 * PI)
      tmp -= 2. * PI;
  }
  return tmp;
}

void CMTRawAnalyzer::fillMAP() {
  HcalLogicalMapGenerator gen;
  HcalLogicalMap lmap = gen.createMap(topo);

  //    HcalElectronicsMap emap=lmap.generateHcalElectronicsMap();

  //    const HcalElectronicsMap* emap=conditions->getHcalMapping();
  const HcalElectronicsMap* emap;
  emap = conditions->getHcalMapping();

  std::string subdet = "";

  MAPfile << "#define LogEleMapdb_h" << std::endl;
  MAPfile << "#include <algorithm>" << std::endl;
  MAPfile << "#include <iostream>" << std::endl;
  MAPfile << "#include <vector>" << std::endl;
  MAPfile << "#include <string>" << std::endl;
  MAPfile << "#include <sstream>" << std::endl;
  MAPfile << std::endl;

  MAPfile << "struct Cell {" << std::endl;
  MAPfile << " std::string subdet;" << std::endl;
  MAPfile << " int Eta;" << std::endl;
  MAPfile << " int Phi;" << std::endl;
  MAPfile << " int Depth;" << std::endl;
  MAPfile << " std::string RBX;" << std::endl;
  MAPfile << " int RM;" << std::endl;
  MAPfile << " int Pixel;" << std::endl;
  MAPfile << " int RMfiber;" << std::endl;
  MAPfile << " int FiberCh;" << std::endl;
  MAPfile << " int QIE;" << std::endl;
  MAPfile << " int ADC;" << std::endl;
  MAPfile << " int VMECardID;" << std::endl;
  MAPfile << " int dccID;" << std::endl;
  MAPfile << " int Spigot;" << std::endl;
  MAPfile << " int FiberIndex;" << std::endl;
  MAPfile << " int HtrSlot;" << std::endl;
  MAPfile << " int HtrTB;" << std::endl;
  MAPfile << std::endl;

  MAPfile << "// the function check, if \"par\" == \"val\" for this cell" << std::endl;
  MAPfile << " bool check(const std::string par, const int val) const " << std::endl;
  MAPfile << " {" << std::endl;
  MAPfile << "       if (par == \"Eta\")    return (val == Eta);" << std::endl;
  MAPfile << "  else if (par == \"Phi\")     return (val == Phi);" << std::endl;
  MAPfile << "  else if (par == \"Depth\")      return (val == Depth);" << std::endl;
  MAPfile << "  else if (par == \"RM\")     return (val == RM);" << std::endl;
  MAPfile << "  else if (par == \"Pixel\") return (val == Pixel);" << std::endl;
  MAPfile << "  else if (par == \"RMfiber\")    return (val == RMfiber);" << std::endl;
  MAPfile << "  else if (par == \"FiberCh\")    return (val == FiberCh);" << std::endl;
  MAPfile << "  else if (par == \"QIE\")     return (val == QIE);" << std::endl;
  MAPfile << "  else if (par == \"ADC\")     return (val == ADC);" << std::endl;
  MAPfile << "  else if (par == \"VMECardID\")     return (val == VMECardID);" << std::endl;
  MAPfile << "  else if (par == \"dccID\")     return (val == dccID);" << std::endl;
  MAPfile << "  else if (par == \"Spigot\")     return (val == Spigot);" << std::endl;
  MAPfile << "  else if (par == \"FiberIndex\")     return (val == FiberIndex);" << std::endl;
  MAPfile << "  else if (par == \"HtrSlot\")     return (val == HtrSlot);" << std::endl;
  MAPfile << "  else if (par == \"HtrTB\")     return (val == HtrTB);" << std::endl;
  MAPfile << "  else return false;" << std::endl;
  MAPfile << " }" << std::endl;
  MAPfile << std::endl;

  MAPfile << " bool check(const std::string par, const std::string val) const" << std::endl;
  MAPfile << " {" << std::endl;
  MAPfile << "       if (par == \"subdet\")    return (val == subdet);" << std::endl;
  MAPfile << "  else if (par == \"RBX\")    return (val == RBX);" << std::endl;
  MAPfile << "  else return false;" << std::endl;
  MAPfile << " }" << std::endl;

  MAPfile << "};" << std::endl;
  MAPfile << std::endl;

  MAPfile << "const Cell AllCells[] = {" << std::endl;
  MAPfile << "//{ SD, Eta, Phi, Depth,     RBX, RM, PIXEL, RMfiber, Fiber Ch., QIE, ADC, VMECrateId, dccid, spigot, "
             "fiberIndex, htrSlot, htrTopBottom }"
          << std::endl;

  // HB

  for (int eta = -16; eta < 0; eta++) {
    for (int phi = 1; phi <= nphi; phi++) {
      for (int depth = 1; depth <= 2; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalBarrel, eta, phi, depth);
        subdet = "HB";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta
  for (int eta = 1; eta <= 16; eta++) {
    for (int phi = 1; phi <= nphi; phi++) {
      for (int depth = 1; depth <= 2; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalBarrel, eta, phi, depth);
        subdet = "HB";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() - 1 << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  // HE
  for (int eta = -20; eta <= -20; eta++) {
    for (int phi = nphi; phi <= nphi; phi++) {
      for (int depth = 1; depth <= 2; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalEndcap, eta, phi, depth);
        subdet = "HE";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  for (int eta = -19; eta <= -16; eta++) {
    for (int phi = nphi; phi <= nphi; phi++) {
      for (int depth = 1; depth <= 3; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalEndcap, eta, phi, depth);
        subdet = "HE";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta
  for (int eta = -29; eta <= -16; eta++) {
    for (int phi = 1; phi <= 71; phi++) {
      for (int depth = 1; depth <= 3; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalEndcap, eta, phi, depth);
        subdet = "HE";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta
  for (int eta = 16; eta <= 29; eta++) {
    for (int phi = 1; phi <= nphi; phi++) {
      for (int depth = 1; depth <= 3; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalEndcap, eta, phi, depth);
        subdet = "HE";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  // HF

  for (int eta = -41; eta <= -29; eta++) {
    for (int phi = 1; phi <= nphi; phi += 2) {
      for (int depth = 1; depth <= 2; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalForward, eta, phi, depth);
        subdet = "HF";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  for (int eta = 29; eta <= 41; eta++) {
    for (int phi = 1; phi <= nphi; phi += 2) {
      for (int depth = 1; depth <= 2; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalForward, eta, phi, depth);
        subdet = "HF";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  // HO

  for (int eta = -15; eta < 0; eta++) {
    for (int phi = 1; phi <= nphi; phi++) {
      for (int depth = 4; depth <= 4; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalOuter, eta, phi, depth);
        subdet = "HO";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta

  for (int eta = 1; eta <= 15; eta++) {
    for (int phi = 1; phi <= nphi; phi++) {
      for (int depth = 4; depth <= 4; depth++) {
        HcalDetId* detid = nullptr;
        detid = new HcalDetId(HcalOuter, eta, phi, depth);
        subdet = "HO";
        HcalFrontEndId lmap_entry = lmap.getHcalFrontEndId(*detid);
        HcalElectronicsId emap_entry = emap->lookup(*detid);
        MAPfile << "  {\"" << subdet << "\" , " << detid->ieta() - 1 << " , " << detid->iphi() - 1 << " ,    "
                << detid->depth() << " ,";
        MAPfile << "\"" << lmap_entry.rbx() << "\" , " << lmap_entry.rm() << " ,   " << lmap_entry.pixel() << " ,      "
                << lmap_entry.rmFiber() << " ,        ";
        MAPfile << lmap_entry.fiberChannel() << " ,  " << lmap_entry.qieCard() << " ,  " << lmap_entry.adc()
                << " ,        ";
        MAPfile << emap_entry.readoutVMECrateId() << " ,    " << emap_entry.dccid() << " ,     " << emap_entry.spigot()
                << " ,         " << emap_entry.fiberIndex() << " ,      ";
        MAPfile << emap_entry.htrSlot() << " ,      " << emap_entry.htrTopBottom();
        MAPfile << "}," << std::endl;
        delete detid;
      }  //Depth
    }    //Phi
  }      //Eta
  MAPfile << "};" << std::endl;
  MAPfile << std::endl;

  MAPfile << "// macro for array length calculation" << std::endl;
  MAPfile << "#define DIM(a) (sizeof(a)/sizeof(a[0]))" << std::endl;
  MAPfile << std::endl;

  MAPfile << "// class for cells array managing" << std::endl;
  MAPfile << "class CellDB {" << std::endl;
  MAPfile << "public:" << std::endl;
  MAPfile << "  CellDB()" << std::endl;
  MAPfile << "  : cells(AllCells,  AllCells + DIM(AllCells))" << std::endl;
  MAPfile << "{}" << std::endl;
  MAPfile << std::endl;

  MAPfile << "// return i-th cell" << std::endl;
  MAPfile << "Cell operator [] (int i) const {return cells[i];}" << std::endl;

  MAPfile << "// number of cells in database" << std::endl;
  MAPfile << "int size() const {return cells.size();}" << std::endl;
  MAPfile << std::endl;

  MAPfile << "// select cells for which \"par\" == \"val\"" << std::endl;
  MAPfile << "template<typename T>" << std::endl;
  MAPfile << "CellDB find(const std::string par, const T val) const" << std::endl;
  MAPfile << "{" << std::endl;
  MAPfile << "  std::vector<Cell> s;" << std::endl;
  MAPfile << "  for (size_t i = 0; i < cells.size(); ++i)" << std::endl;
  MAPfile << "    if (cells[i].check(par, val))" << std::endl;
  MAPfile << "    s.push_back(cells[i]);" << std::endl;
  MAPfile << "  return CellDB(s);" << std::endl;
  MAPfile << "} " << std::endl;
  MAPfile << std::endl;
  MAPfile << "private:" << std::endl;
  MAPfile << " CellDB( const std::vector<Cell> s)" << std::endl;
  MAPfile << " : cells(s)" << std::endl;
  MAPfile << "{}" << std::endl;
  MAPfile << "std::vector<Cell> cells;" << std::endl;
  MAPfile << "};" << std::endl;
  MAPfile.close();
  std::cout << "===== Finish writing Channel MAP =====" << std::endl;
}

#endif
