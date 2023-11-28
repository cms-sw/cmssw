// -*- C++ -*-
//
// Package:    CMTRawAnalyzer
//
#include <fstream>
#include <iostream>
#include <cmath>
#include <iosfwd>
#include <bitset>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
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
double const adc2fC_QIE10[NUMADCS] = {
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
const float bphi = 72.;
const int zneta = 22;
const int znphi = 18;
const int npfit = 220;
const float anpfit = 220.;  // for SiPM:

//
class CMTRawAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
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
  double phi12(double phi1, double en1, double phi2, double en2);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double dPhiWsign(double phi1, double phi2);

  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_qie10_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbheSignal_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbheNoise_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hfSignal_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hfNoise_;

  edm::Service<TFileService> fs_;
  edm::InputTag inputTag_;
  const edm::ESGetToken<HcalDbService, HcalDbRecord> tokDB_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tokTopo_;
  const HcalQIEShape* shape;
  const HcalDbService* conditions;
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
  int flagToUseDigiCollectionsORNot_;
  int flagIterativeMethodCalibrationGroupDigi_;
  int flagIterativeMethodCalibrationGroupReco_;
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
  double lsdep_estimator1_HEdepth4_;
  double lsdep_estimator1_HEdepth5_;
  double lsdep_estimator1_HEdepth6_;
  double lsdep_estimator1_HEdepth7_;
  double lsdep_estimator1_HFdepth3_;
  double lsdep_estimator1_HFdepth4_;
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

  TH2F* h_mapDepth1TS2_HB;
  TH2F* h_mapDepth2TS2_HB;
  TH2F* h_mapDepth3TS2_HB;
  TH2F* h_mapDepth4TS2_HB;
  TH2F* h_mapDepth1TS2_HE;
  TH2F* h_mapDepth2TS2_HE;
  TH2F* h_mapDepth3TS2_HE;
  TH2F* h_mapDepth4TS2_HE;
  TH2F* h_mapDepth5TS2_HE;
  TH2F* h_mapDepth6TS2_HE;
  TH2F* h_mapDepth7TS2_HE;
  TH2F* h_mapDepth1TS1_HF;
  TH2F* h_mapDepth2TS1_HF;
  TH2F* h_mapDepth3TS1_HF;
  TH2F* h_mapDepth4TS1_HF;
  TH2F* h_mapDepth4TS012_HO;

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
  // phy-symmetry, phi-symmetry:
  // Reco:
  TH1F* h_energyhitSignal_HB;
  TH1F* h_energyhitSignal_HE;
  TH1F* h_energyhitSignal_HF;
  TH1F* h_energyhitNoise_HB;
  TH1F* h_energyhitNoise_HE;
  TH1F* h_energyhitNoise_HF;
  //HB:
  TH2F* h_recSignalEnergy0_HB1;
  TH2F* h_recSignalEnergy1_HB1;
  TH2F* h_recSignalEnergy2_HB1;
  TH2F* h_recSignalEnergy0_HB2;
  TH2F* h_recSignalEnergy1_HB2;
  TH2F* h_recSignalEnergy2_HB2;
  TH2F* h_recSignalEnergy0_HB3;
  TH2F* h_recSignalEnergy1_HB3;
  TH2F* h_recSignalEnergy2_HB3;
  TH2F* h_recSignalEnergy0_HB4;
  TH2F* h_recSignalEnergy1_HB4;
  TH2F* h_recSignalEnergy2_HB4;
  TH2F* h_recNoiseEnergy0_HB1;
  TH2F* h_recNoiseEnergy1_HB1;
  TH2F* h_recNoiseEnergy2_HB1;
  TH2F* h_recNoiseEnergy0_HB2;
  TH2F* h_recNoiseEnergy1_HB2;
  TH2F* h_recNoiseEnergy2_HB2;
  TH2F* h_recNoiseEnergy0_HB3;
  TH2F* h_recNoiseEnergy1_HB3;
  TH2F* h_recNoiseEnergy2_HB3;
  TH2F* h_recNoiseEnergy0_HB4;
  TH2F* h_recNoiseEnergy1_HB4;
  TH2F* h_recNoiseEnergy2_HB4;
  //HE:
  TH2F* h_recSignalEnergy0_HE1;
  TH2F* h_recSignalEnergy1_HE1;
  TH2F* h_recSignalEnergy2_HE1;
  TH2F* h_recSignalEnergy0_HE2;
  TH2F* h_recSignalEnergy1_HE2;
  TH2F* h_recSignalEnergy2_HE2;
  TH2F* h_recSignalEnergy0_HE3;
  TH2F* h_recSignalEnergy1_HE3;
  TH2F* h_recSignalEnergy2_HE3;
  TH2F* h_recSignalEnergy0_HE4;
  TH2F* h_recSignalEnergy1_HE4;
  TH2F* h_recSignalEnergy2_HE4;
  TH2F* h_recSignalEnergy0_HE5;
  TH2F* h_recSignalEnergy1_HE5;
  TH2F* h_recSignalEnergy2_HE5;
  TH2F* h_recSignalEnergy0_HE6;
  TH2F* h_recSignalEnergy1_HE6;
  TH2F* h_recSignalEnergy2_HE6;
  TH2F* h_recSignalEnergy0_HE7;
  TH2F* h_recSignalEnergy1_HE7;
  TH2F* h_recSignalEnergy2_HE7;
  TH2F* h_recNoiseEnergy0_HE1;
  TH2F* h_recNoiseEnergy1_HE1;
  TH2F* h_recNoiseEnergy2_HE1;
  TH2F* h_recNoiseEnergy0_HE2;
  TH2F* h_recNoiseEnergy1_HE2;
  TH2F* h_recNoiseEnergy2_HE2;
  TH2F* h_recNoiseEnergy0_HE3;
  TH2F* h_recNoiseEnergy1_HE3;
  TH2F* h_recNoiseEnergy2_HE3;
  TH2F* h_recNoiseEnergy0_HE4;
  TH2F* h_recNoiseEnergy1_HE4;
  TH2F* h_recNoiseEnergy2_HE4;
  TH2F* h_recNoiseEnergy0_HE5;
  TH2F* h_recNoiseEnergy1_HE5;
  TH2F* h_recNoiseEnergy2_HE5;
  TH2F* h_recNoiseEnergy0_HE6;
  TH2F* h_recNoiseEnergy1_HE6;
  TH2F* h_recNoiseEnergy2_HE6;
  TH2F* h_recNoiseEnergy0_HE7;
  TH2F* h_recNoiseEnergy1_HE7;
  TH2F* h_recNoiseEnergy2_HE7;
  //HF:
  TH2F* h_recSignalEnergy0_HF1;
  TH2F* h_recSignalEnergy1_HF1;
  TH2F* h_recSignalEnergy2_HF1;
  TH2F* h_recSignalEnergy0_HF2;
  TH2F* h_recSignalEnergy1_HF2;
  TH2F* h_recSignalEnergy2_HF2;
  TH2F* h_recSignalEnergy0_HF3;
  TH2F* h_recSignalEnergy1_HF3;
  TH2F* h_recSignalEnergy2_HF3;
  TH2F* h_recSignalEnergy0_HF4;
  TH2F* h_recSignalEnergy1_HF4;
  TH2F* h_recSignalEnergy2_HF4;

  TH2F* h_recNoiseEnergy0_HF1;
  TH2F* h_recNoiseEnergy1_HF1;
  TH2F* h_recNoiseEnergy2_HF1;
  TH2F* h_recNoiseEnergy0_HF2;
  TH2F* h_recNoiseEnergy1_HF2;
  TH2F* h_recNoiseEnergy2_HF2;
  TH2F* h_recNoiseEnergy0_HF3;
  TH2F* h_recNoiseEnergy1_HF3;
  TH2F* h_recNoiseEnergy2_HF3;
  TH2F* h_recNoiseEnergy0_HF4;
  TH2F* h_recNoiseEnergy1_HF4;
  TH2F* h_recNoiseEnergy2_HF4;

  TH2F* h_amplitudechannel0_HB1;
  TH2F* h_amplitudechannel1_HB1;
  TH2F* h_amplitudechannel2_HB1;
  TH2F* h_amplitudechannel0_HB2;
  TH2F* h_amplitudechannel1_HB2;
  TH2F* h_amplitudechannel2_HB2;
  TH2F* h_amplitudechannel0_HB3;
  TH2F* h_amplitudechannel1_HB3;
  TH2F* h_amplitudechannel2_HB3;
  TH2F* h_amplitudechannel0_HB4;
  TH2F* h_amplitudechannel1_HB4;
  TH2F* h_amplitudechannel2_HB4;
  TH2F* h_amplitudechannel0_HE1;
  TH2F* h_amplitudechannel1_HE1;
  TH2F* h_amplitudechannel2_HE1;
  TH2F* h_amplitudechannel0_HE2;
  TH2F* h_amplitudechannel1_HE2;
  TH2F* h_amplitudechannel2_HE2;
  TH2F* h_amplitudechannel0_HE3;
  TH2F* h_amplitudechannel1_HE3;
  TH2F* h_amplitudechannel2_HE3;
  TH2F* h_amplitudechannel0_HE4;
  TH2F* h_amplitudechannel1_HE4;
  TH2F* h_amplitudechannel2_HE4;
  TH2F* h_amplitudechannel0_HE5;
  TH2F* h_amplitudechannel1_HE5;
  TH2F* h_amplitudechannel2_HE5;
  TH2F* h_amplitudechannel0_HE6;
  TH2F* h_amplitudechannel1_HE6;
  TH2F* h_amplitudechannel2_HE6;
  TH2F* h_amplitudechannel0_HE7;
  TH2F* h_amplitudechannel1_HE7;
  TH2F* h_amplitudechannel2_HE7;
  TH2F* h_amplitudechannel0_HF1;
  TH2F* h_amplitudechannel1_HF1;
  TH2F* h_amplitudechannel2_HF1;
  TH2F* h_amplitudechannel0_HF2;
  TH2F* h_amplitudechannel1_HF2;
  TH2F* h_amplitudechannel2_HF2;
  TH2F* h_amplitudechannel0_HF3;
  TH2F* h_amplitudechannel1_HF3;
  TH2F* h_amplitudechannel2_HF3;
  TH2F* h_amplitudechannel0_HF4;
  TH2F* h_amplitudechannel1_HF4;
  TH2F* h_amplitudechannel2_HF4;

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
  TH2F* h2_TSnVsAyear2023_HB;
  TH2F* h2_TSnVsAyear2023_HE;
  TH2F* h2_TSnVsAyear2023_HF;
  TH2F* h2_TSnVsAyear2023_HO;
  TH1F* h1_TSnVsAyear2023_HB;
  TH1F* h1_TSnVsAyear2023_HE;
  TH1F* h1_TSnVsAyear2023_HF;
  TH1F* h1_TSnVsAyear2023_HO;
  TH1F* h1_TSnVsAyear20230_HB;
  TH1F* h1_TSnVsAyear20230_HE;
  TH1F* h1_TSnVsAyear20230_HF;
  TH1F* h1_TSnVsAyear20230_HO;

  int calibcapiderror[ndepth][neta][nphi];
  float calibt[ndepth][neta][nphi];
  double caliba[ndepth][neta][nphi];
  double calibw[ndepth][neta][nphi];
  double calib0[ndepth][neta][nphi];
  double signal[ndepth][neta][nphi];
  double calib3[ndepth][neta][nphi];
  double signal3[ndepth][neta][nphi];
  double calib2[ndepth][neta][nphi];
  int badchannels[nsub][ndepth][neta][nphi];
  double sumEstimator0[nsub][ndepth][neta][nphi];
  double sumEstimator1[nsub][ndepth][neta][nphi];
  double sumEstimator2[nsub][ndepth][neta][nphi];
  double sumEstimator3[nsub][ndepth][neta][nphi];
  double sumEstimator4[nsub][ndepth][neta][nphi];
  double sumEstimator5[nsub][ndepth][neta][nphi];
  double sumEstimator6[nsub][ndepth][neta][nphi];
  double sum0Estimator[nsub][ndepth][neta][nphi];

  // phi-symmetry monitoring for calibration group:
  double amplitudechannel0[nsub][ndepth][neta][nphi];
  double amplitudechannel[nsub][ndepth][neta][nphi];
  double amplitudechannel2[nsub][ndepth][neta][nphi];
  double tocamplchannel[nsub][ndepth][neta][nphi];
  double maprphinorm[nsub][ndepth][neta][nphi];

  double recNoiseEnergy0[nsub][ndepth][neta][nphi];
  double recNoiseEnergy1[nsub][ndepth][neta][nphi];
  double recNoiseEnergy2[nsub][ndepth][neta][nphi];

  double recSignalEnergy0[nsub][ndepth][neta][nphi];
  double recSignalEnergy1[nsub][ndepth][neta][nphi];
  double recSignalEnergy2[nsub][ndepth][neta][nphi];

  float TS_data[100];
  float TS_cal[100];
  double mapRADDAM_HE[ndepth][neta][nphi];
  int mapRADDAM0_HE[ndepth][neta][nphi];
  double mapRADDAM_HED2[ndepth][neta];
  int mapRADDAM_HED20[ndepth][neta];
  float binanpfit = anpfit / npfit;
  long int gsmdepth1sipm[npfit][neta][nphi][ndepth];
  /////////////////////////////////////////////
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
  std::ofstream MAPfile;
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

//
/////////////////////////////// -------------------------------------------------------------------
//
void CMTRawAnalyzer::endJob() {
  if (verbosity > 0)
    std::cout << "==============    endJob  ===================================" << std::endl;

  std::cout << " --------------------------------------- " << std::endl;
  std::cout << " for Run = " << run0 << " with runcounter = " << runcounter << " #ev = " << eventcounter << std::endl;
  std::cout << " #LS =  " << lscounterrun << " #LS10 =  " << lscounterrun10 << " Last LS =  " << ls0 << std::endl;
  std::cout << " --------------------------------------------- " << std::endl;
  h_nls_per_run->Fill(float(lscounterrun));
  h_nls_per_run10->Fill(float(lscounterrun10));

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
}
//

//
//
CMTRawAnalyzer::CMTRawAnalyzer(const edm::ParameterSet& iConfig)
    : tokDB_(esConsumes<HcalDbService, HcalDbRecord>()), tokTopo_(esConsumes<HcalTopology, HcalRecNumberingRecord>()) {
  usesResource(TFileService::kSharedResource);
  verbosity = iConfig.getUntrackedParameter<int>("Verbosity");
  MAPcreation = iConfig.getUntrackedParameter<int>("MapCreation");
  recordNtuples_ = iConfig.getUntrackedParameter<bool>("recordNtuples");
  maxNeventsInNtuple_ = iConfig.getParameter<int>("maxNeventsInNtuple");
  tok_calib_ = consumes<HcalCalibDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalCalibDigiCollectionTag"));  //
  tok_hbhe_ = consumes<HBHEDigiCollection>(iConfig.getParameter<edm::InputTag>("hbheDigiCollectionTag"));
  tok_ho_ = consumes<HODigiCollection>(iConfig.getParameter<edm::InputTag>("hoDigiCollectionTag"));
  tok_hf_ = consumes<HFDigiCollection>(iConfig.getParameter<edm::InputTag>("hfDigiCollectionTag"));  //
  tok_qie11_ = consumes<QIE11DigiCollection>(iConfig.getParameter<edm::InputTag>("hbheQIE11DigiCollectionTag"));
  tok_qie10_ = consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("hbheQIE10DigiCollectionTag"));
  // phi-symmetry monitoring for calibration group:
  tok_hbheSignal_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputSignalTag"));
  tok_hbheNoise_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputNoiseTag"));
  tok_hfSignal_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputSignalTag"));
  tok_hfNoise_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputNoiseTag"));
  recordHistoes_ = iConfig.getUntrackedParameter<bool>("recordHistoes");
  studyRunDependenceHist_ = iConfig.getUntrackedParameter<bool>("studyRunDependenceHist");
  studyCapIDErrorsHist_ = iConfig.getUntrackedParameter<bool>("studyCapIDErrorsHist");
  studyRMSshapeHist_ = iConfig.getUntrackedParameter<bool>("studyRMSshapeHist");
  studyRatioShapeHist_ = iConfig.getUntrackedParameter<bool>("studyRatioShapeHist");
  studyTSmaxShapeHist_ = iConfig.getUntrackedParameter<bool>("studyTSmaxShapeHist");
  studyTSmeanShapeHist_ = iConfig.getUntrackedParameter<bool>("studyTSmeanShapeHist");
  studyDiffAmplHist_ = iConfig.getUntrackedParameter<bool>("studyDiffAmplHist");
  studyCalibCellsHist_ = iConfig.getUntrackedParameter<bool>("studyCalibCellsHist");
  studyADCAmplHist_ = iConfig.getUntrackedParameter<bool>("studyADCAmplHist");
  studyPedestalsHist_ = iConfig.getUntrackedParameter<bool>("studyPedestalsHist");
  studyPedestalCorrelations_ = iConfig.getUntrackedParameter<bool>("studyPedestalCorrelations");
  useADCmassive_ = iConfig.getUntrackedParameter<bool>("useADCmassive");
  useADCfC_ = iConfig.getUntrackedParameter<bool>("useADCfC");
  useADCcounts_ = iConfig.getUntrackedParameter<bool>("useADCcounts");
  usePedestalSubtraction_ = iConfig.getUntrackedParameter<bool>("usePedestalSubtraction");
  usecontinuousnumbering_ = iConfig.getUntrackedParameter<bool>("usecontinuousnumbering");
  flagLaserRaddam_ = iConfig.getParameter<int>("flagLaserRaddam");
  flagToUseDigiCollectionsORNot_ = iConfig.getParameter<int>("flagToUseDigiCollectionsORNot");
  flagIterativeMethodCalibrationGroupDigi_ = iConfig.getParameter<int>("flagIterativeMethodCalibrationGroupDigi");
  flagIterativeMethodCalibrationGroupReco_ = iConfig.getParameter<int>("flagIterativeMethodCalibrationGroupReco");
  flagfitshunt1pedorledlowintensity_ = iConfig.getParameter<int>("flagfitshunt1pedorledlowintensity");
  flagabortgaprejected_ = iConfig.getParameter<int>("flagabortgaprejected");
  bcnrejectedlow_ = iConfig.getParameter<int>("bcnrejectedlow");
  bcnrejectedhigh_ = iConfig.getParameter<int>("bcnrejectedhigh");
  ratioHBMin_ = iConfig.getParameter<double>("ratioHBMin");
  ratioHBMax_ = iConfig.getParameter<double>("ratioHBMax");
  ratioHEMin_ = iConfig.getParameter<double>("ratioHEMin");
  ratioHEMax_ = iConfig.getParameter<double>("ratioHEMax");
  ratioHFMin_ = iConfig.getParameter<double>("ratioHFMin");
  ratioHFMax_ = iConfig.getParameter<double>("ratioHFMax");
  ratioHOMin_ = iConfig.getParameter<double>("ratioHOMin");
  ratioHOMax_ = iConfig.getParameter<double>("ratioHOMax");
  flagtodefinebadchannel_ = iConfig.getParameter<int>("flagtodefinebadchannel");
  howmanybinsonplots_ = iConfig.getParameter<int>("howmanybinsonplots");
  splashesUpperLimit_ = iConfig.getParameter<int>("splashesUpperLimit");
  flagtoaskrunsorls_ = iConfig.getParameter<int>("flagtoaskrunsorls");
  flagestimatornormalization_ = iConfig.getParameter<int>("flagestimatornormalization");
  flagcpuoptimization_ = iConfig.getParameter<int>("flagcpuoptimization");
  flagupgradeqie1011_ = iConfig.getParameter<int>("flagupgradeqie1011");
  flagsipmcorrection_ = iConfig.getParameter<int>("flagsipmcorrection");
  flaguseshunt_ = iConfig.getParameter<int>("flaguseshunt");
  lsdep_cut1_peak_HBdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HBdepth1");
  lsdep_cut1_peak_HBdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HBdepth2");
  lsdep_cut1_peak_HEdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth1");
  lsdep_cut1_peak_HEdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth2");
  lsdep_cut1_peak_HEdepth3_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth3");
  lsdep_cut1_peak_HFdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HFdepth1");
  lsdep_cut1_peak_HFdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HFdepth2");
  lsdep_cut1_peak_HOdepth4_ = iConfig.getParameter<int>("lsdep_cut1_peak_HOdepth4");
  lsdep_cut3_max_HBdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HBdepth1");
  lsdep_cut3_max_HBdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HBdepth2");
  lsdep_cut3_max_HEdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth1");
  lsdep_cut3_max_HEdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth2");
  lsdep_cut3_max_HEdepth3_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth3");
  lsdep_cut3_max_HFdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HFdepth1");
  lsdep_cut3_max_HFdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HFdepth2");
  lsdep_cut3_max_HOdepth4_ = iConfig.getParameter<int>("lsdep_cut3_max_HOdepth4");
  lsdep_estimator1_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth1");
  lsdep_estimator1_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth2");
  lsdep_estimator1_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth1");
  lsdep_estimator1_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth2");
  lsdep_estimator1_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth3");
  lsdep_estimator1_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth1");
  lsdep_estimator1_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth2");
  lsdep_estimator1_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HOdepth4");
  lsdep_estimator1_HEdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth4");
  lsdep_estimator1_HEdepth5_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth5");
  lsdep_estimator1_HEdepth6_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth6");
  lsdep_estimator1_HEdepth7_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth7");
  lsdep_estimator1_HFdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth3");
  lsdep_estimator1_HFdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth4");
  lsdep_estimator1_HBdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth3");
  lsdep_estimator1_HBdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth4");
  lsdep_estimator2_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HBdepth1");
  lsdep_estimator2_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HBdepth2");
  lsdep_estimator2_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth1");
  lsdep_estimator2_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth2");
  lsdep_estimator2_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth3");
  lsdep_estimator2_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HFdepth1");
  lsdep_estimator2_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HFdepth2");
  lsdep_estimator2_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator2_HOdepth4");
  lsdep_estimator3_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HBdepth1");
  lsdep_estimator3_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HBdepth2");
  lsdep_estimator3_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth1");
  lsdep_estimator3_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth2");
  lsdep_estimator3_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth3");
  lsdep_estimator3_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HFdepth1");
  lsdep_estimator3_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HFdepth2");
  lsdep_estimator3_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator3_HOdepth4");
  lsdep_estimator4_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HBdepth1");
  lsdep_estimator4_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HBdepth2");
  lsdep_estimator4_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth1");
  lsdep_estimator4_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth2");
  lsdep_estimator4_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth3");
  lsdep_estimator4_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HFdepth1");
  lsdep_estimator4_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HFdepth2");
  lsdep_estimator4_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator4_HOdepth4");
  lsdep_estimator5_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HBdepth1");
  lsdep_estimator5_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HBdepth2");
  lsdep_estimator5_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth1");
  lsdep_estimator5_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth2");
  lsdep_estimator5_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth3");
  lsdep_estimator5_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HFdepth1");
  lsdep_estimator5_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HFdepth2");
  lsdep_estimator5_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator5_HOdepth4");
  forallestimators_amplitude_bigger_ = iConfig.getParameter<double>("forallestimators_amplitude_bigger");
  rmsHBMin_ = iConfig.getParameter<double>("rmsHBMin");
  rmsHBMax_ = iConfig.getParameter<double>("rmsHBMax");
  rmsHEMin_ = iConfig.getParameter<double>("rmsHEMin");
  rmsHEMax_ = iConfig.getParameter<double>("rmsHEMax");
  rmsHFMin_ = iConfig.getParameter<double>("rmsHFMin");
  rmsHFMax_ = iConfig.getParameter<double>("rmsHFMax");
  rmsHOMin_ = iConfig.getParameter<double>("rmsHOMin");
  rmsHOMax_ = iConfig.getParameter<double>("rmsHOMax");
  ADCAmplHBMin_ = iConfig.getParameter<double>("ADCAmplHBMin");
  ADCAmplHEMin_ = iConfig.getParameter<double>("ADCAmplHEMin");
  ADCAmplHOMin_ = iConfig.getParameter<double>("ADCAmplHOMin");
  ADCAmplHFMin_ = iConfig.getParameter<double>("ADCAmplHFMin");
  ADCAmplHBMax_ = iConfig.getParameter<double>("ADCAmplHBMax");
  ADCAmplHEMax_ = iConfig.getParameter<double>("ADCAmplHEMax");
  ADCAmplHOMax_ = iConfig.getParameter<double>("ADCAmplHOMax");
  ADCAmplHFMax_ = iConfig.getParameter<double>("ADCAmplHFMax");
  pedestalwHBMax_ = iConfig.getParameter<double>("pedestalwHBMax");
  pedestalwHEMax_ = iConfig.getParameter<double>("pedestalwHEMax");
  pedestalwHFMax_ = iConfig.getParameter<double>("pedestalwHFMax");
  pedestalwHOMax_ = iConfig.getParameter<double>("pedestalwHOMax");
  pedestalHBMax_ = iConfig.getParameter<double>("pedestalHBMax");
  pedestalHEMax_ = iConfig.getParameter<double>("pedestalHEMax");
  pedestalHFMax_ = iConfig.getParameter<double>("pedestalHFMax");
  pedestalHOMax_ = iConfig.getParameter<double>("pedestalHOMax");
  calibrADCHBMin_ = iConfig.getParameter<double>("calibrADCHBMin");
  calibrADCHEMin_ = iConfig.getParameter<double>("calibrADCHEMin");
  calibrADCHOMin_ = iConfig.getParameter<double>("calibrADCHOMin");
  calibrADCHFMin_ = iConfig.getParameter<double>("calibrADCHFMin");
  calibrADCHBMax_ = iConfig.getParameter<double>("calibrADCHBMax");
  calibrADCHEMax_ = iConfig.getParameter<double>("calibrADCHEMax");
  calibrADCHOMax_ = iConfig.getParameter<double>("calibrADCHOMax");
  calibrADCHFMax_ = iConfig.getParameter<double>("calibrADCHFMax");
  calibrRatioHBMin_ = iConfig.getParameter<double>("calibrRatioHBMin");
  calibrRatioHEMin_ = iConfig.getParameter<double>("calibrRatioHEMin");
  calibrRatioHOMin_ = iConfig.getParameter<double>("calibrRatioHOMin");
  calibrRatioHFMin_ = iConfig.getParameter<double>("calibrRatioHFMin");
  calibrRatioHBMax_ = iConfig.getParameter<double>("calibrRatioHBMax");
  calibrRatioHEMax_ = iConfig.getParameter<double>("calibrRatioHEMax");
  calibrRatioHOMax_ = iConfig.getParameter<double>("calibrRatioHOMax");
  calibrRatioHFMax_ = iConfig.getParameter<double>("calibrRatioHFMax");
  calibrTSmaxHBMin_ = iConfig.getParameter<double>("calibrTSmaxHBMin");
  calibrTSmaxHEMin_ = iConfig.getParameter<double>("calibrTSmaxHEMin");
  calibrTSmaxHOMin_ = iConfig.getParameter<double>("calibrTSmaxHOMin");
  calibrTSmaxHFMin_ = iConfig.getParameter<double>("calibrTSmaxHFMin");
  calibrTSmaxHBMax_ = iConfig.getParameter<double>("calibrTSmaxHBMax");
  calibrTSmaxHEMax_ = iConfig.getParameter<double>("calibrTSmaxHEMax");
  calibrTSmaxHOMax_ = iConfig.getParameter<double>("calibrTSmaxHOMax");
  calibrTSmaxHFMax_ = iConfig.getParameter<double>("calibrTSmaxHFMax");
  calibrTSmeanHBMin_ = iConfig.getParameter<double>("calibrTSmeanHBMin");
  calibrTSmeanHEMin_ = iConfig.getParameter<double>("calibrTSmeanHEMin");
  calibrTSmeanHOMin_ = iConfig.getParameter<double>("calibrTSmeanHOMin");
  calibrTSmeanHFMin_ = iConfig.getParameter<double>("calibrTSmeanHFMin");
  calibrTSmeanHBMax_ = iConfig.getParameter<double>("calibrTSmeanHBMax");
  calibrTSmeanHEMax_ = iConfig.getParameter<double>("calibrTSmeanHEMax");
  calibrTSmeanHOMax_ = iConfig.getParameter<double>("calibrTSmeanHOMax");
  calibrTSmeanHFMax_ = iConfig.getParameter<double>("calibrTSmeanHFMax");
  calibrWidthHBMin_ = iConfig.getParameter<double>("calibrWidthHBMin");
  calibrWidthHEMin_ = iConfig.getParameter<double>("calibrWidthHEMin");
  calibrWidthHOMin_ = iConfig.getParameter<double>("calibrWidthHOMin");
  calibrWidthHFMin_ = iConfig.getParameter<double>("calibrWidthHFMin");
  calibrWidthHBMax_ = iConfig.getParameter<double>("calibrWidthHBMax");
  calibrWidthHEMax_ = iConfig.getParameter<double>("calibrWidthHEMax");
  calibrWidthHOMax_ = iConfig.getParameter<double>("calibrWidthHOMax");
  calibrWidthHFMax_ = iConfig.getParameter<double>("calibrWidthHFMax");
  TSpeakHBMin_ = iConfig.getParameter<double>("TSpeakHBMin");
  TSpeakHBMax_ = iConfig.getParameter<double>("TSpeakHBMax");
  TSpeakHEMin_ = iConfig.getParameter<double>("TSpeakHEMin");
  TSpeakHEMax_ = iConfig.getParameter<double>("TSpeakHEMax");
  TSpeakHFMin_ = iConfig.getParameter<double>("TSpeakHFMin");
  TSpeakHFMax_ = iConfig.getParameter<double>("TSpeakHFMax");
  TSpeakHOMin_ = iConfig.getParameter<double>("TSpeakHOMin");
  TSpeakHOMax_ = iConfig.getParameter<double>("TSpeakHOMax");
  TSmeanHBMin_ = iConfig.getParameter<double>("TSmeanHBMin");
  TSmeanHBMax_ = iConfig.getParameter<double>("TSmeanHBMax");
  TSmeanHEMin_ = iConfig.getParameter<double>("TSmeanHEMin");
  TSmeanHEMax_ = iConfig.getParameter<double>("TSmeanHEMax");
  TSmeanHFMin_ = iConfig.getParameter<double>("TSmeanHFMin");
  TSmeanHFMax_ = iConfig.getParameter<double>("TSmeanHFMax");
  TSmeanHOMin_ = iConfig.getParameter<double>("TSmeanHOMin");
  TSmeanHOMax_ = iConfig.getParameter<double>("TSmeanHOMax");
  lsmin_ = iConfig.getParameter<int>("lsmin");
  lsmax_ = iConfig.getParameter<int>("lsmax");
  alsmin = lsmin_;
  blsmax = lsmax_;
  nlsminmax = lsmax_ - lsmin_ + 1;
  numOfLaserEv = 0;
  local_event = 0;
  numOfTS = 10;
  run0 = -1;
  runcounter = 0;
  eventcounter = 0;
  lumi = 0;
  ls0 = -1;
  lscounter = 0;
  lscounterM1 = 0;
  lscounter10 = 0;
  nevcounter = 0;
  lscounterrun = 0;
  lscounterrun10 = 0;
  nevcounter0 = 0;
  nevcounter00 = 0;
  for (int k0 = 0; k0 < nsub; k0++) {
    for (int k1 = 0; k1 < ndepth; k1++) {
      for (int k2 = 0; k2 < neta; k2++) {
        if (k0 == 1) {
          mapRADDAM_HED2[k1][k2] = 0.;
          mapRADDAM_HED20[k1][k2] = 0.;
        }
        for (int k3 = 0; k3 < nphi; k3++) {
          sumEstimator0[k0][k1][k2][k3] = 0.;
          sumEstimator1[k0][k1][k2][k3] = 0.;
          sumEstimator2[k0][k1][k2][k3] = 0.;
          sumEstimator3[k0][k1][k2][k3] = 0.;
          sumEstimator4[k0][k1][k2][k3] = 0.;
          sumEstimator5[k0][k1][k2][k3] = 0.;
          sumEstimator6[k0][k1][k2][k3] = 0.;
          sum0Estimator[k0][k1][k2][k3] = 0.;
          if (k0 == 1) {
            mapRADDAM_HE[k1][k2][k3] = 0.;
            mapRADDAM0_HE[k1][k2][k3] = 0;
          }
        }
      }
    }
  }
  averSIGNALoccupancy_HB = 0.;
  averSIGNALoccupancy_HE = 0.;
  averSIGNALoccupancy_HF = 0.;
  averSIGNALoccupancy_HO = 0.;
  averSIGNALsumamplitude_HB = 0.;
  averSIGNALsumamplitude_HE = 0.;
  averSIGNALsumamplitude_HF = 0.;
  averSIGNALsumamplitude_HO = 0.;
  averNOSIGNALoccupancy_HB = 0.;
  averNOSIGNALoccupancy_HE = 0.;
  averNOSIGNALoccupancy_HF = 0.;
  averNOSIGNALoccupancy_HO = 0.;
  averNOSIGNALsumamplitude_HB = 0.;
  averNOSIGNALsumamplitude_HE = 0.;
  averNOSIGNALsumamplitude_HF = 0.;
  averNOSIGNALsumamplitude_HO = 0.;
  maxxSUM1 = 0.;
  maxxSUM2 = 0.;
  maxxSUM3 = 0.;
  maxxSUM4 = 0.;
  maxxOCCUP1 = 0.;
  maxxOCCUP2 = 0.;
  maxxOCCUP3 = 0.;
  maxxOCCUP4 = 0.;
  testmetka = 0;
}
CMTRawAnalyzer::~CMTRawAnalyzer() {}
void CMTRawAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  conditions = &iSetup.getData(tokDB_);
  topo = &iSetup.getData(tokTopo_);
  if (MAPcreation > 0) {
    if (flagupgradeqie1011_ == 1)
      fillMAP();
    MAPcreation = 0;
  }
  nevent++;
  nevent50 = nevent / 50;
  Run = iEvent.id().run();
  Nevent = iEvent.id().event();     // event number = global_event
  lumi = iEvent.luminosityBlock();  // lumi section
  bcn = iEvent.bunchCrossing();
  orbitNum = iEvent.orbitNumber();
  int outabortgap = 1;
  if (bcn >= bcnrejectedlow_ && bcn <= bcnrejectedhigh_)
    outabortgap = 0;  //  if(bcn>=3446 && bcn<=3564)

  if ((flagabortgaprejected_ == 1 && outabortgap == 1) || (flagabortgaprejected_ == 0 && outabortgap == 0) ||
      flagabortgaprejected_ == 2) {
    if (run0 != Run) {
      ++runcounter;
      if (runcounter != 1) {
        nevcounter00 = eventcounter;
        std::cout << " --------------------------------------- " << std::endl;
        std::cout << " for Run = " << run0 << " with runcounter = " << runcounter - 1 << " #ev = " << eventcounter
                  << std::endl;
        std::cout << " #LS =  " << lscounterrun << " #LS10 =  " << lscounterrun10 << " Last LS =  " << ls0 << std::endl;
        std::cout << " --------------------------------------------- " << std::endl;
        h_nls_per_run->Fill(float(lscounterrun));
        h_nls_per_run10->Fill(float(lscounterrun10));
        lscounterrun = 0;
        lscounterrun10 = 0;
      }  // runcounter > 1
      std::cout << " ---------***********************------------- " << std::endl;
      std::cout << " New Run =  " << Run << " runcounter =  " << runcounter << std::endl;
      std::cout << " ------- " << std::endl;
      run0 = Run;
      eventcounter = 0;
      ls0 = -1;
    }  // new run
    else {
      nevcounter00 = 0;
    }  //else new run
    ++eventcounter;
    if (ls0 != lumi) {
      if (ls0 != -1) {
        h_nevents_per_eachLS->Fill(float(lscounter), float(nevcounter));  //
        nevcounter0 = nevcounter;
      }  // ls0>-1
      lscounter++;
      lscounterrun++;
      if (usecontinuousnumbering_) {
        lscounterM1 = lscounter - 1;
      } else {
        lscounterM1 = ls0;
      }
      if (ls0 != -1)
        h_nevents_per_eachRealLS->Fill(float(lscounterM1), float(nevcounter));  //
      h_lsnumber_per_eachLS->Fill(float(lscounter), float(lumi));
      if (nevcounter > 10.) {
        ++lscounter10;
        ++lscounterrun10;
      }
      h_nevents_per_LS->Fill(float(nevcounter));
      h_nevents_per_LSzoom->Fill(float(nevcounter));
      nevcounter = 0;
      ls0 = lumi;
    }  // new lumi
    else {
      nevcounter0 = 0;
    }              //else new lumi
    ++nevcounter;  // #ev in LS
                   //////
    if (flagtoaskrunsorls_ == 0) {
      lscounterM1 = runcounter;
      nevcounter0 = nevcounter00;
    }
    if (nevcounter0 != 0 || nevcounter > 99999) {
      if (nevcounter > 99999)
        nevcounter0 = 1;
      ///////  int sub= cell.subdet();  1-HB, 2-HE, 3-HO, 4-HF
      ////////////            k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      ////////////         k1(depth-1): = 0 - 3 or depth: = 1 - 4;
      unsigned long int pcountall1 = 0;
      unsigned long int pcountall3 = 0;
      unsigned long int pcountall6 = 0;
      unsigned long int pcountall8 = 0;
      int pcountmin1 = 0;
      int pcountmin3 = 0;
      int pcountmin6 = 0;
      int pcountmin8 = 0;
      unsigned long int mcountall1 = 0;
      unsigned long int mcountall3 = 0;
      unsigned long int mcountall6 = 0;
      unsigned long int mcountall8 = 0;
      int mcountmin1 = 0;
      int mcountmin3 = 0;
      int mcountmin6 = 0;
      int mcountmin8 = 0;
      int pnnmin1 = 999999999;
      int pnnmin3 = 999999999;
      int pnnmin6 = 999999999;
      int pnnmin8 = 999999999;
      int mnnbins1 = 0;
      int mnnbins3 = 0;
      int mnnbins6 = 0;
      int mnnbins8 = 0;
      int mnnmin1 = 999999999;
      int mnnmin3 = 999999999;
      int mnnmin6 = 999999999;
      int mnnmin8 = 999999999;
      for (int k0 = 0; k0 < nsub; k0++) {
        for (int k1 = 0; k1 < ndepth; k1++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            for (int k2 = 0; k2 < neta; k2++) {
              int ieta = k2 - 41;
              // ------------------------------------------------------------sumEstimator0
              if (sumEstimator0[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator0[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator0[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator0[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS1->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS1->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS2->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS3->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS4->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumPedestalLS5->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS6->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS7->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumPedestalLS8->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator0[k0][k1][k2][k3] != 0.

              // -------------------------------------------------------------------------------------   sumEstimator1
              if (sumEstimator1[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator1[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator1[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator1[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }
                //flag for ask type of Normalization for CMT estimators:
                //=0-normalizationOn#evOfLS;   =1-averagedMeanChannelVariable;   =2-averageVariable-normalizationOn#entriesInLS;
                //flagestimatornormalization = cms.int32(2), !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                // zhokin 18.10.2018 STUDY:               CALL  HFF2 (ID,NID,X,Y,W)
                if (lscounterM1 >= lsmin_ && lscounterM1 < lsmax_) {
                  //                                       INDEXIES:
                  int kkkk2 = (k2 - 1) / 4;
                  if (k2 == 0)
                    kkkk2 = 1.;
                  else
                    kkkk2 += 2;              //kkkk2= 1-22
                  int kkkk3 = (k3) / 4 + 1;  //kkkk3= 1-18
                  //                                       PACKING
                  //kkkk2= 1-22 ;kkkk3= 1-18
                  int ietaphi = 0;
                  ietaphi = ((kkkk2)-1) * znphi + (kkkk3);
                  //  Outout is       ietaphi = 1 - 396 ( # =396; in histo,booking is: 1 - 397 )

                  double bbb3 = 0.;
                  if (bbb1 != 0.)
                    bbb3 = bbbc / bbb1;
                  // very very wrong if below:
                  //		if(bbb3 != 0.) {

                  if (k0 == 0) {
                    h_2DsumADCAmplEtaPhiLs0->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HB
                    h_2DsumADCAmplEtaPhiLs00->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HB
                  }
                  if (k0 == 1) {
                    h_2DsumADCAmplEtaPhiLs1->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HE
                    h_2DsumADCAmplEtaPhiLs10->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HE
                  }
                  if (k0 == 2) {
                    h_2DsumADCAmplEtaPhiLs2->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HO
                    h_2DsumADCAmplEtaPhiLs20->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HO
                  }
                  if (k0 == 3) {
                    h_2DsumADCAmplEtaPhiLs3->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HF
                    h_2DsumADCAmplEtaPhiLs30->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HF
                  }

                  h_sumADCAmplEtaPhiLs->Fill(bbb3);
                  h_sumADCAmplEtaPhiLs_bbbc->Fill(bbbc);
                  h_sumADCAmplEtaPhiLs_bbb1->Fill(bbb1);
                  h_sumADCAmplEtaPhiLs_lscounterM1orbitNum->Fill(float(lscounterM1), float(orbitNum));
                  h_sumADCAmplEtaPhiLs_orbitNum->Fill(float(orbitNum), 1.);
                  h_sumADCAmplEtaPhiLs_lscounterM1->Fill(float(lscounterM1), 1.);
                  h_sumADCAmplEtaPhiLs_ietaphi->Fill(float(ietaphi));

                  //		}// bb3
                }  // lscounterM1 >= lsmin_ && lscounterM1 < lsmax_

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS1copy1->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy2->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy3->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy4->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy5->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                      h_2DsumADCAmplLS1->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth1_)
                      h_2DsumADCAmplLS1_LSselected->Fill(double(ieta), double(k3), bbbc);

                    h_2D0sumADCAmplLS1->Fill(double(ieta), double(k3), bbb1);

                    h_sumADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                      h_sumCutADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS1->Fill(float(lscounterM1), bbb1);

                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 25.) {
                        pcountall1 += bbb1;
                        pcountmin1 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 25.) {
                        mcountall1 += bbb1;
                        mcountmin1 += bbb1;
                      }
                      //////////////////////////////
                    }
                  }
                  // HBdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                      h_2DsumADCAmplLS2->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth2_)
                      h_2DsumADCAmplLS2_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                      h_sumCutADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS2->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }
                  // HBdepth3 upgrade
                  if (k1 + 1 == 3) {
                    h_sumADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                      h_sumCutADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                      h_2DsumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==3)

                  // HBdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                      h_sumCutADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                      h_2DsumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)
                }

                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                      h_2DsumADCAmplLS3->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth1_)
                      h_2DsumADCAmplLS3_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                      h_sumCutADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS3->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 15. && k3 % 2 == 0) {
                        pcountall3 += bbb1;
                        pcountmin3 += bbb1;
                      }
                      //////////////////////////////
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 15. && k3 % 2 == 0) {
                        mcountall3 += bbb1;
                        mcountmin3 += bbb1;
                      }
                      //////////////////////////////
                    }
                  }
                  // HEdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                      h_2DsumADCAmplLS4->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth2_)
                      h_2DsumADCAmplLS4_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                      h_sumCutADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS4->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }
                  // HEdepth3
                  if (k1 + 1 == 3) {
                    h_sumADCAmplLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                      h_2DsumADCAmplLS5->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth3_)
                      h_2DsumADCAmplLS5_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                      h_sumCutADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS5->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }  //if(k1+1  ==3
                  // HEdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                      h_sumCutADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                      h_2DsumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)

                  // HEdepth5 upgrade
                  if (k1 + 1 == 5) {
                    h_sumADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                      h_sumCutADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                      h_2DsumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==5)

                  // HEdepth6 upgrade
                  if (k1 + 1 == 6) {
                    h_sumADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                      h_sumCutADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                      h_2DsumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==6)

                  // HEdepth7 upgrade
                  if (k1 + 1 == 7) {
                    h_sumADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                      h_sumCutADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                      h_2DsumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==7)
                }    //if(k0==1) =HE
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                      h_2DsumADCAmplLS6->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth1_)
                      h_2DsumADCAmplLS6_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                      h_sumCutADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS6->Fill(float(lscounterM1), bbb1);

                    ///////////////////////////////////////////////////////// error-A
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 20.) {
                        pcountall6 += bbb1;
                        pcountmin6 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 20.) {
                        mcountall6 += bbb1;
                        mcountmin6 += bbb1;
                      }
                      //////////////////////////////
                    }
                    /////////////////////////////////////////////////////////
                  }  //if(k1+1  ==1)

                  // HFdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                      h_2DsumADCAmplLS7->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth2_)
                      h_2DsumADCAmplLS7_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                      h_sumCutADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS7->Fill(float(lscounterM1), bbb1);

                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }  //if(k1+1  ==2)

                  // HFdepth3 upgrade
                  if (k1 + 1 == 3) {
                    h_sumADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                      h_sumCutADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS6u->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                      h_2DsumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==3)

                  // HFdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                      h_sumCutADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS7u->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                      h_2DsumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)

                }  //end HF

                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumADCAmplLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                      h_2DsumADCAmplLS8->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HOdepth4_)
                      h_2DsumADCAmplLS8_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                      h_sumCutADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS8->Fill(float(lscounterM1), bbb1);

                    ///////////////////////////////////////////////////////// error-A
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS8_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS8_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 80.) {
                        pcountall8 += bbb1;
                        pcountmin8 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS8_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS8_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 80.) {
                        mcountall8 += bbb1;
                        mcountmin8 += bbb1;
                      }
                      //////////////////////////////
                    }
                    /////////////////////////////////////////////////////////
                  }
                }
              }  //if(sumEstimator1[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator2
              if (sumEstimator2[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator2[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator2[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator2[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                      h_2DsumTSmeanALS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                      h_sumCutTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator2_HBdepth1_)
                      h_sumTSmeanAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                      h_2DsumTSmeanALS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                      h_sumCutTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                      h_2DsumTSmeanALS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                      h_sumCutTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                      h_2DsumTSmeanALS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                      h_sumCutTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumTSmeanALS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                      h_2DsumTSmeanALS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                      h_sumCutTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                      h_2DsumTSmeanALS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                      h_sumCutTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                      h_2DsumTSmeanALS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                      h_sumCutTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumTSmeanALS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                      h_2DsumTSmeanALS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                      h_sumCutTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator2[k0][k1][k2][k3] != 0.

              // ------------------------------------------------------------------------------------------------------------------------sumEstimator3
              if (sumEstimator3[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator3[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator3[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator3[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                      h_2DsumTSmaxALS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                      h_sumCutTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator3_HBdepth1_)
                      h_sumTSmaxAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                      h_2DsumTSmaxALS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                      h_sumCutTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                      h_2DsumTSmaxALS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                      h_sumCutTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                      h_2DsumTSmaxALS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                      h_sumCutTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumTSmaxALS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                      h_2DsumTSmaxALS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                      h_sumCutTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                      h_2DsumTSmaxALS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                      h_sumCutTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                      h_2DsumTSmaxALS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                      h_sumCutTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumTSmaxALS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                      h_2DsumTSmaxALS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                      h_sumCutTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator3[k0][k1][k2][k3] != 0.

              // ------------------------------------------------------------------------------------------------------------------------sumEstimator4
              if (sumEstimator4[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator4[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator4[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator4[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                      h_2DsumAmplitudeLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                      h_sumCutAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator4_HBdepth1_)
                      h_sumAmplitudeperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                      h_2DsumAmplitudeLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                      h_sumCutAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                      h_2DsumAmplitudeLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                      h_sumCutAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                      h_2DsumAmplitudeLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                      h_sumCutAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumAmplitudeLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                      h_2DsumAmplitudeLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                      h_sumCutAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                      h_2DsumAmplitudeLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                      h_sumCutAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                      h_2DsumAmplitudeLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                      h_sumCutAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumAmplitudeLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                      h_2DsumAmplitudeLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                      h_sumCutAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator4[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator5
              if (sumEstimator5[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator5[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator5[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator5[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                      h_2DsumAmplLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                      h_sumCutAmplperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator5_HBdepth1_)
                      h_sumAmplperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                      h_2DsumAmplLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                      h_sumCutAmplperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                      h_2DsumAmplLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                      h_sumCutAmplperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                      h_2DsumAmplLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                      h_sumCutAmplperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumAmplLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                      h_2DsumAmplLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                      h_sumCutAmplperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                      h_2DsumAmplLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                      h_sumCutAmplperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                      h_2DsumAmplLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                      h_sumCutAmplperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumAmplLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                      h_2DsumAmplLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                      h_sumCutAmplperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator5[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator6 (Error-B)
              if (sumEstimator6[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator6[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator6[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator6[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS1->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS1->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS2->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS3->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS4->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumErrorBLS5->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS6->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS7->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumErrorBLS8->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
                ///
              }  //if(sumEstimator6[k0][k1][k2][k3] != 0.
            }    //for k2
            // occupancy distributions for error-A:
            // HB
            if (k0 == 0 && k1 == 0) {
              if (pcountmin1 > 0) {
                if (pcountmin1 < pnnmin1)
                  pnnmin1 = pcountmin1;
                pcountmin1 = 0;
              }
              if (mcountmin1 > 0) {
                if (mcountmin1 < mnnmin1)
                  mnnmin1 = mcountmin1;
                mcountmin1 = 0;
                mnnbins1++;
              }
            }  //
            // HE
            if (k0 == 1 && k1 == 0) {
              if (pcountmin3 > 0) {
                if (pcountmin3 < pnnmin3)
                  pnnmin3 = pcountmin3;
                pcountmin3 = 0;
              }
              if (mcountmin3 > 0) {
                if (mcountmin3 < mnnmin3)
                  mnnmin3 = mcountmin3;
                mcountmin3 = 0;
                mnnbins3++;
              }
            }  //
            // HO
            if (k0 == 2 && k1 == 3) {
              if (pcountmin8 > 0) {
                if (pcountmin8 < pnnmin8)
                  pnnmin8 = pcountmin8;
                pcountmin8 = 0;
              }
              if (mcountmin8 > 0) {
                if (mcountmin8 < mnnmin8)
                  mnnmin8 = mcountmin8;
                mcountmin8 = 0;
                mnnbins8++;
              }
            }  //
            // HF
            if (k0 == 3 && k1 == 0) {
              if (pcountmin6 > 0) {
                if (pcountmin6 < pnnmin6)
                  pnnmin6 = pcountmin6;
                pcountmin6 = 0;
              }
              if (mcountmin6 > 0) {
                if (mcountmin6 < mnnmin6)
                  mnnmin6 = mcountmin6;
                mcountmin6 = 0;
                mnnbins6++;
              }
            }  //

          }  //for k3
        }    //for k1
      }      //for k0
      ///////  int sub= cell.subdet();  1-HB, 2-HE, 3-HO, 4-HF
      ////////////            k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      ////////////         k1(depth-1): = 0 - 3 or depth: = 1 - 4;

      //   cout<<"=============================== lscounterM1 = "<<   (float)lscounterM1    <<endl;

      float patiooccupancy1 = 0.;
      if (pcountall1 != 0)
        patiooccupancy1 = (float)pnnmin1 * mnnbins1 / pcountall1;
      h_RatioOccupancy_HBM->Fill(float(lscounterM1), patiooccupancy1);
      float matiooccupancy1 = 0.;
      if (mcountall1 != 0)
        matiooccupancy1 = (float)mnnmin1 * mnnbins1 / mcountall1;
      h_RatioOccupancy_HBP->Fill(float(lscounterM1), matiooccupancy1);

      float patiooccupancy3 = 0.;
      if (pcountall3 != 0)
        patiooccupancy3 = (float)pnnmin3 * mnnbins3 / pcountall3;
      h_RatioOccupancy_HEM->Fill(float(lscounterM1), patiooccupancy3);
      float matiooccupancy3 = 0.;
      if (mcountall3 != 0)
        matiooccupancy3 = (float)mnnmin3 * mnnbins3 / mcountall3;
      h_RatioOccupancy_HEP->Fill(float(lscounterM1), matiooccupancy3);

      float patiooccupancy6 = 0.;
      if (pcountall6 != 0)
        patiooccupancy6 = (float)pnnmin6 * mnnbins6 / pcountall6;
      h_RatioOccupancy_HFM->Fill(float(lscounterM1), patiooccupancy6);
      float matiooccupancy6 = 0.;
      if (mcountall6 != 0)
        matiooccupancy6 = (float)mnnmin6 * mnnbins6 / mcountall6;
      h_RatioOccupancy_HFP->Fill(float(lscounterM1), matiooccupancy6);

      float patiooccupancy8 = 0.;
      if (pcountall8 != 0)
        patiooccupancy8 = (float)pnnmin8 * mnnbins8 / pcountall8;
      h_RatioOccupancy_HOM->Fill(float(lscounterM1), patiooccupancy8);
      float matiooccupancy8 = 0.;
      if (mcountall8 != 0)
        matiooccupancy8 = (float)mnnmin8 * mnnbins8 / mcountall8;
      h_RatioOccupancy_HOP->Fill(float(lscounterM1), matiooccupancy8);

      for (int k0 = 0; k0 < nsub; k0++) {
        for (int k1 = 0; k1 < ndepth; k1++) {
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              // reset massives:
              sumEstimator0[k0][k1][k2][k3] = 0.;
              sumEstimator1[k0][k1][k2][k3] = 0.;
              sumEstimator2[k0][k1][k2][k3] = 0.;
              sumEstimator3[k0][k1][k2][k3] = 0.;
              sumEstimator4[k0][k1][k2][k3] = 0.;
              sumEstimator5[k0][k1][k2][k3] = 0.;
              sumEstimator6[k0][k1][k2][k3] = 0.;
              sum0Estimator[k0][k1][k2][k3] = 0.;
            }  //for
          }    //for
        }      //for
      }        //for

      //------------------------------------------------------                        averSIGNAL
      averSIGNALoccupancy_HB /= float(nevcounter0);
      h_averSIGNALoccupancy_HB->Fill(float(lscounterM1), averSIGNALoccupancy_HB);
      averSIGNALoccupancy_HE /= float(nevcounter0);
      h_averSIGNALoccupancy_HE->Fill(float(lscounterM1), averSIGNALoccupancy_HE);
      averSIGNALoccupancy_HF /= float(nevcounter0);
      h_averSIGNALoccupancy_HF->Fill(float(lscounterM1), averSIGNALoccupancy_HF);
      averSIGNALoccupancy_HO /= float(nevcounter0);
      h_averSIGNALoccupancy_HO->Fill(float(lscounterM1), averSIGNALoccupancy_HO);

      averSIGNALoccupancy_HB = 0.;
      averSIGNALoccupancy_HE = 0.;
      averSIGNALoccupancy_HF = 0.;
      averSIGNALoccupancy_HO = 0.;

      //------------------------------------------------------
      averSIGNALsumamplitude_HB /= float(nevcounter0);
      h_averSIGNALsumamplitude_HB->Fill(float(lscounterM1), averSIGNALsumamplitude_HB);
      averSIGNALsumamplitude_HE /= float(nevcounter0);
      h_averSIGNALsumamplitude_HE->Fill(float(lscounterM1), averSIGNALsumamplitude_HE);
      averSIGNALsumamplitude_HF /= float(nevcounter0);
      h_averSIGNALsumamplitude_HF->Fill(float(lscounterM1), averSIGNALsumamplitude_HF);
      averSIGNALsumamplitude_HO /= float(nevcounter0);
      h_averSIGNALsumamplitude_HO->Fill(float(lscounterM1), averSIGNALsumamplitude_HO);

      averSIGNALsumamplitude_HB = 0.;
      averSIGNALsumamplitude_HE = 0.;
      averSIGNALsumamplitude_HF = 0.;
      averSIGNALsumamplitude_HO = 0.;

      //------------------------------------------------------                        averNOSIGNAL
      averNOSIGNALoccupancy_HB /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HB->Fill(float(lscounterM1), averNOSIGNALoccupancy_HB);
      averNOSIGNALoccupancy_HE /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HE->Fill(float(lscounterM1), averNOSIGNALoccupancy_HE);
      averNOSIGNALoccupancy_HF /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HF->Fill(float(lscounterM1), averNOSIGNALoccupancy_HF);
      averNOSIGNALoccupancy_HO /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HO->Fill(float(lscounterM1), averNOSIGNALoccupancy_HO);

      averNOSIGNALoccupancy_HB = 0.;
      averNOSIGNALoccupancy_HE = 0.;
      averNOSIGNALoccupancy_HF = 0.;
      averNOSIGNALoccupancy_HO = 0.;

      //------------------------------------------------------
      averNOSIGNALsumamplitude_HB /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HB->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HB);
      averNOSIGNALsumamplitude_HE /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HE->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HE);
      averNOSIGNALsumamplitude_HF /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HF->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HF);
      averNOSIGNALsumamplitude_HO /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HO->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HO);

      averNOSIGNALsumamplitude_HB = 0.;
      averNOSIGNALsumamplitude_HE = 0.;
      averNOSIGNALsumamplitude_HF = 0.;
      averNOSIGNALsumamplitude_HO = 0.;

      //------------------------------------------------------   maxxSA and maxxOccupancy
      h_maxxSUMAmpl_HB->Fill(float(lscounterM1), maxxSUM1);
      h_maxxSUMAmpl_HE->Fill(float(lscounterM1), maxxSUM2);
      h_maxxSUMAmpl_HO->Fill(float(lscounterM1), maxxSUM3);
      h_maxxSUMAmpl_HF->Fill(float(lscounterM1), maxxSUM4);
      maxxSUM1 = 0.;
      maxxSUM2 = 0.;
      maxxSUM3 = 0.;
      maxxSUM4 = 0.;
      //------------------------------------------------------
      h_maxxOCCUP_HB->Fill(float(lscounterM1), maxxOCCUP1);
      h_maxxOCCUP_HE->Fill(float(lscounterM1), maxxOCCUP2);
      h_maxxOCCUP_HO->Fill(float(lscounterM1), maxxOCCUP3);
      h_maxxOCCUP_HF->Fill(float(lscounterM1), maxxOCCUP4);
      maxxOCCUP1 = 0.;
      maxxOCCUP2 = 0.;
      maxxOCCUP3 = 0.;
      maxxOCCUP4 = 0.;

      //------------------------------------------------------
    }  //if(nevcounter0 != 0)
       //  POINT1

    /////////////////////////////////////////////////// over DigiCollections:
    // for upgrade:
    for (int k1 = 0; k1 < ndepth; k1++) {
      for (int k2 = 0; k2 < neta; k2++) {
        for (int k3 = 0; k3 < nphi; k3++) {
          if (studyCalibCellsHist_) {
            signal[k1][k2][k3] = 0.;
            calibt[k1][k2][k3] = 0.;
            calibcapiderror[k1][k2][k3] = 0;
            caliba[k1][k2][k3] = 0.;
            calibw[k1][k2][k3] = 0.;
            calib0[k1][k2][k3] = 0.;
            signal3[k1][k2][k3] = 0.;
            calib3[k1][k2][k3] = 0.;
            calib2[k1][k2][k3] = 0.;
          }
          if (studyRunDependenceHist_) {
            for (int k0 = 0; k0 < nsub; k0++) {
              badchannels[k0][k1][k2][k3] = 0;
            }  //for
          }    //if

        }  //for
      }    //for
    }      //for
    for (int k0 = 0; k0 < nsub; k0++) {
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            amplitudechannel0[k0][k1][k2][k3] = 0.;
            amplitudechannel[k0][k1][k2][k3] = 0.;
            amplitudechannel2[k0][k1][k2][k3] = 0.;

            tocamplchannel[k0][k1][k2][k3] = 0.;
            maprphinorm[k0][k1][k2][k3] = 0.;
            // phi-symmetry monitoring for calibration group:
            // rec energy:
            recSignalEnergy0[k0][k1][k2][k3] = 0.;
            recSignalEnergy1[k0][k1][k2][k3] = 0.;
            recSignalEnergy2[k0][k1][k2][k3] = 0.;
            recNoiseEnergy0[k0][k1][k2][k3] = 0.;
            recNoiseEnergy1[k0][k1][k2][k3] = 0.;
            recNoiseEnergy2[k0][k1][k2][k3] = 0.;

          }  //k3
        }    //k2
      }      //k1
    }        //k0
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////       END of GENERAL NULLING       ////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (flagToUseDigiCollectionsORNot_ != 0) {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////      START of DigiCollections running:          ///////////////////////////////////
      ////////////////////////////////////////////////////////////////////
      if (flagupgradeqie1011_ != 2 && flagupgradeqie1011_ != 3 && flagupgradeqie1011_ != 6 &&
          flagupgradeqie1011_ != 7 && flagupgradeqie1011_ != 8) {
        edm::Handle<HFDigiCollection> hf;
        iEvent.getByToken(tok_hf_, hf);
        bool gotHFDigis = true;
        if (!(iEvent.getByToken(tok_hf_, hf))) {
          gotHFDigis = false;
        }  //this is a boolean set up to check if there are HFdigis in input root file
        if (!(hf.isValid())) {
          gotHFDigis = false;
        }  //if it is not there, leave it false
        if (!gotHFDigis) {
          std::cout << " ******************************  ===========================   No HFDigiCollection found "
                    << std::endl;
        } else {
          ////////////////////////////////////////////////////////////////////   qie8   QIE8 :
          for (HFDigiCollection::const_iterator digi = hf->begin(); digi != hf->end(); digi++) {
            eta = digi->id().ieta();
            phi = digi->id().iphi();
            depth = digi->id().depth();
            nTS = digi->size();
            ///////////////////
            counterhf++;
            ////////////////////////////////////////////////////////////  for zerrors.C script:
            if (recordHistoes_ && studyCapIDErrorsHist_)
              fillDigiErrorsHF(digi);
            //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
            if (recordHistoes_)
              fillDigiAmplitudeHF(digi);
            //////////////////////////////////////////// calibration staff (often not needed):
            if (recordHistoes_ && studyCalibCellsHist_) {
              int iphi = phi - 1;
              int ieta = eta;
              if (ieta > 0)
                ieta -= 1;
              if (nTS <= numOfTS)
                for (int i = 0; i < nTS; i++) {
                  TS_data[i] = adc2fC[digi->sample(i).adc()];
                  signal[3][ieta + 41][iphi] += TS_data[i];
                  if (i > 1 && i < 6)
                    signal3[3][ieta + 41][iphi] += TS_data[i];
                }  // TS
            }      // if(recordHistoes_ && studyCalibCellsHist_)
          }        // for
        }          // hf.isValid
      }            // end flagupgrade

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HFQIE10 DigiCollection
      //////////////////////////////////////////////////////////////////////////////////////////////////upgradeHF upgradehf
      // upgrade:
      if (flagupgradeqie1011_ != 1) {
        edm::Handle<QIE10DigiCollection> hfqie10;
        iEvent.getByToken(tok_qie10_, hfqie10);
        const QIE10DigiCollection& qie10dc =
            *(hfqie10);  ////////////////////////////////////////////////    <<<=========  !!!!
        bool gotQIE10Digis = true;
        if (!(iEvent.getByToken(tok_qie10_, hfqie10))) {
          gotQIE10Digis = false;
        }  //this is a boolean set up to check if there are HFdigis in input root file
        if (!(hfqie10.isValid())) {
          gotQIE10Digis = false;
        }  //if it is not there, leave it false
        if (!gotQIE10Digis) {
          std::cout << " No QIE10DigiCollection collection is found " << std::endl;
        } else {
          ////////////////////////////////////////////////////////////////////   qie10   QIE10 :
          double totalAmplitudeHF = 0.;
          for (unsigned int j = 0; j < qie10dc.size(); j++) {
            QIE10DataFrame qie10df = static_cast<QIE10DataFrame>(qie10dc[j]);
            DetId detid = qie10df.detid();
            HcalDetId hcaldetid = HcalDetId(detid);
            int eta = hcaldetid.ieta();
            int phi = hcaldetid.iphi();
            //	int depth = hcaldetid.depth();
            // loop over the samples in the digi
            nTS = qie10df.samples();
            ///////////////////
            counterhfqie10++;
            ////////////////////////////////////////////////////////////  for zerrors.C script:
            if (recordHistoes_ && studyCapIDErrorsHist_)
              fillDigiErrorsHFQIE10(qie10df);
            //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
            if (recordHistoes_)
              fillDigiAmplitudeHFQIE10(qie10df);
            ///////////////////
            //     if(recordHistoes_ ) {
            if (recordHistoes_ && studyCalibCellsHist_) {
              int iphi = phi - 1;
              int ieta = eta;
              if (ieta > 0)
                ieta -= 1;
              double amplitudefullTSs = 0.;
              double nnnnnnTS = 0.;
              for (int i = 0; i < nTS; ++i) {
                // j - QIE channel
                // i - time sample (TS)
                int adc = qie10df[i].adc();
                // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                //	      float charge = adc2fC_QIE10[ adc ];
                TS_data[i] = adc2fC_QIE10[adc];
                signal[3][ieta + 41][iphi] += TS_data[i];
                totalAmplitudeHF += TS_data[i];
                amplitudefullTSs += TS_data[i];
                nnnnnnTS++;
                if (i > 1 && i < 6)
                  signal3[3][ieta + 41][iphi] += TS_data[i];

              }  // TS
              h_numberofhitsHFtest->Fill(nnnnnnTS);
              h_AmplitudeHFtest->Fill(amplitudefullTSs);
            }  // if(recordHistoes_ && studyCalibCellsHist_)
          }    // for
          h_totalAmplitudeHF->Fill(totalAmplitudeHF);
          h_totalAmplitudeHFperEvent->Fill(float(eventcounter), totalAmplitudeHF);
        }  // hfqie10.isValid
      }    // end flagupgrade
      //end upgrade
      //
      //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HBHEDigiCollection  usual, <=2018
      int qwert1 = 0;
      int qwert2 = 0;
      int qwert3 = 0;
      int qwert4 = 0;
      int qwert5 = 0;
      if (flagupgradeqie1011_ != 2 && flagupgradeqie1011_ != 3) {
        edm::Handle<HBHEDigiCollection> hbhe;
        iEvent.getByToken(tok_hbhe_, hbhe);
        bool gotHBHEDigis = true;
        if (!(iEvent.getByToken(tok_hbhe_, hbhe)))
          gotHBHEDigis = false;  //this is a boolean set up to check if there are HBHEgigis in input root file
        if (!(hbhe.isValid()))
          gotHBHEDigis = false;  //if it is not there, leave it false
        if (!gotHBHEDigis) {
          std::cout << " No HBHEDigiCollection collection is found " << std::endl;
        } else {
          //      unsigned int NHBHEDigiCollectionsize =  hbhe->size();
          double totalAmplitudeHB = 0.;
          double totalAmplitudeHE = 0.;
          double nnnnnnTSHB = 0.;
          double nnnnnnTSHE = 0.;

          for (HBHEDigiCollection::const_iterator digi = hbhe->begin(); digi != hbhe->end(); digi++) {
            eta = digi->id().ieta();
            phi = digi->id().iphi();
            depth = digi->id().depth();
            nTS = digi->size();
            /////////////////////////////////////// counters of event*digis
            nnnnnnhbhe++;
            nnnnnn++;
            //////////////////////////////////  counters of event for subdet & depth
            if (digi->id().subdet() == HcalBarrel && depth == 1 && qwert1 == 0) {
              nnnnnn1++;
              qwert1 = 1;
            }
            if (digi->id().subdet() == HcalBarrel && depth == 2 && qwert2 == 0) {
              nnnnnn2++;
              qwert2 = 1;
            }
            if (digi->id().subdet() == HcalEndcap && depth == 1 && qwert3 == 0) {
              nnnnnn3++;
              qwert3 = 1;
            }
            if (digi->id().subdet() == HcalEndcap && depth == 2 && qwert4 == 0) {
              nnnnnn4++;
              qwert4 = 1;
            }
            if (digi->id().subdet() == HcalEndcap && depth == 3 && qwert5 == 0) {
              nnnnnn5++;
              qwert5 = 1;
            }
            ////////////////////////////////////////////////////////////  for zerrors.C script:
            if (recordHistoes_ && studyCapIDErrorsHist_)
              fillDigiErrors(digi);
            //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
            if (recordHistoes_)
              fillDigiAmplitude(digi);

            if (recordHistoes_ && studyCalibCellsHist_) {
              int iphi = phi - 1;
              int ieta = eta;
              if (ieta > 0)
                ieta -= 1;
              //////////////////////////////////////////    HB:
              if (digi->id().subdet() == HcalBarrel) {
                double amplitudefullTSs = 0.;
                nnnnnnTSHB++;
                if (nTS <= numOfTS)
                  for (int i = 0; i < nTS; i++) {
                    TS_data[i] = adc2fC[digi->sample(i).adc()];
                    signal[0][ieta + 41][iphi] += TS_data[i];
                    amplitudefullTSs += TS_data[i];
                    totalAmplitudeHB += TS_data[i];
                    if (i > 1 && i < 6)
                      signal3[0][ieta + 41][iphi] += TS_data[i];
                  }
                h_AmplitudeHBtest->Fill(amplitudefullTSs);
              }  // HB
              //////////////////////////////////////////    HE:
              if (digi->id().subdet() == HcalEndcap) {
                double amplitudefullTSs = 0.;
                nnnnnnTSHE++;
                if (nTS <= numOfTS)
                  for (int i = 0; i < nTS; i++) {
                    TS_data[i] = adc2fC[digi->sample(i).adc()];
                    signal[1][ieta + 41][iphi] += TS_data[i];
                    totalAmplitudeHE += TS_data[i];
                    amplitudefullTSs += TS_data[i];
                    if (i > 1 && i < 6)
                      signal3[1][ieta + 41][iphi] += TS_data[i];
                  }
                h_AmplitudeHEtest->Fill(amplitudefullTSs);
              }  // HE

            }  //if(recordHistoes_ && studyCalibCellsHist_)
            if (recordNtuples_ && nevent50 < maxNeventsInNtuple_) {
            }  //if(recordNtuples_)
          }    // for HBHE digis
          if (totalAmplitudeHB != 0.) {
            h_numberofhitsHBtest->Fill(nnnnnnTSHB);
            h_totalAmplitudeHB->Fill(totalAmplitudeHB);
            h_totalAmplitudeHBperEvent->Fill(float(eventcounter), totalAmplitudeHB);
          }
          if (totalAmplitudeHE != 0.) {
            h_numberofhitsHEtest->Fill(nnnnnnTSHE);
            h_totalAmplitudeHE->Fill(totalAmplitudeHE);
            h_totalAmplitudeHEperEvent->Fill(float(eventcounter), totalAmplitudeHE);
          }
        }  //hbhe.isValid
      }    // end flagupgrade
      //---------------------------------------------------------------
      //////////////////////////////////////////////////////////////////////////////////////////////////    upgradeHBHE upgradehe       HBHE with SiPM (both >=2020)
      // upgrade:
      if (flagupgradeqie1011_ != 1 && flagupgradeqie1011_ != 4 && flagupgradeqie1011_ != 5 &&
          flagupgradeqie1011_ != 10) {
        edm::Handle<QIE11DigiCollection> heqie11;
        iEvent.getByToken(tok_qie11_, heqie11);
        const QIE11DigiCollection& qie11dc =
            *(heqie11);  ////////////////////////////////////////////////    <<<=========  !!!!
        bool gotQIE11Digis = true;
        if (!(iEvent.getByToken(tok_qie11_, heqie11)))
          gotQIE11Digis = false;  //this is a boolean set up to check if there are QIE11gigis in input root file
        if (!(heqie11.isValid()))
          gotQIE11Digis = false;  //if it is not there, leave it false
        if (!gotQIE11Digis) {
          std::cout << " No QIE11DigiCollection collection is found " << std::endl;
        } else {
          ////////////////////////////////////////////////////////////////////   qie11   QIE11 :
          double totalAmplitudeHBQIE11 = 0.;
          double totalAmplitudeHEQIE11 = 0.;
          double nnnnnnTSHBQIE11 = 0.;
          double nnnnnnTSHEQIE11 = 0.;
          for (unsigned int j = 0; j < qie11dc.size(); j++) {
            QIE11DataFrame qie11df = static_cast<QIE11DataFrame>(qie11dc[j]);
            DetId detid = qie11df.detid();
            HcalDetId hcaldetid = HcalDetId(detid);
            int eta = hcaldetid.ieta();
            int phi = hcaldetid.iphi();
            int depth = hcaldetid.depth();
            if (depth == 0)
              return;
            int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
            // loop over the samples in the digi
            nTS = qie11df.samples();
            ///////////////////
            nnnnnnhbheqie11++;
            nnnnnn++;
            if (recordHistoes_ && studyCapIDErrorsHist_)
              fillDigiErrorsQIE11(qie11df);
            //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
            if (recordHistoes_)
              fillDigiAmplitudeQIE11(qie11df);
            ///////////////////
            //////////////////////////////////  counters of event for subdet & depth
            if (sub == 1 && depth == 1 && qwert1 == 0) {
              nnnnnn1++;
              qwert1 = 1;
            }
            if (sub == 1 && depth == 2 && qwert2 == 0) {
              nnnnnn2++;
              qwert2 = 1;
            }
            if (sub == 2 && depth == 1 && qwert3 == 0) {
              nnnnnn3++;
              qwert3 = 1;
            }
            if (sub == 2 && depth == 2 && qwert4 == 0) {
              nnnnnn4++;
              qwert4 = 1;
            }
            if (sub == 2 && depth == 3 && qwert5 == 0) {
              nnnnnn5++;
              qwert5 = 1;
            }

            if (recordHistoes_ && studyCalibCellsHist_) {
              int iphi = phi - 1;
              int ieta = eta;
              if (ieta > 0)
                ieta -= 1;
              // HB:
              if (sub == 1) {
                double amplitudefullTSs1 = 0.;
                double amplitudefullTSs6 = 0.;
                nnnnnnTSHBQIE11++;
                for (int i = 0; i < nTS; ++i) {
                  int adc = qie11df[i].adc();
                  double charge1 = adc2fC_QIE11_shunt1[adc];
                  double charge6 = adc2fC_QIE11_shunt6[adc];
                  amplitudefullTSs1 += charge1;
                  amplitudefullTSs6 += charge6;
                  double charge = charge6;
                  TS_data[i] = charge;
                  signal[0][ieta + 41][iphi] += charge;
                  if (i > 1 && i < 6)
                    signal3[0][ieta + 41][iphi] += charge;
                  totalAmplitudeHBQIE11 += charge;
                }  //for
                h_AmplitudeHBtest1->Fill(amplitudefullTSs1, 1.);
                h_AmplitudeHBtest6->Fill(amplitudefullTSs6, 1.);
              }  //HB end
              // HE:
              if (sub == 2) {
                double amplitudefullTSs1 = 0.;
                double amplitudefullTSs6 = 0.;
                nnnnnnTSHEQIE11++;
                for (int i = 0; i < nTS; i++) {
                  int adc = qie11df[i].adc();
                  double charge1 = adc2fC_QIE11_shunt1[adc];
                  double charge6 = adc2fC_QIE11_shunt6[adc];
                  amplitudefullTSs1 += charge1;
                  amplitudefullTSs6 += charge6;
                  double charge = charge6;
                  TS_data[i] = charge;
                  signal[1][ieta + 41][iphi] += charge;
                  if (i > 1 && i < 6)
                    signal3[1][ieta + 41][iphi] += charge;
                  totalAmplitudeHEQIE11 += charge;
                }  //for
                h_AmplitudeHEtest1->Fill(amplitudefullTSs1, 1.);
                h_AmplitudeHEtest6->Fill(amplitudefullTSs6, 1.);

              }  //HE end
            }    //if(recordHistoes_ && studyCalibCellsHist_)
          }      // for QIE11 digis

          if (totalAmplitudeHBQIE11 != 0.) {
            h_numberofhitsHBtest->Fill(nnnnnnTSHBQIE11);
            h_totalAmplitudeHB->Fill(totalAmplitudeHBQIE11);
            h_totalAmplitudeHBperEvent->Fill(float(eventcounter), totalAmplitudeHBQIE11);
          }
          if (totalAmplitudeHEQIE11 != 0.) {
            h_numberofhitsHEtest->Fill(nnnnnnTSHEQIE11);
            h_totalAmplitudeHE->Fill(totalAmplitudeHEQIE11);
            h_totalAmplitudeHEperEvent->Fill(float(eventcounter), totalAmplitudeHEQIE11);
          }
        }  //heqie11.isValid
      }    // end flagupgrade

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////   HODigiCollection
      edm::Handle<HODigiCollection> ho;
      iEvent.getByToken(tok_ho_, ho);
      bool gotHODigis = true;
      if (!(iEvent.getByToken(tok_ho_, ho)))
        gotHODigis = false;  //this is a boolean set up to check if there are HOgigis in input root file
      if (!(ho.isValid()))
        gotHODigis = false;  //if it is not there, leave it false
      if (!gotHODigis) {
        //  if(!ho.isValid()) {
        std::cout << " No HO collection is found " << std::endl;
      } else {
        int qwert6 = 0;
        double totalAmplitudeHO = 0.;
        for (HODigiCollection::const_iterator digi = ho->begin(); digi != ho->end(); digi++) {
          eta = digi->id().ieta();
          phi = digi->id().iphi();
          depth = digi->id().depth();
          nTS = digi->size();
          ///////////////////
          counterho++;
          //////////////////////////////////  counters of event
          if (qwert6 == 0) {
            nnnnnn6++;
            qwert6 = 1;
          }
          ////////////////////////////////////////////////////////////  for zerrors.C script:
          if (recordHistoes_ && studyCapIDErrorsHist_)
            fillDigiErrorsHO(digi);
          //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
          if (recordHistoes_)
            fillDigiAmplitudeHO(digi);
          ///////////////////
          if (recordHistoes_ && studyCalibCellsHist_) {
            int iphi = phi - 1;
            int ieta = eta;
            if (ieta > 0)
              ieta -= 1;
            double nnnnnnTS = 0.;
            double amplitudefullTSs = 0.;
            if (nTS <= numOfTS)
              for (int i = 0; i < nTS; i++) {
                TS_data[i] = adc2fC[digi->sample(i).adc()];
                amplitudefullTSs += TS_data[i];
                signal[2][ieta + 41][iphi] += TS_data[i];
                totalAmplitudeHO += TS_data[i];
                if (i > 1 && i < 6)
                  signal3[2][ieta + 41][iphi] += TS_data[i];
                nnnnnnTS++;
              }  //if for
            h_AmplitudeHOtest->Fill(amplitudefullTSs);
            h_numberofhitsHOtest->Fill(nnnnnnTS);
          }  //if(recordHistoes_ && studyCalibCellsHist_)
        }    //for HODigiCollection

        h_totalAmplitudeHO->Fill(totalAmplitudeHO);
        h_totalAmplitudeHOperEvent->Fill(float(eventcounter), totalAmplitudeHO);
      }  //ho.isValid(
    }    // flagToUseDigiCollectionsORNot_

    //////////////////////////////////// RecHits for phi-symmetry monitoring of calibration group:
    // AZ 04.11.2019
    //////////////////////////////////////////////////////
    if (flagIterativeMethodCalibrationGroupReco_ > 0) {
      //////////////////////////////////////////////////////////////////////////////////////////////////////  Noise
      //////////////////////////////////////////////////////////////////////////////////////////////////////  Noise
      //////////////////////////////////////////////////////////////////////////////////////////////////////  Noise
      // HBHE:    HBHERecHitCollection hbheNoise Noise
      edm::Handle<HBHERecHitCollection> hbheNoise;
      iEvent.getByToken(tok_hbheNoise_, hbheNoise);
      bool gotHBHERecHitsNoise = true;
      if (!(iEvent.getByToken(tok_hbheNoise_, hbheNoise)))
        gotHBHERecHitsNoise =
            false;  //this is a boolean set up to check if there are HBHERecHitsNoise in input root file
      if (!(hbheNoise.isValid()))
        gotHBHERecHitsNoise = false;  //if it is not there, leave it false
      if (!gotHBHERecHitsNoise) {
        //  if(!hbheNoise.isValid()) {
        std::cout << " No RecHits HBHENoise collection is found " << std::endl;
      } else {
        for (HBHERecHitCollection::const_iterator hbheItr = hbheNoise->begin(); hbheItr != hbheNoise->end();
             hbheItr++) {
          // Recalibration of energy
          float icalconst = 1.;
          //      DetId mydetid = hbheItr->id().rawId();
          //      if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
          HBHERecHit aHit(hbheItr->id(), hbheItr->eraw() * icalconst, hbheItr->time());
          //        HBHERecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());
          double energyhit = aHit.energy();
          DetId id = (*hbheItr).detid();
          HcalDetId hid = HcalDetId(id);
          int sub = ((hid).rawId() >> 25) & 0x7;

          if (sub == 1)
            h_energyhitNoise_HB->Fill(energyhit, 1.);
          if (sub == 2)
            h_energyhitNoise_HE->Fill(energyhit, 1.);
          //	if(fabs(energyhit) > 40. ) continue;

          //std::cout<<sub<<std::endl;
          if (hid.depth() == 1 && sub == 1 && hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << " Noise,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          if (hid.depth() == 1 && sub == 2 && hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << " Noise,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          int ieta = hid.ieta();  // -15 ... -1; 1... 15 for HB
          if (ieta > 0)
            ieta -= 1;  // -15 ... -1; 0... 14 for HB
          int iphi = hid.iphi() - 1;
          int mdepth = hid.depth();
          recNoiseEnergy0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
          recNoiseEnergy1[sub - 1][mdepth - 1][ieta + 41][iphi] += energyhit;
          recNoiseEnergy2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(energyhit, 2);
        }  // hbheNoise
      }    //hbheNoise.isValid(
      ////////////////////////////////////////////////////// /  HBHE Noise end

      // HF:    HFRecHitCollection hfNoise Noise
      edm::Handle<HFRecHitCollection> hfNoise;
      iEvent.getByToken(tok_hfNoise_, hfNoise);
      bool gotHFRecHitsNoise = true;
      if (!(iEvent.getByToken(tok_hfNoise_, hfNoise)))
        gotHFRecHitsNoise = false;  //this is a boolean set up to check if there are HFRecHitsNoise in input root file
      if (!(hfNoise.isValid()))
        gotHFRecHitsNoise = false;  //if it is not there, leave it false
      if (!gotHFRecHitsNoise) {
        //  if(!hfNoise.isValid()) {
        std::cout << " No RecHits HFNoise collection is found " << std::endl;
      } else {
        for (HFRecHitCollection::const_iterator hfItr = hfNoise->begin(); hfItr != hfNoise->end(); hfItr++) {
          // Recalibration of energy
          float icalconst = 1.;
          //      DetId mydetid = hfItr->id().rawId();
          //      if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
          HFRecHit aHit(hfItr->id(), hfItr->energy() * icalconst, hfItr->time());
          double energyhit = aHit.energy();
          DetId id = (*hfItr).detid();
          HcalDetId hid = HcalDetId(id);
          int sub = ((hid).rawId() >> 25) & 0x7;

          h_energyhitNoise_HF->Fill(energyhit, 1.);
          //	if(fabs(energyhit) > 40. ) continue;

          //std::cout<<sub<<std::endl;
          if (hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << "HF Noise,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          int ieta = hid.ieta();  // -15 ... -1; 1... 15 for HB
          if (ieta > 0)
            ieta -= 1;  // -15 ... -1; 0... 14 for HB
          int iphi = hid.iphi() - 1;
          int mdepth = hid.depth();
          recNoiseEnergy0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
          recNoiseEnergy1[sub - 1][mdepth - 1][ieta + 41][iphi] += energyhit;
          recNoiseEnergy2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(energyhit, 2);
        }  // hfNoise
      }    //hfNoise.isValid(
      ////////////////////////////////////////////////////// /  HF Noise end

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////// Signal
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////// Signal
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////// Signal
      // HBHE:    HBHERecHitCollection hbheSignal Signal
      edm::Handle<HBHERecHitCollection> hbheSignal;
      iEvent.getByToken(tok_hbheSignal_, hbheSignal);
      bool gotHBHERecHitsSignal = true;
      if (!(iEvent.getByToken(tok_hbheSignal_, hbheSignal)))
        gotHBHERecHitsSignal =
            false;  //this is a boolean set up to check if there are HBHERecHitsSignal in input root file
      if (!(hbheSignal.isValid()))
        gotHBHERecHitsSignal = false;  //if it is not there, leave it false
      if (!gotHBHERecHitsSignal) {
        //  if(!hbheSignal.isValid()) {
        std::cout << " No RecHits HBHESignal collection is found " << std::endl;
      } else {
        for (HBHERecHitCollection::const_iterator hbheItr = hbheSignal->begin(); hbheItr != hbheSignal->end();
             hbheItr++) {
          // Recalibration of energy
          float icalconst = 1.;
          //      DetId mydetid = hbheItr->id().rawId();
          //      if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
          HBHERecHit aHit(hbheItr->id(), hbheItr->eraw() * icalconst, hbheItr->time());
          //        HBHERecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());
          double energyhit = aHit.energy();
          DetId id = (*hbheItr).detid();
          HcalDetId hid = HcalDetId(id);
          int sub = ((hid).rawId() >> 25) & 0x7;

          if (sub == 1)
            h_energyhitSignal_HB->Fill(energyhit, 1.);
          if (sub == 2)
            h_energyhitSignal_HE->Fill(energyhit, 1.);

          //std::cout<<sub<<std::endl;
          if (hid.depth() == 1 && sub == 1 && hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << "HBHE Signal,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          if (hid.depth() == 1 && sub == 2 && hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << "HBHE Signal,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          int ieta = hid.ieta();  // -15 ... -1; 1... 15 for HB
          if (ieta > 0)
            ieta -= 1;  // -15 ... -1; 0... 14 for HB
          int iphi = hid.iphi() - 1;
          int mdepth = hid.depth();
          recSignalEnergy0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
          recSignalEnergy1[sub - 1][mdepth - 1][ieta + 41][iphi] += energyhit;
          recSignalEnergy2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(energyhit, 2);
        }  // hbheSignal
      }    //hbheSignal.isValid(
      ////////////////////////////////////////////////////// /  HBHE Signal end

      // HF:    HFRecHitCollection hfSignal Signal
      edm::Handle<HFRecHitCollection> hfSignal;
      iEvent.getByToken(tok_hfSignal_, hfSignal);
      bool gotHFRecHitsSignal = true;
      if (!(iEvent.getByToken(tok_hfSignal_, hfSignal)))
        gotHFRecHitsSignal = false;  //this is a boolean set up to check if there are HFRecHitsSignal in input root file
      if (!(hfSignal.isValid()))
        gotHFRecHitsSignal = false;  //if it is not there, leave it false
      if (!gotHFRecHitsSignal) {
        //  if(!hfSignal.isValid()) {
        std::cout << " No RecHits HFSignal collection is found " << std::endl;
      } else {
        for (HFRecHitCollection::const_iterator hfItr = hfSignal->begin(); hfItr != hfSignal->end(); hfItr++) {
          // Recalibration of energy
          float icalconst = 1.;
          //      DetId mydetid = hfItr->id().rawId();
          //      if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
          HFRecHit aHit(hfItr->id(), hfItr->energy() * icalconst, hfItr->time());
          double energyhit = aHit.energy();
          DetId id = (*hfItr).detid();
          HcalDetId hid = HcalDetId(id);
          int sub = ((hid).rawId() >> 25) & 0x7;

          h_energyhitSignal_HF->Fill(energyhit, 1.);
          //	if(fabs(energyhit) > 40. ) continue;

          //std::cout<<sub<<std::endl;
          if (hid.iphi() == 25) {
            if (verbosity < 0)
              std::cout << "HF Signal,sub = " << sub << " mdepth = " << hid.depth() << "  ieta= " << hid.ieta()
                        << "  iphi= " << hid.iphi() << "  energyhit= " << energyhit << std::endl;
          }
          int ieta = hid.ieta();  // -15 ... -1; 1... 15 for HB
          if (ieta > 0)
            ieta -= 1;  // -15 ... -1; 0... 14 for HB
          int iphi = hid.iphi() - 1;
          int mdepth = hid.depth();
          recSignalEnergy0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
          recSignalEnergy1[sub - 1][mdepth - 1][ieta + 41][iphi] += energyhit;
          recSignalEnergy2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(energyhit, 2);
        }  // hfSignal
      }    //hfSignal.isValid(
      //////////////////////////////////////////////////////  HF Signal end

      //////////////////////////////////////////////////////
      // END of RecHits for phi-symmetry monitoring of calibration group
    }  // flagIterativeMethodCalibrationGroupReco_  >  0
    //////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////  TREATMENT OF OBTAINED DIGI-COLLECTION :
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Phi-Symmetry Monitoring DIGI
    //////////// k0(sub):       =0 HB;      =1 HE;       =2 HO;       =3 HF;
    //////////// k1(depth-1): = 0 - 6 or depth: = 1 - 7;
    if (flagIterativeMethodCalibrationGroupDigi_ > 0) {
      //////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////// Digi Digi Digi Digi Digi Digi
      //////////////////////////////////////////////////////////////////////////
      //	  //	  //	  //	  //	  //	  // tocdefault tocampl tocamplchannel: calibration group, Iterative method, coding start 29.08.2019
      for (int k0 = 0; k0 < nsub; k0++) {
        // HE only, temporary
        if (k0 == 1) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  //if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              //preparation for PHI normalization:
              double sumoverphi = 0;
              int nsumoverphi = 0;
              for (int k3 = 0; k3 < nphi; k3++) {
                if (tocamplchannel[k0][k1][k2][k3] != 0) {
                  sumoverphi += tocamplchannel[k0][k1][k2][k3];
                  ++nsumoverphi;
                  if (verbosity < 0)
                    std::cout << "==== nsumoverphi = " << nsumoverphi << "  sumoverphi = " << sumoverphi
                              << "  k1 = " << k1 << "  k2 = " << k2 << " kkk = " << kkk << "  k3 = " << k3 << std::endl;
                }  //if != 0
              }    //k3
              // PHI normalization into new massive && filling plots:
              for (int k3 = 0; k3 < nphi; k3++) {
                if (nsumoverphi != 0) {
                  maprphinorm[k0][k1][k2][k3] = tocamplchannel[k0][k1][k2][k3] / (sumoverphi / nsumoverphi);
                  if (verbosity < 0)
                    std::cout << "nsumoverphi= " << nsumoverphi << " sumoverphi= " << sumoverphi << " k1= " << k1
                              << " k2= " << k2 << " kkk= " << kkk << " k3= " << k3
                              << " maprphinorm= " << maprphinorm[k0][k1][k2][k3] << std::endl;
                  if (k1 == 0) {
                    h_mapenophinorm_HE1->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE1->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE1->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE1->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE1->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 1) {
                    h_mapenophinorm_HE2->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE2->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE2->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE2->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE2->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 2) {
                    h_mapenophinorm_HE3->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE3->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE3->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE3->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE3->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 3) {
                    h_mapenophinorm_HE4->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE4->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE4->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE4->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE4->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 4) {
                    h_mapenophinorm_HE5->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE5->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE5->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE5->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE5->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 5) {
                    h_mapenophinorm_HE6->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE6->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE6->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE6->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE6->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k1 == 6) {
                    h_mapenophinorm_HE7->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE7->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE7->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE7->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE7->Fill(double(kkk), double(k3), 1.);
                  }
                }  //if nsumoverphi != 0
              }    //k3
            }      //k2
          }        //k1
        }          //if k0 == 1 HE
      }            //k0
      //	  //	  //	  //	  //	  //	    //	  //	  //	  //	  //	  //	    //	  //	  //	  //	  //	  //	    //	  //	  //	  //	  //	  //
      //	  //	  //	  //	  //	  //	  // amplitudechannel amplitudechannel amplitudechannel: calibration group, Iterative method, coding start 11.11.2019
      for (int k0 = 0; k0 < nsub; k0++) {
        // HB:
        if (k0 == 0) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_amplitudechannel0_HB1->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HB1->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HB1->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_amplitudechannel0_HB2->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HB2->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HB2->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 2) {
                  h_amplitudechannel0_HB3->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HB3->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HB3->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_amplitudechannel0_HB4->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HB4->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HB4->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 0 HB

        // HE:
        if (k0 == 1) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_amplitudechannel0_HE1->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE1->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE1->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_amplitudechannel0_HE2->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE2->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE2->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 2) {
                  h_amplitudechannel0_HE3->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE3->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE3->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_amplitudechannel0_HE4->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE4->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE4->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 4) {
                  h_amplitudechannel0_HE5->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE5->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE5->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 5) {
                  h_amplitudechannel0_HE6->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE6->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE6->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 6) {
                  h_amplitudechannel0_HE7->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HE7->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HE7->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 1 HE

        // HF: 4 depthes for Digis and only 2 - for Reco !!!
        if (k0 == 3) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_amplitudechannel0_HF1->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HF1->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HF1->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_amplitudechannel0_HF2->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HF2->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HF2->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 2) {
                  h_amplitudechannel0_HF3->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HF3->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HF3->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_amplitudechannel0_HF4->Fill(double(kkk), double(k3), amplitudechannel0[k0][k1][k2][k3]);
                  h_amplitudechannel1_HF4->Fill(double(kkk), double(k3), amplitudechannel[k0][k1][k2][k3]);
                  h_amplitudechannel2_HF4->Fill(double(kkk), double(k3), amplitudechannel2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 3 HF

      }  //k0

    }  // if(flagIterativeMethodCalibrationGroupDigi

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Phi-Symmetry Monitoring Reco
    //////////// k0(sub):       =0 HB;      =1 HE;       =2 HO;       =3 HF;
    //////////// k1(depth-1): = 0 - 6 or depth: = 1 - 7;
    if (flagIterativeMethodCalibrationGroupReco_ > 0) {
      //////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////// Reco Reco Reco Reco Reco Reco
      //////////////////////////////////////////////////////////////////////////
      //
      for (int k0 = 0; k0 < nsub; k0++) {
        // HB:
        if (k0 == 0) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_recSignalEnergy0_HB1->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HB1->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HB1->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HB1->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HB1->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HB1->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_recSignalEnergy0_HB2->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HB2->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HB2->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HB2->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HB2->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HB2->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 2) {
                  h_recSignalEnergy0_HB3->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HB3->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HB3->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HB3->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HB3->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HB3->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_recSignalEnergy0_HB4->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HB4->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HB4->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HB4->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HB4->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HB4->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 0 HB

        // HE:
        if (k0 == 1) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_recSignalEnergy0_HE1->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE1->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE1->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE1->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE1->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE1->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_recSignalEnergy0_HE2->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE2->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE2->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE2->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE2->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE2->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 2) {
                  h_recSignalEnergy0_HE3->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE3->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE3->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE3->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE3->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE3->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_recSignalEnergy0_HE4->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE4->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE4->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE4->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE4->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE4->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 4) {
                  h_recSignalEnergy0_HE5->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE5->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE5->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE5->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE5->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE5->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 5) {
                  h_recSignalEnergy0_HE6->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE6->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE6->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE6->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE6->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE6->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 6) {
                  h_recSignalEnergy0_HE7->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HE7->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HE7->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HE7->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HE7->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HE7->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 1 HE

        // HF: 4 depthes for Digis and only 2 - for Reco !!! ('ve tried to enter 4 for reco since 31.10.2021 AZ)
        if (k0 == 3) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              //	      int k2plot = k2-41;int kkk = k2; if(k2plot >0 ) kkk=k2+1; //-41 +41 !=0
              int k2plot = k2 - 41;
              int kkk = k2plot;  // if(k2plot >=0 ) kkk=k2plot+1; //-41 +40 !=0
              for (int k3 = 0; k3 < nphi; k3++) {
                if (k1 == 0) {
                  h_recSignalEnergy0_HF1->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HF1->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HF1->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HF1->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HF1->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HF1->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 1) {
                  h_recSignalEnergy0_HF2->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HF2->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HF2->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HF2->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HF2->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HF2->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                // AZ 31.10.2021: k1=3 and 4 added for HF recoSignal,recNoise
                if (k1 == 2) {
                  h_recSignalEnergy0_HF3->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HF3->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HF3->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HF3->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HF3->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HF3->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
                if (k1 == 3) {
                  h_recSignalEnergy0_HF4->Fill(double(kkk), double(k3), recSignalEnergy0[k0][k1][k2][k3]);
                  h_recSignalEnergy1_HF4->Fill(double(kkk), double(k3), recSignalEnergy1[k0][k1][k2][k3]);
                  h_recSignalEnergy2_HF4->Fill(double(kkk), double(k3), recSignalEnergy2[k0][k1][k2][k3]);
                  h_recNoiseEnergy0_HF4->Fill(double(kkk), double(k3), recNoiseEnergy0[k0][k1][k2][k3]);
                  h_recNoiseEnergy1_HF4->Fill(double(kkk), double(k3), recNoiseEnergy1[k0][k1][k2][k3]);
                  h_recNoiseEnergy2_HF4->Fill(double(kkk), double(k3), recNoiseEnergy2[k0][k1][k2][k3]);
                }
              }  //k3
            }    //k2
          }      //k1
        }        //if k0 == 3 HF

      }  //k0

    }  // if(flagIterativeMethodCalibrationGroupReco

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   END PHY-SYMMERTY for calibration group

    //////////////////////////////////////////////////////
    if (flagLaserRaddam_ > 1) {
      ////////////////////////////////////////////////////// RADDAM treatment:
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            if (mapRADDAM0_HE[k1][k2][k3] != 0) {
              // ----------------------------------------    D2 sum over phi before!!! any dividing:
              mapRADDAM_HED2[k1][k2] += mapRADDAM_HE[k1][k2][k3];
              // N phi sectors w/ digihits
              ++mapRADDAM_HED20[k1][k2];
            }  //if
          }    //for
        }      //for
      }        //for

      //////////////---------------------------------------------------------------------------------  2D treatment, zraddam2.cc script
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (mapRADDAM_HED20[k1][k2] != 0) {
            // validation of channels at eta16:
            if (k1 == 2 && k2 == 25) {
              h_sumphiEta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED20[k1][k2]);
            } else if (k1 == 2 && k2 == 56) {
              h_sumphiEta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED20[k1][k2]);
            } else {
              h_sumphiEta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED20[k1][k2]);
            }
            //////////////-----------------------  aver per N-phi_sectors ???
            mapRADDAM_HED2[k1][k2] /= mapRADDAM_HED20[k1][k2];
          }  // if(mapRADDAM_HED20[k1][k2] != 0
        }    //for
      }      //for
      ///////////////////////////////////////////
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (k1 == 2 && (k2 == 25 || k2 == 56)) {
          } else {
            //	if(k2!=25 && k2!=56) {
            int k2plot = k2 - 41;
            int kkk = k2;
            if (k2plot > 0)
              kkk = k2 + 1;
            int kk2 = 25;
            if (k2plot > 0)
              kk2 = 56;
            if (mapRADDAM_HED2[k1][k2] != 0. && mapRADDAM_HED2[2][kk2] != 0) {
              mapRADDAM_HED2[k1][k2] /= mapRADDAM_HED2[2][kk2];
              // (d1 & eta 17-29)                       L1
              int LLLLLL111111 = 0;
              if ((k1 == 0 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 30))
                LLLLLL111111 = 1;
              // (d2 & eta 17-26) && (d3 & eta 27-28)   L2
              int LLLLLL222222 = 0;
              if ((k1 == 1 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 27) ||
                  (k1 == 2 && fabs(kkk - 41) > 26 && fabs(kkk - 41) < 29))
                LLLLLL222222 = 1;
              //
              if (LLLLLL111111 == 1) {
                h_sigLayer1RADDAM5_HED2->Fill(double(kkk - 41), mapRADDAM_HED2[k1][k2]);
                h_sigLayer1RADDAM6_HED2->Fill(double(kkk - 41), 1.);
              }
              if (LLLLLL222222 == 1) {
                h_sigLayer2RADDAM5_HED2->Fill(double(kkk - 41), mapRADDAM_HED2[k1][k2]);
                h_sigLayer2RADDAM6_HED2->Fill(double(kkk - 41), 1.);
              }
            }  //if
          }    // if(k2!=25 && k2!=56
        }      //for
      }        //for

      //////////////---------------------------------------------------------------------------------  3D treatment, zraddam1.cc script

      //------------------------------------------------------        aver per eta 16(depth=3-> k1=2, k2=16(15) :
      //////////// k1(depth-1): = 0 - 6 or depth: = 1 - 7;
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (k1 == 2 && (k2 == 25 || k2 == 56)) {
          } else {
            int k2plot = k2 - 41;
            int kk2 = 25;
            if (k2plot > 0)
              kk2 = 56;
            int kkk = k2;
            if (k2plot > 0)
              kkk = k2 + 1;
            for (int k3 = 0; k3 < nphi; k3++) {
              if (mapRADDAM_HE[k1][k2][k3] != 0. && mapRADDAM_HE[2][kk2][k3] != 0) {
                mapRADDAM_HE[k1][k2][k3] /= mapRADDAM_HE[2][kk2][k3];
                int LLLLLL111111 = 0;
                if ((k1 == 0 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 30))
                  LLLLLL111111 = 1;
                int LLLLLL222222 = 0;
                if ((k1 == 1 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 27) ||
                    (k1 == 2 && fabs(kkk - 41) > 26 && fabs(kkk - 41) < 29))
                  LLLLLL222222 = 1;
                if (LLLLLL111111 == 1) {
                  h_sigLayer1RADDAM5_HE->Fill(double(kkk - 41), mapRADDAM_HE[k1][k2][k3]);
                  h_sigLayer1RADDAM6_HE->Fill(double(kkk - 41), 1.);
                }
                if (LLLLLL222222 == 1) {
                  h_sigLayer2RADDAM5_HE->Fill(double(kkk - 41), mapRADDAM_HE[k1][k2][k3]);
                  h_sigLayer2RADDAM6_HE->Fill(double(kkk - 41), 1.);
                }
              }  //if
            }    //for
          }      // if(k2!=25 && k2!=56
        }        //for
      }          //for
                 //
                 ////////////////////////////////////////////////////////////////////////////////////////////////
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          mapRADDAM_HED2[k1][k2] = 0.;
          mapRADDAM_HED20[k1][k2] = 0.;
          for (int k3 = 0; k3 < nphi; k3++) {
            mapRADDAM_HE[k1][k2][k3] = 0.;
            mapRADDAM0_HE[k1][k2][k3] = 0;
          }  //for
        }    //for
      }      //for

      //////////////////////////////////END of RADDAM treatment:
    }  // END TREATMENT : if(flagLaserRaddam_ == 1
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    //	  //	  //	  //	  //	  //	  //sumamplitudes:
    ////////////////////////////////////////  // k0, k2, k3 loops LOOPS  //////////   /////  ///// NO k1 loop over depthes !!!
    for (int k0 = 0; k0 < nsub; k0++) {
      int sumofchannels = 0;
      double sumamplitudesubdet = 0.;
      int sumofchannels0 = 0;
      double sumamplitudesubdet0 = 0.;
      for (int k2 = 0; k2 < neta; k2++) {
        for (int k3 = 0; k3 < nphi; k3++) {
          // HB
          if (k0 == 0) {
            double sumamplitudechannel_HB = amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3];
            h_sumamplitudechannel_HB->Fill(sumamplitudechannel_HB);
            if (sumamplitudechannel_HB > 80.) {
              sumamplitudesubdet += sumamplitudechannel_HB;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HB > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HB;
                sumofchannels0++;
              }
            }
          }  //

          // HE
          if (k0 == 1) {
            double sumamplitudechannel_HE =
                amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3] + amplitudechannel[k0][2][k2][k3];
            h_sumamplitudechannel_HE->Fill(sumamplitudechannel_HE);
            if (sumamplitudechannel_HE > 200.) {
              sumamplitudesubdet += sumamplitudechannel_HE;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HE > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HE;
                sumofchannels0++;
              }
            }
          }  //

          // HO
          if (k0 == 2) {
            double sumamplitudechannel_HO = amplitudechannel[k0][3][k2][k3];
            h_sumamplitudechannel_HO->Fill(sumamplitudechannel_HO);
            if (sumamplitudechannel_HO > 1200.) {
              sumamplitudesubdet += sumamplitudechannel_HO;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HO > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HO;
                sumofchannels0++;
              }
            }
          }  //
          // HF
          if (k0 == 3) {
            double sumamplitudechannel_HF = amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3];
            h_sumamplitudechannel_HF->Fill(sumamplitudechannel_HF);
            if (sumamplitudechannel_HF > 600.) {
              sumamplitudesubdet += sumamplitudechannel_HF;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HF > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HF;
                sumofchannels0++;
              }
            }
          }  //

        }  //k3
      }    //k2
      //  }//k1
      // SA of each sub-detector DONE. Then: summarize or find maximum throught events of LS
      if (k0 == 0) {
        h_eventamplitude_HB->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HB->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM1)
          maxxSUM1 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP1)
          maxxOCCUP1 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HB += sumofchannels;
        averSIGNALsumamplitude_HB += sumamplitudesubdet;
        averNOSIGNALoccupancy_HB += sumofchannels0;
        averNOSIGNALsumamplitude_HB += sumamplitudesubdet0;
        if ((sumamplitudesubdet + sumamplitudesubdet0) > 60000) {
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              int ieta = k2 - 41;
              /// HB depth1:
              if (amplitudechannel[k0][0][k2][k3] != 0.) {
                h_2DAtaildepth1_HB->Fill(double(ieta), double(k3), amplitudechannel[k0][0][k2][k3]);
                h_2D0Ataildepth1_HB->Fill(double(ieta), double(k3), 1.);
              }
              /// HB depth2:
              if (amplitudechannel[k0][1][k2][k3] != 0.) {
                h_2DAtaildepth2_HB->Fill(double(ieta), double(k3), amplitudechannel[k0][1][k2][k3]);
                h_2D0Ataildepth2_HB->Fill(double(ieta), double(k3), 1.);
              }
            }  //for
          }    //for
        }      //>60000
      }        //HB
      if (k0 == 1) {
        h_eventamplitude_HE->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HE->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM2)
          maxxSUM2 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP2)
          maxxOCCUP2 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HE += sumofchannels;
        averSIGNALsumamplitude_HE += sumamplitudesubdet;
        averNOSIGNALoccupancy_HE += sumofchannels0;
        averNOSIGNALsumamplitude_HE += sumamplitudesubdet0;
      }  //HE
      if (k0 == 2) {
        h_eventamplitude_HO->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HO->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM3)
          maxxSUM3 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP3)
          maxxOCCUP3 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HO += sumofchannels;
        averSIGNALsumamplitude_HO += sumamplitudesubdet;
        averNOSIGNALoccupancy_HO += sumofchannels0;
        averNOSIGNALsumamplitude_HO += sumamplitudesubdet0;
      }  //HO
      if (k0 == 3) {
        h_eventamplitude_HF->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HF->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM4)
          maxxSUM4 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP4)
          maxxOCCUP4 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HF += sumofchannels;
        averSIGNALsumamplitude_HF += sumamplitudesubdet;
        averNOSIGNALoccupancy_HF += sumofchannels0;
        averNOSIGNALsumamplitude_HF += sumamplitudesubdet0;
      }  //HF
    }    //k0

    ///////////////////// ///////////////////// //////////////////////////////////////////
    ///////////////////////////////////////////////  for zRunRatio34.C & zRunNbadchan.C scripts:
    if (recordHistoes_ && studyRunDependenceHist_) {
      int eeeeee;
      eeeeee = lscounterM1;
      if (flagtoaskrunsorls_ == 0)
        eeeeee = runcounter;

      //////////// k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      for (int k0 = 0; k0 < nsub; k0++) {
        //////////// k1(depth-1): = 0 - 3 or depth: = 1 - 4;
        //////////// for upgrade    k1(depth-1): = 0 - 6 or depth: = 1 - 7;
        for (int k1 = 0; k1 < ndepth; k1++) {
          //////////
          int nbadchannels = 0;
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              if (badchannels[k0][k1][k2][k3] != 0)
                ++nbadchannels;
            }  //k3
          }    //k2
          //////////
          //HB
          if (k0 == 0) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HB->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HBdepth1_)
                h_runnbadchannelsC_depth1_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HB->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HBdepth1_)
                h_runbadrateC_depth1_HB->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HB->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HB->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HB->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HB->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HBdepth2_)
                h_runnbadchannelsC_depth2_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HB->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HBdepth2_)
                h_runbadrateC_depth2_HB->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HB->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HB->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HB->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 0)
          //HE
          if (k0 == 1) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth1_)
                h_runnbadchannelsC_depth1_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth1_)
                h_runbadrateC_depth1_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HE->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth2_)
                h_runnbadchannelsC_depth2_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth2_)
                h_runbadrateC_depth2_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HE->Fill(float(bcn), 1.);
            }
            if (k1 == 2) {
              h_nbadchannels_depth3_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth3_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth3_)
                h_runnbadchannelsC_depth3_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth3_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth3_)
                h_runbadrateC_depth3_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth3_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth3_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth3_HE->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 1)
          //HO
          if (k0 == 2) {
            if (k1 == 3) {
              h_nbadchannels_depth4_HO->Fill(float(nbadchannels));
              h_runnbadchannels_depth4_HO->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HOdepth4_)
                h_runnbadchannelsC_depth4_HO->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth4_HO->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HOdepth4_)
                h_runbadrateC_depth4_HO->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth4_HO->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth4_HO->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth4_HO->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 2)
          //HF
          if (k0 == 3) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HF->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HFdepth1_)
                h_runnbadchannelsC_depth1_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HF->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HFdepth1_)
                h_runbadrateC_depth1_HF->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HF->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HF->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HF->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HF->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HFdepth2_)
                h_runnbadchannelsC_depth2_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HF->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HFdepth2_)
                h_runbadrateC_depth2_HF->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HF->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HF->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HF->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 3)

          //////////
        }  //k1
      }    //k0
      ////////////
    }  //if(recordHistoes_&& studyRunDependenceHist_)

    /////////////////////////////////////////////////////////////////////////////////////// HcalCalibDigiCollection
    edm::Handle<HcalCalibDigiCollection> calib;
    iEvent.getByToken(tok_calib_, calib);

    bool gotCALIBDigis = true;
    if (!(iEvent.getByToken(tok_calib_, calib))) {
      gotCALIBDigis = false;  //this is a boolean set up to check if there are CALIBgigis in input root file
    }
    if (!(calib.isValid())) {
      gotCALIBDigis = false;  //if it is not there, leave it false
    }
    if (!gotCALIBDigis) {
    } else {
      for (HcalCalibDigiCollection::const_iterator digi = calib->begin(); digi != calib->end(); digi++) {
        int cal_det = digi->id().hcalSubdet();  // 1-HB,2-HE,3-HO,4-HF
        int cal_phi = digi->id().iphi();
        int cal_eta = digi->id().ieta();
        int cal_cbox = digi->id().cboxChannel();

        /////////////////////////////////////////////
        if (recordHistoes_ && studyCalibCellsHist_) {
          if (cal_det > 0 && cal_det < 5 && cal_cbox == 0) {
            int iphi = cal_phi - 1;
            int ieta = cal_eta;
            if (ieta > 0)
              ieta -= 1;
            nTS = digi->size();
            double max_signal = -100.;
            int ts_with_max_signal = -100;
            double timew = 0.;

            //
            if (nTS <= numOfTS)
              for (int i = 0; i < nTS; i++) {
                double ampldefault = adc2fC[digi->sample(i).adc() & 0xff];
                if (max_signal < ampldefault) {
                  max_signal = ampldefault;
                  ts_with_max_signal = i;
                }
                if (i > 1 && i < 6)
                  calib3[cal_det - 1][ieta + 41][iphi] += ampldefault;
                calib0[cal_det - 1][ieta + 41][iphi] += ampldefault;
                timew += (i + 1) * ampldefault;
              }  // for
            //

            double amplitude = calib0[cal_det - 1][ieta + 41][iphi];
            double aveamplitude = -100.;
            if (amplitude > 0 && timew > 0)
              aveamplitude = timew / amplitude;       // average_TS +1
            double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9
            caliba[cal_det - 1][ieta + 41][iphi] = aveamplitude1;

            double rmsamp = 0.;
            for (int ii = 0; ii < nTS; ii++) {
              double ampldefault = adc2fC[digi->sample(ii).adc() & 0xff];
              double aaaaaa = (ii + 1) - aveamplitude;
              double aaaaaa2 = aaaaaa * aaaaaa;
              rmsamp += (aaaaaa2 * ampldefault);  // fC
            }                                     //for 2
            double rmsamplitude = -100.;
            if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
              rmsamplitude = sqrt(rmsamp / amplitude);
            calibw[cal_det - 1][ieta + 41][iphi] = rmsamplitude;
            //
            calibt[cal_det - 1][ieta + 41][iphi] = ts_with_max_signal;
            //

            if (ts_with_max_signal > -1 && ts_with_max_signal < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] = adc2fC[digi->sample(ts_with_max_signal).adc() & 0xff];
            if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal + 1).adc() & 0xff];
            if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal + 2).adc() & 0xff];
            if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal - 1).adc() & 0xff];
            if (ts_with_max_signal - 2 > -1 && ts_with_max_signal - 2 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal - 2).adc() & 0xff];
            //
            bool anycapid = true;
            bool anyer = false;
            bool anydv = true;
            int error1 = 0, error2 = 0, error3 = 0;
            int lastcapid = 0, capid = 0;
            for (int ii = 0; ii < (*digi).size(); ii++) {
              capid = (*digi)[ii].capid();  // capId (0-3, sequential)
              bool er = (*digi)[ii].er();   // error
              bool dv = (*digi)[ii].dv();   // valid data
              if (ii != 0 && ((lastcapid + 1) % 4) != capid)
                anycapid = false;
              lastcapid = capid;
              if (er)
                anyer = true;
              if (!dv)
                anydv = false;
            }
            if (!anycapid)
              error1 = 1;
            if (anyer)
              error2 = 1;
            if (!anydv)
              error3 = 1;
            if (error1 != 0 || error2 != 0 || error3 != 0)
              calibcapiderror[cal_det - 1][ieta + 41][iphi] = 100;

          }  // if(cal_det>0 && cal_det<5
        }    //if(recordHistoes_ && studyCalibCellsHist_)
        /////////////////////////////////////////////

        if (recordNtuples_ && nevent50 < maxNeventsInNtuple_) {
        }  //if(recordNtuples_) {

      }  //for(HcalCalibDigiCollection
    }    //if(calib.isValid(
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (recordHistoes_ && studyCalibCellsHist_) {
      ////////////////////////////////////for loop for zcalib.C and zgain.C scripts:
      for (int k1 = 0; k1 < nsub; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            int k2plot = k2 - 41;
            if (flagcpuoptimization_ == 0) {
              ////////////////////////////////////////////////////////////////  for zgain.C script:

              if (signal[k1][k2][k3] > 0.) {
                if (k1 == 0) {
                  h_FullSignal3D_HB->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HB->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 1) {
                  h_FullSignal3D_HE->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HE->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 2) {
                  h_FullSignal3D_HO->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HO->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 3) {
                  h_FullSignal3D_HF->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HF->Fill(double(k2plot), double(k3), 1.);
                }
              }

            }  // optimization
            ////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////  for zcalib.C script:
            // k2 = 0-81, k3= 0-71
            // return to real indexes in eta and phi ( k20 and k30)
            int k20 = k2 - 41;  // k20 = -41 - 40
            if (k20 > 0 || k20 == 0)
              k20 += 1;        // k20 = -41 - -1 and +1 - +41
            int k30 = k3 + 1;  // k30= 1-nphi

            // find calibration indexes in eta and phi ( kk2 and kk3)
            int kk2 = 0, kk3 = 0;
            if (k1 == 0 || k1 == 1) {
              if (k20 > 0)
                kk2 = 1;
              else
                kk2 = -1;
              if (k30 == 71 || k30 == nphi || k30 == 1 || k30 == 2)
                kk3 = 71;
              else
                kk3 = ((k30 - 3) / 4) * 4 + 3;
            } else if (k1 == 2) {
              if (abs(k20) <= 4) {
                kk2 = 0;
                if (k30 == 71 || k30 == nphi || k30 == 1 || k30 == 2 || k30 == 3 || k30 == 4)
                  kk3 = 71;
                else
                  kk3 = ((k30 - 5) / 6) * 6 + 5;
              } else {
                if (abs(k20) > 4 && abs(k20) <= 10)
                  kk2 = 1;
                if (abs(k20) > 10 && abs(k20) <= 15)
                  kk2 = 2;
                if (k20 < 0)
                  kk2 = -kk2;
                if (k30 == 71 || k30 == nphi || (k30 >= 1 && k30 <= 10))
                  kk3 = 71;
                else
                  kk3 = ((k30 - 11) / 12) * 12 + 11;
              }
            } else if (k1 == 3) {
              if (k20 > 0)
                kk2 = 1;
              else
                kk2 = -1;
              if (k30 >= 1 && k30 <= 18)
                kk3 = 1;
              if (k30 >= 19 && k30 <= 36)
                kk3 = 19;
              if (k30 >= 37 && k30 <= 54)
                kk3 = 37;
              if (k30 >= 55 && k30 <= nphi)
                kk3 = 55;
            }
            // return to indexes in massiv
            int kkk2 = kk2 + 41;
            if (kk2 > 0)
              kkk2 -= 1;
            int kkk3 = kk3;
            kkk3 -= 1;

            if (flagcpuoptimization_ == 0) {
              double GetRMSOverNormalizedSignal = -1.;
              if (signal[k1][k2][k3] > 0. && calib0[k1][kkk2][kkk3] > 0.) {
                GetRMSOverNormalizedSignal = signal[k1][k2][k3] / calib0[k1][kkk2][kkk3];
                if (k1 == 0) {
                  h_mapGetRMSOverNormalizedSignal_HB->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HB->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 1) {
                  h_mapGetRMSOverNormalizedSignal_HE->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HE->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 2) {
                  h_mapGetRMSOverNormalizedSignal_HO->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HO->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 3) {
                  h_mapGetRMSOverNormalizedSignal_HF->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HF->Fill(double(k2plot), double(k3), 1.);
                }
              }
            }  //optimization
            ////////////////////////////////////////////////////////////////  for zcalib....C script:
            if (signal[k1][k2][k3] > 0.) {
              // ADC
              double adc = 0.;
              if (calib0[k1][kkk2][kkk3] > 0.)
                adc = calib0[k1][kkk2][kkk3];
              // Ratio
              double ratio = 2.;
              if (calib0[k1][kkk2][kkk3] > 0.)
                ratio = calib2[k1][kkk2][kkk3] / calib0[k1][kkk2][kkk3];
              // TSmax
              float calibtsmax = calibt[k1][kkk2][kkk3];
              // TSmean
              float calibtsmean = caliba[k1][kkk2][kkk3];
              // Width
              float calibwidth = calibw[k1][kkk2][kkk3];
              // CapIdErrors
              float calibcap = -100.;
              calibcap = calibcapiderror[k1][kkk2][kkk3];

              //                 HB:
              if (k1 == 0) {
                // ADC
                h_ADCCalib_HB->Fill(adc, 1.);
                h_ADCCalib1_HB->Fill(adc, 1.);
                if (adc < calibrADCHBMin_ || adc > calibrADCHBMax_)
                  h_mapADCCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HB->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HB->Fill(ratio, 1.);
                if (ratio < calibrRatioHBMin_ || ratio > calibrRatioHBMax_)
                  h_mapRatioCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HB->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HB->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHBMin_ || calibtsmax > calibrTSmaxHBMax_)
                    h_mapTSmaxCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HB->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HB->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHBMin_ || calibtsmean > calibrTSmeanHBMax_)
                    h_mapTSmeanCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HB->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HB->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHBMin_ || calibwidth > calibrWidthHBMax_)
                    h_mapWidthCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HB->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HB->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HE:
              if (k1 == 1) {
                // ADC
                h_ADCCalib_HE->Fill(adc, 1.);
                h_ADCCalib1_HE->Fill(adc, 1.);
                if (adc < calibrADCHEMin_ || adc > calibrADCHEMax_)
                  h_mapADCCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HE->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HE->Fill(ratio, 1.);
                if (ratio < calibrRatioHEMin_ || ratio > calibrRatioHEMax_)
                  h_mapRatioCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HE->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HE->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHEMin_ || calibtsmax > calibrTSmaxHEMax_)
                    h_mapTSmaxCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HE->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HE->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHEMin_ || calibtsmean > calibrTSmeanHEMax_)
                    h_mapTSmeanCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HE->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HE->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHEMin_ || calibwidth > calibrWidthHEMax_)
                    h_mapWidthCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HE->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HE->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HO:
              if (k1 == 2) {
                // ADC
                h_ADCCalib_HO->Fill(adc, 1.);
                h_ADCCalib1_HO->Fill(adc, 1.);
                if (adc < calibrADCHOMin_ || adc > calibrADCHOMax_)
                  h_mapADCCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HO->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HO->Fill(ratio, 1.);
                if (ratio < calibrRatioHOMin_ || ratio > calibrRatioHOMax_)
                  h_mapRatioCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HO->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HO->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHOMin_ || calibtsmax > calibrTSmaxHOMax_)
                    h_mapTSmaxCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HO->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HO->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHOMin_ || calibtsmean > calibrTSmeanHOMax_)
                    h_mapTSmeanCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HO->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HO->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHOMin_ || calibwidth > calibrWidthHOMax_)
                    h_mapWidthCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HO->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HO->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HF:
              if (k1 == 3) {
                // ADC
                h_ADCCalib_HF->Fill(adc, 1.);
                h_ADCCalib1_HF->Fill(adc, 1.);
                if (adc < calibrADCHFMin_ || adc > calibrADCHFMax_)
                  h_mapADCCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HF->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HF->Fill(ratio, 1.);
                if (ratio < calibrRatioHFMin_ || ratio > calibrRatioHFMax_)
                  h_mapRatioCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HF->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HF->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHFMin_ || calibtsmax > calibrTSmaxHFMax_)
                    h_mapTSmaxCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HF->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HF->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHFMin_ || calibtsmean > calibrTSmeanHFMax_)
                    h_mapTSmeanCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HF->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HF->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHFMin_ || calibwidth > calibrWidthHFMax_)
                    h_mapWidthCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HF->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HF->Fill(double(k2plot), double(k3), 1.);
              }
              //////////
            }  // if(signal[k1][k2][k3]>0.)
            //////////
          }  // k3
        }    // k2
      }      // k1

      /////

    }  //if(recordHistoes_&& studyCalibCellsHist_)

    ///////////////////////////////////////////////////
    if (recordNtuples_ && nevent50 < maxNeventsInNtuple_)
      myTree->Fill();
    //  if(recordNtuples_ && nevent < maxNeventsInNtuple_) myTree->Fill();

    ///////////////////////////////////////////////////
    if (++local_event % 100 == 0) {
      if (verbosity < 0)
        std::cout << "run " << Run << " processing events " << local_event << " ok, "
                  << ", lumi " << lumi << ", numOfLaserEv " << numOfLaserEv << std::endl;
    }
  }  // bcn

  //EndAnalyzer
}

// ------------ method called once each job just before starting event loop  -----------
void CMTRawAnalyzer::beginJob() {
  if (verbosity > 0)
    std::cout << "========================   beignJob START   +++++++++++++++++++++++++++" << std::endl;
  nnnnnn = 0;
  nnnnnnhbhe = 0;
  nnnnnnhbheqie11 = 0;
  nevent = 0;
  nevent50 = 0;
  counterhf = 0;
  counterhfqie10 = 0;
  counterho = 0;
  nnnnnn1 = 0;
  nnnnnn2 = 0;
  nnnnnn3 = 0;
  nnnnnn4 = 0;
  nnnnnn5 = 0;
  nnnnnn6 = 0;
  //////////////////////////////////////////////////////////////////////////////////    book histoes

  if (recordHistoes_) {
    //  ha2 = fs_->make<TH2F>("ha2"," ", neta, -41., 41., nphi, 0., bphi);

    h_errorGeneral = fs_->make<TH1F>("h_errorGeneral", " ", 5, 0., 5.);
    h_error1 = fs_->make<TH1F>("h_error1", " ", 5, 0., 5.);
    h_error2 = fs_->make<TH1F>("h_error2", " ", 5, 0., 5.);
    h_error3 = fs_->make<TH1F>("h_error3", " ", 5, 0., 5.);
    h_amplError = fs_->make<TH1F>("h_amplError", " ", 100, -2., 98.);
    h_amplFine = fs_->make<TH1F>("h_amplFine", " ", 100, -2., 98.);

    h_errorGeneral_HB = fs_->make<TH1F>("h_errorGeneral_HB", " ", 5, 0., 5.);
    h_error1_HB = fs_->make<TH1F>("h_error1_HB", " ", 5, 0., 5.);
    h_error2_HB = fs_->make<TH1F>("h_error2_HB", " ", 5, 0., 5.);
    h_error3_HB = fs_->make<TH1F>("h_error3_HB", " ", 5, 0., 5.);
    h_error4_HB = fs_->make<TH1F>("h_error4_HB", " ", 5, 0., 5.);
    h_error5_HB = fs_->make<TH1F>("h_error5_HB", " ", 5, 0., 5.);
    h_error6_HB = fs_->make<TH1F>("h_error6_HB", " ", 5, 0., 5.);
    h_error7_HB = fs_->make<TH1F>("h_error7_HB", " ", 5, 0., 5.);
    h_amplError_HB = fs_->make<TH1F>("h_amplError_HB", " ", 100, -2., 98.);
    h_amplFine_HB = fs_->make<TH1F>("h_amplFine_HB", " ", 100, -2., 98.);
    h_mapDepth1Error_HB = fs_->make<TH2F>("h_mapDepth1Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HB = fs_->make<TH2F>("h_mapDepth2Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HB = fs_->make<TH2F>("h_mapDepth3Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HB = fs_->make<TH2F>("h_mapDepth4Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HB = fs_->make<TH1F>("h_fiber0_HB", " ", 10, 0., 10.);
    h_fiber1_HB = fs_->make<TH1F>("h_fiber1_HB", " ", 10, 0., 10.);
    h_fiber2_HB = fs_->make<TH1F>("h_fiber2_HB", " ", 40, 0., 40.);
    h_repetedcapid_HB = fs_->make<TH1F>("h_repetedcapid_HB", " ", 5, 0., 5.);

    h_errorGeneral_HE = fs_->make<TH1F>("h_errorGeneral_HE", " ", 5, 0., 5.);
    h_error1_HE = fs_->make<TH1F>("h_error1_HE", " ", 5, 0., 5.);
    h_error2_HE = fs_->make<TH1F>("h_error2_HE", " ", 5, 0., 5.);
    h_error3_HE = fs_->make<TH1F>("h_error3_HE", " ", 5, 0., 5.);
    h_error4_HE = fs_->make<TH1F>("h_error4_HE", " ", 5, 0., 5.);
    h_error5_HE = fs_->make<TH1F>("h_error5_HE", " ", 5, 0., 5.);
    h_error6_HE = fs_->make<TH1F>("h_error6_HE", " ", 5, 0., 5.);
    h_error7_HE = fs_->make<TH1F>("h_error7_HE", " ", 5, 0., 5.);
    h_amplError_HE = fs_->make<TH1F>("h_amplError_HE", " ", 100, -2., 98.);
    h_amplFine_HE = fs_->make<TH1F>("h_amplFine_HE", " ", 100, -2., 98.);
    h_mapDepth1Error_HE = fs_->make<TH2F>("h_mapDepth1Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HE = fs_->make<TH2F>("h_mapDepth2Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HE = fs_->make<TH2F>("h_mapDepth3Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HE = fs_->make<TH2F>("h_mapDepth4Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Error_HE = fs_->make<TH2F>("h_mapDepth5Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Error_HE = fs_->make<TH2F>("h_mapDepth6Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Error_HE = fs_->make<TH2F>("h_mapDepth7Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HE = fs_->make<TH1F>("h_fiber0_HE", " ", 10, 0., 10.);
    h_fiber1_HE = fs_->make<TH1F>("h_fiber1_HE", " ", 10, 0., 10.);
    h_fiber2_HE = fs_->make<TH1F>("h_fiber2_HE", " ", 40, 0., 40.);
    h_repetedcapid_HE = fs_->make<TH1F>("h_repetedcapid_HE", " ", 5, 0., 5.);

    h_errorGeneral_HF = fs_->make<TH1F>("h_errorGeneral_HF", " ", 5, 0., 5.);
    h_error1_HF = fs_->make<TH1F>("h_error1_HF", " ", 5, 0., 5.);
    h_error2_HF = fs_->make<TH1F>("h_error2_HF", " ", 5, 0., 5.);
    h_error3_HF = fs_->make<TH1F>("h_error3_HF", " ", 5, 0., 5.);
    h_error4_HF = fs_->make<TH1F>("h_error4_HF", " ", 5, 0., 5.);
    h_error5_HF = fs_->make<TH1F>("h_error5_HF", " ", 5, 0., 5.);
    h_error6_HF = fs_->make<TH1F>("h_error6_HF", " ", 5, 0., 5.);
    h_error7_HF = fs_->make<TH1F>("h_error7_HF", " ", 5, 0., 5.);
    h_amplError_HF = fs_->make<TH1F>("h_amplError_HF", " ", 100, -2., 98.);
    h_amplFine_HF = fs_->make<TH1F>("h_amplFine_HF", " ", 100, -2., 98.);
    h_mapDepth1Error_HF = fs_->make<TH2F>("h_mapDepth1Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HF = fs_->make<TH2F>("h_mapDepth2Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HF = fs_->make<TH2F>("h_mapDepth3Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HF = fs_->make<TH2F>("h_mapDepth4Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HF = fs_->make<TH1F>("h_fiber0_HF", " ", 10, 0., 10.);
    h_fiber1_HF = fs_->make<TH1F>("h_fiber1_HF", " ", 10, 0., 10.);
    h_fiber2_HF = fs_->make<TH1F>("h_fiber2_HF", " ", 40, 0., 40.);
    h_repetedcapid_HF = fs_->make<TH1F>("h_repetedcapid_HF", " ", 5, 0., 5.);

    h_errorGeneral_HO = fs_->make<TH1F>("h_errorGeneral_HO", " ", 5, 0., 5.);
    h_error1_HO = fs_->make<TH1F>("h_error1_HO", " ", 5, 0., 5.);
    h_error2_HO = fs_->make<TH1F>("h_error2_HO", " ", 5, 0., 5.);
    h_error3_HO = fs_->make<TH1F>("h_error3_HO", " ", 5, 0., 5.);
    h_error4_HO = fs_->make<TH1F>("h_error4_HO", " ", 5, 0., 5.);
    h_error5_HO = fs_->make<TH1F>("h_error5_HO", " ", 5, 0., 5.);
    h_error6_HO = fs_->make<TH1F>("h_error6_HO", " ", 5, 0., 5.);
    h_error7_HO = fs_->make<TH1F>("h_error7_HO", " ", 5, 0., 5.);
    h_amplError_HO = fs_->make<TH1F>("h_amplError_HO", " ", 100, -2., 98.);
    h_amplFine_HO = fs_->make<TH1F>("h_amplFine_HO", " ", 100, -2., 98.);
    h_mapDepth4Error_HO = fs_->make<TH2F>("h_mapDepth4Error_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HO = fs_->make<TH1F>("h_fiber0_HO", " ", 10, 0., 10.);
    h_fiber1_HO = fs_->make<TH1F>("h_fiber1_HO", " ", 10, 0., 10.);
    h_fiber2_HO = fs_->make<TH1F>("h_fiber2_HO", " ", 40, 0., 40.);
    h_repetedcapid_HO = fs_->make<TH1F>("h_repetedcapid_HO", " ", 5, 0., 5.);

    /////////////////////////////////////////////////////////////////////////////////////////////////             HB

    h_numberofhitsHBtest = fs_->make<TH1F>("h_numberofhitsHBtest", " ", 100, 0., 100.);
    h_AmplitudeHBtest = fs_->make<TH1F>("h_AmplitudeHBtest", " ", 100, 0., 10000.);
    h_AmplitudeHBtest1 = fs_->make<TH1F>("h_AmplitudeHBtest1", " ", 100, 0., 1000000.);
    h_AmplitudeHBtest6 = fs_->make<TH1F>("h_AmplitudeHBtest6", " ", 100, 0., 2000000.);
    h_totalAmplitudeHB = fs_->make<TH1F>("h_totalAmplitudeHB", " ", 100, 0., 3000000.);
    h_totalAmplitudeHBperEvent = fs_->make<TH1F>("h_totalAmplitudeHBperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmpl345Zoom_HB = fs_->make<TH1F>("h_ADCAmpl345Zoom_HB", " ", 100, 0., 400.);
    h_ADCAmpl345Zoom1_HB = fs_->make<TH1F>("h_ADCAmpl345Zoom1_HB", " ", 100, 0., 100.);
    h_ADCAmpl345_HB = fs_->make<TH1F>("h_ADCAmpl345_HB", " ", 100, 10., 3000.);

    h_AmplitudeHBrest = fs_->make<TH1F>("h_AmplitudeHBrest", " ", 100, 0., 10000.);
    h_AmplitudeHBrest1 = fs_->make<TH1F>("h_AmplitudeHBrest1", " ", 100, 0., 1000000.);
    h_AmplitudeHBrest6 = fs_->make<TH1F>("h_AmplitudeHBrest6", " ", 100, 0., 2000000.);

    h_ADCAmpl345_HBCapIdError = fs_->make<TH1F>("h_ADCAmpl345_HBCapIdError", " ", 100, 10., 3000.);
    h_ADCAmpl345_HBCapIdNoError = fs_->make<TH1F>("h_ADCAmpl345_HBCapIdNoError", " ", 100, 10., 3000.);
    h_ADCAmpl_HBCapIdError = fs_->make<TH1F>("h_ADCAmpl_HBCapIdError", " ", 100, 10., 3000.);
    h_ADCAmpl_HBCapIdNoError = fs_->make<TH1F>("h_ADCAmpl_HBCapIdNoError", " ", 100, 10., 3000.);

    h_ADCAmplZoom_HB = fs_->make<TH1F>("h_ADCAmplZoom_HB", " ", 100, 0., 400.);
    h_ADCAmplZoom1_HB = fs_->make<TH1F>("h_ADCAmplZoom1_HB", " ", 100, -20., 80.);
    h_ADCAmpl_HB = fs_->make<TH1F>("h_ADCAmpl_HB", " ", 100, 10., 5000.);
    h_mapDepth1ADCAmpl225_HB = fs_->make<TH2F>("h_mapDepth1ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HB = fs_->make<TH2F>("h_mapDepth2ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HB = fs_->make<TH2F>("h_mapDepth3ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HB = fs_->make<TH2F>("h_mapDepth4ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HB =
        fs_->make<TH2F>("h_mapDepth1ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HB =
        fs_->make<TH2F>("h_mapDepth2ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HB =
        fs_->make<TH2F>("h_mapDepth3ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HB =
        fs_->make<TH2F>("h_mapDepth4ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl_HB = fs_->make<TH2F>("h_mapDepth1ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HB = fs_->make<TH2F>("h_mapDepth2ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HB = fs_->make<TH2F>("h_mapDepth3ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HB = fs_->make<TH2F>("h_mapDepth4ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HB = fs_->make<TH1F>("h_TSmeanA_HB", " ", 100, -1., 11.);
    h_mapDepth1TSmeanA225_HB = fs_->make<TH2F>("h_mapDepth1TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HB = fs_->make<TH2F>("h_mapDepth2TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HB = fs_->make<TH2F>("h_mapDepth3TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HB = fs_->make<TH2F>("h_mapDepth4TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HB = fs_->make<TH2F>("h_mapDepth1TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HB = fs_->make<TH2F>("h_mapDepth2TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HB = fs_->make<TH2F>("h_mapDepth3TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HB = fs_->make<TH2F>("h_mapDepth4TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HB = fs_->make<TH1F>("h_TSmaxA_HB", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HB = fs_->make<TH2F>("h_mapDepth1TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HB = fs_->make<TH2F>("h_mapDepth2TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HB = fs_->make<TH2F>("h_mapDepth3TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HB = fs_->make<TH2F>("h_mapDepth4TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HB = fs_->make<TH2F>("h_mapDepth1TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HB = fs_->make<TH2F>("h_mapDepth2TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HB = fs_->make<TH2F>("h_mapDepth3TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HB = fs_->make<TH2F>("h_mapDepth4TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    // RMS:
    h_Amplitude_HB = fs_->make<TH1F>("h_Amplitude_HB", " ", 100, 0., 5.);
    h_mapDepth1Amplitude225_HB = fs_->make<TH2F>("h_mapDepth1Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HB = fs_->make<TH2F>("h_mapDepth2Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HB = fs_->make<TH2F>("h_mapDepth3Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HB = fs_->make<TH2F>("h_mapDepth4Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HB = fs_->make<TH2F>("h_mapDepth1Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HB = fs_->make<TH2F>("h_mapDepth2Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HB = fs_->make<TH2F>("h_mapDepth3Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HB = fs_->make<TH2F>("h_mapDepth4Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HB = fs_->make<TH1F>("h_Ampl_HB", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HB = fs_->make<TH2F>("h_mapDepth1Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HB = fs_->make<TH2F>("h_mapDepth2Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HB = fs_->make<TH2F>("h_mapDepth3Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HB = fs_->make<TH2F>("h_mapDepth4Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HB = fs_->make<TH2F>("h_mapDepth1Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HB = fs_->make<TH2F>("h_mapDepth2Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HB = fs_->make<TH2F>("h_mapDepth3Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HB = fs_->make<TH2F>("h_mapDepth4Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HB = fs_->make<TH2F>("h_mapDepth1AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HB = fs_->make<TH2F>("h_mapDepth2AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HB = fs_->make<TH2F>("h_mapDepth3AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HB = fs_->make<TH2F>("h_mapDepth4AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HB = fs_->make<TH2F>("h_mapDepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HB = fs_->make<TH2F>("h_mapDepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HB = fs_->make<TH2F>("h_mapDepth3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HB = fs_->make<TH2F>("h_mapDepth4_HB", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1TS2_HB = fs_->make<TH2F>("h_mapDepth1TS2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TS2_HB = fs_->make<TH2F>("h_mapDepth2TS2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TS2_HB = fs_->make<TH2F>("h_mapDepth3TS2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TS2_HB = fs_->make<TH2F>("h_mapDepth4TS2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TS2_HE = fs_->make<TH2F>("h_mapDepth1TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TS2_HE = fs_->make<TH2F>("h_mapDepth2TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TS2_HE = fs_->make<TH2F>("h_mapDepth3TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TS2_HE = fs_->make<TH2F>("h_mapDepth4TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TS2_HE = fs_->make<TH2F>("h_mapDepth5TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TS2_HE = fs_->make<TH2F>("h_mapDepth6TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TS2_HE = fs_->make<TH2F>("h_mapDepth7TS2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HF3 = fs_->make<TH2F>("h_recSignalEnergy0_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HF3 = fs_->make<TH2F>("h_recSignalEnergy1_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HF3 = fs_->make<TH2F>("h_recSignalEnergy2_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HF4 = fs_->make<TH2F>("h_recSignalEnergy0_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HF4 = fs_->make<TH2F>("h_recSignalEnergy1_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HF4 = fs_->make<TH2F>("h_recSignalEnergy2_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HF3 = fs_->make<TH2F>("h_recNoiseEnergy0_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HF3 = fs_->make<TH2F>("h_recNoiseEnergy1_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HF3 = fs_->make<TH2F>("h_recNoiseEnergy2_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HF4 = fs_->make<TH2F>("h_recNoiseEnergy0_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HF4 = fs_->make<TH2F>("h_recNoiseEnergy1_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HF4 = fs_->make<TH2F>("h_recNoiseEnergy2_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TS1_HF = fs_->make<TH2F>("h_mapDepth1TS1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TS1_HF = fs_->make<TH2F>("h_mapDepth2TS1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TS1_HF = fs_->make<TH2F>("h_mapDepth3TS1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TS1_HF = fs_->make<TH2F>("h_mapDepth4TS1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TS012_HO = fs_->make<TH2F>("h_mapDepth4TS012_HO", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////////////             HE

    // stuff regarding summed(total) Amplitude vs iEvent (histo-name is  h_totalAmplitudeHEperEvent)
    // to see from which event ALL channels are available(related to quality of the run)
    h_numberofhitsHEtest = fs_->make<TH1F>("h_numberofhitsHEtest", " ", 100, 0., 10000.);
    h_AmplitudeHEtest = fs_->make<TH1F>("h_AmplitudeHEtest", " ", 100, 0., 1000000.);
    h_AmplitudeHEtest1 = fs_->make<TH1F>("h_AmplitudeHEtest1", " ", 100, 0., 1000000.);
    h_AmplitudeHEtest6 = fs_->make<TH1F>("h_AmplitudeHEtest6", " ", 100, 0., 2000000.);
    h_totalAmplitudeHE = fs_->make<TH1F>("h_totalAmplitudeHE", " ", 100, 0., 10000000000.);
    h_totalAmplitudeHEperEvent = fs_->make<TH1F>("h_totalAmplitudeHEperEvent", " ", 1000, 1., 1001.);

    // Aijk Amplitude:
    h_ADCAmplZoom1_HE = fs_->make<TH1F>("h_ADCAmplZoom1_HE", " ", npfit, 0., anpfit);        // for amplmaxts 1TS w/ max
    h_ADCAmpl345Zoom1_HE = fs_->make<TH1F>("h_ADCAmpl345Zoom1_HE", " ", npfit, 0., anpfit);  // for ampl3ts 3TSs
    h_ADCAmpl345Zoom_HE = fs_->make<TH1F>("h_ADCAmpl345Zoom_HE", " ", npfit, 0., anpfit);    // for ampl 4TSs
    h_amplitudeaveragedbydepthes_HE =
        fs_->make<TH1F>("h_amplitudeaveragedbydepthes_HE", " ", npfit, 0., anpfit);  // for cross-check: A spectrum
    h_ndepthesperamplitudebins_HE =
        fs_->make<TH1F>("h_ndepthesperamplitudebins_HE", " ", 10, 0., 10.);  // for cross-check: ndepthes

    // Ampl12 4TSs to work with "ped-Gsel0" or "led-low-intensity" to clarify gain diff peak2-peak1
    h_mapADCAmplfirstpeak_HE =
        fs_->make<TH2F>("h_mapADCAmplfirstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplfirstpeak0_HE =
        fs_->make<TH2F>("h_mapADCAmplfirstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplsecondpeak_HE =
        fs_->make<TH2F>("h_mapADCAmplsecondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplsecondpeak0_HE = fs_->make<TH2F>(
        "h_mapADCAmplsecondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max

    h_mapADCAmpl11firstpeak_HE =
        fs_->make<TH2F>("h_mapADCAmpl11firstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11firstpeak0_HE =
        fs_->make<TH2F>("h_mapADCAmpl11firstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11secondpeak_HE =
        fs_->make<TH2F>("h_mapADCAmpl11secondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11secondpeak0_HE =
        fs_->make<TH2F>("h_mapADCAmpl11secondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs

    h_mapADCAmpl12firstpeak_HE =
        fs_->make<TH2F>("h_mapADCAmpl12firstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12firstpeak0_HE =
        fs_->make<TH2F>("h_mapADCAmpl12firstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12secondpeak_HE =
        fs_->make<TH2F>("h_mapADCAmpl12secondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12secondpeak0_HE =
        fs_->make<TH2F>("h_mapADCAmpl12secondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs

    // Ampl12 4TSs to work with "ped-Gsel0" or "led-low-intensity" to clarify gain diff peak2-peak1  fit results:
    h_gsmdifferencefit1_HE = fs_->make<TH1F>("h_gsmdifferencefit1_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit2_HE = fs_->make<TH1F>("h_gsmdifferencefit2_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit3_HE = fs_->make<TH1F>("h_gsmdifferencefit3_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit4_HE = fs_->make<TH1F>("h_gsmdifferencefit4_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit5_HE = fs_->make<TH1F>("h_gsmdifferencefit5_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit6_HE = fs_->make<TH1F>("h_gsmdifferencefit6_HE", " ", 80, 20., 60.);

    // Aijk Amplitude:
    h_ADCAmpl_HE = fs_->make<TH1F>("h_ADCAmpl_HE", " ", 200, 0., 2000000.);
    h_ADCAmplrest_HE = fs_->make<TH1F>("h_ADCAmplrest_HE", " ", 100, 0., 500.);
    h_ADCAmplrest1_HE = fs_->make<TH1F>("h_ADCAmplrest1_HE", " ", 100, 0., 100.);
    h_ADCAmplrest6_HE = fs_->make<TH1F>("h_ADCAmplrest6_HE", " ", 100, 0., 10000.);

    h_ADCAmpl345_HE = fs_->make<TH1F>("h_ADCAmpl345_HE", " ", 70, 0., 700000.);

    // SiPM corrections:
    h_corrforxaMAIN_HE = fs_->make<TH1F>("h_corrforxaMAIN_HE", " ", 70, 0., 700000.);
    h_corrforxaMAIN0_HE = fs_->make<TH1F>("h_corrforxaMAIN0_HE", " ", 70, 0., 700000.);
    h_corrforxaADDI_HE = fs_->make<TH1F>("h_corrforxaADDI_HE", " ", 70, 0., 700000.);
    h_corrforxaADDI0_HE = fs_->make<TH1F>("h_corrforxaADDI0_HE", " ", 70, 0., 700000.);

    h_mapDepth1ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth1ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth2ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth3ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth4ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth5ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth6ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl225_HE = fs_->make<TH2F>("h_mapDepth7ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth1ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth2ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth3ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth4ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth5ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth6ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl225Copy_HE =
        fs_->make<TH2F>("h_mapDepth7ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth1ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth2ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth3ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth4ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth5ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth6ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl_HE = fs_->make<TH2F>("h_mapDepth7ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmplSiPM_HE = fs_->make<TH2F>("h_mapDepth1ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmplSiPM_HE = fs_->make<TH2F>("h_mapDepth2ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmplSiPM_HE = fs_->make<TH2F>("h_mapDepth3ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_TSmeanA_HE = fs_->make<TH1F>("h_TSmeanA_HE", " ", 100, -2., 8.);
    h_mapDepth1TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth1TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth2TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth3TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth4TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth5TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth6TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmeanA225_HE = fs_->make<TH2F>("h_mapDepth7TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HE = fs_->make<TH2F>("h_mapDepth1TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HE = fs_->make<TH2F>("h_mapDepth2TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HE = fs_->make<TH2F>("h_mapDepth3TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HE = fs_->make<TH2F>("h_mapDepth4TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmeanA_HE = fs_->make<TH2F>("h_mapDepth5TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmeanA_HE = fs_->make<TH2F>("h_mapDepth6TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmeanA_HE = fs_->make<TH2F>("h_mapDepth7TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HE = fs_->make<TH1F>("h_TSmaxA_HE", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth1TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth2TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth3TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth4TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth5TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth6TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmaxA225_HE = fs_->make<TH2F>("h_mapDepth7TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HE = fs_->make<TH2F>("h_mapDepth1TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HE = fs_->make<TH2F>("h_mapDepth2TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HE = fs_->make<TH2F>("h_mapDepth3TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HE = fs_->make<TH2F>("h_mapDepth4TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmaxA_HE = fs_->make<TH2F>("h_mapDepth5TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmaxA_HE = fs_->make<TH2F>("h_mapDepth6TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmaxA_HE = fs_->make<TH2F>("h_mapDepth7TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    // RMS:
    h_Amplitude_HE = fs_->make<TH1F>("h_Amplitude_HE", " ", 100, 0., 5.5);
    h_mapDepth1Amplitude225_HE = fs_->make<TH2F>("h_mapDepth1Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HE = fs_->make<TH2F>("h_mapDepth2Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HE = fs_->make<TH2F>("h_mapDepth3Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HE = fs_->make<TH2F>("h_mapDepth4Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Amplitude225_HE = fs_->make<TH2F>("h_mapDepth5Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Amplitude225_HE = fs_->make<TH2F>("h_mapDepth6Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Amplitude225_HE = fs_->make<TH2F>("h_mapDepth7Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HE = fs_->make<TH2F>("h_mapDepth1Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HE = fs_->make<TH2F>("h_mapDepth2Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HE = fs_->make<TH2F>("h_mapDepth3Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HE = fs_->make<TH2F>("h_mapDepth4Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Amplitude_HE = fs_->make<TH2F>("h_mapDepth5Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Amplitude_HE = fs_->make<TH2F>("h_mapDepth6Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Amplitude_HE = fs_->make<TH2F>("h_mapDepth7Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);

    // Ratio:
    h_Ampl_HE = fs_->make<TH1F>("h_Ampl_HE", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HE = fs_->make<TH2F>("h_mapDepth1Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HE = fs_->make<TH2F>("h_mapDepth2Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HE = fs_->make<TH2F>("h_mapDepth3Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HE = fs_->make<TH2F>("h_mapDepth4Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Ampl047_HE = fs_->make<TH2F>("h_mapDepth5Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Ampl047_HE = fs_->make<TH2F>("h_mapDepth6Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Ampl047_HE = fs_->make<TH2F>("h_mapDepth7Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HE = fs_->make<TH2F>("h_mapDepth1Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HE = fs_->make<TH2F>("h_mapDepth2Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HE = fs_->make<TH2F>("h_mapDepth3Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HE = fs_->make<TH2F>("h_mapDepth4Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Ampl_HE = fs_->make<TH2F>("h_mapDepth5Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Ampl_HE = fs_->make<TH2F>("h_mapDepth6Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Ampl_HE = fs_->make<TH2F>("h_mapDepth7Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HE = fs_->make<TH2F>("h_mapDepth1AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HE = fs_->make<TH2F>("h_mapDepth2AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HE = fs_->make<TH2F>("h_mapDepth3AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HE = fs_->make<TH2F>("h_mapDepth4AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5AmplE34_HE = fs_->make<TH2F>("h_mapDepth5AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6AmplE34_HE = fs_->make<TH2F>("h_mapDepth6AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7AmplE34_HE = fs_->make<TH2F>("h_mapDepth7AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HE = fs_->make<TH2F>("h_mapDepth1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HE = fs_->make<TH2F>("h_mapDepth2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HE = fs_->make<TH2F>("h_mapDepth3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HE = fs_->make<TH2F>("h_mapDepth4_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5_HE = fs_->make<TH2F>("h_mapDepth5_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6_HE = fs_->make<TH2F>("h_mapDepth6_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7_HE = fs_->make<TH2F>("h_mapDepth7_HE", " ", neta, -41., 41., nphi, 0., bphi);
    ///////////////////////////////////////////////////////////////////////////////////////////////////  IterativeMethodCalibrationGroup
    h_mapenophinorm_HE1 = fs_->make<TH2F>("h_mapenophinorm_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE2 = fs_->make<TH2F>("h_mapenophinorm_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE3 = fs_->make<TH2F>("h_mapenophinorm_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE4 = fs_->make<TH2F>("h_mapenophinorm_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE5 = fs_->make<TH2F>("h_mapenophinorm_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE6 = fs_->make<TH2F>("h_mapenophinorm_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE7 = fs_->make<TH2F>("h_mapenophinorm_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE1 = fs_->make<TH2F>("h_mapenophinorm2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE2 = fs_->make<TH2F>("h_mapenophinorm2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE3 = fs_->make<TH2F>("h_mapenophinorm2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE4 = fs_->make<TH2F>("h_mapenophinorm2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE5 = fs_->make<TH2F>("h_mapenophinorm2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE6 = fs_->make<TH2F>("h_mapenophinorm2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE7 = fs_->make<TH2F>("h_mapenophinorm2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    h_maprphinorm_HE1 = fs_->make<TH2F>("h_maprphinorm_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE2 = fs_->make<TH2F>("h_maprphinorm_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE3 = fs_->make<TH2F>("h_maprphinorm_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE4 = fs_->make<TH2F>("h_maprphinorm_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE5 = fs_->make<TH2F>("h_maprphinorm_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE6 = fs_->make<TH2F>("h_maprphinorm_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE7 = fs_->make<TH2F>("h_maprphinorm_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE1 = fs_->make<TH2F>("h_maprphinorm2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE2 = fs_->make<TH2F>("h_maprphinorm2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE3 = fs_->make<TH2F>("h_maprphinorm2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE4 = fs_->make<TH2F>("h_maprphinorm2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE5 = fs_->make<TH2F>("h_maprphinorm2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE6 = fs_->make<TH2F>("h_maprphinorm2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE7 = fs_->make<TH2F>("h_maprphinorm2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    h_maprphinorm0_HE1 = fs_->make<TH2F>("h_maprphinorm0_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE2 = fs_->make<TH2F>("h_maprphinorm0_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE3 = fs_->make<TH2F>("h_maprphinorm0_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE4 = fs_->make<TH2F>("h_maprphinorm0_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE5 = fs_->make<TH2F>("h_maprphinorm0_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE6 = fs_->make<TH2F>("h_maprphinorm0_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE7 = fs_->make<TH2F>("h_maprphinorm0_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    //
    // Didi as done in Reco
    //HB:
    h_amplitudechannel0_HB1 = fs_->make<TH2F>("h_amplitudechannel0_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HB1 = fs_->make<TH2F>("h_amplitudechannel1_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HB1 = fs_->make<TH2F>("h_amplitudechannel2_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HB2 = fs_->make<TH2F>("h_amplitudechannel0_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HB2 = fs_->make<TH2F>("h_amplitudechannel1_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HB2 = fs_->make<TH2F>("h_amplitudechannel2_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HB3 = fs_->make<TH2F>("h_amplitudechannel0_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HB3 = fs_->make<TH2F>("h_amplitudechannel1_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HB3 = fs_->make<TH2F>("h_amplitudechannel2_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HB4 = fs_->make<TH2F>("h_amplitudechannel0_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HB4 = fs_->make<TH2F>("h_amplitudechannel1_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HB4 = fs_->make<TH2F>("h_amplitudechannel2_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    //HE:
    h_amplitudechannel0_HE1 = fs_->make<TH2F>("h_amplitudechannel0_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE1 = fs_->make<TH2F>("h_amplitudechannel1_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE1 = fs_->make<TH2F>("h_amplitudechannel2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE2 = fs_->make<TH2F>("h_amplitudechannel0_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE2 = fs_->make<TH2F>("h_amplitudechannel1_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE2 = fs_->make<TH2F>("h_amplitudechannel2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE3 = fs_->make<TH2F>("h_amplitudechannel0_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE3 = fs_->make<TH2F>("h_amplitudechannel1_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE3 = fs_->make<TH2F>("h_amplitudechannel2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE4 = fs_->make<TH2F>("h_amplitudechannel0_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE4 = fs_->make<TH2F>("h_amplitudechannel1_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE4 = fs_->make<TH2F>("h_amplitudechannel2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE5 = fs_->make<TH2F>("h_amplitudechannel0_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE5 = fs_->make<TH2F>("h_amplitudechannel1_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE5 = fs_->make<TH2F>("h_amplitudechannel2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE6 = fs_->make<TH2F>("h_amplitudechannel0_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE6 = fs_->make<TH2F>("h_amplitudechannel1_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE6 = fs_->make<TH2F>("h_amplitudechannel2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HE7 = fs_->make<TH2F>("h_amplitudechannel0_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HE7 = fs_->make<TH2F>("h_amplitudechannel1_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HE7 = fs_->make<TH2F>("h_amplitudechannel2_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    //HF:
    h_amplitudechannel0_HF1 = fs_->make<TH2F>("h_amplitudechannel0_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HF1 = fs_->make<TH2F>("h_amplitudechannel1_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HF1 = fs_->make<TH2F>("h_amplitudechannel2_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HF2 = fs_->make<TH2F>("h_amplitudechannel0_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HF2 = fs_->make<TH2F>("h_amplitudechannel1_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HF2 = fs_->make<TH2F>("h_amplitudechannel2_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HF3 = fs_->make<TH2F>("h_amplitudechannel0_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HF3 = fs_->make<TH2F>("h_amplitudechannel1_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HF3 = fs_->make<TH2F>("h_amplitudechannel2_HF3", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel0_HF4 = fs_->make<TH2F>("h_amplitudechannel0_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel1_HF4 = fs_->make<TH2F>("h_amplitudechannel1_HF4", " ", neta, -41., 41., nphi, 0., bphi);
    h_amplitudechannel2_HF4 = fs_->make<TH2F>("h_amplitudechannel2_HF4", " ", neta, -41., 41., nphi, 0., bphi);

    // Reco
    h_energyhitSignal_HB = fs_->make<TH1F>("h_energyhitSignal_HB", " ", npfit, -0.22, 0.22);
    h_energyhitSignal_HE = fs_->make<TH1F>("h_energyhitSignal_HE", " ", npfit, -0.22, 0.22);
    h_energyhitSignal_HF = fs_->make<TH1F>("h_energyhitSignal_HF", " ", npfit, -6.6, 6.6);
    h_energyhitNoise_HB = fs_->make<TH1F>("h_energyhitNoise_HB", " ", npfit, -0.22, 0.22);
    h_energyhitNoise_HE = fs_->make<TH1F>("h_energyhitNoise_HE", " ", npfit, -0.22, 0.22);
    h_energyhitNoise_HF = fs_->make<TH1F>("h_energyhitNoise_HF", " ", npfit, -4.4, 4.4);

    //HB:
    h_recSignalEnergy0_HB1 = fs_->make<TH2F>("h_recSignalEnergy0_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HB1 = fs_->make<TH2F>("h_recSignalEnergy1_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HB1 = fs_->make<TH2F>("h_recSignalEnergy2_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HB2 = fs_->make<TH2F>("h_recSignalEnergy0_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HB2 = fs_->make<TH2F>("h_recSignalEnergy1_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HB2 = fs_->make<TH2F>("h_recSignalEnergy2_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HB3 = fs_->make<TH2F>("h_recSignalEnergy0_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HB3 = fs_->make<TH2F>("h_recSignalEnergy1_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HB3 = fs_->make<TH2F>("h_recSignalEnergy2_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HB4 = fs_->make<TH2F>("h_recSignalEnergy0_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HB4 = fs_->make<TH2F>("h_recSignalEnergy1_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HB4 = fs_->make<TH2F>("h_recSignalEnergy2_HB4", " ", neta, -41., 41., nphi, 0., bphi);

    h_recNoiseEnergy0_HB1 = fs_->make<TH2F>("h_recNoiseEnergy0_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HB1 = fs_->make<TH2F>("h_recNoiseEnergy1_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HB1 = fs_->make<TH2F>("h_recNoiseEnergy2_HB1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HB2 = fs_->make<TH2F>("h_recNoiseEnergy0_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HB2 = fs_->make<TH2F>("h_recNoiseEnergy1_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HB2 = fs_->make<TH2F>("h_recNoiseEnergy2_HB2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HB3 = fs_->make<TH2F>("h_recNoiseEnergy0_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HB3 = fs_->make<TH2F>("h_recNoiseEnergy1_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HB3 = fs_->make<TH2F>("h_recNoiseEnergy2_HB3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HB4 = fs_->make<TH2F>("h_recNoiseEnergy0_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HB4 = fs_->make<TH2F>("h_recNoiseEnergy1_HB4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HB4 = fs_->make<TH2F>("h_recNoiseEnergy2_HB4", " ", neta, -41., 41., nphi, 0., bphi);

    //HE:
    h_recSignalEnergy0_HE1 = fs_->make<TH2F>("h_recSignalEnergy0_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE1 = fs_->make<TH2F>("h_recSignalEnergy1_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE1 = fs_->make<TH2F>("h_recSignalEnergy2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE2 = fs_->make<TH2F>("h_recSignalEnergy0_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE2 = fs_->make<TH2F>("h_recSignalEnergy1_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE2 = fs_->make<TH2F>("h_recSignalEnergy2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE3 = fs_->make<TH2F>("h_recSignalEnergy0_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE3 = fs_->make<TH2F>("h_recSignalEnergy1_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE3 = fs_->make<TH2F>("h_recSignalEnergy2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE4 = fs_->make<TH2F>("h_recSignalEnergy0_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE4 = fs_->make<TH2F>("h_recSignalEnergy1_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE4 = fs_->make<TH2F>("h_recSignalEnergy2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE5 = fs_->make<TH2F>("h_recSignalEnergy0_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE5 = fs_->make<TH2F>("h_recSignalEnergy1_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE5 = fs_->make<TH2F>("h_recSignalEnergy2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE6 = fs_->make<TH2F>("h_recSignalEnergy0_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE6 = fs_->make<TH2F>("h_recSignalEnergy1_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE6 = fs_->make<TH2F>("h_recSignalEnergy2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HE7 = fs_->make<TH2F>("h_recSignalEnergy0_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HE7 = fs_->make<TH2F>("h_recSignalEnergy1_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HE7 = fs_->make<TH2F>("h_recSignalEnergy2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    h_recNoiseEnergy0_HE1 = fs_->make<TH2F>("h_recNoiseEnergy0_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE1 = fs_->make<TH2F>("h_recNoiseEnergy1_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE1 = fs_->make<TH2F>("h_recNoiseEnergy2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE2 = fs_->make<TH2F>("h_recNoiseEnergy0_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE2 = fs_->make<TH2F>("h_recNoiseEnergy1_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE2 = fs_->make<TH2F>("h_recNoiseEnergy2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE3 = fs_->make<TH2F>("h_recNoiseEnergy0_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE3 = fs_->make<TH2F>("h_recNoiseEnergy1_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE3 = fs_->make<TH2F>("h_recNoiseEnergy2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE4 = fs_->make<TH2F>("h_recNoiseEnergy0_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE4 = fs_->make<TH2F>("h_recNoiseEnergy1_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE4 = fs_->make<TH2F>("h_recNoiseEnergy2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE5 = fs_->make<TH2F>("h_recNoiseEnergy0_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE5 = fs_->make<TH2F>("h_recNoiseEnergy1_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE5 = fs_->make<TH2F>("h_recNoiseEnergy2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE6 = fs_->make<TH2F>("h_recNoiseEnergy0_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE6 = fs_->make<TH2F>("h_recNoiseEnergy1_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE6 = fs_->make<TH2F>("h_recNoiseEnergy2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HE7 = fs_->make<TH2F>("h_recNoiseEnergy0_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HE7 = fs_->make<TH2F>("h_recNoiseEnergy1_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HE7 = fs_->make<TH2F>("h_recNoiseEnergy2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    //HF:
    h_recSignalEnergy0_HF1 = fs_->make<TH2F>("h_recSignalEnergy0_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HF1 = fs_->make<TH2F>("h_recSignalEnergy1_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HF1 = fs_->make<TH2F>("h_recSignalEnergy2_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy0_HF2 = fs_->make<TH2F>("h_recSignalEnergy0_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy1_HF2 = fs_->make<TH2F>("h_recSignalEnergy1_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recSignalEnergy2_HF2 = fs_->make<TH2F>("h_recSignalEnergy2_HF2", " ", neta, -41., 41., nphi, 0., bphi);

    h_recNoiseEnergy0_HF1 = fs_->make<TH2F>("h_recNoiseEnergy0_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HF1 = fs_->make<TH2F>("h_recNoiseEnergy1_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HF1 = fs_->make<TH2F>("h_recNoiseEnergy2_HF1", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy0_HF2 = fs_->make<TH2F>("h_recNoiseEnergy0_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy1_HF2 = fs_->make<TH2F>("h_recNoiseEnergy1_HF2", " ", neta, -41., 41., nphi, 0., bphi);
    h_recNoiseEnergy2_HF2 = fs_->make<TH2F>("h_recNoiseEnergy2_HF2", " ", neta, -41., 41., nphi, 0., bphi);

    ///////////////////////////////////////////////////////////////////////////////////////////////////  raddam:
    // RADDAM:
    //    if(flagLaserRaddam_ == 1 ) {
    //    }
    int min80 = -100.;
    int max80 = 9000.;
    // fill for each digi (=each event, each channel)
    h_mapDepth1RADDAM_HE = fs_->make<TH2F>("h_mapDepth1RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2RADDAM_HE = fs_->make<TH2F>("h_mapDepth2RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3RADDAM_HE = fs_->make<TH2F>("h_mapDepth3RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1RADDAM0_HE = fs_->make<TH2F>("h_mapDepth1RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2RADDAM0_HE = fs_->make<TH2F>("h_mapDepth2RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3RADDAM0_HE = fs_->make<TH2F>("h_mapDepth3RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_sigLayer1RADDAM_HE = fs_->make<TH1F>("h_sigLayer1RADDAM_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM_HE = fs_->make<TH1F>("h_sigLayer2RADDAM_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM0_HE = fs_->make<TH1F>("h_sigLayer1RADDAM0_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM0_HE = fs_->make<TH1F>("h_sigLayer2RADDAM0_HE", " ", neta, -41., 41.);
    h_AamplitudewithPedSubtr_RADDAM_HE = fs_->make<TH1F>("h_AamplitudewithPedSubtr_RADDAM_HE", " ", 100, min80, max80);
    h_AamplitudewithPedSubtr_RADDAM_HEzoom0 =
        fs_->make<TH1F>("h_AamplitudewithPedSubtr_RADDAM_HEzoom0", " ", 100, min80, 4000.);
    h_AamplitudewithPedSubtr_RADDAM_HEzoom1 =
        fs_->make<TH1F>("h_AamplitudewithPedSubtr_RADDAM_HEzoom1", " ", 100, min80, 1000.);
    h_mapDepth3RADDAM16_HE = fs_->make<TH1F>("h_mapDepth3RADDAM16_HE", " ", 100, min80, max80);
    h_A_Depth1RADDAM_HE = fs_->make<TH1F>("h_A_Depth1RADDAM_HE", " ", 100, min80, max80);
    h_A_Depth2RADDAM_HE = fs_->make<TH1F>("h_A_Depth2RADDAM_HE", " ", 100, min80, max80);
    h_A_Depth3RADDAM_HE = fs_->make<TH1F>("h_A_Depth3RADDAM_HE", " ", 100, min80, max80);
    int min90 = 0.;
    int max90 = 5000.;
    h_sumphiEta16Depth3RADDAM_HED2 = fs_->make<TH1F>("h_sumphiEta16Depth3RADDAM_HED2", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2 = fs_->make<TH1F>("h_Eta16Depth3RADDAM_HED2", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2 = fs_->make<TH1F>("h_NphiForEta16Depth3RADDAM_HED2", " ", 100, 0, 100.);
    h_sumphiEta16Depth3RADDAM_HED2P = fs_->make<TH1F>("h_sumphiEta16Depth3RADDAM_HED2P", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2P = fs_->make<TH1F>("h_Eta16Depth3RADDAM_HED2P", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2P = fs_->make<TH1F>("h_NphiForEta16Depth3RADDAM_HED2P", " ", 100, 0, 100.);
    h_sumphiEta16Depth3RADDAM_HED2ALL =
        fs_->make<TH1F>("h_sumphiEta16Depth3RADDAM_HED2ALL", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2ALL = fs_->make<TH1F>("h_Eta16Depth3RADDAM_HED2ALL", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2ALL = fs_->make<TH1F>("h_NphiForEta16Depth3RADDAM_HED2ALL", " ", 100, 0, 100.);
    h_sigLayer1RADDAM5_HE = fs_->make<TH1F>("h_sigLayer1RADDAM5_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM5_HE = fs_->make<TH1F>("h_sigLayer2RADDAM5_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM6_HE = fs_->make<TH1F>("h_sigLayer1RADDAM6_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM6_HE = fs_->make<TH1F>("h_sigLayer2RADDAM6_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM5_HED2 = fs_->make<TH1F>("h_sigLayer1RADDAM5_HED2", " ", neta, -41., 41.);
    h_sigLayer2RADDAM5_HED2 = fs_->make<TH1F>("h_sigLayer2RADDAM5_HED2", " ", neta, -41., 41.);
    h_sigLayer1RADDAM6_HED2 = fs_->make<TH1F>("h_sigLayer1RADDAM6_HED2", " ", neta, -41., 41.);
    h_sigLayer2RADDAM6_HED2 = fs_->make<TH1F>("h_sigLayer2RADDAM6_HED2", " ", neta, -41., 41.);

    h_numberofhitsHFtest = fs_->make<TH1F>("h_numberofhitsHFtest", " ", 100, 0., 30000.);
    h_AmplitudeHFtest = fs_->make<TH1F>("h_AmplitudeHFtest", " ", 100, 0., 300000.);
    h_totalAmplitudeHF = fs_->make<TH1F>("h_totalAmplitudeHF", " ", 100, 0., 100000000000.);
    h_totalAmplitudeHFperEvent = fs_->make<TH1F>("h_totalAmplitudeHFperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmplZoom1_HF = fs_->make<TH1F>("h_ADCAmplZoom1_HF", " ", 100, 0., 1000000.);
    h_ADCAmpl_HF = fs_->make<TH1F>("h_ADCAmpl_HF", " ", 250, 0., 500000.);
    h_ADCAmplrest1_HF = fs_->make<TH1F>("h_ADCAmplrest1_HF", " ", 100, 0., 1000.);
    h_ADCAmplrest6_HF = fs_->make<TH1F>("h_ADCAmplrest6_HF", " ", 100, 0., 10000.);

    h_mapDepth1ADCAmpl225_HF = fs_->make<TH2F>("h_mapDepth1ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HF = fs_->make<TH2F>("h_mapDepth2ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HF =
        fs_->make<TH2F>("h_mapDepth1ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HF =
        fs_->make<TH2F>("h_mapDepth2ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl_HF = fs_->make<TH2F>("h_mapDepth1ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HF = fs_->make<TH2F>("h_mapDepth2ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HF = fs_->make<TH2F>("h_mapDepth3ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HF = fs_->make<TH2F>("h_mapDepth4ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HF =
        fs_->make<TH2F>("h_mapDepth3ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HF =
        fs_->make<TH2F>("h_mapDepth4ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HF = fs_->make<TH2F>("h_mapDepth3ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HF = fs_->make<TH2F>("h_mapDepth4ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HF = fs_->make<TH1F>("h_TSmeanA_HF", " ", 100, -1., 11.);
    h_mapDepth1TSmeanA225_HF = fs_->make<TH2F>("h_mapDepth1TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HF = fs_->make<TH2F>("h_mapDepth2TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HF = fs_->make<TH2F>("h_mapDepth1TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HF = fs_->make<TH2F>("h_mapDepth2TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HF = fs_->make<TH2F>("h_mapDepth3TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HF = fs_->make<TH2F>("h_mapDepth4TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HF = fs_->make<TH2F>("h_mapDepth3TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HF = fs_->make<TH2F>("h_mapDepth4TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HF = fs_->make<TH1F>("h_Amplitude_HF", " ", 100, 0., 5.);
    h_TSmaxA_HF = fs_->make<TH1F>("h_TSmaxA_HF", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HF = fs_->make<TH2F>("h_mapDepth1TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HF = fs_->make<TH2F>("h_mapDepth2TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HF = fs_->make<TH2F>("h_mapDepth1TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HF = fs_->make<TH2F>("h_mapDepth2TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HF = fs_->make<TH2F>("h_mapDepth3TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HF = fs_->make<TH2F>("h_mapDepth4TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HF = fs_->make<TH2F>("h_mapDepth3TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HF = fs_->make<TH2F>("h_mapDepth4TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HF = fs_->make<TH1F>("h_Amplitude_HF", " ", 100, 0., 5.);
    h_mapDepth1Amplitude225_HF = fs_->make<TH2F>("h_mapDepth1Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HF = fs_->make<TH2F>("h_mapDepth2Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HF = fs_->make<TH2F>("h_mapDepth1Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HF = fs_->make<TH2F>("h_mapDepth2Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HF = fs_->make<TH2F>("h_mapDepth3Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HF = fs_->make<TH2F>("h_mapDepth4Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HF = fs_->make<TH2F>("h_mapDepth3Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HF = fs_->make<TH2F>("h_mapDepth4Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HF = fs_->make<TH1F>("h_Ampl_HF", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HF = fs_->make<TH2F>("h_mapDepth1Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HF = fs_->make<TH2F>("h_mapDepth2Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HF = fs_->make<TH2F>("h_mapDepth1Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HF = fs_->make<TH2F>("h_mapDepth2Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HF = fs_->make<TH2F>("h_mapDepth1AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HF = fs_->make<TH2F>("h_mapDepth2AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HF = fs_->make<TH2F>("h_mapDepth1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HF = fs_->make<TH2F>("h_mapDepth2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HF = fs_->make<TH2F>("h_mapDepth3Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HF = fs_->make<TH2F>("h_mapDepth4Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HF = fs_->make<TH2F>("h_mapDepth3Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HF = fs_->make<TH2F>("h_mapDepth4Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HF = fs_->make<TH2F>("h_mapDepth3AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HF = fs_->make<TH2F>("h_mapDepth4AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HF = fs_->make<TH2F>("h_mapDepth3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HF = fs_->make<TH2F>("h_mapDepth4_HF", " ", neta, -41., 41., nphi, 0., bphi);

    ////////////////////////////////////////////////////////////////////////////////////////////////                  HO
    h_numberofhitsHOtest = fs_->make<TH1F>("h_numberofhitsHOtest", " ", 100, 0., 30000.);
    h_AmplitudeHOtest = fs_->make<TH1F>("h_AmplitudeHOtest", " ", 100, 0., 300000.);
    h_totalAmplitudeHO = fs_->make<TH1F>("h_totalAmplitudeHO", " ", 100, 0., 100000000.);
    h_totalAmplitudeHOperEvent = fs_->make<TH1F>("h_totalAmplitudeHOperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmpl_HO = fs_->make<TH1F>("h_ADCAmpl_HO", " ", 100, 0., 7000.);
    h_ADCAmplrest1_HO = fs_->make<TH1F>("h_ADCAmplrest1_HO", " ", 100, 0., 150.);
    h_ADCAmplrest6_HO = fs_->make<TH1F>("h_ADCAmplrest6_HO", " ", 100, 0., 500.);

    h_ADCAmplZoom1_HO = fs_->make<TH1F>("h_ADCAmplZoom1_HO", " ", 100, -20., 280.);
    h_ADCAmpl_HO_copy = fs_->make<TH1F>("h_ADCAmpl_HO_copy", " ", 100, 0., 30000.);
    h_mapDepth4ADCAmpl225_HO = fs_->make<TH2F>("h_mapDepth4ADCAmpl225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HO =
        fs_->make<TH2F>("h_mapDepth4ADCAmpl225Copy_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HO = fs_->make<TH2F>("h_mapDepth4ADCAmpl_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HO = fs_->make<TH1F>("h_TSmeanA_HO", " ", 100, 0., 10.);
    h_mapDepth4TSmeanA225_HO = fs_->make<TH2F>("h_mapDepth4TSmeanA225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HO = fs_->make<TH2F>("h_mapDepth4TSmeanA_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HO = fs_->make<TH1F>("h_TSmaxA_HO", " ", 100, 0., 10.);
    h_mapDepth4TSmaxA225_HO = fs_->make<TH2F>("h_mapDepth4TSmaxA225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HO = fs_->make<TH2F>("h_mapDepth4TSmaxA_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HO = fs_->make<TH1F>("h_Amplitude_HO", " ", 100, 0., 5.);
    h_mapDepth4Amplitude225_HO = fs_->make<TH2F>("h_mapDepth4Amplitude225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HO = fs_->make<TH2F>("h_mapDepth4Amplitude_HO", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HO = fs_->make<TH1F>("h_Ampl_HO", " ", 100, 0., 1.1);
    h_mapDepth4Ampl047_HO = fs_->make<TH2F>("h_mapDepth4Ampl047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HO = fs_->make<TH2F>("h_mapDepth4Ampl_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HO = fs_->make<TH2F>("h_mapDepth4AmplE34_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HO = fs_->make<TH2F>("h_mapDepth4_HO", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////
    int baP = 4000;
    float baR = 0.;
    float baR2 = baP;
    h_bcnnbadchannels_depth1_HB = fs_->make<TH1F>("h_bcnnbadchannels_depth1_HB", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HB = fs_->make<TH1F>("h_bcnnbadchannels_depth2_HB", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth1_HE = fs_->make<TH1F>("h_bcnnbadchannels_depth1_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HE = fs_->make<TH1F>("h_bcnnbadchannels_depth2_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth3_HE = fs_->make<TH1F>("h_bcnnbadchannels_depth3_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth4_HO = fs_->make<TH1F>("h_bcnnbadchannels_depth4_HO", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth1_HF = fs_->make<TH1F>("h_bcnnbadchannels_depth1_HF", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HF = fs_->make<TH1F>("h_bcnnbadchannels_depth2_HF", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HB = fs_->make<TH1F>("h_bcnbadrate0_depth1_HB", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HB = fs_->make<TH1F>("h_bcnbadrate0_depth2_HB", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HE = fs_->make<TH1F>("h_bcnbadrate0_depth1_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HE = fs_->make<TH1F>("h_bcnbadrate0_depth2_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth3_HE = fs_->make<TH1F>("h_bcnbadrate0_depth3_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth4_HO = fs_->make<TH1F>("h_bcnbadrate0_depth4_HO", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HF = fs_->make<TH1F>("h_bcnbadrate0_depth1_HF", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HF = fs_->make<TH1F>("h_bcnbadrate0_depth2_HF", " ", baP, baR, baR2);

    h_bcnvsamplitude_HB = fs_->make<TH1F>("h_bcnvsamplitude_HB", " ", baP, baR, baR2);
    h_bcnvsamplitude_HE = fs_->make<TH1F>("h_bcnvsamplitude_HE", " ", baP, baR, baR2);
    h_bcnvsamplitude_HF = fs_->make<TH1F>("h_bcnvsamplitude_HF", " ", baP, baR, baR2);
    h_bcnvsamplitude_HO = fs_->make<TH1F>("h_bcnvsamplitude_HO", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HB = fs_->make<TH1F>("h_bcnvsamplitude0_HB", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HE = fs_->make<TH1F>("h_bcnvsamplitude0_HE", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HF = fs_->make<TH1F>("h_bcnvsamplitude0_HF", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HO = fs_->make<TH1F>("h_bcnvsamplitude0_HO", " ", baP, baR, baR2);

    int zaP = 1000;
    float zaR = 10000000.;
    float zaR2 = 50000000.;
    h_orbitNumvsamplitude_HB = fs_->make<TH1F>("h_orbitNumvsamplitude_HB", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HE = fs_->make<TH1F>("h_orbitNumvsamplitude_HE", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HF = fs_->make<TH1F>("h_orbitNumvsamplitude_HF", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HO = fs_->make<TH1F>("h_orbitNumvsamplitude_HO", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HB = fs_->make<TH1F>("h_orbitNumvsamplitude0_HB", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HE = fs_->make<TH1F>("h_orbitNumvsamplitude0_HE", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HF = fs_->make<TH1F>("h_orbitNumvsamplitude0_HF", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HO = fs_->make<TH1F>("h_orbitNumvsamplitude0_HO", " ", zaP, zaR, zaR2);

    h_2DsumADCAmplEtaPhiLs0 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs0", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs1 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs1", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs2 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs2", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs3 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs3", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);

    h_2DsumADCAmplEtaPhiLs00 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs00", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs10 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs10", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs20 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs20", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs30 = fs_->make<TH2F>(
        "h_2DsumADCAmplEtaPhiLs30", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);

    h_sumADCAmplEtaPhiLs = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs", " ", 1000, 0., 14000.);
    h_sumADCAmplEtaPhiLs_bbbc = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_bbbc", " ", 1000, 0., 300000.);
    h_sumADCAmplEtaPhiLs_bbb1 = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_bbb1", " ", 100, 0., 3000.);
    h_sumADCAmplEtaPhiLs_lscounterM1 = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_lscounterM1", " ", 600, 1., 601.);
    h_sumADCAmplEtaPhiLs_ietaphi = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_ietaphi", " ", 400, 0., 400.);
    h_sumADCAmplEtaPhiLs_lscounterM1orbitNum =
        fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_lscounterM1orbitNum", " ", 600, 1., 601.);
    h_sumADCAmplEtaPhiLs_orbitNum = fs_->make<TH1F>("h_sumADCAmplEtaPhiLs_orbitNum", " ", 1000, 25000000., 40000000.);

    // for LS :

    // for LS binning:
    int bac = howmanybinsonplots_;
    //  int bac= 15;
    float bac2 = bac + 1.;
    // bac,         1.,     bac2  );

    h_nbadchannels_depth1_HB = fs_->make<TH1F>("h_nbadchannels_depth1_HB", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HB = fs_->make<TH1F>("h_runnbadchannels_depth1_HB", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HB = fs_->make<TH1F>("h_runnbadchannelsC_depth1_HB", " ", bac, 1., bac2);
    h_runbadrate_depth1_HB = fs_->make<TH1F>("h_runbadrate_depth1_HB", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HB = fs_->make<TH1F>("h_runbadrateC_depth1_HB", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HB = fs_->make<TH1F>("h_runbadrate0_depth1_HB", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HB = fs_->make<TH1F>("h_nbadchannels_depth2_HB", " ", 100, 1., 501.);
    h_runnbadchannels_depth2_HB = fs_->make<TH1F>("h_runnbadchannels_depth2_HB", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HB = fs_->make<TH1F>("h_runnbadchannelsC_depth2_HB", " ", bac, 1., bac2);
    h_runbadrate_depth2_HB = fs_->make<TH1F>("h_runbadrate_depth2_HB", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HB = fs_->make<TH1F>("h_runbadrateC_depth2_HB", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HB = fs_->make<TH1F>("h_runbadrate0_depth2_HB", " ", bac, 1., bac2);

    h_nbadchannels_depth1_HE = fs_->make<TH1F>("h_nbadchannels_depth1_HE", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HE = fs_->make<TH1F>("h_runnbadchannels_depth1_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HE = fs_->make<TH1F>("h_runnbadchannelsC_depth1_HE", " ", bac, 1., bac2);
    h_runbadrate_depth1_HE = fs_->make<TH1F>("h_runbadrate_depth1_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HE = fs_->make<TH1F>("h_runbadrateC_depth1_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HE = fs_->make<TH1F>("h_runbadrate0_depth1_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HE = fs_->make<TH1F>("h_nbadchannels_depth2_HE", " ", 100, 1., 3001.);
    h_runnbadchannels_depth2_HE = fs_->make<TH1F>("h_runnbadchannels_depth2_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HE = fs_->make<TH1F>("h_runnbadchannelsC_depth2_HE", " ", bac, 1., bac2);
    h_runbadrate_depth2_HE = fs_->make<TH1F>("h_runbadrate_depth2_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HE = fs_->make<TH1F>("h_runbadrateC_depth2_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HE = fs_->make<TH1F>("h_runbadrate0_depth2_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth3_HE = fs_->make<TH1F>("h_nbadchannels_depth3_HE", " ", 100, 1., 501.);
    h_runnbadchannels_depth3_HE = fs_->make<TH1F>("h_runnbadchannels_depth3_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth3_HE = fs_->make<TH1F>("h_runnbadchannelsC_depth3_HE", " ", bac, 1., bac2);
    h_runbadrate_depth3_HE = fs_->make<TH1F>("h_runbadrate_depth3_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth3_HE = fs_->make<TH1F>("h_runbadrateC_depth3_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth3_HE = fs_->make<TH1F>("h_runbadrate0_depth3_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth1_HF = fs_->make<TH1F>("h_nbadchannels_depth1_HF", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HF = fs_->make<TH1F>("h_runnbadchannels_depth1_HF", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HF = fs_->make<TH1F>("h_runnbadchannelsC_depth1_HF", " ", bac, 1., bac2);
    h_runbadrate_depth1_HF = fs_->make<TH1F>("h_runbadrate_depth1_HF", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HF = fs_->make<TH1F>("h_runbadrateC_depth1_HF", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HF = fs_->make<TH1F>("h_runbadrate0_depth1_HF", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HF = fs_->make<TH1F>("h_nbadchannels_depth2_HF", " ", 100, 1., 501.);
    h_runnbadchannels_depth2_HF = fs_->make<TH1F>("h_runnbadchannels_depth2_HF", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HF = fs_->make<TH1F>("h_runnbadchannelsC_depth2_HF", " ", bac, 1., bac2);
    h_runbadrate_depth2_HF = fs_->make<TH1F>("h_runbadrate_depth2_HF", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HF = fs_->make<TH1F>("h_runbadrateC_depth2_HF", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HF = fs_->make<TH1F>("h_runbadrate0_depth2_HF", " ", bac, 1., bac2);

    h_nbadchannels_depth4_HO = fs_->make<TH1F>("h_nbadchannels_depth4_HO", " ", 100, 1., 3001.);
    h_runnbadchannels_depth4_HO = fs_->make<TH1F>("h_runnbadchannels_depth4_HO", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth4_HO = fs_->make<TH1F>("h_runnbadchannelsC_depth4_HO", " ", bac, 1., bac2);
    h_runbadrate_depth4_HO = fs_->make<TH1F>("h_runbadrate_depth4_HO", " ", bac, 1., bac2);
    h_runbadrateC_depth4_HO = fs_->make<TH1F>("h_runbadrateC_depth4_HO", " ", bac, 1., bac2);
    h_runbadrate0_depth4_HO = fs_->make<TH1F>("h_runbadrate0_depth4_HO", " ", bac, 1., bac2);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    h_FullSignal3D_HB = fs_->make<TH2F>("h_FullSignal3D_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HB = fs_->make<TH2F>("h_FullSignal3D0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HE = fs_->make<TH2F>("h_FullSignal3D_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HE = fs_->make<TH2F>("h_FullSignal3D0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HO = fs_->make<TH2F>("h_FullSignal3D_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HO = fs_->make<TH2F>("h_FullSignal3D0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HF = fs_->make<TH2F>("h_FullSignal3D_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HF = fs_->make<TH2F>("h_FullSignal3D0_HF", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////////////////
    h_ADCCalib_HB = fs_->make<TH1F>("h_ADCCalib_HB", " ", 100, 10., 10000.);
    h_ADCCalib1_HB = fs_->make<TH1F>("h_ADCCalib1_HB", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HB = fs_->make<TH2F>("h_mapADCCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HB = fs_->make<TH2F>("h_mapADCCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HB = fs_->make<TH1F>("h_RatioCalib_HB", " ", 100, 0., 1.);
    h_mapRatioCalib047_HB = fs_->make<TH2F>("h_mapRatioCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HB = fs_->make<TH2F>("h_mapRatioCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HB = fs_->make<TH1F>("h_TSmaxCalib_HB", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HB = fs_->make<TH2F>("h_mapTSmaxCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HB = fs_->make<TH2F>("h_mapTSmaxCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HB = fs_->make<TH1F>("h_TSmeanCalib_HB", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HB = fs_->make<TH2F>("h_mapTSmeanCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HB = fs_->make<TH2F>("h_mapTSmeanCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HB = fs_->make<TH1F>("h_WidthCalib_HB", " ", 100, 0., 5.);
    h_mapWidthCalib047_HB = fs_->make<TH2F>("h_mapWidthCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HB = fs_->make<TH2F>("h_mapCapCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HB = fs_->make<TH2F>("h_mapWidthCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HB = fs_->make<TH2F>("h_map_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HE = fs_->make<TH1F>("h_ADCCalib_HE", " ", 100, 10., 10000.);
    h_ADCCalib1_HE = fs_->make<TH1F>("h_ADCCalib1_HE", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HE = fs_->make<TH2F>("h_mapADCCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HE = fs_->make<TH2F>("h_mapADCCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HE = fs_->make<TH1F>("h_RatioCalib_HE", " ", 100, 0., 1.);
    h_mapRatioCalib047_HE = fs_->make<TH2F>("h_mapRatioCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HE = fs_->make<TH2F>("h_mapRatioCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HE = fs_->make<TH1F>("h_TSmaxCalib_HE", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HE = fs_->make<TH2F>("h_mapTSmaxCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HE = fs_->make<TH2F>("h_mapTSmaxCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HE = fs_->make<TH1F>("h_TSmeanCalib_HE", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HE = fs_->make<TH2F>("h_mapTSmeanCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HE = fs_->make<TH2F>("h_mapTSmeanCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HE = fs_->make<TH1F>("h_WidthCalib_HE", " ", 100, 0., 5.);
    h_mapWidthCalib047_HE = fs_->make<TH2F>("h_mapWidthCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HE = fs_->make<TH2F>("h_mapCapCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HE = fs_->make<TH2F>("h_mapWidthCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HE = fs_->make<TH2F>("h_map_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HO = fs_->make<TH1F>("h_ADCCalib_HO", " ", 100, 10., 10000.);
    h_ADCCalib1_HO = fs_->make<TH1F>("h_ADCCalib1_HO", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HO = fs_->make<TH2F>("h_mapADCCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HO = fs_->make<TH2F>("h_mapADCCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HO = fs_->make<TH1F>("h_RatioCalib_HO", " ", 100, 0., 1.);
    h_mapRatioCalib047_HO = fs_->make<TH2F>("h_mapRatioCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HO = fs_->make<TH2F>("h_mapRatioCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HO = fs_->make<TH1F>("h_TSmaxCalib_HO", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HO = fs_->make<TH2F>("h_mapTSmaxCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HO = fs_->make<TH2F>("h_mapTSmaxCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HO = fs_->make<TH1F>("h_TSmeanCalib_HO", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HO = fs_->make<TH2F>("h_mapTSmeanCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HO = fs_->make<TH2F>("h_mapTSmeanCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HO = fs_->make<TH1F>("h_WidthCalib_HO", " ", 100, 0., 5.);
    h_mapWidthCalib047_HO = fs_->make<TH2F>("h_mapWidthCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HO = fs_->make<TH2F>("h_mapCapCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HO = fs_->make<TH2F>("h_mapWidthCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HO = fs_->make<TH2F>("h_map_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HF = fs_->make<TH1F>("h_ADCCalib_HF", " ", 100, 10., 2000.);
    h_ADCCalib1_HF = fs_->make<TH1F>("h_ADCCalib1_HF", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HF = fs_->make<TH2F>("h_mapADCCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HF = fs_->make<TH2F>("h_mapADCCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HF = fs_->make<TH1F>("h_RatioCalib_HF", " ", 100, 0., 1.);
    h_mapRatioCalib047_HF = fs_->make<TH2F>("h_mapRatioCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HF = fs_->make<TH2F>("h_mapRatioCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HF = fs_->make<TH1F>("h_TSmaxCalib_HF", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HF = fs_->make<TH2F>("h_mapTSmaxCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HF = fs_->make<TH2F>("h_mapTSmaxCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HF = fs_->make<TH1F>("h_TSmeanCalib_HF", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HF = fs_->make<TH2F>("h_mapTSmeanCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HF = fs_->make<TH2F>("h_mapTSmeanCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HF = fs_->make<TH1F>("h_WidthCalib_HF", " ", 100, 0., 5.);
    h_mapWidthCalib047_HF = fs_->make<TH2F>("h_mapWidthCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HF = fs_->make<TH2F>("h_mapCapCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HF = fs_->make<TH2F>("h_mapWidthCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HF = fs_->make<TH2F>("h_map_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_nls_per_run = fs_->make<TH1F>("h_nls_per_run", " ", 100, 0., 800.);
    h_nls_per_run10 = fs_->make<TH1F>("h_nls_per_run10", " ", 100, 0., 60.);
    h_nevents_per_LS = fs_->make<TH1F>("h_nevents_per_LS", " ", 100, 0., 600.);
    h_nevents_per_LSzoom = fs_->make<TH1F>("h_nevents_per_LSzoom", " ", 50, 0., 50.);
    h_nevents_per_eachLS = fs_->make<TH1F>("h_nevents_per_eachLS", " ", bac, 1., bac2);
    h_nevents_per_eachRealLS = fs_->make<TH1F>("h_nevents_per_eachRealLS", " ", bac, 1., bac2);
    h_lsnumber_per_eachLS = fs_->make<TH1F>("h_lsnumber_per_eachLS", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator0:
    float pst1 = 30.;
    h_sumPedestalLS1 = fs_->make<TH1F>("h_sumPedestalLS1", " ", 100, 0., pst1);
    h_2DsumPedestalLS1 = fs_->make<TH2F>("h_2DsumPedestalLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS1 = fs_->make<TH1F>("h_sumPedestalperLS1", " ", bac, 1., bac2);
    h_2D0sumPedestalLS1 = fs_->make<TH2F>("h_2D0sumPedestalLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS1 = fs_->make<TH1F>("h_sum0PedestalperLS1", " ", bac, 1., bac2);

    h_sumPedestalLS2 = fs_->make<TH1F>("h_sumPedestalLS2", " ", 100, 0., pst1);
    h_2DsumPedestalLS2 = fs_->make<TH2F>("h_2DsumPedestalLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS2 = fs_->make<TH1F>("h_sumPedestalperLS2", " ", bac, 1., bac2);
    h_2D0sumPedestalLS2 = fs_->make<TH2F>("h_2D0sumPedestalLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS2 = fs_->make<TH1F>("h_sum0PedestalperLS2", " ", bac, 1., bac2);

    h_sumPedestalLS3 = fs_->make<TH1F>("h_sumPedestalLS3", " ", 100, 0., pst1);
    h_2DsumPedestalLS3 = fs_->make<TH2F>("h_2DsumPedestalLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS3 = fs_->make<TH1F>("h_sumPedestalperLS3", " ", bac, 1., bac2);
    h_2D0sumPedestalLS3 = fs_->make<TH2F>("h_2D0sumPedestalLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS3 = fs_->make<TH1F>("h_sum0PedestalperLS3", " ", bac, 1., bac2);

    h_sumPedestalLS4 = fs_->make<TH1F>("h_sumPedestalLS4", " ", 100, 0., pst1);
    h_2DsumPedestalLS4 = fs_->make<TH2F>("h_2DsumPedestalLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS4 = fs_->make<TH1F>("h_sumPedestalperLS4", " ", bac, 1., bac2);
    h_2D0sumPedestalLS4 = fs_->make<TH2F>("h_2D0sumPedestalLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS4 = fs_->make<TH1F>("h_sum0PedestalperLS4", " ", bac, 1., bac2);

    h_sumPedestalLS5 = fs_->make<TH1F>("h_sumPedestalLS5", " ", 100, 0., pst1);
    h_2DsumPedestalLS5 = fs_->make<TH2F>("h_2DsumPedestalLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS5 = fs_->make<TH1F>("h_sumPedestalperLS5", " ", bac, 1., bac2);
    h_2D0sumPedestalLS5 = fs_->make<TH2F>("h_2D0sumPedestalLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS5 = fs_->make<TH1F>("h_sum0PedestalperLS5", " ", bac, 1., bac2);

    h_sumPedestalLS6 = fs_->make<TH1F>("h_sumPedestalLS6", " ", 100, 0., pst1);
    h_2DsumPedestalLS6 = fs_->make<TH2F>("h_2DsumPedestalLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS6 = fs_->make<TH1F>("h_sumPedestalperLS6", " ", bac, 1., bac2);
    h_2D0sumPedestalLS6 = fs_->make<TH2F>("h_2D0sumPedestalLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS6 = fs_->make<TH1F>("h_sum0PedestalperLS6", " ", bac, 1., bac2);

    h_sumPedestalLS7 = fs_->make<TH1F>("h_sumPedestalLS7", " ", 100, 0., pst1);
    h_2DsumPedestalLS7 = fs_->make<TH2F>("h_2DsumPedestalLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS7 = fs_->make<TH1F>("h_sumPedestalperLS7", " ", bac, 1., bac2);
    h_2D0sumPedestalLS7 = fs_->make<TH2F>("h_2D0sumPedestalLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS7 = fs_->make<TH1F>("h_sum0PedestalperLS7", " ", bac, 1., bac2);

    h_sumPedestalLS8 = fs_->make<TH1F>("h_sumPedestalLS8", " ", 100, 0., pst1);
    h_2DsumPedestalLS8 = fs_->make<TH2F>("h_2DsumPedestalLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS8 = fs_->make<TH1F>("h_sumPedestalperLS8", " ", bac, 1., bac2);
    h_2D0sumPedestalLS8 = fs_->make<TH2F>("h_2D0sumPedestalLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS8 = fs_->make<TH1F>("h_sum0PedestalperLS8", " ", bac, 1., bac2);

    //--------------------------------------------------
    // for estimator1:
    h_sumADCAmplLS1copy1 = fs_->make<TH1F>("h_sumADCAmplLS1copy1", " ", 100, 0., 10000);
    h_sumADCAmplLS1copy2 = fs_->make<TH1F>("h_sumADCAmplLS1copy2", " ", 100, 0., 20000);
    h_sumADCAmplLS1copy3 = fs_->make<TH1F>("h_sumADCAmplLS1copy3", " ", 100, 0., 50000);
    h_sumADCAmplLS1copy4 = fs_->make<TH1F>("h_sumADCAmplLS1copy4", " ", 100, 0., 100000);
    h_sumADCAmplLS1copy5 = fs_->make<TH1F>("h_sumADCAmplLS1copy5", " ", 100, 0., 150000);
    h_sumADCAmplLS1 = fs_->make<TH1F>("h_sumADCAmplLS1", " ", 100, 0., lsdep_estimator1_HBdepth1_);
    h_2DsumADCAmplLS1 = fs_->make<TH2F>("h_2DsumADCAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS1_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS1_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS1 = fs_->make<TH1F>("h_sumADCAmplperLS1", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS1 = fs_->make<TH1F>("h_sumCutADCAmplperLS1", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS1 = fs_->make<TH2F>("h_2D0sumADCAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS1 = fs_->make<TH1F>("h_sum0ADCAmplperLS1", " ", bac, 1., bac2);

    h_sumADCAmplLS2 = fs_->make<TH1F>("h_sumADCAmplLS2", " ", 100, 0., lsdep_estimator1_HBdepth2_);
    h_2DsumADCAmplLS2 = fs_->make<TH2F>("h_2DsumADCAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS2_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS2_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS2 = fs_->make<TH1F>("h_sumADCAmplperLS2", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS2 = fs_->make<TH1F>("h_sumCutADCAmplperLS2", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS2 = fs_->make<TH2F>("h_2D0sumADCAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS2 = fs_->make<TH1F>("h_sum0ADCAmplperLS2", " ", bac, 1., bac2);

    h_sumADCAmplLS3 = fs_->make<TH1F>("h_sumADCAmplLS3", " ", 100, 0., lsdep_estimator1_HEdepth1_);
    h_2DsumADCAmplLS3 = fs_->make<TH2F>("h_2DsumADCAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS3_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS3_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS3 = fs_->make<TH1F>("h_sumADCAmplperLS3", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS3 = fs_->make<TH1F>("h_sumCutADCAmplperLS3", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS3 = fs_->make<TH2F>("h_2D0sumADCAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS3 = fs_->make<TH1F>("h_sum0ADCAmplperLS3", " ", bac, 1., bac2);

    h_sumADCAmplLS4 = fs_->make<TH1F>("h_sumADCAmplLS4", " ", 100, 0., lsdep_estimator1_HEdepth2_);
    h_2DsumADCAmplLS4 = fs_->make<TH2F>("h_2DsumADCAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS4_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS4_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS4 = fs_->make<TH1F>("h_sumADCAmplperLS4", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS4 = fs_->make<TH1F>("h_sumCutADCAmplperLS4", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS4 = fs_->make<TH2F>("h_2D0sumADCAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS4 = fs_->make<TH1F>("h_sum0ADCAmplperLS4", " ", bac, 1., bac2);

    h_sumADCAmplLS5 = fs_->make<TH1F>("h_sumADCAmplLS5", " ", 100, 0., lsdep_estimator1_HEdepth3_);
    h_2DsumADCAmplLS5 = fs_->make<TH2F>("h_2DsumADCAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS5_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS5_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS5 = fs_->make<TH1F>("h_sumADCAmplperLS5", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS5 = fs_->make<TH1F>("h_sumCutADCAmplperLS5", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS5 = fs_->make<TH2F>("h_2D0sumADCAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS5 = fs_->make<TH1F>("h_sum0ADCAmplperLS5", " ", bac, 1., bac2);
    // HE upgrade depth4
    h_sumADCAmplperLSdepth4HEu = fs_->make<TH1F>("h_sumADCAmplperLSdepth4HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth4HEu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth4HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth4HEu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth4HEu", " ", bac, 1., bac2);

    // HE upgrade depth5
    h_sumADCAmplperLSdepth5HEu = fs_->make<TH1F>("h_sumADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth5HEu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth5HEu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    // HE upgrade depth6
    h_sumADCAmplperLSdepth6HEu = fs_->make<TH1F>("h_sumADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth6HEu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth6HEu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    // HE upgrade depth7
    h_sumADCAmplperLSdepth7HEu = fs_->make<TH1F>("h_sumADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth7HEu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth7HEu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    // for HE gain stability vs LS:
    h_2DsumADCAmplLSdepth4HEu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth4HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HEu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth4HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth5HEu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth5HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth5HEu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth5HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth6HEu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth6HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth6HEu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth6HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth7HEu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth7HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth7HEu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth7HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth3HFu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth3HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth3HFu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth3HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth4HFu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth4HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HFu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth4HFu", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumADCAmplLS6 = fs_->make<TH1F>("h_sumADCAmplLS6", " ", 100, 0., lsdep_estimator1_HFdepth1_);
    h_2DsumADCAmplLS6 = fs_->make<TH2F>("h_2DsumADCAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS6_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS6_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLS6 = fs_->make<TH2F>("h_2D0sumADCAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS6 = fs_->make<TH1F>("h_sumADCAmplperLS6", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS6 = fs_->make<TH1F>("h_sumCutADCAmplperLS6", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6 = fs_->make<TH1F>("h_sum0ADCAmplperLS6", " ", bac, 1., bac2);
    // HF upgrade depth3
    h_sumADCAmplperLS6u = fs_->make<TH1F>("h_sumADCAmplperLS6u", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS6u = fs_->make<TH1F>("h_sumCutADCAmplperLS6u", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6u = fs_->make<TH1F>("h_sum0ADCAmplperLS6u", " ", bac, 1., bac2);

    h_sumADCAmplLS7 = fs_->make<TH1F>("h_sumADCAmplLS7", " ", 100, 0., lsdep_estimator1_HFdepth2_);
    h_2DsumADCAmplLS7 = fs_->make<TH2F>("h_2DsumADCAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS7_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS7_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLS7 = fs_->make<TH2F>("h_2D0sumADCAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS7 = fs_->make<TH1F>("h_sumADCAmplperLS7", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS7 = fs_->make<TH1F>("h_sumCutADCAmplperLS7", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS7 = fs_->make<TH1F>("h_sum0ADCAmplperLS7", " ", bac, 1., bac2);
    // HF upgrade depth4
    h_sumADCAmplperLS7u = fs_->make<TH1F>("h_sumADCAmplperLS7u", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS7u = fs_->make<TH1F>("h_sumCutADCAmplperLS7u", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS7u = fs_->make<TH1F>("h_sum0ADCAmplperLS7u", " ", bac, 1., bac2);

    h_sumADCAmplLS8 = fs_->make<TH1F>("h_sumADCAmplLS8", " ", 100, 0., lsdep_estimator1_HOdepth4_);
    h_2DsumADCAmplLS8 = fs_->make<TH2F>("h_2DsumADCAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS8_LSselected =
        fs_->make<TH2F>("h_2DsumADCAmplLS8_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS8 = fs_->make<TH1F>("h_sumADCAmplperLS8", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS8 = fs_->make<TH1F>("h_sumCutADCAmplperLS8", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS8 = fs_->make<TH2F>("h_2D0sumADCAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS8 = fs_->make<TH1F>("h_sum0ADCAmplperLS8", " ", bac, 1., bac2);

    // HB upgrade depth3
    h_sumADCAmplperLSdepth3HBu = fs_->make<TH1F>("h_sumADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth3HBu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth3HBu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    // HB upgrade depth4
    h_sumADCAmplperLSdepth4HBu = fs_->make<TH1F>("h_sumADCAmplperLSdepth4HBu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth4HBu = fs_->make<TH1F>("h_sumCutADCAmplperLSdepth4HBu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth4HBu = fs_->make<TH1F>("h_sum0ADCAmplperLSdepth4HBu", " ", bac, 1., bac2);

    // for HB gain stability vs LS:
    h_2DsumADCAmplLSdepth3HBu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth3HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth3HBu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth3HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth4HBu = fs_->make<TH2F>("h_2DsumADCAmplLSdepth4HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HBu = fs_->make<TH2F>("h_2D0sumADCAmplLSdepth4HBu", " ", neta, -41., 41., nphi, 0., bphi);

    // error-A for HB( depth1 only)
    h_sumADCAmplperLS1_P1 = fs_->make<TH1F>("h_sumADCAmplperLS1_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_P1 = fs_->make<TH1F>("h_sum0ADCAmplperLS1_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_P2 = fs_->make<TH1F>("h_sumADCAmplperLS1_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_P2 = fs_->make<TH1F>("h_sum0ADCAmplperLS1_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_M1 = fs_->make<TH1F>("h_sumADCAmplperLS1_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_M1 = fs_->make<TH1F>("h_sum0ADCAmplperLS1_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_M2 = fs_->make<TH1F>("h_sumADCAmplperLS1_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_M2 = fs_->make<TH1F>("h_sum0ADCAmplperLS1_M2", " ", bac, 1., bac2);

    // error-A for HE( depth1 only)
    h_sumADCAmplperLS3_P1 = fs_->make<TH1F>("h_sumADCAmplperLS3_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_P1 = fs_->make<TH1F>("h_sum0ADCAmplperLS3_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_P2 = fs_->make<TH1F>("h_sumADCAmplperLS3_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_P2 = fs_->make<TH1F>("h_sum0ADCAmplperLS3_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_M1 = fs_->make<TH1F>("h_sumADCAmplperLS3_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_M1 = fs_->make<TH1F>("h_sum0ADCAmplperLS3_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_M2 = fs_->make<TH1F>("h_sumADCAmplperLS3_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_M2 = fs_->make<TH1F>("h_sum0ADCAmplperLS3_M2", " ", bac, 1., bac2);

    // error-A for HF( depth1 only)
    h_sumADCAmplperLS6_P1 = fs_->make<TH1F>("h_sumADCAmplperLS6_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_P1 = fs_->make<TH1F>("h_sum0ADCAmplperLS6_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_P2 = fs_->make<TH1F>("h_sumADCAmplperLS6_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_P2 = fs_->make<TH1F>("h_sum0ADCAmplperLS6_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_M1 = fs_->make<TH1F>("h_sumADCAmplperLS6_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_M1 = fs_->make<TH1F>("h_sum0ADCAmplperLS6_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_M2 = fs_->make<TH1F>("h_sumADCAmplperLS6_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_M2 = fs_->make<TH1F>("h_sum0ADCAmplperLS6_M2", " ", bac, 1., bac2);

    // error-A for HO( depth4 only)
    h_sumADCAmplperLS8_P1 = fs_->make<TH1F>("h_sumADCAmplperLS8_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_P1 = fs_->make<TH1F>("h_sum0ADCAmplperLS8_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_P2 = fs_->make<TH1F>("h_sumADCAmplperLS8_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_P2 = fs_->make<TH1F>("h_sum0ADCAmplperLS8_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_M1 = fs_->make<TH1F>("h_sumADCAmplperLS8_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_M1 = fs_->make<TH1F>("h_sum0ADCAmplperLS8_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_M2 = fs_->make<TH1F>("h_sumADCAmplperLS8_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_M2 = fs_->make<TH1F>("h_sum0ADCAmplperLS8_M2", " ", bac, 1., bac2);

    //--------------------------------------------------
    h_sumTSmeanALS1 = fs_->make<TH1F>("h_sumTSmeanALS1", " ", 100, 0., lsdep_estimator2_HBdepth1_);
    h_2DsumTSmeanALS1 = fs_->make<TH2F>("h_2DsumTSmeanALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS1 = fs_->make<TH1F>("h_sumTSmeanAperLS1", " ", bac, 1., bac2);
    h_sumTSmeanAperLS1_LSselected = fs_->make<TH1F>("h_sumTSmeanAperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS1 = fs_->make<TH1F>("h_sumCutTSmeanAperLS1", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS1 = fs_->make<TH2F>("h_2D0sumTSmeanALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS1 = fs_->make<TH1F>("h_sum0TSmeanAperLS1", " ", bac, 1., bac2);

    h_sumTSmeanALS2 = fs_->make<TH1F>("h_sumTSmeanALS2", " ", 100, 0., lsdep_estimator2_HBdepth2_);
    h_2DsumTSmeanALS2 = fs_->make<TH2F>("h_2DsumTSmeanALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS2 = fs_->make<TH1F>("h_sumTSmeanAperLS2", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS2 = fs_->make<TH1F>("h_sumCutTSmeanAperLS2", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS2 = fs_->make<TH2F>("h_2D0sumTSmeanALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS2 = fs_->make<TH1F>("h_sum0TSmeanAperLS2", " ", bac, 1., bac2);

    h_sumTSmeanALS3 = fs_->make<TH1F>("h_sumTSmeanALS3", " ", 100, 0., lsdep_estimator2_HEdepth1_);
    h_2DsumTSmeanALS3 = fs_->make<TH2F>("h_2DsumTSmeanALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS3 = fs_->make<TH1F>("h_sumTSmeanAperLS3", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS3 = fs_->make<TH1F>("h_sumCutTSmeanAperLS3", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS3 = fs_->make<TH2F>("h_2D0sumTSmeanALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS3 = fs_->make<TH1F>("h_sum0TSmeanAperLS3", " ", bac, 1., bac2);

    h_sumTSmeanALS4 = fs_->make<TH1F>("h_sumTSmeanALS4", " ", 100, 0., lsdep_estimator2_HEdepth2_);
    h_2DsumTSmeanALS4 = fs_->make<TH2F>("h_2DsumTSmeanALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS4 = fs_->make<TH1F>("h_sumTSmeanAperLS4", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS4 = fs_->make<TH1F>("h_sumCutTSmeanAperLS4", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS4 = fs_->make<TH2F>("h_2D0sumTSmeanALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS4 = fs_->make<TH1F>("h_sum0TSmeanAperLS4", " ", bac, 1., bac2);

    h_sumTSmeanALS5 = fs_->make<TH1F>("h_sumTSmeanALS5", " ", 100, 0., lsdep_estimator2_HEdepth3_);
    h_2DsumTSmeanALS5 = fs_->make<TH2F>("h_2DsumTSmeanALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS5 = fs_->make<TH1F>("h_sumTSmeanAperLS5", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS5 = fs_->make<TH1F>("h_sumCutTSmeanAperLS5", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS5 = fs_->make<TH2F>("h_2D0sumTSmeanALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS5 = fs_->make<TH1F>("h_sum0TSmeanAperLS5", " ", bac, 1., bac2);

    h_sumTSmeanALS6 = fs_->make<TH1F>("h_sumTSmeanALS6", " ", 100, 0., lsdep_estimator2_HFdepth1_);
    h_2DsumTSmeanALS6 = fs_->make<TH2F>("h_2DsumTSmeanALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS6 = fs_->make<TH1F>("h_sumTSmeanAperLS6", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS6 = fs_->make<TH1F>("h_sumCutTSmeanAperLS6", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS6 = fs_->make<TH2F>("h_2D0sumTSmeanALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS6 = fs_->make<TH1F>("h_sum0TSmeanAperLS6", " ", bac, 1., bac2);

    h_sumTSmeanALS7 = fs_->make<TH1F>("h_sumTSmeanALS7", " ", 100, 0., lsdep_estimator2_HFdepth2_);
    h_2DsumTSmeanALS7 = fs_->make<TH2F>("h_2DsumTSmeanALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS7 = fs_->make<TH1F>("h_sumTSmeanAperLS7", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS7 = fs_->make<TH1F>("h_sumCutTSmeanAperLS7", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS7 = fs_->make<TH2F>("h_2D0sumTSmeanALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS7 = fs_->make<TH1F>("h_sum0TSmeanAperLS7", " ", bac, 1., bac2);

    h_sumTSmeanALS8 = fs_->make<TH1F>("h_sumTSmeanALS8", " ", 100, 0., lsdep_estimator2_HOdepth4_);
    h_2DsumTSmeanALS8 = fs_->make<TH2F>("h_2DsumTSmeanALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS8 = fs_->make<TH1F>("h_sumTSmeanAperLS8", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS8 = fs_->make<TH1F>("h_sumCutTSmeanAperLS8", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS8 = fs_->make<TH2F>("h_2D0sumTSmeanALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS8 = fs_->make<TH1F>("h_sum0TSmeanAperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator3:
    h_sumTSmaxALS1 = fs_->make<TH1F>("h_sumTSmaxALS1", " ", 100, 0., lsdep_estimator3_HBdepth1_);
    h_2DsumTSmaxALS1 = fs_->make<TH2F>("h_2DsumTSmaxALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS1 = fs_->make<TH1F>("h_sumTSmaxAperLS1", " ", bac, 1., bac2);
    h_sumTSmaxAperLS1_LSselected = fs_->make<TH1F>("h_sumTSmaxAperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS1 = fs_->make<TH1F>("h_sumCutTSmaxAperLS1", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS1 = fs_->make<TH2F>("h_2D0sumTSmaxALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS1 = fs_->make<TH1F>("h_sum0TSmaxAperLS1", " ", bac, 1., bac2);

    h_sumTSmaxALS2 = fs_->make<TH1F>("h_sumTSmaxALS2", " ", 100, 0., lsdep_estimator3_HBdepth2_);
    h_2DsumTSmaxALS2 = fs_->make<TH2F>("h_2DsumTSmaxALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS2 = fs_->make<TH1F>("h_sumTSmaxAperLS2", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS2 = fs_->make<TH1F>("h_sumCutTSmaxAperLS2", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS2 = fs_->make<TH2F>("h_2D0sumTSmaxALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS2 = fs_->make<TH1F>("h_sum0TSmaxAperLS2", " ", bac, 1., bac2);

    h_sumTSmaxALS3 = fs_->make<TH1F>("h_sumTSmaxALS3", " ", 100, 0., lsdep_estimator3_HEdepth1_);
    h_2DsumTSmaxALS3 = fs_->make<TH2F>("h_2DsumTSmaxALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS3 = fs_->make<TH1F>("h_sumTSmaxAperLS3", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS3 = fs_->make<TH1F>("h_sumCutTSmaxAperLS3", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS3 = fs_->make<TH2F>("h_2D0sumTSmaxALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS3 = fs_->make<TH1F>("h_sum0TSmaxAperLS3", " ", bac, 1., bac2);

    h_sumTSmaxALS4 = fs_->make<TH1F>("h_sumTSmaxALS4", " ", 100, 0., lsdep_estimator3_HEdepth2_);
    h_2DsumTSmaxALS4 = fs_->make<TH2F>("h_2DsumTSmaxALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS4 = fs_->make<TH1F>("h_sumTSmaxAperLS4", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS4 = fs_->make<TH1F>("h_sumCutTSmaxAperLS4", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS4 = fs_->make<TH2F>("h_2D0sumTSmaxALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS4 = fs_->make<TH1F>("h_sum0TSmaxAperLS4", " ", bac, 1., bac2);

    h_sumTSmaxALS5 = fs_->make<TH1F>("h_sumTSmaxALS5", " ", 100, 0., lsdep_estimator3_HEdepth3_);
    h_2DsumTSmaxALS5 = fs_->make<TH2F>("h_2DsumTSmaxALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS5 = fs_->make<TH1F>("h_sumTSmaxAperLS5", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS5 = fs_->make<TH1F>("h_sumCutTSmaxAperLS5", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS5 = fs_->make<TH2F>("h_2D0sumTSmaxALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS5 = fs_->make<TH1F>("h_sum0TSmaxAperLS5", " ", bac, 1., bac2);

    h_sumTSmaxALS6 = fs_->make<TH1F>("h_sumTSmaxALS6", " ", 100, 0., lsdep_estimator3_HFdepth1_);
    h_2DsumTSmaxALS6 = fs_->make<TH2F>("h_2DsumTSmaxALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS6 = fs_->make<TH1F>("h_sumTSmaxAperLS6", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS6 = fs_->make<TH1F>("h_sumCutTSmaxAperLS6", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS6 = fs_->make<TH2F>("h_2D0sumTSmaxALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS6 = fs_->make<TH1F>("h_sum0TSmaxAperLS6", " ", bac, 1., bac2);

    h_sumTSmaxALS7 = fs_->make<TH1F>("h_sumTSmaxALS7", " ", 100, 0., lsdep_estimator3_HFdepth2_);
    h_2DsumTSmaxALS7 = fs_->make<TH2F>("h_2DsumTSmaxALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS7 = fs_->make<TH1F>("h_sumTSmaxAperLS7", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS7 = fs_->make<TH1F>("h_sumCutTSmaxAperLS7", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS7 = fs_->make<TH2F>("h_2D0sumTSmaxALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS7 = fs_->make<TH1F>("h_sum0TSmaxAperLS7", " ", bac, 1., bac2);

    h_sumTSmaxALS8 = fs_->make<TH1F>("h_sumTSmaxALS8", " ", 100, 0., lsdep_estimator3_HOdepth4_);
    h_2DsumTSmaxALS8 = fs_->make<TH2F>("h_2DsumTSmaxALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS8 = fs_->make<TH1F>("h_sumTSmaxAperLS8", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS8 = fs_->make<TH1F>("h_sumCutTSmaxAperLS8", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS8 = fs_->make<TH2F>("h_2D0sumTSmaxALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS8 = fs_->make<TH1F>("h_sum0TSmaxAperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator4:
    h_sumAmplitudeLS1 = fs_->make<TH1F>("h_sumAmplitudeLS1", " ", 100, 0.0, lsdep_estimator4_HBdepth1_);
    h_2DsumAmplitudeLS1 = fs_->make<TH2F>("h_2DsumAmplitudeLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS1 = fs_->make<TH1F>("h_sumAmplitudeperLS1", " ", bac, 1., bac2);
    h_sumAmplitudeperLS1_LSselected = fs_->make<TH1F>("h_sumAmplitudeperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS1 = fs_->make<TH1F>("h_sumCutAmplitudeperLS1", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS1 = fs_->make<TH2F>("h_2D0sumAmplitudeLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS1 = fs_->make<TH1F>("h_sum0AmplitudeperLS1", " ", bac, 1., bac2);

    h_sumAmplitudeLS2 = fs_->make<TH1F>("h_sumAmplitudeLS2", " ", 100, 0.0, lsdep_estimator4_HBdepth2_);
    h_2DsumAmplitudeLS2 = fs_->make<TH2F>("h_2DsumAmplitudeLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS2 = fs_->make<TH1F>("h_sumAmplitudeperLS2", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS2 = fs_->make<TH1F>("h_sumCutAmplitudeperLS2", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS2 = fs_->make<TH2F>("h_2D0sumAmplitudeLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS2 = fs_->make<TH1F>("h_sum0AmplitudeperLS2", " ", bac, 1., bac2);

    h_sumAmplitudeLS3 = fs_->make<TH1F>("h_sumAmplitudeLS3", " ", 100, 0.0, lsdep_estimator4_HEdepth1_);
    h_2DsumAmplitudeLS3 = fs_->make<TH2F>("h_2DsumAmplitudeLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS3 = fs_->make<TH1F>("h_sumAmplitudeperLS3", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS3 = fs_->make<TH1F>("h_sumCutAmplitudeperLS3", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS3 = fs_->make<TH2F>("h_2D0sumAmplitudeLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS3 = fs_->make<TH1F>("h_sum0AmplitudeperLS3", " ", bac, 1., bac2);

    h_sumAmplitudeLS4 = fs_->make<TH1F>("h_sumAmplitudeLS4", " ", 100, 0.0, lsdep_estimator4_HEdepth2_);
    h_2DsumAmplitudeLS4 = fs_->make<TH2F>("h_2DsumAmplitudeLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS4 = fs_->make<TH1F>("h_sumAmplitudeperLS4", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS4 = fs_->make<TH1F>("h_sumCutAmplitudeperLS4", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS4 = fs_->make<TH2F>("h_2D0sumAmplitudeLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS4 = fs_->make<TH1F>("h_sum0AmplitudeperLS4", " ", bac, 1., bac2);

    h_sumAmplitudeLS5 = fs_->make<TH1F>("h_sumAmplitudeLS5", " ", 100, 0.0, lsdep_estimator4_HEdepth3_);
    h_2DsumAmplitudeLS5 = fs_->make<TH2F>("h_2DsumAmplitudeLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS5 = fs_->make<TH1F>("h_sumAmplitudeperLS5", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS5 = fs_->make<TH1F>("h_sumCutAmplitudeperLS5", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS5 = fs_->make<TH2F>("h_2D0sumAmplitudeLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS5 = fs_->make<TH1F>("h_sum0AmplitudeperLS5", " ", bac, 1., bac2);

    h_sumAmplitudeLS6 = fs_->make<TH1F>("h_sumAmplitudeLS6", " ", 100, 0., lsdep_estimator4_HFdepth1_);
    h_2DsumAmplitudeLS6 = fs_->make<TH2F>("h_2DsumAmplitudeLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS6 = fs_->make<TH1F>("h_sumAmplitudeperLS6", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS6 = fs_->make<TH1F>("h_sumCutAmplitudeperLS6", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS6 = fs_->make<TH2F>("h_2D0sumAmplitudeLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS6 = fs_->make<TH1F>("h_sum0AmplitudeperLS6", " ", bac, 1., bac2);

    h_sumAmplitudeLS7 = fs_->make<TH1F>("h_sumAmplitudeLS7", " ", 100, 0., lsdep_estimator4_HFdepth2_);
    h_2DsumAmplitudeLS7 = fs_->make<TH2F>("h_2DsumAmplitudeLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS7 = fs_->make<TH1F>("h_sumAmplitudeperLS7", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS7 = fs_->make<TH1F>("h_sumCutAmplitudeperLS7", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS7 = fs_->make<TH2F>("h_2D0sumAmplitudeLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS7 = fs_->make<TH1F>("h_sum0AmplitudeperLS7", " ", bac, 1., bac2);

    h_sumAmplitudeLS8 = fs_->make<TH1F>("h_sumAmplitudeLS8", " ", 100, 0., lsdep_estimator4_HOdepth4_);
    h_2DsumAmplitudeLS8 = fs_->make<TH2F>("h_2DsumAmplitudeLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS8 = fs_->make<TH1F>("h_sumAmplitudeperLS8", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS8 = fs_->make<TH1F>("h_sumCutAmplitudeperLS8", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS8 = fs_->make<TH2F>("h_2D0sumAmplitudeLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS8 = fs_->make<TH1F>("h_sum0AmplitudeperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator5:
    h_sumAmplLS1 = fs_->make<TH1F>("h_sumAmplLS1", " ", 100, 0.0, lsdep_estimator5_HBdepth1_);
    h_2DsumAmplLS1 = fs_->make<TH2F>("h_2DsumAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS1 = fs_->make<TH1F>("h_sumAmplperLS1", " ", bac, 1., bac2);
    h_sumAmplperLS1_LSselected = fs_->make<TH1F>("h_sumAmplperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutAmplperLS1 = fs_->make<TH1F>("h_sumCutAmplperLS1", " ", bac, 1., bac2);
    h_2D0sumAmplLS1 = fs_->make<TH2F>("h_2D0sumAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS1 = fs_->make<TH1F>("h_sum0AmplperLS1", " ", bac, 1., bac2);

    h_sumAmplLS2 = fs_->make<TH1F>("h_sumAmplLS2", " ", 100, 0.0, lsdep_estimator5_HBdepth2_);
    h_2DsumAmplLS2 = fs_->make<TH2F>("h_2DsumAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS2 = fs_->make<TH1F>("h_sumAmplperLS2", " ", bac, 1., bac2);
    h_sumCutAmplperLS2 = fs_->make<TH1F>("h_sumCutAmplperLS2", " ", bac, 1., bac2);
    h_2D0sumAmplLS2 = fs_->make<TH2F>("h_2D0sumAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS2 = fs_->make<TH1F>("h_sum0AmplperLS2", " ", bac, 1., bac2);

    h_sumAmplLS3 = fs_->make<TH1F>("h_sumAmplLS3", " ", 100, 0.0, lsdep_estimator5_HEdepth1_);
    h_2DsumAmplLS3 = fs_->make<TH2F>("h_2DsumAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS3 = fs_->make<TH1F>("h_sumAmplperLS3", " ", bac, 1., bac2);
    h_sumCutAmplperLS3 = fs_->make<TH1F>("h_sumCutAmplperLS3", " ", bac, 1., bac2);
    h_2D0sumAmplLS3 = fs_->make<TH2F>("h_2D0sumAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS3 = fs_->make<TH1F>("h_sum0AmplperLS3", " ", bac, 1., bac2);

    h_sumAmplLS4 = fs_->make<TH1F>("h_sumAmplLS4", " ", 100, 0.0, lsdep_estimator5_HEdepth2_);
    h_2DsumAmplLS4 = fs_->make<TH2F>("h_2DsumAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS4 = fs_->make<TH1F>("h_sumAmplperLS4", " ", bac, 1., bac2);
    h_sumCutAmplperLS4 = fs_->make<TH1F>("h_sumCutAmplperLS4", " ", bac, 1., bac2);
    h_2D0sumAmplLS4 = fs_->make<TH2F>("h_2D0sumAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS4 = fs_->make<TH1F>("h_sum0AmplperLS4", " ", bac, 1., bac2);

    h_sumAmplLS5 = fs_->make<TH1F>("h_sumAmplLS5", " ", 100, 0.0, lsdep_estimator5_HEdepth3_);
    h_2DsumAmplLS5 = fs_->make<TH2F>("h_2DsumAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS5 = fs_->make<TH1F>("h_sumAmplperLS5", " ", bac, 1., bac2);
    h_sumCutAmplperLS5 = fs_->make<TH1F>("h_sumCutAmplperLS5", " ", bac, 1., bac2);
    h_2D0sumAmplLS5 = fs_->make<TH2F>("h_2D0sumAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS5 = fs_->make<TH1F>("h_sum0AmplperLS5", " ", bac, 1., bac2);

    h_sumAmplLS6 = fs_->make<TH1F>("h_sumAmplLS6", " ", 100, 0.0, lsdep_estimator5_HFdepth1_);
    h_2DsumAmplLS6 = fs_->make<TH2F>("h_2DsumAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS6 = fs_->make<TH1F>("h_sumAmplperLS6", " ", bac, 1., bac2);
    h_sumCutAmplperLS6 = fs_->make<TH1F>("h_sumCutAmplperLS6", " ", bac, 1., bac2);
    h_2D0sumAmplLS6 = fs_->make<TH2F>("h_2D0sumAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS6 = fs_->make<TH1F>("h_sum0AmplperLS6", " ", bac, 1., bac2);

    h_RatioOccupancy_HBP = fs_->make<TH1F>("h_RatioOccupancy_HBP", " ", bac, 1., bac2);
    h_RatioOccupancy_HBM = fs_->make<TH1F>("h_RatioOccupancy_HBM", " ", bac, 1., bac2);
    h_RatioOccupancy_HEP = fs_->make<TH1F>("h_RatioOccupancy_HEP", " ", bac, 1., bac2);
    h_RatioOccupancy_HEM = fs_->make<TH1F>("h_RatioOccupancy_HEM", " ", bac, 1., bac2);
    h_RatioOccupancy_HOP = fs_->make<TH1F>("h_RatioOccupancy_HOP", " ", bac, 1., bac2);
    h_RatioOccupancy_HOM = fs_->make<TH1F>("h_RatioOccupancy_HOM", " ", bac, 1., bac2);
    h_RatioOccupancy_HFP = fs_->make<TH1F>("h_RatioOccupancy_HFP", " ", bac, 1., bac2);
    h_RatioOccupancy_HFM = fs_->make<TH1F>("h_RatioOccupancy_HFM", " ", bac, 1., bac2);

    h_sumAmplLS7 = fs_->make<TH1F>("h_sumAmplLS7", " ", 100, 0.0, lsdep_estimator5_HFdepth2_);
    h_2DsumAmplLS7 = fs_->make<TH2F>("h_2DsumAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS7 = fs_->make<TH1F>("h_sumAmplperLS7", " ", bac, 1., bac2);
    h_sumCutAmplperLS7 = fs_->make<TH1F>("h_sumCutAmplperLS7", " ", bac, 1., bac2);
    h_2D0sumAmplLS7 = fs_->make<TH2F>("h_2D0sumAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS7 = fs_->make<TH1F>("h_sum0AmplperLS7", " ", bac, 1., bac2);

    h_sumAmplLS8 = fs_->make<TH1F>("h_sumAmplLS8", " ", 100, 0.0, lsdep_estimator5_HOdepth4_);
    h_2DsumAmplLS8 = fs_->make<TH2F>("h_2DsumAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS8 = fs_->make<TH1F>("h_sumAmplperLS8", " ", bac, 1., bac2);
    h_sumCutAmplperLS8 = fs_->make<TH1F>("h_sumCutAmplperLS8", " ", bac, 1., bac2);
    h_2D0sumAmplLS8 = fs_->make<TH2F>("h_2D0sumAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS8 = fs_->make<TH1F>("h_sum0AmplperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator6:
    h_sumErrorBLS1 = fs_->make<TH1F>("h_sumErrorBLS1", " ", 10, 0., 10.);
    h_sumErrorBperLS1 = fs_->make<TH1F>("h_sumErrorBperLS1", " ", bac, 1., bac2);
    h_sum0ErrorBperLS1 = fs_->make<TH1F>("h_sum0ErrorBperLS1", " ", bac, 1., bac2);
    h_2D0sumErrorBLS1 = fs_->make<TH2F>("h_2D0sumErrorBLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS1 = fs_->make<TH2F>("h_2DsumErrorBLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS2 = fs_->make<TH1F>("h_sumErrorBLS2", " ", 10, 0., 10.);
    h_sumErrorBperLS2 = fs_->make<TH1F>("h_sumErrorBperLS2", " ", bac, 1., bac2);
    h_sum0ErrorBperLS2 = fs_->make<TH1F>("h_sum0ErrorBperLS2", " ", bac, 1., bac2);
    h_2D0sumErrorBLS2 = fs_->make<TH2F>("h_2D0sumErrorBLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS2 = fs_->make<TH2F>("h_2DsumErrorBLS2", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS3 = fs_->make<TH1F>("h_sumErrorBLS3", " ", 10, 0., 10.);
    h_sumErrorBperLS3 = fs_->make<TH1F>("h_sumErrorBperLS3", " ", bac, 1., bac2);
    h_sum0ErrorBperLS3 = fs_->make<TH1F>("h_sum0ErrorBperLS3", " ", bac, 1., bac2);
    h_2D0sumErrorBLS3 = fs_->make<TH2F>("h_2D0sumErrorBLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS3 = fs_->make<TH2F>("h_2DsumErrorBLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS4 = fs_->make<TH1F>("h_sumErrorBLS4", " ", 10, 0., 10.);
    h_sumErrorBperLS4 = fs_->make<TH1F>("h_sumErrorBperLS4", " ", bac, 1., bac2);
    h_sum0ErrorBperLS4 = fs_->make<TH1F>("h_sum0ErrorBperLS4", " ", bac, 1., bac2);
    h_2D0sumErrorBLS4 = fs_->make<TH2F>("h_2D0sumErrorBLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS4 = fs_->make<TH2F>("h_2DsumErrorBLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS5 = fs_->make<TH1F>("h_sumErrorBLS5", " ", 10, 0., 10.);
    h_sumErrorBperLS5 = fs_->make<TH1F>("h_sumErrorBperLS5", " ", bac, 1., bac2);
    h_sum0ErrorBperLS5 = fs_->make<TH1F>("h_sum0ErrorBperLS5", " ", bac, 1., bac2);
    h_2D0sumErrorBLS5 = fs_->make<TH2F>("h_2D0sumErrorBLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS5 = fs_->make<TH2F>("h_2DsumErrorBLS5", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS6 = fs_->make<TH1F>("h_sumErrorBLS6", " ", 10, 0., 10.);
    h_sumErrorBperLS6 = fs_->make<TH1F>("h_sumErrorBperLS6", " ", bac, 1., bac2);
    h_sum0ErrorBperLS6 = fs_->make<TH1F>("h_sum0ErrorBperLS6", " ", bac, 1., bac2);
    h_2D0sumErrorBLS6 = fs_->make<TH2F>("h_2D0sumErrorBLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS6 = fs_->make<TH2F>("h_2DsumErrorBLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS7 = fs_->make<TH1F>("h_sumErrorBLS7", " ", 10, 0., 10.);
    h_sumErrorBperLS7 = fs_->make<TH1F>("h_sumErrorBperLS7", " ", bac, 1., bac2);
    h_sum0ErrorBperLS7 = fs_->make<TH1F>("h_sum0ErrorBperLS7", " ", bac, 1., bac2);
    h_2D0sumErrorBLS7 = fs_->make<TH2F>("h_2D0sumErrorBLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS7 = fs_->make<TH2F>("h_2DsumErrorBLS7", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS8 = fs_->make<TH1F>("h_sumErrorBLS8", " ", 10, 0., 10.);
    h_sumErrorBperLS8 = fs_->make<TH1F>("h_sumErrorBperLS8", " ", bac, 1., bac2);
    h_sum0ErrorBperLS8 = fs_->make<TH1F>("h_sum0ErrorBperLS8", " ", bac, 1., bac2);
    h_2D0sumErrorBLS8 = fs_->make<TH2F>("h_2D0sumErrorBLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS8 = fs_->make<TH2F>("h_2DsumErrorBLS8", " ", neta, -41., 41., nphi, 0., bphi);

    //--------------------------------------------------
    // for averSIGNALOCCUPANCY :
    h_averSIGNALoccupancy_HB = fs_->make<TH1F>("h_averSIGNALoccupancy_HB", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HE = fs_->make<TH1F>("h_averSIGNALoccupancy_HE", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HF = fs_->make<TH1F>("h_averSIGNALoccupancy_HF", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HO = fs_->make<TH1F>("h_averSIGNALoccupancy_HO", " ", bac, 1., bac2);

    // for averSIGNALsumamplitude :
    h_averSIGNALsumamplitude_HB = fs_->make<TH1F>("h_averSIGNALsumamplitude_HB", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HE = fs_->make<TH1F>("h_averSIGNALsumamplitude_HE", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HF = fs_->make<TH1F>("h_averSIGNALsumamplitude_HF", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HO = fs_->make<TH1F>("h_averSIGNALsumamplitude_HO", " ", bac, 1., bac2);

    // for averNOSIGNALOCCUPANCY :
    h_averNOSIGNALoccupancy_HB = fs_->make<TH1F>("h_averNOSIGNALoccupancy_HB", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HE = fs_->make<TH1F>("h_averNOSIGNALoccupancy_HE", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HF = fs_->make<TH1F>("h_averNOSIGNALoccupancy_HF", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HO = fs_->make<TH1F>("h_averNOSIGNALoccupancy_HO", " ", bac, 1., bac2);

    // for averNOSIGNALsumamplitude :
    h_averNOSIGNALsumamplitude_HB = fs_->make<TH1F>("h_averNOSIGNALsumamplitude_HB", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HE = fs_->make<TH1F>("h_averNOSIGNALsumamplitude_HE", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HF = fs_->make<TH1F>("h_averNOSIGNALsumamplitude_HF", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HO = fs_->make<TH1F>("h_averNOSIGNALsumamplitude_HO", " ", bac, 1., bac2);

    // for channel SUM over depthes Amplitudes for each sub-detector
    h_sumamplitudechannel_HB = fs_->make<TH1F>("h_sumamplitudechannel_HB", " ", 100, 0., 2000.);
    h_sumamplitudechannel_HE = fs_->make<TH1F>("h_sumamplitudechannel_HE", " ", 100, 0., 3000.);
    h_sumamplitudechannel_HF = fs_->make<TH1F>("h_sumamplitudechannel_HF", " ", 100, 0., 7000.);
    h_sumamplitudechannel_HO = fs_->make<TH1F>("h_sumamplitudechannel_HO", " ", 100, 0., 10000.);

    // for event Amplitudes for each sub-detector
    h_eventamplitude_HB = fs_->make<TH1F>("h_eventamplitude_HB", " ", 100, 0., 80000.);
    h_eventamplitude_HE = fs_->make<TH1F>("h_eventamplitude_HE", " ", 100, 0., 100000.);
    h_eventamplitude_HF = fs_->make<TH1F>("h_eventamplitude_HF", " ", 100, 0., 150000.);
    h_eventamplitude_HO = fs_->make<TH1F>("h_eventamplitude_HO", " ", 100, 0., 250000.);

    // for event Occupancy for each sub-detector
    h_eventoccupancy_HB = fs_->make<TH1F>("h_eventoccupancy_HB", " ", 100, 0., 3000.);
    h_eventoccupancy_HE = fs_->make<TH1F>("h_eventoccupancy_HE", " ", 100, 0., 2000.);
    h_eventoccupancy_HF = fs_->make<TH1F>("h_eventoccupancy_HF", " ", 100, 0., 1000.);
    h_eventoccupancy_HO = fs_->make<TH1F>("h_eventoccupancy_HO", " ", 100, 0., 2500.);

    // for maxxSUMAmplitude
    h_maxxSUMAmpl_HB = fs_->make<TH1F>("h_maxxSUMAmpl_HB", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HE = fs_->make<TH1F>("h_maxxSUMAmpl_HE", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HF = fs_->make<TH1F>("h_maxxSUMAmpl_HF", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HO = fs_->make<TH1F>("h_maxxSUMAmpl_HO", " ", bac, 1., bac2);

    // for maxxOCCUP
    h_maxxOCCUP_HB = fs_->make<TH1F>("h_maxxOCCUP_HB", " ", bac, 1., bac2);
    h_maxxOCCUP_HE = fs_->make<TH1F>("h_maxxOCCUP_HE", " ", bac, 1., bac2);
    h_maxxOCCUP_HF = fs_->make<TH1F>("h_maxxOCCUP_HF", " ", bac, 1., bac2);
    h_maxxOCCUP_HO = fs_->make<TH1F>("h_maxxOCCUP_HO", " ", bac, 1., bac2);
    //--------------------------------------------------
    // pedestals
    h_pedestal0_HB = fs_->make<TH1F>("h_pedestal0_HB", " ", 100, 0., 10.);
    h_pedestal1_HB = fs_->make<TH1F>("h_pedestal1_HB", " ", 100, 0., 10.);
    h_pedestal2_HB = fs_->make<TH1F>("h_pedestal2_HB", " ", 100, 0., 10.);
    h_pedestal3_HB = fs_->make<TH1F>("h_pedestal3_HB", " ", 100, 0., 10.);
    h_pedestalaver4_HB = fs_->make<TH1F>("h_pedestalaver4_HB", " ", 100, 0., 10.);
    h_pedestalaver9_HB = fs_->make<TH1F>("h_pedestalaver9_HB", " ", 100, 0., 10.);
    h_pedestalw0_HB = fs_->make<TH1F>("h_pedestalw0_HB", " ", 100, 0., 2.5);
    h_pedestalw1_HB = fs_->make<TH1F>("h_pedestalw1_HB", " ", 100, 0., 2.5);
    h_pedestalw2_HB = fs_->make<TH1F>("h_pedestalw2_HB", " ", 100, 0., 2.5);
    h_pedestalw3_HB = fs_->make<TH1F>("h_pedestalw3_HB", " ", 100, 0., 2.5);
    h_pedestalwaver4_HB = fs_->make<TH1F>("h_pedestalwaver4_HB", " ", 100, 0., 2.5);
    h_pedestalwaver9_HB = fs_->make<TH1F>("h_pedestalwaver9_HB", " ", 100, 0., 2.5);

    h_pedestal0_HE = fs_->make<TH1F>("h_pedestal0_HE", " ", 100, 0., 10.);
    h_pedestal1_HE = fs_->make<TH1F>("h_pedestal1_HE", " ", 100, 0., 10.);
    h_pedestal2_HE = fs_->make<TH1F>("h_pedestal2_HE", " ", 100, 0., 10.);
    h_pedestal3_HE = fs_->make<TH1F>("h_pedestal3_HE", " ", 100, 0., 10.);
    h_pedestalaver4_HE = fs_->make<TH1F>("h_pedestalaver4_HE", " ", 100, 0., 10.);
    h_pedestalaver9_HE = fs_->make<TH1F>("h_pedestalaver9_HE", " ", 100, 0., 10.);
    h_pedestalw0_HE = fs_->make<TH1F>("h_pedestalw0_HE", " ", 100, 0., 2.5);
    h_pedestalw1_HE = fs_->make<TH1F>("h_pedestalw1_HE", " ", 100, 0., 2.5);
    h_pedestalw2_HE = fs_->make<TH1F>("h_pedestalw2_HE", " ", 100, 0., 2.5);
    h_pedestalw3_HE = fs_->make<TH1F>("h_pedestalw3_HE", " ", 100, 0., 2.5);
    h_pedestalwaver4_HE = fs_->make<TH1F>("h_pedestalwaver4_HE", " ", 100, 0., 2.5);
    h_pedestalwaver9_HE = fs_->make<TH1F>("h_pedestalwaver9_HE", " ", 100, 0., 2.5);

    h_pedestal0_HF = fs_->make<TH1F>("h_pedestal0_HF", " ", 100, 0., 20.);
    h_pedestal1_HF = fs_->make<TH1F>("h_pedestal1_HF", " ", 100, 0., 20.);
    h_pedestal2_HF = fs_->make<TH1F>("h_pedestal2_HF", " ", 100, 0., 20.);
    h_pedestal3_HF = fs_->make<TH1F>("h_pedestal3_HF", " ", 100, 0., 20.);
    h_pedestalaver4_HF = fs_->make<TH1F>("h_pedestalaver4_HF", " ", 100, 0., 20.);
    h_pedestalaver9_HF = fs_->make<TH1F>("h_pedestalaver9_HF", " ", 100, 0., 20.);
    h_pedestalw0_HF = fs_->make<TH1F>("h_pedestalw0_HF", " ", 100, 0., 2.5);
    h_pedestalw1_HF = fs_->make<TH1F>("h_pedestalw1_HF", " ", 100, 0., 2.5);
    h_pedestalw2_HF = fs_->make<TH1F>("h_pedestalw2_HF", " ", 100, 0., 2.5);
    h_pedestalw3_HF = fs_->make<TH1F>("h_pedestalw3_HF", " ", 100, 0., 2.5);
    h_pedestalwaver4_HF = fs_->make<TH1F>("h_pedestalwaver4_HF", " ", 100, 0., 2.5);
    h_pedestalwaver9_HF = fs_->make<TH1F>("h_pedestalwaver9_HF", " ", 100, 0., 2.5);

    h_pedestal0_HO = fs_->make<TH1F>("h_pedestal0_HO", " ", 100, 0., 20.);
    h_pedestal1_HO = fs_->make<TH1F>("h_pedestal1_HO", " ", 100, 0., 20.);
    h_pedestal2_HO = fs_->make<TH1F>("h_pedestal2_HO", " ", 100, 0., 20.);
    h_pedestal3_HO = fs_->make<TH1F>("h_pedestal3_HO", " ", 100, 0., 20.);
    h_pedestalaver4_HO = fs_->make<TH1F>("h_pedestalaver4_HO", " ", 100, 0., 20.);
    h_pedestalaver9_HO = fs_->make<TH1F>("h_pedestalaver9_HO", " ", 100, 0., 20.);
    h_pedestalw0_HO = fs_->make<TH1F>("h_pedestalw0_HO", " ", 100, 0., 2.5);
    h_pedestalw1_HO = fs_->make<TH1F>("h_pedestalw1_HO", " ", 100, 0., 2.5);
    h_pedestalw2_HO = fs_->make<TH1F>("h_pedestalw2_HO", " ", 100, 0., 2.5);
    h_pedestalw3_HO = fs_->make<TH1F>("h_pedestalw3_HO", " ", 100, 0., 2.5);
    h_pedestalwaver4_HO = fs_->make<TH1F>("h_pedestalwaver4_HO", " ", 100, 0., 2.5);
    h_pedestalwaver9_HO = fs_->make<TH1F>("h_pedestalwaver9_HO", " ", 100, 0., 2.5);
    //--------------------------------------------------
    h_mapDepth1pedestalw_HB = fs_->make<TH2F>("h_mapDepth1pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HB = fs_->make<TH2F>("h_mapDepth2pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HB = fs_->make<TH2F>("h_mapDepth3pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HB = fs_->make<TH2F>("h_mapDepth4pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestalw_HE = fs_->make<TH2F>("h_mapDepth1pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HE = fs_->make<TH2F>("h_mapDepth2pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HE = fs_->make<TH2F>("h_mapDepth3pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HE = fs_->make<TH2F>("h_mapDepth4pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5pedestalw_HE = fs_->make<TH2F>("h_mapDepth5pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6pedestalw_HE = fs_->make<TH2F>("h_mapDepth6pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7pedestalw_HE = fs_->make<TH2F>("h_mapDepth7pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestalw_HF = fs_->make<TH2F>("h_mapDepth1pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HF = fs_->make<TH2F>("h_mapDepth2pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HF = fs_->make<TH2F>("h_mapDepth3pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HF = fs_->make<TH2F>("h_mapDepth4pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HO = fs_->make<TH2F>("h_mapDepth4pedestalw_HO", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1pedestal_HB = fs_->make<TH2F>("h_mapDepth1pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HB = fs_->make<TH2F>("h_mapDepth2pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HB = fs_->make<TH2F>("h_mapDepth3pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HB = fs_->make<TH2F>("h_mapDepth4pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestal_HE = fs_->make<TH2F>("h_mapDepth1pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HE = fs_->make<TH2F>("h_mapDepth2pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HE = fs_->make<TH2F>("h_mapDepth3pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HE = fs_->make<TH2F>("h_mapDepth4pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5pedestal_HE = fs_->make<TH2F>("h_mapDepth5pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6pedestal_HE = fs_->make<TH2F>("h_mapDepth6pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7pedestal_HE = fs_->make<TH2F>("h_mapDepth7pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestal_HF = fs_->make<TH2F>("h_mapDepth1pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HF = fs_->make<TH2F>("h_mapDepth2pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HF = fs_->make<TH2F>("h_mapDepth3pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HF = fs_->make<TH2F>("h_mapDepth4pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HO = fs_->make<TH2F>("h_mapDepth4pedestal_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_pedestal00_HB = fs_->make<TH1F>("h_pedestal00_HB", " ", 100, 0., 10.);
    h_gain_HB = fs_->make<TH1F>("h_gain_HB", " ", 100, 0., 1.);
    h_respcorr_HB = fs_->make<TH1F>("h_respcorr_HB", " ", 100, 0., 2.5);
    h_timecorr_HB = fs_->make<TH1F>("h_timecorr_HB", " ", 100, 0., 30.);
    h_lutcorr_HB = fs_->make<TH1F>("h_lutcorr_HB", " ", 100, 0., 10.);
    h_difpedestal0_HB = fs_->make<TH1F>("h_difpedestal0_HB", " ", 100, -3., 3.);
    h_difpedestal1_HB = fs_->make<TH1F>("h_difpedestal1_HB", " ", 100, -3., 3.);
    h_difpedestal2_HB = fs_->make<TH1F>("h_difpedestal2_HB", " ", 100, -3., 3.);
    h_difpedestal3_HB = fs_->make<TH1F>("h_difpedestal3_HB", " ", 100, -3., 3.);

    h_pedestal00_HE = fs_->make<TH1F>("h_pedestal00_HE", " ", 100, 0., 10.);
    h_gain_HE = fs_->make<TH1F>("h_gain_HE", " ", 100, 0., 1.);
    h_respcorr_HE = fs_->make<TH1F>("h_respcorr_HE", " ", 100, 0., 2.5);
    h_timecorr_HE = fs_->make<TH1F>("h_timecorr_HE", " ", 100, 0., 30.);
    h_lutcorr_HE = fs_->make<TH1F>("h_lutcorr_HE", " ", 100, 0., 10.);

    h_pedestal00_HF = fs_->make<TH1F>("h_pedestal00_HF", " ", 100, 0., 10.);
    h_gain_HF = fs_->make<TH1F>("h_gain_HF", " ", 100, 0., 1.);
    h_respcorr_HF = fs_->make<TH1F>("h_respcorr_HF", " ", 100, 0., 2.5);
    h_timecorr_HF = fs_->make<TH1F>("h_timecorr_HF", " ", 100, 0., 30.);
    h_lutcorr_HF = fs_->make<TH1F>("h_lutcorr_HF", " ", 100, 0., 10.);

    h_pedestal00_HO = fs_->make<TH1F>("h_pedestal00_HO", " ", 100, 0., 10.);
    h_gain_HO = fs_->make<TH1F>("h_gain_HO", " ", 100, 0., 1.);
    h_respcorr_HO = fs_->make<TH1F>("h_respcorr_HO", " ", 100, 0., 2.5);
    h_timecorr_HO = fs_->make<TH1F>("h_timecorr_HO", " ", 100, 0., 30.);
    h_lutcorr_HO = fs_->make<TH1F>("h_lutcorr_HO", " ", 100, 0., 10.);
    //--------------------------------------------------
    h2_TSnVsAyear2023_HB = fs_->make<TH2F>("h2_TSnVsAyear2023_HB", " ", 100, 200., 30200., 100, 0., 175.);
    h2_TSnVsAyear2023_HE = fs_->make<TH2F>("h2_TSnVsAyear2023_HE", " ", 100, 200., 75200., 100, 0., 175.);
    h2_TSnVsAyear2023_HF = fs_->make<TH2F>("h2_TSnVsAyear2023_HF", " ", 100, 0., 2000., 100, 0., 50.);
    h2_TSnVsAyear2023_HO = fs_->make<TH2F>("h2_TSnVsAyear2023_HO", " ", 100, 0., 1000., 100, 0., 225.);
    //-----------------------------
    h1_TSnVsAyear2023_HB = fs_->make<TH1F>("h1_TSnVsAyear2023_HB", " ", 100, 200., 15200);
    h1_TSnVsAyear2023_HE = fs_->make<TH1F>("h1_TSnVsAyear2023_HE", " ", 100, 200., 25200);
    h1_TSnVsAyear2023_HF = fs_->make<TH1F>("h1_TSnVsAyear2023_HF", " ", 100, 0., 1500);
    h1_TSnVsAyear2023_HO = fs_->make<TH1F>("h1_TSnVsAyear2023_HO", " ", 100, 0., 750);
    h1_TSnVsAyear20230_HB = fs_->make<TH1F>("h1_TSnVsAyear20230_HB", " ", 100, 200., 15200);
    h1_TSnVsAyear20230_HE = fs_->make<TH1F>("h1_TSnVsAyear20230_HE", " ", 100, 200., 25200);
    h1_TSnVsAyear20230_HF = fs_->make<TH1F>("h1_TSnVsAyear20230_HF", " ", 100, 0., 1500);
    h1_TSnVsAyear20230_HO = fs_->make<TH1F>("h1_TSnVsAyear20230_HO", " ", 100, 0., 750);
    //--------------------------------------------------
    float est6 = 2500.;
    int ist6 = 30;
    int ist2 = 60;
    h2_pedvsampl_HB = fs_->make<TH2F>("h2_pedvsampl_HB", " ", ist2, 0., 7.0, ist2, 0., est6);
    h2_pedwvsampl_HB = fs_->make<TH2F>("h2_pedwvsampl_HB", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HB = fs_->make<TH1F>("h_pedvsampl_HB", " ", ist6, 0., 7.0);
    h_pedwvsampl_HB = fs_->make<TH1F>("h_pedwvsampl_HB", " ", ist6, 0., 2.5);
    h_pedvsampl0_HB = fs_->make<TH1F>("h_pedvsampl0_HB", " ", ist6, 0., 7.);
    h_pedwvsampl0_HB = fs_->make<TH1F>("h_pedwvsampl0_HB", " ", ist6, 0., 2.5);
    h2_amplvsped_HB = fs_->make<TH2F>("h2_amplvsped_HB", " ", ist2, 0., est6, ist2, 0., 7.0);
    h2_amplvspedw_HB = fs_->make<TH2F>("h2_amplvspedw_HB", " ", ist2, 0., est6, ist2, 0., 2.5);
    h_amplvsped_HB = fs_->make<TH1F>("h_amplvsped_HB", " ", ist6, 0., est6);
    h_amplvspedw_HB = fs_->make<TH1F>("h_amplvspedw_HB", " ", ist6, 0., est6);
    h_amplvsped0_HB = fs_->make<TH1F>("h_amplvsped0_HB", " ", ist6, 0., est6);

    h2_pedvsampl_HE = fs_->make<TH2F>("h2_pedvsampl_HE", " ", ist2, 0., 7.0, ist2, 0., est6);
    h2_pedwvsampl_HE = fs_->make<TH2F>("h2_pedwvsampl_HE", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HE = fs_->make<TH1F>("h_pedvsampl_HE", " ", ist6, 0., 7.0);
    h_pedwvsampl_HE = fs_->make<TH1F>("h_pedwvsampl_HE", " ", ist6, 0., 2.5);
    h_pedvsampl0_HE = fs_->make<TH1F>("h_pedvsampl0_HE", " ", ist6, 0., 7.);
    h_pedwvsampl0_HE = fs_->make<TH1F>("h_pedwvsampl0_HE", " ", ist6, 0., 2.5);

    h2_pedvsampl_HF = fs_->make<TH2F>("h2_pedvsampl_HF", " ", ist2, 0., 20.0, ist2, 0., est6);
    h2_pedwvsampl_HF = fs_->make<TH2F>("h2_pedwvsampl_HF", " ", ist2, 0., 2.0, ist2, 0., est6);
    h_pedvsampl_HF = fs_->make<TH1F>("h_pedvsampl_HF", " ", ist6, 0., 20.0);
    h_pedwvsampl_HF = fs_->make<TH1F>("h_pedwvsampl_HF", " ", ist6, 0., 2.0);
    h_pedvsampl0_HF = fs_->make<TH1F>("h_pedvsampl0_HF", " ", ist6, 0., 20.);
    h_pedwvsampl0_HF = fs_->make<TH1F>("h_pedwvsampl0_HF", " ", ist6, 0., 2.0);

    h2_pedvsampl_HO = fs_->make<TH2F>("h2_pedvsampl_HO", " ", ist2, 0., 20.0, ist2, 0., est6);
    h2_pedwvsampl_HO = fs_->make<TH2F>("h2_pedwvsampl_HO", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HO = fs_->make<TH1F>("h_pedvsampl_HO", " ", ist6, 0., 20.0);
    h_pedwvsampl_HO = fs_->make<TH1F>("h_pedwvsampl_HO", " ", ist6, 0., 2.5);
    h_pedvsampl0_HO = fs_->make<TH1F>("h_pedvsampl0_HO", " ", ist6, 0., 20.);
    h_pedwvsampl0_HO = fs_->make<TH1F>("h_pedwvsampl0_HO", " ", ist6, 0., 2.5);
    //--------------------------------------------------
    h_mapDepth1Ped0_HB = fs_->make<TH2F>("h_mapDepth1Ped0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HB = fs_->make<TH2F>("h_mapDepth1Ped1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HB = fs_->make<TH2F>("h_mapDepth1Ped2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HB = fs_->make<TH2F>("h_mapDepth1Ped3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HB = fs_->make<TH2F>("h_mapDepth1Pedw0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HB = fs_->make<TH2F>("h_mapDepth1Pedw1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HB = fs_->make<TH2F>("h_mapDepth1Pedw2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HB = fs_->make<TH2F>("h_mapDepth1Pedw3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HB = fs_->make<TH2F>("h_mapDepth2Ped0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HB = fs_->make<TH2F>("h_mapDepth2Ped1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HB = fs_->make<TH2F>("h_mapDepth2Ped2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HB = fs_->make<TH2F>("h_mapDepth2Ped3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HB = fs_->make<TH2F>("h_mapDepth2Pedw0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HB = fs_->make<TH2F>("h_mapDepth2Pedw1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HB = fs_->make<TH2F>("h_mapDepth2Pedw2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HB = fs_->make<TH2F>("h_mapDepth2Pedw3_HB", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1Ped0_HE = fs_->make<TH2F>("h_mapDepth1Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HE = fs_->make<TH2F>("h_mapDepth1Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HE = fs_->make<TH2F>("h_mapDepth1Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HE = fs_->make<TH2F>("h_mapDepth1Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HE = fs_->make<TH2F>("h_mapDepth1Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HE = fs_->make<TH2F>("h_mapDepth1Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HE = fs_->make<TH2F>("h_mapDepth1Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HE = fs_->make<TH2F>("h_mapDepth1Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HE = fs_->make<TH2F>("h_mapDepth2Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HE = fs_->make<TH2F>("h_mapDepth2Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HE = fs_->make<TH2F>("h_mapDepth2Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HE = fs_->make<TH2F>("h_mapDepth2Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HE = fs_->make<TH2F>("h_mapDepth2Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HE = fs_->make<TH2F>("h_mapDepth2Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HE = fs_->make<TH2F>("h_mapDepth2Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HE = fs_->make<TH2F>("h_mapDepth2Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped0_HE = fs_->make<TH2F>("h_mapDepth3Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped1_HE = fs_->make<TH2F>("h_mapDepth3Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped2_HE = fs_->make<TH2F>("h_mapDepth3Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped3_HE = fs_->make<TH2F>("h_mapDepth3Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw0_HE = fs_->make<TH2F>("h_mapDepth3Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw1_HE = fs_->make<TH2F>("h_mapDepth3Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw2_HE = fs_->make<TH2F>("h_mapDepth3Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw3_HE = fs_->make<TH2F>("h_mapDepth3Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1Ped0_HF = fs_->make<TH2F>("h_mapDepth1Ped0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HF = fs_->make<TH2F>("h_mapDepth1Ped1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HF = fs_->make<TH2F>("h_mapDepth1Ped2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HF = fs_->make<TH2F>("h_mapDepth1Ped3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HF = fs_->make<TH2F>("h_mapDepth1Pedw0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HF = fs_->make<TH2F>("h_mapDepth1Pedw1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HF = fs_->make<TH2F>("h_mapDepth1Pedw2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HF = fs_->make<TH2F>("h_mapDepth1Pedw3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HF = fs_->make<TH2F>("h_mapDepth2Ped0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HF = fs_->make<TH2F>("h_mapDepth2Ped1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HF = fs_->make<TH2F>("h_mapDepth2Ped2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HF = fs_->make<TH2F>("h_mapDepth2Ped3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HF = fs_->make<TH2F>("h_mapDepth2Pedw0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HF = fs_->make<TH2F>("h_mapDepth2Pedw1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HF = fs_->make<TH2F>("h_mapDepth2Pedw2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HF = fs_->make<TH2F>("h_mapDepth2Pedw3_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth4Ped0_HO = fs_->make<TH2F>("h_mapDepth4Ped0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped1_HO = fs_->make<TH2F>("h_mapDepth4Ped1_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped2_HO = fs_->make<TH2F>("h_mapDepth4Ped2_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped3_HO = fs_->make<TH2F>("h_mapDepth4Ped3_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw0_HO = fs_->make<TH2F>("h_mapDepth4Pedw0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw1_HO = fs_->make<TH2F>("h_mapDepth4Pedw1_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw2_HO = fs_->make<TH2F>("h_mapDepth4Pedw2_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw3_HO = fs_->make<TH2F>("h_mapDepth4Pedw3_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_mapDepth1ADCAmpl12_HB = fs_->make<TH2F>("h_mapDepth1ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HB = fs_->make<TH2F>("h_mapDepth2ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HB = fs_->make<TH2F>("h_mapDepth3ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HB = fs_->make<TH2F>("h_mapDepth4ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth1ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth2ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth3ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth4ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth5ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth6ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth7ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl12SiPM_HE = fs_->make<TH2F>("h_mapDepth1ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12SiPM_HE = fs_->make<TH2F>("h_mapDepth2ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12SiPM_HE = fs_->make<TH2F>("h_mapDepth3ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1ADCAmpl12_HF = fs_->make<TH2F>("h_mapDepth1ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HF = fs_->make<TH2F>("h_mapDepth2ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HF = fs_->make<TH2F>("h_mapDepth3ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HF = fs_->make<TH2F>("h_mapDepth4ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth4ADCAmpl12_HO = fs_->make<TH2F>("h_mapDepth4ADCAmpl12_HO", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1linADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth1linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2linADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth2linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3linADCAmpl12_HE = fs_->make<TH2F>("h_mapDepth3linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_mapGetRMSOverNormalizedSignal_HB =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HB =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HE =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HE =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HF =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HF =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HO =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HO =
        fs_->make<TH2F>("h_mapGetRMSOverNormalizedSignal0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_shape_Ahigh_HB0 = fs_->make<TH1F>("h_shape_Ahigh_HB0", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB0 = fs_->make<TH1F>("h_shape0_Ahigh_HB0", " ", 10, 0., 10.);
    h_shape_Alow_HB0 = fs_->make<TH1F>("h_shape_Alow_HB0", " ", 10, 0., 10.);
    h_shape0_Alow_HB0 = fs_->make<TH1F>("h_shape0_Alow_HB0", " ", 10, 0., 10.);
    h_shape_Ahigh_HB1 = fs_->make<TH1F>("h_shape_Ahigh_HB1", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB1 = fs_->make<TH1F>("h_shape0_Ahigh_HB1", " ", 10, 0., 10.);
    h_shape_Alow_HB1 = fs_->make<TH1F>("h_shape_Alow_HB1", " ", 10, 0., 10.);
    h_shape0_Alow_HB1 = fs_->make<TH1F>("h_shape0_Alow_HB1", " ", 10, 0., 10.);
    h_shape_Ahigh_HB2 = fs_->make<TH1F>("h_shape_Ahigh_HB2", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB2 = fs_->make<TH1F>("h_shape0_Ahigh_HB2", " ", 10, 0., 10.);
    h_shape_Alow_HB2 = fs_->make<TH1F>("h_shape_Alow_HB2", " ", 10, 0., 10.);
    h_shape0_Alow_HB2 = fs_->make<TH1F>("h_shape0_Alow_HB2", " ", 10, 0., 10.);
    h_shape_Ahigh_HB3 = fs_->make<TH1F>("h_shape_Ahigh_HB3", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB3 = fs_->make<TH1F>("h_shape0_Ahigh_HB3", " ", 10, 0., 10.);
    h_shape_Alow_HB3 = fs_->make<TH1F>("h_shape_Alow_HB3", " ", 10, 0., 10.);
    h_shape0_Alow_HB3 = fs_->make<TH1F>("h_shape0_Alow_HB3", " ", 10, 0., 10.);
    //--------------------------------------------------
    h_shape_bad_channels_HB = fs_->make<TH1F>("h_shape_bad_channels_HB", " ", 10, 0., 10.);
    h_shape0_bad_channels_HB = fs_->make<TH1F>("h_shape0_bad_channels_HB", " ", 10, 0., 10.);
    h_shape_good_channels_HB = fs_->make<TH1F>("h_shape_good_channels_HB", " ", 10, 0., 10.);
    h_shape0_good_channels_HB = fs_->make<TH1F>("h_shape0_good_channels_HB", " ", 10, 0., 10.);
    h_shape_bad_channels_HE = fs_->make<TH1F>("h_shape_bad_channels_HE", " ", 10, 0., 10.);
    h_shape0_bad_channels_HE = fs_->make<TH1F>("h_shape0_bad_channels_HE", " ", 10, 0., 10.);
    h_shape_good_channels_HE = fs_->make<TH1F>("h_shape_good_channels_HE", " ", 10, 0., 10.);
    h_shape0_good_channels_HE = fs_->make<TH1F>("h_shape0_good_channels_HE", " ", 10, 0., 10.);
    h_shape_bad_channels_HF = fs_->make<TH1F>("h_shape_bad_channels_HF", " ", 10, 0., 10.);
    h_shape0_bad_channels_HF = fs_->make<TH1F>("h_shape0_bad_channels_HF", " ", 10, 0., 10.);
    h_shape_good_channels_HF = fs_->make<TH1F>("h_shape_good_channels_HF", " ", 10, 0., 10.);
    h_shape0_good_channels_HF = fs_->make<TH1F>("h_shape0_good_channels_HF", " ", 10, 0., 10.);
    h_shape_bad_channels_HO = fs_->make<TH1F>("h_shape_bad_channels_HO", " ", 10, 0., 10.);
    h_shape0_bad_channels_HO = fs_->make<TH1F>("h_shape0_bad_channels_HO", " ", 10, 0., 10.);
    h_shape_good_channels_HO = fs_->make<TH1F>("h_shape_good_channels_HO", " ", 10, 0., 10.);
    h_shape0_good_channels_HO = fs_->make<TH1F>("h_shape0_good_channels_HO", " ", 10, 0., 10.);
    //--------------------------------------------------
    //    if(flagcpuoptimization_== 0 ) {

    int spl = 1000;
    float spls = 5000;
    h_sumamplitude_depth1_HB = fs_->make<TH1F>("h_sumamplitude_depth1_HB", " ", spl, 0., spls);
    h_sumamplitude_depth2_HB = fs_->make<TH1F>("h_sumamplitude_depth2_HB", " ", spl, 0., spls);
    h_sumamplitude_depth1_HE = fs_->make<TH1F>("h_sumamplitude_depth1_HE", " ", spl, 0., spls);
    h_sumamplitude_depth2_HE = fs_->make<TH1F>("h_sumamplitude_depth2_HE", " ", spl, 0., spls);
    h_sumamplitude_depth3_HE = fs_->make<TH1F>("h_sumamplitude_depth3_HE", " ", spl, 0., spls);
    h_sumamplitude_depth1_HF = fs_->make<TH1F>("h_sumamplitude_depth1_HF", " ", spl, 0., spls);
    h_sumamplitude_depth2_HF = fs_->make<TH1F>("h_sumamplitude_depth2_HF", " ", spl, 0., spls);
    h_sumamplitude_depth4_HO = fs_->make<TH1F>("h_sumamplitude_depth4_HO", " ", spl, 0., spls);
    int spl0 = 1000;
    float spls0 = 10000;
    h_sumamplitude_depth1_HB0 = fs_->make<TH1F>("h_sumamplitude_depth1_HB0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HB0 = fs_->make<TH1F>("h_sumamplitude_depth2_HB0", " ", spl0, 0., spls0);
    h_sumamplitude_depth1_HE0 = fs_->make<TH1F>("h_sumamplitude_depth1_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HE0 = fs_->make<TH1F>("h_sumamplitude_depth2_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth3_HE0 = fs_->make<TH1F>("h_sumamplitude_depth3_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth1_HF0 = fs_->make<TH1F>("h_sumamplitude_depth1_HF0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HF0 = fs_->make<TH1F>("h_sumamplitude_depth2_HF0", " ", spl0, 0., spls0);
    h_sumamplitude_depth4_HO0 = fs_->make<TH1F>("h_sumamplitude_depth4_HO0", " ", spl0, 0., spls0);
    int spl1 = 1000;
    float spls1 = 100000;
    h_sumamplitude_depth1_HB1 = fs_->make<TH1F>("h_sumamplitude_depth1_HB1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HB1 = fs_->make<TH1F>("h_sumamplitude_depth2_HB1", " ", spl1, 0., spls1);
    h_sumamplitude_depth1_HE1 = fs_->make<TH1F>("h_sumamplitude_depth1_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HE1 = fs_->make<TH1F>("h_sumamplitude_depth2_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth3_HE1 = fs_->make<TH1F>("h_sumamplitude_depth3_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth1_HF1 = fs_->make<TH1F>("h_sumamplitude_depth1_HF1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HF1 = fs_->make<TH1F>("h_sumamplitude_depth2_HF1", " ", spl1, 0., spls1);
    h_sumamplitude_depth4_HO1 = fs_->make<TH1F>("h_sumamplitude_depth4_HO1", " ", spl1, 0., spls1);

    h_Amplitude_forCapIdErrors_HB1 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HB1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HB2 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HB2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE1 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HE1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE2 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HE2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE3 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HE3", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HF1 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HF1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HF2 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HF2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HO4 = fs_->make<TH1F>("h_Amplitude_forCapIdErrors_HO4", " ", 100, 0., 30000.);

    h_Amplitude_notCapIdErrors_HB1 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HB1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HB2 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HB2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE1 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HE1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE2 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HE2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE3 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HE3", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HF1 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HF1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HF2 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HF2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HO4 = fs_->make<TH1F>("h_Amplitude_notCapIdErrors_HO4", " ", 100, 0., 30000.);

    h_2DAtaildepth1_HB = fs_->make<TH2F>("h_2DAtaildepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0Ataildepth1_HB = fs_->make<TH2F>("h_2D0Ataildepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DAtaildepth2_HB = fs_->make<TH2F>("h_2DAtaildepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0Ataildepth2_HB = fs_->make<TH2F>("h_2D0Ataildepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);

    ////////////////////////////////////////////////////////////////////////////////////
  }  //if(recordHistoes_
  if (verbosity > 0)
    std::cout << "========================   booking DONE   +++++++++++++++++++++++++++" << std::endl;
  ///////////////////////////////////////////////////////            ntuples:
  if (recordNtuples_) {
    myTree = fs_->make<TTree>("Hcal", "Hcal Tree");
    myTree->Branch("Nevent", &Nevent, "Nevent/I");
    myTree->Branch("Run", &Run, "Run/I");

  }  //if(recordNtuples_
  if (verbosity > 0)
    std::cout << "========================   beignJob  finish   +++++++++++++++++++++++++++" << std::endl;
  //////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrors(HBHEDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  //    double tool[100];
  if (verbosity < 0)
    std::cout << "**************   in loop over Digis   counter =     " << nnnnnnhbhe << std::endl;
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;
  int TSsize = 10;
  TSsize = digiItr->size();

  ///////////////////////////////////////
  for (int ii = 0; ii < TSsize; ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;

    if (ii == 0)
      firstcapid = capid;

    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }

  }  // for

  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;

  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Energy
  // Energy:

  double ampl = 0.;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }

  ///////////////////////////////////////Digis
  // Digis:
  // HB
  if (sub == 1) {
    h_errorGeneral_HB->Fill(double(errorGeneral), 1.);
    h_error1_HB->Fill(double(error1), 1.);
    h_error2_HB->Fill(double(error2), 1.);
    h_error3_HB->Fill(double(error3), 1.);
    h_error4_HB->Fill(double(error4), 1.);
    h_error5_HB->Fill(double(error5), 1.);
    h_error6_HB->Fill(double(error6), 1.);
    h_error7_HB->Fill(double(error7), 1.);
    h_repetedcapid_HB->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HB->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HB->Fill(double(ieta), double(iphi));
      h_fiber0_HB->Fill(double(ERRORfiber), 1.);
      h_fiber1_HB->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HB->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HB->Fill(ampl, 1.);
    }
  }
  // HE
  if (sub == 2) {
    h_errorGeneral_HE->Fill(double(errorGeneral), 1.);
    h_error1_HE->Fill(double(error1), 1.);
    h_error2_HE->Fill(double(error2), 1.);
    h_error3_HE->Fill(double(error3), 1.);
    h_error4_HE->Fill(double(error4), 1.);
    h_error5_HE->Fill(double(error5), 1.);
    h_error6_HE->Fill(double(error6), 1.);
    h_error7_HE->Fill(double(error7), 1.);
    h_repetedcapid_HE->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HE->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HE->Fill(double(ieta), double(iphi));
      h_fiber0_HE->Fill(double(ERRORfiber), 1.);
      h_fiber1_HE->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HE->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HE->Fill(ampl, 1.);
    }
  }
  //    ha2->Fill(double(ieta), double(iphi));
}  //fillDigiErrors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHBHEQIE11
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsQIE11(QIE11DataFrame qie11df) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  DetId detid = qie11df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int error1 = 0;
  // !!!!!!
  bool anycapid = true;
  //    bool anyer      =  false;
  //    bool anydv      =  true;
  // for help:
  int firstcapid = 0;
  int lastcapid = 0, capid = 0;
  int repetedcapid = 0;
  // loop over the samples in the digi
  nTS = qie11df.samples();
  ///////////////////////////////////////
  for (int ii = 0; ii < nTS; ii++) {
    capid = qie11df[ii].capid();  // capId (0-3, sequential)
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
  }  // for
  ///////////////////////////////////////
  if (!anycapid)
    error1 = 1;
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  double ampl = 0.;
  for (int ii = 0; ii < nTS; ii++) {
    int adc = qie11df[ii].adc();

    double ampldefault = adc2fC_QIE11_shunt6[adc];
    if (flaguseshunt_ == 1)
      ampldefault = adc2fC_QIE11_shunt1[adc];

    ampl += ampldefault;  //
  }
  ///////////////////////////////////////Digis
  // Digis:HBHE
  if (sub == 1) {
    h_error1_HB->Fill(double(error1), 1.);
    h_repetedcapid_HB->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HB->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HB->Fill(double(ieta), double(iphi));
      h_errorGeneral_HB->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HB->Fill(ampl, 1.);
    }
  }
  if (sub == 2) {
    h_error1_HE->Fill(double(error1), 1.);
    h_repetedcapid_HE->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HE->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 5)
        h_mapDepth5Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 6)
        h_mapDepth6Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 7)
        h_mapDepth7Error_HE->Fill(double(ieta), double(iphi));
      h_errorGeneral_HE->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HE->Fill(ampl, 1.);
    }
  }
}  //fillDigiErrorsQIE11
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHF
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHF(HFDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  if (mdepth > 2)
    std::cout << " HF DIGI ??????????????   ERROR       mdepth =  " << mdepth << std::endl;
  // !!!!!!
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;

  int TSsize = 10;
  TSsize = digiItr->size();
  ///////////////////////////////////////
  for (int ii = 0; ii < TSsize; ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
  }  // for
  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;
  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Ampl
  double ampl = 0.;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }
  ///////////////////////////////////////Digis
  // Digis: HF
  if (sub == 4) {
    h_errorGeneral_HF->Fill(double(errorGeneral), 1.);
    h_error1_HF->Fill(double(error1), 1.);
    h_error2_HF->Fill(double(error2), 1.);
    h_error3_HF->Fill(double(error3), 1.);
    h_error4_HF->Fill(double(error4), 1.);
    h_error5_HF->Fill(double(error5), 1.);
    h_error6_HF->Fill(double(error6), 1.);
    h_error7_HF->Fill(double(error7), 1.);
    h_repetedcapid_HF->Fill(double(repetedcapid), 1.);
    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HF->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HF->Fill(double(ieta), double(iphi));
      h_fiber0_HF->Fill(double(ERRORfiber), 1.);
      h_fiber1_HF->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HF->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HF->Fill(ampl, 1.);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHFQIE10
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHFQIE10(QIE10DataFrame qie10df) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  DetId detid = qie10df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int error1 = 0;
  // !!!!!!
  bool anycapid = true;
  //    bool anyer      =  false;
  //    bool anydv      =  true;
  // for help:
  int firstcapid = 0;
  int lastcapid = 0, capid = 0;
  int repetedcapid = 0;
  // loop over the samples in the digi
  nTS = qie10df.samples();
  ///////////////////////////////////////
  for (int ii = 0; ii < nTS; ii++) {
    capid = qie10df[ii].capid();  // capId (0-3, sequential)
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
  }  // for
  ///////////////////////////////////////
  if (!anycapid)
    error1 = 1;
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  double ampl = 0.;
  for (int ii = 0; ii < nTS; ii++) {
    int adc = qie10df[ii].adc();
    double ampldefault = adc2fC_QIE10[adc];
    ampl += ampldefault;  //
  }
  ///////////////////////////////////////Digis
  // Digis:HF
  if (sub == 4) {
    h_error1_HF->Fill(double(error1), 1.);
    h_repetedcapid_HF->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HF->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HF->Fill(double(ieta), double(iphi));
      h_errorGeneral_HF->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HF->Fill(ampl, 1.);
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHO(HODigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE, 3-HO, 4-HF
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;
  for (int ii = 0; ii < (*digiItr).size(); ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;

    if (ii == 0)
      firstcapid = capid;

    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }

  }  // for

  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;

  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Energy
  // Energy:
  double ampl = 0.;
  for (int ii = 0; ii < (*digiItr).size(); ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }
  ///////////////////////////////////////Digis
  // Digis:
  // HO
  if (sub == 3) {
    h_errorGeneral_HO->Fill(double(errorGeneral), 1.);
    h_error1_HO->Fill(double(error1), 1.);
    h_error2_HO->Fill(double(error2), 1.);
    h_error3_HO->Fill(double(error3), 1.);
    h_error4_HO->Fill(double(error4), 1.);
    h_error5_HO->Fill(double(error5), 1.);
    h_error6_HO->Fill(double(error6), 1.);
    h_error7_HO->Fill(double(error7), 1.);
    h_repetedcapid_HO->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HO->Fill(ampl, 1.);
      if (mdepth == 4)
        h_mapDepth4Error_HO->Fill(double(ieta), double(iphi));
      // to be divided by h_mapDepth4_HO

      if (mdepth != 4)
        std::cout << " mdepth HO = " << mdepth << std::endl;
      h_fiber0_HO->Fill(double(ERRORfiber), 1.);
      h_fiber1_HO->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HO->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HO->Fill(ampl, 1.);
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitude(HBHEDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  double toolwithPedSubtr[100];        // TS
  double lintoolwithoutPedSubtr[100];  // TS
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta0 = cell.ieta();     //-41 +41 !=0
  int ieta = ieta0;
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  //    const HcalGainWidth* gainWidth = conditions->getGainWidth(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  //    HcalCalibrations calib = conditions->getHcalCalibrations(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double difpedestal0 = 0.;
  double difpedestal1 = 0.;
  double difpedestal2 = 0.;
  double difpedestal3 = 0.;

  double amplitude = 0.;
  double absamplitude = 0.;
  double amplitude345 = 0.;
  double ampl = 0.;
  double linamplitudewithoutPedSubtr = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  int TSsize = 10;  //HEHB for Run2
  TSsize = digiItr->size();
  if ((*digiItr).size() != 10)
    errorBtype = 1.;
  //     ii = 0 to 9
  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<digiItr->size(); ii++) {
    double ampldefaultwithPedSubtr = 0.;
    double linampldefaultwithoutPedSubtr = 0.;
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive ADCcounts
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts linearized
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    ampldefaultwithPedSubtr = ampldefault0;
    linampldefaultwithoutPedSubtr = ampldefault2;

    int capid = ((*digiItr)[ii]).capid();
    //      double pedestal = calib.pedestal(capid);
    double pedestalINI = pedestal00->getValue(capid);
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    ampldefaultwithPedSubtr -= pedestal;  // pedestal subtraction
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    //      ampldefault*= calib.respcorrgain(capid) ; // fC --> GeV
    tool[ii] = ampldefault;
    toolwithPedSubtr[ii] = ampldefaultwithPedSubtr;
    lintoolwithoutPedSubtr[ii] = linampldefaultwithoutPedSubtr;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal0 = pedestal - pedestalINI;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal1 = pedestal - pedestalINI;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal2 = pedestal - pedestalINI;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal3 = pedestal - pedestalINI;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    ///   for choice TSs, raddam only:
    //     TS = 1 to 10:  1  2  3  4  5  6  7  8  9  10
    //     ii = 0 to  9:  0  1  2  3  4  5  6  7  8   9
    //     var.1             ----------------------
    //     var.2                ----------------
    //     var.3                   ----------
    //     var.4                   -------
    //
    // TS = 2-9      for raddam only  var.1
    // TS = 3-8      for raddam only  var.2
    // TS = 4-7      for raddam only  var.3
    // TS = 4-6      for raddam only  var.4
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //

    if (ii == 3 || ii == 4 || ii == 5)
      amplitude345 += ampldefault;
    if (flagcpuoptimization_ == 0) {
      //////
    }  //flagcpuoptimization
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }  //for 1
  /////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiAmplitude
  // sub=1||2 HBHE
  if (sub == 1 || sub == 2) {
    amplitudechannel0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;                 // 0-neta ; 0-71 HBHE
    amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;           // 0-neta ; 0-71 HBHE
    amplitudechannel2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(amplitude, 2);  // 0-neta ; 0-71 HBHE
  }
  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];

  ///----------------------------------------------------------------------------------------------------  for raddam:
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    linamplitudewithoutPedSubtr = lintoolwithoutPedSubtr[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal - 1];

  double ratio = 0.;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1
  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    //    std::cout << " ii = " << ii  << " capid = " << capid  << " ((lastcapid+1)%4) = " << ((lastcapid+1)%4)  << std::endl;
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2
  if (!anycapid || anyer || !anydv)
    error = 1;

  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9

  // CapIdErrors end  /////////////////////////////////////////////////////////

  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_forCapIdErrors_HE3->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_notCapIdErrors_HE3->Fill(amplitude, 1.);
  }

  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<10; ii++) {
    double ampldefault = tool[ii];
    ///
    if (sub == 1) {
      if (amplitude > 120) {
        h_shape_Ahigh_HB0->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB0->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB0->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB0->Fill(float(ii), 1.);
      }  //HB0
      ///
      if (pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        h_shape_Ahigh_HB1->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB1->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB1->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB1->Fill(float(ii), 1.);
      }  //HB1
      if (error == 0) {
        h_shape_Ahigh_HB2->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB2->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB2->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB2->Fill(float(ii), 1.);
      }  //HB2
      ///
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        h_shape_Ahigh_HB3->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB3->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB3->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB3->Fill(float(ii), 1.);
      }  //HB3

    }  // sub   HB

  }  //for 3 over TSs

  if (sub == 1) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_ || rmsamplitude < rmsHBMin_ ||
        rmsamplitude > rmsHBMax_ || pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ ||
        pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestalw0 < pedestalwHBMax_ ||
        pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ || pedestalw3 < pedestalwHBMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HB->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HB->Fill(float(ii), 1.);
      }
    }
  }  // sub   HB
  if (sub == 2) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_ || rmsamplitude < rmsHEMin_ ||
        rmsamplitude > rmsHEMax_ || pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ ||
        pedestal2 < pedestalHEMax_ || pedestal3 < pedestalHEMax_ || pedestalw0 < pedestalwHEMax_ ||
        pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ || pedestalw3 < pedestalwHEMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HE->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HE->Fill(float(ii), 1.);
      }
    }
  }  // sub   HE

  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HB
  if (sub == 1) {
    if (studyPedestalCorrelations_) {
      //   //   //   //   //   //   //   //   //  HB       PedestalCorrelations :
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HB->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl_HB->Fill(mypedestal, amplitude);
      h_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HB->Fill(mypedestal, 1.);
      h_pedwvsampl0_HB->Fill(mypedestalw, 1.);

      h2_amplvsped_HB->Fill(amplitude, mypedestal);
      h2_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped_HB->Fill(amplitude, mypedestal);
      h_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped0_HB->Fill(amplitude, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HB->Fill(pedestal0, 1.);
      h_pedestal1_HB->Fill(pedestal1, 1.);
      h_pedestal2_HB->Fill(pedestal2, 1.);
      h_pedestal3_HB->Fill(pedestal3, 1.);
      h_pedestalaver4_HB->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HB->Fill(pedestalaver9, 1.);
      h_pedestalw0_HB->Fill(pedestalw0, 1.);
      h_pedestalw1_HB->Fill(pedestalw1, 1.);
      h_pedestalw2_HB->Fill(pedestalw2, 1.);
      h_pedestalw3_HB->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HB->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HB->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HB->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HB->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HB->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HB->Fill(respcorr->getValue(), 1.);
      h_timecorr_HB->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HB->Fill(lutcorr->getValue(), 1.);
      h_difpedestal0_HB->Fill(difpedestal0, 1.);
      h_difpedestal1_HB->Fill(difpedestal1, 1.);
      h_difpedestal2_HB->Fill(difpedestal2, 1.);
      h_difpedestal3_HB->Fill(difpedestal3, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345Zoom1_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345_HB->Fill(amplitude345, 1.);
      if (error == 0) {
        h_ADCAmpl_HBCapIdNoError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdNoError->Fill(amplitude345, 1.);
      }
      if (error == 1) {
        h_ADCAmpl_HBCapIdError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdError->Fill(amplitude345, 1.);
      }
      h_ADCAmplZoom_HB->Fill(amplitude, 1.);
      h_ADCAmplZoom1_HB->Fill(amplitude, 1.);
      h_ADCAmpl_HB->Fill(amplitude, 1.);

      h_AmplitudeHBrest->Fill(amplitude, 1.);
      h_AmplitudeHBrest1->Fill(amplitude, 1.);
      h_AmplitudeHBrest6->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >400.) averSIGNALoccupancy_HB += 1.;
      if (amplitude < 35.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      h_bcnvsamplitude_HB->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HB->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HB->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HB->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HB->Fill(aveamplitude1, 1.);
      //      h2_TSnVsAyear2023_HB->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HB->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HB->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HB->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHBMin_ || aveamplitude1 > TSmeanHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HB->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHBMin_ || ts_with_max_signal > TSpeakHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HB->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHBMin_ || rmsamplitude > rmsHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HB->Fill(ratio, 1.);
      if (ratio < ratioHBMin_ || ratio > ratioHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        // //
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HB All
    if (mdepth == 1)
      h_mapDepth1_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HB->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 1 )

  // HE
  if (sub == 2) {
    //   //   //   //   //   //   //   //   //  HE       PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HE->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl_HE->Fill(mypedestal, amplitude);
      h_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HE->Fill(mypedestal, 1.);
      h_pedwvsampl0_HE->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HE->Fill(pedestal0, 1.);
      h_pedestal1_HE->Fill(pedestal1, 1.);
      h_pedestal2_HE->Fill(pedestal2, 1.);
      h_pedestal3_HE->Fill(pedestal3, 1.);
      h_pedestalaver4_HE->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HE->Fill(pedestalaver9, 1.);
      h_pedestalw0_HE->Fill(pedestalw0, 1.);
      h_pedestalw1_HE->Fill(pedestalw1, 1.);
      h_pedestalw2_HE->Fill(pedestalw2, 1.);
      h_pedestalw3_HE->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HE->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HE->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 3) {
        h_mapDepth3Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth3Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth3Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth3Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth3Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth3Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth3Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth3Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHEMax_ || pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ ||
          pedestalw3 < pedestalwHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ || pedestal2 < pedestalHEMax_ ||
          pedestal3 < pedestalHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HE->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HE->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HE->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HE->Fill(respcorr->getValue(), 1.);
      h_timecorr_HE->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HE->Fill(lutcorr->getValue(), 1.);
    }  //

    //     h_mapDepth1ADCAmpl12SiPM_HE
    //   //   //   //   //   //   //   //   //  HE       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HE->Fill(ampl, 1.);
      h_ADCAmpl345Zoom1_HE->Fill(amplitude345, 1.);
      h_ADCAmpl345_HE->Fill(amplitude345, 1.);
      h_ADCAmpl_HE->Fill(amplitude, 1.);

      h_ADCAmplrest_HE->Fill(amplitude, 1.);
      h_ADCAmplrest1_HE->Fill(amplitude, 1.);
      h_ADCAmplrest6_HE->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HE->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude > 700.) averSIGNALoccupancy_HE += 1.;
      if (amplitude < 500.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);

      if (mdepth == 1) {
        h_mapDepth1ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth1linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }
      if (mdepth == 2) {
        h_mapDepth2ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth2linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }
      if (mdepth == 3) {
        h_mapDepth3ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth3linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }

      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HE->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HE->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HE->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HE->Fill(float(orbitNum), 1.);

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;

    }  //if(studyADCAmplHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HE->Fill(aveamplitude1, 1.);
      //      h2_TSnVsAyear2023_HE->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HE->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HE->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HE->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHEMin_ || aveamplitude1 > TSmeanHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HE->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHEMin_ || ts_with_max_signal > TSpeakHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HE->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHEMin_ || rmsamplitude > rmsHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HE       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HE->Fill(ratio, 1.);
      if (ratio < ratioHEMin_ || ratio > ratioHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       DiffAmplitude:
    if (studyDiffAmplHist_) {
      // gain stability:
      if (mdepth == 1)
        h_mapDepth1AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);

    }  // if(studyDiffAmplHist_)

    // RADDAM filling:
    if (flagLaserRaddam_ > 0) {
      double amplitudewithPedSubtr = 0.;

      //for cut on A_channel:
      if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
        amplitudewithPedSubtr = toolwithPedSubtr[ts_with_max_signal];
      if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal + 2];
      if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal + 1];
      if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal - 1];

      h_AamplitudewithPedSubtr_RADDAM_HE->Fill(amplitudewithPedSubtr);
      h_AamplitudewithPedSubtr_RADDAM_HEzoom0->Fill(amplitudewithPedSubtr);
      h_AamplitudewithPedSubtr_RADDAM_HEzoom1->Fill(amplitudewithPedSubtr);

      if (amplitudewithPedSubtr > 50.) {
        if (flagLaserRaddam_ > 1) {
          mapRADDAM_HE[mdepth - 1][ieta + 41][iphi] += amplitudewithPedSubtr;
          ++mapRADDAM0_HE[mdepth - 1][ieta + 41][iphi];
        }

        if (mdepth == 1) {
          h_mapDepth1RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth1RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth1RADDAM_HE->Fill(amplitudewithPedSubtr);
        }
        if (mdepth == 2) {
          h_mapDepth2RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth2RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth2RADDAM_HE->Fill(amplitudewithPedSubtr);
        }
        if (mdepth == 3) {
          h_mapDepth3RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth3RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth3RADDAM_HE->Fill(amplitudewithPedSubtr);
        }

        // (d1 & eta 17-29)                       L1
        int LLLLLL111111 = 0;
        if ((mdepth == 1 && fabs(ieta0) > 16 && fabs(ieta0) < 30))
          LLLLLL111111 = 1;
        // (d2 & eta 17-26) && (d3 & eta 27-28)   L2
        int LLLLLL222222 = 0;
        if ((mdepth == 2 && fabs(ieta0) > 16 && fabs(ieta0) < 27) ||
            (mdepth == 3 && fabs(ieta0) > 26 && fabs(ieta0) < 29))
          LLLLLL222222 = 1;
        //
        if (LLLLLL111111 == 1) {
          //forStudy	    h_mapLayer1RADDAM_HE->Fill(fabs(double(ieta0)), amplitudewithPedSubtr); h_mapLayer1RADDAM0_HE->Fill(fabs(double(ieta0)), 1.); h_A_Layer1RADDAM_HE->Fill(amplitudewithPedSubtr);
          h_sigLayer1RADDAM_HE->Fill(double(ieta0), amplitudewithPedSubtr);
          h_sigLayer1RADDAM0_HE->Fill(double(ieta0), 1.);
        }
        if (LLLLLL222222 == 1) {
          //forStudy    h_mapLayer2RADDAM_HE->Fill(fabs(double(ieta0)), amplitudewithPedSubtr); h_mapLayer2RADDAM0_HE->Fill(fabs(double(ieta0)), 1.); h_A_Layer2RADDAM_HE->Fill(amplitudewithPedSubtr);
          h_sigLayer2RADDAM_HE->Fill(double(ieta0), amplitudewithPedSubtr);
          h_sigLayer2RADDAM0_HE->Fill(double(ieta0), 1.);
        }

        //
        if (mdepth == 3 && fabs(ieta0) == 16) {
          h_mapDepth3RADDAM16_HE->Fill(amplitudewithPedSubtr);
          // forStudy     h_mapDepth3RADDAM160_HE->Fill(1.);
        }
        //
      }  //amplitude > 60.
    }    // END RADDAM

    ///////////////////////////////    for HE All
    if (mdepth == 1)
      h_mapDepth1_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HE->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 2 )
     //
}  // fillDigiAmplitude
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeQIE11(QIE11DataFrame qie11df) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  DetId detid = qie11df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE QIE11DigiCollection
  nTS = qie11df.samples();
  /////////////////////////////////////////////////////////////////
  if (mdepth == 0 || sub > 4)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 3)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 7)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 8)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 9)
    return;

  // for some CMSSW versions and GT this line uncommented, can help to run job
  //if(mdepth ==4  && sub==1  && (ieta == -16 || ieta == 15)   ) return;// HB depth4 eta=-16, 15 since I did:if(ieta > 0) ieta -= 1;
  /////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////
  const HcalGain* gain = conditions->getGain(hcaldetid);
  //    const HcalGainWidth* gainWidth = conditions->getGainWidth(hcaldetid);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(hcaldetid);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(hcaldetid);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(hcaldetid);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(hcaldetid);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(hcaldetid);
  const HcalPedestal* pedestal00 = conditions->getPedestal(hcaldetid);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(qie11df, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double difpedestal0 = 0.;
  double difpedestal1 = 0.;
  double difpedestal2 = 0.;
  double difpedestal3 = 0.;

  double amplitude = 0.;
  double amplitude0 = 0.;
  double absamplitude = 0.;
  double tocampl = 0.;

  double amplitude345 = 0.;
  double ampl = 0.;
  double ampl3ts = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  int TSsize = 8;  // sub= 1 HB
  if (sub == 2)
    TSsize = 8;  // sub = 2 HE
  if (nTS != TSsize)
    errorBtype = 1.;
  TSsize = nTS;  //nTS = qie11df.samples();
  ///////   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // double ADC_ped = 0.;
  int flagTS2 = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double tocdefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;

    ampldefault0 = adc2fC_QIE11_shunt6[qie11df[ii].adc()];  // massive !!!!!!    (use for local runs as default shunt6)
    if (flaguseshunt_ == 1)
      ampldefault0 = adc2fC_QIE11_shunt1[qie11df[ii].adc()];  // massive !!!!!!
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];  //adcfC
    ampldefault2 = qie11df[ii].adc();   //ADCcounts

    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }  // !!!!!!
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    tocdefault = ampldefault;

    int capid = (qie11df[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    double pedestalINI = pedestal00->getValue(capid);
    tocdefault -= pedestal;  // pedestal subtraction
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    tool[ii] = ampldefault;
    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal0 = pedestal - pedestalINI;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal1 = pedestal - pedestalINI;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal2 = pedestal - pedestalINI;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal3 = pedestal - pedestalINI;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    tocampl += tocdefault;             //

    if (ii == 1 || ii == 2 || ii == 3 || ii == 4 || ii == 5 || ii == 6 || ii == 7 || ii == 8)
      amplitude345 += ampldefault;

    if (flagcpuoptimization_ == 0) {
    }  //flagcpuoptimization

    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;

    if (ii == 2 && ampldefault > 0.)
      flagTS2 = 1;
  }  //for 1
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  amplitude0 = amplitude;

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize) {
    ampl = tool[ts_with_max_signal];
    ampl3ts = tool[ts_with_max_signal];
  }
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize) {
    ampl += tool[ts_with_max_signal - 1];
    ampl3ts += tool[ts_with_max_signal - 1];
  }
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize) {
    ampl += tool[ts_with_max_signal + 1];
    ampl3ts += tool[ts_with_max_signal + 1];
  }
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize) {
    ampl += tool[ts_with_max_signal + 2];
  }
  // HE charge correction for SiPMs:
  if (flagsipmcorrection_ != 0) {
    if (sub == 2) {
      double xa = amplitude / 40.;
      double xb = ampl / 40.;
      double xc = amplitude345 / 40.;
      double xd = ampl3ts / 40.;
      double txa = tocampl / 40.;
      // ADDI case:
      if (((ieta == -16 || ieta == 15) && mdepth == 4) ||
          ((ieta == -17 || ieta == 16) && (mdepth == 2 || mdepth == 3)) ||
          ((ieta == -18 || ieta == 17) && mdepth == 5)) {
        double c0 = 1.000000;
        double b1 = 2.59096e-05;
        double a2 = 4.60721e-11;
        double corrforxa = a2 * xa * xa + b1 * xa + c0;
        double corrforxb = a2 * xb * xb + b1 * xb + c0;
        double corrforxc = a2 * xc * xc + b1 * xc + c0;
        double corrforxd = a2 * xd * xd + b1 * xd + c0;
        double corrfortxa = a2 * txa * txa + b1 * txa + c0;
        h_corrforxaADDI_HE->Fill(amplitude, corrforxa);
        h_corrforxaADDI0_HE->Fill(amplitude, 1.);
        amplitude *= corrforxa;
        ampl *= corrforxb;
        amplitude345 *= corrforxc;
        ampl3ts *= corrforxd;
        tocampl *= corrfortxa;
      }
      // MAIN case:
      else {
        double c0 = 1.000000;
        double b1 = 2.71238e-05;
        double a2 = 1.32877e-10;
        double corrforxa = a2 * xa * xa + b1 * xa + c0;
        double corrforxb = a2 * xb * xb + b1 * xb + c0;
        double corrforxc = a2 * xc * xc + b1 * xc + c0;
        double corrforxd = a2 * xd * xd + b1 * xd + c0;
        double corrfortxa = a2 * txa * txa + b1 * txa + c0;
        h_corrforxaMAIN_HE->Fill(amplitude, corrforxa);
        h_corrforxaMAIN0_HE->Fill(amplitude, 1.);
        amplitude *= corrforxa;
        ampl *= corrforxb;
        amplitude345 *= corrforxc;
        ampl3ts *= corrforxd;
        tocampl *= corrfortxa;
      }
    }  // sub == 2   HE charge correction end
  }    //flagsipmcorrection_
  ///////   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      !!!!!!!!!!!!!!!!!!                      fillDigiAmplitudeQIE11
  // sub=1||2 HBHE
  if (sub == 1 || sub == 2) {
    amplitudechannel0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;                 // 0-neta ; 0-71  HBHE
    amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;           // 0-neta ; 0-71  HBHE
    amplitudechannel2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(amplitude, 2);  // 0-neta ; 0-71  HBHE
  }
  tocamplchannel[sub - 1][mdepth - 1][ieta + 41][iphi] += tocampl;  // 0-neta ; 0-71  HBHE

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude0 > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude0;  // average_TS +1
  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = (qie11df[ii]).capid();
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
  }  //for 2

  if (!anycapid)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude0 > 0 && rmsamp > 0) || (amplitude0 < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude0);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9
  // CapIdErrors end  /////////////////////////////////////////////////////////

  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_forCapIdErrors_HE3->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_notCapIdErrors_HE3->Fill(amplitude, 1.);
  }

  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<10; ii++) {
    double ampldefault = tool[ii];
    ///
    if (sub == 1) {
      if (amplitude0 > 120) {
        h_shape_Ahigh_HB0->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB0->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB0->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB0->Fill(float(ii), 1.);
      }  //HB0
      ///
      if (pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        h_shape_Ahigh_HB1->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB1->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB1->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB1->Fill(float(ii), 1.);
      }  //HB1
      if (error == 0) {
        h_shape_Ahigh_HB2->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB2->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB2->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB2->Fill(float(ii), 1.);
      }  //HB2
      ///
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        h_shape_Ahigh_HB3->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB3->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB3->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB3->Fill(float(ii), 1.);
      }  //HB3

    }  // sub   HB

  }  //for 3 over TSs

  if (sub == 1) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude0 < ADCAmplHBMin_ || amplitude0 > ADCAmplHBMax_ || rmsamplitude < rmsHBMin_ ||
        rmsamplitude > rmsHBMax_ || pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ ||
        pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestalw0 < pedestalwHBMax_ ||
        pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ || pedestalw3 < pedestalwHBMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HB->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HB->Fill(float(ii), 1.);
      }
    }
  }  // sub   HB

  // HE starts:
  if (sub == 2) {
    // shape bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude0 < ADCAmplHEMin_ || amplitude0 > ADCAmplHEMax_ || rmsamplitude < rmsHEMin_ ||
        rmsamplitude > rmsHEMax_ || pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ ||
        pedestal2 < pedestalHEMax_ || pedestal3 < pedestalHEMax_ || pedestalw0 < pedestalwHEMax_ ||
        pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ || pedestalw3 < pedestalwHEMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HE->Fill(float(ii), 1.);
      }
    }
    // shape good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HE->Fill(float(ii), 1.);
      }  // ii
    }    // else for good channels
  }      // sub   HE
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  //    sumEstimator0[sub-1][mdepth-1][ieta+41][iphi] += pedestalw0;//Sig_Pedestals
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HB
  if (sub == 1) {
    if (studyPedestalCorrelations_) {
      //   //   //   //   //   //   //   //   //  HB       PedestalCorrelations :
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HB->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl_HB->Fill(mypedestal, amplitude);
      h_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HB->Fill(mypedestal, 1.);
      h_pedwvsampl0_HB->Fill(mypedestalw, 1.);

      h2_amplvsped_HB->Fill(amplitude, mypedestal);
      h2_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped_HB->Fill(amplitude, mypedestal);
      h_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped0_HB->Fill(amplitude, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HB->Fill(pedestal0, 1.);
      h_pedestal1_HB->Fill(pedestal1, 1.);
      h_pedestal2_HB->Fill(pedestal2, 1.);
      h_pedestal3_HB->Fill(pedestal3, 1.);
      h_pedestalaver4_HB->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HB->Fill(pedestalaver9, 1.);
      h_pedestalw0_HB->Fill(pedestalw0, 1.);
      h_pedestalw1_HB->Fill(pedestalw1, 1.);
      h_pedestalw2_HB->Fill(pedestalw2, 1.);
      h_pedestalw3_HB->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HB->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HB->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth4pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HB->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HB->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HB->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HB->Fill(respcorr->getValue(), 1.);
      h_timecorr_HB->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HB->Fill(lutcorr->getValue(), 1.);
      h_difpedestal0_HB->Fill(difpedestal0, 1.);
      h_difpedestal1_HB->Fill(difpedestal1, 1.);
      h_difpedestal2_HB->Fill(difpedestal2, 1.);
      h_difpedestal3_HB->Fill(difpedestal3, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345Zoom1_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345_HB->Fill(amplitude345, 1.);
      if (error == 0) {
        h_ADCAmpl_HBCapIdNoError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdNoError->Fill(amplitude345, 1.);
      }
      if (error == 1) {
        h_ADCAmpl_HBCapIdError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdError->Fill(amplitude345, 1.);
      }
      h_ADCAmplZoom_HB->Fill(amplitude, 1.);
      h_ADCAmplZoom1_HB->Fill(amplitude, 1.);
      h_ADCAmpl_HB->Fill(amplitude, 1.);

      h_AmplitudeHBrest->Fill(amplitude, 1.);
      h_AmplitudeHBrest1->Fill(amplitude, 1.);
      h_AmplitudeHBrest6->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >400.) averSIGNALoccupancy_HB += 1.;
      if (amplitude < 35.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HB->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HB->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HB->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HB->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HB->Fill(aveamplitude1, 1.);
      //      h2_TSnVsAyear2023_HB->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HB->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HB->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HB->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHBMin_ || aveamplitude1 > TSmeanHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HB->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHBMin_ || ts_with_max_signal > TSpeakHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HB->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHBMin_ || rmsamplitude > rmsHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HB->Fill(ratio, 1.);
      if (ratio < ratioHBMin_ || ratio > ratioHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        // //
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HB All
    if (mdepth == 1)
      h_mapDepth1_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HB->Fill(double(ieta), double(iphi), 1.);

    if (flagTS2 == 1) {
      if (mdepth == 1)
        h_mapDepth1TS2_HB->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 2)
        h_mapDepth2TS2_HB->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 3)
        h_mapDepth3TS2_HB->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 4)
        h_mapDepth4TS2_HB->Fill(double(ieta), double(iphi), 1.);
    }  // select entries only in TS=2

  }  //if ( sub == 1 )

  // HE   QIE11
  if (sub == 2) {
    //   //   //   //   //   //   //   //   //  HE   QIE11    PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      //	double mypedestal  = pedestalaver9;
      //	double mypedestalw = pedestalwaver9;
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HE->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl_HE->Fill(mypedestal, amplitude);
      h_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HE->Fill(mypedestal, 1.);
      h_pedwvsampl0_HE->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE   QIE11    Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HE->Fill(pedestal0, 1.);
      h_pedestal1_HE->Fill(pedestal1, 1.);
      h_pedestal2_HE->Fill(pedestal2, 1.);
      h_pedestal3_HE->Fill(pedestal3, 1.);
      h_pedestalaver4_HE->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HE->Fill(pedestalaver9, 1.);
      h_pedestalw0_HE->Fill(pedestalw0, 1.);
      h_pedestalw1_HE->Fill(pedestalw1, 1.);
      h_pedestalw2_HE->Fill(pedestalw2, 1.);
      h_pedestalw3_HE->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HE->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HE->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 3) {
        h_mapDepth3Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth3Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth3Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth3Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth3Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth3Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth3Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth3Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHEMax_ || pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ ||
          pedestalw3 < pedestalwHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ || pedestal2 < pedestalHEMax_ ||
          pedestal3 < pedestalHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7pedestal_HE->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HE->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HE->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HE->Fill(respcorr->getValue(), 1.);
      h_timecorr_HE->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HE->Fill(lutcorr->getValue(), 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE  QIE11     ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345_HE->Fill(amplitude345, 1.);
      h_ADCAmpl_HE->Fill(amplitude, 1.);
      //	if( ieta <0) h_ADCAmpl_HEM->Fill(amplitude,1.);
      //	if( ieta >0) h_ADCAmpl_HEP->Fill(amplitude,1.);
      h_ADCAmplrest_HE->Fill(amplitude, 1.);
      h_ADCAmplrest1_HE->Fill(amplitude, 1.);
      h_ADCAmplrest6_HE->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude > 700.) averSIGNALoccupancy_HE += 1.;
      if (amplitude < 500.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if

      h_ADCAmplZoom1_HE->Fill(amplitude, 1.);   // for amplitude allTS
      h_ADCAmpl345Zoom1_HE->Fill(ampl3ts, 1.);  // for ampl3ts 3TSs
      h_ADCAmpl345Zoom_HE->Fill(ampl, 1.);      // for ampl 4TSs

      if (amplitude > 110 && amplitude < 150) {
        h_mapADCAmplfirstpeak_HE->Fill(double(ieta), double(iphi), amplitude);
        h_mapADCAmplfirstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (amplitude > 150 && amplitude < 190) {
        h_mapADCAmplsecondpeak_HE->Fill(double(ieta), double(iphi), amplitude);
        h_mapADCAmplsecondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }

      if (ampl3ts > 70 && ampl3ts < 110) {
        h_mapADCAmpl11firstpeak_HE->Fill(double(ieta), double(iphi), ampl3ts);
        h_mapADCAmpl11firstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (ampl3ts > 110 && ampl3ts < 150) {
        h_mapADCAmpl11secondpeak_HE->Fill(double(ieta), double(iphi), ampl3ts);
        h_mapADCAmpl11secondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (ampl > 87 && ampl < 127) {
        h_mapADCAmpl12firstpeak_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapADCAmpl12firstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (ampl > 127 && ampl < 167) {
        h_mapADCAmpl12secondpeak_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapADCAmpl12secondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values of every channel:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 5)
        h_mapDepth5ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 6)
        h_mapDepth6ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 7)
        h_mapDepth7ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 5)
        h_mapDepth5ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 6)
        h_mapDepth6ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 7)
        h_mapDepth7ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      // for averaged values of SiPM channels only:
      if (mdepth == 1)
        h_mapDepth1ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      //
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HE->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HE->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HE->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HE->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    //   //   //   //   //   //   //   //   //  HE  QIE11     TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HE->Fill(aveamplitude1, 1.);
      //    h2_TSnVsAyear2023_HE->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HE->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HE->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HE->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHEMin_ || aveamplitude1 > TSmeanHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 5)
        h_mapDepth5TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 6)
        h_mapDepth6TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 7)
        h_mapDepth7TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE  QIE11     TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HE->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHEMin_ || ts_with_max_signal > TSpeakHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 5)
        h_mapDepth5TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 6)
        h_mapDepth6TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 7)
        h_mapDepth7TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE   QIE11    RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HE->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHEMin_ || rmsamplitude > rmsHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 5)
        h_mapDepth5Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 6)
        h_mapDepth6Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 7)
        h_mapDepth7Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HE  QIE11     Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HE->Fill(ratio, 1.);
      if (ratio < ratioHEMin_ || ratio > ratioHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 5)
        h_mapDepth5Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 6)
        h_mapDepth6Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 7)
        h_mapDepth7Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE   QIE11    DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 5)
        h_mapDepth5AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 6)
        h_mapDepth6AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 7)
        h_mapDepth7AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)
    ///////////////////////////////    for HE All QIE11
    if (mdepth == 1)
      h_mapDepth1_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 5)
      h_mapDepth5_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 6)
      h_mapDepth6_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 7)
      h_mapDepth7_HE->Fill(double(ieta), double(iphi), 1.);

    if (flagTS2 == 1) {
      if (mdepth == 1)
        h_mapDepth1TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 2)
        h_mapDepth2TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 3)
        h_mapDepth3TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 4)
        h_mapDepth4TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 5)
        h_mapDepth5TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 6)
        h_mapDepth6TS2_HE->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 7)
        h_mapDepth7TS2_HE->Fill(double(ieta), double(iphi), 1.);
    }  // select entries only in TS=2

  }  //if ( sub == 2 )
     //
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHF(HFDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // (HFDigiCollection: 4-HF)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;
  int TSsize = 4;  // HF for Run2
  if ((*digiItr).size() != TSsize)
    errorBtype = 1.;
  TSsize = digiItr->size();
  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<digiItr->size(); ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }

    int capid = ((*digiItr)[ii]).capid();
    //      double pedestal = calib.pedestal(capid);
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction

    tool[ii] = ampldefault;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    ///////////////////////////////////

    if (flagcpuoptimization_ == 0) {
    }  //  if(flagcpuoptimization_== 0
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }  //for 1
  /////////////////////////////////////////////////////////////////////////////////////////////////////fillDigiAmplitudeHF
  // sub=4 HF
  if (sub == 4) {
    amplitudechannel0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;                 // 0-neta ; 0-71 HF
    amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;           // 0-neta ; 0-71 HF
    amplitudechannel2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(amplitude, 2);  // 0-neta ; 0-71 HF
  }

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;

  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;

  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1

  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    //    std::cout << " ii = " << ii  << " capid = " << capid  << " ((lastcapid+1)%4) = " << ((lastcapid+1)%4)  << std::endl;
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2

  if (!anycapid || anyer || !anydv)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9
  if (error == 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HF2->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HF2->Fill(amplitude, 1.);
  }

  if (sub == 4) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_ || rmsamplitude < rmsHFMin_ ||
        rmsamplitude > rmsHFMax_ || pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ ||
        pedestal2 < pedestalHFMax_ || pedestal3 < pedestalHFMax_ || pedestalw0 < pedestalwHFMax_ ||
        pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ || pedestalw3 < pedestalwHFMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HF->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HF->Fill(float(ii), 1.);
      }
    }
  }  // sub   HF
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //    Pedestals
  // HF
  if (sub == 4) {
    //   //   //   //   //   //   //   //   //  HF      PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HF->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl_HF->Fill(mypedestal, amplitude);
      h_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HF->Fill(mypedestal, 1.);
      h_pedwvsampl0_HF->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HF       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HF->Fill(pedestal0, 1.);
      h_pedestal1_HF->Fill(pedestal1, 1.);
      h_pedestal2_HF->Fill(pedestal2, 1.);
      h_pedestal3_HF->Fill(pedestal3, 1.);
      h_pedestalaver4_HF->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HF->Fill(pedestalaver9, 1.);
      h_pedestalw0_HF->Fill(pedestalw0, 1.);
      h_pedestalw1_HF->Fill(pedestalw1, 1.);
      h_pedestalw2_HF->Fill(pedestalw2, 1.);
      h_pedestalw3_HF->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HF->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HF->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }

      if (pedestalw0 < pedestalwHFMax_ || pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ ||
          pedestalw3 < pedestalwHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
      }

      if (pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ || pedestal2 < pedestalHFMax_ ||
          pedestal3 < pedestalHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HF->Fill(double(ieta), double(iphi), 1.);
      }

      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HF->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HF->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HF->Fill(respcorr->getValue(), 1.);
      h_timecorr_HF->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HF->Fill(lutcorr->getValue(), 1.);

    }  //

    //   //   //   //   //   //   //   //   //  HF       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HF->Fill(amplitude, 1.);
      h_ADCAmplrest1_HF->Fill(amplitude, 1.);
      h_ADCAmplrest6_HF->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HF->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >1500.) averSIGNALoccupancy_HF += 1.;
      if (amplitude < 20.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if

      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HF->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HF->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HF->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HF->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HF       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HF->Fill(aveamplitude1, 1.);
      //    h2_TSnVsAyear2023_HF->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HF->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HF->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HF->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHFMin_ || aveamplitude1 > TSmeanHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HF->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHFMin_ || ts_with_max_signal > TSpeakHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HF->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHFMin_ || rmsamplitude > rmsHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HF->Fill(ratio, 1.);
      if (ratio < ratioHFMin_ || ratio > ratioHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HF->Fill(double(ieta), double(iphi), ratio);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)

    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HF All
    if (mdepth == 1)
      h_mapDepth1_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HF->Fill(double(ieta), double(iphi), 1.);

  }  //if ( sub == 4 )

  //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHFQIE10(QIE10DataFrame qie10df) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  DetId detid = qie10df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFQIE10DigiCollection: 4-HF)
  nTS = qie10df.samples();       //  ----------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  /*
                                  # flag   HBHE8    HBHE11   HF8   HF10  comments:
                                  #  0       +        +       +     +     all
                                  #  1       +        -       +     -     old
                                  #  2       -        +       -     +     new (2018)
                                  #  3       -        +       -     +     new w/o high depthes
                                  #  4       +        -       +     +     2016fall
                                  #  5       +        -       +     +     2016fall w/o high depthes
                                  #  6       +        +       -     +     2017begin
                                  #  7       +        +       -     +     2017begin w/o high depthes in HEonly
                                  #  8       +        +       -     +     2017begin w/o high depthes
                                  #  9       +        +       +     +     all  w/o high depthes
*/
  if (mdepth == 0 || sub != 4)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 3)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 5)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 8)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 9)
    return;
  /////////////////////////////////////////////////////////////////
  const HcalPedestal* pedestal00 = conditions->getPedestal(hcaldetid);
  const HcalGain* gain = conditions->getGain(hcaldetid);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(hcaldetid);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(hcaldetid);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(hcaldetid);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(hcaldetid);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(hcaldetid);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(qie10df, toolOriginal);
  //    double noiseADC = qie10df[0].adc();
  /////////////////////////////////////////////////////////////////
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  int TSsize = 3;  // HF for Run3
  if (nTS != TSsize)
    errorBtype = 1.;
  TSsize = nTS;  // ------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  int flagTS1 = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC_QIE10[qie10df[ii].adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];  //adcfC
    ampldefault2 = qie10df[ii].adc();   //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }

    int capid = (qie10df[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);

    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction

    tool[ii] = ampldefault;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    ///////////////////////////////////
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
    if (ii == 1 && ampldefault > 0.)
      flagTS1 = 1;
  }  //for 1
  /////////////////////////////////////////////////////////////////////////////////////////////////////fillDigiAmplitudeHFQIE10
  // sub=4 HF
  if (sub == 4) {
    amplitudechannel0[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;                 // 0-neta ; 0-71 HF
    amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;           // 0-neta ; 0-71 HF
    amplitudechannel2[sub - 1][mdepth - 1][ieta + 41][iphi] += pow(amplitude, 2);  // 0-neta ; 0-71 HF
  }  // just in case against any violations

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1

  double rmsamp = 0.;
  int error = 0;
  bool anycapid = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = (qie10df[ii]).capid();
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
  }  //for 2

  if (!anycapid)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9

  // CapIdErrors end  /////////////////////////////////////////////////////////
  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HF2->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HF2->Fill(amplitude, 1.);
  }

  if (sub == 4) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_ || rmsamplitude < rmsHFMin_ ||
        rmsamplitude > rmsHFMax_ || pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ ||
        pedestal2 < pedestalHFMax_ || pedestal3 < pedestalHFMax_ || pedestalw0 < pedestalwHFMax_ ||
        pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ || pedestalw3 < pedestalwHFMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HF->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HF->Fill(float(ii), 1.);
      }
    }
  }  // sub   HFQIE10
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //    Pedestals
  // HFQIE10
  if (sub == 4) {
    //   //   //   //   //   //   //   //   //  HFQIE10      PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HF->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl_HF->Fill(mypedestal, amplitude);
      h_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HF->Fill(mypedestal, 1.);
      h_pedwvsampl0_HF->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HFQIE10       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HF->Fill(pedestal0, 1.);
      h_pedestal1_HF->Fill(pedestal1, 1.);
      h_pedestal2_HF->Fill(pedestal2, 1.);
      h_pedestal3_HF->Fill(pedestal3, 1.);
      h_pedestalaver4_HF->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HF->Fill(pedestalaver9, 1.);
      h_pedestalw0_HF->Fill(pedestalw0, 1.);
      h_pedestalw1_HF->Fill(pedestalw1, 1.);
      h_pedestalw2_HF->Fill(pedestalw2, 1.);
      h_pedestalw3_HF->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HF->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HF->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }

      if (pedestalw0 < pedestalwHFMax_ || pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ ||
          pedestalw3 < pedestalwHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
      }

      if (pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ || pedestal2 < pedestalHFMax_ ||
          pedestal3 < pedestalHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HF->Fill(double(ieta), double(iphi), 1.);
      }

      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HF->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HF->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HF->Fill(respcorr->getValue(), 1.);
      h_timecorr_HF->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HF->Fill(lutcorr->getValue(), 1.);

    }  //

    //   //   //   //   //   //   //   //   //  HFQIE10       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HF->Fill(amplitude, 1.);
      h_ADCAmplrest1_HF->Fill(amplitude, 1.);
      h_ADCAmplrest6_HF->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HF->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >1500.) averSIGNALoccupancy_HF += 1.;
      if (amplitude < 20.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if

      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HF->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HF->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HF->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HF->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    //   //   //   //   //   //   //   //   //  HFQIE10       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HF->Fill(aveamplitude1, 1.);
      //    h2_TSnVsAyear2023_HF->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HF->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HF->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HF->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHFMin_ || aveamplitude1 > TSmeanHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HF->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHFMin_ || ts_with_max_signal > TSpeakHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HF->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHFMin_ || rmsamplitude > rmsHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:

      if (mdepth == 1)
        h_mapDepth1Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HF->Fill(ratio, 1.);
      if (ratio < ratioHFMin_ || ratio > ratioHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HF->Fill(double(ieta), double(iphi), ratio);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)

    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HFQIE10 All
    if (mdepth == 1)
      h_mapDepth1_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HF->Fill(double(ieta), double(iphi), 1.);

    if (flagTS1 == 1) {
      if (mdepth == 1)
        h_mapDepth1TS1_HF->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 2)
        h_mapDepth2TS1_HF->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 3)
        h_mapDepth3TS1_HF->Fill(double(ieta), double(iphi), 1.);
      if (mdepth == 4)
        h_mapDepth4TS1_HF->Fill(double(ieta), double(iphi), 1.);
    }  // for TS = 1

  }  //if ( sub == 4 )

  //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHO(HODigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // (HODigiCollection: 3-HO)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;
  int TSsize = 10;  //HO
  if ((*digiItr).size() != TSsize)
    errorBtype = 1.;
  TSsize = digiItr->size();
  int flagTS012 = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    int capid = ((*digiItr)[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    tool[ii] = ampldefault;
    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;
    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;
    absamplitude += abs(ampldefault);
    ///////////////////////////////////////////
    if (flagcpuoptimization_ == 0) {
    }
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
    if (ii < 3 && ampldefault > 0.)
      flagTS012 = 1;
  }                                                                     //for 1
  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71  HO

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];
  double ratio = 0.;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.04)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1
  double rmsamp = 0.;
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2

  if (!anycapid || anyer || !anydv)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9
  if (error == 1) {
    if (sub == 3 && mdepth == 4)
      h_Amplitude_forCapIdErrors_HO4->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 3 && mdepth == 4)
      h_Amplitude_notCapIdErrors_HO4->Fill(amplitude, 1.);
  }

  if (sub == 3) {
    if (error == 1 || amplitude < ADCAmplHOMin_ || amplitude > ADCAmplHOMax_ || rmsamplitude < rmsHOMin_ ||
        rmsamplitude > rmsHOMax_ || pedestal0 < pedestalHOMax_ || pedestal1 < pedestalHOMax_ ||
        pedestal2 < pedestalHOMax_ || pedestal3 < pedestalHOMax_ || pedestalw0 < pedestalwHOMax_ ||
        pedestalw1 < pedestalwHOMax_ || pedestalw2 < pedestalwHOMax_ || pedestalw3 < pedestalwHOMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HO->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HO->Fill(float(ii), 1.);
      }
    } else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HO->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HO->Fill(float(ii), 1.);
      }
    }
  }  // sub   HO
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HO
  if (sub == 3) {
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HO->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HO->Fill(mypedestalw, amplitude);
      h_pedvsampl_HO->Fill(mypedestal, amplitude);
      h_pedwvsampl_HO->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HO->Fill(mypedestal, 1.);
      h_pedwvsampl0_HO->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HO       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HO->Fill(pedestal0, 1.);
      h_pedestal1_HO->Fill(pedestal1, 1.);
      h_pedestal2_HO->Fill(pedestal2, 1.);
      h_pedestal3_HO->Fill(pedestal3, 1.);
      h_pedestalaver4_HO->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HO->Fill(pedestalaver9, 1.);
      h_pedestalw0_HO->Fill(pedestalw0, 1.);
      h_pedestalw1_HO->Fill(pedestalw1, 1.);
      h_pedestalw2_HO->Fill(pedestalw2, 1.);
      h_pedestalw3_HO->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HO->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HO->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 4) {
        h_mapDepth4Ped0_HO->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth4Ped1_HO->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth4Ped2_HO->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth4Ped3_HO->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth4Pedw0_HO->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth4Pedw1_HO->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth4Pedw2_HO->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth4Pedw3_HO->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHOMax_ || pedestalw1 < pedestalwHOMax_ || pedestalw2 < pedestalwHOMax_ ||
          pedestalw3 < pedestalwHOMax_) {
        if (mdepth == 4)
          h_mapDepth4pedestalw_HO->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHOMax_ || pedestal1 < pedestalHOMax_ || pedestal2 < pedestalHOMax_ ||
          pedestal3 < pedestalHOMax_) {
        if (mdepth == 4)
          h_mapDepth4pedestal_HO->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HO->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HO->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HO->Fill(respcorr->getValue(), 1.);
      h_timecorr_HO->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HO->Fill(lutcorr->getValue(), 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HO       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HO->Fill(amplitude, 1.);
      h_ADCAmplrest1_HO->Fill(amplitude, 1.);
      h_ADCAmplrest6_HO->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HO->Fill(amplitude, 1.);
      h_ADCAmpl_HO_copy->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHOMin_ || amplitude > ADCAmplHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >2000.) averSIGNALoccupancy_HO += 1.;

      if (amplitude < 100.) {
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HO->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HO->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HO->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HO->Fill(float(bcn), 1.);

      h_orbitNumvsamplitude_HO->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HO->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HO->Fill(aveamplitude1, 1.);
      //    h2_TSnVsAyear2023_HO->Fill(25.*aveamplitude1, amplitude);
      h2_TSnVsAyear2023_HO->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear2023_HO->Fill(amplitude, 25. * aveamplitude1);
      h1_TSnVsAyear20230_HO->Fill(amplitude, 1.);
      if (aveamplitude1 < TSmeanHOMin_ || aveamplitude1 > TSmeanHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HO->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HO->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHOMin_ || ts_with_max_signal > TSpeakHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HO->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    if (studyRMSshapeHist_) {
      h_Amplitude_HO->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHOMin_ || rmsamplitude > rmsHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      if (mdepth == 4)
        h_mapDepth4Amplitude_HO->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    if (studyRatioShapeHist_) {
      h_Ampl_HO->Fill(ratio, 1.);
      if (ratio < ratioHOMin_ || ratio > ratioHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4Ampl047_HO->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      if (mdepth == 4)
        h_mapDepth4Ampl_HO->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    if (studyDiffAmplHist_) {
      if (mdepth == 4)
        h_mapDepth4AmplE34_HO->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)
    if (mdepth == 4) {
      h_mapDepth4_HO->Fill(double(ieta), double(iphi), 1.);
      if (flagTS012 == 1)
        h_mapDepth4TS012_HO->Fill(double(ieta), double(iphi), 1.);
    }
  }  //if ( sub == 3 )
}
int CMTRawAnalyzer::getRBX(int& kdet, int& keta, int& kphi) {
  int cal_RBX = 0;
  if (kdet == 1 || kdet == 2) {
    if (kphi == 71)
      cal_RBX = 0;
    else
      cal_RBX = (kphi + 1) / 4;
    cal_RBX = cal_RBX + 18 * (keta + 1) / 2;
  }
  if (kdet == 4) {
    cal_RBX = (int)(kphi / 18) + 1;
  }
  if (kdet == 3) {
    if (keta == -2) {
      if (kphi == 71)
        cal_RBX = 0;
      else
        cal_RBX = kphi / 12 + 1;
    }
    if (keta == -1) {
      if (kphi == 71)
        cal_RBX = 6;
      else
        cal_RBX = kphi / 12 + 1 + 6;
    }
    if (keta == 0) {
      if (kphi == 71)
        cal_RBX = 12;
      else
        cal_RBX = kphi / 6 + 1 + 12;
    }
    if (keta == 1) {
      if (kphi == 71)
        cal_RBX = 24;
      else
        cal_RBX = kphi / 12 + 1 + 24;
    }
    if (keta == 2) {
      if (kphi == 71)
        cal_RBX = 30;
      else
        cal_RBX = kphi / 12 + 1 + 30;
    }
  }
  return cal_RBX;
}
void CMTRawAnalyzer::beginRun(const edm::Run& r, const edm::EventSetup& iSetup) {}
void CMTRawAnalyzer::endRun(const edm::Run& r, const edm::EventSetup& iSetup) {
  if (flagfitshunt1pedorledlowintensity_ != 0) {
  }  // if flag...
  if (usecontinuousnumbering_) {
    lscounterM1 = lscounter - 1;
  } else {
    lscounterM1 = ls0;
  }
  if (ls0 != -1)
    h_nevents_per_eachRealLS->Fill(float(lscounterM1), float(nevcounter));  //
  h_nevents_per_LS->Fill(float(nevcounter));
  h_nevents_per_LSzoom->Fill(float(nevcounter));
  nevcounter0 = nevcounter;
  if (nevcounter0 != 0) {
    for (int k0 = 0; k0 < nsub; k0++) {
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            int ieta = k2 - 41;
            if (sumEstimator0[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator0[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator0[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator0[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }
              if (k0 == 0) {
                if (k1 + 1 == 1) {
                  h_sumPedestalLS1->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS2->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumPedestalLS3->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS4->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumPedestalLS5->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumPedestalLS6->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS7->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumPedestalLS8->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator0[k0][k1][k2][k3] != 0.
            if (sumEstimator1[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator1[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator1[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator1[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }
              if (lscounterM1 >= lsmin_ && lscounterM1 < lsmax_) {
                int kkkk2 = (k2 - 1) / 4;
                if (k2 == 0)
                  kkkk2 = 1.;
                else
                  kkkk2 += 2;              //kkkk2= 1-22
                int kkkk3 = (k3) / 4 + 1;  //kkkk3= 1-18
                int ietaphi = 0;
                ietaphi = ((kkkk2)-1) * znphi + (kkkk3);
                double bbb3 = 0.;
                if (bbb1 != 0.)
                  bbb3 = bbbc / bbb1;
                if (k0 == 0) {
                  h_2DsumADCAmplEtaPhiLs0->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HB
                  h_2DsumADCAmplEtaPhiLs00->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HB
                }
                if (k0 == 1) {
                  h_2DsumADCAmplEtaPhiLs1->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HE
                  h_2DsumADCAmplEtaPhiLs10->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HE
                }
                if (k0 == 2) {
                  h_2DsumADCAmplEtaPhiLs2->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HO
                  h_2DsumADCAmplEtaPhiLs20->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HO
                }
                if (k0 == 3) {
                  h_2DsumADCAmplEtaPhiLs3->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HF
                  h_2DsumADCAmplEtaPhiLs30->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HF
                }

                h_sumADCAmplEtaPhiLs->Fill(bbb3);
                h_sumADCAmplEtaPhiLs_bbbc->Fill(bbbc);
                h_sumADCAmplEtaPhiLs_bbb1->Fill(bbb1);
                h_sumADCAmplEtaPhiLs_lscounterM1orbitNum->Fill(float(lscounterM1), float(orbitNum));
                h_sumADCAmplEtaPhiLs_orbitNum->Fill(float(orbitNum), 1.);
                h_sumADCAmplEtaPhiLs_lscounterM1->Fill(float(lscounterM1), 1.);
                h_sumADCAmplEtaPhiLs_ietaphi->Fill(float(ietaphi));
              }  // lscounterM1 >= lsmin_ && lscounterM1 < lsmax_
              if (k0 == 0) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS1copy1->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy2->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy3->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy4->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy5->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                    h_2DsumADCAmplLS1->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth1_)
                    h_2DsumADCAmplLS1_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                    h_sumCutADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                    h_2DsumADCAmplLS2->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth2_)
                    h_2DsumADCAmplLS2_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                    h_sumCutADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS2->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                    h_sumCutADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbb1);

                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                    h_2DsumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==3)
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                    h_sumCutADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbb1);

                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                    h_2DsumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
              }
              if (k0 == 1) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                    h_2DsumADCAmplLS3->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth1_)
                    h_2DsumADCAmplLS3_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                    h_sumCutADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                    h_2DsumADCAmplLS4->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth2_)
                    h_2DsumADCAmplLS4_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                    h_sumCutADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumADCAmplLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                    h_2DsumADCAmplLS5->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth3_)
                    h_2DsumADCAmplLS5_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                    h_sumCutADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS5->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                    h_sumCutADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                    h_2DsumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
                if (k1 + 1 == 5) {
                  h_sumADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                    h_sumCutADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                    h_2DsumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==5)

                if (k1 + 1 == 6) {
                  h_sumADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                    h_sumCutADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                    h_2DsumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==6)
                if (k1 + 1 == 7) {
                  h_sumADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                    h_sumCutADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                    h_2DsumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==7)

              }  // end HE

              if (k0 == 3) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                    h_2DsumADCAmplLS6->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth1_)
                    h_2DsumADCAmplLS6_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                    h_sumCutADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                    h_2DsumADCAmplLS7->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth2_)
                    h_2DsumADCAmplLS7_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                    h_sumCutADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS7->Fill(float(lscounterM1), bbb1);
                }

                if (k1 + 1 == 3) {
                  h_sumADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                    h_sumCutADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS6u->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                    h_2DsumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==3)
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                    h_sumCutADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS7u->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                    h_2DsumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumADCAmplLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                    h_2DsumADCAmplLS8->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HOdepth4_)
                    h_2DsumADCAmplLS8_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                    h_sumCutADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator1[k0][k1][k2][k3] != 0.
            if (sumEstimator2[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator2[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator2[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator2[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                    h_2DsumTSmeanALS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                    h_sumCutTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator2_HBdepth1_)
                    h_sumTSmeanAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                    h_2DsumTSmeanALS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                    h_sumCutTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              if (k0 == 1) {
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                    h_2DsumTSmeanALS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                    h_sumCutTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                    h_2DsumTSmeanALS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                    h_sumCutTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumTSmeanALS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                    h_2DsumTSmeanALS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                    h_sumCutTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                    h_2DsumTSmeanALS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                    h_sumCutTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                    h_2DsumTSmeanALS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                    h_sumCutTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumTSmeanALS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                    h_2DsumTSmeanALS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                    h_sumCutTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator2[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator3 Tx
            if (sumEstimator3[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator3[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator3[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator3[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                    h_2DsumTSmaxALS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                    h_sumCutTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator3_HBdepth1_)
                    h_sumTSmaxAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                    h_2DsumTSmaxALS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                    h_sumCutTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                    h_2DsumTSmaxALS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                    h_sumCutTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                    h_2DsumTSmaxALS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                    h_sumCutTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumTSmaxALS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                    h_2DsumTSmaxALS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                    h_sumCutTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                    h_2DsumTSmaxALS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                    h_sumCutTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                    h_2DsumTSmaxALS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                    h_sumCutTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumTSmaxALS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                    h_2DsumTSmaxALS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                    h_sumCutTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator3[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator4 W
            if (sumEstimator4[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator4[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator4[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator4[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                    h_2DsumAmplitudeLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                    h_sumCutAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator4_HBdepth1_)
                    h_sumAmplitudeperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                    h_2DsumAmplitudeLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                    h_sumCutAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                    h_2DsumAmplitudeLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                    h_sumCutAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                    h_2DsumAmplitudeLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                    h_sumCutAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumAmplitudeLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                    h_2DsumAmplitudeLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                    h_sumCutAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                    h_2DsumAmplitudeLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                    h_sumCutAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                    h_2DsumAmplitudeLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                    h_sumCutAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumAmplitudeLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                    h_2DsumAmplitudeLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                    h_sumCutAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator4[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator5 R
            if (sumEstimator5[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator5[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator5[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator5[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                    h_2DsumAmplLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                    h_sumCutAmplperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator5_HBdepth1_)
                    h_sumAmplperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                    h_2DsumAmplLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                    h_sumCutAmplperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                    h_2DsumAmplLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                    h_sumCutAmplperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                    h_2DsumAmplLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                    h_sumCutAmplperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumAmplLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                    h_2DsumAmplLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                    h_sumCutAmplperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                    h_2DsumAmplLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                    h_sumCutAmplperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                    h_2DsumAmplLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                    h_sumCutAmplperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumAmplLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                    h_2DsumAmplLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                    h_sumCutAmplperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator5[k0][k1][k2][k3] != 0.
            // ------------------------------------------------------------------------------------------------------------------------sumEstimator6 (Error-B)
            if (sumEstimator6[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator6[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator6[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator6[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS1->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS2->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS3->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS4->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumErrorBLS5->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS6->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS7->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth4
                if (k1 + 1 == 4) {
                  h_sumErrorBLS8->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
              ///
            }  //if(sumEstimator6[k0][k1][k2][k3] != 0.

            ///
            ///
          }  //for
        }    //for
      }      //for
    }        //for

    //------------------------------------------------------   averSIGNAL
    averSIGNALoccupancy_HB /= float(nevcounter0);
    h_averSIGNALoccupancy_HB->Fill(float(lscounterM1), averSIGNALoccupancy_HB);
    averSIGNALoccupancy_HE /= float(nevcounter0);
    h_averSIGNALoccupancy_HE->Fill(float(lscounterM1), averSIGNALoccupancy_HE);
    averSIGNALoccupancy_HF /= float(nevcounter0);
    h_averSIGNALoccupancy_HF->Fill(float(lscounterM1), averSIGNALoccupancy_HF);
    averSIGNALoccupancy_HO /= float(nevcounter0);
    h_averSIGNALoccupancy_HO->Fill(float(lscounterM1), averSIGNALoccupancy_HO);

    averSIGNALoccupancy_HB = 0.;
    averSIGNALoccupancy_HE = 0.;
    averSIGNALoccupancy_HF = 0.;
    averSIGNALoccupancy_HO = 0.;

    //------------------------------------------------------
    averSIGNALsumamplitude_HB /= float(nevcounter0);
    h_averSIGNALsumamplitude_HB->Fill(float(lscounterM1), averSIGNALsumamplitude_HB);
    averSIGNALsumamplitude_HE /= float(nevcounter0);
    h_averSIGNALsumamplitude_HE->Fill(float(lscounterM1), averSIGNALsumamplitude_HE);
    averSIGNALsumamplitude_HF /= float(nevcounter0);
    h_averSIGNALsumamplitude_HF->Fill(float(lscounterM1), averSIGNALsumamplitude_HF);
    averSIGNALsumamplitude_HO /= float(nevcounter0);
    h_averSIGNALsumamplitude_HO->Fill(float(lscounterM1), averSIGNALsumamplitude_HO);

    averSIGNALsumamplitude_HB = 0.;
    averSIGNALsumamplitude_HE = 0.;
    averSIGNALsumamplitude_HF = 0.;
    averSIGNALsumamplitude_HO = 0.;

    //------------------------------------------------------   averNOSIGNAL
    averNOSIGNALoccupancy_HB /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HB->Fill(float(lscounterM1), averNOSIGNALoccupancy_HB);
    averNOSIGNALoccupancy_HE /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HE->Fill(float(lscounterM1), averNOSIGNALoccupancy_HE);
    averNOSIGNALoccupancy_HF /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HF->Fill(float(lscounterM1), averNOSIGNALoccupancy_HF);
    averNOSIGNALoccupancy_HO /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HO->Fill(float(lscounterM1), averNOSIGNALoccupancy_HO);

    averNOSIGNALoccupancy_HB = 0.;
    averNOSIGNALoccupancy_HE = 0.;
    averNOSIGNALoccupancy_HF = 0.;
    averNOSIGNALoccupancy_HO = 0.;

    //------------------------------------------------------
    averNOSIGNALsumamplitude_HB /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HB->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HB);
    averNOSIGNALsumamplitude_HE /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HE->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HE);
    averNOSIGNALsumamplitude_HF /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HF->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HF);
    averNOSIGNALsumamplitude_HO /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HO->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HO);

    averNOSIGNALsumamplitude_HB = 0.;
    averNOSIGNALsumamplitude_HE = 0.;
    averNOSIGNALsumamplitude_HF = 0.;
    averNOSIGNALsumamplitude_HO = 0.;

    h_maxxSUMAmpl_HB->Fill(float(lscounterM1), maxxSUM1);
    h_maxxSUMAmpl_HE->Fill(float(lscounterM1), maxxSUM2);
    h_maxxSUMAmpl_HO->Fill(float(lscounterM1), maxxSUM3);
    h_maxxSUMAmpl_HF->Fill(float(lscounterM1), maxxSUM4);
    maxxSUM1 = 0.;
    maxxSUM2 = 0.;
    maxxSUM3 = 0.;
    maxxSUM4 = 0.;
    //------------------------------------------------------
    h_maxxOCCUP_HB->Fill(float(lscounterM1), maxxOCCUP1);
    h_maxxOCCUP_HE->Fill(float(lscounterM1), maxxOCCUP2);
    h_maxxOCCUP_HO->Fill(float(lscounterM1), maxxOCCUP3);
    h_maxxOCCUP_HF->Fill(float(lscounterM1), maxxOCCUP4);
    maxxOCCUP1 = 0.;
    maxxOCCUP2 = 0.;
    maxxOCCUP3 = 0.;
    maxxOCCUP4 = 0.;

  }  //if( nevcounter0 != 0 )
     /////////////////////////////// -------------------------------------------------------------------

  std::cout << " ==== Edn of run " << std::endl;
}
/////////////////////////////// -------------------------------------------------------------------
//
/////////////////////////////// -------------------------------------------------------------------
//
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

double CMTRawAnalyzer::dR(double eta1, double phi1, double eta2, double phi2) {
  double deltaphi = phi1 - phi2;
  if (phi2 > phi1) {
    deltaphi = phi2 - phi1;
  }
  if (deltaphi > M_PI) {
    deltaphi = 2. * M_PI - deltaphi;
  }
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
  return tmp;
}

double CMTRawAnalyzer::phi12(double phi1, double en1, double phi2, double en2) {
  // weighted mean value of phi1 and phi2

  double a1 = phi1;
  double a2 = phi2;

  if (a1 > 0.5 * M_PI && a2 < 0.)
    a2 += 2 * M_PI;
  if (a2 > 0.5 * M_PI && a1 < 0.)
    a1 += 2 * M_PI;
  double tmp = (a1 * en1 + a2 * en2) / (en1 + en2);
  if (tmp > M_PI)
    tmp -= 2. * M_PI;

  return tmp;
}

double CMTRawAnalyzer::dPhiWsign(double phi1, double phi2) {
  // clockwise      phi2 w.r.t phi1 means "+" phi distance
  // anti-clockwise phi2 w.r.t phi1 means "-" phi distance

  double a1 = phi1;
  double a2 = phi2;
  double tmp = a2 - a1;
  if (a1 * a2 < 0.) {
    if (a1 > 0.5 * M_PI)
      tmp += 2. * M_PI;
    if (a2 > 0.5 * M_PI)
      tmp -= 2. * M_PI;
  }
  return tmp;
}

/////////////////////////////// -------------------------------------------------------------------
/////////////////////////////// -------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(CMTRawAnalyzer);
