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
//using namespace std;
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//using namespace edm;
// this line is to retrieve HCAL RecHitCollections:
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

/////////////////////////////////////////////////////////////////////////
//  SERVICE FUNCTIONS --------------------------------------------------------
/*
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
*/
/////////////////////////////////////////////////////////////////////////

#endif
