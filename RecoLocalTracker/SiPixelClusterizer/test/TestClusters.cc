// File: TestClusters.cc
// Description: T0 test the pixel clusters.
// Author: Danek Kotlinski
// Creation Date:  Initial version. 3/06
// Modify to work with CMSSW354, 11/03/10 d.k.
// Modify to work with CMSSW620, SLC6, CMSSW700, 10/10/13 d.k.
// Change to ByToken (clusters only)
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// For HLT
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// for resyncs
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

// For luminisoty
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrector.h"

// For PVs
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TProfile.h>

#define NEW_ID
#ifdef NEW_ID
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#else
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#endif

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

using namespace std;

#define HISTOS
//#define L1
//#define HLT
//#define HI
//#define ROC_EFF
//#define Lumi
//#define USE_RESYNCS  // get theinfo about recyncs

// special hitos with parameters normalized to intlumi (usually disabled)
//#define INSTLUMI_STUDIES
//#define VDM_STUDIES

//#define BX
#define BX_NEW

//=======================================================================
#ifdef BX_NEW

class getBX {
public:
  getBX(void);
  ~getBX(void);
  static int find(int bx);

private:
};

getBX::getBX(void) { edm::LogPrint("TestClusters") << " bx ctor "; }

getBX::~getBX(void) {}

int getBX::find(int bx) {
  // invalid -1, empty 0, beam1 1, beam2 2, collision 3, collison+1 4, beam1+1 5,  beam2+1 6

  const int limit = 3394;

  // For the 48-35 fill used for VdM
  //   const int coll_num = 35;
  //   const int coll[35] = {
  //        1,41,81,121,161,201,241,721,761,801,841,881,921,961,1581,1621,1661,1701,1741,1781,1821,2161,2201,2241,2281,
  //        2321,2361,2401,2881,2921,2961,3001,3041,3081,3121
  //   };
  //   const int beam1_num=13;
  //   const int beam1[13] = {
  //     301,341,381,541,581,621,661,1335,1375,1415,1455,1495,1535
  //   };
  //   const int beam2_num=13;
  //   const int beam2[13] = {
  //     441,481,521,561,601,641,1192,1232,1272,2001,2041,2081,2121
  //   };

  // For the usuall 1375-1368 50ns fill
  const int coll_num = 1368;
  const int coll[1368] = {
      66,   68,   70,   72,   74,   76,   78,   80,   82,   84,   86,   88,   90,   92,   94,   96,   98,   100,  102,
      104,  106,  108,  110,  112,  114,  116,  118,  120,  122,  124,  126,  128,  130,  132,  134,  136,  146,  148,
      150,  152,  154,  156,  158,  160,  162,  164,  166,  168,  170,  172,  174,  176,  178,  180,  182,  184,  186,
      188,  190,  192,  194,  196,  198,  200,  202,  204,  206,  208,  210,  212,  214,  216,  226,  228,  230,  232,
      234,  236,  238,  240,  242,  244,  246,  248,  250,  252,  254,  256,  258,  260,  262,  264,  266,  268,  270,
      272,  274,  276,  278,  280,  282,  284,  286,  288,  290,  292,  294,  296,  306,  308,  310,  312,  314,  316,
      318,  320,  322,  324,  326,  328,  330,  332,  334,  336,  338,  340,  342,  344,  346,  348,  350,  352,  354,
      356,  358,  360,  362,  364,  366,  368,  370,  372,  374,  376,  413,  415,  417,  419,  421,  423,  425,  427,
      429,  431,  433,  435,  437,  439,  441,  443,  445,  447,  449,  451,  453,  455,  457,  459,  461,  463,  465,
      467,  469,  471,  473,  475,  477,  479,  481,  483,  493,  495,  497,  499,  501,  503,  505,  507,  509,  511,
      513,  515,  517,  519,  521,  523,  525,  527,  529,  531,  533,  535,  537,  539,  541,  543,  545,  547,  549,
      551,  553,  555,  557,  559,  561,  563,  573,  575,  577,  579,  581,  583,  585,  587,  589,  591,  593,  595,
      597,  599,  601,  603,  605,  607,  609,  611,  613,  615,  617,  619,  621,  623,  625,  627,  629,  631,  633,
      635,  637,  639,  641,  643,  653,  655,  657,  659,  661,  663,  665,  667,  669,  671,  673,  675,  677,  679,
      681,  683,  685,  687,  689,  691,  693,  695,  697,  699,  701,  703,  705,  707,  709,  711,  713,  715,  717,
      719,  721,  723,  773,  775,  777,  779,  781,  783,  785,  787,  789,  791,  793,  795,  797,  799,  801,  803,
      805,  807,  809,  811,  813,  815,  817,  819,  821,  823,  825,  827,  829,  831,  833,  835,  837,  839,  841,
      843,  853,  855,  857,  859,  861,  863,  865,  867,  869,  871,  873,  875,  877,  879,  881,  883,  885,  887,
      889,  891,  893,  895,  897,  899,  901,  903,  905,  907,  909,  911,  913,  915,  917,  919,  921,  923,  960,
      962,  964,  966,  968,  970,  972,  974,  976,  978,  980,  982,  984,  986,  988,  990,  992,  994,  996,  998,
      1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018, 1020, 1022, 1024, 1026, 1028, 1030, 1040, 1042, 1044,
      1046, 1048, 1050, 1052, 1054, 1056, 1058, 1060, 1062, 1064, 1066, 1068, 1070, 1072, 1074, 1076, 1078, 1080, 1082,
      1084, 1086, 1088, 1090, 1092, 1094, 1096, 1098, 1100, 1102, 1104, 1106, 1108, 1110, 1120, 1122, 1124, 1126, 1128,
      1130, 1132, 1134, 1136, 1138, 1140, 1142, 1144, 1146, 1148, 1150, 1152, 1154, 1156, 1158, 1160, 1162, 1164, 1166,
      1168, 1170, 1172, 1174, 1176, 1178, 1180, 1182, 1184, 1186, 1188, 1190, 1200, 1202, 1204, 1206, 1208, 1210, 1212,
      1214, 1216, 1218, 1220, 1222, 1224, 1226, 1228, 1230, 1232, 1234, 1236, 1238, 1240, 1242, 1244, 1246, 1248, 1250,
      1252, 1254, 1256, 1258, 1260, 1262, 1264, 1266, 1268, 1270, 1307, 1309, 1311, 1313, 1315, 1317, 1319, 1321, 1323,
      1325, 1327, 1329, 1331, 1333, 1335, 1337, 1339, 1341, 1343, 1345, 1347, 1349, 1351, 1353, 1355, 1357, 1359, 1361,
      1363, 1365, 1367, 1369, 1371, 1373, 1375, 1377, 1387, 1389, 1391, 1393, 1395, 1397, 1399, 1401, 1403, 1405, 1407,
      1409, 1411, 1413, 1415, 1417, 1419, 1421, 1423, 1425, 1427, 1429, 1431, 1433, 1435, 1437, 1439, 1441, 1443, 1445,
      1447, 1449, 1451, 1453, 1455, 1457, 1467, 1469, 1471, 1473, 1475, 1477, 1479, 1481, 1483, 1485, 1487, 1489, 1491,
      1493, 1495, 1497, 1499, 1501, 1503, 1505, 1507, 1509, 1511, 1513, 1515, 1517, 1519, 1521, 1523, 1525, 1527, 1529,
      1531, 1533, 1535, 1537, 1547, 1549, 1551, 1553, 1555, 1557, 1559, 1561, 1563, 1565, 1567, 1569, 1571, 1573, 1575,
      1577, 1579, 1581, 1583, 1585, 1587, 1589, 1591, 1593, 1595, 1597, 1599, 1601, 1603, 1605, 1607, 1609, 1611, 1613,
      1615, 1617, 1667, 1669, 1671, 1673, 1675, 1677, 1679, 1681, 1683, 1685, 1687, 1689, 1691, 1693, 1695, 1697, 1699,
      1701, 1703, 1705, 1707, 1709, 1711, 1713, 1715, 1717, 1719, 1721, 1723, 1725, 1727, 1729, 1731, 1733, 1735, 1737,
      1747, 1749, 1751, 1753, 1755, 1757, 1759, 1761, 1763, 1765, 1767, 1769, 1771, 1773, 1775, 1777, 1779, 1781, 1783,
      1785, 1787, 1789, 1791, 1793, 1795, 1797, 1799, 1801, 1803, 1805, 1807, 1809, 1811, 1813, 1815, 1817, 1854, 1856,
      1858, 1860, 1862, 1864, 1866, 1868, 1870, 1872, 1874, 1876, 1878, 1880, 1882, 1884, 1886, 1888, 1890, 1892, 1894,
      1896, 1898, 1900, 1902, 1904, 1906, 1908, 1910, 1912, 1914, 1916, 1918, 1920, 1922, 1924, 1934, 1936, 1938, 1940,
      1942, 1944, 1946, 1948, 1950, 1952, 1954, 1956, 1958, 1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978,
      1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2014, 2016, 2018, 2020, 2022, 2024,
      2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040, 2042, 2044, 2046, 2048, 2050, 2052, 2054, 2056, 2058, 2060, 2062,
      2064, 2066, 2068, 2070, 2072, 2074, 2076, 2078, 2080, 2082, 2084, 2094, 2096, 2098, 2100, 2102, 2104, 2106, 2108,
      2110, 2112, 2114, 2116, 2118, 2120, 2122, 2124, 2126, 2128, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146,
      2148, 2150, 2152, 2154, 2156, 2158, 2160, 2162, 2164, 2201, 2203, 2205, 2207, 2209, 2211, 2213, 2215, 2217, 2219,
      2221, 2223, 2225, 2227, 2229, 2231, 2233, 2235, 2237, 2239, 2241, 2243, 2245, 2247, 2249, 2251, 2253, 2255, 2257,
      2259, 2261, 2263, 2265, 2267, 2269, 2271, 2281, 2283, 2285, 2287, 2289, 2291, 2293, 2295, 2297, 2299, 2301, 2303,
      2305, 2307, 2309, 2311, 2313, 2315, 2317, 2319, 2321, 2323, 2325, 2327, 2329, 2331, 2333, 2335, 2337, 2339, 2341,
      2343, 2345, 2347, 2349, 2351, 2361, 2363, 2365, 2367, 2369, 2371, 2373, 2375, 2377, 2379, 2381, 2383, 2385, 2387,
      2389, 2391, 2393, 2395, 2397, 2399, 2401, 2403, 2405, 2407, 2409, 2411, 2413, 2415, 2417, 2419, 2421, 2423, 2425,
      2427, 2429, 2431, 2441, 2443, 2445, 2447, 2449, 2451, 2453, 2455, 2457, 2459, 2461, 2463, 2465, 2467, 2469, 2471,
      2473, 2475, 2477, 2479, 2481, 2483, 2485, 2487, 2489, 2491, 2493, 2495, 2497, 2499, 2501, 2503, 2505, 2507, 2509,
      2511, 2549, 2551, 2553, 2555, 2557, 2559, 2561, 2563, 2565, 2567, 2569, 2571, 2573, 2575, 2577, 2579, 2581, 2583,
      2585, 2587, 2589, 2591, 2593, 2595, 2597, 2599, 2601, 2603, 2605, 2607, 2609, 2611, 2613, 2615, 2617, 2619, 2629,
      2631, 2633, 2635, 2637, 2639, 2641, 2643, 2645, 2647, 2649, 2651, 2653, 2655, 2657, 2659, 2661, 2663, 2665, 2667,
      2669, 2671, 2673, 2675, 2677, 2679, 2681, 2683, 2685, 2687, 2689, 2691, 2693, 2695, 2697, 2699, 2736, 2738, 2740,
      2742, 2744, 2746, 2748, 2750, 2752, 2754, 2756, 2758, 2760, 2762, 2764, 2766, 2768, 2770, 2772, 2774, 2776, 2778,
      2780, 2782, 2784, 2786, 2788, 2790, 2792, 2794, 2796, 2798, 2800, 2802, 2804, 2806, 2816, 2818, 2820, 2822, 2824,
      2826, 2828, 2830, 2832, 2834, 2836, 2838, 2840, 2842, 2844, 2846, 2848, 2850, 2852, 2854, 2856, 2858, 2860, 2862,
      2864, 2866, 2868, 2870, 2872, 2874, 2876, 2878, 2880, 2882, 2884, 2886, 2896, 2898, 2900, 2902, 2904, 2906, 2908,
      2910, 2912, 2914, 2916, 2918, 2920, 2922, 2924, 2926, 2928, 2930, 2932, 2934, 2936, 2938, 2940, 2942, 2944, 2946,
      2948, 2950, 2952, 2954, 2956, 2958, 2960, 2962, 2964, 2966, 2976, 2978, 2980, 2982, 2984, 2986, 2988, 2990, 2992,
      2994, 2996, 2998, 3000, 3002, 3004, 3006, 3008, 3010, 3012, 3014, 3016, 3018, 3020, 3022, 3024, 3026, 3028, 3030,
      3032, 3034, 3036, 3038, 3040, 3042, 3044, 3046, 3083, 3085, 3087, 3089, 3091, 3093, 3095, 3097, 3099, 3101, 3103,
      3105, 3107, 3109, 3111, 3113, 3115, 3117, 3119, 3121, 3123, 3125, 3127, 3129, 3131, 3133, 3135, 3137, 3139, 3141,
      3143, 3145, 3147, 3149, 3151, 3153, 3163, 3165, 3167, 3169, 3171, 3173, 3175, 3177, 3179, 3181, 3183, 3185, 3187,
      3189, 3191, 3193, 3195, 3197, 3199, 3201, 3203, 3205, 3207, 3209, 3211, 3213, 3215, 3217, 3219, 3221, 3223, 3225,
      3227, 3229, 3231, 3233, 3243, 3245, 3247, 3249, 3251, 3253, 3255, 3257, 3259, 3261, 3263, 3265, 3267, 3269, 3271,
      3273, 3275, 3277, 3279, 3281, 3283, 3285, 3287, 3289, 3291, 3293, 3295, 3297, 3299, 3301, 3303, 3305, 3307, 3309,
      3311, 3313, 3323, 3325, 3327, 3329, 3331, 3333, 3335, 3337, 3339, 3341, 3343, 3345, 3347, 3349, 3351, 3353, 3355,
      3357, 3359, 3361, 3363, 3365, 3367, 3369, 3371, 3373, 3375, 3377, 3379, 3381, 3383, 3385, 3387, 3389, 3391, 3393};
  const int beam1_num = 6;
  const int beam1[6] = {1, 3, 5, 7, 9, 11};
  const int beam2_num = 6;
  const int beam2[6] = {
      13,
      15,
      17,
      19,
      21,
      23,
  };

  if (bx > limit)
    return (-1);
  //edm::LogPrint("TestClusters")<<bx<<endl;

  // Check collisions
  for (int i = 0; i < coll_num; ++i) {
    if (bx == coll[i])
      return (3);  // collision
    else if (bx == (coll[i] + 1))
      return (4);  // collision+1
  }

  // Check beam1
  for (int i = 0; i < beam1_num; ++i) {
    if (bx == beam1[i])
      return (1);  // collision
    else if (bx == (beam1[i] + 1))
      return (5);  // collision+1
  }

  // Check beam1
  for (int i = 0; i < beam2_num; ++i) {
    if (bx == beam2[i])
      return (2);  // collision
    else if (bx == (beam2[i] + 1))
      return (6);  // collision+1
  }

  return 0;  // return no collision
}

#endif  // BX_NEW

//=======================================================================
#ifdef BX
class getBX {
public:
  getBX(void);
  ~getBX(void);
  int find(int bx);

private:
  int limit;
  int trains;
  int train_start[40];
  int train_stop[40];
  int beam1_start[40];
  int beam1_stop[40];
  int beam2_start[40];
  int beam2_stop[40];
};

getBX::getBX(void) {
  limit = 1841;
  trains = 22;
  int count = 0;

  // train 1
  train_start[count] = 7;
  train_stop[count] = 24;
  beam1_start[count] = 1;
  beam1_stop[count] = 5;
  beam2_start[count] = 25;
  beam2_stop[count] = 29;

  // train 2
  count++;
  train_start[count] = 66;
  train_stop[count] = 137;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 3
  count++;
  train_start[count] = 146;
  train_stop[count] = 217;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 4
  count++;
  train_start[count] = 226;
  train_stop[count] = 297;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 5
  count++;
  train_start[count] = 306;
  train_stop[count] = 377;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 6
  count++;
  train_start[count] = 413;
  train_stop[count] = 484;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 7
  count++;
  train_start[count] = 493;
  train_stop[count] = 564;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 8
  count++;
  train_start[count] = 573;
  train_stop[count] = 644;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 9
  count++;
  train_start[count] = 653;
  train_stop[count] = 724;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 10
  count++;
  train_start[count] = 773;
  train_stop[count] = 844;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 11
  count++;
  train_start[count] = 853;
  train_stop[count] = 924;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 12
  count++;
  train_start[count] = 960;
  train_stop[count] = 1031;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 13
  count++;
  train_start[count] = 1040;
  train_stop[count] = 1111;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 14
  count++;
  train_start[count] = 1120;
  train_stop[count] = 1191;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 15
  count++;
  train_start[count] = 1200;
  train_stop[count] = 1271;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 16
  count++;
  train_start[count] = 1307;
  train_stop[count] = 1378;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 17
  count++;
  train_start[count] = 1387;
  train_stop[count] = 1458;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 18
  count++;
  train_start[count] = 1467;
  train_stop[count] = 1538;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 19
  count++;
  train_start[count] = 1547;
  train_stop[count] = 1618;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 20
  count++;
  train_start[count] = 1667;
  train_stop[count] = 1726;
  beam1_start[count] = 1727;
  beam1_stop[count] = 1734;
  beam2_start[count] = 1655;
  beam2_stop[count] = 1666;

  // train 21
  count++;
  train_start[count] = 1735;
  train_stop[count] = 1738;
  beam1_start[count] = -1;
  beam1_stop[count] = -1;
  beam2_start[count] = -1;
  beam2_stop[count] = -1;

  // train 22
  count++;
  train_start[count] = 1747;
  train_stop[count] = 1806;
  beam1_start[count] = 1807;
  beam1_stop[count] = 1818;
  beam2_start[count] = 1739;
  beam2_stop[count] = 1745;

  edm::LogPrint("TestClusters") << " number of trains " << trains << " " << limit;
}

getBX::~getBX(void) {}

int getBX::find(int bx) {
  // invalid -1, empty 0. beam1 1, beam2 2, collision 3, collison+1 4, beam1+1 5,  beam2+1 6

  if (bx > limit)
    return (-1);
  //edm::LogPrint("TestClusters")<<bx<<endl;

  // Check collisions
  for (int train = 0; train < trains; ++train) {
    //edm::LogPrint("TestClusters")<<bx<<" "<<train<<endl;

    for (int b = train_start[train]; b <= train_stop[train]; b += 2) {
      //edm::LogPrint("TestClusters")<<bx<<" "<<train<<" "<<b<<endl;
      if (bx == b)
        return (3);  // collision
      else if (bx == (b + 1))
        return (4);  // collision+1
    }

    if (beam1_start[train] != -1) {
      for (int b = beam1_start[train]; b <= beam1_stop[train]; b += 2) {
        //edm::LogPrint("TestClusters")<<bx<<" "<<train<<" "<<b<<endl;
        if (bx == b)
          return (1);  // beam1
        else if (bx == (b + 1))
          return (5);  // beam1+1
      }
    }

    if (beam2_start[train] != -1) {
      for (int b = beam2_start[train]; b <= beam2_stop[train]; b += 2) {
        //edm::LogPrint("TestClusters")<<bx<<" "<<train<<" "<<b<<endl;
        if (bx == b)
          return (2);  // beam1
        else if (bx == (b + 1))
          return (6);  // beam1+1
      }
    }

  }  // loop trains

  //edm::LogPrint("TestClusters")<<bx<<endl;

  return 0;  // return no collision
}

#endif
//=========================================================================

//=======================================================================

// decode a simplified ROC address
int rocId(int col, int row) {
  int rocRow = row / 80;
  int rocCol = col / 52;
  int rocId = rocCol + rocRow * 8;
  return rocId;
}

//=========================================================================
#ifdef ROC_EFF

class rocEfficiency {
public:
  rocEfficiency(void);
  ~rocEfficiency(void);
  void addPixel(int layer, int ladder, int module, int roc);
  float getRoc(int layer, int ladder, int module, int roc);
  float getModule(int layer, int ladder, int module, float &half1, float &half2);
  //int analyzeModule(int layer, int ladder, int module);

  // return array index
  inline int transformLadder1(int ladder) {
    if (ladder > 10 || ladder < -10 || ladder == 0) {
      edm::LogPrint("TestClusters") << " wrong ladder-1 id " << ladder << edm::LogPrint("TestClusters");
      return (-1);
    }
    if (ladder < 0)
      return (ladder + 10);
    else
      return (ladder + 10 - 1);
  };
  inline int transformLadder2(int ladder) {
    if (ladder > 16 || ladder < -16 || ladder == 0) {
      edm::LogPrint("TestClusters") << " wrong ladder-2 id " << ladder << edm::LogPrint("TestClusters");
      return (-1);
    }
    if (ladder < 0)
      return (ladder + 16);
    else
      return (ladder + 16 - 1);
  };
  inline int transformLadder3(int ladder) {
    if (ladder > 22 || ladder < -22 || ladder == 0) {
      edm::LogPrint("TestClusters") << " wrong ladder-3 id " << ladder << edm::LogPrint("TestClusters");
      return (-1);
    }
    if (ladder < 0)
      return (ladder + 22);
    else
      return (ladder + 22 - 1);
  };
  inline int transformModule(int module) {
    if (module > 4 || module < -4 || module == 0) {
      edm::LogPrint("TestClusters") << " wrong module id " << module << edm::LogPrint("TestClusters");
      return (-1);
    }
    if (module < 0)
      return (module + 4);
    else
      return (module + 4 - 1);
  };

private:
  int pixelsLayer1[20][8][16];
  int pixelsLayer2[32][8][16];
  int pixelsLayer3[44][8][16];
};

rocEfficiency::rocEfficiency(void) {
  edm::LogPrint("TestClusters") << " clear";
  for (int lad1 = 0; lad1 < 20; ++lad1)
    for (int mod1 = 0; mod1 < 8; ++mod1)
      for (int roc1 = 0; roc1 < 16; ++roc1)
        pixelsLayer1[lad1][mod1][roc1] = 0;

  for (int lad2 = 0; lad2 < 32; ++lad2)
    for (int mod2 = 0; mod2 < 8; ++mod2)
      for (int roc2 = 0; roc2 < 16; ++roc2)
        pixelsLayer2[lad2][mod2][roc2] = 0;

  for (int lad3 = 0; lad3 < 44; ++lad3)
    for (int mod3 = 0; mod3 < 8; ++mod3)
      for (int roc3 = 0; roc3 < 16; ++roc3)
        pixelsLayer3[lad3][mod3][roc3] = 0;
}

rocEfficiency::~rocEfficiency(void) {
  //   edm::LogPrint("TestClusters")<<" clear"<<endl;
  //   for(int lad1=0;lad1<20;++lad1)
  //     for(int mod1=0;mod1<8;++mod1)
  //       for(int col1=0;col1<416;++col1)
  //      for(int row1=0;row1<160;++row1)
  //        if(pixelsLayer1[lad1][mod1][col1][row1]>0)
  //          edm::LogPrint("TestClusters")<<lad1<<" "<<mod1<<" "<<col1<<" "<<row1<<" "
  //              <<pixelsLayer1[lad1][mod1][col1][row1] <<endl;
}

void rocEfficiency::addPixel(int layer, int ladder, int module, int roc) {
  if (roc < 0 || roc >= 16) {
    edm::LogPrint("TestClusters") << " wrong roc number " << roc;
    return;
  }

  int module0 = transformModule(module);
  if (module0 < 0 || module0 >= 8) {
    edm::LogPrint("TestClusters") << " wrong module index " << module0 << " " << module;
    return;
  }

  int ladder0 = 0;
  switch (layer) {
    case 1:
      ladder0 = transformLadder1(ladder);
      //edm::LogPrint("TestClusters")<<layer<<" "<<ladder0<<" "<<module0<<" "<<col<<" "<<row<<endl;
      if (ladder0 < 0 || ladder0 >= 20)
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
      else
        pixelsLayer1[ladder0][module0][roc]++;
      break;
    case 2:
      ladder0 = transformLadder2(ladder);
      if (ladder0 < 0 || ladder0 >= 32)
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
      else
        pixelsLayer2[ladder0][module0][roc]++;
      break;
    case 3:
      ladder0 = transformLadder3(ladder);
      if (ladder0 < 0 || ladder0 >= 44)
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
      else
        pixelsLayer3[ladder0][module0][roc]++;
      break;
    default:
      break;
  }
}

float rocEfficiency::getRoc(int layer, int ladder, int module, int roc) {
  if (roc < 0 || roc >= 16) {
    edm::LogPrint("TestClusters") << " wrong roc number " << roc;
    return -1.;
  }
  int module0 = transformModule(module);
  if (module0 < 0 || module0 >= 8) {
    edm::LogPrint("TestClusters") << " wrong module index " << module0 << " " << module;
    return -1.;
  }

  float count = 0.;
  int ladder0 = 0;

  switch (layer) {
    case 1:
      if ((abs(ladder) == 1 || abs(ladder) == 10) && roc > 7) {  // find half modules
        count = -1.;                                             // invalid
      } else {
        ladder0 = transformLadder1(ladder);
        if (ladder0 < 0 || ladder0 >= 20) {
          edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
          count = -1.;
        } else
          count = float(pixelsLayer1[ladder0][module0][roc]);
        //if(ladder0==10 && module0==5 && col==9 && row==0) edm::LogPrint("TestClusters")<<count<<endl;
      }
      break;
    case 2:
      if ((abs(ladder) == 1 || abs(ladder) == 16) && roc > 7) {  // find half modules
        count = -1.;                                             // invalid
      } else {
        ladder0 = transformLadder2(ladder);
        if (ladder0 < 0 || ladder0 >= 32) {
          edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
          count = -1.;
        } else
          count = float(pixelsLayer2[ladder0][module0][roc]);
      }
      break;
    case 3:
      if ((abs(ladder) == 1 || abs(ladder) == 22) && roc > 7) {  // find half modules
        count = -1;                                              // invalid
      } else {
        ladder0 = transformLadder3(ladder);
        if (ladder0 < 0 || ladder0 >= 44) {
          edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
          count = -1.;
        } else
          count = float(pixelsLayer3[ladder0][module0][roc]);
      }
      break;
    default:
      break;
  }
  return count;
}

// Return counts averaged per ROC (count per module / number of full ROCs )
float rocEfficiency::getModule(int layer, int ladder, int module, float &half1, float &half2) {
  half1 = 0;
  half2 = 0;
  int module0 = transformModule(module);
  if (module0 < 0 || module0 >= 8) {
    edm::LogPrint("TestClusters") << " wrong module index " << module0 << " " << module;
    return -1.;
  }

  float count = 0;
  int rocs = 0;
  int ladder0 = 0;

  switch (layer) {
    case 1:
      ladder0 = transformLadder1(ladder);
      if (ladder0 < 0 || ladder0 >= 20) {
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
        return -1.;
      }
      for (int roc = 0; roc < 16; ++roc) {
        //if( !((abs(ladder)==1 || abs(ladder)==10) && roc>7) ) // treat half-modules
        float tmp = float(pixelsLayer1[ladder0][module0][roc]);
        count += tmp;
        if (roc < 8)
          half1 += tmp;
        else
          half2 += tmp;
        if (tmp > 0)
          rocs++;
      }
      break;

    case 2:
      ladder0 = transformLadder2(ladder);
      if (ladder0 < 0 || ladder0 >= 32) {
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
        return -1.;
      }
      for (int roc = 0; roc < 16; ++roc) {
        //if( !((abs(ladder)==1 || abs(ladder)==22) && roc>7) ) // treat half-modules
        float tmp = float(pixelsLayer2[ladder0][module0][roc]);
        count += tmp;
        if (roc < 8)
          half1 += tmp;
        else
          half2 += tmp;
        if (tmp > 0)
          rocs++;
      }
      break;

    case 3:
      ladder0 = transformLadder3(ladder);
      if (ladder0 < 0 || ladder0 >= 44) {
        edm::LogPrint("TestClusters") << " wrong ladder index " << ladder0 << " " << ladder;
        return -1.;
      }
      for (int roc = 0; roc < 16; ++roc) {
        //if( !((abs(ladder)==1 || abs(ladder)==22) && roc>7) ) // treat half-modules
        float tmp = float(pixelsLayer3[ladder0][module0][roc]);
        count += tmp;
        if (roc < 8)
          half1 += tmp;
        else
          half2 += tmp;
        if (tmp > 0)
          rocs++;
      }
      break;

    default:
      break;
  }
  //edm::LogPrint("TestClusters")<<count<<" "<<rocs<<" "<<ladder<<" "<<module<<" "<<ladder0<<" "<<module0<<endl;
  if (rocs > 0)
    count = count / float(rocs);  // return counts per ROC (full rocs only)
  if (count < 0)
    edm::LogPrint("TestClusters") << " VERY VERY WRONG " << count;
  return count;
}

#endif

//==========================================================================================

class TestClusters : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TestClusters(const edm::ParameterSet &conf);
  virtual ~TestClusters();
  virtual void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  virtual void beginJob() override;
  virtual void endJob() override;

private:
  //const static bool PRINT = false;
  bool PRINT;
  int select1, select2;
  int countEvents, countAllEvents;
  double sumClusters, sumPixels, countLumi;
  //float rocHits[3][10]; // layer, ids, store hits in 30 rocs
  //float moduleHits[3][10]; // layer, ids, store hits in 30 modules

  // Needed for the ByToken method
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > myClus;
  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;
  edm::EDGetTokenT<LumiDetails> lumiDetailsToken_;
  edm::EDGetTokenT<edm::ConditionsInLumiBlock> condToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> l1gtrrToken_;
  edm::EDGetTokenT<Level1TriggerScalersCollection> l1tsToken_;
  edm::EDGetTokenT<edm::TriggerResults> hltToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;

  //TFile* hFile;
  TH1D *hdetunit;
  TH1D *hpixid, *hpixsubid, *hlayerid, *hladder1id, *hladder2id, *hladder3id, *hz1id, *hz2id, *hz3id;

  TH1D *hcharge1, *hcharge2, *hcharge3, *hcharge4, *hcharge5;
  //TH1D *hcharge11,*hcharge12, *hcharge13, *hcharge14;
  TH1D *hpixcharge1, *hpixcharge2, *hpixcharge3, *hpixcharge4, *hpixcharge5;
  //TH1D *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1D *hpcols1, *hpcols2, *hpcols3, *hprows1, *hprows2, *hprows3;
  TH1D *hsize1, *hsize2, *hsize3, *hsizex1, *hsizex2, *hsizex3, *hsizey1, *hsizey2, *hsizey3;
  //TH1D *havCharge1,*havCharge2,*havCharge3,*havCharge4,*havCharge5,*havCharge6;
  //TH1D *hchargen1,*hsizen1, *hsizexn1, *hsizeyn1, *hpixchargen1;
  //TH1D *hchargen2,*hsizen2, *hsizexn2, *hsizeyn2, *hpixchargen2;
  //TH1D *hchargen3,*hchargen4;
  //TH2D *hchargen5;

#if defined(BX) || defined(BX_NEW)
  TH1D *hchargebx1, *hchargebx2, *hchargebx3, *hchargebx4, *hchargebx5, *hchargebx6;
  TH1D *hpixChargebx1, *hpixChargebx2, *hpixChargebx3, *hpixChargebx4, *hpixChargebx5, *hpixChargebx6;
  TH1D *hsizebx1, *hsizebx2, *hsizebx3, *hsizebx4, *hsizebx5, *hsizebx6;

#endif

  TH1D *hclusPerDet1, *hclusPerDet2, *hclusPerDet3;
  TH1D *hpixPerDet1, *hpixPerDet2, *hpixPerDet3;
  TH1D *hpixPerLink1, *hpixPerLink2, *hpixPerLink3;
  TH1D *hclusPerLay1, *hclusPerLay2, *hclusPerLay3;
  TH1D *hpixPerLay1, *hpixPerLay2, *hpixPerLay3;
  TH1D *hdetsPerLay1, *hdetsPerLay2, *hdetsPerLay3;
  TH1D *hclus, *hclusBPix, *hclusFPix, *hdigis, *hdigis2, *hdigisB, *hdigisF;
  //TH1D *hdigis30,*hdigis31,*hdigis32,*hdigis33,*hdigis34,*hdigis35,*hdigis36,*hdigis37,*hdigis38,
  //*hdigis39,*hdigis7,*hdigis18,*hdigis25,*hdigis26;
  TH1D *hclus2, *hclus5;  // *hclus6,*hclus7,*hclus8,*hclus9;
  //*hclus10,*hclus11,*hclus12,*hclus13,*hclus14,*hclus15,*hclus16,*hclus17,*hclus18,*hclus19,
  TH1D *hdets, *hdets2;

  TH1D *hpixDiskR1, *hpixDiskR2;  // *hdetr, *hdetz;

  TH2F *hDetsMap1, *hDetsMap2, *hDetsMap3;  // all modules
  TH2F *hDetMap1, *hDetMap2, *hDetMap3;     // all modules
  TH2F *hpDetMap1, *hpDetMap2, *hpDetMap3;  // all modules
  TH2F *hsizeDetMap1, *hsizeDetMap2, *hsizeDetMap3;
  //TH2F *hsizeXDetMap1, *hsizeXDetMap2, *hsizeXDetMap3;
  //TH2F *hsizeYDetMap1, *hsizeYDetMap2, *hsizeYDetMap3;

  TH2F *hpixDetMap1, *hpixDetMap2, *hpixDetMap3;  // in a  modules
  TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;  // in a  modules

  //TH1D *hncharge1,*hncharge2, *hncharge3;
  //TH1D *hnpixcharge1,*hnpixcharge2,*hnpixcharge3;

  TH2F *hpixDetMap10, *hpixDetMap20, *hpixDetMap30;
  TH2F *hpixDetMap11, *hpixDetMap12, *hpixDetMap13, *hpixDetMap14, *hpixDetMap15;
  TH2F *hpixDetMap21, *hpixDetMap22, *hpixDetMap23, *hpixDetMap24, *hpixDetMap25;
  TH2F *hpixDetMap31, *hpixDetMap32, *hpixDetMap33, *hpixDetMap34, *hpixDetMap35;
  TH2F *hpixDetMap16, *hpixDetMap17, *hpixDetMap18, *hpixDetMap19;
  TH2F *hpixDetMap26, *hpixDetMap27, *hpixDetMap28, *hpixDetMap29;
  TH2F *hpixDetMap36, *hpixDetMap37, *hpixDetMap38, *hpixDetMap39;

  //TH2F *h2d1, *h2d2, *h2d3;

  TH1D *hevent, *hlumi, *hlumi0, *hlumi1, *hlumi10, *hlumi11, *hlumi12, *hlumi13,
      *hlumi14;  //  *hlumi15, *hlumi16, *hlumi17, *hlumi18, *hlumi19;
  TH1D *hbx, *hbx0, *hbx1, *hbx2, *hbx3, *hbx4, *hbx5, *hbx6, *hbx7;  // *hbx8,*hbx9,*hbx10; // *hbx11,*hbx12;
  TH1D *hmaxPixPerDet;

  TH1D *hclusPerDisk1, *hclusPerDisk2, *hclusPerDisk3, *hclusPerDisk4;
  TH1D *hpixPerDisk1, *hpixPerDisk2, *hpixPerDisk3, *hpixPerDisk4;
  TH1D *hl1a, *hl1t, *hltt;
  TH1D *hl1a1, *hl1t1;
  TH1D *hlt1, *hlt2, *hlt3;

  TH1D *hpixPerDet11, *hpixPerDet12, *hpixPerDet13, *hpixPerDet14;
  TH1D *hpixPerDet21, *hpixPerDet22, *hpixPerDet23, *hpixPerDet24;
  TH1D *hpixPerDet31, *hpixPerDet32, *hpixPerDet33, *hpixPerDet34;
  //TH1D *hpixPerDet100,*hpixPerDet101,*hpixPerDet102,*hpixPerDet103,*hpixPerDet104,*hpixPerDet105;

  TH1D *hgz1, *hgz2, *hgz3;
  //TH2F *hclusls2, *hpixls2, *h2clupv, *h2pixpv;
  TH1D *hintg, *hinst, *hpvs;

#ifdef USE_RESYNCS
  TH1D *htest1, *htest2, *htest3, *htest4, *htest5, *htest6;
#endif

  TProfile *hclumult1, *hclumult2, *hclumult3;
  TProfile *hclumultx1, *hclumultx2, *hclumultx3;
  TProfile *hclumulty1, *hclumulty2, *hclumulty3;
  TProfile *hcluchar1, *hcluchar2, *hcluchar3;
  TProfile *hpixchar1, *hpixchar2, *hpixchar3;
  TProfile *hclusls, *hpixls, *hinstls, *hinstlsbx;
  TProfile *hpvls, *hclupv, *hpixpv;  //  *hclupvlsn, *hpixpvlsn;
  TProfile *hcharCluls, *hcharPixls, *hsizeCluls, *hsizeXCluls;
  TProfile *hpixbx, *hclubx, *hpvbx, *hcharClubx, *hcharPixbx, *hsizeClubx, *hsizeYClubx;
  TProfile *hinstbx;
  TProfile *hcharCluLumi, *hcharPixLumi, *hsizeCluLumi, *hsizeXCluLumi, *hsizeYCluLumi;

  TProfile *hcluLumi, *hpixLumi;

  //TProfile *hclus13ls,*hclus23ls,*hclus12ls,*hclusf3ls;
  //TProfile *hclusbx1lsn,*hclusbx2lsn,*hclusbx3lsn,*hclusbx4lsn,*hclusbx5lsn,*hclusbx6lsn,*hclusbx7lsn,
  //*hclusbx8lsn,*hclusbx9lsn;
  //TProfile *hclusbx11lsn,*hclusbx12lsn,*hclusbx13lsn,*hclusbx14lsn,*hclusbx15lsn,*hclusbx16lsn,*hclusbx17lsn,
  //*hclusbx18lsn,*hclusbx19lsn;

#ifdef INSTLUMI_STUDIES
  TProfile *hcluslsn, *hpixlsn;
  TProfile *hpixbl, *hpixb1l, *hpixb2l, *hpixb3l, *hpixfl, *hpixfml, *hpixfpl;
  TProfile *hclusbl, *hclusb1l, *hclusb2l, *hclusb3l, *hclusfl, *hclusfml, *hclusfpl;
  TProfile *hpixbxn, *hclubxn, *hpvbxn;
#endif

#ifdef VDM_STUDIES
  TProfile *hcharCluls1, *hcharPixls1, *hsizeCluls1, *hsizeXCluls1;
  TProfile *hcharCluls2, *hcharPixls2, *hsizeCluls2, *hsizeXCluls2;
  TProfile *hcharCluls3, *hcharPixls3, *hsizeCluls3, *hsizeXCluls3;
  TProfile *hclusls1, *hpixls1, *hclusls2, *hpixls2, *hclusls3, *hpixls3;
#endif

#ifdef ROC_EFF
  rocEfficiency *pixEff;

  TH2F *hbadMap1, *hbadMap2, *hbadMap3;  // modules with bad rocs
  TH2F *hrocHits1ls, *hrocHits2ls, *hrocHits3ls, *hmoduleHits1ls, *hmoduleHits2ls, *hmoduleHits3ls;
  TH1D *hcountInRoc1, *hcountInRoc2, *hcountInRoc3, *hcountInRoc12, *hcountInRoc22, *hcountInRoc32;
#endif

#ifdef BX
  getBX *getbx;
#endif

  // To correct lumi
  LumiCorrector *lumiCorrector;
};

/////////////////////////////////////////////////////////////////
// Contructor, empty.
TestClusters::TestClusters(edm::ParameterSet const &conf) {
  usesResource(TFileService::kSharedResource);
  PRINT = conf.getUntrackedParameter<bool>("Verbosity", false);
  select1 = conf.getUntrackedParameter<int>("Select1", 0);
  select2 = conf.getUntrackedParameter<int>("Select2", 0);
  if (PRINT)
    edm::LogPrint("TestClusters") << " Construct ";

  // For the ByToken method
  myClus = consumes<edmNew::DetSetVector<SiPixelCluster> >(conf.getParameter<edm::InputTag>("src"));
  lumiSummaryToken_ = consumes<LumiSummary>(edm::InputTag("lumiProducer"));
  lumiDetailsToken_ = consumes<LumiDetails>(edm::InputTag("lumiProducer"));
  condToken_ = consumes<edm::ConditionsInLumiBlock>(edm::InputTag("conditionsInEdm"));
  l1gtrrToken_ = consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag("gtDigis"));
  l1tsToken_ = consumes<Level1TriggerScalersCollection>(edm::InputTag("scalersRawToDigi"));
  hltToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", "HLT"));
  vtxToken_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
}
// Virtual destructor needed.
TestClusters::~TestClusters() = default;

// ------------ method called at the begining   ------------
void TestClusters::beginJob() {
  edm::LogPrint("TestClusters") << "Initialize PixelClusterTest ";

  edm::Service<TFileService> fs;

  //=====================================================================

  hladder1id = fs->make<TH1D>("hladder1id", "Ladder L1 id", 23, -11.5, 11.5);
  hladder2id = fs->make<TH1D>("hladder2id", "Ladder L2 id", 35, -17.5, 17.5);
  hladder3id = fs->make<TH1D>("hladder3id", "Ladder L3 id", 47, -23.5, 23.5);
  hz1id = fs->make<TH1D>("hz1id", "Z-index id L1", 11, -5.5, 5.5);
  hz2id = fs->make<TH1D>("hz2id", "Z-index id L2", 11, -5.5, 5.5);
  hz3id = fs->make<TH1D>("hz3id", "Z-index id L3", 11, -5.5, 5.5);

  int sizeH = 200;
  float lowH = -0.5;
  float highH = 199.5;

  hclusPerDet1 = fs->make<TH1D>("hclusPerDet1", "Clus per det l1", sizeH, lowH, highH);
  hclusPerDet2 = fs->make<TH1D>("hclusPerDet2", "Clus per det l2", sizeH, lowH, highH);
  hclusPerDet3 = fs->make<TH1D>("hclusPerDet3", "Clus per det l3", sizeH, lowH, highH);

  sizeH = 1000;
  highH = 1999.5;
  hpixPerDet1 = fs->make<TH1D>("hpixPerDet1", "Pix per det l1", sizeH, lowH, highH);
  hpixPerDet2 = fs->make<TH1D>("hpixPerDet2", "Pix per det l2", sizeH, lowH, highH);
  hpixPerDet3 = fs->make<TH1D>("hpixPerDet3", "Pix per det l3", sizeH, lowH, highH);

  hpixPerDet11 = fs->make<TH1D>("hpixPerDet11", "Pix per det l1 - ring 1", sizeH, lowH, highH);
  hpixPerDet12 = fs->make<TH1D>("hpixPerDet12", "Pix per det l1 - ring 2", sizeH, lowH, highH);
  hpixPerDet13 = fs->make<TH1D>("hpixPerDet13", "Pix per det l1 - ring 3", sizeH, lowH, highH);
  hpixPerDet14 = fs->make<TH1D>("hpixPerDet14", "Pix per det l1 - ring 4", sizeH, lowH, highH);
  hpixPerDet21 = fs->make<TH1D>("hpixPerDet21", "Pix per det l2 - ring 1", sizeH, lowH, highH);
  hpixPerDet22 = fs->make<TH1D>("hpixPerDet22", "Pix per det l2 - ring 2", sizeH, lowH, highH);
  hpixPerDet23 = fs->make<TH1D>("hpixPerDet23", "Pix per det l2 - ring 3", sizeH, lowH, highH);
  hpixPerDet24 = fs->make<TH1D>("hpixPerDet24", "Pix per det l2 - ring 4", sizeH, lowH, highH);
  hpixPerDet31 = fs->make<TH1D>("hpixPerDet31", "Pix per det l3 - ring 1", sizeH, lowH, highH);
  hpixPerDet32 = fs->make<TH1D>("hpixPerDet32", "Pix per det l3 - ring 2", sizeH, lowH, highH);
  hpixPerDet33 = fs->make<TH1D>("hpixPerDet33", "Pix per det l3 - ring 3", sizeH, lowH, highH);
  hpixPerDet34 = fs->make<TH1D>("hpixPerDet34", "Pix per det l3 - ring 4", sizeH, lowH, highH);

  //   hpixPerDet100 = fs->make<TH1D>( "hpixPerDet100", "Pix per det",
  // 			    sizeH, lowH, highH);
  //   hpixPerDet101 = fs->make<TH1D>( "hpixPerDet101", "Pix per det",
  // 			    sizeH, lowH, highH);
  //   hpixPerDet102 = fs->make<TH1D>( "hpixPerDet102", "Pix per det",
  // 			    sizeH, lowH, highH);
  //   hpixPerDet103 = fs->make<TH1D>( "hpixPerDet103", "Pix per det",
  // 			    sizeH, lowH, highH);
  //   hpixPerDet104 = fs->make<TH1D>( "hpixPerDet104", "Pix per det",
  // 			    sizeH, lowH, highH);
  //   hpixPerDet105 = fs->make<TH1D>( "hpixPerDet105", "Pix per det",
  // 			    sizeH, lowH, highH);

  sizeH = 1000;
  highH = 999.5;
  hpixPerLink1 = fs->make<TH1D>("hpixPerLink1", "Pix per link l1", sizeH, lowH, highH);
  hpixPerLink2 = fs->make<TH1D>("hpixPerLink2", "Pix per link l2", sizeH, lowH, highH);
  hpixPerLink3 = fs->make<TH1D>("hpixPerLink3", "Pix per link l3", sizeH, lowH, highH);

  sizeH = 5000;
#ifdef HI
  highH = 19999.5;
#else
  highH = 4999.5;
#endif

  hclusPerLay1 = fs->make<TH1D>("hclusPerLay1", "Clus per layer l1", sizeH, lowH, highH);
  hclusPerLay2 = fs->make<TH1D>("hclusPerLay2", "Clus per layer l2", sizeH, lowH, highH);
  hclusPerLay3 = fs->make<TH1D>("hclusPerLay3", "Clus per layer l3", sizeH, lowH, highH);

  hclus = fs->make<TH1D>("hclus", "Clus per event", sizeH, lowH, 5. * highH);
  hclusBPix = fs->make<TH1D>("hclusBPix", "Bpix Clus per event", sizeH, lowH, 4. * highH);
  hclusFPix = fs->make<TH1D>("hclusFPix", "Fpix Clus per event", sizeH, lowH, 2. * highH);

#ifdef HI
  highH = 3999.5;
#else
  highH = 1999.5;
#endif

  hclusPerDisk1 = fs->make<TH1D>("hclusPerDisk1", "Clus per disk1", sizeH, lowH, highH);
  hclusPerDisk2 = fs->make<TH1D>("hclusPerDisk2", "Clus per disk2", sizeH, lowH, highH);
  hclusPerDisk3 = fs->make<TH1D>("hclusPerDisk3", "Clus per disk3", sizeH, lowH, highH);
  hclusPerDisk4 = fs->make<TH1D>("hclusPerDisk4", "Clus per disk4", sizeH, lowH, highH);

#ifdef HI
  highH = 9999.5;
#else
  highH = 3999.5;
#endif

  hpixPerDisk1 = fs->make<TH1D>("hpixPerDisk1", "Pix per disk1", sizeH, lowH, highH);
  hpixPerDisk2 = fs->make<TH1D>("hpixPerDisk2", "Pix per disk2", sizeH, lowH, highH);
  hpixPerDisk3 = fs->make<TH1D>("hpixPerDisk3", "Pix per disk3", sizeH, lowH, highH);
  hpixPerDisk4 = fs->make<TH1D>("hpixPerDisk4", "Pix per disk4", sizeH, lowH, highH);

  sizeH = 2000;
#ifdef HI
  highH = 99999.5;
#else
  highH = 19999.5;
#endif

  hpixPerLay1 = fs->make<TH1D>("hpixPerLay1", "Pix per layer l1", sizeH, lowH, highH);
  hpixPerLay2 = fs->make<TH1D>("hpixPerLay2", "Pix per layer l2", sizeH, lowH, highH);
  hpixPerLay3 = fs->make<TH1D>("hpixPerLay3", "Pix per layer l3", sizeH, lowH, highH);

  hdigis = fs->make<TH1D>("hdigis", "All Digis in clus per event", sizeH, lowH, 5. * highH);
  hdigisB = fs->make<TH1D>("hdigisB", "BPix Digis in clus per event", sizeH, lowH, 4. * highH);
  hdigisF = fs->make<TH1D>("hdigisF", "FPix Digis in clus per event", sizeH, lowH, 2. * highH);
  hdigis2 = fs->make<TH1D>("hdigis2", "Digis per event - zoomed", 1000, -0.5, 999.5);

  //#ifdef BX
  //   hdigis30 = fs->make<TH1D>("hdigis30","Digis",sizeH,lowH,4.*highH);
  //   hdigis31 = fs->make<TH1D>("hdigis31","Digis",sizeH,lowH,4.*highH);
  //   hdigis32 = fs->make<TH1D>("hdigis32","Digis",sizeH,lowH,4.*highH);
  //   hdigis33 = fs->make<TH1D>("hdigis33","Digis",sizeH,lowH,4.*highH);
  //   hdigis34 = fs->make<TH1D>("hdigis34","Digis",sizeH,lowH,4.*highH);
  //   hdigis35 = fs->make<TH1D>("hdigis35","Digis",sizeH,lowH,4.*highH);
  //   hdigis36 = fs->make<TH1D>("hdigis36","Digis",sizeH,lowH,4.*highH);
  //   hdigis37 = fs->make<TH1D>("hdigis37","Digis",sizeH,lowH,4.*highH);
  //   hdigis38 = fs->make<TH1D>("hdigis38","Digis",sizeH,lowH,4.*highH);
  //   hdigis39 = fs->make<TH1D>("hdigis39","Digis",sizeH,lowH,4.*highH);
  //   hdigis7 = fs->make<TH1D>("hdigis7","Digis",sizeH,lowH,4.*highH);
  //   hdigis18 = fs->make<TH1D>("hdigis18","Digis",sizeH,lowH,4.*highH);
  //   hdigis25 = fs->make<TH1D>("hdigis25","Digis",sizeH,lowH,4.*highH);
  //   hdigis26 = fs->make<TH1D>("hdigis26","Digis",sizeH,lowH,4.*highH);
  //#endif

  sizeH = 1000;
#ifdef HI
  highH = 19999.5;
#else
  highH = 4999.5;
#endif

  hclus2 = fs->make<TH1D>("hclus2", "Clus per event - zoomed", 1000, -0.5, 999.5);
  //hclus3 = fs->make<TH1D>( "hclus3", "Clus per event",sizeH, lowH, highH);
  //hclus4 = fs->make<TH1D>( "hclus4", "Clus per event",sizeH, lowH, highH);
  hclus5 = fs->make<TH1D>("hclus5", "Clus per event", sizeH, lowH, highH);
  //hclus6 = fs->make<TH1D>( "hclus6", "Clus per event",sizeH, lowH, highH);
  //hclus7 = fs->make<TH1D>( "hclus7", "Clus per event",sizeH, lowH, highH);
  //hclus8 = fs->make<TH1D>( "hclus8", "Clus per event",sizeH, lowH, highH);
  //hclus9 = fs->make<TH1D>( "hclus9", "Clus per event",sizeH, lowH, highH);
  //hclus10 = fs->make<TH1D>( "hclus10", "Clus per event",
  //		    sizeH, lowH, highH);

  // #ifdef BX
  //   hclus30=fs->make<TH1D>("hclus30","Clus per event",sizeH, lowH, highH);
  //   hclus31=fs->make<TH1D>("hclus31","Clus per event",sizeH, lowH, highH);
  //   hclus32=fs->make<TH1D>("hclus32","Clus per event",sizeH, lowH, highH);
  //   hclus33=fs->make<TH1D>("hclus33","Clus per event",sizeH, lowH, highH);
  //   hclus34=fs->make<TH1D>("hclus34","Clus per event",sizeH, lowH, highH);
  //   hclus35=fs->make<TH1D>("hclus35","Clus per event",sizeH, lowH, highH);
  //   hclus36=fs->make<TH1D>("hclus36","Clus per event",sizeH, lowH, highH);
  //   hclus37=fs->make<TH1D>("hclus37","Clus per event",sizeH, lowH, highH);
  //   hclus38=fs->make<TH1D>("hclus38","Clus per event",sizeH, lowH, highH);
  //   hclus39=fs->make<TH1D>("hclus39","Clus per event",sizeH, lowH, highH);
  // #endif

  hdetsPerLay1 = fs->make<TH1D>("hdetsPerLay1", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1D>("hdetsPerLay3", "Full dets per layer l3", 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1D>("hdetsPerLay2", "Full dets per layer l2", 257, -0.5, 256.5);

  hdets2 = fs->make<TH1D>("hdets2", "Dets per event", 2000, 0.5, 1999.5);
  hdets = fs->make<TH1D>("hdets", "Dets per event", 2000, -0.5, 1999.5);
  hmaxPixPerDet = fs->make<TH1D>("hmaxPixPerDet", "Max pixels per det", 1000, -0.5, 999.5);

  sizeH = 400;
  lowH = 0.;
  highH = 100.0;                                                             // charge limit in kelec
  hcharge1 = fs->make<TH1D>("hcharge1", "Clu charge l1", sizeH, 0., highH);  //in ke
  hcharge2 = fs->make<TH1D>("hcharge2", "Clu charge l2", sizeH, 0., highH);
  hcharge3 = fs->make<TH1D>("hcharge3", "Clu charge l3", sizeH, 0., highH);
  hcharge4 = fs->make<TH1D>("hcharge4", "Clu charge d1", sizeH, 0., highH);
  hcharge5 = fs->make<TH1D>("hcharge5", "Clu charge d2", sizeH, 0., highH);
  //hchargen = fs->make<TH1D>( "hchargen", "Clu charge", sizeH, 0.,highH);
  //hchargen1 = fs->make<TH1D>( "hchargen1", "Clu charge", sizeH, 0.,highH);
  //hchargen2 = fs->make<TH1D>( "hchargen2", "Clu charge", sizeH, 0.,highH);
  //hchargen3 = fs->make<TH1D>( "hchargen3", "Clu charge", sizeH, 0.,highH);
  //hchargen4 = fs->make<TH1D>( "hchargen4", "Clu charge", sizeH, 0.,highH);
  //hchargen5 = fs->make<TH2D>( "hchargen5", "Clu charge-size", sizeH, 0.,highH,10,0.,10.);

  //havCharge1 = fs->make<TH1D>( "havCharge1", "Clu charge l1", sizeH, 0.,2.*highH); //in ke
  //havCharge2 = fs->make<TH1D>( "havCharge2", "Clu charge l2", sizeH, 0.,2.*highH);
  //havCharge3 = fs->make<TH1D>( "havCharge3", "Clu charge l3", 110, 0.,1.1);
  //havCharge4 = fs->make<TH1D>( "havCharge4", "Clu charge d1", sizeH, 0.,2.*highH);
  //havCharge5 = fs->make<TH1D>( "havCharge5", "Clu charge d2", sizeH, 0.,2.*highH);
  //havCharge6 = fs->make<TH1D>( "havCharge6", "Clu charge d2", sizeH, 0.,2.*highH);

  //hcharge11 = fs->make<TH1D>( "hcharge11", "Clu charge l1", sizeH, 0.,highH);
  //hcharge12 = fs->make<TH1D>( "hcharge12", "Clu charge l1", sizeH, 0.,highH);
  //hcharge13 = fs->make<TH1D>( "hcharge13", "Clu charge l1", sizeH, 0.,highH);
  //hcharge14 = fs->make<TH1D>( "hcharge14", "Clu charge l1", sizeH, 0.,highH);

#if defined(BX) || defined(BX_NEW)
  hchargebx1 = fs->make<TH1D>("hchargebx1", "Clu charge", 100, 0., highH);          //in ke
  hchargebx2 = fs->make<TH1D>("hchargebx2", "Clu charge", 100, 0., highH);          //in ke
  hchargebx3 = fs->make<TH1D>("hchargebx3", "Clu charge", 100, 0., highH);          //in ke
  hchargebx4 = fs->make<TH1D>("hchargebx4", "Clu charge", 100, 0., highH);          //in ke
  hchargebx5 = fs->make<TH1D>("hchargebx5", "Clu charge", 100, 0., highH);          //in ke
  hchargebx6 = fs->make<TH1D>("hchargebx6", "Clu charge", 100, 0., highH);          //in ke
  sizeH = 300;                                                                      // 600
  highH = 60.0;                                                                     // charge limit in kelec
  hpixChargebx1 = fs->make<TH1D>("hpixChargebx1", "Pix charge", sizeH, 0., highH);  //in ke
  hpixChargebx2 = fs->make<TH1D>("hpixChargebx2", "Pix charge", sizeH, 0., highH);  //in ke
  hpixChargebx3 = fs->make<TH1D>("hpixChargebx3", "Pix charge", sizeH, 0., highH);  //in ke
  hpixChargebx5 = fs->make<TH1D>("hpixChargebx5", "Pix charge", sizeH, 0., highH);  //in ke
  hpixChargebx6 = fs->make<TH1D>("hpixChargebx6", "Pix charge", sizeH, 0., highH);  //in ke

  sizeH = 200;
  highH = 199.5;  // charge limit in kelec
  hsizebx1 = fs->make<TH1D>("hsizebx1", "clu size", sizeH, -0.5, highH);
  hsizebx2 = fs->make<TH1D>("hsizebx2", "clu size", sizeH, -0.5, highH);
  hsizebx3 = fs->make<TH1D>("hsizebx3", "clu size", sizeH, -0.5, highH);
  hsizebx5 = fs->make<TH1D>("hsizebx5", "clu size", sizeH, -0.5, highH);
  hsizebx6 = fs->make<TH1D>("hsizebx6", "clu size", sizeH, -0.5, highH);
#endif

  sizeH = 300;                                                                     // 600
  highH = 60.0;                                                                    // charge limit in kelec
  hpixcharge1 = fs->make<TH1D>("hpixcharge1", "Pix charge l1", sizeH, 0., highH);  //in ke
  hpixcharge2 = fs->make<TH1D>("hpixcharge2", "Pix charge l2", sizeH, 0., highH);
  hpixcharge3 = fs->make<TH1D>("hpixcharge3", "Pix charge l3", sizeH, 0., highH);
  hpixcharge4 = fs->make<TH1D>("hpixcharge4", "Pix charge d1", sizeH, 0., highH);
  hpixcharge5 = fs->make<TH1D>("hpixcharge5", "Pix charge d2", sizeH, 0., highH);
  //hpixchargen = fs->make<TH1D>( "hpixchargen", "Pix charge",sizeH, 0.,highH);
  //hpixchargen1 = fs->make<TH1D>( "hpixchargen1", "Pix charge",sizeH, 0.,highH);
  //hpixchargen2 = fs->make<TH1D>( "hpixchargen2", "Pix charge",sizeH, 0.,highH);

  //hnpixcharge1 = fs->make<TH1D>( "hnpixcharge1", "Noise pix charge l1",sizeH, 0.,highH);
  //hnpixcharge2 = fs->make<TH1D>( "hnpixcharge2", "Noise pix charge l2",sizeH, 0.,highH);
  //hnpixcharge3 = fs->make<TH1D>( "hnpixcharge3", "Noise pix charge l3",sizeH, 0.,highH);

  //hcols1 = fs->make<TH1D>( "hcols1", "Layer 1 cols", 500,-0.5,499.5);
  //hcols2 = fs->make<TH1D>( "hcols2", "Layer 2 cols", 500,-0.5,499.5);
  //hcols3 = fs->make<TH1D>( "hcols3", "Layer 3 cols", 500,-0.5,499.5);
  //hrows1 = fs->make<TH1D>( "hrows1", "Layer 1 rows", 200,-0.5,199.5);
  //hrows2 = fs->make<TH1D>( "hrows2", "Layer 2 rows", 200,-0.5,199.5);
  //hrows3 = fs->make<TH1D>( "hrows3", "layer 3 rows", 200,-0.5,199.5);

  hpcols1 = fs->make<TH1D>("hpcols1", "Layer 1 pix cols", 500, -0.5, 499.5);
  hpcols2 = fs->make<TH1D>("hpcols2", "Layer 2 pix cols", 500, -0.5, 499.5);
  hpcols3 = fs->make<TH1D>("hpcols3", "Layer 3 pix cols", 500, -0.5, 499.5);

  hprows1 = fs->make<TH1D>("hprows1", "Layer 1 pix rows", 200, -0.5, 199.5);
  hprows2 = fs->make<TH1D>("hprows2", "Layer 2 pix rows", 200, -0.5, 199.5);
  hprows3 = fs->make<TH1D>("hprows3", "layer 3 pix rows", 200, -0.5, 199.5);

  sizeH = 1000;
  highH = 999.5;  // charge limit in kelec
  hsize1 = fs->make<TH1D>("hsize1", "layer 1 clu size", sizeH, -0.5, highH);
  hsize2 = fs->make<TH1D>("hsize2", "layer 2 clu size", sizeH, -0.5, highH);
  hsize3 = fs->make<TH1D>("hsize3", "layer 3 clu size", sizeH, -0.5, highH);
  //hsizen = fs->make<TH1D>( "hsizen", "clu size",sizeH,-0.5,highH);
  //hsizen1 = fs->make<TH1D>( "hsizen1", "clu size",sizeH,-0.5,highH);
  //hsizen2 = fs->make<TH1D>( "hsizen2", "clu size",sizeH,-0.5,highH);

  hsizex1 = fs->make<TH1D>("hsizex1", "lay1 clu size in x", 100, -0.5, 99.5);
  hsizex2 = fs->make<TH1D>("hsizex2", "lay2 clu size in x", 100, -0.5, 99.5);
  hsizex3 = fs->make<TH1D>("hsizex3", "lay3 clu size in x", 100, -0.5, 99.5);
  //hsizexn = fs->make<TH1D>( "hsizexn", "clu size in x",
  //	      20,-0.5,19.5);
  // hsizexn1 = fs->make<TH1D>( "hsizexn1", "clu size in x",
  // 		      20,-0.5,19.5);
  // hsizexn2 = fs->make<TH1D>( "hsizexn2", "clu size in x",
  // 		      20,-0.5,19.5);
  hsizey1 = fs->make<TH1D>("hsizey1", "lay1 clu size in y", 100, -0.5, 99.5);
  hsizey2 = fs->make<TH1D>("hsizey2", "lay2 clu size in y", 100, -0.5, 99.5);
  hsizey3 = fs->make<TH1D>("hsizey3", "lay3 clu size in y", 100, -0.5, 99.5);
  //hsizeyn = fs->make<TH1D>( "hsizeyn", "lay3 clu size in y",
  //	      30,-0.5,29.5);
  // hsizeyn1 = fs->make<TH1D>( "hsizeyn1", "lay3 clu size in y",
  // 		      30,-0.5,29.5);
  // hsizeyn2 = fs->make<TH1D>( "hsizeyn2", "lay3 clu size in y",
  // 		      30,-0.5,29.5);

  hpixDiskR1 = fs->make<TH1D>("hpixDiskR1", "pix vs. r, disk 1", 200, 0., 20.);
  hpixDiskR2 = fs->make<TH1D>("hpixDiskR2", "pix vs. r, disk 2", 200, 0., 20.);

  hevent = fs->make<TH1D>("hevent", "event", 100, 0, 10000000.);
  //horbit = fs->make<TH1D>("horbit","orbit",100, 0,100000000.);

  hlumi1 = fs->make<TH1D>("hlumi1", "lumi", 2000, 0, 2000.);
  hlumi0 = fs->make<TH1D>("hlumi0", "lumi", 2000, 0, 2000.);
  hlumi = fs->make<TH1D>("hlumi", "lumi", 2000, 0, 2000.);
  hlumi10 = fs->make<TH1D>("hlumi10", "lumi10", 2000, 0, 2000.);
  hlumi11 = fs->make<TH1D>("hlumi11", "lumi11", 2000, 0, 2000.);
  hlumi12 = fs->make<TH1D>("hlumi12", "lumi12", 2000, 0, 2000.);
  hlumi13 = fs->make<TH1D>("hlumi13", "lumi13", 2000, 0, 2000.);
  hlumi14 = fs->make<TH1D>("hlumi14", "lumi14", 2000, 0, 2000.);
  //hlumi15 = fs->make<TH1D>("hlumi15", "lumi15",   2000,0,2000.);
  //hlumi16 = fs->make<TH1D>("hlumi16", "lumi16",   2000,0,2000.);
  //hlumi17 = fs->make<TH1D>("hlumi17", "lumi17",   2000,0,2000.);
  //hlumi18 = fs->make<TH1D>("hlumi18", "lumi18",   2000,0,2000.);
  //hlumi19 = fs->make<TH1D>("hlumi19", "lumi19",   2000,0,2000.);

  //hbx11   = fs->make<TH1D>("hbx11",   "bx",   4000,0,4000.);
  // hbx10   = fs->make<TH1D>("hbx10",   "bx",   4000,0,4000.);
  // hbx9    = fs->make<TH1D>("hbx9",   "bx",   4000,0,4000.);
  // hbx8    = fs->make<TH1D>("hbx8",   "bx",   4000,0,4000.);
  // hbx7    = fs->make<TH1D>("hbx7",   "bx",   4000,0,4000.);
  hbx6 = fs->make<TH1D>("hbx6", "bx", 4000, 0, 4000.);
  hbx5 = fs->make<TH1D>("hbx5", "bx", 4000, 0, 4000.);
  hbx4 = fs->make<TH1D>("hbx4", "bx", 4000, 0, 4000.);
  hbx3 = fs->make<TH1D>("hbx3", "bx", 4000, 0, 4000.);
  hbx2 = fs->make<TH1D>("hbx2", "bx", 4000, 0, 4000.);
  hbx1 = fs->make<TH1D>("hbx1", "bx", 4000, 0, 4000.);
  hbx0 = fs->make<TH1D>("hbx0", "bx", 4000, 0, 4000.);
  hbx = fs->make<TH1D>("hbx", "bx", 4000, 0, 4000.);

  hl1a = fs->make<TH1D>("hl1a", "l1a", 128, -0.5, 127.5);
  hl1t = fs->make<TH1D>("hl1t", "l1t", 128, -0.5, 127.5);
  hltt = fs->make<TH1D>("hltt", "ltt", 128, -0.5, 127.5);
  hl1t1 = fs->make<TH1D>("hl1t1", "l1t1", 128, -0.5, 127.5);
  hl1a1 = fs->make<TH1D>("hl1a1", "l1a1", 128, -0.5, 127.5);

  //hmbits1 = fs->make<TH1D>("hmbits1","hmbits1",50,-0.5,49.5);
  //hmbits2 = fs->make<TH1D>("hmbits2","hmbits2",50,-0.5,49.5);
  //hmbits3 = fs->make<TH1D>("hmbits3","hmbits3",50,-0.5,49.5);

  hlt1 = fs->make<TH1D>("hlt1", "hlt1", 256, -0.5, 255.5);
  hlt2 = fs->make<TH1D>("hlt2", "hlt2", 256, -0.5, 255.5);
  hlt3 = fs->make<TH1D>("hlt3", "hlt3", 256, -0.5, 255.5);

  hgz1 = fs->make<TH1D>("hgz1", "layer1, clu global z", 600, -30., 30.);
  hgz2 = fs->make<TH1D>("hgz2", "layer2, clu global z", 600, -30., 30.);
  hgz3 = fs->make<TH1D>("hgz3", "layer3, clu global z", 600, -30., 30.);

#ifdef USE_RESYNCS
  htest1 = fs->make<TH1D>("htest1", "test1", 1000, 0., 1000.);
  htest2 = fs->make<TH1D>("htest2", "test2", 1000, 0., 1000.);
  htest3 = fs->make<TH1D>("htest3", "test3", 1000, 0., 3.0E8);
  htest4 = fs->make<TH1D>("htest4", "test4", 1000, 0., 3.0E8);
#endif

  // dets hit per event
  hDetsMap1 = fs->make<TH2F>("hDetsMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hDetsMap1->SetOption("colz");
  hDetsMap2 = fs->make<TH2F>("hDetsMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hDetsMap2->SetOption("colz");
  hDetsMap3 = fs->make<TH2F>("hDetsMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hDetsMap3->SetOption("colz");
  // clus per det
  hDetMap1 = fs->make<TH2F>("hDetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hDetMap1->SetOption("colz");
  hDetMap2 = fs->make<TH2F>("hDetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hDetMap2->SetOption("colz");
  hDetMap3 = fs->make<TH2F>("hDetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hDetMap3->SetOption("colz");
  // pix per det
  hpDetMap1 = fs->make<TH2F>("hpDetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hpDetMap1->SetOption("colz");
  hpDetMap2 = fs->make<TH2F>("hpDetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hpDetMap2->SetOption("colz");
  hpDetMap3 = fs->make<TH2F>("hpDetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hpDetMap3->SetOption("colz");

  hsizeDetMap1 = fs->make<TH2F>("hsizeDetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hsizeDetMap1->SetOption("colz");
  hsizeDetMap2 = fs->make<TH2F>("hsizeDetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hsizeDetMap2->SetOption("colz");
  hsizeDetMap3 = fs->make<TH2F>("hsizeDetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hsizeDetMap3->SetOption("colz");

  //hsizeXDetMap1 = fs->make<TH2F>("hsizeXDetMap1"," ",9,-4.5,4.5,21,-10.5,10.5);
  //hsizeXDetMap1->SetOption("colz");
  //hsizeXDetMap2 = fs->make<TH2F>("hsizeXDetMap2"," ",9,-4.5,4.5,33,-16.5,16.5);
  //hsizeXDetMap2->SetOption("colz");
  //hsizeXDetMap3 = fs->make<TH2F>("hsizeXDetMap3"," ",9,-4.5,4.5,45,-22.5,22.5);
  //hsizeXDetMap3->SetOption("colz");

  //hsizeYDetMap1 = fs->make<TH2F>("hsizeYDetMap1"," ",9,-4.5,4.5,21,-10.5,10.5);
  //hsizeYDetMap1->SetOption("colz");
  //hsizeYDetMap2 = fs->make<TH2F>("hsizeYDetMap2"," ",9,-4.5,4.5,33,-16.5,16.5);
  //hsizeYDetMap2->SetOption("colz");
  //hsizeYDetMap3 = fs->make<TH2F>("hsizeYDetMap3"," ",9,-4.5,4.5,45,-22.5,22.5);
  //hsizeYDetMap3->SetOption("colz");

  hpixDetMap1 = fs->make<TH2F>("hpixDetMap1", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap2 = fs->make<TH2F>("hpixDetMap2", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap3 = fs->make<TH2F>("hpixDetMap3", "pix det layer 3", 416, 0., 416., 160, 0., 160.);

  hcluDetMap1 = fs->make<TH2F>("hcluDetMap1", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  hcluDetMap2 = fs->make<TH2F>("hcluDetMap2", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  hcluDetMap3 = fs->make<TH2F>("hcluDetMap3", "clu det layer 1", 416, 0., 416., 160, 0., 160.);

  // Special test hitos for inefficiency effects
  hpixDetMap10 = fs->make<TH2F>("hpixDetMap10", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap20 = fs->make<TH2F>("hpixDetMap20", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap30 = fs->make<TH2F>("hpixDetMap30", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap11 = fs->make<TH2F>("hpixDetMap11", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap12 = fs->make<TH2F>("hpixDetMap12", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap13 = fs->make<TH2F>("hpixDetMap13", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap14 = fs->make<TH2F>("hpixDetMap14", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap15 = fs->make<TH2F>("hpixDetMap15", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap16 = fs->make<TH2F>("hpixDetMap16", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap17 = fs->make<TH2F>("hpixDetMap17", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap18 = fs->make<TH2F>("hpixDetMap18", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap19 = fs->make<TH2F>("hpixDetMap19", "pix det layer 1", 416, 0., 416., 160, 0., 160.);

  hpixDetMap21 = fs->make<TH2F>("hpixDetMap21", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap22 = fs->make<TH2F>("hpixDetMap22", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap23 = fs->make<TH2F>("hpixDetMap23", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap24 = fs->make<TH2F>("hpixDetMap24", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap25 = fs->make<TH2F>("hpixDetMap25", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap26 = fs->make<TH2F>("hpixDetMap26", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap27 = fs->make<TH2F>("hpixDetMap27", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap28 = fs->make<TH2F>("hpixDetMap28", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap29 = fs->make<TH2F>("hpixDetMap29", "pix det layer 2", 416, 0., 416., 160, 0., 160.);

  hpixDetMap31 = fs->make<TH2F>("hpixDetMap31", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap32 = fs->make<TH2F>("hpixDetMap32", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap33 = fs->make<TH2F>("hpixDetMap33", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap34 = fs->make<TH2F>("hpixDetMap34", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap35 = fs->make<TH2F>("hpixDetMap35", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap36 = fs->make<TH2F>("hpixDetMap36", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap37 = fs->make<TH2F>("hpixDetMap37", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap38 = fs->make<TH2F>("hpixDetMap38", "pix det layer 3", 416, 0., 416., 160, 0., 160.);
  hpixDetMap39 = fs->make<TH2F>("hpixDetMap39", "pix det layer 3", 416, 0., 416., 160, 0., 160.);

  //h2d1 = fs->make<TH2F>( "h2d1", "2d 1",200,0.,4000.,50,0., 200.);
  //h2d2 = fs->make<TH2F>( "h2d2", "2d 2", 55,0.,1.1, 150,0., 150.);
  //h2d3 = fs->make<TH2F>( "h2d3", "2d 3", 55,0.,1.1, 300,0.,4000.);

  hclumult1 = fs->make<TProfile>("hclumult1", "cluster size layer 1", 56, -28., 28., 0.0, 100.);
  hclumult2 = fs->make<TProfile>("hclumult2", "cluster size layer 2", 56, -28., 28., 0.0, 100.);
  hclumult3 = fs->make<TProfile>("hclumult3", "cluster size layer 3", 56, -28., 28., 0.0, 100.);

  hclumultx1 = fs->make<TProfile>("hclumultx1", "cluster x-size layer 1", 56, -28., 28., 0.0, 100.);
  hclumultx2 = fs->make<TProfile>("hclumultx2", "cluster x-size layer 2", 56, -28., 28., 0.0, 100.);
  hclumultx3 = fs->make<TProfile>("hclumultx3", "cluster x-size layer 3", 56, -28., 28., 0.0, 100.);

  hclumulty1 = fs->make<TProfile>("hclumulty1", "cluster y-size layer 1", 56, -28., 28., 0.0, 100.);
  hclumulty2 = fs->make<TProfile>("hclumulty2", "cluster y-size layer 2", 56, -28., 28., 0.0, 100.);
  hclumulty3 = fs->make<TProfile>("hclumulty3", "cluster y-size layer 3", 56, -28., 28., 0.0, 100.);

  hcluchar1 = fs->make<TProfile>("hcluchar1", "cluster char layer 1", 56, -28., 28., 0.0, 1000.);
  hcluchar2 = fs->make<TProfile>("hcluchar2", "cluster char layer 2", 56, -28., 28., 0.0, 1000.);
  hcluchar3 = fs->make<TProfile>("hcluchar3", "cluster char layer 3", 56, -28., 28., 0.0, 1000.);

  hpixchar1 = fs->make<TProfile>("hpixchar1", "pix char layer 1", 56, -28., 28., 0.0, 1000.);
  hpixchar2 = fs->make<TProfile>("hpixchar2", "pix char layer 2", 56, -28., 28., 0.0, 1000.);
  hpixchar3 = fs->make<TProfile>("hpixchar3", "pix char layer 3", 56, -28., 28., 0.0, 1000.);

  sizeH = 1000;
  highH = 3000.;
  hclusls = fs->make<TProfile>("hclusls", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  hpixls = fs->make<TProfile>("hpixls", "pix vs ls ", sizeH, 0., highH, 0.0, 100000.);
  hpvls = fs->make<TProfile>("hpvls", "pvs vs ls", sizeH, 0., highH, 0.0, 10000.);
  //hpvlsn = fs->make<TProfile>("hpvlsn","pvs/lumi vs ls",sizeH,0.,highH,0.0,1000.);

  hcharCluls = fs->make<TProfile>("hcharCluls", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharPixls = fs->make<TProfile>("hcharPixls", "pix char vs ls", sizeH, 0., highH, 0.0, 100.);
  hsizeCluls = fs->make<TProfile>("hsizeCluls", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls = fs->make<TProfile>("hsizeXCluls", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);

  // sizeH = 100;
  // highH =  10.;
  // hpixbl   = fs->make<TProfile>("hpixbl",  "pixb vs lumi ", sizeH,0.,highH,0.0,100000.);
  // hclusbl  = fs->make<TProfile>("hclusbl", "clusb vs lumi", sizeH,0.,highH,0.0,30000.);
  // hpixb1l  = fs->make<TProfile>("hpixb1l", "pixb1 vs lumi", sizeH,0.,highH,0.0,100000.);
  // hclusb1l = fs->make<TProfile>("hclusb1l","clusb1 vs lumi",sizeH,0.,highH,0.0,30000.);
  // hpixb2l  = fs->make<TProfile>("hpixb2l", "pixb2 vs lumi ",sizeH,0.,highH,0.0,100000.);
  // hclusb2l = fs->make<TProfile>("hclusb2l","clusb2 vs lumi",sizeH,0.,highH,0.0,30000.);
  // hpixb3l  = fs->make<TProfile>("hpixb3l", "pixb3 vs lumi ",sizeH,0.,highH,0.0,100000.);
  // hclusb3l = fs->make<TProfile>("hclusb3l","clusb3 vs lumi",sizeH,0.,highH,0.0,30000.);
  // hpixfl   = fs->make<TProfile>("hpixfl",  "pixf vs lumi ", sizeH,0.,highH,0.0,100000.);
  // hclusfl  = fs->make<TProfile>("hclusfl", "clusf vs lumi", sizeH,0.,highH,0.0,30000.);
  // hpixfml  = fs->make<TProfile>("hpixfml", "pixfm vs lumi ",sizeH,0.,highH,0.0,100000.);
  // hclusfml = fs->make<TProfile>("hclusfml","clusfm vs lumi",sizeH,0.,highH,0.0,30000.);
  // hpixfpl  = fs->make<TProfile>("hpixfpl", "pixfp vs lumi ",sizeH,0.,highH,0.0,100000.);
  // hclusfpl = fs->make<TProfile>("hclusfpl","clusfp vs lumi",sizeH,0.,highH,0.0,30000.);

  sizeH = 1000;
  highH = 3000.;
  //hpixpvlsn  = fs->make<TProfile>("hpixpvlsn","pix/pv vs ls",sizeH,0.,highH,0.0,100000.);
  //hclupvlsn  = fs->make<TProfile>("hclupvlsn","clu/pv vs ls",sizeH,0.,highH,0.0,30000.);

  hpixpv = fs->make<TProfile>("hpixpv", "pix vs pv", 1000, 0., 20000, 0.0, 1000.);
  hclupv = fs->make<TProfile>("hclupv", "clu vs pv", 1000, 0., 10000, 0.0, 1000.);

  //hintgls  = fs->make<TProfile>("hintgls", "intg lumi vs ls ",sizeH,0.,highH,0.0,10000.);
  hinstls = fs->make<TProfile>("hinstls", "inst bx lumi vs ls ", sizeH, 0., highH, 0.0, 1000.);
  hinstlsbx = fs->make<TProfile>("hinstlsbx", "inst bx lumi vs ls ", sizeH, 0., highH, 0.0, 1000.);
  //hbeam1  = fs->make<TProfile>("hbeam1", "beam1 vs ls ",sizeH,0.,highH,0.0,1000.);
  //hbeam2  = fs->make<TProfile>("hbeam2", "beam2 vs ls ",sizeH,0.,highH,0.0,1000.);

  hpixbx = fs->make<TProfile>("hpixbx", "pix vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  hclubx = fs->make<TProfile>("hclubx", "clu vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  hpvbx = fs->make<TProfile>("hpvbx", "pv vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  //hpixbxn = fs->make<TProfile>("hpixbxn","pix/lumi vs bx ",4000,-0.5,3999.5,0.0,1000000.);
  //hclubxn = fs->make<TProfile>("hclubxn","clu/lumi vs bx ",4000,-0.5,3999.5,0.0,1000000.);
  //hpvbxn  = fs->make<TProfile>("hpvbxn", "pv/lumi vs bx ", 4000,-0.5,3999.5,0.0,1000000.);

  hcharClubx = fs->make<TProfile>("hcharClubx", "clu charge vs bx ", 4000, -0.5, 3999.5, 0.0, 100.);
  hcharPixbx = fs->make<TProfile>("hcharPixbx", "pix charge vs bx ", 4000, -0.5, 3999.5, 0.0, 100.);
  hsizeClubx = fs->make<TProfile>("hsizeClubx", "clu size vs bx ", 4000, -0.5, 3999.5, 0.0, 1000.);
  hsizeYClubx = fs->make<TProfile>("hsizeYClubx", "clu size-y vs bx ", 4000, -0.5, 3999.5, 0.0, 1000.);

  hinstbx = fs->make<TProfile>("hinstbx", "inst lumi vs bx ", 4000, -0.5, 3999.5, 0.0, 100.);

  hcluLumi = fs->make<TProfile>("hcluLumi", "clus vs inst lumi", 100, 0.0, 10., 0.0, 100000.);
  hpixLumi = fs->make<TProfile>("hpixLumi", "pixs vs inst lumi", 100, 0.0, 10., 0.0, 100000.);
  hcharCluLumi = fs->make<TProfile>("hcharCluLumi", "clu char vs inst lumi", 100, 0.0, 10., 0.0, 100.);
  hcharPixLumi = fs->make<TProfile>("hcharPixLumi", "pix char vs inst lumi", 100, 0.0, 10., 0.0, 100.);
  hsizeCluLumi = fs->make<TProfile>("hsizeCluLumi", "clu size vs inst lumi", 100, 0.0, 10., 0.0, 100.);
  hsizeXCluLumi = fs->make<TProfile>("hsizeXCluLumi", "clu size-x vs inst lumi", 100, 0.0, 10., 0.0, 100.);
  hsizeYCluLumi = fs->make<TProfile>("hsizeYCluLumi", "clu size-y vs inst lumi", 100, 0.0, 10., 0.0, 100.);

  //hclus13ls  = fs->make<TProfile>("hclus13ls", "clus1/clus3 vs ls", sizeH,0.,highH,0.0,30000.);
  //hclus23ls  = fs->make<TProfile>("hclus23ls", "clus2/clus3 vs ls", sizeH,0.,highH,0.0,30000.);
  //hclus12ls  = fs->make<TProfile>("hclus12ls", "clus1/clus2 vs ls", sizeH,0.,highH,0.0,30000.);
  //hclusf3ls  = fs->make<TProfile>("hclusf3ls", "clusf/clus3 vs ls", sizeH,0.,highH,0.0,30000.);

  //    hclusbx1lsn  = fs->make<TProfile>("hclusbx1lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx2lsn  = fs->make<TProfile>("hclusbx2lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx3lsn  = fs->make<TProfile>("hclusbx3lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx4lsn  = fs->make<TProfile>("hclusbx4lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx5lsn  = fs->make<TProfile>("hclusbx5lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx6lsn  = fs->make<TProfile>("hclusbx6lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx7lsn  = fs->make<TProfile>("hclusbx7lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx8lsn  = fs->make<TProfile>("hclusbx8lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx9lsn  = fs->make<TProfile>("hclusbx9lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);

  //    hclusbx11lsn  = fs->make<TProfile>("hclusbx11lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx12lsn  = fs->make<TProfile>("hclusbx12lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx13lsn  = fs->make<TProfile>("hclusbx13lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx14lsn  = fs->make<TProfile>("hclusbx14lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx15lsn  = fs->make<TProfile>("hclusbx15lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx16lsn  = fs->make<TProfile>("hclusbx16lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx17lsn  = fs->make<TProfile>("hclusbx17lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx18lsn  = fs->make<TProfile>("hclusbx18lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);
  //    hclusbx19lsn  = fs->make<TProfile>("hclusbx19lsn", "clus vs ls for bx", sizeH,0.,highH,0.0,30000.);

  //hclusls2 = fs->make<TH2F>("hclusls2","clus bs ls", 300,0.,900.,100,0.,5000.);
  //hpixls2  = fs->make<TH2F>("hpixls2", "pix per ls ",300,0.,900.,100,0.,20000.);
  //h2pixpv  = fs->make<TH2F>("h2pixpv","pix vs pv",100,0.,40000, 25,0.0,50.);
  //h2clupv  = fs->make<TH2F>("h2clupv","clu vs pv",100,0.,20000, 25,0.0,50.);

  //hclusls1 = fs->make<TH1D>("hclusls1","av clus/lumi",200,0.,2000.);
  //hpixls1  = fs->make<TH1D>("hpixls1", "av pix/lumi ",500,0.,5000.);

  //hintg  = fs->make<TH1D>("hintg", "intg lumi",100,0.0,1000.);
  hinst = fs->make<TH1D>("hinst", "inst lumi", 100, 0.0, 10.);
  hpvs = fs->make<TH1D>("hpvs", "pvs", 100, -0.5, 99.5);

#ifdef VDM_STUDIES
  hcharCluls1 = fs->make<TProfile>("hcharCluls1", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharPixls1 = fs->make<TProfile>("hcharPixls1", "pix char vs ls", sizeH, 0., highH, 0.0, 100.);
  hsizeCluls1 = fs->make<TProfile>("hsizeCluls1", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls1 = fs->make<TProfile>("hsizeXCluls1", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharCluls2 = fs->make<TProfile>("hcharCluls2", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharPixls2 = fs->make<TProfile>("hcharPixls2", "pix char vs ls", sizeH, 0., highH, 0.0, 100.);
  hsizeCluls2 = fs->make<TProfile>("hsizeCluls2", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls2 = fs->make<TProfile>("hsizeXCluls2", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharCluls3 = fs->make<TProfile>("hcharCluls3", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharPixls3 = fs->make<TProfile>("hcharPixls3", "pix char vs ls", sizeH, 0., highH, 0.0, 100.);
  hsizeCluls3 = fs->make<TProfile>("hsizeCluls3", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls3 = fs->make<TProfile>("hsizeXCluls3", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hclusls1 = fs->make<TProfile>("hclusls1", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  hpixls1 = fs->make<TProfile>("hpixls1", "pix vs ls ", sizeH, 0., highH, 0.0, 100000.);
  hclusls2 = fs->make<TProfile>("hclusls2", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  hpixls2 = fs->make<TProfile>("hpixls2", "pix vs ls ", sizeH, 0., highH, 0.0, 100000.);
  hclusls3 = fs->make<TProfile>("hclusls3", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  hpixls3 = fs->make<TProfile>("hpixls3", "pix vs ls ", sizeH, 0., highH, 0.0, 100000.);
#endif  // VDM_STUDIES

#ifdef INSTLUMI_STUDIES
  hcluslsn = fs->make<TProfile>("hcluslsn", "clus/lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixlsn = fs->make<TProfile>("hpixlsn", "pix/lumi ", sizeH, 0., highH, 0.0, 100000.);
  sizeH = 100;
  highH = 10.;
  hpixbl = fs->make<TProfile>("hpixbl", "pixb vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusbl = fs->make<TProfile>("hclusbl", "clusb vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixb1l = fs->make<TProfile>("hpixb1l", "pixb1 vs lumi", sizeH, 0., highH, 0.0, 100000.);
  hclusb1l = fs->make<TProfile>("hclusb1l", "clusb1 vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixb2l = fs->make<TProfile>("hpixb2l", "pixb2 vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusb2l = fs->make<TProfile>("hclusb2l", "clusb2 vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixb3l = fs->make<TProfile>("hpixb3l", "pixb3 vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusb3l = fs->make<TProfile>("hclusb3l", "clusb3 vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixfl = fs->make<TProfile>("hpixfl", "pixf vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusfl = fs->make<TProfile>("hclusfl", "clusf vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixfml = fs->make<TProfile>("hpixfml", "pixfm vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusfml = fs->make<TProfile>("hclusfml", "clusfm vs lumi", sizeH, 0., highH, 0.0, 30000.);
  hpixfpl = fs->make<TProfile>("hpixfpl", "pixfp vs lumi ", sizeH, 0., highH, 0.0, 100000.);
  hclusfpl = fs->make<TProfile>("hclusfpl", "clusfp vs lumi", sizeH, 0., highH, 0.0, 30000.);

  hpixbxn = fs->make<TProfile>("hpixbxn", "pix/lumi vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  hclubxn = fs->make<TProfile>("hclubxn", "clu/lumi vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  hpvbxn = fs->make<TProfile>("hpvbxn", "pv/lumi vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
#endif  // INSTLUMI_STUDIES

#ifdef ROC_EFF
  // dets with bad rocs
  hbadMap1 = fs->make<TH2F>("hbadMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hbadMap1->SetOption("colz");
  hbadMap2 = fs->make<TH2F>("hbadMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hbadMap2->SetOption("colz");
  hbadMap3 = fs->make<TH2F>("hbadMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hbadMap3->SetOption("colz");

  hrocHits1ls = fs->make<TH2F>("hrocHits1ls", " ", 1000, 0., 3000., 10, 0., 10.);
  hrocHits1ls->SetOption("colz");
  hrocHits2ls = fs->make<TH2F>("hrocHits2ls", " ", 1000, 0., 3000., 10, 0., 10.);
  hrocHits2ls->SetOption("colz");
  hrocHits3ls = fs->make<TH2F>("hrocHits3ls", " ", 1000, 0., 3000., 10, 0., 10.);
  hrocHits3ls->SetOption("colz");

  hcountInRoc1 = fs->make<TH1D>("hcountInRoc1", "roc 1 count", 10000, -0.5, 999999.5);
  hcountInRoc2 = fs->make<TH1D>("hcountInRoc2", "roc 2 count", 10000, -0.5, 999999.5);
  hcountInRoc3 = fs->make<TH1D>("hcountInRoc3", "roc 3 count", 10000, -0.5, 999999.5);
  hcountInRoc12 = fs->make<TH1D>("hcountInRoc12", "roc 1 count norm", 500, -0.5, 4.5);
  hcountInRoc22 = fs->make<TH1D>("hcountInRoc22", "roc 2 count norm", 500, -0.5, 4.5);
  hcountInRoc32 = fs->make<TH1D>("hcountInRoc32", "roc 3 count norm", 500, -0.5, 4.5);

  //   hmoduleHits1ls = fs->make<TH2F>("hmoduleHits1ls"," ",3000,0.,3000.,3,0.,3.);
  //   hmoduleHits1ls->SetOption("colz");
  //   hmoduleHits2ls = fs->make<TH2F>("hmoduleHits2ls"," ",3000,0.,3000.,3,0.,3.);
  //   hmoduleHits2ls->SetOption("colz");
  //   hmoduleHits3ls = fs->make<TH2F>("hmoduleHits3ls"," ",3000,0.,3000.,3,0.,3.);
  //   hmoduleHits3ls->SetOption("colz");

#endif  // ROC_EFF

  countEvents = 0;
  countAllEvents = 0;
  sumClusters = 0., sumPixels = 0.;
  countLumi = 0.;

  //for(int ilayer = 0; ilayer<3; ++ilayer)
  //for(int id=0;id<10;++id) {rocHits[ilayer][id]=0.; moduleHits[ilayer][id]=0.;}

#ifdef ROC_EFF
  // Analyze single pixel efficiency
  pixEff = new rocEfficiency();
#endif

#ifdef BX
  getbx = new getBX();
#endif

  lumiCorrector = new LumiCorrector();
}
// ------------ method called to at the end of the job  ------------
void TestClusters::endJob() {
  double norm = 1;
#ifdef ROC_EFF
  double totClusters = sumClusters;  // save the total cluster number
#endif

  if (countEvents > 0) {
    sumClusters = sumClusters / float(countEvents);
    sumPixels = sumPixels / float(countEvents);
    norm = 1. / float(countEvents);
  }

  countLumi /= 1000.;
  double c1 = 0, c2 = 0;
  double c3 = hinst->GetMean();
  if (c3 > 0.) {
    c1 = sumClusters / c3;
    c2 = sumPixels / c3;
  }

  edm::LogPrint("TestClusters") << " End PixelClusTest, events all/with hits=  " << countAllEvents << "/" << countEvents
                                << " clus/pix per full event " << sumClusters << "/" << sumPixels;
  edm::LogPrint("TestClusters") << " Lumi = " << countLumi << " still the /10 bug? "
                                << "clu and pix per lumi unit" << c1 << " " << c2;

  //Divide the size histos
  hsizeDetMap1->Divide(hsizeDetMap1, hDetMap1, 1., 1.);
  hsizeDetMap2->Divide(hsizeDetMap2, hDetMap2, 1., 1.);
  hsizeDetMap3->Divide(hsizeDetMap3, hDetMap3, 1., 1.);

  // Rescale all 2D plots
  hDetsMap1->Scale(norm);
  hDetsMap2->Scale(norm);
  hDetsMap3->Scale(norm);
  hDetMap1->Scale(norm);
  hDetMap2->Scale(norm);
  hDetMap3->Scale(norm);
  hpDetMap1->Scale(norm);
  hpDetMap2->Scale(norm);
  hpDetMap3->Scale(norm);
  hpixDetMap1->Scale(norm);
  hpixDetMap2->Scale(norm);
  hpixDetMap3->Scale(norm);
  hcluDetMap1->Scale(norm);
  hcluDetMap2->Scale(norm);
  hcluDetMap3->Scale(norm);

#ifdef ROC_EFF
  // Do this only if there is enough statistics
  double clusPerROC = totClusters / 15000.;
  if (clusPerROC < 1000.) {
    edm::LogPrint("TestClusters")
        << " The average number of clusters per ROC is too low to do teh ROF efficiency analysis " << clusPerROC;

  } else {  // do it

    int deadRocs1 = 0, ineffRocs1 = 0, deadRocs2 = 0, ineffRocs2 = 0, deadRocs3 = 0, ineffRocs3 = 0;
    const int ladders1[20] = {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int ladders2[32] = {-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                              1,   2,   3,   4,   5,   6,   7,   8,  9,  10, 11, 12, 13, 14, 15, 16};
    const int ladders3[44] = {-22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8,
                              -7,  -6,  -5,  -4,  -3,  -2,  -1,  1,   2,   3,   4,   5,   6,   7,  8,
                              9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22};
    const int modules[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
    const float effCut = 0.25;
    float half1 = 1, half2 = 0;

    // layer 1
    for (int ilad = 0; ilad < 20; ++ilad) {
      int lad = ladders1[ilad];
      for (int imod = 0; imod < 8; ++imod) {
        int mod = modules[imod];
        half1 = 0;
        half2 = 0;
        float count = pixEff->getModule(1, lad, mod, half1, half2);
        //edm::LogPrint("TestClusters")<<" layer 1 "<<lad<<" "<<mod<<" "<<count<<endl;
        if (count < 1.)
          continue;  // skip dead modules
        for (int roc = 0; roc < 16; ++roc) {
          if (roc < 8 && half1 == 0)
            continue;
          else if (roc > 7 && half2 == 0)
            continue;
          float tmp = pixEff->getRoc(1, lad, mod, roc);
          if (tmp < 0.) {                                          // half-module rocs will show up here
            if ((abs(lad) == 1 || abs(lad) == 10) && (roc > 7)) {  // OK half module
              continue;
            } else {
              edm::LogPrint("TestClusters")
                  << " Layer1, wrong number of hits, roc " << tmp << " " << lad << " " << mod << " " << roc;
            }
          } else if (tmp == 0.) {
            deadRocs1++;
            edm::LogPrint("TestClusters") << " layer1, dead  roc " << lad << " " << mod << " " << roc << " - " << count
                                          << " " << half1 << " " << half2;
          } else {
            hcountInRoc1->Fill(tmp);
            float tmp1 = tmp / count;
            //edm::LogPrint("TestClusters")<<" roc "<<roc<<" "<<tmp<<" "<<tmp1<<endl;
            hcountInRoc12->Fill(tmp1);
            if (abs(1. - tmp1) > effCut) {
              ineffRocs1++;
              edm::LogPrint("TestClusters") << "LOW-EFF/NOISY ROC, Layer 1: " << tmp1 << "/" << tmp << " ladder " << lad
                                            << " module " << mod << " roc(false #) " << roc;
              hbadMap1->Fill(float(mod), float(lad));
            }
          }  //if
        }    // loop over rocs
      }      // mod
    }        // lad

    for (int ilad = 0; ilad < 32; ++ilad) {
      int lad = ladders2[ilad];
      for (int imod = 0; imod < 8; ++imod) {
        int mod = modules[imod];
        half1 = 0;
        half2 = 0;
        float count = pixEff->getModule(2, lad, mod, half1, half2);
        //edm::LogPrint("TestClusters")<<" layer 2 "<<lad<<" "<<mod<<" "<<count<<endl;
        if (count < 1.)
          continue;  // skip dead whole modules
        for (int roc = 0; roc < 16; ++roc) {
          if (roc < 8 && half1 == 0)
            continue;
          else if (roc > 7 && half2 == 0)
            continue;
          float tmp = pixEff->getRoc(2, lad, mod, roc);
          if (tmp < 0.) {
            if ((abs(lad) == 1 || abs(lad) == 16) && (roc > 7)) {  // OK half module
              continue;
            } else {
              edm::LogPrint("TestClusters")
                  << " Layer 2, wrong number of hits, roc " << tmp << " " << lad << " " << mod << " " << roc;
            }
          } else if (tmp == 0.) {
            deadRocs2++;
            edm::LogPrint("TestClusters") << " layer2, dead  roc " << lad << " " << mod << " " << roc << " - " << count
                                          << " " << half1 << " " << half2;
          } else {
            hcountInRoc2->Fill(tmp);
            float tmp1 = tmp / count;
            //edm::LogPrint("TestClusters")<<" roc "<<roc<<" "<<tmp<<" "<<tmp1<<endl;
            hcountInRoc22->Fill(tmp1);
            if (abs(1. - tmp1) > effCut) {
              ineffRocs2++;
              edm::LogPrint("TestClusters") << "LOW-EFF/NOISY ROC, Layer 2: " << tmp1 << "/" << tmp << " ladder " << lad
                                            << " module " << mod << " roc(false) " << roc;
              hbadMap2->Fill(float(mod), float(lad));
            }
          }  //if
        }    // loop over rocs
      }      // mod
    }        // lad

    for (int ilad = 0; ilad < 44; ++ilad) {
      int lad = ladders3[ilad];
      for (int imod = 0; imod < 8; ++imod) {
        int mod = modules[imod];
        half1 = 0;
        half2 = 0;
        float count = pixEff->getModule(3, lad, mod, half1, half2);
        //edm::LogPrint("TestClusters")<<" layer 3"<<lad<<" "<<mod<<" "<<count<<endl;
        if (count < 1.)
          continue;  // skip dead whole modules
        for (int roc = 0; roc < 16; ++roc) {
          float tmp = pixEff->getRoc(3, lad, mod, roc);
          if (tmp < 0.) {
            if ((abs(lad) == 1 || abs(lad) == 22) && (roc > 7)) {  // OK half module
              continue;
            } else {
              edm::LogPrint("TestClusters")
                  << " Layer 3: wrong number of hits, roc " << tmp << " " << lad << " " << mod << " " << roc;
            }
          } else if (tmp == 0.) {
            deadRocs3++;
            edm::LogPrint("TestClusters") << " layer3, dead  roc " << lad << " " << mod << " " << roc << " - " << count
                                          << " " << half1 << " " << half2;
          } else {
            hcountInRoc3->Fill(tmp);
            float tmp1 = tmp / count;
            //edm::LogPrint("TestClusters")<<" roc "<<roc<<" "<<tmp<<" "<<tmp1<<endl;
            hcountInRoc32->Fill(tmp1);
            if (abs(1. - tmp1) > effCut) {
              ineffRocs3++;
              edm::LogPrint("TestClusters") << "LOW-EFF/NOISY ROC, Layer 3: " << tmp1 << "/" << tmp << " ladder " << lad
                                            << " module " << mod << " roc(false) " << roc;
              hbadMap3->Fill(float(mod), float(lad));
            }
          }  //if
        }    // loop over rocs
      }      // mod
    }        // lad

    edm::LogPrint("TestClusters") << " Bad Rocs " << deadRocs1 << " " << deadRocs2 << " " << deadRocs3
                                  << ", Inefficient Rocs " << ineffRocs1 << " " << ineffRocs2 << " " << ineffRocs3;
  }  // if DO IT

#endif  // ROC_EFF
}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void TestClusters::analyze(const edm::Event &e, const edm::EventSetup &es) {
  using namespace edm;
  //const int MAX_CUT = 1000000; unused
  const int selectEvent = -1;
  //bool select = false; unused
  static RunNumber_t runNumberOld = -1;
  static int countRuns = 0;
  //static double pixsum = 0., clussum=0.;
  //static int nsum = 0, lsold=-999, lumiold=0.;

  // Get event setup
  edm::ESHandle<TrackerGeometry> geom = es.getHandle(trackerGeomToken_);
  const TrackerGeometry &theTracker(*geom);

  countAllEvents++;
  RunNumber_t const run = e.id().run();
  EventNumber_t const event = e.id().event();
  LuminosityBlockNumber_t const lumiBlock = e.luminosityBlock();

  int bx = e.bunchCrossing();
  int orbit = e.orbitNumber();

  hbx0->Fill(float(bx));
  hlumi0->Fill(float(lumiBlock));

  //if(lumiBlock<127) return;
  //if(event!=3787937) return;

  float instlumi = 0;
  //int beamint1=0, beamint2=0;

#ifdef Lumi
  float instlumiAv = 0, instlumiBx = 0;

  // Lumi
  edm::LuminosityBlock const &iLumi = e.getLuminosityBlock();
  edm::Handle<LumiSummary> lumi;
  edm::Handle<LumiDetails> ld;
  iLumi.getByToken(lumiSummaryToken_, lumi);
  iLumi.getByToken(lumiDetailsToken_, ld);

  edm::Handle<edm::ConditionsInLumiBlock> cond;
  iLumi.getByToken(condToken_, cond);
  // This will only work when running on RECO until (if) they fix it in the FW
  // When running on RAW and reconstructing, the LumiSummary will not appear
  // in the event before reaching endLuminosityBlock(). Therefore, it is not
  // possible to get this info in the event
  if (lumi.isValid()) {
    //intlumi =(lumi->intgRecLumi())/1000.; // integrated lumi per LS in -pb
    //instlumi=(lumi->avgInsDelLumi())/1000.; //ave. inst lumi per LS in -pb
    float tmp0 = (lumi->avgInsDelLumi());  //ave. inst lumi per LS in -nb
    //beamint1=(cond->totalIntensityBeam1)/1000;
    //beamint2=(cond->totalIntensityBeam2)/1000;
    const int nbx = 1331;  // for 1380 fills
    float corr = lumiCorrector->TotalNormOcc1((tmp0 / 1000.), nbx);
    //float tmp2 = lumiCorrector->TotalNormOcc2(tmp0,nbx);
    //float tmp3 = lumiCorrector->TotalNormET(tmp0,nbx);
    float tmp1 = tmp0 * corr;
    //instlumiAv = tmp1/1000000.;  // in 10^33
    float tmp2 = tmp1 / float(nbx) / 1000.;  // per bx
    instlumiAv = tmp2;                       // use per bx lumi

    if (ld.isValid()) {
      instlumiBx = ld->lumiValue(LumiDetails::kOCC1, bx) * 6.37;  // cor=6.37 in 2011, 7.13 in 2012?
    }

    //edm::LogPrint("TestClusters")<<run<<" "<<lumiBlock<<" "<<tmp0<<" "<<corr<<" "<<tmp1<<" "<<instlumiAv<<" "<<tmp2<<" ";
    //edm::LogPrint("TestClusters")<<instlumiBx<<endl;

  } else {
    //std::edm::LogPrint("TestClusters") << "** ERROR: Event does not get lumi info\n";
  }

  hinst->Fill(float(instlumiBx));
  //hintg->Fill(float(intlumi));
  hinstls->Fill(float(lumiBlock), float(instlumiAv));
  hinstlsbx->Fill(float(lumiBlock), float(instlumiBx));
  hinstbx->Fill(float(bx), float(instlumiBx));
  //hbeam1->Fill(float(lumiBlock),float(beamint1));
  //hbeam2->Fill(float(lumiBlock),float(beamint2));

  // Use this per bx as int lumi
  instlumi = instlumiBx;

#endif

  // PVs
  int numPVsGood = 0;
  if (select2 < 11 && run > 165000) {  // skip for earlier runs, crashes
    edm::Handle<reco::VertexCollection> vertices;
    e.getByToken(vtxToken_, vertices);

    //int numPVs = vertices->size(); // unused
    if (!vertices.failedToGet() && vertices.isValid()) {
      for (reco::VertexCollection::const_iterator iVertex = vertices->begin(); iVertex != vertices->end(); ++iVertex) {
        if (!iVertex->isValid())
          continue;
        if (iVertex->isFake())
          continue;
        numPVsGood++;

        if (PRINT) {
          edm::LogPrint("TestClusters") << "vertex";
          edm::LogPrint("TestClusters") << ": x " << iVertex->x();
          edm::LogPrint("TestClusters") << ", y " << iVertex->y();
          edm::LogPrint("TestClusters") << ", z " << iVertex->z();
          edm::LogPrint("TestClusters") << ", ndof " << iVertex->ndof();
          edm::LogPrint("TestClusters") << ", sumpt " << iVertex->p4().pt();
          edm::LogPrint("TestClusters");
        }  // print
      }    // for loop
    }      // if vertex

    hpvs->Fill(float(numPVsGood));
    hpvls->Fill(float(lumiBlock), float(numPVsGood));
    //if(instlumi>0.) {
    //float tmp = float(numPVsGood)/instlumi;
    //hpvlsn->Fill(float(lumiBlock),tmp);
    //}
  }  // if run

  int bxId = -1;
#ifdef BX
  //edm::LogPrint("TestClusters")<<" for bx "<<bx<<endl;
  bxId = getbx->find(bx);  // get the bunch type
  //edm::LogPrint("TestClusters")<<" id is "<<bxId<<endl;
#endif  // BX

#ifdef BX_NEW
  //edm::LogPrint("TestClusters")<<" for bx "<<bx<<endl;
  bxId = getBX::find(bx);  // get the bunch type
  //edm::LogPrint("TestClusters")<<" id is "<<bxId<<endl;
#endif  // BX_NEW

#if defined(BX) || defined(BX_NEW)
  if (bxId == 3)
    hbx1->Fill(float(bx));  // collision
  else if (bxId == 4)
    hbx2->Fill(float(bx));  // collision+1
  else if (bxId == 1)
    hbx3->Fill(float(bx));  // beam1
  else if (bxId == 2)
    hbx4->Fill(float(bx));  // beam2
  else if (bxId == 5 || bxId == 6)
    hbx5->Fill(float(bx));  // beam1,2+1
  else if (bxId == 0)
    hbx6->Fill(float(bx));  // empty

#endif

  // Get Cluster Collection from InputTag
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusters;
  // New By Token method
  e.getByToken(myClus, clusters);

  const edmNew::DetSetVector<SiPixelCluster> &input = *clusters;
  int numOf = input.size();

  //edm::LogPrint("TestClusters")<<numOf<<endl;
  //if(numOf<1) return; // skip events with no pixels

  //edm::LogPrint("TestClusters")<<"run "<<run<<" event "<<event<<" bx "<<bx<<" lumi "<<lumiBlock<<" orbit "<<orbit<<" "
  //  <<numOf<<" lumi "<<instlumi<<endl;

  if (PRINT)
    edm::LogPrint("TestClusters") << "run " << run << " event " << event << " bx " << bx << " lumi " << lumiBlock
                                  << " orbit " << orbit << " " << numOf << " lumi " << instlumi;

    // For L1
    //bool bit0=false; // , bit126=false, bit121=false,bit122=false;
    //bool bptx_m=false, bptx_p=false, bptxAnd=false,
    //bptx3=false, bptx4=false, bptx5=false, bptx6=false,bptx7=false;
    //bool  bcsOR=false, bit32_33=false, bit40=false, bit41=false, halo=false, splash1=false, splash2=false;

#ifdef L1
  // Get L1
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  e.getByToken(l1gtrrToken_, L1GTRR);

  if (L1GTRR.isValid()) {
    //bool l1a = L1GTRR->decision();  // global decission?  unused
    const L1GtPsbWord psb = L1GTRR->gtPsbWord(0xbb09, 0);  // select PSB#9 and bunch crossing 0
    int techLowNonPres = 65536 * psb.bData(0) + psb.aData(0);
    int techHighNonPres = 65536 * psb.bData(1) + psb.aData(1);

    //edm::LogPrint("TestClusters") << hex << "high word: " << ( 65536*psb.bData(1) + psb.aData(1) )
    // << " low word: " << ( 65536*psb.bData(0) + psb.aData(0) ) << " "
    // <<techLowNonPres<<" "<<techHighNonPres<<dec<<endl;

    //edm::LogPrint("TestClusters")<<" L1 status = "<<l1a<<" : ";
    for (unsigned int i = 0; i < L1GTRR->decisionWord().size(); ++i) {
      int l1flag = L1GTRR->decisionWord()[i];
      int t1flag = L1GTRR->technicalTriggerWord()[i];
      int techflag = 0;
      if (i < 32)
        techflag = (techLowNonPres & (0x1 << i));
      else if (i < 64)
        techflag = (techHighNonPres & (0x1 << i));

      if (l1flag > 0) {  // look at L1A algoritmic bits
        //edm::LogPrint("TestClusters")<<" la "<<i<<" ";
        hl1a->Fill(float(i));
        //if(i==0) bit0=true;
        // 3-pre, 4-bg_bsc, 5-bg_hf, 6-inter_bsc, 7-inter_hf, 8-halo
        //else if(i==121) bit121=true;
        //else if(i==122) bit122=true;
        //else if(i==126) bit126=true; //bscOR and bptx
      }
      if (techflag > 0 && i < 64) {  // look at nonprescaled technical bits
        //edm::LogPrint("TestClusters")<<" t1 "<<i<<" ";
        hl1t->Fill(float(i));  // non- prescaled
                               //if(i==0) bptxAnd=true;  // bptxAnd
                               //else if(i==1) bptx_p=true;
                               //else if(i==2) bptx_m=true;
                               //else if(i==3) bptx3=true; // OR
                               //else if(i==4) bptx4=true; // AND
                               //else if(i==5) bptx5=true; // p not m
                               //else if(i==6) bptx6=true; // m not p
                               //else if(i==7) bptx7=true; // quite
                               //else if(i==9) hf=true;
                               //else if(i==10) hf=true;
      }
      if (t1flag > 0 && i < 64) {  // look at prescaled L1T technical bits
        //edm::LogPrint("TestClusters")<<" lt "<<i<<" ";
        hltt->Fill(float(i));  // prescaled
      }

    }  // for loop
    //edm::LogPrint("TestClusters")<<dec<<endl;

  }  // if l1a
#endif

  //bool bptx_and = bptx_m && bptx_p;
  //bool bptx_or  = bptx_m || bptx_p;
  //bool bptx_xor = bptx_or && !bptx_and;

  //---------------------------------------
  // Analyse HLT
  //bool passHLT1=false,passHLT2=false,passHLT3=false,passHLT4=false,passHLT5=false;
  bool hlt[256];
  for (int i = 0; i < 256; ++i)
    hlt[i] = false;

#ifdef HLT

  edm::TriggerNames TrigNames;
  edm::Handle<edm::TriggerResults> HLTResults;

  // Extract the HLT results
  e.getByToken(hltToken_, HLTResults);
  if ((HLTResults.isValid() == true) && (HLTResults->size() > 0)) {
    //TrigNames.init(*HLTResults);
    const edm::TriggerNames &TrigNames = e.triggerNames(*HLTResults);

    //edm::LogPrint("TestClusters")<<TrigNames.triggerNames().size()<<endl;

    for (unsigned int i = 0; i < TrigNames.triggerNames().size(); i++) {  // loop over trigger
      if (countAllEvents == 1)
        edm::LogPrint("TestClusters") << i << " " << TrigNames.triggerName(i);

      if ((HLTResults->wasrun(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
          (HLTResults->accept(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
          (HLTResults->error(TrigNames.triggerIndex(TrigNames.triggerName(i))) == false)) {
        hlt[i] = true;
        hlt1->Fill(float(i));

        // 	if (      TrigNames.triggerName(i) == "HLT_L1Tech_BSC_minBias") passHLT1=true;
        // 	else if  (TrigNames.triggerName(i) == "HLT_L1Tech_BSC_minBias_OR") passHLT2=true;
        // 	else if  (TrigNames.triggerName(i) == "HLT_L1_BptxXOR_BscMinBiasOR")
        // 	  passHLT3=true;
        // 	else if  (TrigNames.triggerName(i) == "HLT_L1Tech_BSC_halo_forPhysicsBackground") passHLT4=true;
        // 	//
        // 	//else if  (TrigNames.triggerName(i) == "HLT_L1_BPTX") passHLT5=true;
        // 	else if  (TrigNames.triggerName(i) == "HLT_L1_ZeroBias") passHLT5=true;
        // 	//else if  (TrigNames.triggerName(i) == "HLT_L1_BPTX_MinusOnly") passHLT5=true;
        // 	//else if  (TrigNames.triggerName(i) == "HLT_L1_BPTX_PlusOnly") passHLT5=true;
        // 	else if ((TrigNames.triggerName(i) == "HLT_MinBiasPixel_SingleTrac") ||
        // 		 (TrigNames.triggerName(i) == "HLT_MinBiasPixel_DoubleTrack") ||
        // 		 (TrigNames.triggerName(i) == "HLT_MinBiasPixel_DoubleIsoTrack5") )
        //	  passHLT5=true;

      }  // if hlt

    }   // loop
  }     // if valid
#endif  // HLT

#ifdef USE_RESYNCS
  Handle<Level1TriggerScalersCollection> l1ts;
  e.getByToken(l1tsToken_, l1ts);

  if (l1ts->size() > 0) {
    int r1 = (*l1ts)[0].lastResync();
    // int r2 = (*l1ts)[0].lastOrbitCounter0();
    // int r3 = (*l1ts)[0].lastEventCounter0();
    // int r4 = (*l1ts)[0].lastHardReset();
    // int r5 = (*l1ts)[0].eventID();
    // int r6 = (*l1ts)[0].bunchNumber();
    // int r7 = (*l1ts)[0].lumiSegmentNr();
    // int r8 = (*l1ts)[0].lumiSegmentOrbits();
    // int r9 = (*l1ts)[0].orbitNr();
    // int r10 = (*l1ts)[0].lastStart();

    float t1 = r1 / 2.621E5;              // orbit to LS
    float t2 = (orbit - r1) * 88.924E-6;  // orbit diff (event-resync) to time in sec

    //edm::LogPrint("TestClusters")<<"run "<<run<<" event "<<event<<" bx "<<bx<<" lumi "<<lumiBlock<<" orbit "<<orbit<<" "
    //<<numOf<<" lumi "<<instlumi<<" - ";
    //edm::LogPrint("TestClusters") <<t1<<" "<<t2<<" "<<r1 <<" "<<r2 <<" "<<r3 <<" "<<r4 <<" "<<r5 <<" "<<r6 <<" "<<r7 <<" "<<r8 <<" "<<r9 <<" "<<r10<<endl;

    htest1->Fill(t1);
    htest2->Fill(t2);
    htest3->Fill(float(r1));
    htest4->Fill(float(orbit));
  }

#endif

  //----------------------------------------------

  // Select trigger bits  SELECT-EVENTS
  //if( bptx_xor || bptx_and ) return; // select no bptx events
  //if(!bptx_and ) return; // select coll  events
  //if(!bptx_xor ) return; // select single beams

  // if(select2>0 && select2<11) {
  //   if(select2==1 && !bptx_and) return;  // select bptx_and only
  //   if(select2==2 && !bptx_xor) return;  // select bptx_xor
  //   if(select2==3 && !bptx_or) return;  // select bptx_or
  //   if(select2==4 && !bptx_p) return;   // select bptix plus
  //   if(select2==5 && !bptx_m) return;  // select bptx minus
  //   if(select2==6 && bptx_or) return;  // select NO bptx
  // }

  hdets->Fill(float(numOf));  // number of modules with pix

  // Select events with pixels
  //if(numOf<1) return; // skip events with  pixel dets

  if (select1 <= 0) {
    if (numOf < 4)
      return;
  }  // skip events with few pixel dets
  else {
    if (numOf < select1)
      return;
  }  // skip events with few pixel dets

  hevent->Fill(float(event));
  hlumi->Fill(float(lumiBlock));
  hbx->Fill(float(bx));
  //horbit->Fill(float(orbit));
  for (unsigned int i = 0; i < 256; i++)
    if (hlt[i] == true)
      hlt2->Fill(float(i));

  if (run != runNumberOld) {
    runNumberOld = run;
    countRuns++;
  }
  switch (countRuns) {
    case 1: {
      hlumi10->Fill(float(lumiBlock));
      break;
    }
    case 2: {
      hlumi11->Fill(float(lumiBlock));
      break;
    }
    case 3: {
      hlumi12->Fill(float(lumiBlock));
      break;
    }
    case 4: {
      hlumi13->Fill(float(lumiBlock));
      break;
    }
    case 5: {
      hlumi14->Fill(float(lumiBlock));
      break;
    }
    //case 6: {hlumi15->Fill(float(lumiBlock)); break;}
    //case 7: {hlumi16->Fill(float(lumiBlock)); break;}
    //case 8: {hlumi17->Fill(float(lumiBlock)); break;}
    //case 9: {hlumi18->Fill(float(lumiBlock)); break;}
    //case 10: {hlumi19->Fill(float(lumiBlock)); break;}
    default:
      edm::LogPrint("TestClusters") << " too many runs " << countRuns;
  }

#ifdef NEW_ID
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo = es.getHandle(trackerTopoToken_);
#endif

  //---------------------------------------

  countEvents++;
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  int numberOfPixels = 0;
  int numberOfNoneEdgePixels = 0;
  int numberOfDetUnits1 = 0;
  int numOfClustersPerDet1 = 0;
  int numOfClustersPerLay1 = 0;
  int numberOfDetUnits2 = 0;
  int numOfClustersPerDet2 = 0;
  int numOfClustersPerLay2 = 0;
  int numberOfDetUnits3 = 0;
  int numOfClustersPerDet3 = 0;
  int numOfClustersPerLay3 = 0;

  int numOfPixPerLay1 = 0;
  int numOfPixPerLay2 = 0;
  int numOfPixPerLay3 = 0;

  int numOfPixPerDet1 = 0;
  int numOfPixPerDet2 = 0;
  int numOfPixPerDet3 = 0;

  int numOfPixPerLink11 = 0;
  int numOfPixPerLink12 = 0;
  int numOfPixPerLink21 = 0;
  int numOfPixPerLink22 = 0;
  //int numOfPixPerLink3=0;

  int maxClusPerDet = 0;
  int maxPixPerDet = 0;
  unsigned int maxPixPerClu = 0;

  int numOfClustersPerDisk1 = 0;
  int numOfClustersPerDisk2 = 0;
  int numOfClustersPerDisk3 = 0;
  int numOfClustersPerDisk4 = 0;
  int numOfPixPerDisk1 = 0;
  int numOfPixPerDisk2 = 0;
  int numOfPixPerDisk3 = 0;
  int numOfPixPerDisk4 = 0;

  //float avCharge1 = 0., avCharge2 = 0., avCharge3 = 0., avCharge4 = 0., avCharge5 = 0.;

  //static int module1[416][160] = {{0}};
  //static int module2[416][160] = {{0}};
  //static int module3[416][160] = {{0}};

  // get vector of detunit ids
  //--- Loop over detunits.
  edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter = input.begin();
  for (; DSViter != input.end(); DSViter++) {
    //bool valid = false;
    unsigned int detid = DSViter->detId();
    // Det id
    DetId detId = DetId(detid);             // Get the Detid object
    unsigned int detType = detId.det();     // det type, pixel=1
    unsigned int subid = detId.subdetId();  //subdetector type, barrel=1

    if (PRINT)
      edm::LogPrint("TestClusters") << "Det: " << detId.rawId() << " " << detId.null() << " " << detType << " "
                                    << subid;

#ifdef HISTOS
      //hdetunit->Fill(float(detid));
      //hpixid->Fill(float(detType));
      //hpixsubid->Fill(float(subid));
#endif  // HISTOS

    if (detType != 1)
      continue;  // look only at pixels
    ++numberOfDetUnits;

    //const GeomDetUnit * genericDet = geom->idToDet(detId);
    //const PixelGeomDetUnit * pixDet =
    //dynamic_cast<const PixelGeomDetUnit*>(genericDet);

    // Get the geom-detector
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit *>(theTracker.idToDet(detId));
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();

    //const BoundPlane& plane = theGeomDet->surface(); //for transf.

    //double detThick = theGeomDet->specificSurface().bounds().thickness();
    //int cols = theGeomDet->specificTopology().ncolumns();
    //int rows = theGeomDet->specificTopology().nrows();

    const PixelTopology *topol = &(theGeomDet->specificTopology());

    // barrel ids
    unsigned int layerC = 0;
    unsigned int ladderC = 0;
    unsigned int zindex = 0;
    int shell = 0;      // shell id // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
    int sector = 0;     // 1-8
    int ladder = 0;     // 1-22
    int layer = 0;      // 1-3
    int module = 0;     // 1-4
    bool half = false;  //

    // Endcap ids
    unsigned int disk = 0;     //1,2,3
    unsigned int blade = 0;    //1-24
    unsigned int zindexF = 0;  //
    unsigned int side = 0;     //size=1 for -z, 2 for +z
    unsigned int panel = 0;    //panel=1

    edmNew::DetSet<SiPixelCluster>::const_iterator clustIt;

    // Subdet id, pix barrel=1, forward=2
    if (subid == 2) {  // forward

#ifdef NEW_ID
      disk = tTopo->pxfDisk(detid);      //1,2,3
      blade = tTopo->pxfBlade(detid);    //1-24
      zindex = tTopo->pxfModule(detid);  //
      side = tTopo->pxfSide(detid);      //size=1 for -z, 2 for +z
      panel = tTopo->pxfPanel(detid);    //panel=1
#else
      PXFDetId pdetId = PXFDetId(detid);
      disk = pdetId.disk();       //1,2,3
      blade = pdetId.blade();     //1-24
      moduleF = pdetId.module();  // plaquette
      side = pdetId.side();       //size=1 for -z, 2 for +z
      panel = pdetId.panel();     //panel=1
#endif

      if (PRINT)
        edm::LogPrint("TestClusters") << " forward det, disk " << disk << ", blade " << blade << ", module " << zindexF
                                      << ", side " << side << ", panel " << panel << " pos = " << detZ << " " << detR;

    } else if (subid == 1) {  // barrel

#ifdef NEW_ID
      layerC = tTopo->pxbLayer(detid);
      ladderC = tTopo->pxbLadder(detid);
      zindex = tTopo->pxbModule(detid);
      PixelBarrelName pbn(detid);
#else
      PXBDetId pdetId = PXBDetId(detid);
      //unsigned int detTypeP=pdetId.det();
      //unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      layerC = pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladderC = pdetId.ladder();
      // Barrel Z-index=1,8
      zindex = pdetId.module();
      // Convert to online
      PixelBarrelName pbn(pdetId);
#endif

      // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
      PixelBarrelName::Shell sh = pbn.shell();  //enum
      sector = pbn.sectorName();
      ladder = pbn.ladderName();
      layer = pbn.layerName();
      module = pbn.moduleName();
      half = pbn.isHalfModule();
      shell = int(sh);
      // change the module sign for z<0
      if (shell == 1 || shell == 2)
        module = -module;
      // change ladeer sign for Outer )x<0)
      if (shell == 1 || shell == 3)
        ladder = -ladder;

      if (PRINT) {
        edm::LogPrint("TestClusters") << " Barrel layer, ladder, module " << layerC << " " << ladderC << " " << zindex
                                      << " " << sh << "(" << shell << ") " << sector << " " << layer << " " << ladder
                                      << " " << module << " " << half;
        //edm::LogPrint("TestClusters")<<" Barrel det, thick "<<detThick<<" "
        //  <<" layer, ladder, module "
        //  <<layer<<" "<<ladder<<" "<<zindex<<endl;
        //edm::LogPrint("TestClusters")<<" col/row, pitch "<<cols<<" "<<rows<<" "
        //  <<pitchX<<" "<<pitchY<<endl;
      }

    }  // if subid

    if (PRINT) {
      edm::LogPrint("TestClusters") << "List clusters : ";
      edm::LogPrint("TestClusters") << "Num Charge Size SizeX SizeY X Y Xmin Xmax Ymin Ymax Edge";
    }

    // Loop over clusters
    for (clustIt = DSViter->begin(); clustIt != DSViter->end(); clustIt++) {
      sumClusters++;
      numberOfClusters++;
      float ch = float(clustIt->charge()) / 1000.;  // convert ke to electrons
      int size = clustIt->size();
      int sizeX = clustIt->sizeX();  //x=row=rfi,
      int sizeY = clustIt->sizeY();  //y=col=z_global
      float x = clustIt->x();        // row, cluster position in pitch units, as float (int+0.5)
      float y = clustIt->y();        // col, analog average
      // Returns int index of the cluster min/max
      int minPixelRow = clustIt->minPixelRow();  //x
      int maxPixelRow = clustIt->maxPixelRow();
      int minPixelCol = clustIt->minPixelCol();  //y
      int maxPixelCol = clustIt->maxPixelCol();

      //unsigned int geoId = clustIt->geographicalId(); // always 0?!
      // edge method moved to topologu class
      bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || (topol->isItEdgePixelInX(maxPixelRow));
      bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || (topol->isItEdgePixelInY(maxPixelCol));

      bool edgeHitX2 = false;  // edge method moved
      bool edgeHitY2 = false;  // to topologu class

      if (PRINT)
        edm::LogPrint("TestClusters") << numberOfClusters << " " << ch << " " << size << " " << sizeX << " " << sizeY
                                      << " " << x << " " << y << " " << minPixelRow << " " << maxPixelRow << " "
                                      << minPixelCol << " " << maxPixelCol << " " << edgeHitX << " " << edgeHitY;

      // get global z position of teh cluster
      LocalPoint lp = topol->localPosition(MeasurementPoint(x, y));
      float lx = lp.x();  // local cluster position in cm
      float ly = lp.y();

      float zPos = detZ - ly;
      float rPos = detR + lx;

      // Get the pixels in the Cluster
      const vector<SiPixelCluster::Pixel> &pixelsVec = clustIt->pixels();
      if (PRINT)
        edm::LogPrint("TestClusters") << " Pixels in this cluster ";
      //bool bigInX=false, bigInY=false;
      // Look at pixels in this cluster. ADC is calibrated, in electrons
      bool edgeInX = false;  // edge method moved
      bool edgeInY = false;  // to topologu class
      //bool cluBigInX = false; // does this clu include a big pixel
      //bool cluBigInY = false; // does this clu include a big pixel
      //int noisy = 0;

      if (pixelsVec.size() > maxPixPerClu)
        maxPixPerClu = pixelsVec.size();

      for (unsigned int i = 0; i < pixelsVec.size(); ++i) {  // loop over pixels
        sumPixels++;
        numberOfPixels++;
        float pixx = pixelsVec[i].x;  // index as float=iteger, row index
        float pixy = pixelsVec[i].y;  // same, col index
        float adc = (float(pixelsVec[i].adc) / 1000.);
#ifdef ROC_EFF
        int roc = rocId(int(pixy), int(pixx));  // column, row
#endif
        //int chan = PixelChannelIdentifier::pixelToChannel(int(pixx),int(pixy));

        bool bigInX = topol->isItBigPixelInX(int(pixx));
        bool bigInY = topol->isItBigPixelInY(int(pixy));
        if (!(bigInX || bigInY))
          numberOfNoneEdgePixels++;

#ifdef HISTOS
        // Pixel histos
        if (subid == 1 && (selectEvent == -1 || countEvents == selectEvent)) {  // barrel
          if (layer == 1) {
            numOfPixPerDet1++;
            numOfPixPerLay1++;
            //valid = valid || true;
            hpixcharge1->Fill(adc);
            hpixDetMap1->Fill(pixy, pixx);
            hpDetMap1->Fill(float(module), float(ladder));
            //module1[int(pixx)][int(pixy)]++;

            hpcols1->Fill(pixy);
            hprows1->Fill(pixx);

            if (ladder == -1 && module == -1)
              hpixDetMap10->Fill(pixy, pixx);  // ineff
            //else if(ladder==-4 && module==-1) hpixDetMap11->Fill(pixy,pixx); // ineff
            //else if(ladder== 2 && module== 2) hpixDetMap12->Fill(pixy,pixx); // ineff
            else if (ladder == 2 && module == 8)
              hpixDetMap11->Fill(pixy, pixx);  // roc ineff (1run)
            else if (ladder == 2 && module == -2)
              hpixDetMap12->Fill(pixy, pixx);  // roc ineff (1 run)

            else if (ladder == -9 && module == 4)
              hpixDetMap13->Fill(pixy, pixx);  // ineff
            else if (ladder == -8 && module == -4)
              hpixDetMap14->Fill(pixy, pixx);  // bad al, ENE
            else if (ladder == 6 && module == 4)
              hpixDetMap15->Fill(pixy, pixx);  // pix 0,0
            else if (ladder == 9 && module == 4)
              hpixDetMap16->Fill(pixy, pixx);  // gain low off
            else if (ladder == -3 && module == -3)
              hpixDetMap17->Fill(pixy, pixx);  // bad col
            else if (ladder == -7 && module == -4)
              hpixDetMap18->Fill(pixy, pixx);  // bad col
            else if (ladder == -5 && module == -4)
              hpixDetMap19->Fill(pixy, pixx);  // low ROC eff (1 run)

            //if(module1[int(pixx)][int(pixy)]>MAX_CUT)
            //edm::LogPrint("TestClusters")<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
            //  <<pixx<<" "<<pixy<<" "<<module1[int(pixx)][int(pixy)]<<endl;

            if (pixx < 80.)
              numOfPixPerLink11++;
            else
              numOfPixPerLink12++;

            hpixchar1->Fill(zPos, adc);
            hcharPixbx->Fill(bx, adc);
            hcharPixls->Fill(lumiBlock, adc);
            hcharPixLumi->Fill(instlumi, adc);

#ifdef VDM_STUDIES
            hcharPixls1->Fill(lumiBlock, adc);
#endif

#if defined(BX) || defined(BX_NEW)
            if (bxId > -1) {
              if (bxId == 3)
                hpixChargebx1->Fill(adc);  // coll
              else if (bxId == 4)
                hpixChargebx2->Fill(adc);  // coll+1
              else if (bxId == 0)
                hpixChargebx6->Fill(adc);  // empty
              else if (bxId == 1)
                hpixChargebx3->Fill(adc);  // beam1
              else if (bxId == 2)
                hpixChargebx3->Fill(adc);  // beam2
              else if (bxId == 5 || bxId == 6)
                hpixChargebx5->Fill(adc);  // beam1/2+1
              else
                edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
            }
#endif

#ifdef ROC_EFF
            pixEff->addPixel(1, ladder, module, roc);  // count pixels

            //index = lumiBlock/5;
            // 	   if     (ladder== 3 && module== 3)  hmoduleHits1ls->Fill(float(lumiBlock),0);
            // 	   if     (ladder==-9 && module==-3 && roc== 9)  hrocHits1ls->Fill(float(lumiBlock),0);
            // 	   else if(ladder==-9 && module==-1 && roc== 0)  hrocHits1ls->Fill(float(lumiBlock),1);
            // 	   else if(ladder==-8 && module==-1 && roc== 2)  hrocHits1ls->Fill(float(lumiBlock),2);
            // 	   else if(ladder==-8 && module== 1 && roc== 2)  hrocHits1ls->Fill(float(lumiBlock),3);
            // 	   else if(ladder==-4 && module==-2 && roc== 2)  hrocHits1ls->Fill(float(lumiBlock),4);
            // 	   else if(ladder== 2 && module== 2 && roc== 4)  hrocHits1ls->Fill(float(lumiBlock),5);
            // 	   else if(ladder== 3 && module==-2 && roc== 4)  hrocHits1ls->Fill(float(lumiBlock),6);
            // 	   else if(ladder== 3 && module== 1 && roc== 6)  hrocHits1ls->Fill(float(lumiBlock),7);
            // 	   else if(ladder== 4 && module== 3 && roc== 8)  hrocHits1ls->Fill(float(lumiBlock),8);
            // 	   else if(ladder== 8 && module==-4 && roc==11)  hrocHits1ls->Fill(float(lumiBlock),9);
            // run 176286
            //if     (ladder==-3 && module== 1 && roc== 5)  hrocHits1ls->Fill(float(lumiBlock),0);
            //else if(ladder== 2 && module==-1 && roc==13)  hrocHits1ls->Fill(float(lumiBlock),1);
            //else if(ladder== 3 && module== 4 && roc== 9)  hrocHits1ls->Fill(float(lumiBlock),2);
            //else if(ladder== 5 && module== 2 && roc== 3)  hrocHits1ls->Fill(float(lumiBlock),3);
            //else if(ladder== 7 && module==-1 && roc== 8)  hrocHits1ls->Fill(float(lumiBlock),4);
            //else if(ladder== 2 && module== 1 && roc== 0)  hrocHits1ls->Fill(float(lumiBlock),5);
            // run 180250
            if (ladder == -9 && module == 4 && roc == 12)
              hrocHits1ls->Fill(float(lumiBlock), 0);
            else if (ladder == -7 && module == -4 && roc == 3)
              hrocHits1ls->Fill(float(lumiBlock), 1);
            else if (ladder == -5 && module == -4 && roc == 11)
              hrocHits1ls->Fill(float(lumiBlock), 2);
            else if (ladder == 2 && module == -2 && roc == 1)
              hrocHits1ls->Fill(float(lumiBlock), 3);
            else if (ladder == 8 && module == 2 && roc == 1)
              hrocHits1ls->Fill(float(lumiBlock), 4);
              //else if(ladder== 2 && module== 1 && roc== 0)  hrocHits1ls->Fill(float(lumiBlock),5);
#endif

          } else if (layer == 2) {
            numOfPixPerDet2++;
            numOfPixPerLay2++;

            hpcols2->Fill(pixy);
            hprows2->Fill(pixx);

            if (ladder == -11 && module == -2)
              hpixDetMap20->Fill(pixy, pixx);  // ineff
            else if (ladder == -7 && module == -3)
              hpixDetMap21->Fill(pixy, pixx);  // ineff
            else if (ladder == -2 && module == 1)
              hpixDetMap22->Fill(pixy, pixx);  // ineff
            else if (ladder == -14 && module == 4)
              hpixDetMap23->Fill(pixy, pixx);  // ineff
            else if (ladder == -14 && module == 1)
              hpixDetMap24->Fill(pixy, pixx);  // bad al
            else if (ladder == -14 && module == -1)
              hpixDetMap25->Fill(pixy, pixx);  // evn errors
            //else if(ladder== 14 && module== 2) hpixDetMap26->Fill(pixy,pixx); // gain cal poor eff
            else if (ladder == -4 && module == 4)
              hpixDetMap26->Fill(pixy, pixx);  // bad dcol
            else if (ladder == 6 && module == -2)
              hpixDetMap27->Fill(pixy, pixx);  // bad dcol
            //else if(ladder==-4  && module== 2) hpixDetMap28->Fill(pixy,pixx); // pix0

            else if (ladder == -10 && module == -3)
              hpixDetMap28->Fill(pixy, pixx);  // roc eff 1 run
            else if (ladder == 14 && module == 3)
              hpixDetMap29->Fill(pixy, pixx);  // "

            if (pixx < 80.)
              numOfPixPerLink21++;
            else
              numOfPixPerLink22++;

#if defined(BX) || defined(BX_NEW)
            if (bxId > -1) {
              if (bxId == 3)
                hpixChargebx1->Fill(adc);  // coll
              else if (bxId == 4)
                hpixChargebx2->Fill(adc);  // coll+1
              else if (bxId == 0)
                hpixChargebx6->Fill(adc);  // empty
              else if (bxId == 1)
                hpixChargebx3->Fill(adc);  // beam1
              else if (bxId == 2)
                hpixChargebx3->Fill(adc);  // beam2
              else if (bxId == 5 || bxId == 6)
                hpixChargebx5->Fill(adc);  // beam1/2+1
              else
                edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
            }
#endif

#ifdef ROC_EFF
            pixEff->addPixel(2, ladder, module, roc);

            //index = lumiBlock/5;
            // 	    if     (ladder==10 && module== 2)  hmoduleHits2ls->Fill(float(lumiBlock),0); // many resyncs

            // 	   if     (ladder==-2 && module== 1 && roc==10)  hrocHits2ls->Fill(float(lumiBlock),0);
            // 	   else if(ladder== 1 && module== 1 && roc== 3)  hrocHits2ls->Fill(float(lumiBlock),1);
            // 	   else if(ladder== 1 && module== 2 && roc== 4)  hrocHits2ls->Fill(float(lumiBlock),2);
            // 	   else if(ladder== 2 && module==-4 && roc== 3)  hrocHits2ls->Fill(float(lumiBlock),3);
            // 	   else if(ladder==10 && module==-3 && roc== 1)  hrocHits2ls->Fill(float(lumiBlock),4);
            // 	   else if(ladder==10 && module==-1 && roc==10)  hrocHits2ls->Fill(float(lumiBlock),5);
            // 	   else if(ladder==11 && module== 3 && roc==13)  hrocHits2ls->Fill(float(lumiBlock),6);
            // 	   else if(ladder==12 && module==-4 && roc==14)  hrocHits2ls->Fill(float(lumiBlock),7);
            // 	   else if(ladder==12 && module==-1 && roc== 3)  hrocHits2ls->Fill(float(lumiBlock),8);
            // 	   else if(ladder==-5 && module==-2 && roc==10)  hrocHits2ls->Fill(float(lumiBlock),9);
            // 	   else if(ladder==10 && module== 2 && roc== 0)  hrocHits3ls->Fill(float(lumiBlock),9); // many resuncs

            // run 176286
            //if     (ladder==-5 && module==-2 && roc==10)  hrocHits2ls->Fill(float(lumiBlock),0);
            //else if(ladder==-2 && module==-4 && roc==12)  hrocHits2ls->Fill(float(lumiBlock),1);
            //else if(ladder==15 && module==-1 && roc==11)  hrocHits2ls->Fill(float(lumiBlock),2);
            //else if(ladder== 2 && module==1  && roc== 0)  hrocHits2ls->Fill(float(lumiBlock),3);
            // run 180250
            if (ladder == -2 && module == 1 && roc == 10)
              hrocHits2ls->Fill(float(lumiBlock), 0);
            else if (ladder == 10 && module == -3 && roc == 2)
              hrocHits2ls->Fill(float(lumiBlock), 1);
            else if (ladder == 15 && module == 3 && roc == 7)
              hrocHits2ls->Fill(float(lumiBlock), 2);
              //else if(ladder== 2 && module==1  && roc== 0)  hrocHits2ls->Fill(float(lumiBlock),3);
#endif

            hpixcharge2->Fill(adc);
            hpixDetMap2->Fill(pixy, pixx);
            hpDetMap2->Fill(float(module), float(ladder));
            //module2[int(pixx)][int(pixy)]++;

            hpixchar2->Fill(zPos, adc);
            hcharPixbx->Fill(bx, adc);
            hcharPixls->Fill(lumiBlock, adc);
            hcharPixLumi->Fill(instlumi, adc);

#ifdef VDM_STUDIES
            hcharPixls2->Fill(lumiBlock, adc);
#endif
            //if( (ladder==-11 && module==-1) || (ladder==-11 && module== 1) ) {
            //hpixchargen->Fill(adc);
            //if(       ladder==-11 && module==-1 ) hpixchargen1->Fill(adc);
            //else if ( ladder==-11 && module== 1 ) hpixchargen2->Fill(adc);
            //}

            //if(module2[int(pixx)][int(pixy)]>MAX_CUT)
            // edm::LogPrint("TestClusters")<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
            //  <<pixx<<" "<<pixy<<" "<<module2[int(pixx)][int(pixy)]<<endl;

          } else if (layer == 3) {
            numOfPixPerDet3++;
            numOfPixPerLay3++;
            //valid = valid || true;
            hpixcharge3->Fill(adc);
            hpixDetMap3->Fill(pixy, pixx);
            //if(ladder==l3ldr&&module==l3mod) hpixDetMap30->Fill(pixy,pixx,adc);
            hpDetMap3->Fill(float(module), float(ladder));
            //module3[int(pixx)][int(pixy)]++;

            hpcols3->Fill(pixy);
            hprows3->Fill(pixx);

            if (ladder == -5 && module == -4)
              hpixDetMap30->Fill(pixy, pixx);  // ineff
            else if (ladder == 9 && module == 4)
              hpixDetMap31->Fill(pixy, pixx);  // adr errors, NOR
            else if (ladder == 15 && module == -3)
              hpixDetMap32->Fill(pixy, pixx);  // ineff
            else if (ladder == 17 && module == -4)
              hpixDetMap33->Fill(pixy, pixx);  // ineff, NOR
            //else if(ladder== 6   && module== 4) hpixDetMap34->Fill(pixy,pixx); // gain ineff
            //else if(ladder==-14  && module==-3) hpixDetMap35->Fill(pixy,pixx); // gain low slope
            else if (ladder == -6 && module == -1)
              hpixDetMap34->Fill(pixy, pixx);  // roc eff 1 run
            else if (ladder == -13 && module == 4)
              hpixDetMap35->Fill(pixy, pixx);  // "
            else if (ladder == 14 && module == -4)
              hpixDetMap36->Fill(pixy, pixx);  // ineff pixel alive
            else if (ladder == -6 && module == 2)
              hpixDetMap37->Fill(pixy, pixx);  // E pattern
            else if (ladder == 19 && module == -4)
              hpixDetMap38->Fill(pixy, pixx);  // large thr.
            //else if(ladder==-8   && module==-1) hpixDetMap39->Fill(pixy,pixx); // large thr rms
            //else if(ladder== 11 && module== 1) hpixDetMap39->Fill(pixy,pixx); // old ROCs
            else if (ladder == -7 && module == -4)
              hpixDetMap39->Fill(pixy, pixx);  // roc eff (1 run)

            hpixchar3->Fill(zPos, adc);
            hcharPixbx->Fill(bx, adc);
            hcharPixls->Fill(lumiBlock, adc);
            hcharPixLumi->Fill(instlumi, adc);

            // 	    if(     countEvents== 66  && ladder==-17 && module== 4) hpixDetMap300->Fill(pixy,pixx);
            // 	    else if(countEvents==502  && ladder== 14 && module== 3) hpixDetMap301->Fill(pixy,pixx);
            // 	    else if(countEvents==1830 && ladder== 14 && module==-2) hpixDetMap302->Fill(pixy,pixx);
            // 	    else if(countEvents==4720 && ladder== 12 && module==-3) hpixDetMap303->Fill(pixy,pixx);
            // 	    else if(countEvents==7241 && ladder== -2 && module== 4) hpixDetMap304->Fill(pixy,pixx);
            // 	    else if(countEvents==8845 && ladder== -5 && module== 4) hpixDetMap305->Fill(pixy,pixx);
            // 	    else if(countEvents==9267 && ladder==  6 && module== 4) hpixDetMap306->Fill(pixy,pixx);

            //if(module3[int(pixx)][int(pixy)]>MAX_CUT)
            //edm::LogPrint("TestClusters")<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
            //  <<pixx<<" "<<pixy<<" "<<module3[int(pixx)][int(pixy)]<<endl;

#ifdef VDM_STUDIES
            hcharPixls3->Fill(lumiBlock, adc);
#endif

#if defined(BX) || defined(BX_NEW)
            if (bxId > -1) {
              if (bxId == 3)
                hpixChargebx1->Fill(adc);  // coll
              else if (bxId == 4)
                hpixChargebx2->Fill(adc);  // coll+1
              else if (bxId == 0)
                hpixChargebx6->Fill(adc);  // empty
              else if (bxId == 1)
                hpixChargebx3->Fill(adc);  // beam1
              else if (bxId == 2)
                hpixChargebx3->Fill(adc);  // beam2
              else if (bxId == 5 || bxId == 6)
                hpixChargebx5->Fill(adc);  // beam1/2+1
              else
                edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
            }
#endif

#ifdef ROC_EFF
            pixEff->addPixel(3, ladder, module, roc);

            //index = lumiBlock/5;
            // 	    if     (ladder==11 && module== 1)  hmoduleHits3ls->Fill(float(lumiBlock),0);

            // 	    if     (ladder==-18 && module==-3 && roc== 0)  hrocHits3ls->Fill(float(lumiBlock),0);
            // 	    else if(ladder== 5  && module== 1 && roc== 5)  hrocHits3ls->Fill(float(lumiBlock),1);
            // 	    else if(ladder== 9  && module==-1 && roc== 2)  hrocHits3ls->Fill(float(lumiBlock),2);
            // 	    else if(ladder==10  && module==-3 && roc== 8)  hrocHits3ls->Fill(float(lumiBlock),3);
            // 	    else if(ladder==12  && module==-3 && roc==12)  hrocHits3ls->Fill(float(lumiBlock),4);
            // 	    else if(ladder==-18 && module==-2 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),5);
            //else if(ladder==-3  && module==-3 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),6);
            //else if(ladder==-3  && module==-3 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),7);
            //else if(ladder==-3  && module==-3 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),8);
            //else if(ladder==-3  && module==-3 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),9);

            // run 176286
            //if     (ladder==-18 && module==-2 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),0);
            //else if(ladder==-3  && module==-3 && roc==14)  hrocHits3ls->Fill(float(lumiBlock),1);
            //else if(ladder== 3  && module==-1 && roc==12)  hrocHits3ls->Fill(float(lumiBlock),2);
            //else if(ladder==16  && module==-3 && roc== 6)  hrocHits3ls->Fill(float(lumiBlock),3);
            //else if(ladder== 2  && module== 1 && roc== 0)  hrocHits3ls->Fill(float(lumiBlock),4);
            // run 180250
            if (ladder == -13 && module == 4 && roc == 12)
              hrocHits3ls->Fill(float(lumiBlock), 0);
            else if (ladder == -7 && module == -4 && roc == 15)
              hrocHits3ls->Fill(float(lumiBlock), 1);
            else if (ladder == -6 && module == -2 && roc == 10)
              hrocHits3ls->Fill(float(lumiBlock), 2);
              //else if(ladder==16  && module==-3 && roc== 6)  hrocHits3ls->Fill(float(lumiBlock),3);
              //else if(ladder== 2  && module== 1 && roc== 0)  hrocHits3ls->Fill(float(lumiBlock),4);

#endif

          }  // if layer

        } else if (subid == 2 && (selectEvent == -1 || countEvents == selectEvent)) {  // endcap
          // pixels

          if (disk == 1) {  // disk1 -+z
            if (side == 1)
              numOfPixPerDisk2++;  // d1,-z
            else if (side == 2)
              numOfPixPerDisk3++;  // d1, +z
            else
              edm::LogPrint("TestClusters") << " unknown side " << side;

            hpixcharge4->Fill(adc);
            hpixDiskR1->Fill(rPos);

          } else if (disk == 2) {  // disk2 -+z

            if (side == 1)
              numOfPixPerDisk1++;  // d2, -z
            else if (side == 2)
              numOfPixPerDisk4++;  // d2, +z
            else
              edm::LogPrint("TestClusters") << " unknown side " << side;

            hpixcharge5->Fill(adc);
            hpixDiskR2->Fill(rPos);

          } else
            edm::LogPrint("TestClusters") << " unknown disk " << disk;

        }  // end if subdet (pixel loop)

#endif  // HISTOS

        // Find if this cluster includes an edge pixel
        edgeInX = topol->isItEdgePixelInX(int(pixx));
        edgeInY = topol->isItEdgePixelInY(int(pixy));

        if (PRINT)
          edm::LogPrint("TestClusters") << i << " " << pixx << " " << pixy << " " << adc << " " << bigInX << " "
                                        << bigInY << " " << edgeInX << " " << edgeInY;

        if (edgeInX)
          edgeHitX2 = true;
        if (edgeInY)
          edgeHitY2 = true;
        //if(bigInX) cluBigInX=true;
        //if(bigInY) cluBigInY=true;

      }  // pixel loop

#ifdef HISTOS

      // Cluster histos
      if (subid == 1 && (selectEvent == -1 || countEvents == selectEvent)) {  // barrel
        //if (subid==1) {  // barrel

        if (layer == 1) {  // layer 1

          hDetMap1->Fill(float(module), float(ladder));
          hsizeDetMap1->Fill(float(module), float(ladder), float(size));
          //hsizeXDetMap1->Fill(float(module),float(ladder),float(sizeX));
          //hsizeYDetMap1->Fill(float(module),float(ladder),float(sizeY));

          hcluDetMap1->Fill(y, x);
          hcharge1->Fill(ch);
          //hcols1->Fill(y);
          //hrows1->Fill(x);
          hsize1->Fill(float(size));
          hsizex1->Fill(float(sizeX));
          hsizey1->Fill(float(sizeY));
          numOfClustersPerDet1++;
          numOfClustersPerLay1++;
          //avCharge1 += ch;

          //if(numOf<10) hcharge11->Fill(ch);
          //else if(numOf<25) hcharge12->Fill(ch);
          //else if(numOf<100) hcharge13->Fill(ch);
          //else hcharge14->Fill(ch);

          hgz1->Fill(zPos);
          hclumult1->Fill(zPos, size);
          hclumultx1->Fill(zPos, sizeX);
          hclumulty1->Fill(zPos, sizeY);
          hcluchar1->Fill(zPos, ch);

#if defined(BX) || defined(BX_NEW)
          if (bxId > -1) {
            if (bxId == 3) {
              hchargebx1->Fill(ch);
              hsizebx1->Fill(size);
            }  // coll
            else if (bxId == 4) {
              hchargebx2->Fill(ch);
              hsizebx2->Fill(size);
            }  // coll+1
            else if (bxId == 0) {
              hchargebx6->Fill(ch);
              hsizebx6->Fill(size);
            }  // empty
            else if (bxId == 1) {
              hchargebx3->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam1
            else if (bxId == 2) {
              hchargebx4->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam2
            else if (bxId == 5 || bxId == 6) {
              hchargebx5->Fill(ch);
              hsizebx5->Fill(size);
            }  // beam1/2+1
            else
              edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
          }
          //if(bxId==0) htestbx->Fill(bx,ch);
#endif

          hcharClubx->Fill(bx, ch);
          hsizeClubx->Fill(bx, size);
          hsizeYClubx->Fill(bx, sizeY);
          hcharCluls->Fill(lumiBlock, ch);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);

          hcharCluLumi->Fill(instlumi, ch);
          hsizeCluLumi->Fill(instlumi, size);
          hsizeXCluLumi->Fill(instlumi, sizeX);
          hsizeYCluLumi->Fill(instlumi, sizeY);

          //hchargen5->Fill(ch,float(size));
          //hchargen3->Fill(ch);
          //if(size==1) hchargen4->Fill(ch);

#ifdef VDM_STUDIES
          hcharCluls1->Fill(lumiBlock, ch);
          hsizeCluls1->Fill(lumiBlock, size);
          hsizeXCluls1->Fill(lumiBlock, sizeX);
#endif

        } else if (layer == 2) {
          hDetMap2->Fill(float(module), float(ladder));
          hsizeDetMap2->Fill(float(module), float(ladder), float(size));
          //hsizeXDetMap2->Fill(float(module),float(ladder),float(sizeX));
          //hsizeYDetMap2->Fill(float(module),float(ladder),float(sizeY));

          hcluDetMap2->Fill(y, x);
          hcharge2->Fill(ch);
          //hcols2->Fill(y);
          //hrows2->Fill(x);
          hsize2->Fill(float(size));
          hsizex2->Fill(float(sizeX));
          hsizey2->Fill(float(sizeY));
          numOfClustersPerDet2++;
          numOfClustersPerLay2++;
          //avCharge2 += ch;
          hgz2->Fill(zPos);
          hclumult2->Fill(zPos, size);
          hclumultx2->Fill(zPos, sizeX);
          hclumulty2->Fill(zPos, sizeY);
          hcluchar2->Fill(zPos, ch);

          if ((ladder == -11 && module == -1) || (ladder == -11 && module == 1)) {
            //hchargen->Fill(ch);
            //hsizen->Fill(float(size));
            //hsizexn->Fill(float(sizeX));
            //hsizeyn->Fill(float(sizeY));
            // if( (ladder==-11 && module==-1) ) {
            //   hchargen1->Fill(ch);
            //   hsizen1->Fill(float(size));
            //   hsizexn1->Fill(float(sizeX));
            //   hsizeyn1->Fill(float(sizeY));
            // } else {
            //   hchargen2->Fill(ch);
            //   hsizen2->Fill(float(size));
            //   hsizexn2->Fill(float(sizeX));
            //   hsizeyn2->Fill(float(sizeY));
            // }
          }

#if defined(BX) || defined(BX_NEW)
          if (bxId > -1) {
            if (bxId == 3) {
              hchargebx1->Fill(ch);
              hsizebx1->Fill(size);
            }  // coll
            else if (bxId == 4) {
              hchargebx2->Fill(ch);
              hsizebx2->Fill(size);
            }  // coll+1
            else if (bxId == 0) {
              hchargebx6->Fill(ch);
              hsizebx6->Fill(size);
            }  // empty
            else if (bxId == 1) {
              hchargebx3->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam1
            else if (bxId == 2) {
              hchargebx4->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam2
            else if (bxId == 5 || bxId == 6) {
              hchargebx5->Fill(ch);
              hsizebx5->Fill(size);
            }  // beam1/2+1
            else
              edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
          }
#endif

          hcharClubx->Fill(bx, ch);
          hsizeClubx->Fill(bx, size);
          hsizeYClubx->Fill(bx, sizeY);
          hcharCluls->Fill(lumiBlock, ch);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);
          hcharCluLumi->Fill(instlumi, ch);
          hsizeCluLumi->Fill(instlumi, size);
          hsizeXCluLumi->Fill(instlumi, sizeX);
          hsizeYCluLumi->Fill(instlumi, sizeY);

          //hchargen5->Fill(ch,float(size));
          //hchargen3->Fill(ch);
          //if(size==1) hchargen4->Fill(ch);

#ifdef VDM_STUDIES
          hcharCluls2->Fill(lumiBlock, ch);
          hsizeCluls2->Fill(lumiBlock, size);
          hsizeXCluls2->Fill(lumiBlock, sizeX);
#endif

        } else if (layer == 3) {
          hDetMap3->Fill(float(module), float(ladder));
          hsizeDetMap3->Fill(float(module), float(ladder), float(size));
          //hsizeXDetMap3->Fill(float(module),float(ladder),float(sizeX));
          //hsizeYDetMap3->Fill(float(module),float(ladder),float(sizeY));

          hcluDetMap3->Fill(y, x);
          hcharge3->Fill(ch);
          //hcols3->Fill(y);
          //hrows3->Fill(x);
          hsize3->Fill(float(size));
          hsizex3->Fill(float(sizeX));
          hsizey3->Fill(float(sizeY));
          numOfClustersPerDet3++;
          numOfClustersPerLay3++;
          //avCharge3 += ch;
          hgz3->Fill(zPos);
          hclumult3->Fill(zPos, size);
          hclumultx3->Fill(zPos, sizeX);
          hclumulty3->Fill(zPos, sizeY);
          hcluchar3->Fill(zPos, ch);

#if defined(BX) || defined(BX_NEW)
          if (bxId > -1) {
            if (bxId == 3) {
              hchargebx1->Fill(ch);
              hsizebx1->Fill(size);
            }  // coll
            else if (bxId == 4) {
              hchargebx2->Fill(ch);
              hsizebx2->Fill(size);
            }  // coll+1
            else if (bxId == 0) {
              hchargebx6->Fill(ch);
              hsizebx6->Fill(size);
            }  // empty
            else if (bxId == 1) {
              hchargebx3->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam1
            else if (bxId == 2) {
              hchargebx4->Fill(ch);
              hsizebx3->Fill(size);
            }  // beam2
            else if (bxId == 5 || bxId == 6) {
              hchargebx5->Fill(ch);
              hsizebx5->Fill(size);
            }  // beam1/2+1
            else
              edm::LogPrint("TestClusters") << " wrong bx id " << bxId;
          }
#endif

          hcharClubx->Fill(bx, ch);
          hsizeClubx->Fill(bx, size);
          hsizeYClubx->Fill(bx, sizeY);
          hcharCluls->Fill(lumiBlock, ch);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);
          hcharCluLumi->Fill(instlumi, ch);
          hsizeCluLumi->Fill(instlumi, size);
          hsizeXCluLumi->Fill(instlumi, sizeX);
          hsizeYCluLumi->Fill(instlumi, sizeY);

          //hchargen5->Fill(ch,float(size));
          //hchargen3->Fill(ch);
          //if(size==1) hchargen4->Fill(ch);

#ifdef VDM_STUDIES
          hcharCluls3->Fill(lumiBlock, ch);
          hsizeCluls3->Fill(lumiBlock, size);
          hsizeXCluls3->Fill(lumiBlock, sizeX);
#endif

        }  // end if layer

      } else if (subid == 2 && (selectEvent == -1 || countEvents == selectEvent)) {  // endcap

        //edm::LogPrint("TestClusters")<<disk<<" "<<side<<endl;
        if (disk == 1) {  // disk1 -+z
          if (side == 1) {
            numOfClustersPerDisk2++;  // d1,-z
            //avCharge4 += ch;
          } else if (side == 2) {
            numOfClustersPerDisk3++;  // d1, +z
            //avCharge5 += ch;
          } else
            edm::LogPrint("TestClusters") << " unknown side " << side;

          hcharge4->Fill(ch);

        } else if (disk == 2) {  // disk2 -+z

          if (side == 1) {
            numOfClustersPerDisk1++;  // d2, -z
            //avCharge4 += ch;
          } else if (side == 2) {
            numOfClustersPerDisk4++;  // d2, +z
            //avCharge5 += ch;
          } else
            edm::LogPrint("TestClusters") << " unknown side " << side;

          hcharge5->Fill(ch);

        } else
          edm::LogPrint("TestClusters") << " unknown disk " << disk;

      }  // end barrel/forward cluster loop

#endif  // HISTOS

      if ((edgeHitX != edgeHitX2) && sizeX < 64)
        edm::LogPrint("TestClusters") << " wrong egdeX " << edgeHitX << " " << edgeHitX2;
      if ((edgeHitY != edgeHitY2) && sizeY < 64)
        edm::LogPrint("TestClusters") << " wrong egdeY " << edgeHitY << " " << edgeHitY2;

    }  // clusters

    if (numOfClustersPerDet1 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet1;
    if (numOfClustersPerDet2 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet2;
    if (numOfClustersPerDet3 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet3;

    if (PRINT) {
      if (layer == 1)
        edm::LogPrint("TestClusters") << "Lay1: number of clusters per det = " << numOfClustersPerDet1;
      else if (layer == 2)
        edm::LogPrint("TestClusters") << "Lay2: number of clusters per det = " << numOfClustersPerDet1;
      else if (layer == 3)
        edm::LogPrint("TestClusters") << "Lay3: number of clusters per det = " << numOfClustersPerDet1;
    }  // end if PRINT

#ifdef HISTOS
    if (subid == 1 && (selectEvent == -1 || countEvents == selectEvent)) {  // barrel
      //if (subid==1 && countEvents==selectEvent) {  // barrel

      //hlayerid->Fill(float(layer));

      // Det histos
      if (layer == 1) {
        hladder1id->Fill(float(ladder));
        hz1id->Fill(float(module));
        ++numberOfDetUnits1;
        hclusPerDet1->Fill(float(numOfClustersPerDet1));
        hpixPerDet1->Fill(float(numOfPixPerDet1));
        if (numOfPixPerDet1 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet1;
        //if(numOfPixPerLink11>798 || numOfPixPerLink12>798) select=true;
        hpixPerLink1->Fill(float(numOfPixPerLink11));
        hpixPerLink1->Fill(float(numOfPixPerLink12));
        hDetsMap1->Fill(float(module), float(ladder));

        if (abs(module) == 1)
          hpixPerDet11->Fill(float(numOfPixPerDet1));
        else if (abs(module) == 2)
          hpixPerDet12->Fill(float(numOfPixPerDet1));
        else if (abs(module) == 3)
          hpixPerDet13->Fill(float(numOfPixPerDet1));
        else if (abs(module) == 4)
          hpixPerDet14->Fill(float(numOfPixPerDet1));

        numOfClustersPerDet1 = 0;
        numOfPixPerDet1 = 0;
        numOfPixPerLink11 = 0;
        numOfPixPerLink12 = 0;

      } else if (layer == 2) {
        hladder2id->Fill(float(ladder));
        hz2id->Fill(float(module));
        ++numberOfDetUnits2;
        hclusPerDet2->Fill(float(numOfClustersPerDet2));
        hpixPerDet2->Fill(float(numOfPixPerDet2));
        if (numOfPixPerDet2 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet2;
        hpixPerLink2->Fill(float(numOfPixPerLink21));
        hpixPerLink2->Fill(float(numOfPixPerLink22));
        hDetsMap2->Fill(float(module), float(ladder));

        if (abs(module) == 1)
          hpixPerDet21->Fill(float(numOfPixPerDet2));
        else if (abs(module) == 2)
          hpixPerDet22->Fill(float(numOfPixPerDet2));
        else if (abs(module) == 3)
          hpixPerDet23->Fill(float(numOfPixPerDet2));
        else if (abs(module) == 4)
          hpixPerDet24->Fill(float(numOfPixPerDet2));

        numOfClustersPerDet2 = 0;
        numOfPixPerDet2 = 0;
        numOfPixPerLink21 = 0;
        numOfPixPerLink22 = 0;

      } else if (layer == 3) {
        hladder3id->Fill(float(ladder));
        hz3id->Fill(float(module));
        ++numberOfDetUnits3;
        hclusPerDet3->Fill(float(numOfClustersPerDet3));
        hpixPerDet3->Fill(float(numOfPixPerDet3));
        if (numOfPixPerDet3 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet3;
        hDetsMap3->Fill(float(module), float(ladder));

        if (abs(module) == 1)
          hpixPerDet31->Fill(float(numOfPixPerDet3));
        else if (abs(module) == 2)
          hpixPerDet32->Fill(float(numOfPixPerDet3));
        else if (abs(module) == 3)
          hpixPerDet33->Fill(float(numOfPixPerDet3));
        else if (abs(module) == 4)
          hpixPerDet34->Fill(float(numOfPixPerDet3));

        // 	if(      module==-2&& ladder==13)  hpixPerDet100->Fill(float(numOfPixPerDet3));
        // 	else if( module==+1&& ladder==11)  hpixPerDet101->Fill(float(numOfPixPerDet3));
        // 	else if( module==-1&& ladder==9 )  hpixPerDet102->Fill(float(numOfPixPerDet3));
        // 	else if( module==-4&& ladder==11)  hpixPerDet103->Fill(float(numOfPixPerDet3));
        // 	else if( module==-2&& ladder==11)  hpixPerDet104->Fill(float(numOfPixPerDet3));
        // 	else if( module==-2&& ladder==19)  hpixPerDet105->Fill(float(numOfPixPerDet3));

        // 	if( numOfPixPerDet3>191 ) {
        // 	  edm::LogPrint("TestClusters")<<" Layer 3 module "<<ladder<<" "<<module<<" "<<numOfPixPerDet3<<endl;
        // 	  select = true;
        // 	}

        numOfClustersPerDet3 = 0;
        numOfPixPerDet3 = 0;
        //numOfPixPerLink3=0;

      }  // layer

    }  // end barrel/forward

#endif  // HISTOS

  }  // detunits loop

  if (PRINT || countEvents == selectEvent) {  //
    edm::LogPrint("TestClusters") << "run " << run << " event " << event << " bx " << bx << " lumi " << lumiBlock
                                  << " orbit " << orbit << " " << countEvents;
    edm::LogPrint("TestClusters") << "Num of pix " << numberOfPixels << " num of clus " << numberOfClusters
                                  << " max clus per det " << maxClusPerDet << " max pix per clu " << maxPixPerClu
                                  << " count " << countEvents;
    edm::LogPrint("TestClusters") << "Number of clusters per Lay1,2,3: " << numOfClustersPerLay1 << " "
                                  << numOfClustersPerLay2 << " " << numOfClustersPerLay3;
    edm::LogPrint("TestClusters") << "Number of pixels per Lay1,2,3: " << numOfPixPerLay1 << " " << numOfPixPerLay2
                                  << " " << numOfPixPerLay3;
    edm::LogPrint("TestClusters") << "Number of dets with clus in Lay1,2,3: " << numberOfDetUnits1 << " "
                                  << numberOfDetUnits2 << " " << numberOfDetUnits3;
  }  // if PRINT

#ifdef HISTOS

  //if(numberOfClusters<=3) continue; // skip events
  if ((selectEvent == -1 || countEvents == selectEvent)) {
    hdets2->Fill(float(numOf));  // number of modules with pix
    hlumi1->Fill(float(lumiBlock));

    //if(bptx_m && bptx_xor) hbx7->Fill(float(bx));
    //if(bptx_p && bptx_xor) hbx8->Fill(float(bx));
    //if(bptx_and) hbx9->Fill(float(bx));
    //if(bptx_or)  hbx10->Fill(float(bx));

    hdigis->Fill(float(numberOfPixels));   // all pix
    hdigis2->Fill(float(numberOfPixels));  // same zoomed

    int pixb = numOfPixPerLay1 + numOfPixPerLay2 + numOfPixPerLay3;
    hdigisB->Fill(float(pixb));  // pix in bpix
    int pixf = numOfPixPerDisk1 + numOfPixPerDisk2 + numOfPixPerDisk3 + numOfPixPerDisk4;
    hdigisF->Fill(float(pixf));  // pix in fpix

    hclus->Fill(float(numberOfClusters));   // clusters fpix+bpix
    hclus2->Fill(float(numberOfClusters));  // same, zoomed

    //h2clupv->Fill(float(numberOfClusters),float(numPVsGood));
    //h2pixpv->Fill(float(numberOfPixels),float(numPVsGood));
    hclupv->Fill(float(numberOfClusters), float(numPVsGood));
    hpixpv->Fill(float(numberOfPixels), float(numPVsGood));
    //if(numPVsGood>0) {
    //float tmp = float(numberOfClusters)/float(numPVsGood);
    //hclupvlsn->Fill(float(lumiBlock),tmp);
    //tmp = float(numberOfPixels)/float(numPVsGood);
    //hpixpvlsn->Fill(float(lumiBlock),tmp);
    //}

    int clusb = numOfClustersPerLay1 + numOfClustersPerLay2 + numOfClustersPerLay3;
    hclusBPix->Fill(float(clusb));  // clusters in bpix

    int clusf = numOfClustersPerDisk1 + numOfClustersPerDisk2 + numOfClustersPerDisk3 + numOfClustersPerDisk4;
    hclusFPix->Fill(float(clusf));  // clusters in fpix

    hclus5->Fill(float(numberOfNoneEdgePixels));               // count none edge pixels
    hclusls->Fill(float(lumiBlock), float(numberOfClusters));  // clusters fpix+bpix
    hpixls->Fill(float(lumiBlock), float(numberOfPixels));     // pixels fpix+bpix

    hclubx->Fill(float(bx), float(numberOfClusters));  // clusters fpix+bpix
    hpixbx->Fill(float(bx), float(numberOfPixels));    // pixels fpix+bpix
    hpvbx->Fill(float(bx), float(numPVsGood));         // pvs

#ifdef VDM_STUDIES
    hclusls1->Fill(float(lumiBlock), float(numOfClustersPerLay1));  // clusters bpix1
    hpixls1->Fill(float(lumiBlock), float(numOfPixPerLay1));        // pixels bpix1
    hclusls2->Fill(float(lumiBlock), float(numOfClustersPerLay2));  // clusters bpix2
    hpixls2->Fill(float(lumiBlock), float(numOfPixPerLay2));        // pixels bpix2
    hclusls3->Fill(float(lumiBlock), float(numOfClustersPerLay3));  // clusters bpix3
    hpixls3->Fill(float(lumiBlock), float(numOfPixPerLay3));        // pixels bpix3
#endif

    if (instlumi > 0.) {
      hcluLumi->Fill(instlumi, float(numberOfClusters));  // clus
      hpixLumi->Fill(instlumi, float(numberOfPixels));    // pix

#ifdef INSTLUMI_STUDIES

      float tmp = float(numberOfClusters) / instlumi;
      hcluslsn->Fill(float(lumiBlock), tmp);  // clusters fpix+bpix
      tmp = float(numberOfPixels) / instlumi;
      hpixlsn->Fill(float(lumiBlock), tmp);  // pixels fpix+bpix

      // pix & clus per lumi
      tmp = float(pixb);
      hpixbl->Fill(instlumi, tmp);  // pixels bpix

      tmp = float(numOfPixPerLay1);
      hpixb1l->Fill(instlumi, tmp);  // pixels bpix
      tmp = float(numOfPixPerLay2);
      hpixb2l->Fill(instlumi, tmp);  // pixels bpix
      tmp = float(numOfPixPerLay3);
      hpixb3l->Fill(instlumi, tmp);  // pixels bpix

      tmp = float(pixf);
      hpixfl->Fill(instlumi, tmp);  // pixels fpix

      tmp = float(numOfPixPerDisk1 + numOfPixPerDisk2);  // -z
      hpixfml->Fill(instlumi, tmp);                      // pixels fpix
      tmp = float(numOfPixPerDisk3 + numOfPixPerDisk4);  // +z
      hpixfpl->Fill(instlumi, tmp);                      // pixels fpix

      tmp = float(clusb);
      hclusbl->Fill(instlumi, tmp);  // clus bpix
      tmp = float(numOfClustersPerLay1);
      hclusb1l->Fill(instlumi, tmp);  // clus bpix
      tmp = float(numOfClustersPerLay2);
      hclusb2l->Fill(instlumi, tmp);  // clus bpix
      tmp = float(numOfClustersPerLay3);
      hclusb3l->Fill(instlumi, tmp);  // clus bpix

      tmp = float(clusf);
      hclusfl->Fill(instlumi, tmp);  // clus  fpix
      tmp = float(numOfClustersPerDisk1 + numOfClustersPerDisk2);
      hclusfml->Fill(instlumi, tmp);  // clus fpix
      tmp = float(numOfClustersPerDisk3 + numOfClustersPerDisk4);
      hclusfpl->Fill(instlumi, tmp);  // clus fpix

      // bx
      tmp = float(numberOfClusters) / instlumi;
      hclubxn->Fill(float(bx), tmp);  // clusters fpix+bpix
      tmp = float(numberOfPixels) / instlumi;
      hpixbxn->Fill(float(bx), tmp);  // pixels fpix+bpix
      tmp = float(numPVsGood) / instlumi;
      hpvbxn->Fill(float(bx), tmp);  // pvs

      //       tmp = float(numOfClustersPerLay1)/float(numOfClustersPerLay3);
      //       hclus13ls->Fill(float(lumiBlock),tmp); // clus 1/3
      //       tmp = float(numOfClustersPerLay2)/float(numOfClustersPerLay3);
      //       hclus23ls->Fill(float(lumiBlock),tmp); // clus 2/3
      //       tmp = float(numOfClustersPerLay1)/float(numOfClustersPerLay2);
      //       hclus12ls->Fill(float(lumiBlock),tmp); // clus 1/2
      //       tmp = float(clusf)/float(numOfClustersPerLay3);
      //       hclusf3ls->Fill(float(lumiBlock),tmp); // clus f/3

#endif  // INSTLUMI_STUDIES

    }  // if instlumi

    hclusPerLay1->Fill(float(numOfClustersPerLay1));
    hclusPerLay2->Fill(float(numOfClustersPerLay2));
    hclusPerLay3->Fill(float(numOfClustersPerLay3));
    hpixPerLay1->Fill(float(numOfPixPerLay1));
    hpixPerLay2->Fill(float(numOfPixPerLay2));
    hpixPerLay3->Fill(float(numOfPixPerLay3));
    if (numOfClustersPerLay1 > 0)
      hdetsPerLay1->Fill(float(numberOfDetUnits1));
    if (numOfClustersPerLay2 > 0)
      hdetsPerLay2->Fill(float(numberOfDetUnits2));
    if (numOfClustersPerLay3 > 0)
      hdetsPerLay3->Fill(float(numberOfDetUnits3));

    hclusPerDisk1->Fill(float(numOfClustersPerDisk1));
    hclusPerDisk2->Fill(float(numOfClustersPerDisk2));
    hclusPerDisk3->Fill(float(numOfClustersPerDisk3));
    hclusPerDisk4->Fill(float(numOfClustersPerDisk4));

    hpixPerDisk1->Fill(float(numOfPixPerDisk1));
    hpixPerDisk2->Fill(float(numOfPixPerDisk2));
    hpixPerDisk3->Fill(float(numOfPixPerDisk3));
    hpixPerDisk4->Fill(float(numOfPixPerDisk4));

    hmaxPixPerDet->Fill(float(maxPixPerDet));

    //     // Check mod and roc time dependence
    //     if( (lumiBlock%5) == 0 ) { // do every 5 lumi blocks

    //       int id=0, lad=0, mod=0, roc=0, layer=0, index=0;
    //       float tmp=0;
    //       // ROCs
    //       // Layer 1,
    //       layer=1;
    //       id = 0;
    //       lad=-3;mod=1;roc=5;
    //       tmp = pixEff->getRoc(layer,lad,mod,roc);
    //       tmp = tmp - rocHits[layer-1][id];
    //       rocHits[layer-1][id] = tmp;
    //       index = lumiBlock/5;
    //       hrocHits1ls->Fill(float(index),id,tmp);

    //       // Layer 2,
    //       layer=2;
    //       id = 0;
    //       lad=-5;mod=-2;roc=10;
    //       tmp = pixEff->getRoc(layer,lad,mod,roc);
    //       tmp = tmp - rocHits[layer-1][id];
    //       rocHits[layer-1][id] = tmp;
    //       index = lumiBlock/5;
    //       hrocHits2ls->Fill(float(index),id,tmp);
    //       // Layer 3,
    //       layer=3;
    //       id = 0;
    //       lad=-18;mod=-2;roc=14;
    //       tmp = pixEff->getRoc(layer,lad,mod,roc);
    //       tmp = tmp - rocHits[layer-1][id];
    //       rocHits[layer-1][id] = tmp;
    //       index = lumiBlock/5;
    //       hrocHits3ls->Fill(float(index),id,tmp);
    //       // MODULES
    //       float half1=0, half2=0;
    //       // Layer 1,
    //       layer=1;
    //       id = 0;
    //       lad=-3;mod=1;
    //       tmp = pixEff->getModule(layer,lad,mod,half1,half2);
    //       tmp = tmp - moduleHits[layer-1][id];
    //       moduleHits[layer-1][id] = tmp;
    //       index = lumiBlock/5;
    //       hmoduleHits1ls->Fill(float(index),id,tmp);

    //     }

#ifdef SEB
//     float tmp0=0., tmp3=0., tmp4=0., tmp5=0.;
//     //if(numOfClustersPerLay1>0) { tmp0 = avCharge1/float(numOfClustersPerLay1); havCharge1->Fill(tmp0);}
//     //if(numOfClustersPerLay2>0) { tmp0 = avCharge2/float(numOfClustersPerLay2); havCharge1->Fill(tmp0);}
//     //if(numOfClustersPerLay3>0) { tmp0 = avCharge3/float(numOfClustersPerLay3); havCharge1->Fill(tmp0);}
//     if( (numOfClustersPerDisk1 + numOfClustersPerDisk2) > 0 )
//       tmp4 = avCharge4/float(numOfClustersPerDisk2 + numOfClustersPerDisk1);
//     if( (numOfClustersPerDisk3 + numOfClustersPerDisk4) > 0 )
//       tmp5 = avCharge5/float(numOfClustersPerDisk3 + numOfClustersPerDisk4);
//     //tmp4 = float(numOfClustersPerDisk2 + numOfClustersPerDisk1);
//     //tmp5 = float(numOfClustersPerDisk3 + numOfClustersPerDisk4);
//     havCharge6->Fill(float(tmp4));
//     havCharge6->Fill(float(tmp5));
//     float tmp8=1.03;
//     if( (tmp4+tmp5) > 0. ) tmp8 = tmp5/(tmp4+tmp5);
//     else tmp8 = 1.05;
//     havCharge3->Fill(tmp8);
//     if(tmp1>0) {
//       tmp3 = (avCharge1 + avCharge2 + avCharge3)/float(tmp1);  // average over bpix
//       havCharge2->Fill(tmp3);
//       //h2d1->Fill(tmp1,tmp3);  // bpix num of clusters vs av cluster charge
//       //float tmp6 = float(numberOfDetUnits1 + numberOfDetUnits2 + numberOfDetUnits3); //num of dets
//       //float tmp7 = float(tmp1)/float(tmp6);  // ave clusters per module
//       //h2d2->Fill(tmp8,tmp3); //
//       //h2d3->Fill(tmp8,tmp1); //
//       if(tmp8>1.02) { // no signal in fpix
// 	havCharge1->Fill(tmp3);
// 	hclus20->Fill(float(numberOfClusters));
//       } else if (tmp8>0.05 && tmp8<0.95) { // some signal in both disks
// 	havCharge4->Fill(tmp3);
// 	hclus21->Fill(float(numberOfClusters));
//       } else { // tmp=0., 1., signal in only one disk
// 	havCharge5->Fill(tmp3);
// 	hclus22->Fill(float(numberOfClusters));
//       }
//     } else {  // no bpix hits
//       hclus23->Fill(float(numberOfClusters));
//     }
#endif  // SEB

#ifdef L1
    //int numberOfClusters0 = numberOfClusters;  // select all clusters

    //if(bptx3) hclus1->Fill(float(numberOfClusters0));
    //if(bptx4) hclus2->Fill(float(numberOfClusters0));
    //if(bit0)  hclus10->Fill(float(numberOfClusters0));   //
    //if(bptx_and) hclus11->Fill(float(numberOfClusters0));
    //if(bptx_xor) hclus12->Fill(float(numberOfClusters0));
    //if(!bptx_xor && !bptx_and) hclus13->Fill(float(numberOfClusters0));
    //if(bptx_m)   hclus14->Fill(float(numberOfClusters0));
    //if(bptx_p)   hclus15->Fill(float(numberOfClusters0));

    //if(bcs_all)   hclus4->Fill(float(numberOfClusters0));       // or of all BCS bits
    //else      hclus17->Fill(float(numberOfClusters0));      // no BCS
    //if(bit126) hclus6->Fill(float(numberOfClusters0));   // bit 126
    //if(bit124) hclus7->Fill(float(numberOfClusters0));
    //if(bit122) hclus26->Fill(float(numberOfClusters0));
    //if(bcsOR)      hclus8->Fill(float(numberOfClusters0));  // bit 34
    //if(bcs)        hclus9->Fill(float(numberOfClusters0));  // all bcs except 34
    //if(bcs_bptx)   hclus29->Fill(float(numberOfClusters0)); // bits 124,126
    //if(bcs_double) hclus28->Fill(float(numberOfClusters0)); // 36-39, 40-43
    //if(bit32_33)   hclus25->Fill(float(numberOfClusters0)); // bits 32,33? is this usefull
    //if(halo)       hclus3->Fill(float(numberOfClusters0));  // bits 36-39
    //if(bit85) hclus18->Fill(float(numberOfClusters0));      // bit85
    //if(minBias) hclus19->Fill(float(numberOfClusters0));    // bits 40,41
    // #ifdef BX
    //     if     (bxId==1)  {hclus30->Fill(float(numberOfClusters0));hdigis30->Fill(float(numberOfPixels));}
    //     else if(bxId==2)  {hclus31->Fill(float(numberOfClusters0));hdigis31->Fill(float(numberOfPixels));}
    //     else if(bxId==3)  {hclus32->Fill(float(numberOfClusters0));hdigis32->Fill(float(numberOfPixels));}
    //     else if(bxId==4)  {hclus33->Fill(float(numberOfClusters0));hdigis33->Fill(float(numberOfPixels));}
    //     else if(bxId==5)  {hclus34->Fill(float(numberOfClusters0));hdigis34->Fill(float(numberOfPixels));}
    //     else if(bxId==6)  {hclus35->Fill(float(numberOfClusters0));hdigis35->Fill(float(numberOfPixels));}
    //     else if(bxId==-1) {hclus36->Fill(float(numberOfClusters0));hdigis36->Fill(float(numberOfPixels));}
    //     else              {hclus37->Fill(float(numberOfClusters0));hdigis37->Fill(float(numberOfPixels));}
    //     if(run>=160888 && run<=160940) { // 64bx
    //       // Fill 1638
    //       if(bx==442)       {hclus7->Fill(float(numberOfClusters0));  hdigis7->Fill(float(numberOfPixels));} //B1, 1st
    //       else if(bx==3136) {hclus18->Fill(float(numberOfClusters0)); hdigis18->Fill(float(numberOfPixels));} //B1, last
    //       else if(bx==466)  {hclus25->Fill(float(numberOfClusters0)); hdigis25->Fill(float(numberOfPixels));} //B2, last
    //       else if(bx==3112) {hclus26->Fill(float(numberOfClusters0)); hdigis26->Fill(float(numberOfPixels));} //B2, 1st
    //     } else if(run>=160955 && run<=161176) { // 136bx
    //       // Fill 1640
    //       if(bx==149)       hclus7->Fill(float(numberOfClusters0));
    //       else if(bx==3184) hclus18->Fill(float(numberOfClusters0));
    //       else if(bx==173)  hclus25->Fill(float(numberOfClusters0));
    //       else if(bx==3112) hclus26->Fill(float(numberOfClusters0));
    //     } else if(run>=161216 && run<=161312) { // 200bx
    //       // Fill 1645
    //       if(bx==1 || bx==4) {
    // 	hclus7->Fill(float(numberOfClusters0));
    // 	hdigis7->Fill(float(numberOfPixels));
    //       } else if( bx==1950 || bx==1953 || bx==1956 || bx==1959 ) {
    // 	hclus18->Fill(float(numberOfClusters0));
    // 	hdigis18->Fill(float(numberOfPixels));
    //       } else if(bx==25 || bx==28) {
    // 	hclus25->Fill(float(numberOfClusters0));
    // 	hdigis25->Fill(float(numberOfPixels));
    //       } else if( bx==1878 || bx==1881 || bx==1884 || bx==1887 ) {
    // 	hclus26->Fill(float(numberOfClusters0));
    // 	hdigis26->Fill(float(numberOfPixels));      }
    //     }
    // #endif

    // Check L1 bits with pixel selection
    if (L1GTRR.isValid()) {
      //bool l1a = L1GTRR->decision();
      //edm::LogPrint("TestClusters")<<" L1 status :"<<l1a<<" "<<hex;
      for (unsigned int i = 0; i < L1GTRR->decisionWord().size(); ++i) {
        int l1flag = L1GTRR->decisionWord()[i];
        int t1flag = L1GTRR->technicalTriggerWord()[i];
        if (l1flag > 0)
          hl1a1->Fill(float(i));
        if (t1flag > 0 && i < 64)
          hl1t1->Fill(float(i));
      }  // for loop
    }    // if l1a

    // HLT bits
    for (unsigned int i = 0; i < 256; i++)
      if (hlt[i])
        hlt3->Fill(float(i));

#endif  // L1

  }  // if select event

#endif  // HISTOS

}  // end

//define this as a plug-in
DEFINE_FWK_MODULE(TestClusters);
