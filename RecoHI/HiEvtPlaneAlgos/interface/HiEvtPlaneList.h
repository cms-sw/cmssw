#ifndef __HiEvtPlaneList__
#define __HiEvtPlaneList__
/*
Index     Name   Detector Order hmin1 hmax1 hmin2 hmax2 minpt maxpt nsub mcw    rmate1    rmate2
    0      HFm2        HF     2 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp2 trackmid2
    1      HFp2        HF     2  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm2 trackmid2
    2       HF2        HF     2 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm2   trackp2
    3 trackmid2   Tracker     2 -0.50  0.50  0.00  0.00  0.50  3.00 3sub  no      HFm2      HFp2
    4   trackm2   Tracker     2 -2.00 -1.00  0.00  0.00  0.50  3.00 3sub  no      HFp2      HFm2
    5   trackp2   Tracker     2  1.00  2.00  0.00  0.00  0.50  3.00 3sub  no      HFp2      HFm2
    6      HFm3        HF     3 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp3 trackmid3
    7      HFp3        HF     3  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm3 trackmid3
    8       HF3        HF     3 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm3   trackp3
    9 trackmid3   Tracker     3 -0.50  0.50  0.00  0.00  0.50  3.00 3sub  no      HFm3      HFp3
   10   trackm3   Tracker     3 -2.00 -1.00  0.00  0.00  0.50  3.00 3sub  no      HFp3      HFm3
   11   trackp3   Tracker     3  1.00  2.00  0.00  0.00  0.50  3.00 3sub  no      HFp3      HFm3
*/
#include <string>
namespace hi {

  // clang-format off
  enum EPNamesInd {
          HFm2,        HFp2,         HF2,   trackmid2,     trackm2,
       trackp2,        HFm3,        HFp3,         HF3,   trackmid3,
       trackm3,     trackp3,   EPBLANK
  };

  static const int NumEPNames = 12;

  const std::array<std::string, NumEPNames> EPNames = {{
        "HFm2",      "HFp2",       "HF2", "trackmid2",   "trackm2",
     "trackp2",      "HFm3",      "HFp3",       "HF3", "trackmid3",
     "trackm3",   "trackp3"
  }};

  enum Detectors { Tracker, HF, Castor, RPD };

  const std::array<int, NumEPNames> EPDet = {{
          HF,        HF,        HF,   Tracker,   Tracker,
     Tracker,        HF,        HF,        HF,   Tracker,
     Tracker,   Tracker
  }};

  const std::array<int, NumEPNames> EPOrder = {{
             2,           2,           2,           2,           2,
             2,           3,           3,           3,           3,
             3,           3
  }};

  const std::array<double, NumEPNames> EPEtaMin1 = {{
         -5.00,        3.00,       -5.00,       -0.50,       -2.00,
          1.00,       -5.00,        3.00,       -5.00,       -0.50,
         -2.00,        1.00
  }};

  const std::array<double, NumEPNames> EPEtaMax1 = {{
         -3.00,        5.00,       -3.00,        0.50,       -1.00,
          2.00,       -3.00,        5.00,       -3.00,        0.50,
         -1.00,        2.00
  }};

  const std::array<double, NumEPNames> EPEtaMin2 = {{
          0.00,        0.00,        3.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        3.00,        0.00,
          0.00,        0.00
  }};

  const std::array<double, NumEPNames> EPEtaMax2 = {{
          0.00,        0.00,        5.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        5.00,        0.00,
          0.00,        0.00
  }};

  const std::array<double, NumEPNames> minTransverse = {{
          0.01,        0.01,        0.01,        0.50,        0.50,
          0.50,        0.01,        0.01,        0.01,        0.50,
          0.50,        0.50
  }};

  const std::array<double, NumEPNames> maxTransverse = {{
         30.00,       30.00,       30.00,        3.00,        3.00,
          3.00,       30.00,       30.00,       30.00,        3.00,
          3.00,        3.00
  }};

  const std::array<std::string, NumEPNames> ResCalcType = {{
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub"
  }};

  const std::array<std::string, NumEPNames> MomConsWeight = {{
          "no",        "no",        "no",        "no",        "no",
          "no",        "no",        "no",        "no",        "no",
          "no",        "no"
  }};

  const std::array<int, NumEPNames> RCMate1 = {{
        HFp2,      HFm2,   trackm2,      HFm2,      HFp2,
        HFp2,      HFp3,      HFm3,   trackm3,      HFm3,
        HFp3,      HFp3
  }};

  const std::array<int, NumEPNames> RCMate2 = {{
   trackmid2, trackmid2,   trackp2,      HFp2,      HFm2,
        HFm2, trackmid3, trackmid3,   trackp3,      HFp3,
        HFm3,      HFm3
  }};

  // clang-format on
}  // namespace hi
#endif
