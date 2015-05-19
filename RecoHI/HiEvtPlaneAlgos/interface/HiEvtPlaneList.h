#ifndef __HiEvtPlaneList__
#define __HiEvtPlaneList__
/*
Index     Name   Detector Order hmin1 hmax1 hmin2 hmax2 minpt maxpt nsub mcw    rmate1    rmate2
    0      HFm1        HF     1 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp1   trackp1
    1      HFp1        HF     1  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm1   trackm1
    2       HF1        HF     1 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm1   trackp1
    3   trackm1   Tracker     1 -2.00 -1.00  0.00  0.00  0.30  3.00 3sub  no      HFm1      HFp1
    4   trackp1   Tracker     1  1.00  2.00  0.00  0.00  0.30  3.00 3sub  no      HFm1      HFp1
    5   Castor1    Castor     1 -6.55 -5.10  0.00  0.00  0.01 50.00 3sub  no      HFp1   trackp1
    6      HFm2        HF     2 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp2 trackmid2
    7      HFp2        HF     2  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm2 trackmid2
    8       HF2        HF     2 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm2   trackp2
    9 trackmid2   Tracker     2 -0.75  0.75  0.00  0.00  0.30  3.00 3sub  no      HFm2      HFp2
   10   trackm2   Tracker     2 -2.00 -1.00  0.00  0.00  0.30  3.00 3sub  no      HFm2      HFp2
   11   trackp2   Tracker     2  1.00  2.00  0.00  0.00  0.30  3.00 3sub  no      HFm2      HFp2
   12   Castor2    Castor     2 -6.55 -5.10  0.00  0.00  0.01 50.00 3sub  no trackmid2      HFp2
   13      HFm3        HF     3 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp3 trackmid3
   14      HFp3        HF     3  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm3 trackmid3
   15       HF3        HF     3 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm3   trackp3
   16 trackmid3   Tracker     3 -0.75  0.75  0.00  0.00  0.30  3.00 3sub  no      HFm3      HFp3
   17   trackm3   Tracker     3 -2.00 -1.00  0.00  0.00  0.30  3.00 3sub  no      HFm3      HFp3
   18   trackp3   Tracker     3  1.00  2.00  0.00  0.00  0.30  3.00 3sub  no      HFm3      HFp3
   19      HFm4        HF     4 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub  no      HFp4 trackmid4
   20      HFp4        HF     4  3.00  5.00  0.00  0.00  0.01 30.00 3sub  no      HFm4 trackmid4
   21       HF4        HF     4 -5.00 -3.00  3.00  5.00  0.01 30.00 3sub  no   trackm4   trackp4
   22 trackmid4   Tracker     4 -0.75  0.75  0.00  0.00  0.30  3.00 3sub  no      HFm4      HFp4
   23   trackm4   Tracker     4 -2.00 -1.00  0.00  0.00  0.30  3.00 3sub  no      HFm4      HFp4
   24   trackp4   Tracker     4  1.00  2.00  0.00  0.00  0.30  3.00 3sub  no      HFm4      HFp4
   25    HFm1mc        HF     1 -5.00 -3.00  0.00  0.00  0.01 30.00 3sub yes    HFp1mc trackp1mc
   26    HFp1mc        HF     1  3.00  5.00  0.00  0.00  0.01 30.00 3sub yes    HFm1mc trackm1mc
   27 trackm1mc   Tracker     1 -2.20 -1.40  0.00  0.00  0.30  3.00 3sub yes    HFm1mc    HFp1mc
   28 trackp1mc   Tracker     1  1.40  2.20  0.00  0.00  0.30  3.00 3sub yes    HFm1mc    HFp1mc
   29 Castor1mc    Castor     1 -6.55 -5.10  0.00  0.00  0.01 50.00 3sub yes    HFp1mc trackp1mc
*/
#include <string>

namespace hi{

  enum EPNamesInd {
          HFm1,        HFp1,         HF1,     trackm1,     trackp1,
       Castor1,        HFm2,        HFp2,         HF2,   trackmid2,
       trackm2,     trackp2,     Castor2,        HFm3,        HFp3,
           HF3,   trackmid3,     trackm3,     trackp3,        HFm4,
          HFp4,         HF4,   trackmid4,     trackm4,     trackp4,
        HFm1mc,      HFp1mc,   trackm1mc,   trackp1mc,   EPBLANK
  };

  const std::string  EPNames[]  = {
        "HFm1",      "HFp1",       "HF1",   "trackm1",   "trackp1",
     "Castor1",      "HFm2",      "HFp2",       "HF2", "trackmid2",
     "trackm2",   "trackp2",   "Castor2",      "HFm3",      "HFp3",
         "HF3", "trackmid3",   "trackm3",   "trackp3",      "HFm4",
        "HFp4",       "HF4", "trackmid4",   "trackm4",   "trackp4",
      "HFm1mc",    "HFp1mc", "trackm1mc", "trackp1mc" 
  };

  enum Detectors {Tracker, HF, Castor};

  const int  EPDet[]  = {
          HF,        HF,        HF,   Tracker,   Tracker,
      Castor,        HF,        HF,        HF,   Tracker,
     Tracker,   Tracker,    Castor,        HF,        HF,
          HF,   Tracker,   Tracker,   Tracker,        HF,
          HF,        HF,   Tracker,   Tracker,   Tracker,
          HF,        HF,   Tracker,   Tracker 
  };

  const int  EPOrder[]  = {
             1,           1,           1,           1,           1,
             1,           2,           2,           2,           2,
             2,           2,           2,           3,           3,
             3,           3,           3,           3,           4,
             4,           4,           4,           4,           4,
             1,           1,           1,           1 
  };

  const double  EPEtaMin1[]  = {
         -5.00,        3.00,       -5.00,       -2.00,        1.00,
         -6.55,       -5.00,        3.00,       -5.00,       -0.75,
         -2.00,        1.00,       -6.55,       -5.00,        3.00,
         -5.00,       -0.75,       -2.00,        1.00,       -5.00,
          3.00,       -5.00,       -0.75,       -2.00,        1.00,
         -5.00,        3.00,       -2.20,        1.40 
  };

  const double  EPEtaMax1[]  = {
         -3.00,        5.00,       -3.00,       -1.00,        2.00,
         -5.10,       -3.00,        5.00,       -3.00,        0.75,
         -1.00,        2.00,       -5.10,       -3.00,        5.00,
         -3.00,        0.75,       -1.00,        2.00,       -3.00,
          5.00,       -3.00,        0.75,       -1.00,        2.00,
         -3.00,        5.00,       -1.40,        2.20 
  };

  const double  EPEtaMin2[]  = {
          0.00,        0.00,        3.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        3.00,        0.00,
          0.00,        0.00,        0.00,        0.00,        0.00,
          3.00,        0.00,        0.00,        0.00,        0.00,
          0.00,        3.00,        0.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        0.00 
  };

  const double  EPEtaMax2[]  = {
          0.00,        0.00,        5.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        5.00,        0.00,
          0.00,        0.00,        0.00,        0.00,        0.00,
          5.00,        0.00,        0.00,        0.00,        0.00,
          0.00,        5.00,        0.00,        0.00,        0.00,
          0.00,        0.00,        0.00,        0.00 
  };

  const double  minTransverse[]  = {
          0.01,        0.01,        0.01,        0.30,        0.30,
          0.01,        0.01,        0.01,        0.01,        0.30,
          0.30,        0.30,        0.01,        0.01,        0.01,
          0.01,        0.30,        0.30,        0.30,        0.01,
          0.01,        0.01,        0.30,        0.30,        0.30,
          0.01,        0.01,        0.30,        0.30 
  };

  const double  maxTransverse[]  = {
         30.00,       30.00,       30.00,        3.00,        3.00,
         50.00,       30.00,       30.00,       30.00,        3.00,
          3.00,        3.00,       50.00,       30.00,       30.00,
         30.00,        3.00,        3.00,        3.00,       30.00,
         30.00,       30.00,        3.00,        3.00,        3.00,
         30.00,       30.00,        3.00,        3.00 
  };

  const std::string  ResCalcType[]  = {
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub",      "3sub",
        "3sub",      "3sub",      "3sub",      "3sub" 
  };

  const std::string  MomConsWeight[]  = {
          "no",        "no",        "no",        "no",        "no",
          "no",        "no",        "no",        "no",        "no",
          "no",        "no",        "no",        "no",        "no",
          "no",        "no",        "no",        "no",        "no",
          "no",        "no",        "no",        "no",        "no",
         "yes",       "yes",       "yes",       "yes" 
  };

  const int  RCMate1[]  = {
        HFp1,      HFm1,   trackm1,      HFm1,      HFm1,
        HFp1,      HFp2,      HFm2,   trackm2,      HFm2,
        HFm2,      HFm2, trackmid2,      HFp3,      HFm3,
     trackm3,      HFm3,      HFm3,      HFm3,      HFp4,
        HFm4,   trackm4,      HFm4,      HFm4,      HFm4,
      HFp1mc,    HFm1mc,    HFm1mc,    HFm1mc 
  };

  const int  RCMate2[]  = {
     trackp1,   trackm1,   trackp1,      HFp1,      HFp1,
     trackp1, trackmid2, trackmid2,   trackp2,      HFp2,
        HFp2,      HFp2,      HFp2, trackmid3, trackmid3,
     trackp3,      HFp3,      HFp3,      HFp3, trackmid4,
   trackmid4,   trackp4,      HFp4,      HFp4,      HFp4,
   trackp1mc, trackm1mc,    HFp1mc,    HFp1mc 
  };

  static const int NumEPNames = 29;
}
#endif
