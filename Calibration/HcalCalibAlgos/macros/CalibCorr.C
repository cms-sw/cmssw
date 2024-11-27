//////////////////////////////////////////////////////////////////////////////
// Contains several utility functions used by CalibMonitor/CalibSort/CalibTree:
//
// void unpackDetId(detId, subdet, zside, ieta, iphi, depth)
//      Unpacks rawId for Hadron calorimeter
// unsigned int truncateId(detId, truncateFlag, debug)
//      Defines depth of the DetId guided by the value of truncateFlag
// double puFactor(type, ieta, pmom, eHcal, ediff, debug)
//      Returns a multiplicative factor to the energy as PU correction
//        int type = The type of data being considered
//                   (1: Run 1 old; 2: Run1 new; 3: Run2 2016;
//                    4: Run 2 2017; 5: Run2 2018; 6: Run3 Old;
//                    7: Run 3 June 2021; 8: Run3 Mahi Jan2022;
//                    9: Run3 M0 Jan2022l; 97: dlphin Try 3;
//                    98: dlphin Try 2; 99: dlphin Try 1)
// double puFactorRho(type, ieta, rho, double eHcal)
//      Returns a multiplicative factor as PU correction evaluated from rho
//        int type = The type of data being considered
//                   (1: 2017 Data;  2: 2017 MC; 3: 2018 MC; 4: 2018AB;
//                    5: 2018BC; 6: 2016 MC)
// double puweight(vtx)
//      Return PU weight for QCD PU sample
// bool fillChain(chain, inputFileList)
//      Prepares a Tchain by chaining several ROOT files specified
// std::vector<std::string> splitString (fLine)
//      Splits a string into several items which are separated by blank in i/p
// CalibCorrFactor(infile, useScale, scale, etamax, debug)
//      A class which reads a file with correction factors and provides
//        bool   doCorr() : flag saying if correction is available
//        double getCorr(id): correction factor for a given cell
// CalibCorr(infile, flag, debug)
//      A second class which reads a file and provide corrections through
//        float getCorr(run, id): Run and ID dependent correction
//        double getCorr(entry): Entry # (in the file) dependent correction
//        bool absent(entry) : if correction factor absent
//        bool present(entry): or present (relevant for ML motivated)
//        double getPhiCorr(unsigned int id): correction factor from phi
//                                            symmery studies
//        double getPhiCorrs(int run, unsigned int id): era-dependent correction
//                                            factor from phi symmetry studies
//      *infile* is the name of the input file
//      *flag* can have a value between 0:5 specifying the type of correction
//             5 for run-dependent correctons using results from several phi
//             symmetry studies; 4 for using results from one phi-symmetry
//             study; 3 for pileup correction using machine learning method;
//             2 for overall response corrections; 1 for depth dependence
//             corrections; 0 for raddam corrections)
// CalibSelectRBX(rbxFile, debug)
//      A class for selecting a given set of Read Out Box's and provides
//        bool isItRBX(detId): if it/they is in the chosen RBXs
//        bool isItRBX(ieta, iphi): if it is in the chosen RBXs
// CalibDuplicate(infile, flag, debug)
//      A class for either rejecting duplicate entries or giving depth
//        dependent weight. flag is 0 for keeping a list of duplicate
//        emtries; 1 is to keep depth dependent weight for each ieta;
//        2 is to keep a list of ieta, iphi for channels to be selected;
//        3 is to read the gain correction file
//        bool isDuplicate(entry): if it is a duplicate entry
//        double getWeight(ieta, depth): get the dependent weight
//        bool select(int ieta, int iphi): channels to be selected
//        double getCorr(int run, int ieta, int depth): correction factor
//        to modify the rechit energy due to change in gain fctor
// void CalibCorrTest(infile, flag)
//      Tests a file which contains correction factors used by CalibCorr
//      if *flag* is >= 0; for negative values it initializes CalibDuplicate
//      with -flag and *debug* = tue
// void CalibCorrScale(infile, oufile, scale)
//      Scales all contents of correction factors by "scale" from "infile"
//      to "outfile"
// void CalibCorrScale2(infile, oufile, scaleB, scaleT, scaleE)
//      Scales all contents of correction factors in the barrel, transition and
//      endcap regions by "scaleB", "scaleT", "scaleE" from "infile" and writes
//      them to "outfile"
//////////////////////////////////////////////////////////////////////////////
#ifndef CalibrationHcalCalibAlgosCalibCorr_h
#define CalibrationHcalCalibAlgosCalibCorr_h

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include <TChain.h>
#include <TROOT.h>

void unpackDetId(unsigned int detId, int& subdet, int& zside, int& ieta, int& iphi, int& depth) {
  // The maskings are defined in DataFormats/DetId/interface/DetId.h
  //                      and in DataFormats/HcalDetId/interface/HcalDetId.h
  // The macro does not invoke the classes there and use them
  subdet = ((detId >> 25) & (0x7));
  if ((detId & 0x1000000) == 0) {
    ieta = ((detId >> 7) & 0x3F);
    zside = (detId & 0x2000) ? (1) : (-1);
    depth = ((detId >> 14) & 0x1F);
    iphi = (detId & 0x3F);
  } else {
    ieta = ((detId >> 10) & 0x1FF);
    zside = (detId & 0x80000) ? (1) : (-1);
    depth = ((detId >> 20) & 0xF);
    iphi = (detId & 0x3FF);
  }
}

unsigned int packDetId(int subdet, int ieta, int iphi, int depth) {
  // The maskings are defined in DataFormats/DetId/interface/DetId.h
  //                      and in DataFormats/HcalDetId/interface/HcalDetId.h
  // The macro does not invoke the classes there and use them
  const int det(4);
  unsigned int id = (((det & 0xF) << 28) | ((subdet & 0x7) << 25));
  id |= ((0x1000000) | ((depth & 0xF) << 20) | ((ieta > 0) ? (0x80000 | (ieta << 10)) : ((-ieta) << 10)) |
         (iphi & 0x3FF));
  return id;
}

unsigned int matchDetId(unsigned int detId) {
  if ((detId & 0x1000000) == 0) {
    int subdet = ((detId >> 25) & (0x7));
    int ieta = ((detId >> 7) & 0x3F);
    int zside = (detId & 0x2000) ? (1) : (-1);
    int depth = ((detId >> 14) & 0x1F);
    int iphi = (detId & 0x3F);
    return packDetId(subdet, (ieta * zside), iphi, depth);
  } else {
    return detId;
  }
}

unsigned int truncateId(unsigned int detId, int truncateFlag, bool debug = false) {
  //Truncate depth information of DetId's
  unsigned int id(detId);
  if (debug) {
    std::cout << "Truncate 1 " << std::hex << detId << " " << id << std::dec << " Flag " << truncateFlag << std::endl;
  }
  int truncate0 = ((truncateFlag / 1) % 10);
  int truncate1 = ((truncateFlag / 10) % 10);
  int subdet, depth, zside, ieta, iphi;
  unpackDetId(detId, subdet, zside, ieta, iphi, depth);
  if (truncate1 == 1)
    zside = 1;
  if (truncate0 == 1) {
    //Ignore depth index of ieta values of 15 and 16 of HB
    if ((subdet == 1) && (ieta > 14))
      depth = 1;
  } else if (truncate0 == 2) {
    //Ignore depth index of all ieta values
    depth = 1;
  } else if (truncate0 == 3) {
    //Ignore depth index for HE
    if (subdet == 2)
      depth = 1;
  } else if (truncate0 == 4) {
    //Ignore depth index for HB
    if (subdet == 1)
      depth = 1;
  } else if (truncate0 == 5) {
    //Ignore depth index for depth > 1 in HB and HE
    if (depth > 1)
      depth = 2;
  } else if (truncate0 == 6) {
    //Ignore depth index for depth > 2 in HB and HE; depths 1, 2 considered as 1
    if (depth <= 2)
      depth = 1;
    else
      depth = 2;
  } else if (truncate0 == 7) {
    //Ignore depth index for HB; depth 1, 2 considered as 1; deepth > 2 as 2
    //Ignore depth index for depth > 2 in HE; all depths considered as 1
    if (subdet == 1) {
      if (depth <= 2)
        depth = 1;
      else
        depth = 2;
    } else {
      depth = 1;
    }
  } else if (truncate0 == 8) {
    //Assign all depths > 4 as depth = 5
    if (depth >= 5)
      depth = 5;
  } else if (truncate0 == 9) {
    //Depths 1 and 2 considered as depth = 2; rest of the depths are kept as it is
    if (depth <= 2)
      depth = 2;
  }
  id = (subdet << 25) | (0x1000000) | ((depth & 0xF) << 20) | ((zside > 0) ? (0x80000 | (ieta << 10)) : (ieta << 10));
  if (debug) {
    std::cout << "Truncate 2: " << subdet << " " << zside * ieta << " " << depth << " " << std::hex << id << " input "
              << detId << std::dec << std::endl;
  }
  return id;
}

unsigned int repackId(const std::string& det, int eta, int depth) {
  int subdet = (det == "HE") ? 2 : 1;
  int zside = (eta >= 0) ? 1 : -1;
  int ieta = (eta >= 0) ? eta : -eta;
  unsigned int id =
      (subdet << 25) | (0x1000000) | ((depth & 0xF) << 20) | ((zside > 0) ? (0x80000 | (ieta << 10)) : (ieta << 10));
  return id;
}

unsigned int repackId(int eta, int depth) {
  int zside = (eta >= 0) ? 1 : -1;
  int ieta = (eta >= 0) ? eta : -eta;
  int subdet = ((ieta > 16) || ((ieta == 16) && (depth > 3))) ? 2 : 1;
  unsigned int id =
      (subdet << 25) | (0x1000000) | ((depth & 0xF) << 20) | ((zside > 0) ? (0x80000 | (ieta << 10)) : (ieta << 10));
  return id;
}

unsigned int repackId(int subdet, int ieta, int iphi, int depth) {
  unsigned int id = ((subdet & 0x7) << 25);
  id |= ((0x1000000) | ((depth & 0xF) << 20) | ((ieta > 0) ? (0x80000 | (ieta << 10)) : ((-ieta) << 10)) |
         (iphi & 0x3FF));
  return id;
}

bool ifHB(int ieta, int depth) { return ((std::abs(ieta) < 16) || ((std::abs(ieta) == 16) && (depth != 4))); }

int truncateDepth(int ieta, int depth, int truncateFlag) {
  int truncate0 = ((truncateFlag / 1) % 10);
  int d(depth);
  if (truncate0 == 5) {
    d = (depth == 1) ? 1 : 2;
  } else if (truncate0 == 4) {
    d = ifHB(ieta, depth) ? ((depth == 1) ? 1 : 2) : depth;
  } else if (truncate0 == 3) {
    d = (!ifHB(ieta, depth)) ? ((depth == 1) ? 1 : 2) : depth;
  } else if (truncate0 == 2) {
    d = 1;
  } else if (truncate0 == 1) {
    d = ((std::abs(ieta) == 15) || (std::abs(ieta) == 16)) ? 1 : depth;
  }
  return d;
}

double threshold(int subdet, int depth, int form) {
  double cutHE[4][7] = {{0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
			{0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
			{0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
			{0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}};
  double cutHB[4][4] = {{0.1, 0.2, 0.3, 0.3}, 
			{0.25, 0.25, 0.3, 0.3}, 
			{0.4, 0.3, 0.3, 0.3},
			{0.6, 0.4, 0.4, 0.5}};
  double thr(0);
  if (form > 0) {
    if (subdet == 2)
      thr = cutHE[form - 1][depth - 1];
    else
      thr = cutHB[form - 1][depth - 1];
  }
  return thr;
}

double threshold(unsigned int detId, int form) {
  int subdet = ((detId >> 25) & (0x7));
  int depth = ((detId & 0x1000000) == 0) ? ((detId >> 14) & 0x1F) : ((detId >> 20) & 0xF);
  return threshold(subdet, depth, form);
}

double puFactor(int type, int ieta, double pmom, double eHcal, double ediff, bool debug = false) {
  double fac(1.0);
  if (debug)
    std::cout << "Input Type " << type << " ieta " << ieta << " pmon " << pmom << " E " << eHcal << ":" << ediff;
  if (type <= 2) {
    double frac = (type == 1) ? 0.02 : 0.03;
    if (pmom > 0 && ediff > frac * pmom) {
      double a1(0), a2(0);
      if (type == 1) {
        a1 = -0.35;
        a2 = -0.65;
        if (std::abs(ieta) == 25) {
          a2 = -0.30;
        } else if (std::abs(ieta) > 25) {
          a1 = -0.45;
          a2 = -0.10;
        }
      } else {
        a1 = -0.39;
        a2 = -0.59;
        if (std::abs(ieta) >= 25) {
          a1 = -0.283;
          a2 = -0.272;
        } else if (std::abs(ieta) > 22) {
          a1 = -0.238;
          a2 = -0.241;
        }
      }
      fac = (1.0 + a1 * (eHcal / pmom) * (ediff / pmom) * (1 + a2 * (ediff / pmom)));
      if (debug)
        std::cout << " coeff " << a1 << ":" << a2 << " Fac " << fac;
    }
  } else {
    int jeta = std::abs(ieta);
    double d2p = (ediff / pmom);
    const double DELTA_CUT = 0.03;
    if (type == 3) {  // 16pu
      const double CONST_COR_COEF[4] = {0.971, 1.008, 0.985, 1.086};
      const double LINEAR_COR_COEF[4] = {0, -0.359, -0.251, -0.535};
      const double SQUARE_COR_COEF[4] = {0, 0, 0.048, 0.143};
      const int PU_IETA_1 = 9;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3));
      if (d2p > DELTA_CUT)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 4) {  // 17pu
      const double CONST_COR_COEF[4] = {0.974, 1.023, 0.989, 1.077};
      const double LINEAR_COR_COEF[4] = {0, -0.524, -0.268, -0.584};
      const double SQUARE_COR_COEF[4] = {0, 0, 0.053, 0.170};
      const int PU_IETA_1 = 9;
      const int PU_IETA_2 = 18;
      const int PU_IETA_3 = 25;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3));
      if (d2p > DELTA_CUT)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 5) {  // 18pu
      const double CONST_COR_COEF[4] = {0.973, 0.998, 0.992, 0.965};
      const double LINEAR_COR_COEF[4] = {0, -0.318, -0.261, -0.406};
      const double SQUARE_COR_COEF[4] = {0, 0, 0.047, 0.089};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3));
      if (d2p > DELTA_CUT)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 6) {  // 21pu (old)
      const double CONST_COR_COEF[6] = {0.98913, 0.982008, 0.974011, 0.496234, 0.368110, 0.294053};
      const double LINEAR_COR_COEF[6] = {-0.0491388, -0.124058, -0.249718, -0.0667390, -0.0770766, -0.0580492};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0368657, 0.00656337, 0.00724508, 0.00568967};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 97) {  // dlphin Try 3
      const double CONST_COR_COEF[6] = {0.987617, 0.983421, 0.938622, 0.806662, 0.738354, 0.574195};
      const double LINEAR_COR_COEF[6] = {-0.07018610, -0.2494880, -0.1997290, -0.1769320, -0.2427950, -0.1230480};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0263541, 0.0257008, 0.0426584, 0.0200361};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 98) {  // dlphin Try 2
      const double CONST_COR_COEF[6] = {0.987665, 0.983468, 0.938628, 0.807241, 0.739132, 0.529059};
      const double LINEAR_COR_COEF[6] = {-0.0708906, -0.249995, -0.199683, -0.177692, -0.243436, -0.0668783};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0263163, 0.0260158, 0.0426864, 0.00398778};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 99) {  // dlphin Try 1
      const double CONST_COR_COEF[6] = {0.98312, 0.978532, 0.972211, 0.756004, 0.638075, 0.547192};
      const double LINEAR_COR_COEF[6] = {-0.0472436, -0.186206, -0.247339, -0.166062, -0.159781, -0.118747};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0356827, 0.0202461, 0.01785078, 0.0123003};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 7) {  // 21pu (June, 2021)
      const double CONST_COR_COEF[6] = {0.989727, 0.981923, 0.97571, 0.562475, 0.467947, 0.411831};
      const double LINEAR_COR_COEF[6] = {-0.0469558, -0.125805, -0.251383, -0.0668994, -0.0964236, -0.0947158};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0399785, 0.00610104, 0.00952528, 0.0100645};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 9) {  // M0 22pu (Jan, 2022)
      const double CONST_COR_COEF[6] = {0.980941, 0.973156, 0.970749, 0.726582, 0.532628, 0.473727};
      const double LINEAR_COR_COEF[6] = {-0.0770642, -0.178295, -0.241338, -0.122956, -0.122346, -0.112574};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0401732, 0.00989908, 0.0108291, 0.0100508};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else if (type == 10) {  // MAHI (Jan, 2023)
      const double CONST_COR_COEF[6] = {0.987967, 0.983376, 0.954840, 0.676950, 0.513111, 0.430349};
      const double LINEAR_COR_COEF[6] = {-0.0399269, -0.101755, -0.156848, -0.0969012 - 0.107831, -0.0911755};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0133473, 0.00727513, 0.00863409, 0.00727055};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    } else {  // Mahi 22pu (Jan, 2022)
      const double CONST_COR_COEF[6] = {0.995902, 0.991240, 0.981019, 0.788052, 0.597956, 0.538731};
      const double LINEAR_COR_COEF[6] = {-0.0540563, -0.104361, -0.215936, -0.147801, -0.160845, -0.154359};
      const double SQUARE_COR_COEF[6] = {0, 0, 0.0365911, 0.0161266, 0.0180053, 0.0184295};
      const int PU_IETA_1 = 7;
      const int PU_IETA_2 = 16;
      const int PU_IETA_3 = 25;
      const int PU_IETA_4 = 26;
      const int PU_IETA_5 = 27;
      unsigned icor = (unsigned(jeta >= PU_IETA_1) + unsigned(jeta >= PU_IETA_2) + unsigned(jeta >= PU_IETA_3) +
                       unsigned(jeta >= PU_IETA_4) + unsigned(jeta >= PU_IETA_5));
      double deltaCut = (icor > 2) ? 1.0 : DELTA_CUT;
      if (d2p > deltaCut)
        fac = (CONST_COR_COEF[icor] + LINEAR_COR_COEF[icor] * d2p + SQUARE_COR_COEF[icor] * d2p * d2p);
      if (debug)
        std::cout << " d2p " << d2p << ":" << DELTA_CUT << " coeff " << icor << ":" << CONST_COR_COEF[icor] << ":"
                  << LINEAR_COR_COEF[icor] << ":" << SQUARE_COR_COEF[icor] << " Fac " << fac;
    }
  }
  if (fac < 0 || fac > 1)
    fac = 0;
  if (debug)
    std::cout << " Final factor " << fac << std::endl;
  return fac;
}

double puFactorRho(int type, int ieta, double rho, double eHcal) {
  // type = 1: 2017 Data;  2: 2017 MC; 3: 2018 MC; 4: 2018AB; 5: 2018BC
  //        6: 2016 MC;
  double par[36] = {0.0205395,  -43.0914,   2.67115,    0.239674,    -0.0228009, 0.000476963, 0.137566,  -32.8386,
                    3.25886,    0.0863636,  -0.0165639, 0.000413894, 0.206168,   -145.828,    10.3191,   0.531418,
                    -0.0578416, 0.00118905, 0.175356,   -175.543,    14.3414,    0.294718,    -0.049836, 0.00106228,
                    0.134314,   -175.809,   13.5307,    0.395943,    -0.0539062, 0.00111573,  0.145342,  -98.1904,
                    8.14001,    0.205526,   -0.0327818, 0.000726059};
  double energy(eHcal);
  if (type >= 1 && type <= 6) {
    int eta = std::abs(ieta);
    int it = 6 * (type - 1);
    double ea =
        (eta < 20)
            ? par[it]
            : ((((par[it + 5] * eta + par[it + 4]) * eta + par[it + 3]) * eta + par[it + 2]) * eta + par[it + 1]);
    energy -= (rho * ea);
  }
  return energy;
}

double puweight(double vtx) {  ///////for QCD PU sample
  double a(1.0);
  if (vtx < 11)
    a = 0.120593;
  else if (vtx < 21)
    a = 0.58804;
  else if (vtx < 31)
    a = 1.16306;
  else if (vtx < 41)
    a = 1.45892;
  else if (vtx < 51)
    a = 1.35528;
  else if (vtx < 61)
    a = 1.72032;
  else if (vtx < 71)
    a = 3.34812;
  else if (vtx < 81)
    a = 9.70097;
  else if (vtx < 91)
    a = 9.24839;
  else if (vtx < 101)
    a = 23.0816;
  return a;
}

bool fillChain(TChain* chain, const char* inputFileList) {
  std::string fname(inputFileList);
  if (fname.substr(fname.size() - 5, 5) == ".root") {
    chain->Add(fname.c_str());
  } else {
    std::ifstream infile(inputFileList);
    if (!infile.is_open()) {
      std::cout << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
      return false;
    }
    while (1) {
      infile >> fname;
      if (!infile.good())
        break;
      chain->Add(fname.c_str());
    }
    infile.close();
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  return true;
}

std::vector<std::string> splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

class CalibCorrFactor {
public:
  CalibCorrFactor(const char* infile, int useScale, double scale, bool etamax, bool marina, bool debug);
  ~CalibCorrFactor() {}

  bool doCorr() const { return (corrE_ || (useScale_ != 0)); }
  double getCorr(unsigned int id);

private:
  bool readCorrFactor(const char* fName, bool marina);
  double getFactor(const int& ieta);

  const int useScale_;
  const double scale_;
  const bool etaMax_, debug_;
  static const int depMax_ = 10;
  bool corrE_;
  int etamp_, etamn_;
  double cfacmp_[depMax_], cfacmn_[depMax_];
  std::map<std::pair<int, int>, double> cfactors_;
};

class CalibCorr {
public:
  CalibCorr(const char* infile, int flag, bool debug);
  ~CalibCorr() {}

  float getCorr(int run, unsigned int id);
  double getCorr(const Long64_t& entry);
  double getTrueCorr(const Long64_t& entry);
  double getPhiCorr(unsigned int id);
  double getPhiCorrs(int run, unsigned int id);
  bool absent(const Long64_t& entry);
  bool absent() { return (good_ == 0); }
  bool present(const Long64_t& entry);

private:
  unsigned int runId(const int& run);
  unsigned int readCorrRun(const char* infile);
  unsigned int readCorrDepth(const char* infile);
  unsigned int readCorrResp(const char* infile);
  unsigned int readCorrPU(const char* infile);
  unsigned int readCorrPhi(const char* infile);
  unsigned int readCorrPhis(const char* infile);
  unsigned int getDetIdHE(int ieta, int iphi, int depth);
  unsigned int getDetId(int subdet, int ieta, int iphi, int depth);
  unsigned int correctDetId(const unsigned int& detId);

  static const unsigned int nmax_ = 10;
  int flag_;
  bool debug_;
  unsigned int good_;
  std::map<unsigned int, float> corrFac_[nmax_], corrFacDepth_, corrFacResp_;
  std::map<Long64_t, double> cfactors_;
  std::vector<int> runlow_;
  std::map<unsigned int, double> corrPhiSym_;
  std::vector<std::pair<int, int> > runs_, ids_;
  std::map<std::pair<unsigned int, unsigned int>, double> corrPhiSyms_;
};

class CalibSelectRBX {
public:
  CalibSelectRBX(const char* rbxFile, bool debug = false);
  ~CalibSelectRBX() {}

  bool isItRBX(const unsigned int);
  bool isItRBX(const std::vector<unsigned int>*);
  bool isItRBX(const int, const int);

private:
  bool debug_;
  std::vector<int> zsphis_;
};

class CalibDuplicate {
public:
  CalibDuplicate(const char* infile, int flag, bool debug);
  ~CalibDuplicate() {}

  bool isDuplicate(long entry);
  double getWeight(const unsigned int);
  bool doCorr() { return (((flag_ == 1) || (flag_ == 3)) && ok_); }
  bool doCorr(int flag) { return ((flag_ == flag) && ok_); }
  bool select(int ieta, int iphi);
  double getCorr(int run, int ieta, int depth);

private:
  bool readCorrFactor(const char* fName);
  unsigned int runId(const int& run);

  int flag_;
  double debug_, ok_;
  std::vector<Long64_t> entries_;
  std::map<int, std::vector<double> > weights_;
  std::vector<std::pair<int, int> > etaphi_;
  std::vector<std::pair<int, int> > runs_, ids_;
  std::map<std::pair<int, int>, std::vector<double> > corrs_;
};

CalibCorrFactor::CalibCorrFactor(const char* infile, int useScale, double scale, bool etamax, bool marina, bool debug)
    : useScale_(useScale), scale_(scale), etaMax_(etamax), debug_(debug), etamp_(0), etamn_(0) {
  for (int i = 0; i < depMax_; ++i) {
    cfacmp_[i] = 1.0;
    cfacmn_[i] = 1.0;
  }
  if (std::string(infile) != "") {
    corrE_ = readCorrFactor(infile, marina);
    std::cout << "Reads " << cfactors_.size() << " correction factors from " << infile << " with flag " << corrE_
              << std::endl
              << "Flag for scale " << useScale_ << " with scale " << scale_ << "; flag for etaMax " << etaMax_
              << " and flag for Format " << marina << std::endl;
  } else {
    corrE_ = false;
    std::cout << "No correction factors provided; Flag for scale " << useScale_ << " with scale " << scale_
              << "; flag for etaMax " << etaMax_ << " and flag for Format " << marina << std::endl;
  }
}

double CalibCorrFactor::getCorr(unsigned int id) {
  double cfac(1.0);
  if (corrE_) {
    if (cfactors_.size() > 0) {
      int subdet, zside, ieta, iphi, depth;
      unpackDetId(id, subdet, zside, ieta, iphi, depth);
      std::map<std::pair<int, int>, double>::const_iterator itr =
          cfactors_.find(std::pair<int, int>(zside * ieta, depth));
      if (itr != cfactors_.end()) {
        cfac = itr->second;
      } else if (etaMax_) {
        if (zside > 0 && ieta > etamp_)
          cfac = (depth < depMax_) ? cfacmp_[depth] : cfacmp_[depMax_ - 1];
        if (zside < 0 && ieta > -etamn_)
          cfac = (depth < depMax_) ? cfacmn_[depth] : cfacmn_[depMax_ - 1];
      }
    }
  } else if (useScale_ != 0) {
    int subdet, zside, ieta, iphi, depth;
    unpackDetId(id, subdet, zside, ieta, iphi, depth);
    cfac = getFactor(ieta);
  }
  return cfac;
}

bool CalibCorrFactor::readCorrFactor(const char* fname, bool marina) {
  bool ok(false);
  if (std::string(fname) != "") {
    std::ifstream fInput(fname);
    if (!fInput.good()) {
      std::cout << "Cannot open file " << fname << std::endl;
    } else {
      char buffer[1024];
      unsigned int all(0), good(0);
      while (fInput.getline(buffer, 1024)) {
        ++all;
        if (buffer[0] == '#')
          continue;  //ignore comment
        std::vector<std::string> items = splitString(std::string(buffer));
        if (items.size() != 5) {
          std::cout << "Ignore  line: " << buffer << std::endl;
        } else {
          ++good;
          int ieta = (marina) ? std::atoi(items[0].c_str()) : std::atoi(items[1].c_str());
          int depth = (marina) ? std::atoi(items[1].c_str()) : std::atoi(items[2].c_str());
          float corrf = std::atof(items[3].c_str());
          double scale = getFactor(std::abs(ieta));
          cfactors_[std::pair<int, int>(ieta, depth)] = scale * corrf;
          if (ieta > etamp_ && depth == 1)
            etamp_ = ieta;
          if (ieta == etamp_ && depth < depMax_)
            cfacmp_[depth] = scale * corrf;
          if (ieta < etamn_ && depth == 1)
            etamn_ = ieta;
          if (ieta == etamn_ && depth < depMax_)
            cfacmn_[depth] = scale * corrf;
        }
      }
      fInput.close();
      std::cout << "Reads total of " << all << " and " << good << " good records"
                << " Max eta (z>0) " << etamp_ << " eta (z<0) " << etamn_ << std::endl;
      for (int i = 0; i < depMax_; ++i)
        std::cout << "[" << i << "] C+ " << cfacmp_[i] << " C- " << cfacmn_[i] << std::endl;
      if (good > 0)
        ok = true;
    }
  }
  return ok;
}

double CalibCorrFactor::getFactor(const int& ieta) {
  double scale(1.0);
  if (ieta < 16) {
    if ((useScale_ == 1) || (useScale_ == 3))
      scale = scale_;
  } else {
    if ((useScale_ == 2) || (useScale_ == 3))
      scale = scale_;
  }
  return scale;
}

CalibCorr::CalibCorr(const char* infile, int flag, bool debug) : flag_(flag), debug_(debug) {
  std::cout << "CalibCorr is created with flag " << flag << ":" << flag_ << " for i/p file " << infile << std::endl;
  if (flag == 1)
    good_ = readCorrDepth(infile);
  else if (flag == 2)
    good_ = readCorrResp(infile);
  else if (flag == 3)
    good_ = readCorrPU(infile);
  else if (flag == 4)
    good_ = readCorrPhi(infile);
  else if (flag == 5)
    good_ = readCorrPhis(infile);
  else
    good_ = readCorrRun(infile);
}

float CalibCorr::getCorr(int run, unsigned int id) {
  float cfac(1.0);
  if (good_ == 0)
    return cfac;
  unsigned idx = correctDetId(id);
  if (flag_ == 1) {
    std::map<unsigned int, float>::iterator itr = corrFacDepth_.find(idx);
    if (itr != corrFacDepth_.end())
      cfac = itr->second;
  } else if (flag_ == 2) {
    std::map<unsigned int, float>::iterator itr = corrFacResp_.find(idx);
    if (itr != corrFacResp_.end())
      cfac = itr->second;
  } else if (flag_ == 4) {
    cfac = getPhiCorr(idx);
  } else if (flag_ == 5) {
    cfac = getPhiCorrs(run, idx);
  } else {
    int ip(-1);
    for (unsigned int k = 0; k < runlow_.size(); ++k) {
      unsigned int i = runlow_.size() - k - 1;
      if (run >= runlow_[i]) {
        ip = (int)(i);
        break;
      }
    }
    if (debug_) {
      std::cout << "Run " << run << " Perdiod " << ip << std::endl;
    }
    if (ip >= 0) {
      std::map<unsigned int, float>::iterator itr = corrFac_[ip].find(idx);
      if (itr != corrFac_[ip].end())
        cfac = itr->second;
    }
  }
  if (debug_) {
    int subdet, zside, ieta, iphi, depth;
    unpackDetId(idx, subdet, zside, ieta, iphi, depth);
    std::cout << "ID " << std::hex << id << std::dec << " (Sub " << subdet << " eta " << zside * ieta << " phi " << iphi
              << " depth " << depth << ")  Factor " << cfac << std::endl;
  }
  return cfac;
}

double CalibCorr::getCorr(const Long64_t& entry) {
  if (good_ == 0)
    return 1.0;
  double cfac(0.0);
  std::map<Long64_t, double>::iterator itr = cfactors_.find(entry);
  if (itr != cfactors_.end())
    cfac = std::min(itr->second, 10.0);
  return cfac;
}

double CalibCorr::getTrueCorr(const Long64_t& entry) {
  if (good_ == 0)
    return 1.0;
  double cfac(0.0);
  std::map<Long64_t, double>::iterator itr = cfactors_.find(entry);
  if (itr != cfactors_.end())
    cfac = itr->second;
  return cfac;
}

double CalibCorr::getPhiCorr(unsigned int idx) {
  double cfac(1.0);
  if (good_ == 0)
    return cfac;
  std::map<unsigned int, double>::iterator itr = corrPhiSym_.find(idx);
  if (itr != corrPhiSym_.end())
    cfac = itr->second;
  if (debug_) {
    int subdet, zside, ieta, iphi, depth;
    unpackDetId(idx, subdet, zside, ieta, iphi, depth);
    std::cout << "ID " << std::hex << idx << std::dec << " (Sub " << subdet << " eta " << zside * ieta << " phi "
              << iphi << " depth " << depth << ")  Factor " << cfac << std::endl;
  }
  return cfac;
}

double CalibCorr::getPhiCorrs(int run, unsigned int idx) {
  double cfac(1.0);
  if (good_ == 0)
    return cfac;
  unsigned int runid = runId(run);
  std::map<std::pair<unsigned int, unsigned int>, double>::iterator itr =
      corrPhiSyms_.find(std::pair<unsigned int, unsigned int>(runid, idx));
  if (itr != corrPhiSyms_.end())
    cfac = itr->second;
  if (debug_) {
    int subdet, zside, ieta, iphi, depth;
    unpackDetId(idx, subdet, zside, ieta, iphi, depth);
    std::cout << "Run " << run << " ID " << runid << ":" << std::hex << idx << std::dec << " (Sub " << subdet << " eta "
              << zside * ieta << " phi " << iphi << " depth " << depth << ")  Factor " << cfac << std::endl;
  }
  return cfac;
}

bool CalibCorr::absent(const Long64_t& entry) { return (cfactors_.find(entry) == cfactors_.end()); }

bool CalibCorr::present(const Long64_t& entry) { return (cfactors_.find(entry) != cfactors_.end()); }

unsigned int CalibCorr::readCorrRun(const char* infile) {
  std::cout << "Enters readCorrRun for " << infile << std::endl;
  std::ifstream fInput(infile);
  unsigned int all(0), good(0), ncorr(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 5) == "#IOVs") {
        std::vector<std::string> items = splitString(bufferString.substr(6));
        ncorr = items.size() - 1;
        for (unsigned int n = 0; n < ncorr; ++n) {
          int run = std::atoi(items[n].c_str());
          runlow_.push_back(run);
        }
        std::cout << ncorr << ":" << runlow_.size() << " Run ranges" << std::endl;
        for (unsigned int n = 0; n < runlow_.size(); ++n)
          std::cout << " [" << n << "] " << runlow_[n];
        std::cout << std::endl;
      } else if (buffer[0] == '#') {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() != ncorr + 3) {
          std::cout << "Ignore  line: " << buffer << std::endl;
        } else {
          ++good;
          int ieta = std::atoi(items[0].c_str());
          int iphi = std::atoi(items[1].c_str());
          int depth = std::atoi(items[2].c_str());
          unsigned int id = getDetIdHE(ieta, iphi, depth);
          for (unsigned int n = 0; n < ncorr; ++n) {
            float corrf = std::atof(items[n + 3].c_str());
            if (n < nmax_)
              corrFac_[n][id] = corrf;
          }
          if (debug_) {
            std::cout << "ID " << std::hex << id << std::dec << ":" << id << " (eta " << ieta << " phi " << iphi
                      << " depth " << depth << ")";
            for (unsigned int n = 0; n < ncorr; ++n)
              std::cout << " " << corrFac_[n][id];
            std::cout << std::endl;
          }
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records of run dependent corrections from "
              << infile << std::endl;
  }
  return good;
}

unsigned int CalibCorr::readCorrDepth(const char* infile) {
  std::cout << "Enters readCorrDepth for " << infile << std::endl;
  unsigned int all(0), good(0);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 5) == "depth") {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() != 3) {
          std::cout << "Ignore  line: " << buffer << " Size " << items.size();
          for (unsigned int k = 0; k < items.size(); ++k)
            std::cout << " [" << k << "] : " << items[k];
          std::cout << std::endl;
        } else {
          ++good;
          int ieta = std::atoi(items[1].c_str());
          int depth = std::atoi(items[0].c_str());
          float corrf = std::atof(items[2].c_str());
          int nphi = (std::abs(ieta) > 20) ? 36 : 72;
          for (int i = 1; i <= nphi; ++i) {
            int iphi = (nphi > 36) ? i : (2 * i - 1);
            unsigned int id = getDetIdHE(ieta, iphi, depth);
            corrFacDepth_[id] = corrf;
            if (debug_) {
              std::cout << "ID " << std::hex << id << std::dec << ":" << id << " (eta " << ieta << " phi " << iphi
                        << " depth " << depth << ") " << corrFacDepth_[id] << std::endl;
            }
          }
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records of depth dependent factors from "
              << infile << std::endl;
  }
  return good;
}

unsigned int CalibCorr::readCorrResp(const char* infile) {
  std::cout << "Enters readCorrResp for " << infile << std::endl;
  unsigned int all(0), good(0), other(0);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 1) == "#") {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() < 5) {
          std::cout << "Ignore  line: " << buffer << " Size " << items.size();
          for (unsigned int k = 0; k < items.size(); ++k)
            std::cout << " [" << k << "] : " << items[k];
          std::cout << std::endl;
        } else if (items[3] == "HB" || items[3] == "HE") {
          ++good;
          int ieta = std::atoi(items[0].c_str());
          int iphi = std::atoi(items[1].c_str());
          int depth = std::atoi(items[2].c_str());
          int subdet = (items[3] == "HE") ? 2 : 1;
          float corrf = std::atof(items[4].c_str());
          unsigned int id = getDetId(subdet, ieta, iphi, depth);
          corrFacResp_[id] = corrf;
          if (debug_) {
            std::cout << "ID " << std::hex << id << std::dec << ":" << id << " (subdet " << items[3] << ":" << subdet
                      << " eta " << ieta << " phi " << iphi << " depth " << depth << ") " << corrFacResp_[id]
                      << std::endl;
          }
        } else {
          ++other;
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good and " << other
              << " detector records of depth dependent factors from " << infile << std::endl;
  }
  return good;
}

unsigned int CalibCorr::readCorrPU(const char* infile) {
  if (std::string(infile) != "") {
    std::ifstream fInput(infile);
    if (!fInput.good()) {
      std::cout << "Cannot open file " << infile << std::endl;
    } else {
      double val1, val2;
      cfactors_.clear();
      while (1) {
        fInput >> val1 >> val2;
        if (!fInput.good())
          break;
        Long64_t entry = (Long64_t)(val1);
        cfactors_[entry] = val2;
      }
      fInput.close();
    }
  }
  std::cout << "Reads " << cfactors_.size() << " PU correction factors from " << infile << std::endl;
  return cfactors_.size();
}

unsigned int CalibCorr::readCorrPhi(const char* infile) {
  std::cout << "Enters readCorrPhi for " << infile << std::endl;
  unsigned int all(0), good(0);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 1) == "#") {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() < 5) {
          std::cout << "Ignore  line: " << buffer << " Size " << items.size();
          for (unsigned int k = 0; k < items.size(); ++k)
            std::cout << " [" << k << "] : " << items[k];
          std::cout << std::endl;
        } else {
          ++good;
          int subdet = std::atoi(items[0].c_str());
          int ieta = std::atoi(items[1].c_str());
          int iphi = std::atoi(items[2].c_str());
          int depth = std::atoi(items[3].c_str());
          double corrf = std::atof(items[4].c_str());
          unsigned int id = packDetId(subdet, ieta, iphi, depth);
          corrPhiSym_[id] = corrf;
          if (debug_)
            std::cout << "ID " << std::hex << id << std::dec << ":" << id << " (subdet " << subdet << " eta " << ieta
                      << " phi " << iphi << " depth " << depth << ") " << corrPhiSym_[id] << std::endl;
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records of phi-symmetry factors from " << infile
              << std::endl;
  }
  return good;
}

unsigned int CalibCorr::readCorrPhis(const char* infile0) {
  std::cout << "Enters readCorrPhis for " << infile0 << std::endl;
  unsigned int all(0), good(0);
  std::ifstream fIn(infile0);
  if (!fIn.good()) {
    std::cout << "Cannot open file " << infile0 << std::endl;
  } else {
    char buffer0[1024];
    while (fIn.getline(buffer0, 1024)) {
      std::string bufferString0(buffer0);
      if (bufferString0.substr(0, 1) == "#") {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> item = splitString(bufferString0);
        if (item.size() > 2) {
          int run1 = std::atoi(item[0].c_str());
          int run2 = std::atoi(item[1].c_str());
          std::string infile(item[2]);
          unsigned int runid = runs_.size();
          runs_.push_back(std::pair<int, int>(run1, run2));
          std::ifstream fInput(infile.c_str());
          if (!fInput.good()) {
            std::cout << "Cannot open file " << infile << std::endl;
          } else {
            char buffer[1024];
            while (fInput.getline(buffer, 1024)) {
              ++all;
              std::string bufferString(buffer);
              if (bufferString.substr(0, 1) == "#") {
                continue;  //ignore other comments
              } else {
                std::vector<std::string> items = splitString(bufferString);
                if (items.size() < 5) {
                  std::cout << "Ignore  line: " << buffer << " Size " << items.size();
                  for (unsigned int k = 0; k < items.size(); ++k)
                    std::cout << " [" << k << "] : " << items[k];
                  std::cout << std::endl;
                } else {
                  ++good;
                  int subdet = std::atoi(items[0].c_str());
                  int ieta = std::atoi(items[1].c_str());
                  int iphi = std::atoi(items[2].c_str());
                  int depth = std::atoi(items[3].c_str());
                  double corrf = std::atof(items[4].c_str());
                  unsigned int id = packDetId(subdet, ieta, iphi, depth);
                  corrPhiSyms_[std::pair<int, int>(runid, id)] = corrf;
                  if (debug_)
                    std::cout << "ID " << runid << ":" << std::hex << id << std::dec << ":" << id << " (subdet "
                              << subdet << " eta " << ieta << " phi " << iphi << " depth " << depth << ") "
                              << corrPhiSym_[id] << std::endl;
                }
              }
            }
            fInput.close();
          }
        }
      }
    }
    fIn.close();
    std::cout << "Reads total of " << all << " and " << good << " good records of phi-symmetry factors from "
              << runs_.size() << " files as given in " << infile0 << std::endl;
  }
  return good;
}

unsigned int CalibCorr::runId(const int& run) {
  unsigned int id(runs_.size());
  for (unsigned int k = 0; k < runs_.size(); ++k) {
    if ((run >= runs_[k].first) && (run <= runs_[k].second)) {
      id = k;
      break;
    }
  }
  return id;
}

unsigned int CalibCorr::getDetIdHE(int ieta, int iphi, int depth) { return getDetId(2, ieta, iphi, depth); }

unsigned int CalibCorr::getDetId(int subdet, int ieta, int iphi, int depth) {
  // All numbers used here are described as masks/offsets in
  // DataFormats/HcalDetId/interface/HcalDetId.h
  unsigned int id_ = ((4 << 28) | ((subdet & 0x7) << 25));
  id_ |= ((0x1000000) | ((depth & 0xF) << 20) | ((ieta > 0) ? (0x80000 | (ieta << 10)) : ((-ieta) << 10)) |
          (iphi & 0x3FF));
  return id_;
}

unsigned int CalibCorr::correctDetId(const unsigned int& detId) {
  int subdet, ieta, zside, depth, iphi;
  unpackDetId(detId, subdet, zside, ieta, iphi, depth);
  if (subdet == 0) {
    if (ieta > 16)
      subdet = 2;
    else if (ieta == 16 && depth > 2)
      subdet = 2;
    else
      subdet = 1;
  }
  unsigned int id = getDetId(subdet, ieta * zside, iphi, depth);
  if ((id != detId) && debug_) {
    std::cout << "Correct Id " << std::hex << detId << " to " << id << std::dec << "(Sub " << subdet << " eta "
              << ieta * zside << " phi " << iphi << " depth " << depth << ")" << std::endl;
  }
  return id;
}

CalibSelectRBX::CalibSelectRBX(const char* rbxFile, bool debug) : debug_(debug) {
  std::cout << "Enters CalibSelectRBX for " << rbxFile << std::endl;
  unsigned int all(0), good(0);
  std::vector<int> rbxs;
  std::ifstream fInput(rbxFile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << rbxFile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 1) == "#") {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() != 1) {
          std::cout << "Ignore  line: " << buffer << " Size " << items.size() << std::endl;
        } else {
          ++good;
          int rbx = std::atoi(items[0].c_str());
          rbxs.push_back(rbx);
          int zside = (rbx > 0) ? 1 : -1;
          int subdet = (std::abs(rbx) / 100) % 10;
          if (subdet != 1)
            subdet = 2;
          int iphis = std::abs(rbx) % 100;
          if (iphis > 0 && iphis <= 18) {
            for (int i = 0; i < 4; ++i) {
              int iphi = (iphis - 2) * 4 + 3 + i;
              if (iphi < 1)
                iphi += 72;
              int zsphi = zside * (subdet * 100 + iphi);
              zsphis_.push_back(zsphi);
            }
          }
        }
      }
    }
    fInput.close();
  }
  std::cout << "Select a set of RBXs " << rbxs.size() << " by reading " << all << ":" << good << " records from "
            << rbxFile << " with " << zsphis_.size() << " iphi values:";
  for (unsigned int i = 0; i < zsphis_.size(); ++i)
    std::cout << " " << zsphis_[i];
  std::cout << std::endl;
}

bool CalibSelectRBX::isItRBX(const unsigned int detId) {
  bool ok(true);
  if (zsphis_.size() > 0) {
    int subdet, ieta, zside, depth, iphi;
    unpackDetId(detId, subdet, zside, ieta, iphi, depth);
    int zsphi = zside * (subdet * 100 + iphi);
    ok = (std::find(zsphis_.begin(), zsphis_.end(), zsphi) != zsphis_.end());

    if (debug_) {
      std::cout << "isItRBX:subdet|zside|iphi " << subdet << ":" << zside << ":" << iphi << " OK " << ok << std::endl;
    }
  }
  return ok;
}

bool CalibSelectRBX::isItRBX(const std::vector<unsigned int>* detId) {
  bool ok(true);
  if (zsphis_.size() > 0) {
    ok = true;
    for (unsigned int i = 0; i < detId->size(); ++i) {
      int subdet, ieta, zside, depth, iphi;
      unpackDetId((*detId)[i], subdet, zside, ieta, iphi, depth);
      int zsphi = zside * (subdet * 100 + iphi);
      ok = (std::find(zsphis_.begin(), zsphis_.end(), zsphi) != zsphis_.end());
      if (debug_) {
        std::cout << "isItRBX: subdet|zside|iphi " << subdet << ":" << zside << ":" << iphi << " OK " << ok
                  << std::endl;
      }
      if (ok)
        break;
    }
  }
  if (debug_)
    std::cout << "isItRBX: size " << detId->size() << " OK " << ok << std::endl;
  return ok;
}

bool CalibSelectRBX::isItRBX(const int ieta, const int iphi) {
  bool ok(true);
  if (zsphis_.size() == 4) {
    int zside = (ieta > 0) ? 1 : -1;
    int subd1 = (std::abs(ieta) <= 16) ? 1 : 2;
    int subd2 = (std::abs(ieta) >= 16) ? 2 : 1;
    int zsphi1 = zside * (subd1 * 100 + iphi);
    int zsphi2 = zside * (subd2 * 100 + iphi);
    ok = ((std::find(zsphis_.begin(), zsphis_.end(), zsphi1) != zsphis_.end()) ||
          (std::find(zsphis_.begin(), zsphis_.end(), zsphi2) != zsphis_.end()));
  }
  if (debug_) {
    std::cout << "isItRBX: ieta " << ieta << " iphi " << iphi << " OK " << ok << std::endl;
  }
  return ok;
}

CalibDuplicate::CalibDuplicate(const char* fname, int flag, bool debug) : flag_(flag), debug_(debug), ok_(false) {
  if (flag_ == 0) {
    if (strcmp(fname, "") != 0) {
      std::ifstream infile(fname);
      if (!infile.is_open()) {
        std::cout << "Cannot open duplicate file " << fname << std::endl;
      } else {
        while (1) {
          Long64_t jentry;
          infile >> jentry;
          if (!infile.good())
            break;
          entries_.push_back(jentry);
        }
        infile.close();
        std::cout << "Reads a list of " << entries_.size() << " events from " << fname << std::endl;
        if (entries_.size() > 0)
          ok_ = true;
      }
    } else {
      std::cout << "No duplicate events in the input file" << std::endl;
    }
  } else if (flag_ == 1) {
    if (strcmp(fname, "") != 0) {
      std::ifstream infile(fname);
      if (!infile.is_open()) {
        std::cout << "Cannot open depth dependence file " << fname << std::endl;
      } else {
        unsigned int all(0), good(0);
        char buffer[1024];
        while (infile.getline(buffer, 1024)) {
          ++all;
          std::string bufferString(buffer);
          if (bufferString.substr(0, 1) == "#") {
            continue;  //ignore other comments
          } else {
            std::vector<std::string> items = splitString(bufferString);
            if (items.size() < 3) {
              std::cout << "Ignore  line: " << buffer << " Size " << items.size();
              for (unsigned int k = 0; k < items.size(); ++k)
                std::cout << " [" << k << "] : " << items[k];
              std::cout << std::endl;
            } else {
              ++good;
              int ieta = std::atoi(items[0].c_str());
              std::vector<double> weights;
              for (unsigned int k = 1; k < items.size(); ++k) {
                double corrf = std::atof(items[k].c_str());
                weights.push_back(corrf);
              }
              weights_[ieta] = weights;
              if (debug_) {
                std::cout << "Eta " << ieta << " with " << weights.size() << " depths having weights:";
                for (unsigned int k = 0; k < weights.size(); ++k)
                  std::cout << " " << weights[k];
                std::cout << std::endl;
              }
            }
          }
        }
        infile.close();
        std::cout << "Reads total of " << all << " and " << good << " good records of depth dependent factors from "
                  << fname << std::endl;
        if (good > 0)
          ok_ = true;
      }
    }
  } else if (flag_ == 3) {
    ok_ = readCorrFactor(fname);
  } else {
    flag_ = 2;
    if (strcmp(fname, "") != 0) {
      std::ifstream infile(fname);
      if (!infile.is_open()) {
        std::cout << "Cannot open rejection file " << fname << std::endl;
      } else {
        unsigned int all(0), good(0);
        char buffer[1024];
        while (infile.getline(buffer, 1024)) {
          ++all;
          std::string bufferString(buffer);
          if (bufferString.substr(0, 1) == "#") {
            continue;  //ignore other comments
          } else {
            std::vector<std::string> items = splitString(bufferString);
            if (items.size() != 2) {
              std::cout << "Ignore  line: " << buffer << " Size " << items.size();
              for (unsigned int k = 0; k < items.size(); ++k)
                std::cout << " [" << k << "] : " << items[k];
              std::cout << std::endl;
            } else {
              ++good;
              int ieta = std::atoi(items[0].c_str());
              int iphi = std::atoi(items[1].c_str());
              etaphi_.push_back(std::pair<int, int>(ieta, iphi));
              if (debug_)
                std::cout << "Select channels with iEta " << ieta << " iPhi " << iphi << std::endl;
            }
          }
        }
        infile.close();
        std::cout << "Reads total of " << all << " and " << good << " good records of rejection candidates from "
                  << fname << std::endl;
        if (good > 0)
          ok_ = true;
      }
    }
  }
}

bool CalibDuplicate::isDuplicate(long entry) {
  if (ok_)
    return (std::find(entries_.begin(), entries_.end(), entry) != entries_.end());
  else
    return false;
}

double CalibDuplicate::getWeight(unsigned int detId) {
  double wt(1.0);
  if (ok_) {
    int subdet, ieta, zside, depth, iphi;
    unpackDetId(detId, subdet, zside, ieta, iphi, depth);
    std::map<int, std::vector<double> >::const_iterator itr = weights_.find(ieta);
    --depth;
    if (depth < 0)
      std::cout << "Strange Depth value " << depth << " in " << std::hex << detId << std::dec << std::endl;
    if (itr != weights_.end()) {
      if (depth < static_cast<int>(itr->second.size()))
        wt = itr->second[depth];
    }
  }
  return wt;
}

bool CalibDuplicate::select(int ieta, int iphi) {
  bool flag(false);
  if ((ok_) && (flag_ == 2))
    flag = (std::find(etaphi_.begin(), etaphi_.end(), std::pair<int, int>(ieta, iphi)) != etaphi_.end());
  if (debug_)
    std::cout << "Input " << ieta << ":" << iphi << " Flags " << ok_ << ":" << flag_ << ":" << flag << std::endl;
  return flag;
}

double CalibDuplicate::getCorr(int run, int ieta, int depth) {
  std::map<std::pair<int, int>, std::vector<double> >::const_iterator itr =
      corrs_.find(std::pair<int, int>(ieta, depth));
  double corr(1.0);
  if (itr != corrs_.end()) {
    unsigned int irun = runId(run);
    corr = (itr->second)[irun];
  }
  return corr;
}

bool CalibDuplicate::readCorrFactor(const char* infile) {
  bool flag(false);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "CalibDuplicate::readCorrFactor:Cannot open file " << infile << std::endl;
  } else {
    int nrun, neta;
    fInput >> nrun >> neta;
    int indx, run1, run2;
    for (int k = 0; k < nrun; ++k) {
      fInput >> indx >> run1 >> run2;
      runs_.push_back(std::pair<int, int>(run1, run2));
    }
    int eta, depth;
    double corr;
    for (int k = 0; k < neta; ++k) {
      fInput >> eta >> depth;
      std::vector<double> corrs;
      for (int i = 0; i < nrun; ++i) {
        fInput >> corr;
        corrs.push_back(corr);
      }
      corrs_[std::pair<int, int>(eta, depth)] = corrs;
    }
    fInput.close();
    std::cout << "CalibDuplicate::readCorrFactor:Reads information of " << runs_.size() << " runs and " << corrs_.size()
              << " channels from " << infile << std::endl;
    flag = true;
    if (debug_) {
      for (unsigned int k = 0; k < runs_.size(); ++k)
        std::cout << "Run range[" << k << "] " << runs_[k].first << ":" << runs_[k].second << std::endl;
      std::map<std::pair<int, int>, std::vector<double> >::const_iterator itr;
      for (itr = corrs_.begin(); itr != corrs_.end(); ++itr) {
        std::cout << "eta:depth [" << (itr->first).first << ":" << (itr->first).second << "]";
        for (unsigned int i = 0; i < (itr->second).size(); ++i)
          std::cout << " " << (itr->second)[i];
        std::cout << std::endl;
      }
    }
  }
  return flag;
}

unsigned int CalibDuplicate::runId(const int& run) {
  unsigned int id(runs_.size());
  for (unsigned int k = 0; k < runs_.size(); ++k) {
    if ((run >= runs_[k].first) && (run <= runs_[k].second)) {
      id = k;
      break;
    }
  }
  return id;
}

void CalibCorrTest(const char* infile, int flag) {
  if (flag >= 0) {
    CalibCorr* c1 = new CalibCorr(infile, flag, true);
    for (int ieta = 1; ieta < 29; ++ieta) {
      int subdet = (ieta > 16) ? 2 : 1;
      int depth = (ieta > 16) ? 2 : 1;
      unsigned int id1 = ((4 << 28) | ((subdet & 0x7) << 25));
      id1 |= ((0x1000000) | ((depth & 0xF) << 20) | (ieta << 10) | 1);
      c1->getCorr(0, id1);
      id1 |= (0x80000);
      c1->getCorr(0, id1);
    }
    delete c1;
  } else {
    CalibDuplicate* c1 = new CalibDuplicate(infile, -flag, true);
    delete c1;
  }
}

unsigned int stringTest(const std::string& str) {
  std::cout << str << " has " << str.size() << " characters\n";
  return str.size();
}

void CalibCorrScale(const char* infile, const char* outfile, double scale) {
  std::ofstream myfile;
  myfile.open(outfile);
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open '" << outfile << std::endl;
  } else {
    if (std::string(infile) != "") {
      std::ifstream fInput(infile);
      if (!fInput.good()) {
        std::cout << "Cannot open file " << infile << std::endl;
      } else {
        char buffer[1024];
        unsigned int all(0), good(0), comment(0);
        while (fInput.getline(buffer, 1024)) {
          ++all;
          if (buffer[0] == '#') {
            myfile << buffer << std::endl;
            ++comment;
            continue;  //ignore comment
          }
          std::vector<std::string> items = splitString(std::string(buffer));
          if (items.size() != 5) {
            std::cout << "Ignore  line: " << buffer << std::endl;
          } else {
            ++good;
            int ieta = std::atoi(items[1].c_str());
            int depth = std::atoi(items[2].c_str());
            float corrf = scale * std::atof(items[3].c_str());
            float dcorr = scale * std::atof(items[4].c_str());
            myfile << std::setw(10) << items[0] << std::setw(10) << std::dec << ieta << std::setw(10) << depth
                   << std::setw(10) << corrf << " " << std::setw(10) << dcorr << std::endl;
          }
        }
        fInput.close();
        std::cout << "Reads total of " << all << ", " << comment << " and " << good << " good records from " << infile
                  << " and copied to " << outfile << std::endl;
      }
    }
    myfile.close();
  }
}

void CalibCorrScale2(const char* infile, const char* outfile, double scaleB, double scaleT, double scaleE) {
  int ietasL[3] = {0, 13, 17};
  int ietasH[3] = {14, 18, 29};
  double scale[3] = {scaleB, scaleT, scaleE};
  std::ofstream myfile;
  myfile.open(outfile);
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open '" << outfile << std::endl;
  } else {
    if (std::string(infile) != "") {
      std::ifstream fInput(infile);
      if (!fInput.good()) {
        std::cout << "Cannot open file " << infile << std::endl;
      } else {
        char buffer[1024];
        unsigned int all(0), good(0), comment(0);
        while (fInput.getline(buffer, 1024)) {
          ++all;
          if (buffer[0] == '#') {
            myfile << buffer << std::endl;
            ++comment;
            continue;  //ignore comment
          }
          std::vector<std::string> items = splitString(std::string(buffer));
          if (items.size() != 5) {
            std::cout << "Ignore  line: " << buffer << std::endl;
          } else {
            ++good;
            int ieta = std::atoi(items[1].c_str());
            int depth = std::atoi(items[2].c_str());
            int jp(-1);
            for (int j = 0; j < 3; ++j) {
              if (std::abs(ieta) > ietasL[j] && std::abs(ieta) <= ietasH[j]) {
                if (jp < 0)
                  jp = j;
              }
            }
            if (jp < 0)
              jp = 2;
            float corrf = scale[jp] * std::atof(items[3].c_str());
            float dcorr = scale[jp] * std::atof(items[4].c_str());
            myfile << std::setw(10) << items[0] << std::setw(10) << std::dec << ieta << std::setw(10) << depth
                   << std::setw(10) << corrf << " " << std::setw(10) << dcorr << std::endl;
          }
        }
        fInput.close();
        std::cout << "Reads total of " << all << ", " << comment << " and " << good << " good records from " << infile
                  << " and copied to " << outfile << std::endl;
      }
    }
    myfile.close();
  }
}
#endif
