///////////////////////////////////////////////////////////////////////////////
// File: HFRecalibration.cc
// Description: simple helper class containing parameterized 
//              function for HF damade recovery for Upgrade studies  
//              evaluated using SimG4CMS/Calo/ HFDarkening   
///////////////////////////////////////////////////////////////////////////////

#include "HFRecalibration.h"

HFRecalibration::HFRecalibration() { }
HFRecalibration::~HFRecalibration() { }

double HFRecalibration::getCorr(int ieta, int depth, double lumi) {

  // parameterizations provided by James Wetzel
 
  ieta = abs(ieta);

   if (depth == 1)
   {
      switch (ieta) {
         case 30:
            return (1 + 0.004123*sqrt(lumi) + -0.000004*lumi);
            break;
         case 31:
            return (1 + 0.006020*sqrt(lumi) + -0.000002*lumi);
            break;
         case 32:
            return (1 + 0.008201*sqrt(lumi) + 0.000000*lumi);
            break;
         case 33:
            return (1 + 0.010489*sqrt(lumi) + 0.000004*lumi);
            break;
         case 34:
            return (1 + 0.013379*sqrt(lumi) + 0.000015*lumi);
            break;
         case 35:
            return (1 + 0.016997*sqrt(lumi) + 0.000026*lumi);
            break;
         case 36:
            return (1 + 0.021464*sqrt(lumi) + 0.000063*lumi);
            break;
         case 37:
            return (1 + 0.027371*sqrt(lumi) + 0.000084*lumi);
            break;
         case 38:
            return (1 + 0.034195*sqrt(lumi) + 0.000160*lumi);
            break;
         case 39:
            return (1 + 0.044807*sqrt(lumi) + 0.000107*lumi);
            break;
         case 40:
            return (1 + 0.058939*sqrt(lumi) + 0.000425*lumi);
            break;
         case 41:
            return (1 + 0.125497*sqrt(lumi) + 0.000209*lumi);
            break;
         default:
            return 1.0;
            break;
      }
   }
   else if (depth == 2)
   {
      switch (ieta) {
         case 30:
            return (1 + 0.002861*sqrt(lumi) + -0.000002*lumi);
            break;
         case 31:
            return (1 + 0.004168*sqrt(lumi) + -0.000000*lumi);
            break;
         case 32:
            return (1 + 0.006400*sqrt(lumi) + -0.000007*lumi);
            break;
         case 33:
            return (1 + 0.008388*sqrt(lumi) + -0.000006*lumi);
            break;
         case 34:
            return (1 + 0.011601*sqrt(lumi) + -0.000002*lumi);
            break;
         case 35:
            return (1 + 0.014425*sqrt(lumi) + 0.000001*lumi);
            break;
         case 36:
            return (1 + 0.018633*sqrt(lumi) + 0.000019*lumi);
            break;
         case 37:
            return (1 + 0.023232*sqrt(lumi) + 0.000031*lumi);
            break;
         case 38:
            return (1 + 0.028274*sqrt(lumi) + 0.000067*lumi);
            break;
         case 39:
            return (1 + 0.035447*sqrt(lumi) + 0.000012*lumi);
            break;
         case 40:
            return (1 + 0.051579*sqrt(lumi) + 0.000157*lumi);
            break;
         case 41:
            return (1 + 0.086593*sqrt(lumi) + -0.000003*lumi);
            break;
         default:
            return 1.0;
            break;
      }
   }
   else
      return 1.0;
}
