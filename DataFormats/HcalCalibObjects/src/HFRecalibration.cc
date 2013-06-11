///////////////////////////////////////////////////////////////////////////////
// File: HFRecalibration.cc
// Description: simple helper class containing parameterized 
//              function for HF damade recovery for Upgrade studies  
//              evaluated using SimG4CMS/Calo/ HFDarkening   
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/HcalCalibObjects/interface/HFRecalibration.h"

HFRecalibration::HFRecalibration() { }
HFRecalibration::~HFRecalibration() { }

double HFRecalibration::getCorr(int ieta, int depth, double lumi) {

  // parameterizations provided by James Wetzel
 
  ieta = abs(ieta);

   if (depth == 1)
   {
      switch (ieta) {
         case 29:
            return (1 + 0.00311431*sqrt(lumi) - 1.2533e-05*lumi);
            break;
         case 30:
            return (1 + 0.00465755*sqrt(lumi) - 2.13133e-05*lumi);
            break;
         case 31:
            return (1 + 0.000977613*sqrt(lumi) - 4.07977e-06*lumi);
            break;
         case 32:
            return (1 + 0.000524346*sqrt(lumi) + 4.9271e-06*lumi);
            break;
         case 33:
            return (1 + 0.00173415*sqrt(lumi) - 2.24236e-06*lumi);
            break;
         case 34:
            return (1 + 0.00232593*sqrt(lumi) - 4.52383e-06*lumi);
            break;
         case 35:
            return (1 + 0.00320855*sqrt(lumi) - 2.99755e-06*lumi);
            break;
         case 36:
            return (1 + 0.0041762*sqrt(lumi) - 7.45472e-06*lumi);
            break;
         case 37:
            return (1 + 0.00470535*sqrt(lumi) - 8.54648e-06*lumi);
            break;
         case 38:
            return (1 + 0.0128799*sqrt(lumi) - 2.08496e-05*lumi);
            break;
         case 39:
            return (1 + 0.0132013*sqrt(lumi) - 2.20553e-06*lumi);
            break;
         case 40:
            return (1 + 0.0175412*sqrt(lumi) + 9.27508e-06*lumi);
            break;
         case 41:
            return (1 + 0.0225411*sqrt(lumi) + 6.27762e-06*lumi);
            break;
         default:
            return 1.0;
            break;
      }
   }
   else if (depth == 2)
   {
      switch (ieta) {
         case 29:
            return (1 + 0.00761858*sqrt(lumi) - 5.27388e-05*lumi);
            break;
         case 30:
            return (1 + 0.00504187*sqrt(lumi) - 2.37517e-05*lumi);
            break;
         case 31:
            return (1 + 0.000662802*sqrt(lumi) - 1.59053e-06*lumi);
            break;
         case 32:
            return (1 + 0.000309601*sqrt(lumi) + 5.12078e-06*lumi);
            break;
         case 33:
            return (1 + 0.00136072*sqrt(lumi) - 1.00731e-06*lumi);
            break;
         case 34:
            return (1 + 0.00162751*sqrt(lumi) - 9.75138e-07*lumi);
            break;
         case 35:
            return (1 + 0.00276588*sqrt(lumi) - 2.7936e-06*lumi);
            break;
         case 36:
            return (1 + 0.00350136*sqrt(lumi) - 5.80201e-06*lumi);
            break;
         case 37:
            return (1 + 0.00373748*sqrt(lumi) - 4.18957e-06*lumi);
            break;
         case 38:
            return (1 + 0.0114058*sqrt(lumi) - 2.09587e-05*lumi);
            break;
         case 39:
            return (1 + 0.0114551*sqrt(lumi) - 5.59902e-06*lumi);
            break;
         case 40:
            return (1 + 0.0168812*sqrt(lumi) - 1.00741e-05*lumi);
            break;
         case 41:
            return (1 + 0.018887*sqrt(lumi) - 6.77675e-06*lumi);
            break;
         default:
            return 1.0;
            break;
      }
   }
   else
      return 1.0;
}
