#include <cmath>
#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

#include "BFit3D_data.h"

using namespace magfieldparam;

//_______________________________________________________________________________
void BFit3D::SetCoeff_Linear(const double B)
{
   unsigned jj, kk = 1;
   double w_0, w_1, B_mod = fabs(B);
   if (B_mod <= B_nom[0]) {
      w_0 = B / B_nom[0];
      for (jj = 0; jj < 360; ++jj) {
         C[jj] = w_0*C0[jj][0];
      }
   } else if (B_mod >= B_nom[3]) {
      w_0 = B / B_nom[3];
      for (jj = 0; jj < 360; ++jj) {
         C[jj] = w_0*C0[jj][3];
      }
   } else {
      while (B_nom[kk] < B_mod) ++kk; //Simple linear search
      w_1 = (B_mod - B_nom[kk-1])/(B_nom[kk] - B_nom[kk-1]);
      w_0 = 1.0 - w_1;
      if (B < 0.) {
         w_0 = -w_0;
         w_1 = -w_1;
      }
      for (jj = 0; jj < 360; ++jj) {
         C[jj] = w_0*C0[jj][kk-1] + w_1*C0[jj][kk];
      }
   }
}

//_______________________________________________________________________________
void BFit3D::SetCoeff_Spline(const double B)
{
   int jc, k0 = 0, k1 = 1;
   double dB2, dB = fabs(B);
   if (dB >= B_nom[3]) { //Linear extrapolation for a large field
      dB -= B_nom[3];
      for (jc = 0; jc < 360; ++jc) C[jc] = C0[jc][3] + C1[jc][4]*dB;
   } else {
      if (dB < B_nom[0]) {
         dB2 = dB*dB / (3.*B_nom[0]);
         for (jc = 0; jc < 360; ++jc) C[jc] = (C2[jc][0]*dB2 + C1[jc][0])*dB;
      } else {
         while (B_nom[k1] < dB) ++k1; //Simple linear search
         k0 = k1-1;
         dB2 = (dB -= B_nom[k0]) / (3.*(B_nom[k1] - B_nom[k0]));
         if (k1 < 3) { //"Regular" interval
            for (jc = 0; jc < 360; ++jc)
               C[jc] = (((C2[jc][k1] - C2[jc][k0])*dB2+C2[jc][k0])*dB + C1[jc][k1])*dB + C0[jc][k0];
         } else {      //The last interval
            dB2 = (1.- dB2)*dB;
            for (jc = 0; jc < 360; ++jc)
               C[jc] = (C2[jc][k0]*dB2 + C1[jc][k1])*dB + C0[jc][k0];
         }
      }
   }
   if (B < 0) for (jc = 0; jc < 360; ++jc) C[jc] = -C[jc];
}

//_______________________________________________________________________________
void BFit3D::GetField(const double r, const double z, const double phi,
                      double &Br, double &Bz, double &Bphi)
{
//Return field components in Br, Bz, Bphi. SetField must be called before use.
//
   if (signed_rad || (r >= 0.)) HB->SetPoint(r, z, phi);
   else HB->SetPoint(-r, z, phi+M_PI);
   HB->EvalBr();
   HB->EvalBz();
   HB->EvalBphi();
   Br   = HB->GetBr  (C);
   Bz   = HB->GetBz  (C);
   Bphi = HB->GetBphi(C);
}

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//                           T E S T  A R E A                                  //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

