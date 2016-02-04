#ifndef BFit3D_h
#define BFit3D_h

#include <iostream>
#include "HarmBasis3DCyl.h"

//_______________________________________________________________________________
namespace magfieldparam {
class BFit3D {

private:

   //The following constants are defined in BFit3D_data.h
   static const double B_nom[4];   //Nom. field values at measurements
   static const double C0[360][4]; //Expansion coeffs. at measurements
   static const double C1[360][5]; //First derivatives of coeffs.
   static const double C2[360][3]; //Second derivatives of coeffs.

   double C[360]; //interpolated expansion coeefs. for the field B_set
   
   bool   use_spline;
   bool   signed_rad;
   double B_set;

   HarmBasis3DCyl *HB;
 
   void SetCoeff_Linear(const double B);
   void SetCoeff_Spline(const double B);

public:

   //Defaults: piecewise linear interpolation is used to calculate
   //expansion coefficients for intermidiate field values,
   //Signed "R" coordinate is accepted
   BFit3D() : use_spline(false), signed_rad(true), B_set(0.), 
              HB(new HarmBasis3DCyl(18)) {}

   virtual ~BFit3D() { delete HB;}
   
   //Set the interpolation type (cubic spline or linear piecewise)
   void UseSpline   (const bool flag = true) { use_spline = flag;}

   //Switch between signed and unsigned "R" modes
   void UseSignedRad(const bool flag = true) { signed_rad = flag;}
   
   //BASIC FUNCTION: Set nominal field
   void SetField(const double B) {
      if (use_spline) SetCoeff_Spline(B); else SetCoeff_Linear(B);
      B_set = B;
   }

   //BASIC FUNCTION: Return field components at the point (r,z,phi)
   void GetField(const double r, const double z, const double phi,
                 double &Br, double &Bz, double &Bphi);

   //All the following functions are provided for diagnostic purposes

   unsigned GetLen() { return HB->GetLen();} //Ret. the basis length
   double GetBnom() { return B_set;}         //Ret. nominal field
   double GetC(const int k) { return C[k];}  //Ret. k-th expansion coefficient

   //The following functions return values of the k-th basis component
   //(B_r, B_z or B_phi) at the point which is set by last GetField(...)
   double GetBr_k  (const unsigned k) { return HB->GetBr_k  (k);}
   double GetBz_k  (const unsigned k) { return HB->GetBz_k  (k);}
   double GetBphi_k(const unsigned k) { return HB->GetBphi_k(k);}
   
   //The following functions prints the basis polynomials for the scalar
   //field potential, B_r, B_z or B_phi.
   void PrintPtnPoly (std::ostream &out = std::cout) { HB->PrintPtB  (out);}
   void PrintBrPoly  (std::ostream &out = std::cout) { HB->PrintBrB  (out);}
   void PrintBzPoly  (std::ostream &out = std::cout) { HB->PrintBzB  (out);}
   void PrintBphiPoly(std::ostream &out = std::cout) { HB->PrintBphiB(out);}
   void PrintPoly    (std::ostream &out = std::cout) { HB->Print     (out);}

};
}
#endif
