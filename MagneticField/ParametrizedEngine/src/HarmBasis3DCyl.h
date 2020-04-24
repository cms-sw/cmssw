#ifndef HarmBasis3DCyl_h
#define HarmBasis3DCyl_h

#include "rz_harm_poly.h"

namespace magfieldparam {

typedef std::vector<rz_harm_poly>  harm_poly_vec;
typedef std::vector<harm_poly_vec> harm_poly_arr;

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  HarmBasis3DCyl: set of basis harmonic polynomials in cylindrical CS        //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

class HarmBasis3DCyl {

private:
   unsigned Dim;   //Dimension of the basis
   unsigned Len;   //Length of the basis, accounting negative M's
   
   int *L_k, *M_k; //Translation arrays from linear to (L,M) address;
   double *P_k, *Br_k, *Bz_k, *Bphi_k; //Calculated values for (r,z) terms

   harm_poly_arr PtB;   //Potential basis
   harm_poly_arr BrB;   //Br basis
   harm_poly_arr BzB;   //Bz basis
   harm_poly_arr BphiB; //phi basis

   void   EvalRZ(harm_poly_arr &B, double *val);
   double GetVal(double *coeff, double *basis);

   void Print(harm_poly_arr &B,std::ostream &out = std::cout);
   
public:
   HarmBasis3DCyl(const unsigned N = 18); //The only legal constructor
   virtual ~HarmBasis3DCyl();

   unsigned GetDim() { return Dim;}
   unsigned GetLen() { return Len;}
   void     GetLM(const unsigned j, int &Lj, int &Mj) { Lj = L_k[j]; Mj = M_k[j];}
   
   //Sets point for the basis components evaluation
   void SetPoint(const double r, const double z, const double phi)
   { rz_harm_poly::SetPoint(r, z, phi);}
   
   //Fill tables with the basis component values. SetPoint(r,z,phi)
   //must be called before EvalXXX() calls.
   void EvalPtn() { EvalRZ(PtB, P_k);}
   void EvalBr()  { EvalRZ(BrB, Br_k);}
   void EvalBz()  { EvalRZ(BzB, Bz_k);}
   void EvalBphi();
   
   //Return the basis component value for the linear address k.
   //EvalXXX() must be called before GetXXX_k() call
   double GetPtn_k (const unsigned k) { return P_k[k];}
   double GetBr_k  (const unsigned k) { return Br_k[k];}
   double GetBz_k  (const unsigned k) { return Bz_k[k];}
   double GetBphi_k(const unsigned k) { return Bphi_k[k];}
   
   //Return the the potential and the field component values
   //resulted by the basis expansion with coefficients in <coeff>
   //EvalXXX() must be called before GetXXX() call
   double GetPtn (double *coeff) { return GetVal(coeff, P_k);}
   double GetBr  (double *coeff) { return GetVal(coeff, Br_k);}
   double GetBz  (double *coeff) { return GetVal(coeff, Bz_k);}
   double GetBphi(double *coeff) { return GetVal(coeff, Bphi_k);}

   void PrintPtB  (std::ostream &out = std::cout) { Print(PtB,   out);}
   void PrintBrB  (std::ostream &out = std::cout) { Print(BrB,   out);}
   void PrintBzB  (std::ostream &out = std::cout) { Print(BzB,   out);}
   void PrintBphiB(std::ostream &out = std::cout) { Print(BphiB, out);}
   void Print     (std::ostream &out = std::cout);

}; //class HarmBasis3DCyl
}

#endif
