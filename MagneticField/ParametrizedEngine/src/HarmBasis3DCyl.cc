/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  HarmBasis3DCyl: set of basis harmonic polynomials in cylindrical CS        //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

#include "HarmBasis3DCyl.h"

using namespace magfieldparam;


//_______________________________________________________________________________
HarmBasis3DCyl::HarmBasis3DCyl(const unsigned N)
{
//Construct a basis of dimension N
//
   Dim = N;
   Len = N*(N+2);
   
   L_k = new int [Len];
   M_k = new int [Len];
   
   P_k    = new double [Len];
   Br_k   = new double [Len];
   Bz_k   = new double [Len];
   Bphi_k = new double [Len];
   
   PtB.reserve(N);
   BrB.reserve(N);
   BzB.reserve(N);
   BphiB.reserve(N);
   
   rz_harm_poly::IncNPwr(N); //In order to prevent GetMaxPow() calls
   unsigned M, vLen, k = 0;
   for (unsigned L = 1; L <= N; ++L) {
      vLen = L+1;
      harm_poly_vec Pt_vec;   Pt_vec.reserve(vLen);
      harm_poly_vec Br_vec;   Br_vec.reserve(vLen);
      harm_poly_vec Bz_vec;   Bz_vec.reserve(vLen);
      harm_poly_vec Bphi_vec; Bphi_vec.reserve(vLen);

      Pt_vec.push_back  (rz_harm_poly(L));
      Br_vec.push_back  (Pt_vec[0].GetDiff(0));
      Bz_vec.push_back  (Pt_vec[0].GetDiff(1));
      Bphi_vec.push_back(rz_harm_poly());
      Bphi_vec[0].CheatL(L);
      
      L_k[k] = L; M_k[k] = 0; ++k;

      for (M = 1; M <= L; ++M) {
         Pt_vec.push_back  (Pt_vec[M-1].LadderUp());
         Br_vec.push_back  (Pt_vec[M].GetDiff(0));
         Bz_vec.push_back  (Pt_vec[M].GetDiff(1));
         Bphi_vec.push_back(Pt_vec[M].GetDecPow(0));
         Bphi_vec[M].Scale(M);
         L_k[k] = L; M_k[k] =  M; ++k;
         L_k[k] = L; M_k[k] = -M; ++k;
      }
      PtB.push_back  (Pt_vec);
      BrB.push_back  (Br_vec);
      BzB.push_back  (Bz_vec);
      BphiB.push_back(Bphi_vec);
   }
}

//_______________________________________________________________________________
HarmBasis3DCyl::~HarmBasis3DCyl()
{
   delete [] Bphi_k;
   delete [] Bz_k;
   delete [] Br_k;
   delete [] P_k;
   delete [] M_k;
   delete [] L_k;
}

//_______________________________________________________________________________
void HarmBasis3DCyl::EvalRZ(harm_poly_arr &B, double *val)
{
//Fills the linear array val[Len] with values of basis polynomials.
//Private function, intended for internal use only.
//
   unsigned M;
   double   V;
   rz_harm_poly *P;
   for (unsigned L = 1, k = 0; L <= Dim; ++L, ++k) {
      (*val) = B[k][0].Eval(); ++val;
      for (M = 1; M <= L; ++M) {
         P = &(B[k][M]);
         V = P->Eval();
         (*val) = V*P->GetCos(); ++val;
         (*val) = V*P->GetSin(); ++val;
      }
   }
}

//_______________________________________________________________________________
void HarmBasis3DCyl::EvalBphi()
{
//Fills the array Bphi_k[Len] with values of phi-basis polynomials.
//
   unsigned M;
   double   V;
   double  *val = Bphi_k;
   rz_harm_poly *P;
   for (unsigned L = 1, k = 0; L <= Dim; ++L, ++k) {
      (*val) = 0.; ++val;
      for (M = 1; M <= L; ++M) {
         P = &(BphiB[k][M]);
         V = P->Eval();
         (*val) = -V*P->GetSin(); ++val;
         (*val) =  V*P->GetCos(); ++val;
      }
   }
}

//_______________________________________________________________________________
double HarmBasis3DCyl::GetVal(double *coeff, double *basis)
{
//return value of the expansion with coefficients coeff[Len] for the basis
//Private function, intended for internal use only.
//
   double S = 0.;
   for (unsigned k = 0; k < Len; ++k) S += coeff[k]*basis[k];
   return S;
}

//_______________________________________________________________________________
void HarmBasis3DCyl::Print(harm_poly_arr &B, std::ostream &out)
{
   unsigned jL, jM, wdt = 60;
   char fc1 = '-', fc0 = out.fill(fc1);
   for (jL = 0; jL < B.size(); ++jL) {
      out << std::setw(wdt) << fc1 << std::endl;
      out << "Basis subset " << jL+1 << std::endl;
      out << std::setw(wdt) << fc1 << std::endl;
      for (jM = 0; jM < B[jL].size(); ++jM) {
         B[jL][jM].Print(out);
      }
   }
   out.fill(fc0);
}

//_______________________________________________________________________________
void HarmBasis3DCyl::Print(std::ostream &out)
{
   out << "BASIS POLYNOMIALS FOR THE POTENTIAL:\n" << std::endl;
   PrintPtB(out);
   out << "\nBASIS POLYNOMIALS FOR R-COMPONENT   OF THE FIELD:\n" << std::endl;
   PrintBrB(out);
   out << "\nBASIS POLYNOMIALS FOR Z-COMPONENT   OF THE FIELD:\n" << std::endl;
   PrintBzB(out);
   out << "\nBASIS POLYNOMIALS FOR PHI-COMPONENT OF THE FIELD:\n" << std::endl;
   PrintBphiB(out);
}

