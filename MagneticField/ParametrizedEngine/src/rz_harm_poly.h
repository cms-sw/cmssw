#ifndef rz_harm_poly_h
#define rz_harm_poly_h

#include <cmath>
#include "poly2d_base.h"

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Pair (Cos(phi),Sin(Phi)). Intended for internal use by rz_harm_poly.       //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
namespace magfieldparam {
struct trig_pair {
   double CosPhi;
   double SinPhi;

   trig_pair() : CosPhi(1.), SinPhi(0.) {}
   trig_pair(const trig_pair &tp) : CosPhi(tp.CosPhi), SinPhi(tp.SinPhi) {}
   trig_pair(const double C, const double S) : CosPhi(C), SinPhi(S) {}
   trig_pair(const double phi) : CosPhi(cos(phi)), SinPhi(sin(phi)) {}
   
   //Return trig_pair fo angle increased by angle of tp.
   trig_pair Add(const trig_pair &tp) {
      return trig_pair(this->CosPhi*tp.CosPhi - this->SinPhi*tp.SinPhi,
                       this->SinPhi*tp.CosPhi + this->CosPhi*tp.SinPhi);
   }
};

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Harmonic homogeneous polynomial in cylindrical system.                     //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
class rz_harm_poly : public poly2d_base {

private:
   unsigned L;
   int      M;
   
   static unsigned   Cnt;      //Number of the "rz_harm_poly" objects
   static double     phival;   //Last phi value used
   static bool       phi_set;  //TRUE if phi value is set
   static unsigned   MaxM;     //Max. M among "rz_harm_poly" objects

   static unsigned   TASize;   //TrigArr size
   static trig_pair *TrigArr;  //Array with angular data

   static void SetTrigArrSize(const unsigned N);
   static void FillTrigArr   (const double phi);
   
   void PrintLM(std::ostream &out = std::cout)
   {
      out <<  "L=" << std::setw(3)  << std::left << L
          <<", M=" << std::setw(3)  << std::left << M << "; ";
   }

public:

   static int      GetMaxM(); //return Max. M for the class
   static unsigned ParentCount() { return poly2d_base::Count();}
   static unsigned Count() { return Cnt;}
   static void     SetPhi(const double phi);
   static void     SetPoint(const double r, const double z, const double phi)
   {
      poly2d_base::SetPoint(r, z); SetPhi(phi);
   }
   
   rz_harm_poly() : poly2d_base(), L(0), M(0) {++Cnt;} 
   rz_harm_poly(const poly2d_base &S) : poly2d_base(S), L(0), M(0) {++Cnt;}
   rz_harm_poly(const rz_harm_poly &S) : poly2d_base(S), L(S.L), M(S.M) {++Cnt;}
   rz_harm_poly(const unsigned N);
   ~rz_harm_poly() override;
   
   bool IsPhiSet() { return phi_set;}
   
   rz_harm_poly GetDiff  (int nvar) { rz_harm_poly R(*this); R.Diff  (nvar); return R;}
   rz_harm_poly GetInt   (int nvar) { rz_harm_poly R(*this); R.Int   (nvar); return R;}
   rz_harm_poly GetIncPow(int nvar) { rz_harm_poly R(*this); R.IncPow(nvar); return R;}
   rz_harm_poly GetDecPow(int nvar) { rz_harm_poly R(*this); R.DecPow(nvar); return R;}

   rz_harm_poly LadderUp();
   rz_harm_poly LadderDwn();
   
   unsigned GetL() { return L;}
   int      GetM() { return M;}
   
   //Next functions return value of angular terms. 
   //No check is made, wheither the TrigArr is initialized.
   //User can check if IsPhiSet() == true
   double GetCos() { return TrigArr[M].CosPhi;}
   double GetSin() { return TrigArr[M].SinPhi;}
   
   void CheatL(const unsigned newL) { L = newL;}
   void Print(std::ostream &out = std::cout, const std::streamsize prec = 5)
   { PrintLM(out); poly2d_base::Print(out, prec);}
 

   static void PrintTrigArr(std::ostream &out = std::cout, const std::streamsize prec = 5);

}; //class rz_harm_poly
}

#endif
