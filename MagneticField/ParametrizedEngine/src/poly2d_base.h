#ifndef poly2d_base_h
#define poly2d_base_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <set>
#include <cstring>

#include <cmath>
#include <cfloat> //in order to use DBL_EPSILON (1+DBL_EPSILON > 1)

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  The "poly2d_term" represent a term of a polynomial of 2 variables.         //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
namespace magfieldparam {

struct poly2d_term {
   double   coeff;  //Coefficient of the term
   unsigned np[2];  //Powers of variables

   poly2d_term() {memset(this, 0, sizeof(*this));}
   poly2d_term(double C, unsigned nr, unsigned nz)
   {
      coeff = C; np[0] = nr; np[1] = nz;
   }
   void Print(std::ostream &out = std::cout, bool first_term = true);
};

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Base class that represent a polynomial of 2 variables. It isn't supposed   //
//  to be used directly and provides no way of setting coefficients directly.  //
//  Such methods must be defined in derived classes.                           //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________________
class poly2d_base {   // a general polynomial of 2 variables

protected:
   //Group of static members for the class memory management
   //----------------------------------------------------------------------------
   static double     rval;   //last r-value used in calculation
   static double     zval;   //last z-value used in calculation

   static double   **rz_pow; //table with calculated r^n*z^m values
   static unsigned   NTab;   //rz_pow table size
   static unsigned   NPwr;   //max power in use by CLASS
   static bool       rz_set;

   static const double MIN_COEFF; //Threshold for assigning a coeff. to 0

   static std::set<poly2d_base*> poly2d_base_set;  //Set of all poly2d_base objects
//   static std::set<poly2d_base*, less<poly2d_base*> > poly2d_base_set;  //Set of all poly2d_base objects

   static void SetTabSize(const unsigned N); //Set rz-table size
   static void FillTable (const double r, const double z);

   static void AdjustTab();
   //----------------------------------------------------------------------------

   std::vector<poly2d_term> data; //polynomial terms
   unsigned max_pwr;         //max power in use by INSTANCE
   
public:
   static void     IncNPwr(const unsigned N) {if (N > NPwr) NPwr = N;}
   static int      GetMaxPow();
   static unsigned Count() { return poly2d_base_set.size();}
   static void     PrintTab(std::ostream &out = std::cout, const std::streamsize prec = 5);

   static void SetPoint(const double r, const double z);

   poly2d_base() {
      max_pwr = 0;
      poly2d_base_set.insert(this);
   }
   poly2d_base(const poly2d_base &S) {
      data    = S.data;
      max_pwr = S.max_pwr;
      poly2d_base_set.insert(this);
   }

   virtual ~poly2d_base();

   bool IsOn()    { return bool(!data.empty());}
   bool IsRZSet() { return rz_set;}

   void Collect(); //Collect terms and remove zero terms
   void Compress() { Collect();}
   
   void Diff  (int nvar); //differentiate the polynomial by variable# nvar
   void Int   (int nvar); //Integrate the polynomial by variable# nvar
   void IncPow(int nvar); //Multiply the polynomial by variable# nvar
   void DecPow(int nvar); //Divide the polynomial by variable# nvar

   void Scale(const double C);
//   poly2d_base& operator*=(const double C) { Scale(C); return *this;}

   double Eval(); //Evaluation with no check that rz_pow table exist
   double GetVal() {if (rz_set) return Eval(); else return 0.;}
   double GetVal(const double r, const double z) { SetPoint(r,z); return Eval();}
   
   void Print(std::ostream &out = std::cout, const std::streamsize prec = 5);

}; //Class poly2d_base
}

#endif
