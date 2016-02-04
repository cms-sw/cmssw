#ifndef rz_poly_h
#define rz_poly_h

/** \class magfieldparam::rz_poly
 *
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author V. Maroussov
 */

#include <vector>

namespace magfieldparam {

//_______________________________________________________________________________
struct poly_term {
   double coeff;
   int    np[2];
};

//_______________________________________________________________________________
typedef std::vector<poly_term> poly_vect;
//_______________________________________________________________________________
typedef std::vector<poly_vect> poly_arr;

//_______________________________________________________________________________
class rz_poly {   // represent a set of homogeneous polynomials

private:

   poly_arr data;
   int max_nr, max_nz, n_active;
   double *r_pow;
   double *z_pow;
   bool   *is_off;

public:

   rz_poly() : data(), max_nr(0), max_nz(0), n_active(0), 
               r_pow(0), z_pow(0), is_off(0) {};

   rz_poly(int N);
   rz_poly(const rz_poly& S);
   ~rz_poly();
   
   void SetOFF  (int npoly);
   void SetON   (int npoly);
   void SetAllON(int npoly) {if (is_off) std::fill(is_off, is_off+data.size(), false);}
   
   rz_poly Diff(int nvar, bool keep_empty = false);
   rz_poly Int (int nvar);
   
   rz_poly& operator*=(double  C);
   rz_poly& operator*=(double *C);
   
   double  GetSVal(double r, double z, double *C);
   double *GetVVal(double r, double z, double *rez_out = 0);
   
   int GetMaxRPow() {return max_nr-1;}
   int GetMaxZPow() {return max_nz-1;}
   int GetLength()  {return (int)data.size();}
   int GetNActive() {return n_active;}
   
   double *Expand(double *C);
   
   void Print();

};
}

#endif
