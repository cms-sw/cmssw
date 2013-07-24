#ifndef BFit_h
#define BFit_h

/** \class magfieldparam::BFit
 *
 *  2D parametrization of MTCC data
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author V. Maroussov
 */

#include "rz_poly.h"



//_______________________________________________________________________________
namespace magfieldparam {
class BFit {

private:

#ifdef BFit_PW
   static const double Z_nom[4];
   static const double B_nom[4];
   static const double C_nom[4][16];
#else
   static const double dZ_0;
   static const double dZ_2;

   static const double C_0[16];   //4-fold expansion coefficients
   static const double C_2[16];   //of the expansion coefficients :)
   static const double C_4[16];
#endif   
   double dZ;    //Z-shift
   double C[16]; //the expansion coeeficients 

   rz_poly *Bz_base;
   rz_poly *Br_base;
 
public:

   BFit();
   ~BFit() {delete Bz_base; delete Br_base;};
   
   void SetField(double B);
   void GetField(double r,   double z,   double phi,
                 double &Br, double &Bz, double &Bphi);
};
}

#endif
