/***********************************************************************/
/*                                                                     */
/*              Finding MT2W                                           */
/*              Reference:  arXiv:1203.4813 [hep-ph]                   */
/*              Authors: Yang Bai, Hsin-Chia Cheng,                    */
/*                       Jason Gallicchio, Jiayin Gu                   */
/*              Based on MT2 by: Hsin-Chia Cheng, Zhenyu Han           */ 
/*              May 8, 2012, v1.00a                                    */
/*                                                                     */  
/***********************************************************************/
#ifndef PhysicsTools_Heppy_mt2w_bisect_H
#define PhysicsTools_Heppy_mt2w_bisect_H

namespace heppy {

namespace mt2w_bisect
{
class mt2w
{  
 public:
  static const float RELATIVE_PRECISION; 
  static const float ABSOLUTE_PRECISION;
  static const float MIN_MASS;
  static const float ZERO_MASS;
  static const float SCANSTEP;
      
  mt2w(double upper_bound=500.0, double error_value=499.0, double scan_step=0.5);
  // Constructor where:
  //    upper_bound:  the upper bound of search for MT2W, default value is 500 GeV 
  //    error_value:  if we couldn't find any compatible region below upper_bound, this value gets returned.  
  //                  -1.0 is a reasonable to indicate error, but upper_bound-1.0 allows a simple greater-than cut for signal
  //    scan_step:    if we need to scan to find the compatible region, this is the step of the scan
  void   set_momenta(double *pl0, double *pb10, double *pb20, double* pmiss0);  //b1 pairs with l
  void   set_momenta(double El,  double plx,  double ply,  double plz,
		     double Eb1, double pb1x, double pb1y, double pb1z,
		     double Eb2, double pb2x, double pb2y, double pb2z,
		     double pmissx, double pmissy);  // Same as above without pointers/arrays
  // Where the input 4-vector information represents:
  //    l is the visible lepton
  //    b1 is the bottom on the same side as the visible lepton
  //    b2 is the other bottom (paired with the invisible W)
  //    pmiss is missing momentum with only x and y components.
  double get_mt2w();  // Calculates result, which is cached until set_momenta is called.
  //  void   print();
      
 protected:
  void   mt2w_bisect();  // The real work is done here.
      
 private:  

  bool   solved;
  bool   momenta_set;
  double upper_bound;
  double error_value;
  double scan_step;
  double mt2w_b;

  int    teco(double mtop);   // test the compatibility of a given trial top mass mtop
  inline int    signchange_n( long double t1, long double t2, long double t3, long double t4, long double t5);
  inline int    signchange_p( long double t1, long double t2, long double t3, long double t4, long double t5);

  //data members
  double plx, ply, plz, ml, El;      // l is the visible lepton
  double pb1x, pb1y, pb1z, mb1, Eb1;   // b1 is the bottom on the same side as the visible lepton
  double pb2x, pb2y, pb2z, mb2, Eb2;   // b2 is the other bottom
  double pmissx, pmissy;              // x and y component of missing p_T
  double mv,mw;           //mass of neutrino and W-boson
     
  //auxiliary definitions
  double mlsq, Elsq;
  double mb1sq, Eb1sq;
  double mb2sq, Eb2sq;

  //auxiliary coefficients
  double a1, b1, c1, a2, b2, c2, d1, e1, f1, d2, e2, f2;
  double d2o, e2o, f2o;

  double precision;
};

}//end namespace mt2w_bisect
}//end namespace heppy

#endif
