//
//  VVIObjF.h (v2.0)
//
//  Vavilov density and distribution functions
//
//
// Created by Morris Swartz on 1/14/10.
// 2010 __TheJohnsHopkinsUniversity__.
//
// V1.1 - make dzero call both fcns with a switch
// V1.2 - remove inappriate initializers and add methods to return non-zero/normalized region
// V2.0 - restructuring and speed improvements by V. Innocente
//
 
#ifndef VVIObjF_h
#define VVIObjF_h 1

// ***********************************************************************************************************************
//! \class VVIObjF 
//!  
//!  Port of CERNLIB routines vvidis/vviden (G116) to calculate higher quality Vavilov density and distribution functions
//! 
// ***********************************************************************************************************************
class VVIObjF {

 public:
  
  VVIObjF(float kappa = 0.01, float beta2 = 1., int mode = 0); //!< Constructor  
  
  float fcn(float x) const; //! density (mode=0) or distribution (mode=1) function
  void limits(float& xl, float& xu) const; //! returns the limits on the non-zero (mode=0) or normalized region (mode=1)
  
private:
  
  // Vavilov distribution parameters (inputs and common block /G116C1/)
  
  const int mode_;          //!< set to 0 to calculate the density function and to 1 to calculate the distribution function
  float t0_;         
  float t1_;         
  float t_;          
  float omega_;
  float x0_;        
  float a_[155];    
  float b_[155];     
};


#endif
