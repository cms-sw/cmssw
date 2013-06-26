//
//  VVIObj.h (v2.0)
//
//  Vavilov density and distribution functions
//
//
// Created by Morris Swartz on 1/14/10.
// Copyright 2010 __TheJohnsHopkinsUniversity__. All rights reserved.
//
// V1.1 - make dzero call both fcns with a switch
// V1.2 - remove inappriate initializers and add methods to return non-zero/normalized region
// V2.0 - restructuring and speed improvements by V. Innocente
//
 
#ifndef VVIObj_h
#define VVIObj_h 1

namespace sistripvvi {
// ***********************************************************************************************************************
//! \class VVIObj 
//!  
//!  Port of CERNLIB routines vvidis/vviden (G116) to calculate higher quality Vavilov density and distribution functions
//! 
// ***********************************************************************************************************************
class VVIObj {

 public:
  
  VVIObj(double kappa = 0.01, double beta2 = 1., int mode = 0); //!< Constructor  
  
  double fcn(double x) const; //! density (mode=0) or distribution (mode=1) function
  void limits(double& xl, double& xu) const; //! returns the limits on the non-zero (mode=0) or normalized region (mode=1)
  
private:
  
  // Vavilov distribution parameters (inputs and common block /G116C1/)
  
  const int mode_;          //!< set to 0 to calculate the density function and to 1 to calculate the distribution function
  double t0_;         
  double t1_;         
  double t_;          
  double omega_;
  double x0_;        
  double a_[155];    
  double b_[155];     
};
}

#endif
