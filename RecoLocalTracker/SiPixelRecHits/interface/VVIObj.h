//
//  VVIObj.h (v1.2)
//
//  Vavilov density and distribution functions
//
//
// Created by Morris Swartz on 1/14/10.
// Copyright 2010 __TheJohnsHopkinsUniversity__. All rights reserved.
//
// V1.1 - make dzero call both fcns with a switch
// V1.2 - remove inappriate initializers and add methods to return non-zero/normalized region
//
 
#ifndef VVIObj_h
#define VVIObj_h 1

// ***********************************************************************************************************************
//! \class VVIObj 
//!  
//!  Port of CERNLIB routines vvidis/vviden (G116) to calculate higher quality Vavilov density and distribution functions
//! 
// ***********************************************************************************************************************
class VVIObj {

 public:
  
	 VVIObj(double kappa = 0.01, double beta2 = 1., double mode = 0); //!< Constructor  
  
  	 double fcn(double x); //! density (mode=0) or distribution (mode=1) function
	 void limits(double& xl, double& xu); //! returns the limits on the non-zero (mode=0) or normalized region (mode=1)
  
 private:
  
// Vavilov distribution parameters (inputs and common black /G116C1/)
  
    mutable double kappa_;      //!< Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10. (Gaussian-like)]
    mutable double beta2_;      //!< Vavilov beta2 parameter (speed of particle in v/c units)
	 const int mode_;          //!< set to 0 to calculate the density function and to 1 to calculate the distribution function
    mutable double h_[7];     //!< these reproduce the auxilliary common block /G116C1/
    mutable double t0_;         
    mutable double t1_;         
    mutable double t_;          
    mutable double omega_;      
    mutable double a_[155];    
    mutable double b_[155];     
    mutable double x0_;        
	
	 double f1(double x);    //! Private function f1 called from constructor
	 double f2(double x);    //! Private function f2 called from constructor
	
	 double cosint(double x);    //! Private version of the cosine integral
	 double sinint(double x);    //! Private version of the sine integral
	 double expint(double x);    //! Private version of the exponential integral
	 int dzero(double a, double b, double& x0, double& rv, double eps, int mxf, int fsel);    //! Private version of the CERNLIB root finder
	
} ;


#endif
