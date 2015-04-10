#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "TMath.h"
#include "TString.h"
#include "TF1.h"
#include "TRandom.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/SigmaPtDiff.h"

/**
 * Used to define parameters inside the functions.
 */
struct ParameterSet
{
  ParameterSet() {}
  ParameterSet(const TString & inputName, const double & inputStep, const double & inputMini, const double & inputMaxi) :
    step(inputStep),
    mini(inputMini),
    maxi(inputMaxi)
  {
    std::cout << "setting name = " << inputName << std::endl;
    name = inputName;
  }
  TString name;
  double step, mini, maxi;
};

// ----------------------- //
// Bias and scale functors //
// ----------------------- //
/** The correct functor is selected at job start in the constructor.
 * The pt value is taken by reference and modified internally.
 * eta, phi and chg are taken by const reference.<br>
 * Made into a template so that it can be used with arrays too
 * (parval for the scale fit is an array, because Lykelihood is an
 * extern C function, because TMinuit asks it).<br>
 * Note that in the array case it takes the pointer by const reference,
 * thus the elements of the array are modifiable.
 */
template <class T>
class scaleFunctionBase {
 public:
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const = 0;
  virtual ~scaleFunctionBase() = 0;
  /// This method is used to reset the scale parameters to neutral values (useful for iterations > 0)
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    std::cout << "ERROR: the resetParameters method must be defined in each scale function" << std::endl;
    std::cout << "Please add it to the scaleFunction you are using" << std::endl;
    exit(1);
  }
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) = 0;
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    std::cout << "The method setParameters must be implemented for this scale function" << std::endl;
    exit(1);
  }
  virtual int parNum() const { return parNum_; }
 protected:
  int parNum_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                      TString* parname, const T & parScale, const std::vector<int> & parScaleOrder,
                      double* thisStep, double* thisMini, double* thisMaxi, TString* thisParName ) {
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = parScale[iPar];
      Step[iPar] = thisStep[iPar];
      Mini[iPar] = thisMini[iPar];
      Maxi[iPar] = thisMaxi[iPar];
      ind[iPar] = parScaleOrder[iPar];
      parname[iPar] = thisParName[iPar];
    }
  }
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
         TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const std::vector<ParameterSet> & parSet ) {
    if( int(parSet.size()) != this->parNum_ ) {
      std::cout << "Error: wrong number of parameter initializations = " << parSet.size() << ". Number of parameters is " << this->parNum_ << std::endl;
      exit(1);
    }
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = parResol[iPar];
      Step[iPar] = parSet[iPar].step;
      Mini[iPar] = parSet[iPar].mini;
      Maxi[iPar] = parSet[iPar].maxi;
      ind[iPar] = parResolOrder[iPar];
      parname[iPar] = parSet[iPar].name;
    }
  }
};

// SLIMMED VERSION
// The full set of scaleFunction developed in the past can be found at 
// https://github.com/scasasso/cmssw/blob/test_binned_function/MuonAnalysis/MomentumScaleCalibration/interface/Functions.h#L100-L5177

template <class T> inline scaleFunctionBase<T>::~scaleFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.

// No scale
// --------
template <class T>
class scaleFunctionType0 : public scaleFunctionBase<T> {
public:
  scaleFunctionType0() {
    // One of the two is required. This follows from when templates are used by the compiler and the names lookup rules in c++.
    // scaleFunctionBase<T>::parNum_ = 0;
    this->parNum_ = 0;
  }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const { return pt; }
  virtual void resetParameters(std::vector<double> * scaleVec) const {}
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {}
};

//
// Curvature: (linear eta + sinusoidal in phi (both in 5 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType50 : public scaleFunctionBase<T> {
public:
  scaleFunctionType50() { this->parNum_ = 27; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), twist(0), ampl2(0), freq2(0), phase2(0);

// very bwd bin
    if ( eta  < parScale[4] ) {
      ampl = parScale[1]; phase = parScale[2]; ampl2 = parScale[21]; freq2 = parScale[22]; phase2 = parScale[23];
      twist = parScale[3]*(eta-parScale[4])+parScale[7]*(parScale[4]-parScale[8])+parScale[11]*parScale[8]; 
// bwd bin
    } else if ( parScale[4] <= eta && eta < parScale[8] ) {
      ampl = parScale[5]; phase = parScale[6];
      twist = parScale[7]*(eta-parScale[8])+parScale[11]*parScale[8] ; 
// barrel bin
    } else if ( parScale[8] <= eta && eta < parScale[12] ) {
      ampl = parScale[9]; phase = parScale[10];
      twist = parScale[11]*eta; 
// fwd bin
    } else if ( parScale[12] <= eta && eta < parScale[16] ) {
      ampl = parScale[13]; phase = parScale[14];
      twist = parScale[15]*(eta-parScale[12])+parScale[11]*parScale[12]; 
// very fwd bin
    } else if ( parScale[16] < eta ) {
      ampl = parScale[17]; phase = parScale[18]; ampl2 = parScale[24]; freq2 = parScale[25]; phase2 = parScale[26];
      twist = parScale[19]*(eta-parScale[16])+parScale[15]*(parScale[16]-parScale[12])+parScale[11]*parScale[12]; 
    }

// apply the correction
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -ampl2*sin((int)freq2*phi+phase2)
				    -0.5*parScale[20]);
    return 1./((double)chg*curv);
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    //    scaleVec->push_back(1);
    for( int i=0; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {

    double thisStep[] = {0.000001, 
			 0.000001, 0.01, 0.000001, 0.01,  
			 0.000001, 0.01, 0.000001, 0.01, 
			 0.000001, 0.01, 0.000001, 0.01, 
			 0.000001, 0.01, 0.000001, 0.01, 
			 0.000001, 0.01, 0.000001,
			 0.000001,
			 0.000001, 0.01, 0.01,
			 0.000001, 0.01, 0.01}; 

    TString thisParName[] = { "Curv global scale"    	
			      , "Phi ampl eta vbwd"  , "Phi phase eta vbwd" , "Twist eta vbwd",   "vbwd/bwd bndry"
			      , "Phi ampl eta  bwd"  , "Phi phase eta  bwd" , "Twist eta  bwd",   "bwd/bar bndry"   
			      , "Phi ampl eta  bar"  , "Phi phase eta  bar" , "Twist eta  bar",   "bar/fwd bndry"  
			      , "Phi ampl eta  fwd"  , "Phi phase eta  fwd" , "Twist eta  fwd",   "fwd/vfwd bndry"  
			      , "Phi ampl eta vfwd"  , "Phi phase eta vfwd" , "Twist eta vfwd"
			      , "Charge depend bias"
			      , "Phi ampl eta vbwd (2nd harmon.)", "Phi freq. eta vbwd (2nd harmon.)", "Phi phase eta vbwd (2nd harmon.)"
			      , "Phi ampl eta vfwd (2nd harmon.)", "Phi freq. eta vfwd (2nd harmon.)", "Phi phase eta vfwd (2nd harmon.)"};

    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1, -0.3, 1., -3.1416, -0.3, 1., -3.1416};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3,  0.1,  0.3, 5.,  3.1416,  0.3, 5.,  3.1416};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1, -0.3, 1., -3.1416, -0.3, 1., -3.1416};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3,  0.1,  0.3, 5.,  3.1416,  0.3, 5.,  3.1416};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parScale, const std::vector<int> & parScaleOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    if( (int(parStep.size()) != this->parNum_) || (int(parMin.size()) != this->parNum_) || (int(parMax.size()) != this->parNum_) ) {
      std::cout << "Error: par step or min or max do not match with number of parameters" << std::endl;
      std::cout << "parNum = " << this->parNum_ << std::endl;
      std::cout << "parStep.size() = " << parStep.size() << std::endl;
      std::cout << "parMin.size() = " << parMin.size() << std::endl;
      std::cout << "parMax.size() = " << parMax.size() << std::endl;
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Curv global scale",   parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Phi ampl  vbwd",      parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Phi phase vbwd",      parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Twist vbwd",          parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "vbwd/bwd bndry",      parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Phi ampl   bwd",      parStep[5], parMin[5], parMax[5] );
    parSet[6]  = ParameterSet( "Phi phase  bwd",      parStep[6], parMin[6], parMax[6] );
    parSet[7]  = ParameterSet( "Twist  bwd",          parStep[7], parMin[7], parMax[7] );
    parSet[8]  = ParameterSet( "bwd/bar bndry",       parStep[8], parMin[8], parMax[8] );
    parSet[9]  = ParameterSet( "Phi ampl   bar",      parStep[9], parMin[9], parMax[9] );
    parSet[10] = ParameterSet( "Phi phase  bar",      parStep[10],parMin[10],parMax[10] );
    parSet[11] = ParameterSet( "Twist  bar",          parStep[11],parMin[11],parMax[11] );
    parSet[12] = ParameterSet( "bar/fwd bndry",       parStep[12],parMin[12],parMax[12] );
    parSet[13] = ParameterSet( "Phi ampl   fwd",      parStep[13],parMin[13],parMax[13] );
    parSet[14] = ParameterSet( "Phi phase  fwd",      parStep[14],parMin[14],parMax[14] );
    parSet[15] = ParameterSet( "Twist  fwd",          parStep[15],parMin[15],parMax[15] );
    parSet[16] = ParameterSet( "fwd/vfwd bndry",      parStep[16],parMin[16],parMax[16] );
    parSet[17] = ParameterSet( "Phi ampl  vfwd",      parStep[17],parMin[17],parMax[17] );
    parSet[18] = ParameterSet( "Phi phase vfwd",      parStep[18],parMin[18],parMax[18] );
    parSet[19] = ParameterSet( "Twist vfwd",          parStep[19],parMin[19],parMax[19] );
    parSet[20] = ParameterSet( "Charge depend bias",  parStep[20],parMin[20],parMax[20] );
    parSet[21] = ParameterSet( "Phi ampl vbwd (2nd)", parStep[21],parMin[21],parMax[21] );
    parSet[22] = ParameterSet( "Phi freq vbwd (2nd)", parStep[22],parMin[22],parMax[22] );
    parSet[23] = ParameterSet( "Phi phase vbwd (2nd)",parStep[23],parMin[23],parMax[23] );
    parSet[24] = ParameterSet( "Phi ampl vfwd (2nd)", parStep[24],parMin[24],parMax[24] );
    parSet[25] = ParameterSet( "Phi freq vfwd (2nd)", parStep[25],parMin[25],parMax[25] );
    parSet[26] = ParameterSet( "Phi phase vfwd (2nd)",parStep[26],parMin[26],parMax[26] );


    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, parSet );
  }

};

//
// Curvature: binned function w/ constraints on eta rings
// good results on Legacy 2011 MC
// phi bins = [8,8,8,8,8,8,8] in 7 eta bins
// ------------------------------------------------------------
template <class T>
class scaleFunctionType64 : public scaleFunctionBase<T> {
public:
  scaleFunctionType64() { 
    this->parNum_ = 50; 
  }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double deltaK(0);
    double p0 = parScale[0];
            
    if ( eta<-2.1 && eta>-2.4 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[1];
    else if ( eta<-2.1 && eta>-2.4 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[2];
    else if ( eta<-2.1 && eta>-2.4 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[3];
    else if ( eta<-2.1 && eta>-2.4 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[4];
    else if ( eta<-2.1 && eta>-2.4 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[5];
    else if ( eta<-2.1 && eta>-2.4 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[6];
    else if ( eta<-2.1 && eta>-2.4 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[7];
    else if ( eta<-2.1 && eta>-2.4 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[1]+parScale[2]+parScale[3]+parScale[4]+parScale[5]+parScale[6]+parScale[7]);

    else if ( eta<-1.5 && eta>-2.1 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[8];
    else if ( eta<-1.5 && eta>-2.1 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[9];
    else if ( eta<-1.5 && eta>-2.1 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[10];
    else if ( eta<-1.5 && eta>-2.1 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[11];
    else if ( eta<-1.5 && eta>-2.1 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[12];
    else if ( eta<-1.5 && eta>-2.1 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[13];
    else if ( eta<-1.5 && eta>-2.1 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[14];
    else if ( eta<-1.5 && eta>-2.1 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[8]+parScale[9]+parScale[10]+parScale[11]+parScale[12]+parScale[13]+parScale[14]);

    else if ( eta<-0.9 && eta>-1.5 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[15];
    else if ( eta<-0.9 && eta>-1.5 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[16];
    else if ( eta<-0.9 && eta>-1.5 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[17];
    else if ( eta<-0.9 && eta>-1.5 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[18];
    else if ( eta<-0.9 && eta>-1.5 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[19];
    else if ( eta<-0.9 && eta>-1.5 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[20];
    else if ( eta<-0.9 && eta>-1.5 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[21];
    else if ( eta<-0.9 && eta>-1.5 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[15]+parScale[16]+parScale[17]+parScale[18]+parScale[19]+parScale[20]+parScale[21]);

    else if ( eta<0.9 && eta>-0.9 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[22];
    else if ( eta<0.9 && eta>-0.9 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[23];
    else if ( eta<0.9 && eta>-0.9 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[24];
    else if ( eta<0.9 && eta>-0.9 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[25];
    else if ( eta<0.9 && eta>-0.9 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[26];
    else if ( eta<0.9 && eta>-0.9 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[27];
    else if ( eta<0.9 && eta>-0.9 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[28];
    else if ( eta<0.9 && eta>-0.9 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[22]+parScale[23]+parScale[24]+parScale[25]+parScale[26]+parScale[27]+parScale[28]);

    else if ( eta<1.5 && eta>0.9 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[29];
    else if ( eta<1.5 && eta>0.9 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[30];
    else if ( eta<1.5 && eta>0.9 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[31];
    else if ( eta<1.5 && eta>0.9 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[32];
    else if ( eta<1.5 && eta>0.9 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[33];
    else if ( eta<1.5 && eta>0.9 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[34];
    else if ( eta<1.5 && eta>0.9 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[35];
    else if ( eta<1.5 && eta>0.9 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[29]+parScale[30]+parScale[31]+parScale[32]+parScale[33]+parScale[34]+parScale[35]);

    else if ( eta<2.1 && eta>1.5 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[36];
    else if ( eta<2.1 && eta>1.5 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[37];
    else if ( eta<2.1 && eta>1.5 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[38];
    else if ( eta<2.1 && eta>1.5 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[39];
    else if ( eta<2.1 && eta>1.5 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[40];
    else if ( eta<2.1 && eta>1.5 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[41];
    else if ( eta<2.1 && eta>1.5 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[42];
    else if ( eta<2.1 && eta>1.5 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[36]+parScale[37]+parScale[38]+parScale[39]+parScale[40]+parScale[41]+parScale[42]);

    else if ( eta<2.4 && eta>2.1 && phi<-2.35619449019 && phi>-3.14159265359 ) deltaK = parScale[43];
    else if ( eta<2.4 && eta>2.1 && phi<-1.57079632679 && phi>-2.35619449019 ) deltaK = parScale[44];
    else if ( eta<2.4 && eta>2.1 && phi<-0.785398163397 && phi>-1.57079632679 ) deltaK = parScale[45];
    else if ( eta<2.4 && eta>2.1 && phi<0.0 && phi>-0.785398163397 ) deltaK = parScale[46];
    else if ( eta<2.4 && eta>2.1 && phi<0.785398163397 && phi>0.0 ) deltaK = parScale[47];
    else if ( eta<2.4 && eta>2.1 && phi<1.57079632679 && phi>0.785398163397 ) deltaK = parScale[48];
    else if ( eta<2.4 && eta>2.1 && phi<2.35619449019 && phi>1.57079632679 ) deltaK = parScale[49];
    else if ( eta<2.4 && eta>2.1 && phi<3.14159265359 && phi>2.35619449019 ) deltaK = -(parScale[43]+parScale[44]+parScale[45]+parScale[46]+parScale[47]+parScale[48]+parScale[49]);
    else {
      std::cout << "This should really not happen, this muon has eta = " << eta << "and phi = " << phi << std::endl;
      exit(1);
    }

    // apply the correction
    double curv = (double)chg/pt;
    return 1./((double)chg*(1+p0)*(curv+deltaK));
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    //    scaleVec->push_back(1);
    for( int i=0; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {

    double thisStep[] = {
      0.000001, 
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
    }; 

    TString thisParName[] = { 
      "Curv global scale", 
      "deltaK bin1", "deltaK bin2", "deltaK bin3", "deltaK bin4", "deltaK bin5", "deltaK bin6", "deltaK bin7", "deltaK bin8", "deltaK bin9", "deltaK bin10", "deltaK bin11", "deltaK bin12", "deltaK bin13", "deltaK bin14", "deltaK bin15", "deltaK bin16", "deltaK bin17", "deltaK bin18", "deltaK bin19", "deltaK bin20", "deltaK bin21", "deltaK bin22", "deltaK bin23", "deltaK bin24", "deltaK bin25", "deltaK bin26", "deltaK bin27", "deltaK bin28", "deltaK bin29", "deltaK bin30", "deltaK bin31", "deltaK bin32", "deltaK bin33", "deltaK bin34", "deltaK bin35", "deltaK bin36", "deltaK bin37", "deltaK bin38", "deltaK bin39", "deltaK bin40", "deltaK bin41", "deltaK bin42", "deltaK bin43", "deltaK bin44", "deltaK bin45", "deltaK bin46", "deltaK bin47", "deltaK bin48", "deltaK bin49"
    };

    if( muonType == 1 ) {
      double thisMini[] = {
      -0.1,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
      -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005
      };
      double thisMaxi[] = {
	0.1,
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
      };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {
	-0.1,
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, 
	-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005
      };
      double thisMaxi[] = {
	0.1,
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 
	0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
      };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parScale, const std::vector<int> & parScaleOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    if( (int(parStep.size()) != this->parNum_) || (int(parMin.size()) != this->parNum_) || (int(parMax.size()) != this->parNum_) ) {
      std::cout << "Error: par step or min or max do not match with number of parameters" << std::endl;
      std::cout << "parNum = " << this->parNum_ << std::endl;
      std::cout << "parStep.size() = " << parStep.size() << std::endl;
      std::cout << "parMin.size() = " << parMin.size() << std::endl;
      std::cout << "parMax.size() = " << parMax.size() << std::endl;
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]   = ParameterSet( "Curv global scale",   parStep[0], parMin[0], parMax[0] );
    parSet[1]   = ParameterSet( "deltaK bin1",    parStep[1],  parMin[1],  parMax[1] );
    parSet[2]   = ParameterSet( "deltaK bin2",    parStep[2],  parMin[2],  parMax[2] );
    parSet[3]   = ParameterSet( "deltaK bin3",    parStep[3],  parMin[3],  parMax[3] );
    parSet[4]   = ParameterSet( "deltaK bin4",    parStep[4],  parMin[4],  parMax[4] );
    parSet[5]   = ParameterSet( "deltaK bin5",    parStep[5],  parMin[5],  parMax[5] );
    parSet[6]   = ParameterSet( "deltaK bin6",    parStep[6],  parMin[6],  parMax[6] );
    parSet[7]   = ParameterSet( "deltaK bin7",    parStep[7],  parMin[7],  parMax[7] );
    parSet[8]   = ParameterSet( "deltaK bin9",    parStep[8],  parMin[8],  parMax[8] );
    parSet[9]   = ParameterSet( "deltaK bin10",   parStep[9],  parMin[9],  parMax[9] );
    parSet[10]  = ParameterSet( "deltaK bin11",   parStep[10], parMin[10], parMax[10] );
    parSet[11]  = ParameterSet( "deltaK bin12",   parStep[11], parMin[11], parMax[11] );
    parSet[12]  = ParameterSet( "deltaK bin13",   parStep[12], parMin[12], parMax[12] );
    parSet[13]  = ParameterSet( "deltaK bin14",   parStep[13], parMin[13], parMax[13] );
    parSet[14]  = ParameterSet( "deltaK bin15",   parStep[14], parMin[14], parMax[14] );
    parSet[15]  = ParameterSet( "deltaK bin17",   parStep[15], parMin[15], parMax[15] );
    parSet[16]  = ParameterSet( "deltaK bin18",   parStep[16], parMin[16], parMax[16] );
    parSet[17]  = ParameterSet( "deltaK bin19",   parStep[17], parMin[17], parMax[17] );
    parSet[18]  = ParameterSet( "deltaK bin20",   parStep[18], parMin[18], parMax[18] );
    parSet[19]  = ParameterSet( "deltaK bin21",   parStep[19], parMin[19], parMax[19] );
    parSet[20]  = ParameterSet( "deltaK bin22",   parStep[20], parMin[20], parMax[20] );
    parSet[21]  = ParameterSet( "deltaK bin23",   parStep[21], parMin[21], parMax[21] );
    parSet[22]  = ParameterSet( "deltaK bin25",   parStep[22], parMin[22], parMax[22] );
    parSet[23]  = ParameterSet( "deltaK bin26",   parStep[23], parMin[23], parMax[23] );
    parSet[24]  = ParameterSet( "deltaK bin27",   parStep[24], parMin[24], parMax[24] );
    parSet[25]  = ParameterSet( "deltaK bin28",   parStep[25], parMin[25], parMax[25] );
    parSet[26]  = ParameterSet( "deltaK bin29",   parStep[26], parMin[26], parMax[26] );
    parSet[27]  = ParameterSet( "deltaK bin30",   parStep[27], parMin[27], parMax[27] );
    parSet[28]  = ParameterSet( "deltaK bin31",   parStep[28], parMin[28], parMax[28] );
    parSet[29]  = ParameterSet( "deltaK bin33",   parStep[29], parMin[29], parMax[29] );
    parSet[30]  = ParameterSet( "deltaK bin34",   parStep[30], parMin[30], parMax[30] );
    parSet[31]  = ParameterSet( "deltaK bin35",   parStep[31], parMin[31], parMax[31] );
    parSet[32]  = ParameterSet( "deltaK bin36",   parStep[32], parMin[32], parMax[32] );
    parSet[33]  = ParameterSet( "deltaK bin37",   parStep[33], parMin[33], parMax[33] );
    parSet[34]  = ParameterSet( "deltaK bin38",   parStep[34], parMin[34], parMax[34] );
    parSet[35]  = ParameterSet( "deltaK bin39",   parStep[35], parMin[35], parMax[35] );
    parSet[36]  = ParameterSet( "deltaK bin41",   parStep[36], parMin[36], parMax[36] );
    parSet[37]  = ParameterSet( "deltaK bin42",   parStep[37], parMin[37], parMax[37] );
    parSet[38]  = ParameterSet( "deltaK bin43",   parStep[38], parMin[38], parMax[38] );
    parSet[39]  = ParameterSet( "deltaK bin44",   parStep[39], parMin[39], parMax[39] );
    parSet[40]  = ParameterSet( "deltaK bin45",   parStep[40], parMin[40], parMax[40] );
    parSet[41]  = ParameterSet( "deltaK bin46",   parStep[41], parMin[41], parMax[41] );
    parSet[42]  = ParameterSet( "deltaK bin47",   parStep[42], parMin[42], parMax[42] );
    parSet[43]  = ParameterSet( "deltaK bin49",   parStep[43], parMin[43], parMax[43] );
    parSet[44]  = ParameterSet( "deltaK bin50",   parStep[44], parMin[44], parMax[44] );
    parSet[45]  = ParameterSet( "deltaK bin51",   parStep[45], parMin[45], parMax[45] );
    parSet[46]  = ParameterSet( "deltaK bin52",   parStep[46], parMin[46], parMax[46] );
    parSet[47]  = ParameterSet( "deltaK bin53",   parStep[47], parMin[47], parMax[47] );
    parSet[48]  = ParameterSet( "deltaK bin54",   parStep[48], parMin[48], parMax[48] );
    parSet[49]  = ParameterSet( "deltaK bin55",   parStep[49], parMin[49], parMax[49] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, parSet );
  }

};


/// Service to build the scale functor corresponding to the passed identifier
scaleFunctionBase<double * > * scaleFunctionService( const int identifier );

/// Service to build the scale functor corresponding to the passed identifier when receiving a std::vector<double>
scaleFunctionBase<std::vector<double> > * scaleFunctionVecService( const int identifier );

// -------------- //
// Smear functors //
// -------------- //

class smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) = 0;
  smearFunctionBase() {
    cotgth_ = 0.;
    gRandom_ = new TRandom();
  }
  virtual ~smearFunctionBase() = 0;
protected:
  void smearEta(double & eta) {
    double theta;
    if (cotgth_!=0) {
      theta = atan(1/cotgth_);
    } else {
      theta = TMath::Pi()/2;
    }
    if (theta<0) theta += TMath::Pi();
    eta = -log(tan(theta/2));
  }
  double cotgth_;
  TRandom * gRandom_;
};
inline smearFunctionBase::~smearFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.

// No smearing
// -----------
class smearFunctionType0 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) { }
};
// The 3 parameters of smearType1 are: pt dependence of pt smear, phi smear and
// cotgtheta smear.
class smearFunctionType1 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt);
    phi = phi*(1.0+y[1]*parSmear[1]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[2]*parSmear[2]);
    smearEta(eta);
  }
};

class smearFunctionType2 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*std::fabs(eta));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]);
    smearEta(eta);
  }
};

class smearFunctionType3 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*std::fabs(eta));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*std::fabs(eta));
    smearEta(eta);
  }
};
// The six parameters of SmearType=4 are respectively:
// Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res.,
// |eta| dep. of |eta| res., Pt^2 dep. of Pt res.
class smearFunctionType4 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*std::fabs(eta)+y[5]*parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*std::fabs(eta));
    smearEta(eta);
  }
};

class smearFunctionType5 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*std::fabs(eta)+y[5]*parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*parSmear[2]+y[6]*parSmear[6]*pt);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*std::fabs(eta));
    smearEta(eta);
  }
};

//Smearing for MC correction based on the resolution function Type 15 for misaligned MC
class smearFunctionType6 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear) {
    double sigmaSmear = 0;
    double sigmaPtAl = 0;
    double sigmaPtMisal = 0;
    double ptPart = parSmear[0] + parSmear[1]*1/pt + pt*parSmear[2];
    double fabsEta = std::fabs(eta);

    sigmaPtAl = parSmear[14]*etaByPoints(eta, parSmear[15]);

    if (std::fabs(eta)<=1.4){
      sigmaPtMisal = ptPart + parSmear[3] + parSmear[4]*std::fabs(eta) + parSmear[5]*eta*eta;
      sigmaSmear = sqrt(std::fabs(pow(sigmaPtMisal,2)-pow(sigmaPtAl,2)));
      pt = pt*gRandom_->Gaus(1,sigmaSmear);
    }
    else if (eta>1.4){//eta in right endcap
      double par = parSmear[3] + parSmear[4]*1.4 + parSmear[5]*1.4*1.4 - (parSmear[6] + parSmear[7]*(1.4-parSmear[8]) + parSmear[9]*(1.4-parSmear[8])*(1.4-parSmear[8]));
      sigmaPtMisal = par + ptPart + parSmear[6] + parSmear[7]*std::fabs((fabsEta-parSmear[8])) + parSmear[9]*(fabsEta-parSmear[8])*(fabsEta-parSmear[8]);
      sigmaSmear = sqrt(std::fabs(pow(sigmaPtMisal,2)-pow(sigmaPtAl,2)));
      pt = pt*gRandom_->Gaus(1,sigmaSmear);
    }
    else{//eta in left endcap
      double par =  parSmear[3] + parSmear[4]*1.4 + parSmear[5]*1.4*1.4 - (parSmear[10] + parSmear[11]*(1.4-parSmear[12]) + parSmear[13]*(1.4-parSmear[12])*(1.4-parSmear[12]));
      sigmaPtMisal = par + ptPart + parSmear[10] + parSmear[11]*std::fabs((fabsEta-parSmear[12])) + parSmear[13]*(fabsEta-parSmear[12])*(fabsEta-parSmear[12]);
      sigmaSmear = sqrt(std::fabs(pow(sigmaPtMisal,2)-pow(sigmaPtAl,2)));
      pt = pt*gRandom_->Gaus(1,sigmaSmear);
    }
  }
 protected:
  /**
   * This is the pt vs eta resolution by points. It uses std::fabs(eta) assuming symmetry.
   * The values are derived from 100k events of MuonGun with 5<pt<100 and |eta|<3.
   */
  double etaByPoints(const double & inEta, const double & border) {
    Double_t eta = std::fabs(inEta);
    if( 0. <= eta && eta <= 0.2 ) return 0.00942984;
    else if( 0.2 < eta && eta <= 0.4 ) return 0.0104489;
    else if( 0.4 < eta && eta <= 0.6 ) return 0.0110521;
    else if( 0.6 < eta && eta <= 0.8 ) return 0.0117338;
    else if( 0.8 < eta && eta <= 1.0 ) return 0.0138142;
    else if( 1.0 < eta && eta <= 1.2 ) return 0.0165826;
    else if( 1.2 < eta && eta <= 1.4 ) return 0.0183663;
    else if( 1.4 < eta && eta <= 1.6 ) return 0.0169904;
    else if( 1.6 < eta && eta <= 1.8 ) return 0.0173289;
    else if( 1.8 < eta && eta <= 2.0 ) return 0.0205821;
    else if( 2.0 < eta && eta <= 2.2 ) return 0.0250032;
    else if( 2.2 < eta && eta <= 2.4 ) return 0.0339477;
    // ATTENTION: This point has a big error and it is very displaced from the rest of the distribution.
    else if( 2.4 < eta && eta <= 2.6 ) return border;
    return ( 0. );
  }
};

class smearFunctionType7 : public smearFunctionBase
{
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const std::vector<double> & parSmear)
  {
    double sigmaSquared = sigmaPtDiff.squaredDiff(eta);
    TF1 G("G", "[0]*exp(-0.5*pow(x,2)/[1])", -5., 5.);
    double norm = 1/(sqrt(2*TMath::Pi()*sigmaSquared));
    G.SetParameter (0,norm);
    G.SetParameter (1,sigmaSquared);
    pt = pt*(1-G.GetRandom());
  }
  SigmaPtDiff sigmaPtDiff;
};

/// Service to build the smearing functor corresponding to the passed identifier
smearFunctionBase * smearFunctionService( const int identifier );

// // Defined globally...
// static smearFunctionBase * smearFunctionArray[] = {
//   new smearFunctionType0,
//   new smearFunctionType1,
//   new smearFunctionType2,
//   new smearFunctionType3,
//   new smearFunctionType4,
//   new smearFunctionType5
// };

/**
 * Resolution functions. <br>
 * Need to use templates to make it work with both array and std::vector<double>.
 */
template <class T>
class resolutionFunctionBase {
 public:
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    return 0.;
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) = 0;
  virtual double covPt1Pt2(const double & pt1, const double & eta1, const double & pt2, const double & eta2, const T & parval)
  {
    return 0.;
  }
  resolutionFunctionBase() {}
  virtual ~resolutionFunctionBase() = 0;
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {};
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    std::cout << "The method setParameters must be implemented for this resolution function" << std::endl;
    exit(1);
  }
  virtual int parNum() const { return parNum_; }
 protected:
  int parNum_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
         TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
         double* thisStep, double* thisMini, double* thisMaxi, TString* thisParName ) {
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = parResol[iPar];
      Step[iPar] = thisStep[iPar];
      Mini[iPar] = thisMini[iPar];
      Maxi[iPar] = thisMaxi[iPar];
      ind[iPar] = parResolOrder[iPar];
      parname[iPar] = thisParName[iPar];
    }
  }
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
         TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const std::vector<ParameterSet> & parSet ) {
    if( int(parSet.size()) != this->parNum_ ) {
      std::cout << "Error: wrong number of parameter initializations = " << parSet.size() << ". Number of parameters is " << this->parNum_ << std::endl;
      exit(1);
    }
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = parResol[iPar];
      Step[iPar] = parSet[iPar].step;
      Mini[iPar] = parSet[iPar].mini;
      Maxi[iPar] = parSet[iPar].maxi;
      ind[iPar] = parResolOrder[iPar];
      parname[iPar] = parSet[iPar].name;
    }
  }
};
template <class T> inline resolutionFunctionBase<T>::~resolutionFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.

// SLIMMED VERSION
// The full set of resolutionFunction developed in the past can be found at 
// https://github.com/cms-sw/cmssw/blob/CMSSW_5_3_X/MuonAnalysis/MomentumScaleCalibration/interface/Functions.h#L3082-L5052

// null
// --------
template <class T>
class resolutionFunctionType0 : public resolutionFunctionBase<T> {
public:
  resolutionFunctionType0() {
    // One of the two is required. This follows from when templates are used by the compiler and the names lookup rules in c++.
    this->parNum_ = 0;
  }
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) { return 0; }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) { return 0; }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) { return 0.; }
  // derivatives ---------------
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError) { return 0; }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {}
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType) {}

};

// Binned in eta to fit the Z (parametrization as linear sum)
template <class T>
class resolutionFunctionType45 : public resolutionFunctionBase<T> {
 public:
  int etaBin(const double & eta)
  {
    // 12 bins from -2.4 to 2.4
    // std::cout << "for eta = " << eta << ", bin = " << bin << std::endl;
    if( eta < -2.0 ) return 1;
    if( eta < -1.8 ) return 2;
    if( eta < -1.6 ) return 3;
    if( eta < -1.2 ) return 4;
    if( eta < -0.8 ) return 5;
    if( eta < 0. )  return 6;
    if( eta < 0.8 ) return 7;
    if( eta < 1.2 ) return 8;
    if( eta < 1.6 ) return 9;
    if( eta < 1.8 ) return 10;
    if( eta < 2.0 ) return 11;
    return 12;
  }
   
  resolutionFunctionType45() { this->parNum_ = 13; }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    return (parval[0]*pt + parval[etaBin(eta)]); 
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    // Use the etaBin function to select the right bin for the parameter
    return sqrt( pow(pt*parError[0], 2) + pow(parError[etaBin(eta)], 2));
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
			     TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
			     const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", 0.002,      -0.1,  0.1  );
    parSet[1]  = ParameterSet( "eta bin 1",   0.00002,    -0.01, 0.01 );
    parSet[2]  = ParameterSet( "eta bin 2",   0.00002,    -0.01, 0.01 );
    parSet[3]  = ParameterSet( "eta bin 3",   0.00002,    -0.01, 0.01 );
    parSet[4]  = ParameterSet( "eta bin 4",   0.00002,    -0.01, 0.01 );
    parSet[5]  = ParameterSet( "eta bin 5",   0.00002,    -0.01, 0.01 );
    parSet[6]  = ParameterSet( "eta bin 6",   0.00002,    -0.01, 0.01 );
    parSet[7]  = ParameterSet( "eta bin 7",   0.00002,    -0.01, 0.01 );
    parSet[8]  = ParameterSet( "eta bin 8",   0.00002,    -0.01, 0.01 );
    parSet[9]  = ParameterSet( "eta bin 9",   0.00002,    -0.01, 0.01 );
    parSet[10] = ParameterSet( "eta bin 10",  0.00002,    -0.01, 0.01 );
    parSet[11] = ParameterSet( "eta bin 11",  0.00002,    -0.01, 0.01 );
    parSet[12] = ParameterSet( "eta bin 12",  0.00002,    -0.01, 0.01 );

    std::cout << "setting parameters" << std::endl;
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    if( (int(parStep.size()) != this->parNum_) || (int(parMin.size()) != this->parNum_) || (int(parMax.size()) != this->parNum_) ) {
      std::cout << "Error: par step or min or max do not match with number of parameters" << std::endl;
      std::cout << "parNum = " << this->parNum_ << std::endl;
      std::cout << "parStep.size() = " << parStep.size() << std::endl;
      std::cout << "parMin.size() = " << parMin.size() << std::endl;
      std::cout << "parMax.size() = " << parMax.size() << std::endl;
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta bin 1",   parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "eta bin 2",   parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "eta bin 3",   parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "eta bin 4",   parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "eta bin 5",   parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "eta bin 6",   parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "eta bin 7",   parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "eta bin 8",   parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "eta bin 9",   parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "eta bin 10",  parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "eta bin 11",  parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "eta bin 12",  parStep[12], parMin[12], parMax[12] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};


// Binned in eta to fit the Z (parametrization as sum in quadrature)
template <class T>
class resolutionFunctionType46 : public resolutionFunctionBase<T> {
 public:
  int etaBin(const double & eta)
  {
    // 12 bins from -2.4 to 2.4
    // std::cout << "for eta = " << eta << ", bin = " << bin << std::endl;
    if( eta < -2.0 ) return 1;
    if( eta < -1.8 ) return 2;
    if( eta < -1.6 ) return 3;
    if( eta < -1.2 ) return 4;
    if( eta < -0.8 ) return 5;
    if( eta < 0. )  return 6;
    if( eta < 0.8 ) return 7;
    if( eta < 1.2 ) return 8;
    if( eta < 1.6 ) return 9;
    if( eta < 1.8 ) return 10;
    if( eta < 2.0 ) return 11;
    return 12;
  }
   
  resolutionFunctionType46() { this->parNum_ = 13; }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    return sqrt(pow(parval[0]*pt,2) + pow(parval[etaBin(eta)],2));
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    // Use the etaByBin function to select the right bin for the parameter
    double r = sqrt(pow(parval[0]*pt,2) + pow(parval[etaBin(eta)],2));
    return sqrt( pow(pt*pt*parval[0]*parError[0],2) + pow(parval[etaBin(eta)]*parError[etaBin(eta)],2) )/r;
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
			     TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
			     const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", 0.0002,    0.,  0.1  );
    parSet[1]  = ParameterSet( "eta bin 1",   0.00002,   0.,  0.01 );
    parSet[2]  = ParameterSet( "eta bin 2",   0.00002,   0.,  0.01 );
    parSet[3]  = ParameterSet( "eta bin 3",   0.00002,   0.,  0.01 );
    parSet[4]  = ParameterSet( "eta bin 4",   0.00002,   0.,  0.01 );
    parSet[5]  = ParameterSet( "eta bin 5",   0.00002,   0.,  0.01 );
    parSet[6]  = ParameterSet( "eta bin 6",   0.00002,   0.,  0.01 );
    parSet[7]  = ParameterSet( "eta bin 7",   0.00002,   0.,  0.01 );
    parSet[8]  = ParameterSet( "eta bin 8",   0.00002,   0.,  0.01 );
    parSet[9]  = ParameterSet( "eta bin 9",   0.00002,   0.,  0.01 );
    parSet[10] = ParameterSet( "eta bin 10",  0.00002,   0.,  0.01 );
    parSet[11] = ParameterSet( "eta bin 11",  0.00002,   0.,  0.01 );
    parSet[12] = ParameterSet( "eta bin 12",  0.00002,   0.,  0.01 );

    std::cout << "setting parameters" << std::endl;
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    if( (int(parStep.size()) != this->parNum_) || (int(parMin.size()) != this->parNum_) || (int(parMax.size()) != this->parNum_) ) {
      std::cout << "Error: par step or min or max do not match with number of parameters" << std::endl;
      std::cout << "parNum = " << this->parNum_ << std::endl;
      std::cout << "parStep.size() = " << parStep.size() << std::endl;
      std::cout << "parMin.size() = " << parMin.size() << std::endl;
      std::cout << "parMax.size() = " << parMax.size() << std::endl;
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta bin 1",   parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "eta bin 2",   parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "eta bin 3",   parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "eta bin 4",   parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "eta bin 5",   parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "eta bin 6",   parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "eta bin 7",   parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "eta bin 8",   parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "eta bin 9",   parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "eta bin 10",  parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "eta bin 11",  parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "eta bin 12",  parStep[12], parMin[12], parMax[12] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

// Binned in eta to fit the Z (parametrization as sum in quadrature) and including an overall covariance
template <class T>
class resolutionFunctionType47 : public resolutionFunctionBase<T> {
 public:
  int etaBin(const double & eta)
  {
    // 12 bins from -2.4 to 2.4
    // std::cout << "for eta = " << eta << ", bin = " << bin << std::endl;
    if( eta < -2.0 ) return 1;
    if( eta < -1.8 ) return 2;
    if( eta < -1.6 ) return 3;
    if( eta < -1.2 ) return 4;
    if( eta < -0.8 ) return 5;
    if( eta < 0. )  return 6;
    if( eta < 0.8 ) return 7;
    if( eta < 1.2 ) return 8;
    if( eta < 1.6 ) return 9;
    if( eta < 1.8 ) return 10;
    if( eta < 2.0 ) return 11;
    return 12;
  }
  
  resolutionFunctionType47() { this->parNum_ = 14; }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    return sqrt(pow(parval[0]*pt,2) + pow(parval[etaBin(eta)],2) + pow(parval[13],2));
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  virtual double covPt1Pt2(const double & pt1, const double & eta1, const double & pt2, const double & eta2, const T & parval)
  {
    return parval[14];
  }

  // derivatives ---------------
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    // Use the etaByBin function to select the right bin for the parameter
    double r = sqrt(pow(parval[0]*pt,2) + pow(parval[etaBin(eta)],2) + pow(parval[13],2));
    return sqrt( pow(pt*pt*parval[0]*parError[0],2) + pow(parval[etaBin(eta)]*parError[etaBin(eta)],2) + pow(parval[13]*parError[13],2) )/r;
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
			     TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
			     const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", 0.0002,    0.,  0.1  );
    parSet[1]  = ParameterSet( "eta bin 1",   0.00002,   0.,  0.01 );
    parSet[2]  = ParameterSet( "eta bin 2",   0.00002,   0.,  0.01 );
    parSet[3]  = ParameterSet( "eta bin 3",   0.00002,   0.,  0.01 );
    parSet[4]  = ParameterSet( "eta bin 4",   0.00002,   0.,  0.01 );
    parSet[5]  = ParameterSet( "eta bin 5",   0.00002,   0.,  0.01 );
    parSet[6]  = ParameterSet( "eta bin 6",   0.00002,   0.,  0.01 );
    parSet[7]  = ParameterSet( "eta bin 7",   0.00002,   0.,  0.01 );
    parSet[8]  = ParameterSet( "eta bin 8",   0.00002,   0.,  0.01 );
    parSet[9]  = ParameterSet( "eta bin 9",   0.00002,   0.,  0.01 );
    parSet[10] = ParameterSet( "eta bin 10",  0.00002,   0.,  0.01 );
    parSet[11] = ParameterSet( "eta bin 11",  0.00002,   0.,  0.01 );
    parSet[12] = ParameterSet( "eta bin 12",  0.00002,   0.,  0.01 );
    parSet[13] = ParameterSet( "cov(pt1,pt2)",  0.00002,   0.,  0.01 );

    std::cout << "setting parameters" << std::endl;
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType)
  {
    if( (int(parStep.size()) != this->parNum_) || (int(parMin.size()) != this->parNum_) || (int(parMax.size()) != this->parNum_) ) {
      std::cout << "Error: par step or min or max do not match with number of parameters" << std::endl;
      std::cout << "parNum = " << this->parNum_ << std::endl;
      std::cout << "parStep.size() = " << parStep.size() << std::endl;
      std::cout << "parMin.size() = " << parMin.size() << std::endl;
      std::cout << "parMax.size() = " << parMax.size() << std::endl;
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.", parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta bin 1",   parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "eta bin 2",   parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "eta bin 3",   parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "eta bin 4",   parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "eta bin 5",   parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "eta bin 6",   parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "eta bin 7",   parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "eta bin 8",   parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "eta bin 9",   parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "eta bin 10",  parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "eta bin 11",  parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "eta bin 12",  parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "cov(pt1,pt2)",parStep[13], parMin[13], parMax[13] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};


// ------------ ATTENTION ----------- //
// Other functions are not in for now //
// ---------------------------------- //

/// Service to build the resolution functor corresponding to the passed identifier
resolutionFunctionBase<double *> * resolutionFunctionService( const int identifier );

/// Service to build the resolution functor corresponding to the passed identifier when receiving a std::vector<double>
resolutionFunctionBase<std::vector<double> > * resolutionFunctionVecService( const int identifier );

/**
 * Background functors. <br>
 * MuScleFit uses different background functions for each resonance. This is done because the
 * background shape can change from a region to another so that it is not well described just one of the following functions.
 * Since we are only interested in getting the correct shape and fraction in the resonance region we can split it in several
 * parts and fit them separately. <br>
 * <br>
 * When fitting the background function: <br>
 * - we define three regions: <br>
 * -- Psis region: centered in the mean value of J/Psi and Psi2S masses. <br>
 * -- Upsilons region: centered in the mean value of all the Upsilon masses. <br>
 * -- Z region. <br>
 * - In this case we thus have only three background functions, that is three sets of parameters. <br>
 * When not fitting the background function: <br>
 * - the windows considered are much narrower and centered around each resonance. <br>
 * - In this case we compute a rescaled background fraction parameter for each resonance and we use the rest of the parameters
 * for the corresponding region. <br>
 * ATTENTION: we are assuming that J/Psi and Psi2S have the same bin width. The same assumption is done for the Upsilons. <br>
 * This is the case in the present probability root file. If this changes, the background function for the different regions
 * must keep it into account. <br>
 * <br>
 * All this is handled in MuScleFit by a single functor: BackgroundHandler defined in another file in this dir. <br>
 * ATTENTION: The BackgroundHandler assumes that the background fraction is always the first parameter (parBgr[0]).
 */
class backgroundFunctionBase {
 public:
  backgroundFunctionBase(const double & lowerLimit, const double & upperLimit) :
    lowerLimit_(lowerLimit), upperLimit_(upperLimit) {}
  virtual ~backgroundFunctionBase()
  {
    delete functionForIntegral_;
  };
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const = 0;
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    return operator()(parval, mass, eta1);
  }
  virtual int parNum() const { return parNum_; }
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) = 0;
  virtual TF1* functionForIntegral(const std::vector<double>::const_iterator & parBgrIt) const
  {
    functionForIntegral_ = new FunctionForIntegral(this, parBgrIt);
    TF1 * backgroundFunctionForIntegral = new TF1("backgroundFunctionForIntegral", functionForIntegral_,
                                                  lowerLimit_, upperLimit_, this->parNum_);
    return( backgroundFunctionForIntegral );
  }
  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const { return 1.; }

protected:
  int parNum_;
  double lowerLimit_;
  double upperLimit_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
                      const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt,
                      double* thisStep, double* thisMini, double* thisMaxi, TString* thisParName ) {
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = *(parBgrIt+iPar);
      Step[iPar] = thisStep[iPar];
      Mini[iPar] = thisMini[iPar];
      Maxi[iPar] = thisMaxi[iPar];
      ind[iPar] = *(parBgrOrderIt+iPar);
      // EM 2012.05.22 this line is crashing cmsRun (need to be fixed)      parname[iPar] = thisParName[iPar];
    }
  }
  class FunctionForIntegral
  {
  public:
    FunctionForIntegral( const backgroundFunctionBase * function,
                         const std::vector<double>::const_iterator & parBgrIt ) :
      function_(function)
    {
      parval_ = new double[function_->parNum()];
      for( int i=0; i < function_->parNum(); ++i ) {
        parval_[i] = *(parBgrIt+i);
      }
    }
    ~FunctionForIntegral()
    {
      delete parval_;
    }
    double operator()(const double * mass, const double *) const
    {
      // FIXME: this is a gross approximation. The function should be integrated in eta over the sample.
      return( (*function_)(parval_, *mass, 0.) );
    }
  protected:
    const backgroundFunctionBase * function_;
    double * parval_;
  };
  mutable FunctionForIntegral * functionForIntegral_;
};

/// Linear
// -------
class backgroundFunctionType1 : public backgroundFunctionBase
{
 public:
  /**
   * Returns the value of the linear function f(M) = 1 + b*M for M < -1/b, 0 otherwise. <br>
   * b is chosen to be negative (background decreasing when M increases). <br>
   * Note that this form describes only cases with a != 0 (keep in mind that the relative height
   * with respect to the signal is controlled by the fraction parameter).
   */
  backgroundFunctionType1(const double & lowerLimit, const double & upperLimit) :
    backgroundFunctionBase(lowerLimit, upperLimit)
    { this->parNum_ = 2; }
    virtual double operator()( const double * parval, const double & mass, const double & eta ) const
  {
    double a = 1.;
    double b = parval[1];

    double norm = -(a*lowerLimit_ + b*lowerLimit_*lowerLimit_/2.);

    if( -a/b > upperLimit_ ) norm += a*upperLimit_ + b*upperLimit_*upperLimit_/2.;
    else norm += -a*a/(2*b);

    if( mass < -a/b && norm != 0 ) return (a + b*mass)/norm;
    else return 0;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Constant", "Linear"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, -300.};
      double thisMaxi[] = {1.0,    0.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, -300.};
      double thisMaxi[] = {1.0,    0.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
/// Exponential
// ------------
class backgroundFunctionType2 : public backgroundFunctionBase {
 public:
  /**
   * In case of an exponential, we normalize it such that it has integral in any window
   * equal to unity, and then, when adding together all the resonances, one gets a meaningful
   * result for the overall background fraction.
   */
  backgroundFunctionType2(const double & lowerLimit, const double & upperLimit) :
    backgroundFunctionBase(lowerLimit, upperLimit)
    { this->parNum_ = 2; }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const
  {
    double Bgrp2 = parval[1];
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }




  // virtual double fracVsEta(const double * parval, const double & resEta) const
  // {
  //   // return( 0.6120 - 0.0225*eta*eta );
  //   return( 1. - 0.0225*resEta*resEta ); // so that a = 1 for eta = 0.
  // }




};







///// Constant + Exponential
//// -------------------------------------------------------------------------------- //
//// ATTENTION: TODO: the normalization must be adapted to the asymmetric mass window //
//// -------------------------------------------------------------------------------- //
//
//class backgroundFunctionType3 : public backgroundFunctionBase {
// public:
//  // pass parval[shift]
//  backgroundFunctionType3(const double & lowerLimit, const double & upperLimit) :
//    backgroundFunctionBase(lowerLimit, upperLimit)
//    { this->parNum_ = 3; }
//  virtual double operator()( const double * parval, const int resTotNum, const int ires, const bool * resConsidered,
//                             const double * ResMass, const double ResHalfWidth[], const int MuonType, const double & mass, const int nbins ) {
//    double PB = 0.;
//    double Bgrp2 = parval[1];
//    double Bgrp3 = parval[2];
//    for (int ires=0; ires<resTotNum; ires++) {
//      // In this case, by integrating between A and B, we get for f=exp(a-bx)+k:
//      // INT = exp(a)/b*(exp(-bA)-exp(-bB))+k*(B-A) so our function, which in 1000 bins between A and B
//      // gets a total of 1, is f = (exp(a-bx)+k)*(B-A)/nbins / (INT)
//      // ----------------------------------------------------------------------------------------------
//      if (resConsidered[ires]) {
//	if (exp(-Bgrp2*(ResMass[ires]-ResHalfWidth[ires]))-exp(-Bgrp2*(ResMass[ires]+ResHalfWidth[ires]))>0) {
//	  PB += (exp(-Bgrp2*mass)+Bgrp3) *
//	    2*ResHalfWidth[ires]/(double)nbins /
//	    ( (exp(-Bgrp2*(ResMass[ires]-ResHalfWidth[ires]))-exp(-Bgrp2*(ResMass[ires]+ResHalfWidth[ires])))/
//	      Bgrp2 + Bgrp3*2*ResHalfWidth[ires] );
//	} else {
//	  std::cout << "Impossible to compute Background probability! - some fix needed - Bgrp2=" << Bgrp2 << std::endl;
//	}
//      }
//    }
//    return PB;
//  }
//  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
//    double thisStep[] = {0.1, 0.001, 0.1};
//    TString thisParName[] = {"Bgr fraction", "Bgr slope", "Bgr constant"};
//    if( muonType == 1 ) {
//      double thisMini[] = {0.0, 0.000000001, 0.0};
//      double thisMaxi[] = {1.0, 0.2, 1000};
//      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
//    } else {
//      double thisMini[] = {0.0, 0.000000001, 0.0};
//      double thisMaxi[] = {1.0, 0.2, 1000};
//      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
//    }
//  }
//  virtual TF1* functionForIntegral(const std::vector<double>::const_iterator & parBgrIt) const {return 0;};
//};



/// Exponential with eta dependence
// --------------------------------
class backgroundFunctionType4 : public backgroundFunctionBase
{
 public:
  /**
   * In case of an exponential, we normalize it such that it has integral in any window
   * equal to unity, and then, when adding together all the resonances, one gets a meaningful
   * result for the overall background fraction.
   */
  backgroundFunctionType4(const double & lowerLimit, const double & upperLimit) :
    backgroundFunctionBase(lowerLimit, upperLimit)
    { this->parNum_ = 4; }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const
  {
    double Bgrp2 = parval[1] + parval[2]*eta*eta;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01, 0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope", "Bgr slope eta^2 dependence", "background fraction eta dependence"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.,   0., -1.};
      double thisMaxi[] = {1.0, 10., 10.,  1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0., -1., -1.};
      double thisMaxi[] = {1.0, 10., 1.,  1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
  /* virtual double fracVsEta(const double * parval, const double & resEta) const */
  /* { */
  /*   return( 1. - parval[3]*resEta*resEta ); // so that a = 1 for eta = 0. */
  /* } */
};

/// Linear with eta dependence
// ---------------------------
class backgroundFunctionType5 : public backgroundFunctionBase
{
 public:
  /**
   * Returns the value of the linear function f(M) = a + b*M for M < -a/b, 0 otherwise. <br>
   * Where a = 1 + c*eta*eta and b is chosen to be negative (background decreasing when M increases).
   */
  backgroundFunctionType5(const double & lowerLimit, const double & upperLimit) :
    backgroundFunctionBase(lowerLimit, upperLimit)
    { this->parNum_ = 3; }
    virtual double operator()( const double * parval, const double & mass, const double & eta ) const
  {
    double b = parval[1];
    // double c = parval[2];
    double a = 1 + parval[2]*eta*eta;

    double norm = -(a*lowerLimit_ + b*lowerLimit_*lowerLimit_/2.);

    if( -a/b > upperLimit_ ) norm += a*upperLimit_ + b*upperLimit_*upperLimit_/2.;
    else norm += -a*a/(2*b);

    if( mass < -a/b && norm != 0 ) return (a + b*mass)/norm;
    else return 0;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Constant", "Linear"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0,  0., -300.};
      double thisMaxi[] = {1.0, 300.,   0.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0,  0., -300.};
      double thisMaxi[] = {1.0, 300.,   0.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


/// Exponential binned in eta
// --------------------------
class backgroundFunctionType6 : public backgroundFunctionBase {
 public:
  backgroundFunctionType6(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;
    if( fabs(eta1) <= 1.3 && fabs(eta2) <= 1.3 ) {
      Bgrp2 = -1.20528;
    }
    else if( (fabs(eta1) <= 1.6 && fabs(eta1) > 1.3) && (fabs(eta2) <= 1.6 && fabs(eta2) > 1.3) ) {
      Bgrp2 = 0.234713;
    }
    else if( fabs(eta1) > 1.6 && fabs(eta2) > 1.6 ) {
      Bgrp2 = -0.667103;
    }
    else if( (fabs(eta1) <= 1.3 && (fabs(eta2) > 1.3 && fabs(eta2) <= 1.6)) ||
	     (fabs(eta2) <= 1.3 && (fabs(eta1) > 1.3 && fabs(eta1) <= 1.6)) ) {
      Bgrp2 = -0.656904;
    }
    else if( (fabs(eta1) <= 1.3 && fabs(eta2) > 1.6) ||
	     (fabs(eta2) <= 1.3 && fabs(eta1) > 1.6) ) {
      Bgrp2 = 0.155328;
    }
    else if( ((fabs(eta1) > 1.3 && fabs(eta1) <= 1.6) && fabs(eta2) > 1.6) ||
	     ((fabs(eta2) > 1.3 && fabs(eta2) <= 1.6) && fabs(eta1) > 1.6) ) {
      Bgrp2 = -0.177154;
    }
    else {
      std::cout << "WARNING: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      Bgrp2 = -0.667103;
    }
    Bgrp2*=-1.;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;

  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const
  {
    if( fabs(eta1) <= 1.3 && fabs(eta2) <= 1.3 ) {
      return (1.-0.910903);
    }
    else if( (fabs(eta1) <= 1.6 && fabs(eta1) > 1.3) && (fabs(eta2) <= 1.6 && fabs(eta2) > 1.3) ) {
      return (1.-0.801469);
    }
    else if( fabs(eta1) > 1.6 && fabs(eta2) > 1.6 ) {
      return (1.-0.658196);
    }
    else if( (fabs(eta1) <= 1.3 && (fabs(eta2) > 1.3 && fabs(eta2) <= 1.6)) ||
	     (fabs(eta2) <= 1.3 && (fabs(eta1) > 1.3 && fabs(eta1) <= 1.6)) ) {
      return (1.-0.873411);
    }
    else if( (fabs(eta1) <= 1.3 && fabs(eta2) > 1.6) ||
	     (fabs(eta2) <= 1.3 && fabs(eta1) > 1.6) ) {
      return (1.-0.784674);
    }
    else if( ((fabs(eta1) > 1.3 && fabs(eta1) <= 1.6) && fabs(eta2) > 1.6) ||
	     ((fabs(eta2) > 1.3 && fabs(eta2) <= 1.6) && fabs(eta1) > 1.6) ) {
      return (1.-0.714398);
    }
    else {
      std::cout << "WARNING: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      return  (1.-0.658196);
    }
  }
};

/// Exponential binned in eta, much finer binning then type6
// ---------------------------------------------------------
class backgroundFunctionType7 : public backgroundFunctionBase {
 public:
  backgroundFunctionType7(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;
    if( (fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 0. && fabs(eta2) < 0.9) ) {
      Bgrp2 = (-1.42465);
    }
    else if( (fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) ) {
      Bgrp2 = (-1.38576);
    }
    else if( (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) ) {
      Bgrp2 = (-0.333728);
    }
    else if( (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) ) {
      Bgrp2 = (0.94066);
    }
    else if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) ) {
      Bgrp2 = (0.371026);
    }
    else if( (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) ) {
      Bgrp2 = (-0.959101);
    }
    else if( (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) ) {
      Bgrp2 = (-1.13829);
    }
    else if( (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0) ) {
      Bgrp2 = (-0.921581);
    }
    else if( (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.) ) {
      Bgrp2 = (-0.664338);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 0.9 && fabs(eta2) < 1.3)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 0.9 && fabs(eta1) < 1.3)) ) {
      Bgrp2 = (-1.07581);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5)) ) {
      Bgrp2 = (-0.250272);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      Bgrp2 = (0.101785);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      Bgrp2 = (0.360397);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (0.689136);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (0.860723);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (1.21908);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (2.4453);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5)) ) {
      Bgrp2 = (-1.14152);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      Bgrp2 = (-0.77241);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      Bgrp2 = (-0.516479);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (-0.361401);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (-0.33143);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.20813);
    }
    else if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (0.158002);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      Bgrp2 = (0.273222);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      Bgrp2 = (0.247639);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (-0.148616);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (-0.413175);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.230031);
    }
    else if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.122756);
    }
    else if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      Bgrp2 = (0.650851);
    }
    else if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (-0.0985001);
    }
    else if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (-0.402548);
    }
    else if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.27401);
    }
    else if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.22863);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (-0.436959);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (-0.506041);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.31618);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.365653);
    }
    else if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      Bgrp2 = (-1.16783);
    }
    else if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.730701);
    }
    else if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.5271);
    }
    else if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.99893);
    }
    else if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.687263);
    }
    else if( ((fabs(eta1) >= 1.9 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.9 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      Bgrp2 = (-0.722394);
    }
    else {
      std::cout << "WARNING: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      Bgrp2 = -0.664338;
    }
    Bgrp2*=-1.;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;

  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const
  {
    if( (fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 0. && fabs(eta2) < 0.9) ) {
      return (1.-0.915365);
    }
    if( (fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) ) {
      return (1.-0.914149);
    }
    if( (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) ) {
      return (1.-0.855918);
    }
    if( (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) ) {
      return (1.-0.70221);
    }
    if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) ) {
      return (1.-0.701489);
    }
    if( (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) ) {
      return (1.-0.651162);
    }
    if( (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) ) {
      return (1.-0.639839);
    }
    if( (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0) ) {
      return (1.-0.64915);
    }
    if( (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.) ) {
      return (1.-0.687878);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 0.9 && fabs(eta2) < 1.3)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 0.9 && fabs(eta1) < 1.3)) ) {
      return (1.-0.903486);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5)) ) {
      return (1.-0.882516);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      return (1.-0.85477);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      return (1.-0.804919);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      return (1.-0.75411);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.714128);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.645403);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.9) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.9) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.588049);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.3 && fabs(eta2) < 1.5)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.3 && fabs(eta1) < 1.5)) ) {
      return (1.-0.901123);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      return (1.-0.87852);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      return (1.-0.862266);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      return (1.-0.846385);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.825401);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.812449);
    }
    if( ((fabs(eta1) >= 0.9 && fabs(eta1) < 1.3) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 0.9 && fabs(eta2) < 1.3) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.753754);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.5 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.5 && fabs(eta1) < 1.6)) ) {
      return (1.-0.794143);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      return (1.-0.761375);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      return (1.-0.765572);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.749438);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.750941);
    }
    if( ((fabs(eta1) >= 1.3 && fabs(eta1) < 1.5) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.3 && fabs(eta2) < 1.5) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.722832);
    }
    if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.7)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.7)) ) {
      return (1.-0.699723);
    }
    if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      return (1.-0.734044);
    }
    if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.719434);
    }
    if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.718889);
    }
    if( ((fabs(eta1) >= 1.5 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.5 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.689382);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.7 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.7 && fabs(eta1) < 1.8)) ) {
      return (1.-0.681106);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.685783);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.695924);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.7) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.7) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.670977);
    }
    if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.8 && fabs(eta2) < 1.9)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.8 && fabs(eta1) < 1.9)) ) {
      return (1.-0.654816);
    }
    if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.670969);
    }
    if( ((fabs(eta1) >= 1.7 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.7 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.659082);
    }
    if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 1.9 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) && (fabs(eta1) >= 1.9 && fabs(eta1) < 2.0)) ) {
      return (1.-0.648371);
    }
    if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 1.9) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 1.9) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.659114);
    }
    if( ((fabs(eta1) >= 1.9 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.0 && fabs(eta2) < 1000.)) ||
	((fabs(eta2) >= 1.9 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.0 && fabs(eta1) < 1000.)) ) {
      return (1.-0.660482);
    }
    else {
      std::cout << "WARNING: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      return  (1.-0.687878);
    }
  }
};
//Function 8
class backgroundFunctionType8 : public backgroundFunctionBase {
 public:
  backgroundFunctionType8(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;
if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
  Bgrp2 = (-2.17047);
}
else if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
  Bgrp2 = (-2.12913);
}
else if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
  Bgrp2 = (-2.19963);
}
else if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1000) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000) ) {
  Bgrp2 = (-0.386394);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
  Bgrp2 = (-1.71339);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  Bgrp2 = (-0.206566);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (4.4815);
}
else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  Bgrp2 = (-1.87985);
}
else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (-0.163569);
}
else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (-1.67935);
}
    Bgrp2*=-1.;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;

  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const
  {
if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
  return (1.-0.907727);
}
if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
  return (1.-0.907715);
}
if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
  return (1.-0.912233);
}
if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1000) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000) ) {
  return (1.-0.876776);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
  return (1.-0.913046);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  return (1.-0.916765);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.6);
}
if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  return (1.-0.907471);
}
if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.899253);
}
if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.879108);
}
  return 0;}
};

/// Service to build the background functor corresponding to the passed identifier
backgroundFunctionBase * backgroundFunctionService( const int identifier, const double & lowerLimit, const double & upperLimit );

//Function Type 9

class backgroundFunctionType9 : public backgroundFunctionBase {
 public:
  backgroundFunctionType9(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;

if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
  Bgrp2 = (-1.80833);
}
else if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
  Bgrp2 = (-1.98281);
}
else if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
  Bgrp2 = (-1.79632);
}
else if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1000) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000) ) {
  Bgrp2 = (-1.14645);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
  Bgrp2 = (-1.55747);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  Bgrp2 = (-0.337598);
}
else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (5.36513);
}
else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  Bgrp2 = (-1.44363);
}
else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (-0.54614);
}
else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (-1.41059);
}
else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  Bgrp2 = (-1.41059);
}
    Bgrp2*=-1.;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;

  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const

 {
   // std::cout << "Eta1 " << eta1 << " eta2 = " << eta2 << std::endl;
if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
  return (1.-0.893683);
}
if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
  return (1.-0.888968);
}
if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
  return (1.-0.885926);
}
if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1000) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000) ) {
  return (1.-0.866615);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
  return (1.-0.892856);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  return (1.-0.884864);
}
if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.6);
}
if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
  return (1.-0.894739);
}
if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.880597);
}
if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.869165);
}
if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
    ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
  return (1.-0.869165);
}
 return 0;}
};


class backgroundFunctionType10 : public backgroundFunctionBase {
 public:
  backgroundFunctionType10(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;
    if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
      Bgrp2 = (-1.80833);
    }
    else if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
      Bgrp2 = (-1.98281);
    }
    else if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
      Bgrp2 = (-1.79632);
    }
    else if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) ) {
      Bgrp2 = (-1.50855);
    }
    else if( (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) ) {
      Bgrp2 = (-0.498511);
    }
    else if( (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2) ) {
      Bgrp2 = (-0.897031);
    }
    else if( (fabs(eta1) >= 2.2 && fabs(eta1) < 1000) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000) ) {
      Bgrp2 = (-0.75954);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
	     ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
      Bgrp2 = (-1.55747);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
	     ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
      Bgrp2 = (-0.337598);
    }
    else if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (3.5163);
    }
    else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
	     ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
      Bgrp2 = (-1.44363);
    }
    else if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (-0.54614);
    }
    else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.8)) ||
	     ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.8)) ) {
      Bgrp2 = (-1.36442);
    }
    else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0)) ||
	     ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-1.66202);
    }
    else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	     ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      Bgrp2 = (-0.62038);
    }
    else if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (0.662449);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0)) ||
	     ((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0)) ) {
      Bgrp2 = (-0.723325);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	     ((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      Bgrp2 = (-1.54405);
    }
    else if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (-1.1104);
    }
    else if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	     ((fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      Bgrp2 = (-1.56277);
    }
    else if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (-1.0827);
    }
    else if( ((fabs(eta1) >= 2.0 && fabs(eta1) < 2.2) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	     ((fabs(eta2) >= 2.0 && fabs(eta2) < 2.2) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      Bgrp2 = (-1.05662);
    }
    Bgrp2*=-1.;
    double norm = -(exp(-Bgrp2*upperLimit_) - exp(-Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(-Bgrp2*mass)/norm;
    else return 0.;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const
  {
    if( (fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0. && fabs(eta2) < 0.85) ) {
      return (1.-0.893683);
    }
    if( (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) ) {
      return (1.-0.888968);
    }
    if( (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) ) {
      return (1.-0.885926);
    }
    if( (fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) ) {
      return (1.-0.892823);
    }
    if( (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) ) {
      return (1.-0.888735);
    }
    if( (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2) ) {
      return (1.-0.87497);
    }
    if( (fabs(eta1) >= 2.2 && fabs(eta1) < 1000) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000) ) {
      return (1.-0.895275);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 0.85 && fabs(eta2) < 1.25)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 0.85 && fabs(eta1) < 1.25)) ) {
      return (1.-0.892856);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
      return (1.-0.884864);
    }
    if( ((fabs(eta1) >= 0. && fabs(eta1) < 0.85) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 0. && fabs(eta2) < 0.85) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
      return (1.-0.834572);
    }
    if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.25 && fabs(eta2) < 1.6)) ||
	((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.25 && fabs(eta1) < 1.6)) ) {
      return (1.-0.894739);
    }
    if( ((fabs(eta1) >= 0.85 && fabs(eta1) < 1.25) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 0.85 && fabs(eta2) < 1.25) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1000)) ) {
      return (1.-0.880597);
    }
    if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.6 && fabs(eta2) < 1.8)) ||
	((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.6 && fabs(eta1) < 1.8)) ) {
      return (1.-0.892911);
    }
    if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0)) ) {
      return (1.-0.880506);
    }
    if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      return (1.-0.885718);
    }
    if( ((fabs(eta1) >= 1.25 && fabs(eta1) < 1.6) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 1.25 && fabs(eta2) < 1.6) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      return (1.-0.853141);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 1.8 && fabs(eta2) < 2.0)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 1.8 && fabs(eta1) < 2.0)) ) {
      return (1.-0.88822);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      return (1.-0.87028);
    }
    if( ((fabs(eta1) >= 1.6 && fabs(eta1) < 1.8) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 1.6 && fabs(eta2) < 1.8) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      return (1.-0.869603);
    }
    if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.0 && fabs(eta2) < 2.2)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.0 && fabs(eta1) < 2.2)) ) {
      return (1.-0.877922);
    }
    if( ((fabs(eta1) >= 1.8 && fabs(eta1) < 2.0) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 1.8 && fabs(eta2) < 2.0) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      return (1.-0.865997);
    }
    if( ((fabs(eta1) >= 2.0 && fabs(eta1) < 2.2) && (fabs(eta2) >= 2.2 && fabs(eta2) < 1000)) ||
	((fabs(eta2) >= 2.0 && fabs(eta2) < 2.2) && (fabs(eta1) >= 2.2 && fabs(eta1) < 1000)) ) {
      return (1.-0.886109);
    }
    return 0;
  }
};


/// Exponential binned in eta (Z, Run2012C PromptReco-v1 + PromptReco-v2)
// --------------------------
class backgroundFunctionType11 : public backgroundFunctionBase {
 public:
  backgroundFunctionType11(const double & lowerLimit, const double & upperLimit) :
  backgroundFunctionBase(lowerLimit, upperLimit)
  {
    this->parNum_ = 2;
  }
  virtual double operator()( const double * parval, const double & mass, const double & eta ) const {return 0.;}
  virtual double operator()( const double * parval, const double & mass, const double & eta1, const double & eta2 ) const
  {
    double Bgrp2 = 0.;
    if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= -100. && eta2 < -0.8) ) {
      Bgrp2 = (-0.0512353);
    }
    else if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= -0.8 && eta2 < 0.) ) {
      Bgrp2 = (-0.0448482);
    }
    else if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= 0. && eta2 < 0.8) ) {
      Bgrp2 = (-0.0193726);
    }
    else if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= 0.8 && eta2 < 100.) ) {
      Bgrp2 = (0.0225765);
    }
    else if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= -100. && eta2 < -0.8) ) {
      Bgrp2 = (-0.0822936);
    }
    else if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= -0.8 && eta2 < 0.) ) {
      Bgrp2 = (-0.0676357);
    }
    else if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= 0. && eta2 < 0.8) ) {
      Bgrp2 = (-0.0591544);
    }
    else if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= 0.8 && eta2 < 100.) ) {
      Bgrp2 = (-0.0235858);
    }
    else if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= -100. && eta2 < -0.8) ) {
      Bgrp2 = (-0.0317051);
    }
    else if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= -0.8 && eta2 < 0.) ) {
      Bgrp2 = (-0.06139);
    }
    else if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= 0. && eta2 < 0.8) ) {
      Bgrp2 = (-0.0747737);
    }
    else if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= 0.8 && eta2 < 100.) ) {
      Bgrp2 = (-0.0810139);
    }
    else if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= -100. && eta2 < -0.8) ) {
      Bgrp2 = (0.0229602);
    }
    else if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= -0.8 && eta2 < 0.) ) {
      Bgrp2 = (-0.0224212);
    }
    else if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= 0. && eta2 < 0.8) ) {
      Bgrp2 = (-0.0446273);
    }
    else if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= 0.8 && eta2 < 100.) ) {
      Bgrp2 = (-0.0554561);
    }
    else {
      std::cout << "WARNING, backgroundFunctionType11: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      return (-0.05);
    }    
    double norm = (exp(Bgrp2*upperLimit_) - exp(Bgrp2*lowerLimit_))/Bgrp2;
    if( norm != 0 ) return exp(Bgrp2*mass)/norm;
    else return 0.;

  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double>::const_iterator & parBgrIt, const std::vector<int>::const_iterator & parBgrOrderIt, const int muonType) {
    double thisStep[] = {0.01, 0.01};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {-1.0, 10.};
      double thisMaxi[] = {1.0 , 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-1.0, 10.};
      double thisMaxi[] = { 1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgrIt, parBgrOrderIt, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

  virtual double fracVsEta(const double * parval, const double & eta1, const double & eta2) const
  {
    if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= -100. && eta2 < -0.8) ) {
      return (1.-0.966316);
    }
    if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= -0.8 && eta2 < 0.) ) {
      return (1.-0.966875);
    }
    if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= 0. && eta2 < 0.8) ) {
      return (1.-0.955311);
    }
    if( (eta1 >= -100. && eta1 < -0.8) && (eta2 >= 0.8 && eta2 < 100.) ) {
      return (1.-0.928771);
    }
    if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= -100. && eta2 < -0.8) ) {
      return (1.-0.983255);
    }
    if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= -0.8 && eta2 < 0.) ) {
      return (1.-0.982203);
    }
    if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= 0. && eta2 < 0.8) ) {
      return (1.-0.972127);
    }
    if( (eta1 >= -0.8 && eta1 < 0.) && (eta2 >= 0.8 && eta2 < 100.) ) {
      return (1.-0.962929);
    }
    if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= -100. && eta2 < -0.8) ) {
      return (1.-0.965597);
    }
    if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= -0.8 && eta2 < 0.) ) {
      return (1.-0.969461);
    }
    if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= 0. && eta2 < 0.8) ) {
      return (1.-0.979922);
    }
    if( (eta1 >= 0. && eta1 < 0.8) && (eta2 >= 0.8 && eta2 < 100.) ) {
      return (1.-0.984247);
    }
    if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= -100. && eta2 < -0.8) ) {
      return (1.-0.934252);
    }
    if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= -0.8 && eta2 < 0.) ) {
      return (1.-0.952914);
    }
    if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= 0. && eta2 < 0.8) ) {
      return (1.-0.960191);
    }
    if( (eta1 >= 0.8 && eta1 < 100.) && (eta2 >= 0.8 && eta2 < 100.) ) {
      return (1.-0.966175);
    }
    else {
      std::cout << "WARNING, backgroundFunctionType11: this should not happen for eta1 = " << eta1 << " and eta2 = " << eta2 << std::endl;
      return (1.-0.97);
    }
  }
};


#endif // FUNCTIONS_H
