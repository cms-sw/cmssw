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
// Linear in pt
// ------------
template <class T>
class scaleFunctionType1 : public scaleFunctionBase<T> {
public:
  scaleFunctionType1() { this->parNum_ = 2; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return ( (parScale[0] + parScale[1]*pt)*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.97, -0.1};
      double thisMaxi[] = {1.03, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.1};
      double thisMaxi[] = {1.03, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
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

    parSet[0]  = ParameterSet( "Pt offset", parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "Pt slope" , parStep[1],  parMin[1],  parMax[1]  );


    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};
// Linear in |eta|
// ---------------
template <class T>
class scaleFunctionType2 : public scaleFunctionBase<T> {
public:
  scaleFunctionType2() { this->parNum_ = 2; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return ( (parScale[0] + parScale[1]*std::fabs(eta))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01};
    TString thisParName[] = {"Eta offset", "Eta slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3};
      double thisMaxi[] = {1.1, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.1};
      double thisMaxi[] = {1.03, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Sinusoidal in phi
// -----------------
template <class T>
class scaleFunctionType3 : public scaleFunctionBase<T> {
public:
  scaleFunctionType3() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*sin(parScale[2]*phi + parScale[3]))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01};
    TString thisParName[] = {"Phi offset", "Phi ampl", "Phi freq", "Phi phase"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.1, -0.1};
      double thisMaxi[] = {1.1, 0.1, 0.1, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.05, 6, -3.14};
      double thisMaxi[] = {1.03, 0.05, 0.1, 3.14};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Linear in pt and |eta|
// ----------------------
template <class T>
class scaleFunctionType4 : public scaleFunctionBase<T> {
public:
  scaleFunctionType4() { this->parNum_ = 3; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*std::fabs(eta))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.1};
      double thisMaxi[] = {1.1, 0.1, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.02};
      double thisMaxi[] = {1.03, 0.02, 0.02};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Linear in pt and sinusoidal in phi
// ----------------------------------
template <class T>
class scaleFunctionType5 : public scaleFunctionBase<T> {
public:
  scaleFunctionType5() { this->parNum_ = 3; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*sin(phi))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Phi ampl"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.3};
      double thisMaxi[] = {1.1, 0.1, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.3};
      double thisMaxi[] = {1.03, 0.02, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Linear in |eta| and sinusoidal in phi
// -------------------------------------
template <class T>
class scaleFunctionType6 : public scaleFunctionBase<T> {
public:
  scaleFunctionType6() { this->parNum_ = 3; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return (parScale[0] + parScale[1]*std::fabs(eta) +
             parScale[2]*sin(phi))*pt;
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    //    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01};
    TString thisParName[] = {"Eta offset", "Eta slope", "Phi ampl"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.3};
      double thisMaxi[] = {1.1, 0.1, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.3};
      double thisMaxi[] = {1.03, 0.02, 0.3};
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
    parSet[0]  = ParameterSet( "Eta offset",        parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Eta slope",         parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Phi ampl",          parStep[2], parMin[2], parMax[2] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, parSet );
  }

};
// Linear in pt and |eta| and sinusoidal in phi
// --------------------------------------------
template <class T>
class scaleFunctionType7 : public scaleFunctionBase<T> {
public:
  scaleFunctionType7() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*std::fabs(eta) +
             parScale[3]*sin(phi))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope", "Phi ampl"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.3, -0.3};
      double thisMaxi[] = {1.1, 0.1, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.3, -0.3};
      double thisMaxi[] = {1.03, 0.02, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Linear in pt and parabolic in |eta|
// -----------------------------------
template <class T>
class scaleFunctionType8 : public scaleFunctionBase<T> {
public:
  scaleFunctionType8() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*std::fabs(eta) +
             parScale[3]*eta*eta)*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00001, 0.000001, 0.0000001, 0.0000001};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope", "Eta quadr"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3, -0.3, -0.3};
      double thisMaxi[] = {1.1, 0.3, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      // double thisMini[] = {0.985, -0.002, -0.005, -0.005};
      // double thisMaxi[] = {1.015, 0.002, 0.005, 0.005};
      double thisMini[] = {0.9, -0.002, -0.01, -0.005};
      double thisMaxi[] = {1.1,  0.002,  0.01,  0.005};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
// Exponential in pt
// -----------------
template <class T>
class scaleFunctionType9 : public scaleFunctionBase<T> {
public:
  scaleFunctionType9() { this->parNum_ = 2; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + exp(parScale[1]*pt))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01};
    TString thisParName[] = {"Pt offset", "Pt expon"};
    double thisMini[] = {0.97, -0.1};
    double thisMaxi[] = {1.03, 0.1};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};
// Parabolic in pt
// ---------------
template <class T>
class scaleFunctionType10 : public scaleFunctionBase<T> {
public:
  scaleFunctionType10() { this->parNum_ = 3; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*pt*pt)*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Pt quadr"};
    double thisMini[] = {0.97, -0.1, -0.001};
    double thisMaxi[] = {1.03, 0.1, 0.001};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};
// Linear in pt, sinusoidal in phi with muon sign
// ----------------------------------------------

// UNCOMMENT to get the original one ------
// template <class T>
// class scaleFunctionType11 : public scaleFunctionBase<T> {
// public:
//   scaleFunctionType11() { this->parNum_ = 4; }
//   virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
//     return( (parScale[0] + parScale[1]*pt +
//              (double)chg*parScale[2]*sin(phi+parScale[3]))*pt );
//   }
//   // Fill the scaleVec with neutral parameters
//   virtual void resetParameters(std::vector<double> * scaleVec) const {
//     scaleVec->push_back(1);
//     for( int i=1; i<this->parNum_; ++i ) {
//       scaleVec->push_back(0);
//     }
//   }
//   virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
//     double thisStep[] = {0.001, 0.001, 0.01, 0.1};
//     TString thisParName[] = {"Pt scale", "Pt slope", "Phi ampl", "Phi phase"};
//     double thisMini[] = {0.97, -0.01, -0.02, -3.1416};
//     double thisMaxi[] = {1.03, 0.01, 0.02, 3.1416};
//     this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
//   }
// };

// VALID FUNCTION 23-01-2010 ---------
template <class T>
class scaleFunctionType11 : public scaleFunctionBase<T> {
public:
  scaleFunctionType11() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt + (double)chg*parScale[4]*eta +
	     (double)chg*parScale[2]*sin(phi+parScale[3]))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.1,0.01};
    TString thisParName[] = {"Pt scale", "Pt slope", "Phi ampl", "Phi phase","Eta coeff"};
    double thisMini[] = {0.9, -0.1, -0.02, -3.1416,-0.2};
    double thisMaxi[] = {1.1, 0.1, 0.02, 3.1416,0.2};

    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};

/// TESTED on 23-01-2011
// template <class T>
// class scaleFunctionType11 : public scaleFunctionBase<T> {
// public:
//   scaleFunctionType11() { this->parNum_ = 8; }
//   virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    
//     if (chg>0){
//       return( (parScale[0] + parScale[1]*pt + parScale[2]*eta +
// 	       parScale[3]*sin(phi+parScale[4]))*pt );
//     }
//     // if (chg<0){
//     return( (parScale[0] + parScale[1]*pt + parScale[5]*eta +
// 	     parScale[6]*sin(phi+parScale[7]))*pt );
//     // }
    
//   }
//   // Fill the scaleVec with neutral parameters
//   virtual void resetParameters(std::vector<double> * scaleVec) const {
//     scaleVec->push_back(1);
//     for( int i=1; i<this->parNum_; ++i ) {
//       scaleVec->push_back(0);
//     }
//   }
//   virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
//     double thisStep[] = {0.001, 0.01, 0.01, 0.1,0.01,0.01, 0.1,0.01};
//     TString thisParName[] = {"Pt scale", "Pt slope", "Eta coeff pos.","Phi ampl pos.", "Phi phase pos.","Eta coeff neg.","Phi ampl neg.", "Phi phase neg."};
//     double thisMini[] = {0.9, -0.1,-0.2, -0.02, -3.1416,-0.2, -0.02, -3.1416};
//     double thisMaxi[] = {1.1, 0.1, 0.2, 0.02, 3.1416, 0.2, 0.02, 3.1416};
//     this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
//   }
// };


// // Linear in pt, parabolic in eta, sinusoidal in phi with muon sign
// ----------------------------------------------------------------
template <class T>
class scaleFunctionType12 : public scaleFunctionBase<T> {
public:
  scaleFunctionType12() { this->parNum_ = 6; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*std::fabs(eta) +
             parScale[3]*eta*eta +
             (double)chg*parScale[4]*sin(phi+parScale[5]))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01, 0.01, 0.1};
    TString thisParName[] = {"Pt scale", "Pt slope", "Eta slope", "Eta quadr", "Phi ampl", "Phi phase"};
    double thisMini[] = {0.97, -0.1, -0.2, -0.2, -0.02, 0.0};
    double thisMaxi[] = {1.03, 0.1, 0.2, 0.2, 0.02, 3.1416};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};





// Linear in pt, parabolic in eta, sinusoidal in phi with muon sign
// ----------------------------------------------------------------
template <class T>
class scaleFunctionType13 : public scaleFunctionBase<T> {
public:
  scaleFunctionType13() { this->parNum_ = 8; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    if (chg>0) {
      return( (parScale[0] + parScale[1]*pt +
               parScale[2]*std::fabs(eta) +
               parScale[3]*eta*eta +
               parScale[4]*sin(phi+parScale[5]))*pt );
    }
    // else {
    return( (parScale[0] + parScale[1]*pt +
             parScale[2]*std::fabs(eta) +
             parScale[3]*eta*eta +
             parScale[6]*sin(phi+parScale[7]))*pt );
    // }
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.1};
    TString thisParName[] = {"Pt scale", "Pt slope", "Eta slope", "Eta quadr", "Phi ampl+", "Phi phase+", "Phi ampl-", "Phi phase-"};
    double thisMini[] = {0.99, -0.01, -0.02, -0.02, -0.02, 0.0, -0.02, 0.0};
    double thisMaxi[] = {1.01, 0.01, 0.02, 0.02, 0.02, 3.1416, 0.02, 3.1416};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};

// linear in pt and cubic in |eta|
// --------------------------------------
template <class T>
class scaleFunctionType14 : public scaleFunctionBase<T> {
public:
  scaleFunctionType14() { this->parNum_ = 10; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
//     for( int i=0; i<10; ++i ) {
//       std::cout << " parScale["<<i<<"] = " << parScale[i];
//     }
//     std::cout << "   newPt = " << ( parScale[0] +
//                                parScale[1]*pt + parScale[2]*pt*pt + parScale[3]*pt*pt*pt +
//                                parScale[4]*std::fabs(eta) + parScale[5]*eta*eta + parScale[6]*std::fabs(eta*eta*eta) +
//                                parScale[7]*eta*eta*eta*eta + parScale[8]*std::fabs(eta*eta*eta*eta*eta) +
//                                parScale[9]*eta*eta*eta*eta*eta*eta )*pt << std::endl;
    return( ( parScale[0] +
              parScale[1]*pt + parScale[2]*pt*pt + parScale[3]*pt*pt*pt +
              parScale[4]*std::fabs(eta) + parScale[5]*eta*eta + parScale[6]*std::fabs(eta*eta*eta) +
              parScale[7]*eta*eta*eta*eta + parScale[8]*std::fabs(eta*eta*eta*eta*eta) +
              parScale[9]*eta*eta*eta*eta*eta*eta )*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001,
                         0.01, 0.01, 0.001,
                         0.01, 0.00001, 0.0000001, 0.00000001, 0.00000001, 0.000000001};
    TString thisParName[] = {"Pt offset",
                             "Pt slope", "Pt quadr", "Pt cubic",
                             "Eta slope", "Eta quadr", "Eta cubic", "Eta quartic", "Eta fifth grade", "Eta sixth grade"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9,
                           -0.3, -0.3, -0.3,
                           -0.3, -0.3, -0.01, -0.001, -0.0001, -0.00001};
      double thisMaxi[] = {1.1,
                           0.3,  0.3,  0.3,
                           0.3,  0.3,  0.01,  0.001,  0.0001, 0.00001};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97,
                           -0.1, -0.001, -0.001,
                           -0.1, -0.1, -0.1, -0.0001, -0.00001, -0.000001};
      double thisMaxi[] = {1.03,
                           0.1, 0.001, 0.001,
                           0.1, 0.1, 0.1, 0.0001, 0.00001, 0.000001};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// linear in pt and cubic in |eta|
// --------------------------------------
template <class T>
class scaleFunctionType15 : public scaleFunctionBase<T> {
public:
  scaleFunctionType15() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    if( pt > parScale[0] ) {
      return( ( parScale[1] + parScale[3]*std::fabs(eta) + parScale[4]*pow(eta,2) )*pt );
    }
    else {
      return( ( parScale[2] + parScale[3]*std::fabs(eta) + parScale[4]*pow(eta,2) )*pt );
    }
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    // For the first, reset to the original pt region separator
    scaleVec->push_back(originalPtRegionSeparator_);
    // The next two are the scale in the two regions
    scaleVec->push_back(1);
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    originalPtRegionSeparator_ = parScale[0];
    double thisStep[] = {0.001,
                         0.01, 0.01,
                         0.01, 0.00001};
    TString thisParName[] = {"Pt offset",
                             "Pt slope 1", "Pt slope 2",
                             "Eta slope", "Eta quadr"};
    if( muonType == 1 ) {
      double thisMini[] = {30.,
                           0.9, 0.9,
                           -0.3, -0.3};
      double thisMaxi[] = {50.,
                           1.1,  1.1,
                           0.3,  0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {30.,
                           0.97, 0.97,
                           -0.1, -0.01};
      double thisMaxi[] = {50.,
                           1.03, 1.03,
                           0.1, 0.01};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
protected:
  double originalPtRegionSeparator_;
};

//
// --------------------------------------
template <class T>
class scaleFunctionType16 : public scaleFunctionBase<T> {
public:
  scaleFunctionType16() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return (parScale[0] + parScale[1]*std::fabs(eta)+ parScale[2]*eta*eta + parScale[3]*pt + parScale[4]/(pt*pt))*pt;
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.0000001, 0.00000001, 0.0000001, 0.00000001, 0.00000001};
    TString thisParName[] = {"offset", "linearEta", "parabEta", "linearPt", "coeffOverPt2"};
    if( muonType == 1 ) {
      double thisMini[] = {30.,
                           0.9, 0.9,
                           -0.3, -0.3};
      double thisMaxi[] = {50.,
                           1.1,  1.1,
                           0.3,  0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.000001, -0.000001, -0.00001, -0.00001};
      double thisMaxi[] = { 1.1,  0.01,   0.001,   0.001, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

template <class T>
class scaleFunctionType17 : public scaleFunctionBase<T> {
public:
  scaleFunctionType17() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return (parScale[0]*std::fabs(eta)+ parScale[1]*eta*eta + pt/(parScale[2]*pt + parScale[3]))*pt;
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00000001, 0.000000001, 0.00000001, 0.00000001};
    TString thisParName[] = {"linearEta", "parabEta", "coeffPt", "coeffOverPt"};
    if( muonType == 1 ) {
      double thisMini[] = {30.,
                           0.9, 0.9,
                           -0.3};
      double thisMaxi[] = {50.,
                           1.1,  1.1,
                           0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.000001, -0.000001, 0.8, -0.001};
      double thisMaxi[] = { 0.01,   0.005,   1.2, 0.001};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
template <class T>
class scaleFunctionType18 : public scaleFunctionBase<T> {
public:
  scaleFunctionType18() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {

    if(std::fabs(eta)<0.2)
      return parScale[0]*pt;
    else if(std::fabs(eta)>0.2 && std::fabs(eta)<1.1)
      return parScale[1]*pt;
    else if(std::fabs(eta)>1.1 && std::fabs(eta)<1.5)
      return parScale[2]*pt;
    else
      return parScale[3]*pt;
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(1);
    }
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00000001, 0.000000001, 0.00000001, 0.00000001};
    TString thisParName[] = {"etaCentr", "barrel", "overlap", "endcaps"};
    if( muonType == 1 ) {
      double thisMini[] = {30.,
                           0.9, 0.9,
                           -0.3};
      double thisMaxi[] = {50.,
                           1.1,  1.1,
                           0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, 0.9, 0.9, 0.9};
      double thisMaxi[] = {1.1,  1.1,   1.1, 1.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


// ---- R.C.Nov.09 ---
// Scale function for Z mass (misalignment STARTUP scenario) corrections
// Linear in pt, sinusoidal in phi (muon-charge dependent) and parabolic in Eta

template <class T>
class scaleFunctionType19 : public scaleFunctionBase<T> {
public:
  scaleFunctionType19() { this->parNum_ = 9; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
  if (chg>0) {
      return( (parScale[0] + parScale[1]*sin(parScale[2]*phi + parScale[3])+ parScale[4]*std::fabs(eta) + parScale[5]*eta*eta )*pt);
  }
  return( (parScale[0] + parScale[6]*sin(parScale[7]*phi + parScale[8])+ parScale[4]*std::fabs(eta) + parScale[5]*eta*eta )*pt );
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const
			     std::vector<int> & parScaleOrder, const int muonType)
  {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    TString thisParName[] = {"Phi offset", "Phi ampl Pos","Phi freq Pos", "Phi phase Pos","Eta slope", "Eta quadr","Phi ampl Neg","Phi freq Neg", "Phi phase Neg"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.1, -0.1, -0.02, -0.02, -0.1, -0.1, -0.1};
      double thisMaxi[] = {1.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.1, 0.1, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.1, -2.0, 0., -0.1, -0.01, -0.1, -2.0, 0. };
      double thisMaxi[] = {1.1, 0.1, 2.0, 6.28, 0.1, 0.01, 0.1, 2.0, 3.14 };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// This function allows to use three different pt functions for three pt ranges
template <class T>
class scaleFunctionType20 : public scaleFunctionBase<T> {
public:
  scaleFunctionType20() { this->parNum_ = 10; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    if( pt < parScale[8] ) {
      return( (parScale[0] + parScale[3] + parScale[6]*std::fabs(eta) + parScale[7]*eta*eta )*pt);
    }
    else if( pt < parScale[9] ) {
      return( (parScale[1] + parScale[4] + parScale[6]*std::fabs(eta) + parScale[7]*eta*eta )*pt);
    }
    return( (parScale[2] + parScale[5] + parScale[6]*std::fabs(eta) + parScale[7]*eta*eta )*pt);
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    scaleVec->push_back(1);
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_-2; ++i ) {
      scaleVec->push_back(0);
    }
    scaleVec->push_back(this->originalTransition1_);
    scaleVec->push_back(this->originalTransition2_);
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const
			     std::vector<int> & parScaleOrder, const int muonType) {

    originalTransition1_ = parScale[8];
    originalTransition2_ = parScale[9];

    double thisStep[] = {0.001, 0.01, 0.01, 0.1, 0.01, 0.01, 0.001, 0.001, 0.1, 0.1};

    TString thisParName[] = {"offset1", "offset2", "offset3", "linearPt1", "linearPt2", "linearPt3",
                             "linearEta", "parabEta", "transition1", "transition2"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, 0.9, 0.9, -1., -1., -1., -1., -1.,   0.,   0.};
      double thisMaxi[] = {1.1, 1.1, 1.1,  1.,  1.,  1.,  1.,  1., 100., 100.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, 0.9, 0.9, -1., -1., -1., -1., -1.,   0.,   0.};
      double thisMaxi[] = {1.1, 1.1, 1.1,  1.,  1.,  1.,  1.,  1., 100., 100.};

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }

 protected:

  double originalTransition1_;
  double originalTransition2_;
};


// Linear in pt and up to cubic in |eta| with possible eta asymmetry: two parabolic branches are used one for eta+ and one for eta-
// --------------------------------------------------------------------------------------------------------------------------------
template <class T>
class scaleFunctionType21 : public scaleFunctionBase<T> {
public:
  scaleFunctionType21() { this->parNum_ = 8; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    double ptPart = parScale[0] + parScale[1]*pt;
    if( eta >= 0 ) {
      return( (ptPart+
	       parScale[2]*eta +
	       parScale[3]*eta*eta +
	       parScale[4]*eta*eta*eta)*pt );
    }
    return( (ptPart +
             parScale[5]*(-eta) +
             parScale[6]*eta*eta +
	     parScale[7]*(-eta*eta*eta))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00001, 0.000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope pos eta", "Eta quadr pos eta", "Eta cubic pos eta", "Eta slope neg eta", "Eta quadr neg eta", "Eta cubic neg eta"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3};
      double thisMaxi[] = {1.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.002, -0.01, -0.005, -0.005, -0.01, -0.005, -0.005};
      double thisMaxi[] = {1.1,  0.002,  0.01,  0.005,  0.005, 0.01,  0.005,  0.005};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


// Function built to correct STARTUP MC
template <class T>
class scaleFunctionType22 : public scaleFunctionBase<T>
{
public:
  scaleFunctionType22() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const
  {
    // Set to 0: use the same parameters for negative and positive muons
    int negChg = 0;
    if( chg > 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + parScale[1]*TMath::Abs(phi)*sin(2*phi + parScale[2]))*pt;
      }
      else {
	return (parScale[0] + parScale[3]*TMath::Abs(phi)*sin(2*phi + parScale[4]))*pt;
      }
    }
    else if( chg < 0 ) {
      if( phi > 0 ) {
	return (parScale[0] - parScale[1+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[2+negChg]))*pt;
      }
      else {
	return (parScale[0] - parScale[3+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[4+negChg]))*pt;
      }
    }
    std::cout << "Error: we should not be here." << std::endl;
    exit(1);
    return 1;
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType)
  {
    double thisStep[] = {0.0001, 0.0001, 0.01, 0.0001, 0.01};
                         // , 0.0001, 0.01, 0.0001, 0.01};
    TString thisParName[] = {"Phi offset",
			     "amplitude pos phi", "phase pos phi",
			     "amplitude neg phi", "phase neg phi"};
			     // "amplitude pos charge pos phi", "phase pos charge pos phi",
			     // "amplitude pos charge neg phi", "phase pos charge neg phi",
			     // "amplitude neg charge pos phi", "phase neg charge pos phi",
			     // "amplitude neg charge neg phi", "phase neg charge neg phi" };
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3, -0.3, -0.3, -0.3};
			   // , -0.3, -0.3, -0.3, -0.3};
      double thisMaxi[] = {1.1,  0.3,  0.3,  0.3,  0.3};
			   // , 0.3,  0.3,  0.3,  0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.1, -3, -0.1, -3};
                           // , -0.1, -3, -0.1, -3};
      double thisMaxi[] = {1.1,   0.1,  3,  0.1,  3};
                           // , 0.1,  3,  0.1,  3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


// Function built to correct STARTUP MC
// Independent parameters for mu+ and mu-
template <class T>
class scaleFunctionType23 : public scaleFunctionBase<T>
{
public:
  scaleFunctionType23() { this->parNum_ = 11; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const
  {
    // Set to 0: use the same parameters for negative and positive muons
    int negChg = 4;
    if( chg > 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[10]*eta*eta + parScale[1]*TMath::Abs(phi)*sin(2*phi + parScale[2]))*pt;
      }
      else {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[10]*eta*eta + parScale[3]*TMath::Abs(phi)*sin(2*phi + parScale[4]))*pt;
      }
    }
    else if( chg < 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[10]*eta*eta - parScale[1+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[2+negChg]))*pt;
      }
      else {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[10]*eta*eta - parScale[3+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[4+negChg]))*pt;
      }
    }
    std::cout << "Error: we should not be here." << std::endl;
    exit(1);
    return 1;
  }
  double etaCorrection(const double & eta) const
  {
    double fabsEta = std::fabs(eta);
    if( fabsEta < 0.2) return -0.00063509;
    else if( fabsEta < 0.4 ) return -0.000585369;
    else if( fabsEta < 0.6 ) return -0.00077363;
    else if( fabsEta < 0.8 ) return -0.000547868;
    else if( fabsEta < 1.0 ) return -0.000954819;
    else if( fabsEta < 1.2 ) return -0.000162139;
    else if( fabsEta < 1.4 ) return 0.0026909;
    else if( fabsEta < 1.6 ) return 0.000684376;
    else if( fabsEta < 1.8 ) return -0.00174534;
    else if( fabsEta < 2.0 ) return -0.00177076;
    else if( fabsEta < 2.2 ) return 0.00117463;
    else if( fabsEta < 2.4 ) return 0.000985705;
    else if( fabsEta < 2.6 ) return 0.00163941;
    return 0.;
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType)
  {
    double thisStep[] = {0.0001,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.001,
			 0.00001};
    TString thisParName[] = {"Phi offset",
			     // "amplitude pos phi", "phase pos phi",
			     // "amplitude neg phi", "phase neg phi"};
			     "amplitude pos charge pos phi", "phase pos charge pos phi",
			     "amplitude pos charge neg phi", "phase pos charge neg phi",
			     "amplitude neg charge pos phi", "phase neg charge pos phi",
			     "amplitude neg charge neg phi", "phase neg charge neg phi",
			     "amplitude of eta correction",
			     "quadratic eta"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.3, -0.3, -0.3, -0.3,
			   -10.,
			   -1.};
      double thisMaxi[] = {1.1,
			   0.3,  0.3,  0.3,  0.3,
			   0.3,  0.3,  0.3,  0.3,
			   10.,
			   1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9,
			   -0.1, -3, -0.1, -3,
                           -0.1, -3, -0.1, -3,
			   -10.,
			   -1.};
      double thisMaxi[] = {1.1,
			   0.1,  3,  0.1,  3,
                           0.1,  3,  0.1,  3,
			   10.,
			   1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
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

    parSet[0]  = ParameterSet( "Phi offset",                   parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "amplitude pos charge pos phi", parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "phase pos charge pos phi",     parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "amplitude pos charge neg phi", parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "phase pos charge neg phi",     parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "amplitude neg charge pos phi", parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "phase neg charge pos phi",     parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "amplitude neg charge neg phi", parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "phase neg charge neg phi",     parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "amplitude of eta correction",  parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "quadratic eta",                parStep[10], parMin[10], parMax[10] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

// Function built to correct STARTUP MC
// Independent parameters for mu+ and mu-
template <class T>
class scaleFunctionType24 : public scaleFunctionBase<T>
{
public:
  scaleFunctionType24() { this->parNum_ = 10; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const
  {
    // Set to 0: use the same parameters for negative and positive muons
    int negChg = 4;
    if( chg > 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[1]*TMath::Abs(phi)*sin(2*phi + parScale[2]))*pt;
      }
      else {
	return (parScale[0] + parScale[9]*etaCorrection(eta) + parScale[3]*TMath::Abs(phi)*sin(2*phi + parScale[4]))*pt;
      }
    }
    else if( chg < 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + parScale[9]*etaCorrection(eta) - parScale[1+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[2+negChg]))*pt;
      }
      else {
	return (parScale[0] + parScale[9]*etaCorrection(eta) - parScale[3+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[4+negChg]))*pt;
      }
    }
    std::cout << "Error: we should not be here." << std::endl;
    exit(1);
    return 1;
  }
  double etaCorrection(const double & eta) const
  {
    if( eta < -2.6 ) return 0.;
    else if( eta < -2.4) return 0.00205594;
    else if( eta < -2.2) return 0.000880532;
    else if( eta < -2.0) return 0.0013714;
    else if( eta < -1.8) return -0.00153122;
    else if( eta < -1.6) return -0.000894437;
    else if( eta < -1.4) return 0.000883338;
    else if( eta < -1.2) return 0.0027599;
    else if( eta < -1.0) return 8.57009e-05;
    else if( eta < -0.8) return -0.00092294;
    else if( eta < -0.6) return -0.000492001;
    else if( eta < -0.4) return -0.000948406;
    else if( eta < -0.2) return -0.000478767;
    else if( eta <  0.0) return -0.0006909;
    else if( eta <  0.2) return -0.000579281;
    else if( eta <  0.4) return -0.000691971;
    else if( eta <  0.6) return -0.000598853;
    else if( eta <  0.8) return -0.000603736;
    else if( eta <  1.0) return -0.000986699;
    else if( eta <  1.2) return -0.00040998;
    else if( eta <  1.4) return 0.00262189;
    else if( eta <  1.6) return 0.000485414;
    else if( eta <  1.8) return -0.00259624;
    else if( eta <  2.0) return -0.00201031;
    else if( eta <  2.2) return 0.000977849;
    else if( eta <  2.5) return 0.00109088;
    else if( eta <  2.6) return 0.00122289;
    return 0.;
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType)
  {
    double thisStep[] = {0.0001,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.001};
    TString thisParName[] = {"Phi offset",
			     // "amplitude pos phi", "phase pos phi",
			     // "amplitude neg phi", "phase neg phi"};
			     "amplitude pos charge pos phi", "phase pos charge pos phi",
			     "amplitude pos charge neg phi", "phase pos charge neg phi",
			     "amplitude neg charge pos phi", "phase neg charge pos phi",
			     "amplitude neg charge neg phi", "phase neg charge neg phi",
			     "amplitude of eta correction"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.3, -0.3, -0.3, -0.3,
			   -10.};
      double thisMaxi[] = {1.1,
			   0.3,  0.3,  0.3,  0.3,
			   0.3,  0.3,  0.3,  0.3,
			   10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9,
			   -0.1, -3, -0.1, -3,
                           -0.1, -3, -0.1, -3,
			   -10.};
      double thisMaxi[] = {1.1,
			   0.1,  3,  0.1,  3,
                           0.1,  3,  0.1,  3,
			   10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


// Function built to correct STARTUP MC
// Analytical description in eta
template <class T>
class scaleFunctionType25 : public scaleFunctionBase<T>
{
public:
  scaleFunctionType25() { this->parNum_ = 19; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const
  {
    // Set to 0: use the same parameters for negative and positive muons
    int negChg = 4;
    if( chg > 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + etaCorrection(eta, parScale) + parScale[1]*TMath::Abs(phi)*sin(2*phi + parScale[2]))*pt;
      }
      else {
	return (parScale[0] + etaCorrection(eta, parScale) + parScale[3]*TMath::Abs(phi)*sin(2*phi + parScale[4]))*pt;
      }
    }
    else if( chg < 0 ) {
      if( phi > 0 ) {
	return (parScale[0] + etaCorrection(eta, parScale) - parScale[1+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[2+negChg]))*pt;
      }
      else {
	return (parScale[0] + etaCorrection(eta, parScale) - parScale[3+negChg]*TMath::Abs(phi)*sin(2*phi + parScale[4+negChg]))*pt;
      }
    }
    std::cout << "Error: we should not be here." << std::endl;
    exit(1);
    return 1;
  }
  double etaCorrection(const double & eta, const T & parScale) const
  {
    if( eta < -2.06 ) return cubicEta(-2.06, parScale, 9);
    else if( eta < -1.06 ) return cubicEta(eta, parScale, 9);
    else if( eta < 1.1 ) return( parScale[13] + parScale[14]*eta );
    else if( eta < 2. ) return cubicEta(eta, parScale, 15);
    return cubicEta(2., parScale, 15);
  }
  double cubicEta(const double & eta, const T & parScale, const int shift ) const
  {
    return( parScale[shift] + parScale[shift+1]*eta + parScale[shift+2]*eta*eta + parScale[shift+3]*eta*eta*eta );
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType)
  {
    double thisStep[] = {0.0001,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.0001, 0.01, 0.0001, 0.01,
			 0.0001, 0.0001, 0.0001, 0.0001,
			 0.0001, 0.0001,
			 0.0001, 0.0001, 0.0001, 0.0001};
    TString thisParName[] = {"Phi offset",
			     // "amplitude pos phi", "phase pos phi",
			     // "amplitude neg phi", "phase neg phi"};
			     "amplitude pos charge pos phi", "phase pos charge pos phi",
			     "amplitude pos charge neg phi", "phase pos charge neg phi",
			     "amplitude neg charge pos phi", "phase neg charge pos phi",
			     "amplitude neg charge neg phi", "phase neg charge neg phi",
			     "etaNeg0", "etaNeg1", "etaNeg2", "etaNeg3",
			     "etaCent0", "etaCent1",
    			     "etaPos0", "etaPos1", "etaPos2", "etaPos3"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.1, -1., -1., -1.,
			   -0.1, -1.,
			   -0.1, -1., -1., -1.};
      double thisMaxi[] = {1.1,
			   0.3,  0.3,  0.3,  0.3,
			   0.3,  0.3,  0.3,  0.3,
			   0.1, 1., 1., 1.,
			   0.1, 1.,
			   0.1, 1., 1., 1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9,
			   -0.1, -3, -0.1, -3,
                           -0.1, -3, -0.1, -3,
			   -0.1, -0.6, -0.5, -0.08,
			   -0.1, -0.001,
			   -0.1, -0.1, -0.4, -0.01};
      double thisMaxi[] = {1.1,
			   0.1,  3,  0.1,  3,
                           0.1,  3,  0.1,  3,
			   0.1, 0.1, 0.1, 0.01,
			   0.1, 0.002,
			   0.1, 0.8, 0.1, 0.2};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// Built for the first 100/nb of J/Psi in data
// It has eta dependent corrections only for |eta| > parScale[6] and separate parabolic corrections for eta > 0 or < 0. 
template <class T>
class scaleFunctionType26 : public scaleFunctionBase<T> {
public:
  scaleFunctionType26() { this->parNum_ = 9; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    double ptPart = parScale[0] + parScale[1]*pt;
    double fabsEta = std::fabs(eta);

    if( fabsEta > parScale[8] ) {
      if( eta > 0 ) {
	return( (ptPart+
		 parScale[2]*eta +
		 parScale[3]*eta*eta)*pt );
      }
      else {
	return( (ptPart+
		 parScale[4]*eta +
		 parScale[5]*eta*eta)*pt );
      }
    }
    return( (ptPart + parScale[6]*fabsEta + parScale[7]*eta*eta)*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00001, 0.000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.00001, 0.000001};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope pos eta", "Eta quadr pos eta", "Eta slope neg eta", "Eta quadr neg eta", "Eta splope barrel", "Eta quadr barrel", "Eta corr region"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 0.3, 0.};
      double thisMaxi[] = {1.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.002, -0.01, -0.005, -0.005, -0.01, -0.01, -0.005, 0.};
      double thisMaxi[] = {1.1,  0.002,  0.01,  0.005,  0.005, 0.01, 0.01, 0.005, 2.4};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};



// Built for the first 100/nb of J/Psi in data
// It has eta dependent corrections only for |eta| > parScale[6] and separate parabolic corrections for eta > 0 or < 0. 
template <class T>
class scaleFunctionType27 : public scaleFunctionBase<T> {
public:
  scaleFunctionType27() { this->parNum_ = 13; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    double ptPart = parScale[0] + parScale[1]*pt;
    double fabsEta = std::fabs(eta);

    if( fabsEta > parScale[12] ) {
      if( eta > 0 ) {
	return( (ptPart+parScale[2]+
		 parScale[3]*(fabsEta - parScale[5]) +
		 parScale[4]*(fabsEta - parScale[5])*(fabsEta - parScale[5]))*pt );
      }
      else {
	return( (ptPart+parScale[6]+
		 parScale[7]*(fabsEta - parScale[9]) +
		 parScale[8]*(fabsEta - parScale[9])*(fabsEta - parScale[9]))*pt );
      }
    }
    return( (ptPart + parScale[10]*fabsEta + parScale[11]*eta*eta)*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.00001, 0.000001,
			 0.000001, 0.0000001, 0.0000001, 0.0000001,
			 0.000001, 0.0000001, 0.0000001, 0.0000001,
			 0.0000001, 0.0000001,
			 0.00001};
    TString thisParName[] = {"Pt offset", "Pt slope",
			     "Eta shift pos eta", "Eta slope pos eta", "Eta quadr pos eta", "Eta center pos eta",
			     "Eta shift neg eta", "Eta slope neg eta", "Eta quadr neg eta", "Eta center neg eta",
			     "Eta splope barrel", "Eta quadr barrel",
			     "Eta corr region"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.3, -0.3, -0.3, -0.3,
			   -0.3, -0.3,
			   0.};
      double thisMaxi[] = {1.1, 0.3,
			   0.3, 0.3, 0.3, 0.3,
			   0.3, 0.3, 0.3, 0.3,
			   0.3, 0.3,
			   0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.9, -0.002,
			   -0.01, -0.01, -0.005, 0.,
			   -0.01, -0.01, -0.005, 0.,
			   -0.01, -0.005,
			   0.};
      double thisMaxi[] = {1.1,  0.002,
			   0.01, 0.01, 0.005, 2.4,
			   0.01, 0.01, 0.005, 2.4,
			   0.01, 0.005,
			   2.4};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

template <class T>
class scaleFunctionType28 : public scaleFunctionBase<T> {
public:
  scaleFunctionType28() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt + (double)chg*parScale[4]*eta +
             (double)chg*parScale[2]*sin(phi+parScale[3]))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const
 std::vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.1,0.01};
    TString thisParName[] = {"Pt scale", "Pt slope", "Phi ampl", "Phi phase","Eta coeff"};
    double thisMini[] = {0.9, -0.1, -0.02, -3.1416,-0.2};
    double thisMaxi[] = {1.1, 0.1, 0.02, 3.1416,0.2};    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};


// As type 11 but with no Pt constant dependence -i.e. no "a" term + explicit dependence from pT 
// of phi coefficient (assuming only alignment effects). Suitable for 4Nov (Zmumu 36/pb) or 
// 22Dec (note: enable eta linear coefficient!)

template <class T>
class scaleFunctionType29 : public scaleFunctionBase<T> {
public:
  scaleFunctionType29() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (1+ parScale[0]*pt +
	     (double)chg*pt*parScale[1]*eta
	     +parScale[2]*eta*eta +
	     pt*(double)chg*parScale[3]*sin(phi+parScale[4]))*pt );
  }
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const {
    //    scaleVec->push_back(1);
    for( int i=0; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
			     TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
			     const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt bias",             0.0001, -0.02,   0.02   );
    parSet[1]  = ParameterSet( "Eta linear coeff",    0.01,   -0.2,    0.2    );
    parSet[2]  = ParameterSet( "Eta parabolic coeff", 0.0001, -0.02,   0.02   );
    parSet[3]  = ParameterSet( "Phi ampl",            0.001,  -0.02,   0.02   );
    parSet[4]  = ParameterSet( "Phi phase",           0.1,    -3.1416, 3.1416 );

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
    parSet[0]  = ParameterSet( "Pt bias",             parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Eta linear coeff",    parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Eta parabolic coeff", parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Phi ampl",            parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi phase",           parStep[4], parMin[4], parMax[4] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

template <class T>
class scaleFunctionType30 : public scaleFunctionBase<T> {
 public:
  scaleFunctionType30() { this->parNum_ = 8; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & par) const
  //Bir alt satirdan Gul'u sonuna kadar ben ekledim bu fonksiyonu.   
  {
    double etaCorr = 0.;
    if( (chg < 0 && eta > par[4]) || (chg > 0 && eta < -par[4]) ) {
      etaCorr = par[1]+par[2]*fabs(fabs(eta)-par[4])+par[3]*(fabs(eta)-par[4])*(fabs(eta)-par[4]);
    }

    //don't forget par[3] = 1.6

    double ptCorr = 0.;
    if( pt < par[7] ) {
      ptCorr = 1+par[5]*(pt - par[7]) + par[6]*(pt - par[7])*(pt - par[7]);
    }

    //don't forget par[6] = 6

    return par[0]*pt/(1 + etaCorr + ptCorr);
  } //Gul araya yaz
  
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {}
  
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
    // name, step, mini, maxi Bunun altindakileri degistirdik. Parametre sayisini 6'ya dusurduk.

    parSet[0]  = ParameterSet( "Overall scale term",           parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta scale term ", parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "multiplies eta", parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "phase pos charge pos phi",     parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "amplitude pos charge neg phi", parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "phase pos charge neg phi",     parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "amplitude neg charge pos phi", parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "phase neg charge pos phi",     parStep[7],  parMin[7],  parMax[7]  );
    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

template <class T>
class scaleFunctionType31 : public scaleFunctionBase<T> {
 public:
  scaleFunctionType31() { this->parNum_ = 8; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & par) const
  //Bir alt satirdan Gul'u sonuna kadar ben ekledim bu fonksiyonu.   
  {
    double etaCorr = 0.;
    if( (chg < 0 && eta > par[4]) || (chg > 0 && eta < -par[4]) ) {
      etaCorr = par[1]+par[2]*fabs(fabs(eta)-par[4])+par[3]*(fabs(eta)-par[4])*(fabs(eta)-par[4]);
    }

    //don't forget par[3] = 1.6

    double ptCorr = 0.;
    if( pt < par[7] ) {
      ptCorr = par[5]*(pt - par[7]) + par[6]*(pt - par[7])*(pt - par[7]);
    }

    //don't forget par[6] = 6

    return par[0]*pt*(1 + etaCorr + ptCorr);
  } //Gul araya yaz
  
  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {}
  
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
    // name, step, mini, maxi Bunun altindakileri degistirdik. Parametre sayisini 6'ya dusurduk.

    parSet[0]  = ParameterSet( "Overall scale term",           parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta scale term ", parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "multiplies eta", parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "phase pos charge pos phi",     parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "amplitude pos charge neg phi", parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "phase pos charge neg phi",     parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "amplitude neg charge pos phi", parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "phase neg charge pos phi",     parStep[7],  parMin[7],  parMax[7]  );
    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

template <class T>
class scaleFunctionType32 : public scaleFunctionBase<T> {
 public:
  scaleFunctionType32() { this->parNum_ = 22; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & par) const
  {
    double fabsEta = fabs(eta);
    double etaCorr = 0.;
    // Eta bins
    if( fabsEta < 0.8 ) etaCorr = par[8];
    else if( fabsEta < 1.2 ) etaCorr = par[9];
    else if( fabsEta < 1.4 ) etaCorr = par[10];
    else if( fabsEta < 1.6 ) etaCorr = par[11];
    else if( fabsEta < 1.8 ) etaCorr = par[12];
    else if( fabsEta < 2.0 ) etaCorr = par[13];
    else if( fabsEta < 2.2 ) etaCorr = par[14];
    else etaCorr = par[15];

    // Charge-asymmetric eta-dependent correction
    if( (chg < 0 && eta > par[4]) || (chg > 0 && eta < -par[4]) ) {
      etaCorr += par[1]+par[2]*fabs(fabsEta-par[4])+par[3]*(fabsEta-par[4])*(fabsEta-par[4]);
    }

    // Phi bins
    double phiCorr = 0.;
    if( phi < -2.0 ) phiCorr = par[16];
    else if( phi < -1.6 ) phiCorr = par[17];
    else if( phi < -1.2 ) phiCorr = par[18];
    else if( phi <  1.2 ) phiCorr = par[19];
    else if( phi <  1.6 ) phiCorr = par[20];
    else if( phi <  2.0 ) phiCorr = par[21];
    // Removed to remove the degeneracy. The overall shift of the eta bins will account for this paramter.
    // else phiCorr = par[22];

    double ptCorr = 0.;
    if( pt < par[7] ) {
      ptCorr = par[5]*(pt - par[7]) + par[6]*(pt - par[7])*(pt - par[7]);
    }

    return par[0]*pt*(1 + etaCorr + ptCorr + phiCorr);
  }

  // Fill the scaleVec with neutral parameters
  virtual void resetParameters(std::vector<double> * scaleVec) const
  {
    //    scaleVec->push_back(1);
    for( int i=1; i<this->parNum_; ++i ) {
      scaleVec->push_back(0);
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const std::vector<int> & parScaleOrder, const int muonType) {}
  
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
    parSet[0]  = ParameterSet( "Overall scale term",           parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "eta scale term ",              parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "multiplies eta",               parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "phase pos charge pos phi",     parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "amplitude pos charge neg phi", parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "phase pos charge neg phi",     parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "amplitude neg charge pos phi", parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "phase neg charge pos phi",     parStep[7],  parMin[7],  parMax[7]  );
    // Eta bins
    parSet[8]  = ParameterSet( "eta bin scale",                parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "eta bin scale",                parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "eta bin scale",                parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "eta bin scale",                parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "eta bin scale",                parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "eta bin scale",                parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "eta bin scale",                parStep[14], parMin[14], parMax[14] );
    parSet[15] = ParameterSet( "eta bin scale",                parStep[15], parMin[15], parMax[15] );
    // Phi bins
    parSet[16] = ParameterSet( "phi bin scale",                parStep[16], parMin[16], parMax[16] );
    parSet[17] = ParameterSet( "phi bin scale",                parStep[17], parMin[17], parMax[17] );
    parSet[18] = ParameterSet( "phi bin scale",                parStep[18], parMin[18], parMax[18] );
    parSet[19] = ParameterSet( "phi bin scale",                parStep[19], parMin[19], parMax[19] );
    parSet[20] = ParameterSet( "phi bin scale",                parStep[20], parMin[20], parMax[20] );
    parSet[21] = ParameterSet( "phi bin scale",                parStep[21], parMin[21], parMax[21] );
    // parSet[22] = ParameterSet( "phi bin scale",                parStep[22], parMin[22], parMax[22] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};


//
// Curvature: (linear in eta + sinusoidal in phi) * global scale
// ------------------------------------------------------------
template <class T>
class scaleFunctionType33 : public scaleFunctionBase<T> {
public:
  scaleFunctionType33() { this->parNum_ = 5; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double curv = (1.+parScale[0])*((double)chg/pt-(parScale[1]*eta+parScale[2]*eta*eta)-parScale[3]*sin(phi+parScale[4]));
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
    double thisStep[] = {0.000001, 0.000001, 0.000001, 0.000001, 0.01};
    TString thisParName[] = {"Curv global scale", "Eta slope", "Eta parabolic" "Phi ampl", "Phi phase"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.3, -0.3, -3.1416};
      double thisMaxi[] = {1.1,  0.1,  0.3,  0.3, 3.1416};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.3, -0.3, -3.1416};
      double thisMaxi[] = {1.03,  0.02,  0.3,  0.3, 3.1416};
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
    parSet[0]  = ParameterSet( "Curv global scale",          parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Eta slope",           parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Eta parabolic",       parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Phi ampl",            parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi phase",           parStep[4], parMin[4], parMax[4] );

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
// Curvature: (constant shift + linear in eta + sinusoidal in phi) * global scale
// ------------------------------------------------------------
template <class T>
class scaleFunctionType34 : public scaleFunctionBase<T> {
public:
  scaleFunctionType34() { this->parNum_ = 6; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double curv = (1.+parScale[0])*((double)chg/pt-(double)chg*parScale[1]-(parScale[2]*eta+parScale[3]*eta*eta)-parScale[4]*sin(phi+parScale[5]));
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
    double thisStep[] = {0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001};
    TString thisParName[] = {"Curv global scale", "Curv offset", "Eta slope", "Eta parabolic" "Phi ampl", "Phi phase"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.1, -0.3, -0.3, -3.1416};
      double thisMaxi[] = {1.1,  0.1, 0.1,  0.3,  0.3, 3.1416};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.02, -0.3, -0.3, -3.1416};
      double thisMaxi[] = {1.03,  0.02,  0.02,  0.3,  0.3, 3.1416};
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
    parSet[0]  = ParameterSet( "Curv scale",          parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Curv offset",         parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Eta slope",           parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Eta parabolic",       parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi ampl",            parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Phi phase",           parStep[5], parMin[5], parMax[5] );

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
// Curvature: (linear in eta + sinusoidal in phi (in 7 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType35 : public scaleFunctionBase<T> {
public:
  scaleFunctionType35() { this->parNum_ = 18; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0);
    if ( eta  < -2.1 ) {
      ampl = parScale[3]; phase = parScale[4];
    } else if ( -2.1 <= eta && eta < -1.7 ) {
      ampl = parScale[5]; phase = parScale[6];
    } else if ( -1.7 <= eta && eta < -0.9 ) {
      ampl = parScale[7]; phase = parScale[8];
    } else if ( -0.9 <= eta && eta <= +0.9 ) {
      ampl = parScale[9]; phase = parScale[10];
    } else if ( +0.9 < eta && eta <= +1.7 ) {
      ampl = parScale[11]; phase = parScale[12];
    } else if ( +1.7 < eta && eta <= 2.1 ) {
      ampl = parScale[13]; phase = parScale[14];
    } else if ( +2.1 < eta ) {
      ampl = parScale[15]; phase = parScale[16];
    }

    double curv = (1.+parScale[0])*((double)chg/pt
				    -(parScale[1]*eta+parScale[2]*eta*eta*eta)
				    -ampl*sin(phi+phase)
				    -0.5*(double)chg*parScale[17]);
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
    double thisStep[] = {0.000001, 0.000001, 0.000001, 0.000001, 0.01, 0.000001, 0.01, 0.000001, 0.01,0.000001, 0.01, 0.000001, 0.01, 0.000001};
    TString thisParName[] = {"Curv global scale"     , "Eta slope", "Eta parabolic", 
			     "Phi ampl eta<-2.1"     , "Phi phase eta<-2.1", 
			     "Phi ampl -2.1<eta<-1.7", "Phi phase -2.1<eta<-1.7", 
			     "Phi ampl -1.7<eta<-0.9", "Phi phase -1.7<eta<-0.9", 
			     "Phi ampl |eta|<0.9"    , "Phi phase |eta|<0.9", 
			     "Phi ampl 0.9<eta<1.7"  , "Phi phase 0.9<eta<1.7", 
			     "Phi ampl 1.7<eta<2.1"  , "Phi phase 1.7<eta<2.1", 
			     "Phi ampl eta>+2.1"     , "Phi phase eta>+2.1",
			     "Charge depend bias"                      };

    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1, -0.3, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.1};
      double thisMaxi[] = {1.1,  0.1,  0.3,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.02, -0.3, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.3, -3.1416, -0.1};
      double thisMaxi[] = {1.03,  0.02,  0.3,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.3,  3.1416,  0.1};
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
    parSet[0]  = ParameterSet( "Curv global scale",      parStep[0],  parMin[0],  parMax[0] );
    parSet[1]  = ParameterSet( "Eta slope",              parStep[1],  parMin[1],  parMax[1] );
    parSet[2]  = ParameterSet( "Eta parabolic",          parStep[2],  parMin[2],  parMax[2] );
    parSet[3]  = ParameterSet( "Phi ampl  eta<-2.1",     parStep[3],  parMin[3],  parMax[3] );
    parSet[4]  = ParameterSet( "Phi phase eta<-2.1",     parStep[4],  parMin[4],  parMax[4] );
    parSet[5]  = ParameterSet( "Phi ampl  -2.1<eta<-1.7",parStep[5],  parMin[5],  parMax[5] );
    parSet[6]  = ParameterSet( "Phi phase -2.1<eta<-1.7",parStep[6],  parMin[6],  parMax[6] );
    parSet[7]  = ParameterSet( "Phi ampl  -1.7<eta<-0.9",parStep[7],  parMin[7],  parMax[7] );
    parSet[8]  = ParameterSet( "Phi phase -1.7<eta<-0.9",parStep[8],  parMin[8],  parMax[8] );
    parSet[9]  = ParameterSet( "Phi ampl  |eta|<0.9",    parStep[9],  parMin[9],  parMax[9] );
    parSet[10] = ParameterSet( "Phi phase |eta|<0.9",    parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "Phi ampl  0.9<eta<1.7",  parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "Phi phase 0.9<eta<1.7",  parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "Phi ampl  1.7<eta<2.1",  parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "Phi phase 1.7<eta<2.1",  parStep[14], parMin[14], parMax[14] );
    parSet[15] = ParameterSet( "Phi ampl  eta>2.1",      parStep[15], parMin[15], parMax[15] );
    parSet[16] = ParameterSet( "Phi phase eta>2.1",      parStep[16], parMin[16], parMax[16] );
    parSet[17] = ParameterSet( "Charge depend bias",     parStep[17], parMin[17], parMax[17] );

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
// Curvature: (SinHyp in eta + sinusoidal in phi (both in 3 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType36 : public scaleFunctionBase<T> {
public:
  scaleFunctionType36() { this->parNum_ = 13; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), twist(0);
    if ( eta  < parScale[11] ) {
      ampl = parScale[1]; phase = parScale[2];
      twist = parScale[3]*TMath::SinH(eta-parScale[11])+parScale[6]*TMath::SinH(parScale[11]) ; 
    } else if ( parScale[11] <= eta && eta <= parScale[12] ) {
      ampl = parScale[4]; phase = parScale[5];
      twist = parScale[6]*TMath::SinH(eta); 
    } else if ( parScale[12] < eta ) {
      ampl = parScale[7]; phase = parScale[8];
      twist = parScale[9]*TMath::SinH(eta-parScale[12])+parScale[6]*TMath::SinH(parScale[12]) ; 
    }
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -0.5*(double)chg*parScale[10]);
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
			 0.000001, 0.01, 0.000001, 
			 0.000001, 0.01, 0.000001,
			 0.000001, 0.01, 0.000001,
			 0.000001, 
			 -0.01, 0.01};
    TString thisParName[] = {"Curv global scale"     ,  
			     "Phi ampl eta<-1.5"     , "Phi phase eta<-1.5" , "Twist eta<-1.5"     ,  
			     "Phi ampl |eta|<1.5"    , "Phi phase |eta|<1.5", "Twist |eta|<1.5"    ,
			     "Phi ampl eta>+1.5"     , "Phi phase eta>+1.5" , "Twist eta>+1.5"     ,
			     "Charge depend bias",
			     "eta neg boundary", "eta pos boundary"};
    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
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
    parSet[0]  = ParameterSet( "Curv global scale",      parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Phi ampl  eta<-1.5",     parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Phi phase eta<-1.5",     parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Twist eta<-1.5",         parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi ampl  |eta|<1.5",    parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Phi phase |eta|<1.5",    parStep[5], parMin[5], parMax[5] );
    parSet[6]  = ParameterSet( "Twist |eta|<1.5",        parStep[6], parMin[6], parMax[6] );
    parSet[7]  = ParameterSet( "Phi ampl  eta>1.5",      parStep[7], parMin[7], parMax[7] );
    parSet[8]  = ParameterSet( "Phi phase eta>1.5",      parStep[8], parMin[8], parMax[8] );
    parSet[9]  = ParameterSet( "Twist eta>1.5" ,         parStep[9], parMin[9], parMax[9] );
    parSet[10] = ParameterSet( "Charge depend bias",     parStep[10],parMin[10],parMax[10] );
    parSet[11] = ParameterSet( "Eta neg boundary",       parStep[11],parMin[11],parMax[11] );
    parSet[12] = ParameterSet( "Eta pos boundary",       parStep[12],parMin[12],parMax[12] );


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
// Curvature: (SinHyp in eta + sinusoidal in phi (both in 3 eta bins)) * global scale  >> Not continuous in eta
// ------------------------------------------------------------
template <class T>
class scaleFunctionType37 : public scaleFunctionBase<T> {
public:
  scaleFunctionType37() { this->parNum_ = 13; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), twist(0);
    if ( eta  < parScale[11] ) {
      ampl = parScale[1]; phase = parScale[2];
      twist = parScale[3]*TMath::SinH(eta); 
    } else if ( parScale[11] <= eta && eta <= parScale[12] ) {
      ampl = parScale[4]; phase = parScale[5];
      twist = parScale[6]*TMath::SinH(eta); 
    } else if ( parScale[12] < eta ) {
      ampl = parScale[7]; phase = parScale[8];
      twist = parScale[9]*TMath::SinH(eta); 
    }
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -0.5*parScale[10]);
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
			 0.000001, 0.01, 0.000001, 
			 0.000001, 0.01, 0.000001,
			 0.000001, 0.01, 0.000001,
			 0.000001, 
			 -0.01, 0.01};
    TString thisParName[] = {"Curv global scale"     ,  
			     "Phi ampl eta<-1.5"     , "Phi phase eta<-1.5" , "Twist eta<-1.5"     ,  
			     "Phi ampl |eta|<1.5"    , "Phi phase |eta|<1.5", "Twist |eta|<1.5"    ,
			     "Phi ampl eta>+1.5"     , "Phi phase eta>+1.5" , "Twist eta>+1.5"     ,
			     "Charge depend bias",
			     "eta neg boundary", "eta pos boundary"};
    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
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
    parSet[0]  = ParameterSet( "Curv global scale",      parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Phi ampl  eta<-1.5",     parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Phi phase eta<-1.5",     parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Twist eta<-1.5",         parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi ampl  |eta|<1.5",    parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Phi phase |eta|<1.5",    parStep[5], parMin[5], parMax[5] );
    parSet[6]  = ParameterSet( "Twist |eta|<1.5",        parStep[6], parMin[6], parMax[6] );
    parSet[7]  = ParameterSet( "Phi ampl  eta>1.5",      parStep[7], parMin[7], parMax[7] );
    parSet[8]  = ParameterSet( "Phi phase eta>1.5",      parStep[8], parMin[8], parMax[8] );
    parSet[9]  = ParameterSet( "Twist eta>1.5" ,         parStep[9], parMin[9], parMax[9] );
    parSet[10] = ParameterSet( "Charge depend bias",     parStep[10],parMin[10],parMax[10] );
    parSet[11] = ParameterSet( "Eta neg boundary",       parStep[11],parMin[11],parMax[11] );
    parSet[12] = ParameterSet( "Eta pos boundary",       parStep[12],parMin[12],parMax[12] );


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
// Curvature: (SinHyp in eta + sinusoidal in phi (both in 3 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType38 : public scaleFunctionBase<T> {
public:
  scaleFunctionType38() { this->parNum_ = 13; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), twist(0);
    if ( eta  < parScale[11] ) {
      ampl = parScale[1]; phase = parScale[2];
      twist = parScale[3]*(eta-parScale[11])+parScale[6]*parScale[11] ; 
    } else if ( parScale[11] <= eta && eta <= parScale[12] ) {
      ampl = parScale[4]; phase = parScale[5];
      twist = parScale[6]*TMath::SinH(eta); 
    } else if ( parScale[12] < eta ) {
      ampl = parScale[7]; phase = parScale[8];
      twist = parScale[9]*(eta-parScale[12])+parScale[6]*parScale[12]; 
    }
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -0.5*parScale[10]);
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
			 0.000001, 0.01, 0.000001, 
			 0.000001, 0.01, 0.000001,
			 0.000001, 0.01, 0.000001,
			 0.000001, 
			 0.01, 0.01};
    TString thisParName[] = {"Curv global scale"     ,  
			     "Phi ampl eta<-1.5"     , "Phi phase eta<-1.5" , "Twist eta<-1.5"     ,  
			     "Phi ampl |eta|<1.5"    , "Phi phase |eta|<1.5", "Twist |eta|<1.5"    ,
			     "Phi ampl eta>+1.5"     , "Phi phase eta>+1.5" , "Twist eta>+1.5"     ,
			     "Charge depend bias",
			     "eta neg boundary", "eta pos boundary"};
    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -3.1416, -0.3, -0.3, -2.6, 0.};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  3.1416,  0.3,  0.3,  0.,  2.6};
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
    parSet[0]  = ParameterSet( "Curv global scale",      parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Phi ampl  eta<-1.5",     parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Phi phase eta<-1.5",     parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Twist eta<-1.5",         parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Phi ampl  |eta|<1.5",    parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Phi phase |eta|<1.5",    parStep[5], parMin[5], parMax[5] );
    parSet[6]  = ParameterSet( "Twist |eta|<1.5",        parStep[6], parMin[6], parMax[6] );
    parSet[7]  = ParameterSet( "Phi ampl  eta>1.5",      parStep[7], parMin[7], parMax[7] );
    parSet[8]  = ParameterSet( "Phi phase eta>1.5",      parStep[8], parMin[8], parMax[8] );
    parSet[9]  = ParameterSet( "Twist eta>1.5" ,         parStep[9], parMin[9], parMax[9] );
    parSet[10] = ParameterSet( "Charge depend bias",     parStep[10],parMin[10],parMax[10] );
    parSet[11] = ParameterSet( "Eta neg boundary",       parStep[11],parMin[11],parMax[11] );
    parSet[12] = ParameterSet( "Eta pos boundary",       parStep[12],parMin[12],parMax[12] );


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
// Curvature: (linear eta (5 eta bins) + sinusoidal in phi (6 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType51 : public scaleFunctionBase<T> {
public:
  scaleFunctionType51() { this->parNum_ = 23; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), twist(0);

// very bwd bin
    if ( eta  < parScale[4] ) {
      ampl = parScale[1]; phase = parScale[2];
      twist = parScale[3]*(eta-parScale[4])+parScale[7]*(parScale[4]-parScale[8])+parScale[11]*parScale[8]; 
// bwd bin
    } else if ( parScale[4] <= eta && eta < parScale[8] ) {
      ampl = parScale[5]; phase = parScale[6];
      twist = parScale[7]*(eta-parScale[8])+parScale[11]*parScale[8] ; 
// barrel bin 1
    } else if ( parScale[8] <= eta && eta < parScale[14] ) {
      if ( eta < 0 ) { 
	ampl = parScale[9]; phase = parScale[10];
      } else if ( eta > 0 ) {
	ampl = parScale[11]; phase = parScale[12];
      }
      twist = parScale[13]*eta; 
// fwd bin
    } else if ( parScale[14] <= eta && eta < parScale[18] ) {
      ampl = parScale[15]; phase = parScale[16];
      twist = parScale[17]*(eta-parScale[14])+parScale[13]*parScale[14]; 
// very fwd bin
    } else if ( parScale[18] < eta ) {
      ampl = parScale[19]; phase = parScale[20];
      twist = parScale[21]*(eta-parScale[18])+parScale[17]*(parScale[18]-parScale[14])+parScale[13]*parScale[14]; 
    }

// apply the correction
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -0.5*parScale[22]);
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
			 0.000001, 0.01, 0.000001, 0.01, 
			 0.000001, 0.01, 0.000001,
			 0.000001}; 

    TString thisParName[] = { "Curv global scale"    	
			     , "Phi ampl eta vbwd"  , "Phi phase eta vbwd" , "Twist eta vbwd",    "vbwd/bwd bndry"
			     , "Phi ampl eta  bwd"  , "Phi phase eta  bwd" , "Twist eta  bwd",    "bwd/bar bndry"   
			     , "Phi ampl eta  bar-"  ,"Phi phase eta  bar-", "Phi ampl eta  bar+","Phi phase eta  bar+" , "Twist eta  bar",   "bar/fwd bndry"  
			     , "Phi ampl eta  fwd"  , "Phi phase eta  fwd" , "Twist eta  fwd",    "fwd/vfwd bndry"  
			     , "Phi ampl eta vfwd"  , "Phi phase eta vfwd" , "Twist eta vfwd"
			     , "Charge depend bias"};

    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3,  0.1};
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
    parSet[9]  = ParameterSet( "Phi ampl   bar-",     parStep[9], parMin[9], parMax[9] );
    parSet[10] = ParameterSet( "Phi phase  bar-",     parStep[10],parMin[10],parMax[10] );
    parSet[11] = ParameterSet( "Phi ampl   bar+",     parStep[11],parMin[11],parMax[11] );
    parSet[12] = ParameterSet( "Phi phase  bar+",     parStep[12],parMin[12],parMax[12] );
    parSet[13] = ParameterSet( "Twist  bar",          parStep[13],parMin[13],parMax[13] );
    parSet[14] = ParameterSet( "bar/fwd bndry",       parStep[14],parMin[14],parMax[14] );
    parSet[15] = ParameterSet( "Phi ampl   fwd",      parStep[15],parMin[15],parMax[15] );
    parSet[16] = ParameterSet( "Phi phase  fwd",      parStep[16],parMin[16],parMax[16] );
    parSet[17] = ParameterSet( "Twist  fwd",          parStep[17],parMin[17],parMax[17] );
    parSet[18] = ParameterSet( "fwd/vfwd bndry",      parStep[18],parMin[18],parMax[18] );
    parSet[19] = ParameterSet( "Phi ampl  vfwd",      parStep[19],parMin[19],parMax[19] );
    parSet[20] = ParameterSet( "Phi phase vfwd",      parStep[20],parMin[20],parMax[20] );
    parSet[21] = ParameterSet( "Twist vfwd",          parStep[21],parMin[21],parMax[21] );
    parSet[22] = ParameterSet( "Charge depend bias",  parStep[22],parMin[22],parMax[22] );

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
// Curvature: (linear eta + sinusoidal in phi (mode 1+2) (both in 5 eta bins)) * global scale 
// ------------------------------------------------------------
template <class T>
class scaleFunctionType52 : public scaleFunctionBase<T> {
public:
  scaleFunctionType52() { this->parNum_ = 31; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {    
    double ampl(0), phase(0), ampl2(0), phase2(0), twist(0);

// very bwd bin
    if ( eta  < parScale[4] ) {
      ampl = parScale[1]; phase = parScale[2];
      ampl2 = parScale[21]; phase2 = parScale[22];
      twist = parScale[3]*(eta-parScale[4])+parScale[7]*(parScale[4]-parScale[8])+parScale[11]*parScale[8]; 
// bwd bin
    } else if ( parScale[4] <= eta && eta < parScale[8] ) {
      ampl = parScale[5]; phase = parScale[6];
      ampl2 = parScale[23]; phase2 = parScale[24];
      twist = parScale[7]*(eta-parScale[8])+parScale[11]*parScale[8] ; 
// barrel bin
    } else if ( parScale[8] <= eta && eta < parScale[12] ) {
      ampl = parScale[9]; phase = parScale[10];
      ampl2 = parScale[25]; phase2 = parScale[26];
      twist = parScale[11]*eta; 
// fwd bin
    } else if ( parScale[12] <= eta && eta < parScale[16] ) {
      ampl = parScale[13]; phase = parScale[14];
      ampl2 = parScale[27]; phase2 = parScale[28];
      twist = parScale[15]*(eta-parScale[12])+parScale[11]*parScale[12]; 
// very fwd bin
    } else if ( parScale[16] < eta ) {
      ampl = parScale[17]; phase = parScale[18];
      ampl2 = parScale[29]; phase2 = parScale[30];
      twist = parScale[19]*(eta-parScale[16])+parScale[15]*(parScale[16]-parScale[12])+parScale[11]*parScale[12]; 
    }

// apply the correction
    double curv = (1.+parScale[0])*((double)chg/pt
				    -twist
				    -ampl*sin(phi+phase)
				    -ampl2*sin(2*phi+phase2)
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
			 0.000001, 0.01, 0.000001, 0.01,  0.000001, 0.01, 0.000001, 0.01, 0.000001, 0.01
			 }; 

    TString thisParName[] = { "Curv global scale"    	
			     , "Phi ampl eta vbwd"  , "Phi phase eta vbwd" , "Twist eta vbwd",   "vbwd/bwd bndry"
			     , "Phi ampl eta  bwd"  , "Phi phase eta  bwd" , "Twist eta  bwd",   "bwd/bar bndry"   
			     , "Phi ampl eta  bar"  , "Phi phase eta  bar" , "Twist eta  bar",   "bar/fwd bndry"  
			     , "Phi ampl eta  fwd"  , "Phi phase eta  fwd" , "Twist eta  fwd",   "fwd/vfwd bndry"  
			     , "Phi ampl eta vfwd"  , "Phi phase eta vfwd" , "Twist eta vfwd"
			     , "Charge depend bias"
			     , "Phi ampl2 eta vbwd"  , "Phi phase2 eta vbwd" 
			     , "Phi ampl2 eta  bwd"  , "Phi phase2 eta  bwd" 
                             , "Phi ampl2 eta  bar"  , "Phi phase2 eta  bar" 
                             , "Phi ampl2 eta  fwd"  , "Phi phase2 eta  fwd" 
                             , "Phi ampl2 eta vfwd"  , "Phi phase2 eta vfwd" 
			       };

    if( muonType == 1 ) {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3,  0.1, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {-0.1, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3, -2.6, -0.3, -3.1416, -0.3,  0.,  -0.3, -3.1416, -0.3,  0.  , -0.3, -3.1416, -0.3, -0.1,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416,-0.3, -3.1416};
      double thisMaxi[] = { 0.1,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  0. ,  0.3,  3.1416,  0.3,  2.6 , 0.3,  3.1416,  0.3,  2.6 ,  0.3,  3.1416,  0.3,  0.1, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416, 0.3,  3.1416};
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
    parSet[21] = ParameterSet( "Phi ampl2  vbwd",      parStep[21],parMin[21],parMax[21] );
    parSet[22] = ParameterSet( "Phi phase2 vbwd",      parStep[22],parMin[22],parMax[22] );
    parSet[23] = ParameterSet( "Phi ampl2   bwd",      parStep[23],parMin[23],parMax[23] );
    parSet[24] = ParameterSet( "Phi phase2  bwd",      parStep[24],parMin[24],parMax[24] );
    parSet[25] = ParameterSet( "Phi ampl2   bar",      parStep[25],parMin[25],parMax[25] );
    parSet[26] = ParameterSet( "Phi phase2  bar",      parStep[26],parMin[26],parMax[26] );
    parSet[27] = ParameterSet( "Phi ampl2   fwd",      parStep[27],parMin[27],parMax[27] );
    parSet[28] = ParameterSet( "Phi phase2  fwd",      parStep[28],parMin[28],parMax[28] );
    parSet[29] = ParameterSet( "Phi ampl2  vfwd",      parStep[29],parMin[29],parMax[29] );
    parSet[30] = ParameterSet( "Phi phase2 vfwd",      parStep[30],parMin[30],parMax[30] );



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

// Resolution Type 1
template <class T>
class resolutionFunctionType1 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType1() { this->parNum_ = 3; }
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return parval[0];
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return parval[1];
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return parval[2];
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = {0.001, 0.001, 0.001};
    TString thisParName[] = {"Pt res. sc.", "Phi res. sc.", "CotgThs res. sc."};
    double thisMini[] = {0., 0., 0.};
    if( muonType == 1 ) {
      double thisMaxi[] = {0.4, 0.4, 0.4};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = {0.01, 0.02, 0.02};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
			     const T & parResol, const std::vector<int> & parResolOrder,
			     const std::vector<double> & parStep,
			     const std::vector<double> & parMin,
			     const std::vector<double> & parMax,
			     const int muonType) {}
};

// Resolution Type 6
template <class T>
class resolutionFunctionType6 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType6() { this->parNum_ = 15; }
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt+parval[2]*pow(pt,2)+parval[3]*pow(pt,3)+parval[4]*pow(pt,4)+parval[5]*std::fabs(eta)+parval[6]*pow(eta,2) );
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7]+parval[8]/pt+parval[9]*std::fabs(eta)+parval[10]*pow(eta,2) );
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[11]+parval[12]/pt+parval[13]*std::fabs(eta)+parval[14]*pow(eta,2) );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.005, 0.0005 ,0.000005 ,0.00000005 ,0.0000000005 ,0.0005 ,0.000005,
                          0.000005 ,0.0005 ,0.00000005 ,0.0000005,
                          0.00005 ,0.0005 ,0.00000005 ,0.000005};
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Pt^2 sc.", "Pt res. Pt^3 sc.", "Pt res. Pt^4 sc.", "Pt res. Eta sc.", "Pt res. Eta^2 sc.",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = { 0.0, -0.01, -0.001, -0.01, -0.001, -0.001, -0.001,
                          0.0, 0.0, -0.001, -0.001,
                          0.0, 0.0, -0.001, -0.001 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1., 1., 1., 1.,
                            0.1, 1., 1., 1.,
                            1., 1., 1., 1. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1,
                            0.01, 0.01, 0.01, 0.01,
                            0.01, 0.01, 0.01, 0.01 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// Resolution Type 7
template <class T>
class resolutionFunctionType7 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType7() { this->parNum_ = 12; }
  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt + parval[2]*std::fabs(eta)+parval[3]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[4]+parval[5]/pt + parval[6]*std::fabs(eta)+parval[7]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*std::fabs(eta)+parval[11]*pow(eta,2) );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.002, 0.00002, 0.000002, 0.0002,
                          0.00002, 0.0002, 0.0000002, 0.000002,
                          0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Eta sc.", "Pt res. Eta^2 sc.",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = { 0.0, -0.01, -0.001, -0.0001,
                          0.0, -0.001, -0.001, -0.00001,
                          0.0, -0.001, -0.0001, -0.0001 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.1, 0.01, 0.01, 0.1,
                            0.01, 0.01, 0.1, 0.01,
                            0.01, 0.01, 0.01, 0.01 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// Resolution Type 8
template <class T>
class resolutionFunctionType8 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType8() { this->parNum_ = 12; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt + parval[2]*etaByPoints(eta, parval[3]) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[4]+parval[5]/pt + parval[6]*std::fabs(eta)+parval[7]*eta*eta );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*std::fabs(eta)+parval[11]*eta*eta );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.0000002, 0.0000001, 0.00001, 0.02,
                          0.00002, 0.0002, 0.0000002, 0.00002,
                          0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Eta sc.", "Pt res. eta border",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = {  -0.03, -0.0000001, 0.1, 0.01,
                           -0.001, 0.002, -0.0001, -0.0001,
                           -0.0001, 0.0005, -0.0001, -0.00001 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.03, 0.1, 1.4, 0.6,
                            0.001, 0.005, 0.00004, 0.0007,
                            0.001, 0.01, -0.0000015, 0.0004 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
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
    else if( 2.4 < eta && eta <= 2.6 ) return border;
    return ( 0. );
  }
};

// Resolution Type 9
template <class T>
class resolutionFunctionType9 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType9() { this->parNum_ = 31; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double ptPart = 0.;

    if( pt < 3 ) ptPart = parval[15];
    else if( pt < 4 ) ptPart = parval[16];
    else if( pt < 5 ) ptPart = parval[17];
    else if( pt < 6 ) ptPart = parval[18];
    else if( pt < 7 ) ptPart = parval[19];
    else if( pt < 8 ) ptPart = parval[20];
    else if( pt < 9 ) ptPart = parval[21];
    else if( pt < 10 ) ptPart = parval[22];

    else ptPart = parval[0] + parval[1]*pt + parval[2]*pt*pt + parval[3]*pt*pt*pt + parval[4]*pt*pt*pt*pt;

    double fabsEta = std::fabs(eta);
    double etaCoeff = parval[5];
    if( fabsEta < 0.1 ) etaCoeff = parval[23];
    else if( fabsEta < 0.2 ) etaCoeff = parval[24];
    else if( fabsEta < 0.3 ) etaCoeff = parval[25];
    else if( fabsEta < 0.4 ) etaCoeff = parval[26];
    else if( fabsEta < 0.5 ) etaCoeff = parval[27];
    else if( fabsEta < 0.6 ) etaCoeff = parval[28];
    else if( fabsEta < 0.7 ) etaCoeff = parval[29];
    else if( fabsEta < 0.8 ) etaCoeff = parval[30];

    return( ptPart + etaCoeff*etaByPoints(eta, parval[6]) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7]+parval[8]/pt + parval[9]*std::fabs(eta)+parval[10]*eta*eta );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[11]+parval[12]/pt + parval[13]*std::fabs(eta)+parval[14]*eta*eta );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.0002, 0.000002, 0.0000002, 0.00000002, 0.000000002, 0.02, 0.02,
                          0.00002, 0.0002, 0.0000002, 0.00002,
                          0.00002, 0.0002, 0.00000002, 0.000002,
                          0.001, 0.001, 0.001, 0.001,
                          0.001, 0.001, 0.001, 0.001,
                          0.001, 0.001, 0.001, 0.001,
                          0.001, 0.001, 0.001, 0.001 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Pt^2 sc.", "Pt res. Pt^3 sc.", "Pt res. Pt^4 sc",
                              "Pt res. Eta sc.", "Pt res. eta border",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc.",
                              "Pt by points sc. 0-3", "Pt by points sc. 3-4", "Pt by points sc. 4-5", "Pt by points sc. 5-6",
                              "Pt by points sc. 6-7", "Pt by points sc. 7-8", "Pt by points sc. 8-9", "Pt by points sc. 9-10",
                              "Eta scale for eta < 0.1", "Eta scale for eta < 0.2", "Eta scale for eta < 0.3", "Eta scale for eta < 0.4",
                              "Eta scale for eta < 0.5", "Eta scale for eta < 0.6", "Eta scale for eta < 0.7", "Eta scale for eta < 0.8" };
    double thisMini[] = {  -0.1, -0.001, -0.001, -0.001, -0.001, 0.001, 0.0001,
                           -0.001, 0.002, -0.0001, -0.0001,
                           -0.0001, 0.0005, -0.0001, -0.00001,
                           -1., -1., -1., -1.,
                           -1., -1., -1., -1.,
                           -1., -1., -1., -1.,
                           -1., -1., -1., -1. };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            3., 3., 3., 3. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.1, 0.001, 0.001, 0.001, 0.001, 1.5, 1.,
                            0.001, 0.005, 0.00004, 0.0007,
                            0.001, 0.01, -0.0000015, 0.0004,
                            3., 3., 3., 3.,
                            3., 3., 3., 3.,
                            3., 3., 3., 3.,
                            3., 3., 3., 3. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
protected:
  /**
   * This is the pt vs eta resolution by points. It uses std::fabs(eta) assuming symmetry.<br>
   * The values are derived from Upsilon2S redigi events.
   */
  double etaByPoints(const double & inEta, const double & border) {
    Double_t eta = std::fabs(inEta);
    // More detailed taken from Upsilon2S
//     if( 0.0 < eta && eta <= 0.1 ) return( (0.006496598 + 0.006713836)/2 );
//     else if( 0.1 < eta && eta <= 0.2 ) return( (0.006724315 + 0.006787474)/2 );
//     else if( 0.2 < eta && eta <= 0.3 ) return( (0.007284029 + 0.007293643)/2 );
//     else if( 0.3 < eta && eta <= 0.4 ) return( (0.008138282 + 0.008187387)/2 );
//     else if( 0.4 < eta && eta <= 0.5 ) return( (0.008174111 + 0.008030496)/2 );
//     else if( 0.5 < eta && eta <= 0.6 ) return( (0.008126558 + 0.008100443)/2 );
//     else if( 0.6 < eta && eta <= 0.7 ) return( (0.008602069 + 0.008626195)/2 );
//     else if( 0.7 < eta && eta <= 0.8 ) return( (0.009187699 + 0.009090244)/2 );
//     else if( 0.8 < eta && eta <= 0.9 ) return( (0.009835283 + 0.009875661)/2 );
//     else if( 0.9 < eta && eta <= 1.0 ) return( (0.01156847 + 0.011774)/2);
//     else if( 1.0 < eta && eta <= 1.1 ) return( (0.01319311 + 0.01312528)/2 );
//     else if( 1.1 < eta && eta <= 1.2 ) return( (0.01392963 + 0.01413793)/2 );
//     else if( 1.2 < eta && eta <= 1.3 ) return( (0.01430238 + 0.01385749)/2 );
//     else if( 1.3 < eta && eta <= 1.4 ) return( (0.01409375 + 0.01450355)/2 );
//     else if( 1.4 < eta && eta <= 1.5 ) return( (0.01395235 + 0.01419122)/2 );
//     else if( 1.5 < eta && eta <= 1.6 ) return( (0.01384032 + 0.01354162)/2 );
//     else if( 1.6 < eta && eta <= 1.7 ) return( (0.01325593 + 0.01302663)/2 );
//     else if( 1.7 < eta && eta <= 1.8 ) return( (0.01365382 + 0.01361993)/2 );
//     else if( 1.8 < eta && eta <= 1.9 ) return( (0.01516075 + 0.01514115)/2 );
//     else if( 1.9 < eta && eta <= 2.0 ) return( (0.01587837 + 0.01561742)/2 );
//     else if( 2.0 < eta && eta <= 2.1 ) return( (0.01696865 + 0.01760318)/2 );
//     else if( 2.1 < eta && eta <= 2.2 ) return( (0.01835451 + 0.01887852)/2 );
//     else if( 2.2 < eta && eta <= 2.3 ) return( (0.02116863 + 0.02254953)/2 );
//     else if( 2.3 < eta && eta <= 2.4 ) return( (0.0224906 + 0.02158211)/2 );

    // Less detailed
//     if( 0.0 < eta && eta <= 0.2 ) return( (0.006496598 + 0.006713836 + 0.006724315 + 0.006787474)/4 );
//     else if( 0.2 < eta && eta <= 0.4 ) return( (0.007284029 + 0.007293643 + 0.008138282 + 0.008187387)/4 );
//     else if( 0.4 < eta && eta <= 0.6 ) return( (0.008174111 + 0.008030496 + 0.008126558 + 0.008100443)/4 );
//     else if( 0.6 < eta && eta <= 0.8 ) return( (0.008602069 + 0.008626195 + 0.009187699 + 0.009090244)/4 );
//     else if( 0.8 < eta && eta <= 1.0 ) return( (0.009835283 + 0.009875661 + 0.01156847 + 0.011774)/4 );
//     else if( 1.0 < eta && eta <= 1.2 ) return( (0.01319311 + 0.01312528 + 0.01392963 + 0.01413793)/4 );
//     else if( 1.2 < eta && eta <= 1.4 ) return( (0.01430238 + 0.01385749 + 0.01409375 + 0.01450355)/4 );
//     else if( 1.4 < eta && eta <= 1.6 ) return( (0.01395235 + 0.01419122 + 0.01384032 + 0.01354162)/4 );
//     else if( 1.6 < eta && eta <= 1.8 ) return( (0.01325593 + 0.01302663 + 0.01365382 + 0.01361993)/4 );
//     else if( 1.8 < eta && eta <= 2.0 ) return( (0.01516075 + 0.01514115 + 0.01587837 + 0.01561742)/4 );
//     else if( 2.0 < eta && eta <= 2.2 ) return( (0.01696865 + 0.01760318 + 0.01835451 + 0.01887852)/4 );
//     // else if( 2.2 < eta && eta <= 2.4 ) return( (0.02116863 + 0.02254953 + 0.0224906 + 0.02158211)/4 );

//     return ( border );

    // From MuonGun
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
    else if( 2.4 < eta && eta <= 2.6 ) return border;
    return ( 0. );

  }
};

/// This is resolution function where sigmaPt/Pt is described by f(Pt) = polynomial(4th grade) and f(Eta) = polynomial(8th grade).
// Resolution Type 10
template <class T>
class resolutionFunctionType10 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType10() { this->parNum_ = 21; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    return( parval[0] + parval[1]*pt + parval[2]*pt*pt + parval[3]*pt*pt*pt + parval[4]*pt*pt*pt*pt
            + parval[5]*fabsEta + parval[6]*fabsEta*fabsEta + parval[7]*pow(fabsEta,3) + parval[8]*pow(fabsEta,4)
            + parval[9]*pow(fabsEta,5) + parval[10]*pow(fabsEta,6) + parval[11]*pow(fabsEta,7) + parval[12]*pow(fabsEta,8) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[13]+parval[14]/pt + parval[15]*std::fabs(eta)+parval[16]*eta*eta );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[17]+parval[18]/pt + parval[19]*std::fabs(eta)+parval[20]*eta*eta );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.0002,  0.000002, 0.0000002, 0.00000002, 0.000000002,
                          0.02,    0.02,     0.002,     0.0002,
                          0.00002, 0.000002, 0.0000002, 0.00000002,
                          0.00002, 0.0002, 0.0000002, 0.00002,
                          0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Pt^2 sc.", "Pt res. Pt^3 sc.", "Pt res. Pt^4 sc",
                              "Pt res. Eta sc.", "Pt res. Eta^2 sc." ,"Pt res. Eta^3 sc.", "Pt res. Eta^4 sc.",
                              "Pt res. Eta^5 sc.", "Pt res. Eta^6 sc.", "Pt res. Eta^7 sc.", "Pt res. Eta^8 sc.",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = {  -0.1, -0.001, -0.001, -0.001, -0.001,
                           -2., -1., -0.1, -0.1,
                           -0.1, -0.1, -0.1, -0.1,
                           -0.001, 0.002, -0.0001, -0.0001,
                           -0.0001, 0.0005, -0.0001, -0.00001,
                           0.};
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.1, 0.001, 0.001, 0.001, 0.001,
                            2., 1., 0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1, 0.1, 0.1,
                            0.001, 0.005, 0.00004, 0.0007,
                            0.001, 0.01, -0.0000015, 0.0004 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

/// This is resolution function where sigmaPt/Pt is described by f(Pt) = a + b/pt + pt/(pt+c)and f(Eta) = 2 parabolas for fabsEta<1.2 or fabsEta>1.2
// Resolution Type 11
template <class T>
class resolutionFunctionType11 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType11() { this->parNum_ = 8; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    if(fabsEta<1.2)
      return (parval[0]+ parval[2]*1./pt + pt/(pt+parval[3]) + parval[4]*fabsEta + parval[5]*eta*eta);
    else
      return (parval[1]+ parval[2]*1./pt + pt/(pt+parval[3]) + parval[6]*std::fabs((fabsEta-1.6)) + parval[7]*(fabsEta-1.6)*(fabsEta-1.6));
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.004 );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.00001, 0.00001, 0.0000001, 0.00000001,
                          0.00000001, 0.00000001, 0.00000001, 0.00000001 };
    TString thisParName[] = { "offsetEtaCentral", "offsetEtaHigh", "coeffOverPt", "coeffHighPt", "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh" };
    double thisMini[] = { -1.1,  -1.1,   -0.1,           -0.1  ,     0.0001,      0.0005,     0.0005,     0.001};
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { -0.8,   -0.8,   -0.001,     -0.001 ,     0.005,        0.05,      0.05,    0.05};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

/**
 * Same as type11 but with free parameters for transition region and center of second parabola.
 * It also imposes continuity of the two fuctions.
 * Adds also two additional parameters to allow a linear and a quadratic dependence from pt (the
 * resolution vs Pt has been seen to grow with Pt for misaligned samples.
 * Also replaced the sigmaCotgTh and sigmaPhi with those from type8.
 */
// Resolution Type 12
template <class T>
class resolutionFunctionType12 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType12() { this->parNum_ = 15; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);

    double ptPart = parval[2]*1./pt + pt/(pt+parval[3]) + pt*parval[9] + pt*pt*parval[10];

    if(fabsEta<parval[0]) {
      // To impose continuity we require that the parval[0] of type11 is
      double par = parval[1] + parval[6]*std::fabs((parval[0]-parval[8])) + parval[7]*(parval[0]-parval[8])*(parval[0]-parval[8]) - (parval[4]*parval[0] + parval[5]*parval[0]*parval[0]);
      return( par + ptPart + parval[4]*fabsEta + parval[5]*eta*eta );
    }
    else {
      return( parval[1]+ ptPart + parval[6]*std::fabs((fabsEta-parval[8])) + parval[7]*(fabsEta-parval[8])*(fabsEta-parval[8]) );
    }
  }

  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[11]+parval[12]/pt + parval[13]*std::fabs(eta)+parval[14]*eta*eta );
  }

  // // 1/pt in pt and quadratic in eta
  // virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
  //   return( parval[15]+parval[16]/pt + parval[17]*std::fabs(eta)+parval[18]*eta*eta );
  // }

  // constant sigmaCotgTh
  // virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
  //   return( 0.004 );
  // }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.001, 0.00001, 0.0000001, 0.00000001,
                          0.00000001, 0.00000001, 0.00000001, 0.00000001,
                          0.001, 0.0001, 0.000001,
			  0.00002, 0.0002, 0.0000002, 0.00002 };
			  // 0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "etaTransition", "offsetEtaHigh", "coeffOverPt", "coeffHighPt",
                              "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh",
                              "secondParabolaCenter", "linearPt", "quadraticPt",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc." };
			      // "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = { 0.8, -1.1, 0., -1.1,
                          0., 0.0005, 0.0005, 0.001,
                          1.4, 0., 0.,
                          // -0.001, 0.002, -0.0001, -0.0001 };
                          -0.1, 0., -0.1, -0.1 };
			  // -0.0001, 0.0005, -0.0001, -0.00001 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1.,
                            1., 1., 1., 1. };
			    // 1., 1., 1., 1. };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 1.8, 0.8, 0.1, 0.1,
                            0.005, 0.05, 0.05, 0.05,
                            2.4, 2.0, 2.0,
                            // 0.001, 0.005, 0.00004, 0.0007 };
			    0.1, 0.05, 0.1, 0.1 };
			    // 0.001, 0.01, -0.0000015, 0.0004 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi

    parSet[0]  = ParameterSet( "etaTransition",            parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "offsetEtaHigh",            parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "coeffOverPt",              parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "coeffHighPt",              parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "linearEtaCentral",         parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "parabEtaCentral",          parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "linearEtaHigh",            parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "parabEtaHigh",             parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "secondParabolaCenter",     parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "linearPt",                 parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "quadraticPt",              parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "Cth res. sc.",             parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "Cth res. 1/Pt sc.",        parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "Cth res. Eta sc.",         parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "Cth res. Eta^2 sc.",       parStep[14], parMin[14], parMax[14] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

/**
 * Same as type12 but introduces an additional parabola with starting parameters
 * putting it in 0.9-1.2 in eta. This is done to take into account the transition
 * to TOB at eta = 0.9.
 */
// Resolution Type 13
template <class T>
class resolutionFunctionType13 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType13() { this->parNum_ = 15; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);

    double ptPart = parval[2]*1./pt + pt/(pt+parval[3]) + pt*parval[13] + pt*pt*parval[14];

    if(fabsEta<parval[9]) {
      // To impose continuity we require that
      double par2 = parval[1] + parval[6]*std::fabs(parval[9] - parval[8]) + parval[7]*pow(parval[9] - parval[8], 2) - parval[10]*std::fabs(parval[9] - parval[12]) - parval[11]*pow(parval[9] - parval[12], 2);
      if( fabsEta<parval[0]) {
        double par1 = par2 + parval[10]*std::fabs(parval[0] - parval[12]) + parval[11]*pow(parval[0] - parval[12], 2) - parval[4]*parval[0] - parval[5]*parval[0]*parval[0];
        return( par1 + ptPart + parval[4]*fabsEta + parval[5]*eta*eta );
        // return( parval[15] + ptPart + parval[4]*fabsEta + parval[5]*eta*eta );
      }
      // return( par2+ ptPart + parval[10]*std::fabs((fabsEta-parval[12])) + parval[11]*(fabsEta-parval[12])*(fabsEta-parval[12]) );
      return( par2+ ptPart + parval[10]*std::fabs((fabsEta-parval[12])) + parval[11]*(fabsEta-parval[12])*(fabsEta-parval[12]) );
    }
    return( parval[1]+ ptPart + parval[6]*std::fabs((fabsEta-parval[8])) + parval[7]*(fabsEta-parval[8])*(fabsEta-parval[8]) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.004 );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.001, 0.00001, 0.0000001, 0.00000001,
                          0.00000001, 0.00000001, 0.00000001, 0.00000001,
                          0.00000001, 0.00000001, 0.00000001, 0.00000001,
                          0.001, 0.0001, 0.000001 };
//                          0.001 };
    TString thisParName[] = { "etaTransition1", "offsetEtaHigh", "coeffOverPt", "coeffHighPt",
                              "linearEtaCentral", "parabEtaCentral", "linearEtaHighEta", "parabEtaHighEta",
                              "centerEtaHighEta", "etaTransition2", "linearEtaSecondEta", "parabEtaSecondEta",
                              "centerEtaSecondEta", "linearPt", "quadraticPt" };
//                              "scale2" };
    double thisMini[] = { 0.8, -1.1, -1.1, -1.1,
                          -1., 0.0005, 0.0005, 0.001,
                          0.8, 1.1, 0.0005, 0.001,
                          0.0, -1.0, -1.0 };
//                          -1.0 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1., 1.,
                            1., 1., 1. };
//                            1. };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 1.0, -0.8, 0.1, 0.1,
                            0.005, 0.05, 0.05, 0.05,
                            2.4, 1.4, 0.05, 0.05,
                            1.8, 2.0, 2.0 };
//                            2.0 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

template <class T>
class resolutionFunctionType14 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType14() { this->parNum_ = 7; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    
    if(fabsEta<1.3)
      return (parval[0] + parval[6]*pt + parval[2]*fabsEta + parval[3]*eta*eta);
    //if((1.3<eta<1.6) || (-1.6<eta<-1.3) )
    //  return (parval[7] + parval[8]*fabsEta);
    else
      return (parval[1]+ parval[4]*std::fabs((fabsEta-1.6)) + parval[5]*(fabsEta-1.6)*(fabsEta-1.6));

//     if(fabsEta<1.35)
//       return (parval[0] + parval[6]*pt + parval[2]*fabsEta + parval[3]*eta*eta);
//     else
//       return (parval[0] + parval[6]*pt + parval[2]*fabsEta + parval[3]*eta*eta+ parval[4]*std::fabs((fabsEta-1.35)) + parval[5]*(fabsEta-1.35)*(fabsEta-1.35));

   }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.004 );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.00001, 0.00001, 0.0000001, 0.00000001, 0.00000001, 0.00000001, 0.00000001 , 0.00002};
    TString thisParName[] = { "offsetEtaCentral", "offsetEtaHigh", "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh", "linearPt"};
			      //, "offsetEtaOverlap","linaerEtaOverlap"};
    double thisMini[] = {           0.0,               0.0,            -0.01,               0.,                -0.05,     0. , 0.};
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1., 1.,
                            1., 1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = {           0.01,                  0.02,             0.01,             0.05,           0.05,      0.1, 0.01};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
protected:
};

// par6,par7,par10
// Resolution Type 15. For misaligned data. Linear in pt, parabolic in eta, regions separated: barrl+overlap, right endcap, left endcap.
template <class T>
class resolutionFunctionType15 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType15() { this->parNum_ = 7; }
  // linear in pt and parabolic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    double ptPart = pt*parval[0];

    if(fabsEta<=0.6) 
      return (ptPart);
    else if(fabsEta>0.6 && fabsEta<=1.3) {//eta in barrel + overlap
      double par = - parval[1]*0.6*0.6;
      return( par + ptPart + parval[1]*eta*eta );
    }
    else if (eta>1.3){//eta in right endcap
      double par =  parval[1]*1.3*1.3 - (parval[2]*(1.3-parval[3])*(1.3-parval[3]));
      return( par + ptPart + parval[2]*(fabsEta-parval[3])*(fabsEta-parval[3]) );
    }
    else{//eta in left endcap
      double par = parval[1]*1.3*1.3 - (parval[4]*(1.3-parval[5]) + parval[6]*(1.3-parval[5])*(1.3-parval[5]));
      return( par + ptPart + parval[4]*std::fabs((fabsEta-parval[5])) + parval[6]*(fabsEta-parval[5])*(fabsEta-parval[5]) );
    }
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    //    if (std::fabs(eta)<=1.2) return(0.000949148 );
    //    else if(eta<-1.2) return(-0.00645458 + -0.00579458*eta);
    //    else return(-0.00306283 + 0.00346136*eta);
    return 0;

  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    //return( 0.000211 + 0.00001*std::fabs(eta) + 0.0000789*eta*eta);
    return 0;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.000001, 0.0000001,
			  0.000001, 0.001, 
			  0.00000001, 0.001, 0.0001};
    TString thisParName[] = { "linearPt", "parabEtaCentral", 
			      "parabolicEtaRight", "rightParabCenter",
			      "linearEtaLeft", "leftParabCenter", "parabolicEtaLeft" };
    double thisMini[] = { 0.00001, 0.0001,
                          0.005, -5,
			  -0.00006, 0.1, 0.002
                        };

    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 
			    1., 1., 
                            1., 1. ,1.,
      };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.0005, 0.05,
			    0.15,  1.99,
                            0.005, 1.99, 0.15, 
      };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

//Divides the resolution vs pt in three regions, barrel, overlap and endcaps. It gives excellent results for at least 15K Z
template <class T>
class resolutionFunctionType17 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType17() { this->parNum_ = 18; }
  // linear in pt and parabolic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    double ptPartBar =  parval[0] + pt*parval[2];
    double ptPartOvlap = parval[16] + pt*parval[17];
    double ptPartEndc = parval[14] + pt*parval[15];
    if(fabsEta<=0.9) {//eta in barrel
      return( ptPartBar + parval[3] + parval[4]*fabsEta);
    }
    else if( (eta > 0.9 && eta <= 1.4) || (eta < -0.9 && eta > -1.4)){ //eta in overlap
      return( ptPartOvlap + parval[3] + parval[4]*eta + parval[5]*eta*eta);
    }
    else if (eta>1.4){//eta in right endcap
      double par = parval[3] + parval[4]*1.4 + parval[5]*1.4*1.4 - (parval[6] + parval[7]*(1.4-parval[8]) + parval[9]*(1.4-parval[8])*(1.4-parval[8]));
      return( par + ptPartEndc + parval[6] + parval[7]*std::fabs((fabsEta-parval[8])) + parval[9]*(fabsEta-parval[8])*(fabsEta-parval[8]) );
    }
    else {//eta in left endcap
      double par =  parval[3] + parval[4]*1.4 + parval[5]*1.4*1.4 - (parval[10] + parval[11]*(1.4-parval[12]) + parval[13]*(1.4-parval[12])*(1.4-parval[12]));
      return( par + ptPartEndc + parval[10] + parval[11]*std::fabs((fabsEta-parval[12])) + parval[13]*(fabsEta-parval[12])*(fabsEta-parval[12]) );
    }
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.004 );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.0001, 0.00001, 0.00001,
			  0.001, 0.0001, 0.0000001,
                          0.01, 0.001, 0.01, 0.001,
			  0.01, 0.001, 0.01, 0.001,
			  0.01, 0.00001, 0.01, 0.00001};
    TString thisParName[] = { "offsetPt", "hyperbolicPt", "linearPt",
                              "offsetEtaCentral", "linaerEtaCentral", "parabEtaCentral",
			      "offsetEtaEndcapRight", "linearEtaRight", "rightParabCenter", "parabolicEtaRight",
                              "offsetEtaEndcapLeft", "linearEtaLeft", "leftParabCenter", "parabolicEtaLeft",
			      "offsetPtEndc", "linearPtEndc", "offsetPtOvlap", "linearPtOvlap"
                            };
    double thisMini[] = { -0.15, -0.001, 0.00005,
                          -0.05, -0.1, 0.0,
                          -0.6, -0.0009, 0., 0.0005,
			  -0.6, -0.1, 1., 0.01,
			  -1.5, 0.00005, -1.5, 0.00005
                        };

    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1.,
			    1., 1., 1.,
                            1., 1. ,1., 1.,
			    1., 1., 1., 1.,
			    1., 1., 1., 1.,
                          };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.15, 0.8, 0.005,
			    0.05, 0.1, 0.08,
                            0.9, 0.5, 1.99, 0.15,
			    0.9, 0.5, 1.99, 0.15,
			    1.1, 0.005, 1.1, 0.005
      };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

template <class T>
class resolutionFunctionType18 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType18() { this->parNum_ = 14; }
  // linear in pt and parabolic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    double ptPart =  parval[0] + parval[1]*1./pt + pt*parval[2];

    if(fabsEta<=0.6)
      return( ptPart + parval[3]);
    else if((eta>0.6 && eta<=1.3) || (eta>=-1.3 && eta<-0.6)) {//eta in barrel + overlap
      double par = parval[3] - 0.6*parval[4] - 0.6*0.6*parval[5];
      return( ptPart + par + parval[4]*fabsEta + parval[5]*eta*eta );
    }
    else if (eta>1.3){//eta in right endcap
      double par = parval[3] - 0.6*parval[4] - 0.6*0.6*parval[5] + parval[4]*1.3 + parval[5]*1.3*1.3 - (parval[6] + parval[7]*std::fabs(1.3-parval[8]) + parval[9]*(1.3-parval[8])*(1.3-parval[8]));
      return( par +  ptPart + parval[6] + parval[7]*std::fabs((fabsEta-parval[8])) + parval[9]*(fabsEta-parval[8])*(fabsEta-parval[8]) );
    }
    else{//eta in left endcap
      double par = parval[3] - 0.6*parval[4] - 0.6*0.6*parval[5] + parval[4]*1.3 + parval[5]*1.3*1.3 - (parval[10] + parval[11]*std::fabs(1.3-parval[12]) + parval[13]*(1.3-parval[12])*(1.3-parval[12]));
      return( par + ptPart + parval[10] + parval[11]*std::fabs((fabsEta-parval[12])) + parval[13]*(fabsEta-parval[12])*(fabsEta-parval[12]) );
    }
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.004 );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.01, 0.0001, 0.00001,
			  0.001, 0.0001, 0.000001,
                          0.01, 0.001, 0.01, 0.001,
			  0.01, 0.001, 0.01, 0.001};
    TString thisParName[] = { "offsetPt", "hyperbolicPt", "linearPt",
                              "offsetEtaCentral", "linaerEtaCentral", "parabEtaCentral",
			      "offsetEtaEndcapRight", "linearEtaRight", "rightParabCenter", "parabolicEtaRight",
                              "offsetEtaEndcapLeft", "linearEtaLeft", "leftParabCenter", "parabolicEtaLeft" };
    double thisMini[] = { -1.5, -0.001, 0.00005,
                          -0.05, -0.1, 0.0,
                          -0.6, -0.0009, 0., 0.0005,
			  -0.6, -0.1, 1., 0.01
                        };

    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1.,
			    1., 1., 1.,
                            1., 1. ,1., 1.,
			    1., 1., 1., 1.
                          };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 1.1, 0.8, 0.005,
			    0.05, 0.1, 0.08,
                            0.9, 0.5, 1.99, 0.15,
			    0.9, 0.5, 1.99, 0.15
      };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

// Resolution Type 19
// Same as type 8, but the sigmaPhi and sigmaCotgTh are not free. This way the function results as having less parameters.
// This was done to verify if fixed parameters have an influence in the computation of errors by minuit.
template <class T>
class resolutionFunctionType19 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType19() { this->parNum_ = 4; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double value = parval[0]+parval[1]*pt + parval[2]*etaByPoints(eta, parval[3]);
    if( value != value ) {
      std::cout << "parval[0] = " << parval[0] << ", parval[1]*"<<pt<<" = " << parval[1]*pt << "parval[2] = " << parval[1] << ",etaByPoints("<<eta<<", "<<parval[3]<<") = " << etaByPoints(eta, parval[3]) << std::endl;
    }
    return( parval[0] + parval[1]*pt + parval[2]*etaByPoints(eta, parval[3]) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( 0.00043 + 0.0041/pt + (2.8e-06)*std::fabs(eta) + (7.7e-05)*eta*eta );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.00011 + 0.0018/pt - (9.4e-07)*std::fabs(eta) + (2.2e-05)*eta*eta );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    double thisStep[] = { 0.0000002, 0.0000001, 0.00001, 0.001 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Eta sc.", "Pt res. eta border"};
    double thisMini[] = {  -0.03, -0.0000001, 0.001, 0.01};
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.03, 0.1, 2., 0.6};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
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
    else if( 2.4 < eta && eta <= 2.6 ) return border;
    return ( 0. );
  }
};

template <class T>
class resolutionFunctionType20 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType20() { this->parNum_ = 9; }
  // linear in pt and by points in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);

    if(fabsEta<parval[0]) {
      // To impose continuity we require that the parval[0] of type11 is
      double par = parval[1] + parval[4]*std::fabs((parval[0]-parval[6])) + parval[5]*(parval[0]-parval[6])*(parval[0]-parval[6]) - (parval[2]*parval[0] + parval[3]*parval[0]*parval[0]);
      return( par + parval[2]*fabsEta + parval[3]*eta*eta );
    }
    else {
      return( parval[1]+ parval[4]*std::fabs((fabsEta-parval[6])) + parval[5]*(fabsEta-parval[6])*(fabsEta-parval[6]) );
    }
  }

  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7]+parval[8]/pt  );
  }

  // // 1/pt in pt and quadratic in eta
  // virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
  //   return( parval[15]+parval[16]/pt + parval[17]*std::fabs(eta)+parval[18]*eta*eta );
  // }

  // constant sigmaCotgTh
  // virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
  //   return( 0.004 );
  // }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( 0.001 );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.001, 0.00001, 
                          0.00000001, 0.00000001, 0.00000001, 0.00000001,
                          0.001,
			  0.00002, 0.0002 };
			  // 0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "etaTransition", "offsetEtaHigh", 
                              "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh",
                              "secondParabolaCenter",
                              "Cth res. sc.", "Cth res. 1/Pt sc." };
			      // "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = { 0.8, -0.1,
                          0., 0., 0., 0.,
                          1.0,
                          -0.1, 0. };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 1.,1.};

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 2., 0.1,
                            0.01, 0.1, 0.1, 1.,
                            4., 
			    0.1, 0.1 };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};


/**
 * Same as type12, but improves sigmaCotgTh parameterization.
 */
// Resolution Type 30
template <class T>
class resolutionFunctionType30 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType30() { this->parNum_ = 27; }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    double fabsEta = std::fabs(eta);

    double ptPart = parval[13]*pt;
    if( fabsEta > 2.0 ) {
      ptPart = parval[22]*pt + parval[23]*pt*pt;
    }
    else if( fabsEta > 1.4 ) {
      ptPart = parval[20]*pt + parval[21]*pt*pt;
    }
    if(fabsEta<parval[0]) {
      return( ptPart + parval[1] + parval[2]*fabsEta + parval[3]*eta*eta );
    }
    // Return a line connecting the two parabolas
    else if( fabsEta < parval[14] ) {
      double x_1 = parval[0];
      double y_1 = parval[1] + parval[2]*parval[0] + parval[3]*parval[0]*parval[0];
      double x_2 = parval[14];
      double y_2 = parval[4] + parval[5]*std::fabs((parval[14]-parval[7])) + parval[6]*(parval[14]-parval[7])*(parval[14]-parval[7]);
      return( (fabsEta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1 );
    }
    else if( fabsEta < parval[15] ) {
      return( ptPart + parval[4] + parval[5]*std::fabs(fabsEta-parval[7]) + parval[6]*(fabsEta-parval[7])*(fabsEta-parval[7]) );
    }
    else {
      return( ptPart + parval[16] + parval[17]*std::fabs(fabsEta-parval[19]) + parval[18]*(fabsEta-parval[19])*(fabsEta-parval[19]) );
    }
  }

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);

    double ptPart = parError[13]*pt;
    if( fabsEta > 2.0 ) {
      ptPart = parError[22]*pt + parError[23]*pt*pt;
    }
    else if( fabsEta > 1.4 ) {
      ptPart = parError[20]*pt + parError[21]*pt*pt;
    }
    if(fabsEta<parval[0]) {
      return( ptPart + parError[1] + parError[2]*fabsEta + parError[3]*eta*eta );
    }
    // Note: this is a rough approximation, it should be fixed
    else if( fabsEta < parval[14] ) {
      double x_1 = parval[0];
      double y_1 = parval[1] + parval[2]*parval[0] + parval[3]*parval[0]*parval[0];
      double x_2 = parval[14];
      double y_2 = parval[4] + parval[5]*std::fabs((parval[14]-parval[7])) + parval[6]*(parval[14]-parval[7])*(parval[14]-parval[7]);
      double lineValue = (fabsEta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1;

      // x_1 = parval[0];
      y_1 = parval[1] + parError[1] + (parval[2] + parError[2])*parval[0] + (parval[3] + parError[3])*parval[0]*parval[0];
      // x_2 = parval[14];
      y_2 = parval[4] + parError[4] + (parval[5] + parError[5])*std::fabs((parval[14]-parval[7])) + (parval[6] + parError[6])*(parval[14]-parval[7])*(parval[14]-parval[7]);
      double lineValuePlusError = (fabsEta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1;
      
      return(lineValuePlusError - lineValue );
    }
    else if( fabsEta < parval[15] ) {
      return( ptPart + parError[4] + parError[5]*std::fabs(fabsEta-parval[7]) + parError[6]*(fabsEta-parval[7])*(fabsEta-parval[7]) );
    }
    else {
      return( ptPart + parError[16] + parError[17]*std::fabs(fabsEta-parval[19]) + parError[18]*(fabsEta-parval[19])*(fabsEta-parval[19]) );
    }
  }

  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    // return( parval[8] + parval[9]/(pt + parval[10]) + parval[11]*pt );
    double fabsEta = std::fabs(eta);
    double value = parval[8] + parval[9]*fabsEta + parval[10]*eta*eta + parval[11]*fabsEta*fabsEta*fabsEta;
    if( value > 0 ) {
      return( value );
    }
    return 0;
  }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
  //std::cout << "parval[12] = " << parval[12] << std::endl;
  
  return( parval[12] );
  
}

  virtual double covPt1Pt2(const double & pt1, const double & eta1, const double & pt2, const double & eta2, const T & parval)
  {
    return parval[24] + std::fabs(pt1 - pt2)*parval[25] + std::fabs(eta1 - eta2)*parval[26];
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "etaTransition",            0.0001,    0., 2. );
    parSet[1]  = ParameterSet( "constantCentral",          0.00001,   0., 1. );
    parSet[2]  = ParameterSet( "linearEtaCentral",         0.00001,   0., 1. );
    parSet[3]  = ParameterSet( "quadraticEtaCentral",      0.000001,  0., 1. );
    parSet[4]  = ParameterSet( "constantForward",          0.00001,   0., 1. );
    parSet[5]  = ParameterSet( "linearEtaForward",         0.00001,   0., 1. );
    parSet[6]  = ParameterSet( "quadraticEtaForward",      0.000001,  0., 1. );
    parSet[7]  = ParameterSet( "vertexForward",            0.0001,    0., 3. );
    parSet[8]  = ParameterSet( "cotgThetaConstant",        0.00001,  -1., 1. );
    parSet[9]  = ParameterSet( "cotgThetaFactor",          0.00001,  -1., 1. );
    parSet[10] = ParameterSet( "cotgThetaDenominatorTerm", 0.000001, -1., 1. );
    parSet[11] = ParameterSet( "cotgThetaLinearPt",        0.000001, -1., 1. );
    parSet[12] = ParameterSet( "sigmaPhi",                 0.0001,    0., 1. );
    parSet[13] = ParameterSet( "barrelLinearPt",           0.00001,   0., 1. );
    parSet[14] = ParameterSet( "split",                    0.0001,    0., 3. );
    parSet[15] = ParameterSet( "veryForwardSplit",         0.0001,    0., 3. );
    parSet[16] = ParameterSet( "constantVeryForward",      0.00001,   0., 1. );
    parSet[17] = ParameterSet( "linearEtaVeryForward",     0.00001,   0., 1. );
    parSet[18] = ParameterSet( "quadraticEtaVeryForward",  0.000001,  0., 1. );
    parSet[19] = ParameterSet( "vertexVeryForward",        0.0001,    0., 3. );
    parSet[20] = ParameterSet( "endcapsLinearPt",          0.00001,  -1., 1. );
    parSet[21] = ParameterSet( "endcapsQuadraticPt",       0.000001, -1., 1. );
    parSet[22] = ParameterSet( "veryForwardLinearPt",      0.00001,  -1., 1. );
    parSet[23] = ParameterSet( "veryForwardQuadraticPt",   0.000001, -1., 1. );
    parSet[24] = ParameterSet( "covPt1Pt2Constant",        0.000001, -1., 1. );
    parSet[25] = ParameterSet( "covPt1Pt2DeltaPt",         0.000001, -1., 1. );
    parSet[26] = ParameterSet( "covPt1Pt2DeltaEta",        0.000001, -1., 1. );

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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi

    parSet[0]  = ParameterSet( "etaTransition",            parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "constantCentral",          parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "linearEtaCentral",         parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "quadraticEtaCentral",      parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "constantForward",          parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "linearEtaForward",         parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "quadraticEtaForward",      parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "vertexForward",            parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "cotgThetaConstant",        parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "cotgThetaFactor",          parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "cotgThetaDenominatorTerm", parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "cotgThetaLinearPt",        parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "sigmaPhi",                 parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "barrelLinearPt",           parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "split",                    parStep[14], parMin[14], parMax[14] );
    parSet[15] = ParameterSet( "veryForwardSplit",         parStep[15], parMin[15], parMax[15] );
    parSet[16] = ParameterSet( "constantVeryForward",      parStep[16], parMin[16], parMax[16] );
    parSet[17] = ParameterSet( "linearEtaVeryForward",     parStep[17], parMin[17], parMax[17] );
    parSet[18] = ParameterSet( "quadraticEtaVeryForward",  parStep[18], parMin[18], parMax[18] );
    parSet[19] = ParameterSet( "vertexVeryForward",        parStep[19], parMin[19], parMax[19] );
    parSet[20] = ParameterSet( "endcapsLinearPt",          parStep[20], parMin[20], parMax[20] );
    parSet[21] = ParameterSet( "endcapsQuadraticPt",       parStep[21], parMin[21], parMax[21] );
    parSet[22] = ParameterSet( "veryForwardLinearPt",      parStep[22], parMin[22], parMax[22] );
    parSet[23] = ParameterSet( "veryForwardQuadraticPt",   parStep[23], parMin[23], parMax[23] );
    parSet[24] = ParameterSet( "covPt1Pt2Constant",        parStep[24], parMin[24], parMax[24] );
    parSet[25] = ParameterSet( "covPt1Pt2DeltaPt",         parStep[25], parMin[25], parMax[25] );
    parSet[26] = ParameterSet( "covPt1Pt2DeltaEta",        parStep[26], parMin[26], parMax[26] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};
template <class T>
class resolutionFunctionType31 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType31() { this->parNum_ = 14; }
  
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    double fabsEta = std::fabs(eta);
    
    if(fabsEta < 0.2 ) return parval[0]; 
    if(fabsEta < 0.4 ) return parval[1]; 
    if(fabsEta < 0.6 ) return parval[2]; 
    if(fabsEta < 0.8 ) return parval[3]; 
    if(fabsEta < 1.0 ) return parval[4]; 
    if(fabsEta < 1.2 ) return parval[5]; 
    if(fabsEta < 1.4 ) return parval[6]; 
    if(fabsEta < 1.6 ) return parval[7]; 
    if(fabsEta < 1.8 ) return parval[8]; 
    if(fabsEta < 2.0 ) return parval[9]; 
    if(fabsEta < 2.2 ) return parval[10]; 
    return parval[11]; 
  }
  
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    return 0;
  }
  
  
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval)
  {
    // return( parval[8] + parval[9]/(pt + parval[10]) + parval[11]*pt );
    // double fabsEta = std::fabs(eta);
    double value = parval[12] ;
    if( value > 0 ) {
      return( value );
    }
    return 0;
  }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval)
  {
    //std::cout << "parval[12] = " << parval[12] << std::endl;
    return( parval[13] );
  }

  virtual double covPt1Pt2(const double & pt1, const double & eta1, const double & pt2, const double & eta2, const T & parval)
  {
    return 0;
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "sigmaPt |eta|<0.2", 0.00001,   0., 1. );
    parSet[1]  = ParameterSet( "sigmaPt |eta|<0.4", 0.00001,   0., 1. );
    parSet[2]  = ParameterSet( "sigmaPt |eta|<0.6", 0.00001,   0., 1. );
    parSet[3]  = ParameterSet( "sigmaPt |eta|<0.8", 0.00001,   0., 1. );
    parSet[4]  = ParameterSet( "sigmaPt |eta|<1.0", 0.00001,   0., 1. );
    parSet[5]  = ParameterSet( "sigmaPt |eta|<1.2", 0.00001,   0., 1. );
    parSet[6]  = ParameterSet( "sigmaPt |eta|<1.4", 0.00001,   0., 1. );
    parSet[7]  = ParameterSet( "sigmaPt |eta|<1.6", 0.00001,   0., 1. );
    parSet[8]  = ParameterSet( "sigmaPt |eta|<1.8", 0.00001,   0., 1. );
    parSet[9]  = ParameterSet( "sigmaPt |eta|<2.0", 0.00001,   0., 1. );
    parSet[10] = ParameterSet( "sigmaPt |eta|<2.2", 0.00001,   0., 1. );
    parSet[11] = ParameterSet( "sigmaPt |eta|>2.2", 0.00001,   0., 1. );
    parSet[12] = ParameterSet( "sigmacotheta",             0.0001,    0., 1. );
    parSet[13] = ParameterSet( "sigmaPhi",                 0.0001, 0. ,1.);
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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi

    parSet[0]  = ParameterSet( "sigmaPt |eta|<0.2",  parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "sigmaPt |eta|<0.4",  parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "sigmaPt |eta|<0.6",  parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "sigmaPt |eta|<0.8",  parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "sigmaPt |eta|<1.0",  parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "sigmaPt |eta|<1.2",  parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "sigmaPt |eta|<1.4",  parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "sigmaPt |eta|<1.6",  parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "sigmaPt |eta|<1.8",  parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "sigmaPt |eta|<2.0",  parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "sigmaPt |eta|<2.2",  parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "sigmaPt |eta|>2.2",  parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "sigmacotTheta",            parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "sigmaPhi",                 parStep[13], parMin[13], parMax[13] );
    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

template <class T>
class resolutionFunctionType32 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType32()
  {
    this->parNum_ = 26;
    double tempBins[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			  0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
			  1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3 };
    std::copy( tempBins, tempBins+23, bins );
  }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    double fabsEta = std::fabs(eta);
    for( int i=0; i<23; ++i ) {
      if( fabsEta < bins[i] ) return parval[i];
    }
    // The last bin is |eta| > 2.3
    return parval[23];
  }
  
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);
    for( int i=0; i<23; ++i ) {
      if( fabsEta < bins[i] ) return parError[i];
    }
    return parError[23];
  }
  
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval)
  {
    // double fabsEta = std::fabs(eta);
    double value = parval[24] ;
    if( value > 0 ) {
      return( value );
    }
    return 0;
  }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval)
  {
    return( parval[25] );
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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    for( int i=0; i<24; ++i ) {
      parSet[i]  = ParameterSet( "eta bin", parStep[i],  parMin[i],  parMax[i]  );
    }
    parSet[24] = ParameterSet( "sigmaCotgTheta", parStep[24], parMin[24], parMax[24] );
    parSet[25] = ParameterSet( "sigmaPhi", parStep[25], parMin[25], parMax[25] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
  // Data members
  double bins[23];
};

// Daniele's function for Zmumu (36/pb)--------------------
template <class T>
class resolutionFunctionType40 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType40() { this->parNum_ = 10; }
  // linear in pt and by points in eta    parSet[0]  = ParameterSet( "Phi ampl bin0 (neg muon)"           ,      parStep[0], parMin[0], parMax[0] );

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);    
    if(fabsEta<parval[0]) {
      // To impose continuity we require that the parval[0] of type11 is
    double par = parval[1];// + parval[5]*(parval[0]-parval[6])*(parval[0]-parval[6]) - parval[3]*parval[0]*parval[0]
    return( par + parval[3]*eta*eta + parval[9]*pt);
    }
    else {
      double coeff, centre;
      if(eta>0.) {    coeff=parval[5]; centre=parval[6];
      }
      else {  coeff=parval[2];centre=parval[4];
      }
      return( parval[1] + coeff*(fabsEta-centre)*(fabsEta-centre) + parval[9]*pt);
    }
  }

  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7] );
  }
  
  // // 1/pt in pt and quadratic in eta
  // virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
  //   return( parval[15]+parval[16]/pt + parval[17]*std::fabs(eta)+parval[18]*eta*eta );
  // }

  // constant sigmaCotgTh
  // virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
  //   return( 0.004 );
  // }

  // constant sigmaPhi
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8] );
    //return( 0.001 );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.001, 0.00001,
                          0.00000001, 0.00000001, 0.001, 0.00000001,
                          0.001,
              0.00002, 
              0.0001, 
              0.000002 };
              // 0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "etaTransition", "offsetEtaHigh", 
                              "parabEtaHighNeg", "parabEtaCentral", "secondParabolaCenterNeg", "parabEtaHighPos",
                              "secondParabolaCenterPos",
                              "Cth res. sc.", "Phi res. sc.", "linearPt" };
                  // "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
    double thisMini[] = { 0.8, -0.1,
                          0., 0., 0., 0.,
                          0.,
                          0., 
			  0.,
			  0. };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1., 1.,
                            1., 1., 1.,
			    0.1,
			    0.001 };

      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 2., 0.1,
                            1., 0.1, 2.5, 1.,
                            2.5, 
			    0.1, 
			    0.1,
			    0.0005 };
      
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

//------------------------ 4Nov and 22 Dec data/MC Zmumu (36/pb) ---------
template <class T>
class resolutionFunctionType41 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType41() { this->parNum_ = 18; }
  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    double fabsEta = std::fabs(eta);
    if(eta>=parval[16] && eta<=parval[17]){
      return( parval[0]+parval[1]*pt + parval[2]*std::fabs(eta)+parval[3]*pow(eta,2) );
    }
    else if(eta<parval[16]){ //eta in left endcap
      double par = - parval[12]*parval[16]*parval[16] + parval[0]+parval[1]*pt + parval[2]*parval[16]+parval[3]*pow(parval[16],2);
      return( par + parval[12]*(fabsEta-parval[13])*(fabsEta-parval[13]) );
    }
    
    else{ //eta in righ endcap  ///
      double par = - parval[14]*parval[17]*parval[17] + parval[0]+parval[1]*pt + parval[2]*parval[17]+parval[3]*pow(parval[17],2);
      return( par + parval[14]*(fabsEta-parval[15])*(fabsEta-parval[15]) ); 
    }
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[4]+parval[5]/pt + parval[6]*std::fabs(eta)+parval[7]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*std::fabs(eta)+parval[11]*pow(eta,2) );
  }

  // derivatives ---------------

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);
    if(eta>=parval[16] && eta<=parval[17]){  //central region
      //  return( parval[0]+parval[1]*pt + parval[2]*std::fabs(eta)+parval[3]*pow(eta,2) );
      return( sqrt(parError[0]*parError[0]+(pt*pt)*parError[1]*parError[1]+(eta*eta)*parError[2]*parError[2]+pow(eta,4)*parError[3]*parError[3]) );
    }
    else if(eta<parval[16]){ // Left endcap  
      return( sqrt(pow(parval[16]+(fabsEta-parval[13])*(fabsEta-parval[13]),2)*parError[12]*parError[12]+
		   pow(parError[16]*(2*parval[16]*parval[12]+parval[2]+2*parval[16]*parval[3]),2)+
		   pow(parError[0],2)+
		   pow(parError[1]*pt,2)+
		   pow(parError[2]*parval[16],2)+
		   pow(parval[16],4)*pow(parError[3],2)+
		   pow(2*(fabsEta-parval[13])*parError[13],2)
		   )  
	      );
    }
    
    else{ //eta in righ endcap  ///

      return( sqrt(pow(parval[17]+(fabsEta-parval[15])*(fabsEta-parval[15]),2)*parError[14]*parError[14]+
		   pow(parError[17]*(2*parval[17]*parval[14]+parval[2]+2*parval[17]*parval[3]),2)+
		   pow(parError[0],2)+
		   pow(parError[1]*pt,2)+
		   pow(parError[2]*parval[17],2)+
		   pow(parval[17],4)*pow(parError[3],2)+
		   pow(2*(fabsEta-parval[15])*parError[15],2)
		   )  
	      );


    }
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType) {
    double thisStep[] = { 0.002, 0.00002, 0.000002, 0.0002,
                          0.00002, 0.0002, 0.0000002, 0.000002,
                          0.00002, 0.0002, 0.00000002, 0.000002,
			  0.0002, 0.001, 0.0002, 0.001,
			  0.01,0.01
    };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Eta sc.", "Pt res. Eta^2 sc.",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc.",
			      "Pt res. Eta^2 sc./offset (right)","Pt res. Eta^2 sc.(left)","Pt res. Eta^2 sc./offset (right)","Pt res. Eta^2 sc.(left)",
			      "floating point right","floating poin left"
    };
    double thisMini[] = { 0.0, -0.01, -0.001, -0.0001,
                          0.0, -0.001, -0.001, -0.00001,
                          0.0, -0.001, -0.0001, -0.0001,
			  -0.0001, 0.01, -0.0001, 0.01,
			  -2.,0.
    };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1.,
			    1., 1. ,1. ,1.,
			    1.,1.
      };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMaxi[] = { 0.1, 0.01, 0.01, 0.1,
                            0.01, 0.01, 0.1, 0.01,
                            0.01, 0.01, 0.01, 0.01,
			    0.1, 1.99, 0.1, 1.99,
			    0.,2.
			 
      };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};







//------------------------ 4Nov and 22 Dec data/MC Zmumu (36/pb) -- 3 parabolas (for MU-10-004) ---------
template <class T>
class resolutionFunctionType42 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType42() { this->parNum_ = 15; }
  
  inline double centralParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0]+parval[1]*pt + parval[14]*pt*pt + parval[2]*std::fabs(eta) + parval[3]*eta*eta );
  }
  inline double leftParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[1]*pt + parval[14]*pt*pt + parval[5]*fabs(eta-parval[7]) +
    parval[6]*(eta-parval[7])*(eta-parval[7]) );
  }
  inline double rightParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[1]*pt + parval[14]*pt*pt + parval[9]*fabs(eta-parval[11]) +
    parval[10]*(eta-parval[11])*(eta-parval[11]) );
  }

  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    //double fabsEta = std::fabs(eta);
    if(eta>=parval[12] && eta<=parval[13]){
      return centralParabola(pt, eta, parval);
    }
    else if(eta<parval[12]) { //eta in left endcap
      return( centralParabola(pt, parval[12], parval) - leftParabola(pt, parval[12], parval) + leftParabola(pt, eta, parval) );
    }
    // std::cout << "parval[13] = " << parval[13] << ", eta = " << eta << std::endl;
    return( centralParabola(pt, parval[13], parval) - rightParabola(pt, parval[13], parval) + rightParabola(pt, eta, parval) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
      //0.00035 + eta*eta*0.00015; // fixed from MC (Mar. 2011)
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);
    if( eta >= parval[12] && eta <= parval[13] ) {
      return sqrt( pow(parError[0], 2) + pow(pt*parError[1], 2) + pow(pt*pt*parError[14], 2) + pow(fabsEta*parError[2], 2) + pow(eta*eta*parError[3], 2));
    }
    else if( eta < parval[12] ) {
      double sign1 = 1;
      // if( parval[12] < parval[7] ) sign1 = -1;
      double sign2 = 1;
      if( eta < parval[7] ) sign2 = -1;
      double sign3 = 1;
      if( parval[12] < 0 ) sign3 = -1; // This should be always true

      return(sqrt(pow(parError[0],2) +  // parval[0]
      pow(pt*parError[1], 2) +           // parval[1]
      pow(parval[12]*parError[2], 2) +   // parval[2]
      pow(parval[12]*parval[12]*parError[3], 2) + // parval[3]
      pow((-fabs(parval[12]-parval[7])+fabs(eta-parval[7]))*parError[5], 2) + // parval[5]
      pow((- pow(parval[12]-parval[7], 2) + pow(eta-parval[7], 2))*parError[6], 2) + // parval[6]
      pow((sign1*parval[5] + 2*(parval[12]-parval[7])*parval[6] - sign2*parval[5] + - 2*(eta-parval[7])*parval[6])*parError[7], 2) +
      pow((sign3*parval[2] + 2*parval[12]*parval[3] - sign1*parval[5] - 2*(parval[12]-parval[7])*parval[6])*parError[12], 2) + // parval[12]
      pow(pt*pt*parError[14], 2)) );

/*      return sqrt( pow(parError[4], 2) + pow(pt*parError[1], 2) + pow(pt*pt*parError[14], 2) +
		   pow((fabsEta-parval[7])*parError[5], 2) +
		   pow(pow(fabsEta-parval[7], 2)*parError[6], 2) +
		   pow(-parval[5] - 2*parval[6]*(fabsEta - parval[7]), 2)*pow(parError[7], 2) );*/
    }

    double sign1 = 1;
    // if( parval[13] < parval[11] ) sign1 = -1;
    double sign2 = 1;
    if( eta < parval[11] ) sign2 = -1;
    double sign3 = 1;
    if( parval[13] < 0 ) sign3 = -1; // This should never be true

    return(sqrt(pow(parError[0],2) +  // parval[0]
		pow(pt*parError[1], 2) +           // parval[1]
		pow(parval[13]*parError[2], 2) +   // parval[2]
		pow(parval[13]*parval[13]*parError[3], 2) + // parval[3]
		pow((-fabs(parval[13]-parval[11])+fabs(eta-parval[11]))*parError[9], 2) + // parval[9]
		pow((- pow(parval[13]-parval[11], 2) + pow(eta-parval[11], 2))*parError[10], 2) + // parval[10]
		pow((sign1*parval[9] + 2*(parval[13]-parval[11])*parval[10] - sign2*parval[9] + - 2*(eta-parval[11])*parval[10])*parError[11], 2) + // parval[11]
		pow((sign3*parval[2] + 2*parval[13]*parval[3] - sign1*parval[9] - 2*(parval[13]-parval[11])*parval[10])*parError[13], 2) + // parval[13]
		pow(pt*pt*parError[14], 2)) );
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
			     TString* parname, const T & parResol, const std::vector<int> & parResolOrder,
			     const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",                      0.002,      -0.1,  0.1  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",             0.00002,    -0.01, 0.01 );
    parSet[2]  = ParameterSet( "Pt res. Eta sc.",                  0.000002,   0.,    0.01 );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc.",                0.0002,     -0.01, 0.02 );
    parSet[4]  = ParameterSet( "Pt res. sc. (left)",               0.00002,    0.,    0.01 );
    parSet[5]  = ParameterSet( "Pt res. Eta sc. (left)",           0.0002,     0.,    0.2  );
    parSet[6]  = ParameterSet( "Pt res. Eta^2 sc. (left)",         0.0000002,  -0.2,  0.5  );
    parSet[7]  = ParameterSet( "Pt res. Eta^2 sc./offset (left)",  0.0002,     -2.2,  -0.8 );
    parSet[8]  = ParameterSet( "Pt res. sc. (right)",              0.00002,    0.,    0.01 );
    parSet[9]  = ParameterSet( "Pt res. Eta sc. (right)",          0.0002,     -0.2,  0.1  );
    parSet[10] = ParameterSet( "Pt res. Eta^2 sc. (right)",        0.000002,   -0.1,  0.5  );
    parSet[11] = ParameterSet( "Pt res. Eta^2 sc./offset (right)", 0.0002,     0.,    3.   );
    parSet[12] = ParameterSet( "floating point left",              0.001,      -2.2,  -1.6 );
    parSet[13] = ParameterSet( "floating point right",             0.001,      1.,    2.2  );
    parSet[14] = ParameterSet( "pt^2 sc.",                         0.0001,     0.,    0.01 );

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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",                      parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",             parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "Pt res. Eta sc.",                  parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc.",                parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "Pt res. sc. (left)",               parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "Pt res. Eta sc. (left)",           parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "Pt res. Eta^2 sc. (left)",         parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "Pt res. Eta^2 sc./offset (left)",  parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "Pt res. sc. (right)",              parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "Pt res. Eta sc. (right)",          parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "Pt res. Eta^2 sc. (right)",        parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "Pt res. Eta^2 sc./offset (right)", parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "floating point left",              parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "floating point right",             parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "pt^2 sc.",                         parStep[14], parMin[14], parMax[14] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};



//------------------------ 4Nov and 22 Dec data/MC Zmumu (36/pb) -- 3 parabolas (for MU-10-004) ---------
template <class T>
class resolutionFunctionType43 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType43() { this->parNum_ = 17; }

  inline double centralParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0]+parval[1]*pt + parval[14]*pt*pt + parval[2]*std::fabs(eta)+parval[3]*eta*eta );
  }
  inline double leftParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0] + fabs(parval[12])*parval[2] + parval[12]*parval[12]*parval[3] -
	    parval[5]*fabs(parval[12]-parval[7]) -
	    parval[6]*(parval[12]-parval[7])*(parval[12]-parval[7]) +
	    parval[1]*pt + parval[14]*pt*pt + parval[5]*fabs(eta-parval[7]) +
	    parval[6]*(eta-parval[7])*(eta-parval[7]) );
  }
  inline double rightParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0] + fabs(parval[13])*parval[2] + parval[13]*parval[13]*parval[3] -
	    parval[9]*fabs(parval[13]-parval[11]) -
	    parval[10]*(parval[13]-parval[11])*(parval[13]-parval[11]) +
	    parval[1]*pt + parval[14]*pt*pt + parval[9]*fabs(eta-parval[11]) +
	    parval[10]*(eta-parval[11])*(eta-parval[11]) );
  }
  inline double leftLine(const double & pt, const double & eta, const T & parval)
  {
    double x_1 = parval[15];
    double y_1 = centralParabola(pt, parval[15], parval);
    double x_2 = parval[12];
    double y_2 = leftParabola(pt, parval[12], parval);
    return( (eta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1 );
  }
  inline double rightLine(const double & pt, const double & eta, const T & parval)
  {
    double x_1 = parval[16];
    double y_1 = centralParabola(pt, parval[16], parval);
    double x_2 = parval[13];
    double y_2 = rightParabola(pt, parval[13], parval);
    return( (eta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1 );
  }

  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    //double fabsEta = std::fabs(eta);
    if(eta>=parval[15] && eta<=parval[16]){
      return centralParabola(pt, eta, parval);
    }
    // Return a line connecting the two parabolas
    else if( (eta >= parval[12]) && (eta < parval[15]) ) {
      return leftLine(pt, eta, parval);
    }
    else if( eta < parval[12] ){ //eta in left endcap
      return leftParabola(pt, eta, parval);
    }
    // Return a line connecting the two parabolas
    else if( (eta > parval[16]) && (eta <= parval[13]) ) {
      return rightLine(pt, eta, parval);
    }
    return rightParabola(pt, eta, parval);
  }

  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval)
  {
    return 0;
      //0.00035 + eta*eta*0.00015; // fixed from MC (Mar. 2011)
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval)
  {
    return 0.;
  }

  // derivatives ---------------
  inline double centralParabolaError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);
    return sqrt( pow(parError[0], 2) + pow(pt*parError[1], 2) + pow(pt*pt*parError[14], 2) + pow(fabsEta*parError[2], 2) + pow(eta*eta*parError[3], 2));
  }

  inline double leftParabolaError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double sign1 = 1;
    if( parval[12] < parval[7] ) sign1 = -1;
    double sign2 = 1;
    if( eta < parval[7] ) sign2 = -1;
    double sign3 = 1;
    if( parval[12] < 0 ) sign3 = -1; // This should be always true

    return(sqrt(pow(parError[0],2) +  // parval[0]
		pow(pt*parError[1], 2) +           // parval[1]
		pow(parval[12]*parError[2], 2) +   // parval[2]
		pow(parval[12]*parval[12]*parError[3], 2) + // parval[3]
		pow((-fabs(parval[12]-parval[7])+fabs(eta-parval[7]))*parError[5], 2) + // parval[5]
		pow((- pow(parval[12]-parval[7], 2) + pow(eta-parval[7], 2))*parError[6], 2) + // parval[6]
		pow((sign1*parval[5] + 2*(parval[12]-parval[7])*parval[6] - sign2*parval[5] + - 2*(eta-parval[7])*parval[6])*parError[7], 2) +
		pow((sign3*parval[2] + 2*parval[12]*parval[3] - sign1*parval[5] - 2*(parval[12]-parval[7])*parval[6])*parError[12], 2) + // parval[12]
		pow(pt*pt*parError[14], 2)) );
  }

  inline double rightParabolaError(const double & pt, const double & eta, const T & parval, const T & parError) {
    double sign1 = 1;
    if( parval[13] < parval[11] ) sign1 = -1;
    double sign2 = 1;
    if( eta < parval[11] ) sign2 = -1;
    double sign3 = 1;
    if( parval[13] < 0 ) sign3 = -1; // This should never be true

    return(sqrt(pow(parError[0],2) +  // parval[0]
		pow(pt*parError[1], 2) +           // parval[1]
		pow(parval[13]*parError[2], 2) +   // parval[2]
		pow(parval[13]*parval[13]*parError[3], 2) + // parval[3]
		pow((-fabs(parval[13]-parval[11])+fabs(eta-parval[11]))*parError[9], 2) + // parval[9]
		pow((- pow(parval[13]-parval[11], 2) + pow(eta-parval[11], 2))*parError[10], 2) + // parval[10]
		pow((sign1*parval[9] + 2*(parval[13]-parval[11])*parval[10] - sign2*parval[9] + - 2*(eta-parval[11])*parval[10])*parError[11], 2) + // parval[11]
		pow((sign3*parval[2] + 2*parval[13]*parval[3] - sign1*parval[9] - 2*(parval[13]-parval[11])*parval[10])*parError[13], 2) + // parval[13]
		pow(pt*pt*parError[14], 2)) );
  }

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    // double fabsEta = std::fabs(eta);
    if( eta >= parval[15] && eta <= parval[16] ) {
      return centralParabolaError(pt, eta, parval, parError);
    }
    else if( (eta >= parval[12]) && (eta < parval[15]) ) {
      double lineValue = leftLine(pt, eta, parval);

      double x_1 = parval[15];
      double y_1 = centralParabola(pt, parval[15], parval) + centralParabolaError(pt, parval[15], parval, parError);
      double x_2 = parval[12];
      double y_2 = leftParabola(pt, parval[12], parval) + leftParabolaError(pt, parval[12], parval, parError);
      double lineValuePlusError = (eta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1;
      
      return( lineValuePlusError - lineValue );
    }
    else if( eta < parval[12] ) {
      return leftParabolaError(pt, eta, parval, parError);
    }
    else if( (eta > parval[16]) && (eta <= parval[13]) ) {
      double lineValue = rightLine(pt, eta, parval);

      double x_1 = parval[16];
      double y_1 = centralParabola(pt, parval[16], parval) + centralParabolaError(pt, parval[16], parval, parError);
      double x_2 = parval[13];
      double y_2 = rightParabola(pt, parval[13], parval) + leftParabolaError(pt, parval[13], parval, parError);
      double lineValuePlusError = (eta - x_1)*(y_2 - y_1)/(x_2 - x_1) + y_1;
      
      return( lineValuePlusError - lineValue );
    }
    return rightParabolaError(pt, eta, parval, parError);
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",			   0.002,      0.,      0.1  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",		   0.00002,    -0.01,   0.01 );
    parSet[2]  = ParameterSet( "Pt res. Eta sc. central",	   0.000002,   0.,      0.01 );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. central",	   0.0002,     -0.0001, 0.1  );
    parSet[4]  = ParameterSet( "Not used",      		   0.00002,    0.,      0.01 );
    parSet[5]  = ParameterSet( "Pt res. Eta sc. (left)",	   0.0002,     -0.01,   0.1  );
    parSet[6]  = ParameterSet( "Pt res. Eta^2 sc. (left)",	   0.0000002,  0.,      1.   );
    parSet[7]  = ParameterSet( "Pt res. Eta^2 sc./offset (left)",  0.0002,     -2.,     0.   );
    parSet[8]  = ParameterSet( "Not used",	        	   0.00002,    0.,      0.01 );
    parSet[9]  = ParameterSet( "Pt res. Eta sc. (right)",	   0.0002,     -0.01,   0.1  );
    parSet[10] = ParameterSet( "Pt res. Eta^2 sc. (right)",	   0.0000002,  -1.,     1.   );
    parSet[11] = ParameterSet( "Pt res. Eta^2 sc./offset (right)", 0.0002,     0.,      2.   );
    parSet[12] = ParameterSet( "floating point left",		   0.001,      -2.,     1.   );
    parSet[13] = ParameterSet( "floating point right",		   0.001,      1.,      3.   );
    parSet[14] = ParameterSet( "pt^2 sc.",                         0.0001,     0.,      0.01 );
    parSet[15] = ParameterSet( "left line point",                  0.001,      -2.,     0.8  );
    parSet[16] = ParameterSet( "right line point",                 0.001,      0.8,     2.   );

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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",			   parStep[0],  parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",		   parStep[1],  parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "Pt res. Eta sc. central",	   parStep[2],  parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. central",	   parStep[3],  parMin[3],  parMax[3]  );
    parSet[4]  = ParameterSet( "Not used",      		   parStep[4],  parMin[4],  parMax[4]  );
    parSet[5]  = ParameterSet( "Pt res. Eta sc. (left)",	   parStep[5],  parMin[5],  parMax[5]  );
    parSet[6]  = ParameterSet( "Pt res. Eta^2 sc. (left)",	   parStep[6],  parMin[6],  parMax[6]  );
    parSet[7]  = ParameterSet( "Pt res. Eta^2 sc./offset (left)",  parStep[7],  parMin[7],  parMax[7]  );
    parSet[8]  = ParameterSet( "Not used",	        	   parStep[8],  parMin[8],  parMax[8]  );
    parSet[9]  = ParameterSet( "Pt res. Eta sc. (right)",	   parStep[9],  parMin[9],  parMax[9]  );
    parSet[10] = ParameterSet( "Pt res. Eta^2 sc. (right)",	   parStep[10], parMin[10], parMax[10] );
    parSet[11] = ParameterSet( "Pt res. Eta^2 sc./offset (right)", parStep[11], parMin[11], parMax[11] );
    parSet[12] = ParameterSet( "floating point left",		   parStep[12], parMin[12], parMax[12] );
    parSet[13] = ParameterSet( "floating point right",		   parStep[13], parMin[13], parMax[13] );
    parSet[14] = ParameterSet( "pt^2 sc.",                         parStep[14], parMin[14], parMax[14] );
    parSet[15] = ParameterSet( "left line point",                  parStep[15], parMin[15], parMax[15] );
    parSet[16] = ParameterSet( "right line point",                 parStep[16], parMin[16], parMax[16] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};




// Binned in eta to fit the Z with 40/pb
template <class T>
class resolutionFunctionType45 : public resolutionFunctionBase<T> {
 public:
  int etaBin(const double & eta)
  {
    // 24 bins from -2.4 to 2.4, first shift the range to be positive and then compute the index by x/k
    double shiftedEta = eta+2.4;

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
   
  // resolutionFunctionType45() { this->parNum_ = 21; }
  resolutionFunctionType45() { this->parNum_ = 13; }
  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval)
  {
    // std::cout << "parval["<<etaBin(eta)<<"] = " << parval[etaBin(eta)] << std::endl;

    if( eta < -2.0 ) return( parval[0]*pt + parval[1] );
    if( eta < -1.8 ) return( parval[0]*pt + parval[2] );
    if( eta < -1.6 ) return( parval[0]*pt + parval[3] );
    if( eta < -1.2 ) return( parval[0]*pt + parval[4] );
    if( eta < -0.8 ) return( parval[0]*pt + parval[5] );
    if( eta < 0. )   return( parval[0]*pt + parval[6] );
    if( eta < 0.8 )  return( parval[0]*pt + parval[7] );
    if( eta < 1.2 )  return( parval[0]*pt + parval[8] );
    if( eta < 1.6 )  return( parval[0]*pt + parval[9] );
    if( eta < 1.8 )  return( parval[0]*pt + parval[10] );
    if( eta < 2.0 )  return( parval[0]*pt + parval[11] );
    return( parval[0]*pt + parval[12] );

  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
      //0.00035 + eta*eta*0.00015; // fixed from MC (Mar. 2011)
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    // Use the etaByPoints function to select the right bin for the parameter
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
    parSet[2]  = ParameterSet( "eta bin 2",   0.000002,   0.,    0.01 );
    parSet[3]  = ParameterSet( "eta bin 3",   0.000002,   0.,    0.01 );
    parSet[4]  = ParameterSet( "eta bin 4",   0.00002,    -0.01, 0.01 );
    parSet[5]  = ParameterSet( "eta bin 5",   0.000002,   0.,    0.01 );
    parSet[6]  = ParameterSet( "eta bin 6",   0.00002,    -0.01, 0.01 );
    parSet[7]  = ParameterSet( "eta bin 7",   0.000002,   0.,    0.01 );
    parSet[8]  = ParameterSet( "eta bin 8",   0.00002,    -0.01, 0.01 );
    parSet[9]  = ParameterSet( "eta bin 9",   0.000002,   0.,    0.01 );
    parSet[10] = ParameterSet( "eta bin 10",  0.00002,    -0.01, 0.01 );
    parSet[11] = ParameterSet( "eta bin 11",  0.000002,   0.,    0.01 );
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





//------------------------ 4Nov and 22 Dec data/MC Zmumu (36/pb) -- 3 parabolas (for MU-10-004) ---------
template <class T>
class resolutionFunctionType44 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType44() { this->parNum_ = 6; }
  
  inline double leftParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0] + parval[1]*pt + parval[2]*std::fabs(eta) + parval[3]*eta*eta );
  }
  inline double rightParabola(const double & pt, const double & eta, const T & parval)
  {
    return( parval[0] + parval[1]*pt + parval[4]*fabs(eta) + parval[5]*eta*eta );
  }

  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    //double fabsEta = std::fabs(eta);
    if( eta <= 0 ){
      return leftParabola(pt, eta, parval);
    }
    else
      return rightParabola(pt, eta, parval);

  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------

  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError)
  {
    double fabsEta = std::fabs(eta);
    if( eta <= 0 ) {
      return sqrt( pow(parError[0], 2) + pow(pt*parError[1], 2) + pow(fabsEta*parError[2], 2) + pow(eta*eta*parError[3], 2));
    }
    else {
      return sqrt( pow(parError[0], 2) + pow(pt*parError[1], 2) + pow(fabsEta*parError[4], 2) + pow(eta*eta*parError[5], 2));
    }
  }

  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",               0.002,    -0.1,  0.1  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",      0.00002,  -0.01, 0.01 );
    parSet[2]  = ParameterSet( "Pt res. Eta sc. (left)",    0.000002, 0.,    0.01 );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. (left)",  0.0002,   -0.01, 0.02 );
    parSet[4]  = ParameterSet( "Pt res. Eta sc. (right)",   0.000002, 0.,    0.01 );
    parSet[5]  = ParameterSet( "Pt res. Eta^2 sc. (right)", 0.0002,   -0.01, 0.02 );

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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",              parStep[0], parMin[0], parMax[0] );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",     parStep[1], parMin[1], parMax[1] );
    parSet[2]  = ParameterSet( "Pt res. Eta sc. (left)",   parStep[2], parMin[2], parMax[2] );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. (left)", parStep[3], parMin[3], parMax[3] );
    parSet[4]  = ParameterSet( "Pt res. sc. (right)",      parStep[4], parMin[4], parMax[4] );
    parSet[5]  = ParameterSet( "Pt res. Eta sc. (right)",  parStep[5], parMin[5], parMax[5] );

    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

//--- 2012.04.05 ---//

template <class T>
class resolutionFunctionType99 : public resolutionFunctionBase<T> {
public:
  resolutionFunctionType99() { this->parNum_ = 4; }

  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {

    if( eta <=0 ) {
      return (parval[0] + parval[1]*pt + parval[2]*eta*eta);
    }
    else {
      return (parval[0] + parval[1]*pt + parval[3]*eta*eta);
    }
    
  }

  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return 0;
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return 0.;
  }

  // derivatives ---------------
  virtual double sigmaPtError(const double & pt, const double & eta, const T & parval, const T & parError) {
    if( eta <=0 ) {
      return( parError[0] + parError[1]*pt + parError[2]*eta*eta );
    }
    else {
      return( parError[0] + parError[1]*pt + parError[3]*eta*eta );
    }
  }



  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const std::vector<int> & parResolOrder, const int muonType)
  {
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",			   0.002,      0.,      0.1  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",		   0.00002,    -0.01,   0.01 );
    parSet[2]  = ParameterSet( "Pt res. Eta^2 sc. (left)",	   0.0000002,  -1.,     1.   );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. (right)",	   0.0000002,  -1.,     1.   );


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
      exit(1);
    }
    std::vector<ParameterSet> parSet(this->parNum_);
    // name, step, mini, maxi
    parSet[0]  = ParameterSet( "Pt res. sc.",			   parStep[0], parMin[0],  parMax[0]  );
    parSet[1]  = ParameterSet( "Pt res. Pt sc. (all)",		   parStep[1], parMin[1],  parMax[1]  );
    parSet[2]  = ParameterSet( "Pt res. Eta^2 sc. (left)",	   parStep[2], parMin[2],  parMax[2]  );
    parSet[3]  = ParameterSet( "Pt res. Eta^2 sc. (right)",	   parStep[3], parMin[3],  parMax[3] );



    std::cout << "setting parameters" << std::endl;
    for( int i=0; i<this->parNum_; ++i ) {
      std::cout << "parStep["<<i<<"] = " << parStep[i]
		<< ", parMin["<<i<<"] = " << parMin[i]
		<< ", parMax["<<i<<"] = " << parMin[i] << std::endl;
    }
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, parSet );
  }
};

//-----------------//

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

#endif // FUNCTIONS_H
