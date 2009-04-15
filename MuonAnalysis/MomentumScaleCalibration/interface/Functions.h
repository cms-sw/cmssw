#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <vector>
#include "TMath.h"
#include "TString.h"

using namespace std;

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
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) = 0;
  virtual int parNum() const { return parNum_; }
 protected:
  int parNum_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                      TString* parname, const T & parScale, const vector<int> & parScaleOrder,
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {}
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
};
// Linear in |eta|
// ---------------
template <class T>
class scaleFunctionType2 : public scaleFunctionBase<T> {
public:
  scaleFunctionType2() { this->parNum_ = 2; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return ( (parScale[0] + parScale[1]*fabs(eta))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
  scaleFunctionType3() { this->parNum_ = 2; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*sin(phi))*pt ); 
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01};
    TString thisParName[] = {"Phi offset", "Phi ampl"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.1};
      double thisMaxi[] = {1.1, 0.1};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.05};
      double thisMaxi[] = {1.03, 0.05};
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
             parScale[2]*fabs(eta))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
    return( (parScale[0] + parScale[1]*fabs(eta) + 
             parScale[2]*sin(phi))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
};
// Linear in pt and |eta| and sinusoidal in phi
// --------------------------------------------
template <class T>
class scaleFunctionType7 : public scaleFunctionBase<T> {
public:
  scaleFunctionType7() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt + 
             parScale[2]*fabs(eta) + 
             parScale[3]*sin(phi))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                             TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
             parScale[2]*fabs(eta) +
             parScale[3]*eta*eta)*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Eta slope", "Eta quadr"};
    if( muonType == 1 ) {
      double thisMini[] = {0.9, -0.3, -0.3, -0.3};
      double thisMaxi[] = {1.1, 0.3, 0.3, 0.3};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.97, -0.1, -0.1, -0.1};
      double thisMaxi[] = {1.03, 0.1, 0.1, 0.1};
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01};
    TString thisParName[] = {"Pt offset", "Pt slope", "Pt quadr"};
    double thisMini[] = {0.97, -0.1, -0.001};
    double thisMaxi[] = {1.03, 0.1, 0.001};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};
// Linear in pt, sinusoidal in phi with muon sign
// ----------------------------------------------
template <class T>
class scaleFunctionType11 : public scaleFunctionBase<T> {
public:
  scaleFunctionType11() { this->parNum_ = 4; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt + 
             (double)chg*parScale[2]*sin(phi+parScale[3]))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
    double thisStep[] = {0.001, 0.01, 0.01, 0.1};
    TString thisParName[] = {"Pt scale", "Pt slope", "Phi ampl", "Phi phase"};
    double thisMini[] = {0.97, -0.1, -0.02, 0.};
    double thisMaxi[] = {1.03, 0.1, 0.02, 3.1416};
    this->setPar( Start, Step, Mini, Maxi, ind, parname, parScale, parScaleOrder, thisStep, thisMini, thisMaxi, thisParName );
  }
};
// Linear in pt, parabolic in eta, sinusoidal in phi with muon sign
// ---------------------------------------------------------------- 
template <class T>
class scaleFunctionType12 : public scaleFunctionBase<T> {
public:
  scaleFunctionType12() { this->parNum_ = 6; }
  virtual double scale(const double & pt, const double & eta, const double & phi, const int chg, const T & parScale) const {
    return( (parScale[0] + parScale[1]*pt + 
             parScale[2]*fabs(eta) +
             parScale[3]*eta*eta + 
             (double)chg*parScale[4]*sin(phi+parScale[5]))*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
               parScale[2]*fabs(eta) +
               parScale[3]*eta*eta + 
               parScale[4]*sin(phi+parScale[5]))*pt );
    }
    // else {
    return( (parScale[0] + parScale[1]*pt + 
             parScale[2]*fabs(eta) +
             parScale[3]*eta*eta + 
             parScale[6]*sin(phi+parScale[7]))*pt );
    // }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
    return( ( parScale[0] +
              parScale[1]*pt + parScale[2]*pow(pt,2) + parScale[3]*pow(pt,3) +
              parScale[4]*fabs(eta) + parScale[5]*pow(eta,2) + parScale[6]*fabs(pow(eta,3)) +
              parScale[7]*pow(eta,4) + parScale[8]*fabs(pow(eta,5)) + parScale[9]*pow(eta,6) )*pt );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
                           -0.1, -0.1, -0.1,
                           -0.1, -0.01, -0.001, -0.0001, -0.00001, -0.000001};
      double thisMaxi[] = {1.03,
                           0.1, 0.1, 0.1,
                           0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001};
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
      return( ( parScale[1] + parScale[3]*fabs(eta) + parScale[4]*pow(eta,2) )*pt );
    }
    else {
      return( ( parScale[2] + parScale[3]*fabs(eta) + parScale[4]*pow(eta,2) )*pt );
    }
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parScale, const vector<int> & parScaleOrder, const int muonType) {
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
};

/// Service to build the scale functor corresponding to the passed identifier
scaleFunctionBase<double * > * scaleFunctionService( const int identifier );

/// Service to build the scale functor corresponding to the passed identifier when receiving a vector<double>
scaleFunctionBase<vector<double> > * scaleFunctionVecService( const int identifier );

// -------------- //
// Smear functors //
// -------------- //

class smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) = 0;
  smearFunctionBase() { cotgth_ = 0.; }
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
};
inline smearFunctionBase::~smearFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.

// No smearing
// -----------
class smearFunctionType0 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) { }
};
// The 3 parameters of smearType1 are: pt dependence of pt smear, phi smear and
// cotgtheta smear.
class smearFunctionType1 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt);
    phi = phi*(1.0+y[1]*parSmear[1]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[2]*parSmear[2]);
    smearEta(eta);
  }
};

class smearFunctionType2 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*fabs(eta));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]);
    smearEta(eta);
  }
};

class smearFunctionType3 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*fabs(eta));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*fabs(eta));
    smearEta(eta);
  }
};
// The six parameters of SmearType=4 are respectively:
// Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
// |eta| dep. of |eta| res., Pt^2 dep. of Pt res.
class smearFunctionType4 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*fabs(eta)+y[5]*parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*fabs(eta));
    smearEta(eta);
  }
};

class smearFunctionType5 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y, const vector<double> & parSmear) {
    pt = pt*(1.0+y[0]*parSmear[0]*pt+y[1]*parSmear[1]*fabs(eta)+y[5]*parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*parSmear[2]+y[6]*parSmear[6]*pt);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*parSmear[3]+y[4]*parSmear[4]*fabs(eta));
    smearEta(eta);
  }
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
 * Need to use templates to make it work with both array and vector<double>.
 */
template <class T>
class resolutionFunctionBase {
 public:
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) = 0;
  resolutionFunctionBase() {}
  virtual ~resolutionFunctionBase() = 0;
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const vector<int> & parResolOrder, const int muonType) = 0;
  virtual int parNum() const { return parNum_; }
 protected:
  int parNum_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
         TString* parname, const T & parResol, const vector<int> & parResolOrder,
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
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const vector<int> & parResolOrder, const int muonType) {
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
};

// Resolution Type 6
template <class T>
class resolutionFunctionType6 : public resolutionFunctionBase<T> {
 public:
  resolutionFunctionType6() { this->parNum_ = 15; }
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt+parval[2]*pow(pt,2)+parval[3]*pow(pt,3)+parval[4]*pow(pt,4)+parval[5]*fabs(eta)+parval[6]*pow(eta,2) );
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7]+parval[8]/pt+parval[9]*fabs(eta)+parval[10]*pow(eta,2) );
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[11]+parval[12]/pt+parval[13]*fabs(eta)+parval[14]*pow(eta,2) );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const vector<int> & parResolOrder, const int muonType) {
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
    return( parval[0]+parval[1]*pt + parval[2]*fabs(eta)+parval[3]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[4]+parval[5]/pt + parval[6]*fabs(eta)+parval[7]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*fabs(eta)+parval[11]*pow(eta,2) );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const vector<int> & parResolOrder, const int muonType) {
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
      double thisMaxi[] = { 0.1, 0.01, 0.01, 0.01,
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
    return( parval[4]+parval[5]/pt + parval[6]*fabs(eta)+parval[7]*eta*eta );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*fabs(eta)+parval[11]*eta*eta );
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const T & parResol, const vector<int> & parResolOrder, const int muonType) {

    double thisStep[] = { 0.0002, 0.000002, 0.02, 0.02,
                          0.00002, 0.0002, 0.0000002, 0.00002,
                          0.00002, 0.0002, 0.00000002, 0.000002 };
    TString thisParName[] = { "Pt res. sc.", "Pt res. Pt sc.", "Pt res. Eta sc.", "Pt res. eta border",
                              "Cth res. sc.", "Cth res. 1/Pt sc.", "Cth res. Eta sc.", "Cth res. Eta^2 sc.",
                              "Phi res. sc.", "Phi res. 1/Pt sc.", "Phi res. Eta sc.", "Phi res. Eta^2 sc." };
//     double thisMini[] = {  -0.01, 0.00000001, 0.5,
//                            -0.0004, 0.003, 0.000002, 0.0004,
//                            0.0001, 0.001, -0.0000007, 0.00008 };
    double thisMini[] = {  -0.1, -0.001, 0.4, 0.01,
                           -0.001, 0.002, -0.0001, -0.0001,
                           -0.0001, 0.0005, -0.0001, -0.00001 };
//     double thisMini[] = {  -0.006, 0.00005, 0.8,
//                            -0.0004, 0.003, 0.000002, 0.0004,
//                            0.0001, 0.001, -0.0000007, 0.00008 };
    if( muonType == 1 ) {
      double thisMaxi[] = { 1., 1., 1., 1.,
                            1., 1., 1., 0.1,
                            1., 1., 1., 1. };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
//      double thisMaxi[] = { 0.1, 0.0004, 1.2,
//                            -0.0002, 0.005, 0.000004, 0.0007,
//                            0.0003, 0.003, -0.0000011, 0.00012 };
      double thisMaxi[] = { 0.1, 0.001, 1.5, 1.,
                            0.001, 0.005, 0.00004, 0.0007,
                            0.001, 0.01, -0.0000015, 0.0004 };
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
protected:
  /**
   * This is the pt vs eta resolution by points. It uses fabs(eta) assuming symmetry.
   * The values are derived from 100k events of MuonGun with 5<pt<100 and |eta|<3.
   */
  double etaByPoints(const double & inEta, const double & border) {
    Double_t eta = fabs(inEta);
//     if( 0. <= eta && eta <= 0.2 )      return 0.0120913;
//     else if( 0.2 < eta && eta <= 0.4 ) return 0.0122204;
//     else if( 0.4 < eta && eta <= 0.6 ) return 0.0136937;
//     else if( 0.6 < eta && eta <= 0.8 ) return 0.0142069;
//     else if( 0.8 < eta && eta <= 1.0 ) return 0.0177526;
//     else if( 1.0 < eta && eta <= 1.2 ) return 0.0243587;
//     else if( 1.2 < eta && eta <= 1.4 ) return 0.019994;
//     else if( 1.4 < eta && eta <= 1.6 ) return 0.0185132;
//     else if( 1.6 < eta && eta <= 1.8 ) return 0.0177141;
//     else if( 1.8 < eta && eta <= 2.0 ) return 0.0211577;
//     else if( 2.0 < eta && eta <= 2.2 ) return 0.0255051;
//     else if( 2.2 < eta && eta <= 2.4 ) return 0.0338104;
//     // ATTENTION: This point has a big error and it is very displaced from the rest of the distribution.
//     else if( 2.4 < eta && eta <= 2.6 ) return 0.31;
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
    // else if( 2.4 < eta && eta <= 2.6 ) return 0.445473;
    else if( 2.4 < eta && eta <= 2.6 ) return border;
    return ( 0. );
  }
};

// ------------ ATTENTION ------------ //
// Other functions are not put for now //
// ----------------------------------- //

/// Service to build the resolution functor corresponding to the passed identifier
resolutionFunctionBase<double *> * resolutionFunctionService( const int identifier );

/// Service to build the resolution functor corresponding to the passed identifier when receiving a vector<double>
resolutionFunctionBase<vector<double> > * resolutionFunctionVecService( const int identifier );

// // Defined globally...
// static resolutionFunctionBase<double *> * resolutionFunctionArray[] = {
//   0,
//   new resolutionFunctionType1<double *>,
//   0,
//   0,
//   0,
//   0,
//   new resolutionFunctionType6<double *>,
//   new resolutionFunctionType7<double *>,
//   new resolutionFunctionType8<double *>
// };

// static resolutionFunctionBase<vector<double> > * resolutionFunctionArrayForVec[] = {
//   0,
//   new resolutionFunctionType1<vector<double> >,
//   0,
//   0,
//   0,
//   0,
//   new resolutionFunctionType6<vector<double> >,
//   new resolutionFunctionType7<vector<double> >,
//   new resolutionFunctionType8<vector<double> >
// };

/**
 * Background functors.
 */
class backgroundFunctionBase {
 public:
  virtual ~backgroundFunctionBase() {};
  virtual double operator()( const double * parval, const int resTotNum, const int nres, const bool * resConsidered,
                             const double * ResMass, const double ResHalfWidth[][3], const int MuonType, const double & mass, const int nbins ) = 0;
  virtual int parNum() const { return parNum_; }
  /// This method is used to differentiate parameters among the different functions
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType) = 0;
  virtual void setLeftWindowFactor(const double & leftWindowFactor) { leftWindowFactor_ = leftWindowFactor; }
  virtual void setRightWindowFactor(const double & rightWindowFactor) { rightWindowFactor_ = rightWindowFactor; }
protected:
  int parNum_;
  double leftWindowFactor_;
  double rightWindowFactor_;
  /// This method sets the parameters
  virtual void setPar(double* Start, double* Step, double* Mini, double* Maxi, int* ind,
                      TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder,
                      double* thisStep, double* thisMini, double* thisMaxi, TString* thisParName ) {
    for( int iPar=0; iPar<this->parNum_; ++iPar ) {
      Start[iPar] = parBgr[iPar];
      Step[iPar] = thisStep[iPar];
      Mini[iPar] = thisMini[iPar];
      Maxi[iPar] = thisMaxi[iPar];
      ind[iPar] = parBgrOrder[iPar];
      parname[iPar] = thisParName[iPar];
    }
  }
};
/// Constant
// ---------
class backgroundFunctionType1 : public backgroundFunctionBase {
 public:
  /**
   * This is a constant normalized to unity in the span of the window (1000 bins in mass)
   * NB: wherever there are more than a single resonance contributing, the background fraction
   * gets multiplied by the number of signals. This allows to have the same normalization 
   * throughout the spectrum: the background fraction (Bgrp1 in this fit) will represent 
   * the right fraction overall. This is because where two resonances overlap their windows
   * a given background fraction will contribute only half to Bgrp1.
   */
  backgroundFunctionType1() { this->parNum_ = 1; }
  virtual double operator()( const double * parval, const int resTotNum, const int nres, const bool * resConsidered,
                             const double * ResMass, const double ResHalfWidth[][3], const int MuonType, const double & mass, const int nbins ) {
    return( nres/(double)nbins ); 
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType) {
    double thisStep[] = {0.1};
    TString thisParName[] = {"Bgr fraction"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0};
      double thisMaxi[] = {1.0};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0};
      double thisMaxi[] = {1.0};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
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
  backgroundFunctionType2() { this->parNum_ = 2; }
  virtual double operator()( const double * parval, const int resTotNum, const int nres, const bool * resConsidered,
                             const double * ResMass, const double ResHalfWidth[][3], const int MuonType, const double & mass, const int nbins ) {
    double PB = 0.;
    double Bgrp2 = parval[1];
    for (int ires=0; ires<resTotNum; ires++) {
      if (resConsidered[ires]) {

        PB += Bgrp2*exp(-Bgrp2*mass) * (2*ResHalfWidth[ires][MuonType])/(double)nbins;

        // 	if (exp(-Bgrp2*(ResMass[ires]-leftWindowFactor_*ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+rightWindowFactor_*ResHalfWidth[ires][MuonType]))>0) {
        // 	  PB += Bgrp2*exp(-Bgrp2*mass) * 
        // 	    (2*ResHalfWidth[ires][MuonType])/(double)nbins /
        // 	    (exp(-Bgrp2*(ResMass[ires]-leftWindowFactor_*ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+rightWindowFactor_*ResHalfWidth[ires][MuonType])));
        // 	} else {
        // 	  cout << "Impossible to compute Background probability! - some fix needed - Bgrp2=" << Bgrp2 << endl;  
        // 	}
        //         cout << "exp(-Bgrp2*(ResMass[ires]-ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+ResHalfWidth[ires][MuonType])) = "
        //              << exp(-Bgrp2*(ResMass[ires]-leftWindowFactor_*ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+rightWindowFactor_*ResHalfWidth[ires][MuonType])) << endl;

        // cout << "Bgrp2 = " << Bgrp2 << endl;
      }
    }
    return PB;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType) {
    double thisStep[] = {0.01, 0.001};
    TString thisParName[] = {"Bgr fraction", "Bgr slope"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.0};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.0};
      double thisMaxi[] = {1.0, 10.};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};
/// Constant + Exponential
// -------------------------------------------------------------------------------- //
// ATTENTION: TODO: the normalization must be adapted to the asymmetric mass window //
// -------------------------------------------------------------------------------- //

class backgroundFunctionType3 : public backgroundFunctionBase {
 public:
  // pass parval[shift]
  backgroundFunctionType3() { this->parNum_ = 3; }
  virtual double operator()( const double * parval, const int resTotNum, const int nres, const bool * resConsidered,
                             const double * ResMass, const double ResHalfWidth[][3], const int MuonType, const double & mass, const int nbins ) {
    double PB = 0.;
    double Bgrp2 = parval[1];
    double Bgrp3 = parval[2];
    for (int ires=0; ires<resTotNum; ires++) {
      // In this case, by integrating between A and B, we get for f=exp(a-bx)+k:
      // INT = exp(a)/b*(exp(-bA)-exp(-bB))+k*(B-A) so our function, which in 1000 bins between A and B
      // gets a total of 1, is f = (exp(a-bx)+k)*(B-A)/nbins / (INT)
      // ----------------------------------------------------------------------------------------------
      if (resConsidered[ires]) {
	if (exp(-Bgrp2*(ResMass[ires]-ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+ResHalfWidth[ires][MuonType]))>0) {
	  PB += (exp(-Bgrp2*mass)+Bgrp3) *
	    2*ResHalfWidth[ires][MuonType]/(double)nbins / 
	    ( (exp(-Bgrp2*(ResMass[ires]-ResHalfWidth[ires][MuonType]))-exp(-Bgrp2*(ResMass[ires]+ResHalfWidth[ires][MuonType])))/
	      Bgrp2 + Bgrp3*2*ResHalfWidth[ires][MuonType] );
	} else {
	  cout << "Impossible to compute Background probability! - some fix needed - Bgrp2=" << Bgrp2 << endl;  
	}
      }
    }
    return PB;
  }
  virtual void setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType) {
    double thisStep[] = {0.1, 0.001, 0.1};
    TString thisParName[] = {"Bgr fraction", "Bgr slope", "Bgr constant"};
    if( muonType == 1 ) {
      double thisMini[] = {0.0, 0.000000001, 0.0};
      double thisMaxi[] = {1.0, 0.2, 1000};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
    } else {
      double thisMini[] = {0.0, 0.000000001, 0.0};
      double thisMaxi[] = {1.0, 0.2, 1000};
      this->setPar( Start, Step, Mini, Maxi, ind, parname, parBgr, parBgrOrder, thisStep, thisMini, thisMaxi, thisParName );
    }
  }
};

/// Service to build the background functor corresponding to the passed identifier
backgroundFunctionBase * backgroundFunctionService( const int identifier );

// static backgroundFunctionBase * backgroundFunctionArray[] = {
//   new backgroundFunctionType1,
//   new backgroundFunctionType2,
//   new backgroundFunctionType3
// };

#endif // FUNCTIONS_H
