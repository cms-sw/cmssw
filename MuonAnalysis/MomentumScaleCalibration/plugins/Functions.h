#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>

using namespace std;

// ----------------------- //
// Bias and scale functors //
// ----------------------- //

/** The correct functor is selected at job start in the constructor.
 * The pt value is taken by reference and modified internally.
 * eta, phi and chg are taken by const reference.</br>
 * Made into a template so that it can be used with arrays too
 * (parval for the scale fit is an array, because Lykelihood is an
 * extern C function, because TMinuit asks it).</br>
 * Note that in the array case it takes the pointer by const reference,
 * thus the elements of the array are modifiable.
 */
template <class T>
class scaleFunctionBase {
 public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) = 0;
  virtual ~scaleFunctionBase() = 0;
  // Possibly use this to set the parameters for the fit
  // virtual void setScaleFitParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname) = 0;
};
template <class T> inline scaleFunctionBase<T>::~scaleFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.
// No scale
// --------
template <class T>
class scaleFunctionType0 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {}
  // virtual void setScaleFitParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname) {}
};
// Linear in pt
// ------------
template <class T>
class scaleFunctionType1 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt)*pt;
  }
//   virtual void setScaleFitParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname) {
//     Start[0]   = parScale[0]; // 1.0
//     Start[1]   = parScale[1]; // 0.0
//     Step[0]    = 0.001;
//     Step[1]    = 0.01;
//     Mini[0]    = 0.97;
//     Mini[1]    = -0.1;
//     Maxi[0]    = 1.03;
//     Maxi[1]    = 0.1;
//     ind[0]     = parScaleOrder[0];
//     ind[1]     = parScaleOrder[1];
//     parname[0] = "Pt offset";
//     parname[1] = "Pt slope";
//   }
};
// Linear in |eta|
// ---------------
template <class T>
class scaleFunctionType2 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*fabs(eta))*pt; 
  }
};
// Sinusoidal in phi
// -----------------
template <class T>
class scaleFunctionType3 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*sin(phi))*pt; 
  }
};
// Linear in pt and |eta|
// ----------------------
template <class T>
class scaleFunctionType4 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*fabs(eta))*pt;
  }
};
// Linear in pt and sinusoidal in phi
// ----------------------------------
template <class T>
class scaleFunctionType5 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*sin(phi))*pt;
  }
};
// Linear in |eta| and sinusoidal in phi
// -------------------------------------
template <class T>
class scaleFunctionType6 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*fabs(eta) + 
	  parScale[2]*sin(phi))*pt;
  }
};
// Linear in pt and |eta| and sinusoidal in phi
// --------------------------------------------
template <class T>
class scaleFunctionType7 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*fabs(eta) + 
	  parScale[3]*sin(phi))*pt;
  }
};
// Linear in pt and parabolic in |eta|
// -----------------------------------
template <class T>
class scaleFunctionType8 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*fabs(eta) +
	  parScale[3]*eta*eta)*pt;
  }
};
// Exponential in pt
// -----------------
template <class T>
class scaleFunctionType9 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + exp(parScale[1]*pt))*pt;
  }
};
// Parabolic in pt
// ---------------
template <class T>
class scaleFunctionType10 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*pt*pt)*pt;
  }
};
// Linear in pt, sinusoidal in phi with muon sign
// ----------------------------------------------
template <class T>
class scaleFunctionType11 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    pt = (parScale[0] + parScale[1]*pt + 
	  (double)chg*parScale[2]*sin(phi+parScale[3]))*pt;
  }
};
// Linear in pt, parabolic in eta, sinusoidal in phi with muon sign
// ---------------------------------------------------------------- 
template <class T>
class scaleFunctionType12 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
   pt = (parScale[0] + parScale[1]*pt + 
	  parScale[2]*fabs(eta) +
	  parScale[3]*eta*eta + 
	  (double)chg*parScale[4]*sin(phi+parScale[5]))*pt;
  }
};
// Linear in pt, parabolic in eta, sinusoidal in phi with muon sign
// ----------------------------------------------------------------
template <class T>
class scaleFunctionType13 : public scaleFunctionBase<T> {
public:
  virtual void scale(double & pt, const double & eta, const double & phi, const int chg, const T & parScale) {
    if (chg>0) {
      pt = (parScale[0] + parScale[1]*pt + 
	    parScale[2]*fabs(eta) +
	    parScale[3]*eta*eta + 
	    parScale[4]*sin(phi+parScale[5]))*pt;
    } else {
      pt = (parScale[0] + parScale[1]*pt + 
	    parScale[2]*fabs(eta) +
	    parScale[3]*eta*eta + 
	    parScale[6]*sin(phi+parScale[7]))*pt;
    }
  }
};

// Bias functions, they get vector<double>
static scaleFunctionBase<vector<double> > * biasFunctionArray[] = {
  new scaleFunctionType0<vector<double> >,
  new scaleFunctionType1<vector<double> >,
  new scaleFunctionType2<vector<double> >,
  new scaleFunctionType3<vector<double> >,
  new scaleFunctionType4<vector<double> >,
  new scaleFunctionType5<vector<double> >,
  new scaleFunctionType6<vector<double> >,
  new scaleFunctionType7<vector<double> >,
  new scaleFunctionType8<vector<double> >,
  new scaleFunctionType9<vector<double> >,
  new scaleFunctionType10<vector<double> >,
  new scaleFunctionType11<vector<double> >,
  new scaleFunctionType12<vector<double> >,
  new scaleFunctionType13<vector<double> >
};

// Scale functions, they get arrays of doubles
static scaleFunctionBase<double* > * scaleFunctionArray[] = {
  new scaleFunctionType0<double* >,
  new scaleFunctionType1<double* >,
  new scaleFunctionType2<double* >,
  new scaleFunctionType3<double* >,
  new scaleFunctionType4<double* >,
  new scaleFunctionType5<double* >,
  new scaleFunctionType6<double* >,
  new scaleFunctionType7<double* >,
  new scaleFunctionType8<double* >,
  new scaleFunctionType9<double* >,
  new scaleFunctionType10<double* >,
  new scaleFunctionType11<double* >,
  new scaleFunctionType12<double* >,
  new scaleFunctionType13<double* >
};

// -------------- //
// Smear functors //
// -------------- //

class smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) = 0;
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
  virtual void smear(double & pt, double & eta, double & phi, const double * y) { }
};
// The 3 parameters of smearType1 are: pt dependence of pt smear, phi smear and
// cotgtheta smear.
class smearFunctionType1 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) {
    pt = pt*(1.0+y[0]*MuScleFitUtils::parSmear[0]*pt);
    phi = phi*(1.0+y[1]*MuScleFitUtils::parSmear[1]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[2]*MuScleFitUtils::parSmear[2]);
    smearEta(eta);
  }
};

class smearFunctionType2 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) {
    pt = pt*(1.0+y[0]*MuScleFitUtils::parSmear[0]*pt+y[1]*MuScleFitUtils::parSmear[1]*fabs(eta));
    phi = phi*(1.0+y[2]*MuScleFitUtils::parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*MuScleFitUtils::parSmear[3]);
    smearEta(eta);
  }
};

class smearFunctionType3 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) {
    pt = pt*(1.0+y[0]*MuScleFitUtils::parSmear[0]*pt+y[1]*MuScleFitUtils::parSmear[1]*fabs(eta));
    phi = phi*(1.0+y[2]*MuScleFitUtils::parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*MuScleFitUtils::parSmear[3]+y[4]*MuScleFitUtils::parSmear[4]*fabs(eta));
    smearEta(eta);
  }
};
// The six parameters of SmearType=4 are respectively:
// Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
// |eta| dep. of |eta| res., Pt^2 dep. of Pt res.
class smearFunctionType4 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) {
    pt = pt*(1.0+y[0]*MuScleFitUtils::parSmear[0]*pt+y[1]*MuScleFitUtils::parSmear[1]*fabs(eta)+y[5]*MuScleFitUtils::parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*MuScleFitUtils::parSmear[2]);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*MuScleFitUtils::parSmear[3]+y[4]*MuScleFitUtils::parSmear[4]*fabs(eta));
    smearEta(eta);
  }
};

class smearFunctionType5 : public smearFunctionBase {
 public:
  virtual void smear(double & pt, double & eta, double & phi, const double * y) {
    pt = pt*(1.0+y[0]*MuScleFitUtils::parSmear[0]*pt+y[1]*MuScleFitUtils::parSmear[1]*fabs(eta)+y[5]*MuScleFitUtils::parSmear[5]*pow(pt,2));
    phi = phi*(1.0+y[2]*MuScleFitUtils::parSmear[2]+y[6]*MuScleFitUtils::parSmear[6]*pt);
    double tmp = 2*atan(exp(-eta));
    cotgth_ = cos(tmp)/sin(tmp)*(1.0+y[3]*MuScleFitUtils::parSmear[3]+y[4]*MuScleFitUtils::parSmear[4]*fabs(eta));
    smearEta(eta);
  }
};

// Defined globally...
static smearFunctionBase * smearFunctionArray[] = {
  new smearFunctionType0,
  new smearFunctionType1,
  new smearFunctionType2,
  new smearFunctionType3,
  new smearFunctionType4,
  new smearFunctionType5
};

/**
 * Resolution functions. </br>
 * Need to use templates to make it works with both array and vector<double>.
 */
template <class T>
class resolutionFunctionBase {
 public:
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) = 0;
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) = 0;
  resolutionFunctionBase() {}
  virtual ~resolutionFunctionBase() = 0;
};
template <class T> inline resolutionFunctionBase<T>::~resolutionFunctionBase() { }  // defined even though it's pure virtual; should be faster this way.

// Resolution Type 1
template <class T>
class resolutionFunctionType1 : public resolutionFunctionBase<T> {
 public:
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return parval[0];
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return parval[1];
  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return parval[2];
  }
};

// Resolution Type 6
template <class T>
class resolutionFunctionType6 : public resolutionFunctionBase<T> {
 public:
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt+parval[2]*pow(pt,2)+parval[3]*pow(pt,3)+parval[4]*pow(pt,4)+parval[5]*fabs(eta)+parval[6]*pow(eta,2) );
  }
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[11]+parval[12]/pt+parval[13]*fabs(eta)+parval[14]*pow(eta,2) );

  }
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[7]+parval[8]/pt+parval[9]*fabs(eta)+parval[10]*pow(eta,2) );
  }
};

// Resolution Type 7
template <class T>
class resolutionFunctionType7 : public resolutionFunctionBase<T> {
 public:
  // linear in pt and quadratic in eta
  virtual double sigmaPt(const double & pt, const double & eta, const T & parval) {
    return( parval[0]+parval[1]*pt + parval[2]*fabs(eta)+parval[3]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaCotgTh(const double & pt, const double & eta, const T & parval) {
    return( parval[8]+parval[9]/pt + parval[10]*fabs(eta)+parval[11]*pow(eta,2) );
  }
  // 1/pt in pt and quadratic in eta
  virtual double sigmaPhi(const double & pt, const double & eta, const T & parval) {
    return( parval[4]+parval[5]/pt + parval[6]*fabs(eta)+parval[7]*pow(eta,2) );
  }
};

// ------------ ATTENTION ------------ //
// Other functions are not put for now //
// ----------------------------------- //

// Defined globally...
static resolutionFunctionBase<double *> * resolutionFunctionArray[] = {
  0,
  new resolutionFunctionType1<double *>,
  0,
  0,
  0,
  0,
  new resolutionFunctionType6<double *>,
  new resolutionFunctionType7<double *>
};

static resolutionFunctionBase<vector<double> > * resolutionFunctionArrayForVec[] = {
  0,
  new resolutionFunctionType1<vector<double> >,
  0,
  0,
  0,
  0,
  new resolutionFunctionType6<vector<double> >,
  new resolutionFunctionType7<vector<double> >
};

#endif // FUNCTIONS_H
