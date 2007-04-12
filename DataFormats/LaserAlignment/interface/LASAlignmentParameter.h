#ifndef DataFormats_LaserAlignment_LASAlignmentParameter_h
#define DataFormats_LaserAlignment_LASAlignmentParameter_h

/** \class LASAlignmentParameter
 *  Store the calculated alignment parameters for the laser alignment system. Available are
 *  methods to access
 *  - \f$ \Delta\varphi_0 \f$
 *  - \f$ \Delta\varphi_t \f$
 *  - \f$ \Delta\varphi_k \f$
 *  - \f$ \Delta x_0 \f$
 *  - \f$ \Delta x_t \f$
 *  - \f$ \Delta x_k \f$
 *  - \f$ \Delta y_0 \f$
 *  - \f$ \Delta y_t \f$
 *  - \f$ \Delta y_k \f$
 *
 *  $Date: 2007/04/05 13:19:24 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include <iostream>
#include <valarray>

class LASAlignmentParameter
{
 public:
	typedef std::valarray<double> LASvec;
	/// default constructor
  LASAlignmentParameter() : name_(), dphi0_(0), dphit_(0), dphik_(),
    dx0_(0), dxt_(0), dxk_(), dy0_(0), dyt_(0), dyk_() {}
	/// constructor
  LASAlignmentParameter(std::string name, double dphi0, double dphit, LASvec dphik, 
		    double dx0, double dxt, LASvec dxk, double dy0, double dyt, LASvec dyk) : name_(name), 
    dphi0_(dphi0), dphit_(dphit), dphik_(dphik), dx0_(dx0), dxt_(dxt), 
    dxk_(dxk), dy0_(dy0), dyt_(dyt) {}

  // Access to the information from the fit
	/// get the name of the current set of parameters
  std::string name() const { return name_; }
	/// get \f$ \Delta\varphi_0 \f$
  double dphi0() const { return dphi0_; }
	/// get \f$ \Delta\varphi_t \f$
  double dphit() const { return dphit_; }
	/// get \f$ \Delta\varphi_k \f$
  LASvec dphik() const { return dphik_; }
	/// get \f$ \Delta x_0 \f$
  double dx0() const { return dx0_; }
	/// get \f$ \Delta x_t \f$
  double dxt() const { return dxt_; }
	/// get \f$ \Delta x_k \f$
	LASvec dxk() const { return dxk_; }
	/// get \f$ \Delta y_0 \f$
  double dy0() const { return dy0_; }
	/// get \f$ \Delta y_t \f$
	double dyt() const { return dyt_; }
	/// get \f$ \Delta y_k \f$
	LASvec dyk() const { return dyk_; }

 private:
  std::string name_;
  double dphi0_;
  double dphit_;
  LASvec dphik_;
  double dx0_;
  double dxt_;
	LASvec dxk_;
  double dy0_;
  double dyt_;
	LASvec dyk_;
};

#endif

