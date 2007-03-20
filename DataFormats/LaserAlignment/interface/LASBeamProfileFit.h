#ifndef DataFormats_LaserAlignment_LASBeamProfileFit_h
#define DataFormats_LaserAlignment_LASBeamProfileFit_h

/** \class LASBeamProfileFit
 *  The results of the fit of the Laser Beam Profiles in the Tracker. It has
 *  methods to access
 *  - fitted mean (strip number) both corrected and uncorrected for the Beamsplitter kink
 *  - error of the mean
 *  - sigma (in strips)
 *  - error of sigma
 *  - phi position (calculated from the fitted and corrected mean)
 *  - phi postion error
 *  - name of beam and layer to identify the beam profile
 *
 *  $Date: Mon Mar 19 12:44:40 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include <iostream>

class LASBeamProfileFit
{
 public:
	/// default constructor
  LASBeamProfileFit() : name_(), mean_(0), meanError_(0), uncorrectedMean_(0),
    sigma_(0), sigmaError_(0), phi_(0), phiError_(0) {}
	/// constructor
  LASBeamProfileFit(const char * name, double mean, double meanError, double sigma, double sigmaError) : name_(name),
    mean_(mean), meanError_(meanError), uncorrectedMean_(mean), sigma_(sigma), sigmaError_(sigmaError), 
    phi_(0), phiError_(0) {}
	/// constructor
  LASBeamProfileFit(const char * name, double mean, double meanError, double uncorMean, 
		    double sigma, double sigmaError, double phi, double phiError) : name_(name), 
    mean_(mean), meanError_(meanError), uncorrectedMean_(uncorMean), sigma_(sigma), sigmaError_(sigmaError), 
    phi_(phi), phiError_(phiError) {}


  // Access to the information from the fit
	/// get the name of the current beam
  std::string name() const { return name_; }
	/// get mean of gauss fit
  double mean() const { return mean_; }
	/// get error on mean
  double meanError() const { return meanError_; }
	/// get mean without correction for beamsplitter kink
  double uncorrectedMean() const { return uncorrectedMean_; }
	/// get sigma of gauss fit
  double sigma() const { return sigma_; }
	/// get error on sigma
  double sigmaError() const { return sigmaError_; }
	/// get phi position of the beam
  double phi() const { return phi_; }
	/// get error on phi position
  double phiError() const { return phiError_; }

 private:
  std::string name_;
  double mean_;
  double meanError_;
  double uncorrectedMean_;
  double sigma_;
  double sigmaError_;
  double phi_;
  double phiError_;
};

#endif

