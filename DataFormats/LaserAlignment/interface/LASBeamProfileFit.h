/*                               -*- Mode: C -*- 
 * LASBeamProfileFit.h --- this product contains the results of the Minuit
 *                         fit of the Laser Beam Profiles in the Tracker
 * Author          : Maarten Thomas
 * Created On      : Tue Apr  4 16:44:47 2006
 * Last Modified By: Maarten Thomas
 * Last Modified On: Tue Sep 26 15:43:36 2006
 * Update Count    : 16
 * Status          : Unknown, Use with caution!
 */

/**
 * The results of the fit of the Laser Beam Profiles in the Tracker. It has
 * methods to access
 * - fitted mean (strip number) both corrected and uncorrected for the Beamsplitter kink
 * - error of the mean
 * - sigma (in strips)
 * - error of sigma
 * - phi position (calculated from the fitted and corrected mean)
 * - phi postion error
 * - name of beam and layer to identify the beam profile
 */

#ifndef DataFormats_LaserAlignment_LASBeamProfileFit_h
#define DataFormats_LaserAlignment_LASBeamProfileFit_h

#include <iostream>

class LASBeamProfileFit
{
 public:
  LASBeamProfileFit() : name_(), mean_(0), meanError_(0), uncorrectedMean_(0),
    sigma_(0), sigmaError_(0), phi_(0), phiError_(0) {}

  LASBeamProfileFit(const char * name, double mean, double meanError, double sigma, double sigmaError) : name_(name),
    mean_(mean), meanError_(meanError), uncorrectedMean_(mean), sigma_(sigma), sigmaError_(sigmaError), 
    phi_(0), phiError_(0) {}

  LASBeamProfileFit(const char * name, double mean, double meanError, double uncorMean, 
		    double sigma, double sigmaError, double phi, double phiError) : name_(name), 
    mean_(mean), meanError_(meanError), uncorrectedMean_(uncorMean), sigma_(sigma), sigmaError_(sigmaError), 
    phi_(phi), phiError_(phiError) {}


  // Access to the information from the fit
  std::string name() const { return name_; }
  double mean() const { return mean_; }
  double meanError() const { return meanError_; }
  double uncorrectedMean() const { return uncorrectedMean_; }
  double sigma() const { return sigma_; }
  double sigmaError() const { return sigmaError_; }
  double phi() const { return phi_; }
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

