#ifndef CalibMuon_DTTimeBoxFitter_H
#define CalibMuon_DTTimeBoxFitter_H

/** \class DTTimeBoxFitter
 *  Fit the rising edge of the time box with the integral
 *  of a gaussian returning the mean value and the sigma.
 *
 *  $Date: 2006/07/21 12:26:31 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include <utility>


class TH1F;
class TFile;

class DTTimeBoxFitter {
public:
  /// Constructor
  DTTimeBoxFitter();

  /// Destructor
  virtual ~DTTimeBoxFitter();

  // Operations

  /// Fit the rising edge of the time box returning mean value and sigma (first and second respectively)
  std::pair<double, double> fitTimeBox(TH1F *hTimeBox);

  
  /// Automatically compute the seeds the range to be used for time box fit
  void getFitSeeds(TH1F *hTBox, double& mean, double& sigma, double& tBoxMax, double& xFitMin,
		   double& xFitMax);

  /// Ask the user to provide the seeds
  void getInteractiveFitSeeds(TH1F *hTBox, double& mean, double& sigma, double& tBoxMax,
			      double& xFitMin, double& xFitMax);


  /// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
  void setVerbosity(unsigned int lvl) {
    theVerbosityLevel = lvl;
  }

  /// Switch to interactive fit
  void setInteractiveFit(bool isInteractive) {
    interactiveFit = isInteractive;
  }

protected:

private:

  TFile *hDebugFile;

  unsigned int theVerbosityLevel;
  bool interactiveFit;
};

// Define the integral of the gaussian to be used in the fit
double intGauss(double *x, double *par);

#endif
