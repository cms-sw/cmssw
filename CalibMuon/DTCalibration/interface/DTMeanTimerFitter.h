#ifndef CalibMuon_DTMeanTimerFitter_H
#define CalibMuon_DTMeanTimerFitter_H

/** \class DTMeanTimerFitter
 *  Fit the Tmax histograms with a gaussian 
 *  returning the mean values and the sigmas.
 *
 *  $Date: 2007/01/22 11:10:27 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi - INFN Torino
 */
#include <vector> 
#include "TString.h"

class TH1F;
class TFile;
class TF1;



class DTMeanTimerFitter {
public:
  /// Constructor
  DTMeanTimerFitter(TFile *file);

  /// Destructor
  virtual ~DTMeanTimerFitter();

  /// Fit the TMax histos and evaluate VDrift and resolution
  std::vector<float> evaluateVDriftAndReso (TString N);

  /// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
  void setVerbosity(unsigned int lvl) {
    theVerbosityLevel = lvl;
  }

  /// Really do the fit
  TF1* fitTMax(TH1F* histo);
protected:

private:

  TFile *hDebugFile;
  TFile *hInputFile;

  unsigned int theVerbosityLevel;
};

#endif

