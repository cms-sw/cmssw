#ifndef DTMeanTimerPlotter_H
#define DTMeanTimerPlotter_H

/** \class DTMeanTimerPlotter
 *  Utility class to plot MeanTimer produced with DTVDriftCalibration.
 *  The tmax histograms can be fitted using exactly the same code that is
 *  used by the calibration application.
 *
 *  $Date: 2007/01/22 11:11:10 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "TString.h"
#include <vector>


class TFile;
class TCanvas;
class TH1D;
class TF1;

class DTMeanTimerPlotter {
public:
  /// Constructor
  DTMeanTimerPlotter(TFile *file);


  /// Destructor
  virtual ~DTMeanTimerPlotter();

  // Operations

  /// Plot the time box of a given superlayer.
  /// Options: <br> 
  /// "same" -> histo drawn in the active canvas, <br>
  /// "SingleDeltaT0" or "SingleFormula" -> fit and draw each TMax histo separately (in the same canvas) <br>
  /// "fit" -> draw and fit the histos <br>
  void plotMeanTimer(int wheel, int station, int sector, int sl,
		    const TString& drawOptions = "");
 
   /// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
  void setVerbosity(unsigned int lvl);

  /// Set rebin number
  void setRebinning(unsigned int rebin);

  /// Reset the counter for histos color
  void resetColor();

protected:

private:

  //Plot the TMax histogram for each deltaT0: THESE ARE NOT USED TO COMPUTE VDRIFT
  std::vector<TH1D*>  plotSingleTMaxDeltaT0 (TString& name);
  //Plot the TMax histogram for each formula
  std::vector<TH1D*> plotSingleTMaxFormula (TString& name);
  //Plot the total TMax histograms: THIS IS NOT USED TO COMPUTE VDRIFT
  std::vector<TH1D*> plotTotalTMax (TString& name);

  void plotHistos(std::vector<TH1D*> hTMaxes, TString& name, const TString& drawOptions);
  TString getHistoNameSuffix(int wheel, int station, int sector, int sl);
  std::vector<TF1*> fitTMaxes(std::vector<TH1D*> histo);
  double getMaximum(std::vector<TH1D*> hTMaxes);

  TFile *theFile;
  unsigned int theVerbosityLevel;
  unsigned int theRebinning;
  unsigned int color;
};
#endif
