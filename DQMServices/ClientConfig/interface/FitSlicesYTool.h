#ifndef FitSlicesYTool_h
#define FitSlicesYTool_h

/** \class FitSlicesYTool
 *  Class to fill Monitor Elements using the ROOT FitSlicesY tool
 *
 *  $Date: 2009/03/27 00:16:49 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include <TH2F.h>
#include <TH1F.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <string>

class FitSlicesYTool {
 public:
  FitSlicesYTool(MonitorElement*);
  /// Constructor: needs a TH2F
  /*   FitSlicesYTool(TH2F*); */
  /// Destructor
  ~FitSlicesYTool();
  /// Fill the ME with the mean value of the gaussian fit in each slice
  void getFittedMean(MonitorElement*);
  /// Fill the ME with the sigma value of the gaussian fit in each slice
  void getFittedSigma(MonitorElement*);
  /// Fill the ME with the mean value (with error) of the gaussian fit in each slice
  void getFittedMeanWithError(MonitorElement*);
  /// Fill the ME with the sigma value (with error) of the gaussian fit in each slice
  void getFittedSigmaWithError(MonitorElement*);
 private:
  TH1* h0;
  TH1* h1;
  TH1* h2;
  TH1* h3;
};

#endif
