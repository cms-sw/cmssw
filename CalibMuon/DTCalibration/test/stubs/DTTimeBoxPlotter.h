#ifndef DTTimeBoxPlotter_H
#define DTTimeBoxPlotter_H

/** \class DTTimeBoxPlotter
 *  Utility class to plot TimeBoxes produced with DTTTrigCalibration.
 *  The time box rising edge can be fitted using exactly the same code that is
 *  used by the calibration application.
 *
 *  $Date: 2007/05/15 14:43:20 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include "TString.h"


class TFile;
class TCanvas;
class TH1F;
class TH2F;
class DTTimeBoxFitter;

class DTTimeBoxPlotter {
public:
  /// Constructor
  DTTimeBoxPlotter(TFile *file);


  /// Destructor
  virtual ~DTTimeBoxPlotter();

  // Operations

  /// Plot the time box of a given chamber.
  /// Options: "same" -> histo drawn in the active canvas, "fit" -> fit the rising edge of the time box
  TH1F* plotTimeBox(int wheel, int station, int sector,
		    const TString& drawOptions = "");
  /// Plot the time box of a given superlayer.
  /// Options: "same" -> histo drawn in the active canvas, "fit" -> fit the rising edge of the time box
  TH1F* plotTimeBox(int wheel, int station, int sector, int sl,
		    const TString& drawOptions = "");
  /// Plot the time box of a given layer.
  /// Options: "same" -> histo drawn in the active canvas, "fit" -> fit the rising edge of the time box
  TH1F* plotTimeBox(int wheel, int station, int sector, int sl, int layer,
		    const TString& drawOptions = "");
  /// Plot the time box of a given wire.
  /// Options: "same" -> histo drawn in the active canvas, "fit" -> fit the rising edge of the time box
  TH1F* plotTimeBox(int wheel, int station, int sector, int sl, int layer, int wire,
		    const TString& drawOptions = "");

  /// Print all canvases in a pdf file.
  void printPDF();

  /// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
  void setVerbosity(unsigned int lvl);
  void setInteractiveFit(bool isInteractive);
  void setRebinning(int rebin);

protected:

private:
  TString getHistoNameSuffix(int wheel, int station, int sector);
  TString getHistoNameSuffix(int wheel, int station, int sector, int sl);
  TString getHistoNameSuffix(int wheel, int station, int sector, int sl, int layer);
  TString getHistoNameSuffix(int wheel, int station, int sector, int sl, int layer, int wire);

  TH1F* plotHisto(const TString& histoName, const TString& drawOptions = "");
  TH2F* plotHisto2D(const TString& histoName, const TString& drawOptions = "");

  TCanvas * newCanvas(TString name="",
		      TString title="",
		      int xdiv=0,
		      int ydiv=0,
		      int form = 1,
		      int w=-1);

  TCanvas * newCanvas(TString name, int xdiv, int ydiv, int form, int w);
  TCanvas * newCanvas(int xdiv, int ydiv, int form = 1);
  TCanvas * newCanvas(int form = 1);
  TCanvas * newCanvas(TString name, int form, int w=-1);

  DTTimeBoxFitter *theFitter;
  TFile *theFile;
  unsigned int theVerbosityLevel;
};
#endif
