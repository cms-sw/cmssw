// Original Author: Gero Flucke
// last change    : $Date: 2009/01/20 18:28:38 $
// by             : $Author: flucke $
#ifndef PLOTMILLEMONITOR_H
#define PLOTMILLEMONITOR_H

#include "PlotMilleMonitor.h"

#include <Rtypes.h>
#include <TString.h>
#include <vector>
#include <utility>

class TFile;
class GFHistManager;

class PlotMilleMonitor
{
 public:
  explicit PlotMilleMonitor(const char *fileName);
  virtual ~PlotMilleMonitor();

  void DrawAllByHit(const char *xy = "X", Option_t *option = "");// option: 'add','sum','sumonly','norm','gaus'
  void DrawResidualsByHit(const char *histName, const char *xy = "X", Option_t *option = "");// histName='resid','reduResid','sigma','angle'; option: 'add','sum','sumonly','norm,'gaus'

  GFHistManager* GetHistManager() { return fHistManager;}

  TString Unique(const char *name) const;
  //  void CopyAddBinning(TString &name, const TH1 *hist) const;// extend 'name' taking binning from hist
 private: 
  Int_t PrepareAdd(bool addPlots);
  bool OpenFilesLegends(const char *fileLegendList);
  Int_t AddResidualsByHit(const char *histName, std::pair<TFile*,TString> &fileLeg,
			  Int_t layer, const char *xy, Option_t *option);


  GFHistManager *fHistManager;
  //  TFile         *fFile;
  std::vector<std::pair<TFile*, TString> > fFileLegends;
//  std::vector<TFile*>  fFiles
};

#endif
