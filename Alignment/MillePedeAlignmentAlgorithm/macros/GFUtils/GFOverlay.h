#ifndef GFUTILSGFOVERLAY_H
#define GFUTILSGFOVERLAY_H

#include "TString.h"
#include "TObjArray.h"
#include <vector>

//   Author:      Gero Flucke
//   Date:        October 2007
//   last update: $Date: 2012/03/29 08:48:50 $  
//   by:          $Author: flucke $
//
// Class to overlay hists with same names from (several) different files.
// Constructor argument 'fileLegendList' is a comma separated list of files 
// and the legend entry corresponding to it, e.g.
// "file1.root=label1,file2.root=label2,..." 
// Finds recursively all hists in all directories of the first file.
// The hists of one directory are grouped in one layer of the GFHistManager.
//
// Following options are recognised:
// 1)
// If 'option' contains 'norm', the hists are normalised to their number of entries.
// 2)
// If 'option' contains 'skip(XYZ)' or 'skipHist(XYZ)', hists are skipped if their names
// contain 'XYZ', similarly with 'skip(XYZ)' or 'skipDir(XYZ)' for directories.
// Can be given several times.
// 3)
// If 'option' contains 'name(XYZ)' or 'nameHist(XYZ)', hists are skipped if their names
// do NOT contain 'XYZ', similarly with 'name(XYZ)' or 'nameHist(XYZ)' for directories.
// Can be given several times.
// 4)
// If 'option' contains 'sumperdir', for each directory hists created containing
// the means and RMS of all (1D-)hists of that directory and puts these in the following
// 'layer' of the GFHistManager.
// 
// (Lower and upper case are ignored for 'norm', 'skip', 'name' and 'sumperdir'.)
//
// CAVEAT:
// If you want to skip all hists that contain 'Norm' in their name by option 'skip(Norm)',
// this will also lead to the normalisation option being switched on...

class TFile;
class TDirectory;
class GFHistManager;
class TH1;

class GFOverlay {
 public:
  GFOverlay(const char *fileLegendList, Option_t *option = "");
  ~GFOverlay();
  GFHistManager* GetHistManager() { return fHistMan;}

 private:
  TObjArray FindAllBetween(const TString &text, const char *startStr, const char *endStr) const;
  TString FindNextBetween(const TString &input, Ssiz_t startInd,
			  const char *startStr, const char *endStr) const;
  bool OpenFilesLegends(const char *fileLegendList);
  void Overlay(const TObjArray &dirs, const TObjArray &legends);
  bool KeyContainsListMember(const TString &key, const TObjArray &list) const;
  TObjArray GetTypeWithNameFromDirs(const TClass *aType, const char *name,
				    const TObjArray &dirs) const;
  Int_t AddHistsAt(const TObjArray &hists, const TObjArray &legends, Int_t layer,Int_t pos);
  void CreateFillMeanRms(const TObjArray &hists, Int_t layer, const char *dirName,
			 std::vector<TH1*> &meanHists, std::vector<TH1*> &rmsHists) const;
  GFHistManager *fHistMan;
  Int_t          fLayer;
  TObjArray      fFiles;
  TObjArray      fLegends;
  Bool_t         fNormalise;
  Bool_t         fSummaries;
  TObjArray      fDirNames;
  TObjArray      fSkipDirNames;
  TObjArray      fHistNames;
  TObjArray      fSkipHistNames;
};

#endif
