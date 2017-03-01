#ifndef ApeOverview_h
#define ApeOverview_h

#include <vector>
#include <map>

#include "TString.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"




class ApeOverview{
   public:
      ApeOverview(const TString inputFileName);
      ~ApeOverview();
      
      void whichModuleInFile(int);  // if several modules were registered in TFileService, give nr. of wanted one (alphabetical order)
      
      void onlyZoomedHists();  // if wanted, has to be set before getOverview()
      
      void setSectorsForOverview(const TString& sectors);  // comma separated list; if wanted, has to be set before getOverview()
      
      void getOverview();
      
      enum HistLevel{event, track, sector};
      void printOverview(const TString& outputFileName = "apeOverview.ps", const HistLevel& histLevel = ApeOverview::event);  //ApeOverview::event, ApeOverview::track, ApeOverview::sector
      
   private:
      
      TString setCanvasName()const;
      
      void eventAndTrackHistos();
      
      int drawHistToPad(const TString histName, const bool setLogScale = true);
      
      
      enum PlotDimension{dim1,dim2};
      int setNewCanvas(const PlotDimension& pDim);
      
      
      
      // --------------------------------------------------- member data ---------------------------------------------------
      
      TFile* inputFile_;
      
      int moduleNo_;
      
      bool onlyZoomedHists_;
      
      std::vector<unsigned int> vSelectedSector_;
      
      TString firstSelectedSector_;
      
      TString pluginDir_, histDir_;
      
      HistLevel histLevel_;
      
      typedef std::pair<unsigned int, unsigned int> PadCounterPair;
      PadCounterPair eventPadCounter_, trackPadCounter_;
      std::map<unsigned int, PadCounterPair> mSectorPadCounter_;
      
      unsigned int sectorCounter_;
      
      typedef std::pair<std::vector<TCanvas*>, std::vector<TCanvas*> > CanvasPair;  //contain (1DHists, 2DAndProfileHists)
      CanvasPair eventPair_, trackPair_;
      std::map<unsigned int,CanvasPair> mSectorPair_;
      
};


#endif
