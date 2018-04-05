#ifndef __GFHISTMANAGER_H
#define __GFHISTMANAGER_H

 // ROOT includes
#include <TObject.h>
#include <TArrayI.h>
#include <TArrayC.h>
#include <TH1.h> // for type Color_t etc.

class TObjArray;
class TCollection;
// class TH1;
class TCanvas;
class TVirtualPad;
class TFile; 
class TLegend; 
class TLegendEntry; 
class GFHistArray;

class GFHistManager : public TObject{
public:
  GFHistManager();
  explicit GFHistManager(TH1* hist); 
  explicit GFHistManager(TCollection* hists); 
  virtual ~GFHistManager(); 
  
  virtual TLegendEntry* AddHist(TH1* hist, Int_t layer = 0, const char* legendTitle = NULL,
				const char* legOpt = NULL);
  virtual void AddHists(TCollection* hists, Int_t layer = 0,
			const char* legendTitle = NULL, const char* legOpt = NULL);
  virtual TLegendEntry* AddHistSame(TH1* hist, Int_t layer, Int_t nPad, 
				    const char* legendTitle=NULL, const char* legOpt=NULL);
  virtual void AddHistsSame(GFHistArray* hists, Int_t layer,
			    const char* legendTitle = NULL, const char* legOpt = NULL);
  virtual void AddHistsSame(GFHistArray* hists, Int_t layer, Int_t nPad);
  virtual void AddLayers(GFHistManager* other);
  virtual void AddLayer(GFHistManager* other, Int_t layer);
  virtual void Overlay(GFHistManager* other, Int_t otherLayer, Int_t myLayer,
		       const char* legendTitle);
  virtual TLegend* AddLegend(Int_t layer, Int_t nPad, const char* header = NULL, 
			     Bool_t referAll = kTRUE);
  virtual Bool_t RemoveLegend(Int_t layer, Int_t nPad);
  virtual void AddLegend(TLegend* leg, Int_t layer, Int_t nPad);
  virtual void AddObject(TObject* obj, Int_t layer, Int_t histoNum, Option_t* opt = NULL);

  virtual void Draw(Option_t * opt = "");
  virtual void Draw(Int_t layer);
  using TObject::Print;
  virtual void Print(const char* filename, Bool_t add = kFALSE);
//   virtual void Print(const char* filename, Int_t layer);
  virtual void Clear(Bool_t deleteHists);
  void Clear(Option_t * = "") {this->Clear(kFALSE);}
  void Delete(Option_t * = "") {this->Clear(kTRUE);}
  virtual void SetLegendX1Y1X2Y2(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
  virtual void SetLegendX1(Double_t x1);// {fLegendX1 = x1;}
  virtual void SetLegendY1(Double_t y1);// {fLegendY1 = y1;}
  virtual void SetLegendX2(Double_t x2);// {fLegendX2 = x2;}
  virtual void SetLegendY2(Double_t y2);// {fLegendY2 = y2;}
  virtual void SetStatsX1Y1X2Y2(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
  virtual void SetNumHistsX(UInt_t numX);
  virtual void SetNumHistsX(UInt_t numX, Int_t layer);
  virtual void SetNumHistsY(UInt_t numY);
  virtual void SetNumHistsY(UInt_t numY, Int_t layer);
  virtual void SetNumHistsXY(UInt_t numX, UInt_t numY);
  virtual void SetNumHistsXY(UInt_t numX, UInt_t numY, Int_t layer);
  virtual void SetLogY(Bool_t yesNo = kTRUE);
  virtual void SetLogY(Int_t layer, Bool_t yesNo = kTRUE);
  void SetCanvasName(const TString& name);
  virtual void SetCanvasWidth(Int_t w) {fCanvasWidth = w;}
  virtual void SetCanvasHeight(Int_t h) {fCanvasHeight = h;}
  virtual void SetHistsOption(Option_t* option);
  virtual void SetHistsOption(Option_t* option, Int_t layer);
  virtual void SetHistsMinMax(Double_t minMax, Bool_t min);
  virtual void SetHistsMinMax(Double_t minMax, Bool_t min, Int_t layer);
  static void SetLegendEntryOption(const char* option) {fgLegendEntryOption = option;} // some of "lpf"
  virtual void AddHistsOption(Option_t* option);
  virtual void AddHistsOption(Option_t* option, Int_t layer);
  virtual void SetHistsXTitle(const char* title);
  virtual void SetHistsXTitle(const char* title, Int_t layer);
  virtual void SetHistsYTitle(const char* title);
  virtual void SetHistsYTitle(const char* title, Int_t layer);
  virtual void SetHistsFillColor(Color_t color);
  virtual void SetHistsFillColor(Color_t color, Int_t layer);
  virtual void SetHistsLineWidth(Width_t width);
  virtual void SetHistsLineWidth(Width_t  width, Int_t layer);
  virtual void SetHistsLineStyle(Int_t style);
  virtual void SetHistsLineStyle(Int_t style, Int_t layer, Int_t numHistInPad = -1);
  virtual void SetHistsLineColor(Color_t color);
  virtual void SetHistsLineColor(Color_t color, Int_t layer);
  virtual void WriteCanvases(TFile* file);
  virtual void WriteHistos(TFile* file);
  virtual void Update();
  virtual void Update(Int_t layer);
  virtual Bool_t SetBatch(Bool_t set = kTRUE) {Bool_t r = fBatch; fBatch = set; return r;}
  virtual Bool_t IsBatch() const {return fBatch;}
  virtual void ExpandTo(Int_t newDepth);

  virtual GFHistArray* GetHistsOf(Int_t layer, Int_t histNum, Bool_t mayExpand = kFALSE);
  virtual TList* GetObjectsOf(Int_t layer, Int_t histNum);
  virtual Int_t GetNumHistsOf(Int_t layer);
  virtual TLegend* GetLegendOf(Int_t layer, Int_t nPad);
  virtual Int_t GetCanvasWidth() const {return fCanvasWidth;}
  virtual Int_t GetCanvasHeight() const {return fCanvasHeight;}
  virtual Int_t GetNumHistsX(Int_t layer) const;
  virtual Int_t GetNumHistsY(Int_t layer) const;
  virtual Double_t MaxOfHist(const TH1* h) const;
  virtual Double_t MaxOfHists(const TObjArray* hists) const;
  virtual Double_t MinOfHist(const TH1* h) const;
  virtual Double_t MinOfHists(const TObjArray* hists) const;
  virtual TVirtualPad* GetPad(Int_t layer, Int_t histNum);
  virtual TCanvas* GetCanvas(Int_t layer, Int_t number = 0);// after draw!!
  Int_t GetNumLayers() const {return fDepth;}
  virtual void GetLegendX1Y1X2Y2(Double_t& x1, Double_t& y1, Double_t& x2, Double_t& y2) const;

  static const Int_t kDefaultPadsPerCanX;// = 2; Doesn't work! Why ???
  static const Int_t kDefaultPadsPerCanY;// = 2; It should. And it DOES, if we make
  static const Int_t kDefaultDepth;// = 1;        these static const datamembers protected!

  static  Int_t NumberOfSubPadsOf(TCanvas* can);
  static void MakeDifferentStyle(GFHistArray* hists);
  Bool_t   DrawDiffStyle(Bool_t yesNo) {
    const Bool_t old = fDrawDiffStyle; fDrawDiffStyle = yesNo; return old;}
  Bool_t SameWithStats(Bool_t yesNo) {
    const Bool_t old = fSameWithStats; fSameWithStats = yesNo; return old;}

protected:
  virtual void DrawReally(Int_t layer);
  virtual void DrawLegend(Int_t layer, Int_t histNo);
  virtual void DrawObjects(Int_t layer, Int_t histNo);
  virtual void MakeCanvases(Int_t layer);
  virtual TObjArray* MakeLegends(Int_t layer);
  virtual TList* MakeObjList(Int_t layer, Int_t histoNum);
  virtual void Initialise();
  virtual Bool_t CheckDepth(const char* method, Int_t layer, Bool_t mayExpand = kTRUE);
  virtual Bool_t CheckHistNum(const char* method, Int_t layer, Int_t histNum, 
			      Bool_t mayExpand = kFALSE);
  void DrawFuncs(const TH1* hist) const;
  void ColourStatsBoxes(GFHistArray *hists) const;
  void ColourFuncs(GFHistArray *hists) const;

private:
  Int_t        fDepth;         // how many layers of histograms in arrays?
  TArrayI      fNoX;           // how many hists in x...
  TArrayI      fNoY;           // ... and in y in each canvas array
  TArrayC      fLogY;          // whether or not a layer should be plotted in log(y)
  Double_t     fLegendX1;      // default position
  Double_t     fLegendY1;      // ...of TLegends
  Double_t     fLegendX2;      // ... in x
  Double_t     fLegendY2;      // .. and y
  Double_t     fStatsX1;       // default positions in x and y
  Double_t     fStatsX2;       // ...of first statsbox in case
  Double_t     fStatsY1;       // ... many have to be drawn
  Double_t     fStatsY2;       // ... (subsequent boxes are shifted)
  TString      fCanvasName;
  Int_t        fCanvasWidth;   // pixel width
  Int_t        fCanvasHeight;  //       height of canvases (maybe relativly manipulated...) 
  static TString fgLegendEntryOption; // option used for legend entry style
  TObjArray*   fHistArrays;    // array of arrays of arrays of histograms
  TObjArray*   fLegendArrays;     // array of arrays to hold potential TLegend's
  TObjArray*   fObjLists;      // array of array of lists to hold potential objects
  TObjArray*   fCanArrays;     // array of arrays to hold canvases for drawing
  Bool_t       fBatch;         // if true: ignore Draw()
  Bool_t       fDrawDiffStyle; // if true(default): call MakeDifferentStyle while Draw
  Bool_t       fSameWithStats;  // if true(non-default): use SAMES option to add all stats boxes

  ClassDef(GFHistManager, 0)   // Gero's histogram manager (not writable!)
};

#endif
