#ifndef ZTR_TEcnaParHistos
#define ZTR_TEcnaParHistos

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

#include "TCanvas.h"
#include "TRootCanvas.h"
#include "TH1.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TString.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TVectorD.h"
#include "TMatrixD.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

///-----------------------------------------------------------
///   TEcnaParHistos.h
///   Update: 05/10/2012
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_main_page.htm
///-----------------------------------------------------------
///
///    Values of different parameters for plots in the framework of TEcnaHistos
///    (see TEcnaHistos documentation)
///
///    Examples of parameters:  ymin and ymax values for histos, title sizes,
///                             margins for plots, etc...
///

class TEcnaParHistos : public TObject {
private:
  //..... Attributes

  // static const Int_t fgMaxCar = 512;                   // <=== HYPER DANGEREUX !!!

  Int_t fgMaxCar;  // Max nb of caracters for char*

  Int_t fCnew, fCdelete;
  Int_t fCnewRoot, fCdeleteRoot;

  TString fTTBELL;

  Int_t fCnaCommand, fCnaError;

  TEcnaParEcal* fEcal;
  TEcnaNumbering* fEcalNumbering;

  TString fFlagSubDet;

  //...............................................................

  Int_t fMaxColorNumber;
  Double_t fMarginAutoMinMax;
  Int_t fNbOfRunsDisplayed;
  Int_t fMaxNbOfRunsInLists;

  TString fOnlyOnePlot;
  TString fSeveralPlot;
  TString fSameOnePlot;
  TString fAllXtalsInStinPlot;
  Int_t fPlotAllXtalsInStin;

public:
  //..... Methods

  TEcnaParHistos();
  TEcnaParHistos(TEcnaObject*, const TString&);
  TEcnaParHistos(const TString&, TEcnaParEcal*, TEcnaNumbering*);
  ~TEcnaParHistos() override;

  void Init();
  void SetEcalSubDetector(const TString&);
  void SetEcalSubDetector(const TString&, TEcnaParEcal*, TEcnaNumbering*);

  //...................................................... PLOT methods
  UInt_t SetCanvasWidth(const TString&, const TString&);
  UInt_t SetCanvasHeight(const TString&, const TString&);
  UInt_t CanvasFormatW(const TString&);
  UInt_t CanvasFormatH(const TString&);

  Double_t BoxLeftX(const TString&);
  Double_t BoxRightX(const TString&);
  Double_t BoxBottomY(const TString&);
  Double_t BoxTopY(const TString&);

  void SetColorPalette(const TString&);
  Color_t ColorTab(const Int_t&);
  Color_t ColorDefinition(const TString&);
  Int_t GetMaxNbOfColors();

  Int_t GetNbOfRunsDisplayed();
  Double_t GetMarginAutoMinMax();

  void SetViewHistoStyle(const TString&);
  void SetViewHistoPadMargins(const TString&, const TString&);
  void SetViewHistoStats(TH1D*, const TString&);
  void SetViewHistoOffsets(TH1D*, const TString&, const TString&);
  void SetViewGraphOffsets(TGraph*, const TString&);

  Float_t AxisTitleOffset();
  Float_t AxisTitleOffset(const TString&);
  Float_t AxisTitleSize();
  Float_t AxisTitleSize(const TString&);
  Float_t AxisLabelOffset();
  Float_t AxisLabelOffset(const TString&);
  Float_t AxisLabelSize();
  Float_t AxisLabelSize(const TString&);
  Float_t AxisTickSize();
  Float_t AxisTickSize(const TString&);

  Float_t DeeOffsetX(const TString&, const Int_t&);
  Float_t DeeNameOffsetX(const Int_t&);
  Float_t DeeNumberOffsetX(const TString&, const Int_t&);

  TPaveText* SetPaveGeneralComment(const TString&);
  TPaveText* SetPaveAnalysisRun(
      const TString&, const Int_t&, const Int_t&, const TString&, const Int_t&, const Int_t&, const TString&);
  TPaveText* SetPaveNbOfEvts(const Int_t&, const TString&, const TString&, const TString&);
  TPaveText* SetPaveEvolNbOfEvtsAna(const TString&, const Int_t&, const Int_t&, const Int_t&, const TString&);
  TPaveText* SetPaveEvolRuns(const Int_t&, const TString&, const Int_t&, const TString&, const TString&, const TString&);

  TPaveText* SetOptionSamePaveBorder(const TString&, const TString&);

  TPaveText* SetPaveStas();
  TPaveText* SetPaveSM(const TString&, const Int_t&, const TString&);
  TPaveText* SetPaveTower(const Int_t&);
  TPaveText* SetPaveTowersXY(const Int_t&, const Int_t&);
  TPaveText* SetPaveLVRB(const Int_t&, const Int_t&);
  Color_t SetColorsForNumbers(const TString&);

  TPaveText* SetPaveDee(const TString&, const Int_t&, const TString&);
  TPaveText* SetPaveSC(const Int_t&, const Int_t&);
  TPaveText* SetPaveSCsXY(const Int_t&, const Int_t&);
  TPaveText* SetPaveCxyz(const Int_t&);

  TPaveText* SetPaveStex(const TString&, const Int_t&);
  TPaveText* SetPaveStin(const Int_t&, const Int_t&);
  TPaveText* SetPaveStinsXY(const Int_t&, const Int_t&);
  TPaveText* SetPaveCrystal(const Int_t&, const Int_t&, const Int_t&);
  TPaveText* SetPaveCrystal(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  TPaveText* SetPaveCrystalSample(const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  TString GetHistoType(const TString&);
  TString GetXVarHisto(const TString&, const TString&, const Int_t&);
  TString GetYVarHisto(const TString&, const TString&, const Int_t&);
  TString GetQuantityName(const TString&);

  Double_t GetYminDefaultValue(const TString&);
  Double_t GetYmaxDefaultValue(const TString&);

  Int_t MaxNbOfRunsInLists();

  //...............................................................
  TString BuildStandardDetectorCode(const TString&);
  TString BuildStandardPlotOption(const TString&, const TString&);
  TString BuildStandard1DHistoCodeX(const TString&, const TString&);
  TString BuildStandard1DHistoCodeY(const TString&, const TString&);
  TString BuildStandard1DHistoCodeXY(const TString&);
  TString BuildStandardCovOrCorCode(const TString&, const TString&);
  TString BuildStandardBetweenWhatCode(const TString&, const TString&);

  void ListOfStandardCodes(const TString&);

  TString GetTechHistoCode(const TString&);
  TString GetTechHistoCode(const TString&, const TString&);

  TString GetCodeOnlyOnePlot();
  TString GetCodeSeveralPlot();
  TString GetCodeSameOnePlot();
  TString GetCodeAllXtalsInStinPlot();
  Int_t GetCodePlotAllXtalsInStin();

  ClassDefOverride(TEcnaParHistos, 1)  // Parameter management for CNA (Correlated Noises Analysis)
};

#endif  //    ZTR_TEcnaParameter
