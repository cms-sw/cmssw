#ifndef ZTR_TCnaViewEB
#define ZTR_TCnaViewEB

#include "TObject.h"
#include <Riostream.h>
#include <time.h>
#include "TSystem.h"

#include "TGraph.h"
#include "TVectorD.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TPaveText.h"
#include "TString.h"
#include "TColor.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaReadEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBNumbering.h"

//------------------------ TCnaViewEB.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TCnaViewEB : public TObject {

 private:

  //..... Attributes

  //  static  const  Int_t        fgMaxCar    = 512;   <== DANGEROUS !

  Int_t              fgMaxCar;                    // Max nb of caracters for char*

  Int_t              fCnaCommand,  fCnaError;
  Int_t              fCnew,        fCdelete;
  Int_t              fCnewRoot,    fCdeleteRoot;

  TString            fTTBELL;
  TCnaParameters*    fParameters;

  ifstream           fFcin_f;
  ifstream           fFcin_rr;
  ifstream           fFcin_lor;

  Int_t              fFapMaxNbOfRuns;  // Maximum Number of runs
  Int_t              fFapNbOfRuns;     // Number of runs
  TString            fFapFileRuns;     // name of the file containing the list of run parameters

  TString*           fT1DAnaType;             // Type of analysis
  Int_t*             fT1DRunNumber;           // Run number
  Int_t*             fT1DFirstEvt;            // First taken event number             
  Int_t*             fT1DNbOfEvts;            // Number of taken events             
  Int_t*             fT1DSuMoNumber;          // Super-Module number
  TString*           fT1DResultsRootFilePath; // absolute path for the ROOT files (/afs/etc... )
  TString*           fT1DListOfRunsFilePath;  // absolute path for the list-of-runs .ascii files (/afs/etc...)

  TString            fFapAnaType;                 // Type of analysis
  Int_t              fFapRunNumber;               // Run number
  Int_t              fFapFirstEvt;                // First taken event number
  Int_t              fFapNbOfEvts;                // Number of taken events
  Int_t              fFapSuMoNumber;              // Super-Module number
  TString            fFapSuMoBarrel;              // Barrel type of the SuperModule (barrel+ OR barrel-)
  TString            fFileForResultsRootFilePath; // name of the file containing the results .root file path
  TString            fFileForListOfRunFilePath;   // name of the file containing the list-of-run file path
  TString            fCfgResultsRootFilePath;     // absolute path for the results .root files (/afs/etc...)
  TString            fCfgListOfRunsFilePath;      // absolute path for the list-of-runs .ascii files (/afs/etc...)

  Int_t              fFapTowXNumber;   // Tower X number
  Int_t              fFapTowYNumber;   // Tower Y number
  Int_t              fFapChanNumber;   // Channel number
  Int_t              fFapSampNumber;   // Sample number

  time_t             fStartTime,      fStopTime;
  TString            fStartDate,      fStopDate;

  time_t             fStartEvolTime,  fStopEvolTime;
  TString            fStartEvolDate,  fStopEvolDate;

  Int_t              fStartEvolRun,   fStopEvolRun;
  Int_t              fNbOfExistingRuns;

  Int_t              fTowerSizeInCrystals;   // Size of one tower in term of crystals
                     // (a tower contains fTowerSizeInCrystals*fTowerSizeInCrystals crystals)

  Int_t  fOptVisLego,         fOptVisColz,        fOptVisSurf1,     fOptVisSurf4;
  Int_t  fOptVisLine,         fOptVisPolm;
  Int_t  fOptScaleLiny,       fOptScaleLogy;

  TString fFlagScaleX;
  TString fFlagScaleY;

  Int_t fOptMatCov,          fOptMatCor;

  Int_t   fTextPaveAlign;
  Int_t   fTextPaveFont;
  Float_t fTextPaveSize;

  TString fOptMcc;
  TString fOptMss;
  TString fOptMtt;

  Double_t fSMFoundEvtsGlobalYmin;
  Double_t fSMFoundEvtsGlobalYmax;
  Double_t fSMFoundEvtsProjYmin;
  Double_t fSMFoundEvtsProjYmax;
  Double_t fSMEvEvGlobalYmin;
  Double_t fSMEvEvGlobalYmax;
  Double_t fSMEvEvProjYmin;
  Double_t fSMEvEvProjYmax;
  Double_t fSMEvSigGlobalYmin;
  Double_t fSMEvSigGlobalYmax;
  Double_t fSMEvSigProjYmin;
  Double_t fSMEvSigProjYmax;
  Double_t fSMEvCorssGlobalYmin;
  Double_t fSMEvCorssGlobalYmax;
  Double_t fSMEvCorssProjYmin;
  Double_t fSMEvCorssProjYmax;
  Double_t fSMSigEvGlobalYmin;
  Double_t fSMSigEvGlobalYmax;
  Double_t fSMSigEvProjYmin;
  Double_t fSMSigEvProjYmax;
  Double_t fSMSigSigGlobalYmin;
  Double_t fSMSigSigGlobalYmax;
  Double_t fSMSigSigProjYmin;
  Double_t fSMSigSigProjYmax;
  Double_t fSMSigCorssGlobalYmin;
  Double_t fSMSigCorssGlobalYmax;
  Double_t fSMSigCorssProjYmin;
  Double_t fSMSigCorssProjYmax;

  Double_t fSMEvCorttMatrixYmin;
  Double_t fSMEvCorttMatrixYmax;
  Double_t fSMEvCovttMatrixYmin;
  Double_t fSMEvCovttMatrixYmax;
  Double_t fSMCorccInTowersYmin;
  Double_t fSMCorccInTowersYmax;


  Double_t fEvYmin;
  Double_t fEvYmax;
  Double_t fSigmaYmin;
  Double_t fSigmaYmax;
  Double_t fEvtsYmin;
  Double_t fEvtsYmax;
  Double_t fSampTimeYmin;
  Double_t fSampTimeYmax;
  Double_t fEvolEvEvYmin;
  Double_t fEvolEvEvYmax;
  Double_t fEvolEvSigYmin;
  Double_t fEvolEvSigYmax;
  Double_t fEvolEvCorssYmin;
  Double_t fEvolEvCorssYmax;

  //============================================== Canvases attributes, options

  TPaveText* ftitle_g1; 
  TPaveText* fcom_top_left; 
  TPaveText* fcom_top_left_memo; 
  TPaveText* fcom_top_mid;
  TPaveText* fcom_top_right;
  TPaveText* fcom_bot_left;
  TPaveText* fcom_bot_mid;
  TPaveText* fcom_bot_right;

  TString fOnlyOnePlot;
  TString fSeveralPlot;

  Int_t  fOptGlobal,     fOptProj;

  Int_t  fMemoPlotSMFoundEvtsGlobal, fMemoPlotSMFoundEvtsProj;
  Int_t  fMemoPlotSMEvEvGlobal,      fMemoPlotSMEvEvProj;
  Int_t  fMemoPlotSMEvSigGlobal,     fMemoPlotSMEvSigProj; 
  Int_t  fMemoPlotSMEvCorssGlobal,   fMemoPlotSMEvCorssProj;
  Int_t  fMemoPlotSMSigEvGlobal,     fMemoPlotSMSigEvProj; 
  Int_t  fMemoPlotSMSigSigGlobal,    fMemoPlotSMSigSigProj; 
  Int_t  fMemoPlotSMSigCorssGlobal,  fMemoPlotSMSigCorssProj; 
  Int_t  fMemoPlotEv,           fMemoPlotSigma,       fMemoPlotEvts,        fMemoPlotSampTime;
  Int_t  fMemoPlotEvolEvEv,     fMemoPlotEvolEvSig,   fMemoPlotEvolEvCorss;

  Int_t  fMemoColorSMFoundEvtsGlobal, fMemoColorSMFoundEvtsProj;
  Int_t  fMemoColorSMEvEvGlobal,      fMemoColorSMEvEvProj;
  Int_t  fMemoColorSMEvSigGlobal,     fMemoColorSMEvSigProj; 
  Int_t  fMemoColorSMEvCorssGlobal,   fMemoColorSMEvCorssProj;
  Int_t  fMemoColorSMSigEvGlobal,     fMemoColorSMSigEvProj; 
  Int_t  fMemoColorSMSigSigGlobal,    fMemoColorSMSigSigProj; 
  Int_t  fMemoColorSMSigCorssGlobal,  fMemoColorSMSigCorssProj; 
  Int_t  fMemoColorEv,                fMemoColorSigma,           fMemoColorEvts,        fMemoColorSampTime;
  Int_t  fMemoColorEvolEvEv,          fMemoColorEvolEvSig,       fMemoColorEvolEvCorss; 

  Int_t  fNbBinsProj,                 fMaxNbColLine;

  TCanvas*  fCanvSMFoundEvtsGlobal;
  TCanvas*  fCanvSMFoundEvtsProj;
  TCanvas*  fCanvSMEvEvGlobal;
  TCanvas*  fCanvSMEvEvProj;
  TCanvas*  fCanvSMEvSigGlobal;   
  TCanvas*  fCanvSMEvSigProj; 
  TCanvas*  fCanvSMEvCorssGlobal; 
  TCanvas*  fCanvSMEvCorssProj;
  TCanvas*  fCanvSMSigEvGlobal;
  TCanvas*  fCanvSMSigEvProj; 
  TCanvas*  fCanvSMSigSigGlobal;   
  TCanvas*  fCanvSMSigSigProj; 
  TCanvas*  fCanvSMSigCorssGlobal; 
  TCanvas*  fCanvSMSigCorssProj; 
  TCanvas*  fCanvEv;
  TCanvas*  fCanvSigma;  
  TCanvas*  fCanvEvts;     
  TCanvas*  fCanvSampTime;
  TCanvas*  fCanvEvolEvEv;
  TCanvas*  fCanvEvolEvSig;
  TCanvas*  fCanvEvolEvCorss;

  TVirtualPad*  fCurrentPad;

  TVirtualPad*  fPadSMFoundEvtsGlobal;
  TVirtualPad*  fPadSMFoundEvtsProj;
  TVirtualPad*  fPadSMEvEvGlobal;
  TVirtualPad*  fPadSMEvEvProj;
  TVirtualPad*  fPadSMEvSigGlobal;   
  TVirtualPad*  fPadSMEvSigProj; 
  TVirtualPad*  fPadSMEvCorssGlobal; 
  TVirtualPad*  fPadSMEvCorssProj;
  TVirtualPad*  fPadSMSigEvGlobal;
  TVirtualPad*  fPadSMSigEvProj; 
  TVirtualPad*  fPadSMSigSigGlobal;   
  TVirtualPad*  fPadSMSigSigProj; 
  TVirtualPad*  fPadSMSigCorssGlobal; 
  TVirtualPad*  fPadSMSigCorssProj; 
  TVirtualPad*  fPadEv;
  TVirtualPad*  fPadSigma;  
  TVirtualPad*  fPadEvts;     
  TVirtualPad*  fPadSampTime;
  TVirtualPad*  fPadEvolEvEv;
  TVirtualPad*  fPadEvolEvSig;
  TVirtualPad*  fPadEvolEvCorss;

  TCanvasImp*  fImpSMFoundEvtsGlobal;
  TCanvasImp*  fImpSMFoundEvtsProj;
  TCanvasImp*  fImpSMEvEvGlobal;
  TCanvasImp*  fImpSMEvEvProj;
  TCanvasImp*  fImpSMEvSigGlobal;   
  TCanvasImp*  fImpSMEvSigProj; 
  TCanvasImp*  fImpSMEvCorssGlobal; 
  TCanvasImp*  fImpSMEvCorssProj;
  TCanvasImp*  fImpSMSigEvGlobal;
  TCanvasImp*  fImpSMSigEvProj; 
  TCanvasImp*  fImpSMSigSigGlobal;   
  TCanvasImp*  fImpSMSigSigProj; 
  TCanvasImp*  fImpSMSigCorssGlobal; 
  TCanvasImp*  fImpSMSigCorssProj; 
  TCanvasImp*  fImpEv;
  TCanvasImp*  fImpSigma;  
  TCanvasImp*  fImpEvts;     
  TCanvasImp*  fImpSampTime;
  TCanvasImp*  fImpEvolEvEv;
  TCanvasImp*  fImpEvolEvSig;
  TCanvasImp*  fImpEvolEvCorss;

  TPaveText*  fPavTxtSMFoundEvtsGlobal;
  TPaveText*  fPavTxtSMFoundEvtsProj;
  TPaveText*  fPavTxtSMEvEvGlobal;
  TPaveText*  fPavTxtSMEvEvProj;
  TPaveText*  fPavTxtSMEvSigGlobal;   
  TPaveText*  fPavTxtSMEvSigProj; 
  TPaveText*  fPavTxtSMEvCorssGlobal; 
  TPaveText*  fPavTxtSMEvCorssProj;
  TPaveText*  fPavTxtSMSigEvGlobal;
  TPaveText*  fPavTxtSMSigEvProj; 
  TPaveText*  fPavTxtSMSigSigGlobal;   
  TPaveText*  fPavTxtSMSigSigProj; 
  TPaveText*  fPavTxtSMSigCorssGlobal; 
  TPaveText*  fPavTxtSMSigCorssProj; 
  TPaveText*  fPavTxtEv;
  TPaveText*  fPavTxtSigma;  
  TPaveText*  fPavTxtEvts;     
  TPaveText*  fPavTxtSampTime;
  TPaveText*  fPavTxtEvolEvEv;
  TPaveText*  fPavTxtEvolEvSig;
  TPaveText*  fPavTxtEvolEvCorss;

  Int_t  fCanvSameSMFoundEvtsGlobal, fCanvSameSMFoundEvtsProj;
  Int_t  fCanvSameSMEvEvGlobal,      fCanvSameSMEvEvProj;
  Int_t  fCanvSameSMEvSigGlobal,     fCanvSameSMEvSigProj; 
  Int_t  fCanvSameSMEvCorssGlobal,   fCanvSameSMEvCorssProj;
  Int_t  fCanvSameSMSigEvGlobal,     fCanvSameSMSigEvProj; 
  Int_t  fCanvSameSMSigSigGlobal,    fCanvSameSMSigSigProj; 
  Int_t  fCanvSameSMSigCorssGlobal,  fCanvSameSMSigCorssProj; 
  Int_t  fCanvSameEv,                fCanvSameSigma,           fCanvSameEvts,        fCanvSameSampTime;
  Int_t  fCanvSameEvolEvEv,          fCanvSameEvolEvSig,       fCanvSameEvolEvCorss;

  Int_t   fNbOfListFileEvolEvEv,  fNbOfListFileEvolEvSig,  fNbOfListFileEvolEvCorss;   // List file numbers

  Double_t    fXinf,    fXsup,    fYinf,    fYsup;

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22

 public:
           TCnaViewEB();
  virtual  ~TCnaViewEB();

  void Init();

  //................. methods to get the paths of the result files
  void GetPathForResultsRootFiles();
  void GetPathForResultsRootFiles(const TString);
  void GetPathForListOfRunFiles();
  void GetPathForListOfRunFiles(const TString);

  //................. methods to set the parameters values from arguments
  void SetFile(const Int_t&);
  void SetFile(const TString,  const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  void SetFile(const TString,  const Int_t&, const Int_t&, const Int_t&, const Int_t&,
	       const TString,  const TString);

  //................. methods calling ViewMatrix(...)
  void CorrelationsBetweenTowers(const TString);
  void CovariancesBetweenTowers(const TString);
  void CorrelationsBetweenCrystals(const Int_t&, const Int_t&, const TString);
  void CovariancesBetweenCrystals(const Int_t&, const Int_t&, const TString);
  void CorrelationsBetweenSamples(const Int_t&, const Int_t&, const TString);
  void CovariancesBetweenSamples(const Int_t&, const Int_t&, const TString);

  //................. methods calling ViewTower(...)
  void CorrelationsBetweenSamples(const Int_t& tower);
  void CovariancesBetweenSamples(const Int_t& tower);

  //................. methods calling ViewSuperModule(...) (eta,phi)
  void EtaPhiSuperModuleFoundEvents();
  void EtaPhiSuperModuleMeanPedestals();
  void EtaPhiSuperModuleMeanOfSampleSigmas();
  void EtaPhiSuperModuleMeanOfCorss();
  void EtaPhiSuperModuleSigmaPedestals();
  void EtaPhiSuperModuleSigmaOfSampleSigmas();
  void EtaPhiSuperModuleSigmaOfCorss();
  void EtaPhiSuperModuleCorccMeanOverSamples();

  //................. methods showing tower/crystal numbering
  void SuperModuleTowerNumbering(const Int_t&); 
  void TowerCrystalNumbering(const Int_t&, const Int_t&);

  //................. methods calling ViewHisto(...)
  void HistoSuperModuleFoundEventsOfCrystals(const TString);
  void HistoSuperModuleMeanPedestalsOfCrystals(const TString);
  void HistoSuperModuleMeanOfSampleSigmasOfCrystals(const TString);
  void HistoSuperModuleMeanOfCorssOfCrystals(const TString);
  void HistoSuperModuleSigmaPedestalsOfCrystals(const TString);
  void HistoSuperModuleSigmaOfSampleSigmasOfCrystals(const TString);
  void HistoSuperModuleSigmaOfCorssOfCrystals(const TString);

  void HistoSuperModuleFoundEventsDistribution(const TString);
  void HistoSuperModuleMeanPedestalsDistribution(const TString);
  void HistoSuperModuleMeanOfSampleSigmasDistribution(const TString);
  void HistoSuperModuleMeanOfCorssDistribution(const TString);
  void HistoSuperModuleSigmaPedestalsDistribution(const TString);
  void HistoSuperModuleSigmaOfSampleSigmasDistribution(const TString);
  void HistoSuperModuleSigmaOfCorssDistribution(const TString);

  void HistoCrystalExpectationValuesOfSamples(const Int_t&, const Int_t&, const TString);
  void HistoCrystalSigmasOfSamples(const Int_t&, const Int_t&, const TString);
  void HistoCrystalPedestalEventNumber(const Int_t&, const Int_t&, const TString);

  void HistoSampleEventDistribution(const Int_t&, const Int_t&, const Int_t&, const TString);

  //................. methods calling ViewHistime(...) (evolution in time)
  void HistimeCrystalMeanPedestals(const TString, const Int_t&, const Int_t&, const TString);
  void HistimeCrystalMeanSigmas(const TString, const Int_t&, const Int_t&, const TString);
  void HistimeCrystalMeanCorss(const TString, const Int_t&, const Int_t&, const TString);

  //....................................................................

  void ViewSuperModule(const TString);
  void ViewTower(const Int_t&, const Int_t&);
  void ViewMatrix(const Int_t&, const Int_t&,  const Int_t&,
		  const Int_t&, const TString, const TString);

  void ViewHisto(const Int_t&,  const Int_t&, const Int_t&,
		 const TString, const Int_t&, const TString);
  void ViewHistime(const TString, const Int_t&, const Int_t&,
		   const TString, const Int_t&, const TString);

  Int_t GetListOfRunParameters(const TString, const TString);

  void  TopAxisForTowerNumbers(TH1D*, const TString, const Int_t&,
			       const Int_t&, const Int_t&, const Int_t& );
  Int_t HistoSetMinMax(TH1D*,   const TString);
  Int_t GraphSetMinMax(TGraph*, const TString);

  void ViewSuperModuleTowerNumberingPad(TEBParameters*, TEBNumbering*, const Int_t&);
  void ViewTowerGrid(TEBNumbering*, const Int_t&, const Int_t&, const Int_t&,
		     const Int_t&,   const Int_t&, const TString);
  void ViewSuperModuleGrid(TEBParameters*, TEBNumbering*, const Int_t&, const TString);
  void SqrtContourLevels(const Int_t&, Double_t*);

  void HistoPlot(TH1D*, TCnaReadEB*, TEBNumbering*, const Int_t&, const TString, const TString,
		 const Int_t&,     const Int_t&,   const Int_t&, const Int_t&,  const Int_t&,
		 const Int_t&,     const TString,  const Int_t&);

  void HistimePlot(TGraph*, Axis_t , Axis_t, TCnaReadEB*, TEBNumbering*, const TString,
		   const TString, const Int_t&, const Int_t&,
		   const Int_t&, const Int_t&, const Int_t&, const Int_t&, const TString, const Int_t&);

  TPaveText* PutPaveGeneralComment();
  TPaveText* PutPaveSuperModule(const TString);
  TPaveText* PutPaveTower(const Int_t&);
  TPaveText* PutPaveTowersXY(const Int_t&, const Int_t&);
  TPaveText* PutPaveCrystal(TEBNumbering*, const Int_t&, const Int_t&);
  TPaveText* PutPaveCrystalSample(TCnaReadEB*, const Int_t&, const Int_t&, const Int_t&);
  TPaveText* PutPaveAnalysisRun(TCnaReadEB*);
  TPaveText* PutPaveTakenEvents(TCnaReadEB*);
  TPaveText* PutPaveAnalysisRunList(TCnaReadEB*);
  TPaveText* PutPaveTakenEventsRunList(TCnaReadEB*);
  TPaveText* PutPaveLVRB(TEBNumbering*, const Int_t&, const Int_t&);

  void PutAllPavesViewMatrix(TCnaReadEB*, TEBNumbering*, const TString, const Int_t&,
			     const Int_t&, const Int_t&);
  void PutAllPavesViewTower(TCnaReadEB*, const Int_t&);
  void PutAllPavesViewSuperModule();
  void PutAllPavesViewSuperModule(TCnaReadEB*);
  void PutAllPavesViewTowerCrysNb(TEBNumbering*, const Int_t&, const Int_t&);
  void PutAllPavesViewHisto(TCnaReadEB*, TEBNumbering*, const TString,
			    const Int_t&, const Int_t&, const Int_t&, const TString);

  Int_t GetXSampInTow(TEBNumbering*, TEBParameters*, const Int_t&, const Int_t&,
		      const Int_t&, const Int_t&);
  Int_t GetYSampInTow(TEBNumbering*, TEBParameters*, const Int_t&, const Int_t&,
		      const Int_t&, const Int_t&);

  Int_t GetXCrysInSM(TEBNumbering*, TEBParameters*, const Int_t&, const Int_t&, const Int_t&);
  Int_t GetYCrysInSM(TEBNumbering*, TEBParameters*, const Int_t&, const Int_t&, const Int_t&);

  TString GetEtaPhiAxisTitle(const TString);

  TString GetQuantityType(const TString);
  TString GetQuantityName(const TString);
  Bool_t  GetOkViewHisto(TCnaReadEB*, const Int_t&, const Int_t&, const Int_t&, const TString);
  Int_t   GetHistoSize(TCnaReadEB*,   const TString);

  TVectorD GetHistoValues(TCnaReadEB*,     const TString, const Int_t&,
			  const Int_t&,  const Int_t&,  const Int_t&, Int_t&);

  TString SetHistoXAxisTitle(const TString);
  TString SetHistoYAxisTitle(const TString);

  Axis_t SetHistoXinf(TCnaReadEB*,     const TString, const Int_t&,
		      const Int_t&,  const Int_t&,  const Int_t&);
  Axis_t SetHistoXsup(TCnaReadEB*,     const TString, const Int_t&,
		      const Int_t&,  const Int_t&,  const Int_t&);

  Int_t  SetHistoNumberOfBins(const TString,  const Int_t&); 

  void     PutYmin(const TString, const Double_t&);
  void     PutYmax(const TString, const Double_t&);
  Double_t GetYmin(const TString);
  Double_t GetYmax(const TString);


  void SetHistoScaleX(const TString);
  void SetHistoScaleY(const TString);

  void FillHisto(TH1D*,  const TVectorD, const TString, const Int_t&,
		 const Axis_t, const Axis_t,   const Int_t&);

  TString GetMemoFlag(const TString);
  void    SetMemoFlagFree(const TString);
  void    SetMemoFlagBusy(const TString);

  void         CreateCanvas(const TString, const TString, UInt_t,  UInt_t);
  void         SetParametersCanvas(const TString);
  TVirtualPad* ActivePad(const TString);
  TPaveText*   ActivePavTxt(const TString);

  void SetHistoPresentation(TH1D*, const TString);
  void SetGraphPresentation(TGraph*, const TString);

  void SetViewHistoStyle(const TString);
  void SetViewHistoPadMargins(const TString);

  void SetViewHistoOffsets(TH1D*, const TString);
  void SetViewGraphOffsets(TGraph*, const TString); 
  
  void SetViewHistoStats(TH1D*, const TString);

  void SetViewHistoColors(TH1D*, const TString, const TString);
  void SetViewGraphColors(TGraph*, const TString, const TString);

  Color_t  GetViewHistoColor(const TString);

  Int_t GetListFileNumber(const TString);
  void  ReInitCanvas(const TString);

  void InitQuantityYmin(const TString);
  void InitQuantityYmax(const TString);

  TString  SetCanvasName(const TString, const Int_t&, const Int_t&,  const TString,
			 const Int_t&,  const Int_t&, const Int_t&);

  UInt_t SetCanvasWidth(const TString);
  UInt_t SetCanvasHeight(const TString);
  UInt_t CanvasFormatW(const TString);
  UInt_t CanvasFormatH(const TString);

  Color_t SetColorsForNumbers(const TString);
  Color_t ColorTab(const Int_t&);
  Color_t ColorDefinition(const TString);

  Double_t BoxLeftX(const TString);
  Double_t BoxRightX(const TString);
  Double_t BoxBottomY(const TString);
  Double_t BoxTopY(const TString);

  void AllocArraysForEvol();

ClassDef(TCnaViewEB,1)// methods for plots from CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TCnaViewEB
