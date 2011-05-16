#ifndef ZTR_TEcnaHistos
#define ZTR_TEcnaHistos

#include "TObject.h"
#include <Riostream.h>
#include <time.h>
#include "TSystem.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TRootCanvas.h"
#include "TVectorD.h"
#include "TH1.h"
#include "TH2D.h"
#include "TF1.h"
#include "TPaveText.h"
#include "TString.h"
#include "TColor.h"
#include "TGaxis.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

//------------------------ TEcnaHistos.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TEcnaHistos : public TObject {

 private:

  //..... Attributes

  //  static  const  Int_t        fgMaxCar    = 512;   <== DANGEROUS !

  Int_t fgMaxCar;                    // Max nb of caracters for char*
  Int_t fZerv;                       // = 0 , for ViewHisto non used arguments


  Int_t fCnaCommand, fCnaError;
  Int_t fCnew,       fCdelete;
  Int_t fCnewRoot,   fCdeleteRoot;

  TString fTTBELL;

  //....................... Current subdetector flag and codes
  TString fFlagSubDet;
  TString fCodeEB;
  TString fCodeEE;

  //...........................................
  TEcnaParHistos*   fCnaParHistos;
  TEcnaParPaths*    fCnaParPaths;
  TEcnaParCout*     fCnaParCout;
  TEcnaWrite*       fCnaWrite;
  TEcnaParEcal* fEcal;
  TEcnaNumbering*  fEcalNumbering;
  TEcnaHeader*      fFileHeader;

  TEcnaRead*        fMyRootFile;

  ifstream fFcin_f;

  TString fFapAnaType;             // Type of analysis
  Int_t   fFapNbOfSamples;         // Nb of required samples
  Int_t   fFapRunNumber;           // Run number
  Int_t   fFapFirstReqEvtNumber;   // First requested event number
  Int_t   fFapLastReqEvtNumber;    // Last requested event number
  Int_t   fFapReqNbOfEvts;         // Requested number of events
  Int_t   fFapStexNumber;          // Stex number

  Int_t   fFapNbOfEvts;           // Number of found events

  Int_t   fFapMaxNbOfRuns;         // Maximum Number of runs
  Int_t   fFapNbOfRuns;            // Number of runs
  TString fFapFileRuns;            // name of the file containing the list of run parameters

  Int_t fStartEvolRun, fStopEvolRun;
  Int_t fNbOfExistingRuns;

  time_t  fStartEvolTime, fStopEvolTime;
  TString fStartEvolDate, fStopEvolDate;

  TString fFapStexBarrel;          // Barrel type of the Stex (barrel+ OR barrel-)   (EB only)
  TString fFapStexType;            // type of the Dee (EE+F, EE+N, EE-F, EE-N)       (EE only)
  TString fFapStexDir;             // direction of the Dee (right, left)             (EE only)
  TString fFapStinQuadType;        // quadrant type of the SC (top, bottom)          (EE only)
 
  TString fFapStexName;            // Stex name: "SM"      (for EB) or "Dee"     (for EE)
  TString fFapStinName;            // Stin name: "tower"   (for EB) or "SC"      (for EE)
  TString fFapEchaName;            // Echa name: "channel" (for EB) or "crystal" (for EE)


  TString fMyRootFileName;  // memo Root file name used in SetFile() for obtaining the number of found events 

  TString fCfgResultsRootFilePath;     // absolute path for the results .root files (/afs/etc...)
  TString fCfgHistoryRunListFilePath;  // absolute path for the list-of-runs .ascii files (/afs/etc...)
                                       // MUST BE KEPT BECAUSE OF HISTIME PLOTS
  TString fAsciiFileName;

  Bool_t fStatusFileFound;
  Bool_t fStatusDataExist;

  time_t  fStartTime, fStopTime;
  TString fStartDate, fStopDate;
  TString fRunType;


  TString* fT1DAnaType;             // Type of analysis
  Int_t*   fT1DRunNumber;           // Run number

  TString* fT1DResultsRootFilePath; // absolute path for the ROOT files (/afs/etc... )
  TString* fT1DHistoryRunListFilePath;   // absolute path for the list-of-runs .ascii files (/afs/etc...)

  Int_t fStinSizeInCrystals;   // Size of one Stin in term of crystals
                               // (a Stin contains fStinSizeInCrystals*fStinSizeInCrystals crystals)
  TString fFlagScaleX;
  TString fFlagScaleY;
  TString fFlagColPal;
  TString fFlagGeneralTitle;

  Double_t fUserHistoMin,     fUserHistoMax;
  TString  fFlagUserHistoMin, fFlagUserHistoMax;
  //  TString  fCurQuantCode;

  Int_t fOptVisLego,   fOptVisColz,   fOptVisSurf1,  fOptVisSurf4;
  Int_t fOptVisLine,   fOptVisPolm;

  Int_t fOptScaleLinx, fOptScaleLogx, fOptScaleLiny, fOptScaleLogy;

  TString fCovarianceMatrix, fCorrelationMatrix;

  TString fBetweenSamples;
  TString fLFBetweenChannels, fHFBetweenChannels;
  TString fLFBetweenStins, fHFBetweenStins;

  Int_t   fTextPaveAlign;
  Int_t   fTextPaveFont;
  Float_t fTextPaveSize;
  Int_t   fTextBorderSize;

  Double_t fXinf, fXsup, fYinf, fYsup;

  Double_t fXinfProj, fXsupProj;

  //.................................... Xinf, Xsup
  Axis_t fH1SameOnePlotXinf;
  Axis_t fH1SameOnePlotXsup;

  Axis_t fD_NOE_ChNbXinf;
  Axis_t fD_NOE_ChNbXsup;
  Axis_t fD_NOE_ChDsXinf;
  Axis_t fD_NOE_ChDsXsup;
  Axis_t fD_Ped_ChNbXinf;
  Axis_t fD_Ped_ChNbXsup;
  Axis_t fD_Ped_ChDsXinf;
  Axis_t fD_Ped_ChDsXsup;
  Axis_t fD_TNo_ChNbXinf;
  Axis_t fD_TNo_ChNbXsup;
  Axis_t fD_TNo_ChDsXinf;
  Axis_t fD_TNo_ChDsXsup;
  Axis_t fD_MCs_ChNbXinf;
  Axis_t fD_MCs_ChNbXsup;
  Axis_t fD_MCs_ChDsXinf;
  Axis_t fD_MCs_ChDsXsup;
  Axis_t fD_LFN_ChNbXinf;
  Axis_t fD_LFN_ChNbXsup;
  Axis_t fD_LFN_ChDsXinf;
  Axis_t fD_LFN_ChDsXsup;
  Axis_t fD_HFN_ChNbXinf;
  Axis_t fD_HFN_ChNbXsup;
  Axis_t fD_HFN_ChDsXinf;
  Axis_t fD_HFN_ChDsXsup;
  Axis_t fD_SCs_ChNbXinf;
  Axis_t fD_SCs_ChNbXsup;
  Axis_t fD_SCs_ChDsXinf;
  Axis_t fD_SCs_ChDsXsup;

  Axis_t fD_MSp_SampXinf;
  Axis_t fD_MSp_SampXsup;
  Axis_t fD_SSp_SampXinf;
  Axis_t fD_SSp_SampXsup;
  Axis_t fD_Adc_EvDsXinf;
  Axis_t fD_Adc_EvDsXsup;
  Axis_t fD_Adc_EvNbXinf;
  Axis_t fD_Adc_EvNbXsup;
  Axis_t fH_Ped_DateXinf;
  Axis_t fH_Ped_DateXsup;
  Axis_t fH_TNo_DateXinf;
  Axis_t fH_TNo_DateXsup;
  Axis_t fH_MCs_DateXinf;
  Axis_t fH_MCs_DateXsup;
  Axis_t fH_LFN_DateXinf;
  Axis_t fH_LFN_DateXsup;
  Axis_t fH_HFN_DateXinf;
  Axis_t fH_HFN_DateXsup;
  Axis_t fH_SCs_DateXinf;
  Axis_t fH_SCs_DateXsup;

  Axis_t fH_Ped_RuDsXinf;
  Axis_t fH_Ped_RuDsXsup;
  Axis_t fH_TNo_RuDsXinf;
  Axis_t fH_TNo_RuDsXsup;
  Axis_t fH_MCs_RuDsXinf;
  Axis_t fH_MCs_RuDsXsup;
  Axis_t fH_LFN_RuDsXinf;
  Axis_t fH_LFN_RuDsXsup;
  Axis_t fH_HFN_RuDsXinf;
  Axis_t fH_HFN_RuDsXsup;
  Axis_t fH_SCs_RuDsXinf;
  Axis_t fH_SCs_RuDsXsup;

  //.................................... Ymin, Ymax

  TString  fHistoCodeFirst;  // HistoCode of the first histo in option SAME n
  Double_t fD_NOE_ChNbYmin;
  Double_t fD_NOE_ChNbYmax;
  Double_t fD_NOE_ChDsYmin;
  Double_t fD_NOE_ChDsYmax;
  Double_t fD_Ped_ChNbYmin;
  Double_t fD_Ped_ChNbYmax;
  Double_t fD_Ped_ChDsYmin;
  Double_t fD_Ped_ChDsYmax;
  Double_t fD_TNo_ChNbYmin;
  Double_t fD_TNo_ChNbYmax;
  Double_t fD_TNo_ChDsYmin;
  Double_t fD_TNo_ChDsYmax;
  Double_t fD_MCs_ChNbYmin;
  Double_t fD_MCs_ChNbYmax;
  Double_t fD_MCs_ChDsYmin;
  Double_t fD_MCs_ChDsYmax;
  Double_t fD_LFN_ChNbYmin;
  Double_t fD_LFN_ChNbYmax;
  Double_t fD_LFN_ChDsYmin;
  Double_t fD_LFN_ChDsYmax;
  Double_t fD_HFN_ChNbYmin;
  Double_t fD_HFN_ChNbYmax;
  Double_t fD_HFN_ChDsYmin;
  Double_t fD_HFN_ChDsYmax;
  Double_t fD_SCs_ChNbYmin;
  Double_t fD_SCs_ChNbYmax;
  Double_t fD_SCs_ChDsYmin;
  Double_t fD_SCs_ChDsYmax;

  Double_t fD_MSp_SampYmin;
  Double_t fD_MSp_SampYmax;
  Double_t fD_SSp_SampYmin;
  Double_t fD_SSp_SampYmax;
  Double_t fD_Adc_EvDsYmin;
  Double_t fD_Adc_EvDsYmax;
  Double_t fD_Adc_EvNbYmin;
  Double_t fD_Adc_EvNbYmax;
  Double_t fH_Ped_DateYmin;
  Double_t fH_Ped_DateYmax;
  Double_t fH_TNo_DateYmin;
  Double_t fH_TNo_DateYmax;
  Double_t fH_MCs_DateYmin;
  Double_t fH_MCs_DateYmax;
  Double_t fH_LFN_DateYmin;
  Double_t fH_LFN_DateYmax;
  Double_t fH_HFN_DateYmin;
  Double_t fH_HFN_DateYmax;
  Double_t fH_SCs_DateYmin;
  Double_t fH_SCs_DateYmax;

  Double_t fH_Ped_RuDsYmin;
  Double_t fH_Ped_RuDsYmax;
  Double_t fH_TNo_RuDsYmin;
  Double_t fH_TNo_RuDsYmax;
  Double_t fH_MCs_RuDsYmin;
  Double_t fH_MCs_RuDsYmax;
  Double_t fH_LFN_RuDsYmin;
  Double_t fH_LFN_RuDsYmax;
  Double_t fH_HFN_RuDsYmin;
  Double_t fH_HFN_RuDsYmax;
  Double_t fH_SCs_RuDsYmin;
  Double_t fH_SCs_RuDsYmax;

  Double_t fH2LFccMosMatrixYmin;
  Double_t fH2LFccMosMatrixYmax;
  Double_t fH2HFccMosMatrixYmin;
  Double_t fH2HFccMosMatrixYmax;
  Double_t fH2CorccInStinsYmin;
  Double_t fH2CorccInStinsYmax;

  //============================================== Canvases attributes, options
  TPaveText* fPavComGeneralTitle;
  TPaveText* fPavComStas;
  TPaveText* fPavComStex;
  TPaveText* fPavComStin;
  TPaveText* fPavComXtal;
  TPaveText* fPavComAnaRun;
  TPaveText* fPavComNbOfEvts;
  TPaveText* fPavComSeveralChanging;
  TPaveText* fPavComLVRB;                 // specific EB
  TPaveText* fPavComCxyz;                 // specific EE
  TPaveText* fPavComEvolRuns;
  TPaveText* fPavComEvolNbOfEvtsAna;

  TString fOnlyOnePlot;
  TString fSeveralPlot;
  TString fSameOnePlot;

  Int_t  fMemoPlotH1SamePlus;
  Int_t  fMemoPlotD_NOE_ChNb, fMemoPlotD_NOE_ChDs;
  Int_t  fMemoPlotD_Ped_ChNb, fMemoPlotD_Ped_ChDs;
  Int_t  fMemoPlotD_TNo_ChNb, fMemoPlotD_TNo_ChDs; 
  Int_t  fMemoPlotD_MCs_ChNb, fMemoPlotD_MCs_ChDs;
  Int_t  fMemoPlotD_LFN_ChNb, fMemoPlotD_LFN_ChDs; 
  Int_t  fMemoPlotD_HFN_ChNb, fMemoPlotD_HFN_ChDs; 
  Int_t  fMemoPlotD_SCs_ChNb, fMemoPlotD_SCs_ChDs; 
  Int_t  fMemoPlotD_MSp_Samp, fMemoPlotD_SSp_Samp;
  Int_t  fMemoPlotD_Adc_EvNb, fMemoPlotD_Adc_EvDs;
  Int_t  fMemoPlotH_Ped_Date, fMemoPlotH_Ped_RuDs;
  Int_t  fMemoPlotH_TNo_Date, fMemoPlotH_TNo_RuDs;
  Int_t  fMemoPlotH_LFN_Date, fMemoPlotH_LFN_RuDs;
  Int_t  fMemoPlotH_HFN_Date, fMemoPlotH_HFN_RuDs;
  Int_t  fMemoPlotH_MCs_Date, fMemoPlotH_MCs_RuDs;
  Int_t  fMemoPlotH_SCs_Date, fMemoPlotH_SCs_RuDs;

  Int_t  fMemoColorH1SamePlus;
  Int_t  fMemoColorD_NOE_ChNb, fMemoColorD_NOE_ChDs;
  Int_t  fMemoColorD_Ped_ChNb, fMemoColorD_Ped_ChDs;
  Int_t  fMemoColorD_TNo_ChNb, fMemoColorD_TNo_ChDs; 
  Int_t  fMemoColorD_MCs_ChNb, fMemoColorD_MCs_ChDs;
  Int_t  fMemoColorD_LFN_ChNb, fMemoColorD_LFN_ChDs; 
  Int_t  fMemoColorD_HFN_ChNb, fMemoColorD_HFN_ChDs; 
  Int_t  fMemoColorD_SCs_ChNb, fMemoColorD_SCs_ChDs; 
  Int_t  fMemoColorD_MSp_Samp, fMemoColorD_SSp_Samp;
  Int_t  fMemoColorD_Adc_EvNb, fMemoColorD_Adc_EvDs;
  Int_t  fMemoColorH_Ped_Date, fMemoColorH_Ped_RuDs;
  Int_t  fMemoColorH_TNo_Date, fMemoColorH_TNo_RuDs;
  Int_t  fMemoColorH_LFN_Date, fMemoColorH_LFN_RuDs;
  Int_t  fMemoColorH_HFN_Date, fMemoColorH_HFN_RuDs;
  Int_t  fMemoColorH_MCs_Date, fMemoColorH_MCs_RuDs; 
  Int_t  fMemoColorH_SCs_Date, fMemoColorH_SCs_RuDs;

  Int_t  fNbBinsProj;

  TString  fXMemoH1SamePlus;
  TString  fXMemoD_NOE_ChNb;
  TString  fXMemoD_NOE_ChDs;
  TString  fXMemoD_Ped_ChNb;
  TString  fXMemoD_Ped_ChDs;
  TString  fXMemoD_TNo_ChNb;
  TString  fXMemoD_TNo_ChDs; 
  TString  fXMemoD_MCs_ChNb; 
  TString  fXMemoD_MCs_ChDs;
  TString  fXMemoD_LFN_ChNb;
  TString  fXMemoD_LFN_ChDs; 
  TString  fXMemoD_HFN_ChNb;   
  TString  fXMemoD_HFN_ChDs; 
  TString  fXMemoD_SCs_ChNb; 
  TString  fXMemoD_SCs_ChDs; 
  TString  fXMemoD_MSp_Samp;
  TString  fXMemoD_SSp_Samp;
  TString  fXMemoD_Adc_EvDs;     
  TString  fXMemoD_Adc_EvNb;
  TString  fXMemoH_Ped_Date;
  TString  fXMemoH_TNo_Date;
  TString  fXMemoH_MCs_Date;
  TString  fXMemoH_LFN_Date;
  TString  fXMemoH_HFN_Date;
  TString  fXMemoH_SCs_Date;
  TString  fXMemoH_Ped_RuDs;
  TString  fXMemoH_TNo_RuDs;
  TString  fXMemoH_MCs_RuDs;
  TString  fXMemoH_LFN_RuDs;
  TString  fXMemoH_HFN_RuDs;
  TString  fXMemoH_SCs_RuDs;

  TString  fYMemoH1SamePlus;
  TString  fYMemoD_NOE_ChNb;
  TString  fYMemoD_NOE_ChDs;
  TString  fYMemoD_Ped_ChNb;
  TString  fYMemoD_Ped_ChDs;
  TString  fYMemoD_TNo_ChNb;   
  TString  fYMemoD_TNo_ChDs; 
  TString  fYMemoD_MCs_ChNb; 
  TString  fYMemoD_MCs_ChDs;
  TString  fYMemoD_LFN_ChNb;
  TString  fYMemoD_LFN_ChDs; 
  TString  fYMemoD_HFN_ChNb;   
  TString  fYMemoD_HFN_ChDs; 
  TString  fYMemoD_SCs_ChNb; 
  TString  fYMemoD_SCs_ChDs; 
  TString  fYMemoD_MSp_Samp;
  TString  fYMemoD_SSp_Samp;  
  TString  fYMemoD_Adc_EvDs;     
  TString  fYMemoD_Adc_EvNb;
  TString  fYMemoH_Ped_Date;
  TString  fYMemoH_TNo_Date;
  TString  fYMemoH_MCs_Date;
  TString  fYMemoH_LFN_Date;
  TString  fYMemoH_HFN_Date;
  TString  fYMemoH_SCs_Date;
  TString  fYMemoH_Ped_RuDs;
  TString  fYMemoH_TNo_RuDs;
  TString  fYMemoH_MCs_RuDs;
  TString  fYMemoH_LFN_RuDs;
  TString  fYMemoH_HFN_RuDs;
  TString  fYMemoH_SCs_RuDs;

  Int_t  fNbBinsMemoH1SamePlus;
  Int_t  fNbBinsMemoD_NOE_ChNb;
  Int_t  fNbBinsMemoD_NOE_ChDs;
  Int_t  fNbBinsMemoD_Ped_ChNb;
  Int_t  fNbBinsMemoD_Ped_ChDs;
  Int_t  fNbBinsMemoD_TNo_ChNb;   
  Int_t  fNbBinsMemoD_TNo_ChDs; 
  Int_t  fNbBinsMemoD_MCs_ChNb; 
  Int_t  fNbBinsMemoD_MCs_ChDs;
  Int_t  fNbBinsMemoD_LFN_ChNb;
  Int_t  fNbBinsMemoD_LFN_ChDs; 
  Int_t  fNbBinsMemoD_HFN_ChNb;   
  Int_t  fNbBinsMemoD_HFN_ChDs; 
  Int_t  fNbBinsMemoD_SCs_ChNb; 
  Int_t  fNbBinsMemoD_SCs_ChDs; 
  Int_t  fNbBinsMemoD_MSp_Samp;
  Int_t  fNbBinsMemoD_SSp_Samp;  
  Int_t  fNbBinsMemoD_Adc_EvDs;     
  Int_t  fNbBinsMemoD_Adc_EvNb;
  Int_t  fNbBinsMemoH_Ped_Date;
  Int_t  fNbBinsMemoH_TNo_Date;
  Int_t  fNbBinsMemoH_MCs_Date;
  Int_t  fNbBinsMemoH_LFN_Date;
  Int_t  fNbBinsMemoH_HFN_Date;
  Int_t  fNbBinsMemoH_SCs_Date;
  Int_t  fNbBinsMemoH_Ped_RuDs;
  Int_t  fNbBinsMemoH_TNo_RuDs;
  Int_t  fNbBinsMemoH_MCs_RuDs;
  Int_t  fNbBinsMemoH_LFN_RuDs;
  Int_t  fNbBinsMemoH_HFN_RuDs;
  Int_t  fNbBinsMemoH_SCs_RuDs;
  //.......................................................
  TString   fCurrentCanvasName;
  TCanvas*  fCurrentCanvas;

  TCanvas*  fCanvH1SamePlus;
  TCanvas*  fCanvD_NOE_ChNb;
  TCanvas*  fCanvD_NOE_ChDs;
  TCanvas*  fCanvD_Ped_ChNb;
  TCanvas*  fCanvD_Ped_ChDs;
  TCanvas*  fCanvD_TNo_ChNb;
  TCanvas*  fCanvD_TNo_ChDs;
  TCanvas*  fCanvD_MCs_ChNb;
  TCanvas*  fCanvD_MCs_ChDs;
  TCanvas*  fCanvD_LFN_ChNb;
  TCanvas*  fCanvD_LFN_ChDs;
  TCanvas*  fCanvD_HFN_ChNb; 
  TCanvas*  fCanvD_HFN_ChDs;
  TCanvas*  fCanvD_SCs_ChNb;
  TCanvas*  fCanvD_SCs_ChDs;
  TCanvas*  fCanvD_MSp_Samp;
  TCanvas*  fCanvD_SSp_Samp;  
  TCanvas*  fCanvD_Adc_EvDs;     
  TCanvas*  fCanvD_Adc_EvNb;
  TCanvas*  fCanvH_Ped_Date;
  TCanvas*  fCanvH_TNo_Date;
  TCanvas*  fCanvH_MCs_Date;
  TCanvas*  fCanvH_LFN_Date;
  TCanvas*  fCanvH_HFN_Date;
  TCanvas*  fCanvH_SCs_Date;
  TCanvas*  fCanvH_Ped_RuDs;
  TCanvas*  fCanvH_TNo_RuDs;
  TCanvas*  fCanvH_MCs_RuDs;
  TCanvas*  fCanvH_LFN_RuDs;
  TCanvas*  fCanvH_HFN_RuDs;
  TCanvas*  fCanvH_SCs_RuDs;

  TVirtualPad*  fCurrentPad;

  TVirtualPad*  fPadH1SamePlus;
  TVirtualPad*  fPadD_NOE_ChNb;
  TVirtualPad*  fPadD_NOE_ChDs;
  TVirtualPad*  fPadD_Ped_ChNb;
  TVirtualPad*  fPadD_Ped_ChDs;
  TVirtualPad*  fPadD_TNo_ChNb;  
  TVirtualPad*  fPadD_TNo_ChDs; 
  TVirtualPad*  fPadD_MCs_ChNb;
  TVirtualPad*  fPadD_MCs_ChDs;
  TVirtualPad*  fPadD_LFN_ChNb;
  TVirtualPad*  fPadD_LFN_ChDs;
  TVirtualPad*  fPadD_HFN_ChNb;   
  TVirtualPad*  fPadD_HFN_ChDs;
  TVirtualPad*  fPadD_SCs_ChNb;
  TVirtualPad*  fPadD_SCs_ChDs; 
  TVirtualPad*  fPadD_MSp_Samp;
  TVirtualPad*  fPadD_SSp_Samp;
  TVirtualPad*  fPadD_Adc_EvDs;     
  TVirtualPad*  fPadD_Adc_EvNb;
  TVirtualPad*  fPadH_Ped_Date;
  TVirtualPad*  fPadH_TNo_Date;
  TVirtualPad*  fPadH_MCs_Date;
  TVirtualPad*  fPadH_LFN_Date;
  TVirtualPad*  fPadH_HFN_Date;
  TVirtualPad*  fPadH_SCs_Date;
  TVirtualPad*  fPadH_Ped_RuDs;
  TVirtualPad*  fPadH_TNo_RuDs;
  TVirtualPad*  fPadH_MCs_RuDs;
  TVirtualPad*  fPadH_LFN_RuDs;
  TVirtualPad*  fPadH_HFN_RuDs;
  TVirtualPad*  fPadH_SCs_RuDs;

  TPaveText*  fPavTxtH1SamePlus;
  TPaveText*  fPavTxtD_NOE_ChNb;
  TPaveText*  fPavTxtD_NOE_ChDs;
  TPaveText*  fPavTxtD_Ped_ChNb;
  TPaveText*  fPavTxtD_Ped_ChDs;
  TPaveText*  fPavTxtD_TNo_ChNb;   
  TPaveText*  fPavTxtD_TNo_ChDs; 
  TPaveText*  fPavTxtD_MCs_ChNb; 
  TPaveText*  fPavTxtD_MCs_ChDs;
  TPaveText*  fPavTxtD_LFN_ChNb;
  TPaveText*  fPavTxtD_LFN_ChDs; 
  TPaveText*  fPavTxtD_HFN_ChNb;   
  TPaveText*  fPavTxtD_HFN_ChDs; 
  TPaveText*  fPavTxtD_SCs_ChNb; 
  TPaveText*  fPavTxtD_SCs_ChDs; 
  TPaveText*  fPavTxtD_MSp_Samp;
  TPaveText*  fPavTxtD_SSp_Samp;  
  TPaveText*  fPavTxtD_Adc_EvDs;     
  TPaveText*  fPavTxtD_Adc_EvNb;
  TPaveText*  fPavTxtH_Ped_Date;
  TPaveText*  fPavTxtH_TNo_Date;
  TPaveText*  fPavTxtH_MCs_Date;
  TPaveText*  fPavTxtH_LFN_Date;
  TPaveText*  fPavTxtH_HFN_Date;
  TPaveText*  fPavTxtH_SCs_Date;
  TPaveText*  fPavTxtH_Ped_RuDs;
  TPaveText*  fPavTxtH_TNo_RuDs;
  TPaveText*  fPavTxtH_MCs_RuDs;
  TPaveText*  fPavTxtH_LFN_RuDs;
  TPaveText*  fPavTxtH_HFN_RuDs;
  TPaveText*  fPavTxtH_SCs_RuDs;

  TCanvasImp*  fImpH1SamePlus;
  TCanvasImp*  fImpD_NOE_ChNb;
  TCanvasImp*  fImpD_NOE_ChDs;
  TCanvasImp*  fImpD_Ped_ChNb;
  TCanvasImp*  fImpD_Ped_ChDs;
  TCanvasImp*  fImpD_TNo_ChNb;
  TCanvasImp*  fImpD_TNo_ChDs;
  TCanvasImp*  fImpD_MCs_ChNb;
  TCanvasImp*  fImpD_MCs_ChDs;
  TCanvasImp*  fImpD_LFN_ChNb;
  TCanvasImp*  fImpD_LFN_ChDs;
  TCanvasImp*  fImpD_HFN_ChNb;
  TCanvasImp*  fImpD_HFN_ChDs;
  TCanvasImp*  fImpD_SCs_ChNb;
  TCanvasImp*  fImpD_SCs_ChDs; 
  TCanvasImp*  fImpD_MSp_Samp;
  TCanvasImp*  fImpD_SSp_Samp;  
  TCanvasImp*  fImpD_Adc_EvDs; 
  TCanvasImp*  fImpD_Adc_EvNb;
  TCanvasImp*  fImpH_Ped_Date;
  TCanvasImp*  fImpH_TNo_Date;
  TCanvasImp*  fImpH_MCs_Date;
  TCanvasImp*  fImpH_LFN_Date;
  TCanvasImp*  fImpH_HFN_Date;
  TCanvasImp*  fImpH_SCs_Date;
  TCanvasImp*  fImpH_Ped_RuDs;
  TCanvasImp*  fImpH_TNo_RuDs;
  TCanvasImp*  fImpH_MCs_RuDs;
  TCanvasImp*  fImpH_LFN_RuDs;
  TCanvasImp*  fImpH_HFN_RuDs;
  TCanvasImp*  fImpH_SCs_RuDs;

  Int_t  fCanvSameH1SamePlus;
  Int_t  fCanvSameD_NOE_ChNb, fCanvSameD_NOE_ChDs;
  Int_t  fCanvSameD_Ped_ChNb, fCanvSameD_Ped_ChDs;
  Int_t  fCanvSameD_TNo_ChNb, fCanvSameD_TNo_ChDs; 
  Int_t  fCanvSameD_MCs_ChNb, fCanvSameD_MCs_ChDs;
  Int_t  fCanvSameD_LFN_ChNb, fCanvSameD_LFN_ChDs; 
  Int_t  fCanvSameD_HFN_ChNb, fCanvSameD_HFN_ChDs; 
  Int_t  fCanvSameD_SCs_ChNb, fCanvSameD_SCs_ChDs; 
  Int_t  fCanvSameD_MSp_Samp, fCanvSameD_SSp_Samp;
  Int_t  fCanvSameD_Adc_EvDs, fCanvSameD_Adc_EvNb;
  Int_t  fCanvSameH_Ped_Date, fCanvSameH_Ped_RuDs;
  Int_t  fCanvSameH_TNo_Date, fCanvSameH_TNo_RuDs;
  Int_t  fCanvSameH_LFN_Date, fCanvSameH_LFN_RuDs;
  Int_t  fCanvSameH_HFN_Date, fCanvSameH_HFN_RuDs;
  Int_t  fCanvSameH_MCs_Date, fCanvSameH_MCs_RuDs;      
  Int_t  fCanvSameH_SCs_Date, fCanvSameH_SCs_RuDs;

  Int_t  fNbOfListFileH_Ped_Date, fNbOfListFileH_TNo_Date, fNbOfListFileH_MCs_Date; // List file numbers
  Int_t  fNbOfListFileH_LFN_Date, fNbOfListFileH_HFN_Date, fNbOfListFileH_SCs_Date; // List file numbers
  Int_t  fNbOfListFileH_Ped_RuDs, fNbOfListFileH_TNo_RuDs, fNbOfListFileH_MCs_RuDs; // List file numbers
  Int_t  fNbOfListFileH_LFN_RuDs, fNbOfListFileH_HFN_RuDs, fNbOfListFileH_SCs_RuDs; // List file numbers

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

 public:

  //...................................... methods
  TEcnaHistos();
  TEcnaHistos(const TString);
  TEcnaHistos(const TString,
	     const TEcnaParPaths*,
	     const TEcnaParCout*,
	     const TEcnaParEcal*, 
	     const TEcnaParHistos*,
	     const TEcnaNumbering*,
	     const TEcnaWrite*);
  
  virtual  ~TEcnaHistos();

  void Init();
  void SetEcalSubDetector(const TString,
			  const TEcnaParEcal*, 
			  const TEcnaParHistos*,
			  const TEcnaNumbering*,
			  const TEcnaWrite*);

  //==================================== METHODS FOR THE USER =======================================

  //............ method to set the result file name parameters (from values in argument)
  //............ FileParameters(AnaType, [RunNumber], FirstEvent, NbOfEvts, [SM or Dee number])
  //             RunNumber = 0 => history plots , SM or Dee number = 0 => EB or EE Plots

  void FileParameters(const TString, const Int_t&, const Int_t&,
		      const Int_t&,  const Int_t&,  const Int_t&, const Int_t&);

  //................. methods for displaying the correlations and covariances matrices
  void LowFrequencyMeanCorrelationsBetweenTowers(const TString);  // USER: specific EB
  void LowFrequencyMeanCorrelationsBetweenSCs(const TString);     // USER: specific EE
  void HighFrequencyMeanCorrelationsBetweenTowers(const TString); // USER: specific EB
  void HighFrequencyMeanCorrelationsBetweenSCs(const TString);    // USER: specific EE

  void LowFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const TString); // USER: EB or EE
  void LowFrequencyCovariancesBetweenChannels(const Int_t&, const Int_t&, const TString);  // USER: EB or EE
  void HighFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const TString); // USER: EB or EE
  void HighFrequencyCovariancesBetweenChannels(const Int_t&, const Int_t&, const TString);  // USER: EB or EE

  void CorrelationsBetweenSamples(const Int_t&, const Int_t&, const TString);  // USER: EB or EE
  void CovariancesBetweenSamples(const Int_t&, const Int_t&, const TString);   // USER: EB or EE

  //................. methods for displaying 2D view of the whole detector; 2D(eta,Phi) for EB, 2D(IX,IY) for EE
  void EBEtaPhiAveragedNumberOfEvents();     // USER: specific EB
  void EBEtaPhiAveragedPedestals();          // USER: specific EB
  void EBEtaPhiAveragedTotalNoise();         // USER: specific EB
  void EBEtaPhiAveragedMeanOfCorss();        // USER: specific EB
  void EBEtaPhiAveragedLowFrequencyNoise();  // USER: specific EB
  void EBEtaPhiAveragedHighFrequencyNoise(); // USER: specific EB
  void EBEtaPhiAveragedSigmaOfCorss();       // USER: specific EB

  void EEIXIYAveragedNumberOfEvents();     // USER: specific EE
  void EEIXIYAveragedPedestals();          // USER: specific EE
  void EEIXIYAveragedTotalNoise();         // USER: specific EE
  void EEIXIYAveragedMeanOfCorss();        // USER: specific EE
  void EEIXIYAveragedLowFrequencyNoise();  // USER: specific EE
  void EEIXIYAveragedHighFrequencyNoise(); // USER: specific EE
  void EEIXIYAveragedSigmaOfCorss();       // USER: specific EE

  //................. methods for displaying the SM 2D(eta,phi) view
  void SMEtaPhiNumberOfEvents();       // USER: specific EB
  void SMEtaPhiPedestals();            // USER: specific EB
  void SMEtaPhiTotalNoise();           // USER: specific EB
  void SMEtaPhiMeanOfCorss();          // USER: specific EB
  void SMEtaPhiLowFrequencyNoise();    // USER: specific EB
  void SMEtaPhiHighFrequencyNoise();   // USER: specific EB
  void SMEtaPhiSigmaOfCorss();         // USER: specific EB

  void SMEtaPhiLowFrequencyCorcc();    // USER: specific EB
  void SMEtaPhiHighFrequencyCorcc();   // USER: specific EB

  //................. methods for displaying the Dee 2D(IX,IY) view
  void DeeIXIYNumberOfEvents();        // USER: specific EE
  void DeeIXIYPedestals();             // USER: specific EE
  void DeeIXIYTotalNoise();            // USER: specific EE
  void DeeIXIYMeanOfCorss();           // USER: specific EE
  void DeeIXIYLowFrequencyNoise();     // USER: specific EE
  void DeeIXIYHighFrequencyNoise();    // USER: specific EE
  void DeeIXIYSigmaOfCorss();          // USER: specific EE

  void DeeIXIYLowFrequencyCorcc();     // USER: specific EE
  void DeeIXIYHighFrequencyCorcc();    // USER: specific EE

  //...... methods for displaying 1D histos, explicit option plot argument ("ONLYONE", "SAME" or "ASCII")

  //............ EE or EE, "Averaged"
  void EBXtalsAveragedNumberOfEvents(const TString);     // USER: specific EB
  void EBXtalsAveragedPedestals(const TString);          // USER: specific EB
  void EBXtalsAveragedTotalNoise(const TString);         // USER: specific EB
  void EBXtalsAveragedMeanOfCorss(const TString);        // USER: specific EB
  void EBXtalsAveragedLowFrequencyNoise(const TString);  // USER: specific EB
  void EBXtalsAveragedHighFrequencyNoise(const TString); // USER: specific EB
  void EBXtalsAveragedSigmaOfCorss(const TString);       // USER: specific EB

  void EBAveragedNumberOfEventsXtals(const TString);     // USER: specific EB
  void EBAveragedPedestalsXtals(const TString);          // USER: specific EB
  void EBAveragedTotalNoiseXtals(const TString);         // USER: specific EB
  void EBAveragedMeanOfCorssXtals(const TString);        // USER: specific EB
  void EBAveragedLowFrequencyNoiseXtals(const TString);  // USER: specific EB
  void EBAveragedHighFrequencyNoiseXtals(const TString); // USER: specific EB
  void EBAveragedSigmaOfCorssXtals(const TString);       // USER: specific EB
 
  void EEXtalsAveragedNumberOfEvents(const TString);     // USER: specific EE
  void EEXtalsAveragedPedestals(const TString);          // USER: specific EE
  void EEXtalsAveragedTotalNoise(const TString);         // USER: specific EE
  void EEXtalsAveragedMeanOfCorss(const TString);        // USER: specific EE
  void EEXtalsAveragedLowFrequencyNoise(const TString);  // USER: specific EE
  void EEXtalsAveragedHighFrequencyNoise(const TString); // USER: specific EE
  void EEXtalsAveragedSigmaOfCorss(const TString);       // USER: specific EE

  void EEAveragedNumberOfEventsXtals(const TString);     // USER: specific EE
  void EEAveragedPedestalsXtals(const TString);          // USER: specific EE
  void EEAveragedTotalNoiseXtals(const TString);         // USER: specific EE
  void EEAveragedMeanOfCorssXtals(const TString);        // USER: specific EE
  void EEAveragedLowFrequencyNoiseXtals(const TString);  // USER: specific EE
  void EEAveragedHighFrequencyNoiseXtals(const TString); // USER: specific EE
  void EEAveragedSigmaOfCorssXtals(const TString);       // USER: specific EE

  //............ SM or Dee
  void SMXtalsNumberOfEvents(const TString);     // USER: specific EB
  void SMXtalsPedestals(const TString);          // USER: specific EB
  void SMXtalsTotalNoise(const TString);         // USER: specific EB
  void SMXtalsMeanOfCorss(const TString);        // USER: specific EB
  void SMXtalsLowFrequencyNoise(const TString);  // USER: specific EB
  void SMXtalsHighFrequencyNoise(const TString); // USER: specific EB
  void SMXtalsSigmaOfCorss(const TString);       // USER: specific EB

  void SMNumberOfEventsXtals(const TString);     // USER: specific EB
  void SMPedestalsXtals(const TString);          // USER: specific EB
  void SMTotalNoiseXtals(const TString);         // USER: specific EB
  void SMMeanOfCorssXtals(const TString);        // USER: specific EB
  void SMLowFrequencyNoiseXtals(const TString);  // USER: specific EB
  void SMHighFrequencyNoiseXtals(const TString); // USER: specific EB
  void SMSigmaOfCorssXtals(const TString);       // USER: specific EB
 
  void DeeXtalsNumberOfEvents(const TString);     // USER: specific EE
  void DeeXtalsPedestals(const TString);          // USER: specific EE
  void DeeXtalsTotalNoise(const TString);         // USER: specific EE
  void DeeXtalsMeanOfCorss(const TString);        // USER: specific EE
  void DeeXtalsLowFrequencyNoise(const TString);  // USER: specific EE
  void DeeXtalsHighFrequencyNoise(const TString); // USER: specific EE
  void DeeXtalsSigmaOfCorss(const TString);       // USER: specific EE

  void DeeNumberOfEventsXtals(const TString);     // USER: specific EE
  void DeePedestalsXtals(const TString);          // USER: specific EE
  void DeeTotalNoiseXtals(const TString);         // USER: specific EE
  void DeeMeanOfCorssXtals(const TString);        // USER: specific EE
  void DeeLowFrequencyNoiseXtals(const TString);  // USER: specific EE
  void DeeHighFrequencyNoiseXtals(const TString); // USER: specific EE
  void DeeSigmaOfCorssXtals(const TString);       // USER: specific EE

  //.......... Others
  void XtalSamplesEv(const Int_t&, const Int_t&, const TString);
  void XtalSamplesSigma(const Int_t&, const Int_t&, const TString);

  void XtalSampleValues(const Int_t&, const Int_t&, const Int_t&, const TString);
  void SampleADCEvents(const Int_t&, const Int_t&, const Int_t&, const TString);

  //........ methods for displaying 1D histos no option plot argument (default = "ONLYONE")

  //............ EE or EE, "Averaged"
  void EBXtalsAveragedNumberOfEvents();     // USER: specific EB
  void EBXtalsAveragedPedestals();          // USER: specific EB
  void EBXtalsAveragedTotalNoise();         // USER: specific EB
  void EBXtalsAveragedMeanOfCorss();        // USER: specific EB
  void EBXtalsAveragedLowFrequencyNoise();  // USER: specific EB
  void EBXtalsAveragedHighFrequencyNoise(); // USER: specific EB
  void EBXtalsAveragedSigmaOfCorss();       // USER: specific EB

  void EBAveragedNumberOfEventsXtals();     // USER: specific EB
  void EBAveragedPedestalsXtals();          // USER: specific EB
  void EBAveragedTotalNoiseXtals();         // USER: specific EB
  void EBAveragedMeanOfCorssXtals();        // USER: specific EB
  void EBAveragedLowFrequencyNoiseXtals();  // USER: specific EB
  void EBAveragedHighFrequencyNoiseXtals(); // USER: specific EB
  void EBAveragedSigmaOfCorssXtals();       // USER: specific EB
 
  void EEXtalsAveragedNumberOfEvents();     // USER: specific EE
  void EEXtalsAveragedPedestals();          // USER: specific EE
  void EEXtalsAveragedTotalNoise();         // USER: specific EE
  void EEXtalsAveragedMeanOfCorss();        // USER: specific EE
  void EEXtalsAveragedLowFrequencyNoise();  // USER: specific EE
  void EEXtalsAveragedHighFrequencyNoise(); // USER: specific EE
  void EEXtalsAveragedSigmaOfCorss();       // USER: specific EE

  void EEAveragedNumberOfEventsXtals();     // USER: specific EE
  void EEAveragedPedestalsXtals();          // USER: specific EE
  void EEAveragedTotalNoiseXtals();         // USER: specific EE
  void EEAveragedMeanOfCorssXtals();        // USER: specific EE
  void EEAveragedLowFrequencyNoiseXtals();  // USER: specific EE
  void EEAveragedHighFrequencyNoiseXtals(); // USER: specific EE
  void EEAveragedSigmaOfCorssXtals();       // USER: specific EE

  //............ SM or Dee
  void SMXtalsNumberOfEvents();     // USER: specific EB
  void SMXtalsPedestals();          // USER: specific EB // (sample ADC value->mean over events)->mean over samples
  void SMXtalsTotalNoise();         // USER: specific EB // 
  void SMXtalsMeanOfCorss();        // USER: specific EB // MeanOfCorss
  void SMXtalsLowFrequencyNoise();  // USER: specific EB // 
  void SMXtalsHighFrequencyNoise(); // USER: specific EB // 
  void SMXtalsSigmaOfCorss();       // USER: specific EB // 

  void SMNumberOfEventsXtals();      // USER: specific EB
  void SMPedestalsXtals();           // USER: specific EB // 
  void SMTotalNoiseXtals();          // USER: specific EB // 
  void SMMeanOfCorssXtals();         // USER: specific EB //
  void SMLowFrequencyNoiseXtals();   // USER: specific EB // 
  void SMHighFrequencyNoiseXtals();  // USER: specific EB // 
  void SMSigmaOfCorssXtals();        // USER: specific EB //

  void DeeXtalsNumberOfEvents();     // USER: specific EE
  void DeeXtalsPedestals();          // USER: specific EE
  void DeeXtalsTotalNoise();         // USER: specific EE
  void DeeXtalsMeanOfCorss();        // USER: specific EE
  void DeeXtalsLowFrequencyNoise();  // USER: specific EE
  void DeeXtalsHighFrequencyNoise(); // USER: specific EE
  void DeeXtalsSigmaOfCorss();       // USER: specific EE

  void DeeNumberOfEventsXtals();     // USER: specific EE
  void DeePedestalsXtals();          // USER: specific EE
  void DeeTotalNoiseXtals();         // USER: specific EE
  void DeeMeanOfCorssXtals();        // USER: specific EE
  void DeeLowFrequencyNoiseXtals();  // USER: specific EE
  void DeeHighFrequencyNoiseXtals(); // USER: specific EE
  void DeeSigmaOfCorssXtals();       // USER: specific EE

  //.......... Others
  void XtalSamplesEv(const Int_t&, const Int_t&);
  void XtalSamplesSigma(const Int_t&, const Int_t&);
  void XtalSampleValues(const Int_t&, const Int_t&, const Int_t&);

  void SampleADCEvents(const Int_t&, const Int_t&, const Int_t&);

  //....... methods for displaying evolution in time histos, explicit option plot argument
  void XtalTimePedestals(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTimeTotalNoise(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTimeMeanOfCorss(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTimeLowFrequencyNoise(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTimeHighFrequencyNoise(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTimeSigmaOfCorss(const TString, const Int_t&, const Int_t&, const TString);

  void XtalPedestalsRuns(const TString, const Int_t&, const Int_t&, const TString);
  void XtalTotalNoiseRuns(const TString, const Int_t&, const Int_t&, const TString);
  void XtalMeanOfCorssRuns(const TString, const Int_t&, const Int_t&, const TString);
  void XtalLowFrequencyNoiseRuns(const TString, const Int_t&, const Int_t&, const TString);
  void XtalHighFrequencyNoiseRuns(const TString, const Int_t&, const Int_t&, const TString);
  void XtalSigmaOfCorssRuns(const TString, const Int_t&, const Int_t&, const TString);

  //....... methods for displaying evolution in time histos, no option plot argument (default)
  void XtalTimePedestals(const TString, const Int_t&, const Int_t&);
  void XtalTimeTotalNoise(const TString, const Int_t&, const Int_t&);
  void XtalTimeMeanOfCorss(const TString, const Int_t&, const Int_t&);
  void XtalTimeLowFrequencyNoise(const TString, const Int_t&, const Int_t&);
  void XtalTimeHighFrequencyNoise(const TString, const Int_t&, const Int_t&);
  void XtalTimeSigmaOfCorss(const TString, const Int_t&, const Int_t&);

  void XtalPedestalsRuns(const TString, const Int_t&, const Int_t&);
  void XtalTotalNoiseRuns(const TString, const Int_t&, const Int_t&);
  void XtalMeanOfCorssRuns(const TString, const Int_t&, const Int_t&);
  void XtalLowFrequencyNoiseRuns(const TString, const Int_t&, const Int_t&);
  void XtalHighFrequencyNoiseRuns(const TString, const Int_t&, const Int_t&);
  void XtalSigmaOfCorssRuns(const TString, const Int_t&, const Int_t&);

  //................. methods for displaying Tower, SC, crystal numbering

  void SMTowerNumbering(const Int_t&);  // USER: specific EB
  void DeeSCNumbering(const Int_t&);             // USER: specific EE

  void TowerCrystalNumbering(const Int_t&, const Int_t&);  // USER: specific EB
  void SCCrystalNumbering(const Int_t&, const Int_t&);     // USER: specific EE

  //.................... General title
  void GeneralTitle(const TString);

  //.................................. Lin:Log scale, ColorPalette, General Title
  void SetHistoScaleX(const TString);
  void SetHistoScaleY(const TString);
  void SetHistoColorPalette(const TString);

  //.................................. 1D and 2D histo min,max user's values
  void SetHistoMin(const Double_t&);
  void SetHistoMax(const Double_t&);
  //.................................. 1D and 2D histo min,max from histo values
  void SetHistoMin();
  void SetHistoMax();

  //======================= TECHNICAL METHODS (in principle not for the user) ========================

  void SetGeneralTitle(const TString);

  void SetRunNumberFromList(const Int_t&, const Int_t&);  // called by Histime
  void InitSpecParBeforeFileReading();                    // set parameters from the file reading

  Int_t GetNumberOfEvents(TEcnaRead* , const Int_t&);

  //................. methods for displaying the cor(s,s) and cov(s,s) corresponding to a Stin
  void CorrelationsBetweenSamples(const Int_t&);
  void CovariancesBetweenSamples(const Int_t&);

  void LowFrequencyMeanCorrelationsBetweenStins(const TString);  
  void HighFrequencyMeanCorrelationsBetweenStins(const TString);   

  void StasHocoVecoAveragedNumberOfEvents(); 
  void StasHocoVecoAveragedPedestals(); 
  void StasHocoVecoAveragedTotalNoise(); 
  void StasHocoVecoAveragedMeanOfCorss(); 
  void StasHocoVecoAveragedLowFrequencyNoise(); 
  void StasHocoVecoAveragedHighFrequencyNoise(); 
  void StasHocoVecoAveragedSigmaOfCorss(); 

  void StexHocoVecoNumberOfEvents();               
  void StexHocoVecoPedestals();             
  void StexHocoVecoTotalNoise();        
  void StexHocoVecoMeanOfCorss();               
  void StexHocoVecoLowFrequencyNoise();            
  void StexHocoVecoHighFrequencyNoise();       
  void StexHocoVecoSigmaOfCorss();              
  void StexHocoVecoLHFCorcc(const TString);

  void StexStinNumbering(const Int_t&);                    
  void StinCrystalNumbering(const Int_t&, const Int_t&);   

  void StexXtalsNumberOfEvents(const TString);          
  void StexXtalsPedestals(const TString);        
  void StexXtalsTotalNoise(const TString);   
  void StexXtalsMeanOfCorss(const TString);          
  void StexXtalsLowFrequencyNoise(const TString);       
  void StexXtalsHighFrequencyNoise(const TString);  
  void StexXtalsSigmaOfCorss(const TString);

  void StexNumberOfEventsXtals(const TString);
  void StexPedestalsXtals(const TString);
  void StexTotalNoiseXtals(const TString);
  void StexMeanOfCorssXtals(const TString);
  void StexLowFrequencyNoiseXtals(const TString);
  void StexHighFrequencyNoiseXtals(const TString);
  void StexSigmaOfCorssXtals(const TString);

  void StexXtalsNumberOfEvents();
  void StexXtalsPedestals();
  void StexXtalsTotalNoise();
  void StexXtalsMeanOfCorss();
  void StexXtalsLowFrequencyNoise();
  void StexXtalsHighFrequencyNoise();
  void StexXtalsSigmaOfCorss();

  void StexNumberOfEventsXtals();
  void StexPedestalsXtals();
  void StexTotalNoiseXtals();
  void StexMeanOfCorssXtals();
  void StexLowFrequencyNoiseXtals();
  void StexHighFrequencyNoiseXtals();
  void StexSigmaOfCorssXtals();

  void ViewStas(const TString);
  void ViewStex(const TString);
  void ViewStin(const Int_t&, const TString);
  void ViewMatrix(const Int_t&,  const Int_t&,  const Int_t&,
		  const TString, const TString, const TString);
  void ViewHisto(const Int_t&,  const Int_t&, const Int_t&,
		 const TString, const TString);

  Int_t GetDSOffset(const Int_t&, const Int_t&);
  Int_t GetSCOffset(const Int_t&, const Int_t&, const Int_t&);

  void ViewHistime(const TString, const Int_t&, const Int_t&,
		   const TString, const TString);

  Int_t GetHistoryRunListParameters(const TString, const TString);

  void TopAxisForHistos(TH1D*,
			const TString, const Int_t&, const Int_t&, const Int_t&,
			const Int_t&,  const Int_t& );

  //--------------------------------------------------------------- xinf, xsup management
  void     SetXinfMemoFromValue(const TString, const Double_t&);
  void     SetXsupMemoFromValue(const TString, const Double_t&);
  void     SetXinfMemoFromValue(const Double_t&);
  void     SetXsupMemoFromValue(const Double_t&);

  Double_t GetXinfValueFromMemo(const TString);
  Double_t GetXsupValueFromMemo(const TString);
  Double_t GetXinfValueFromMemo();
  Double_t GetXsupValueFromMemo();

  Axis_t   GetHistoXinf(const TString, const Int_t&, const TString);
  Axis_t   GetHistoXsup(const TString, const Int_t&, const TString);

  Int_t    GetHistoNumberOfBins(const TString,  const Int_t&); 

  //--------------------------------------------------------------- ymin, ymax management
  void     SetYminMemoFromValue(const TString, const Double_t&);
  void     SetYmaxMemoFromValue(const TString, const Double_t&);

  Double_t GetYminValueFromMemo(const TString);
  Double_t GetYmaxValueFromMemo(const TString);

  void     SetYminMemoFromPreviousMemo(const TString);
  void     SetYmaxMemoFromPreviousMemo(const TString);

  Int_t    SetHistoFrameYminYmaxFromMemo(TH1D*,   const TString);
  Int_t    SetGraphFrameYminYmaxFromMemo(TGraph*, const TString);

  Double_t GetYminFromHistoFrameAndMarginValue(TH1D*, const Double_t);
  Double_t GetYmaxFromHistoFrameAndMarginValue(TH1D*, const Double_t);

  Double_t GetYminFromGraphFrameAndMarginValue(TGraph*, const Double_t);
  Double_t GetYmaxFromGraphFrameAndMarginValue(TGraph*, const Double_t);

  //.................................. 1D and 2D histo min,max default values
  void SetAllYminYmaxMemoFromDefaultValues();

  //------------------------------------------------- Memo Same, Same n management
  void    SetXVarMemo(const TString, const TString, const TString);
  TString GetXVarFromMemo(const TString, const TString);

  void    SetYVarMemo(const TString, const TString, const TString);
  TString GetYVarFromMemo(const TString, const TString);

  void    SetNbBinsMemo(const TString, const TString, const Int_t&);
  Int_t   GetNbBinsFromMemo(const TString, const TString);

  //--------------------------------------------------------------------------------
  void ViewStexStinNumberingPad(const Int_t&);
  void ViewSMTowerNumberingPad(const Int_t&);   // specific EB
  void ViewDeeSCNumberingPad(const Int_t&);     // specific EE

  void ViewStinGrid(const Int_t&, const Int_t&, const Int_t&,
		    const Int_t&, const Int_t&, const TString);
  void ViewTowerGrid(const Int_t&, const Int_t&, const Int_t&,
		     const Int_t&, const Int_t&, const TString);  // specific EB
  void ViewSCGrid(const Int_t&, const Int_t&, const Int_t&,
		  const Int_t&, const Int_t&, const TString);     // specific EE

  void ViewStexGrid(const Int_t&, const TString);
  void ViewSMGrid(const Int_t&, const TString);    // specific EB
  void ViewDeeGrid(const Int_t&, const TString);   // specific EE

  void ViewStasGrid(const Int_t&);
  void ViewEBGrid();
  void ViewEEGrid(const Int_t&);

  void EEDataSectors(const Float_t&,  const Float_t&, const Int_t&, const TString);
  void EEGridAxis(const Float_t&,  const Float_t&, const Int_t&, const TString, const TString);

  void SqrtContourLevels(const Int_t&, Double_t*);

  TString StexNumberToString(const Int_t&);

  void HistoPlot(TH1D*,
		 const Int_t&,  const Axis_t&,  const Axis_t&,  const TString, const TString,
		 const Int_t&,  const Int_t&,   const Int_t&,   const Int_t&,
		 const Int_t&,  const TString,  const Int_t&);

  Double_t NotConnectedSCH1DBin(const Int_t&);
  Int_t    GetNotConnectedDSSCFromIndex(const Int_t&);
  Int_t    GetNotConnectedSCForConsFromIndex(const Int_t&);
  Int_t    ModifiedSCEchaForNotConnectedSCs(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  Double_t NotCompleteSCH1DBin(const Int_t&);
  Int_t    GetNotCompleteDSSCFromIndex(const Int_t&);
  Int_t    GetNotCompleteSCForConsFromIndex(const Int_t&);

  void HistimePlot(TGraph*,       Axis_t,        Axis_t,
		   const TString, const TString, const Int_t&, const Int_t&,
		   const Int_t&,  const Int_t&,  const Int_t&, const TString, const Int_t&);

  void SetAllPavesViewMatrix(const TString, const Int_t&,
			     const Int_t&, const Int_t&);
  void SetAllPavesViewStin(const Int_t&);
  void SetAllPavesViewStex(const TString, const Int_t&);
  void SetAllPavesViewStex(const Int_t&);
  void SetAllPavesViewStas();
  void SetAllPavesViewStinCrysNb(const Int_t&, const Int_t&);
  void SetAllPavesViewHisto(const TString,
			    const Int_t&, const Int_t&, const Int_t&, const TString);

  Int_t GetXSampInStin(const Int_t&, const Int_t&,
		       const Int_t&,  const Int_t&);
  Int_t GetYSampInStin(const Int_t&, const Int_t&,
		       const Int_t&,  const Int_t&);

  Int_t GetXCrysInStex(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetYCrysInStex(const Int_t&, const Int_t&, const Int_t&);

  Int_t GetXStinInStas(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetYStinInStas(const Int_t&, const Int_t&);


  TString GetHocoVecoAxisTitle(const TString);
  TString GetEtaPhiAxisTitle(const TString);        // specific EB
  TString GetIXIYAxisTitle(const TString);          // specific EE

  Bool_t   GetOkViewHisto(TEcnaRead*, const Int_t&, const Int_t&, const Int_t&, const TString);
  Int_t    GetHistoSize(const TString, const TString);
  TVectorD GetHistoValues(TEcnaRead*,    const TString, const Int_t&, const Int_t&,
			  const Int_t&, const Int_t&,  const Int_t&, Int_t&);

  TString SetHistoXAxisTitle(const TString);
  TString SetHistoYAxisTitle(const TString);

  void FillHisto(TH1D*, const TVectorD&, const TString, const Int_t&);

  TString GetMemoFlag(const TString);
  TString GetMemoFlag(const TString, const TString);

  TCanvas* CreateCanvas(const TString, const TString, const TString, UInt_t,  UInt_t);
  TCanvas* GetCurrentCanvas(const TString, const TString);
  TCanvas* GetCurrentCanvas();
  TString  GetCurrentCanvasName();
  void     PlotCloneOfCurrentCanvas();

  void SetParametersCanvas(const TString, const TString);
  void SetParametersPavTxt(const TString, const TString);

  TVirtualPad* ActivePad(const TString, const TString);
  TPaveText*   ActivePavTxt(const TString, const TString);

  void SetHistoPresentation(TH1D*,   const TString);
  void SetHistoPresentation(TH1D*,   const TString, const TString);
  void SetGraphPresentation(TGraph*, const TString, const TString);

  void SetViewHistoColors(TH1D*,   const TString, const TString);
  void SetViewGraphColors(TGraph*, const TString, const TString);

  Color_t GetViewHistoColor(const TString, const TString);

  Int_t GetListFileNumber(const TString);
  void  ReInitCanvas(const TString, const TString);
  void  NewCanvas(const TString);

  TString SetCanvasName(const TString, const Int_t&, const Int_t&, 
			const TString, const Int_t&, const Int_t&, const Int_t&);

  Color_t GetSCColor(const TString, const TString, const TString);     // specific EE

  void WriteMatrixAscii(const TString, const TString, const Int_t&, const Int_t&, const Int_t&, const TMatrixD&);
  void WriteHistoAscii(const TString, const Int_t&, const TVectorD&);

  TString  AsciiFileName();
  Bool_t StatusFileFound();
  Bool_t StatusDataExist();

ClassDef(TEcnaHistos,1)// methods for plots from ECNA (Ecal Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaHistos
