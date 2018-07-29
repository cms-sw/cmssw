#ifndef ZTR_TEcnaHistos
#define ZTR_TEcnaHistos

#include "TObject.h"
#include <TQObject.h>
#include <RQ_OBJECT.h>
//#include <Riostream.h>
#include <iostream>
#include "TSystem.h"
#include <ctime>
#include "TString.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TVectorD.h"
#include "TH1.h"
#include "TH2D.h"
#include "TF1.h"
#include "TPaveText.h"
#include "TColor.h"
#include "TGaxis.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

///-----------------------------------------------------------
///   TEcnaHistos.h
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
///==============> INTRODUCTION
///
///    This class provides methods for displaying plots of various types:
///    1D, 2D and 3D histograms for different quantities (pedestals, noises,
///    correlations, .etc..). The data are read from files which has been
///    previously written by using the class TEcnaRun (.root result files).
///    The reading is performed by appropriate methods of the class TEcnaRead.
///
///      ***   I N S T R U C T I O N S   F O R   U S E   ***
///
///   PS: examples of programs using TEcnaHistos are situated in directory test (.cc files)
///
///   // (A) --> Object declarations:
///
///      TEcnaHistos* MyHistosEB = new TEcnaHistos("EB");
///      TEcnaHistos* MyHistosEE = new TEcnaHistos("EE");
///      
///   // (B) --> Specification of the file which has to be read.
///   //         This file is a .root result file which has previously been written by means of
///   //         the class TEcnaRun (see documentation of this class)
///
///   //  1) specify the parameter values of the file name:
///
///      TString AnalysisName      = "StdPed12"; (AnalysisName -> explanation in TEcnaRun documentation)
///      Int_t   NbOfSamples       = 10;
///      Int_t   RunNumber         = 112206;
///      Int_t   FirstReqEvtNumber = 100;  |    (numbering starting from 1)
///      Int_t   LastReqEvtNumber  = 300;  | => treats 150 evts between evt#100 and evt#300 (included)
///      Int_t   ReqNbOfEvts       = 150;  |
///      Int_t   SMNumber          = 2;
///
///   //  2) call method "FileParameters" to say that the file to be read is the file
///   //     which name parameters are those specified above:
/// 
///      MyHistosEB->FileParameters(AnalysisName, NbOfSamples, RunNumber,
///                                 FirstReqEvtNumber, LastReqEvtNumber, ReqNbOfEvts, SMNumber); 
///
///   //  Now, the class TEcnaHistos knowns that it has to work
///   //  with the file StdPed12_S1_10_R112206_1_150_150_SM2.root
///
///   // (C) -->  E X A M P L E S   O F   U S E 
/// 
///   //--> Plot correlation matrices between samples
///   //    for channel 21 (electronic channel number in Tower)
///   //    and for the two towers 10 and 33
///
///      Int_t SMTow = 14;       // (Tower number in SM)
///      Int_t TowEcha = 21;     // (Electronic channel number in tower)
///      MyHistosEB->PlotMatrix("Cor", "Mss", SMTow, TowEcha, "SURF1Z"); // correlations between samples
///
///    // - - - - - -   SYNTAX FOR ARGUMENTS CODES ("Cor", "Mss", ...):
///
///      Multi-syntax codes are available, for example:
///
///      "Cor", "correl", "correlations", "Correlations"
///
///      "Mss", "ss", "samp", "Samp", "BetweenSamples", "Between samples", "between samples", "Samples","samples"
///
///      If a wrong code is used a message "code not found" is displayed
///      and information on available codes is given.
///
///
///   // (D) -->  O T H E R   E X A M P L E S
///
///    //--> Plot Pedestals as a function of SC number for Dee 4 
///
///      TString AnalysisName      = "StdPed12"; (AnalysisName -> explanation in TEcnaRun documentation)
///      Int_t   NbOfSamples       = 10;
///      Int_t   RunNumber         = 132440;
///      Int_t   FirstReqEvtNumber = 1;    |
///      Int_t   LastReqEvtNumber  = 0;    | => treats 150 evts from evt#1 until EOF if necessary
///      Int_t   ReqNbOfEvts       = 150;  |
///
///      Int_t   DeeNumber = 4;
///      MyHistosEE->FileParameters(AnalysisName, NbOfSamples, RunNumber,
///                                 FirstReqEvtNumber, LastReqEvtNumber, ReqNbOfEvts, DeeNumber); 
///
///      MyHistoEE->PlotDetector("Ped", "Dee");          // 2D Histo: Z = pedestal, detector = Dee
///      MyHistoEE->Plot1DHisto("Tow", "TNo", "EE");     // 1D Histo: X = tower#, Y = Total noise, detector = EE
///
///    //--> Plot total noise history for channel 12 of tower 38
///    //    (electronic channel number in tower)
///
///      MyHistosEB->FileParameters(AnalysisName, NbOfSamples, RunNumber,
///                                 FirstReqEvtNumber, LastReqEvtNumber, ReqNbOfEvts, SMNumber);
///      Int_t SMTower = 38;
///      Int_t TowEcha = 12;
///      TString list_of_run_file_name = "HistoryRunList_132440_132665.ascii";
///      MyHistoEB->PlotHistory("Time", "MeanCorss", list_of_run_file_name, SMTower, TowEcha);
///      
///    // the .ascii file "HistoryRunList_132440_132665.ascii" must contain a list of
///    // the run numbers according to the following syntax:
///
///   //.......... SYNTAX OF THE FILE "HistoryRunList_SM6.ascii" ("runlist history plot" file):
///
///   HistoryRunList_132440_132665.ascii  <- 1rst line: comment (name of the file, for example)
///
///   132440                <- other lines: run numbers
///   132442
///                        <- (empty lines can be used)
///   132561
///   132562
///   112584
///
///
///      etc...
///
///   132665
///
///......................................... end of exammples ...........................................
///
///   PS: it is also possible to use the methods PlotMatrix, PlotDetector and Plot1DHisto after
///       reading file with TEcnaRead. Then, pointers to the read arrays have to be used
///       as arguments (see examples in directory test)
///
///------------------------------------------- LIST OF METHODS ------------------------------------------
///
///  //==================================================================================================
///  //   method to set the result file name parameters (from values in argument)
///  //   FileParameters(AnaType, NbOfSamples, Run#,
///  //                  FirstRequestedEvt#, LastRequestedEvt#, ReqNbOfEvts#, SM# or Dee#)
///  //   RunNumber = 0 => history plots, SM or Dee number = 0 => Plots for EB or EE
///  //==================================================================================================
///
///  void FileParameters(const TString& Analysis,         
///                      const Int_t&  NbOfSamples,
///                      const Int_t&  Run#,               // RunNumber = 0 => history plots
///		         const Int_t&  FirstRequestedEvt#,
///                      const Int_t&  LastRequestedEvt#,
///                      const Int_t&  ReqNbOfEvts#,
///                      const Int_t&  SMOrDee#);          // SM or Dee number = 0 => Plots for EB or EE
///
///        -----------------------------------------------------------------------------------
///        In the following:
///
///        TowOrSC# = Tower number in case of EB  or  SC number FOR CONSTRUCTION in case of EE
///
///        -----------------------------------------------------------------------------------
///
///  //==================================================================================================
///  //                  methods for displaying the correlations and covariances matrices
///  //                  PlotOption = ROOT DrawOption ("SAME", "LEGO", "COLZ", etc...)
///  //                               + option "ASCII": write histo in ASCII file 
///  //==================================================================================================
///  //..................... Corcc[for 1 Stex] (big matrix)
///    void PlotMatrix
///   (const TMatrixD&,   const TString&, const TString&,   [const TString&]);
///    read_matrix_corcc, UserCorOrCov,  UserBetweenWhat, [PlotOption]
///
///    void PlotMatrix   (const TString&, const TString&,   [const TString&]);
///                       UserCorOrCov,  UserBetweenWhat, [PlotOption]
///  
///    //..................... Corcc[for 1 Stin], Corss[for 1 Echa], Covss[for 1 Echa]
///    void PlotMatrix
///    (const TMatrixD&, const TString&, const TString&,   const Int_t&, const Int_t&, [const TString&]);
///     read_matrix,     UserCorOrCov,  UserBetweenWhat, arg_n1,       arg_n2,       [PlotOption]
///
///    void PlotMatrix  (const TString&, const TString&,   const Int_t&, const Int_t&, [const TString&]);
///                      UserCorOrCov,  UserBetweenWhat, arg_n1,       arg_n2,       [PlotOption]
///  
///  //==================================================================================================
///  //                  methods for displaying the 2D views of the detector
///  //
///  //                  Detector = SM,Dee,EB,EE
///  //
///  //==================================================================================================
///    void PlotDetector(const TVectorD&, const TString&, const TString&);
///                      read_histo,      UserHistoCode, Detector,
///
///    void PlotDetector(const TString&, const TString&);
///                      UserHistoCode, Detector
///  
///  //==================================================================================================
///  //                             methods for displaying 1D histos
///  //
///  //     PlotOption: optional argument ("ONLYONE", "SAME","SAME n"  or "ASCII")
///  //      
///  //  "ONLYONE" :  display only one histo (default; same as without argument)
///  //  "SAME"    :  Same as Draw Option "SAME" in ROOT: superimpose on previous picture in the same pad
///  //               1D histos of only one quantity
///  //  "SAME n"  :  Same as Draw Option "SAME" in ROOT: superimpose on previous picture in the same pad
///  //               1D histos of possibly several quantities
///  //  "ASCII"   :  write histo contents in ASCII file
///  //
///  //==================================================================================================
///
///    void Plot1DHisto
///    (const TVectorD&, const TString&,   const TString&,   const TString&, [const TString&]);
///     InputHisto,      User_X_Quantity, User_Y_Quantity, Detector,      [PlotOption]
///
///    void Plot1DHisto (const TString&,   const TString&,   const TString&, [const TString&]);
///                      User_X_Quantity, User_Y_Quantity, Detector,	  [PlotOption])
///
///
///    void Plot1DHisto
///    (const TVectorD&, const TString&,   const TString&,   const Int_t&, const Int_t&, [const TString&]);
///     InputHisto,      User_X_Quantity, User_Y_Quantity, n1StexStin,   i0StinEcha,   [PlotOption]
///
///    void Plot1DHisto (const TString&,   const TString&,   const Int_t&, const Int_t&,  [const TString&]);
///                      User_X_Quantity, User_Y_Quantity, n1StexStin,   i0StinEcha,    [PlotOption]
///
///
///    void Plot1DHisto
///    (const TVectorD&, const TString&,   const TString&,   const Int_t&, const Int_t&, const Int_t&, [const TString&]);
///     InputHisto,      User_X_Quantity, User_Y_Quantity, n1StexStin,   i0StinEcha,   n1Sample,     [PlotOption]
///
///    void Plot1DHisto (const TString&,   const TString&,   const Int_t&, const Int_t&, const Int_t&, [const TString&]);
///                      User_X_Quantity, User_Y_Quantity, n1StexStin,   i0StinEcha,   n1Sample,     [PlotOption]
///
///
///    void Plot1DHisto(const TVectorD&, const TString&,   const TString&,   const Int_t&, [const TString&]);
///                     InputHisto,      User_X_Quantity, User_Y_Quantity, n1StexStin,   [PlotOption]
///
///  //==================================================================================================
///  //                     method for displaying 1D history plots
///  //==================================================================================================
/// 
///  void PlotHistory
///       (const TString&,   const TString&,   const TString&,         const Int_t&, const Int_t&, [const TString&]);
///        User_X_Quantity, User_Y_Quantity, list_of_run_file_name, StexStin_A,   i0StinEcha,   [PlotOption]
///
///  //==================================================================================================
///  //             methods for displaying Tower, SC, crystal numbering
///  //==================================================================================================
///
///  void SMTowerNumbering(const Int_t& SM#);
///  void DeeSCNumbering  (const Int_t& Dee#);
///
///  void TowerCrystalNumbering(const Int_t& SM#,  const Int_t& Tow#);
///  void SCCrystalNumbering   (const Int_t& Dee#, const Int_t& SC#);   // (SC# for construction)
///
///  //==================================================================================================
///  //                General title
///  //==================================================================================================
///  void GeneralTitle(const TString& Title);
///
///  //==================================================================================================
///  //                        Lin:Log scale (SCALE = "LIN" or "LOG") 
///  //==================================================================================================
///  void SetHistoScaleX(const TString& SCALE);
///  void SetHistoScaleY(const TString& SCALE);
///
///  //==================================================================================================
///  //                   ColorPalette (OPTION = "ECNAColor" or "Rainbow")
///  //==================================================================================================
///  void SetHistoColorPalette(const TString& OPTION);
///
///  //==================================================================================================
///  //                            histo ymin, ymax management
///  //==================================================================================================
///
///  //  These methods must be called before calls to the display methods
///
///  //...................... 1D histo (ymin,ymax) forced to (YminValue,YmaxValue) values
///  void SetHistoMin(const Double_t& YminValue);
///  void SetHistoMax(const Double_t& YmaxValue);
///
///  //...................... 1D histo (ymin,ymax) calculated from histo values
///  void SetHistoMin();
///  void SetHistoMax();
///
///  if SetHistoMin and SetHistoMax are not called, default values are applied. These default values
///  are in methods GetYminDefaultValue(...) and GetYmaxDefaultValue(...) of class TEcnaParHistos
///
///------------------------------------------------------------------------------------------------------
///
///   ECNA web page:
///
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_main_page.htm
///
///   For questions or comments, please send e-mail to: bernard.fabbro@cea.fr 
///

// ------- methods called by ReadAnd[Plot1DHisto]
//       (const TString&, const TString&, [const TVectorD&], const Int_t&, const Int_t&, const TString&)
//
//  void XtalSamplesEv(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, [const TString&]); //  EB or EE
//                     n1StexStin,   i0StinEcha,   [PlotOption]
//
//  void EvSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, [const TString&]);
//                      n1StexStin,   i0StinEcha,   [PlotOption]
//
//
//  void XtalSamplesSigma(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, [const TString&]); //  EB or EE
//                        n1StexStin,   i0StinEcha,   [PlotOption]
//
//  void SigmaSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, [const TString&]) //  EB or EE
//                         n1StexStin,   i0StinEcha,   [PlotOption]
//
//  void XtalSampleValues(const Int_t&, const Int_t&, const Int_t&, [const TString&]); // EB or EE
//                        n1StexStin,   i0StinEcha,   iSample,      [PlotOption]
//
//  void SampleADCEvents(const Int_t&, const Int_t&, const Int_t&, [const TString&]);  // EB or EE
//                       n1StexStin,   i0StinEcha,   iSample,      [PlotOption]
//

class TEcnaHistos : public TObject {

  RQ_OBJECT("TEcnaHistos")

 private:

  //..... Attributes

  constexpr static int charArrLen = 512;  // Max nb of caracters for char*
  Int_t fZerv;                            // = 0 , for ViewHisto non used arguments
  Int_t fUnev;                            // = 1 , for ViewHisto non used arguments


  Int_t fCnaCommand, fCnaError;
  Int_t fCnew,       fCdelete;
  Int_t fCnewRoot,   fCdeleteRoot;

  TString fTTBELL;

  //....................... Current subdetector flag and codes
  TString fFlagSubDet;
  TString fCodeEB;
  TString fCodeEE;

  //...........................................
  TEcnaParHistos* fCnaParHistos;
  TEcnaParPaths*  fCnaParPaths;
  TEcnaParCout*   fCnaParCout;
  TEcnaWrite*     fCnaWrite;
  TEcnaParEcal*   fEcal;
  TEcnaNumbering* fEcalNumbering;
  TEcnaHeader*    fFileHeader;

  TEcnaRead*      fMyRootFile;
  Int_t           fAlreadyRead;
  Int_t           fMemoAlreadyRead;
  Int_t           fTobeRead;
  TVectorD        fReadHistoDummy;
  TMatrixD        fReadMatrixDummy;

  std::ifstream fFcin_f;

  TString fFapAnaType;             // Type of analysis
  Int_t   fFapNbOfSamples;         // Nb of required samples
  Int_t   fFapRunNumber;           // Run number
  Int_t   fFapFirstReqEvtNumber;   // First requested event number
  Int_t   fFapLastReqEvtNumber;    // Last requested event number
  Int_t   fFapReqNbOfEvts;         // Requested number of events
  Int_t   fFapStexNumber;          // Stex number

  Int_t   fFapNbOfEvts;            // Number of found events

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
 
  TString fFapStexName;            // Stex name:               "SM"      (for EB) or "Dee"     (for EE)
  TString fFapStinName;            // Stin name:               "tower"   (for EB) or "SC"      (for EE)
  TString fFapXtalName;            // Xtal name:               "xtal"    (for EB) or "Xtal"    (for EE)
  TString fFapEchaName;            // Electronic channel name: "Chan"    (for EB) or "Chan"    (for EE)

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

  Axis_t fD_MSp_SpNbXinf;
  Axis_t fD_MSp_SpNbXsup;
  Axis_t fD_MSp_SpDsXinf;
  Axis_t fD_MSp_SpDsXsup;
  Axis_t fD_SSp_SpNbXinf;
  Axis_t fD_SSp_SpNbXsup;
  Axis_t fD_SSp_SpDsXinf;
  Axis_t fD_SSp_SpDsXsup;
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

  Double_t fD_MSp_SpNbYmin;
  Double_t fD_MSp_SpNbYmax;
  Double_t fD_MSp_SpDsYmin;
  Double_t fD_MSp_SpDsYmax;
  Double_t fD_SSp_SpNbYmin;
  Double_t fD_SSp_SpNbYmax;
  Double_t fD_SSp_SpDsYmin;
  Double_t fD_SSp_SpDsYmax;
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
  TString fAllXtalsInStinPlot;
  Int_t   fPlotAllXtalsInStin;

  Int_t  fMemoPlotH1SamePlus;
  Int_t  fMemoPlotD_NOE_ChNb, fMemoPlotD_NOE_ChDs;
  Int_t  fMemoPlotD_Ped_ChNb, fMemoPlotD_Ped_ChDs;
  Int_t  fMemoPlotD_TNo_ChNb, fMemoPlotD_TNo_ChDs; 
  Int_t  fMemoPlotD_MCs_ChNb, fMemoPlotD_MCs_ChDs;
  Int_t  fMemoPlotD_LFN_ChNb, fMemoPlotD_LFN_ChDs; 
  Int_t  fMemoPlotD_HFN_ChNb, fMemoPlotD_HFN_ChDs; 
  Int_t  fMemoPlotD_SCs_ChNb, fMemoPlotD_SCs_ChDs; 
  Int_t  fMemoPlotD_MSp_SpNb, fMemoPlotD_SSp_SpNb; 
  Int_t  fMemoPlotD_MSp_SpDs, fMemoPlotD_SSp_SpDs;
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
  Int_t  fMemoColorD_MSp_SpNb, fMemoColorD_SSp_SpNb;
  Int_t  fMemoColorD_MSp_SpDs, fMemoColorD_SSp_SpDs;
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
  TString  fXMemoD_MSp_SpNb; 
  TString  fXMemoD_MSp_SpDs;
  TString  fXMemoD_SSp_SpNb; 
  TString  fXMemoD_SSp_SpDs;
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
  TString  fYMemoD_MSp_SpNb; 
  TString  fYMemoD_MSp_SpDs;
  TString  fYMemoD_SSp_SpNb;
  TString  fYMemoD_SSp_SpDs;  
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
  Int_t  fNbBinsMemoD_MSp_SpNb;
  Int_t  fNbBinsMemoD_MSp_SpDs;
  Int_t  fNbBinsMemoD_SSp_SpNb;
  Int_t  fNbBinsMemoD_SSp_SpDs;
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
  TCanvas*  fCanvD_MSp_SpNb;
  TCanvas*  fCanvD_MSp_SpDs;
  TCanvas*  fCanvD_SSp_SpNb;
  TCanvas*  fCanvD_SSp_SpDs;  
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

  Bool_t  fClosedH1SamePlus;
  Bool_t  fClosedD_NOE_ChNb;
  Bool_t  fClosedD_NOE_ChDs;
  Bool_t  fClosedD_Ped_ChNb;
  Bool_t  fClosedD_Ped_ChDs;
  Bool_t  fClosedD_TNo_ChNb;
  Bool_t  fClosedD_TNo_ChDs;
  Bool_t  fClosedD_MCs_ChNb;
  Bool_t  fClosedD_MCs_ChDs;
  Bool_t  fClosedD_LFN_ChNb;
  Bool_t  fClosedD_LFN_ChDs;
  Bool_t  fClosedD_HFN_ChNb; 
  Bool_t  fClosedD_HFN_ChDs;
  Bool_t  fClosedD_SCs_ChNb;
  Bool_t  fClosedD_SCs_ChDs;
  Bool_t  fClosedD_MSp_SpNb;
  Bool_t  fClosedD_MSp_SpDs;
  Bool_t  fClosedD_SSp_SpNb;
  Bool_t  fClosedD_SSp_SpDs;    
  Bool_t  fClosedD_Adc_EvNb; 
  Bool_t  fClosedD_Adc_EvDs;  
  Bool_t  fClosedH_Ped_Date;
  Bool_t  fClosedH_TNo_Date;
  Bool_t  fClosedH_MCs_Date;
  Bool_t  fClosedH_LFN_Date;
  Bool_t  fClosedH_HFN_Date;
  Bool_t  fClosedH_SCs_Date;
  Bool_t  fClosedH_Ped_RuDs;
  Bool_t  fClosedH_TNo_RuDs;
  Bool_t  fClosedH_MCs_RuDs;
  Bool_t  fClosedH_LFN_RuDs;
  Bool_t  fClosedH_HFN_RuDs;
  Bool_t  fClosedH_SCs_RuDs;

  TString fCurrentHistoCode;
  TString fCurrentOptPlot;

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
  TVirtualPad*  fPadD_MSp_SpNb; 
  TVirtualPad*  fPadD_MSp_SpDs;
  TVirtualPad*  fPadD_SSp_SpNb;
  TVirtualPad*  fPadD_SSp_SpDs;
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
  TPaveText*  fPavTxtD_MSp_SpNb; 
  TPaveText*  fPavTxtD_MSp_SpDs;
  TPaveText*  fPavTxtD_SSp_SpNb; 
  TPaveText*  fPavTxtD_SSp_SpDs;  
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
  TCanvasImp*  fImpD_MSp_SpNb;
  TCanvasImp*  fImpD_MSp_SpDs;
  TCanvasImp*  fImpD_SSp_SpNb; 
  TCanvasImp*  fImpD_SSp_SpDs;  
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
  Int_t  fCanvSameD_MSp_SpNb, fCanvSameD_SSp_SpNb; 
  Int_t  fCanvSameD_MSp_SpDs, fCanvSameD_SSp_SpDs;
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
  TEcnaHistos(TEcnaObject*, const TString&);

  //TEcnaHistos(const TString&);
  //TEcnaHistos(const TString&, const TEcnaParPaths*);
  //TEcnaHistos(const TString&,
//	      const TEcnaParPaths*,
//	      const TEcnaParCout*,
//	      const TEcnaParEcal*, 
//	      const TEcnaParHistos*,
//	      const TEcnaNumbering*,
//	      const TEcnaWrite*);
  
   ~TEcnaHistos() override;
  
  void Init();
  void SetEcalSubDetector(const TString&);
//  void SetEcalSubDetector(const TString&,
//  			  const TEcnaParEcal*, 
//  			  const TEcnaParHistos*,
//  			  const TEcnaNumbering*,
//  			  const TEcnaWrite*);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                     METHODS FOR THE USER
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  //............ method to set the result file name parameters (from values in argument)
  //............ FileParameters(AnaType, [RunNumber], FirstEvent, NbOfEvts, [SM or Dee number])
  //             RunNumber = 0 => history plots , SM or Dee number = 0 => EB or EE Plots

  void FileParameters(const TString&, const Int_t&, const Int_t&,
		      const Int_t&,  const Int_t&, const Int_t&, const Int_t&);

  void FileParameters(TEcnaRead*);

  //================================================================================================
  //                    methods for displaying matrices (correlations, covariances)
  //  The last argument is optional: option plot ("COLZ", "LEGO" etc..., or "ASCII") default = COLZ
  //================================================================================================ 
  //..................... Corcc[for 1 Stex] (big matrix)
  void PlotMatrix(const TMatrixD&,
		  const TString&, const TString&);
  void PlotMatrix(const TMatrixD&,
		  const TString&, const TString&, const TString&);

  void PlotMatrix(const TString&, const TString&);
  void PlotMatrix(const TString&, const TString&, const TString&);

  //..................... Corcc[for 1 Stin], Corss[for 1 Echa], Covss[for 1 Echa]
  void PlotMatrix(const TMatrixD&,
		  const TString&, const TString&, const Int_t&, const Int_t&);
  void PlotMatrix(const TMatrixD&,
		  const TString&, const TString&, const Int_t&, const Int_t&, const TString&);

  void PlotMatrix(const TString&, const TString&, const Int_t&, const Int_t&);
  void PlotMatrix(const TString&, const TString&, const Int_t&, const Int_t&, const TString&);

  //================================================================================================
  //                    methods for displaying 2D views of the detector
  //                    detector = SM, Dee, EB, EE
  //================================================================================================ 
  void PlotDetector(const TVectorD&,
		    const TString&, const TString&);
  void PlotDetector(const TString&, const TString&);

  //================================================================================================
  //                    methods for displaying 1D histos OR history histos
  //  The last argument is optional: option plot ("COLZ", "LEGO" etc..., or "ASCII") default = COLZ
  //================================================================================================
  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const TString&);
  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const TString&, const TString&);

  void Plot1DHisto(const TString&, const TString&, const TString&);
  void Plot1DHisto(const TString&, const TString&, const TString&, const TString&);

  //.......................................................
  //=> BUG SCRAM? (voir test/TEcnaHistosExample2.cc et src/TEcnaHistos.cc)
  void Plot1DHisto(const TVectorD&, const TString&, const TString&, const Int_t&);
  void Plot1DHisto(const TVectorD&, const TString&, const TString&, const Int_t&, const TString&);

  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const Int_t&, const Int_t&);
  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const Int_t&, const Int_t&, const TString&);

  void Plot1DHisto(const TString&, const TString&, const Int_t&, const Int_t&);
  void Plot1DHisto(const TString&, const TString&, const Int_t&, const Int_t&, const TString&);

 //.......................................................
  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const Int_t&, const Int_t&, const Int_t&);
  void Plot1DHisto(const TVectorD&,
		   const TString&, const TString&, const Int_t&, const Int_t&, const Int_t&, const TString&);

  void Plot1DHisto(const TString&, const TString&, const Int_t&, const Int_t&, const Int_t&);
  void Plot1DHisto(const TString&, const TString&, const Int_t&, const Int_t&, const Int_t&, const TString&);

 //.......................................................
  void PlotHistory(const TString&, const TString&, const TString&, const Int_t&, const Int_t&);
  void PlotHistory(const TString&, const TString&, const TString&, const Int_t&, const Int_t&, const TString&);

  //====================================================================================
  //
  //                   methods for displaying Tower, SC, crystal numbering
  // 
  //====================================================================================
  void SMTowerNumbering(const Int_t&);  // USER: specific EB
  void DeeSCNumbering(const Int_t&);    // USER: specific EE

  void TowerCrystalNumbering(const Int_t&, const Int_t&);  // USER: specific EB
  void SCCrystalNumbering(const Int_t&, const Int_t&);     // USER: specific EE

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //             "TECHNICAL" METHODS
  //
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  void XtalSamplesEv(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, const TString&);
  void EvSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, const TString&);
  void XtalSamplesSigma(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, const TString&);
  void SigmaSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&, const TString&);

  void XtalSamplesEv(const TVectorD&, const Int_t&, const Int_t&, const Int_t&);
  void EvSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&);
  void XtalSamplesSigma(const TVectorD&, const Int_t&, const Int_t&, const Int_t&);
  void SigmaSamplesXtals(const TVectorD&, const Int_t&, const Int_t&, const Int_t&);

  //======================================= Miscellaneous methods
  //.................... General title
  void GeneralTitle(const TString&);

  //.................................. Lin:Log scale, ColorPalette, General Title
  void SetHistoScaleX(const TString&);
  void SetHistoScaleY(const TString&);
  void SetHistoColorPalette(const TString&);

  //.................................. 1D and 2D histo min,max user's values
  void SetHistoMin(const Double_t&);
  void SetHistoMax(const Double_t&);
  //.................................. 1D and 2D histo min,max from histo values
  void SetHistoMin();
  void SetHistoMax();

  //.................................. set fStartDate and fStopDate attributes from external values
  void StartStopDate(const TString&, const TString&);

  //.................................. set fRunType attribute from external value
  void RunType(const TString&);

  //.................................. set fFapNbOfEvts attribute from external value
  void NumberOfEvents(const Int_t&);

  //======================= TECHNICAL METHODS (in principle not for the user) ========================

  void SetRunNumberFromList(const Int_t&, const Int_t&);  // called by Histime
  void InitSpecParBeforeFileReading();                    // set parameters from the file reading

  //................. methods for displaying the cor(s,s) and cov(s,s) corresponding to a Stin
  void CorrelationsBetweenSamples(const Int_t&);
  void CovariancesBetweenSamples(const Int_t&);

  //.................        
  void StexHocoVecoLHFCorcc(const TString&);

  void StexStinNumbering(const Int_t&);                    
  void StinCrystalNumbering(const Int_t&, const Int_t&);   

  void ViewStas(const TVectorD&, const Int_t&, const TString&);
  void ViewStex(const TVectorD&, const Int_t&, const TString&);
  void ViewStin(const Int_t&, const TString&);
  void ViewMatrix(const TMatrixD&, const Int_t&, const Int_t&,  const Int_t&,  const Int_t&,
		  const TString&, const TString&, const TString&);
  void ViewHisto(const TVectorD&, const Int_t&, const Int_t&,  const Int_t&, const Int_t&,
		 const TString&,   const TString&);

  Int_t GetDSOffset(const Int_t&, const Int_t&);
  Int_t GetSCOffset(const Int_t&, const Int_t&, const Int_t&);

  void ViewHistime(const TString&, const Int_t&, const Int_t&,
		   const TString&, const TString&);

  Int_t GetHistoryRunListParameters(const TString&, const TString&);

  void TopAxisForHistos(TH1D*,
			const TString&, const Int_t&, const Int_t&, const Int_t&,
			const Int_t&,  const Int_t& );

  //--------------------------------------------------------------- xinf, xsup management
  void     SetXinfMemoFromValue(const TString&, const Double_t&);
  void     SetXsupMemoFromValue(const TString&, const Double_t&);
  void     SetXinfMemoFromValue(const Double_t&);
  void     SetXsupMemoFromValue(const Double_t&);

  Double_t GetXinfValueFromMemo(const TString&);
  Double_t GetXsupValueFromMemo(const TString&);
  Double_t GetXinfValueFromMemo();
  Double_t GetXsupValueFromMemo();

  Axis_t   GetHistoXinf(const TString&, const Int_t&, const TString&);
  Axis_t   GetHistoXsup(const TString&, const Int_t&, const TString&);

  Int_t    GetHistoNumberOfBins(const TString&,  const Int_t&); 

  //--------------------------------------------------------------- ymin, ymax management
  void     SetYminMemoFromValue(const TString&, const Double_t&);
  void     SetYmaxMemoFromValue(const TString&, const Double_t&);

  Double_t GetYminValueFromMemo(const TString&);
  Double_t GetYmaxValueFromMemo(const TString&);

  void     SetYminMemoFromPreviousMemo(const TString&);
  void     SetYmaxMemoFromPreviousMemo(const TString&);

  Int_t    SetHistoFrameYminYmaxFromMemo(TH1D*,   const TString&);
  Int_t    SetGraphFrameYminYmaxFromMemo(TGraph*, const TString&);

  Double_t GetYminFromHistoFrameAndMarginValue(TH1D*, const Double_t);
  Double_t GetYmaxFromHistoFrameAndMarginValue(TH1D*, const Double_t);

  Double_t GetYminFromGraphFrameAndMarginValue(TGraph*, const Double_t);
  Double_t GetYmaxFromGraphFrameAndMarginValue(TGraph*, const Double_t);

  //.................................. 1D and 2D histo min,max default values
  void SetAllYminYmaxMemoFromDefaultValues();

  //------------------------------------------------- Memo Same, Same n management
  void    SetXVarMemo(const TString&, const TString&, const TString&);
  TString GetXVarFromMemo(const TString&, const TString&);

  void    SetYVarMemo(const TString&, const TString&, const TString&);
  TString GetYVarFromMemo(const TString&, const TString&);

  void    SetNbBinsMemo(const TString&, const TString&, const Int_t&);
  Int_t   GetNbBinsFromMemo(const TString&, const TString&);

  //--------------------------------------------------------------------------------
  void ViewStexStinNumberingPad(const Int_t&);
  void ViewSMTowerNumberingPad(const Int_t&);   // specific EB
  void ViewDeeSCNumberingPad(const Int_t&);     // specific EE

  void ViewStinGrid(const Int_t&, const Int_t&, const Int_t&,
		    const Int_t&, const Int_t&, const TString&);
  void ViewTowerGrid(const Int_t&, const Int_t&, const Int_t&,
		     const Int_t&, const Int_t&, const TString&);  // specific EB
  void ViewSCGrid(const Int_t&, const Int_t&, const Int_t&,
		  const Int_t&, const Int_t&, const TString&);     // specific EE

  void ViewStexGrid(const Int_t&, const TString&);
  void ViewSMGrid(const Int_t&, const TString&);    // specific EB
  void ViewDeeGrid(const Int_t&, const TString&);   // specific EE

  void ViewStasGrid(const Int_t&);
  void ViewEBGrid();               // specific EB
  void ViewEEGrid(const Int_t&);   // specific EE

  void EEDataSectors(const Float_t&,  const Float_t&, const Int_t&, const TString&);                // specific EE
  void EEGridAxis(const Int_t&, const TString&, const TString&);   // specific EE

  void SqrtContourLevels(const Int_t&, Double_t*);

  TString StexNumberToString(const Int_t&);

  void HistoPlot(TH1D*,
		 const Int_t&,  const Axis_t&,  const Axis_t&,  const TString&, const TString&,
		 const Int_t&,  const Int_t&,   const Int_t&,   const Int_t&,
		 const Int_t&,  const TString&,  const Int_t&,   const Int_t&);

  Double_t NotConnectedSCH1DBin(const Int_t&);
  Int_t    GetNotConnectedDSSCFromIndex(const Int_t&);
  Int_t    GetNotConnectedSCForConsFromIndex(const Int_t&);
  Int_t    ModifiedSCEchaForNotConnectedSCs(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  Double_t NotCompleteSCH1DBin(const Int_t&);
  Int_t    GetNotCompleteDSSCFromIndex(const Int_t&);
  Int_t    GetNotCompleteSCForConsFromIndex(const Int_t&);

  void HistimePlot(TGraph*,       Axis_t,        Axis_t,
		   const TString&, const TString&, const Int_t&, const Int_t&,
		   const Int_t&,  const Int_t&,  const Int_t&, const TString&, const Int_t&);

  void SetAllPavesViewMatrix(const TString&, const Int_t&, const Int_t&, const Int_t&);
  void SetAllPavesViewStin(const Int_t&);
  void SetAllPavesViewStex(const TString&, const Int_t&);
  void SetAllPavesViewStex(const Int_t&);
  void SetAllPavesViewStas();
  void SetAllPavesViewStinCrysNb(const Int_t&, const Int_t&);
  void SetAllPavesViewHisto(const TString&, const Int_t&, const Int_t&, const Int_t&, const TString&);
  void SetAllPavesViewHisto(const TString&, const Int_t&, const Int_t&, const Int_t&, const TString&, const Int_t&);

  Int_t GetXSampInStin(const Int_t&, const Int_t&,
		       const Int_t&,  const Int_t&);
  Int_t GetYSampInStin(const Int_t&, const Int_t&,
		       const Int_t&,  const Int_t&);

  Int_t GetXCrysInStex(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetYCrysInStex(const Int_t&, const Int_t&, const Int_t&);

  Int_t GetXStinInStas(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetYStinInStas(const Int_t&, const Int_t&);


  TString GetHocoVecoAxisTitle(const TString&);
  TString GetEtaPhiAxisTitle(const TString&);        // specific EB
  TString GetIXIYAxisTitle(const TString&);          // specific EE

  Bool_t   GetOkViewHisto(TEcnaRead*, const Int_t&, const Int_t&, const Int_t&, const TString&);
  Int_t    GetHistoSize(const TString&, const TString&);
  TVectorD GetHistoValues(const TVectorD&, const Int_t&, TEcnaRead*,  const TString&,
			  const Int_t&,    const Int_t&,
			  const Int_t&,    const Int_t&,  const Int_t&, Int_t&);

  TString SetHistoXAxisTitle(const TString&);
  TString SetHistoYAxisTitle(const TString&);

  void FillHisto(TH1D*, const TVectorD&, const TString&, const Int_t&);

  TString GetMemoFlag(const TString&);
  TString GetMemoFlag(const TString&, const TString&);

  TCanvas* CreateCanvas(const TString&, const TString&, const TString&, UInt_t,  UInt_t);
  TCanvas* GetCurrentCanvas(const TString&, const TString&);
  TCanvas* GetCurrentCanvas();
  TString  GetCurrentCanvasName();
  void     PlotCloneOfCurrentCanvas();

  void SetParametersCanvas(const TString&, const TString&);
  void SetParametersPavTxt(const TString&, const TString&);

  TVirtualPad* ActivePad(const TString&, const TString&);
  TPaveText*   ActivePavTxt(const TString&, const TString&);
  void         DoCanvasClosed();

  void SetHistoPresentation(TH1D*,   const TString&);
  void SetHistoPresentation(TH1D*,   const TString&, const TString&);
  void SetGraphPresentation(TGraph*, const TString&, const TString&);

  void SetViewHistoColors(TH1D*,   const TString&, const TString&, const Int_t&);
  void SetViewGraphColors(TGraph*, const TString&, const TString&);

  Color_t GetViewHistoColor(const TString&, const TString&);

  Int_t GetListFileNumber(const TString&);
  void  ReInitCanvas(const TString&, const TString&);
  void  NewCanvas(const TString&);

  TString SetCanvasName(const TString&, const Int_t&, const Int_t&, 
			const TString&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  Color_t GetSCColor(const TString&, const TString&, const TString&);     // specific EE

  void WriteMatrixAscii(const TString&, const TString&, const Int_t&, const Int_t&, const Int_t&, const TMatrixD&);
  void WriteHistoAscii(const TString&, const Int_t&, const TVectorD&);

  TString  AsciiFileName();
  Bool_t StatusFileFound();
  Bool_t StatusDataExist();

ClassDefOverride(TEcnaHistos,1)// methods for plots from ECNA (Ecal Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaHistos
