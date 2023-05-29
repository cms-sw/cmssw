#ifndef CL_TEcnaRun_H
#define CL_TEcnaRun_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstdio>
#include "Riostream.h"

// ROOT include files
#include "TObject.h"
#include "TSystem.h"
#include "TString.h"
#include "TVectorD.h"

// user include files
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRootFile.h"

///-----------------------------------------------------------
///   TEcnaRun.h
///   Update: 05/10/2012
///   Authors:   B.Fabbro (bernard.fabbro@cea.fr), FX Gentit
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_main_page.htm
///-----------------------------------------------------------
///
/// TEcnaRun + ECNA (Ecal Correlated Noise Analysis) instructions for use
///            in the framework of CMSSW.
///
///==============> INTRODUCTION
///
///    The present documentation contains:
///
///    [1] a brief description of the ECNA package with instructions for use
///        in the framework of the CMS Software
///
///    [2] the documentation for the class TEcnaRun
///
///==[1]=====================================================================================
///
///         DOCUMENTATION FOR THE INTERFACE: ECNA package / CMSSW / SCRAM
///
///==========================================================================================
///
///  ECNA consists in 2 packages named: EcalCorrelatedNoiseAnalysisModules and
///  EcalCorrelatedNoiseAnalysisAlgos.
///
///  The directory tree is the following:
///
///      <local path>/CMSSW_a_b_c/src/----CalibCalorimetry/---EcalCorrelatedNoiseAnalysisModules/BuildFile
///                              |   |                    |                                     |---interface/
///                              |   |                    |                                     |---src/
///                                  |                    |                                     |---data/
///                                  |                    |
///                                  |                    |---EcalCorrelatedNoiseAnalysisAlgos/BuildFile
///                                  |                    |                                   |---interface/
///                                  |                    |                                   |---src/
///                                  |                    |                                   |---test/
///                                  |                    |
///                                  |                    |
///                                  |                    \--- <other packages of CalibCalorimetry>
///                                  |
///                                  \----<other subsystems...>
///
///
///    The package EcalCorrelatedNoiseAnalysisModules contains one standard analyzer
///    (EcnaAnalyzer). The user can edit its own analyzer.
///    A detailed description is given here after in the class TEcnaRun documentation.
///    The package EcalCorrelatedNoiseAnalysisAlgos contains the basic ECNA classes
///    (in src and interface) and standalone executables (in directory test).
///
///==[2]======================================================================================
///
///                         CLASS TEcnaRun DOCUMENTATION
///
///===========================================================================================
///TEcnaRun.
///
///
/// Brief and general description
/// -----------------------------
///
///   This class allows the user to calculate pedestals, noises,
///   correlations and other quantities of interest for correlated
///   noise studies on the CMS/ECAL (EB and EE).
///
///   Three main operations are performed by the class TEcnaRun. Each of them is
///   associated with a specific method of the analyzer EcnaAnalyzer:
///
///    (1) Initialization and calls to "preparation methods".
///        This task is done in the constructor of the analyzer:
///        EcnaAnalyzer::EcnaAnalyzer(const edm::ParameterSet& pSet)
///
///    (2) Building of the event distributions (distributions of the sample ADC
///        values for each sample, each channel, etc...)
///        This task is done in the method "analyze" of the analyzer:
///        EcnaAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
///
///    (3) Calculation of the different quantities (correlations, pedestals, noises, etc...)
///        from the distributions obtained in (2) and writing of these quantities
///        in results ROOT files and also in ASCII files.
///        This task is done in the destructor of the analyzer:
///        EcnaAnalyzer::~EcnaAnalyzer()
///
///
/// Use of the class TEcnaRun by the analyzer EcnaAnalyzer
/// ------------------------------------------------------
///
///           see files EcnaAnalyzer.h and EcnaAnalyzer.cc
///           in package EcalCorrelatedNoiseAnalysisModules
///
/// More detailled description of the class TEcnaRun
/// -----------------------------------------------
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///                     Declaration and Print Methods
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///     Just after the declaration with the constructor,
///     you can set a "Print Flag" by means of the following "Print Methods":
///
///     TEcnaRun* MyCnaRun = new TEcnaRun(...); // declaration of the object MyCnaRun
///
///   // Print Methods:
///
///    MyCnaRun->PrintNoComment();  // Set flag to forbid printing of all the comments
///                                 // except ERRORS.
///
///    MyCnaRun->PrintWarnings();   // (DEFAULT)
///                                 // Set flag to authorize printing of some warnings.
///                                 // WARNING/INFO: information on something unusual
///                                 // in the data.
///                                 // WARNING/CORRECTION: something wrong (but not too serious)
///                                 // in the value of some argument.
///                                 // Automatically modified to a correct value.
///
///   MyCnaRun->PrintComments();    // Set flag to authorize printing of infos
///                                 // and some comments concerning initialisations
///
///   MyCnaRun->PrintAllComments(); // Set flag to authorize printing of all the comments
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///           Method GetReadyToReadData(...) and associated methods
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///      MyCnaRun->GetReadyToReadData(AnalysisName,      NbOfSamples,       RunNumber,
///		                      FirstReqEvtNumber, LastReqEvtNumber,  ReqNbOfEvts,
///                                   StexNumber,        [RunType]);
///
///   Explanations for the arguments (all of them are input arguments):
///
///      TString  AnalysisName: code for the analysis. According to this code,
///                             the analyzer EcnaAnalyser selects the event type
///                             (PEDESTAL_STD, PEDESTAL_GAP, LASER_STD, etc...)
///                             and some other event characteristics
///                             (example: the gain in pedestal runs:
///                              AnalysisName = "Ped1" or "Ped6" or "Ped12")
///                             See EcnaAnalyser.h for a list of available codes.
///                             The string AnalysisName is automatically
///                             included in the name of the results files
///                             (see below: results files paragraph).
///
///      Int_t     NbOfSamples         number of samples (=10 maximum)
///      Int_t     RunNumber:          run number
///      Int_t     FirstReqEvtNumber:  first requested event number (numbering starting from 1)
///      Int_t     LastReqEvtNumber:   last  requested event number
///      Int_t     ReqNbOfEvts:        requested number of events
///      Int_t     StexNumber:         Stex number (Stex = SM if EB, Dee if EE)
///
///     The different quantities (correlations, etc...) will be calculated
///     for ReqNbOfEvts events between event# FirstReqEvtNumber and event# LastReqEvtNumber.
///     If LastReqEvtNumber = 0, the calculations will be performed from event# FirstReqEvtNumber
///     until EOF if necessary (i.e. if the number of treated events is < ReqNbOfEvts)
///
///      Int_t     RunType [optional]: run type
///
///                          PEDESTAL_STD =  9
///                          LASER_STD    =  4
///                          PEDESTAL_GAP = 18, etc...
///      (see CMSSSW/DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h)
///
///      if RunType is specified, the run type will be displayed on the plots
///
///==============> Method to set the start and stop times of the analysis (optional)
///
///  A method can be used to set the fStartDate and fStopDate attributes
///  of the class TEcnaHeader from start and stop time given by the user provided
///  these values have been recovered from the event reading:
///
///      void  MyCnaRun->StartStopDate(const TString& StartDate, const TString& StopDate);
///
///     // TString StartDate, StopDate:  start and stop time of the run
///     //                               in "date" format. Example:
///     //                               Wed Oct  8 04:14:23 2003
///
///     If the method is not called, the values of the attributes
///     fStartDate and fStopDate are set to: "!Start date> no info"
///     and "!Stop date> no info" at the level of Init() method of the class TEcnaHeader.
///     The values of StartDate and StopDate are written in the header of
///     the .root result file.
///
///  PS: another similar method exists, with time_t type arguments:
///
///     void  MyCnaRun->StartStopTime(time_t  StartTime, time_t  StopTime);
///
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///                       Calculation methods
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///    The "calculation methods" are methods which compute the different
///    quantities of interest. They use the ADC sample values which can be
///    recovered by the method void SampleValues():
///
///  void SampleValues(); // 3D histo of the sample ADC value
///                       // for each triple (channel,sample,event)
///
///    List of the calculation methods with associated formulae:
///
///.......... Calculation methods ( need previous call to GetReadyToReadData(...) )
///
///   //   A(t,c,s,e) : ADC value for Stin t channel c, sample s, event e
///                                  (Stin = tower if EB, SC if EE)
///
///   //   E_e , Cov_e : average, covariance over the events
///   //   E_s , Cov_s : average, covariance over the samples
///   //   E_s,s'      : average  over couples of samples (half correlation matrix)
///
///   //   e* : random variable associated to events
///   //   s* : random variable associated to samples
///
///  void SampleMeans();  // Expectation values for each couple (channel,sample)
///  //    SMean(t,c,s)  = E_e[A(t,c,s,e*)]
///
///  void SampleSigmas(); // Sigmas for each couple (channel,sample)
///  //    SSigma(t,c,s) = sqrt{ Cov_e[A(t,c,s,e*),A(t,c,s,e*)] }
///
///  //...........................................
///  void CovariancesBetweenSamples();  // (s,s') covariances  for each channel
///  //    Cov(t,c;s,s') = Cov_e[ A(t,c,s,e*) , A(t,c,s',e*) ]
///  //                = E_e[ ( A(t,c,s,e*) - E_e[A(t,c,s,e*)] )*( A(t,c,s',e*) - E_e[A(t,c,s',e*)] ) ]
///
///  void CorrelationsBetweenSamples(); // (s,s') correlations for each channel
///  //    Cor(t,c;s,s') = Cov(t,c;s,s')/sqrt{ Cov(t,c;s,s)*Cov(t,c;s',s') }
///
///  //.............................................. *==> Stin = tower if EB, SuperCrystal if EE
///  void LowFrequencyCovariancesBetweenChannels();   // LF (t;c,c') covariances  for each Stin
///  void HighFrequencyCovariancesBetweenChannels();  // HF (t;c,c') covariances  for each Stin
///  void LowFrequencyCorrelationsBetweenChannels();  // LF (t;c,c') correlations for each Stin
///  void HighFrequencyCorrelationsBetweenChannels(); // HF (t;c,c') correlations for each Stin
///
///   //    LFCov(t;c,c') = Cov_e[ E_s[A(t,c,s*,e*)] , E_s[A(t,c',s*,e*) ]
///   //
///   //                = E_e[ ( E_s[A(t,c ,s*,e*)] - E_e[ E_s[A(t,c ,s*,e*)] ] )*
///   //                        ( E_s[A(t,c',s*,e*)] - E_e[ E_s[A(t,c',s*,e*)] ] ) ]
///   //
///   //    HFCov(t;c,c') = E_e[ Cov_s[ A(t,c,s*,e*) , A(t,c',s*,e*) ] ]
///   //
///   //                = E_e[ E_s[ ( A(t,c ,s*,e*) - E_s[A(t,c ,s*,e*)] )*
///   //                            ( A(t,c',s*,e*) - E_s[A(t,c',s*,e*)] ) ] ]
///   //
///   //    LFCor(t;c,c') = LFCov(t;c,c')/sqrt{ LFCov(t;c,c)*LFCov(t;c',c') }
///   //
///   //    HFCor(t;c,c') = HFCov(t;c,c')/sqrt{ HFCov(t;c,c)*HFCov(t;c',c') }
///
///  //.............................................. . *==> Stex = SM if EB, Dee if EE
///  void LowFrequencyMeanCorrelationsBetweenTowers();  // LF (tow,tow') correlations for each SM
///  void HighFrequencyMeanCorrelationsBetweenTowers(); // HF (tow,tow') correlations for each SM
///
///  void LowFrequencyMeanCorrelationsBetweenSCs();     // LF (sc,sc') correlations for each Dee
///  void HighFrequencyMeanCorrelationsBetweenSCs();    // HF (sc,sc') correlations for each Dee
///
/// //.................................................... Quantities as a function of Xtal#
///  void Pedestals();
///  void TotalNoise();
///  void LowFrequencyNoise();
///  void HighFrequencyNoise();
///  void MeanCorrelationsBetweenSamples();
///  void SigmaOfCorrelationsBetweenSamples();
///
///  // Pedestal(t,c)    = E_e[ E_s[A(t,c,s*,e*)] ]
///  // TotalNoise(t,c)  = E_s[ sqrt{ E_e[ ( A(t,c,s*,e*) - E_e[A(t,c,s*,e*)] )^2 ] } ]
///  // LowFqNoise(t,c)  = sqrt{ E_e[ ( E_s[A(t,c,s*,e*)] - E_e[ E_s[A(t,c,s*,e*)] ] )^2 ] }
///  // HighFqNoise(t,c) = E_e[ sqrt{ E_s[ (A(t,c,s*,e*) - E_s[A(t,c,s*,e*)] )^2 ] } ]
///  // MeanCorss(t,c)   = E_s,s'[ Cor(t,c;s,s') ]
///  // SigmaCorss(t,c)  = E_s,s'[ Cor(t,c;s,s') - E_s,s'[ Cor(t,c;s,s') ] ]
///
///  //............ Quantities as a function of tower# (EB) or SC# (EE), average over the Xtals
///  void AveragePedestals();
///  void AverageTotalNoise();
///  void AverageLowFrequencyNoise();
///  void AverageHighFrequencyNoise();
///  void AverageMeanCorrelationsBetweenSamples();
///  void AverageSigmaOfCorrelationsBetweenSamples();
///
///  // t = tower if EB , SC if EE , c = channel (Xtal)
///  // AveragePedestal(t) = E_c[Pedestal(t,c*)]
///  // TotalNoise(t)      = E_c[TotalNoise(t,c*)]
///  // LowFqNoise(t)      = E_c[LowFqNoise(t,c*)]
///  // HighFqNoise(t)     = E_c[HighFqNoise(t,c*)]
///  // MeanCorss(t)       = E_c[MeanCorss(t,c*)]
///  // SigmaCorss(t)      = E_c[SigmaCorss(t,c*)]
///
///==============> RESULTS FILES
///
///  The calculation methods above provide results which can be used directly
///  in the user's code. However, these results can also be written in results
///  files by appropriate methods.
///  The names of the results files are automaticaly generated.
///
///  It is also possible to write results in ASCII files  => See TEcnaWrite and TEcnaGui
///  It is also possible to plot results in ROOT canvases => See TEcnaHistos and TEcnaGui
///
/// *-----------> Codification for the name of the ROOT file:
///
///  The name of the ROOT file is the following:
///
///       aaa_S1_sss_Rrrr_fff_lll_SMnnn.root     for EB
///       aaa_S1_sss_Rrrr_fff_lll_Deennn.root    for EE
///
///  with:
///       aaa = Analysis name
///       sss = number of samples
///       rrr = Run number
///       fff = First requested event number
///       lll = Last  requested events
///       mmm = Requested number of events
///       nnn = SM number or Dee number
///
///  This name is automatically generated from the values of the arguments
///  of the method "GetReadyToReadData".
///
/// *-----------> Method which writes the results in the ROOT file:
///
///       Bool_t  MyCnaRun->WriteRootFile();
///
///===================================================================================================
///

class TEcnaRun : public TObject {
private:
  //............ attributes

  Int_t fgMaxCar;  // Max nb of caracters for char*

  Int_t fCnaCommand, fCnaError;

  Int_t fCnew;  // flags for dynamical allocation
  Int_t fCdelete;

  TString fTTBELL;

  Int_t* fMiscDiag;  // Counters for miscellaneous diagnostics
  Int_t fNbOfMiscDiagCounters;
  Int_t fMaxMsgIndexForMiscDiag;

  TEcnaObject* fObjectManager;     // for ECNA object management
  TEcnaHeader* fFileHeader;        // header for result type file
  TEcnaParEcal* fEcal;             // for access to the Ecal current subdetector parameters
  TEcnaNumbering* fEcalNumbering;  // for access to the Ecal channel, Stin and Stex numbering
  TEcnaParCout* fCnaParCout;       // for comment/error messages
  TEcnaParPaths* fCnaParPaths;     // for file access
  TEcnaWrite* fCnaWrite;           // for access to the results files

  //  TEcnaRootFile *gCnaRootFile;

  TString fFlagSubDet;
  TString fStexName, fStinName;

  Bool_t fOpenRootFile;  // flag open ROOT file (open = kTRUE, close = kFALSE)
  Int_t fReadyToReadData;

  TString fRootFileName;
  TString fRootFileNameShort;
  TString fNewRootFileName;
  TString fNewRootFileNameShort;

  Int_t fSpecialStexStinNotIndexed;  // management of event distribution building
  Int_t fStinIndexBuilt;
  Int_t fBuildEvtNotSkipped;

  Int_t fNbSampForFic;
  Int_t fNbSampForCalc;

  Int_t fNumberOfEvents;

  Int_t fMemoReadNumberOfEventsforSamples;

  Double_t*** fT3d_AdcValues;  // 3D array[channel][sample][event] ADC values
  Double_t** fT3d2_AdcValues;
  Double_t* fT3d1_AdcValues;
  Int_t* fTagAdcEvt;

  Int_t** fT2d_NbOfEvts;  // 2D array[channel][sample] max nb of evts read for a given (channel,sample)
  Int_t* fT1d_NbOfEvts;
  Int_t* fTagNbOfEvts;

  Int_t* fT1d_StexStinFromIndex;  // 1D array[Stin] Stin Number as a function of the index Stin
  Int_t* fTagStinNumbers;

  Double_t** fT2d_ev;  // 2D array[channel][sample] for expectation values
  Double_t* fT1d_ev;
  Int_t* fTagMSp;

  Double_t** fT2d_sig;  // 2D array[channel][sample] for sigmass
  Double_t* fT1d_sig;
  Int_t* fTagSSp;

  Double_t*** fT3d_cov_ss;  // 3D array[channel][sample][sample] for (sample,sample) covariances
  Double_t** fT3d2_cov_ss;
  Double_t* fT3d1_cov_ss;
  Int_t* fTagCovCss;

  Double_t*** fT3d_cor_ss;  // 3D array[channel][sample][sample] for (sample,sample) correlations
  Double_t** fT3d2_cor_ss;
  Double_t* fT3d1_cor_ss;
  Int_t* fTagCorCss;

  //...........................................................................
  Double_t* fT1d_ev_ev;    // 1D array[channel] for expectation values of the expectation values of the samples
  Int_t* fTagPed;          // (PEDESTAL)
  Double_t* fT1d_av_mped;  // 1D array[Stin] for expectation values of the Pesdestals of the Stins
  Int_t* fTagAvPed;        // (AVERAGED PEDESTAL)

  Double_t* fT1d_evsamp_of_sigevt;  // 1D array[channel] for expectation values of the sigmas of the samples
  Int_t* fTagTno;                   // (TOTAL NOISE)
  Double_t* fT1d_av_totn;           // 1D array[Stin] for expectation values of the total noise
  Int_t* fTagAvTno;                 //(AVERAGED TOTAL NOISE)

  Double_t* fT1d_ev_cor_ss;    // 1D array[channel] for expectation values of the cor(s,s)
  Int_t* fTagMeanCorss;        // (MEAN COR(S,S))
  Double_t* fT1d_av_ev_corss;  // 1D array[Stin] for expectation values of the mean cor(s,s)
  Int_t* fTagAvMeanCorss;      // (AVERAGED MEAN COR(S,S))

  Double_t* fT1d_sigevt_of_evsamp;  // 1D array[channel] for sigmas of the expectation values of the samples
  Int_t* fTagLfn;                   // (LOW FREQUENCY NOISE)
  Double_t* fT1d_av_lofn;           // 1D array[Stin]  the expectation values of the low frequency noise
  Int_t* fTagAvLfn;                 // (AVERAGED LOW FREQUENCY NOISE)

  Double_t* fT1d_evevt_of_sigsamp;  // 1D array[channel] for sigmas of the sigmas of the samples
  Int_t* fTagHfn;                   // (HIGH FREQUENCY NOISE)
  Double_t* fT1d_av_hifn;           // 1D array[channel] for expectation values of the high frequency noise
  Int_t* fTagAvHfn;                 // (AVERAGED HIGH FREQUENCY NOISE)

  Double_t* fT1d_sig_cor_ss;    // 1D array[channel] for sigmas of the cor(s,s)
  Int_t* fTagSigCorss;          // (SIGMA OF COR(S,S))
  Double_t* fT1d_av_sig_corss;  // 1D array[channel] for expectation values of sigmas  the  of the cor(s,s)
  Int_t* fTagAvSigCorss;        // (AVERAGED SIGMA OF COR(S,S))

  //...........................................................................
  Double_t** fT2d_lf_cov;  // 2D array[channel][channel] for (channel,channel) low frequency covariances
  Double_t* fT2d1_lf_cov;
  Int_t* fTagLfCov;

  Double_t** fT2d_lf_cor;  // 2D array[channel][channel] for (channel,channel) low frequency correlations
  Double_t* fT2d1_lf_cor;
  Int_t* fTagLfCor;

  //...........................................................................
  Double_t** fT2d_hf_cov;  // 2D array[channel][channel] for (channel,channel) low frequency covariances
  Double_t* fT2d1_hf_cov;
  Int_t* fTagHfCov;

  Double_t** fT2d_hf_cor;  // 2D array[channel][channel] for (channel,channel) low frequency correlations
  Double_t* fT2d1_hf_cor;
  Int_t* fTagHfCor;

  //------------------------- 2 tableaux (ci,cj)
  Double_t** fT2d_lfcc_mostins;  // 2D array[Stin][Stin] for (Stin,Stin) mean cov(c,c)
  Double_t* fT2d1_lfcc_mostins;  // (relevant ones) averaged over samples
  Int_t* fTagLFccMoStins;

  Double_t** fT2d_hfcc_mostins;  // 2D array[Stin][Stin] for (Stin,Stin) mean cor(c,c)
  Double_t* fT2d1_hfcc_mostins;  // (relevant ones) averaged over samples
  Int_t* fTagHFccMoStins;

  //------------------------------------------------------------------------------------

  Int_t** fT2dCrysNumbersTable;
  Int_t* fT1dCrysNumbersTable;

  std::ofstream fFcout_f;

  Int_t fFlagPrint;
  Int_t fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

public:
  //................. constructors

  TEcnaRun();                              //  constructor without argument
  TEcnaRun(TEcnaObject*, const TString&);  //  constructors with argument (FOR USER'S DECLARATION)
  TEcnaRun(TEcnaObject*, const TString&, const Int_t&);

  //TEcnaRun(const TString&, const Int_t&, const TEcnaParPaths*, const TEcnaParCout*);
  //TEcnaRun(const TString&);               //  constructors with argument (FOR USER'S DECLARATION)
  //TEcnaRun(const TString&, const Int_t&);

  TEcnaRun(const TEcnaRun&);  //  copy constructor

  //.................... C++ methods

  //TEcnaRun&  operator=(const TEcnaRun&);  //  overloading of the operator=

  //................. destructor

  ~TEcnaRun() override;

  //...................................................... methods that will (should) be private

  void Init(TEcnaObject*);

  void SetEcalSubDetector(const TString&);

  Bool_t GetPathForResults();

  Bool_t OpenRootFile(const Text_t*, const TString&);
  Bool_t CloseRootFile(const Text_t*);

  //======================================= methods for the user =========================================
  void GetReadyToReadData(const TString&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  void GetReadyToReadData(
      const TString&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);

  Bool_t GetSampleAdcValues(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Double_t&);
  Bool_t ReadSampleAdcValues();
  Bool_t ReadSampleAdcValues(const Int_t&);

  void StartStopDate(const TString&, const TString&);
  void StartStopTime(time_t, time_t);

  //................... Calculation methods ( associated to GetReadyToReadData(...) )
  //-------------------- Standard Calculations
  void StandardCalculations();  // see list in the method itself (.cc file)

  void SampleMeans();   // Calculation of the expectation values over the events
                        // for each sample and for each channel
  void SampleSigmas();  // Calculation of the variances over the events
                        // for each sample and for each channel
  //...........................................
  void CovariancesBetweenSamples();   // Calculation of the (s,s) covariances over the events
                                      // for each channel
  void CorrelationsBetweenSamples();  // Calculation of the (s,s) correlations over the events
                                      // for each channel
  //..........................................................
  void Pedestals();
  void TotalNoise();
  void LowFrequencyNoise();
  void HighFrequencyNoise();
  void MeanCorrelationsBetweenSamples();
  void SigmaOfCorrelationsBetweenSamples();

  //..........................................................
  void AveragePedestals();
  void AverageTotalNoise();
  void AverageLowFrequencyNoise();
  void AverageHighFrequencyNoise();
  void AverageMeanCorrelationsBetweenSamples();
  void AverageSigmaOfCorrelationsBetweenSamples();

  //---------- Calculations involving cov and cor between channels
  //
  //   Recommended calling sequences: expert1, expert1 + expert2,  expert2
  //
  //   NOT recommended: expert2 + expert1  (lost of time and place)
  //
  //-------------------- Expert 1 Calculations
  void Expert1Calculations();  // see list in the method itself (.cc file)

  void LowFrequencyCovariancesBetweenChannels();
  void HighFrequencyCovariancesBetweenChannels();

  void LowFrequencyCorrelationsBetweenChannels();
  void HighFrequencyCorrelationsBetweenChannels();

  //-------------------- Expert 2 Calculations
  void Expert2Calculations();  // see list in the method itself (.cc file)

  void LowFrequencyMeanCorrelationsBetweenTowers();
  void HighFrequencyMeanCorrelationsBetweenTowers();

  void LowFrequencyMeanCorrelationsBetweenSCs();
  void HighFrequencyMeanCorrelationsBetweenSCs();

  //===================================== "technical" methods ==========================================

  void GetReadyToCompute();  // Make result root file name and check events
  void SampleValues();       // 3D histo of the sample ADC value for each triple (channel,sample,event)

  //.................................... Technical calculation methods (Stin = Tower or SC)
  void LowFrequencyMeanCorrelationsBetweenStins();
  void HighFrequencyMeanCorrelationsBetweenStins();

  //...................................... ROOT file methods
  const TString& GetRootFileName() const;
  const TString& GetRootFileNameShort() const;
  const TString& GetNewRootFileName() const;
  const TString& GetNewRootFileNameShort() const;

  Bool_t WriteRootFile();
  Bool_t WriteNewRootFile(const TString&);
  Bool_t WriteRootFile(const Text_t*, Int_t&);

  void TRootStinNumbers();
  void TRootNbOfEvts(const Int_t&);

  void TRootAdcEvt(const Int_t&, const Int_t&);

  void TRootMSp(const Int_t&);
  void TRootSSp(const Int_t&);

  void TRootCovCss(const Int_t&, const Int_t&);
  void TRootCorCss(const Int_t&, const Int_t&);

  void TRootLfCov();
  void TRootLfCor();

  void TRootHfCov();
  void TRootHfCor();

  void TRootLFccMoStins();
  void TRootHFccMoStins();

  void TRootPed();
  void TRootTno();
  void TRootMeanCorss();

  void TRootLfn();
  void TRootHfn();
  void TRootSigCorss();

  void TRootAvPed();
  void TRootAvEvCorss();
  void TRootAvSigCorss();
  void TRootAvTno();
  void TRootAvLfn();
  void TRootAvHfn();

  //................................ Flags Print Comments/Debug
  void PrintNoComment();    // (default) Set flags to forbid the printing of all the comments
                            // except ERRORS
  void PrintWarnings();     // Set flags to authorize printing of some warnings
  void PrintComments();     // Set flags to authorize printing of infos and some comments
                            // concerning initialisations
  void PrintAllComments();  // Set flags to authorize printing of all the comments

  ClassDefOverride(TEcnaRun, 1)  // Calculation of correlated noises from data
};

#endif  //  CL_TEcnaRun_H
