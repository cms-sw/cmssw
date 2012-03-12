#ifndef   CL_TEcnaRun_H
#define   CL_TEcnaRun_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include "Riostream.h"

// ROOT include files
#include "TObject.h"
#include "TSystem.h"
#include "TString.h"
#include "TVectorD.h"

// user include files
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"

//-------------------------------- TEcnaRun.h ----------------------------
// 
//   Creation: 03 Dec  2002
//   Update  : 21 Oct  2009
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//-----------------------------------------------------------------------

class TEcnaRun: public TObject {
  
 private:

  //............ attributes

  Int_t fgMaxCar;      // Max nb of caracters for char* 

  Int_t fCnaCommand,  fCnaError;

  Int_t fCnew;          // flags for dynamical allocation
  Int_t fCdelete;       

  TString fTTBELL;

  Int_t* fMiscDiag;                          // Counters for miscellaneous diagnostics
  Int_t  fNbOfMiscDiagCounters;
  Int_t  fMaxMsgIndexForMiscDiag;

  TEcnaHeader    *fFileHeader;    // header for result type file
  TEcnaParEcal   *fEcal;          // for access to the Ecal current subdetector parameters
  TEcnaNumbering *fEcalNumbering; // for access to the Ecal channel, Stin and Stex numbering
  TEcnaParCout   *fCnaParCout;    // for comment/error messages
  TEcnaParPaths  *fCnaParPaths;   // for file access
  TEcnaWrite     *fCnaWrite;      // for access to the results files

  TString fFlagSubDet;
  TString fStexName, fStinName;

  Bool_t  fOpenRootFile;   // flag open ROOT file (open = kTRUE, close = kFALSE)
  Int_t   fReadyToReadData;

  TString  fRootFileName;
  TString  fRootFileNameShort;
  TString  fNewRootFileName;
  TString  fNewRootFileNameShort;

  Int_t   fSpecialStexStinNotIndexed;  // management of event distribution building
  Int_t   fStinIndexBuilt;
  Int_t   fBuildEvtNotSkipped;

  Int_t   fNbSampForFic;
  Int_t   fNbSampForCalc;

  Int_t   fNumberOfEvents;

  Double_t*** fT3d_distribs;   // 3D array[channel][sample][event] ADC values distibutions
  Double_t**  fT3d2_distribs; 
  Double_t*   fT3d1_distribs;
  Int_t*      fTagAdcEvt;

  Int_t**     fT2d_NbOfEvts; // 2D array[channel][sample] max nb of evts read for a given (channel,sample) 
  Int_t*      fT1d_NbOfEvts;
  Int_t*      fTagNbOfEvts;

  Int_t*      fT1d_StexStinFromIndex; // 1D array[Stin] Stin Number as a function of the index Stin
  Int_t*      fTagStinNumbers;

  Double_t**  fT2d_ev;        // 2D array[channel][sample] for expectation values
  Double_t*   fT1d_ev;
  Int_t*      fTagMSp;

  Double_t**  fT2d_sig;       // 2D array[channel][sample] for sigmass
  Double_t*   fT1d_sig;
  Int_t*      fTagSSp;
 
  Double_t*** fT3d_cov_ss;    // 3D array[channel][sample][sample] for (sample,sample) covariances
  Double_t**  fT3d2_cov_ss;
  Double_t*   fT3d1_cov_ss;
  Int_t*      fTagCovCss;

  Double_t*** fT3d_cor_ss;    // 3D array[channel][sample][sample] for (sample,sample) correlations
  Double_t**  fT3d2_cor_ss;
  Double_t*   fT3d1_cor_ss;
  Int_t*      fTagCorCss;

//...........................................................................
  Double_t*   fT1d_ev_ev;     // 1D array[channel] for expectation values of the expectation values of the samples
  Int_t*      fTagPed;        // (PEDESTAL)
  Double_t*   fT1d_av_mped;   // 1D array[Stin] for expectation values of the Pesdestals of the Stins
  Int_t*      fTagAvPed;      // (AVERAGED PEDESTAL)

  Double_t*   fT1d_evsamp_of_sigevt; // 1D array[channel] for expectation values of the sigmas of the samples
  Int_t*      fTagTno;               // (TOTAL NOISE)
  Double_t*   fT1d_av_totn;          // 1D array[Stin] for expectation values of the total noise
  Int_t*      fTagAvTno;             //(AVERAGED TOTAL NOISE)

  Double_t*   fT1d_ev_cor_ss;   // 1D array[channel] for expectation values of the cor(s,s)
  Int_t*      fTagMeanCorss;    // (MEAN OF COR(S,S))
  Double_t*   fT1d_av_ev_corss; // 1D array[Stin] for expectation values of the mean of cor(s,s)
  Int_t*      fTagAvMeanCorss;  // (AVERAGED MEAN OF COR(S,S))

  Double_t*   fT1d_sigevt_of_evsamp;  // 1D array[channel] for sigmas of the expectation values of the samples
  Int_t*      fTagLfn;                // (LOW FREQUENCY NOISE)
  Double_t*   fT1d_av_lofn;           // 1D array[Stin]  the expectation values of the low frequency noise
  Int_t*      fTagAvLfn;              // (AVERAGED LOW FREQUENCY NOISE)

  Double_t*   fT1d_evevt_of_sigsamp; // 1D array[channel] for sigmas of the sigmas of the samples
  Int_t*      fTagHfn;               // (HIGH FREQUENCY NOISE)
  Double_t*   fT1d_av_hifn;          // 1D array[channel] for expectation values of the high frequency noise
  Int_t*      fTagAvHfn;             // (AVERAGED HIGH FREQUENCY NOISE)

  Double_t*   fT1d_sig_cor_ss;   // 1D array[channel] for sigmas of the cor(s,s)
  Int_t*      fTagSigCorss;      // (SIGMA OF COR(S,S))
  Double_t*   fT1d_av_sig_corss; // 1D array[channel] for expectation values of sigmas  the  of the cor(s,s)
  Int_t*      fTagAvSigCorss;    // (AVERAGED SIGMA OF COR(S,S))

  //...........................................................................
  Double_t**  fT2d_lf_cov; // 2D array[channel][channel] for (channel,channel) low frequency covariances
  Double_t*   fT2d1_lf_cov;
  Int_t*      fTagLfCov;

  Double_t**  fT2d_lf_cor; // 2D array[channel][channel] for (channel,channel) low frequency correlations
  Double_t*   fT2d1_lf_cor;
  Int_t*      fTagLfCor;

  //...........................................................................
  Double_t**  fT2d_hf_cov; // 2D array[channel][channel] for (channel,channel) low frequency covariances
  Double_t*   fT2d1_hf_cov;
  Int_t*      fTagHfCov;

  Double_t**  fT2d_hf_cor; // 2D array[channel][channel] for (channel,channel) low frequency correlations
  Double_t*   fT2d1_hf_cor;
  Int_t*      fTagHfCor;

  //------------------------- 2 tableaux (ci,cj)
  Double_t**  fT2d_lfcc_mostins;  // 2D array[Stin][Stin] for (Stin,Stin) mean of the cov(c,c)
  Double_t*   fT2d1_lfcc_mostins; // (relevant ones) averaged over samples
  Int_t*      fTagLFccMoStins;

  Double_t**  fT2d_hfcc_mostins;  // 2D array[Stin][Stin] for (Stin,Stin) mean of the cor(c,c)
  Double_t*   fT2d1_hfcc_mostins; // (relevant ones) averaged over samples
  Int_t*      fTagHFccMoStins;

  //------------------------------------------------------------------------------------

  Int_t**     fT2dCrysNumbersTable;
  Int_t*      fT1dCrysNumbersTable;

  ofstream    fFcout_f;

  Int_t       fFlagPrint;
  Int_t       fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;


 public: 

  //................. constructors
  
  TEcnaRun();                            //  constructor without argument
  TEcnaRun(const TString);               //  constructor with argument (FOR USER'S DECLARATION)
  TEcnaRun(const TString, const Int_t&); //  constructor with arguments (FOR USER'S DECLARATION)

  TEcnaRun(const TEcnaRun&);   //  copy constructor

  //.................... C++ methods

  //TEcnaRun&  operator=(const TEcnaRun&);  //  overloading of the operator=

  //................. destructor
  
  virtual ~TEcnaRun();
  
  //...................................................... methods that will (should) be private

  void Init();

  void SetEcalSubDetector(const TString);

  Bool_t OpenRootFile(const Text_t *, TString);
  Bool_t CloseRootFile(const Text_t *);

  //======================================= methods for the user =========================================
  void GetReadyToReadData(TString, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  void GetReadyToReadData(TString, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Int_t&);
  
  Bool_t BuildEventDistributions(const Int_t&, const Int_t&, const Int_t&, const Int_t&, const Double_t&);
  Bool_t ReadEventDistributions();
  Bool_t ReadEventDistributions(const Int_t&);

  void GetReadyToCompute();

  void StartStopDate(TString, TString);
  void StartStopTime(time_t, time_t);

  //................... Calculation methods ( associated to GetReadyToReadData(...) )
  void SampleValues();               // 3D histo of the sample ADC value 
                                     // for each triple (channel,sample,event)
  void SampleMeans();                // Calculation of the expectation values over the events
                                     // for each sample and for each channel
  void SampleSigmas();               // Calculation of the variances over the events
                                     // for each sample and for each channel
  //...........................................
  void CovariancesBetweenSamples();  // Calculation of the (s,s) covariances over the events
                                     // for each channel
  void CorrelationsBetweenSamples(); // Calculation of the (s,s) correlations over the events
                                     // for each channel
  //..........................................................
  void Pedestals();                        
  void TotalNoise();
  void LowFrequencyNoise();
  void HighFrequencyNoise();
  void MeanOfCorrelationsBetweenSamples();
  void SigmaOfCorrelationsBetweenSamples();

  //..........................................................
  void AveragedPedestals();
  void AveragedTotalNoise();
  void AveragedLowFrequencyNoise();
  void AveragedHighFrequencyNoise();
  void AveragedMeanOfCorrelationsBetweenSamples();
  void AveragedSigmaOfCorrelationsBetweenSamples();

  //..........................................................
  void LowFrequencyCovariancesBetweenChannels();
  void HighFrequencyCovariancesBetweenChannels();
  void LowFrequencyCorrelationsBetweenChannels();
  void HighFrequencyCorrelationsBetweenChannels();
  //..........................................................
  void LowFrequencyMeanCorrelationsBetweenTowers();
  void HighFrequencyMeanCorrelationsBetweenTowers();

  void LowFrequencyMeanCorrelationsBetweenSCs();
  void HighFrequencyMeanCorrelationsBetweenSCs();

  //===================================================================================================

  //.................................... Technical calculation methods (Stin = Tower or SC)
  void LowFrequencyMeanCorrelationsBetweenStins();
  void HighFrequencyMeanCorrelationsBetweenStins();

  //...................................... ROOT file methods
  TString GetRootFileName();
  TString GetRootFileNameShort();
  TString GetNewRootFileName();
  TString GetNewRootFileNameShort();

  Bool_t WriteRootFile();
  Bool_t WriteNewRootFile(const TString);
  Bool_t WriteRootFile(const Text_t *, Int_t&);

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
  void PrintNoComment();   // (default) Set flags to forbid the printing of all the comments
                           // except ERRORS
  void PrintWarnings();    // Set flags to authorize printing of some warnings
  void PrintComments();    // Set flags to authorize printing of infos and some comments
                           // concerning initialisations
  void PrintAllComments(); // Set flags to authorize printing of all the comments

ClassDef(TEcnaRun,1) // Calculation of correlated noises from data
};  

#endif    //  CL_TEcnaRun_H











