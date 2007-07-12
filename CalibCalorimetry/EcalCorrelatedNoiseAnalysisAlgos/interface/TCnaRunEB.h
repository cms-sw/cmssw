#ifndef   CL_TCnaRunEB_H
#define   CL_TCnaRunEB_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include "Riostream.h"

// ROOT include files
#include "TObject.h"
#include "TSystem.h"
#include "TString.h"
#include "TVectorD.h"

// user include files
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRootFile.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaResultType.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaHeaderEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TDistrib.h"


//-------------------------------- TCnaRunEB.h ----------------------------
// 
//   Creation: 03 Dec  2002
//   Update  : 03 May  2007
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//-----------------------------------------------------------------------

class TCnaRunEB: public TObject {
  
 private:

  //............ attributes

  Int_t       fgMaxCar;      // Max nb of caracters for char*  
  Int_t       fDim_name;

  Int_t       fCnaCommand,  fCnaError;

  Int_t       fCnew;          // flags for dynamical allocation
  Int_t       fCdelete;       

  TString     fTTBELL;

  Int_t*      fMiscDiag;                          // Counters for miscellaneous diagnostics
  Int_t       fNbOfMiscDiagCounters;
  Int_t       fMaxMsgIndexForMiscDiag;

  TCnaHeaderEB *fFileHeader;   // header for result type file

  Bool_t      fOpenRootFile;   // flag open ROOT file (open = kTRUE, close = kFALSE)
  Int_t       fCodeHeaderAscii;
  Int_t       fCodeRoot;
  Int_t       fCodeCorresp;

  Int_t       fReadyToReadData;

  Int_t       fSpecialSMTowerNotIndexed;
  Int_t       fTowerIndexBuilt;

  Double_t*** fT3d_distribs;   // 3D array[channel][sample][event] ADC values distibutions
  Double_t**  fT3d2_distribs; 
  Double_t*   fT3d1_distribs;
  Int_t       fCodeSampTime;
  Int_t*      fTagSampTime;

  TDistrib**  fVal_data;          // 2D array[channel][sample] distributions of the ADC events
  TDistrib*   fVal_dat2;          //

  Int_t**     fT2d_EvtNbInLoop;   // 2D array[tower][cna event number] event number in the loop of the data reading   
  Int_t*      fT1d_EvtNbInLoop;
  Int_t*      fTagEvtNbInLoop;

  Int_t**     fT2d_LastEvtNumber; // 2D array[channel][sample] max nb of evts read for a given (channel,sample) 
  Int_t*      fT1d_LastEvtNumber;
  Int_t*      fTagLastEvtNumber;

  Int_t*      fT1d_SMtowFromIndex; // 1D array[tower] tower Number as a function of the index tower
  Int_t*      fTagTowerNumbers;

  Double_t**  fT2d_ev;        // 2D array[channel][sample] for expectation values
  Double_t*   fT1d_ev;
  Int_t       fCodeEv;
  Int_t*      fTagEv;

  Double_t**  fT2d_var;       // 2D array[channel][sample] for variances
  Double_t*   fT1d_var;
  Int_t       fCodeVar;
  Int_t*      fTagVar;

  Double_t*** fT3d_his_s;     // 3D array[channel][sample][bin for ADC value] histograms for ADC values
  Double_t**  fT2d_his_s;      
  Double_t*   fT1d_his_s;
  Double_t**  fT2d_xmin;      // 2D array[channel][sample] minimum ADC values of the ADC histograms
  Double_t*   fT1d_xmin;
  Double_t**  fT2d_xmax;      // 2D array[channel][sample] maximum ADC values of the ADC histograms
  Double_t*   fT1d_xmax;
  Int_t       fCodeEvts;
  Int_t*      fTagEvts;
  
  Double_t*** fT3d_cov_cc;    // 3D array[sample][channel][channel] for (channel,channel) covariances 
  Double_t**  fT3d2_cov_cc;
  Double_t*   fT3d1_cov_cc;
  Int_t       fCodeCovScc;
  Int_t*      fTagCovScc;

  Double_t*** fT3d_cor_cc;    // 3D array[sample][channel][channel] for (channel,channel) correlations
  Double_t**  fT3d2_cor_cc;
  Double_t*   fT3d1_cor_cc;
  Int_t       fCodeCorScc;
  Int_t*      fTagCorScc;

  Double_t**  fT2d_cov_cc_mos; // 2D array[channel][channel] for (channel,channel) covariances, mean over samples
  Double_t*   fT2d1_cov_cc_mos;
  Int_t       fCodeCovSccMos;
  Int_t*      fTagCovSccMos;

  Double_t**  fT2d_cor_cc_mos; // 2D array[channel][channel] for (channel,channel) correlations, mean over samples
  Double_t*   fT2d1_cor_cc_mos;
  Int_t       fCodeCorSccMos;
  Int_t*      fTagCorSccMos;

  Double_t**  fT2d_cov_moscc_mot;  // 2D array[tower][tower] for (tower,tower) mean of the cov(c,c)
  Double_t*   fT2d1_cov_moscc_mot; // (relevant ones) averaged over samples
  Int_t       fCodeCovMosccMot;
  Int_t*      fTagCovMosccMot;

  Double_t**  fT2d_cor_moscc_mot;  // 2D array[tower][tower] for (tower,tower) mean of the cor(c,c)
  Double_t*   fT2d1_cor_moscc_mot; // (relevant ones) averaged over samples
  Int_t       fCodeCorMosccMot;
  Int_t*      fTagCorMosccMot;

  Double_t*** fT3d_cov_ss;    // 3D array[channel][sample][sample] for (sample,sample) covariances
  Double_t**  fT3d2_cov_ss;
  Double_t*   fT3d1_cov_ss;
  Int_t       fCodeCovCss;
  Int_t*      fTagCovCss;

  Double_t*** fT3d_cor_ss;    // 3D array[channel][sample][sample] for (sample,sample) correlations
  Double_t**  fT3d2_cor_ss;
  Double_t*   fT3d1_cor_ss;
  Int_t       fCodeCorCss;
  Int_t*      fTagCorCss;

  Double_t*   fT1d_ev_ev;     // 1D array[channel] for expectation values of the expectation values of the samples
  Int_t       fCodeEvEv;
  Int_t*      fTagEvEv;

  Double_t*   fT1d_ev_sig;    // 1D array[channel] for expectation values of the sigmas of the samples
  Int_t       fCodeEvSig;
  Int_t*      fTagEvSig;

  Double_t*   fT1d_ev_cor_ss;  // 1D array[channel] for expectation values of the cor(s,s)
  Int_t       fCodeEvCorCss;
  Int_t*      fTagEvCorCss;

  Double_t*   fT1d_sig_ev;  // 1D array[channel] for sigmas of the expectation values of the samples
  Int_t       fCodeSigEv;
  Int_t*      fTagSigEv;

  Double_t*   fT1d_sig_sig; // 1D array[channel] for sigmas of the sigmas of the samples
  Int_t       fCodeSigSig;
  Int_t*      fTagSigSig;

  Double_t*   fT1d_sig_cor_ss;  // 1D array[channel] for sigmas of the cor(s,s)
  Int_t       fCodeSigCorCss;
  Int_t*      fTagSigCorCss;

  Double_t**  fT2d_sv_correc_covss_s;  // 2D array[channel][sample] sample corrections from cov(s,s)  
  Double_t*   fT2d1_sv_correc_covss_s;
  Int_t       fCodeSvCorrecCovCss;
  Int_t*      fTagSvCorrecCovCss;

  Double_t*** fT3d_cov_correc_covss_s; // 3D array[channel][sample][sample] cov correc factors from cov(s,s)
  Double_t**  fT3d2_cov_correc_covss_s;
  Double_t*   fT3d1_cov_correc_covss_s;
  Int_t       fCodeCovCorrecCovCss;
  Int_t*      fTagCovCorrecCovCss;

  Double_t*** fT3d_cor_correc_covss_s; // 3D array[channel][sample][sample] cor correc factors from cov(s,s)
  Double_t**  fT3d2_cor_correc_covss_s;
  Double_t*   fT3d1_cor_correc_covss_s;
  Int_t       fCodeCorCorrecCovCss;
  Int_t*      fTagCorCorrecCovCss;

  Int_t**     fT2dCrysNumbersTable;
  Int_t*      fT1dCrysNumbersTable;

  Double_t**  fjustap_2d_ev;
  Double_t*   fjustap_1d_ev;

  Double_t**  fjustap_2d_var;
  Double_t*   fjustap_1d_var;

  Double_t**  fjustap_2d_cc;
  Double_t*   fjustap_1d_cc;

  Double_t**  fjustap_2d_ss;
  Double_t*   fjustap_1d_ss;

  ofstream    fFcout_f;
  ifstream    fFcin_rr;
  ifstream    fFcin_ra;

  TString     fCfgResultsRootFilePath; // absolute path for the results .root files (/afs/etc...)
  TString     fRootFileNameShort;      // name of  the results ROOT files 
  TString     fRootFileName;           // name of  the results ROOT files = fPathRoot/fRootFileNameShort

  TString     fCfgResultsAsciiFilePath; // absolute path for the results .ascii files (/afs/etc...)
  TString     fAsciiFileName;       // name of  the results ASCII files
  TString     fAsciiFileNameShort;  // name of  the results ASCII files = fPathAscii/fA1sciiFileNameShort

  TString     fFileForResultsRootFilePath;  // name of the file containing the results .root  file path
  TString     fFileForResultsAsciiFilePath; // name of the file containing the results .ascii file path

  Int_t       fSectChanSizeX, fSectChanSizeY;
  Int_t       fSectSampSizeX, fSectSampSizeY;

  Int_t       fNbChanByLine;  // Nb channels by line (for ASCII results file)
  Int_t       fNbSampByLine;  // Nb samples by line  (for ASCII results file)
  Int_t       fUserSamp;      // Current sample  number (for ASCII results file)    
  Int_t       fUserSMEcha;    // Current electronic channel number in SM (for ASCII results file) 

  Int_t       fFlagPrint;
  Int_t       fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //.......................................... private methods

  void        fCopy(const TCnaRunEB&);
  void        fMakeResultsFileName(const Int_t&);
  void        fAsciiFileWriteHeader(const Int_t&);
  void        fT1dWriteAscii(const Int_t&, const Int_t&, const Int_t&);
  void        fT2dWriteAscii(const Int_t&, const Int_t&, const Int_t&);

 public: 

  //................. constructors
  
  TCnaRunEB();                            //  constructor without argument (FOR USER'S DECLARATION)
  TCnaRunEB(const TCnaRunEB&);              //  copy constructor

  //.................... C++ methods

  TCnaRunEB&  operator=(const TCnaRunEB&);  //  overloading of the operator=

  //................. destructor
  
  virtual ~TCnaRunEB();
  
  //...................................................... methods that will (should) be private

  void         Init();

  void         GetPathForResultsRootFiles();
  void         GetPathForResultsRootFiles(const TString);

  void         GetPathForResultsAsciiFiles();
  void         GetPathForResultsAsciiFiles(const TString);

  Bool_t       OpenRootFile(Text_t *, TString);
  Bool_t       CloseRootFile(Text_t *);

  //............................................................... Genuine public user's methods
  
  void         GetReadyToReadData(TString,      const Int_t&,       Int_t&,       Int_t&,
				  const Int_t&);
  void         GetReadyToReadData(TString,      const Int_t&,       Int_t&,       Int_t&,
				  const Int_t&, const Int_t&);
  
  Bool_t       BuildEventDistributions(const Int_t&, const Int_t&, const Int_t&,
				       const Int_t&, const Double_t&);
  void         GetReadyToCompute();

  void         StartStopDate(TString, TString);
  void         StartStopTime(time_t, time_t);

  TString      GetRootFileNameShort();

  TString      GetAnalysisName();
  Int_t        GetRunNumber();
  Int_t        GetFirstTakenEvent();
  Int_t        GetNumberOfTakenEvents();
  Int_t        GetSMNumber();
  Int_t        GetNentries();


  Int_t        PickupNumberOfEvents(const Int_t&, const Int_t&); // Number of events provided
                                                                 // by the data reading for a given
                                                                 // (channel, sample)

  //................... Calculation methods ( associated to GetReadyToReadData(...) )

  void  ComputeExpectationValuesOfSamples(); // Calculation of the expectation values of the samples
                                             // for all the channels (i.e. for all the towers and
                                             // all the crystals)
 
  void  ComputeVariancesOfSamples();         // Calculation of the variances of the samples
                                             // for all the channels (i.e. for all the towers and
                                             // all the crystals)
 
  void  MakeHistosOfSampleDistributions();   // Histogram making of the sample ADC distributions
                                             // for all the pairs (channel,sample)

  void  MakeHistosOfSamplesAsFunctionOfEvent(); // Histogram making of the sample ADC value as a function
                                                // of the event for all the pairs (channel,sample)

  void  ComputeCovariancesBetweenSamples();  // Calculation of the (s,s) covariances for all the samples
  void  ComputeCorrelationsBetweenSamples(); // Calculation of the (s,s) correlations for all the samples

  void  ComputeCovariancesBetweenChannels(); // Calculation of the (channel,channel) covariances for each sample
  void  ComputeCovariancesBetweenChannelsMeanOverSamples();
        // Calculation of the (channel,channel) covariances averaged over the samples

  void  ComputeCorrelationsBetweenChannels(); // Calculation of the (channel,channel) correlations for each sample
  void  ComputeCorrelationsBetweenChannelsMeanOverSamples();
        // Calculation of the (channel,channel) correlations averaged over the samples

  void  ComputeCovariancesBetweenTowersMeanOverSamplesAndChannels();
        // Calculation of the cov(channel,channel) averaged over the samples and
        // calculation of the mean of these cov(channel,channel) for all the towers

  void  ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels();
        // Calculation of the cor(channel,channel) averaged over the samples and
        // calculation of the mean of these cor(channel,channel) for all the towers

  //------------------------------ quantites calculees a partir des quantites brutes --------------------------
  void  ComputeExpectationValuesOfExpectationValuesOfSamples(); 
        // Calc. of the exp.val. of the exp.val. of the samples for all the channels
  void  ComputeExpectationValuesOfSigmasOfSamples(); 
        // Calc. of the exp.val. of the sigmas of the samples for all the channels
  void  ComputeExpectationValuesOfCorrelationsBetweenSamples(); 
        // Calc. of the exp.val. of the (sample,sample) correlations for all the channels

  void  ComputeSigmasOfExpectationValuesOfSamples();
        // Calc. of the sigmas of the exp.val. of the samples for all the channels
  void  ComputeSigmasOfSigmasOfSamples();
        // Calc. of the sigmas of the of the sigmas of the samples for all the channels
  void  ComputeSigmasOfCorrelationsBetweenSamples();
        // Calc. of the sigmas of the (sample,sample) correlations for all the channels

  void  ComputeCorrectionsToSamplesFromCovss(const Int_t&);  // Calculation of the corrections coefficients
                                                             // to the samples from cov(s,s)
                                                             // void of (nb of first samples)
  void  ComputeCorrectionFactorsToCovss(); // Calculation of the corrections factors to cov(s,s) (OLD? CHECK)
  void  ComputeCorrectionFactorsToCorss(); // Calculation of the corrections factors to cor(s,s) (OLD? CHECK)
  //-----------------------------------------------------------------------------------------------------------

  //...................................... ROOT file writing methods

  Bool_t  WriteRootFile();
  Bool_t  WriteRootFile(Text_t *);

  void    TRootTowerNumbers();
  void    TRootLastEvtNumber();

  void    TRootSampTime(const Int_t&);

  void    TRootEv();
  void    TRootVar();
  void    TRootEvts(const Int_t&);
  void    TRootEvtsXmin(const Int_t&);
  void    TRootEvtsXmax(const Int_t&);

  void    TRootCovCss(const Int_t&);
  void    TRootCorCss(const Int_t&);
  void    TRootCovSccMos();
  void    TRootCorSccMos();
  void    TRootCovMosccMot();
  void    TRootCorMosccMot();

  void    TRootEvEv();
  void    TRootEvSig();
  void    TRootEvCorCss();

  void    TRootSigEv();
  void    TRootSigSig();
  void    TRootSigCorCss();

  void    TRootSvCorrecCovCss();
  void    TRootCovCorrecCovCss(const Int_t&);
  void    TRootCorCorrecCovCss(const Int_t&);

  //.................................... ASCII writing file methods
  void    WriteAsciiExpectationValuesOfSamples();
  void    WriteAsciiVariancesOfSamples();

  void    WriteAsciiCovariancesBetweenSamples(const Int_t&);
  void    WriteAsciiCorrelationsBetweenSamples(const Int_t&);

  void    WriteAsciiExpectationValuesOfCorrelationsBetweenSamples();
  void    WriteAsciiSigmasOfCorrelationsBetweenSamples();

  void    WriteAsciiSvCorrecCovCss();
  void    WriteAsciiCovCorrecCovCss(const Int_t&);
  void    WriteAsciiCorCorrecCovCss(const Int_t&);

  void    WriteAsciiCnaChannelTable();

  //............... Flags Print Comments/Debug

  void  PrintNoComment();   // (default) Set flags to forbid the printing of all the comments
                            // except ERRORS
  void  PrintWarnings();    // Set flags to authorize printing of some warnings
  void  PrintComments();    // Set flags to authorize printing of infos and some comments
                            // concerning initialisations
  void  PrintAllComments(); // Set flags to authorize printing of all the comments

ClassDef(TCnaRunEB,1) // Calculation of correlated noises from data
};  

#endif    //  CL_TCnaRunEB_H











