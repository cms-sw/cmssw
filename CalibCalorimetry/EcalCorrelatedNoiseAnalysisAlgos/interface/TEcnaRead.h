#ifndef   CL_TEcnaRead_H
#define   CL_TEcnaRead_H

#include <time.h>
#include <math.h>

#include "TSystem.h"
#include "TObject.h"
#include "TString.h"
#include "Riostream.h"
#include "TVectorD.h"
#include "TMatrixD.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParHistos.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaResultType.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRootFile.h"

///-----------------------------------------------------------
///   TEcnaRead.h
///   Update: 15/02/2011
///   Authors:   B.Fabbro (bernard.fabbro@cea.fr), FX Gentit
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///==============> INTRODUCTION
///
///    This class allows the user to read the .root results files (containing
///   expectation values, variances, covariances, correlations and other
///   quantities of interest) previously computed by the class TEcnaRun.
///   The results are available in arrays.
///
///==============> PRELIMINARY REMARK
///
///   Another class named TEcnaHistos can be used directly to make plots of results
///   previously computed by means of the class TEcnaRun. The class TEcnaHistos
///   calls TEcnaRead and manage the reading of the .root result file
///
///
///            ***   I N S T R U C T I O N S   F O R   U S E   ***
///
///   //==============> TEcnaRead DECLARATION
///
///   // The declaration is done by calling the constructor without argument:
///
///       TEcnaRead* MyCnaRead = new TEcnaRead();
///   
///   //==============> PREPARATION METHOD FileParameters(...)
///
///   // There is a preparation method named: FileParameters(...);
///
///   //    FileParameters(...) is used to read the quantities written
///   //    in the ROOT files in order to use these quantities for analysis.
///
///   //.......  Example of program using FileParameters(...)
///
///   //  This example describes the reading of one result file. This file is situated in a
///   //  directory which name is given by the contents of a TString named PathForRootFile
///
///   //................ Set values for the arguments and call to the method
///
///      TString AnalysisName      = "AdcPed12"
///      Int_t   RunNumber         = 132440;
///      Int_t   FirstReqEvtNumber = 1;   |    (numbering starting from 1)
///      Int_t   LastReqEvtNumber  = 300; | => treats 150 evts between evt#100 and evt#300 (included)
///      Int_t   ReqNbOfEvts       = 150; |
///      TString PathForRootFile   = "/afs/cern.ch/etc..." // .root result files directory
///
///      TEcnaRead*  MyCnaRead = new TEcnaRead();
///      MyCnaRead->FileParameters(AnalysisName,      RunNumber,
///                                FirstReqEvtNumber, LastReqEvtNumber, ReqNbOfEvts, StexNumber,
///                                PathForRootFile);
///
///                                *==> Stex = SM if EB, Dee if EE
///
///    //==============>  CALL TO THE METHOD: Bool_t LookAtRootFile() (MANDATORY)
///    //                 and method: Bool_t   DataExist();  // if data exist:  kTRUE , if not: kFALSE 
///    
///    // This methods returns a boolean. It tests the existence
///    // of the ROOT file corresponding to the argument values given
///    // in the call to the method FileParameters(...).
///    // It is recommended to test the return value of the method.
///
///    //....... Example of use:
///
///     if( MyCnaRead->LookAtRootFile() == kFALSE )
///        {
///          cout << "*** ERROR: ROOT file not found" << endl;
///        }
///      else
///        {
///         //........... The ROOT file exists and has been found
///         //
///         //---> CALLS TO THE METHODS WHICH RECOVER THE QUANTITIES. EXAMPLE:
///         //     (see the complete list of the methods hereafter)
///
///           Int_t   MaxSamples  = 10;
///           TMatrixD CorMat(MaxSamples,MaxSamples);
///           Int_t Tower = 59;
///           Int_t Channel =  4;
///           CorMat = MyCnaRead->ReadMatrix(MaxSamples, "Cor", "Samples", Tower, Channel);
///
///           //   arguments: "Cor" = correlation matrix,   "Samples" = between samples
///           //   large amount of possibilities for syntax: "Cor", "cor", "correlation", etc...
///           //                                             "Samples", "samples", "samp", etc...
///
///           if( MyCnaRead->DataExist() == kFALSE )
///             {          :
///                Analysis of the correlations, etc...
///                        :
///             }
///           else
///             {
///               cout << "problem while reading file. data not available. " << endl;
////            }
///
///        }
///
///******************************************************************************
///
///                      *=======================*
///                      | DETAILLED DESCRIPTION |
///                      *=======================*
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///       Method FileParameters(...) and associated methods
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///   TEcnaRead* MyCnaRead = new TEcnaRead();  // declaration of the object MyCnaRead
///
///   MyCnaRead->FileParameters(AnalysisName, RunNumber,    NbOfSamples
///                             FirstReqEvtNumber, LastReqEvtNumber,  ReqNbOfEvts, StexNumber,
///                             PathForRootFile);           
///      
///   Arguments:
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
///      Int_t    NbOfSamples         number of samples (=10 maximum) 
///      Int_t    RunNumber           run number
///      Int_t    FirstReqEvtNumber   first requested event number (numbering starting from 1)
///      Int_t    LastReqEvtNumber    last  requested event number
///      Int_t    ReqNbOfEvts         requested number of events
///      Int_t    StexNumber          Stex number (Stex = SM if EB, Dee if EE)
///      
///     
///      TString  PathForRootFile: Path of the directory containing the ROOT file.
///               The path must be complete: /afs/cern.ch/user/... etc...
///
///==============> METHODS TO RECOVER THE QUANTITIES FROM THE ROOT FILE
///
///                SM = SuperModule  (EB) equivalent to a Dee   (EE)
///                SC = SuperCrystal (EE) equivalent to a Tower (EB)
///
///                Stex = SM    in case of EB  ,  Dee in case of EE
///                Stin = Tower in case of EB  ,  SC  in case of EE
///
///  n1StexStin = Stin#  in Stex
///             = Tower# in SM  (RANGE = [1,68])   if EB
///             = SC#    in Dee (RANGE = [1,149])  if EE
///
///  i0StexEcha = Channel# in Stex
///             = Channel# in SM  (RANGE = [0,1699])  if EB
///             = Channel# in Dee (RANGE = [0,3724])  if EE
///
///  i0StinEcha = Channel# in Stin
///             = Channel# in tower (RANGE = [0,24])  if EB 
///             = Channel# in SC    (RANGE = [0,24])  if EE
///
///  MaxCrysInStin     = Maximum number of Xtals in a tower or a SC (25)
///  MaxCrysEcnaInStex = Maximum number of Xtals in SM (1700) or in the matrix including Dee (5000)
///  MaxStinEcnaInStex = Maximum number of towers in SM (68)  or in the matrix including Dee (200)
///  MaxSampADC        = Maximum number of samples (10)
///  NbOfSample        = Number of samples used to perform the calculations
///                      (example: for the 3 first samples, NbOfSample = 3)
///
///=============================== Standard methods for the user ===============================
///
///  Intervals:     Tower or SC: [1,...]        Channel:[0,24]          Sample:[1,10]
///
///...........................................................................
///  TVectorD Read1DHisto(const Int_t& VecDim,  const TString& Quantity, const Int_t& Tower or SC,
///                       const Int_t& Channel, const Int_t& Sample);
///
///  Example:
///  Int_t Tower = 59; Int_t Channel = 10; Int_t Sample = 4;
///  NbOfEvts = 150;
///  TVectorD Adc(NbOfEvts);
///  Adc = Read1DHisto(NbOfEvts, "AdcValue", Tower, Channel, Sample;
///
///...........................................................................
///  TVectorD Read1DHisto(const Int_t& VecDim, const TString& Quantity, const Int_t& Tower or SC);
///
///  Example:
///  Int_t Tower = 59;
///  TVectorD SampMean(fEcal->MaxCrysInTow()*fEcal->MaxSampADC());
///  SampMean = Read1DHisto(fEcal->MaxCrysInTow()*fEcal->MaxSampADC(), "SampleMean", Tower);
///
///...........................................................................
///  TVectorD Read1DHisto(const Int_t& VecDim, const TString& Quantity, const TString& Detector);
///
///  Example:
///  TVectorD Pedestal(fEcal->MaxCrysInTow());
///  Pedestal = Read1DHisto(fEcal->MaxCrysInTow(), "Ped","SM");
///
///...........................................................................
///  TMatrixD ReadMatrix(const Int_t&, const TString&, const TString&, const Int_t&, const Int_t&);
///  TMatrixD ReadMatrix(const Int_t&, const TString&, const TString&);
///
///=============================== more "technical" methods ===============================
///
///  TVectorD and TMatrixD sizes are indicated after the argument lists
///
///
///  TMatrixD ReadNumberOfEventsForSamples(const Int_t& n1StexStin, const Int_t& MaxCrysInStin,
///                                        const Int_t& NbOfSamples);
/// // TMatrixD(MaxCrysInStin,NbOfSamples)
///
///  TVectorD ReadSampleAdcValues(const Int_t& i0StexEcha, const Int_t& sample,
///                               const Int_t& ReqNbOfEvts);        
/// // TVectorD(ReqNbOfEvts)
/// 
///  TVectorD ReadSampleMeans(const Int_t& n1StexStin, const Int_t& i0StinEcha,
///                           const Int_t& NbOfSamples);
/// // TVectorD(NbOfSamples)
///
///  TVectorD ReadSampleMeans(const Int_t& n1StexStin, const Int_t& MaxCrysInStin*NbOfSamples);
/// // TVectorD(MaxCrysInStin*NbOfSamples)
///
///  TVectorD ReadSampleSigmas(const Int_t& n1StexStin, const Int_t& i0StinEcha,
///                            const Int_t& NbOfSamples);
/// // TVectorD(NbOfSamples)
///
///  TVectorD ReadSampleSigmas(const Int_t& n1StexStin, const Int_t& MaxCrysInStin*NbOfSamples);
/// // TVectorD(MaxCrysInStin*NbOfSamples)
///
///  TMatrixD ReadCovariancesBetweenSamples(const Int_t& n1StexStin, const Int_t& i0StinEcha);
/// // TMatrixD(NbOfSamples,NbOfSamples)
///
///  TMatrixD ReadCorrelationsBetweenSamples(const Int_t& n1StexStin, const Int_t& i0StinEcha);
/// // TMatrixD(NbOfSamples,NbOfSamples)
///
/// -----------------------------------------------------------  TMatrixD
///
///  TMatrixD size is (MaxCrysInStin, MaxCrysInStin)
///
///  TMatrixD ReadLowFrequencyCovariancesBetweenChannels(const Int_t& n1StexStin_X,
///                                                      const Int_t& n1StexStin_Y,
///                                                      const Int_t& MaxCrysInStin); 
/// 
///  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels(const Int_t& n1StexStin_X,
///                                                       const Int_t& n1StexStin_Y,
///                                                       const Int_t& MaxCrysInStin);
///
///  TMatrixD ReadHighFrequencyCovariancesBetweenChannels(const Int_t& n1StexStin_X,
///                                                       const Int_t& n1StexStin_Y,
///                                                       const Int_t& MaxCrysInStin);
///
///  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels(const Int_t& n1StexStin_X,
///                                                        const Int_t& n1StexStin_Y,
///                                                        const Int_t& MaxCrysInStin);
///
/// ----------------------------------------------------------- TMatrixD
///
///  TMatrixD size is (MaxCrysEcnaInStex, MaxCrysEcnaInStex) (BIG!: 1700x1700 for EB and 5000x5000 for EE)
/// 
///  TMatrixD ReadLowFrequencyCovariancesBetweenChannels(const Int_t& MaxCrysEcnaInStex);
///  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels(const Int_t& MaxCrysEcnaInStex);
///
///  TMatrixD ReadHighFrequencyCovariancesBetweenChannels(const Int_t& MaxCrysEcnaInStex);
///  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels(const Int_t& MaxCrysEcnaInStex);
///
/// ----------------------------------------------------------- TMatrixD
///
///  TMatrixD size is (MaxStinEcnaInStex, MaxStinEcnaInStex)
///
///  TMatrixD ReadLowFrequencyMeanCorrelationsBetweenStins(const Int_t& MaxStinEcnaInStex);
///  TMatrixD ReadHighFrequencyMeanCorrelationsBetweenStins(const Int_t& MaxStinEcnaInStex);
///
/// ----------------------------------------------------------- TVectorD
///
///  TVectorD sizes are indicated after the argument lists
///
///  TVectorD ReadPedestals(const Int_t& MaxCrysEcnaInStex);                         // TVectorD(MaxCrysEcnaInStex)
///  TVectorD ReadTotalNoise(const Int_t& MaxCrysEcnaInStex);                        // TVectorD(MaxCrysEcnaInStex)
///  TVectorD ReadMeanCorrelationsBetweenSamples(const Int_t& MaxCrysEcnaInStex);    // TVectorD(MaxCrysEcnaInStex)
///
///  TVectorD ReadLowFrequencyNoise(const Int_t& MaxCrysEcnaInStex);                 // TVectorD(MaxCrysEcnaInStex)
///  TVectorD ReadHighFrequencyNoise(const Int_t& MaxCrysEcnaInStex);                // TVectorD(MaxCrysEcnaInStex)
///  TVectorD ReadSigmaOfCorrelationsBetweenSamples(const Int_t& MaxCrysEcnaInStex); // TVectorD(MaxCrysEcnaInStex)
///
///----------------------------------------------------------------------------------------------------------------
///  TString GetStartDate()
///  TString GetStopDate()
///  TString GetRunType()
///  Int_t   GetFirstReqEvtNumber();
///  Int_t   GetReqNbOfEvts();
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///                         Print Methods
///
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
///
///     Just after the declaration with the constructor,
///     you can set a "Print Flag" by means of the following "Print Methods":
///
///     TEcnaRead* MyCnaRead = new TEcnaRead(....); // declaration of the object MyCnaRead
///
///    // Print Methods: 
///
///    MyCnaRead->PrintNoComment();  // Set flag to forbid printing of all the comments
///                                  // except ERRORS.
///
///    MyCnaRead->PrintWarnings();   // (DEFAULT)
///                                  // Set flag to authorize printing of some warnings.
///                                  // WARNING/INFO: information on something unusual
///                                  // in the data.
///                                  // WARNING/CORRECTION: something wrong (but not too serious)
///                                  // in the value of some argument.
///                                  // Automatically modified to a correct value.
///
///    MyCnaRead->PrintComments();    // Set flag to authorize printing of infos
///                                   // and some comments concerning initialisations
///
///    MyCnaRead->PrintAllComments(); // Set flag to authorize printing of all the comments
///
///------------------------------------------------------------------------------------------------

class TEcnaRead: public TObject {
  
 private:

  //............ attributes

  // static  const  Int_t        fgMaxCar    = 512;          <== DANGEROUS ! 

  Int_t fgMaxCar;   // Max nb of caracters for char*

  Int_t fCnew;      // flags for dynamical allocation
  Int_t fCdelete;       

  TString fTTBELL;

  TEcnaObject    *fObjectManager;  // pointer to TEcnaObject keeped in attribute

  TEcnaParCout   *fCnaParCout;  // for comments or error messages
  TEcnaParPaths  *fCnaParPaths; // for NbBinsADC 

  TEcnaHeader    *fFileHeader;    // header for result file
  TEcnaParEcal   *fEcal;          // for Ecal parameters
  TEcnaNumbering *fEcalNumbering; // for Ecal numbering
  TEcnaParHistos *fCnaParHistos;  // for Histo codes
  TEcnaWrite     *fCnaWrite;      // for writing in ascii files

  //  TEcnaRootFile *gCnaRootFile;

  TString fFlagSubDet;
  TString fStexName, fStinName;

  Bool_t  fOpenRootFile; // flag open ROOT file (open = kTRUE, close = kFALSE)
  TString fCurrentlyOpenFileName; // Name of the file currently open
  TString fFlagNoFileOpen; // Flag to indicate that no file is open

  Int_t fReadyToReadRootFile;
  Int_t fLookAtRootFile;

  Int_t* fT1d_StexStinFromIndex; // 1D array[Stin] Stin Number as a function of the index Stin
  Int_t* fTagStinNumbers;
  Int_t  fMemoStinNumbers;

  //  Int_t fMemoReadNumberOfEventsforSamples;

  TString fPathRoot;  // path for results .root files directory

  Int_t fNbChanByLine;  // Nb channels by line (for ASCII results file)
  Int_t fNbSampByLine;  // Nb samples by line  (for ASCII results file)

  Int_t fFlagPrint;
  Int_t fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //............... flag data exists

  Bool_t fDataExist;

  //................ 3d array for sample ADC value fast transfert
  Double_t*** fT3d_AdcValues;   // 3D array[channel][sample][event] ADC values distibutions
  Double_t**  fT3d2_AdcValues; 
  Double_t*   fT3d1_AdcValues;

  //==========================================================================================
  //
  //                                      M E T H O D S
  //
  //==========================================================================================
  //.......................................... private methods

  void        fCopy(const TEcnaRead&);

 public: 

  //................. constructors
  TEcnaRead();                  //  constructor without argument (FOR USER'S DECLARATION)
  //  constructor with argument (FOR USER'S DECLARATION):
  TEcnaRead(TEcnaObject*, const TString&);

  TEcnaRead(const TEcnaRead&);   //  copy constructor

  //.................... C++ methods
  TEcnaRead&  operator=(const TEcnaRead&);  //  overloading of the operator=

  //................. destructor
  virtual ~TEcnaRead();
  
  //========================================================================
  //
  //                    METHODS FOR THE USER
  //
  //========================================================================
  void     FileParameters(const TString&,
			  const Int_t&, const Int_t&, const Int_t&,
			  const Int_t&, const Int_t&, const Int_t&, const TString&);
  
  Bool_t   LookAtRootFile();  // if file exists: kTRUE , if not: kFALSE
  Bool_t   DataExist();       // if data exist:  kTRUE , if not: kFALSE

  TVectorD Read1DHisto(const Int_t&, const TString&, const Int_t&, const Int_t&, const Int_t&);
  TVectorD Read1DHisto(const Int_t&, const TString&, const Int_t&);
  TVectorD Read1DHisto(const Int_t&, const TString&, const TString&);

  TMatrixD ReadMatrix(const Int_t&, const TString&, const TString&, const Int_t&, const Int_t&);
  TMatrixD ReadMatrix(const Int_t&, const TString&, const TString&);

  //========================================================================
  //
  //                       "TECHNICAL" METHODS
  //
  //========================================================================
  //...................................................... methods that will (should) be private
  void     Init();
  void     SetEcalSubDetector(const TString&);

  void     Anew(const TString&);
  void     Adelete(const TString&);

  Bool_t   OpenRootFile(const Text_t *, const TString&);
  Bool_t   CloseRootFile(const Text_t *);
  void     TestArrayDimH1(const TString&, const TString&, const Int_t&, const Int_t&);
  void     TestArrayDimH2(const TString&, const TString&, const Int_t&, const Int_t&);

  Bool_t   ReadRootFileHeader(const Int_t&);

  TVectorD ReadSampleAdcValues(const Int_t&, const Int_t&, const Int_t&, const Int_t&);  //(nb evts in burst) of (StinEcha,samp)
  //------------------------------------------------------------------------------------------------
  TVectorD ReadSampleMeans(const Int_t&,  const Int_t&, const Int_t&);    // (sample) of (StexStin,Xtal)
  TVectorD ReadSampleMeans(const Int_t&,  const Int_t&);                  // (MaxCrysInStin*sample) of (StexStin)

  TVectorD ReadSampleSigmas(const Int_t&, const Int_t&, const Int_t&);    // (sample) of (StexStin,Xtal)
  TVectorD ReadSampleSigmas(const Int_t&, const Int_t&);                  // (MaxCrysInStin*sample) of (StexStin)

  //------------------------------------------------------------------------------------------------
  TVectorD ReadNumberOfEvents(const Int_t&);                    // EcnaStexCrys of (StexEcha)
  TVectorD ReadPedestals(const Int_t&);                         // EcnaStexCrys of (StexEcha)
  TVectorD ReadTotalNoise(const Int_t&);                        // EcnaStexCrys of (StexEcha)
  TVectorD ReadLowFrequencyNoise(const Int_t&);                 // EcnaStexCrys of (StexEcha)
  TVectorD ReadHighFrequencyNoise(const Int_t&);                // EcnaStexCrys of (StexEcha)
  TVectorD ReadMeanCorrelationsBetweenSamples(const Int_t&);    // EcnaStexCrys of (StexEcha)
  TVectorD ReadSigmaOfCorrelationsBetweenSamples(const Int_t&); // EcnaStexCrys of (StexEcha)

  //------------------------------------------------------------------------------------------------
  TVectorD ReadAverageNumberOfEvents(const Int_t&);                    // EcnaStexStin of (StexStin)
  TVectorD ReadAveragePedestals(const Int_t&);                         // EcnaStexStin of (StexStin)
  TVectorD ReadAverageTotalNoise(const Int_t&);                        // EcnaStexStin of (StexStin)
  TVectorD ReadAverageLowFrequencyNoise(const Int_t&);                 // EcnaStexStin of (StexStin)
  TVectorD ReadAverageHighFrequencyNoise(const Int_t&);                // EcnaStexStin of (StexStin)
  TVectorD ReadAverageMeanCorrelationsBetweenSamples(const Int_t&);    // EcnaStexStin of (StexStin)
  TVectorD ReadAverageSigmaOfCorrelationsBetweenSamples(const Int_t&); // EcnaStexStin of (StexStin)

  //------------------------------------------------------------------------------------------------
  TMatrixD ReadCovariancesBetweenSamples(const Int_t&, const Int_t&, const Int_t&);  // (samp,samp) of (StexStin,Xtal)
  TMatrixD ReadCorrelationsBetweenSamples(const Int_t&, const Int_t&, const Int_t&); // (samp,samp) of (StexStin,Xtal)

  //-----------------------------------------------------------------------------------------------
  TMatrixD ReadLowFrequencyCovariancesBetweenChannels(const Int_t&, const Int_t&, const Int_t&);
  TMatrixD ReadHighFrequencyCovariancesBetweenChannels(const Int_t&, const Int_t&, const Int_t&);
                                       // (Xtal in Stin X, Xtal in Stin Y) of (Stin_X, Stin_Y)

  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const Int_t&);
  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels(const Int_t&, const Int_t&, const Int_t&);
                                       // (Xtal in Stin X, Xtal in Stin Y) of (Stin_X, Stin_Y)

  //------------------- (BIG MATRIX 1700x1700 for barrel, 5000x5000? for endcap) ------------------
  TMatrixD ReadLowFrequencyCovariancesBetweenChannels(const Int_t&);  // (Xtal in Stin X, Xtal in Stin Y) for all Stins
  TMatrixD ReadHighFrequencyCovariancesBetweenChannels(const Int_t&);
  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels(const Int_t&); // (Xtal in Stin X, Xtal in Stin Y) for all Stins
  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels(const Int_t&);

  //-----------------------------------------------------------------------------------------------
  TMatrixD ReadLowFrequencyMeanCorrelationsBetweenStins(const Int_t&);  // 1 of (Stin,Stin) 
  TMatrixD ReadHighFrequencyMeanCorrelationsBetweenStins(const Int_t&); // 1 of (Stin,Stin) 

  //------------------------------------------------------------------------------------------------
  TString  GetAnalysisName();
  Int_t    GetNbOfSamples();
  Int_t    GetRunNumber();
  Int_t    GetFirstReqEvtNumber();
  Int_t    GetLastReqEvtNumber();
  Int_t    GetReqNbOfEvts();
  Int_t    GetStexNumber();

  time_t   GetStartTime();
  time_t   GetStopTime();
  TString  GetStartDate();
  TString  GetStopDate();
  TString  GetRootFileName();
  TString  GetRootFileNameShort();

  TString  GetRunType();

  //-------------------------------------------------------------------------------  "technical" methods
  TVectorD ReadRelevantCorrelationsBetweenSamples(const Int_t&, const Int_t&, const Int_t&);
  // N(N-1)/2 of (StexStin, Xtal)

  Int_t    GetStexStinFromIndex(const Int_t&);      // (no read in the ROOT file)
  Int_t    GetStinIndex(const Int_t&);              // Stin index from Stin number (StexStin)

  TVectorD ReadStinNumbers(const Int_t&);
  TMatrixD ReadNumberOfEventsForSamples(const Int_t&, const Int_t&, const Int_t&); // (Xtal,sample) of (StexStin)

  Double_t*** ReadSampleAdcValuesSameFile(const Int_t&, const Int_t&, const Int_t&); 

  Int_t   GetNumberOfEvents(const Int_t&, const Int_t&);
  Int_t   GetNumberOfBinsSampleAsFunctionOfTime();

  TString GetTypeOfQuantity(const CnaResultTyp);

  TString GetTechReadCode(const TString&, const TString&);

  //------------------------------------------------------------------------------------------------
  //............... Flags Print Comments/Debug

  void  PrintNoComment();   // (default) Set flags to forbid the printing of all the comments
                            // except ERRORS
  void  PrintWarnings();    // Set flags to authorize printing of some warnings
  void  PrintComments();    // Set flags to authorize printing of infos and some comments
                            // concerning initialisations
  void  PrintAllComments(); // Set flags to authorize printing of all the comments

ClassDef(TEcnaRead,1) // Calculation of correlated noises from data
};  

#endif    //  CL_TEcnaRead_H


