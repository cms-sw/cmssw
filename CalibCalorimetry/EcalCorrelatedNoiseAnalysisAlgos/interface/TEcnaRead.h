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

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaHeader.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParCout.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaResultType.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRootFile.h"

//-------------------------------- TEcnaRead.h ----------------------------
// 
//   Creation: 31 May    2005
//
//   Update:   see TEcnaRead.cc
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//-----------------------------------------------------------------------

class TEcnaRead: public TObject {
  
 private:

  //............ attributes

  // static  const  Int_t        fgMaxCar    = 512;          <== DANGEROUS ! 

  Int_t fgMaxCar;   // Max nb of caracters for char*

  Int_t fCnew;      // flags for dynamical allocation
  Int_t fCdelete;       

  TString     fTTBELL;

  TEcnaParCout   *fCnaParCout;  // for comments or error messages
  TEcnaParPaths  *fCnaParPaths; // for NbBinsADC 

  TEcnaHeader    *fFileHeader;    // header for result file
  TEcnaWrite     *fCnaWrite;      // for writing in ascii files
  TEcnaParEcal   *fEcal;          // for Ecal parameters
  TEcnaNumbering *fEcalNumbering; // for Ecal numbering

  TString fFlagSubDet;
  TString fStexName, fStinName;

  Bool_t fOpenRootFile;  // flag open ROOT file (open = kTRUE, close = kFALSE)

  Int_t fReadyToReadRootFile;
  Int_t fLookAtRootFile;

  Int_t* fT1d_StexStinFromIndex; // 1D array[Stin] Stin Number as a function of the index Stin
  Int_t* fTagStinNumbers;
  Int_t  fMemoStinNumbers;

  TString fPathRoot;

  Int_t fSectChanSizeX, fSectChanSizeY,
        fSectSampSizeX, fSectSampSizeY;

  Int_t fNbChanByLine;  // Nb channels by line (for ASCII results file)
  Int_t fNbSampByLine;  // Nb samples by line  (for ASCII results file)
  Int_t fUserSamp;      // Current sample  number (for ASCII results file)    
  Int_t fUserChan;      // Current channel number (for ASCII results file) 

  Int_t fFlagPrint;
  Int_t fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //............... flag data exists

  Bool_t fDataExist;

  //................ 3d array for sample ADC value fast transfert
  Double_t*** fT3d_distribs;   // 3D array[channel][sample][event] ADC values distibutions
  Double_t**  fT3d2_distribs; 
  Double_t*   fT3d1_distribs;

  //.......................................... private methods

  void        fCopy(const TEcnaRead&);

 public: 

  //................. constructors
  
  TEcnaRead();                  //  constructor without argument (FOR USER'S DECLARATION)
  //  constructor with argument (FOR USER'S DECLARATION):
  TEcnaRead(const TString,     const TEcnaParPaths*,   const TEcnaParCout*,
	   const TEcnaHeader*, const TEcnaNumbering*, const TEcnaWrite*);  

  TEcnaRead(const TEcnaRead&);   //  copy constructor

  //.................... C++ methods

  TEcnaRead&  operator=(const TEcnaRead&);  //  overloading of the operator=

  //................. destructor
  
  virtual ~TEcnaRead();
  
  //...................................................... methods that will (should) be private

  void     Init();
  void     SetEcalSubDetector(const TString, const TEcnaNumbering*, const TEcnaWrite*);

  void     Anew(const TString);
  void     Adelete(const TString);

  Bool_t   OpenRootFile(const Text_t *, TString);
  Bool_t   CloseRootFile(const Text_t *);
  Bool_t   ReadRootFileHeader(const Int_t&);
  Bool_t   DataExist();    // if data is present: kTRUE , if not: kFALSE

  //............................................................... Genuine public user's methods
                            
  void     GetReadyToReadRootFile(TString,
				  const Int_t&, const Int_t&, const Int_t&,
				  const Int_t&, const Int_t&, const Int_t&, TString);
  time_t   GetStartTime();
  time_t   GetStopTime();
  TString  GetStartDate();
  TString  GetStopDate();
  TString  GetRootFileName();
  TString  GetRootFileNameShort();

  TString  GetRunType();

  //.................. Recovering methods from ROOT file ( associated to GetReadyToReadRootFile(...) )

  Bool_t   LookAtRootFile();

  Int_t    GetStexStinFromIndex(const Int_t&);      // (no read in the ROOT file)
  Int_t    GetStinIndex(const Int_t&);              // Stin index from Stin number (StexStin)

  TVectorD ReadStinNumbers(const Int_t&);
  TMatrixD ReadNumberOfEventsForSamples(const Int_t&, const Int_t&, const Int_t&); // (Xtal,sample) of (StexStin)

  TVectorD ReadSampleValues(const Int_t&, const Int_t&, const Int_t&);  //(nb evts in burst) of (StexEcha,samp)
  Double_t*** ReadSampleValuesSameFile(const Int_t&, const Int_t&, const Int_t&); 

  TVectorD ReadSampleMeans(const Int_t&,  const Int_t&, const Int_t&);    // (sample) of (StexStin,Xtal)
  TVectorD ReadSampleSigmas(const Int_t&, const Int_t&, const Int_t&);    // (sample) of (StexStin,Xtal) 
  //TVectorD ReadSigmasOfSamples(const Int_t&, const Int_t&, const Int_t&); // (sample) of (StexStin,Xtal)

  //------------------------------------------------------------------------------------------------
  TMatrixD ReadCovariancesBetweenSamples(const Int_t&, const Int_t&, const Int_t&);  // (samp,samp) of (StexStin,Xtal)
  TMatrixD ReadCorrelationsBetweenSamples(const Int_t&, const Int_t&, const Int_t&); // (samp,samp) of (StexStin,Xtal)
  TVectorD ReadRelevantCorrelationsBetweenSamples(const Int_t&, const Int_t&, const Int_t&);
  // N(N-1)/2 of (StexStin, Xtal)

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

  TVectorD ReadNumberOfEvents(const Int_t&);                    // 1 of (StexEcha)
  TVectorD ReadPedestals(const Int_t&);                         // 1 of (StexEcha)
  TVectorD ReadTotalNoise(const Int_t&);                        // 1 of (StexEcha)
  TVectorD ReadLowFrequencyNoise(const Int_t&);                 // 1 of (StexEcha)
  TVectorD ReadHighFrequencyNoise(const Int_t&);                // 1 of (StexEcha)
  TVectorD ReadMeanOfCorrelationsBetweenSamples(const Int_t&);  // 1 of (StexEcha)
  TVectorD ReadSigmaOfCorrelationsBetweenSamples(const Int_t&); // 1 of (StexEcha)

  //------------------------------------------------------------------------------------------------
  TVectorD ReadAveragedNumberOfEvents(const Int_t&);                    // 1 of (StexStin)
  TVectorD ReadAveragedPedestals(const Int_t&);                         // 1 of (StexStin)
  TVectorD ReadAveragedTotalNoise(const Int_t&);                        // 1 of (StexStin)
  TVectorD ReadAveragedLowFrequencyNoise(const Int_t&);                 // 1 of (StexStin)
  TVectorD ReadAveragedHighFrequencyNoise(const Int_t&);                // 1 of (StexStin)
  TVectorD ReadAveragedMeanOfCorrelationsBetweenSamples(const Int_t&);  // 1 of (StexStin)
  TVectorD ReadAveragedSigmaOfCorrelationsBetweenSamples(const Int_t&); // 1 of (StexStin)

  //------------------------------------------------------------------------------------------------
  TString GetAnalysisName();
  Int_t   GetFirstReqEvtNumber();
  Int_t   GetReqNbOfEvts();

  Int_t GetNumberOfBinsSampleAsFunctionOfTime();

  TString GetTypeOfQuantity(const CnaResultTyp);

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


