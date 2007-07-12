#ifndef   CL_TCnaReadEB_H
#define   CL_TCnaReadEB_H

#include "TObject.h"
#include "TString.h"
#include "Riostream.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaHeaderEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaResultType.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRootFile.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"

//-------------------------------- TCnaReadEB.h ----------------------------
// 
//   Creation: 31 May    2005
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//-----------------------------------------------------------------------

class TCnaReadEB: public TObject {
  
 private:

  //............ attributes

  // static  const  Int_t        fgMaxCar    = 512;          <== DANGEROUS ! 

  Int_t       fgMaxCar;   // Max nb of caracters for char*

  Int_t       fCnew;      // flags for dynamical allocation
  Int_t       fCdelete;       

  TString     fTTBELL;

  TCnaHeaderEB *fFileHeader;    // header for result file
  Int_t        fCodeHeader;

  Bool_t      fOpenRootFile;  // flag open ROOT file (open = kTRUE, close = kFALSE)
  Int_t       fCodeRoot;
  Int_t       fCodeCorresp;

  Int_t       fReadyToReadRootFile;
  Int_t       fLookAtRootFile;

  Int_t       fSpecialSMTowerNotIndexed;

  Int_t*      fT1d_SMtowFromIndex; // 1D array[tower] tower Number as a function of the index tower
  Int_t*      fTagTowerNumbers;
  Int_t       fMemoTowerNumbers;

  Double_t**  fjustap_2d_ev;
  Double_t*   fjustap_1d_ev;

  Double_t**  fjustap_2d_var;
  Double_t*   fjustap_1d_var;

  Double_t**  fjustap_2d_cc;
  Double_t*   fjustap_1d_cc;

  Double_t**  fjustap_2d_ss;
  Double_t*   fjustap_1d_ss;

  ofstream    fFcout_f;
  ifstream    fFcin_f;
  
  Int_t       fDim_name;

  TString     fRootFileName;
  TString     fRootFileNameShort;

  TString     fAsciiFileName;
  TString     fAsciiFileNameShort;

  TString     fPathAscii;
  TString     fPathRoot;

  TString     fKeyFileNameForCnaPaths;

  TString     fKeyWorkingDirPath;
  TString     fKeyRootFilePath;

  Int_t       fSectChanSizeX, fSectChanSizeY,
              fSectSampSizeX, fSectSampSizeY;

  Int_t       fNbChanByLine;  // Nb channels by line (for ASCII results file)
  Int_t       fNbSampByLine;  // Nb samples by line  (for ASCII results file)
  Int_t       fUserSamp;      // Current sample  number (for ASCII results file)    
  Int_t       fUserChan;      // Current channel number (for ASCII results file) 

  Int_t       fFlagPrint;
  Int_t       fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //............... flag data exists

  Bool_t      fDataExist;

  //.......................................... private methods

  void        fCopy(const TCnaReadEB&);
  void        fMakeResultsFileName(const Int_t&);

 public: 

  //................. constructors
  
  TCnaReadEB();                            //  constructor without argument (FOR USER'S DECLARATION)
  TCnaReadEB(const TCnaReadEB&);              //  copy constructor

  //.................... C++ methods

  TCnaReadEB&  operator=(const TCnaReadEB&);  //  overloading of the operator=

  //................. destructor
  
  virtual ~TCnaReadEB();
  
  //...................................................... methods that will (should) be private

  void         Init();

  Bool_t       OpenRootFile(Text_t *, TString);
  Bool_t       CloseRootFile(Text_t *);
  Bool_t       ReadRootFileHeader(const Int_t&);
  Bool_t       DataExist();    // if data is present: kTRUE , if not: kFALSE

  //............................................................... Genuine public user's methods
                            
  void         GetReadyToReadRootFile(     TString, const Int_t&, const Int_t&,
				      const Int_t&, const Int_t&, TString);

  time_t       GetStartTime();
  time_t       GetStopTime();
  TString      GetStartDate();
  TString      GetStopDate();
  void         DefineResultsRootFilePath(TString);
  TString      GetRootFileNameShort();

  //.................. Recovering methods from ROOT file ( associated to GetReadyToReadRootFile(...) )

  Bool_t   LookAtRootFile();

  TVectorD ReadTowerNumbers();
  Int_t    GetSMTowFromIndex(const Int_t&);   // (no read in the ROOT file)
  Int_t    GetTowerIndex(const Int_t&);                     // Tower index from tower number (SMtower)
  Int_t    GetSMEcha(const Int_t&, const Int_t&); // CNA-Channel number from tower and crystal numbers 

  TMatrixD ReadNumbersOfFoundEventsForSamples(const Int_t&);                    // (crystal,sample) of (SMtower)
  TVectorD ReadSampleAsFunctionOfTime(const Int_t&, const Int_t&, const Int_t&); // (bins) of (SMtow,crys,sampl) 
  TVectorD ReadExpectationValuesOfSamples(const Int_t&,  const Int_t&);     // (sample) of (SMtower,crystal)
  TVectorD ReadVariancesOfSamples(const Int_t&, const Int_t&);              // (sample) of (SMtower,crystal) 
  TVectorD ReadSigmasOfSamples(const Int_t&, const Int_t&);                 // (sample) of (SMtower,crystal)
  TVectorD ReadEventDistribution(const Int_t&, const Int_t&, const Int_t&);  // (bins) of (SMtower,crystal,sample)
  Double_t ReadEventDistributionXmin(const Int_t&, const Int_t&, const Int_t&); // (1) of (SMtower,crystal,sample)
  Double_t ReadEventDistributionXmax(const Int_t&, const Int_t&, const Int_t&); // (1) of (SMtower,crystal,sample)

  TMatrixD ReadCovariancesBetweenCrystalsMeanOverSamples(const Int_t&, const Int_t&);
                                       // (crystal in tower X, crystal in tower Y) of (SMtower_X, SMtower_Y) 
  TMatrixD ReadCorrelationsBetweenCrystalsMeanOverSamples(const Int_t&, const Int_t&);
                                       // (crystal in tower X, crystal in tower Y) of (SMtower_X, SMtower_Y)

  TMatrixD ReadCovariancesBetweenCrystalsMeanOverSamples();  // (Xtal in tower X, Xtal in tower Y) for all towers 
  TMatrixD ReadCorrelationsBetweenCrystalsMeanOverSamples(); // (Xtal in tower X, Xtal in tower Y) for all towers

  TMatrixD ReadCovariancesBetweenTowersMeanOverSamplesAndChannels();  // 1 of (tower,tower) 
  TMatrixD ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels(); // 1 of (tower,tower) 

  TMatrixD ReadCovariancesBetweenSamples(const Int_t&, const Int_t&);   // (sample,sample) of (SMtower, crystal)
  TMatrixD ReadCorrelationsBetweenSamples(const Int_t&, const Int_t&);  // (sample,sample) of (SMtower, crystal)
  TVectorD ReadRelevantCorrelationsBetweenSamples(const Int_t&, const Int_t&); // N(N-1)/2 of (SMtower, crystal)

  TVectorD ReadExpectationValuesOfExpectationValuesOfSamples();  // 1 of (SMEcha)
  TVectorD ReadExpectationValuesOfSigmasOfSamples();             // 1 of (SMEcha)
  TVectorD ReadExpectationValuesOfCorrelationsBetweenSamples();  // 1 of (SMEcha)

  TVectorD ReadSigmasOfExpectationValuesOfSamples();             // 1  of (SMEcha)
  TVectorD ReadSigmasOfSigmasOfSamples();                        // 1  of (SMEcha)
  TVectorD ReadSigmasOfCorrelationsBetweenSamples();             // 1  of (SMEcha)

  TMatrixD ReadCorrectionsToSamplesFromCovss(const Int_t&);                 // (crystal, sample) of (SMtower)
  TVectorD ReadCorrectionsToSamplesFromCovss(const Int_t&, const Int_t&);   // (sample) of (SMtower, crystal)
  TMatrixD ReadCorrectionFactorsToCovss(const Int_t&, const Int_t&);
  TMatrixD ReadCorrectionFactorsToCorss(const Int_t&, const Int_t&);

  Int_t MaxTowEtaInSM();
  Int_t MaxTowPhiInSM();
  Int_t MaxTowInSM();

  Int_t MaxCrysEtaInTow();           // Tower size X in terms of crystal
  Int_t MaxCrysPhiInTow();           // Tower size Y in terms of crystal
  Int_t MaxCrysInTow();
  Int_t MaxCrysInSM();

  Int_t MaxSampADC();

  TString GetAnalysisName();
  Int_t   GetFirstTakenEventNumber();
  Int_t   GetNumberOfTakenEvents();

  Int_t GetNumberOfBinsEventDistributions();
  Int_t GetNumberOfBinsSampleAsFunctionOfTime();

  TString GetTypeOfQuantity(const CnaResultTyp);

  //............... Flags Print Comments/Debug

  void  PrintNoComment();   // (default) Set flags to forbid the printing of all the comments
                            // except ERRORS
  void  PrintWarnings();    // Set flags to authorize printing of some warnings
  void  PrintComments();    // Set flags to authorize printing of infos and some comments
                            // concerning initialisations
  void  PrintAllComments(); // Set flags to authorize printing of all the comments

ClassDef(TCnaReadEB,1) // Calculation of correlated noises from data
};  

#endif    //  CL_TCnaReadEB_H


