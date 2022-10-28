//----------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 30/01/2014

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaWrite.h"

//--------------------------------------
//  TEcnaWrite.cc
//  Class creation: 19 May 2005
//  Documentation: see TEcnaWrite.h
//--------------------------------------

ClassImp(TEcnaWrite);
//______________________________________________________________________________
//

TEcnaWrite::~TEcnaWrite() {
  //destructor

  //if (fEcal          != 0){delete fEcal;          fCdelete++;}
  //if (fEcalNumbering != 0){delete fEcalNumbering; fCdelete++;}
  //if (fCnaParPaths   != 0){delete fCnaParPaths;   fCdelete++;}
  //if (fCnaParCout    != 0){delete fCnaParCout;    fCdelete++;}

  // std::cout << "[Info Management] CLASS: TEcnaWrite.         DESTROY OBJECT: this = " << this << std::endl;
}

//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaWrite::TEcnaWrite() {
  // std::cout << "[Info Management] CLASS: TEcnaWrite.         CREATE OBJECT: this = " << this << std::endl;

  Init();
}

TEcnaWrite::TEcnaWrite(TEcnaObject* pObjectManager, const TString& SubDet) {
  // std::cout << "[Info Management] CLASS: TEcnaWrite.         CREATE OBJECT: this = " << this << std::endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaWrite", i_this);

  //............................ fCnaParCout
  fCnaParCout = nullptr;
  Long_t iCnaParCout = pObjectManager->GetPointerValue("TEcnaParCout");
  if (iCnaParCout == 0) {
    fCnaParCout = new TEcnaParCout(pObjectManager); /*fCnew++*/
  } else {
    fCnaParCout = (TEcnaParCout*)iCnaParCout;
  }
  fCodePrintAllComments = fCnaParCout->GetCodePrint("AllComments");

  //............................ fCnaParPaths
  fCnaParPaths = nullptr;
  Long_t iCnaParPaths = pObjectManager->GetPointerValue("TEcnaParPaths");
  if (iCnaParPaths == 0) {
    fCnaParPaths = new TEcnaParPaths(pObjectManager); /*fCnew++*/
  } else {
    fCnaParPaths = (TEcnaParPaths*)iCnaParPaths;
  }

  //fCfgResultsRootFilePath    = fCnaParPaths->ResultsRootFilePath();
  //fCfgHistoryRunListFilePath = fCnaParPaths->HistoryRunListFilePath();

  fCnaParPaths->GetPathForResultsRootFiles();
  fCnaParPaths->GetPathForResultsAsciiFiles();

  //............................ fEcal  => should be changed in fParEcal
  fEcal = nullptr;
  Long_t iParEcal = pObjectManager->GetPointerValue("TEcnaParEcal");
  if (iParEcal == 0) {
    fEcal = new TEcnaParEcal(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fEcal = (TEcnaParEcal*)iParEcal;
  }

  //............................ fEcalNumbering
  fEcalNumbering = nullptr;
  Long_t iEcalNumbering = pObjectManager->GetPointerValue("TEcnaNumbering");
  if (iEcalNumbering == 0) {
    fEcalNumbering = new TEcnaNumbering(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fEcalNumbering = (TEcnaNumbering*)iEcalNumbering;
  }

  SetEcalSubDetector(SubDet.Data());
}

TEcnaWrite::TEcnaWrite(const TString& SubDet,
                       TEcnaParPaths* pCnaParPaths,
                       TEcnaParCout* pCnaParCout,
                       TEcnaParEcal* pEcal,
                       TEcnaNumbering* pEcalNumbering) {
  // std::cout << "[Info Management] CLASS: TEcnaWrite.         CREATE OBJECT: this = " << this << std::endl;

  Init();

  //----------------------------- Object management
  fCnaParPaths = nullptr;
  if (pCnaParPaths == nullptr) {
    fCnaParPaths = new TEcnaParPaths(); /*fCnew++*/
    ;
  } else {
    fCnaParPaths = pCnaParPaths;
  }

  //................. Get paths from ECNA directory

  fCnaParPaths->GetPathForResultsRootFiles();
  fCnaParPaths->GetPathForResultsAsciiFiles();

  fCnaParCout = nullptr;
  if (pCnaParCout == nullptr) {
    fCnaParCout = new TEcnaParCout(); /*fCnew++*/
    ;
  } else {
    fCnaParCout = pCnaParCout;
  }
  fCodePrintAllComments = fCnaParCout->GetCodePrint("AllComments");

  fEcal = nullptr;
  if (pEcal == nullptr) {
    fEcal = new TEcnaParEcal(SubDet.Data()); /*fCnew++*/
    ;
  } else {
    fEcal = pEcal;
  }

  fEcalNumbering = nullptr;
  if (pEcalNumbering == nullptr) {
    fEcalNumbering = new TEcnaNumbering(SubDet.Data(), fEcal); /*fCnew++*/
    ;
  } else {
    fEcalNumbering = pEcalNumbering;
  }

  SetEcalSubDetector(SubDet.Data(), fEcal, fEcalNumbering);
}

void TEcnaWrite::Init() {
  //----------------------------- Parameters values
  fTTBELL = '\007';

  fgMaxCar = (Int_t)512;  // max number of characters in TStrings
  fCodeHeaderAscii = 0;
  fCodeRoot = 1;

  //------------------------------------------- Codes
  //................. presently used codes
  fCodeNbOfEvts = 101;
  fCodePed = 102;
  fCodeTno = 103;
  fCodeLfn = 104;
  fCodeHfn = 105;
  fCodeMeanCorss = 106;
  fCodeSigCorss = 107;

  fCodeCovCss = 201;
  fCodeCorCss = 202;

  //................. not yet used codes
  //fCodeAdcEvt      =  3;
  //fCodeMSp         =  4;
  //fCodeSSp         =  5;

  //fCodeAvPed       = 17;
  //fCodeAvTno       =  6;
  //fCodeAvMeanCorss = 18;
  //fCodeAvSigCorss  = 19;

  //fCodeLfCov       = 11;
  //fCodeLfCor       = 12;
  //fCodeHfCov       =  9;
  //fCodeHfCor       = 10;

  //fCodeLFccMoStins = 13;
  //fCodeHFccMoStins = 14;

  //----------------------------------------

  fUserSamp = 0;
  fStexStinUser = 0;
  fStinEchaUser = 0;

  fjustap_2d_ev = nullptr;
  fjustap_1d_ev = nullptr;

  fjustap_2d_var = nullptr;
  fjustap_1d_var = nullptr;

  fjustap_2d_cc = nullptr;
  fjustap_1d_cc = nullptr;

  fjustap_2d_ss = nullptr;
  fjustap_1d_ss = nullptr;
}
// end of Init()

//===================================================================
//
//                   Methods
//
//===================================================================

void TEcnaWrite::SetEcalSubDetector(const TString& SubDet) {
  // Set Subdetector (EB or EE)

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();

  //........................................................................
  //
  //             (for ASCII files writing methods only) ...
  //
  //                DEFINITION OF THE SECTOR SIZES
  //       FOR THE CORRELATION AND COVARIANCE MATRICES DISPLAY
  //
  //            MUST BE A DIVISOR OF THE TOTAL NUMBER.
  //            ======================================
  //
  //     Examples:
  //
  //      (1)    25 channels => size = 25 or 5 (divisors of 25)
  //
  //             25 => matrix = 1 x 1 sector  of size (25 x 25)
  //                          = (1 x 1) x (25 x 25) = 1 x 625 = 625
  //              5 => matrix = 5 x 5 sectors of size (5 x 5)
  //                          = (5 x 5) x ( 5 x  5) = 25 x 25 = 625
  //
  //      (2)    10 samples  => size = 10, 5 or 2 (divisors of 10)
  //
  //             10 => matrix = 1 X 1 sectors of size (10 x 10)
  //                          = (1 x 1) x (10 x 10) =  1 x 100 = 100
  //              5 => matrix = 2 x 2 sectors of size (5 x 5)
  //                          = (2 x 2) x ( 5 x  5) =  4 x  25 = 100
  //              2 => matrix = 5 x 5 sectors of size (2 x 2)
  //                          = (5 x 5) x ( 2 x  2) = 25 x  4  = 100
  //
  //........................................................................
  fSectChanSizeX = fEcal->MaxCrysHocoInStin();
  fSectChanSizeY = fEcal->MaxCrysVecoInStin();
  fSectSampSizeX = fEcal->MaxSampADC();
  fSectSampSizeY = fEcal->MaxSampADC();

  //........................................................................
  //
  //                DEFINITION OF THE NUMBER OF VALUES BY LINE
  //                for the Expectation Values, Variances and.
  //                Event distributions by (channel,sample)
  //
  //               MUST BE A DIVISOR OF THE TOTAL NUMBER.
  //               ======================================
  //
  //     Examples:
  //                1) For expectation values and variances:
  //
  //                25 channels => size = 5
  //                => sample sector = 5 lines of 5 values
  //                                 = 5 x 5 = 25 values
  //
  //                10 samples  => size = 10
  //                => channel sector = 1 lines of 10 values
  //                                  = 1 x 10 = 10 values
  //
  //                2) For event distributions:
  //
  //                100 bins  => size = 10
  //                => sample sector = 10 lines of 10 values
  //                                 = 10 x 10 = 100 values
  //
  //........................................................................
  fNbChanByLine = fEcal->MaxCrysHocoInStin();
  fNbSampByLine = fEcal->MaxSampADC();
}  //---------- (end of SetEcalSubDetector) ------------------

void TEcnaWrite::SetEcalSubDetector(const TString& SubDet, TEcnaParEcal* pEcal, TEcnaNumbering* pEcalNumbering) {
  // Set Subdetector (EB or EE)

  fEcal = nullptr;
  if (pEcal == nullptr) {
    fEcal = new TEcnaParEcal(SubDet.Data());
    fCnew++;
  } else {
    fEcal = pEcal;
  }

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();

  fEcalNumbering = nullptr;
  if (pEcalNumbering == nullptr) {
    fEcalNumbering = new TEcnaNumbering(SubDet.Data(), fEcal);
    fCnew++;
  } else {
    fEcalNumbering = pEcalNumbering;
  }

  //........................................................................
  //
  //             (for ASCII files writing methods only) ...
  //
  //                DEFINITION OF THE SECTOR SIZES
  //       FOR THE CORRELATION AND COVARIANCE MATRICES DISPLAY
  //
  //            MUST BE A DIVISOR OF THE TOTAL NUMBER.
  //            ======================================
  //
  //     Examples:
  //
  //      (1)    25 channels => size = 25 or 5 (divisors of 25)
  //
  //             25 => matrix = 1 x 1 sector  of size (25 x 25)
  //                          = (1 x 1) x (25 x 25) = 1 x 625 = 625
  //              5 => matrix = 5 x 5 sectors of size (5 x 5)
  //                          = (5 x 5) x ( 5 x  5) = 25 x 25 = 625
  //
  //      (2)    10 samples  => size = 10, 5 or 2 (divisors of 10)
  //
  //             10 => matrix = 1 X 1 sectors of size (10 x 10)
  //                          = (1 x 1) x (10 x 10) =  1 x 100 = 100
  //              5 => matrix = 2 x 2 sectors of size (5 x 5)
  //                          = (2 x 2) x ( 5 x  5) =  4 x  25 = 100
  //              2 => matrix = 5 x 5 sectors of size (2 x 2)
  //                          = (5 x 5) x ( 2 x  2) = 25 x  4  = 100
  //
  //........................................................................
  fSectChanSizeX = fEcal->MaxCrysHocoInStin();
  fSectChanSizeY = fEcal->MaxCrysVecoInStin();
  fSectSampSizeX = fEcal->MaxSampADC();
  fSectSampSizeY = fEcal->MaxSampADC();

  //........................................................................
  //
  //                DEFINITION OF THE NUMBER OF VALUES BY LINE
  //                for the Expectation Values, Variances and.
  //                Event distributions by (channel,sample)
  //
  //               MUST BE A DIVISOR OF THE TOTAL NUMBER.
  //               ======================================
  //
  //     Examples:
  //                1) For expectation values and variances:
  //
  //                25 channels => size = 5
  //                => sample sector = 5 lines of 5 values
  //                                 = 5 x 5 = 25 values
  //
  //                10 samples  => size = 10
  //                => channel sector = 1 lines of 10 values
  //                                  = 1 x 10 = 10 values
  //
  //                2) For event distributions:
  //
  //                100 bins  => size = 10
  //                => sample sector = 10 lines of 10 values
  //                                 = 10 x 10 = 100 values
  //
  //........................................................................
  fNbChanByLine = fEcal->MaxCrysHocoInStin();
  fNbSampByLine = fEcal->MaxSampADC();
}  //---------- (end of SetEcalSubDetector) ------------------

//-------------------------------------------------------------------------
//
//    Get methods for different run or file parameters
//
//    W A R N I N G :  some of these methods are called by external code
//
//                D O N ' T    S U P P R E S S !
//
//
//     TString  fRootFileNameShort
//     TString  fAnaType       = typ_ana
//     Int_t    fRunNumber     = run_number
//     Int_t    fFirstReqEvtNumber = nfirst
//     Int_t    fReqNbOfEvts = nevents
//     Int_t    fStexNumber    = super_module
//
//-------------------------------------------------------------------------
const TString& TEcnaWrite::GetAsciiFileName() const { return fAsciiFileName; }
const TString& TEcnaWrite::GetRootFileName() const { return fRootFileName; }
const TString& TEcnaWrite::GetRootFileNameShort() const { return fRootFileNameShort; }
const TString& TEcnaWrite::GetAnalysisName() const { return fAnaType; }
Int_t TEcnaWrite::GetNbOfSamples() { return fNbOfSamples; }
Int_t TEcnaWrite::GetRunNumber() { return fRunNumber; }
Int_t TEcnaWrite::GetFirstReqEvtNumber() { return fFirstReqEvtNumber; }
Int_t TEcnaWrite::GetReqNbOfEvts() { return fReqNbOfEvts; }
Int_t TEcnaWrite::GetStexNumber() { return fStexNumber; }

//---------------------------------------------------------------------------------------------------
//
//  NumberOfEventsAnalysis(...) : Analyses the number of events found in data
//                                channel by channel OR sample by sample
//                                an output ERROR message is delivered if differences are detected
//
//            channel by channel: called by TEcnaRead object (3 arguments)
//            sample  by sample : called by TEcnaRun  object (4 arguments)
//
//    This method must be a TEcnaWrite method since it can be called by objects
//    of class TEcnaRead OR TEcnaRun
//
//---------------------------------------------------------------------------------------------------
//.............................................................................................
Int_t TEcnaWrite::NumberOfEventsAnalysis(Int_t* ArrayNbOfEvts,
                                         const Int_t& MaxArray,
                                         const Int_t& NbOfReqEvts,
                                         const Int_t& StexNumber) {
  //  CHECK THE NUMBER OF FOUND EVENTS, return rNumberOfEvents (NumberOfEvents())
  //  (number used to compute the average values over the events)
  //
  //  1D array: called by TEcnaRead object

  Int_t rNumberOfEvents = 0;
  Int_t PresentNumber = 0;
  Int_t EmptyChannel = 0;
  Int_t DifferentMinusValue = 0;
  Int_t DifferentPlusValue = 0;

  //........................................................ i_SSoSE = StexStin or StinEcha
  for (Int_t i_SSoSE = 0; i_SSoSE < MaxArray; i_SSoSE++) {
    Int_t NbOfEvts = ArrayNbOfEvts[i_SSoSE];

    if (NbOfEvts > 0) {
      if (PresentNumber == 0)  // first channel
      {
        PresentNumber = NbOfEvts;
      } else  // current channel
      {
        if (NbOfEvts > PresentNumber)  // warning
        {
          PresentNumber = NbOfEvts;
          DifferentPlusValue++;
        }
        if (NbOfEvts < PresentNumber)  // warning
        {
          DifferentMinusValue++;
        }
      }
    } else {
      EmptyChannel++;
    }
  }

  rNumberOfEvents = PresentNumber;

  if (EmptyChannel > 0) {
    if (MaxArray == fEcal->MaxCrysInSM())  // (EB)
    {
      std::cout << "!TEcnaWrite::NumberOfEventsAnalysis()> *** WARNING *** " << EmptyChannel
                << " empty channels detected in SM " << StexNumber << " (EB "
                << fEcalNumbering->PlusMinusSMNumber(StexNumber) << ")" << std::endl;
    }
    if (MaxArray == fEcal->MaxCrysEcnaInDee())  // (EE)
    {
      EmptyChannel -= fEcal->EmptyChannelsInDeeMatrixIncompleteSCIncluded();
      if (EmptyChannel > 0) {
        std::cout << "!TEcnaWrite::NumberOfEventsAnalysis()> *** WARNING *** " << EmptyChannel
                  << " empty channels detected in Dee " << StexNumber << std::endl;
      }
    }
  }

  if (DifferentMinusValue > 0 || DifferentPlusValue > 0) {
    std::cout << "!TEcnaWrite::NumberOfEventsAnalysis()> " << std::endl;

    if (MaxArray == fEcal->MaxCrysInSM())  // (EB)
    {
      std::cout << "************** W A R N I N G  :  NUMBER OF EVENTS NOT CONSTANT FOR SM " << StexNumber << " (EB "
                << fEcalNumbering->PlusMinusSMNumber(StexNumber) << ")  *********************";
    }

    if (MaxArray == fEcal->MaxCrysEcnaInDee())  // (EE)
    {
      std::cout << "****************** W A R N I N G  :  NUMBER OF EVENTS NOT CONSTANT FOR Dee " << StexNumber
                << " **************************";
    }

    std::cout
        << std::endl
        << "  Result ROOT file: " << fRootFileName << std::endl
        << "  The number of events is not the same for all the non-empty channels." << std::endl
        << "  The maximum number (" << rNumberOfEvents << ") is considered as the number of events for calculations "
        << std::endl
        << "  of pedestals, noises and correlations." << std::endl
        << "  Number of channels with 0 < nb of evts < " << rNumberOfEvents << " : " << DifferentMinusValue
        << std::endl
        // << "  Number of channels with nb of evts > " << rNumberOfEvents << " = " << DifferentPlusValue  << std::endl
        << "  Number of empty channels : " << EmptyChannel << std::endl
        << "  Some values of pedestals, noises and correlations may be wrong for channels" << std::endl
        << "  with number of events different from " << rNumberOfEvents << "." << std::endl
        << "  Please, check the histogram 'Numbers of events'." << std::endl
        << "*******************************************************************************************************"
        << std::endl;
  } else {
    if (fFlagPrint == fCodePrintAllComments) {
      if (rNumberOfEvents < NbOfReqEvts) {
        std::cout << "*TEcnaWrite::NumberOfEventsAnalysis()> *** INFO *** Number of events found in data = "
                  << rNumberOfEvents << ": less than number of requested events ( = " << NbOfReqEvts << ")"
                  << std::endl;
      }
    }
  }
  return rNumberOfEvents;
}

//.............................................................................................
Int_t TEcnaWrite::NumberOfEventsAnalysis(Int_t** T2d_NbOfEvts,
                                         const Int_t& MaxCrysEcnaInStex,
                                         const Int_t& MaxNbOfSamples,
                                         const Int_t& NbOfReqEvts) {
  //  CHECK OF THE NUMBER OF FOUND EVENTS, return rNumberOfEvents (NumberOfEvents())
  //       (number used to compute the average values over the events)
  //
  //  2D array: called by TEcnaRun object

  Int_t rNumberOfEvents = 0;
  Int_t PresentNumber = 0;
  Int_t DifferentMinusValue = 0;
  Int_t DifferentPlusValue = 0;

  for (Int_t i0StexEcha = 0; i0StexEcha < MaxCrysEcnaInStex; i0StexEcha++) {
    for (Int_t i_samp = 0; i_samp < MaxNbOfSamples; i_samp++) {
      Int_t NbOfEvts = T2d_NbOfEvts[i0StexEcha][i_samp];

      if (NbOfEvts > 0) {
        if (PresentNumber == 0) {
          PresentNumber = NbOfEvts;
        } else {
          if (NbOfEvts > PresentNumber) {
            PresentNumber = NbOfEvts;
            DifferentPlusValue++;
          }
          if (NbOfEvts < PresentNumber) {
            DifferentMinusValue++;
          }
        }
      }
    }
  }
  //.............................................................  (NumberOfEvents())
  rNumberOfEvents = PresentNumber;

  if (DifferentMinusValue > 0 || DifferentPlusValue > 0) {
    std::cout
        << "!TEcnaWrite::NumberOfEventsAnalysis()> " << std::endl
        << "****************** W A R N I N G  :  NUMBER OF EVENTS NOT CONSTANT !  *********************************"
        << std::endl
        << "  Result ROOT file: " << fRootFileName << std::endl
        << "  The number of events is not the same for all the non-empty channels" << std::endl
        << "  The maximum number (" << rNumberOfEvents << ") is considered as the number of events " << std::endl
        << "  for calculations of pedestals, noises and correlations." << std::endl
        << "  Number of channels with 0 < nb of evts < " << rNumberOfEvents << " : " << DifferentMinusValue
        << std::endl
        // << "  Number of channels with nb of evts > " << rNumberOfEvents << " : " << DifferentPlusValue  << std::endl
        // << "  Number of empty channels : " << EmptyChannel << std::endl
        << "  Some values of pedestals, noises and correlations may be wrong for channels" << std::endl
        << "  with number of events different from " << rNumberOfEvents << "." << std::endl
        << "  Please, check the histogram 'Numbers of events'." << std::endl
        << "*******************************************************************************************************"
        << std::endl;
  } else {
    if (fFlagPrint == fCodePrintAllComments) {
      if (rNumberOfEvents < NbOfReqEvts) {
        std::cout << "*TEcnaWrite::NumberOfEventsAnalysis()> *** INFO *** Number of events found in data = "
                  << rNumberOfEvents << ": less than number of requested events ( = " << NbOfReqEvts << ")"
                  << std::endl;
      }
    }
  }
  return rNumberOfEvents;

}  //----- ( end of NumberOfEvents(...) ) ----------------

void TEcnaWrite::RegisterFileParameters(const TString& ArgAnaType,
                                        const Int_t& ArgNbOfSamples,
                                        const Int_t& ArgRunNumber,
                                        const Int_t& ArgFirstReqEvtNumber,
                                        const Int_t& ArgLastReqEvtNumber,
                                        const Int_t& ArgReqNbOfEvts,
                                        const Int_t& ArgStexNumber) {
  fAnaType = ArgAnaType;
  fNbOfSamples = ArgNbOfSamples;
  fRunNumber = ArgRunNumber;
  fFirstReqEvtNumber = ArgFirstReqEvtNumber;
  fLastReqEvtNumber = ArgLastReqEvtNumber;
  fReqNbOfEvts = ArgReqNbOfEvts;
  fStexNumber = ArgStexNumber;
}

void TEcnaWrite::RegisterFileParameters(const TString& ArgAnaType,
                                        const Int_t& ArgNbOfSamples,
                                        const Int_t& ArgRunNumber,
                                        const Int_t& ArgFirstReqEvtNumber,
                                        const Int_t& ArgLastReqEvtNumber,
                                        const Int_t& ArgReqNbOfEvts,
                                        const Int_t& ArgStexNumber,
                                        const TString& ArgStartDate,
                                        const TString& ArgStopDate,
                                        const time_t ArgStartTime,
                                        const time_t ArgStopTime) {
  fAnaType = ArgAnaType;
  fNbOfSamples = ArgNbOfSamples;
  fRunNumber = ArgRunNumber;
  fFirstReqEvtNumber = ArgFirstReqEvtNumber;
  fLastReqEvtNumber = ArgLastReqEvtNumber;
  fReqNbOfEvts = ArgReqNbOfEvts;
  fStexNumber = ArgStexNumber;
  fStartDate = ArgStartDate;

  fStopDate = ArgStopDate;
  fStartTime = ArgStartTime;
  fStopTime = ArgStopTime;
}
//=======================================================================
//
//                      FILE NAMES MANAGEMENT
//
//=======================================================================
//==============================================================================
//
//                     Results Filename Making  (private)
//
//==============================================================================
void TEcnaWrite::fMakeResultsFileName() { fMakeResultsFileName(fCodeRoot); }
void TEcnaWrite::fMakeResultsFileName(const Int_t& i_code) {
  //Results filename making  (private)

  //----------------------------------------------------------------------
  //
  //     Making of the name of the result file from the parameters set
  //     by call to RegisterFileParameters(...)
  //
  //     Put the names in the following class attributes:
  //
  //     fRootFileName,  fRootFileNameShort,
  //     fAsciiFileName, fAsciiFileNameShort
  //
  //     (Short means: without the directory path)
  //
  //     set indications (run number, type of quantity, ...)
  //     and add the extension ".ascii" or ".root"
  //
  //     ROOT:  only one ROOT file:  i_code = fCodeRoot.
  //                                          All the types of quantities
  //
  //     ASCII: several ASCII files: i_code = code for one type of quantity
  //            each i_code which is not equal to fCodeRoot is also implicitly
  //            a code "fCodeAscii" (this last attribute is not in the class)
  //
  //----------------------------------------------------------------------

  char* f_in = new char[fgMaxCar];
  fCnew++;
  char* f_in_short = new char[fgMaxCar];
  fCnew++;

  Int_t MaxCar = fgMaxCar;
  fStexName.Resize(MaxCar);
  fStexName = "SM or Dee?";

  MaxCar = fgMaxCar;
  fStinName.Resize(MaxCar);
  fStinName = "tower or SC?";

  if (fFlagSubDet == "EB") {
    fStexName = "SM";
    fStinName = "tower";
  }
  if (fFlagSubDet == "EE") {
    fStexName = "Dee";
    fStinName = "SC";
  }

  //  switch (i_code){

  //===================================  R O O T  =========================  (fMakeResultsFileName)
  TString sPointInterrog = "?";
  TString sDollarHome = "$HOME";

  if (i_code == fCodeRoot) {
    if (fCnaParPaths->ResultsRootFilePath().Data() == sPointInterrog.Data()) {
      std::cout << "!TEcnaWrite::fMakeResultsFileName>  * * * W A R N I N G * * * " << std::endl
                << std::endl
                << "    Path for results .root file not defined. Default option will be used here:" << std::endl
                << "    your results files will be written in your HOME directory." << std::endl
                << std::endl
                << "    In order to write the .root results file in a specific directory," << std::endl
                << "    you have to create a file named path_results_root in a subdirectory named ECNA" << std::endl
                << "    previously created in your home directory." << std::endl
                << "    This file must have only one line containing the path of the directory" << std::endl
                << "    where must be the .root result files." << std::endl
                << std::endl;

      TString home_path = gSystem->Getenv("HOME");
      fCnaParPaths->SetResultsRootFilePath(home_path.Data());
    }

    if (fCnaParPaths->BeginningOfResultsRootFilePath().Data() == sDollarHome.Data()) {
      fCnaParPaths->TruncateResultsRootFilePath(0, 5);
      const Text_t* t_file_nohome = (const Text_t*)fCnaParPaths->ResultsRootFilePath().Data();  //  /scratch0/cna/...

      TString home_path = gSystem->Getenv("HOME");
      fCnaParPaths->SetResultsRootFilePath(home_path.Data());  //  /afs/cern.ch/u/USER
      fCnaParPaths->AppendResultsRootFilePath(t_file_nohome);  //  /afs/cern.ch/u/USER/scratch0/cna/...
    }

    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d",
            fCnaParPaths->ResultsRootFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  //===================================  A S C I I  ====================  (fMakeResultsFileName)
  //fCnaParPaths->GetPathForResultsAsciiFiles();
  if (i_code != fCodeRoot) {
    if (i_code == fCodeHeaderAscii) {
      if (fCnaParPaths->ResultsAsciiFilePath().Data() == sPointInterrog.Data()) {
        std::cout << "!TEcnaWrite::fMakeResultsFileName>  * * * W A R N I N G * * * " << std::endl
                  << std::endl
                  << "    Path for results .ascii file not defined. Default option will be used here:" << std::endl
                  << "    your results files will be written in your HOME directory." << std::endl
                  << std::endl
                  << "    In order to write the .ascii results file in a specific directory," << std::endl
                  << "    you have to create a file named path_results_ascii in a subdirectory named ECNA" << std::endl
                  << "    previously created in your home directory." << std::endl
                  << "    This file must have only one line containing the path of the directory" << std::endl
                  << "    where must be the .ascii result files." << std::endl
                  << std::endl;

        TString home_path = gSystem->Getenv("HOME");
        fCnaParPaths->SetResultsAsciiFilePath(home_path.Data());
      }

      if (fCnaParPaths->BeginningOfResultsAsciiFilePath().Data() == sDollarHome.Data()) {
        fCnaParPaths->TruncateResultsAsciiFilePath(0, 5);
        const Text_t* t_file_nohome = (const Text_t*)fCnaParPaths->ResultsAsciiFilePath().Data();  // /scratch0/cna/...

        TString home_path = gSystem->Getenv("HOME");
        fCnaParPaths->SetResultsAsciiFilePath(home_path.Data());  //  /afs/cern.ch/u/USER
        fCnaParPaths->AppendResultsAsciiFilePath(t_file_nohome);  //  /afs/cern.ch/u/USER/scratch0/cna/...
      }
    }

    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_header",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_header",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  //--------------------------------------------------------------  (fMakeResultsFileName)
  if (i_code == fCodeNbOfEvts) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_NbOfEvents",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_NbOfEvents",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodePed) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_Pedestals",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_Pedestals",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeTno) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_TotalNoise",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_TotalNoise",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeLfn) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_LFNoise",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_LFNoise",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeHfn) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_HFNoise",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_HFNoise",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeMeanCorss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_MeanCorss",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_MeanCorss",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeSigCorss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_SigmaCorss",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_SigmaCorss",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeCovCss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_Covss_%s%d_Channel_%d",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinName.Data(),
            fStexStinUser,
            fStinEchaUser);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_Covss_%s%d_Channel_%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinName.Data(),
            fStexStinUser,
            fStinEchaUser);
  }

  if (i_code == fCodeCorCss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_Corss_%s%d_Channel_%d",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinName.Data(),
            fStexStinUser,
            fStinEchaUser);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_Corss_%s%d_Channel_%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinName.Data(),
            fStexStinUser,
            fStinEchaUser);
  }

  //------- (not used yet)
#define OCOD
#ifndef OCOD
  if (i_code == fCodeMSp) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_SampleMeans",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_SampleMeans",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeSSp) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_SampleSigmas",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_SampleSigmas",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeAvTno) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageTotalNoise_c%d",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageTotalNoise_c%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
  }

  if (i_code == fCodeLfCov) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_LF_cov",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_LF_cov",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeLfCor) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_LF_cor",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_LF_cor",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeAvPed) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_AveragePedestals",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_AveragePedestals",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber);
  }

  if (i_code == fCodeAvMeanCorss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageMeanCorss%d",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageMeanCorss%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
  }

  if (i_code == fCodeAvSigCorss) {
    sprintf(f_in,
            "%s/%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageSigmaCorss%d",
            fCnaParPaths->ResultsAsciiFilePath().Data(),
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
    sprintf(f_in_short,
            "%s_S1_%d_R%d_%d_%d_%d_%s%d_AverageSigmaCorss%d",
            fAnaType.Data(),
            fNbOfSamples,
            fRunNumber,
            fFirstReqEvtNumber,
            fLastReqEvtNumber,
            fReqNbOfEvts,
            fStexName.Data(),
            fStexNumber,
            fStinEchaUser);
  }
#endif  // OCOD

  //----------------------------------------------------------- (fMakeResultsFileName)

  // default:
  //    std::cout << "*TEcnaWrite::fMakeResultsFileName(const Int_t&  i_code)> "
  //	 << "wrong header code , i_code = " << i_code << std::endl;
  //  }

  //======================================= f_name

  char* f_name = new char[fgMaxCar];
  fCnew++;

  for (Int_t i = 0; i < fgMaxCar; i++) {
    f_name[i] = '\0';
  }

  Int_t ii = 0;
  for (Int_t i = 0; i < fgMaxCar; i++) {
    if (f_in[i] != '\0') {
      f_name[i] = f_in[i];
      ii++;
    } else {
      break;
    }  // va directement a if ( ii+5 < fgMaxCar ) puis... f_name[ii] = '.';
  }

  if (ii + 5 < fgMaxCar) {
    //.......... writing of the file extension (.root or .ascii)  (fMakeResultsFileName)

    //------------------------------------------- extension .ascii
    if (i_code != fCodeRoot || i_code == fCodeNbOfEvts) {
      f_name[ii] = '.';
      f_name[ii + 1] = 'a';
      f_name[ii + 2] = 's';
      f_name[ii + 3] = 'c';
      f_name[ii + 4] = 'i';
      f_name[ii + 5] = 'i';

      fAsciiFileName = f_name;
    }
    //------------------------------------------- extension .root
    if (i_code == fCodeRoot) {
      f_name[ii] = '.';
      f_name[ii + 1] = 'r';
      f_name[ii + 2] = 'o';
      f_name[ii + 3] = 'o';
      f_name[ii + 4] = 't';

      fRootFileName = f_name;
    }
  } else {
    std::cout << "*TEcnaWrite::fMakeResultsFileName(...)> Name too long (for f_name)."
              << " No room enough for the extension. (ii = " << ii << ")" << fTTBELL << std::endl;
  }

  //====================================== f_name_short  (fMakeResultsFileName)

  char* f_name_short = new char[fgMaxCar];
  fCnew++;

  for (Int_t i = 0; i < fgMaxCar; i++) {
    f_name_short[i] = '\0';
  }

  ii = 0;
  for (Int_t i = 0; i < fgMaxCar; i++) {
    if (f_in_short[i] != '\0') {
      f_name_short[i] = f_in_short[i];
      ii++;
    } else {
      break;
    }  // va directement a f_name_short[ii] = '.';
  }

  if (ii + 5 < fgMaxCar) {
    //.......... writing of the file extension (.root or .ascii)

    //-------------------------------------------extension .ascii
    if (i_code != fCodeRoot || i_code == fCodeNbOfEvts) {
      f_name_short[ii] = '.';
      f_name_short[ii + 1] = 'a';
      f_name_short[ii + 2] = 's';
      f_name_short[ii + 3] = 'c';
      f_name_short[ii + 4] = 'i';
      f_name_short[ii + 5] = 'i';

      fAsciiFileNameShort = f_name_short;
    }

    //-------------------------------------------- extension .root
    if (i_code == fCodeRoot) {
      f_name_short[ii] = '.';
      f_name_short[ii + 1] = 'r';
      f_name_short[ii + 2] = 'o';
      f_name_short[ii + 3] = 'o';
      f_name_short[ii + 4] = 't';

      fRootFileNameShort = f_name_short;
    }
  } else {
    std::cout << "*TEcnaWrite::fMakeResultsFileName(...)> Name too long (for f_name_short)."
              << " No room enough for the extension. (ii = " << ii << ")" << fTTBELL << std::endl;
  }
  delete[] f_name;
  f_name = nullptr;
  fCdelete++;
  delete[] f_name_short;
  f_name_short = nullptr;
  fCdelete++;

  delete[] f_in;
  f_in = nullptr;
  fCdelete++;
  delete[] f_in_short;
  f_in_short = nullptr;
  fCdelete++;

}  // end of fMakeResultsFileName

//==========================================================================================
//
//
//
//==========================================================================================

void TEcnaWrite::fAsciiFileWriteHeader(const Int_t& i_code) {
  //Ascii results file header writing  (private). Called by the WriteAscii...() methods

  //-----------------------------------------------
  //
  //     opening of the ASCII results file
  //     and writing of its header
  //
  //-----------------------------------------------

  if (fAsciiFileName.BeginsWith("$HOME")) {
    fAsciiFileName.Remove(0, 5);
    TString EndOfAsciiFileName = fAsciiFileName;
    const Text_t* t_file_nohome = (const Text_t*)EndOfAsciiFileName.Data();  //  const Text_t* -> EndOfAsciiFileName
    // ( /scratch0/cna/... )
    TString home_path = gSystem->Getenv("HOME");
    fAsciiFileName = home_path;            // fAsciiFileName = absolute HOME path ( /afs/cern.ch/u/USER )
    fAsciiFileName.Append(t_file_nohome);  // Append  (const Text_t* -> EndOfAsciiFileName) to fAsciiFileName
    // ( /afs/cern.ch/u/USER/scratch0/cna/... )
  }

  fFcout_f.open(fAsciiFileName.Data());

  fFcout_f << "*** File: " << fAsciiFileName << " *** " << std::endl << std::endl;
  fFcout_f << "*Analysis name                : " << fAnaType << std::endl;
  fFcout_f << "*First-Last samples           : 1 - " << fNbOfSamples << std::endl;
  fFcout_f << "*Run number                   : " << fRunNumber << std::endl;
  fFcout_f << "*First requested event number : " << fFirstReqEvtNumber << std::endl;
  fFcout_f << "*Last  requested event number : " << fLastReqEvtNumber << std::endl;
  fFcout_f << "*Requested number of events   : " << fReqNbOfEvts << std::endl;
  if (fFlagSubDet == "EB") {
    fFcout_f << "*SuperModule number           : " << fStexNumber << std::endl;
  }
  if (fFlagSubDet == "EE") {
    fFcout_f << "*Dee number                   : " << fStexNumber << std::endl;
  }
  fFcout_f << "*Date first requested event   : " << fStartDate;
  fFcout_f << "*Date last requested event    : " << fStopDate << std::endl;
  fFcout_f << std::endl;

  //=========================================================================
  //   closing of the results file if i_code = fCodeHeaderAscii only.
  //   closing is done in the fT1dWriteAscii() and fT2dWriteAscii() methods
  //   except for i_code = fCodeHeaderAscii
  //=========================================================================
  if (i_code == fCodeHeaderAscii) {
    fFcout_f.close();
  }
}

//=========================================================================
//
//         W R I T I N G   M E T H O D S :    R O O T    F I L E S
//
//    The root files are written by TEcnaRun since the arrays are computed
//    by this class (arrays fT2d_..., fT1d_..) and are private attributes.
//    No writing method in ROOT file here.
//    Just the making of the file name
//
//=========================================================================

//=========================================================================
//
//    W R I T I N G   M E T H O D S :    A S C I I    F I L E S
//
//    Ascii file are treated on the same footing as plots in the
//    TEcnaHistos methods.
//    The arrays are TVectorD or TMatrixD and are arguments of the
//    writting methods (WriteAsciiHisto, fT2dWriteAscii)
//
//=========================================================================
//
//  In the following, we describe:
//
//     (1) The method which gets the path for the results ASCII files
//         from a "cna-configuration" file
//     (2) The codification for the names of the ASCII files
//     (3) The methods which writes the results in the ASCII files
//
//
//
// (1)-----------> Method to set the path for the results ASCII files
//
//      void  MyCnaRun->GetPathForResultsAsciiFiles(pathname);
//
//            TString pathname = name of a "cna-config" file located
//            in the user's HOME directory and containing one line which
//            specifies the path where must be written the .ascii result files.
//            (no slash at the end of the string)
//
//   DEFAULT:
//            void  MyCnaRun->GetPathForResultsAsciiFiles();
//            If there is no argument, the "cna-config" file must be named
//            "path_results_ascii.ecna" and must be located in the user's HOME
//            directory
//
//
// (2)-----------> Codification for the names of the ASCII files (examples):
//
//       aaa_nnnS_Rrrr_fff_ttt_SMnnn_pedestals.ascii   OR  aaa_nnnS_Rrrr_fff_ttt_Deennn_pedestals.ascii
//       aaa_nnnS_Rrrr_fff_ttt_SMnnn_HF_noise.ascii    OR  aaa_nnnS_Rrrr_fff_ttt_Deennn_HF_noise.ascii
//       aaa_nnnS_Rrrr_fff_ttt_SMnnn_corss_cCCC.ascii  OR  aaa_nnnS_Rrrr_fff_ttt_Deennn_corss_cCCC.ascii
//       etc...
//
//  with:
//       aaa = Analysis name
//       rrr = Run number
//       fff = First requested event number
//       ttt = Number of requested events
//       CCC = Electronic Channel number in Stex
//       nnn = SM number or Dee number
//
//  Examples:
//       StdPed12_10S_R66689_1_50_SM1_pedestals.ascii
//       StdPed6_10S_R66689_1_50_Dee3_cor_ss_c2234.ascii
//
//
// (3)-----------> Methods which write the ASCII files:
//
//  The methods which write the ASCII files are the following:
//
//      void  WriteAsciiCovariancesBetweenSamples(Channel, MatrixSize, TMatrixD matrix);
//      void  WriteAsciiCorrelationsBetweenSamples(Channel, MatrixSize, TMatrixD matrix);
//      void  fT2dWriteAscii(code, px, py, MatrixSize, TMatrixD matrix) [private]
//      void  WriteAsciiHisto(CodeHisto, HistoSize, TVectorD histo)
//
//  Each of these methods corresponds to a "calculation method" of TEcnaRun.
//  The calculation method of TEcnaRun must have been been called before
//  using the writing method of TEcnaWrite.
//==========================================================================================

//-------------------------------------------------------------------------------------------
//
//          WriteAsciiHisto()
//
//          Stex = SM or Dee , Stin = Tower or SC
//
//-------------------------------------------------------------------------------------------
void TEcnaWrite::WriteAsciiHisto(const TString& HistoCode, const Int_t& HisSize, const TVectorD& read_histo) {
  //Write histo with correspondance CNA-channel <-> xtal number in SM or Dee

  // BuildCrysTable() is called in the method Init() which is called by the constructor

  Int_t i_code = fCodeNbOfEvts;

  //--------------------------------------------------------------------------------
  if (HistoCode == "D_NOE_ChNb") {
    i_code = fCodeNbOfEvts;
  }
  if (HistoCode == "D_Ped_ChNb") {
    i_code = fCodePed;
  }
  if (HistoCode == "D_TNo_ChNb") {
    i_code = fCodeTno;
  }
  if (HistoCode == "D_LFN_ChNb") {
    i_code = fCodeLfn;
  }
  if (HistoCode == "D_HFN_ChNb") {
    i_code = fCodeHfn;
  }
  if (HistoCode == "D_MCs_ChNb") {
    i_code = fCodeMeanCorss;
  }
  if (HistoCode == "D_SCs_ChNb") {
    i_code = fCodeSigCorss;
  }

  fMakeResultsFileName(i_code);   // => Making of the results .ascii file name
  fAsciiFileWriteHeader(i_code);  // => Open of the file associated with stream fFcout_f

  //..................................... format numerical values
  fFcout_f << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
  fFcout_f << std::setprecision(3) << std::setw(6);
  fFcout_f.setf(std::ios::dec, std::ios::basefield);
  fFcout_f.setf(std::ios::fixed, std::ios::floatfield);
  fFcout_f.setf(std::ios::left, std::ios::adjustfield);
  fFcout_f.setf(std::ios::right, std::ios::adjustfield);

  std::cout << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
  std::cout << std::setprecision(3) << std::setw(6);
  std::cout.setf(std::ios::dec, std::ios::basefield);
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::left, std::ios::adjustfield);
  std::cout.setf(std::ios::right, std::ios::adjustfield);

  //........................................................ WriteAsciiHisto
  TString aStexName;
  Int_t MaxCar = fgMaxCar;
  aStexName.Resize(MaxCar);
  aStexName = "SM or Dee?";

  TString aStinName;
  MaxCar = fgMaxCar;
  aStinName.Resize(MaxCar);
  aStinName = "Tower or SC?";

  TString aHoco;
  MaxCar = fgMaxCar;
  aHoco.Resize(MaxCar);
  aHoco = "Eta or IX?";

  TString aVeco;
  MaxCar = fgMaxCar;
  aVeco.Resize(MaxCar);
  aVeco = "Phi or IY?";

  TString aSpecifa;
  MaxCar = fgMaxCar;
  aSpecifa.Resize(MaxCar);
  aSpecifa = " ";

  TString aSpecifc;
  MaxCar = fgMaxCar;
  aSpecifc.Resize(MaxCar);
  aSpecifc = " ";

  TString aSpecifd;
  MaxCar = fgMaxCar;
  aSpecifd.Resize(MaxCar);
  aSpecifd = " ";

  TString aSpecife;
  MaxCar = fgMaxCar;
  aSpecife.Resize(MaxCar);
  aSpecife = " ";

  TString aSpecif1;
  MaxCar = fgMaxCar;
  aSpecif1.Resize(MaxCar);
  aSpecif1 = " ";

  TString aSpecif2;
  MaxCar = fgMaxCar;
  aSpecif2.Resize(MaxCar);
  aSpecif2 = " ";

  if (fFlagSubDet == "EB") {
    aStexName = "SM   ";
    aStinName = "tower";
    aSpecifa = " channel# ";
    aHoco = "   Eta    ";
    aVeco = "   Phi    ";
    aSpecifc = " channel# ";
    aSpecifd = " crystal# ";
    aSpecife = "SM    ";
  }
  if (fFlagSubDet == "EE") {
    aStexName = "Dee  ";
    aStinName = " SC  ";
    aSpecifa = "  Sector# ";
    aHoco = "   IX     ";
    aVeco = "   IY     ";
    aSpecifc = " crystal# ";
    aSpecifd = "   SC #   ";
    aSpecife = "Sector";
  }

  //.............................................................. WriteAsciiHisto
  for (Int_t i0StexEcha = 0; i0StexEcha < HisSize; i0StexEcha++) {
    Int_t n1StexStin = 0;
    Int_t StexStinEcna = 0;
    Int_t i0StinEcha = 0;
    Int_t n1StinEcha = 0;
    Int_t n1StexCrys = 0;
    Int_t n1DataSector = 0;
    Int_t n1SCinDS = 0;

    if (fFlagSubDet == "EB") {
      n1StexStin = fEcalNumbering->Get1SMTowFrom0SMEcha(i0StexEcha);
      StexStinEcna = n1StexStin;
      i0StinEcha = fEcalNumbering->Get0TowEchaFrom0SMEcha(i0StexEcha);
      n1StexCrys = fEcalNumbering->Get1SMCrysFrom1SMTowAnd0TowEcha(n1StexStin, i0StinEcha);
    }
    if (fFlagSubDet == "EE") {
      StexStinEcna = fEcalNumbering->Get1DeeSCEcnaFrom0DeeEcha(i0StexEcha);
      n1DataSector = fEcalNumbering->GetDSFrom1DeeSCEcna(fStexNumber, StexStinEcna);
      n1SCinDS = fEcalNumbering->GetDSSCFrom1DeeSCEcna(fStexNumber, StexStinEcna);
      n1StexStin = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fStexNumber, StexStinEcna);
      n1StinEcha = fEcalNumbering->Get1SCEchaFrom0DeeEcha(i0StexEcha);
    }

    if (n1StexStin > 0) {
      if ((fFlagSubDet == "EB" && i0StinEcha == 0) || (fFlagSubDet == "EE" && n1StinEcha == 1)) {
        if (HistoCode == "D_NOE_ChNb") {
          aSpecif1 = "Number of";
          aSpecif2 = "     events (requested)";
        }
        if (HistoCode == "D_Ped_ChNb") {
          aSpecif1 = "Pedestals";
          aSpecif2 = "             ";
        }
        if (HistoCode == "D_TNo_ChNb") {
          aSpecif1 = "   Total ";
          aSpecif2 = "   noise     ";
        }
        if (HistoCode == "D_MCs_ChNb") {
          aSpecif1 = "    Mean ";
          aSpecif2 = "   cor(s,s)  ";
        }
        if (HistoCode == "D_LFN_ChNb") {
          aSpecif1 = "   Low Fq";
          aSpecif2 = "   noise     ";
        }
        if (HistoCode == "D_HFN_ChNb") {
          aSpecif1 = "  High Fq";
          aSpecif2 = "   noise     ";
        }
        if (HistoCode == "D_SCs_ChNb") {
          aSpecif1 = " Sigma of";
          aSpecif2 = "   cor(s,s)  ";
        }

        fFcout_f << std::endl;

        fFcout_f << aSpecifa.Data() << "  " << aStinName.Data() << "#   " << aSpecifc.Data() << aSpecifd.Data()
                 << aHoco.Data() << aVeco.Data() << aSpecif1.Data() << std::endl;

        fFcout_f << "  in " << aStexName.Data() << "  in " << aStexName.Data() << "  in " << aStinName.Data() << "  in "
                 << aSpecife.Data() << "  in " << aStexName.Data() << "  in " << aStexName.Data() << aSpecif2.Data()
                 << std::endl
                 << std::endl;
      }

      Double_t value = read_histo(i0StexEcha);

      if (fFlagSubDet == "EB") {
        fFcout_f << std::setw(7) << i0StexEcha << std::setw(8) << n1StexStin << std::setw(11)
                 << i0StinEcha  // (Electronic channel number in tower)
                 << std::setw(10) << n1StexCrys << std::setw(10)
                 << (Int_t)fEcalNumbering->GetEta(fStexNumber, StexStinEcna, i0StinEcha) << std::setw(10)
                 << (Int_t)fEcalNumbering->GetPhiInSM(fStexNumber, StexStinEcna, i0StinEcha);
      }
      if (fFlagSubDet == "EE") {
        Int_t n1StinEcha_m = n1StinEcha - 1;
        fFcout_f << std::setw(7) << n1DataSector << std::setw(8) << n1StexStin << std::setw(11)
                 << n1StinEcha  // (Xtal number for construction in SC)
                 << std::setw(10) << n1SCinDS << std::setw(10)
                 << fEcalNumbering->GetIXCrysInDee(fStexNumber, StexStinEcna, n1StinEcha_m) << std::setw(10)
                 << fEcalNumbering->GetJYCrysInDee(fStexNumber, StexStinEcna, n1StinEcha_m);
      }

      if (HistoCode == "D_NOE_ChNb") {
        Int_t ivalue = (Int_t)value;
        fFcout_f << std::setw(13) << ivalue;
        fFcout_f << std::setw(4) << "(" << std::setw(6) << fReqNbOfEvts << ")";
      } else {
        fFcout_f << std::setw(13) << value;
      }

      fFcout_f << std::endl;
    }
  }  // end of loop:  for (Int_t i0StexEcha=0; i0StexEcha<HisSize; i0StexEcha++)

  fFcout_f.close();

  // if(fFlagPrint != fCodePrintNoComment)
  // {
  std::cout << "*TEcnaWrite::WriteAsciiHisto(...)> INFO: "
            << "histo has been written in file: " << std::endl
            << "            " << fAsciiFileName.Data() << std::endl;
  // }
}  // end of TEcnaWrite::WriteAsciiHisto

//================================================================================
//
//        W R I T I N G  O F  T H E   C O V   A N D   C O R  M A T R I C E S
//
//================================================================================

//--------------------------------------------------------------------------------
//
//      Writing of the covariances between samples
//      for a given StexEcha in an ASCII file
//
//--------------------------------------------------------------------------------
void TEcnaWrite::WriteAsciiCovariancesBetweenSamples(const Int_t& user_StexStin,
                                                     const Int_t& user_StinEcha,
                                                     const Int_t& MatSize,
                                                     const TMatrixD& read_matrix) {
  //Writing of the covariances between samples for a given StexEcha in an ASCII file

  if (fFlagSubDet == "EB") {
    fStexStinUser = user_StexStin;
  }
  if (fFlagSubDet == "EE") {
    fStexStinUser = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fStexNumber, user_StexStin);
  }

  if (fFlagSubDet == "EB") {
    fStinEchaUser = user_StinEcha;
  }
  if (fFlagSubDet == "EE") {
    fStinEchaUser = user_StinEcha + 1;
  }

  Int_t i_code = fCodeCovCss;  // code for covariances between samples
  fMakeResultsFileName(i_code);
  fAsciiFileWriteHeader(i_code);

  Int_t i_pasx = fSectSampSizeX;
  Int_t i_pasy = fSectSampSizeY;

  fT2dWriteAscii(i_code, i_pasx, i_pasy, MatSize, read_matrix);
}

//---------------------------------------------------------------------------------
//
//   Writing of the correlations between samples
//   for a given StexEcha in an ASCII file
//
//---------------------------------------------------------------------------------
void TEcnaWrite::WriteAsciiCorrelationsBetweenSamples(const Int_t& user_StexStin,
                                                      const Int_t& user_StinEcha,
                                                      const Int_t& MatSize,
                                                      const TMatrixD& read_matrix) {
  //Writing of the correlations between samples for a given StexEcha in an ASCII file

  if (fFlagSubDet == "EB") {
    fStexStinUser = user_StexStin;
  }
  if (fFlagSubDet == "EE") {
    fStexStinUser = fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fStexNumber, user_StexStin);
  }

  if (fFlagSubDet == "EB") {
    fStinEchaUser = user_StinEcha;
  }
  if (fFlagSubDet == "EE") {
    fStinEchaUser = user_StinEcha + 1;
  }

  Int_t i_code = fCodeCorCss;  // code for correlations between samples
  fMakeResultsFileName(i_code);
  fAsciiFileWriteHeader(i_code);

  Int_t i_pasx = fSectSampSizeX;
  Int_t i_pasy = fSectSampSizeY;

  fT2dWriteAscii(i_code, i_pasx, i_pasy, MatSize, read_matrix);
}

//----------------------------------------------------------------------
//
//            fT2dWriteAscii: Array 2D of (n_sctx , n_scty) sectors
//                            of size: i_pasx_arg * i_pasy_arg
//
//                       (private)
//
//----------------------------------------------------------------------

void TEcnaWrite::fT2dWriteAscii(const Int_t& i_code,
                                const Int_t& i_pasx_arg,
                                const Int_t& i_pasy_arg,
                                const Int_t& MatSize,
                                const TMatrixD& read_matrix) {
  //Writing of a matrix by sectors (private)

  Int_t i_pasx = i_pasx_arg;  // taille secteur en x
  Int_t i_pasy = i_pasy_arg;  // taille secteur en y

  //------------ formatage des nombres en faisant appel a la classe ios

  fFcout_f << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
  fFcout_f.setf(std::ios::dec, std::ios::basefield);
  fFcout_f.setf(std::ios::fixed, std::ios::floatfield);
  fFcout_f.setf(std::ios::left, std::ios::adjustfield);
  fFcout_f.setf(std::ios::right, std::ios::adjustfield);
  fFcout_f << std::setprecision(3) << std::setw(6);

  std::cout << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
  std::cout.setf(std::ios::dec, std::ios::basefield);
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::left, std::ios::adjustfield);
  std::cout.setf(std::ios::right, std::ios::adjustfield);
  std::cout << std::setprecision(3) << std::setw(6);

  //--------------------- fin du formatage standard C++ -------------------

  //-----------------------------------------------------------------------
  //  Reservation dynamique d'un array Double_t** de dimensions
  //  les multiples de 5 juste au-dessus des dimensions de l'array 2D
  //  a ecrire ( array de dimensions
  //  (fEcal->MaxSampADC(),fEcal->MaxSampADC())
  //  (fEcal->MaxCrysEcnaInStex(),fEcal->MaxCrysEcnaInStex()) )
  //-----------------------------------------------------------------------
  // Determination des tailles multiples de fSectChanSizeX ou fSectSampSizeX

#define NOUC
#ifndef NOUC

  //*************** channels *************
  Int_t justap_chan = 0;

  if (fEcal->MaxCrysEcnaInStex() % fSectChanSizeX == 0) {
    justap_chan = fEcal->MaxCrysEcnaInStex();
  } else {
    justap_chan = ((fEcal->MaxCrysEcnaInStex() / fSectChanSizeX) + 1) * fSectChanSizeX;
  }

  //....................... Allocation fjustap_2d_cc

  if (i_code == fCodeHfCov || i_code == fCodeHfCor || i_code == fCodeLfCov || i_code == fCodeLfCor) {
    if (fjustap_2d_cc == 0) {
      //................... Allocation
      fjustap_2d_cc = new Double_t*[justap_chan];
      fCnew++;
      fjustap_1d_cc = new Double_t[justap_chan * justap_chan];
      fCnew++;
      for (Int_t i = 0; i < justap_chan; i++) {
        fjustap_2d_cc[i] = &fjustap_1d_cc[0] + i * justap_chan;
      }
    }

    //............................... Transfert des valeurs dans fjustap_2d_cc  (=init)
    for (Int_t i = 0; i < fEcal->MaxCrysEcnaInStex(); i++) {
      for (Int_t j = 0; j < fEcal->MaxCrysEcnaInStex(); j++) {
        if (i_code == fCodeHfCov) {
          fjustap_2d_cc[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeHfCor) {
          fjustap_2d_cc[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeLfCov) {
          fjustap_2d_cc[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeLfCor) {
          fjustap_2d_cc[i][j] = read_matrix(i, j);
        }
      }
    }

    //.......................... mise a zero du reste de la matrice (=init)
    for (Int_t i = fEcal->MaxCrysEcnaInStex(); i < justap_chan; i++) {
      for (Int_t j = fEcal->MaxCrysEcnaInStex(); j < justap_chan; j++) {
        fjustap_2d_cc[i][j] = (Double_t)0.;
      }
    }
  }

#endif  //NOUC

  //************************************ Samples ***************************
  Int_t justap_samp = 0;

  if (fEcal->MaxSampADC() % fSectSampSizeX == 0) {
    justap_samp = fEcal->MaxSampADC();
  } else {
    justap_samp = ((fEcal->MaxSampADC() / fSectSampSizeX) + 1) * fSectSampSizeX;
  }

  //....................... allocation fjustap_2d_ss

  if (i_code == fCodeCovCss || i_code == fCodeCorCss || i_code == fCodeAvMeanCorss || i_code == fCodeAvSigCorss) {
    if (fjustap_2d_ss == nullptr) {
      //................... Allocation
      fjustap_2d_ss = new Double_t*[justap_samp];
      fCnew++;
      fjustap_1d_ss = new Double_t[justap_samp * justap_samp];
      fCnew++;
      for (Int_t i = 0; i < justap_samp; i++) {
        fjustap_2d_ss[i] = &fjustap_1d_ss[0] + i * justap_samp;
      }
    }

    //.............................. Transfert des valeurs dans fjustap_2d_ss (=init)
    for (Int_t i = 0; i < fEcal->MaxSampADC(); i++) {
      for (Int_t j = 0; j < fEcal->MaxSampADC(); j++) {
        if (i_code == fCodeCovCss) {
          fjustap_2d_ss[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeCorCss) {
          fjustap_2d_ss[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeAvMeanCorss) {
          fjustap_2d_ss[i][j] = read_matrix(i, j);
        }
        if (i_code == fCodeAvSigCorss) {
          fjustap_2d_ss[i][j] = read_matrix(i, j);
        }
      }
    }

    //.......................... mise a zero du reste de la matrice (=init)
    for (Int_t i = fEcal->MaxSampADC(); i < justap_samp; i++) {
      for (Int_t j = fEcal->MaxSampADC(); j < justap_samp; j++) {
        fjustap_2d_ss[i][j] = (Double_t)0.;
      }
    }
  }

  //..................... impressions + initialisations selon i_code

  Int_t isx_max = 0;
  Int_t isy_max = 0;

#define COCC
#ifndef COCC
  if (i_code == fCodeHfCov) {
    fFcout_f << "Covariance matrix between channels "
             << "for sample number " << fUserSamp;
    isx_max = justap_chan;
    isy_max = justap_chan;
  }
  if (i_code == fCodeHfCor) {
    fFcout_f << "*Correlation matrix between channels "
             << "for sample number " << fUserSamp;
    isx_max = justap_chan;
    isy_max = justap_chan;
  }

  if (i_code == fCodeLfCov) {
    fFcout_f << "Covariance matrix between channels "
             << "averaged on the samples ";
    isx_max = justap_chan;
    isy_max = justap_chan;
  }
  if (i_code == fCodeLfCor) {
    fFcout_f << "Correlation matrix between channels "
             << "averaged on the samples ";
    isx_max = justap_chan;
    isy_max = justap_chan;
  }
#endif  // COCC

  Int_t n1StexStin = 0;
  Int_t i0StinEcha = 0;
  Int_t n1StinEcha = 0;

  if (fFlagSubDet == "EB") {
    n1StexStin = fStexStinUser;
    i0StinEcha = fEcalNumbering->Get0TowEchaFrom0SMEcha(fStinEchaUser);
  }
  if (fFlagSubDet == "EE") {
    n1StexStin = fStexStinUser;
    Int_t fStinEchaUser_m = fStinEchaUser - 1;
    n1StinEcha = fEcalNumbering->Get1SCEchaFrom0DeeEcha(fStinEchaUser_m);

    TString sDeeDir = fEcalNumbering->GetDeeDirViewedFromIP(fStexNumber);
  }

  if (i_code == fCodeCovCss) {
    if (fFlagSubDet == "EB") {
      fFcout_f << "Covariance matrix between samples "
               << "for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin << " , channel in "
               << fStinName << ": " << i0StinEcha << ")";
    }
    if (fFlagSubDet == "EE") {
      fFcout_f << "Covariance matrix between samples "
               << "for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin << " , channel in "
               << fStinName << ": " << n1StinEcha << ")";
    }
    isx_max = justap_samp;
    isy_max = justap_samp;
  }
  if (i_code == fCodeCorCss) {
    if (fFlagSubDet == "EB") {
      fFcout_f << "Correlation matrix between samples "
               << "for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin << " , channel in "
               << fStinName << ": " << i0StinEcha << ")";
    }
    if (fFlagSubDet == "EE") {
      fFcout_f << "Correlation matrix between samples "
               << "for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin << " , channel in "
               << fStinName << ": " << n1StinEcha << ")";
    }
    isx_max = justap_samp;
    isy_max = justap_samp;
  }

  if (i_code == fCodeAvMeanCorss) {
    if (fFlagSubDet == "EB") {
      fFcout_f << "Correction factors to the covariances "
               << "between samples for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin
               << " , channel in " << fStinName << ": " << i0StinEcha << ")";
    }
    if (fFlagSubDet == "EE") {
      fFcout_f << "Correction factors to the covariances "
               << "between samples for channel number " << fStinEchaUser << " (" << fStinName << ": " << n1StexStin
               << " , channel in " << fStinName << ": " << n1StinEcha << ")";
    }
    isx_max = justap_samp;
    isy_max = justap_samp;
  }

  if (i_code == fCodeAvSigCorss) {
    if (fFlagSubDet == "EB") {
      fFcout_f << "Correction factors to the correlations "
               << "between samples for channel number " << fStinEchaUser << " ( " << fStinName << ": " << n1StexStin
               << " , channel in " << fStinName << ": " << i0StinEcha << ")";
    }
    if (fFlagSubDet == "EE") {
      fFcout_f << "Correction factors to the correlations "
               << "between samples for channel number " << fStinEchaUser << " ( " << fStinName << ": " << n1StexStin
               << " , channel in " << fStinName << ": " << n1StinEcha << ")";
    }
    isx_max = justap_samp;
    isy_max = justap_samp;
  }

  fFcout_f << std::endl;

  //............... Calcul des nombres de secteurs selon x
  //                i_pasx  = taille secteur en x
  //                isx_max = taille de la matrice en x
  //                n_sctx  = nombre de secteurs en x
  //
  if (i_pasx > isx_max) {
    i_pasx = isx_max;
  }
  Int_t n_sctx = 1;
  Int_t max_verix;
  if (i_pasx > 0) {
    n_sctx = isx_max / i_pasx;
  }
  max_verix = n_sctx * i_pasx;
  if (max_verix < isx_max) {
    n_sctx++;
  }

  //............... Calcul des nombres de secteurs selon y
  //                i_pasy  = taille secteur en y
  //                isy_max = taille de la matrice en y
  //                n_scty  = nombre de secteurs en x
  //
  if (i_pasy > isy_max) {
    i_pasy = isy_max;
  }
  Int_t n_scty = 1;
  Int_t max_veriy;
  if (i_pasy > 0) {
    n_scty = isy_max / i_pasy;
  }
  max_veriy = n_scty * i_pasy;
  if (max_veriy < isy_max) {
    n_scty++;
  }

#define NBSC
#ifndef NBSC
  //................ Ecriture de la taille et du nombre des secteurs
  if (i_code == fCodeCovCss || i_code == fCodeCorCss || i_code == fCodeAvMeanCorss || i_code == fCodeAvSigCorss) {
    fFcout_f << "sector size = " << fSectSampSizeX << " , number of sectors = " << n_sctx << " x " << n_scty << endl;
  }
  if (i_code == fCodeHfCov || i_code == fCodeHfCor || i_code == fCodeLfCov || i_code == fCodeLfCor) {
    fFcout_f << "sector size = " << fSectChanSizeX << " , number of sectors = " << n_sctx << " x " << n_scty
             << std::endl;
  }
#endif  // NBSC

  fFcout_f << std::endl;

  //............... impression matrice par secteurs i_pas x i_pas
  //........................... boucles pour display des secteurs
  Int_t ix_inf = -i_pasx;

  for (Int_t nsx = 0; nsx < n_sctx; nsx++) {
    //......................... calcul limites secteur
    ix_inf = ix_inf + i_pasx;
    Int_t ix_sup = ix_inf + i_pasx;

    Int_t iy_inf = -i_pasy;

    for (Int_t nsy = 0; nsy < n_scty; nsy++) {
      iy_inf = iy_inf + i_pasy;
      Int_t iy_sup = iy_inf + i_pasy;

      //......................... display du secteur (nsx,nsy)

      if (i_code == fCodeHfCov || i_code == fCodeCovCss || i_code == fCodeAvMeanCorss || i_code == fCodeAvSigCorss) {
        fFcout_f << "        ";
      }
      if (i_code == fCodeHfCor || i_code == fCodeCorCss) {
        fFcout_f << "      ";
      }

      for (Int_t iy_c = iy_inf; iy_c < iy_sup; iy_c++) {
        if (i_code == fCodeHfCov || i_code == fCodeLfCov || i_code == fCodeCovCss || i_code == fCodeAvMeanCorss ||
            i_code == fCodeAvSigCorss) {
          fFcout_f.width(8);
        }
        if (i_code == fCodeHfCor || i_code == fCodeLfCor || i_code == fCodeCorCss) {
          fFcout_f.width(6);
        }
        fFcout_f << iy_c << "  ";
      }
      fFcout_f << std::endl << std::endl;

      for (Int_t ix_c = ix_inf; ix_c < ix_sup; ix_c++) {
        if (i_code == fCodeHfCov || i_code == fCodeLfCov || i_code == fCodeCovCss || i_code == fCodeAvMeanCorss ||
            i_code == fCodeAvSigCorss) {
          fFcout_f.width(8);
        }
        if (i_code == fCodeHfCor || i_code == fCodeLfCor || i_code == fCodeCorCss) {
          fFcout_f.width(6);
        }
        fFcout_f << ix_c << "   ";

        for (Int_t iy_c = iy_inf; iy_c < iy_sup; iy_c++) {
          if (i_code == fCodeHfCov || i_code == fCodeLfCov || i_code == fCodeCovCss || i_code == fCodeAvMeanCorss ||
              i_code == fCodeAvSigCorss) {
            fFcout_f.width(8);
          }

          if (i_code == fCodeHfCor || i_code == fCodeLfCor || i_code == fCodeCorCss) {
            fFcout_f.width(6);
          }

          if (i_code == fCodeHfCov || i_code == fCodeLfCov || i_code == fCodeHfCor) {
            fFcout_f << fjustap_2d_cc[ix_c][iy_c] << "  ";
          }

          if (i_code == fCodeCovCss || i_code == fCodeCorCss || i_code == fCodeAvMeanCorss ||
              i_code == fCodeAvSigCorss) {
            fFcout_f << fjustap_2d_ss[ix_c][iy_c] << "  ";
          }
        }
        fFcout_f << std::endl;
      }
      fFcout_f << std::endl;
    }
  }

  //........... closing of the results file

  fFcout_f.close();

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaWrite::fT2dWriteAscii(....)> INFO: "
              << "matrix has been written in file: " << std::endl
              << "            " << fAsciiFileName.Data() << std::endl;
  }

}  // end of TEcnaWrite::fT2dWriteAscii

//=========================================================================
//
//   ci-dessous: ===> methodes a implementer plus tard?
//
//=========================================================================
#define WASC
#ifndef WASC
//------------------------------------------------------------
//
//      Writing of the expectation values in an ASCII file
//
//------------------------------------------------------------

void TEcnaWrite::WriteAsciiSampleMeans() {
  //Writing of the expectation values in an ASCII file

  Int_t i_code = fCodeMSp;
  fMakeResultsFileName(i_code);
  fAsciiFileWriteHeader(i_code);

  Int_t i_lic1 = fNbChanByLine;
  Int_t i_lic2 = fNbSampByLine;

  fT1dWriteAscii(i_code, i_lic1, i_lic2);
}

//-------------------------------------------------------
//
//      Writing of the sigmas in an ASCII file
//
//-------------------------------------------------------

void TEcnaWrite::WriteAsciiSampleSigmas() {
  //Writing of the variances in an ASCII file

  Int_t i_code = fCodeVar;  // code for variance
  fMakeResultsFileName(i_code);
  fAsciiFileWriteHeader(i_code);

  Int_t i_lic1 = fNbChanByLine;
  Int_t i_lic2 = fNbSampByLine;

  fT1dWriteAscii(i_code, i_lic1, i_lic2);
}
#endif  // WASC
