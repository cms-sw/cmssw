//----------Author's Name: B.Fabbro, FX Gentit DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 04/07/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"

//--------------------------------------
//  TEcnaRead.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaRead.h
//--------------------------------------

R__EXTERN TEcnaRootFile *gCnaRootFile;

ClassImp(TEcnaRead);
//___________________________________________________________________________
//

TEcnaRead::TEcnaRead() {
  Init();
  // std::cout << "[Info Management] CLASS: TEcnaRead.          CREATE OBJECT: this = " << this << std::endl;
}
//Constructor without argument

TEcnaRead::TEcnaRead(TEcnaObject *pObjectManager, const TString &SubDet) {
  fObjectManager = (TEcnaObject *)pObjectManager;
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaRead", i_this);

  //----------------------- Object management
  //............................ fCnaParCout
  fCnaParCout = nullptr;
  Long_t iCnaParCout = pObjectManager->GetPointerValue("TEcnaParCout");
  if (iCnaParCout == 0) {
    fCnaParCout = new TEcnaParCout(pObjectManager); /*fCnew++*/
  } else {
    fCnaParCout = (TEcnaParCout *)iCnaParCout;
  }

  //............................ fCnaParPaths
  fCnaParPaths = nullptr;
  Long_t iCnaParPaths = pObjectManager->GetPointerValue("TEcnaParPaths");
  if (iCnaParPaths == 0) {
    fCnaParPaths = new TEcnaParPaths(pObjectManager); /*fCnew++*/
  } else {
    fCnaParPaths = (TEcnaParPaths *)iCnaParPaths;
  }

  //............................ fEcalNumbering
  fEcalNumbering = nullptr;
  Long_t iEcalNumbering = pObjectManager->GetPointerValue("TEcnaNumbering");
  if (iEcalNumbering == 0) {
    fEcalNumbering = new TEcnaNumbering(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fEcalNumbering = (TEcnaNumbering *)iEcalNumbering;
  }

  //............................ fCnaParHistos
  fCnaParHistos = nullptr;
  Long_t iCnaParHistos = pObjectManager->GetPointerValue("TEcnaParHistos");
  if (iCnaParHistos == 0) {
    fCnaParHistos = new TEcnaParHistos(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fCnaParHistos = (TEcnaParHistos *)iCnaParHistos;
  }

  //............................ fCnaWrite
  fCnaWrite = nullptr;
  Long_t iCnaWrite = pObjectManager->GetPointerValue("TEcnaWrite");
  if (iCnaWrite == 0) {
    fCnaWrite = new TEcnaWrite(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fCnaWrite = (TEcnaWrite *)iCnaWrite;
  }

  // fEcal = 0; fEcal = new TEcnaParEcal(SubDet.Data());            // Anew("fEcal");
  //............................ fEcal  => to be changed in fParEcal
  fEcal = nullptr;
  Long_t iParEcal = pObjectManager->GetPointerValue("TEcnaParEcal");
  if (iParEcal == 0) {
    fEcal = new TEcnaParEcal(pObjectManager, SubDet.Data()); /*fCnew++*/
  } else {
    fEcal = (TEcnaParEcal *)iParEcal;
  }

  //fFileHeader  = 0;
  //const Text_t *h_name  = "CnaHeader";  //==> voir cette question avec FXG
  //const Text_t *h_title = "CnaHeader";  //==> voir cette question avec FXG
  //fFileHeader = new TEcnaHeader(h_name, h_title);  // Anew("fFileHeader");

  //............................ fFileHeader
  const Text_t *h_name = "CnaHeader";   //==> voir cette question avec FXG
  const Text_t *h_title = "CnaHeader";  //==> voir cette question avec FXG

  // "CnaHeader" est le nom du fichier utilisé dans ReadRootFileHeader(...) sous la forme:
  //
  // TEcnaHeader *h;
  // h =(TEcnaHeader*)gCnaRootFile->fRootFile->Get("CnaHeader");
  //

  fFileHeader = nullptr;
  Long_t iFileHeader = pObjectManager->GetPointerValue("TEcnaHeader");
  if (iFileHeader == 0) {
    fFileHeader = new TEcnaHeader(pObjectManager, h_name, h_title); /*fCnew++*/
  } else {
    fFileHeader = (TEcnaHeader *)iFileHeader;
  }

  Init();
  SetEcalSubDetector(SubDet.Data());

  // std::cout << "[Info Management] CLASS: TEcnaRead.          CREATE OBJECT: this = " << this << std::endl;
}

void TEcnaRead::Init() {
  //Initialisation concerning the ROOT file

  fCnew = 0;
  fCdelete = 0;

  fTTBELL = '\007';

  fgMaxCar = (Int_t)512;

  fCodePrintNoComment = fCnaParCout->GetCodePrint("NoComment");
  fCodePrintWarnings = fCnaParCout->GetCodePrint("Warnings ");
  fCodePrintComments = fCnaParCout->GetCodePrint("Comments");
  fCodePrintAllComments = fCnaParCout->GetCodePrint("AllComments");

  //.................................. Set flag print to "Warnings"   (Init)
  fFlagPrint = fCodePrintWarnings;

  //................................ tags and array Stin numbers
  fTagStinNumbers = nullptr;
  fMemoStinNumbers = 0;
  fT1d_StexStinFromIndex = nullptr;

  //................................
  //  fMemoReadNumberOfEventsforSamples = 0;

  //.......................... flag data exist (utile ici?)
  fDataExist = kFALSE;

  //......................... transfert Sample ADC Values 3D array   (Init)
  fT3d_AdcValues = nullptr;
  fT3d2_AdcValues = nullptr;
  fT3d1_AdcValues = nullptr;

  //................................. path for .root files
  Int_t MaxCar = fgMaxCar;
  fPathRoot.Resize(MaxCar);
  fPathRoot = "fPathRoot not defined";

  //.................................. Pointer and Flags for Root File   (Init)
  gCnaRootFile = nullptr;

  fOpenRootFile = kFALSE;
  fReadyToReadRootFile = 0;
  fLookAtRootFile = 0;

  //................................. currently open file
  fFlagNoFileOpen.Resize(MaxCar);
  fFlagNoFileOpen = "No file is open";

  fCurrentlyOpenFileName.Resize(MaxCar);
  fCurrentlyOpenFileName = fFlagNoFileOpen;

}  // end of Init()

//============================================================================================================

void TEcnaRead::SetEcalSubDetector(const TString &SubDet) {
  // Set Subdetector (EB or EE)

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();

  if (fFlagSubDet == "EB") {
    fStexName = "SM";
    fStinName = "tower";
  }
  if (fFlagSubDet == "EE") {
    fStexName = "Dee";
    fStinName = "SC";
  }
}

//============================================================================================================
void TEcnaRead::Anew(const TString &VarName) {
  // allocation survey for new

  fCnew++;
  // std::cout << "TEcnaRead::Anew---> new " << std::setw(4) << fCnew << " --------------> " << std::setw(25)
  //      << VarName.Data() << " / object(this): " << this << std::endl;
}

void TEcnaRead::Adelete(const TString &VarName) {
  // allocation survey for delete

  fCdelete++;
  // std::cout << "TEcnaRead::Adelete> ========== delete" << std::setw(4) << fCdelete << " -> " << std::setw(25)
  //      << VarName.Data() << " / object(this): " << this << std::endl;
}

//=========================================== private copy ==========

void TEcnaRead::fCopy(const TEcnaRead &rund) {
  //Private copy

  fFileHeader = rund.fFileHeader;
  fOpenRootFile = rund.fOpenRootFile;

  //........................................ Codes

  fCodePrintComments = rund.fCodePrintComments;
  fCodePrintWarnings = rund.fCodePrintWarnings;
  fCodePrintAllComments = rund.fCodePrintAllComments;
  fCodePrintNoComment = rund.fCodePrintNoComment;

  //.................................................. Tags
  fTagStinNumbers = rund.fTagStinNumbers;

  fFlagPrint = rund.fFlagPrint;
  fPathRoot = rund.fPathRoot;

  fCnew = rund.fCnew;
  fCdelete = rund.fCdelete;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    copy constructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead::TEcnaRead(const TEcnaRead &dcop) : TObject::TObject(dcop) {
  std::cout << "*TEcnaRead::TEcnaRead(const TEcnaRead& dcop)> "
            << " It is time to write a copy constructor" << std::endl;

  // { Int_t cintoto;  cin >> cintoto; }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead &TEcnaRead::operator=(const TEcnaRead &dcop) {
  //Overloading of the operator=

  fCopy(dcop);
  return *this;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead::~TEcnaRead() {
  //Destructor

  // std::cout << "[Info Management] CLASS: TEcnaRead.          DESTROY OBJECT: this = " << this << std::endl;

  if (fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments) {
    std::cout << "*TEcnaRead::~TEcnaRead()> Entering destructor" << std::endl;
  }

  //if (fFileHeader    != 0){delete fFileHeader;    Adelete("fFileHeader");}
  //if (fEcal          != 0){delete fEcal;          Adelete("fEcal");}
  //if (fCnaParCout    != 0){delete fCnaParCout;    Adelete("fCnaParCout");}
  //if (fCnaParPaths   != 0){delete fCnaParPaths;   Adelete("fCnaParPaths");}
  //if (fCnaWrite      != 0){delete fCnaWrite;      Adelete("fCnaWrite");}
  //if (fEcalNumbering != 0){delete fEcalNumbering; Adelete("fEcalNumbering");}

  if (fT1d_StexStinFromIndex != nullptr) {
    delete[] fT1d_StexStinFromIndex;
    Adelete("fT1d_StexStinFromIndex");
  }
  if (fTagStinNumbers != nullptr) {
    delete[] fTagStinNumbers;
    Adelete("fTagStinNumbers");
  }

  if (fT3d_AdcValues != nullptr) {
    delete[] fT3d_AdcValues;
    Adelete("fT3d_AdcValues");
  }
  if (fT3d2_AdcValues != nullptr) {
    delete[] fT3d2_AdcValues;
    Adelete("fT3d2_AdcValues");
  }
  if (fT3d1_AdcValues != nullptr) {
    delete[] fT3d1_AdcValues;
    Adelete("fT3d1_AdcValues");
  }

  if (fCnew != fCdelete) {
    std::cout << "!TEcnaRead/destructor> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = " << fCnew
              << ", fCdelete = " << fCdelete << fTTBELL << std::endl;
  } else {
    // std::cout << "*TEcnaRead/destructor> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
    //      << fCnew << ", fCdelete = " << fCdelete << std::endl;
  }

  if (fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments) {
    std::cout << "*TEcnaRead::~TEcnaRead()> End of destructor " << std::endl;
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                             M  E  T  H  O  D  S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//============================================================================
//
//                       1 D   H I S T O S
//
//============================================================================
TVectorD TEcnaRead::Read1DHisto(const Int_t &VecDim,
                                const TString &UserQuantity,
                                const Int_t &n1StexStin,
                                const Int_t &i0StinEcha,
                                const Int_t &n1Sample) {
  Int_t VecDimTest = fFileHeader->fReqNbOfEvts;

  if (VecDim == VecDimTest) {
    TVectorD vec(VecDim);

    TString CallingMethod = "1D";
    TString StandardQuantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, UserQuantity);

    if (StandardQuantity == "Adc") {
      Int_t i0Sample = n1Sample - 1;
      vec = ReadSampleAdcValues(n1StexStin, i0StinEcha, i0Sample, VecDim);
    } else {
      for (Int_t i = 0; i < VecDim; i++) {
        vec(i) = (double_t)0.;
      }
      std::cout << "!TEcnaRead::Read1DHisto(...)>  UserQuantity = " << UserQuantity
                << "(StandardQuantity = " << StandardQuantity << "). Wrong code, no file reading." << fTTBELL
                << std::endl;
    }
    return vec;
  } else {
    TVectorD vec(VecDim);
    for (Int_t i = 0; i < VecDim; i++) {
      vec(i) = (double_t)0.;
    }
    std::cout << "!TEcnaRead::Read1DHisto(...)> UserQuantity = " << UserQuantity << ", VecDim = " << VecDim
              << "(VecDimTest = " << VecDimTest << ")"
              << ". Wrong code or array dimension. No file reading." << fTTBELL << std::endl;
    return vec;
  }
}  // end of Read1DHisto / ReadSampleAdcValues

TVectorD TEcnaRead::Read1DHisto(const Int_t &VecDim, const TString &UserQuantity, const Int_t &n1StexStin) {
  Int_t VecDimTest = fEcal->MaxCrysInStin() * fEcal->MaxSampADC();
  if (VecDim == VecDimTest) {
    TVectorD vec(VecDim);

    TString CallingMethod = "1D";
    TString StandardQuantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, UserQuantity);

    if (StandardQuantity == "MSp" || StandardQuantity == "SSp") {
      if (StandardQuantity == "MSp") {
        vec = ReadSampleMeans(n1StexStin, VecDim);
      }
      if (StandardQuantity == "SSp") {
        vec = ReadSampleSigmas(n1StexStin, VecDim);
      }
    } else {
      for (Int_t i = 0; i < VecDim; i++) {
        vec(i) = (double_t)0.;
      }
      std::cout << "!TEcnaRead::Read1DHisto(...)>  UserQuantity = " << UserQuantity
                << ", StandardQuantity = " << StandardQuantity << ". Wrong code, no file reading." << fTTBELL
                << std::endl;
    }
    return vec;
  } else {
    TVectorD vec(VecDim);
    for (Int_t i = 0; i < VecDim; i++) {
      vec(i) = (double_t)0.;
    }
    std::cout << "!TEcnaRead::Read1DHisto(...)> UserQuantity = " << UserQuantity << ", VecDim = " << VecDim
              << "(VecDimTest = " << VecDimTest << ")"
              << ". Wrong code or array dimension. No file reading." << fTTBELL << std::endl;
    return vec;
  }
}  // end of Read1DHisto / ReadSampleMeans , ReadSampleSigmas

TVectorD TEcnaRead::Read1DHisto(const Int_t &VecDim, const TString &UserQuantity, const TString &UserDetector) {
  // VecDim = fEcal->MaxCrysEcnaInStex if StandardDetector = "SM" or "Dee"
  // VecDim = fEcal->MaxStinEcnaInStas if StandardDetector = "EB" or "EE"

  Int_t VecDimTest = 1;
  TString StandardDetector = fCnaParHistos->BuildStandardDetectorCode(UserDetector);
  if (StandardDetector == "SM" || StandardDetector == "Dee") {
    VecDimTest = fEcal->MaxCrysEcnaInStex();
  }
  if (StandardDetector == "EB" || StandardDetector == "EE") {
    VecDimTest = fEcal->MaxStinEcnaInStas();
  }

  if (VecDim == VecDimTest) {
    TVectorD vec(VecDim);

    TString CallingMethod = "1D";
    TString StandardQuantity = "?";
    StandardQuantity = fCnaParHistos->BuildStandard1DHistoCodeY(CallingMethod, UserQuantity);
    TString rTechReadCode = GetTechReadCode(StandardQuantity, StandardDetector);

    if (rTechReadCode != "?") {
      if (StandardDetector == "SM" || StandardDetector == "Dee") {
        if (rTechReadCode == "NOEStex") {
          vec = ReadNumberOfEvents(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "PedStex") {
          vec = ReadPedestals(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "TNoStex") {
          vec = ReadTotalNoise(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "LFNStex") {
          vec = ReadLowFrequencyNoise(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "HFNStex") {
          vec = ReadHighFrequencyNoise(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "MCsStex") {
          vec = ReadMeanCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());
        }
        if (rTechReadCode == "SCsStex") {
          vec = ReadSigmaOfCorrelationsBetweenSamples(fEcal->MaxCrysEcnaInStex());
        }
      }

      if (StandardDetector == "EB" || StandardDetector == "EE") {
        TVectorD vecStex(fEcal->MaxStinEcnaInStex());
        for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
          vecStex(i) = (Double_t)0.;
        }

        time_t xStartTime = GetStartTime();
        time_t xStopTime = GetStopTime();
        TString xStartDate = "sStartDate";
        TString xStopDate = "sStopDate";

        for (Int_t i0Stex = 0; i0Stex < fEcal->MaxStexInStas(); i0Stex++) {
          Int_t n1Stex = i0Stex + 1;
          FileParameters(fFileHeader->fTypAna,
                         fFileHeader->fNbOfSamples,
                         fFileHeader->fRunNumber,
                         fFileHeader->fFirstReqEvtNumber,
                         fFileHeader->fLastReqEvtNumber,
                         fFileHeader->fReqNbOfEvts,
                         n1Stex,
                         fPathRoot);

          if (LookAtRootFile() == kTRUE) {
            if (rTechReadCode == "NOEStas") {
              vecStex = ReadAverageNumberOfEvents(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "PedStas") {
              vecStex = ReadAveragePedestals(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "TNoStas") {
              vecStex = ReadAverageTotalNoise(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "LFNStas") {
              vecStex = ReadAverageLowFrequencyNoise(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "HFNStas") {
              vecStex = ReadAverageHighFrequencyNoise(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "MCsStas") {
              vecStex = ReadAverageMeanCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());
            }
            if (rTechReadCode == "SCsStas") {
              vecStex = ReadAverageSigmaOfCorrelationsBetweenSamples(fEcal->MaxStinEcnaInStex());
            }

            for (Int_t i0Stin = 0; i0Stin < fEcal->MaxStinEcnaInStex(); i0Stin++) {
              vec(fEcal->MaxStinEcnaInStex() * i0Stex + i0Stin) = vecStex(i0Stin);
            }

            //............ Get start and stop date for the Stas (Stas = EB or EE)
            if (i0Stex == 0) {
              xStartTime = GetStartTime();
              xStopTime = GetStopTime();
              xStartDate = GetStartDate();
              xStopDate = GetStopDate();
            }
            time_t cStartTime = GetStartTime();
            time_t cStopTime = GetStopTime();
            TString cStartDate = GetStartDate();
            TString cStopDate = GetStopDate();

            if (cStartTime < xStartTime) {
              xStartTime = cStartTime;
              xStartDate = cStartDate;
            }
            if (cStopTime > xStopTime) {
              xStopTime = cStopTime;
              xStopDate = cStopDate;
            }

            fFileHeader->fStartDate = xStartDate;
            fFileHeader->fStopDate = xStopDate;
          } else {
            std::cout << "!TEcnaRead::Read1DHisto(const TString&, const TString&)> *ERROR* =====> "
                      << " ROOT file not found" << fTTBELL << std::endl;
          }
        }
      }
    } else {
      for (Int_t i = 0; i < VecDim; i++) {
        vec(i) = (double_t)0.;
      }
      std::cout << "!TEcnaRead::Read1DHisto(...)> UserQuantity = " << UserQuantity
                << ", UserDetector = " << UserDetector << ". Wrong code(s). No file reading." << fTTBELL << std::endl;
    }
    return vec;
  } else {
    TVectorD vec(VecDim);
    for (Int_t i = 0; i < VecDim; i++) {
      vec(i) = (double_t)0.;
    }
    std::cout << "!TEcnaRead::Read1DHisto(...)> UserQuantity = " << UserQuantity << ", UserDetector = " << UserDetector
              << ", VecDim = " << VecDim << ". Wrong code(s) or array dimension. No file reading." << fTTBELL
              << std::endl;
    return vec;
  }
}  // end of Read1DHisto / Stex and Stas histos

//============================================================================
//
//                       2 D   H I S T O S
//
//============================================================================
TMatrixD TEcnaRead::ReadMatrix(const Int_t &MatDim,
                               const TString &UserCorOrCov,
                               const TString &UserBetweenWhat,
                               const Int_t &nb_arg_1,
                               const Int_t &nb_arg_2) {
  TMatrixD mat(MatDim, MatDim);
  TString CallingMethod = "2D";
  TString StandardMatrixType = "?";
  TString StandardBetweenWhat = "?";

  StandardMatrixType = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);
  StandardBetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);

  if (StandardMatrixType != "?" && StandardBetweenWhat != "?") {
    if (StandardBetweenWhat == "Mss") {
      Int_t n1StexStin = nb_arg_1;
      Int_t i0StinEcha = nb_arg_2;

      if (StandardMatrixType == "Cov") {
        mat = ReadCovariancesBetweenSamples(n1StexStin, i0StinEcha, MatDim);
      }

      if (StandardMatrixType == "Cor") {
        mat = ReadCorrelationsBetweenSamples(n1StexStin, i0StinEcha, MatDim);
      }
    }

    if (StandardBetweenWhat != "Mss") {
      Int_t n1StexStin_a = nb_arg_1;
      Int_t n1StexStin_b = nb_arg_2;

      if (StandardMatrixType == "Cov" && StandardBetweenWhat == "MccLF") {
        mat = ReadLowFrequencyCovariancesBetweenChannels(n1StexStin_a, n1StexStin_b, MatDim);
      }

      if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MccLF") {
        mat = ReadLowFrequencyCorrelationsBetweenChannels(n1StexStin_a, n1StexStin_b, MatDim);
      }

      if (StandardMatrixType == "Cov" && StandardBetweenWhat == "MccHF") {
        mat = ReadHighFrequencyCovariancesBetweenChannels(n1StexStin_a, n1StexStin_b, MatDim);
      }

      if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MccHF") {
        mat = ReadHighFrequencyCorrelationsBetweenChannels(n1StexStin_a, n1StexStin_b, MatDim);
      }
    }
  } else {
    for (Int_t i = 0; i - MatDim < 0; i++) {
      for (Int_t j = 0; j - MatDim < 0; j++) {
        mat(i, j) = (double_t)0.;
      }
    }
    std::cout << "!TEcnaRead::ReadMatrix(...)> UserCorOrCov = " << UserCorOrCov
              << ", UserBetweenWhat = " << UserBetweenWhat << ". Wrong code(s), no file reading." << fTTBELL
              << std::endl;
  }
  return mat;
}

TMatrixD TEcnaRead::ReadMatrix(const Int_t &MatDim, const TString &UserCorOrCov, const TString &UserBetweenWhat) {
  //------------------- (BIG MATRIX 1700x1700 for barrel, 5000x5000 for endcap) ------------------
  TMatrixD mat(MatDim, MatDim);
  TString CallingMethod = "2D";
  TString StandardMatrixType = "?";
  TString StandardBetweenWhat = "?";

  StandardMatrixType = fCnaParHistos->BuildStandardCovOrCorCode(CallingMethod, UserCorOrCov);
  StandardBetweenWhat = fCnaParHistos->BuildStandardBetweenWhatCode(CallingMethod, UserBetweenWhat);

  if (StandardMatrixType != "?" && StandardBetweenWhat != "?") {
    //......................... between channels (covariances, correlations)
    if (StandardMatrixType == "Cov" && StandardBetweenWhat == "MccLF") {
      mat = ReadLowFrequencyCovariancesBetweenChannels(MatDim);
    }

    if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MccLF") {
      mat = ReadLowFrequencyCorrelationsBetweenChannels(MatDim);
    }

    if (StandardMatrixType == "Cov" && StandardBetweenWhat == "MccHF") {
      mat = ReadHighFrequencyCovariancesBetweenChannels(MatDim);
    }

    if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MccHF") {
      mat = ReadHighFrequencyCorrelationsBetweenChannels(MatDim);
    }

    //......................... between Stins (mean correlations)
    if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MttLF") {
      mat = ReadLowFrequencyMeanCorrelationsBetweenStins(MatDim);
    }

    if (StandardMatrixType == "Cor" && StandardBetweenWhat == "MttHF") {
      mat = ReadHighFrequencyMeanCorrelationsBetweenStins(MatDim);
    }
  } else {
    for (Int_t i = 0; i - MatDim < 0; i++) {
      for (Int_t j = 0; j - MatDim < 0; j++) {
        mat(i, j) = (double_t)0.;
      }
    }
    std::cout << "!TEcnaRead::ReadMatrix(...)> UserCorOrCov = " << UserCorOrCov
              << ", UserBetweenWhat = " << UserBetweenWhat << ". Wrong code(s), no file reading." << fTTBELL
              << std::endl;
  }
  return mat;
}

//============================================================================
TString TEcnaRead::GetTechReadCode(const TString &StandardQuantity, const TString &StandardDetector) {
  TString rTechReadCode = "?";
  TString dTechDetector = "?";

  if (StandardDetector == "SM" || StandardDetector == "Dee") {
    dTechDetector = "Stex";
  }
  if (StandardDetector == "EB" || StandardDetector == "EE") {
    dTechDetector = "Stas";
  }

  if (dTechDetector == "?") {
    std::cout << "!TEcnaRead::GetTechReadCode(...)> *** ERROR: wrong standard code *** dTechDetector = "
              << dTechDetector << ", StandardDetector = " << StandardDetector << fTTBELL << std::endl;
  } else {
    if (StandardQuantity == "NOE" && dTechDetector == "Stex") {
      rTechReadCode = "NOEStex";
    }
    if (StandardQuantity == "NOE" && dTechDetector == "Stas") {
      rTechReadCode = "NOEStas";
    }
    if (StandardQuantity == "Ped" && dTechDetector == "Stex") {
      rTechReadCode = "PedStex";
    }
    if (StandardQuantity == "Ped" && dTechDetector == "Stas") {
      rTechReadCode = "PedStas";
    }
    if (StandardQuantity == "TNo" && dTechDetector == "Stex") {
      rTechReadCode = "TNoStex";
    }
    if (StandardQuantity == "TNo" && dTechDetector == "Stas") {
      rTechReadCode = "TNoStas";
    }
    if (StandardQuantity == "LFN" && dTechDetector == "Stex") {
      rTechReadCode = "LFNStex";
    }
    if (StandardQuantity == "LFN" && dTechDetector == "Stas") {
      rTechReadCode = "LFNStas";
    }
    if (StandardQuantity == "HFN" && dTechDetector == "Stex") {
      rTechReadCode = "HFNStex";
    }
    if (StandardQuantity == "HFN" && dTechDetector == "Stas") {
      rTechReadCode = "HFNStas";
    }
    if (StandardQuantity == "MCs" && dTechDetector == "Stex") {
      rTechReadCode = "MCsStex";
    }
    if (StandardQuantity == "MCs" && dTechDetector == "Stas") {
      rTechReadCode = "MCsStas";
    }
    if (StandardQuantity == "SCs" && dTechDetector == "Stex") {
      rTechReadCode = "SCsStex";
    }
    if (StandardQuantity == "SCs" && dTechDetector == "Stas") {
      rTechReadCode = "SCsStas";
    }
  }

  if (rTechReadCode == "?") {
    std::cout << "!TEcnaRead::GetTechReadCode(...)> *** ERROR: wrong standard code *** rTechReadCode = "
              << rTechReadCode << ", StandardQuantity = " << StandardQuantity << fTTBELL << std::endl;
  }

  return rTechReadCode;
}

//==========================================================================================
//
//                             FileParameters(...)
//
//==========================================================================================
void TEcnaRead::FileParameters(const TString &typ_ana,
                               const Int_t &nb_of_samples,
                               const Int_t &run_number,
                               const Int_t &nfirst,
                               const Int_t &nlast,
                               const Int_t &nreqevts,
                               const Int_t &Stex,
                               const TString &path_root) {
  // Preparation for reading the ROOT file
  // Preliminary save of the arguments values because they can be of the form: fFileHeader->...
  // and because fFileHeader can be deleted and re-created in this method

  const TString &sTypAna = typ_ana;
  Int_t nNbOfSamples = nb_of_samples;
  Int_t nRunNumber = run_number;
  Int_t nFirstEvt = nfirst;
  Int_t nLastEvt = nlast;
  Int_t nReqNbOfEvts = nreqevts;
  Int_t nStexNumber = Stex;

  //................................................................................................
  const Text_t *h_name = "CnaHeader";   //==> voir cette question avec FXG
  const Text_t *h_title = "CnaHeader";  //==> voir cette question avec FXG

  // "CnaHeader" est le nom du fichier utilisé dans ReadRootFileHeader(...) sous la forme:
  //
  // TEcnaHeader *h;
  // h =(TEcnaHeader*)gCnaRootFile->fRootFile->Get("CnaHeader");
  //

  //----------- old version, with arguments h_name, h_title, (FXG) ----------
  //
  // fFileHeader->HeaderParameters(h_name,    h_title,
  //			           sTypAna,   nNbOfSamples, nRunNumber,
  //			           nFirstEvt, nLastEvt, nReqNbOfEvts, nStexNumber);
  //
  //-------------------------------------------------------------------------

  //---------- new version
  if (fFileHeader == nullptr) {
    fFileHeader = new TEcnaHeader(fObjectManager, h_name, h_title); /* Anew("fFileHeader") */
    ;
  }
  fFileHeader->HeaderParameters(sTypAna, nNbOfSamples, nRunNumber, nFirstEvt, nLastEvt, nReqNbOfEvts, nStexNumber);

  // After this call to TEcnaHeader, we have:
  //     fFileHeader->fTypAna            = sTypAna
  //     fFileHeader->fNbOfSamples       = nNbOfSamples
  //     fFileHeader->fRunNumber         = nRunNumber
  //     fFileHeader->fFirstReqEvtNumber = nFirstEvt
  //     fFileHeader->fLastReqEvtNumber  = nLastEvt
  //     fFileHeader->fReqNbOfEvts       = nReqNbOfEvts
  //     fFileHeader->fStex              = nStexNumber                       ( FileParameters(...) )
  //.......................... path_root
  fPathRoot = path_root;

  //-------- gets the arguments for the file names (long and short) and makes these names
  fCnaWrite->RegisterFileParameters(typ_ana, nb_of_samples, run_number, nfirst, nlast, nreqevts, Stex);
  fCnaWrite->fMakeResultsFileName();
  //         names can be now recovered by call to TEcnaWrite methods: GetRootFileName() and GetRootFileNameShort()

  //------------------------- init Stin numbers memo flags
  fMemoStinNumbers = 0;

  if (fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments) {
    std::cout << std::endl;
    std::cout << "*TEcnaRead::FileParameters(...)>" << std::endl
              << "          The method has been called with the following argument values:" << std::endl
              << "          Analysis name                = " << fFileHeader->fTypAna << std::endl
              << "          Nb of required samples       = " << fFileHeader->fNbOfSamples << std::endl
              << "          Run number                   = " << fFileHeader->fRunNumber << std::endl
              << "          First requested event number = " << fFileHeader->fFirstReqEvtNumber << std::endl
              << "          Last requested event number  = " << fFileHeader->fLastReqEvtNumber << std::endl
              << "          Requested number of events   = " << fFileHeader->fReqNbOfEvts << std::endl
              << "          Stex number                  = " << fFileHeader->fStex << std::endl
              << "          Path for the ROOT file       = " << fPathRoot << std::endl
              << std::endl;
  }

  fReadyToReadRootFile = 1;  // set flag

}  //----------------- end of FileParameters(...)

//=========================================================================
//
//   GetAnalysisName, GetNbOfSamples, GetRunNumber, GetFirstReqEvtNumber
//   GetLastReqEvtNumber, GetReqNbOfEvts, GetStexNumber
//
//=========================================================================
TString TEcnaRead::GetAnalysisName() { return fFileHeader->fTypAna; }
Int_t TEcnaRead::GetNbOfSamples() { return fFileHeader->fNbOfSamples; }
Int_t TEcnaRead::GetRunNumber() { return fFileHeader->fRunNumber; }
Int_t TEcnaRead::GetFirstReqEvtNumber() { return fFileHeader->fFirstReqEvtNumber; }
Int_t TEcnaRead::GetLastReqEvtNumber() { return fFileHeader->fLastReqEvtNumber; }
Int_t TEcnaRead::GetReqNbOfEvts() { return fFileHeader->fReqNbOfEvts; }
Int_t TEcnaRead::GetStexNumber() { return fFileHeader->fStex; }
//=========================================================================
//
//     GetStartDate, GetStopDate, GetRunType
//
//=========================================================================
time_t TEcnaRead::GetStartTime() { return fFileHeader->fStartTime; }
time_t TEcnaRead::GetStopTime() { return fFileHeader->fStopTime; }
TString TEcnaRead::GetStartDate() { return fFileHeader->fStartDate; }
TString TEcnaRead::GetStopDate() { return fFileHeader->fStopDate; }
TString TEcnaRead::GetRunType() {
  TString cType = "run type not defined";
  Int_t numtype = fFileHeader->fRunType;
  //----------------------------------------- run types

  if (numtype == 0) {
    cType = "COSMICS";
  }
  if (numtype == 1) {
    cType = "BEAMH4";
  }
  if (numtype == 2) {
    cType = "BEAMH2";
  }
  if (numtype == 3) {
    cType = "MTCC";
  }
  if (numtype == 4) {
    cType = "LASER_STD";
  }
  if (numtype == 5) {
    cType = "LASER_POWER_SCAN";
  }
  if (numtype == 6) {
    cType = "LASER_DELAY_SCAN";
  }
  if (numtype == 7) {
    cType = "TESTPULSE_SCAN_MEM";
  }
  if (numtype == 8) {
    cType = "TESTPULSE_MGPA";
  }
  if (numtype == 9) {
    cType = "PEDESTAL_STD";
  }
  if (numtype == 10) {
    cType = "PEDESTAL_OFFSET_SCAN";
  }
  if (numtype == 11) {
    cType = "PEDESTAL_25NS_SCAN";
  }
  if (numtype == 12) {
    cType = "LED_STD";
  }

  if (numtype == 13) {
    cType = "PHYSICS_GLOBAL";
  }
  if (numtype == 14) {
    cType = "COSMICS_GLOBAL";
  }
  if (numtype == 15) {
    cType = "HALO_GLOBAL";
  }

  if (numtype == 16) {
    cType = "LASER_GAP";
  }
  if (numtype == 17) {
    cType = "TESTPULSE_GAP";
  }
  if (numtype == 18) {
    cType = "PEDESTAL_GAP";
  }
  if (numtype == 19) {
    cType = "LED_GAP";
  }

  if (numtype == 20) {
    cType = "PHYSICS_LOCAL";
  }
  if (numtype == 21) {
    cType = "COSMICS_LOCAL";
  }
  if (numtype == 22) {
    cType = "HALO_LOCAL";
  }
  if (numtype == 23) {
    cType = "CALIB_LOCAL";
  }

  if (numtype == 24) {
    cType = "PEDSIM";
  }

  return cType;
}
//==========================================================================
//
//                       R E A D    M E T H O D S
//                      (   R O O T    F I L E  )
//
//==========================================================================
//-------------------------------------------------------------
//
//                      OpenRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRead::OpenRootFile(const Text_t *name, const TString &status) {
  //Open the Root file

  Bool_t ok_open = kFALSE;

  TString s_name;
  s_name = fPathRoot;
  s_name.Append('/');
  s_name.Append(name);

  // if( gCnaRootFile != 0 )
  //  {
  //    Int_t iPointer = (Int_t)gCnaRootFile;
  //   std::cout << "*TEcnaRead::OpenRootFile(...)> RootFile pointer not (re)initialized to 0. gCnaRootFile = "
  //	   << gCnaRootFile << ", pointer =  " << iPointer << fTTBELL << std::endl;
  //
  //   delete gCnaRootFile; gCnaRootFile = 0;  Adelete("gCnaRootFile");
  //  }

  //if( gCnaRootFile != 0 ){gCnaRootFile->ReStart(s_name.Data(), status);}
  //if( gCnaRootFile == 0 )
  //  {
  // gCnaRootFile = new TEcnaRootFile(fObjectManager, s_name.Data(), status);  Anew("gCnaRootFile");

  Long_t iCnaRootFile = fObjectManager->GetPointerValue("TEcnaRootFile");
  if (iCnaRootFile == 0) {
    gCnaRootFile = new TEcnaRootFile(fObjectManager, s_name.Data(), status); /* Anew("gCnaRootFile");*/
  } else {
    gCnaRootFile = (TEcnaRootFile *)iCnaRootFile;
    gCnaRootFile->ReStart(s_name.Data(), status);
  }
  //  }

  if (gCnaRootFile->fRootFileStatus == "RECREATE") {
    ok_open = gCnaRootFile->OpenW();
  }
  if (gCnaRootFile->fRootFileStatus == "READ") {
    ok_open = gCnaRootFile->OpenR();
  }

  if (ok_open == kFALSE) {
    std::cout << "!TEcnaRead::OpenRootFile> " << s_name.Data() << ": file not found." << std::endl;
    //if( gCnaRootFile != 0 )
    //  {delete gCnaRootFile; gCnaRootFile = 0;  Adelete("gCnaRootFile");}
  } else {
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRead::OpenRootFile> Open ROOT file " << s_name.Data() << " OK "
                << ", gCnaRootFile = " << gCnaRootFile << std::endl;
    }
    fOpenRootFile = kTRUE;
    fCurrentlyOpenFileName = s_name;
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRead::OpenRootFile> Open ROOT file: " << fCurrentlyOpenFileName.Data() << " => OK "
                << ", gCnaRootFile = " << gCnaRootFile << std::endl
                << std::endl;
    }
  }
  return ok_open;
}  // end of OpenRootFile()

//-------------------------------------------------------------
//
//                      CloseRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRead::CloseRootFile(const Text_t *name) {
  //Close the Root file

  Bool_t ok_close = kFALSE;

  if (fOpenRootFile == kTRUE) {
    if (gCnaRootFile != nullptr) {
      gCnaRootFile->CloseFile();

      if (fFlagPrint == fCodePrintAllComments) {
        TString e_path;
        e_path.Append(name);
        std::cout << "*TEcnaRead::CloseRootFile> Close ROOT file " << e_path.Data() << " OK " << std::endl;
      }
      if (fFlagPrint == fCodePrintAllComments) {
        Long_t pointer_value = (Long_t)gCnaRootFile;
        std::cout << "*TEcnaRead::CloseRootFile(...)> going to delete gCnaRootFile, gCnaRootFile = " << gCnaRootFile
                  << ", pointer = " << pointer_value << std::endl;
      }

      //delete gCnaRootFile;   gCnaRootFile = 0;  Adelete("gCnaRootFile");

      ok_close = kTRUE;
      fOpenRootFile = kFALSE;
      fCurrentlyOpenFileName = fFlagNoFileOpen;
      fReadyToReadRootFile = 0;
    } else {
      std::cout << "*TEcnaRead::CloseRootFile(...)> RootFile pointer equal to zero. Close not possible. gCnaRootFile = "
                << gCnaRootFile << fTTBELL << std::endl;
    }
  } else {
    std::cout << "*TEcnaRead::CloseRootFile(...)> no close since no file is open. fOpenRootFile = " << fOpenRootFile
              << std::endl;
  }
  return ok_close;
}

//============================================================================
//
//                      LookAtRootFile()
//                   Called by TEcnaHistos
//
//============================================================================
Bool_t TEcnaRead::LookAtRootFile() {
  //---------- Reads the ROOT file header and makes allocations and some other things

  fLookAtRootFile = 0;  // set flag to zero before looking for the file

  if (fReadyToReadRootFile == 1) {
    //------------ Call to ReadRootFileHeader
    Int_t iprint = 0;
    if (ReadRootFileHeader(iprint) == kTRUE)  //    (1) = print, (0) = no print
    {
      //........................................ allocation tags
      if (fTagStinNumbers == nullptr) {
        fTagStinNumbers = new Int_t[1];
        Anew("fTagStinNumbers");
      }

      //...................... allocation for fT1d_StexStinFromIndex[]
      if (fT1d_StexStinFromIndex == nullptr) {
        fT1d_StexStinFromIndex = new Int_t[fEcal->MaxStinEcnaInStex()];
        Anew("fT1d_StexStinFromIndex");
      }

      //.. recover of the Stin numbers from the ROOT file (= init fT1d_StexStinFromIndex+init TagStin)
      TVectorD vec(fEcal->MaxStinEcnaInStex());
      for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
        vec(i) = (Double_t)0.;
      }
      vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

      for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
        fT1d_StexStinFromIndex[i] = (Int_t)vec(i);
      }

      fTagStinNumbers[0] = 1;
      fFileHeader->fStinNumbersCalc++;

      fLookAtRootFile = 1;  // set flag
      return kTRUE;
    } else {
      std::cout << "!TEcnaRead::LookAtRootFile()> *** ERROR ***>"
                << " ROOT file not found " << fTTBELL << std::endl;
      return kFALSE;
    }
  } else {
    std::cout << "!TEcnaRead::LookAtRootFile()> *** ERROR ***>"
              << " FileParameters not called " << fTTBELL << std::endl;
    return kFALSE;
  }
  return kFALSE;
}  //----------------- end of LookAtRootFile()
//-------------------------------------------------------------------------
//
//                     DataExist()
//
//     DON'T SUPPRESS: CALLED BY ANOTHER CLASSES
//
//-------------------------------------------------------------------------
Bool_t TEcnaRead::DataExist() {
  // return kTRUE if the data are present in the ROOT file, kFALSE if not.
  // fDataExist is set in the read methods

  return fDataExist;
}
//-------------------------------------------------------------------------
//
//                     ReadRootFileHeader
//
//-------------------------------------------------------------------------
Bool_t TEcnaRead::ReadRootFileHeader(const Int_t &i_print) {
  //Read the header of the Root file => test the file existence

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  if (i_print == 1) {
    std::cout << "*TEcnaRead::ReadRootFileHeader> file_name = " << fCnaWrite->fRootFileNameShort.Data() << std::endl;
  }

  Bool_t ok_open = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadRootFileHeader(...)*** ERROR ***> "
  //	   << "Reading header on file already open." << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");
    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadRootFileHeader(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    TEcnaHeader *headerFile;
    headerFile = (TEcnaHeader *)gCnaRootFile->fRootFile->Get("CnaHeader");

    //..... get the attributes which are not already set by the call to TEcnaHeader
    //      in FileParameters(...) and are only available in the ROOT file

    fFileHeader->fStartTime = headerFile->fStartTime;
    fFileHeader->fStopTime = headerFile->fStopTime;
    fFileHeader->fStartDate = headerFile->fStartDate;
    fFileHeader->fStopDate = headerFile->fStopDate;

    fFileHeader->fRunType = headerFile->fRunType;

    //....... Les f..Calc dans le header: pour acces direct a la taille des differentes data du fichier
    fFileHeader->fStinNumbersCalc = headerFile->fStinNumbersCalc;
    fFileHeader->fAdcEvtCalc = headerFile->fAdcEvtCalc;
    fFileHeader->fMSpCalc = headerFile->fMSpCalc;
    fFileHeader->fSSpCalc = headerFile->fSSpCalc;
    fFileHeader->fAvTnoCalc = headerFile->fAvTnoCalc;
    fFileHeader->fAvLfnCalc = headerFile->fAvLfnCalc;
    fFileHeader->fAvHfnCalc = headerFile->fAvHfnCalc;

    fFileHeader->fCovCssCalc = headerFile->fCovCssCalc;
    fFileHeader->fCorCssCalc = headerFile->fCorCssCalc;
    fFileHeader->fHfCovCalc = headerFile->fHfCovCalc;
    fFileHeader->fHfCorCalc = headerFile->fHfCorCalc;
    fFileHeader->fLfCovCalc = headerFile->fLfCovCalc;
    fFileHeader->fLfCorCalc = headerFile->fLfCorCalc;
    fFileHeader->fLFccMoStinsCalc = headerFile->fLFccMoStinsCalc;
    fFileHeader->fHFccMoStinsCalc = headerFile->fHFccMoStinsCalc;
    fFileHeader->fMeanCorssCalc = headerFile->fMeanCorssCalc;
    fFileHeader->fSigCorssCalc = headerFile->fSigCorssCalc;

    fFileHeader->fAvPedCalc = headerFile->fAvPedCalc;
    fFileHeader->fAvMeanCorssCalc = headerFile->fAvMeanCorssCalc;
    fFileHeader->fAvSigCorssCalc = headerFile->fAvSigCorssCalc;

    if (i_print == 1) {
      fFileHeader->Print();
    }

    CloseRootFile(file_name);
    return kTRUE;
  }
  return kFALSE;
}
//-------------------------------------------------------------------------
void TEcnaRead::TestArrayDimH1(const TString &CallingMethod,
                               const TString &MaxName,
                               const Int_t &MaxValue,
                               const Int_t &VecDim) {
  // array dim test

  if (MaxValue != VecDim) {
    std::cout << "!TEcnaRead::TestArrayDimH1(...)> No matching for array dimension: CallingMethod: "
              << CallingMethod.Data() << ", MaxName: " << MaxName.Data() << ", Maxvalue = " << MaxValue
              << ", VecDim = " << VecDim << fTTBELL << std::endl;
  }
#define NOPM
#ifndef NOPM
  else {
    std::cout << "!TEcnaRead::TestArrayDimH1(...)> matching array dimension: OK. CallingMethod: "
              << CallingMethod.Data() << ", MaxName: " << MaxName.Data() << ", Maxvalue = " << MaxValue
              << ", VecDim = " << VecDim << std::endl;
  }
#endif  // NOPM
}
//-------------------------------------------------------------------------
void TEcnaRead::TestArrayDimH2(const TString &CallingMethod,
                               const TString &MaxName,
                               const Int_t &MaxValue,
                               const Int_t &MatDim) {
  // array dim test

  if (MaxValue != MatDim) {
    std::cout << "!TEcnaRead::TestArrayDimH2(...)> No matching for array dimension: CallingMethod: "
              << CallingMethod.Data() << ", MaxName: " << MaxName.Data() << ", Maxvalue = " << MaxValue
              << ", MatDim = " << MatDim << fTTBELL << std::endl;
  }
#define NOPN
#ifndef NOPN
  else {
    std::cout << "!TEcnaRead::TestArrayDimH2(...)> matching array dimension: OK. CallingMethod: "
              << CallingMethod.Data() << ", MaxName: " << MaxName.Data() << ", Maxvalue = " << MaxValue
              << ", MatDim = " << MatDim << std::endl;
  }
#endif  // NOPN
}

//-------------------------------------------------------------------------
//
//                     ReadStinNumbers(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadStinNumbers(const Int_t &VecDim) {
  //Get the Stin numbers and put them in a TVectorD
  //Read the ROOT file at first call and load in a TVectorD attribute
  //Get directly the TVectorD attribute at other times
  //
  // Possible values for VecDim:
  //          (1) VecDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);

  TestArrayDimH1("ReadStinNumbers", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  if (fMemoStinNumbers == 0) {
    CnaResultTyp typ = cTypNumbers;
    const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
    const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

    //.............. reading of the ROOT file data type TResultTyp = cTypStinsNumbers
    //               to get the conversion: Stin index -> Stin number (n1StexStin)

    Bool_t ok_open = kFALSE;
    Bool_t ok_read = kFALSE;

    TString FileNameLong = fCnaWrite->GetRootFileName();
    Bool_t allowed_to_read = kFALSE;

    //      if ( fOpenRootFile )
    //	{
    //	  std::cout << "!TEcnaRead::ReadStinNumbers(...) *** ERROR ***> Reading on file already open."
    //	       << fTTBELL << std::endl;
    //	}

    if (FileNameLong == fCurrentlyOpenFileName) {
      allowed_to_read = kTRUE;
    } else {
      if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
        CloseRootFile(current_file_name);
      }
      ok_open = OpenRootFile(file_name, "READ");

      if (ok_open) {
        allowed_to_read = kTRUE;
      } else {
        std::cout << "!TEcnaRead::ReadStinNumbers(...) *** ERROR ***> Open .root file failed for file: " << file_name
                  << fTTBELL << std::endl;
        allowed_to_read = kFALSE;
      }
    }

    if (allowed_to_read == kTRUE) {
      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);

      if (ok_read == kTRUE) {
        fDataExist = kTRUE;
        //......... Get the Stin numbers and put them in TVectorD vec()
        for (Int_t i_Stin = 0; i_Stin < VecDim; i_Stin++) {
          vec(i_Stin) = gCnaRootFile->fCnaIndivResult->fMatHis(0, i_Stin);
          fT1d_StexStinFromIndex[i_Stin] = (Int_t)vec(i_Stin);
        }
        fMemoStinNumbers++;
      } else {
        fDataExist = kFALSE;
        std::cout << "!TEcnaRead::ReadStinNumbers(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                  << ": .root file failed" << std::endl
                  << "                                                -> quantity: <" << GetTypeOfQuantity(typ)
                  << "> not available in file." << fTTBELL << std::endl;
      }
      CloseRootFile(file_name);
    }

    if (ok_read == kTRUE) {
      //........................... Print the Stin numbers
      if (fFlagPrint == fCodePrintAllComments) {
        for (Int_t i = 0; i < VecDim; i++) {
          std::cout << "*TEcnaRead::ReadStinNumbers(...)> StinNumber[" << i << "] = " << vec[i] << std::endl;
        }
      }
    }
  } else {
    fDataExist = kTRUE;
    for (Int_t i_Stin = 0; i_Stin < VecDim; i_Stin++) {
      vec(i_Stin) = fT1d_StexStinFromIndex[i_Stin];
    }
  }
  return vec;
}  // ----------------- ( end of ReadStinNumbers(...) ) -----------------

//============================================================================
//
//                       1 D   H I S T O S   (TECHNICAL METHODS)
//
//============================================================================

//--------------------------------------------------------------------------------------
//
//                 ReadSampleAdcValues(i0StexEcha,sample,fFileHeader->fReqNbOfEvts)
//
//--------------------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleAdcValues(const Int_t &n1StexStin,
                                        const Int_t &i0StinEcha,
                                        const Int_t &sample,
                                        const Int_t &VecDim) {
  //Read the sample ADC values for each event for a given i0StexEcha and a given sample
  //in the results ROOT file and return it in a TVectorD(requested nb of events)
  //
  //Possible values for VecDim: (1) VecDim = fFileHeader->fReqNbOfEvts

  TestArrayDimH1("ReadSampleAdcValues", "fFileHeader->fReqNbOfEvts", fFileHeader->fReqNbOfEvts, VecDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAdcEvt;  //  sample as a function of time type

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //   {
  //     std::cout << "!TEcnaRead::ReadSampleAdcValues(...) *** ERROR ***> "
  // 	   << "Reading on file already open." << fTTBELL << std::endl;
  //   }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadSampleAdcValues(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_bin = 0; i_bin < VecDim; i_bin++) {
        vec(i_bin) = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadSampleAdcValues(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                 -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}
//--- (end of ReadSampleAdcValues) ----------

//-------------------------------------------------------------------------
//
//                  ReadSampleMeans
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleMeans(const Int_t &n1StexStin, const Int_t &i0StinEcha, const Int_t &VecDim) {
  //Read the expectation values of the samples
  //for a given Stin and a given channel
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples

  TestArrayDimH1("ReadSampleMeans", "fFileHeader->fNbOfSamples", fFileHeader->fNbOfSamples, VecDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypMSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  // if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> "
  // 	   << " Reading on file already open." << fTTBELL << std::endl;
  //  }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");
    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_samp = 0; i_samp < VecDim; i_samp++) {
        vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha, i_samp);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}
//------------------------------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleMeans(const Int_t &n1StexStin, const Int_t &VecDim) {
  //Read the expectation values of the samples
  //for all the channel of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples*fEcal->MaxCrysInStin()

  TestArrayDimH1("ReadSampleMeans",
                 "fFileHeader->fNbOfSamples*fEcal->MaxCrysInStin()",
                 fFileHeader->fNbOfSamples * fEcal->MaxCrysInStin(),
                 VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypMSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> "
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //  }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;

      for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
        Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
        for (Int_t i_samp = 0; i_samp < fFileHeader->fNbOfSamples; i_samp++) {
          vec(i0StinEcha * fFileHeader->fNbOfSamples + i_samp) =
              gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha, i_samp);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadSampleSigmas
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleSigmas(const Int_t &n1StexStin, const Int_t &i0StinEcha, const Int_t &VecDim) {
  //Read the expectation values of the samples
  //for a given Stin and a given channel
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples

  TestArrayDimH1("ReadSampleSigmas", "fFileHeader->fNbOfSamples", fFileHeader->fNbOfSamples, VecDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

  TVectorD vec(VecDim);
  vec.Zero();

  CnaResultTyp typ = cTypSSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  TString FileNameLong = fCnaWrite->GetRootFileName();

  //if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //  }

  if (FileNameLong != fCurrentlyOpenFileName) {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen)
      CloseRootFile(current_file_name);

    if (!(OpenRootFile(file_name, "READ"))) {
      std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      return vec;
    }
  }

  Int_t i_zero = 0;

  if (gCnaRootFile->ReadElement(typ, i_zero)) {
    fDataExist = kTRUE;
    for (Int_t i_samp = 0; i_samp < VecDim; i_samp++) {
      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha, i_samp);
    }
  } else {
    fDataExist = kFALSE;
    std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
              << ": .root file failed" << std::endl
              << "                                                 -> quantity: <" << GetTypeOfQuantity(typ)
              << "> not available in file." << fTTBELL << std::endl;
  }
  CloseRootFile(file_name);
  return vec;
}
//------------------------------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleSigmas(const Int_t &n1StexStin, const Int_t &VecDim) {
  //Read the expectation values of the samples
  //for all the channel of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples*fEcal->MaxCrysInStin()

  TestArrayDimH1("ReadSampleSigmas",
                 "fFileHeader->fNbOfSamples*fEcal->MaxCrysInStin()",
                 fFileHeader->fNbOfSamples * fEcal->MaxCrysInStin(),
                 VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypSSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  // }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;

      for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
        Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
        for (Int_t i_samp = 0; i_samp < fFileHeader->fNbOfSamples; i_samp++) {
          vec(i0StinEcha * fFileHeader->fNbOfSamples + i_samp) =
              gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha, i_samp);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                 -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-----------------------------------------------------------------------------
//
//                  ReadNumberOfEvents(...)
//
//-----------------------------------------------------------------------------
TVectorD TEcnaRead::ReadNumberOfEvents(const Int_t &VecDim) {
  //Read the numbers of found events in the data
  //for the crystals and for the samples for all the Stin's in the Stex
  //in the ROOT file, compute the average on the samples
  //and return them in a TVectorD(MaxCrysEcnaInStex)
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1("ReadNumberOfEvents", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  TMatrixD mat(fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);

  for (Int_t iStexStin = 0; iStexStin < fEcal->MaxStinEcnaInStex(); iStexStin++) {
    //............. set mat(,) to zero before reading it
    for (Int_t i = 0; i < fEcal->MaxCrysInStin(); i++) {
      for (Int_t j = 0; j < fFileHeader->fNbOfSamples; j++) {
        mat(i, j) = (Double_t)0.;
      }
    }
    //............. read mat(,)
    Int_t n1StexStin = iStexStin + 1;
    mat = ReadNumberOfEventsForSamples(n1StexStin, fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);

    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
      vec(i0StexEcha) = 0;
      //.... average value over the samples
      for (Int_t i_samp = 0; i_samp < fFileHeader->fNbOfSamples; i_samp++) {
        vec(i0StexEcha) += mat(i0StinEcha, i_samp);
      }
      vec(i0StexEcha) = vec(i0StexEcha) / fFileHeader->fNbOfSamples;
    }
  }
  return vec;
}

//-----------------------------------------------------------------------------
//
//                  ReadNumberOfEventsForSamples
//
//-----------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadNumberOfEventsForSamples(const Int_t &n1StexStin, const Int_t &MatDimX, const Int_t &MatDimY) {
  //Read the numbers of found events in the data
  //for the crystals and for the samples, for a given Stin in the Stex
  //in the ROOT file and return them in a TMatrixD(MaxCrysInStin,NbOfSamples)
  //
  //Possible values for MatDimX and MatDimY:
  //  (1) MatDimX = fEcal->MaxCrysInStin(), MatDimY = fFileHeader->fNbOfSamples

  TMatrixD mat(MatDimX, MatDimY);
  for (Int_t i = 0; i - MatDimX < 0; i++) {
    for (Int_t j = 0; j - MatDimY < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  Int_t Stin_index = GetStinIndex(n1StexStin);
  if (Stin_index >= 0) {
    if (fLookAtRootFile == 1) {
      CnaResultTyp typ = cTypNbOfEvts;
      const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
      const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

      Bool_t ok_open = kFALSE;
      Bool_t ok_read = kFALSE;

      TString FileNameLong = fCnaWrite->GetRootFileName();
      Bool_t allowed_to_read = kFALSE;

      //	  if ( fOpenRootFile )
      //	    {
      //	      std::cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
      //		   << " Reading on file already open." << fTTBELL << std::endl;
      //	    }

      if (FileNameLong == fCurrentlyOpenFileName) {
        allowed_to_read = kTRUE;
      } else {
        if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
          CloseRootFile(current_file_name);
        }
        ok_open = OpenRootFile(file_name, "READ");  // set fOpenRootFile to kTRUE
        if (ok_open) {
          allowed_to_read = kTRUE;
        } else {
          std::cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> Open .root file failed for file: "
                    << file_name << fTTBELL << std::endl;
          allowed_to_read = kFALSE;
        }
      }

      if (allowed_to_read == kTRUE) {
        Int_t i_zero = 0;
        ok_read = gCnaRootFile->ReadElement(typ, i_zero);

        if (ok_read == kTRUE) {
          fDataExist = kTRUE;
          for (Int_t i_crys = 0; i_crys - MatDimX < 0; i_crys++) {
            Int_t j_cna_chan = Stin_index * MatDimX + i_crys;
            for (Int_t i_samp = 0; i_samp - MatDimY < 0; i_samp++) {
              mat(i_crys, i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(j_cna_chan, i_samp);
            }
          }
        } else {
          fDataExist = kFALSE;
          std::cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
                    << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                    << "                                                                 -> quantity: <"
                    << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
        }
      }
      CloseRootFile(file_name);
    }  // end of if (fLookAtRootFile == 1)
    else {
      std::cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
                << "It is not possible to access the number of found events: the ROOT file has not been read."
                << fTTBELL << std::endl;
    }
  }  // end of if (Stin_index >= 0)
  return mat;
}  // ----------------- end of ReadNumberOfEventsForSamples(...)

//-------------------------------------------------------------------------
//
//        ReadPedestals(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadPedestals(const Int_t &VecDim) {
  //Read the expectation values of the expectation values of the samples
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1("ReadPedestals", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypPed;  // pedestals type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //    if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadPedestals(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");
    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadPedestals(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadPedestals(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                              -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadTotalNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadTotalNoise(const Int_t &VecDim) {
  //Read the expectation values of the sigmas of the samples
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1("ReadTotalNoise", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypTno;  // Total noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadTotalNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //  }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");
    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadTotalNoise(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadTotalNoise(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                               -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}
//-------------------------------------------------------------------------
//
//          ReadMeanCorrelationsBetweenSamples(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadMeanCorrelationsBetweenSamples(const Int_t &VecDim) {
  //Read the Expectation values of the (sample,sample) correlations
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1(
      "ReadMeanCorrelationsBetweenSamples", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypMeanCorss;  // mean corss type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //if ( fOpenRootFile )
  //  {
  //    std::cout << "!TEcnaRead::ReadMeanCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //  }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");
    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout
          << "!TEcnaRead::ReadMeanCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file failed for file: "
          << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadMeanCorrelationsBetweenSamples(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                   ->  quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadLowFrequencyNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadLowFrequencyNoise(const Int_t &VecDim) {
  //Read the sigmas of the expectation values of the samples
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1("ReadLowFrequencyNoise", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypLfn;  // low frequency noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyNoise(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyNoise(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                      -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadHighFrequencyNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadHighFrequencyNoise(const Int_t &VecDim) {
  //Read the sigmas of the sigmas of the samples
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1("ReadHighFrequencyNoise", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypHfn;  // high frequency noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyNoise(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyNoise(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                       -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadSigmaOfCorrelationsBetweenSamples(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(const Int_t &VecDim) {
  //Read the Expectation values of the (sample,sample) correlations
  //for all the channels of a given Stin
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH1(
      "ReadSigmaOfCorrelationsBetweenSamples", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypSigCorss;  // sigma of corss type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout
          << "!TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file failed for file: "
          << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++) {
        vec(i_StexCrys) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                      -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}
//==================================================================================================
//-----------------------------------------------------------------------------
//
//                  ReadAverageNumberOfEvents(...)
//
//       NB: read "direct" numbers of evts and compute the average HERE
//           (different from ReadAveragePedestals, Noises, etc...)
//
//-----------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageNumberOfEvents(const Int_t &VecDim) {
  //Read the numbers of found events in the data
  //for the crystals and for the samples for all the Stin's in the Stex
  //in the ROOT file, compute the average on the samples and on the crystals
  //and return them in a TVectorD(MaxStinEcnaInStex)
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1("ReadAverageNumberOfEvents", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vecAverage(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vecAverage(i) = (Double_t)0.;
  }

  TVectorD vecMean(fEcal->MaxCrysEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxCrysEcnaInStex(); i++) {
    vecMean(i) = (Double_t)0.;
  }

  vecMean = ReadNumberOfEvents(fEcal->MaxCrysEcnaInStex());

  for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
    vecAverage(i0StexStin) = 0;
    //.... average value over the crystals
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t n1StexStin = i0StexStin + 1;
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

      if (fStexName == "SM") {
        vecAverage(i0StexStin) += vecMean(i0StexEcha);
      }

      if (fStexName == "Dee") {
        //--------- EE --> Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStin == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStin == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStin == 29 || n1StexStin == 32) && n1StinEcha == 11)) {
          vecAverage(i0StexStin) += vecMean(i0StexEcha);
        }
      }
    }

    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      Int_t n1StexStin = i0StexStin + 1;
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStin, "TEcnaRead");
    }

    vecAverage(i0StexStin) = vecAverage(i0StexStin) / xdivis;
  }
  return vecAverage;
}

//-------------------------------------------------------------------------
//
//        ReadAveragePedestals(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragePedestals(const Int_t &VecDim) {
  //Read the expectation values of the Pedestals
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1("ReadAveragePedestals", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvPed;  // averaged pedestals type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAveragePedestals(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAveragePedestals(...) *** ERROR ***> Open .root file failed for file: " << file_name
                << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAveragePedestals(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                     -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}  // end of ReadAveragePedestals

//-------------------------------------------------------------------------
//
//        ReadAverageTotalNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageTotalNoise(const Int_t &VecDim) {
  //Read the expectation values of the Total Noise
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1("ReadAverageTotalNoise", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvTno;  // averaged Total Noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAverageTotalNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAverageTotalNoise(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAverageTotalNoise(...) *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                      -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}

//-------------------------------------------------------------------------
//
//        ReadAverageLowFrequencyNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageLowFrequencyNoise(const Int_t &VecDim) {
  //Read the expectation values of the Pedestals
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1("ReadAverageLowFrequencyNoise", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvLfn;  // averaged Low FrequencyNoise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAverageLowFrequencyNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAverageLowFrequencyNoise(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAverageLowFrequencyNoise(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                             -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}  // end of ReadAverageLowFrequencyNoise

//-------------------------------------------------------------------------
//
//        ReadAverageHighFrequencyNoise(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageHighFrequencyNoise(const Int_t &VecDim) {
  //Read the expectation values of the Pedestals
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1("ReadAverageHighFrequencyNoise", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvHfn;  // averaged High FrequencyNoise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAverageHighFrequencyNoise(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAverageHighFrequencyNoise(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAverageHighFrequencyNoise(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                              -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}  // end of ReadAverageHighFrequencyNoise

//-------------------------------------------------------------------------
//
//        ReadAverageMeanCorrelationsBetweenSamples(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageMeanCorrelationsBetweenSamples(const Int_t &VecDim) {
  //Read the expectation values of the Pedestals
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1(
      "ReadAverageMeanCorrelationsBetweenSamples", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvMeanCorss;  // averaged MeanCorrelationsBetweenSamples type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAverageMeanCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAverageMeanCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAverageMeanCorrelationsBetweenSamples(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                          -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}  // end of ReadAverageMeanCorrelationsBetweenSamples

//-------------------------------------------------------------------------
//
//        ReadAverageSigmaOfCorrelationsBetweenSamples(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAverageSigmaOfCorrelationsBetweenSamples(const Int_t &VecDim) {
  //Read the expectation values of the Pedestals
  //for all the Stins of a given Stex
  //in the ROOT file and return them in a TVectorD
  //
  //Possible values for VecDim: (1) VecDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH1(
      "ReadAverageSigmaOfCorrelationsBetweenSamples", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), VecDim);

  TVectorD vec(VecDim);
  for (Int_t i = 0; i < VecDim; i++) {
    vec(i) = (Double_t)0.;
  }

  CnaResultTyp typ = cTypAvSigCorss;  // averaged SigmaOfCorrelationsBetweenSamples type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadAverageSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadAverageSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file "
                   "failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++) {
        vec(i0StexStin) = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadAverageSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                             -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec;
}  // end of ReadAverageSigmaOfCorrelationsBetweenSamples

//============================================================================
//
//                       2 D   H I S T O S   (TECHNICAL METHODS)
//
//============================================================================
//-------------------------------------------------------------------------
//
//  ReadCovariancesBetweenSamples(n1StexStin,StinEcha,fFileHeader->fNbOfSamples)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadCovariancesBetweenSamples(const Int_t &n1StexStin,
                                                  const Int_t &i0StinEcha,
                                                  const Int_t &MatDim) {
  //Read the (sample,sample) covariances for a given channel
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fFileHeader->fNbOfSamples

  TestArrayDimH2("ReadCovariancesBetweenSamples", "fFileHeader->fNbOfSamples", fFileHeader->fNbOfSamples, MatDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypCovCss;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadCovariancesBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadCovariancesBetweenSamples(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_samp = 0; i_samp - MatDim < 0; i_samp++) {
        for (Int_t j_samp = 0; j_samp - MatDim < 0; j_samp++) {
          mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadCovariancesBetweenSamples() *** ERROR ***> " << fCnaWrite->fRootFileNameShort.Data()
                << ": .root file failed" << std::endl
                << "                                                           -> quantity: <" << GetTypeOfQuantity(typ)
                << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}

//-------------------------------------------------------------------------
//
//  ReadCorrelationsBetweenSamples(n1StexStin,StinEcha,fFileHeader->fNbOfSamples)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadCorrelationsBetweenSamples(const Int_t &n1StexStin,
                                                   const Int_t &i0StinEcha,
                                                   const Int_t &MatDim) {
  //Read the (sample,sample) correlations for a given channel
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fFileHeader->fNbOfSamples

  TestArrayDimH2("ReadCorrelationsBetweenSamples", "fFileHeader->fNbOfSamples", fFileHeader->fNbOfSamples, MatDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypCorCss;
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_samp = 0; i_samp - MatDim < 0; i_samp++) {
        for (Int_t j_samp = 0; j_samp - MatDim < 0; j_samp++) {
          mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadCorrelationsBetweenSamples() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                            -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//-------------------------------------------------------------------------
//
//  ReadRelevantCorrelationsBetweenSamples(n1StexStin,i0StinEcha)
//                 (NOT USED)
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadRelevantCorrelationsBetweenSamples(const Int_t &n1StexStin,
                                                           const Int_t &i0StinEcha,
                                                           const Int_t &InPutMatDim) {
  //Read the (sample,sample) correlations for a given channel
  //in ROOT file and return the relevant correlations in a TVectorD
  //
  //Possible values for InPutMatDim: (1) InPutMatDim = fFileHeader->fNbOfSamples
  //
  //  *===>  OutPut TVectorD dimension value = InPutMatDim*(InPutMatDim-1)/2

  TestArrayDimH2(
      "ReadRelevantCorrelationsBetweenSamples", "fFileHeader->fNbOfSamples", fFileHeader->fNbOfSamples, InPutMatDim);

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  Int_t nb_of_relevant = InPutMatDim * (InPutMatDim - 1) / 2;
  TVectorD vec_rel(nb_of_relevant);
  for (Int_t i = 0; i < nb_of_relevant; i++) {
    vec_rel(i) = (Double_t)0.;
  }
  CnaResultTyp typ = cTypCorCss;
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadRelevantCorrelationsBetweenSamples(...) *** ERROR ***> "
  //	   << "Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout
          << "!TEcnaRead::ReadRelevantCorrelationsBetweenSamples(...) *** ERROR ***> Open .root file failed for file: "
          << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      Int_t k_cor = 0;
      for (Int_t i_samp = 0; i_samp < InPutMatDim; i_samp++) {
        for (Int_t j_samp = 0; j_samp < i_samp; j_samp++) {
          vec_rel(k_cor) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp);
          k_cor++;
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadRelevantCorrelationsBetweenSamples() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                    -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return vec_rel;
}
//----- end of (ReadRelevantCorrelationsBetweenSamples ) -------

//-----------------------------------------------------------------------------------------
//
//        ReadLowFrequencyCovariancesBetweenChannels(Stin_a, Stin_b)
//
//-----------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(const Int_t &n1StexStin_a,
                                                               const Int_t &n1StexStin_b,
                                                               const Int_t &MatDim) {
  //Read the Low Frequency cov(i0StinEcha of Stin_a, i0StinEcha of Stin b)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  TestArrayDimH2(
      "ReadLowFrequencyCovariancesBetweenChannels", "fEcal->MaxCrysInStin()", fEcal->MaxCrysInStin(), MatDim);

  Int_t index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypLfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_crys = 0; i_crys - MatDim < 0; i_crys++) {
        Int_t i_cna_chan = index_Stin_a * MatDim + i_crys;
        for (Int_t j_crys = 0; j_crys - MatDim < 0; j_crys++) {
          Int_t j_cna_chan = index_Stin_b * MatDim + j_crys;
          mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                           -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadLowFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------------------------
//
//         ReadLowFrequencyCorrelationsBetweenChannels(Stin_a, Stin_b)
//
//-------------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(const Int_t &n1StexStin_a,
                                                                const Int_t &n1StexStin_b,
                                                                const Int_t &MatDim) {
  //Read the Low Frequency cor(i0StinEcha of Stin_a, i0StinEcha of Stin b)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  TestArrayDimH2(
      "ReadLowFrequencyCorrelationsBetweenChannels", "fEcal->MaxCrysInStin()", fEcal->MaxCrysInStin(), MatDim);

  Int_t index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypLfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;

    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_crys = 0; i_crys - MatDim < 0; i_crys++) {
        Int_t i_cna_chan = index_Stin_a * MatDim + i_crys;
        for (Int_t j_crys = 0; j_crys - MatDim < 0; j_crys++) {
          Int_t j_cna_chan = index_Stin_b * MatDim + j_crys;
          mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                            -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadLowFrequencyCorrelationsBetweenChannels(...) ) -------

//-----------------------------------------------------------------------------------------
//
//        ReadHighFrequencyCovariancesBetweenChannels(Stin_a, Stin_b)
//
//-----------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(const Int_t &n1StexStin_a,
                                                                const Int_t &n1StexStin_b,
                                                                const Int_t &MatDim) {
  //Read the High Frequency cov(i0StinEcha of Stin_a, i0StinEcha of Stin b)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  TestArrayDimH2(
      "ReadHighFrequencyCovariancesBetweenChannels", "fEcal->MaxCrysInStin()", fEcal->MaxCrysInStin(), MatDim);

  Int_t index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypHfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_crys = 0; i_crys - MatDim < 0; i_crys++) {
        Int_t i_cna_chan = index_Stin_a * MatDim + i_crys;
        for (Int_t j_crys = 0; j_crys - MatDim < 0; j_crys++) {
          Int_t j_cna_chan = index_Stin_b * MatDim + j_crys;
          mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                            -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadHighFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------------------------
//
//         ReadHighFrequencyCorrelationsBetweenChannels(Stin_a, Stin_b)
//
//-------------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(const Int_t &n1StexStin_a,
                                                                 const Int_t &n1StexStin_b,
                                                                 const Int_t &MatDim) {
  //Read the High Frequency Cor(i0StinEcha of Stin_a, i0StinEcha of Stin b)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  TestArrayDimH2(
      "ReadHighFrequencyCorrelationsBetweenChannels", "fEcal->MaxCrysInStin()", fEcal->MaxCrysInStin(), MatDim);

  Int_t index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  CnaResultTyp typ = cTypHfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> Open .root file "
                   "failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;

    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t i_crys = 0; i_crys - MatDim < 0; i_crys++) {
        Int_t i_cna_chan = index_Stin_a * MatDim + i_crys;
        for (Int_t j_crys = 0; j_crys - MatDim < 0; j_crys++) {
          Int_t j_cna_chan = index_Stin_b * MatDim + j_crys;
          mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                             -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadHighFrequencyCorrelationsBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadLowFrequencyCovariancesBetweenChannels(...)
//                  (NOT USED)
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(const Int_t &MatDim) {
  //Read all the Low Frequency covariances
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH2(
      "ReadLowFrequencyCovariancesBetweenChannels", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), MatDim);

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypLfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++) {
        if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex()) {
          for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex()) {
              for (Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++) {
                Int_t i_cna_chan = index_Stin_a * fEcal->MaxCrysInStin() + i_crys;
                Int_t i_chan_sm = (Int_t)(vec(index_Stin_a) - 1) * fEcal->MaxCrysInStin() + i_crys;
                for (Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++) {
                  Int_t j_cna_chan = index_Stin_b * fEcal->MaxCrysInStin() + j_crys;
                  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b) - 1) * fEcal->MaxCrysInStin() + j_crys;
                  mat(i_chan_sm, j_chan_sm) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
                }
              }
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                        -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadLowFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadLowFrequencyCorrelationsBetweenChannels(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(const Int_t &MatDim) {
  //Read all the Low Frequency correlations
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH2(
      "ReadLowFrequencyCorrelationsBetweenChannels", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), MatDim);

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypLfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++) {
        if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex()) {
          for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex()) {
              for (Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++) {
                Int_t i_cna_chan = index_Stin_a * fEcal->MaxCrysInStin() + i_crys;
                Int_t i_chan_sm = (Int_t)(vec(index_Stin_a) - 1) * fEcal->MaxCrysInStin() + i_crys;
                for (Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++) {
                  Int_t j_cna_chan = index_Stin_b * fEcal->MaxCrysInStin() + j_crys;
                  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b) - 1) * fEcal->MaxCrysInStin() + j_crys;
                  mat(i_chan_sm, j_chan_sm) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
                }
              }
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                         ->  quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of (ReadLowFrequencyCorrelationsBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadHighFrequencyCovariancesBetweenChannels(...)
//                  (NOT USED)
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(const Int_t &MatDim) {
  //Read all the High Frequency covariances
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH2(
      "ReadHighFrequencyCovariancesBetweenChannels", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), MatDim);

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypHfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***> Open .root file failed "
                   "for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++) {
        if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex()) {
          for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex()) {
              for (Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++) {
                Int_t i_cna_chan = index_Stin_a * fEcal->MaxCrysInStin() + i_crys;
                Int_t i_chan_sm = (Int_t)(vec(index_Stin_a) - 1) * fEcal->MaxCrysInStin() + i_crys;
                for (Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++) {
                  Int_t j_cna_chan = index_Stin_b * fEcal->MaxCrysInStin() + j_crys;
                  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b) - 1) * fEcal->MaxCrysInStin() + j_crys;
                  mat(i_chan_sm, j_chan_sm) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
                }
              }
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                         -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//----- end of ( ReadHighFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadHighFrequencyCorrelationsBetweenChannels(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(const Int_t &MatDim) {
  //Read all the High Frequency correlations
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TestArrayDimH2(
      "ReadHighFrequencyCorrelationsBetweenChannels", "fEcal->MaxCrysEcnaInStex()", fEcal->MaxCrysEcnaInStex(), MatDim);

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypHfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> Open .root file "
                   "failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++) {
        if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex()) {
          for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex()) {
              for (Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++) {
                Int_t i_cna_chan = index_Stin_a * fEcal->MaxCrysInStin() + i_crys;
                Int_t i_chan_sm = (Int_t)(vec(index_Stin_a) - 1) * fEcal->MaxCrysInStin() + i_crys;
                for (Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++) {
                  Int_t j_cna_chan = index_Stin_b * fEcal->MaxCrysInStin() + j_crys;
                  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b) - 1) * fEcal->MaxCrysInStin() + j_crys;
                  mat(i_chan_sm, j_chan_sm) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan, j_cna_chan);
                }
              }
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                          -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//-------------- ( end of ReadHighFrequencyCorrelationsBetweenChannels(...) ) ---------

//-------------------------------------------------------------------------
//
//         ReadLowFrequencyMeanCorrelationsBetweenStins(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins(const Int_t &MatDim) {
  //Read all the Low Frequency Mean Correlations Between Stins the for all (Stin_X, Stin_Y)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH2(
      "ReadLowFrequencyMeanCorrelationsBetweenStins", "fEcal->MaxStinEcnaInStex()", fEcal->MaxStinEcnaInStex(), MatDim);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypLFccMoStins;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins(...) *** ERROR ***> Open .root file "
                   "failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a - MatDim < 0; index_Stin_a++) {
        for (Int_t index_Stin_b = 0; index_Stin_b - MatDim < 0; index_Stin_b++) {
          if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= MatDim) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= MatDim) {
              Int_t vec_ia_m = (Int_t)vec(index_Stin_a) - 1;
              Int_t vec_ib_m = (Int_t)vec(index_Stin_b) - 1;
              mat((Int_t)vec_ia_m, vec_ib_m) = gCnaRootFile->fCnaIndivResult->fMatMat(index_Stin_a, index_Stin_b);
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                          -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//-------- ( end of ReadLowFrequencyMeanCorrelationsBetweenStins) --------

//-------------------------------------------------------------------------
//
//         ReadHighFrequencyMeanCorrelationsBetweenStins(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins(const Int_t &MatDim) {
  //Read all the High Frequency Mean Correlations Between Stins the for all (Stin_X, Stin_Y)
  //in ROOT file and return them in a TMatrixD
  //
  //Possible values for MatDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TestArrayDimH2("ReadHighFrequencyMeanCorrelationsBetweenStins",
                 "fEcal->MaxStinEcnaInStex()",
                 fEcal->MaxStinEcnaInStex(),
                 MatDim);

  TMatrixD mat(MatDim, MatDim);
  for (Int_t i = 0; i - MatDim < 0; i++) {
    for (Int_t j = 0; j - MatDim < 0; j++) {
      mat(i, j) = (Double_t)0.;
    }
  }

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypHFccMoStins;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
  const Text_t *current_file_name = (const Text_t *)fCurrentlyOpenFileName.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  TString FileNameLong = fCnaWrite->GetRootFileName();
  Bool_t allowed_to_read = kFALSE;

  //  if ( fOpenRootFile )
  //    {
  //      std::cout << "!TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins() *** ERROR ***>"
  //	   << " Reading on file already open." << fTTBELL << std::endl;
  //    }

  if (FileNameLong == fCurrentlyOpenFileName) {
    allowed_to_read = kTRUE;
  } else {
    if (fCurrentlyOpenFileName != fFlagNoFileOpen) {
      CloseRootFile(current_file_name);
    }
    ok_open = OpenRootFile(file_name, "READ");

    if (ok_open) {
      allowed_to_read = kTRUE;
    } else {
      std::cout << "!TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins(...) *** ERROR ***> Open .root file "
                   "failed for file: "
                << file_name << fTTBELL << std::endl;
      allowed_to_read = kFALSE;
    }
  }

  if (allowed_to_read == kTRUE) {
    Int_t i_zero = 0;
    ok_read = gCnaRootFile->ReadElement(typ, i_zero);

    if (ok_read == kTRUE) {
      fDataExist = kTRUE;
      for (Int_t index_Stin_a = 0; index_Stin_a - MatDim < 0; index_Stin_a++) {
        for (Int_t index_Stin_b = 0; index_Stin_b - MatDim < 0; index_Stin_b++) {
          if (vec(index_Stin_a) > 0 && vec(index_Stin_a) <= MatDim) {
            if (vec(index_Stin_b) > 0 && vec(index_Stin_b) <= MatDim) {
              Int_t vec_ia_m = (Int_t)vec(index_Stin_a) - 1;
              Int_t vec_ib_m = (Int_t)vec(index_Stin_b) - 1;
              mat((Int_t)vec_ia_m, (Int_t)vec_ib_m) =
                  gCnaRootFile->fCnaIndivResult->fMatMat(index_Stin_a, index_Stin_b);
            }
          }
        }
      }
    } else {
      fDataExist = kFALSE;
      std::cout << "!TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins() *** ERROR ***> "
                << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                << "                                                                           -> quantity: <"
                << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
    }
    CloseRootFile(file_name);
  }
  return mat;
}
//-------- ( end of ReadHighFrequencyMeanCorrelationsBetweenStins) --------

//=============================================================================
//
//             M I S C E L L A N E O U S     R E A D      M E T H O D S
//
//=============================================================================

//--------------------------------------------------------------------------------------
//
//       ReadSampleAdcValuesSameFile(fEcal->MaxCrysEcnaInStex(),
//                                   fFileHeader->fNbOfSamples, fFileHeader->fReqNbOfEvts)
//
//--------------------------------------------------------------------------------------
Double_t ***TEcnaRead::ReadSampleAdcValuesSameFile(const Int_t &DimX, const Int_t &DimY, const Int_t &DimZ) {
  //Possible values for DimX, DimY, DimZ : (1) DimX = fEcal->MaxCrysEcnaInStex()
  //                                           DimY = fFileHeader->fNbOfSamples
  //                                           DimZ = fFileHeader->fReqNbOfEvts

  if (fT3d_AdcValues == nullptr) {
    //............ Allocation for the 3d array
    fT3d_AdcValues = new Double_t **[DimX];
    fCnew++;
    fT3d2_AdcValues = new Double_t *[DimX * DimY];
    fCnew++;
    fT3d1_AdcValues = new Double_t[DimX * DimY * DimZ];
    fCnew++;

    for (Int_t i0StexEcha = 0; i0StexEcha < DimX; i0StexEcha++) {
      fT3d_AdcValues[i0StexEcha] = &fT3d2_AdcValues[0] + i0StexEcha * DimY;
      for (Int_t j_samp = 0; j_samp < DimY; j_samp++) {
        fT3d2_AdcValues[DimY * i0StexEcha + j_samp] = &fT3d1_AdcValues[0] + DimZ * (DimY * i0StexEcha + j_samp);
      }
    }
  }

  //................................. Init to zero                 (ReadSampleAdcValuesSameFile)
  for (Int_t iza = 0; iza < DimX; iza++) {
    for (Int_t izb = 0; izb < DimY; izb++) {
      for (Int_t izc = 0; izc < DimZ; izc++) {
        if (fT3d_AdcValues[iza][izb][izc] != (Double_t)0) {
          fT3d_AdcValues[iza][izb][izc] = (Double_t)0;
        }
      }
    }
  }

  //-------------------------------------------------------------------------- (ReadSampleAdcValuesSameFile)
  CnaResultTyp typ = cTypAdcEvt;  //  sample as a function of time type

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  Int_t i_entry = 0;
  Int_t i_entry_fail = 0;

  ok_open = OpenRootFile(file_name, "READ");

  if (ok_open == kTRUE) {
    for (Int_t i0StexEcha = 0; i0StexEcha < DimX; i0StexEcha++) {
      if (i0StexEcha == 0) {
        i_entry = gCnaRootFile->ReadElementNextEntryNumber(typ, i0StexEcha);
        if (i_entry >= 0) {
          ok_read = kTRUE;
        }
      }
      if (i_entry >= 0)  //  (ReadSampleAdcValuesSameFile)
      {
        if (i0StexEcha > 0) {
          ok_read = gCnaRootFile->ReadElement(i_entry);
          i_entry++;
        }

        if (ok_read == kTRUE) {
          fDataExist = kTRUE;
          for (Int_t sample = 0; sample < DimY; sample++) {
            for (Int_t i_bin = 0; i_bin < DimZ; i_bin++) {
              fT3d_AdcValues[i0StexEcha][sample][i_bin] = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
            }
          }
        } else  //  (ReadSampleAdcValuesSameFile)
        {
          fDataExist = kFALSE;
          std::cout << "!TEcnaRead::ReadSampleAdcValuesSameFile(...) *** ERROR ***> "
                    << fCnaWrite->fRootFileNameShort.Data() << ": .root file failed" << std::endl
                    << "                                                         -> quantity: <"
                    << GetTypeOfQuantity(typ) << "> not available in file." << fTTBELL << std::endl;
        }
      } else {
        i_entry_fail++;
      }
    }
    CloseRootFile(file_name);
  } else {
    std::cout << "*TEcnaRead::ReadSampleAdcValuesSameFile(...)> *ERROR* =====> "
              << " ROOT file not found" << fTTBELL << std::endl;
  }

  if (i_entry_fail > 0) {
    std::cout << "*TEcnaRead::ReadSampleAdcValuesSameFile(...)> *ERROR* =====> "
              << " Entry reading failure(s). i_entry_fail = " << i_entry_fail << fTTBELL << std::endl;
  }
  return fT3d_AdcValues;
}
//--- (end of ReadSampleAdcValuesSameFile) ----------

//=========================================================================
//
//          M I S C E L L A N E O U S    G E T    M E T H O D S
//
//=========================================================================
Int_t TEcnaRead::GetNumberOfEvents(const Int_t &xFapNbOfReqEvts, const Int_t &xStexNumber) {
  //...... Calculate the number of found events  (file existence already tested in calling method)
  Int_t xFapNbOfEvts = 0;

  TVectorD NOFE_histp(fEcal->MaxCrysEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxCrysEcnaInStex(); i++) {
    NOFE_histp(i) = (Double_t)0.;
  }
  NOFE_histp = ReadNumberOfEvents(fEcal->MaxCrysEcnaInStex());

  //... Call to fCnaWrite->NumberOfEventsAnalysis(...) 1rst argument must be Int_t, not TVectorD,
  //    duplicate NOFE_histp to NOFE_int to obtain fFapNbOfEvts from fCnaWrite->NumberOfEventsAnalysis(...)
  Int_t *NOFE_int = new Int_t[fEcal->MaxCrysEcnaInStex()];
  fCnew++;
  for (Int_t i = 0; i < fEcal->MaxCrysEcnaInStex(); i++) {
    NOFE_int[i] = (Int_t)NOFE_histp(i);
  }

  xFapNbOfEvts = fCnaWrite->NumberOfEventsAnalysis(NOFE_int, fEcal->MaxCrysEcnaInStex(), xFapNbOfReqEvts, xStexNumber);

  delete[] NOFE_int;
  NOFE_int = nullptr;
  fCdelete++;

  return xFapNbOfEvts;
}

//-------------------------------------------------------------------------
//
//    Get the name of the quantity from its "CnaResultTyp" type
//
//-------------------------------------------------------------------------
TString TEcnaRead::GetTypeOfQuantity(const CnaResultTyp arg_typ) {
  TString quantity_name = "?";

  if (arg_typ == cTypNumbers) {
    if (fFlagSubDet == "EB") {
      quantity_name = "SM numbers";
    }
    if (fFlagSubDet == "EE") {
      quantity_name = "Dee numbers";
    }
  }
  if (arg_typ == cTypMSp) {
    quantity_name = "Mean samples";
  }
  if (arg_typ == cTypSSp) {
    quantity_name = "Sigma of samples";
  }

  if (arg_typ == cTypNbOfEvts) {
    quantity_name = "Number of events";
  }
  if (arg_typ == cTypPed) {
    quantity_name = "Pedestals";
  }
  if (arg_typ == cTypTno) {
    quantity_name = "Total noise";
  }
  if (arg_typ == cTypLfn) {
    quantity_name = "LF noise";
  }
  if (arg_typ == cTypHfn) {
    quantity_name = "HF noise";
  }
  if (arg_typ == cTypMeanCorss) {
    quantity_name = "Mean cor(s,s')";
  }
  if (arg_typ == cTypSigCorss) {
    quantity_name = "Sigma of cor(s,s')";
  }

  if (arg_typ == cTypAvPed) {
    quantity_name = "Average pedestals";
  }
  if (arg_typ == cTypAvTno) {
    quantity_name = "Average total noise";
  }
  if (arg_typ == cTypAvLfn) {
    quantity_name = "Average LF noise";
  }
  if (arg_typ == cTypAvHfn) {
    quantity_name = "Average HF noise";
  }
  if (arg_typ == cTypAvMeanCorss) {
    quantity_name = "Average mean cor(s,s')";
  }
  if (arg_typ == cTypAvSigCorss) {
    quantity_name = "Average sigma of cor(s,s')";
  }

  if (arg_typ == cTypAdcEvt) {
    quantity_name = "Sample ADC a.f.o event number";
  }

  if (arg_typ == cTypCovCss) {
    quantity_name = "Cov(s,s')";
  }
  if (arg_typ == cTypCorCss) {
    quantity_name = "Cor(s,s')";
  }
  if (arg_typ == cTypLfCov) {
    quantity_name = "LF Cov(c,c')";
  }
  if (arg_typ == cTypLfCor) {
    quantity_name = "LF Cor(c,c')";
  }
  if (arg_typ == cTypHfCov) {
    quantity_name = "HF Cov(c,c')";
  }
  if (arg_typ == cTypHfCor) {
    quantity_name = "HF Cor(c,c')";
  }

  if (fFlagSubDet == "EB") {
    if (arg_typ == cTypLFccMoStins) {
      quantity_name = "Mean LF |Cor(c,c')| in (tow,tow')";
    }
    if (arg_typ == cTypHFccMoStins) {
      quantity_name = "Mean HF |Cor(c,c')| in (tow,tow')";
    }
  }
  if (fFlagSubDet == "EE") {
    if (arg_typ == cTypLFccMoStins) {
      quantity_name = "Mean LF |Cor(c,c')| in (SC,SC')";
    }
    if (arg_typ == cTypHFccMoStins) {
      quantity_name = "Mean HF |Cor(c,c')| in (SC,SC')";
    }
  }
  return quantity_name;
}

//-------------------------------------------------------------------------
//
//    Get the ROOT file name (long and short)
//
//-------------------------------------------------------------------------
TString TEcnaRead::GetRootFileName() { return fCnaWrite->GetRootFileName(); }
TString TEcnaRead::GetRootFileNameShort() { return fCnaWrite->GetRootFileNameShort(); }
//-------------------------------------------------------------------------
//
//                     GetStexStinFromIndex
//
//  *==> DON'T SUPPRESS: this method is called by TEcnaRun and TEcnaHistos
//
//-------------------------------------------------------------------------
Int_t TEcnaRead::GetStexStinFromIndex(const Int_t &i0StexStinEcna) {
  // Get the Stin number in Stex from the Stin index

  Int_t number = -1;
  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  number = (Int_t)vec(i0StexStinEcna);
  return number;
}

//------------------------------------------------------------------------

Int_t TEcnaRead::GetNumberOfBinsSampleAsFunctionOfTime() { return GetReqNbOfEvts(); }
//-------------------------------------------------------------------------
//
//                     GetStinIndex(n1StexStin)
//
//-------------------------------------------------------------------------
Int_t TEcnaRead::GetStinIndex(const Int_t &n1StexStin) {
  //Get the index of the Stin from its number in Stex

  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "*TEcnaRead::GetStinIndex(...)> fEcal->MaxStinEcnaInStex() = " << fEcal->MaxStinEcnaInStex()
              << std::endl
              << "                              n1StexStin = " << n1StexStin << std::endl
              << std::endl;
  }

  Int_t Stin_index = n1StexStin - 1;  // suppose les 68 tours

#define NOGT
#ifndef NOGT
  Int_t Stin_index = -1;
  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    vec(i) = (Double_t)0.;
  }
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  //........................... Get the Stin index

  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRead::GetStinIndex(...)> StinNumber[" << i << "] = " << vec[i] << std::endl;
    }
    if (vec[i] == n1StexStin) {
      Stin_index = i;
    }
  }

  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << std::endl;
    std::cout << "*TEcnaRead::GetStinIndex> Stin number: " << n1StexStin << std::endl
              << "                          Stin index : " << Stin_index << std::endl;
    std::cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << std::endl;
  }

  if (Stin_index < 0) {
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "!TEcnaRead::GetStinIndex *** WARNING ***> n1StexStin" << n1StexStin << " : "
                << "index Stin not found" << fTTBELL << std::endl;
    }
  }
#endif  // NOGT

  return Stin_index;
}

//=========================================================================
//
//         METHODS TO SET FLAGS TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void TEcnaRead::PrintComments() {
  // Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  std::cout << "*TEcnaRead::PrintComments()> Warnings and some comments on init will be printed" << std::endl;
}

void TEcnaRead::PrintWarnings() {
  // Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  std::cout << "*TEcnaRead::PrintWarnings()> Warnings will be printed" << std::endl;
}

void TEcnaRead::PrintAllComments() {
  // Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  std::cout << "*TEcnaRead::PrintAllComments()> All the comments will be printed" << std::endl;
}

void TEcnaRead::PrintNoComment() {
  // Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}
