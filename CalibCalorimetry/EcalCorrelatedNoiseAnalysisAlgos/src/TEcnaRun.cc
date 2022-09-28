//----------Author's Names: B.Fabbro, FX Gentit DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 30/01/2014

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRun.h"

//--------------------------------------
//  TEcnaRun.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaRun.h
//--------------------------------------

R__EXTERN TEcnaRootFile* gCnaRootFile;

ClassImp(TEcnaRun);
//___________________________________________________________________________
//

TEcnaRun::TEcnaRun() {
  //Constructor without argument: nothing special done

  // std::cout << "[Info Management] CLASS: TEcnaRun.           CREATE OBJECT: this = " << this << std::endl;
}

TEcnaRun::TEcnaRun(TEcnaObject* pObjectManager, const TString& SubDet) {
  //Constructor with argument: call to Init() and declare fEcal according to SubDet value ("EB" or "EE")

  // std::cout << "[Info Management] CLASS: TEcnaRun.           CREATE OBJECT: this = " << this << std::endl;

  Init(pObjectManager);

  //fCfgResultsRootFilePath    = fCnaParPaths->ResultsRootFilePath();
  //fCfgHistoryRunListFilePath = fCnaParPaths->HistoryRunListFilePath();

  //ffFileHeader = 0;
  //fconst Text_t *h_name  = "CnaHeader";   //==> voir cette question avec FXG
  //fconst Text_t *h_title = "CnaHeader";   //==> voir cette question avec FXG
  //ffFileHeader = new TEcnaHeader(h_name, h_title);     //fCnew++;

  //............................ fFileHeader
  const Text_t* h_name = "CnaHeader";   //==> voir cette question avec FXG
  const Text_t* h_title = "CnaHeader";  //==> voir cette question avec FXG

  fFileHeader = nullptr;
  //Int_t iFileHeader = pObjectManager->GetPointerValue("TEcnaHeader");
  Long_t iFileHeader = 0;  // one TEcnaHeader object for each file since they can be open simultaneously
  if (iFileHeader == 0) {
    fFileHeader = new TEcnaHeader(pObjectManager, h_name, h_title); /*fCnew++*/
  } else {
    fFileHeader = (TEcnaHeader*)iFileHeader;
  }

  SetEcalSubDetector(SubDet.Data());
  fNbSampForFic = fEcal->MaxSampADC();  // DEFAULT Number of samples for ROOT file
}

TEcnaRun::TEcnaRun(TEcnaObject* pObjectManager, const TString& SubDet, const Int_t& NbOfSamples) {
  Init(pObjectManager);

  //fCfgResultsRootFilePath    = fCnaParPaths->ResultsRootFilePath();
  //fCfgHistoryRunListFilePath = fCnaParPaths->HistoryRunListFilePath();

  //............................ fFileHeader
  const Text_t* h_name = "CnaHeader";   //==> voir cette question avec FXG
  const Text_t* h_title = "CnaHeader";  //==> voir cette question avec FXG

  fFileHeader = nullptr;
  //Long_t iFileHeader = pObjectManager->GetPointerValue("TEcnaHeader");
  Long_t iFileHeader = 0;  // one TEcnaHeader object for each file since they can be open simultaneously
  if (iFileHeader == 0) {
    fFileHeader = new TEcnaHeader(pObjectManager, h_name, h_title); /*fCnew++*/
  } else {
    fFileHeader = (TEcnaHeader*)iFileHeader;
  }

  SetEcalSubDetector(SubDet.Data());
  if (NbOfSamples > 0 && NbOfSamples <= fEcal->MaxSampADC()) {
    fNbSampForFic = NbOfSamples;
  } else {
    std::cout << "TEcnaRun/CONSTRUCTOR> Number of required samples = " << NbOfSamples
              << ": OUT OF RANGE. Set to the default value (= " << fEcal->MaxSampADC() << ")." << fTTBELL << std::endl;
    fNbSampForFic = fEcal->MaxSampADC();  // DEFAULT Number of samples for file reading
  }
}

//.... return true or false according to the existence of the path. The path itself is in an attribute of fCnaParPaths.
Bool_t TEcnaRun::GetPathForResults() { return fCnaParPaths->GetPathForResultsRootFiles(); }

void TEcnaRun::Init(TEcnaObject* pObjectManager) {
  //Initialisation

  fCnew = 0;
  fCdelete = 0;
  fCnaCommand = 0;
  fCnaError = 0;

  fTTBELL = '\007';

  //........................... TString file names init
  fgMaxCar = (Int_t)512;

  //................ MiscDiag counters .................
  fMaxMsgIndexForMiscDiag = (Int_t)10;
  fNbOfMiscDiagCounters = (Int_t)50;
  fMiscDiag = nullptr;

  fNumberOfEvents = 0;
  //............................. init pointers  ( Init() )
  fT3d_AdcValues = nullptr;
  fT3d2_AdcValues = nullptr;
  fT3d1_AdcValues = nullptr;

  fT1d_StexStinFromIndex = nullptr;

  fT2d_NbOfEvts = nullptr;
  fT1d_NbOfEvts = nullptr;

  fT2d_ev = nullptr;
  fT1d_ev = nullptr;
  fT2d_sig = nullptr;
  fT1d_sig = nullptr;

  fT3d_cov_ss = nullptr;
  fT3d2_cov_ss = nullptr;
  fT3d1_cov_ss = nullptr;

  fT3d_cor_ss = nullptr;
  fT3d2_cor_ss = nullptr;
  fT3d1_cor_ss = nullptr;

  fT2d_lf_cov = nullptr;
  fT2d1_lf_cov = nullptr;

  fT2d_lf_cor = nullptr;
  fT2d1_lf_cor = nullptr;

  fT2d_hf_cov = nullptr;
  fT2d1_hf_cov = nullptr;

  fT2d_hf_cor = nullptr;
  fT2d1_hf_cor = nullptr;

  fT2d_lfcc_mostins = nullptr;
  fT2d1_lfcc_mostins = nullptr;

  fT2d_hfcc_mostins = nullptr;
  fT2d1_hfcc_mostins = nullptr;

  fT1d_ev_ev = nullptr;
  fT1d_evsamp_of_sigevt = nullptr;
  fT1d_ev_cor_ss = nullptr;
  fT1d_av_mped = nullptr;
  fT1d_av_totn = nullptr;
  fT1d_av_lofn = nullptr;
  fT1d_av_hifn = nullptr;
  fT1d_av_ev_corss = nullptr;
  fT1d_av_sig_corss = nullptr;

  fT1d_sigevt_of_evsamp = nullptr;
  fT1d_evevt_of_sigsamp = nullptr;
  fT1d_sig_cor_ss = nullptr;

  fT2dCrysNumbersTable = nullptr;
  fT1dCrysNumbersTable = nullptr;

  //................................ tags   ( Init() )
  fTagStinNumbers = nullptr;

  fTagNbOfEvts = nullptr;

  fTagAdcEvt = nullptr;

  fTagMSp = nullptr;
  fTagSSp = nullptr;

  fTagCovCss = nullptr;
  fTagCorCss = nullptr;

  fTagHfCov = nullptr;
  fTagHfCor = nullptr;
  fTagLfCov = nullptr;
  fTagLfCor = nullptr;

  fTagLFccMoStins = nullptr;
  fTagHFccMoStins = nullptr;

  fTagPed = nullptr;
  fTagTno = nullptr;
  fTagMeanCorss = nullptr;

  fTagLfn = nullptr;
  fTagHfn = nullptr;
  fTagSigCorss = nullptr;

  fTagAvPed = nullptr;
  fTagAvTno = nullptr;
  fTagAvLfn = nullptr;
  fTagAvHfn = nullptr;

  fTagAvMeanCorss = nullptr;
  fTagAvSigCorss = nullptr;

  fObjectManager = (TEcnaObject*)pObjectManager;
  pObjectManager->RegisterPointer("TEcnaRun", (Long_t)this);

  //............................ fCnaParCout
  Long_t iCnaParCout = pObjectManager->GetPointerValue("TEcnaParCout");
  if (iCnaParCout == 0) {
    fCnaParCout = new TEcnaParCout(pObjectManager); /*fCnew++*/
  } else {
    fCnaParCout = (TEcnaParCout*)iCnaParCout;
  }

  //............................ fCnaParPaths
  Long_t iCnaParPaths = pObjectManager->GetPointerValue("TEcnaParPaths");
  if (iCnaParPaths == 0) {
    fCnaParPaths = new TEcnaParPaths(pObjectManager); /*fCnew++*/
  } else {
    fCnaParPaths = (TEcnaParPaths*)iCnaParPaths;
  }

  //................................................... Code Print  ( Init() )
  fCodePrintNoComment = fCnaParCout->GetCodePrint("NoComment");
  fCodePrintWarnings = fCnaParCout->GetCodePrint("Warnings ");  // => default value
  fCodePrintComments = fCnaParCout->GetCodePrint("Comments");
  fCodePrintAllComments = fCnaParCout->GetCodePrint("AllComments");

  fFlagPrint = fCodePrintWarnings;

  //...................................................
  gCnaRootFile = nullptr;
  fOpenRootFile = kFALSE;
  fReadyToReadData = 0;

  //.............................................. Miscellaneous
  fSpecialStexStinNotIndexed = -1;

  fStinIndexBuilt = 0;
  fBuildEvtNotSkipped = 0;

  fMemoReadNumberOfEventsforSamples = 0;

}  // end of Init()

//========================================================================
void TEcnaRun::SetEcalSubDetector(const TString& SubDet) {
  // Set Subdetector (EB or EE)

  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = SubDet.Data();

  fEcal = nullptr;
  fEcal = new TEcnaParEcal(fFlagSubDet.Data());  //fCnew++;
  fEcalNumbering = nullptr;
  fEcalNumbering = new TEcnaNumbering(fFlagSubDet.Data(), fEcal);  //fCnew++;
  fCnaWrite = nullptr;

  fCnaWrite = new TEcnaWrite(fFlagSubDet.Data(), fCnaParPaths, fCnaParCout, fEcal, fEcalNumbering);  //fCnew++;

  if (fFlagSubDet == "EB") {
    fStexName = "SM ";
    fStinName = "tower";
  }
  if (fFlagSubDet == "EE") {
    fStexName = "Dee";
    fStinName = " SC  ";
  }
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    copy constructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRun::TEcnaRun(const TEcnaRun& dcop) : TObject::TObject(dcop) {
  std::cout << "*TEcnaRun::TEcnaRun(const TEcnaRun& dcop)> "
            << " Now is the time to write a copy constructor" << std::endl;

  //{ Int_t cintoto;  cin >> cintoto; }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//TEcnaRun& TEcnaRun::operator=(const TEcnaRun& dcop)
//{
//Overloading of the operator=
//
//  fCopy(dcop);
//  return *this;
//}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRun::~TEcnaRun() {
  //Destructor

  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "*TEcnaRun::~TEcnaRun()> Entering destructor." << std::endl;
  }

  if (fFlagPrint != fCodePrintNoComment || fFlagPrint == fCodePrintWarnings) {
    if (fBuildEvtNotSkipped > 0) {
      std::cout << "************************************************************************************* "
                << std::endl;
      std::cout << "*TEcnaRun::~TEcnaRun()> Nb of calls to GetSampleAdcValues by cmsRun: " << fBuildEvtNotSkipped
                << std::endl;
      std::cout << "************************************************************************************* "
                << std::endl;
    }
  }

  if (fFlagPrint == fCodePrintAllComments) {
    Int_t misc_czero = 0;
    for (Int_t i = 0; i < fNbOfMiscDiagCounters; i++) {
      if (fMiscDiag[i] != 0) {
        std::cout << "                          fMiscDiag Counter " << std::setw(3) << i << " = " << std::setw(9)
                  << fMiscDiag[i] << " (INFO: alloc on non zero freed zone) " << std::endl;
      } else {
        misc_czero++;
      }
    }
    std::cout << "                          Nb of fMiscDiag counters at zero: " << misc_czero
              << " (total nb of counters: " << fNbOfMiscDiagCounters << ")" << std::endl;
  }

  if (fMiscDiag != nullptr) {
    delete[] fMiscDiag;
    fCdelete++;
  }

  //if (fFileHeader              != 0){delete fFileHeader;                  fCdelete++;}
  //if (fEcal                    != 0){delete fEcal;                        fCdelete++;}
  //if (fEcalNumbering           != 0){delete fEcalNumbering;               fCdelete++;}
  //if (fCnaParCout              != 0){delete fCnaParCout;                  fCdelete++;}
  //if (fCnaParPaths             != 0){delete fCnaParPaths;                 fCdelete++;}
  //if (fCnaWrite                != 0){delete fCnaWrite;                    fCdelete++;}

  if (fT1d_StexStinFromIndex != nullptr) {
    delete[] fT1d_StexStinFromIndex;
    fCdelete++;
  }

  if (fT2d_NbOfEvts != nullptr) {
    delete[] fT2d_NbOfEvts;
    fCdelete++;
  }
  if (fT1d_NbOfEvts != nullptr) {
    delete[] fT1d_NbOfEvts;
    fCdelete++;
  }

  if (fT3d_AdcValues != nullptr) {
    delete[] fT3d_AdcValues;
    fCdelete++;
  }
  if (fT3d2_AdcValues != nullptr) {
    delete[] fT3d2_AdcValues;
    fCdelete++;
  }
  if (fT3d1_AdcValues != nullptr) {
    delete[] fT3d1_AdcValues;
    fCdelete++;
  }

  if (fT2d_ev != nullptr) {
    delete[] fT2d_ev;
    fCdelete++;
  }
  if (fT1d_ev != nullptr) {
    delete[] fT1d_ev;
    fCdelete++;
  }

  if (fT2d_sig != nullptr) {
    delete[] fT2d_sig;
    fCdelete++;
  }
  if (fT1d_sig != nullptr) {
    delete[] fT1d_sig;
    fCdelete++;
  }

  if (fT3d_cov_ss != nullptr) {
    delete[] fT3d_cov_ss;
    fCdelete++;
  }
  if (fT3d2_cov_ss != nullptr) {
    delete[] fT3d2_cov_ss;
    fCdelete++;
  }
  if (fT3d1_cov_ss != nullptr) {
    delete[] fT3d1_cov_ss;
    fCdelete++;
  }

  if (fT3d_cor_ss != nullptr) {
    delete[] fT3d_cor_ss;
    fCdelete++;
  }
  if (fT3d2_cor_ss != nullptr) {
    delete[] fT3d2_cor_ss;
    fCdelete++;
  }
  if (fT3d1_cor_ss != nullptr) {
    delete[] fT3d1_cor_ss;
    fCdelete++;
  }

  if (fT2d_lf_cov != nullptr) {
    delete[] fT2d_lf_cov;
    fCdelete++;
  }
  if (fT2d1_lf_cov != nullptr) {
    delete[] fT2d1_lf_cov;
    fCdelete++;
  }

  if (fT2d_lf_cor != nullptr) {
    delete[] fT2d_lf_cor;
    fCdelete++;
  }
  if (fT2d1_lf_cor != nullptr) {
    delete[] fT2d1_lf_cor;
    fCdelete++;
  }

  if (fT2d_hf_cov != nullptr) {
    delete[] fT2d_hf_cov;
    fCdelete++;
  }
  if (fT2d1_hf_cov != nullptr) {
    delete[] fT2d1_hf_cov;
    fCdelete++;
  }

  if (fT2d_hf_cor != nullptr) {
    delete[] fT2d_hf_cor;
    fCdelete++;
  }
  if (fT2d1_hf_cor != nullptr) {
    delete[] fT2d1_hf_cor;
    fCdelete++;
  }

  if (fT2d_lfcc_mostins != nullptr) {
    delete[] fT2d_lfcc_mostins;
    fCdelete++;
  }
  if (fT2d1_lfcc_mostins != nullptr) {
    delete[] fT2d1_lfcc_mostins;
    fCdelete++;
  }

  if (fT2d_hfcc_mostins != nullptr) {
    delete[] fT2d_hfcc_mostins;
    fCdelete++;
  }
  if (fT2d1_hfcc_mostins != nullptr) {
    delete[] fT2d1_hfcc_mostins;
    fCdelete++;
  }

  if (fT1d_ev_ev != nullptr) {
    delete[] fT1d_ev_ev;
    fCdelete++;
  }
  if (fT1d_evsamp_of_sigevt != nullptr) {
    delete[] fT1d_evsamp_of_sigevt;
    fCdelete++;
  }
  if (fT1d_ev_cor_ss != nullptr) {
    delete[] fT1d_ev_cor_ss;
    fCdelete++;
  }
  if (fT1d_av_mped != nullptr) {
    delete[] fT1d_av_mped;
    fCdelete++;
  }
  if (fT1d_av_totn != nullptr) {
    delete[] fT1d_av_totn;
    fCdelete++;
  }
  if (fT1d_av_lofn != nullptr) {
    delete[] fT1d_av_lofn;
    fCdelete++;
  }
  if (fT1d_av_hifn != nullptr) {
    delete[] fT1d_av_hifn;
    fCdelete++;
  }
  if (fT1d_av_ev_corss != nullptr) {
    delete[] fT1d_av_ev_corss;
    fCdelete++;
  }
  if (fT1d_av_sig_corss != nullptr) {
    delete[] fT1d_av_sig_corss;
    fCdelete++;
  }

  if (fT1d_sigevt_of_evsamp != nullptr) {
    delete[] fT1d_sigevt_of_evsamp;
    fCdelete++;
  }
  if (fT1d_evevt_of_sigsamp != nullptr) {
    delete[] fT1d_evevt_of_sigsamp;
    fCdelete++;
  }
  if (fT1d_sig_cor_ss != nullptr) {
    delete[] fT1d_sig_cor_ss;
    fCdelete++;
  }

  if (fT2dCrysNumbersTable != nullptr) {
    delete[] fT2dCrysNumbersTable;
    fCdelete++;
  }
  if (fT1dCrysNumbersTable != nullptr) {
    delete[] fT1dCrysNumbersTable;
    fCdelete++;
  }

  if (fTagStinNumbers != nullptr) {
    delete[] fTagStinNumbers;
    fCdelete++;
  }
  if (fTagNbOfEvts != nullptr) {
    delete[] fTagNbOfEvts;
    fCdelete++;
  }
  if (fTagAdcEvt != nullptr) {
    delete[] fTagAdcEvt;
    fCdelete++;
  }
  if (fTagMSp != nullptr) {
    delete[] fTagMSp;
    fCdelete++;
  }
  if (fTagSSp != nullptr) {
    delete[] fTagSSp;
    fCdelete++;
  }

  if (fTagCovCss != nullptr) {
    delete[] fTagCovCss;
    fCdelete++;
  }
  if (fTagCorCss != nullptr) {
    delete[] fTagCorCss;
    fCdelete++;
  }

  if (fTagHfCov != nullptr) {
    delete[] fTagHfCov;
    fCdelete++;
  }
  if (fTagHfCor != nullptr) {
    delete[] fTagHfCor;
    fCdelete++;
  }
  if (fTagLfCov != nullptr) {
    delete[] fTagLfCov;
    fCdelete++;
  }
  if (fTagLfCor != nullptr) {
    delete[] fTagLfCor;
    fCdelete++;
  }

  if (fTagLFccMoStins != nullptr) {
    delete[] fTagLFccMoStins;
    fCdelete++;
  }
  if (fTagHFccMoStins != nullptr) {
    delete[] fTagHFccMoStins;
    fCdelete++;
  }

  if (fTagPed != nullptr) {
    delete[] fTagPed;
    fCdelete++;
  }
  if (fTagTno != nullptr) {
    delete[] fTagTno;
    fCdelete++;
  }
  if (fTagMeanCorss != nullptr) {
    delete[] fTagMeanCorss;
    fCdelete++;
  }

  if (fTagLfn != nullptr) {
    delete[] fTagLfn;
    fCdelete++;
  }
  if (fTagHfn != nullptr) {
    delete[] fTagHfn;
    fCdelete++;
  }
  if (fTagSigCorss != nullptr) {
    delete[] fTagSigCorss;
    fCdelete++;
  }

  if (fTagAvPed != nullptr) {
    delete[] fTagAvPed;
    fCdelete++;
  }
  if (fTagAvTno != nullptr) {
    delete[] fTagAvTno;
    fCdelete++;
  }
  if (fTagAvLfn != nullptr) {
    delete[] fTagAvLfn;
    fCdelete++;
  }
  if (fTagAvHfn != nullptr) {
    delete[] fTagAvHfn;
    fCdelete++;
  }
  if (fTagAvMeanCorss != nullptr) {
    delete[] fTagAvMeanCorss;
    fCdelete++;
  }
  if (fTagAvSigCorss != nullptr) {
    delete[] fTagAvSigCorss;
    fCdelete++;
  }

  if (fCnew != fCdelete) {
    std::cout << "!TEcnaRun::~TEcnaRun()> WRONG MANAGEMENT OF MEMORY ALLOCATIONS: fCnew = " << fCnew
              << ", fCdelete = " << fCdelete << fTTBELL << std::endl;
  } else {
    // std::cout << "*TEcnaRun::~TEcnaRun()> Management of memory allocations: OK. fCnew = "
    //   << fCnew << ", fCdelete = " << fCdelete << std::endl;
  }

  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "*TEcnaRun::~TEcnaRun()> Exiting destructor (this = " << this << ")." << std::endl
              << "~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#"
              << std::endl;
  }

  // std::cout << "[Info Management] CLASS: TEcnaRun.           DESTROY OBJECT: this = " << this << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                             M  E  T  H  O  D  S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//============================================================================
//
//                           GetReadyToReadData(...)
//
//  Preparation of the result file name + tags allocations
//  + ADC event distribution array allocation + nb of events array allocation
//
//============================================================================

void TEcnaRun::GetReadyToReadData(const TString& typ_ana,
                                  const Int_t& run_number,
                                  const Int_t& nfirst,
                                  const Int_t& nlast,
                                  const Int_t& nbevts,
                                  const Int_t& Stex) {
  //Preparation of the data reading. Set part of the header. No RunType as argument.
  //Use default value = 99999999 and call method with all the arguments (see below)

  Int_t RunType = 99999999;
  GetReadyToReadData(typ_ana, run_number, nfirst, nlast, nbevts, Stex, RunType);
}
//--------------------------------------------------------------------------------
void TEcnaRun::GetReadyToReadData(const TString& typ_ana,
                                  const Int_t& run_number,
                                  const Int_t& nfirst,
                                  const Int_t& nlast,
                                  const Int_t& nbevts,
                                  const Int_t& Stex,
                                  const Int_t& run_type) {
  //Preparation of the data reading. Set part of the header
  //
  //   [nfirst, nlast] = [1,50] (50 events) or [151,300] (150 events) or etc...

  Int_t nrangeevts = nlast - nfirst + 1;  // number of events in range

  if (nrangeevts < nbevts) {
    if (nlast >= nfirst) {
      std::cout << "*TEcnaRun::GetReadyToReadData(...)> --- WARNING ---> number of events = " << nbevts
                << ", out of range (range = " << nfirst << "," << nlast << ")" << std::endl
                << "                                    The number of found events will be less " << std::endl
                << "                                    than the number of requested events." << std::endl;
    }
    if (nlast < nfirst) {
      std::cout << "*TEcnaRun::GetReadyToReadData(...)> --- INFO ---> last requested event number = " << nlast
                << ", less than first requested event number (= " << nfirst << ")" << std::endl
                << "                                    File will be read until EOF if the number of found events"
                << std::endl
                << "                                    remains less than the number of requested events." << std::endl;
    }
  }

  //............. allocation for counters
  fMiscDiag = new Int_t[fNbOfMiscDiagCounters];
  fCnew++;
  for (Int_t iz = 0; iz < fNbOfMiscDiagCounters; iz++) {
    fMiscDiag[iz] = (Int_t)0;
  }

  //************** CHECK OF ARGUMENTS: nfirst_arg and nbevts_arg
  Int_t nentries = 99999999;  // => to be reintroduced as argument (like run_type) (?)
  if (nfirst <= nentries) {
    //--------------------- test positivity of nfirst_arg        (GetReadyToReadData)
    if (nfirst > 0) {
      //-------- test compatibility between the last requested event number
      //         and the number of entries
      if (nlast <= nentries) {
        const Text_t* h_name = "CnaHeader";   //==> voir cette question avec FXG
        const Text_t* h_title = "CnaHeader";  //==> voir cette question avec FXG
        //fFileHeader->HeaderParameters(h_name,     h_title ,
        //	         	      typ_ana,    fNbSampForFic,
        //			      run_number, nfirst,  nlast,  nbevts,
        //			      Stex,       nentries);                 fCnew++;

        if (fEcal->MaxStinEcnaInStex() > 0 && fEcal->MaxCrysInStin() > 0 && fNbSampForFic > 0) {
          if (fFileHeader == nullptr) {
            fFileHeader = new TEcnaHeader(fObjectManager, h_name, h_title);
          }  // fCnew++;

          fFileHeader->HeaderParameters(typ_ana, fNbSampForFic, run_number, nfirst, nlast, nbevts, Stex, run_type);

          // After this call to TEcnaHeader, we have:               (GetReadyToReadData)
          //     fFileHeader->fTypAna            = typ_ana
          //     fFileHeader->fNbOfSamples       = fNbSampForFic
          //     fFileHeader->fRunNumber         = run_number
          //     fFileHeader->fFirstReqEvtNumber = nfirst
          //     fFileHeader->fLastReqEvtNumber  = nlast
          //     fFileHeader->fReqNbOfEvts       = nbevts
          //     fFileHeader->fStex              = Stex number
          //     fFileHeader->fRunType           = run_type

          // fFileHeader->Print();

          // {Int_t cintoto; std::cout << "taper 0 pour continuer" << std::endl; cin >> cintoto;}

          //  fFileHeader->SetName("CnaHeader");              *======> voir FXG
          //  fFileHeader->SetTitle("CnaHeader");

          //......................................... allocation tags + init of them (GetReadyToReadData)

          fTagStinNumbers = new Int_t[1];
          fCnew++;
          fTagStinNumbers[0] = (Int_t)0;
          fTagNbOfEvts = new Int_t[1];
          fCnew++;
          fTagNbOfEvts[0] = (Int_t)0;

          fTagAdcEvt = new Int_t[fEcal->MaxCrysEcnaInStex()];
          fCnew++;
          for (Int_t iz = 0; iz < fEcal->MaxCrysEcnaInStex(); iz++) {
            fTagAdcEvt[iz] = (Int_t)0;
          }

          fTagMSp = new Int_t[1];
          fCnew++;
          fTagMSp[0] = (Int_t)0;
          fTagSSp = new Int_t[1];
          fCnew++;
          fTagSSp[0] = (Int_t)0;

          fTagCovCss = new Int_t[fEcal->MaxCrysEcnaInStex()];
          fCnew++;
          for (Int_t iz = 0; iz < fEcal->MaxCrysEcnaInStex(); iz++) {
            fTagCovCss[iz] = (Int_t)0;
          }

          fTagCorCss = new Int_t[fEcal->MaxCrysEcnaInStex()];
          fCnew++;
          for (Int_t iz = 0; iz < fEcal->MaxCrysEcnaInStex(); iz++) {
            fTagCorCss[iz] = (Int_t)0;
          }

          fTagLfCov = new Int_t[1];
          fCnew++;
          fTagLfCov[0] = (Int_t)0;
          fTagLfCor = new Int_t[1];
          fCnew++;
          fTagLfCor[0] = (Int_t)0;

          fTagHfCov = new Int_t[1];
          fCnew++;
          fTagHfCov[0] = (Int_t)0;
          fTagHfCor = new Int_t[1];
          fCnew++;
          fTagHfCor[0] = (Int_t)0;

          fTagLFccMoStins = new Int_t[1];
          fCnew++;
          fTagLFccMoStins[0] = (Int_t)0;
          fTagHFccMoStins = new Int_t[1];
          fCnew++;
          fTagHFccMoStins[0] = (Int_t)0;

          fTagPed = new Int_t[1];
          fCnew++;
          fTagPed[0] = (Int_t)0;
          fTagTno = new Int_t[1];
          fCnew++;
          fTagTno[0] = (Int_t)0;
          fTagMeanCorss = new Int_t[1];
          fCnew++;
          fTagMeanCorss[0] = (Int_t)0;

          fTagLfn = new Int_t[1];
          fCnew++;
          fTagLfn[0] = (Int_t)0;
          fTagHfn = new Int_t[1];
          fCnew++;
          fTagHfn[0] = (Int_t)0;
          fTagSigCorss = new Int_t[1];
          fCnew++;
          fTagSigCorss[0] = (Int_t)0;

          fTagAvPed = new Int_t[1];
          fCnew++;
          fTagAvPed[0] = (Int_t)0;
          fTagAvTno = new Int_t[1];
          fCnew++;
          fTagAvTno[0] = (Int_t)0;
          fTagAvLfn = new Int_t[1];
          fCnew++;
          fTagAvLfn[0] = (Int_t)0;
          fTagAvHfn = new Int_t[1];
          fCnew++;
          fTagAvHfn[0] = (Int_t)0;
          fTagAvMeanCorss = new Int_t[1];
          fCnew++;
          fTagAvMeanCorss[0] = (Int_t)0;
          fTagAvSigCorss = new Int_t[1];
          fCnew++;
          fTagAvSigCorss[0] = (Int_t)0;

          //====================================================================================
          //
          //   allocation for fT1d_StexStinFromIndex[] and init to fSpecialStexStinNotIndexed
          //
          //====================================================================================

          if (fT1d_StexStinFromIndex == nullptr) {
            fT1d_StexStinFromIndex = new Int_t[fEcal->MaxStinEcnaInStex()];
            fCnew++;
          }
          for (Int_t i0_Stin = 0; i0_Stin < fEcal->MaxStinEcnaInStex(); i0_Stin++) {
            fT1d_StexStinFromIndex[i0_Stin] = fSpecialStexStinNotIndexed;
          }

          //-------------------------------------------------------------  (GetReadyToReadData)

          //====================================================================================
          //
          //   allocation of the 3D array fT3d_AdcValues[channel][sample][events] (ADC values)
          //
          //   This array is filled in the GetSampleAdcValues(...) method
          //
          //====================================================================================

          if (fT3d_AdcValues == nullptr) {
            //............ Allocation for the 3d array
            std::cout << "*TEcnaRun::GetReadyToReadData(...)> Allocation of 3D array for ADC distributions."
                      << " Nb of requested evts = " << fFileHeader->fReqNbOfEvts << std::endl
                      << "                                    This number must not be too large"
                      << " (no failure after this message means alloc OK)." << std::endl;

            fT3d_AdcValues = new Double_t**[fEcal->MaxCrysEcnaInStex()];
            fCnew++;

            fT3d2_AdcValues = new Double_t*[fEcal->MaxCrysEcnaInStex() * fNbSampForFic];
            fCnew++;

            fT3d1_AdcValues = new Double_t[fEcal->MaxCrysEcnaInStex() * fNbSampForFic * fFileHeader->fReqNbOfEvts];
            fCnew++;

            for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
              fT3d_AdcValues[i0StexEcha] = &fT3d2_AdcValues[0] + i0StexEcha * fNbSampForFic;
              for (Int_t j0Sample = 0; j0Sample < fNbSampForFic; j0Sample++) {
                fT3d2_AdcValues[fNbSampForFic * i0StexEcha + j0Sample] =
                    &fT3d1_AdcValues[0] + fFileHeader->fReqNbOfEvts * (fNbSampForFic * i0StexEcha + j0Sample);
              }
            }
          }
          //................................. Init to zero
          for (Int_t iza = 0; iza < fEcal->MaxCrysEcnaInStex(); iza++) {
            for (Int_t izb = 0; izb < fNbSampForFic; izb++) {
              for (Int_t izc = 0; izc < fFileHeader->fReqNbOfEvts; izc++) {
                if (fT3d_AdcValues[iza][izb][izc] != (Double_t)0) {
                  fMiscDiag[0]++;
                  fT3d_AdcValues[iza][izb][izc] = (Double_t)0;
                }
              }
            }
          }

          //--------------------------------------------------------- (GetReadyToReadData)
          //====================================================================================
          //
          //   allocation of the 2D array fT2d_NbOfEvts[channel][sample] (Max nb of evts)
          //
          //====================================================================================

          if (fT2d_NbOfEvts == nullptr) {
            fT2d_NbOfEvts = new Int_t*[fEcal->MaxCrysEcnaInStex()];
            fCnew++;
            fT1d_NbOfEvts = new Int_t[fEcal->MaxCrysEcnaInStex() * fNbSampForFic];
            fCnew++;

            for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
              fT2d_NbOfEvts[i0StexEcha] = &fT1d_NbOfEvts[0] + i0StexEcha * fNbSampForFic;
            }

            //................ Init the array to 0
            for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
              for (Int_t i0Sample = 0; i0Sample < fNbSampForFic; i0Sample++) {
                fT2d_NbOfEvts[i0StexEcha][i0Sample] = 0;
              }
            }
          } else {
            std::cerr << "!TEcnaRun::GetReadyToReadData(...)> *** ERROR *** No allocation for fT2d_NbOfEvts!"
                      << " Pointer already not NULL " << fTTBELL << std::endl;
            // {Int_t cintoto; std::cout << "Enter: 0 and RETURN to continue or: CTRL C to exit"
            //		   << std::endl; std::cin >> cintoto;}
          }
        } else {
          std::cerr << std::endl
                    << "!TEcnaRun::GetReadyToReadData(...)> "
                    << " *** ERROR *** " << std::endl
                    << " --------------------------------------------------" << std::endl
                    << " NULL or NEGATIVE values for arguments" << std::endl
                    << " with expected positive values:" << std::endl
                    << " Number of Stins in Stex = " << fEcal->MaxStinEcnaInStex() << std::endl
                    << " Number of crystals in Stin     = " << fEcal->MaxCrysInStin() << std::endl
                    << " Number of samples by channel    = " << fNbSampForFic << std::endl
                    << std::endl
                    << std::endl
                    << " hence, no memory allocation for array member has been performed." << std::endl;

          std::cout << "Enter: 0 and RETURN to continue or: CTRL C to exit";
          Int_t toto;
          std::cin >> toto;
        }
        //----------------------------------------------------------- (GetReadyToReadData)
        if (fFlagPrint == fCodePrintAllComments) {
          std::cout << std::endl;
          std::cout << "*TEcnaRun::GetReadyToReadData(...)>" << std::endl
                    << "          The method has been called with the following argument values:" << std::endl
                    << "          Analysis name                = " << fFileHeader->fTypAna << std::endl
                    << "          Run number                   = " << fFileHeader->fRunNumber << std::endl
                    << "          Run type                     = " << fFileHeader->fRunType << std::endl
                    << "          First requested event number = " << fFileHeader->fFirstReqEvtNumber << std::endl
                    << "          Last requested event number  = " << fFileHeader->fLastReqEvtNumber << std::endl
                    << "          " << fStexName.Data() << " number                  = " << fFileHeader->fStex
                    << std::endl
                    << "          Number of " << fStinName.Data() << " in " << fStexName.Data() << "       = "
                    << fEcal->MaxStinEcnaInStex() << std::endl
                    << "          Number of crystals in " << fStinName.Data() << "  = " << fEcal->MaxCrysInStin()
                    << std::endl
                    << "          Number of samples by channel = " << fNbSampForFic << std::endl
                    << std::endl;
        }

        fReadyToReadData = 1;  // set flag
      } else {
        if (fFlagPrint != fCodePrintNoComment) {
          std::cout << "!TEcnaRun::GetReadyToReadData(...) > WARNING/CORRECTION:" << std::endl
                    << "! The fisrt requested event number is not positive (nfirst = " << nfirst << ") " << fTTBELL
                    << std::endl;
        }
      }
    } else {
      if (fFlagPrint != fCodePrintNoComment) {
        std::cout << std::endl
                  << "!TEcnaRun::GetReadyToReadData(...)> WARNING/CORRECTION:" << std::endl
                  << "! The number of requested events (nbevts = " << nbevts << ") is too large." << std::endl
                  << "! Last event number = " << nlast << " > number of entries = " << nentries << ". " << fTTBELL
                  << std::endl
                  << std::endl;
      }
    }
  } else {
    std::cout << "!TEcnaRun::GetReadyToReadData(...) *** ERROR ***> "
              << " The first requested event number is greater than the number of entries." << fTTBELL << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "*TEcnaRun::GetReadyToReadData(...)> Leaving the method. fReadyToReadData = " << fReadyToReadData
              << std::endl;
  }

}  // end of GetReadyToReadData

//====================================================================================================
//
//     GetSampleAdcValues: method called by the CMSSW analyzer (cmsRun)
//
//     At each event, put the Sample ADC value in the 3D array: fT3d_AdcValues[i0StexEcha][i0Sample][i0EventIndex]
//
//         |============================================================|
//         |                                                            |
//         | (Stex,Stin) means: (SM,tower) for EB and: (Dee,SC) for EE  |
//         |                                                            |
//         |============================================================|
//
//    (Stin number <-> Stin index correspondance, ADC sample values)
//
//        THIS METHOD IS CALLED INSIDE THE LOOPS OVER:
//          ( EVENTS (tower or SC (CRYSTAL in tower or SC (SAMPLES))))
//
//  Arguments: Event    = event number.              Range = [ 1, fFileHeader->fReqNbOfEvts ]
//             n1StexStin = Stin number in Stex.     Range = [ 1, fEcal->MaxStinEcnaInStex() ]
//             i0StinEcha = channel number in Stin.  Range = [ 0, fEcal->MaxCrysInStin() - 1 ]
//             sample   = ADC sample number.         Range = [ 0, fNbSampForFic - 1 ]
//             adcvalue = ADC sample value.
//
//====================================================================================================
Bool_t TEcnaRun::GetSampleAdcValues(const Int_t& n1EventNumber,
                                    const Int_t& n1StexStin,
                                    const Int_t& i0StinEcha,
                                    const Int_t& i0Sample,
                                    const Double_t& adcvalue) {
  //Building of the arrays fT1d_StexStinFromIndex[] and fT3d_AdcValues[][][]

  fBuildEvtNotSkipped++;  // event not skipped by cmsRun

  Bool_t ret_code = kFALSE;

  Int_t i0EventIndex = n1EventNumber - 1;  // INDEX FOR Event number
  Int_t i0StexStinEcna = n1StexStin - 1;   // INDEX FOR StexStin = Number_of_the_Stin_in_Stex - 1

  Int_t i_trouve = 0;
  //.................................................................. (GetSampleAdcValues)
  if (fReadyToReadData == 1) {
    if (n1StexStin >= 1 && n1StexStin <= fEcal->MaxStinEcnaInStex()) {
      if (i0StinEcha >= 0 && i0StinEcha < fEcal->MaxCrysInStin()) {
        if (i0Sample >= 0 && i0Sample < fEcal->MaxSampADC()) {
          //..... Put the StexStin number in 1D array fT1d_StexStinFromIndex[] = Stin index + 1
          if (fT1d_StexStinFromIndex != nullptr)  // table fT1d_StexStinFromIndex[index] already allocated
          {
            ret_code = kTRUE;

            // StexStin already indexed
            if (n1StexStin == fT1d_StexStinFromIndex[i0StexStinEcna]) {
              i_trouve = 1;
            }

            // StexStin index not found: set index for new StexStin
            if (i_trouve != 1) {
              if (fT1d_StexStinFromIndex[i0StexStinEcna] == fSpecialStexStinNotIndexed) {
                fT1d_StexStinFromIndex[i0StexStinEcna] = n1StexStin;
                fFileHeader->fStinNumbersCalc = 1;
                fTagStinNumbers[0] = 1;
                fStinIndexBuilt++;  //  number of found Stins

                if (fFlagPrint == fCodePrintAllComments) {
                  if (fStinIndexBuilt == 1) {
                    std::cout << std::endl
                              << "*TEcnaRun::GetSampleAdcValues(...)> event " << n1EventNumber << " : first event for "
                              << fStexName.Data() << " " << fFileHeader->fStex << "; " << fStinName.Data() << "s : ";
                  }
                  if (fFlagSubDet == "EB") {
                    std::cout << fT1d_StexStinFromIndex[i0StexStinEcna] << ", ";
                  }
                  if (fFlagSubDet == "EE") {
                    std::cout << fEcalNumbering->GetDeeSCConsFrom1DeeSCEcna(fFileHeader->fStex,
                                                                            fT1d_StexStinFromIndex[i0StexStinEcna])
                              << ", ";
                  }
                }
                //.................................................... (GetSampleAdcValues)
                if (fFlagPrint == fCodePrintAllComments) {
                  std::cout << " (" << fStinIndexBuilt << " " << fStinName.Data() << " found), channel " << i0StinEcha
                            << ", i0Sample " << i0Sample << std::endl;
                }
                ret_code = kTRUE;
              }  // if ( fT1d_StexStinFromIndex[i0StexStinEcna] == fSpecialStexStinNotIndexed )
              else {
                std::cout << "!TEcnaRun::GetSampleAdcValues(...)> *** ERROR ***> NOT ALLOWED if RESULT. "
                          << " n1StexStin = " << n1StexStin << ", fT1d_StexStinFromIndex[" << i0StexStinEcna
                          << "] = " << fT1d_StexStinFromIndex[i0StexStinEcna]
                          << ", fStinIndexBuilt = " << fStinIndexBuilt << fTTBELL << std::endl;
                ret_code = kFALSE;
              }
            }  //  if (i_trouve != 1 )

          }  //  if( fT1d_StexStinFromIndex != 0 )
          else {
            std::cout << "!TEcnaRun, GetSampleAdcValues *** ERROR ***> "
                      << " fT1d_StexStinFromIndex = " << fT1d_StexStinFromIndex
                      << " fT1d_StexStinFromIndex[] ALLOCATION NOT DONE" << fTTBELL << std::endl;
            ret_code = kFALSE;
          }  //.................................................................. (GetSampleAdcValues)
        }    // end of if( i0Sample >= 0 && i0Sample < fNbSampForFic )
        else {
          //.......Reading data => Message and error only if sample >= fEcal->MaxSampADC()
          //       (not fNbSampForFic, the later is used only for calculations)
          if (i0Sample >= fEcal->MaxSampADC()) {
            std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> "
                      << " sample number = " << i0Sample << ". OUT OF BOUNDS"
                      << " (max = " << fNbSampForFic << ")" << fTTBELL << std::endl;
            ret_code = kFALSE;
          } else {
            ret_code = kTRUE;
          }
        }  // else of if( i0Sample >= 0 && i0Sample < fNbSampForFic )
      }    // end of if( i0StinEcha >= 0 && i0StinEcha < fEcal->MaxCrysInStin() )
      else {
        std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> "
                  << " channel number in " << fStinName.Data() << " = " << i0StinEcha << ". OUT OF BOUNDS"
                  << " (max = " << fEcal->MaxCrysInStin() << ")" << fTTBELL << std::endl;
        ret_code = kFALSE;
      }  // else of if( i0StinEcha >= 0 && i0StinEcha < fEcal->MaxCrysInStin() )
    } else {
      std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> " << fStinName.Data() << " number in "
                << fStexName.Data() << " = " << n1StexStin << ". OUT OF BOUNDS"
                << " (max = " << fEcal->MaxStinEcnaInStex() << ")" << fTTBELL << std::endl;
      ret_code = kFALSE;
    }

    //.................................................................. (GetSampleAdcValues)
    //........ Filling of the 2D array of the event numbers in the data reading loop and
    //         filling of the 3D array of the ADC sample values
    //
    //                           ONLY if ret_code == kTRUE

    if (ret_code == kTRUE) {
      //............ 1) Conversion (Stin,i0StinEcha) -> i0StexEcha  (same numbering for EB and EE)
      //==========================================================================================
      //   n1StexStin (Tower or SC):      1            2            3
      //   iStexStin                      0            1            2
      //
      //   i0StinEcha:                 0......24    0......24    0......24
      //
      //   i0StexEcha         :        0......24   25......49   50......74  grouped by StexStin's
      //   i0StexEcha+1 (Xtal):        1......25   26......50   51......75
      //
      //==========================================================================================

      Int_t i0StexEcha = i0StexStinEcna * fEcal->MaxCrysInStin() + i0StinEcha;

      //--------------------------------------------------------- (GetSampleAdcValues)
      if (i0StexEcha >= 0 && i0StexEcha < fEcal->MaxCrysEcnaInStex()) {
        //............ 2) Increase of the nb of evts for (StexEcha,sample) (events found in the data)
        (fT2d_NbOfEvts[i0StexEcha][i0Sample])++;  // value after first incrementation = 1
        fTagNbOfEvts[0] = 1;
        fFileHeader->fNbOfEvtsCalc = 1;

        //............ 3) Filling of the 3D array of the ADC values
        if (i0EventIndex >= 0 && i0EventIndex < fFileHeader->fReqNbOfEvts) {
          if (i0Sample >= 0 && i0Sample < fNbSampForFic) {
            fT3d_AdcValues[i0StexEcha][i0Sample][i0EventIndex] = adcvalue;
          } else {
            std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> "
                      << " sample index = " << i0Sample << ". OUT OF BOUNDS"
                      << " (max = " << fNbSampForFic << ")" << fTTBELL << std::endl;
          }
        } else {
          std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> "
                    << " event number = " << n1EventNumber << ". OUT OF BOUNDS"
                    << " (max = " << fFileHeader->fReqNbOfEvts << ")" << fTTBELL << std::endl;
          ret_code = kFALSE;
        }
      } else {
        std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> "
                  << " CHANNEL NUMBER OUT OF BOUNDS" << std::endl
                  << " i0StexEcha number = " << i0StexEcha << " , n1StexStin = " << n1StexStin
                  << " , i0StinEcha = " << i0StinEcha
                  << " , fEcal->MaxCrysEcnaInStex() = " << fEcal->MaxCrysEcnaInStex() << fTTBELL << std::endl;
        ret_code = kFALSE;
        // {Int_t cintoto; std::cout << "TAPER 0 POUR CONTINUER" << std::endl; cin >> cintoto;}
      }
    }  // end of if( ret_code == kTRUE )
    else {
      std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> ret_code = kFALSE " << fTTBELL << std::endl;
    }
  }  // end of if(fReadyToReadData == 1)
  else {
    std::cout << "!TEcnaRun::GetSampleAdcValues(...) *** ERROR ***> GetReadyToReadData(...) not called." << fTTBELL
              << std::endl;
    ret_code = kFALSE;
  }
  //.................................................................. (GetSampleAdcValues)
  if (ret_code == kFALSE) {
    std::cout << "!TEcnaRun::GetSampleAdcValues(...)> *** ERROR ***> ret_code = " << ret_code
              << " (FALSE). Event: " << n1EventNumber << ", " << fStexName.Data() << ": " << fFileHeader->fStex << ", "
              << fStinName.Data() << ": " << n1StexStin << ", channel: " << i0StinEcha << ", Sample: " << i0Sample
              << ", ADC value: " << adcvalue << std::endl;
  }
  return ret_code;
}
//------------- ( end of GetSampleAdcValues ) -----------------------
//====================================================================================================
//
//  ReadSampleAdcValues: Get the Sample ADC values from file by using TEcnaRead.
//
//====================================================================================================
Bool_t TEcnaRun::ReadSampleAdcValues() { return ReadSampleAdcValues(fEcal->MaxSampADC()); }

Bool_t TEcnaRun::ReadSampleAdcValues(const Int_t& nb_samp_for_calc) {
  // read the Sample ADC values from "ADC" result root files                     (ReadSampleAdcValues)

  // put the number of sample for calculations in attribute fNbSampForCalc
  // and call the method without arguments
  // We must have: nb_samp_for_calc <= fFileHeader->fNbOfSamples (= nb of samples in ROOT file)

  fNbSampForCalc = nb_samp_for_calc;

  //   TEcnaRead* MyRootFile = new TEcnaRead(fFlagSubDet.Data(), fCnaParPaths, fCnaParCout,
  // 		 			   fFileHeader, fEcalNumbering, fCnaWrite);          //  fCnew++;

  TEcnaRead* MyRootFile = new TEcnaRead(fObjectManager, fFlagSubDet.Data());  //  fCnew++;

  MyRootFile->PrintNoComment();

  MyRootFile->FileParameters(fFileHeader->fTypAna,
                             fFileHeader->fNbOfSamples,
                             fFileHeader->fRunNumber,
                             fFileHeader->fFirstReqEvtNumber,
                             fFileHeader->fLastReqEvtNumber,
                             fFileHeader->fReqNbOfEvts,
                             fFileHeader->fStex,
                             fCnaParPaths->ResultsRootFilePath().Data());

  Bool_t ok_read = MyRootFile->LookAtRootFile();

  fFileHeader->fStartTime = MyRootFile->GetStartTime();
  fFileHeader->fStopTime = MyRootFile->GetStopTime();
  fFileHeader->fStartDate = MyRootFile->GetStartDate();
  fFileHeader->fStopDate = MyRootFile->GetStopDate();

  if (ok_read == kTRUE) {
    fRootFileName = MyRootFile->GetRootFileName();
    fRootFileNameShort = MyRootFile->GetRootFileNameShort();
    std::cout << "*TEcnaRun::ReadSampleAdcValues> Reading sample ADC values from file: " << std::endl
              << "           " << fRootFileName << std::endl;

    size_t i_no_data = 0;

    //.......... Read the StinNumbers in the old file                     (ReadSampleAdcValues)
    TVectorD vec(fEcal->MaxStinEcnaInStex());
    for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++) {
      vec(i) = (Double_t)0.;
    }
    vec = MyRootFile->ReadStinNumbers(fEcal->MaxStinEcnaInStex());
    if (MyRootFile->DataExist() == kTRUE) {
      fTagStinNumbers[0] = 1;
      fFileHeader->fStinNumbersCalc = 1;
      for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
        fT1d_StexStinFromIndex[i0StexStinEcna] = (Int_t)vec(i0StexStinEcna);
      }
    } else {
      i_no_data++;
    }
    //.......... Read the Numbers of Events in the old file                      (ReadSampleAdcValues)
    TMatrixD partial_matrix(fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);
    for (Int_t i = 0; i < fEcal->MaxCrysInStin(); i++) {
      for (Int_t j = 0; j < fFileHeader->fNbOfSamples; j++) {
        partial_matrix(i, j) = (Double_t)0.;
      }
    }

    for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
      Int_t n1StexStin = MyRootFile->GetStexStinFromIndex(i0StexStinEcna);
      if (n1StexStin != -1) {
        partial_matrix =
            MyRootFile->ReadNumberOfEventsForSamples(n1StexStin, fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);

        if (MyRootFile->DataExist() == kTRUE) {
          fTagNbOfEvts[0] = 1;
          fFileHeader->fNbOfEvtsCalc = 1;
          for (Int_t i0StinCrys = 0; i0StinCrys < fEcal->MaxCrysInStin(); i0StinCrys++) {
            Int_t i0StexEcha = (n1StexStin - 1) * fEcal->MaxCrysInStin() + i0StinCrys;
            for (Int_t i0Sample = 0; i0Sample < fFileHeader->fNbOfSamples; i0Sample++) {
              fT2d_NbOfEvts[i0StexEcha][i0Sample] = (Int_t)partial_matrix(i0StinCrys, i0Sample);
            }
          }
        } else {
          i_no_data++;
        }
      }
    }

    //.......... Read the Sample ADC values in the old file                     (ReadSampleAdcValues)
    Double_t*** fT3d_read_AdcValues = MyRootFile->ReadSampleAdcValuesSameFile(
        fEcal->MaxCrysEcnaInStex(), fFileHeader->fNbOfSamples, fFileHeader->fReqNbOfEvts);

    if (MyRootFile->DataExist() == kTRUE) {
      for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
        for (Int_t i0Sample = 0; i0Sample < fFileHeader->fNbOfSamples; i0Sample++) {
          for (Int_t i_event = 0; i_event < fFileHeader->fReqNbOfEvts; i_event++) {
            fT3d_AdcValues[i0StexEcha][i0Sample][i_event] = fT3d_read_AdcValues[i0StexEcha][i0Sample][i_event];
          }
        }
      }
    } else {
      i_no_data++;
    }
    if (i_no_data) {
      std::cout << "!TEcnaRun::ReadSampleAdcValues(...)> *ERROR* =====> "
                << " Read failure. i_no_data = " << i_no_data << fTTBELL << std::endl;
    }
  } else {
    std::cout << "!TEcnaRun::ReadSampleAdcValues(...)> *ERROR* =====> "
              << " ROOT file not found" << fTTBELL << std::endl;
  }
  delete MyRootFile;  //  fCdelete++;
  return ok_read;
}
//------------- ( end of ReadSampleAdcValues ) -----------------------
//-------------------------------------------------------------------------
//
//    Get the ROOT file name (long and short)
//
//-------------------------------------------------------------------------
const TString& TEcnaRun::GetRootFileName() const { return fRootFileName; }
const TString& TEcnaRun::GetRootFileNameShort() const { return fRootFileNameShort; }

//###################################################################################################
//
// THE FOLLOWING METHODS ARE CALLED AFTER THE LOOPS OVER EVENTS, STINS, CRYSTALS AND SAMPLES
//
//###################################################################################################
//=========================================================================
//
//     Set start time, stop time, StartDate, StopDate
//
//=========================================================================
void TEcnaRun::StartStopTime(time_t t_startime, time_t t_stoptime) {
  // Put the start an stop time (if they exist) in fFileHeader class attributes.

  fFileHeader->fStartTime = t_startime;
  fFileHeader->fStopTime = t_stoptime;
}

void TEcnaRun::StartStopDate(const TString& c_startdate, const TString& c_stopdate) {
  // Put the start an stop date (if they exist) in fFileHeader class attributes.

  fFileHeader->fStartDate = c_startdate;
  fFileHeader->fStopDate = c_stopdate;
}

//=========================================================================
//
//                         GetReadyToCompute()   (technical)
//
//=========================================================================
void TEcnaRun::GetReadyToCompute() {
  //
  // MAKE THE RESULTS FILE NAME and
  // CHECK OF THE NUMBER OF FOUND EVENTS AND init fNumberOfEvents
  // (number used to compute the average values over the events)
  // The number of events fNumberOfEvents is extracted from the array fT2d_NbOfEvts[]
  // which has been filled by the GetSampleAdcValues(...) method

  //..................... Making of the Root File name that will be written
  fCnaWrite->RegisterFileParameters(fFileHeader->fTypAna.Data(),
                                    fFileHeader->fNbOfSamples,
                                    fFileHeader->fRunNumber,
                                    fFileHeader->fFirstReqEvtNumber,
                                    fFileHeader->fLastReqEvtNumber,
                                    fFileHeader->fReqNbOfEvts,
                                    fFileHeader->fStex);

  fCnaWrite->fMakeResultsFileName();  // set fRootFileName, fRootFileNameShort

  //..................... Checking numbers of found events channel by channel
  if (fT2d_NbOfEvts != nullptr) {
    fNumberOfEvents = fCnaWrite->NumberOfEventsAnalysis(
        fT2d_NbOfEvts, fEcal->MaxCrysEcnaInStex(), fNbSampForFic, fFileHeader->fReqNbOfEvts);
  } else {
    std::cout << "*TEcnaRun::GetReadyToCompute()> no data? fT2d_NbOfEvts = " << fT2d_NbOfEvts << std::endl;
  }
}
//  end of GetReadyToCompute()

//-------------------------------------------------------------------
//
//                      SampleValues()      (technical)
//
//  Written in .root file corresponding to analysis name
//  beginning with: "Adc" (see EcnaAnalyzer.cc in package "Modules")
//
//-------------------------------------------------------------------
void TEcnaRun::SampleValues() {
  //3D histo of the sample ADC values for all the triples (StexEcha, sample, event)

  // The histo is already in fT3d_AdcValues[][][]
  // this method sets the "Tag", increment the "f...Calc" (and must be kept for that)
  // f...Calc > 0  => allow writing on file.

  if (fFileHeader->fAdcEvtCalc > 0) {
    fFileHeader->fAdcEvtCalc = 0;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    fTagAdcEvt[i0StexEcha] = 1;
    fFileHeader->fAdcEvtCalc++;
  }
}

//=========================================================================
//
//               C A L C U L A T I O N    M E T H O D S
//
//     fTag... = 1 => Calculation done. OK for writing on result file
//     ...Calc++   => Incrementation for result file size.
//
//=========================================================================
void TEcnaRun::StandardCalculations() {
  SampleMeans();
  SampleSigmas();
  CorrelationsBetweenSamples();

  Pedestals();  // => mean over Xtal's
  TotalNoise();
  LowFrequencyNoise();
  HighFrequencyNoise();
  MeanCorrelationsBetweenSamples();
  SigmaOfCorrelationsBetweenSamples();

  AveragePedestals();  // Average => mean over Stin's (Tower if EB, SC if EE)
  AverageTotalNoise();
  AverageLowFrequencyNoise();
  AverageHighFrequencyNoise();
  AverageMeanCorrelationsBetweenSamples();
  AverageSigmaOfCorrelationsBetweenSamples();
}

void TEcnaRun::Expert1Calculations() {
  // long time, big file

  LowFrequencyCorrelationsBetweenChannels();
  HighFrequencyCorrelationsBetweenChannels();
}

void TEcnaRun::Expert2Calculations() {
  // long time, no big file
  // expert 1 is called (if not called before) without writing in file.
  // Results are used only in memory to compute expert2 calculations

  LowFrequencyMeanCorrelationsBetweenStins();
  HighFrequencyMeanCorrelationsBetweenStins();
}
//====================================================================
//
//       E X P E C T A T I O N   V A L U E S  ,  V A R I A N C E S
//
//====================================================================
//----------------------------------------------------------------
//  Calculation of the expectation values of the samples
//  for all the StexEchas
//
//  SMean(c,s)  = E_e[A(c,s,e*)]
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  E_e : average over the events
//----------------------------------------------------------------
void TEcnaRun::SampleMeans() {
  // Calculation of the expectation values over events
  // for the samples 0 to fNbSampForCalc and for all the StexEchas

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::SampleMeans() " << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation: sample expectation values over the events"
              << " for each channel." << std::endl;
  }

  //................... Allocation fT2d_ev
  if (fT2d_ev == nullptr) {
    Int_t n_samp = fNbSampForCalc;
    Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT2d_ev = new Double_t*[n_StexEcha];
    fCnew++;
    fT1d_ev = new Double_t[n_StexEcha * n_samp];
    fCnew++;
    for (Int_t i = 0; i < n_StexEcha; i++) {
      fT2d_ev[i] = &fT1d_ev[0] + i * n_samp;
    }
  }
  //................... init fT2d_ev to zero
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      if (fT2d_ev[i0StexEcha][i0Sample] != (Double_t)0) {
        fMiscDiag[1]++;
        fT2d_ev[i0StexEcha][i0Sample] = (Double_t)0;
      }
    }
  }

  //................... Calculation
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      for (Int_t i_event = 0; i_event < fNumberOfEvents; i_event++) {
        fT2d_ev[i0StexEcha][i0Sample] += fT3d_AdcValues[i0StexEcha][i0Sample][i_event];
      }
      fT2d_ev[i0StexEcha][i0Sample] /= fNumberOfEvents;
    }
  }
  fTagMSp[0] = 1;
  fFileHeader->fMSpCalc++;
}

//--------------------------------------------------------
//  Calculation of the sigmas of the samples
//  for all the StexEchas
//
//  SSigma(c,s) = sqrt{ Cov_e[A(c,s,e*),A(c,s,e*)] }
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  Cov_e : covariance over the events
//--------------------------------------------------------
void TEcnaRun::SampleSigmas() {
  //Calculation of the sigmas of the samples for all the StexEchas

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::SampleSigmas()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation: sample ADC sigmas over the events "
              << " for each channel." << std::endl;
  }

  //... preliminary calculation of the expectation values if not done yet.
  //    The tag is set to 1 after call to the method. It is reset to 0
  //    because the expectation values must not be written in the result ROOT file
  //    (since the tag was equal to 0)
  if (fTagMSp[0] != 1) {
    SampleMeans();
    fTagMSp[0] = 0;
  }

  //................... Allocation fT2d_sig
  if (fT2d_sig == nullptr) {
    Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    Int_t n_samp = fNbSampForCalc;
    fT2d_sig = new Double_t*[n_StexEcha];
    fCnew++;
    fT1d_sig = new Double_t[n_StexEcha * n_samp];
    fCnew++;
    for (Int_t i0StexEcha = 0; i0StexEcha < n_StexEcha; i0StexEcha++) {
      fT2d_sig[i0StexEcha] = &fT1d_sig[0] + i0StexEcha * n_samp;
    }
  }
  // ................... init fT2d_sig to zero
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      if (fT2d_sig[i0StexEcha][i0Sample] != (Double_t)0) {
        fMiscDiag[2]++;
        fT2d_sig[i0StexEcha][i0Sample] = (Double_t)0;
      }
    }
  }

  //................... Calculation
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      Double_t variance = (Double_t)0.;
      for (Int_t i_event = 0; i_event < fNumberOfEvents; i_event++) {
        Double_t ecart = fT3d_AdcValues[i0StexEcha][i0Sample][i_event] - fT2d_ev[i0StexEcha][i0Sample];
        variance += ecart * ecart;
      }
      variance /= fNumberOfEvents;
      fT2d_sig[i0StexEcha][i0Sample] = sqrt(variance);
    }
  }
  fTagSSp[0] = 1;
  fFileHeader->fSSpCalc++;
}

//====================================================================
//
//       C O V A R I A N C E S   &   C O R R E L A T I O N S
//
//                 B E T W E E N   S A M P L E S
//
//====================================================================
//-----------------------------------------------------------
//  Calculation of the covariances between samples
//  for all the StexEchas
//  Cov(c;s,s') = Cov_e[ A(c,s,e*) , A(c,s',e*) ]
//              = E_e[ ( A(c,s,e*) - E_e[A(c,s,e*)] )*
//                     ( A(c,s',e*) - E_e[A(c,s',e*)] ) ]
//  A(c,s,e)    : ADC value for channel c, sample s, event e
//  E_e , Cov_e : average, covariance over the events
//-----------------------------------------------------------
void TEcnaRun::CovariancesBetweenSamples() {
  //Calculation of the covariances between samples for all the StexEchas

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::CovariancesBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation: covariances between samples"
              << " for each channel." << std::endl;
  }

  //................... Allocations cov_ss
  if (fT3d_cov_ss == nullptr) {
    const Int_t n_samp = fNbSampForCalc;
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT3d_cov_ss = new Double_t**[n_StexEcha];
    fCnew++;
    fT3d2_cov_ss = new Double_t*[n_StexEcha * n_samp];
    fCnew++;
    fT3d1_cov_ss = new Double_t[n_StexEcha * n_samp * n_samp];
    fCnew++;
    for (Int_t i = 0; i < n_StexEcha; i++) {
      fT3d_cov_ss[i] = &fT3d2_cov_ss[0] + i * n_samp;
      for (Int_t j = 0; j < n_samp; j++) {
        fT3d2_cov_ss[n_samp * i + j] = &fT3d1_cov_ss[0] + n_samp * (n_samp * i + j);
      }
    }
  }

  //.................. Calculation (= init)
  //.................. computation of half of the matrix, diagonal included)

  //... preliminary calculation of the expectation values if not done yet.
  //    The tag is set to 1 after call to the method. It is reset to 0
  //    because the expectation values must not be written in the result ROOT file
  //    (since the tag was equal to 0)
  //    Results in array  fT2d_ev[j0StexEcha][i0Sample]
  if (fTagMSp[0] != 1) {
    SampleMeans();
    fTagMSp[0] = 0;
  }

  for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample <= i0Sample; j0Sample++) {
        fT3d_cov_ss[j0StexEcha][i0Sample][j0Sample] = (Double_t)0;
        for (Int_t i_event = 0; i_event < fNumberOfEvents; i_event++) {
          fT3d_cov_ss[j0StexEcha][i0Sample][j0Sample] +=
              (fT3d_AdcValues[j0StexEcha][i0Sample][i_event] - fT2d_ev[j0StexEcha][i0Sample]) *
              (fT3d_AdcValues[j0StexEcha][j0Sample][i_event] - fT2d_ev[j0StexEcha][j0Sample]);
        }
        fT3d_cov_ss[j0StexEcha][i0Sample][j0Sample] /= (Double_t)fNumberOfEvents;
        fT3d_cov_ss[j0StexEcha][j0Sample][i0Sample] = fT3d_cov_ss[j0StexEcha][i0Sample][j0Sample];
      }
    }
    fTagCovCss[j0StexEcha] = 1;
    fFileHeader->fCovCssCalc++;
  }
}

//-----------------------------------------------------------
//
//  Calculation of the correlations between samples
//  for all the StexEchas
//  Cor(c;s,s') = Cov(c;s,s')/sqrt{ Cov(c;s,s)*Cov(c;s',s') }
//-----------------------------------------------------------
void TEcnaRun::CorrelationsBetweenSamples() {
  //Calculation of the correlations between samples for all the StexEchas

  //... preliminary calculation of the covariances if not done yet.
  //    Test only the first tag since the cov are computed globaly
  //    but set all the tags to 0 because we don't want to write
  //    the covariances in the result ROOT file
  if (fTagCovCss[0] != 1) {
    CovariancesBetweenSamples();
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      fTagCovCss[j0StexEcha] = 0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::CorrelationsBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation: correlations between samples"
              << " for each channel." << std::endl;
  }

  //................... Allocations cor_ss
  if (fT3d_cor_ss == nullptr) {
    const Int_t n_samp = fNbSampForCalc;
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT3d_cor_ss = new Double_t**[n_StexEcha];
    fCnew++;
    fT3d2_cor_ss = new Double_t*[n_StexEcha * n_samp];
    fCnew++;
    fT3d1_cor_ss = new Double_t[n_StexEcha * n_samp * n_samp];
    fCnew++;
    for (Int_t i = 0; i < n_StexEcha; i++) {
      fT3d_cor_ss[i] = &fT3d2_cor_ss[0] + i * n_samp;
      for (Int_t j = 0; j < n_samp; j++) {
        fT3d2_cor_ss[n_samp * i + j] = &fT3d1_cor_ss[0] + n_samp * (n_samp * i + j);
      }
    }
  }

  //..................... calculation of the correlations (=init)
  //......................computation of half of the matrix, diagonal included (verif = 1)

  for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample <= i0Sample; j0Sample++) {
        if ((fT3d_cov_ss[j0StexEcha][i0Sample][i0Sample] > 0) && (fT3d_cov_ss[j0StexEcha][j0Sample][j0Sample] > 0)) {
          fT3d_cor_ss[j0StexEcha][i0Sample][j0Sample] =
              fT3d_cov_ss[j0StexEcha][i0Sample][j0Sample] /
              (sqrt(fT3d_cov_ss[j0StexEcha][i0Sample][i0Sample]) * sqrt(fT3d_cov_ss[j0StexEcha][j0Sample][j0Sample]));
        } else {
          (fT3d_cor_ss)[j0StexEcha][i0Sample][j0Sample] = (Double_t)0;  // prevoir compteur + fTTBELL
        }
        fT3d_cor_ss[j0StexEcha][j0Sample][i0Sample] = fT3d_cor_ss[j0StexEcha][i0Sample][j0Sample];
      }
    }
    fTagCorCss[j0StexEcha] = 1;
    fFileHeader->fCorCssCalc++;
  }
}

//===========================================================================
//
//     M E A N   P E D E S T A L S ,   T O T A L    N O I S E ,
//     L O W     F R E Q U E N C Y    N O I S E ,
//     H I G H   F R E Q U E N C Y    N O I S E
//     M E A N   O F  C O R ( S , S ),  S I G M A   O F  C O R ( S , S )
//
//===========================================================================
//-------------------------------------------------------------------------
//
//  Calculation of the Pedestals for each channel in Stex
//  tag: Ped
//  Pedestal(c ) = E_e[ E_s[A(c ,s*,e*)] ]
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  E_e : average over the events
//  E_s : average over the samples
//
//-------------------------------------------------------------------------
void TEcnaRun::Pedestals() {
  // Calculation, for each channel, of the expectation values
  // (over the samples 0 to fNbSampForCalc-1) of the ADC expectation values
  // (over the events)

  //... preliminary calculation of the expectation values if not done yet
  if (fTagMSp[0] != 1) {
    SampleMeans();
    fTagMSp[0] = 0;
  }

  //................... Allocation ev_ev + init to zero (mandatory)
  if (fT1d_ev_ev == nullptr) {
    fT1d_ev_ev = new Double_t[fEcal->MaxCrysEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_ev_ev[i0StexEcha] != (Double_t)0) {
      fMiscDiag[11]++;
      fT1d_ev_ev[i0StexEcha] = (Double_t)0;
    }
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::Pedestals()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the channels, of the expectation values (over the samples 1 to "
              << fNbSampForCalc << ")" << std::endl
              << "          of the ADC expectation values (over the events)." << std::endl;
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      fT1d_ev_ev[i0StexEcha] += fT2d_ev[i0StexEcha][i0Sample];
    }
    fT1d_ev_ev[i0StexEcha] /= fNbSampForCalc;
  }
  fTagPed[0] = 1;
  fFileHeader->fPedCalc++;
}
//------------------------ (end of Pedestals) ----------------------------

//------------------------------------------------------------------------------
//
//  Calculation of the TN (Total Noise)
//  tag: Tno
//
//  TotalNoise(c)  = E_s[ sqrt{ E_e[ ( A(c ,s*,e*) - E_e[A(c ,s*,e*)] )^2 ] } ]
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  E_e : average over the events
//  E_s : average over the samples
//
//------------------------------------------------------------------------------
void TEcnaRun::TotalNoise() {
  // Calculation, for each channel, of the expectation values
  // (over the samples 0 to fNbSampForCalc-1) of the sigmas
  // (over the events)

  //... preliminary calculation of the sigmas if not done yet
  if (fTagSSp[0] != 1) {
    SampleSigmas();
    fTagSSp[0] = 0;
  }

  //................... Allocation ev_ev + init to zero (mandatory)
  if (fT1d_evsamp_of_sigevt == nullptr) {
    fT1d_evsamp_of_sigevt = new Double_t[fEcal->MaxCrysEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_evsamp_of_sigevt[i0StexEcha] != (Double_t)0) {
      fMiscDiag[12]++;
      fT1d_evsamp_of_sigevt[i0StexEcha] = (Double_t)0;
    }
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::TotalNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the channels, of the expectation values (over the samples 1 to "
              << fNbSampForCalc << ")" << std::endl
              << "          of the ADC expectation values (over the events)." << std::endl;
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      if (fT2d_sig[i0StexEcha][i0Sample] < 0) {
        std::cout << "!TEcnaRun::TotalNoise() *** ERROR ***> Negative sigma!" << fTTBELL << std::endl;
      } else {
        fT1d_evsamp_of_sigevt[i0StexEcha] += fT2d_sig[i0StexEcha][i0Sample];
      }
    }
    fT1d_evsamp_of_sigevt[i0StexEcha] /= fNbSampForCalc;
  }
  fTagTno[0] = 1;
  fFileHeader->fTnoCalc++;
}
//------------------------ (end of TotalNoise) ----------------------------

//---------------------------------------------------------------------------------
//
//  Calculation of the LFN  (Low Frequency Noise)
//  tag: Lfn
//
//  LowFqNoise(c) = sqrt{ E_e[ ( E_s[A(c ,s*,e*)] - E_e[ E_s[A(c ,s*,e*)] ] )^2 ] }
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  E_e : average over the events
//  E_s : average over the samples
//
//---------------------------------------------------------------------------------
void TEcnaRun::LowFrequencyNoise() {
  // Calculation, for each channel, of the sigma (over the events)
  // of the ADC expectation values (over the samples 0 to fNbSampForCalc-1)

  //................... Allocation fT1d_sigevt_of_evsamp + init to zero (mandatory)
  if (fT1d_sigevt_of_evsamp == nullptr) {
    fT1d_sigevt_of_evsamp = new Double_t[fEcal->MaxCrysEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_sigevt_of_evsamp[i0StexEcha] != (Double_t)0) {
      fMiscDiag[13]++;
      fT1d_sigevt_of_evsamp[i0StexEcha] = (Double_t)0;
    }
  }

  //................... Allocation mean_over_samples
  TVectorD mean_over_samples(fNumberOfEvents);
  for (Int_t i = 0; i < fNumberOfEvents; i++) {
    mean_over_samples(i) = (Double_t)0.;
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::LowFrequencyNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for each channel, of the sigma (over the events)" << std::endl
              << "           of the ADC expectation values (over the samples 1 to " << fNbSampForCalc << ")."
              << std::endl;
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    // Calculation of the mean over the events of the mean over the samples
    Double_t mean_over_events = (Double_t)0;
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      // Calculation, for each event, of the mean over the samples
      mean_over_samples(n_event) = (Double_t)0.;
      for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
        mean_over_samples(n_event) += fT3d_AdcValues[i0StexEcha][i0Sample][n_event];
      }
      mean_over_samples(n_event) /= (Double_t)fNbSampForCalc;

      mean_over_events += mean_over_samples(n_event);
    }
    mean_over_events /= (Double_t)fNumberOfEvents;

    // Calculation of the sigma over the events of the mean over the samples
    Double_t var = (Double_t)0;
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      Double_t ecart = mean_over_samples(n_event) - mean_over_events;
      var += ecart * ecart;
    }
    var /= (Double_t)fNumberOfEvents;

    fT1d_sigevt_of_evsamp[i0StexEcha] = sqrt(var);
  }
  fTagLfn[0] = 1;
  fFileHeader->fLfnCalc++;
}
//------------------------ (end of LowFrequencyNoise) ----------------------------

//---------------------------------------------------------------------------------
//
//  Calculation of the HFN  (High Frequency Noise)
//  tag: Hfn
//
//  HighFqNoise(c) = E_e[ sqrt{ E_s[ (A(c ,s*,e*) - E_s[A(c ,s*,e*)] )^2 ] } ]
//  A(c,s,e) : ADC value for channel c, sample s, event e
//  E_e : average over the events
//  E_s : average over the samples
//
//---------------------------------------------------------------------------------
void TEcnaRun::HighFrequencyNoise() {
  // Calculation, for each channel, of the mean (over the events)
  // of the ADC sigmas (over the samples 0 to fNbSampForCalc-1)

  //................... Allocation fT1d_evevt_of_sigsamp + init to zero (mandatory)
  if (fT1d_evevt_of_sigsamp == nullptr) {
    fT1d_evevt_of_sigsamp = new Double_t[fEcal->MaxCrysEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_evevt_of_sigsamp[i0StexEcha] != (Double_t)0) {
      fMiscDiag[14]++;
      fT1d_evevt_of_sigsamp[i0StexEcha] = (Double_t)0;
    }
  }

  //................... Allocations mean_over_samples, sigma_over_sample
  TVectorD mean_over_samples(fNumberOfEvents);
  for (Int_t i = 0; i < fNumberOfEvents; i++) {
    mean_over_samples(i) = (Double_t)0.;
  }
  TVectorD sigma_over_samples(fNumberOfEvents);
  for (Int_t i = 0; i < fNumberOfEvents; i++) {
    sigma_over_samples(i) = (Double_t)0.;
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::HighFrequencyNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for each channel, of the sigma (over the events)" << std::endl
              << "           of the ADC expectation values (over the samples 1 to " << fNbSampForCalc << ")."
              << std::endl;
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    //..................... Calculation of the sigma over samples
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      // Calculation, for each event, of the mean over the samples
      mean_over_samples(n_event) = (Double_t)0.;
      for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
        mean_over_samples(n_event) += fT3d_AdcValues[i0StexEcha][i0Sample][n_event];
      }
      mean_over_samples(n_event) /= (Double_t)fNbSampForCalc;

      // Calculation, for each event, of the sigma over the samples
      Double_t var_over_samples = (Double_t)0;
      for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
        Double_t deviation = fT3d_AdcValues[i0StexEcha][i0Sample][n_event] - mean_over_samples(n_event);
        var_over_samples += deviation * deviation;
      }
      var_over_samples /= (Double_t)fNbSampForCalc;

      if (var_over_samples < 0) {
        std::cout << "!TEcnaRun::HighFrequencyNoise() *** ERROR ***> Negative variance! " << fTTBELL << std::endl;
      } else {
        sigma_over_samples(n_event) = sqrt(var_over_samples);
      }
    }

    //....... Calculation of the mean over the events of the sigma over samples
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      fT1d_evevt_of_sigsamp[i0StexEcha] += sigma_over_samples(n_event);
    }

    fT1d_evevt_of_sigsamp[i0StexEcha] /= (Double_t)fNumberOfEvents;
  }
  fTagHfn[0] = 1;
  fFileHeader->fHfnCalc++;
}
//------------------------ (end of HighFrequencyNoise) ----------------------------

//-------------------------------------------------------------------------
//
//  Calculation of the expectation values of (sample,sample)
//  correlations for all the channels (mean cor(s,s))
//  tag: MeanCorss
//
//  MeanCorss(c)   = E_s,s'[ Cor(c;s,s') ]
//  E_s,s': average  over couples of samples (half correlation matrix)
//
//-------------------------------------------------------------------------
void TEcnaRun::MeanCorrelationsBetweenSamples() {
  // Calculation, for all the channels, of the expectation values
  // of the correlations between the first fNbSampForCalc samples

  //... preliminary calculation of the correlations if not done yet
  //    (test only the first element since the cor are computed globaly)
  if (fTagCorCss[0] != 1) {
    CorrelationsBetweenSamples();
    fTagCorCss[0] = 0;
  }

  //................... Allocations ev_cor_ss + init to zero (mandatory)
  if (fT1d_ev_cor_ss == nullptr) {
    Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT1d_ev_cor_ss = new Double_t[n_StexEcha];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_ev_cor_ss[i0StexEcha] != (Double_t)0) {
      fMiscDiag[15]++;
      fT1d_ev_cor_ss[i0StexEcha] = (Double_t)0;
    }
  }

  //.......... 1D array half_cor_ss[N(N-1)/2] to put the N (sample,sample) correlations
  //           ( half of (them minus the diagonal) )
  Int_t ndim = (Int_t)(fNbSampForCalc * (fNbSampForCalc - 1) / 2);

  TVectorD half_cor_ss(ndim);
  for (Int_t i = 0; i < ndim; i++) {
    half_cor_ss(i) = (Double_t)0.;
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::MeanCorrelationsBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the channels, of the expectation values of the" << std::endl
              << "           correlations between the first " << fNbSampForCalc << " samples." << std::endl;
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    //..................... half_cor_ss() array filling
    Int_t i_count = 0;
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample < i0Sample; j0Sample++) {
        half_cor_ss(i_count) = fT3d_cor_ss[i0StexEcha][i0Sample][j0Sample];
        i_count++;
      }
    }
    //...................... mean cor(s,s') calculation
    fT1d_ev_cor_ss[i0StexEcha] = (Double_t)0;
    for (Int_t i_rcor = 0; i_rcor < ndim; i_rcor++) {
      fT1d_ev_cor_ss[i0StexEcha] += half_cor_ss(i_rcor);
    }
    fT1d_ev_cor_ss[i0StexEcha] /= (Double_t)ndim;
  }
  fTagMeanCorss[0] = 1;
  fFileHeader->fMeanCorssCalc++;
}
//--------------- (end of MeanCorrelationsBetweenSamples) -----------

//-------------------------------------------------------------------------
//
// Calculation of the sigmas of the (sample,sample) correlations
// for all the channels (sigma of cor(s,s))
// tag: SigCorss
//
// SigmaCorss(c)  = E_s,s'[ Cor(c;s,s') - E_s,s'[ Cor(c;s,s') ] ]
// E_s,s': average  over couples of samples (half correlation matrix)
//
//--------------------------------------------------------------------------
void TEcnaRun::SigmaOfCorrelationsBetweenSamples() {
  //Calculation of the sigmas of the (sample,sample) correlations for all the StexEchas

  //... preliminary calculation of the mean cor(s,s') if not done yet
  //    (test only the first element since the cor are computed globaly)
  //    Results available in array fT1d_ev_cor_ss[i0StexEcha]
  if (fTagMeanCorss[0] != 1) {
    MeanCorrelationsBetweenSamples();
    fTagMeanCorss[0] = 0;
  }

  //................... Allocations sig_cor_ss + init to zero
  if (fT1d_sig_cor_ss == nullptr) {
    Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT1d_sig_cor_ss = new Double_t[n_StexEcha];
    fCnew++;
  }
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if (fT1d_sig_cor_ss[i0StexEcha] != (Double_t)0) {
      fMiscDiag[16]++;
      fT1d_sig_cor_ss[i0StexEcha] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::SigmasOfCorrelationsBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the sigmas of the (sample,sample)" << std::endl
              << "           correlations for all the channels." << std::endl;
  }

  //.......... 1D array half_cor_ss[N(N-1)/2] to put the N (sample,sample) correlations
  //           (half of them minus the diagonal)
  Int_t ndim = (Int_t)(fNbSampForCalc * (fNbSampForCalc - 1) / 2);

  TVectorD half_cor_ss(ndim);
  for (Int_t i = 0; i < ndim; i++) {
    half_cor_ss(i) = (Double_t)0.;
  }

  //.................. Calculation
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    //..................... half_cor_ss() array filling
    Int_t i_count = 0;
    for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample < i0Sample; j0Sample++) {
        half_cor_ss(i_count) = fT3d_cor_ss[i0StexEcha][i0Sample][j0Sample];
        i_count++;
      }
    }

    //...................... sigma of cor(s,s') calculation
    Double_t var = (Double_t)0;
    for (Int_t i_rcor = 0; i_rcor < ndim; i_rcor++) {
      Double_t ecart = half_cor_ss(i_rcor) - fT1d_ev_cor_ss[i0StexEcha];
      var += ecart * ecart;
    }
    var /= (Double_t)ndim;
    fT1d_sig_cor_ss[i0StexEcha] = sqrt(var);
  }
  fTagSigCorss[0] = 1;
  fFileHeader->fSigCorssCalc++;
}
//--------------- (end of SigmaOfCorrelationsBetweenSamples) -----------

//-----------------------------------------------------------------------------
//
//  Calculation of the average Pedestals for each Stin in Stex
//  tag: AvPed
//
//-----------------------------------------------------------------------------
void TEcnaRun::AveragePedestals() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the Pedestals

  //... preliminary calculation of the Pedestals if not done yet
  if (fTagPed[0] != 1) {
    Pedestals();
    fTagPed[0] = 0;
  }
  //................... Allocation av_mped + init to zero (mandatory)
  if (fT1d_av_mped == nullptr) {
    fT1d_av_mped = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_mped[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[41]++;
      fT1d_av_mped[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AveragePedestals()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average Pedestals"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_mped[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_mped[i0StexStinEcna] += fT1d_ev_ev[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_mped[i0StexStinEcna] += fT1d_ev_ev[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_mped[i0StexStinEcna] = fT1d_av_mped[i0StexStinEcna] / xdivis;
  }

  fTagAvPed[0] = 1;
  fFileHeader->fAvPedCalc++;
}
//-----------------------------------------------------------------------------
//
// Calculation of the average total noise for each Stin in Stex
// tag: AvTno
//
//-----------------------------------------------------------------------------
void TEcnaRun::AverageTotalNoise() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the Total Noise

  //... preliminary calculation of the averaged Total Noise if not done yet
  if (fTagTno[0] != 1) {
    TotalNoise();
    fTagTno[0] = 0;
  }
  //................... Allocation av_totn + init to zero (mandatory)
  if (fT1d_av_totn == nullptr) {
    fT1d_av_totn = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_totn[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[42]++;
      fT1d_av_totn[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AverageTotalNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average total Noise"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_totn[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_totn[i0StexStinEcna] += fT1d_evsamp_of_sigevt[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_totn[i0StexStinEcna] += fT1d_evsamp_of_sigevt[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_totn[i0StexStinEcna] = fT1d_av_totn[i0StexStinEcna] / xdivis;
  }
  fTagAvTno[0] = 1;
  fFileHeader->fAvTnoCalc++;
}
//-----------------------------------------------------------------------------
//
// Calculation of the average Low Frequency noise for each Stin in Stex
// tag: AvLfn
//
//-----------------------------------------------------------------------------
void TEcnaRun::AverageLowFrequencyNoise() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the Low Frequency Noise

  //... preliminary calculation of the Low Frequency Noise if not done yet
  if (fTagLfn[0] != 1) {
    LowFrequencyNoise();
    fTagLfn[0] = 0;
  }
  //................... Allocation av_lofn + init to zero (mandatory)
  if (fT1d_av_lofn == nullptr) {
    fT1d_av_lofn = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_lofn[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[43]++;
      fT1d_av_lofn[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AverageLowFrequencyNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average Low Frequency Noise"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_lofn[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_lofn[i0StexStinEcna] += fT1d_sigevt_of_evsamp[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_lofn[i0StexStinEcna] += fT1d_sigevt_of_evsamp[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_lofn[i0StexStinEcna] = fT1d_av_lofn[i0StexStinEcna] / xdivis;
  }
  fTagAvLfn[0] = 1;
  fFileHeader->fAvLfnCalc++;
}
//-----------------------------------------------------------------------------
//
// Calculation of the average high frequency noise for each Stin in Stex
// tag: AvHfn
//
//-----------------------------------------------------------------------------
void TEcnaRun::AverageHighFrequencyNoise() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the High Frequency Noise

  //... preliminary calculation of the High Frequency Noise if not done yet
  if (fTagHfn[0] != 1) {
    HighFrequencyNoise();
    fTagHfn[0] = 0;
  }
  //................... Allocation av_hifn + init to zero (mandatory)
  if (fT1d_av_hifn == nullptr) {
    fT1d_av_hifn = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_hifn[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[44]++;
      fT1d_av_hifn[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AverageHighFrequencyNoise()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average High Frequency Noise"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_hifn[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_hifn[i0StexStinEcna] += fT1d_evevt_of_sigsamp[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_hifn[i0StexStinEcna] += fT1d_evevt_of_sigsamp[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_hifn[i0StexStinEcna] = fT1d_av_hifn[i0StexStinEcna] / xdivis;
  }
  fTagAvHfn[0] = 1;
  fFileHeader->fAvHfnCalc++;
}
//-----------------------------------------------------------------------------
//
// Calculation of the average mean cor(s,s) for each Stin in Stex
// tag: AvMeanCorss
//
//-----------------------------------------------------------------------------
void TEcnaRun::AverageMeanCorrelationsBetweenSamples() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the mean cor(s,s)

  //... preliminary calculation of the mean cor(s,s) if not done yet
  if (fTagMeanCorss[0] != 1) {
    MeanCorrelationsBetweenSamples();
    fTagMeanCorss[0] = 0;
  }
  //................... Allocation av_ev_corss + init to zero (mandatory)
  if (fT1d_av_ev_corss == nullptr) {
    fT1d_av_ev_corss = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_ev_corss[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[45]++;
      fT1d_av_ev_corss[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AverageMeanCorrelationsBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average mean cor(s,s)"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_ev_corss[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_ev_corss[i0StexStinEcna] += fT1d_ev_cor_ss[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_ev_corss[i0StexStinEcna] += fT1d_ev_cor_ss[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_ev_corss[i0StexStinEcna] = fT1d_av_ev_corss[i0StexStinEcna] / xdivis;
  }
  fTagAvMeanCorss[0] = 1;
  fFileHeader->fAvMeanCorssCalc++;
}
//-----------------------------------------------------------------------------
//
// Calculation of the average sigma of cor(s,s) for each Stin in Stex
// tag: AvSigCorss
//
//-----------------------------------------------------------------------------
void TEcnaRun::AverageSigmaOfCorrelationsBetweenSamples() {
  // Calculation of the average
  // (over the Stin's 0 to fEcal->MaxStinInStex()) of the sigma of cor(s,s)

  //... preliminary calculation of the sigma of cor(s,s) if not done yet
  if (fTagSigCorss[0] != 1) {
    SigmaOfCorrelationsBetweenSamples();
    fTagSigCorss[0] = 0;
  }
  //................... Allocation av_sig_corss + init to zero (mandatory)
  if (fT1d_av_sig_corss == nullptr) {
    fT1d_av_sig_corss = new Double_t[fEcal->MaxStinEcnaInStex()];
    fCnew++;
  }
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    if (fT1d_av_sig_corss[i0StexStinEcna] != (Double_t)0) {
      fMiscDiag[46]++;
      fT1d_av_sig_corss[i0StexStinEcna] = (Double_t)0;
    }
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::AverageSigmaOfCorrelationsBetweenSamples()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for all the " << fStinName.Data() << "s, of the average sigma of cor(s,s)"
              << std::endl;
  }

  //................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    Int_t n1StexStinEcna = i0StexStinEcna + 1;
    fT1d_av_sig_corss[i0StexStinEcna] = (Double_t)0;
    for (Int_t i0StinEcha = 0; i0StinEcha < fEcal->MaxCrysInStin(); i0StinEcha++) {
      Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStinEcna, i0StinEcha);

      if (fStexName == "SM ") {
        fT1d_av_sig_corss[i0StexStinEcna] += fT1d_sig_cor_ss[i0StexEcha];
      }

      if (fStexName == "Dee") {
        //---------------- Special translation for mixed SCEcna (29 and 32)
        //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
        //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
        Int_t n1StinEcha = i0StinEcha + 1;
        if (n1StexStinEcna == 10 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);
        }
        if (n1StexStinEcna == 11 && n1StinEcha == 11) {
          i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);
        }
        if (!((n1StexStinEcna == 29 || n1StexStinEcna == 32) && n1StinEcha == 11)) {
          fT1d_av_sig_corss[i0StexStinEcna] += fT1d_sig_cor_ss[i0StexEcha];
        }
      }
    }
    Double_t xdivis = (Double_t)0.;
    if (fStexName == "SM ") {
      xdivis = (Double_t)fEcal->MaxCrysInStin();
    }
    if (fStexName == "Dee") {
      xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, n1StexStinEcna, "TEcnaRun");
    }

    fT1d_av_sig_corss[i0StexStinEcna] = fT1d_av_sig_corss[i0StexStinEcna] / xdivis;
  }
  fTagAvSigCorss[0] = 1;
  fFileHeader->fAvSigCorssCalc++;
}

//======================================================================
//
//       C O V A R I A N C E S   &   C O R R E L A T I O N S
//
//                 B E T W E E N   C H A N N E L S
//
//======================================================================
//----------------------------------------------------------------------
//
//  Calculation of the Low Frequency Covariances between channels
//
//  LFCov(Ci,Cj) = Cov_e[ E_s[A(Ci,s*,e*)] , E_s[A(Cj,s*,e*) ]
//
//               = E_e[ ( E_s[A(Ci,s*,e*)] - E_e[ E_s[A(Ci,s*,e*)] ] )*
//                      ( E_s[A(Cj,s*,e*)] - E_e[ E_s[A(Cj,s*,e*)] ] ) ]
//
//   A(Ci,s,e) : ADC value for channel Ci, sample s, event e
//
//   E_e , Cov_e : average, covariance over the events
//   E_s :         average over the samples
//
//   e* : random variable associated to events
//   s* : random variable associated to samples
//
//----------------------------------------------------------------------
void TEcnaRun::LowFrequencyCovariancesBetweenChannels() {
  //Calculation of the Low Frequency Covariances between channels

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::LowFrequencyCovariancesBetweenChannels()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the Low Frequency Covariances between channels" << std::endl;
  }

  //................. allocation fT2d_lf_cov + init to zero (mandatory)
  if (fT2d_lf_cov == nullptr) {
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT2d_lf_cov = new Double_t*[n_StexEcha];
    fCnew++;
    fT2d1_lf_cov = new Double_t[n_StexEcha * n_StexEcha];
    fCnew++;
    for (Int_t i0StexEcha = 0; i0StexEcha < n_StexEcha; i0StexEcha++) {
      fT2d_lf_cov[i0StexEcha] = &fT2d1_lf_cov[0] + i0StexEcha * n_StexEcha;
    }
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      if (fT2d_lf_cov[i0StexEcha][j0StexEcha] != (Double_t)0) {
        fMiscDiag[21]++;
        fT2d_lf_cov[i0StexEcha][j0StexEcha] = (Double_t)0;
      }
    }
  }
  //........................................... Calculation  (LowFrequencyCovariancesBetweenChannels)
  //................... Allocation mean_over_samples(i0StexEcha, n_event)
  TMatrixD mean_over_samples(fEcal->MaxCrysEcnaInStex(), fNumberOfEvents);
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      mean_over_samples(i0StexEcha, n_event) = (Double_t)0.;
    }
  }
  //................... Allocation MoeOfMos(i0StexEcha)
  TVectorD MoeOfMos(fEcal->MaxCrysEcnaInStex());
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    MoeOfMos(i0StexEcha) = (Double_t)0.;
  }

  //................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "          Calculation, for each pair of channels, of the covariance (over the events)" << std::endl
              << "          between the ADC expectation values (over the samples 1 to " << fNbSampForCalc << ")."
              << std::endl;
  }

  std::cout << " Please, wait (end at i= " << fEcal->MaxCrysEcnaInStex() << "): " << std::endl;

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    MoeOfMos(i0StexEcha) = (Double_t)0;

    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
        // Calculation, for each event, of the mean over the samples  ( = E_s[A(c_i,s*,e_n] )
        mean_over_samples(i0StexEcha, n_event) = (Double_t)0.;
        for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
          mean_over_samples(i0StexEcha, n_event) += fT3d_AdcValues[i0StexEcha][i0Sample][n_event];
        }
        mean_over_samples(i0StexEcha, n_event) /= (Double_t)fNbSampForCalc;
      }
      //Calculation of the mean over the events of E_s[A(c_i,s*,e_n] ( = E_e[E_s[A(c_i,s*,e*]] )
      for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
        MoeOfMos(i0StexEcha) += mean_over_samples(i0StexEcha, n_event);
      }
      MoeOfMos(i0StexEcha) /= (Double_t)fNumberOfEvents;
    }
  }

  //... Calculation of half of the matrix, diagonal included        (LowFrequencyCovariancesBetweenChannels)
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t j0StexEcha = 0; j0StexEcha <= i0StexEcha; j0StexEcha++) {
        if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, j0StexEcha) > 0) ||
            (fFlagSubDet == "EB")) {
          fT2d_lf_cov[i0StexEcha][j0StexEcha] = (Double_t)0;
          for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
            fT2d_lf_cov[i0StexEcha][j0StexEcha] += (mean_over_samples(i0StexEcha, n_event) - MoeOfMos(i0StexEcha)) *
                                                   (mean_over_samples(j0StexEcha, n_event) - MoeOfMos(j0StexEcha));
          }
          fT2d_lf_cov[i0StexEcha][j0StexEcha] /= (Double_t)fNumberOfEvents;

          fT2d_lf_cov[j0StexEcha][i0StexEcha] = fT2d_lf_cov[i0StexEcha][j0StexEcha];
        }
      }
      if (i0StexEcha % 100 == 0) {
        std::cout << i0StexEcha << "[LFN Cov], ";
      }
    }
  }
  std::cout << std::endl;
  fTagLfCov[0] = 1;
  fFileHeader->fLfCovCalc++;
}
//---------- (end of LowFrequencyCovariancesBetweenChannels ) --------------------

//------------------------------------------------------------------
//
//  Calculation of the Low Frequency Correlations between channels
//
//  LFCor(Ci,Cj) = LFCov(Ci,Cj)/sqrt(LFCov(Ci,Ci)*LFCov(Cj,Cj))
//
//------------------------------------------------------------------
void TEcnaRun::LowFrequencyCorrelationsBetweenChannels() {
  //Calculation of the Low Frequency Correlations between channels

  //... preliminary calculation of the covariances if not done yet.
  if (fTagLfCov[0] != 1) {
    LowFrequencyCovariancesBetweenChannels();
    fTagLfCov[0] = 0;
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::LowFrequencyCorrelationsBetweenChannels()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "          Calculation of the Low Frequency Correlations between channels" << std::endl
              << "          Starting allocation. " << std::endl;
  }

  //................. allocation fT2d_lf_cor + init to zero (mandatory)
  if (fT2d_lf_cor == nullptr) {
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT2d_lf_cor = new Double_t*[n_StexEcha];
    fCnew++;
    fT2d1_lf_cor = new Double_t[n_StexEcha * n_StexEcha];
    fCnew++;
    for (Int_t i0StexEcha = 0; i0StexEcha < n_StexEcha; i0StexEcha++) {
      fT2d_lf_cor[i0StexEcha] = &fT2d1_lf_cor[0] + i0StexEcha * n_StexEcha;
    }
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      if (fT2d_lf_cor[i0StexEcha][j0StexEcha] != (Double_t)0) {
        fMiscDiag[22]++;
        fT2d_lf_cor[i0StexEcha][j0StexEcha] = (Double_t)0;
      }
    }
  }

  //................. calculation
  //........................... computation of half of the matrix, diagonal included
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t j0StexEcha = 0; j0StexEcha <= i0StexEcha; j0StexEcha++) {
        if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, j0StexEcha) > 0) ||
            (fFlagSubDet == "EB")) {
          if (fT2d_lf_cov[i0StexEcha][i0StexEcha] > 0 && fT2d_lf_cov[j0StexEcha][j0StexEcha] > 0) {
            fT2d_lf_cor[i0StexEcha][j0StexEcha] =
                fT2d_lf_cov[i0StexEcha][j0StexEcha] /
                ((Double_t)sqrt(fT2d_lf_cov[i0StexEcha][i0StexEcha] * fT2d_lf_cov[j0StexEcha][j0StexEcha]));
          } else {
            fT2d_lf_cor[i0StexEcha][j0StexEcha] = (Double_t)0.;
          }
          fT2d_lf_cor[j0StexEcha][i0StexEcha] = fT2d_lf_cor[i0StexEcha][j0StexEcha];
        }
      }
    }
    if (i0StexEcha % 100 == 0) {
      std::cout << i0StexEcha << "[LFN Cor], ";
    }
  }
  std::cout << std::endl;

  fTagLfCor[0] = 1;
  fFileHeader->fLfCorCalc++;
}
//--------------- (end of LowFrequencyCorrelationsBetweenChannels) --------------------

//------------------------------------------------------------------
//
//  Calculation of the High Frequency Covariances between channels
//
//  HFCov(Ci,Cj) = E_e[ Cov_s[ A(Ci,s*,e*) , A(Cj,s*,e*) ] ]
//
//               = E_e[ E_s[ ( A(Ci,s*,e*) - E_s[A(Ci,s*,e*)] )*
//                           ( A(Cj,s*,e*) - E_s[A(Cj,s*,e*)] ) ] ]
//
//   A(Ci,s,e) : ADC value for channel Ci, sample s, event e
//
//   E_e         : average over the events
//   E_s , Cov_s : average, covariance over the samples
//
//------------------------------------------------------------------
void TEcnaRun::HighFrequencyCovariancesBetweenChannels() {
  //Calculation of the High Frequency Covariances between channels

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::HighFrequencyCovariancesBetweenChannels()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the High Frequency Covariances between channels" << std::endl;
  }

  //................. allocation fT2d_hf_cov + init to zero (mandatory)
  if (fT2d_hf_cov == nullptr) {
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT2d_hf_cov = new Double_t*[n_StexEcha];
    fCnew++;
    fT2d1_hf_cov = new Double_t[n_StexEcha * n_StexEcha];
    fCnew++;
    for (Int_t i0StexEcha = 0; i0StexEcha < n_StexEcha; i0StexEcha++) {
      fT2d_hf_cov[i0StexEcha] = &fT2d1_hf_cov[0] + i0StexEcha * n_StexEcha;
    }
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      if (fT2d_hf_cov[i0StexEcha][j0StexEcha] != (Double_t)0) {
        fMiscDiag[23]++;
        fT2d_hf_cov[i0StexEcha][j0StexEcha] = (Double_t)0;
      }
    }
  }

  //................... Allocation mean_over_samples(i0StexEcha, n_event)
  TMatrixD mean_over_samples(fEcal->MaxCrysEcnaInStex(), fNumberOfEvents);
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
      mean_over_samples(i0StexEcha, n_event) = (Double_t)0.;
    }
  }
  //................... Allocation cov_over_samp(i0StexEcha,j0StexEcha)
  TMatrixD cov_over_samp(fEcal->MaxCrysEcnaInStex(), fEcal->MaxCrysEcnaInStex());
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      cov_over_samp(i0StexEcha, j0StexEcha) = (Double_t)0.;
    }
  }

  //........................................... Calculation    (HighFrequencyCovariancesBetweenChannels)
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "          Calculation of the mean (over the events)" << std::endl
              << "          of the covariances between the channels (over the samples 1 to " << fNbSampForCalc << ")."
              << std::endl;
  }

  std::cout << " Please, wait (end at i= " << fEcal->MaxCrysEcnaInStex() << "): " << std::endl;

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
        // Calculation, for each event, of the mean over the samples  ( = E_s[A(c_i,s*,e_n] )
        mean_over_samples(i0StexEcha, n_event) = (Double_t)0.;
        for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
          mean_over_samples(i0StexEcha, n_event) += fT3d_AdcValues[i0StexEcha][i0Sample][n_event];
        }
        mean_over_samples(i0StexEcha, n_event) /= (Double_t)fNbSampForCalc;
      }
    }
    if (i0StexEcha % 100 == 0) {
      std::cout << i0StexEcha << "[HFNa Cov], ";
    }
  }
  std::cout << std::endl;

  std::cout << " Please, wait (end at i= " << fEcal->MaxCrysEcnaInStex() << "): " << std::endl;

  //... Calculation of half of the matrix, diagonal included    (HighFrequencyCovariancesBetweenChannels)
  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t j0StexEcha = 0; j0StexEcha <= i0StexEcha; j0StexEcha++) {
        if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, j0StexEcha) > 0) ||
            (fFlagSubDet == "EB")) {
          for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
            // Calculation, for each event, of the covariance over the samples
            cov_over_samp(i0StexEcha, j0StexEcha) = (Double_t)0;
            for (Int_t i0Sample = 0; i0Sample < fNbSampForCalc; i0Sample++) {
              cov_over_samp(i0StexEcha, j0StexEcha) +=
                  (fT3d_AdcValues[i0StexEcha][i0Sample][n_event] - mean_over_samples(i0StexEcha, n_event)) *
                  (fT3d_AdcValues[j0StexEcha][i0Sample][n_event] - mean_over_samples(j0StexEcha, n_event));
            }
            cov_over_samp(i0StexEcha, j0StexEcha) /= (Double_t)fNbSampForCalc;
          }
          //....... Calculation of the mean over the events of Cov_s[A(c_i,s*,e*),A(c_j,s*,e*)]
          //......... Calculation of half of the matrix, diagonal included
          fT2d_hf_cov[i0StexEcha][j0StexEcha] = (Double_t)0;
          for (Int_t n_event = 0; n_event < fNumberOfEvents; n_event++) {
            fT2d_hf_cov[i0StexEcha][j0StexEcha] += cov_over_samp(i0StexEcha, j0StexEcha);
          }
          fT2d_hf_cov[i0StexEcha][j0StexEcha] /= (Double_t)fNumberOfEvents;

          fT2d_hf_cov[j0StexEcha][i0StexEcha] = fT2d_hf_cov[i0StexEcha][j0StexEcha];
        }
      }
    }
    if (i0StexEcha % 100 == 0) {
      std::cout << i0StexEcha << "[HFNb Cov], ";
    }
  }
  std::cout << std::endl;

  fTagHfCov[0] = 1;
  fFileHeader->fHfCovCalc++;
}
//---------- (end of HighFrequencyCovariancesBetweenChannels ) --------------------

//------------------------------------------------------------------
//
//  Calculation of the High Frequency Correlations between channels
//
//  HFCor(Ci,Cj) = HFCov(Ci,Cj)/sqrt(HFCov(Ci,Ci)*HFCov(Cj,Cj))
//
//------------------------------------------------------------------
void TEcnaRun::HighFrequencyCorrelationsBetweenChannels() {
  //Calculation of the High Frequency Correlations between channels

  //... preliminary calculation of the covariances if not done yet.
  if (fTagHfCov[0] != 1) {
    HighFrequencyCovariancesBetweenChannels();
    fTagHfCov[0] = 0;
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::HighFrequencyCorrelationsBetweenChannels()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the High Frequency Correlations between channels" << std::endl
              << "          Starting allocation. " << std::endl;
  }

  //................. allocation fT2d_hf_cor + init to zero (mandatory)
  if (fT2d_hf_cor == nullptr) {
    const Int_t n_StexEcha = fEcal->MaxCrysEcnaInStex();
    fT2d_hf_cor = new Double_t*[n_StexEcha];
    fCnew++;
    fT2d1_hf_cor = new Double_t[n_StexEcha * n_StexEcha];
    fCnew++;
    for (Int_t i0StexEcha = 0; i0StexEcha < n_StexEcha; i0StexEcha++) {
      fT2d_hf_cor[i0StexEcha] = &fT2d1_hf_cor[0] + i0StexEcha * n_StexEcha;
    }
  }

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      if (fT2d_hf_cor[i0StexEcha][j0StexEcha] != (Double_t)0) {
        fMiscDiag[24]++;
        fT2d_hf_cor[i0StexEcha][j0StexEcha] = (Double_t)0;
      }
    }
  }

  //................. calculation
  //........................... computation of half of the matrix, diagonal included

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, i0StexEcha) > 0) ||
        (fFlagSubDet == "EB")) {
      for (Int_t j0StexEcha = 0; j0StexEcha <= i0StexEcha; j0StexEcha++) {
        if ((fFlagSubDet == "EE" && fEcalNumbering->StexEchaForCons(fFileHeader->fStex, j0StexEcha) > 0) ||
            (fFlagSubDet == "EB")) {
          if (fT2d_hf_cov[i0StexEcha][i0StexEcha] > 0 && fT2d_hf_cov[j0StexEcha][j0StexEcha] > 0) {
            fT2d_hf_cor[i0StexEcha][j0StexEcha] =
                fT2d_hf_cov[i0StexEcha][j0StexEcha] / ((Double_t)sqrt(fT2d_hf_cov[i0StexEcha][i0StexEcha]) *
                                                       (Double_t)sqrt(fT2d_hf_cov[j0StexEcha][j0StexEcha]));
          } else {
            fT2d_hf_cor[i0StexEcha][j0StexEcha] = (Double_t)0.;
          }

          fT2d_hf_cor[j0StexEcha][i0StexEcha] = fT2d_hf_cor[i0StexEcha][j0StexEcha];
        }
      }
    }
    if (i0StexEcha % 100 == 0) {
      std::cout << i0StexEcha << "[HFN Cor], ";
    }
  }
  std::cout << std::endl;

  fTagHfCor[0] = 1;
  fFileHeader->fHfCorCalc++;
}
//------- (end of HighFrequencyCorrelationsBetweenChannels) ----------

//=================================================================================
//
//          L O W  &  H I G H    F R E Q U E N C Y    C O R R E L A T I O N S
//
//       B E T W E E N   T O W E R S  ( E B )  O R   S C s   ( E E )
//
//=================================================================================
//-----------------------------------------------------------------------------
//      Calculation of the mean Low Frequency Correlations
//      between channels for each Stin
//-----------------------------------------------------------------------------
void TEcnaRun::LowFrequencyMeanCorrelationsBetweenTowers() { LowFrequencyMeanCorrelationsBetweenStins(); }
void TEcnaRun::LowFrequencyMeanCorrelationsBetweenSCs() { LowFrequencyMeanCorrelationsBetweenStins(); }

void TEcnaRun::LowFrequencyMeanCorrelationsBetweenStins() {
  //Calculation of the mean Low Frequency Correlations
  //between channels for each Stin

  //... preliminary calculation of the Low Frequency Cor(c,c) if not done yet
  //    Only one tag (dim=1) to set to 0 (no write in the result ROOT file)
  if (fTagLfCor[0] != 1) {
    LowFrequencyCorrelationsBetweenChannels();
    fTagLfCor[0] = 0;
  }

  //..... mean fT2d_lfcc_mostins for each pair (Stin_X,Stin_Y)
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::LowFrequencyMeanCorrelationsBetweenStins()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the mean, for each " << fStinName.Data() << ", of the" << std::endl
              << "           Low Frequency Correlations between channels." << std::endl;
  }

  //................. allocation fT2d_lfcc_mostins + init to zero (mandatory)
  if (fT2d_lfcc_mostins == nullptr) {
    const Int_t n_Stin = fEcal->MaxStinEcnaInStex();
    fT2d_lfcc_mostins = new Double_t*[n_Stin];
    fCnew++;
    fT2d1_lfcc_mostins = new Double_t[n_Stin * n_Stin];
    fCnew++;
    for (Int_t i0StexStinEcna = 0; i0StexStinEcna < n_Stin; i0StexStinEcna++) {
      fT2d_lfcc_mostins[i0StexStinEcna] = &fT2d1_lfcc_mostins[0] + i0StexStinEcna * n_Stin;
    }
  }

  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      if (fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna] != (Double_t)0) {
        fMiscDiag[31]++;
        fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna] = (Double_t)0;
      }
    }
  }

  //..... Calculation of the mean LF Cor(c,c) for each pair (Stin_X,Stin_Y)
  //
  //           ! => Warning: this matrix is NOT symmetric => take N*N elements
  //                Only (Stin,Stin) matrix is symmetric.
  //                (StinEcha,StinEcha) matrix inside a (Stin,Stin) element is NOT symmetric
  //                (except for the (Stin,Stin) DIAGONAL elements)
  //      Then:
  //            1D array half_LFccMos[N*N] to put the (channel,channel) correlations

  Int_t ndim = (Int_t)(fEcal->MaxCrysInStin() * fEcal->MaxCrysInStin());

  TVectorD half_LFccMos(ndim);
  for (Int_t i = 0; i < ndim; i++) {
    half_LFccMos(i) = (Double_t)0.;
  }

  //..................... Calculation
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::LowFrequencyMeanCorrelationsBetweenStins()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for each " << fStinName.Data() << ", of the mean " << std::endl
              << "           Low Frequency cor(c,c)." << std::endl;
  }

  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      //................... .Copy the Mean Correlations(c,c') in 1D vector half_LFccMos()
      Int_t i_count = 0;
      for (Int_t i0StinCrys = 0; i0StinCrys < fEcal->MaxCrysInStin(); i0StinCrys++) {
        Int_t i0StexEcha = i0StexStinEcna * fEcal->MaxCrysInStin() + i0StinCrys;
        for (Int_t j0StinCrys = 0; j0StinCrys < fEcal->MaxCrysInStin(); j0StinCrys++) {
          Int_t j0StexEcha = j0StexStinEcna * fEcal->MaxCrysInStin() + j0StinCrys;
          if ((i0StexEcha >= 0 && i0StexEcha < fEcal->MaxCrysEcnaInStex()) &&
              (j0StexEcha >= 0 && j0StexEcha < fEcal->MaxCrysEcnaInStex())) {
            half_LFccMos(i_count) = fT2d_lf_cor[i0StexEcha][j0StexEcha];
            i_count++;
          } else {
            std::cout << "!TEcnaRun::LowFrequencyMeanCorrelationsBetweenStins()> Channel number out of range."
                      << "i0StexEcha = " << i0StexEcha << ", j0StexEcha = " << j0StexEcha << fTTBELL << std::endl;
          }
        }
      }
      //...... Calculation of the mean absolute values of the LF mean Correlations(c,c')
      fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna] = (Double_t)0;
      for (Int_t i_rcor = 0; i_rcor < ndim; i_rcor++) {
        fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna] += fabs(half_LFccMos(i_rcor));
      }
      fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna] /= (Double_t)ndim;
    }
    if (i0StexStinEcna % 10 == 0) {
      std::cout << i0StexStinEcna << "[LFN MCtt], ";
    }
  }
  std::cout << std::endl;

  fTagLFccMoStins[0] = 1;
  fFileHeader->fLFccMoStinsCalc++;
}  // ------- end of LowFrequencyMeanCorrelationsBetweenStins() -------

//-----------------------------------------------------------------------------
//      Calculation of the mean High Frequency Correlations
//      between channels for each Stin
//-----------------------------------------------------------------------------
void TEcnaRun::HighFrequencyMeanCorrelationsBetweenTowers() { HighFrequencyMeanCorrelationsBetweenStins(); }
void TEcnaRun::HighFrequencyMeanCorrelationsBetweenSCs() { HighFrequencyMeanCorrelationsBetweenStins(); }

void TEcnaRun::HighFrequencyMeanCorrelationsBetweenStins() {
  //Calculation of the mean High Frequency Correlations
  //between channels for each Stin

  //... preliminary calculation of the High Frequency Cor(c,c) if not done yet
  //    Only one tag (dim=1) to set to 0 (no write in the result ROOT file)
  if (fTagHfCor[0] != 1) {
    HighFrequencyCorrelationsBetweenChannels();
    fTagHfCor[0] = 0;
  }

  //..... mean fT2d_hfcc_mostins for each pair (Stin_X,Stin_Y)
  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::HighFrequencyMeanCorrelationsBetweenStins()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation of the mean, for each " << fFlagSubDet.Data() << ", of the" << std::endl
              << "           High Frequency Correlations between channels." << std::endl;
  }

  //................. allocation fT2d_hfcc_mostins + init to zero (mandatory)
  if (fT2d_hfcc_mostins == nullptr) {
    const Int_t n_Stin = fEcal->MaxStinEcnaInStex();
    fT2d_hfcc_mostins = new Double_t*[n_Stin];
    fCnew++;
    fT2d1_hfcc_mostins = new Double_t[n_Stin * n_Stin];
    fCnew++;
    for (Int_t i0StexStinEcna = 0; i0StexStinEcna < n_Stin; i0StexStinEcna++) {
      fT2d_hfcc_mostins[i0StexStinEcna] = &fT2d1_hfcc_mostins[0] + i0StexStinEcna * n_Stin;
    }
  }

  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      if (fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna] != (Double_t)0) {
        fMiscDiag[32]++;
        fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna] = (Double_t)0;
      }
    }
  }

  //..... Calculation of the mean HF Cor(c,c) for each pair (Stin_X,Stin_Y)
  //
  //           ! => Warning: this matrix is NOT symmetric => take N*N elements
  //                Only (Stin,Stin) matrix is symmetric.
  //                (StinEcha,StinEcha) matrix inside a (Stin,Stin) element is NOT symmetric
  //                (except for the (Stin,Stin) DIAGONAL elements)
  //      Then:
  //            1D array half_LFccMos[N*N] to put the (channel,channel) correlations

  Int_t ndim = (Int_t)(fEcal->MaxCrysInStin() * fEcal->MaxCrysInStin());

  TVectorD half_HFccMos(ndim);
  for (Int_t i = 0; i < ndim; i++) {
    half_HFccMos(i) = (Double_t)0.;
  }

  if (fFlagPrint != fCodePrintNoComment) {
    std::cout << "*TEcnaRun::HighFrequencyMeanCorrelationsBetweenStins()" << std::endl;
  }
  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "           Calculation, for each " << fFlagSubDet.Data() << ", of the mean " << std::endl
              << "           High Frequency cor(c,c)." << std::endl;
  }

  //..................... Calculation
  for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      //.................... Copy the relevant Mean Correlations(c,c') in 1D vector half_HFccMos()
      Int_t i_count = 0;
      for (Int_t i0StinCrys = 0; i0StinCrys < fEcal->MaxCrysInStin(); i0StinCrys++) {
        Int_t i0StexEcha = i0StexStinEcna * fEcal->MaxCrysInStin() + i0StinCrys;
        for (Int_t j0StinCrys = 0; j0StinCrys < fEcal->MaxCrysInStin(); j0StinCrys++) {
          Int_t j0StexEcha = j0StexStinEcna * fEcal->MaxCrysInStin() + j0StinCrys;
          if ((i0StexEcha >= 0 && i0StexEcha < fEcal->MaxCrysEcnaInStex()) &&
              (j0StexEcha >= 0 && j0StexEcha < fEcal->MaxCrysEcnaInStex())) {
            half_HFccMos(i_count) = fT2d_hf_cor[i0StexEcha][j0StexEcha];
            i_count++;
          } else {
            std::cout << "!TEcnaRun::HighFrequencyMeanCorrelationsBetweenStins()> Channel number out of range."
                      << "i0StexEcha = " << i0StexEcha << ", j0StexEcha = " << j0StexEcha << fTTBELL << std::endl;
          }
        }
      }
      //..... Calculation of the mean absolute values of the HF mean Correlations(c,c')
      fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna] = (Double_t)0;
      for (Int_t i_rcor = 0; i_rcor < ndim; i_rcor++) {
        fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna] += fabs(half_HFccMos(i_rcor));
      }
      fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna] /= (Double_t)ndim;
    }
    if (i0StexStinEcna % 10 == 0) {
      std::cout << i0StexStinEcna << "[HFN MCtt], ";
    }
  }
  std::cout << std::endl;

  fTagHFccMoStins[0] = 1;
  fFileHeader->fHFccMoStinsCalc++;
}  // ------- end of HighFrequencyMeanCorrelationsBetweenStins() -------

//=========================================================================
//
//                  W R I T I N G     M E T H O D S
//
//=========================================================================

//=========================================================================
//
//         W R I T I N G   M E T H O D S :    R O O T    F I L E S
//
//=========================================================================
//-------------------------------------------------------------
//
//                      OpenRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRun::OpenRootFile(const Text_t* name, const TString& status) {
  //Open the Root file

  Bool_t ok_open = kFALSE;

  TString s_name;
  s_name = fCnaParPaths->ResultsRootFilePath();
  s_name.Append('/');
  s_name.Append(name);

  //gCnaRootFile = new TEcnaRootFile(fObjectManager, s_name.Data(), status);     fCnew++;

  Long_t iCnaRootFile = fObjectManager->GetPointerValue("TEcnaRootFile");
  if (iCnaRootFile == 0) {
    gCnaRootFile = new TEcnaRootFile(fObjectManager, s_name.Data(), status); /* Anew("gCnaRootFile");*/
  } else {
    gCnaRootFile = (TEcnaRootFile*)iCnaRootFile;
    gCnaRootFile->ReStart(s_name.Data(), status);
  }

  if (gCnaRootFile->fRootFileStatus == "RECREATE") {
    ok_open = gCnaRootFile->OpenW();
  }
  if (gCnaRootFile->fRootFileStatus == "READ") {
    ok_open = gCnaRootFile->OpenR();
  }

  if (!ok_open)  // unable to open file
  {
    std::cout << "TEcnaRun::OpenRootFile> Cannot open file " << s_name.Data() << std::endl;
  } else {
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::OpenRootFile> Open ROOT file OK for file " << s_name.Data() << std::endl;
    }
    fOpenRootFile = kTRUE;
  }
  return ok_open;
}
//-------------------------------------------------------------
//
//                      CloseRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRun::CloseRootFile(const Text_t* name) {
  //Close the Root file

  TString s_name;
  s_name = fCnaParPaths->ResultsRootFilePath();
  s_name.Append('/');
  s_name.Append(name);

  Bool_t ok_close = kFALSE;

  if (fOpenRootFile == kTRUE) {
    gCnaRootFile->CloseFile();

    if (fFlagPrint != fCodePrintAllComments) {
      std::cout << "*TEcnaRun::CloseRootFile> ROOT file " << s_name.Data() << " closed." << std::endl;
    }

    //     delete gCnaRootFile;     gCnaRootFile = 0;          fCdelete++;

    fOpenRootFile = kFALSE;
    ok_close = kTRUE;
  } else {
    std::cout << "*TEcnaRun::CloseRootFile(...)> No close since no file is open." << fTTBELL << std::endl;
  }
  return ok_close;
}
//-------------------------------------------------------------
//
//   WriteRootFile without arguments.
//   Call WriteRootFile WITH argument (file name)
//   after an automatic generation of the file name.
//
//   Codification for the file name:
//            see comment at the beginning of this file
//
//-------------------------------------------------------------

//=================================================================================
//
//         WriteRootFile()   ====>  D O N ' T    S U P P R E S S  ! ! !
//                                  Called by the analyzer in package: "Modules"
//
//=================================================================================
Bool_t TEcnaRun::WriteRootFile() {
  //Write the Root file.
  //File name automatically generated by fCnaWrite->fMakeResultsFileName()
  //previously called in GetReadyToCompute().

  Bool_t ok_write = kFALSE;

  //============================= check number of found events
  Int_t nCountEvts = 0;

  for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
    for (Int_t i0Sample = 0; i0Sample < fFileHeader->fNbOfSamples; i0Sample++) {
      nCountEvts += fT2d_NbOfEvts[i0StexEcha][i0Sample];
    }
  }

  if (nCountEvts <= 0) {
    //============== no write if no event found
    std::cout << "!TEcnaRun::WriteRootFile()> No event found for file " << fCnaWrite->GetRootFileNameShort().Data()
              << ". File will not be written." << std::endl;
    ok_write = kTRUE;
  } else {
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile()> Results are going to be written in the ROOT file: " << std::endl
                << "                           " << fCnaWrite->GetRootFileName().Data() << std::endl;
    }

    const Text_t* FileShortName = (const Text_t*)fCnaWrite->GetRootFileNameShort().Data();
    ok_write = WriteRootFile(FileShortName, fFileHeader->fNbOfSamples);

    if (ok_write == kTRUE) {
      if (fFlagPrint != fCodePrintNoComment) {
        std::cout << "*TEcnaRun::WriteRootFile()> Writing OK for file " << fCnaWrite->GetRootFileName().Data()
                  << std::endl;
      }
    } else {
      std::cout << "!TEcnaRun::WriteRootFile()> Writing FAILLED for file " << fCnaWrite->GetRootFileName().Data()
                << fTTBELL << std::endl;
    }
  }
  return ok_write;
}  // end of WriteRootFile()

//--------------------------------------------------------------------
//
//               WriteNewRootFile with argument
//    Called by TEcnaGui for results file of the Calculations method
//    analysis type and nb of samples changed, other arguments kept
//
//--------------------------------------------------------------------
Bool_t TEcnaRun::WriteNewRootFile(const TString& TypAna) {
  //Write a new Root file. File name automatically generated by fCnaWrite->fMakeResultsFileName()
  //called here.

  Bool_t ok_write = kFALSE;

  fCnaWrite->RegisterFileParameters(TypAna.Data(),
                                    fNbSampForCalc,
                                    fFileHeader->fRunNumber,
                                    fFileHeader->fFirstReqEvtNumber,
                                    fFileHeader->fLastReqEvtNumber,
                                    fFileHeader->fReqNbOfEvts,
                                    fFileHeader->fStex,
                                    fFileHeader->fStartDate,
                                    fFileHeader->fStopDate,
                                    fFileHeader->fStartTime,
                                    fFileHeader->fStopTime);

  fCnaWrite->fMakeResultsFileName();  // set fRootFileName, fRootFileNameShort
  fNewRootFileName = fCnaWrite->GetRootFileName();
  fNewRootFileNameShort = fCnaWrite->GetRootFileNameShort();

  const Text_t* FileShortName = (const Text_t*)fNewRootFileNameShort.Data();

  if (fFlagPrint == fCodePrintAllComments) {
    std::cout << "*TEcnaRun::WriteNewRootFile()> Results are going to be written in the ROOT file: " << std::endl
              << "                              " << fNewRootFileNameShort.Data() << std::endl;
  }

  ok_write = WriteRootFile(FileShortName, fNbSampForCalc);

  return ok_write;
}

//-------------------------------------------------------------------------
//
//    Get the new ROOT file name (long and short)
//   (called by TEcnaGui in Calculations method)
//
//-------------------------------------------------------------------------
const TString& TEcnaRun::GetNewRootFileName() const { return fNewRootFileName; }
const TString& TEcnaRun::GetNewRootFileNameShort() const { return fNewRootFileNameShort; }

//--------------------------------------------------------------------
//
//               WriteRootFile with argument
//
//--------------------------------------------------------------------
Bool_t TEcnaRun::WriteRootFile(const Text_t* name, Int_t& argNbSampWrite) {
  //Write the Root file

  const Text_t* file_name = name;

  Bool_t ok_write = kFALSE;

  if (fOpenRootFile) {
    std::cout << "!TEcnaRun::WriteRootFile(...) *** ERROR ***> Writing on file already open." << fTTBELL << std::endl;
  } else {
    // List of the different element types and associated parameters as ordered in the ROOT file (smaller -> larger)
    //                                                                 ==========
    //
    //         WARNING  *** HERE SIZES ARE THESE FOR THE BARREL (1700 Xtals) and for 10 samples ***
    //
    //   Nb of   Type of element            Type      Type                                    Size    Comment
    // elements                             Number    Name
    //
    //        1  fMatHis(1,StexStin)         ( 0)  cTypNumbers             1*(   1,  68) =         68

    //        1  fMatHis(1,StexStin)         (12)  cTypAvPed               1*(   1,  68) =         68
    //        1  fMatHis(1,StexStin)         ( 3)  cTypAvTno               1*(   1,  68) =         68
    //        1  fMatHis(1,StexStin)         ( 4)  cTypAvLfn               1*(   1,  68) =         68
    //        1  fMatHis(1,StexStin)         ( 5)  cTypAvHfn               1*(   1,  68) =         68
    //        1  fMatHis(1,StexStin)         (13)  cTypAvMeanCorss         1*(   1,  68) =         68
    //        1  fMatHis(1,StexStin)         (14)  cTypAvSigCorss          1*(   1,  68) =         68

    //        1  fMatHis(1,StexEcha)         (16)  cTypPed                 1*(   1,1700) =      1 700
    //        1  fMatHis(1,StexEcha)         (17)  cTypTno                 1*(   1,1700) =      1 700
    //        1  fMatHis(1,StexEcha)         (10)  cTypMeanCorss           1*(   1,1700) =      1 700
    //        1  fMatHis(1,StexEcha)         (18)  cTypLfn                 1*(   1,1700) =      1 700
    //        1  fMatHis(1,StexEcha)         (19)  cTypHfn                 1*(   1,1700) =      1 700
    //        1  fMatHis(1,StexEcha)         (11)  cTypSigCorss            1*(   1,1700) =      1 700

    //        1  fMatMat(Stin,Stin)          (23)  cTypLFccMoStins         1*(  68,  68) =      4 624
    //        1  fMatMat(Stin,Stin)          (24)  cTypHFccMoStins         1*(  68,  68) =      4 624

    //        1  fMatHis(StexEcha, sample)   (15)  cTypNbOfEvts            1*(1700,  10) =     17 000
    //        1  fMatHis(StexEcha, sample)   ( 1)  cTypMSp                 1*(1700,  10) =     17 000
    //        1  fMatHis(StexEcha, sample)   ( 2)  cTypSSp                 1*(1700,  10) =     17 000

    //   StexEcha  fMatMat(sample, sample)   ( 8)  cTypCovCss           1700*(  10,  10) =    170 000
    //   StexEcha  fMatMat(sample, sample    ( 9)  cTypCorCss           1700*(  10,  10) =    170 000

    //   StexEcha  fMatHis(sample, bin_evt)  (20)  cTypAdcEvt,          1700*(  10, 150) =  2 550 000

    //        1  fMatMat(StexEcha, StexEcha) (21)  cTypLfCov               1*(1700,1700) =  2 890 000
    //        1  fMatMat(StexEcha, StexEcha) (22)  cTypLfCor               1*(1700,1700) =  2 890 000

    //        1  fMatMat(StexEcha, StexEcha) ( 6)  cTypHfCov               1*(1700,1700) =  2 890 000 // (06/05/08)
    //        1  fMatMat(StexEcha, StexEcha) ( 7)  cTypHfCor               1*(1700,1700) =  2 890 000 // (06/05/08)

    //......................................................................................................

    OpenRootFile(file_name, "RECREATE");

    TString typ_name = "?";
    Int_t v_nb_times = 0;
    Int_t v_dim_one = 0;
    Int_t v_dim_two = 0;
    Int_t v_size = 0;
    Int_t v_tot = 0;
    Int_t v_tot_writ = 0;

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //
    //  ===> no general method and no translation to TEcnaWrite
    //       because the fT1d.. and fT2d... arrays
    //       are attributes of TEcnaRun (calls to the "TRootXXXX" methods)
    //
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    //-------------------------- Stin numbers
    //       1   fMatHis(1,Stin)           ( 0)  cTypNumbers        1*(   1,  68) =         68

    Int_t MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "StinNumbers";
    v_nb_times = fFileHeader->fStinNumbersCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagStinNumbers[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypNumbers;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootStinNumbers();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Average Pedestals (1 value per Stin)
    //       1   fMatHis(1, StexStin)   (12)  cTypAvPed      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvPed";
    v_nb_times = fFileHeader->fAvPedCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvPed[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvPed;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvPed();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Average Total noise
    // StexEcha   fMatHis(1, StexStin)     ( 3)  cTypAvTno      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvTno";
    v_nb_times = fFileHeader->fAvTnoCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvTno[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvTno;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvTno();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Average Low frequency noise
    //       1   fMatHis(1, StexStin)   ( 4)  cTypAvLfn      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvLfn";
    v_nb_times = fFileHeader->fAvLfnCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvLfn[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvLfn;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvLfn();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Average High frequency noise
    //       1   fMatHis(1, StexStin)   ( 5)  cTypAvHfn      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvHfn";
    v_nb_times = fFileHeader->fAvHfnCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvHfn[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvHfn;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvHfn();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Average mean cor(s,s)
    //       1   fMatHis(1, StexStin)   (13)  cTypAvMeanCorss      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvMeanCorss";
    v_nb_times = fFileHeader->fAvMeanCorssCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvMeanCorss[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvMeanCorss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvEvCorss();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //--------------------------  Average sigma of cor(s,s)
    //       1   fMatHis(1, StexStin)    (14)  cTypAvSigCorss      1*(1,  68) =     68

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AvSigCorss";
    v_nb_times = fFileHeader->fAvSigCorssCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagAvSigCorss[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAvSigCorss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAvSigCorss();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Expectation values of the expectation values of the samples (pedestals)
    //       1   fMatHis(1,StexEcha)         (16)  cTypPed                1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "Ped";
    v_nb_times = fFileHeader->fPedCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagPed[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypPed;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootPed();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Expectation values of the sigmas the samples
    //       1   fMatHis(1,StexEcha)         (17)  cTypTno               1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "Tno";
    v_nb_times = fFileHeader->fTnoCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagTno[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypTno;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootTno();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Expectation values of the correlations between the samples
    //       1   fMatHis(1,StexEcha)         (10)  cTypMeanCorss            1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "MeanCorss";
    v_nb_times = fFileHeader->fMeanCorssCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagMeanCorss[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypMeanCorss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootMeanCorss();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Sigmas of the expectation values of the samples
    //       1   fMatHis(1,StexEcha)         (18)  cTypLfn               1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "Lfn";
    v_nb_times = fFileHeader->fLfnCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagLfn[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypLfn;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootLfn();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Sigmas of the sigmas of the samples
    //       1   fMatHis(1,StexEcha)         (19)  cTypHfn              1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "Hfn";
    v_nb_times = fFileHeader->fHfnCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagHfn[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypHfn;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootHfn();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Sigmas of the correlations between the samples
    //       1   fMatHis(1,StexEcha)         (11)  cTypSigCorss           1*(   1,1700) =      1 700

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "SigCorss";
    v_nb_times = fFileHeader->fSigCorssCalc;
    v_dim_one = 1;
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagSigCorss[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSigCorss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootSigCorss();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //----- Mean Covariances between StexEchas (averaged over samples) for all (Stin_X,Stin_Y)
    //       1   fMatMat(Stin,Stin)       (23)  cTypLFccMoStins         1*(  68,  68) =      4 624

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "LFccMoStins";
    v_nb_times = fFileHeader->fLFccMoStinsCalc;
    v_dim_one = fEcal->MaxStinEcnaInStex();
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagLFccMoStins[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypLFccMoStins;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootLFccMoStins();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //----- Mean Correlations between StexEchas (averaged over samples) for all (Stin_X,Stin_Y)
    //       1   fMatMat(Stin,Stin)       (24)  cTypHFccMoStins         1*(  68,  68) =      4 624

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "HFccMoStins";
    v_nb_times = fFileHeader->fHFccMoStinsCalc;
    v_dim_one = fEcal->MaxStinEcnaInStex();
    v_dim_two = fEcal->MaxStinEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagHFccMoStins[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypHFccMoStins;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootHFccMoStins();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Numbers of found events (NbOfEvts)
    //       1   fMatHis(StexEcha, sample)   (15)  cTypNbOfEvts       1*(1700,  10) =     17 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "NbOfEvts";
    v_nb_times = fFileHeader->fNbOfEvtsCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = argNbSampWrite;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagNbOfEvts[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypNbOfEvts;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootNbOfEvts(argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Expectation values of the samples
    //       1   fMatHis(StexEcha, sample)   ( 1)  cTypMSp                  1*(1700,  10) =     17 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "MSp";
    v_nb_times = fFileHeader->fMSpCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = argNbSampWrite;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagMSp[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypMSp;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootMSp(argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Sigmas of the samples
    //       1   fMatHis(StexEcha, sample)   ( 2)  cTypSSp                 1*(1700,  10) =     17 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "SSp";
    v_nb_times = fFileHeader->fSSpCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = argNbSampWrite;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagSSp[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSSp;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootSSp(argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Covariances between samples

    // StexEcha   fMatMat(sample,  sample)   ( 8)  cTypCovCss           1700*(  10,  10) =    170 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "CovCss";
    v_nb_times = fFileHeader->fCovCssCalc;
    v_dim_one = argNbSampWrite;
    v_dim_two = argNbSampWrite;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i0StexEcha = 0; i0StexEcha < v_nb_times; i0StexEcha++) {
      if (fTagCovCss[i0StexEcha] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCovCss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i0StexEcha;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootCovCss(i0StexEcha, argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i0StexEcha == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Correlations between samples
    // StexEcha   fMatMat(sample,  sample)   ( 9)  cTypCorCss           1700*(  10,  10) =    170 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "CorCss";
    v_nb_times = fFileHeader->fCorCssCalc;
    v_dim_one = argNbSampWrite;
    v_dim_two = argNbSampWrite;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i0StexEcha = 0; i0StexEcha < v_nb_times; i0StexEcha++) {
      if (fTagCorCss[i0StexEcha] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCorCss;
        gCnaRootFile->fCnaIndivResult->fIthElement = i0StexEcha;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootCorCss(i0StexEcha, argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i0StexEcha == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Samples as a function of event = events distributions
    // StexEcha   fMatHis(sample,  bins)     (20)  cTypAdcEvt,        1700*(  10, 150) =  2 550 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "AdcEvt";
    v_nb_times = fFileHeader->fAdcEvtCalc;
    v_dim_one = argNbSampWrite;
    v_dim_two = fFileHeader->fReqNbOfEvts;
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i0StexEcha = 0; i0StexEcha < v_nb_times; i0StexEcha++) {
      if (fTagAdcEvt[i0StexEcha] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypAdcEvt;
        gCnaRootFile->fCnaIndivResult->fIthElement = i0StexEcha;
        gCnaRootFile->fCnaIndivResult->SetSizeHis(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1, 1);
        TRootAdcEvt(i0StexEcha, argNbSampWrite);
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i0StexEcha == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Low Frequency Covariances between StexEchas
    //  sample   fMatMat(StexEcha, StexEcha)  (21)  cTypLfCov           1*(1700,1700) =  2 890 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "LfCov";
    v_nb_times = fFileHeader->fLfCovCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {  //=================================== Record type EB
      if (fTagLfCov[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypLfCov;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootLfCov();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- Low Frequency Correlations between StexEchas
    //  sample   fMatMat(StexEcha, StexEcha)  (22)  cTypLfCor           1*(1700,1700) =  2 890 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "LfCor";
    v_nb_times = fFileHeader->fLfCorCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagLfCor[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypLfCor;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootLfCor();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- High Frequency Covariances between StexEchas
    //  sample   fMatMat(StexEcha, StexEcha)  (6)  cTypHfCov           1*(1700,1700) =  2 890 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "HfCov";
    v_nb_times = fFileHeader->fHfCovCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagHfCov[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypHfCov;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootHfCov();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //-------------------------- High Frequency Correlations between StexEchas
    //  sample   fMatMat(StexEcha, StexEcha)  (7)  cTypHfCor           1*(1700,1700) =  2 890 000

    MaxCar = fgMaxCar;
    typ_name.Resize(MaxCar);
    typ_name = "HfCor";
    v_nb_times = fFileHeader->fHfCorCalc;
    v_dim_one = fEcal->MaxCrysEcnaInStex();
    v_dim_two = fEcal->MaxCrysEcnaInStex();
    v_size = v_nb_times * v_dim_one * v_dim_two;
    v_tot += v_size;

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(18) << typ_name << ": " << std::setw(4) << v_nb_times
                << " * (" << std::setw(4) << v_dim_one << "," << std::setw(4) << v_dim_two << ") = " << std::setw(9)
                << v_size;
    }

    for (Int_t i = 0; i < v_nb_times; i++) {
      if (fTagHfCor[0] == 1) {
        gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypHfCor;
        gCnaRootFile->fCnaIndivResult->fIthElement = i;
        gCnaRootFile->fCnaIndivResult->SetSizeMat(v_dim_one, v_dim_two);
        gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1, 1);
        TRootHfCor();
        gCnaRootFile->fCnaResultsTree->Fill();
        if (i == 0 && fFlagPrint == fCodePrintAllComments) {
          std::cout << " => WRITTEN ON FILE ";
          v_tot_writ += v_size;
        }
      }
    }
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << std::endl;
    }

    //---------------------------------------------- WRITING
    //...................................... file
    gCnaRootFile->fRootFile->Write();
    //...................................... header
    fFileHeader->Write();

    //...................................... status message
    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> " << std::setw(20) << "TOTAL: " << std::setw(21)
                << "CALCULATED = " << std::setw(9) << v_tot << " => WRITTEN ON FILE = " << std::setw(9) << v_tot_writ
                << std::endl;
    }

    if (fFlagPrint == fCodePrintAllComments) {
      std::cout << "*TEcnaRun::WriteRootFile(...)> Write OK in file " << file_name << " in directory:" << std::endl
                << "                           " << fCnaParPaths->ResultsRootFilePath().Data() << std::endl;
    }

    ok_write = kTRUE;

    //...................................... close
    CloseRootFile(file_name);
  }
  return ok_write;
}  //-------------- End of WriteRootFile(...) -----------------------

//======================== "PREPA FILL" METHODS ===========================

//-------------------------------------------------------------------------
//
//  Prepa Fill Stin numbers as a function of the Stin index
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootStinNumbers() {
  if (fTagStinNumbers[0] == 1) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_StexStinFromIndex[j0StexStinEcna];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill last evt numbers for all the (StexEcha,sample)
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootNbOfEvts(const Int_t& argNbSampWrite) {
  if (fTagNbOfEvts[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
        gCnaRootFile->fCnaIndivResult->fMatHis(j0StexEcha, i0Sample) = fT2d_NbOfEvts[j0StexEcha][i0Sample];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill histogram of samples as a function of event
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAdcEvt(const Int_t& user_StexEcha, const Int_t& argNbSampWrite) {
  if (fTagAdcEvt[user_StexEcha] == 1) {
    for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
      //...................... all the bins set to zero
      for (Int_t j_bin = 0; j_bin < fFileHeader->fReqNbOfEvts; j_bin++) {
        gCnaRootFile->fCnaIndivResult->fMatHis(i0Sample, j_bin) = (Double_t)0.;
      }
      //...................... fill the non-zero bins
      for (Int_t j_bin = 0; j_bin < fFileHeader->fReqNbOfEvts; j_bin++) {
        gCnaRootFile->fCnaIndivResult->fMatHis(i0Sample, j_bin) = fT3d_AdcValues[user_StexEcha][i0Sample][j_bin];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the samples for all the StexEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootMSp(const Int_t& argNbSampWrite) {
  if (fTagMSp[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
        gCnaRootFile->fCnaIndivResult->fMatHis(j0StexEcha, i0Sample) = fT2d_ev[j0StexEcha][i0Sample];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the samples for all the StexEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootSSp(const Int_t& argNbSampWrite) {
  if (fTagSSp[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
        gCnaRootFile->fCnaIndivResult->fMatHis(j0StexEcha, i0Sample) = fT2d_sig[j0StexEcha][i0Sample];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill mean covariances between StexEchas, mean over samples
//  for all (Stin_X, Stin_Y)
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootLFccMoStins() {
  if (fTagLFccMoStins[0] == 1) {
    for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
      for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexStinEcna, j0StexStinEcna) =
            fT2d_lfcc_mostins[i0StexStinEcna][j0StexStinEcna];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill mean correlations between StexEchas, mean over samples
//  for all (Stin_X, Stin_Y)
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootHFccMoStins() {
  if (fTagHFccMoStins[0] == 1) {
    for (Int_t i0StexStinEcna = 0; i0StexStinEcna < fEcal->MaxStinEcnaInStex(); i0StexStinEcna++) {
      for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexStinEcna, j0StexStinEcna) =
            fT2d_hfcc_mostins[i0StexStinEcna][j0StexStinEcna];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions of the samples for all the StexEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvTno() {
  if (fTagAvTno[0] == 1) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_totn[j0StexStinEcna];
    }
  }
}
//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions xmin of the samples for all the StexEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvLfn() {
  if (fTagAvLfn[0] == 1) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_lofn[j0StexStinEcna];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions xmax of the samples for all the StexEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvHfn() {
  if (fTagAvHfn[0] == 1) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_hifn[j0StexStinEcna];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill Low Frequency covariances between StexEchas
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootLfCov() {
  if (fTagLfCov[0] == 1) {
    for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
      for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexEcha, j0StexEcha) = fT2d_lf_cov[i0StexEcha][j0StexEcha];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill Low Frequency correlations between StexEchas
//                         (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootLfCor() {
  if (fTagLfCor[0] == 1) {
    for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
      for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexEcha, j0StexEcha) = fT2d_lf_cor[i0StexEcha][j0StexEcha];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill High Frequency covariances between StexEchas
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootHfCov() {
  if (fTagHfCov[0] == 1) {
    for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
      for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexEcha, j0StexEcha) = fT2d_hf_cov[i0StexEcha][j0StexEcha];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill High Frequency correlations between StexEchas
//                         (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootHfCor() {
  if (fTagHfCor[0] == 1) {
    for (Int_t i0StexEcha = 0; i0StexEcha < fEcal->MaxCrysEcnaInStex(); i0StexEcha++) {
      for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0StexEcha, j0StexEcha) = fT2d_hf_cor[i0StexEcha][j0StexEcha];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill covariances between samples for a given StexEcha
//                      (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootCovCss(const Int_t& user_StexEcha, const Int_t& argNbSampWrite) {
  if (fTagCovCss[user_StexEcha] == 1) {
    for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample < argNbSampWrite; j0Sample++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0Sample, j0Sample) = fT3d_cov_ss[user_StexEcha][i0Sample][j0Sample];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill correlations between samples for a given StexEcha
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootCorCss(const Int_t& user_StexEcha, const Int_t& argNbSampWrite) {
  if (fTagCorCss[user_StexEcha] == 1) {
    for (Int_t i0Sample = 0; i0Sample < argNbSampWrite; i0Sample++) {
      for (Int_t j0Sample = 0; j0Sample < argNbSampWrite; j0Sample++) {
        gCnaRootFile->fCnaIndivResult->fMatMat(i0Sample, j0Sample) = fT3d_cor_ss[user_StexEcha][i0Sample][j0Sample];
      }
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the expectation values of the samples
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootPed() {
  if (fTagPed[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_ev_ev[j0StexEcha];
    }
  }
}
//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the sigmas of the samples
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootTno() {
  if (fTagTno[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_evsamp_of_sigevt[j0StexEcha];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the (sample,sample) correlations
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootMeanCorss() {
  if (fTagMeanCorss[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_ev_cor_ss[j0StexEcha];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the expectation values of the samples
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootLfn() {
  if (fTagLfn[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_sigevt_of_evsamp[j0StexEcha];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the expectation values of the sigmas
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootHfn() {
  if (fTagHfn[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_evevt_of_sigsamp[j0StexEcha];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the (sample,sample) correlations
//  for all the StexEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootSigCorss() {
  if (fTagSigCorss[0] == 1) {
    for (Int_t j0StexEcha = 0; j0StexEcha < fEcal->MaxCrysEcnaInStex(); j0StexEcha++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexEcha) = fT1d_sig_cor_ss[j0StexEcha];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill Average Pedestals
//  for all the StexStins
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvPed() {
  if (fTagAvPed[0] == 1) {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_mped[j0StexStinEcna];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill
//
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvEvCorss() {
  if (fTagAvMeanCorss[0] == 1)  // test 1st elt only since global calc
  {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_ev_corss[j0StexStinEcna];
    }
  }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill
//
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TEcnaRun::TRootAvSigCorss() {
  if (fTagAvSigCorss[0] == 1)  // test 1st elt only since global calc
  {
    for (Int_t j0StexStinEcna = 0; j0StexStinEcna < fEcal->MaxStinEcnaInStex(); j0StexStinEcna++) {
      gCnaRootFile->fCnaIndivResult->fMatHis(0, j0StexStinEcna) = fT1d_av_sig_corss[j0StexStinEcna];
    }
  }
}

//=========================================================================
//
//         METHODS TO SET FLAGS TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void TEcnaRun::PrintComments() {
  // Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  std::cout << "*TEcnaRun::PrintComments()> Warnings and some comments on init will be printed" << std::endl;
}

void TEcnaRun::PrintWarnings() {
  // Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  std::cout << "*TEcnaRun::PrintWarnings()> Warnings will be printed" << std::endl;
}

void TEcnaRun::PrintAllComments() {
  // Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  std::cout << "*TEcnaRun::PrintAllComments()> All the comments will be printed" << std::endl;
}

void TEcnaRun::PrintNoComment() {
  // Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}
//=========================== E N D ======================================
