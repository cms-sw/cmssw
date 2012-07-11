//----------Author's Name: B.Fabbro, FX Gentit DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 08/04/2010

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRead.h"

R__EXTERN TEcnaRootFile *gCnaRootFile;

ClassImp(TEcnaRead)
//___________________________________________________________________________
//
// TEcnaRead.
//==============> INTRODUCTION
//
//    This class allows the user to read the .root results files (containing
//   expectation values, variances, covariances, correlations and other
//   quantities of interest) previously computed by the class TEcnaRun.
//   (see documentation of this class)
//
//==============> PRELIMINARY REMARK
//
//    The user is not obliged to use directly this class. Another class
//   named TEcnaHistos can be used to make plots of the results computed
//   previously by means of the class TEcnaRun. The class TEcnaHistos
//   calls TEcnaRead and manage the reading of the .root result files.
//   (see the documentation of the class TEcnaHistos)
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//      Class TEcnaRead      ***   I N S T R U C T I O N S   F O R   U S E   ***
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//   //==============> TEcnaRead DECLARATION
//
//   // The declaration is done by calling the constructor without argument:
//
//       TEcnaRead* MyCnaRead = new TEcnaRead();
//   
//   //==============> PREPARATION METHOD GetReadyToReadRootFile(...)
//
//   // There is a preparation method named: GetReadyToReadRootFile(...);
//
//   //    GetReadyToReadRootFile(...) is used to read the quantities written
//   //    in the ROOT files in order to use these quantities for analysis.
//
//   //.......  Example of program using GetReadyToReadRootFile(...)
//
//   //  This example describes the reading of one result file. This file is in a
//   //  directory which name is given by the contents of a TString named PathForRootFile
//
//   //................ Set values for the arguments and call to the method
//
//      TString AnalysisName      = "AdcPed12"
//      Int_t   RunNumber         = 22770;
//      Int_t   FirstReqEvtNumber = 0;    
//      Int_t   LastReqEvtNumber  = 150; 
//      Int_t   NbOfEvts          = 150;
//      TString PathForRootFile   = "/afs/cern.ch/etc..." // .root result files directory
//
//      TEcnaRead*  MyCnaRead = new TEcnaRead();
//      MyCnaRead->GetReadyToReadRootFile(AnalysisName,      RunNumber,
//                                        FirstReqEvtNumber, LastReqEvtNumber, ReqNbOfEvts, Stex,
//                                        PathForRootFile);
//
//   //==============>  CALL TO THE METHOD: Bool_t LookAtRootFile() (MANDATORY)
//    
//    // This methods returns a boolean. It tests the existence
//    // of the ROOT file corresponding to the argument values given
//    // in the call to the method GetReadyToReadRootFile(...).
//    // It is recommended to test the return value of the method.
//
//    //....... Example of use:
//
//     if( MyCnaRead->LookAtRootFile() == kFALSE )
//        {
//          cout << "*** ERROR: ROOT file not found" << endl;
//        }
//      else
//        {
//         //........... The ROOT file exists and has been found
//         //
//         //---> CALLS TO THE METHODS WHICH RECOVER THE QUANTITIES. EXAMPLE:
//         //     (see the complete list of the methods hereafter)
//
//           Int_t   MaxSamples  = 10;
//           TMatrixD CorMat(MaxSamples,MaxSamples);
//           Int_t smStin = 59;
//           Int_t i0StinEcha =  4;
//           CorMat = MyCnaRead->ReadCorrelationsBetweenSamples(smStin,i0StinEcha);
//                        :
//            (Analysis of the correlations, etc...)
//                        :
//        }
//
//******************************************************************************
//
//                      *=======================*
//                      | DETAILLED DESCRIPTION |
//                      *=======================*
//

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//       Method GetReadyToReadRootFile(...) and associated methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//   TEcnaRead* MyCnaRead = new TEcnaRead();  // declaration of the object MyCnaRead
//
//   MyCnaRead->GetReadyToReadRootFile(AnalysisName, RunNumber,    NbOfSamples
//                                     FirstReqEvtNumber, LastReqEvtNumber,  ReqEvtNumber, Stexumber,
//                                     PathForRootFile);           
//      
//   Arguments:
//
//      TString  AnalysisName: code for the analysis. This code is
//                             necessary to distinguish between different
//                             analyses on the same events of a same run
//                             (example: pedestal run for the 3 gains:
//                                       AnalysisName = "Ped1" or "Ped6" or "Ped12")
//                             The string AnalysisName is automatically
//                             included in the name of the results files
//                             (see below: results files paragraph)
//
//      Int_t    NbOfSamples         number of samples (=10 maximum) 
//      Int_t    RunNumber:          run number
//      Int_t    FirstReqEvtNumber:  first requested event number (numbering starting from 1)
//      Int_t    LastReqEvtNumber:   last  requested event number
//      Int_t    StexNumber:         SM or Dee number (Stex = SM if EB, Dee if EE)
//      
//     
//      TString  PathForRootFile: Path of the directory containing the ROOT file.
//               The path must be complete: /afs/cern.ch/user/... etc...
//
//==============> METHODS TO RECOVER THE QUANTITIES FROM THE ROOT FILE
//
//                SM = SuperModule  (EB) equivalent to a Dee   (EE)
//                SC = SuperCrystal (EE) equivalent to a Tower (EB)
//
//                Stex = SM    in case of EB  ,  Dee in case of EE
//                Stin = Tower in case of EB  ,  SC  in case of EE
//
//  n1StexStin = Stin#    in Stex = Tower number in SM (RANGE = [1,68]) .OR.  SC number in Dee (RANGE = [1,149])
//  i0StexEcha = Channel# in Stex = Electronic channel number in SM    (RANGE = [0,1699]) .OR. in Dee (RANGE = [0,3724])
//  i0StinEcha = Channel# in Stin = Electronic channel number in tower (RANGE = [0,24])   .OR. in SC  (RANGE = [0,24])
//
//  MaxCrysInStin     = Maximum number of Xtals in a tower or a SC (25)
//  MaxCrysEcnaInStex = Maximum number of Xtals in SM (1700) or in the matrix including Dee (5000)
//  MaxStinEcnaInStex = Maximum number of towers in SM (68)  or in the matrix including Dee (200)
//  MaxSampADC        = Maximum number of samples (10)
//  NbOfSample        = Number of samples used to perform the calculations
//                      (example: for the 3 first samples, NbOfSample = 3)
//
//
//  TVectorD and TMatrixD sizes are indicated after the argument lists or at the beginning of method sub-lists
//
//
//  TMatrixD ReadNumberOfEventsForSamples
//  (const Int_t& n1StexStin, const Int_t& MaxCrysInStin, const Int_t& NbOfSamples); // TMatrixD(MaxCrysInStin,NbOfSamples)
//
//  TVectorD ReadSampleValues
//  (const Int_t& i0StexEcha, const Int_t& sample, const Int_t& NbOfReqEvts)         // TVectorD(Nb of requested evts)
// 
//  TVectorD ReadSampleMeans
//           (const Int_t& n1StexStin,  const Int_t& i0StinEcha, const Int_t& NbOfSamples); // TVectorD(NbOfSamples)
//  TVectorD ReadSampleSigmas
//           (const Int_t& n1StexStin, const Int_t& i0StinEcha, const Int_t& NbOfSamples);  // TVectorD(NbOfSamples)
//  TVectorD ReadSigmasOfSamples
//           (const Int_t& n1StexStin, const Int_t& i0StinEcha, const Int_t& NbOfSamples);  // TVectorD(NbOfSamples)
//
//  TMatrixD ReadCovariancesBetweenSamples
//               (const Int_t& n1StexStin, const Int_t& i0StinEcha); // TMatrixD(NbOfSamples,NbOfSamples)
//  TMatrixD ReadCorrelationsBetweenSamples
//               (const Int_t& n1StexStin, const Int_t& i0StinEcha); // TMatrixD(NbOfSamples,NbOfSamples)
//
// -----------------------------------------------------------  TMatrixD(MaxCrysInStin, MaxCrysInStin)
//  TMatrixD ReadLowFrequencyCovariancesBetweenChannels
//              (const Int_t& n1StexStin_X, const Int_t& n1StexStin_Y, const Int_t& MaxCrysInStin); 
// 
//  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels
//              (const Int_t& n1StexStin_X, const Int_t& n1StexStin_Y, const Int_t& MaxCrysInStin);
//
//  TMatrixD ReadHighFrequencyCovariancesBetweenChannels
//              (const Int_t& n1StexStin_X, const Int_t& n1StexStin_Y, const Int_t& MaxCrysInStin);
//  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels
//              (const Int_t& n1StexStin_X, const Int_t& n1StexStin_Y, const Int_t& MaxCrysInStin);
//
// ----------------- TMatrixD(MaxCrysEcnaInStex, MaxCrysEcnaInStex) (BIG!: 1700x1700 for EB and 5000x5000 for EE) 
//  TMatrixD ReadLowFrequencyCovariancesBetweenChannels(const Int_t& MaxCrysEcnaInStex);
//  TMatrixD ReadLowFrequencyCorrelationsBetweenChannels(const Int_t& MaxCrysEcnaInStex);
//
//  TMatrixD ReadHighFrequencyCovariancesBetweenChannels(const Int_t& MaxCrysEcnaInStex);
//  TMatrixD ReadHighFrequencyCorrelationsBetweenChannels(const Int_t& MaxCrysEcnaInStex);
//
// ---------------------------------------------------- TMatrixD(MaxStinEcnaInStex, MaxStinEcnaInStex)
//  TMatrixD ReadLowFrequencyMeanCorrelationsBetweenStins(const Int_t& MaxStinEcnaInStex);
//  TMatrixD ReadHighFrequencyMeanCorrelationsBetweenStins(const Int_t& MaxStinEcnaInStex);
//
// ---------------------------------------------------------------------------------------------------------------
//  TVectorD ReadPedestals(const Int_t& MaxCrysEcnaInStex);                         // TVectorD(MaxCrysEcnaInStex)
//  TVectorD ReadTotalNoise(const Int_t& MaxCrysEcnaInStex);                        // TVectorD(MaxCrysEcnaInStex)
//  TVectorD ReadMeanOfCorrelationsBetweenSamples(const Int_t& MaxCrysEcnaInStex);  // TVectorD(MaxCrysEcnaInStex)
//
//  TVectorD ReadLowFrequencyNoise(const Int_t& MaxCrysEcnaInStex);                 // TVectorD(MaxCrysEcnaInStex)
//  TVectorD ReadHighFrequencyNoise(const Int_t& MaxCrysEcnaInStex);                // TVectorD(MaxCrysEcnaInStex)
//  TVectorD ReadSigmaOfCorrelationsBetweenSamples(const Int_t& MaxCrysEcnaInStex); // TVectorD(MaxCrysEcnaInStex)
//
//----------------------------------------------------------------------------------------------------------------
//  TString GetStartDate()
//  TString GetStopDate()
//  TString GetRunType()
//  Int_t   GetFirstReqEvtNumber();
//  Int_t   GetReqNbOfEvts();
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//                         Print Methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//     Just after the declaration with the constructor without arguments,
//     you can set a "Print Flag" by means of the following "Print Methods":
//
//     TEcnaRead* MyCnaRead = new TEcnaRead(); // declaration of the object MyCnaRead
//
//    // Print Methods: 
//
//    MyCnaRead->PrintNoComment();  // Set flag to forbid printing of all the comments
//                                  // except ERRORS.
//
//    MyCnaRead->PrintWarnings();   // (DEFAULT)
//                                  // Set flag to authorize printing of some warnings.
//                                  // WARNING/INFO: information on something unusual
//                                  // in the data.
//                                  // WARNING/CORRECTION: something wrong (but not too serious)
//                                  // in the value of some argument.
//                                  // Automatically modified to a correct value.
//
//    MyCnaRead->PrintComments();    // Set flag to authorize printing of infos
//                                   //  and some comments concerning initialisations
//
//    MyCnaRead->PrintAllComments(); // Set flag to authorize printing of all the comments
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//
//------------------------------ TEcnaRead.cxx -----------------------------
//  
//   Creation (first version): 03 Dec 2002
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

TEcnaRead::TEcnaRead()
{
//Constructor without argument
  
 // cout << "[Info Management] CLASS: TEcnaRead.          CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaRead::TEcnaRead(const TString SubDet,
		     const TEcnaParPaths*  pCnaParPaths,
		     const TEcnaParCout*   pCnaParCout,
		     const TEcnaHeader*    pFileHeader,
		     const TEcnaNumbering* pEcalNumbering,
		     const TEcnaWrite*     pCnaWrite)
{
//Constructor with argument

 // cout << "[Info Management] CLASS: TEcnaRead.          CREATE OBJECT: this = " << this << endl;

  fCnaParPaths = 0;
  if( pCnaParPaths == 0 )
    {fCnaParPaths = new TEcnaParPaths(); /* Anew("fCnaParPaths");*/ }
  else
    {fCnaParPaths = (TEcnaParPaths*)pCnaParPaths;}

  fCnaParCout  = 0;
  if( pCnaParCout == 0 )
    {fCnaParCout = new TEcnaParCout();   /*Anew("fCnaParCout");*/ }
  else
    {fCnaParCout = (TEcnaParCout*)pCnaParCout;}

  fFileHeader  = 0; 
  if( pFileHeader == 0 )
    {
      const Text_t *h_name  = "CnaHeader";  //==> voir cette question avec FXG
      const Text_t *h_title = "CnaHeader";  //==> voir cette question avec FXG
      fFileHeader = new TEcnaHeader(h_name, h_title);  // Anew("fFileHeader");
    }
  else
    {fFileHeader = (TEcnaHeader*)pFileHeader;}

  Init();
  SetEcalSubDetector(SubDet.Data(), pEcalNumbering, pCnaWrite);
}

void TEcnaRead::Init()
{
//Initialisation concerning the ROOT file

  fCnew    = 0;
  fCdelete = 0;

  fTTBELL = '\007';

  fgMaxCar = (Int_t)512;

  fCodePrintNoComment   = fCnaParCout->GetCodePrint("NoComment");
  fCodePrintWarnings    = fCnaParCout->GetCodePrint("Warnings ");
  fCodePrintComments    = fCnaParCout->GetCodePrint("Comments");
  fCodePrintAllComments = fCnaParCout->GetCodePrint("AllComments");

  //.................................. Set flag print to "Warnings"   (Init)
  fFlagPrint = fCodePrintWarnings;

  fUserSamp      = -1;
  fUserChan      = -1;
 
  fSectChanSizeX = 0;
  fSectChanSizeY = 0;
  fSectSampSizeX = 0;
  fSectSampSizeY = 0;

  //.................................. Flags for Root File   (Init)
  fOpenRootFile = kFALSE;

  fReadyToReadRootFile = 0;
  fLookAtRootFile      = 0;

  fT1d_StexStinFromIndex = 0;

  //................................ tags Stin numbers
  fTagStinNumbers  = 0;
  fMemoStinNumbers = 0;

  //.......................... flag data exist
  fDataExist = kFALSE;

  //......................... transfert Sample ADC Values 3D array   (Init)
  fT3d_distribs  = 0;
  fT3d2_distribs = 0;
  fT3d1_distribs = 0;

  //................................. others
  Int_t MaxCar = fgMaxCar;

  MaxCar = fgMaxCar;
  fPathRoot.Resize(MaxCar);
  fPathRoot  = "fPathRoot> not defined";

}// end of Init()

//============================================================================================================
void  TEcnaRead::SetEcalSubDetector(const TString SubDet,
				    const TEcnaNumbering* pEcalNumbering,
				    const TEcnaWrite* pCnaWrite)
{
 // Set Subdetector (EB or EE)

  fEcal = 0; fEcal = new TEcnaParEcal(SubDet.Data());            // Anew("fEcal");
  Int_t MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = fEcal->GetEcalSubDetector();

  if( fFlagSubDet == "EB" ){fStexName = "SM";  fStinName = "tower";}
  if( fFlagSubDet == "EE" ){fStexName = "Dee"; fStinName = "SC";}

  fEcalNumbering = 0;
  if( pEcalNumbering == 0 )
    {fEcalNumbering = new TEcnaNumbering(fFlagSubDet.Data(), fEcal);   /*Anew("fEcalNumbering")*/ ;}
  else
    {fEcalNumbering = (TEcnaNumbering*)pEcalNumbering;}

  fCnaWrite = 0;
  if( pCnaWrite == 0 )
    {fCnaWrite =
       new TEcnaWrite(fFlagSubDet.Data(), fCnaParPaths, fCnaParCout, fEcal, fEcalNumbering); /*Anew("fCnaWrite")*/ ;}
  else
    {fCnaWrite = (TEcnaWrite*)pCnaWrite;}
}
//============================================================================================================
void TEcnaRead::Anew(const TString VarName)
{
  // allocation survey for new
  
  fCnew++;
  // cout << "TEcnaRead::Anew---> new " << setw(4) << fCnew << " --------------> " << setw(25)
  //      << VarName.Data() << " / object(this): " << this << endl;
}

void TEcnaRead::Adelete(const TString VarName)
{
  // allocation survey for delete
  
  fCdelete++;
  // cout << "TEcnaRead::Adelete> ========== delete" << setw(4) << fCdelete << " -> " << setw(25)
  //      << VarName.Data() << " / object(this): " << this << endl;
}

//=========================================== private copy ==========

void  TEcnaRead::fCopy(const TEcnaRead& rund)
{
//Private copy

  fFileHeader   = rund.fFileHeader;
  fOpenRootFile = rund.fOpenRootFile;

  fUserSamp     = rund.fUserSamp;
  fUserChan     = rund.fUserChan;

  fSectChanSizeX = rund.fSectChanSizeX;
  fSectChanSizeY = rund.fSectChanSizeY;
  fSectSampSizeX = rund.fSectSampSizeX;
  fSectSampSizeY = rund.fSectSampSizeY;

  //........................................ Codes 

  fCodePrintComments    = rund.fCodePrintComments;
  fCodePrintWarnings    = rund.fCodePrintWarnings;
  fCodePrintAllComments = rund.fCodePrintAllComments;
  fCodePrintNoComment   = rund.fCodePrintNoComment;

  //.................................................. Tags
  fTagStinNumbers  = rund.fTagStinNumbers;

  fFlagPrint = rund.fFlagPrint;
  fPathRoot  = rund.fPathRoot;

  fCnew    = rund.fCnew;
  fCdelete = rund.fCdelete;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    copy constructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead::TEcnaRead(const TEcnaRead& dcop)
{
  cout << "*TEcnaRead::TEcnaRead(const TEcnaRead& dcop)> "
       << " It is time to write a copy constructor" << endl
    // << " type an integer value and then RETURN to continue"
       << endl;
  
  // { Int_t cintoto;  cin >> cintoto; }
  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead& TEcnaRead::operator=(const TEcnaRead& dcop)
{
//Overloading of the operator=

  fCopy(dcop);
  return *this;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TEcnaRead::~TEcnaRead()
{
//Destructor
  
  // cout << "[Info Management] CLASS: TEcnaRead.          DESTROY OBJECT: this = " << this << endl;

  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments){
    cout << "*TEcnaRead::~TEcnaRead()> Entering destructor" << endl;}
  
  //if (fFileHeader    != 0){delete fFileHeader;    Adelete("fFileHeader");}
  //if (fEcal          != 0){delete fEcal;          Adelete("fEcal");}
  //if (fCnaParCout    != 0){delete fCnaParCout;    Adelete("fCnaParCout");}
  //if (fCnaParPaths   != 0){delete fCnaParPaths;   Adelete("fCnaParPaths");}
  //if (fCnaWrite      != 0){delete fCnaWrite;      Adelete("fCnaWrite");}
  //if (fEcalNumbering != 0){delete fEcalNumbering; Adelete("fEcalNumbering");}

  if (fT1d_StexStinFromIndex != 0){delete [] fT1d_StexStinFromIndex; Adelete("fT1d_StexStinFromIndex");}
  if (fTagStinNumbers        != 0){delete [] fTagStinNumbers;        Adelete("fTagStinNumbers");}

  if (fT3d_distribs  != 0){delete [] fT3d_distribs;  Adelete("fT3d_distribs");}
  if (fT3d2_distribs != 0){delete [] fT3d2_distribs; Adelete("fT3d2_distribs");}
  if (fT3d1_distribs != 0){delete [] fT3d1_distribs; Adelete("fT3d1_distribs");}

  if ( fCnew != fCdelete )
    {
      cout << "!TEcnaRead/destructor> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << fTTBELL << endl;
    }
  else
    {
      // cout << "*TEcnaRead/destructor> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
      //      << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  
  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments){
    cout << "*TEcnaRead::~TEcnaRead()> End of destructor " << endl;}
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                             M  E  T  H  O  D  S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//============================================================================
//
//                  GetReadyToReadRootFile(...)
//                  
//============================================================================
void TEcnaRead::GetReadyToReadRootFile(TString      typ_ana,    const Int_t& nb_of_samples,
				      const Int_t& run_number, const Int_t& nfirst,
				      const Int_t& nlast,      const Int_t& nreqevts,  const Int_t& Stex,
				      TString      path_root)
{
  //Preparation for reading the ROOT file

  // Preliminary save of the arguments values because it can be of the form: fFileHeader->...
  // and because fFileHeader can be deleted and re-created in this method

  TString sTypAna      = typ_ana;
  Int_t   nNbOfSamples = nb_of_samples;
  Int_t   nRunNumber   = run_number;
  Int_t   nFirstEvt    = nfirst;
  Int_t   nLastEvt     = nlast;
  Int_t   nReqNbOfEvts = nreqevts;
  Int_t   nStexNumber  = Stex;

  //................................................................................................
  const Text_t *h_name  = "CnaHeader";   //==> voir cette question avec FXG
  const Text_t *h_title = "CnaHeader";   //==> voir cette question avec FXG

  //----------- old version, with arguments h_name, h_title, (FXG) -----(----
  //
  // fFileHeader->HeaderParameters(h_name,    h_title,
  //			           sTypAna,   nNbOfSamples, nRunNumber,
  //			           nFirstEvt, nLastEvt, nReqNbOfEvts, nStexNumber);
  //
  //-------------------------------------------------------------------------

  //---------- new version
  if( fFileHeader == 0 ){fFileHeader = new TEcnaHeader(h_name, h_title);  /* Anew("fFileHeader") */ ;}
  fFileHeader->HeaderParameters(sTypAna,   nNbOfSamples, nRunNumber,
				nFirstEvt, nLastEvt,     nReqNbOfEvts, nStexNumber);

  // After this call to TEcnaHeader, we have:
  //     fFileHeader->fTypAna            = sTypAna
  //     fFileHeader->fNbOfSamples       = nNbOfSamples
  //     fFileHeader->fRunNumber         = nRunNumber
  //     fFileHeader->fFirstReqEvtNumber = nFirstEvt
  //     fFileHeader->fLastReqEvtNumber  = nLastEvt
  //     fFileHeader->fReqNbOfEvts       = nReqNbOfEvts
  //     fFileHeader->fStex              = nStexNumber                       ( GetReadyToReadRootFile(...) )
  //.......................... path_root
  fPathRoot = path_root;

  //-------- gets the arguments for the file names (long and short) and makes these names
  fCnaWrite->RegisterFileParameters(typ_ana, nb_of_samples, run_number, nfirst, nlast, nreqevts, Stex);
  fCnaWrite->fMakeResultsFileName();

  //------------------------- init Stin numbers memo flags
  fMemoStinNumbers = 0;

  if( fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments )
    {
      cout << endl;
      cout << "*TEcnaRead::GetReadyToReadRootFile(...)>" << endl
	   << "          The method has been called with the following argument values:" << endl
	   << "          Analysis name                = "
	   << fFileHeader->fTypAna << endl
	   << "          Nb of required samples       = "
	   << fFileHeader->fNbOfSamples << endl
	   << "          Run number                   = "
	   << fFileHeader->fRunNumber << endl
	   << "          First requested event number = "
	   << fFileHeader->fFirstReqEvtNumber << endl
	   << "          Last requested event number  = "
	   << fFileHeader->fLastReqEvtNumber << endl
	   << "          Requested number of events   = "
	   << fFileHeader->fReqNbOfEvts << endl
	   << "          Stex number                  = "
	   << fFileHeader->fStex << endl
	   << "          Path for the ROOT file       = "
	   << fPathRoot << endl
	   << endl;
    }
  
  fReadyToReadRootFile = 1;           // set flag
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TEcnaRead::GetReadyToReadRootFile(...)> Leaving the method."
	 << endl;}
} //----------------- end of GetReadyToReadRootFile(...)

//=========================================================================
//
//     Set start time and stop time, GetStartDate, GetStopDate
//
//=========================================================================
time_t  TEcnaRead::GetStartTime(){return fFileHeader->fStartTime;}
time_t  TEcnaRead::GetStopTime() {return fFileHeader->fStopTime;}
TString TEcnaRead::GetStartDate(){return fFileHeader->fStartDate;}
TString TEcnaRead::GetStopDate() {return fFileHeader->fStopDate;}
TString TEcnaRead::GetRunType()
{
  TString cType = "run type not defined";
  Int_t numtype = fFileHeader->fRunType;
  //----------------------------------------- run types

  if( numtype ==  0 ){cType = "COSMICS";}
  if( numtype ==  1 ){cType = "BEAMH4";}
  if( numtype ==  2 ){cType = "BEAMH2";}
  if( numtype ==  3 ){cType = "MTCC";}
  if( numtype ==  4 ){cType = "LASER_STD";}
  if( numtype ==  5 ){cType = "LASER_POWER_SCAN";}
  if( numtype ==  6 ){cType = "LASER_DELAY_SCAN";}
  if( numtype ==  7 ){cType = "TESTPULSE_SCAN_MEM";}
  if( numtype ==  8 ){cType = "TESTPULSE_MGPA";}
  if( numtype ==  9 ){cType = "PEDESTAL_STD";}
  if( numtype == 10 ){cType = "PEDESTAL_OFFSET_SCAN";}
  if( numtype == 11 ){cType = "PEDESTAL_25NS_SCAN";}
  if( numtype == 12 ){cType = "LED_STD";}

  if( numtype == 13 ){cType = "PHYSICS_GLOBAL";}
  if( numtype == 14 ){cType = "COSMICS_GLOBAL";}
  if( numtype == 15 ){cType = "HALO_GLOBAL";}

  if( numtype == 16 ){cType = "LASER_GAP";}
  if( numtype == 17 ){cType = "TESTPULSE_GAP";}
  if( numtype == 18 ){cType = "PEDESTAL_GAP";}
  if( numtype == 19 ){cType = "LED_GAP";}

  if( numtype == 20 ){cType = "PHYSICS_LOCAL";}
  if( numtype == 21 ){cType = "COSMICS_LOCAL";}
  if( numtype == 22 ){cType = "HALO_LOCAL";}
  if( numtype == 23 ){cType = "CALIB_LOCAL";}

  if( numtype == 24 ){cType = "PEDSIM";}

  return cType;
}
//==========================================================================
//
//                       R E A D    M E T H O D S  
//                      (   R O O T    F I L E  )
//
//==========================================================================

//============================================================================
//
//                      LookAtRootFile()
//                  
//============================================================================
Bool_t TEcnaRead::LookAtRootFile()
{
//---------- Reads the ROOT file header and makes allocations and some other things

  fLookAtRootFile = 0;          // set flag to zero before looking for the file

  Bool_t ok_read = kFALSE;

  if(fReadyToReadRootFile == 1)
    {
      //------------ Call to ReadRootFileHeader
      if( ReadRootFileHeader(0) == kTRUE )   //    (1) = print, (0) = no print
	{
	  //........................................ allocation tags      
	  if( fTagStinNumbers == 0 ){fTagStinNumbers = new Int_t[1]; Anew("fTagStinNumbers");}

	  //...................... allocation for fT1d_StexStinFromIndex[]
	  if(fT1d_StexStinFromIndex == 0)
	    {fT1d_StexStinFromIndex = new Int_t[fEcal->MaxStinEcnaInStex()]; Anew("fT1d_StexStinFromIndex");}
	  
	  //.. recover of the Stin numbers from the ROOT file (= init fT1d_StexStinFromIndex+init TagStin)
	  TVectorD vec(fEcal->MaxStinEcnaInStex());
	  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
	  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
	  
	  for (Int_t i = 0; i < fEcal->MaxStinEcnaInStex(); i++ ){
	    fT1d_StexStinFromIndex[i] = (Int_t)vec(i);}

	  fTagStinNumbers[0] = 1;                fFileHeader->fStinNumbersCalc++;
	  ok_read = kTRUE;
	  
	  fLookAtRootFile = 1;           // set flag
	}
      else
	{
	  cout << "!TEcnaRead::LookAtRootFile()> *** ERROR ***>"
	       << " ROOT file not found " << fTTBELL << endl;
	  ok_read = kFALSE; 
	}
    }
  else
    {
      cout << "!TEcnaRead::LookAtRootFile()> *** ERROR ***>"
	   << " GetReadyToReadRootFile not called " << fTTBELL << endl;
      ok_read = kFALSE;      
    }
  return ok_read;
} //----------------- end of LookAtRootFile()

//-------------------------------------------------------------------------
//
//                     ReadRootFileHeader
//
//-------------------------------------------------------------------------
Bool_t TEcnaRead::ReadRootFileHeader(const Int_t& i_print)
{
//Read the header of the Root file

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  if( i_print == 1 ){cout << "*TEcnaRead::ReadRootFileHeader> file_name = "
			 << fCnaWrite->fRootFileNameShort.Data() << endl;}

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadRootFileHeader(...)*** ERROR ***> "
	   << "Reading header on file already open." << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      if(ok_open)
	{
	  TEcnaHeader *h;
	  h =(TEcnaHeader*)gCnaRootFile->fRootFile->Get("CnaHeader");

	  //..... get the attributes which are not already set by the call to TEcnaHeader
	  //      in GetReadyToReadRootFile(...) and are only available in the ROOT file

	  fFileHeader->fStartTime = h->fStartTime;
	  fFileHeader->fStopTime  = h->fStopTime;
	  fFileHeader->fStartDate = h->fStartDate;
	  fFileHeader->fStopDate  = h->fStopDate;

	  fFileHeader->fRunType   = h->fRunType;

	  //........................................................................
	  fFileHeader->fStinNumbersCalc = h->fStinNumbersCalc;
	  fFileHeader->fAdcEvtCalc      = h->fAdcEvtCalc;
	  fFileHeader->fMSpCalc         = h->fMSpCalc;
	  fFileHeader->fSSpCalc         = h->fSSpCalc;
	  fFileHeader->fAvTnoCalc       = h->fAvTnoCalc;
	  fFileHeader->fAvLfnCalc       = h->fAvLfnCalc;
	  fFileHeader->fAvHfnCalc       = h->fAvHfnCalc;

	  fFileHeader->fCovCssCalc      = h->fCovCssCalc;
	  fFileHeader->fCorCssCalc      = h->fCorCssCalc;
	  fFileHeader->fHfCovCalc       = h->fHfCovCalc;
	  fFileHeader->fHfCorCalc       = h->fHfCorCalc;
	  fFileHeader->fLfCovCalc       = h->fLfCovCalc;
	  fFileHeader->fLfCorCalc       = h->fLfCorCalc;
	  fFileHeader->fLFccMoStinsCalc = h->fLFccMoStinsCalc;
	  fFileHeader->fHFccMoStinsCalc = h->fHFccMoStinsCalc;
	  fFileHeader->fMeanCorssCalc   = h->fMeanCorssCalc;
	  fFileHeader->fSigCorssCalc    = h->fSigCorssCalc;

	  fFileHeader->fAvPedCalc       = h->fAvPedCalc;
	  fFileHeader->fAvMeanCorssCalc = h->fAvMeanCorssCalc;
	  fFileHeader->fAvSigCorssCalc  = h->fAvSigCorssCalc;
	  
	  if(i_print == 1){fFileHeader->Print();}
          CloseRootFile(file_name);
	  ok_read = kTRUE;
	}
      else
	{
	  cout << "!TEcnaRead::ReadRootFileHeader(...) *** ERROR ***> Open ROOT file failed for file: "
	       << file_name << fTTBELL << endl;
	  ok_read = kFALSE;
	}
    }
  return ok_read;
}

//-------------------------------------------------------------
//
//                      OpenRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRead::OpenRootFile(const Text_t *name, TString status) {
//Open the Root file

  TString s_path;
  s_path = fPathRoot;
  s_path.Append('/');
  s_path.Append(name);

  gCnaRootFile = 0; gCnaRootFile = new TEcnaRootFile(s_path.Data(), status);     Anew("gCnaRootFile");

  Bool_t ok_open = kFALSE;

  if ( gCnaRootFile->fRootFileStatus == "RECREATE" ){ok_open = gCnaRootFile->OpenW();}
  if ( gCnaRootFile->fRootFileStatus == "READ"     ){ok_open = gCnaRootFile->OpenR();}

  if (!ok_open)
    {
      cout << "!TEcnaRead::OpenRootFile> " << s_path.Data() << ": file not found." << endl;
      if( gCnaRootFile != 0 ){delete gCnaRootFile; gCnaRootFile = 0;  Adelete("gCnaRootFile");}
    }
  else
    {
      if(fFlagPrint == fCodePrintAllComments){
	TString e_path;  e_path.Append(name);
	cout << "*TEcnaRead::OpenRootFile>  Open  ROOT file " << e_path.Data() << " OK " << endl;}      
      fOpenRootFile  = kTRUE;
    }

  return ok_open;
}                     // end of OpenRootFile()

//-------------------------------------------------------------
//
//                      CloseRootFile
//
//-------------------------------------------------------------
Bool_t TEcnaRead::CloseRootFile(const Text_t *name) {
//Close the Root file
 
  Bool_t ok_close = kFALSE;

  if (fOpenRootFile == kTRUE ) 
    {
      gCnaRootFile->CloseFile();
      
      if(fFlagPrint == fCodePrintAllComments){
	TString e_path;  e_path.Append(name);
	cout << "*TEcnaRead::CloseRootFile> Close ROOT file " << e_path.Data() << " OK " << endl;}
      
      if( gCnaRootFile != 0 ){delete gCnaRootFile;   gCnaRootFile = 0;  Adelete("gCnaRootFile");}
      fOpenRootFile = kFALSE;
      ok_close      = kTRUE;
    }
  else
    {
      cout << "*TEcnaRead::CloseRootFile(...)> no close since no file is open"
	   << fTTBELL << endl;
    }

  return ok_close;
}
//-------------------------------------------------------------------------
//
//                     DataExist()
//
//     DON'T SUPPRESS: CALLED BY ANOTHER CLASSES
//
//-------------------------------------------------------------------------
Bool_t TEcnaRead::DataExist()
{
  // return kTRUE if the data are present in the ROOT file, kFALSE if not.
  // fDataExist is set in the read methods

  return fDataExist;
}
//-------------------------------------------------------------------------
//
//                     ReadStinNumbers(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadStinNumbers(const Int_t& VecDim)
{
//Get the Stin numbers and put them in a TVectorD
//Read the ROOT file at first call and load in a TVectorD attribute
//Get directly the TVectorD attribute at other times
//
// Possible values for VecDim:
//          (1) VecDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim); 
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  if (fMemoStinNumbers == 0)
    {
      CnaResultTyp typ = cTypNumbers;
      const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

      //.............. reading of the ROOT file data type TResultTyp = cTypStinsNumbers
      //               to get the conversion: Stin index -> Stin number (n1StexStin)

      Bool_t ok_open = kFALSE;
      Bool_t ok_read = kFALSE;

      if ( fOpenRootFile )
	{
	  cout << "!TEcnaRead::ReadStinNumbers(...) *** ERROR ***> Reading on file already open."
	       << fTTBELL << endl;
	} 
      else
	{
	  ok_open = OpenRootFile(file_name, "READ");
	  
	  Int_t i_zero = 0;
	  ok_read = gCnaRootFile->ReadElement(typ, i_zero);
	  
	  if ( ok_read == kTRUE )
	    {
	      fDataExist = kTRUE;
	      //......... Get the Stin numbers and put them in TVectorD vec()
	      for ( Int_t i_Stin = 0; i_Stin < VecDim; i_Stin++)
		{
		  vec(i_Stin) = gCnaRootFile->fCnaIndivResult->fMatHis(0,i_Stin);
		  fT1d_StexStinFromIndex[i_Stin] = (Int_t)vec(i_Stin);
		}
	      fMemoStinNumbers++;
	    }
	  else
	    {
	      fDataExist = kFALSE;
	      cout << "!TEcnaRead::ReadStinNumbers(...) *** ERROR ***> "
		   << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
		   <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
		   << fTTBELL << endl;
	    }
	}
      CloseRootFile(file_name);
      
      if( ok_read == kTRUE )
	{
	  //........................... Print the Stin numbers 
	  if(fFlagPrint == fCodePrintAllComments)
	    {
	      for(Int_t i=0; i < VecDim; i++)
		{
		  cout << "*TEcnaRead::ReadStinNumbers(...)> StinNumber[" << i << "] = "
		       << vec[i] << endl;
		}
	    }
	}
    }
  else
    {
      fDataExist = kTRUE;
      for ( Int_t i_Stin = 0; i_Stin < VecDim; i_Stin++)
	{vec(i_Stin) = fT1d_StexStinFromIndex[i_Stin];}
    }

  return vec;
} // ----------------- ( end of ReadStinNumbers(...) ) -----------------

//-----------------------------------------------------------------------------
//
//                  ReadNumberOfEventsForSamples 
//
//-----------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadNumberOfEventsForSamples(const Int_t& n1StexStin,
						 const Int_t& MatDimX, const Int_t& MatDimY)
{
//Read the numbers of found events in the data
//for the crystals and for the samples, for a given Stin in the Stex
//in the ROOT file and return them in a TMatrixD(MaxCrysInStin,NbOfSamples)
//
//Possible values for MatDimX and MatDimY:
//  (1) MatDimX = fEcal->MaxCrysInStin(), MatDimY = fFileHeader->fNbOfSamples

  TMatrixD mat(MatDimX, MatDimY);
  for(Int_t i=0; i<MatDimX; i++)
    {for(Int_t j=0; j<MatDimY; j++){mat(i,j)=(Double_t)0.;}}

  Int_t Stin_index = GetStinIndex(n1StexStin);
  if( Stin_index >= 0 )
    {
      if(fLookAtRootFile == 1)
	{
	  CnaResultTyp typ = cTypNbOfEvts;
	  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();
	  
	  Bool_t ok_open = kFALSE;
	  Bool_t ok_read = kFALSE;
	  
	  if ( fOpenRootFile )
	    {
	      cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
		   << " Reading on file already open." << fTTBELL << endl;
	    } 
	  else
	    {
	      ok_open = OpenRootFile(file_name, "READ");
	      
	      Int_t i_zero = 0;
	      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
	      
	      if ( ok_read == kTRUE )
		{
		  fDataExist = kTRUE;
		  for (Int_t i_crys=0; i_crys<MatDimX; i_crys++)
		    {		      
		      Int_t j_cna_chan = Stin_index*MatDimX + i_crys;
		      for ( Int_t i_samp=0; i_samp<MatDimY; i_samp++)
			{
			  mat(i_crys, i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(j_cna_chan, i_samp);
			}
		    }
		}
	      else
		{
		  fDataExist = kFALSE;
		  cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
		   << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
		   <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
		   << fTTBELL << endl;
		}
	    }
	  CloseRootFile(file_name);
	}
      else
	{
	  cout << "!TEcnaRead::ReadNumberOfEventsForSamples(...) *** ERROR ***> "
	       << "It is not possible to access the number of found events: the ROOT file has not been read."
	       << fTTBELL << endl;
	}
    }  // end of if (Stin_index >= 0)
  return mat;
}

//--------------------------------------------------------------------------------------
//
//                 ReadSampleValues(i0StexEcha,sample,fFileHeader->fReqNbOfEvts)  
//
//--------------------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleValues(const Int_t& i0StexEcha, const Int_t& sample, const Int_t& VecDim)
{
//Read the event distribution for a given i0StexEcha and a given sample
//in the results ROOT file and return it in a TVectorD(nb of evts in burst)
//
//Possible values for VecDim: (1) VecDim = fFileHeader->fReqNbOfEvts

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAdcEvt;   //  sample as a function of time type

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadSampleValues(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_bin=0; i_bin<VecDim; i_bin++)
	    {
	      vec(i_bin) = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadSampleValues(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);

  return vec;
}
//--- (end of ReadSampleValues) ----------

//--------------------------------------------------------------------------------------
//
//       ReadSampleValuesSameFile(fEcal->MaxCrysEcnaInStex(),
//                                fFileHeader->fNbOfSamples, fFileHeader->fReqNbOfEvts)  
//
//--------------------------------------------------------------------------------------
Double_t*** TEcnaRead::ReadSampleValuesSameFile(const Int_t& DimX, const Int_t& DimY, const Int_t& DimZ)
{

//Possible values for DimX, DimY, DimZ : (1) DimX = fEcal->MaxCrysEcnaInStex()
//                                           DimY = fFileHeader->fNbOfSamples
//                                           DimZ = fFileHeader->fReqNbOfEvts

  if(fT3d_distribs == 0)
    {
      //............ Allocation for the 3d array 
      fT3d_distribs   = new Double_t**[DimX];                         fCnew++;  
      fT3d2_distribs  = new  Double_t*[DimX*DimY];                    fCnew++;  
      fT3d1_distribs  = new   Double_t[DimX*DimY*DimZ];               fCnew++;
      
      for(Int_t i0StexEcha = 0 ; i0StexEcha < DimX ; i0StexEcha++){
	fT3d_distribs[i0StexEcha] = &fT3d2_distribs[0] + i0StexEcha*DimY;
	for(Int_t j_samp = 0 ; j_samp < DimY ; j_samp++){
	  fT3d2_distribs[DimY*i0StexEcha + j_samp] = &fT3d1_distribs[0]+
	    DimZ*(DimY*i0StexEcha+j_samp);}}
    }
  
  //................................. Init to zero                 (ReadSampleValuesSameFile)
  for (Int_t iza=0; iza<DimX; iza++)
    {
      for (Int_t izb=0; izb<DimY; izb++)
	{
	  for (Int_t izc=0; izc<DimZ; izc++)
	    {
	      if( fT3d_distribs[iza][izb][izc] != (Double_t)0 )
		{
		  fT3d_distribs[iza][izb][izc] = (Double_t)0;
		}
	    }
	}
    }	  
  
  //-------------------------------------------------------------------------- (ReadSampleValuesSameFile)
  CnaResultTyp typ = cTypAdcEvt;   //  sample as a function of time type

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;

  Int_t  i_entry      = 0;  
  Int_t  i_entry_fail = 0;  

  ok_open = OpenRootFile(file_name, "READ");
  
  if( ok_open == kTRUE )
    {
      for(Int_t i0StexEcha=0; i0StexEcha<DimX; i0StexEcha++)
	{
	  if( i0StexEcha == 0 )
	    {
	      i_entry = gCnaRootFile->ReadElementNextEntryNumber(typ, i0StexEcha);
	      if( i_entry >= 0 ){ok_read = kTRUE;}
	    }
	  if( i_entry >= 0 )                                                //  (ReadSampleValuesSameFile)
	    {
	      if( i0StexEcha > 0 ){ok_read = gCnaRootFile->ReadElement(i_entry); i_entry++;}
	      
	      if ( ok_read == kTRUE )
		{
		  fDataExist = kTRUE;	  
		  for(Int_t sample=0; sample<DimY; sample++)
		    {
		      for ( Int_t i_bin=0; i_bin<DimZ; i_bin++)
			{
			  fT3d_distribs[i0StexEcha][sample][i_bin]
			    = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
			}  
		    }
		} 
	      else                                                        //  (ReadSampleValuesSameFile)
		{
		  fDataExist = kFALSE;
		  cout << "!TEcnaRead::ReadSampleValuesSameFile(...) *** ERROR ***> "
		       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
		       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
		       << fTTBELL << endl;
		}
	    }
	  else
	    {
	      i_entry_fail++;
	    }
	}
      CloseRootFile(file_name);
    }
  else
    {
      cout  << "*TEcnaRead::ReadSampleValuesSameFile(...)> *ERROR* =====> "
	    << " ROOT file not found" << fTTBELL << endl;
    }

  if(i_entry_fail > 0 )
    {
      cout  << "*TEcnaRead::ReadSampleValuesSameFile(...)> *ERROR* =====> "
	    << " Entry reading failure(s). i_entry_fail = "
	    << i_entry_fail << fTTBELL << endl;
    }
  return fT3d_distribs;
}
//--- (end of ReadSampleValuesSameFile) ----------

//-------------------------------------------------------------------------
//
//                  ReadSampleMeans  
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleMeans(const Int_t & n1StexStin,
						  const Int_t & i0StinEcha, const Int_t & VecDim)
{
//Read the expectation values of the samples
//for a given Stin and a given channel
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  
  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypMSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> "
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < VecDim; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha, i_samp);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadSampleMeans(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadSampleSigmas  
//                   
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSampleSigmas(const Int_t & n1StexStin,
					  const Int_t & i0StinEcha, const Int_t & VecDim)
{
//Read the expectation values of the samples
//for a given Stin and a given channel
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  
  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypSSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < VecDim; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha,i_samp);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadSampleSigmas(...) *** ERROR ***> "
		   << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
		   <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
		   << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

#define RSOS
#ifndef RSOS
//-------------------------------------------------------------------------
//
//                  ReadSigmasOfSamples(...)
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSigmasOfSamples(const Int_t & n1StexStin,
				       const Int_t & i0StinEcha, const Int_t & VecDim)
{
//Read the expectation values of the samples
//for a given Stin and a given channel
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim : (1) VecDim = fFileHeader->fNbOfSamples

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  
  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypSSp;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadSigmasOfSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < VecDim; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i0StexEcha,i_samp);
	      if( vec(i_samp) >= (Double_t)0. )
		{
		  vec(i_samp) = (Double_t)sqrt((Double_t)vec(i_samp));
		}
	      else
		{
                  vec(i_samp) = (Double_t)(-1.);
		  cout << cout << "!TEcnaRead::ReadSigmasOfSamples(...) *** ERROR ***> "
		       << "Negative variance! Sigma forced to -1" << fTTBELL << endl;
		}
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadSigmasOfSamples(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}
#endif // RSOS

//-------------------------------------------------------------------------
//
//  ReadCovariancesBetweenSamples(n1StexStin,StinEcha,fFileHeader->fNbOfSamples)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadCovariancesBetweenSamples(const Int_t & n1StexStin, const Int_t & i0StinEcha,
						 const Int_t& MatDim)
{
//Read the (sample,sample) covariances for a given channel
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fFileHeader->fNbOfSamples

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
  
  CnaResultTyp typ = cTypCovCss;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadCovariancesBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for( Int_t i_samp = 0; i_samp < MatDim; i_samp++ )
	    {
	      for ( Int_t j_samp = 0; j_samp < MatDim; j_samp++)
		{
		 mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadCovariancesBetweenSamples() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       << " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------
//
//  ReadCorrelationsBetweenSamples(n1StexStin,StinEcha,fFileHeader->fNbOfSamples)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadCorrelationsBetweenSamples(const Int_t & n1StexStin, const Int_t & i0StinEcha,
						  const Int_t& MatDim)
{
//Read the (sample,sample) correlations for a given channel
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fFileHeader->fNbOfSamples

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
 
  CnaResultTyp typ = cTypCorCss;
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < MatDim; i_samp++){
	    for ( Int_t j_samp = 0; j_samp < MatDim; j_samp++){
	      mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);}}
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadCorrelationsBetweenSamples() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       << " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return mat;
}
//-------------------------------------------------------------------------
//
//  ReadRelevantCorrelationsBetweenSamples(n1StexStin,i0StinEcha)
//                 (NOT USED)
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadRelevantCorrelationsBetweenSamples(const Int_t & n1StexStin, const Int_t & i0StinEcha,
							  const Int_t & InPutMatDim )
{
//Read the (sample,sample) correlations for a given channel
//in ROOT file and return the relevant correlations in a TVectorD
//
//Possible values for InPutMatDim: (1) InPutMatDim = fFileHeader->fNbOfSamples
//
//  *===>  OutPut TVectorD dimension value = InPutMatDim*(InPutMatDim-1)/2

  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(n1StexStin, i0StinEcha);
  Int_t nb_of_relevant = InPutMatDim*(InPutMatDim-1)/2;
  TVectorD vec_rel(nb_of_relevant); for(Int_t i=0; i<nb_of_relevant; i++){vec_rel(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypCorCss;
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadRelevantCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, i0StexEcha);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  Int_t k_cor = 0;
	  for ( Int_t i_samp = 0; i_samp < InPutMatDim; i_samp++){
	    for ( Int_t j_samp = 0; j_samp < i_samp; j_samp++){
	      vec_rel(k_cor) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
	      k_cor++;}}
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadRelevantCorrelationsBetweenSamples() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec_rel;
}
//----- end of (ReadRelevantCorrelationsBetweenSamples ) -------

//-----------------------------------------------------------------------------------------
//
//        ReadLowFrequencyCovariancesBetweenChannels(Stin_a, Stin_b)
//
//-----------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(const Int_t& n1StexStin_a,
							      const Int_t& n1StexStin_b,
							      const Int_t& MatDim)
{
//Read the Low Frequency cov(i0StinEcha of Stin_a, i0StinEcha of Stin b)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  Int_t   index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t   index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
  
  CnaResultTyp typ = cTypLfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ,i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_crys = 0; i_crys < MatDim; i_crys++)
	    {
	      Int_t i_cna_chan = index_Stin_a*MatDim + i_crys;
	      for ( Int_t j_crys = 0; j_crys < MatDim; j_crys++)
		{
		  Int_t j_cna_chan = index_Stin_b*MatDim + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadLowFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------------------------
//
//         ReadLowFrequencyCorrelationsBetweenChannels(Stin_a, Stin_b)
//
//-------------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(const Int_t & n1StexStin_a,
							       const Int_t & n1StexStin_b,
							       const Int_t& MatDim)
{
//Read the Low Frequency cor(i0StinEcha of Stin_a, i0StinEcha of Stin b)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  Int_t   index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t   index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
  
  CnaResultTyp typ = cTypLfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;

      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_crys = 0; i_crys < MatDim; i_crys++)
	    {
	      Int_t i_cna_chan = index_Stin_a*MatDim + i_crys;
	      for ( Int_t j_crys = 0; j_crys < MatDim; j_crys++)
		{
		  Int_t j_cna_chan = index_Stin_b*MatDim + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadLowFrequencyCorrelationsBetweenChannels(...) ) -------

//-----------------------------------------------------------------------------------------
//
//        ReadHighFrequencyCovariancesBetweenChannels(Stin_a, Stin_b)
//
//-----------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(const Int_t & n1StexStin_a,
							       const Int_t & n1StexStin_b,
							       const Int_t& MatDim)
{
//Read the High Frequency cov(i0StinEcha of Stin_a, i0StinEcha of Stin b)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  Int_t   index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t   index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
  
  CnaResultTyp typ = cTypHfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ,i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_crys = 0; i_crys < MatDim; i_crys++)
	    {
	      Int_t i_cna_chan = index_Stin_a*MatDim + i_crys;
	      for ( Int_t j_crys = 0; j_crys < MatDim; j_crys++)
		{
		  Int_t j_cna_chan = index_Stin_b*MatDim + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadHighFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------------------------
//
//         ReadHighFrequencyCorrelationsBetweenChannels(Stin_a, Stin_b)
//
//-------------------------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(const Int_t & n1StexStin_a,
								const Int_t & n1StexStin_b,
								const Int_t& MatDim)
{
//Read the High Frequency Cor(i0StinEcha of Stin_a, i0StinEcha of Stin b)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysInStin()

  Int_t   index_Stin_a = GetStinIndex(n1StexStin_a);
  Int_t   index_Stin_b = GetStinIndex(n1StexStin_b);

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}
  
  CnaResultTyp typ = cTypHfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;

      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_crys = 0; i_crys < MatDim; i_crys++)
	    {
	      Int_t i_cna_chan = index_Stin_a*MatDim + i_crys;
	      for ( Int_t j_crys = 0; j_crys < MatDim; j_crys++)
		{
		  Int_t j_cna_chan = index_Stin_b*MatDim + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadHighFrequencyCorrelationsBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadLowFrequencyCovariancesBetweenChannels(...)
//                  (NOT USED)
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels(const Int_t& MatDim)
{
//Read all the Low Frequency covariances
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++){for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  
  CnaResultTyp typ = cTypLfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++)
	    {
	      if ( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex())
		{
		  for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++)
		    {
		      if ( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex())
			{
			  for ( Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++)
			    {
			      Int_t i_cna_chan = index_Stin_a*fEcal->MaxCrysInStin() + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_Stin_a)-1)*fEcal->MaxCrysInStin() +i_crys;
			      for ( Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++)
				{
				  Int_t j_cna_chan = index_Stin_b*fEcal->MaxCrysInStin() + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b)-1)*fEcal->MaxCrysInStin() +j_crys;
				  mat(i_chan_sm, j_chan_sm) =
				    gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
				}
			    }
			}
		    }
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyCovariancesBetweenChannels() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadLowFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadLowFrequencyCorrelationsBetweenChannels(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels(const Int_t& MatDim)
{
//Read all the Low Frequency correlations
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++){for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  
  CnaResultTyp typ = cTypLfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++)
	    {
	      if ( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex())
		{
		  for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++)
		    {
		      if ( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex())
			{
			  for ( Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++)
			    {
			      Int_t i_cna_chan = index_Stin_a*fEcal->MaxCrysInStin() + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_Stin_a)-1)*fEcal->MaxCrysInStin() + i_crys;
			      for ( Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++)
				{
				  Int_t j_cna_chan = index_Stin_b*fEcal->MaxCrysInStin() + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b)-1)*fEcal->MaxCrysInStin() + j_crys;
				  mat(i_chan_sm, j_chan_sm) =
				    gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
				}
			    }
			}
		    }
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyCorrelationsBetweenChannels() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of (ReadLowFrequencyCorrelationsBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadHighFrequencyCovariancesBetweenChannels(...)
//                  (NOT USED)
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels(const Int_t& MatDim)
{
//Read all the High Frequency covariances
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++){for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  
  CnaResultTyp typ = cTypHfCov;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++)
	    {
	      if ( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex())
		{
		  for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++)
		    {
		      if ( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex())
			{
			  for ( Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++)
			    {
			      Int_t i_cna_chan = index_Stin_a*fEcal->MaxCrysInStin() + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_Stin_a)-1)*fEcal->MaxCrysInStin() +i_crys;
			      for ( Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++)
				{
				  Int_t j_cna_chan = index_Stin_b*fEcal->MaxCrysInStin() + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b)-1)*fEcal->MaxCrysInStin() +j_crys;
				  mat(i_chan_sm, j_chan_sm) =
				    gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
				}
			    }
			}
		    }
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyCovariancesBetweenChannels() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//----- end of ( ReadHighFrequencyCovariancesBetweenChannels(...) ) -------

//-------------------------------------------------------------------------
//
//         ReadHighFrequencyCorrelationsBetweenChannels(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels(const Int_t& MatDim)
{
//Read all the High Frequency correlations
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++){for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  
  CnaResultTyp typ = cTypHfCor;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for (Int_t index_Stin_a = 0; index_Stin_a < fEcal->MaxStinEcnaInStex(); index_Stin_a++)
	    {
	      if ( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= fEcal->MaxStinEcnaInStex())
		{
		  for (Int_t index_Stin_b = 0; index_Stin_b < fEcal->MaxStinEcnaInStex(); index_Stin_b++)
		    {
		      if ( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= fEcal->MaxStinEcnaInStex())
			{
			  for ( Int_t i_crys = 0; i_crys < fEcal->MaxCrysInStin(); i_crys++)
			    {
			      Int_t i_cna_chan = index_Stin_a*fEcal->MaxCrysInStin() + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_Stin_a)-1)*fEcal->MaxCrysInStin() + i_crys;
			      for ( Int_t j_crys = 0; j_crys < fEcal->MaxCrysInStin(); j_crys++)
				{
				  Int_t j_cna_chan = index_Stin_b*fEcal->MaxCrysInStin() + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_Stin_b)-1)*fEcal->MaxCrysInStin() + j_crys;
				  mat(i_chan_sm, j_chan_sm) =
				    gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
				}
			    }
			}
		    }
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyCorrelationsBetweenChannels() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}
//-------------- ( end of ReadHighFrequencyCorrelationsBetweenChannels(...) ) ---------


//-------------------------------------------------------------------------
//
//         ReadLowFrequencyMeanCorrelationsBetweenStins(...)
//
//-------------------------------------------------------------------------
TMatrixD TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins(const Int_t& MatDim)
{
//Read all the Low Frequency Mean Correlations Between Towers the for all (Stin_X, Stin_Y)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypLFccMoStins;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for(Int_t index_Stin_a = 0; index_Stin_a < MatDim; index_Stin_a++)
	    {
	      for(Int_t index_Stin_b = 0; index_Stin_b < MatDim; index_Stin_b++)   
		{
		  if( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= MatDim)
		    {
		      if( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= MatDim)
			{
			  mat((Int_t)vec(index_Stin_a)-1, (Int_t)vec(index_Stin_b)-1) =
			    gCnaRootFile->fCnaIndivResult->fMatMat(index_Stin_a,index_Stin_b);
			}
		    }
		}
	    }
	} 
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyMeanCorrelationsBetweenStins() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
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
TMatrixD TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins(const Int_t& MatDim)
{
//Read all the High Frequency Mean Correlations Between Towers the for all (Stin_X, Stin_Y)
//in ROOT file and return them in a TMatrixD
//
//Possible values for MatDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TMatrixD mat(MatDim, MatDim);
  for(Int_t i=0; i<MatDim; i++)
    {for(Int_t j=0; j<MatDim; j++){mat(i,j)=(Double_t)0.;}}

  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  CnaResultTyp typ = cTypHFccMoStins;

  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins() *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for(Int_t index_Stin_a = 0; index_Stin_a < MatDim; index_Stin_a++)
	    {
	      for(Int_t index_Stin_b = 0; index_Stin_b < MatDim; index_Stin_b++)   
		{
		  if( vec(index_Stin_a) > 0 && vec(index_Stin_a) <= MatDim)
		    {
		      if( vec(index_Stin_b) > 0 && vec(index_Stin_b) <= MatDim)
			{
			  mat((Int_t)vec(index_Stin_a)-1, (Int_t)vec(index_Stin_b)-1) =
			    gCnaRootFile->fCnaIndivResult->fMatMat(index_Stin_a,index_Stin_b);
			}
		    }
		}
	    }
	} 
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyMeanCorrelationsBetweenStins() *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}

      CloseRootFile(file_name);
    }

  return mat;
}
//-------- ( end of ReadHighFrequencyMeanCorrelationsBetweenStins) --------

//============================================================================
//
//                       1 D   H I S T O S :  M E A N
//
//============================================================================
//-----------------------------------------------------------------------------
//
//                  ReadNumberOfEvents(...)
//
//-----------------------------------------------------------------------------
TVectorD TEcnaRead::ReadNumberOfEvents(const Int_t& VecDim)
{
//Read the numbers of found events in the data
//for the crystals and for the samples for all the Stin's in the Stex
//in the ROOT file, compute the average on the samples
//and return them in a TVectorD(MaxCrysEcnaInStex)
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  TMatrixD mat(fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);

  for(Int_t iStexStin=0; iStexStin<fEcal->MaxStinEcnaInStex(); iStexStin++)
    {
      //............. set mat(,) to zero before reading it
      for(Int_t i=0; i<fEcal->MaxCrysInStin(); i++)
	{for(Int_t j=0; j<fFileHeader->fNbOfSamples; j++){mat(i,j)=(Double_t)0.;}}
      //............. read mat(,)
      mat = ReadNumberOfEventsForSamples(iStexStin+1, fEcal->MaxCrysInStin(), fFileHeader->fNbOfSamples);

      for(Int_t i0StinEcha=0; i0StinEcha<fEcal->MaxCrysInStin(); i0StinEcha++)
	{
	  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(iStexStin+1, i0StinEcha);
	  vec(i0StexEcha) = 0; 
	  //.... average value over the samples
	  for(Int_t i_samp=0; i_samp<fFileHeader->fNbOfSamples; i_samp++)
	    {vec(i0StexEcha) += mat(i0StinEcha, i_samp);}
	  vec(i0StexEcha) = vec(i0StexEcha)/fFileHeader->fNbOfSamples;
	} 
    }
  return vec;
}

//-------------------------------------------------------------------------
//
//        ReadPedestals(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadPedestals(const Int_t& VecDim)
{
//Read the expectation values of the expectation values of the samples
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypPed;    // pedestals type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadPedestals(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadPedestals(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadTotalNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadTotalNoise(const Int_t& VecDim)
{
//Read the expectation values of the sigmas of the samples
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypTno;   // Total noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadTotalNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadTotalNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}
//-------------------------------------------------------------------------
//
//          ReadMeanOfCorrelationsBetweenSamples(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadMeanOfCorrelationsBetweenSamples(const Int_t& VecDim)
{
//Read the Expectation values of the (sample,sample) correlations
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypMeanCorss;     // mean of corss type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadMeanOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadMeanOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadLowFrequencyNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadLowFrequencyNoise(const Int_t& VecDim)
{
//Read the sigmas of the expectation values of the samples
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypLfn;        // low frequency noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadLowFrequencyNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadLowFrequencyNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadHighFrequencyNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadHighFrequencyNoise(const Int_t& VecDim)
{
//Read the sigmas of the sigmas of the samples
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()
  
  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypHfn;        // high frequency noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadHighFrequencyNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadHighFrequencyNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadSigmaOfCorrelationsBetweenSamples(...)       
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(const Int_t& VecDim)
{
//Read the Expectation values of the (sample,sample) correlations
//for all the channels of a given Stin
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxCrysEcnaInStex()

  TVectorD vec(VecDim); for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}
  CnaResultTyp typ = cTypSigCorss;  // sigma of corss type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_StexCrys = 0; i_StexCrys < VecDim; i_StexCrys++)
	    {
	      vec(i_StexCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_StexCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}
//============================================================================
//
//                  1 D   H I S T O S :  A V E R A G E D   M E A N
//
//============================================================================
//-----------------------------------------------------------------------------
//
//                  ReadAveragedNumberOfEvents(...)
//
//       NB: read "direct" numbers of evts and compute the average HERE
//           (different from ReadAveragedPedestals, Noises, etc...)
//
//-----------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedNumberOfEvents(const Int_t& VecDim)
{
//Read the numbers of found events in the data
//for the crystals and for the samples for all the Stin's in the Stex
//in the ROOT file, compute the average on the samples and on the crystals
//and return them in a TVectorD(MaxStinEcnaInStex)
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vecAveraged(VecDim);
  for(Int_t i=0; i<VecDim; i++){vecAveraged(i)=(Double_t)0.;}

  TVectorD vecMean(fEcal->MaxCrysEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxCrysEcnaInStex(); i++){vecMean(i)=(Double_t)0.;}

  vecMean = ReadNumberOfEvents(fEcal->MaxCrysEcnaInStex());

  for(Int_t i0StexStin=0; i0StexStin<VecDim; i0StexStin++)
    {
      Int_t n1StexStin = i0StexStin+1;
      vecAveraged(i0StexStin) = 0;
      //.... average value over the crystals
      for(Int_t i0StinEcha=0; i0StinEcha<fEcal->MaxCrysInStin(); i0StinEcha++)
	{
	  Int_t i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(i0StexStin+1, i0StinEcha);
	  
	  if( fStexName == "SM" )
	    {vecAveraged(i0StexStin) += vecMean(i0StexEcha);}

	  if( fStexName == "Dee" )
	    {
	      //--------- EE --> Special translation for mixed SCEcna (29 and 32)
	      //                 Xtal 11 of SCEcna 29 -> Xtal 11 of SCEcna 10
	      //                 Xtal 11 of SCEcna 32 -> Xtal 11 of SCEcna 11
	      Int_t n1StinEcha = i0StinEcha+1;
	      if( n1StexStin == 10 && n1StinEcha == 11 )
		{i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(29, i0StinEcha);}
	      if( n1StexStin == 11 && n1StinEcha == 11 )
		{i0StexEcha = fEcalNumbering->Get0StexEchaFrom1StexStinAnd0StinEcha(32, i0StinEcha);}
	      if( !( (n1StexStin == 29 || n1StexStin == 32) && n1StinEcha == 11 )  )
		{vecAveraged(i0StexStin) += vecMean(i0StexEcha);}
	    }
	}

      Double_t xdivis = (Double_t)0.;
      if( fStexName == "SM"  )
	{xdivis = (Double_t)fEcal->MaxCrysInStin();}
      if( fStexName == "Dee" )
	{xdivis = (Double_t)fEcalNumbering->MaxCrysInStinEcna(fFileHeader->fStex, i0StexStin+1, "TEcnaRead");}

      vecAveraged(i0StexStin) = vecAveraged(i0StexStin)/xdivis;   
    }
  return vecAveraged;
}

//-------------------------------------------------------------------------
//
//        ReadAveragedPedestals(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedPedestals(const Int_t& VecDim)
{
//Read the expectation values of the Pedestals
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvPed;    // averaged pedestals type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedPedestals(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedPedestals(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
} // end of ReadAveragedPedestals

//-------------------------------------------------------------------------
//
//        ReadAveragedTotalNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedTotalNoise(const Int_t& VecDim)
{
//Read the expectation values of the Total Noise
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvTno;    // averaged Total Noise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedTotalNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedTotalNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//        ReadAveragedLowFrequencyNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedLowFrequencyNoise(const Int_t& VecDim)
{
//Read the expectation values of the Pedestals
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
  for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvLfn;    // averaged Low FrequencyNoise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedLowFrequencyNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedLowFrequencyNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
} // end of ReadAveragedLowFrequencyNoise

//-------------------------------------------------------------------------
//
//        ReadAveragedHighFrequencyNoise(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedHighFrequencyNoise(const Int_t& VecDim)
{
//Read the expectation values of the Pedestals
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
 for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvHfn;    // averaged High FrequencyNoise type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedHighFrequencyNoise(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedHighFrequencyNoise(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
} // end of ReadAveragedHighFrequencyNoise

//-------------------------------------------------------------------------
//
//        ReadAveragedMeanOfCorrelationsBetweenSamples(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedMeanOfCorrelationsBetweenSamples(const Int_t& VecDim)
{
//Read the expectation values of the Pedestals
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
 for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvMeanCorss;    // averaged MeanOfCorrelationsBetweenSamples type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedMeanOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedMeanOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
} // end of ReadAveragedMeanOfCorrelationsBetweenSamples

//-------------------------------------------------------------------------
//
//        ReadAveragedSigmaOfCorrelationsBetweenSamples(...)      
//
//-------------------------------------------------------------------------
TVectorD TEcnaRead::ReadAveragedSigmaOfCorrelationsBetweenSamples(const Int_t& VecDim)
{
//Read the expectation values of the Pedestals
//for all the Stins of a given Stex
//in the ROOT file and return them in a TVectorD
//
//Possible values for VecDim: (1) MatDim = fEcal->MaxStinEcnaInStex()

  TVectorD vec(VecDim);
 for(Int_t i=0; i<VecDim; i++){vec(i)=(Double_t)0.;}

  CnaResultTyp typ = cTypAvSigCorss;    // averaged SigmaOfCorrelationsBetweenSamples type
  const Text_t *file_name = (const Text_t *)fCnaWrite->fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TEcnaRead::ReadAveragedSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = 0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i0StexStin = 0; i0StexStin < VecDim; i0StexStin++)
	    {
	      vec(i0StexStin)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i0StexStin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TEcnaRead::ReadAveragedSigmaOfCorrelationsBetweenSamples(...) *** ERROR ***> "
	       << fCnaWrite->fRootFileNameShort.Data() << ": ROOT file failed ->"
	       <<  " quantity: <" << GetTypeOfQuantity(typ) << "> not available in file."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
} // end of ReadAveragedSigmaOfCorrelationsBetweenSamples

//=========================================================================
//
//          M I S C E L L A N E O U S    G E T    M E T H O D S   
//
//=========================================================================
//-------------------------------------------------------------------------
//
//    Get the name of the quantity from its "CnaResultTyp" type
//

//=========================================================================
//
//          M I S C E L L A N E O U S    G E T    M E T H O D S   
//
//=========================================================================
//-------------------------------------------------------------------------
//
//    Get the name of the quantity from its "CnaResultTyp" type
//
//-------------------------------------------------------------------------
TString TEcnaRead::GetTypeOfQuantity(const CnaResultTyp arg_typ)
{
  TString quantity_name = "?";

  if( arg_typ == cTypNumbers)
    {
      if( fFlagSubDet == "EB" ){quantity_name = "SM numbers";}
      if( fFlagSubDet == "EE" ){quantity_name = "Dee numbers";}
    }
  if( arg_typ == cTypMSp    ){quantity_name = "Mean of samples";}
  if( arg_typ == cTypSSp    ){quantity_name = "Sigma of samples";}

  if( arg_typ == cTypNbOfEvts ){quantity_name = "Number of events";}
  if( arg_typ == cTypPed      ){quantity_name = "Pedestals";}
  if( arg_typ == cTypTno      ){quantity_name = "Total noise";}
  if( arg_typ == cTypLfn      ){quantity_name = "LF noise";}
  if( arg_typ == cTypHfn      ){quantity_name = "HF noise";}
  if( arg_typ == cTypMeanCorss){quantity_name = "Mean of cor(s,s')";}
  if( arg_typ == cTypSigCorss ){quantity_name = "Sigma of cor(s,s')";}

  if( arg_typ == cTypAvPed      ){quantity_name = "Averaged pedestals";}
  if( arg_typ == cTypAvTno      ){quantity_name = "Averaged total noise";}
  if( arg_typ == cTypAvLfn      ){quantity_name = "Averaged LF noise";}
  if( arg_typ == cTypAvHfn      ){quantity_name = "Averaged HF noise";}
  if( arg_typ == cTypAvMeanCorss){quantity_name = "Averaged mean of cor(s,s')";}
  if( arg_typ == cTypAvSigCorss ){quantity_name = "Averaged sigma of cor(s,s')";}

  if( arg_typ == cTypAdcEvt ){quantity_name = "Sample ADC a.f.o event number";}

  if( arg_typ == cTypCovCss ){quantity_name = "Cov(s,s')";}
  if( arg_typ == cTypCorCss ){quantity_name = "Cor(s,s')";}
  if( arg_typ == cTypLfCov  ){quantity_name = "LF Cov(c,c')";}
  if( arg_typ == cTypLfCor  ){quantity_name = "LF Cor(c,c')";}
  if( arg_typ == cTypHfCov  ){quantity_name = "HF Cov(c,c')";}
  if( arg_typ == cTypHfCor  ){quantity_name = "HF Cor(c,c')";}

  if( fFlagSubDet == "EB" )
    {
      if( arg_typ == cTypLFccMoStins){quantity_name = "Mean LF |Cor(c,c')| in (tow,tow')";}
      if( arg_typ == cTypHFccMoStins){quantity_name = "Mean HF |Cor(c,c')| in (tow,tow')";}
    }
  if( fFlagSubDet == "EE" )
    {
      if( arg_typ == cTypLFccMoStins){quantity_name = "Mean LF |Cor(c,c')| in (SC,SC')";}
      if( arg_typ == cTypHFccMoStins){quantity_name = "Mean HF |Cor(c,c')| in (SC,SC')";}
    }
  return quantity_name;
}

//-------------------------------------------------------------------------
//
//    Get the ROOT file name (long and short)
//
//-------------------------------------------------------------------------
TString TEcnaRead::GetRootFileName(){return fCnaWrite->GetRootFileName();}
TString TEcnaRead::GetRootFileNameShort(){return fCnaWrite->GetRootFileNameShort();}
//-------------------------------------------------------------------------
//
//                     GetStexStinFromIndex
//
//  *====> DON'T SUPPRESS: this method is called by TEcnaRun and TEcnaHistos
//
//-------------------------------------------------------------------------
Int_t TEcnaRead::GetStexStinFromIndex(const Int_t& i0StexStinEcna)
{
// Get the Stin number in Stex from the Stin index

  Int_t number = -1;
  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());
  number = (Int_t)vec(i0StexStinEcna);
  return number;
}

TString  TEcnaRead::GetAnalysisName()
{
  TString astring = "?";
  if (fFileHeader != 0){astring = fFileHeader->fTypAna;}
  else {cout << "!TEcnaRead::GetAnalysisName()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return astring;
}
//------------------------------------------------------------------------
Int_t  TEcnaRead::GetFirstReqEvtNumber()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fFirstReqEvtNumber;}
  else {cout << "!TEcnaRead::GetFirstReqEvtNumber()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------
Int_t  TEcnaRead::GetReqNbOfEvts()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fReqNbOfEvts;}
  else {cout << "!TEcnaRead::GetReqNbOfEvts()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------

Int_t  TEcnaRead::GetNumberOfBinsSampleAsFunctionOfTime(){return GetReqNbOfEvts();}
//-------------------------------------------------------------------------
//
//                     GetStinIndex(n1StexStin)
//
//-------------------------------------------------------------------------
Int_t  TEcnaRead::GetStinIndex(const Int_t & n1StexStin)
{
//Get the index of the Stin from its number in Stex

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TEcnaRead::GetStinIndex(...)> fEcal->MaxStinEcnaInStex() = "
	 << fEcal->MaxStinEcnaInStex() << endl
	 << "                              n1StexStin = " << n1StexStin
	 << endl << endl;}

  Int_t Stin_index = n1StexStin-1;    // suppose les 68 tours 

#define NOGT
#ifndef NOGT
  Int_t Stin_index = -1;
  TVectorD vec(fEcal->MaxStinEcnaInStex());
  for(Int_t i=0; i<fEcal->MaxStinEcnaInStex(); i++){vec(i)=(Double_t)0.;}
  vec = ReadStinNumbers(fEcal->MaxStinEcnaInStex());

  //........................... Get the Stin index

  for(Int_t i=0; i < fEcal->MaxStinEcnaInStex(); i++)
    {
      if(fFlagPrint == fCodePrintAllComments){
        cout << "*TEcnaRead::GetStinIndex(...)> StinNumber[" << i << "] = "
             << vec[i] << endl;}
      if ( vec[i] == n1StexStin ){Stin_index = i;}
    }

  if(fFlagPrint == fCodePrintAllComments){
    cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;
    cout << "*TEcnaRead::GetStinIndex> Stin number: " << n1StexStin  << endl
         << "                          Stin index : " << Stin_index << endl;
    cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;}

  if ( Stin_index < 0 )
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "!TEcnaRead::GetStinIndex *** WARNING ***> n1StexStin" << n1StexStin << " : "
	     << "index Stin not found"
	     << fTTBELL << endl;}
    }
#endif // NOGT

  return Stin_index;
}

//=========================================================================
//
//         METHODS TO SET FLAGS TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void  TEcnaRead::PrintComments()
{
// Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  cout << "*TEcnaRead::PrintComments()> Warnings and some comments on init will be printed" << endl;
}

void  TEcnaRead::PrintWarnings()
{
// Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  cout << "*TEcnaRead::PrintWarnings()> Warnings will be printed" << endl;
}

void  TEcnaRead::PrintAllComments()
{
// Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  cout << "*TEcnaRead::PrintAllComments()> All the comments will be printed" << endl;
}

void  TEcnaRead::PrintNoComment()
{
// Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}
