//----------Author's Name: B.Fabbro, FX Gentit DSM/DAPNIA/SPP CEA-Saclay
//---------Copyright: Those valid for CEA sofware
//----------Modified: 07/06/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaReadEB.h"

#include "TSystem.h"

R__EXTERN TCnaRootFile *gCnaRootFile;

ClassImp(TCnaReadEB)
//___________________________________________________________________________
//
// TCnaReadEB.
//==============> INTRODUCTION
//
//    This class allows the user to read the .root results files (containing
//   expectation values, variances, covariances, correlations and other
//   quantities of interest) previously computed by the class TCnaRunEB.
//   (see documentation of this class)
//
//==============> PRELIMINARY REMARK
//
//    The user is not obliged to use directly this class. Another class
//   named TCnaViewEB can be used to make plots of the results computed
//   previously by means of the class TCnaRunEB. The class TCnaviewEB
//   calls TCnaReadEB and manage the reading of the .root result files.
//   (see the documentation of the class TCnaViewEB)
//
//==============> TCnaReadEB DECLARATION
//
//   The declaration is done by calling the constructor without argument:
//
//       TCnaReadEB* MyCnaRead = new TCnaReadEB();
//   
//==============> PREPARATION METHOD
//
//   There is a preparation method named: GetReadyToReadRootFile(...);
//
//       GetReadyToReadRootFile(...) is used to read the quantities written
//       in the ROOT files in order to use these quantities for analysis.
//
//                   *------------------------------------------------------*
// %%%%%%%%%%%%%%%%%%| Example of program using GetReadyToReadRootFile(...) |%%%%%%%%%%%%%%%%%%%%%%%
//                   *------------------------------------------------------*
//
//    This example describes the reading of one result file. This file is in a
//    directory which name is given by the contents of a TSTring named PathForRootFile
//
//    //................ Set values for the arguments
//      TString AnalysisName    = "cosmics"
//      Int_t   RunNumber       = 22770;
//      Int_t   FirstEvt        = 300;   
//      Int_t   NbOfEvents      = 150;
//      TString PathForRootFile = "/afs/cern.ch/etc..." // .root result files directory
//
//      TCnaReadEB*  MyCnaRead = new TCnaReadEB();
//      MyCnaRead->GetReadyToReadRootFile(AnalysisName,      RunNumber,
//                                        FirstTakenEvent,   NbOfTakenEvents,  Supermodule,
//                                        PathForRootFile);
//
//    //##########> CALL TO THE METHOD: Bool_t LookAtRootFile() (MANDATORY)
//    //
//    //             This methods returns a boolean. It tests the existence
//    //            of the ROOT file corresponding to the argument values given
//    //            in the call to the mwthod GetReadyToReadRootFile(...).
//    //            It is recommended to test the return value of the method.
//
//    // Example of use:
//
//     if( MyCnaRead->LookAtRootFile() == kFALSE )
//        {
//          cout << "*** ERROR: ROOT file not found" << endl;
//        }
//      else
//        {
//         //........... The ROOT file exists and has been found
//         //
//         //#########> CALLS TO THE METHODS WHICH RECOVER THE QUANTITIES. EXAMPLE:
//         //           (see the complete list of the methods hereafter)
//
//           Int_t   MaxSamples  = 10;
//           TMatrixD CorMat(MaxSamples,MaxSamples);
//           Int_t smtower = 59;
//           Int_t TowEcha =  4;
//           CorMat = MyCnaRead->ReadCorrelationsBetweenSamples(smtower,TowEcha);
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
//                     Declaration and Print Methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//     Just after the declaration with the constructor without arguments,
//     you can set a "Print Flag" by means of the following "Print Methods":
//
//     TCnaReadEB* MyCnaRead = new TCnaReadEB(); // declaration of the object MyCnaRead
//
//    // Print Methods: 
//
//    MyCnaRead->PrintNoComment();  // Set flag to forbid printing of all the comments
//                                  // except ERRORS.
//
//    MyCnaRead->PrintWarnings();   // (DEFAULT)
//                                  // Set flag to authorize printing of some warnings.
//                                  // WARNING/INFO: information on something unusual
//                                  // in the data is pointed out.
//                                  // WARNING/CORRECTION: something wrong (but not too serious)
//                                  // in the value of some argument is pointed out and
//                                  // automatically modified to a correct value.
//
//    MyCnaRead->PrintComments();    // Set flag to authorize printing of infos
//                                   //  and some comments concerning initialisations
//
//    MyCnaRead->PrintAllComments(); // Set flag to authorize printing of all the comments
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//       Method GetReadyToReadRootFile(...) and associated methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//   TCnaReadEB* MyCnaRead = new TCnaReadEB();
//   MyCnaRead->GetReadyToReadRootFile(AnalysisName,    RunNumber,
//                                     FirstEvent,      NbOfEvents,  SuperModule,
//                                     PathForRootFile);           
//      
//   Explanations for the arguments:
//
//      * TString  AnalysisName --> see the method "GetReadyToReadData" of the class TCnaRunEB
//      * Int_t    RunNumber    --> see the method "GetReadyToReadData" of the class TCnaRunEB     
//      * Int_t    FirstEvent   --> see the method "GetReadyToReadData" of the class TCnaRunEB     
//      * Int_t    NbOfEvents   --> see the method "GetReadyToReadData" of the class TCnaRunEB
//      * Int_t    SuperModule  --> see the method "GetReadyToReadData" of the class TCnaRunEB     
//     
//      * TString  PathForRootFile: Path of the directory containing the ROOT file.
//                 The path must be complete: /afs/cern.ch/user/... etc...
//
//
//==============> METHODS TO RECOVER THE QUANTITIES FROM THE ROOT FILE
//
//  Bool_t   LookAtRootFile();
//
//  TVectorD ReadTowerNumbers(); 
//
//  TMatrixD ReadNumbersOfFoundEventsForSamples(const Int_t& SMTow);  // TMatrixD(MaxCrysInTow,MaxSampADC)
//  TVectorD ReadSampleAsFunctionOfTime
//               (const Int_t& SMTow, const Int_t& TowEcha, const Int_t& sample);         // TVectorD(MaxBins) 
//  TVectorD ReadExpectationValuesOfSamples(const Int_t& SMTow,  const Int_t& TowEcha));  // TVectorD(MaxSampADC)
//  TVectorD ReadVariancesOfSamples(const Int_t& SMTow, const Int_t& TowEcha);            // TVectorD(MaxSampADC)
//  TVectorD ReadSigmasOfSamples(const Int_t& SMTow, const Int_t& TowEcha);               // TVectorD(MaxSampADC)
//  TVectorD ReadEventDistribution
//               (const Int_t& SMTow, const Int_t& TowEcha, const Int_t& sample);         // TVectorD(Maxbins)
//  Double_t ReadEventDistributionXmin(const Int_t& SMTow, const Int_t& TowEcha, const Int_t& sample);
//  Double_t ReadEventDistributionXmax(const Int_t& SMTow, const Int_t& TowEcha, const Int_t& sample);
//
//  TMatrixD ReadCovariancesBetweenCrystalsMeanOverSamples
//              (const Int_t& SMTow_X, const Int_t& SMTow_Y);  // TMatrixD(Tow_XEcha, Tow_YEcha)
//  TMatrixD ReadCorrelationsBetweenCrystalsMeanOverSamples
//              (const Int_t& SMTow_X, const Int_t& SMTow_Y);  // TMatrixD(Tow_XEcha, Tow_YEcha)
//
//  TMatrixD ReadCovariancesBetweenCrystalsMeanOverSamples();  // TMatrixD(SMEcha, SMEcha) (BIG!: 1700x1700)
//  TMatrixD ReadCorrelationsBetweenCrystalsMeanOverSamples(); // TMatrixD(SMEcha, SMEcha) (BIG!: 1700x1700) 
//
//  TMatrixD ReadCovariancesBetweenTowersMeanOverSamplesAndChannels();  // TMatrixD(SMTow,SMTow) (68x68)
//  TMatrixD ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels(); // TMatrixD(SMTow,SMTow) (68x68)
//
//  TMatrixD ReadCovariancesBetweenSamples
//               (const Int_t& SMTow, const Int_t& TowEcha); // TMatrixD(MaxSampADC,MaxSampADC)
//  TMatrixD ReadCorrelationsBetweenSamples
//               (const Int_t& SMTow, const Int_t& TowEcha); // TMatrixD(MaxSampADC,MaxSampADC)
//  TVectorD ReadRelevantCorrelationsBetweenSamples
//               (const Int_t& SMTow, const Int_t& TowEcha); // TVectorD( MaxSampADC(MaxSampADC-1)/2 )
//
//  TVectorD ReadExpectationValuesOfExpectationValuesOfSamples();  // TVectorD(MaxCrysInSM)
//  TVectorD ReadExpectationValuesOfSigmasOfSamples();             // TVectorD(MaxCrysInSM)
//  TVectorD ReadExpectationValuesOfCorrelationsBetweenSamples();  // TVectorD(MaxCrysInSM)
//
//  TVectorD ReadSigmasOfExpectationValuesOfSamples();             // TVectorD(MaxSampADC)
//  TVectorD ReadSigmasOfSigmasOfSamples();                        // TVectorD(MaxSampADC)
//  TVectorD ReadSigmasOfCorrelationsBetweenSamples();             // TVectorD(MaxSampADC)
//
//  Int_t MaxTowEtaInSM();
//  Int_t MaxTowPhiInSM();
//  Int_t MaxTowInSM();
//
//  Int_t MaxCrysEtaInTow();           // Tower size X in terms of crystals
//  Int_t MaxCrysPhiInTow();           // Tower size Y in terms of crystals
//  Int_t MaxCrysInTow();
//
//  Int_t MaxSampADC();
//
//  Int_t GetNumberOfTakenEvents();
//
//  Int_t GetNumberOfBinsEventDistributions();
//  Int_t GetNumberOfBinsSampleAsFunctionOfTime();
//  Int_t GetNumberOfBinsEvolution();
//
//  Int_t GetTowerIndex(const Int_t& SMTow);
//  Int_t GetSMTowFromIndex(const Int_t& tower_index);
//  Int_t GetSMEcha(const Int_t& SMTow, const Int_t& TowEcha); 
//
// // Int_t GetTowerNumber(const Int_t&);    // SM-Tower number from CNA-channel number
// // Int_t GetCrystalNumber(const Int_t&);  // Crystal number in tower from CNA-channel number
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//

//------------------------------ TCnaReadEB.cxx -----------------------------
//  
//   Creation (first version): 03 Dec 2002
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

TCnaReadEB::TCnaReadEB()
{
//Constructor without argument: initialisation concerning the class
// and call to Init() (initialisations concerning the ROOT file)

  fTTBELL = '\007';

  fgMaxCar = (Int_t)512;
  fDim_name = fgMaxCar;

  //............................... codes (all the values must be different)

  fCodeHeader          =  0;
  fCodeRoot            =  1; 
  fCodeCorresp         =  2; 

  TCnaParameters* MyParameters = new TCnaParameters();   fCnew++;

  fCodePrintNoComment   = MyParameters->GetCodePrint("NoComment");
  fCodePrintWarnings    = MyParameters->GetCodePrint("Warnings ");
  fCodePrintComments    = MyParameters->GetCodePrint("Comments");
  fCodePrintAllComments = MyParameters->GetCodePrint("AllComments");

  delete MyParameters;                                   fCdelete++;

  //.................................. Set flag print to "Warnings"
  fFlagPrint = fCodePrintWarnings;
  //.................................. call to Init()
  Init();
}

void TCnaReadEB::Init()
{
//Initialisation concerning the ROOT file (called by GetReadyToReadRootFile(...)

  fCnew           = 0;
  fCdelete        = 0;

  fFileHeader    = new TCnaHeaderEB();
  fOpenRootFile  = kFALSE;

  fReadyToReadRootFile = 0;
  fLookAtRootFile      = 0;

  fUserSamp      = -1;
  fUserChan      = -1;
 
  fSectChanSizeX = 0;
  fSectChanSizeY = 0;
  fSectSampSizeX = 0;
  fSectSampSizeY = 0;

  fT1d_SMtowFromIndex = 0;

  fSpecialSMTowerNotIndexed = -1;

  //................................ tags tower numbers
  fMemoTowerNumbers = 0;
  fTagTowerNumbers  = 0;

  //.......................... flag data exist
  fDataExist = kFALSE;

  //................................. others
  Int_t MaxCar = fgMaxCar;
  fPathAscii.Resize(MaxCar);
  fPathAscii = "fPathAscii> not defined";

  MaxCar = fgMaxCar;
  fPathRoot.Resize(MaxCar);
  fPathRoot  = "fPathRoot> not defined";

  MaxCar = fgMaxCar;
  fRootFileName.Resize(MaxCar);
  fRootFileName      = "fRootFileName> not defined";

  MaxCar = fgMaxCar;
  fRootFileNameShort.Resize(MaxCar);
  fRootFileNameShort = "fRootFileNameShort> not defined";

  MaxCar = fgMaxCar;
  fAsciiFileName.Resize(MaxCar);
  fAsciiFileName      = "fAsciiFileName> not defined";

  MaxCar = fgMaxCar;
  fAsciiFileNameShort.Resize(MaxCar);
  fAsciiFileNameShort = "fAsciiFileNameShort> not defined";

  fUserSamp   =  0;
  fUserChan   =  0;
}
//=========================================== private copy ==========

void  TCnaReadEB::fCopy(const TCnaReadEB& rund)
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

  //  fT2dCrysNumbersTable  = rund.fT2dCrysNumbersTable;
  //  fT1dCrysNumbersTable  = rund.fT1dCrysNumbersTable;

  //........................................ Codes   
  fCodeHeader          = rund.fCodeHeader;
  fCodeRoot            = rund.fCodeRoot;

  fCodePrintComments    = rund.fCodePrintComments;
  fCodePrintWarnings    = rund.fCodePrintWarnings;
  fCodePrintAllComments = rund.fCodePrintAllComments;
  fCodePrintNoComment   = rund.fCodePrintNoComment;

  //.................................................. Tags
  fTagTowerNumbers  = rund.fTagTowerNumbers;

  fFlagPrint          = rund.fFlagPrint;

  fRootFileName         = rund.fRootFileName;
  fRootFileNameShort    = rund.fRootFileNameShort;
  fAsciiFileName        = rund.fAsciiFileName;
  fAsciiFileNameShort   = rund.fAsciiFileNameShort;

  fDim_name             = rund.fDim_name;
  fPathRoot             = rund.fPathRoot;
  fPathAscii            = rund.fPathAscii;

  fCnew    = rund.fCnew;
  fCdelete = rund.fCdelete;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    copy constructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaReadEB::TCnaReadEB(const TCnaReadEB& dcop)
{
  cout << "*TCnaReadEB::TCnaReadEB(const TCnaReadEB& dcop)> "
       << " It is time to write a copy constructor" << endl
       << " type an integer value and then RETURN to continue"
       << endl;
  
  { Int_t cintoto;  cin >> cintoto; }
  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaReadEB& TCnaReadEB::operator=(const TCnaReadEB& dcop)
{
//Overloading of the operator=

  fCopy(dcop);
  return *this;
}

//============================================================================
//
//                  GetReadyToReadRootFile(...)
//                  
//============================================================================

void TCnaReadEB::GetReadyToReadRootFile(TString      typ_ana,         
				      const Int_t& run_number, const Int_t& nfirst,
				      const Int_t& nevents,    const Int_t& super_module,
				      TString      path_root)
{
  //Preparation for reading the ROOT file

  Init();

  Text_t *h_name  = "CnaHeader";   //==> voir cette question avec FXG
  Text_t *h_title = "CnaHeader";   //==> voir cette question avec FXG

  fFileHeader = new TCnaHeaderEB(h_name,   h_title ,
			       typ_ana,  run_number,  nfirst,  nevents, super_module);
  
  // After this call to TCnaHeaderEB, we have:
  //     fFileHeader->fTypAna        = typ_ana
  //     fFileHeader->fRunNumber     = run_number
  //     fFileHeader->fFirstEvt      = nfirst
  //     fFileHeader->fNbOfTakenEvts = nevents
  //     fFileHeader->fSuperModule   = super_module
  
  fPathRoot = path_root;

  //-------- gets the names (long and short) of the ROOT file
  DefineResultsRootFilePath(path_root);        //  (by a call to fMakeResultsFileName())

  if( fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments ){
    cout << endl;
    cout << "*TCnaReadEB::GetReadyToReadRootFile(...)>" << endl
	 << "          The method has been called with the following argument values:" << endl
	 << "          Analysis name          = "
	 << fFileHeader->fTypAna << endl
	 << "          Run number             = "
	 << fFileHeader->fRunNumber << endl
	 << "          First taken event      = "
	 << fFileHeader->fFirstEvt << endl
	 << "          Number of taken events = "
	 << fFileHeader->fNbOfTakenEvts << endl
	 << "          Super-module number    = "
	 << fFileHeader->fSuperModule << endl
	 << "          Path for the ROOT file = "
	 << fPathRoot << endl
	 << endl;}

  fReadyToReadRootFile = 1;           // set flag

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaReadEB::GetReadyToReadRootFile(...)> Leaving the method."
	 << endl;}
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaReadEB::~TCnaReadEB()
{
//Destructor

  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments){
    cout << "*TCnaReadEB::~TCnaReadEB()> Entering destructor" << endl;}

  if (fT1d_SMtowFromIndex      != 0){delete [] fT1d_SMtowFromIndex;       fCdelete++;}
  if (fTagTowerNumbers         != 0){delete [] fTagTowerNumbers;          fCdelete++;}

  if ( fCnew != fCdelete )
    {
      cout << "!TCnaReadEB/destructor> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << fTTBELL << endl;
    }
  else
    {
      // cout << "*TCnaReadEB/destructor> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
      //	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  
  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments){
    cout << "*TCnaReadEB::~TCnaReadEB()> End of destructor " << endl;}
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                             M  E  T  H  O  D  S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//=========================================================================
//
//     ROOT file directory path and making of the ROOT file name
//
//=========================================================================
void   TCnaReadEB::DefineResultsRootFilePath(TString path_name)
{
// Gets ROOT file directory path and makes the ROOT file name.

  fPathRoot = path_name;
  Int_t i_code = fCodeRoot;
  fMakeResultsFileName(i_code); 
}

//=========================================================================
//
//     Set start time and stop time, GetStartDate, GetStopDate
//
//=========================================================================

time_t TCnaReadEB::GetStartTime()
{
  return fFileHeader->fStartTime;
}

time_t TCnaReadEB::GetStopTime()
{
  return fFileHeader->fStopTime;
}

TString TCnaReadEB::GetStartDate()
{
  return fFileHeader->fStartDate;
}

TString TCnaReadEB::GetStopDate()
{
  return fFileHeader->fStopDate;
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

Bool_t TCnaReadEB::LookAtRootFile()
{
//---------- Reads the ROOT file header and makes allocations and some other things

  fLookAtRootFile = 0;          // set flag to zero before looking for the file

  Bool_t ok_read = kFALSE;

  if (fReadyToReadRootFile == 1)
    {
      if ( ReadRootFileHeader(0) == kTRUE )   //    (1) = print, (0) = no print
	{
	  //........................................ allocation tags      
	  if ( fTagTowerNumbers == 0 ){fTagTowerNumbers    = new Int_t[1];         fCnew++;}

	  //...................... allocation for fT1d_SMtowFromIndex[]
	  
	  if (fT1d_SMtowFromIndex == 0)
	    {
	      fT1d_SMtowFromIndex = new Int_t[fFileHeader->fMaxTowInSM];           fCnew++;
	    }
	  
	  //.... recover of the tower numbers from the ROOT file ( = init fT1d_SMtowFromIndex + init TagTower)
	  TVectorD vec(fFileHeader->fMaxTowInSM);
	  vec = ReadTowerNumbers();
	  
	  for (Int_t i = 0; i < fFileHeader->fMaxTowInSM; i++ ){
	    fT1d_SMtowFromIndex[i] = (Int_t)vec(i);}

	  fTagTowerNumbers[0] = 1;                fFileHeader->fTowerNumbersCalc++;
	  ok_read = kTRUE;
	  
	  fLookAtRootFile = 1;           // set flag
	}
      else
	{
	  cout << "!TCnaReadEB::LookAtRootFile()> *** ERROR ***>"
	       << " ROOT file not found " << fTTBELL << endl;
	  ok_read = kFALSE; 
	}
    }
  else
    {
      cout << "!TCnaReadEB::LookAtRootFile()> *** ERROR ***>"
	   << " GetReadyToReadRootFile not called " << fTTBELL << endl;
      ok_read = kFALSE;      
    }
  return ok_read;
}

//-------------------------------------------------------------------------
//
//                     ReadRootFileHeader
//
//-------------------------------------------------------------------------
Bool_t TCnaReadEB::ReadRootFileHeader(const Int_t& i_print)
{
//Read the header of the Root file

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  if( i_print == 1){cout << "*TCnaReadEB::ReadRootFileHeader> file_name = "
			 << file_name << endl;}

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadRootFileHeader(...)*** ERROR ***> "
	   << "Reading header on file already open."
	   << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      if(ok_open)
	{
	  TCnaHeaderEB *h;
	  h =(TCnaHeaderEB *)gCnaRootFile->fRootFile->Get("CnaHeader");

	  //..... get the attributes which are not already set by the call to TCnaHeaderEB
	  //      in GetReadyToReadRootFile(...) and are only available in the ROOT file

	  fFileHeader->fStartTime  = h->fStartTime;
	  fFileHeader->fStopTime   = h->fStopTime;
	  fFileHeader->fStartDate  = h->fStartDate;
	  fFileHeader->fStopDate   = h->fStopDate;

	  fFileHeader->fNentries   = h->fNentries;

	  //.......................... En principe deja initialise a la declaration
	  fFileHeader->fMaxTowInSM     = h->fMaxTowInSM;
	  fFileHeader->fMaxCrysInTow   = h->fMaxCrysInTow;
	  fFileHeader->fMaxSampADC     = h->fMaxSampADC;
	  fFileHeader->fNbBinsADC      = h->fNbBinsADC;
	  //fFileHeader->fNbBinsEvol     = h->fNbBinsEvol;
	  fFileHeader->fNbBinsSampTime = h->fNbBinsSampTime;
	  
	  fFileHeader->fMaxCrysInSM    = h->fMaxCrysInSM;
	  //........................................................................

	  fFileHeader->fTowerNumbersCalc    = h->fTowerNumbersCalc;
	  fFileHeader->fSampTimeCalc        = h->fSampTimeCalc;
	  fFileHeader->fEvCalc              = h->fEvCalc;
	  fFileHeader->fVarCalc             = h->fVarCalc;
	  fFileHeader->fEvtsCalc            = h->fEvtsCalc;
	  fFileHeader->fCovCssCalc          = h->fCovCssCalc;
	  fFileHeader->fCorCssCalc          = h->fCorCssCalc;
	  fFileHeader->fCovSccCalc          = h->fCovSccCalc;
	  fFileHeader->fCorSccCalc          = h->fCorSccCalc;
	  fFileHeader->fCovSccMosCalc       = h->fCovSccMosCalc;
	  fFileHeader->fCorSccMosCalc       = h->fCorSccMosCalc;
	  fFileHeader->fCovMosccMotCalc     = h->fCovMosccMotCalc;
	  fFileHeader->fCorMosccMotCalc     = h->fCorMosccMotCalc;
	  fFileHeader->fEvCorCssCalc        = h->fEvCorCssCalc;
	  fFileHeader->fSigCorCssCalc       = h->fSigCorCssCalc;
	  fFileHeader->fSvCorrecCovCssCalc  = h->fSvCorrecCovCssCalc;
	  fFileHeader->fCovCorrecCovCssCalc = h->fCovCorrecCovCssCalc;
	  fFileHeader->fCorCorrecCovCssCalc = h->fCorCorrecCovCssCalc;
	  
	  if(i_print == 1){fFileHeader->Print();}
          CloseRootFile(file_name);
	  ok_read = kTRUE;
	}
      else
	{
	  cout << "!TCnaReadEB::ReadRootFileHeader(...) *** ERROR ***> Open ROOT file failed for file: "
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
Bool_t TCnaReadEB::OpenRootFile(Text_t *name, TString status) {
//Open the Root file

  TString s_path;
  s_path = fPathRoot;
  s_path.Append('/');
  s_path.Append(name);

  gCnaRootFile   = new TCnaRootFile(s_path.Data(), status);     fCnew++;

  Bool_t ok_open = kFALSE;

  if ( gCnaRootFile->fRootFileStatus == "RECREATE" )
    {
      ok_open = gCnaRootFile->OpenW();
    }
  if ( gCnaRootFile->fRootFileStatus == "READ"     )
    {
      ok_open = gCnaRootFile->OpenR();
    }

  if (!ok_open)
    {
      cout << "TCnaReadEB::OpenRootFile> " << s_path.Data() << ": file not found." << endl;
      delete gCnaRootFile;                                     fCdelete++;
    }
  else
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "*TCnaReadEB::OpenRootFile> Open ROOT file OK " << endl;}      
      fOpenRootFile  = kTRUE;
    }
  return ok_open;
}                     // end of OpenRootFile()

//-------------------------------------------------------------
//
//                      CloseRootFile
//
//-------------------------------------------------------------
Bool_t TCnaReadEB::CloseRootFile(Text_t *name) {
//Close the Root file
 
  Bool_t ok_close = kFALSE;

  if (fOpenRootFile == kTRUE ) 
    {
      gCnaRootFile->CloseFile();

      if(fFlagPrint == fCodePrintAllComments){
	cout << "*TCnaReadEB::CloseRootFile> Close ROOT file OK " << endl;}

      delete gCnaRootFile;                                     fCdelete++;
      fOpenRootFile = kFALSE;
      ok_close      = kTRUE;
    }
  else
    {
      cout << "*TCnaReadEB::CloseRootFile(...)> no close since no file is open"
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
Bool_t TCnaReadEB::DataExist()
{
  // return kTRUE if the data are present in the ROOT file, kFALSE if not.
  // fDataExist is set in the read methods

  return fDataExist;
}
//-------------------------------------------------------------------------
//
//                     ReadTowerNumbers()
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadTowerNumbers()
{
//Get the tower numbers and put them in a TVectorD
//Read the ROOT file at first call and load in a TVectorD attribute
//Get directly the TVectorD attribute at other times

  TVectorD vec(fFileHeader->fMaxTowInSM);
  
  if (fMemoTowerNumbers == 0)
    {
      CnaResultTyp typ = cTypTowerNumbers;
      Text_t *file_name = (Text_t *)fRootFileNameShort.Data();
      
      //.............. reading of the ROOT file data type TResultTyp = cTypTowersNumbers
      //               to get the conversion: Tower index -> Tower number (SMtow)
      
      Bool_t ok_open = kFALSE;
      Bool_t ok_read = kFALSE;
      
      if ( fOpenRootFile )
	{
	  cout << "!TCnaReadEB::ReadTowerNumbers(...) *** ERROR ***> Reading on file already open."
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
	      //......... Get the tower numbers and put them in TVectorD vec()
	      for ( Int_t i_tow = 0; i_tow < fFileHeader->fMaxTowInSM; i_tow++)
		{
		  vec(i_tow) = gCnaRootFile->fCnaIndivResult->fMatHis(0,i_tow);
		  fT1d_SMtowFromIndex[i_tow] = (Int_t)vec(i_tow);
		  fMemoTowerNumbers = 1;
		}
	    }
	  else
	    {
	      fDataExist = kFALSE;
	      cout << "!TCnaReadEB::ReadTowerNumbers(...) *** ERROR ***> "
		   << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
		   << " File = " << fRootFileNameShort.Data()
		   << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
		   << fTTBELL << endl;
	    }
	}
      
      CloseRootFile(file_name);
      
      if( ok_read == kTRUE )
	{
	  //........................... Print the tower numbers
	  
	  if(fFlagPrint != fCodePrintNoComment)
	    {
	      for(Int_t i=0; i < fFileHeader->fMaxTowInSM; i++)
		{
		  cout << "*TCnaReadEB::ReadTowerNumbers(...)> TowerNumber[" << i << "] = "
		       << vec[i] << endl;
		}
	    }
	}
    }
  else
    {
      for ( Int_t i_tow = 0; i_tow < fFileHeader->fMaxTowInSM; i_tow++)
	{
	  vec(i_tow) = fT1d_SMtowFromIndex[i_tow];
	}
    }
  return vec;
}

//-----------------------------------------------------------------------------
//
//                  ReadNumbersOfFoundEventsForSamples 
//
//-----------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadNumbersOfFoundEventsForSamples(const Int_t& SMTow)
{
//Read the numbers of found events in the data
//for the crystals and for the samples, for a given tower
//in the ROOT file and return them in a TMatrixD(MaxCrysInTow,MaxSampADC)

  Int_t tow_index = GetTowerIndex(SMTow);

  TMatrixD mat(fFileHeader->fMaxCrysInTow, fFileHeader->fMaxSampADC);

  if( tow_index >= 0 )
    {
      if(fLookAtRootFile == 1)
	{
	  CnaResultTyp typ = cTypLastEvtNumber;
	  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();
	  
	  Bool_t ok_open = kFALSE;
	  Bool_t ok_read = kFALSE;
	  
	  if ( fOpenRootFile )
	    {
	      cout << "!TCnaReadEB::ReadNumbersOfFoundEventsForSamples(...) *** ERROR ***> "
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
		  for (Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
		    {		      
		      Int_t j_cna_chan = tow_index*fFileHeader->fMaxCrysInTow + i_crys;
		      for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
			{
			  mat(i_crys, i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(j_cna_chan, i_samp);
			}
		    }
		}
	      else
		{
		  fDataExist = kFALSE;
		  cout << "!TCnaReadEB::ReadNumbersOfFoundEventsForSamples(...) *** ERROR ***> "
		       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
		       << " File = " << fRootFileNameShort.Data()
		       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
		       << fTTBELL << endl;
		}
	    }
	  CloseRootFile(file_name);
	}
      else
	{
	  cout << "!TCnaReadEB::ReadNumbersOfFoundEventsForSamples(...) *** ERROR ***> "
	       << "It is not possible to access the number of found events: the ROOT file has not been read."
	       << fTTBELL << endl;
	}
    }  // end of if (tow_index >= 0)
  return mat;
}

//-------------------------------------------------------------------------
//
//                 ReadSampleAsFunctionOfTime(SMTow,TowEcha,sample)  
//
//-------------------------------------------------------------------------

TVectorD TCnaReadEB::ReadSampleAsFunctionOfTime(const Int_t& SMTow,
					      const Int_t& TowEcha,
					      const Int_t& sample)
{
//Read the histo of sample as a function of time for a given sample and
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  
  TVectorD vec(fFileHeader->fNbBinsSampTime);

  CnaResultTyp typ = cTypSampTime;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadSampleAsFunctionOfTime(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_bin = 0; i_bin < fFileHeader->fNbBinsSampTime; i_bin++)
	    {
	      vec(i_bin) = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadSampleAsFunctionOfTime(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << ", user_cna_chan = " << user_cna_chan
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadExpectationValuesOfSamples  
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadExpectationValuesOfSamples(const Int_t & SMTow,
						  const Int_t & TowEcha)
{
//Read the expectation values of the samples
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  
  TVectorD vec(fFileHeader->fMaxSampADC);

  CnaResultTyp typ = cTypEv;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadExpectationValuesOfSamples(...) *** ERROR ***> "
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
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(user_cna_chan, i_samp);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadExpectationValuesOfSamples(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadVariancesOfSamples  
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadVariancesOfSamples(const Int_t & SMTow,
					  const Int_t & TowEcha)
{
//Read the expectation values of the samples
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  
  TVectorD vec(fFileHeader->fMaxSampADC);

  CnaResultTyp typ = cTypVar;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadVariancesOfSamples(...) *** ERROR ***> "
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
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(user_cna_chan,i_samp);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadVariancesOfSamples(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}
//-------------------------------------------------------------------------
//
//                  ReadSigmasOfSamples  
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadSigmasOfSamples(const Int_t & SMTow,
				       const Int_t & TowEcha)
{
//Read the expectation values of the samples
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  
  TVectorD vec(fFileHeader->fMaxSampADC);

  CnaResultTyp typ = cTypVar;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadSigmasOfSamples(...) *** ERROR ***> "
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
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      vec(i_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(user_cna_chan,i_samp);
	      if( vec(i_samp) >= (Double_t)0. )
		{
		  vec(i_samp) = (Double_t)sqrt((Double_t)vec(i_samp));
		}
	      else
		{
                  vec(i_samp) = (Double_t)(-1.);
		  cout << cout << "!TCnaReadEB::ReadSigmasOfSamples(...) *** ERROR ***> "
		       << "Negative variance! Sigma forced to -1" << fTTBELL << endl;
		}
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadSigmasOfSamples(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadEventDistribution(SMTow,TowEcha,sample)  
//
//-------------------------------------------------------------------------

TVectorD TCnaReadEB::ReadEventDistribution(const Int_t& SMTow,
					 const Int_t& TowEcha,
					 const Int_t& sample)
{
//Read the event distribution for a given sample and
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  
  TVectorD vec(fFileHeader->fNbBinsADC);

  CnaResultTyp typ = cTypEvts;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadEventDistribution(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_bin = 0; i_bin < fFileHeader->fNbBinsADC; i_bin++)
	    {
	      vec(i_bin) = gCnaRootFile->fCnaIndivResult->fMatHis(sample, i_bin);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadEventDistribution(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return vec;
}

//-------------------------------------------------------------------------
//
//               ReadEventDistributionXmin(SMTow,TowEcha,sample)  
//
//-------------------------------------------------------------------------

Double_t TCnaReadEB::ReadEventDistributionXmin(const Int_t& SMTow,
					     const Int_t& TowEcha,
					     const Int_t& sample)
{
//Read the xmin of the event distribution for a given sample and
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Double_t value = 0.;
  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);

  CnaResultTyp typ = cTypEvtsXmin;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadEventDistributionXmin(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  value = gCnaRootFile->fCnaIndivResult->fMatHis(0, sample);
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadEventDistributionXmin(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);

  return value;
}
//-------------------------------------------------------------------------
//
//               ReadEventDistributionXmax(SMTow,TowEcha,sample)  
//
//-------------------------------------------------------------------------

Double_t TCnaReadEB::ReadEventDistributionXmax(const Int_t& SMTow,
					     const Int_t& TowEcha,
					     const Int_t& sample)
{
//Read the xmax of the event distribution for a given sample and
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  Double_t value = 0.;
  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);

  CnaResultTyp typ = cTypEvtsXmax;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadEventDistributionXmax(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  value = gCnaRootFile->fCnaIndivResult->fMatHis(0, sample);
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadEventDistributionXmax(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return value;
}

//-------------------------------------------------------------------------
//
//             ReadCovariancesBetweenSamples(SMTow,TowEcha)
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCovariancesBetweenSamples(const Int_t & SMTow,
						 const Int_t & TowEcha)
{
//Read the (sample,sample) covariances for a given cna_chan
//in ROOT file and return them in a TMatrixD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);

  TMatrixD mat(fFileHeader->fMaxSampADC, fFileHeader->fMaxSampADC);
  
  CnaResultTyp typ = cTypCovCss;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCovariancesBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
		{
		 mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCovariancesBetweenSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------
//
//     TMatrixD    ReadCorrelationsBetweenSamples(SMTow,TowEcha)
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrelationsBetweenSamples(const Int_t & SMTow,
						  const Int_t & TowEcha)
{
//Read the (sample,sample) correlations for a given cna_chan
//in ROOT file and return them in a TMatrixD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  TMatrixD mat(fFileHeader->fMaxSampADC, fFileHeader->fMaxSampADC);  
  CnaResultTyp typ = cTypCorCss;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++){
	    for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++){
	      mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);}}
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrelationsBetweenSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return mat;
}
//-------------------------------------------------------------------------
//
//   TVectorD   ReadRelevantCorrelationsBetweenSamples(SMTow,TowEcha)
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadRelevantCorrelationsBetweenSamples(const Int_t & SMTow,
							  const Int_t & TowEcha)
{
//Read the (sample,sample) correlations for a given cna_chan
//in ROOT file and return the relevant correlations in a TVectorD

  Int_t user_cna_chan = GetSMEcha(SMTow, TowEcha);
  Int_t nb_of_relevant = fFileHeader->fMaxSampADC*(fFileHeader->fMaxSampADC-1)/2;
  TVectorD vec_rel(nb_of_relevant);  
  CnaResultTyp typ = cTypCorCss;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadRelevantCorrelationsBetweenSamples(...) *** ERROR ***> "
	   << "Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  Int_t k_cor = 0;
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++){
	    for ( Int_t j_samp = 0; j_samp < i_samp; j_samp++){
	      vec_rel(k_cor) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
	      k_cor++;}}
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadRelevantCorrelationsBetweenSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec_rel;
}

//-----------------------------------------------------------------------------------------
//
//        ReadCovariancesBetweenCrystalsMeanOverSamples(tower_a, tower_b)
//
//-----------------------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples(const Int_t & SMTow_a,
								 const Int_t & SMTow_b)
{
//Read the (TowEcha of tower_a, TowEcha of tower b) covariances averaged over samples
//in ROOT file and return them in a TMatrixD

  Int_t   index_tow_a = GetTowerIndex(SMTow_a);
  Int_t   index_tow_b = GetTowerIndex(SMTow_b);

  TMatrixD mat(fFileHeader->fMaxCrysInTow, fFileHeader->fMaxCrysInTow);
  
  CnaResultTyp typ = cTypCovSccMos;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples(...) *** ERROR ***>"
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
	  for ( Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
	    {
	      Int_t i_cna_chan = index_tow_a*fFileHeader->fMaxCrysInTow + i_crys;
	      for ( Int_t j_crys = 0; j_crys < fFileHeader->fMaxCrysInTow; j_crys++)
		{
		  Int_t j_cna_chan = index_tow_b*fFileHeader->fMaxCrysInTow + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------------------------
//
//         ReadCorrelationsBetweenCrystalsMeanOverSamples(tower_a, tower_b)
//
//-------------------------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples(const Int_t & SMTow_a,
								  const Int_t & SMTow_b)
{
//Read the (TowEcha of tower_a, TowEcha of tower b) correlations averaged over samples
//in ROOT file and return them in a TMatrixD

  Int_t   index_tow_a = GetTowerIndex(SMTow_a);
  Int_t   index_tow_b = GetTowerIndex(SMTow_b);

  TMatrixD mat(fFileHeader->fMaxCrysInTow, fFileHeader->fMaxCrysInTow);
  
  CnaResultTyp typ = cTypCorSccMos;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples(...) *** ERROR ***>"
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
	  for ( Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
	    {
	      Int_t i_cna_chan = index_tow_a*fFileHeader->fMaxCrysInTow + i_crys;
	      for ( Int_t j_crys = 0; j_crys < fFileHeader->fMaxCrysInTow; j_crys++)
		{
		  Int_t j_cna_chan = index_tow_b*fFileHeader->fMaxCrysInTow + j_crys;
		  mat(i_crys, j_crys) = gCnaRootFile->fCnaIndivResult->fMatMat(i_cna_chan,j_cna_chan);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------
//
//         ReadCovariancesBetweenCrystalsMeanOverSamples()
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples()
{
//Read all the covariances for a given sample
//in ROOT file and return them in a TMatrixD

  Int_t MaxChannels = fFileHeader->fMaxTowInSM*fFileHeader->fMaxCrysInTow;

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MaxChannels, MaxChannels);
  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();
  
  CnaResultTyp typ = cTypCovSccMos;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples() *** ERROR ***>"
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
	  for (Int_t index_tow_a = 0; index_tow_a < fFileHeader->fMaxTowInSM; index_tow_a++)
	    {
	      if ( vec(index_tow_a) > 0 && vec(index_tow_a) <= fFileHeader->fMaxTowInSM)
		{
		  for (Int_t index_tow_b = 0; index_tow_b < fFileHeader->fMaxTowInSM; index_tow_b++)
		    {
		      if ( vec(index_tow_b) > 0 && vec(index_tow_b) <= fFileHeader->fMaxTowInSM)
			{
			  for ( Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
			    {
			      Int_t i_cna_chan = index_tow_a*fFileHeader->fMaxCrysInTow + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_tow_a)-1)*fFileHeader->fMaxCrysInTow +i_crys;
			      for ( Int_t j_crys = 0; j_crys < fFileHeader->fMaxCrysInTow; j_crys++)
				{
				  Int_t j_cna_chan = index_tow_b*fFileHeader->fMaxCrysInTow + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_tow_b)-1)*fFileHeader->fMaxCrysInTow +j_crys;
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
	  cout << "!TCnaReadEB::ReadCovariancesBetweenCrystalsMeanOverSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}


//-------------------------------------------------------------------------
//
//         ReadCorrelationsBetweenCrystalsMeanOverSamples()
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples()
{
//Read all the correlations for a given sample
//in ROOT file and return them in a TMatrixD

  Int_t MaxChannels = fFileHeader->fMaxTowInSM*fFileHeader->fMaxCrysInTow;

  //=====> WARNING: BIG MATRIX (1700x1700)
  TMatrixD mat(MaxChannels, MaxChannels);
  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();
  
  CnaResultTyp typ = cTypCorSccMos;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples() *** ERROR ***>"
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
	  for (Int_t index_tow_a = 0; index_tow_a < fFileHeader->fMaxTowInSM; index_tow_a++)
	    {
	      if ( vec(index_tow_a) > 0 && vec(index_tow_a) <= fFileHeader->fMaxTowInSM)
		{
		  for (Int_t index_tow_b = 0; index_tow_b < fFileHeader->fMaxTowInSM; index_tow_b++)
		    {
		      if ( vec(index_tow_b) > 0 && vec(index_tow_b) <= fFileHeader->fMaxTowInSM)
			{
			  for ( Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
			    {
			      Int_t i_cna_chan = index_tow_a*fFileHeader->fMaxCrysInTow + i_crys;
			      Int_t i_chan_sm = (Int_t)(vec(index_tow_a)-1)*fFileHeader->fMaxCrysInTow + i_crys;
			      for ( Int_t j_crys = 0; j_crys < fFileHeader->fMaxCrysInTow; j_crys++)
				{
				  Int_t j_cna_chan = index_tow_b*fFileHeader->fMaxCrysInTow + j_crys;
				  Int_t j_chan_sm = (Int_t)(vec(index_tow_b)-1)*fFileHeader->fMaxCrysInTow + j_crys;
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
	  cout << "!TCnaReadEB::ReadCorrelationsBetweenCrystalsMeanOverSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------
//
//         ReadCovariancesBetweenTowersMeanOverSamplesAndChannels()
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCovariancesBetweenTowersMeanOverSamplesAndChannels()
{

//Read all the mean cov(c,c) averaged on sample for all (tow_X, tow_Y)
//in ROOT file and return them in a TMatrixD

  TMatrixD mat(fFileHeader->fMaxTowInSM, fFileHeader->fMaxTowInSM);

  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();

  CnaResultTyp typ = cTypCovMosccMot;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCovariancesBetweenTowersMeanOverSamplesAndChannels() *** ERROR ***>"
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
	  for(Int_t index_tow_a = 0; index_tow_a < fFileHeader->fMaxTowInSM; index_tow_a++)
	    {
	      for(Int_t index_tow_b = 0; index_tow_b < fFileHeader->fMaxTowInSM; index_tow_b++)   
		{
		  if( vec(index_tow_a) > 0 && vec(index_tow_a) <= fFileHeader->fMaxTowInSM)
		    {
		      if( vec(index_tow_b) > 0 && vec(index_tow_b) <= fFileHeader->fMaxTowInSM)
			{
			  mat((Int_t)vec(index_tow_a)-1, (Int_t)vec(index_tow_b)-1) =
			    gCnaRootFile->fCnaIndivResult->fMatMat(index_tow_a,index_tow_b);
			}
		    }
		}
	    }
	} 
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCovariancesBetweenTowersMeanOverSamplesAndChannels() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}

      CloseRootFile(file_name);
    }

  return mat;
}


//-------------------------------------------------------------------------
//
//         ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels()
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels()
{

//Read all the mean cor(c,c) averaged over samples for all pairs (tow_X, tow_Y)
//in ROOT file and return them in a TMatrixD

  TMatrixD mat(fFileHeader->fMaxTowInSM, fFileHeader->fMaxTowInSM);

  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();

  CnaResultTyp typ = cTypCorMosccMot;

  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels() *** ERROR ***>"
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
	  for(Int_t index_tow_a = 0; index_tow_a < fFileHeader->fMaxTowInSM; index_tow_a++)
	    {
	      for(Int_t index_tow_b = 0; index_tow_b < fFileHeader->fMaxTowInSM; index_tow_b++)   
		{
		  if( vec(index_tow_a) > 0 && vec(index_tow_a) <= fFileHeader->fMaxTowInSM)
		    {
		      if( vec(index_tow_b) > 0 && vec(index_tow_b) <= fFileHeader->fMaxTowInSM)
			{
			  mat((Int_t)vec(index_tow_a)-1, (Int_t)vec(index_tow_b)-1) =
			    gCnaRootFile->fCnaIndivResult->fMatMat(index_tow_a,index_tow_b);
			}
		    }
		}
	    }
	} 
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrelationsBetweenTowersMeanOverSamplesAndChannels() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}

      CloseRootFile(file_name);
    }

  return mat;
}

//-------------------------------------------------------------------------
//
//        ReadExpectationValuesOfExpectationValuesOfSamples()      
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadExpectationValuesOfExpectationValuesOfSamples()
{
//Read the expectation values of the expectation values of the samples
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxCrysInSM);

  CnaResultTyp typ = cTypEvEv;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadExpectationValuesOfExpectationValuesOfSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero, i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadExpectationValuesOfExpectationValuesOfSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadExpectationValuesOfSigmasOfSamples()      
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadExpectationValuesOfSigmasOfSamples()
{
//Read the expectation values of the sigmas of the samples
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxCrysInSM);
  CnaResultTyp typ = cTypEvSig;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadExpectationValuesOfSigmasOfSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadExpectationValuesOfSigmasOfSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  return vec;

}

//-------------------------------------------------------------------------
//
//          ReadExpectationValuesOfCorrelationsBetweenSamples()      
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadExpectationValuesOfCorrelationsBetweenSamples()
{
//Read the Expectation values of the (sample,sample) correlations
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxCrysInSM);
  CnaResultTyp typ = cTypEvCorCss;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadExpectationValuesOfCorrelationsBetweenSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadExpectationValuesOfCorrelationsBetweenSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadSigmasOfExpectationValuesOfSamples()      
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadSigmasOfExpectationValuesOfSamples()
{
//Read the sigmas of the expectation values of the samples
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxCrysInSM);
  CnaResultTyp typ = cTypSigEv;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadSigmasOfExpectationValuesOfSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadSigmasOfExpectationValuesOfSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadSigmasOfSigmasOfSamples()      
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadSigmasOfSigmasOfSamples()
{
//Read the sigmas of the sigmas of the samples
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD
  
  TVectorD vec(fFileHeader->fMaxCrysInSM);
  CnaResultTyp typ = cTypSigSig;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadSigmasOfSigmasOfSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadSigmasOfSigmasOfSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}

//-------------------------------------------------------------------------
//
//              ReadSigmasOfCorrelationsBetweenSamples()       
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadSigmasOfCorrelationsBetweenSamples()
{
//Read the Expectation values of the (sample,sample) correlations
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxCrysInSM);
  CnaResultTyp typ = cTypSigCorCss;
  Text_t *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadSigmasOfCorrelationsBetweenSamples() *** ERROR ***> "
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
	  for ( Int_t i_SMCrys = 0; i_SMCrys < fFileHeader->fMaxCrysInSM; i_SMCrys++)
	    {
	      vec(i_SMCrys)  = gCnaRootFile->fCnaIndivResult->fMatHis(i_zero,i_SMCrys);
	    }   
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadSigmasOfCorrelationsBetweenSamples() *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }

  CloseRootFile(file_name);
  return vec;
}
//############################### CORRECTIONS ####################################
//-------------------------------------------------------------------------
//
//          TMatrixD     ReadCorrectionsToSamplesFromCovss   
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrectionsToSamplesFromCovss(const Int_t& SMTow)
{
//Read the corrections to the sample values (all of them)
//for all the TowEchas of a given tower
//in the ROOT file and return them in a TMatrixD

  TMatrixD mat(fFileHeader->fMaxCrysInTow, fFileHeader->fMaxSampADC);

  Int_t         index_tow = GetTowerIndex(SMTow);  
  CnaResultTyp  typ       = cTypSvCorrecCovCss;
  Text_t       *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrectionsToSamplesFromCovss(smtower) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = (Int_t)0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_crys = 0; i_crys < fFileHeader->fMaxCrysInTow; i_crys++)
	    {
	      Int_t i_cna_chan = index_tow*fFileHeader->fMaxCrysInTow + i_crys;
	      for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
		{
		  mat(i_crys, j_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(i_cna_chan,j_samp);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;	
	  cout << "!TCnaReadEB::ReadCorrectionsToSamplesFromCovss(smtower) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);
  
  return mat;
}

//-------------------------------------------------------------------------
//
//        TVectorD    ReadCorrectionsToSamplesFromCovss   
//
//-------------------------------------------------------------------------
TVectorD TCnaReadEB::ReadCorrectionsToSamplesFromCovss(const Int_t& SMTow,
						     const Int_t& TowEcha)
{
//Read the corrections to the sample values (all of them)
//for a given tower and a given TowEcha
//in the ROOT file and return them in a TVectorD

  TVectorD vec(fFileHeader->fMaxSampADC);

  Int_t  user_cna_chan = GetSMEcha(SMTow, TowEcha);

  CnaResultTyp  typ       = cTypSvCorrecCovCss;
  Text_t       *file_name = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrectionsToSamplesFromCovss(smtower,TowEcha) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");

      Int_t i_zero = (Int_t)0;
      ok_read = gCnaRootFile->ReadElement(typ, i_zero);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
	    {
	      vec(j_samp) = gCnaRootFile->fCnaIndivResult->fMatHis(user_cna_chan,j_samp);
	    } 
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrectionsToSamplesFromCovss(smtower,TowEcha) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);

  return vec;
}

//-------------------------------------------------------------------------
//
//                  ReadCorrectionFactorsToCovss
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrectionFactorsToCovss(const Int_t& SMTow,
					       const Int_t & TowEcha)
{
//Read the corrections factors to the covariances
//for a given TowEcha and for a given tower
//in the ROOT file and return them in a TMatrixD

  TMatrixD mat(fFileHeader->fMaxSampADC, fFileHeader->fMaxSampADC);
 
  Int_t         user_cna_chan = GetSMEcha(SMTow, TowEcha);
  CnaResultTyp  typ          = cTypCovCorrecCovCss;
  Text_t       *file_name    = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrectionFactorsToCovss(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxCrysInTow; i_samp++)
	    {
	      for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
		{
		  mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrectionFactorsToCovss(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);

  return mat;
}

//-------------------------------------------------------------------------
//
//                ReadCorrectionFactorsToCorss  
//
//-------------------------------------------------------------------------
TMatrixD TCnaReadEB::ReadCorrectionFactorsToCorss(const Int_t& SMTow,
						const Int_t & TowEcha)
{
//Read the corrections factors to the correlations
//for a given TowEcha and for a given tower
//in the ROOT file and return them in a TMatrixD

  TMatrixD mat(fFileHeader->fMaxSampADC, fFileHeader->fMaxSampADC);

  Int_t         user_cna_chan = GetSMEcha(SMTow, TowEcha);  
  CnaResultTyp  typ          = cTypCorCorrecCovCss;
  Text_t       *file_name    = (Text_t *)fRootFileNameShort.Data();

  Bool_t ok_open = kFALSE;
  Bool_t ok_read = kFALSE;
  
  if ( fOpenRootFile )
    {
      cout << "!TCnaReadEB::ReadCorrectionFactorsToCorss(...) *** ERROR ***>"
	   << " Reading on file already open." << fTTBELL << endl;
    } 
  else
    {
      ok_open = OpenRootFile(file_name, "READ");
      ok_read = gCnaRootFile->ReadElement(typ, user_cna_chan);
      
      if ( ok_read == kTRUE )
	{
	  fDataExist = kTRUE;
	  for ( Int_t i_samp = 0; i_samp < fFileHeader->fMaxCrysInTow; i_samp++)
	    {
	      for ( Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
		{
		  mat(i_samp, j_samp) = gCnaRootFile->fCnaIndivResult->fMatMat(i_samp,j_samp);
		}
	    }
	}
      else
	{
	  fDataExist = kFALSE;
	  cout << "!TCnaReadEB::ReadCorrectionFactorsToCorss(...) *** ERROR ***> "
	       << "Read ROOT file failed. Quantity type = " << GetTypeOfQuantity(typ) << endl
	       << " File = " << fRootFileNameShort.Data()
	       << ". QUANTITY NOT PRESENT IN THE ROOT FILE."
	       << fTTBELL << endl;
	}
    }
  CloseRootFile(file_name);

  return mat;
}
//########################## (END FOR CORRECTIONS) #################################

//=========================================================================
//
//          M I S C E L L A N E O U S    G E T    M E T H O D S   
//
//=========================================================================
//-------------------------------------------------------------------------
//
//    Get the name of the quantity from its "CnaresultTyp" type
//
//-------------------------------------------------------------------------
TString TCnaReadEB::GetTypeOfQuantity(const CnaResultTyp arg_typ)
{
  TString quantity_name = "?";

  if( arg_typ == cTypTowerNumbers ){quantity_name = "Tower numbers";}
  if( arg_typ == cTypEv           ){quantity_name = "Expectation values";}
  if( arg_typ == cTypVar          ){quantity_name = "Variances";}
  if( arg_typ == cTypEvts         ){quantity_name = "Event distributions";}

  if( arg_typ == cTypCovScc ){quantity_name = "Covariances between channels";}
  if( arg_typ == cTypCorScc ){quantity_name = "Correlations between channels";}
  if( arg_typ == cTypCovCss ){quantity_name = "Covariances between samples";}
  if( arg_typ == cTypCorCss ){quantity_name = "Correlations between samples";}
  if( arg_typ == cTypEvCorCss      ){quantity_name = "Mean of Correlations between samples";}
  if( arg_typ == cTypSigCorCss     ){quantity_name = "Sigma of Correlations between samples";}
  if( arg_typ == cTypLastEvtNumber ){quantity_name = "Number of events";}
  if( arg_typ == cTypEvEv       ){quantity_name = "Mean pedestals";}
  if( arg_typ == cTypEvSig      ){quantity_name = "Mean of sample sigmas";}
  if( arg_typ == cTypSigEv      ){quantity_name = "Sigma of sample means";}
  if( arg_typ == cTypSigSig     ){quantity_name = "Sigma of sample sigmas";}
  if( arg_typ == cTypSampTime   ){quantity_name = "Pedestal a.f.o event number";}
  if( arg_typ == cTypCovSccMos  ){quantity_name = "Covariances between channels (mean over samples)";}
  if( arg_typ == cTypCorSccMos  ){quantity_name = "Correlations between channels (mean over samples)";}
  if( arg_typ == cTypCovMosccMot){quantity_name = "Covariances between towers (mean samp. & chan.)";}
  if( arg_typ == cTypCorMosccMot){quantity_name = "Correlations between towers (mean samp. & chan.)";}

  return quantity_name;
}

//-------------------------------------------------------------------------
//
//    Get the ROOT file name (short)
//
//-------------------------------------------------------------------------
TString TCnaReadEB::GetRootFileNameShort()
{
  return fRootFileNameShort;
}

//-------------------------------------------------------------------------
//
//                  MaxCrysEtaInTow
//
//-------------------------------------------------------------------------
Int_t TCnaReadEB::MaxCrysEtaInTow()
{
// Get the X size of the tower in terms of TowEchas

  Int_t size = (Int_t)sqrt((Double_t)fFileHeader->fMaxCrysInTow);
  return size;
} 
//-------------------------------------------------------------------------
//
//                  MaxCrysPhiInTow
//
//-------------------------------------------------------------------------
  Int_t  TCnaReadEB::MaxCrysPhiInTow()
{
// Get the Y size of the tower in terms of crystals

  Int_t size = (Int_t)sqrt((Double_t)fFileHeader->fMaxCrysInTow);
  return size;
} 
//-------------------------------------------------------------------------
//
//                     GetSMTowFromIndex
//
//  *====>  DON'T SUPPRESS: this method is called by TCnaViewEB 
//
//-------------------------------------------------------------------------
Int_t TCnaReadEB::GetSMTowFromIndex(const Int_t& i_tower)
{
// Get the Tower number in Super-module from the tower index

  Int_t number = -1;
  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();
  number = (Int_t)vec(i_tower);
  return number;
}

//--------------------------------------------------------------------------------
//  MaxTowEtaInSM(), MaxTowPhiInSM(), 
//  MaxTowInSM(), MaxCrysInTow(), MaxSampADC(),
//  GetNumberOfTakenEvents()
//  GetNumberOfBinsEventDistributions(), GetNumberOfBinsSampleAsFunctionOfTime()
//
//--------------------------------------------------------------------------------
Int_t  TCnaReadEB::MaxTowEtaInSM()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxTowEtaInSM;}
  else {cout << "!TCnaReadEB::MaxTowEtaInSM()> *** ERROR *** fFileHeader pointer = "
	     << fFileHeader << fTTBELL <<endl;}
  return number;
}

Int_t  TCnaReadEB::MaxTowPhiInSM()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxTowPhiInSM;}
  else {cout << "!TCnaReadEB::MaxTowPhiInSM()> *** ERROR *** fFileHeader pointer = "
	     << fFileHeader << fTTBELL <<endl;}
  return number;
}
Int_t  TCnaReadEB::MaxTowInSM()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxTowInSM;}
  else {cout << "!TCnaReadEB::MaxTowInSM()> *** ERROR *** fFileHeader pointer = "
	     << fFileHeader << fTTBELL <<endl;}
  return number;
}
//------------------------------------------------------------------------
Int_t  TCnaReadEB::MaxCrysInTow()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxCrysInTow;}
  else {cout << "!TCnaReadEB::MaxCrysInTow()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------
Int_t  TCnaReadEB::MaxCrysInSM()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxCrysInSM;}
  else {cout << "!TCnaReadEB::MaxCrysInSM()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------
Int_t  TCnaReadEB::MaxSampADC()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fMaxSampADC;}
  else {cout << "!TCnaReadEB::MaxSampADC()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------
TString  TCnaReadEB::GetAnalysisName()
{
  TString astring = "?";
  if (fFileHeader != 0){astring = fFileHeader->fTypAna;}
  else {cout << "!TCnaReadEB::GetAnalysisName()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return astring;
}
//------------------------------------------------------------------------
Int_t  TCnaReadEB::GetFirstTakenEventNumber()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fFirstEvt;}
  else {cout << "!TCnaReadEB::GetFirstTakenEventNumber()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------
Int_t  TCnaReadEB::GetNumberOfTakenEvents()
{
  Int_t number = -1;
  if (fFileHeader != 0){number = fFileHeader->fNbOfTakenEvts;}
  else {cout << "!TCnaReadEB::GetNumberOfTakenEvents()> fFileHeader pointer = "
	     << fFileHeader << endl;}
  return number;
}
//------------------------------------------------------------------------

Int_t  TCnaReadEB::GetNumberOfBinsEventDistributions()
{
  return fFileHeader->fNbBinsADC;
}

//------------------------------------------------------------------------
Int_t  TCnaReadEB::GetNumberOfBinsSampleAsFunctionOfTime()
{
  return fFileHeader->fNbBinsSampTime;
}

//------------------------------------------------------------------------
//Int_t  TCnaReadEB::GetNumberOfBinsEvolution()
//{
//  return fFileHeader->fNbBinsEvol;
//}

//-------------------------------------------------------------------------
//
//                     GetSMEcha(SMTow, TowEcha)
//
//-------------------------------------------------------------------------
Int_t  TCnaReadEB::GetSMEcha(const Int_t & SMTow, const Int_t & TowEcha)
{
  Int_t j_cna_chan = -1;
  Int_t tow_index = GetTowerIndex(SMTow);

  if ( tow_index >= 0 )
    {
      j_cna_chan = tow_index*fFileHeader->fMaxCrysInTow + TowEcha;
      
      if(fFlagPrint == fCodePrintAllComments){
        cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;
        cout << "*TCnaReadEB::GetSMEcha> SMtow    : " << SMTow << endl
             << "                      TowEcha  : " << TowEcha    << endl
             << "                   => SMECha   = " << j_cna_chan << endl;
        cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;}
    }
  else
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "!TCnaReadEB::GetSMEcha *** ERROR ***> channel number not found."
	     << " Forced to -1. Argument values: SMTow = " << SMTow
	     << ", TowEcha = " << TowEcha
	     << fTTBELL << endl;}
    }
  return j_cna_chan;
}

//-------------------------------------------------------------------------
//
//                     GetTowerIndex(SMTow)
//
//-------------------------------------------------------------------------
Int_t  TCnaReadEB::GetTowerIndex(const Int_t & SMTow)
{
//Get the index of the tower from its number in SuperModule

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaReadEB::GetTowerIndex(...)> fFileHeader->fMaxTowInSM = "
	 << fFileHeader->fMaxTowInSM << endl
	 << "                              SMTow = " << SMTow
	 << endl << endl;}

  Int_t tower_index = SMTow-1;    // suppose les 68 tours 

#define NOGT
#ifndef NOGT
  Int_t tower_index = -1;
  TVectorD vec(fFileHeader->fMaxTowInSM);
  vec = ReadTowerNumbers();

  //........................... Get the tower index

  for(Int_t i=0; i < fFileHeader->fMaxTowInSM; i++)
    {
      if(fFlagPrint == fCodePrintAllComments){
        cout << "*TCnaReadEB::GetTowerIndex(...)> TowerNumber[" << i << "] = "
             << vec[i] << endl;}
      if ( vec[i] == SMTow ){tower_index = i;}
    }

  if(fFlagPrint == fCodePrintAllComments){
    cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;
    cout << "*TCnaReadEB::GetTowerIndex> Tower number: " << SMTow  << endl
         << "                          Tower index : " << tower_index << endl;
    cout << "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-" << endl;}

  if ( tower_index < 0 )
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "!TCnaReadEB::GetTowerIndex *** WARNING ***> SMTow" << SMTow << " : "
	     << "index tower not found"
	     << fTTBELL << endl;}
    }
#endif // NOGT

  return tower_index;
}

//=========================================================================
//
//         METHODS TO SET FLAGS TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void  TCnaReadEB::PrintComments()
{
// Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  cout << "*TCnaReadEB::PrintComments()> Warnings and some comments on init will be printed" << endl;
}

void  TCnaReadEB::PrintWarnings()
{
// Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  cout << "*TCnaReadEB::PrintWarnings()> Warnings will be printed" << endl;
}

void  TCnaReadEB::PrintAllComments()
{
// Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  cout << "*TCnaReadEB::PrintAllComments()> All the comments will be printed" << endl;
}

void  TCnaReadEB::PrintNoComment()
{
// Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//                 +--------------------------------------+
//                 |    P R I V A T E     M E T H O D S   |
//                 +--------------------------------------+
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//=========================================================================
//
//                 Results Filename Making  (private)
//
//=========================================================================

void  TCnaReadEB::fMakeResultsFileName(const Int_t&  i_code)
{
//Results filename making  (private)
  
  //----------------------------------------------------------------------
  //
  //     making of the name of the results file:
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
  
  char* f_in       = new char[fDim_name];                 fCnew++;
  char* f_in_short = new char[fDim_name];                 fCnew++;
  
  //  switch (i_code){  
  
  //===================================  R O O T  =====================================
  if (i_code == fCodeRoot)
    {
      sprintf(f_in, "%s/%s_%d_%d_%d_SM%d",
	      fPathRoot.Data(), fFileHeader->fTypAna.Data(),  fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  //===================================  A S C I I  ===================================  

  if (i_code == fCodeHeader)
    {
      sprintf(f_in, "%s/%s_%d_header_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_header_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeCorresp)
    {
      sprintf(f_in, "%s/%s_%d_%d_%d_SM%d_cna",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_%d_%d_SM%d_cna",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

#define FCEV
#ifndef FCEV
  if (i_code == fCodeEv)
    {
      sprintf(f_in, "%s/%s_%d_ev_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeVar)
    {
      sprintf(f_in, "%s/%s_%d_var_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if( i_code == fCodeEvts)
    {
      sprintf(f_in, "%s/%s_%d_evts_s_c%d_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserChan,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if(  i_code == fCodeCovSccMos)
    {
      sprintf(f_in, "%s/%s_%d_cov_cc_mos_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if ( i_code == fCodeCorSccMos)
    {
      sprintf(f_in, "%s/%s_%d_cor_cc_mos_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code ==  fCodeCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cov_ss_c%d_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserChan,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeCorCss)
    {
      sprintf(f_in, "%s/%s_%d_cor_ss_c%d_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserChan, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeEvCorCss)
    {
      sprintf(f_in, "%s/%s_%d_ev_cor_ss_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeSigCorCss)
    {
      sprintf(f_in, "%s/%s_%d_sig_cor_ss_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeSvCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_sv_correc_covss_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeCovCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cov_correc_covss_c%d_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserChan, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeCorCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cor_correc_covss_c%d_%d_%d_SM%d",
	      fPathAscii.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserChan, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    } 
#endif // FCEV


  // default:
  //    cout << "*RunDistribs::fMakeResultsFileName(const Int_t&  i_code)> "
  //	 << "wrong header code , i_code = " << i_code << endl; 
  //  }

  //======================================= f_name
  
  char* f_name = new char[fDim_name];                   fCnew++;
  
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      f_name[i] = '\0';
    }
  
  Int_t ii = 0;
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      if ( f_in[i] != '\0' ){f_name[i] = f_in[i]; ii++;}
      else {break;}  // va directement a f_name[ii] = '.';
    }

  //.......... writing of the file extension (.root or .ascii)

  //------------------------------------------- extension .ascii
  if ( i_code != fCodeRoot  || i_code == fCodeCorresp )
    {
      f_name[ii] = '.';   f_name[ii+1] = 'a';
      f_name[ii+2] = 's'; f_name[ii+3] = 'c';
      f_name[ii+4] = 'i'; f_name[ii+5] = 'i';

      fAsciiFileName = f_name;
    }
  //------------------------------------------- extension .root
  if ( i_code == fCodeRoot )
    {
      f_name[ii] = '.';   f_name[ii+1] = 'r';
      f_name[ii+2] = 'o'; f_name[ii+3] = 'o';  f_name[ii+4] = 't';

      fRootFileName = f_name;
    }

  //====================================== f_name_short
  
  char* f_name_short = new char[fDim_name];          fCnew++;

  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      f_name_short[i] = '\0';
    }
  
  ii = 0;
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      if ( f_in_short[i] != '\0' ){f_name_short[i] = f_in_short[i]; ii++;}
      else {break;}  // va directement a f_name_short[ii] = '.';
    }

  //.......... writing of the file extension (.root or .ascii)

  //-------------------------------------------extension .ascii
  if ( i_code != fCodeRoot || i_code == fCodeCorresp )
    {
      f_name_short[ii] = '.';   f_name_short[ii+1] = 'a';
      f_name_short[ii+2] = 's'; f_name_short[ii+3] = 'c';
      f_name_short[ii+4] = 'i'; f_name_short[ii+5] = 'i';

      fAsciiFileNameShort = f_name_short;
    }

  //-------------------------------------------- extension .root
  if ( i_code == fCodeRoot )
    {
      f_name_short[ii] = '.';   f_name_short[ii+1] = 'r';
      f_name_short[ii+2] = 'o'; f_name_short[ii+3] = 'o';
      f_name_short[ii+4] = 't';

      fRootFileNameShort = f_name_short;
    }

    delete [] f_name;                                        fCdelete++;
    delete [] f_name_short;                                  fCdelete++;

    delete [] f_in;                                          fCdelete++;
    delete [] f_in_short;                                    fCdelete++;
}
