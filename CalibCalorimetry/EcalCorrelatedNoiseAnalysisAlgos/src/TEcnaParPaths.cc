//----------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 30/01/2014

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"
#include <cstdlib>

//--------------------------------------
//  TEcnaParPaths.cc
//  Class creation: 19 May 2005
//  Documentation: see TEcnaParPaths.h
//--------------------------------------

ClassImp(TEcnaParPaths);
//______________________________________________________________________________
//

TEcnaParPaths::~TEcnaParPaths() {
  //destructor

  // std::cout << "[Info Management] CLASS: TEcnaParPaths.      DESTROY OBJECT: this = " << this << std::endl;
}

//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaParPaths::TEcnaParPaths() {
  // Constructor without argument
  // std::cout << "[Info Management] CLASS: TEcnaParPaths.      CREATE OBJECT: this = " << this << std::endl;
  Init();
}

TEcnaParPaths::TEcnaParPaths(TEcnaObject *pObjectManager) {
  // Constructor without argument
  // std::cout << "[Info Management] CLASS: TEcnaParPaths.      CREATE OBJECT: this = " << this << std::endl;
  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaParPaths", i_this);
}

void TEcnaParPaths::Init() {
  fgMaxCar = (Int_t)512;  // max number of characters in TStrings

  fTTBELL = '\007';

  //................ Init CNA Command and error numbering
  fCnaCommand = (Int_t)0;
  fCnaError = (Int_t)0;

}  // end of Init()

Bool_t TEcnaParPaths::GetPaths() {
  //............................... Get user's parameters from user's directory
  Bool_t FileHere = kFALSE;
  Int_t NbFileHere = 0;
  if (GetPathForResultsRootFiles() == kTRUE) {
    NbFileHere++;
  }  //  Init fCfgResultsRootFilePath
  if (GetPathForResultsAsciiFiles() == kTRUE) {
    NbFileHere++;
  }  //  Init fCfgResultsAsciiFilePath
  if (GetPathForHistoryRunListFiles() == kTRUE) {
    NbFileHere++;
  }  //  Init fCfgHistoryRunListFilePath

  GetCMSSWParameters();  //  Init fCfgCMSSWBase, fCfgCMSSWSubsystem, fCfgSCRAMArch <== done in TEcnaGui

  if (NbFileHere == 3) {
    FileHere = kTRUE;
  }
  return FileHere;
}

//=======================================================================================
//
//     M E T H O D S    T O    G E T    T H E    P A R A M E T E R S
//
//     F R O M    T H E    U S E R ' S    D I R E C T O R Y
//
//=======================================================================================
Bool_t TEcnaParPaths::GetPathForResultsRootFiles() { return GetPathForResultsRootFiles(""); }

Bool_t TEcnaParPaths::GetPathForResultsRootFiles(const TString &argFileName) {
  // Init fCfgResultsRootFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_results_root" (located in $HOME user's directory - default - )

  Bool_t FileHere = kFALSE;
  Int_t MaxCar = fgMaxCar;
  fCfgResultsRootFilePath.Resize(MaxCar);
  fCfgResultsRootFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsRootFilePath

  if (argFileName == "") {
    std::string cFileNameForCnaPaths = "ECNA/path_results_root";  // config file name
    TString s_file_name = cFileNameForCnaPaths.c_str();
    const Text_t *t_file_name = (const Text_t *)s_file_name.Data();

    TString s_path_name = gSystem->Getenv("HOME");  // get user's home directory path

    fFileForResultsRootFilePath = s_path_name;
    fFileForResultsRootFilePath.Append('/');
    fFileForResultsRootFilePath.Append(t_file_name);
  } else {
    fFileForResultsRootFilePath = argFileName.Data();
  }

  //........ Reading of the paths in the file named fFileForResultsRootFilePath.Data()
  //
  fFcin_rr.open(fFileForResultsRootFilePath.Data());

  if (fFcin_rr.fail() == kFALSE) {
    fFcin_rr.clear();
    std::string xResultsFileP;
    fFcin_rr >> xResultsFileP;
    fCfgResultsRootFilePath = xResultsFileP.c_str();
    fFcin_rr.close();
    FileHere = kTRUE;
  } else {
    fFcin_rr.clear();
    fCnaError++;
    std::cout << fTTBELL << std::endl
              << " ***************************************************************************************** "
              << std::endl;
    std::cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << std::endl
              << std::endl
              << "     " << fFileForResultsRootFilePath.Data() << ": file not found. " << std::endl
              << std::endl
              << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)"
              << std::endl
              << "     and create a file named path_results_root in the subdirectory ECNA." << std::endl
              << std::endl
              << "     The file " << fFileForResultsRootFilePath.Data() << " is a configuration file" << std::endl
              << "     for ECNA and must contain one line with the following syntax:" << std::endl
              << std::endl
              << "        PATH_FOR_THE_RESULTS_ROOT_FILE (without slash at the end of line)" << std::endl
              << "                                        ================================" << std::endl
              << std::endl
              << "     Example: $HOME/scratch0/ecna/results_root" << std::endl
              << std::endl
              << " ***************************************************************************************** "
              << fTTBELL << std::endl;

    fFcin_rr.close();
    FileHere = kFALSE;
  }

  return FileHere;

}  // ----------- (end of GetPathForResultsRootFiles) --------------------

//================================================================================================

Bool_t TEcnaParPaths::GetPathForResultsAsciiFiles() { return GetPathForResultsAsciiFiles(""); }

Bool_t TEcnaParPaths::GetPathForResultsAsciiFiles(const TString &argFileName) {
  // Init fCfgResultsAsciiFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_results_ascii" (located in $HOME user's directory - default - )

  Bool_t FileHere = kFALSE;

  Int_t MaxCar = fgMaxCar;
  fCfgResultsAsciiFilePath.Resize(MaxCar);
  fCfgResultsAsciiFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsAsciiFilePath

  if (argFileName == "") {
    std::string cFileNameForCnaPaths = "ECNA/path_results_ascii";  // config file name
    TString s_file_name = cFileNameForCnaPaths.c_str();
    const Text_t *t_file_name = (const Text_t *)s_file_name.Data();

    TString s_path_name = gSystem->Getenv("HOME");  // get user's home directory path

    fFileForResultsAsciiFilePath = s_path_name;
    fFileForResultsAsciiFilePath.Append('/');
    fFileForResultsAsciiFilePath.Append(t_file_name);
  } else {
    fFileForResultsAsciiFilePath = argFileName.Data();
  }

  //........ Reading of the paths in the file named fFileForResultsAsciiFilePath.Data()
  //
  fFcin_ra.open(fFileForResultsAsciiFilePath.Data());
  if (fFcin_ra.fail() == kFALSE) {
    fFcin_ra.clear();
    std::string xResultsFileP;
    fFcin_ra >> xResultsFileP;
    fCfgResultsAsciiFilePath = xResultsFileP.c_str();
    fFcin_ra.close();
    FileHere = kTRUE;
  } else {
    fFcin_ra.clear();
    fCnaError++;
    std::cout << fTTBELL << std::endl
              << " ***************************************************************************************** "
              << std::endl;
    std::cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << std::endl
              << std::endl
              << "     " << fFileForResultsAsciiFilePath.Data() << ": file not found. " << std::endl
              << std::endl
              << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)"
              << std::endl
              << "     and create a file named path_results_ascii in the subdirectory ECNA." << std::endl
              << std::endl
              << "     The file " << fFileForResultsAsciiFilePath.Data() << " is a configuration file" << std::endl
              << "     for ECNA and must contain one line with the following syntax:" << std::endl
              << std::endl
              << "        PATH_FOR_THE_RESULTS_ASCII_FILE (without slash at the end of line)" << std::endl
              << "                                         ================================" << std::endl
              << std::endl
              << "     Example: $HOME/scratch0/ecna/results_ascii" << std::endl
              << std::endl
              << " ***************************************************************************************** "
              << fTTBELL << std::endl;

    fFcin_ra.close();
    FileHere = kFALSE;
  }

  return FileHere;

}  // ----------- (end of GetPathForResultsAsciiFiles) --------------------

//================================================================================================

Bool_t TEcnaParPaths::GetPathForHistoryRunListFiles() { return GetPathForHistoryRunListFiles(""); }

Bool_t TEcnaParPaths::GetPathForHistoryRunListFiles(const TString &argFileName) {
  // Init fCfgHistoryRunListFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_runlist_history_plots" (located in $HOME user's directory - default - )

  Bool_t FileHere = kFALSE;

  Int_t MaxCar = fgMaxCar;
  fCfgHistoryRunListFilePath.Resize(MaxCar);
  fCfgHistoryRunListFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForHistoryRunListFilePath

  if (argFileName == "") {
    std::string cFileNameForCnaPaths = "ECNA/path_runlist_history_plots";  // config file name
    TString s_file_name = cFileNameForCnaPaths.c_str();
    const Text_t *t_file_name = (const Text_t *)s_file_name.Data();

    TString s_path_name = gSystem->Getenv("HOME");  // get user's home directory path

    fFileForHistoryRunListFilePath = s_path_name;
    fFileForHistoryRunListFilePath.Append('/');
    fFileForHistoryRunListFilePath.Append(t_file_name);
  } else {
    fFileForHistoryRunListFilePath = argFileName.Data();
  }

  //........ Reading of the paths in the file named fFileForHistoryRunListFilePath.Data()
  //
  fFcin_lor.open(fFileForHistoryRunListFilePath.Data());
  if (fFcin_lor.fail() == kFALSE) {
    fFcin_lor.clear();
    std::string xHistoryRunListP;
    fFcin_lor >> xHistoryRunListP;
    fCfgHistoryRunListFilePath = xHistoryRunListP.c_str();
    fFcin_lor.close();
    FileHere = kTRUE;
  } else {
    fFcin_lor.clear();
    fCnaError++;
    std::cout
        << fTTBELL << std::endl
        << " ******************************************************************************************************** "
        << std::endl;
    std::cout
        << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << std::endl
        << std::endl
        << "     " << fFileForHistoryRunListFilePath.Data() << ": file not found. " << std::endl
        << std::endl
        << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)" << std::endl
        << "     and create a file named path_runlist_history_plots in the subdirectory ECNA." << std::endl
        << std::endl
        << "     The file " << fFileForHistoryRunListFilePath.Data() << " is a configuration file" << std::endl
        << "     for ECNA and must contain one line with the following syntax:" << std::endl
        << std::endl
        << "        PATH_FOR_THE_LIST_OF_RUNS_FOR_HISTORY_PLOTS_FILE (without slash at the end of line)" << std::endl
        << "                                                          ================================" << std::endl
        << std::endl
        << "     Example: $HOME/scratch0/ecna/runlist_history_plots" << std::endl
        << std::endl
        << " ******************************************************************************************************** "
        << fTTBELL << std::endl;

    fFcin_lor.close();
    FileHere = kFALSE;
  }

  return FileHere;

}  // ----------- (end of GetPathForHistoryRunListFiles) --------------------

//================================================================================================
void TEcnaParPaths::GetCMSSWParameters() {
  // Init fCfgCMSSWBase, fCfgSCRAMArch and fCfgCMSSWSubsystem

  Int_t MaxCar = fgMaxCar;
  fCfgCMSSWBase.Resize(MaxCar);
  fCfgCMSSWBase = "?";

  fCfgSCRAMArch.Resize(MaxCar);
  fCfgSCRAMArch = "?";

  fCfgCMSSWSubsystem.Resize(MaxCar);
  fCfgCMSSWSubsystem = "?";

  //------------ CMSSW_BASE

  char *ch_cmssw_base = std::getenv("CMSSW_BASE");
  if (ch_cmssw_base == nullptr) {
    std::cout << "*TEcnaParPaths::GetCMSSWParameters()> CMSSW_BASE not defined."
              << " Please, set up the environment (command: eval `scramv1 runtime -csh`)" << fTTBELL << std::endl;
  } else {
    fCfgCMSSWBase = (TString)ch_cmssw_base;
  }

  //------------ SCRAM_ARCH

  char *ch_scram_arch = std::getenv("SCRAM_ARCH");
  if (ch_scram_arch == nullptr) {
    std::cout << "*TEcnaParPaths::GetCMSSWParameters()> SCRAM_ARCH not defined."
              << " Please, set up the environment (command: eval `scramv1 runtime -csh`)" << fTTBELL << std::endl;
  } else {
    fCfgSCRAMArch = (TString)ch_scram_arch;
  }

  //------------ SUBSYSTEM: CalibCalorimetry

  fCfgCMSSWSubsystem = "CalibCalorimetry";

}  // ----------- (end of GetCMSSWParameters) --------------------

//=======================================================================================
//
//        M E T H O D S    T O    R E T U R N    T H E    P A R A M E T E R S
//
//=======================================================================================
TString TEcnaParPaths::ResultsRootFilePath() { return fCfgResultsRootFilePath; }
TString TEcnaParPaths::ResultsAsciiFilePath() { return fCfgResultsAsciiFilePath; }
TString TEcnaParPaths::HistoryRunListFilePath() { return fCfgHistoryRunListFilePath; }
TString TEcnaParPaths::CMSSWBase() { return fCfgCMSSWBase; }
TString TEcnaParPaths::CMSSWSubsystem() { return fCfgCMSSWSubsystem; }
TString TEcnaParPaths::SCRAMArch() { return fCfgSCRAMArch; }

//.....................................................................................
TString TEcnaParPaths::PathModulesData() {
  // ----- return the path of data subdirectory in package "Modules"
  TString ModulesdataPath = "";

  const Text_t *t_cmssw_base = (const Text_t *)CMSSWBase().Data();
  ModulesdataPath.Append(t_cmssw_base);
  ModulesdataPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/

  TString s_src = "src";
  const Text_t *t_src = (const Text_t *)s_src.Data();
  ModulesdataPath.Append(t_src);
  ModulesdataPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src

  const Text_t *t_cmssw_subsystem = (const Text_t *)CMSSWSubsystem().Data();
  ModulesdataPath.Append(t_cmssw_subsystem);
  ModulesdataPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/

  TString s_package_data_path = "EcalCorrelatedNoiseAnalysisModules/data";
  const Text_t *t_package_data_path = (const Text_t *)s_package_data_path.Data();
  ModulesdataPath.Append(t_package_data_path);
  ModulesdataPath.Append('/');
  // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/

  return ModulesdataPath;
}
//.....................................................................................
TString TEcnaParPaths::PathTestScramArch() {
  // ----- return the path of test/slc... subdirectory in CMSSW_BASE
  TString TestslcPath = "";

  const Text_t *t_cmssw_base = (const Text_t *)CMSSWBase().Data();
  TestslcPath.Append(t_cmssw_base);
  TestslcPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/

  TString s_test = "test";
  const Text_t *t_test = (const Text_t *)s_test.Data();
  TestslcPath.Append(t_test);
  TestslcPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test

  const Text_t *t_cmssw_arch = (const Text_t *)SCRAMArch().Data();
  TestslcPath.Append(t_cmssw_arch);
  TestslcPath.Append('/');  //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test/slc4_ia32_gcc345/

  return TestslcPath;
}

//=======================================================================================
//
//       A N C I L L A R Y   M E T H O D S   C O N C E R N I N G   P A T H S
//
//=======================================================================================
void TEcnaParPaths::SetResultsRootFilePath(const TString &ImposedPath) { fCfgResultsRootFilePath = ImposedPath; }
void TEcnaParPaths::SetResultsAsciiFilePath(const TString &ImposedPath) { fCfgResultsAsciiFilePath = ImposedPath; }
void TEcnaParPaths::SetHistoryRunListFilePath(const TString &ImposedPath) { fCfgHistoryRunListFilePath = ImposedPath; }

void TEcnaParPaths::TruncateResultsRootFilePath(const Int_t &n1, const Int_t &nbcar) {
  fCfgResultsRootFilePath.Remove(n1, nbcar);
}

void TEcnaParPaths::TruncateResultsAsciiFilePath(const Int_t &n1, const Int_t &nbcar) {
  fCfgResultsAsciiFilePath.Remove(n1, nbcar);
}

TString TEcnaParPaths::BeginningOfResultsRootFilePath() {
  TString sBegin = "?";
  if (fCfgResultsRootFilePath.BeginsWith("$HOME")) {
    sBegin = "$HOME";
  }
  return sBegin;
}

TString TEcnaParPaths::BeginningOfResultsAsciiFilePath() {
  TString sBegin = "?";
  if (fCfgResultsAsciiFilePath.BeginsWith("$HOME")) {
    sBegin = "$HOME";
  }
  return sBegin;
}

void TEcnaParPaths::AppendResultsRootFilePath(const Text_t *t_file_nohome) {
  fCfgResultsRootFilePath.Append(t_file_nohome);
}

void TEcnaParPaths::AppendResultsAsciiFilePath(const Text_t *t_file_nohome) {
  fCfgResultsAsciiFilePath.Append(t_file_nohome);
}
