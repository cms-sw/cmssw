//---------Author's Name: B.Fabbro DSM/IRFU/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 13/10/2010
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"

ClassImp(TEcnaParPaths)
//______________________________________________________________________________
//
// TEcnaParPaths.
//
//    Values of different parameters for plots in the framework of TEcnaHistos
//    (see description of this class)
//
//    Examples of parameters:  ymin and ymax values for histos, title sizes,
//                             margins for plots, etc...
//                             
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//
//---------------------- TEcnaParPaths.cc -------------------------------
//  
//   Creation (first version): 19 May 2005
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

  TEcnaParPaths::~TEcnaParPaths()
{
//destructor

 // cout << "[Info Management] CLASS: TEcnaParPaths.      DESTROY OBJECT: this = " << this << endl;
}

//===================================================================
//
//                   Constructors
//
//===================================================================
TEcnaParPaths::TEcnaParPaths()
{
// Constructor without argument

 // cout << "[Info Management] CLASS: TEcnaParPaths.      CREATE OBJECT: this = " << this << endl;

  Init();

  //............................... Get user's parameters from user's directory
  GetPathForResultsRootFiles();        //  Init the values of fCfgResultsRootFilePath
  GetPathForResultsAsciiFiles();       //  Init the values of fCfgResultsAsciiFilePath
  GetPathForHistoryRunListFiles();     //  Init the values of fCfgHistoryRunListFilePath
  //  GetPathForAnalyzerParametersFiles(); //  Init the values of fCfgAnalyzerParametersFilePath
  GetCMSSWParameters();                //  Init the values of fCfgCMSSWVersion, fCfgCMSSWSubsystem and fCfgCMSSWSlc
}

void  TEcnaParPaths::Init()
{
  fgMaxCar = (Int_t)512;              // max number of characters in TStrings

  fTTBELL = '\007';

  //................ Init CNA Command and error numbering
  fCnaCommand = 0;
  fCnaError   = 0;

  //................ Init path flags

  fPathForResultsRootFiles    = kFALSE;
  fPathForResultsAsciiFiles   = kFALSE;
  fPathForHistoryRunListFiles = kFALSE;

}// end of Init()


//=======================================================================================
//
//     P R I V A T E    M E T H O D S    T O    G E T    T H E    P A R A M E T E R S 
//
//     F R O M    T H E    U S E R ' S    D I R E C T O R Y 
//
//=======================================================================================
void TEcnaParPaths::GetPathForResultsRootFiles()
{
  GetPathForResultsRootFiles("");
}

void TEcnaParPaths::GetPathForResultsRootFiles(const TString argFileName)
{
  // Init fCfgResultsRootFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_results_root" (located in $HOME user's directory - default - )

  Int_t MaxCar = fgMaxCar;
  fCfgResultsRootFilePath.Resize(MaxCar);
  fCfgResultsRootFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsRootFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "ECNA/path_results_root";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      const Text_t *t_file_name = (const Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForResultsRootFilePath = s_path_name;
      fFileForResultsRootFilePath.Append('/');
      fFileForResultsRootFilePath.Append(t_file_name);
    }
  else
    {
      fFileForResultsRootFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForResultsRootFilePath.Data()
  //

  fFcin_rr.open(fFileForResultsRootFilePath.Data());
  if(fFcin_rr.fail() == kFALSE)
    {
      fFcin_rr.clear();
      string xResultsFileP;
      fFcin_rr >> xResultsFileP;
      fCfgResultsRootFilePath = xResultsFileP.c_str();
      fFcin_rr.close();
      fPathForResultsRootFiles = kTRUE;
    }
  else
    {
      fFcin_rr.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************************** " << endl;
      cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "     " << fFileForResultsRootFilePath.Data() << ": file not found. " << endl << endl
	   << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)" << endl
	   << "     and create a file named path_results_root in the subdirectory ECNA." << endl << endl
           << "     The file " << fFileForResultsRootFilePath.Data() << " is a configuration file" << endl
	   << "     for the CNA and must contain one line with the following syntax:" << endl << endl
	   << "        PATH_OF_THE_RESULTS_ROOT_FILE ($HOME/etc...) (without slash at the end of line)" << endl
	   << "                                                      ================================"
	   << endl << endl
	   << "     Example: $HOME/scratch0/cna/results_root" << endl << endl
	   << " ***************************************************************************************** "
	   << fTTBELL << endl;

      fFcin_rr.close();
      fPathForResultsRootFiles = kFALSE;
    }
} // ----------- (end of GetPathForResultsRootFiles) --------------------

//================================================================================================

void TEcnaParPaths::GetPathForResultsAsciiFiles()
{
  GetPathForResultsAsciiFiles("");
}

void TEcnaParPaths::GetPathForResultsAsciiFiles(const TString argFileName)
{
  // Init fCfgResultsAsciiFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_results_ascii" (located in $HOME user's directory - default - )

  Int_t MaxCar = fgMaxCar;
  fCfgResultsAsciiFilePath.Resize(MaxCar);
  fCfgResultsAsciiFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsAsciiFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "ECNA/path_results_ascii";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      const Text_t *t_file_name = (const Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForResultsAsciiFilePath = s_path_name;
      fFileForResultsAsciiFilePath.Append('/');
      fFileForResultsAsciiFilePath.Append(t_file_name);
    }
  else
    {
      fFileForResultsAsciiFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForResultsAsciiFilePath.Data()
  //

  fFcin_ra.open(fFileForResultsAsciiFilePath.Data());
  if(fFcin_ra.fail() == kFALSE)
    {
      fFcin_ra.clear();
      string xResultsFileP;
      fFcin_ra >> xResultsFileP;
      fCfgResultsAsciiFilePath = xResultsFileP.c_str();
      fFcin_ra.close();
      fPathForResultsAsciiFiles = kTRUE;
    }
  else
    {
      fFcin_ra.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************************** " << endl;
      cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "     " << fFileForResultsAsciiFilePath.Data() << ": file not found. " << endl << endl
	   << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)" << endl
	   << "     and create a file named path_results_ascii in the subdirectory ECNA." << endl << endl
           << "     The file " << fFileForResultsAsciiFilePath.Data() << " is a configuration file" << endl
	   << "     for the CNA and must contain one line with the following syntax:" << endl << endl
	   << "        PATH_OF_THE_RESULTS_ASCII_FILE ($HOME/etc...) (without slash at the end of line)" << endl
	   << "                                                       ================================"
	   << endl << endl
	   << "     Example: $HOME/scratch0/cna/results_ascii" << endl << endl
	   << " ***************************************************************************************** "
	   << fTTBELL << endl;

      fFcin_ra.close();
      fPathForResultsAsciiFiles = kFALSE;
    }
} // ----------- (end of GetPathForResultsAsciiFiles) --------------------

//================================================================================================

void TEcnaParPaths::GetPathForHistoryRunListFiles()
{
  GetPathForHistoryRunListFiles("");
}

void TEcnaParPaths::GetPathForHistoryRunListFiles(const TString argFileName)
{
  // Init fCfgHistoryRunListFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/path_runlist_history_plots" (located in $HOME user's directory - default - )

  Int_t MaxCar = fgMaxCar;
  fCfgHistoryRunListFilePath.Resize(MaxCar);
  fCfgHistoryRunListFilePath          = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForHistoryRunListFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "ECNA/path_runlist_history_plots";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      const Text_t *t_file_name = (const Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForHistoryRunListFilePath = s_path_name;
      fFileForHistoryRunListFilePath.Append('/');
      fFileForHistoryRunListFilePath.Append(t_file_name);
    }
  else
    {
      fFileForHistoryRunListFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForHistoryRunListFilePath.Data()
  //

  fFcin_lor.open(fFileForHistoryRunListFilePath.Data());
  if(fFcin_lor.fail() == kFALSE)
    {
      fFcin_lor.clear();
      string xHistoryRunListP;
      fFcin_lor >> xHistoryRunListP;
      fCfgHistoryRunListFilePath = xHistoryRunListP.c_str();
      fFcin_lor.close();
      fPathForHistoryRunListFiles = kTRUE;
    }
  else
    {
      fFcin_lor.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ******************************************************************************************************** " << endl;
      cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "     " << fFileForHistoryRunListFilePath.Data() << ": file not found. " << endl << endl
	   << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)" << endl
	   << "     and create a file named path_runlist_history_plots in the subdirectory ECNA." << endl << endl
           << "     The file " << fFileForHistoryRunListFilePath.Data() << " is a configuration file" << endl
	   << "     for the CNA and must contain one line with the following syntax:" << endl << endl
	   << "        PATH_OF_THE_LIST_OF_RUNS_FOR_HISTORY_PLOTS_FILE ($HOME/etc...) (without slash at the end of line)" << endl
	   << "                                                                        ================================"
	   << endl << endl
	   << "     Example: $HOME/scratch0/cna/runlist_history_plots" << endl << endl
	   << " ******************************************************************************************************** "
	   << fTTBELL << endl;

      fFcin_lor.close();
      fPathForHistoryRunListFiles = kFALSE;
    }
} // ----------- (end of GetPathForHistoryRunListFiles) --------------------

//================================================================================================

#define ANAP
#ifndef ANAP

void TEcnaParPaths::GetPathForAnalyzerParametersFiles()
{
  GetPathForAnalyzerParametersFiles("");
}

void TEcnaParPaths::GetPathForAnalyzerParametersFiles(const TString argFileName)
{
  // Init fCfgAnalyzerParametersFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "path_analyzer_parameters.ecna" and file located
  // in $HOME user's directory (default)

  Int_t MaxCar = fgMaxCar;
  fCfgAnalyzerParametersFilePath.Resize(MaxCar);
  fCfgAnalyzerParametersFilePath          = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForAnalyzerParametersFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "path_analyzer_parameters.ecna";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      const Text_t *t_file_name = (const Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForAnalyzerParametersFilePath = s_path_name;
      fFileForAnalyzerParametersFilePath.Append('/');
      fFileForAnalyzerParametersFilePath.Append(t_file_name);
    }
  else
    {
      fFileForAnalyzerParametersFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForAnalyzerParametersFilePath.Data()
  //

  fFcin_anapar.open(fFileForAnalyzerParametersFilePath.Data());
  if(fFcin_anapar.fail() == kFALSE)
    {
      fFcin_anapar.clear();
      string xAnalyzerParametersP;
      fFcin_anapar >> xAnalyzerParametersP;
      fCfgAnalyzerParametersFilePath = xAnalyzerParametersP.c_str();
      fFcin_anapar.close();
    }
  else
    {
      fFcin_anapar.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************** " << endl;
      cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "    "
	   << fFileForAnalyzerParametersFilePath.Data() << " : file not found. " << endl
	   << "    "
	   << " Please create this file in your HOME directory and then restart."
	   << endl << endl
	   << "    "
           << " The file " << fFileForAnalyzerParametersFilePath.Data()
	   << " is a configuration file for the CNA and"
	   << " must contain one line with the following syntax:" << endl << endl
	   << "    "
	   << "   path of the analyzer parameters files ($HOME/etc...) " << endl
	   << "    "
	   << "          (without slash at the end of line)" << endl
	   << endl << endl
	   << "    "
	   << " EXAMPLE:" << endl << endl
	   << "    "
	   << "  $HOME/scratch0/cna/analyzer_parameters" << endl << endl
	   << " ***************************************************************************** "
	   << fTTBELL << endl;

      fFcin_anapar.close();
    }
} // ----------- (end of GetPathForAnalyzerParametersFiles) --------------------
#endif // ANAP

//================================================================================================
void TEcnaParPaths::GetCMSSWParameters()
{
  GetCMSSWParameters("");
}

void TEcnaParPaths::GetCMSSWParameters(const TString argFileName)
{
  // Init fCfgCMSSWVersion, fCfgCMSSWSubsystem and fCfgCMSSWSlc
  // and get them from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "ECNA/cmssw_parameters" (located in $HOME user's directory - default - )

  Int_t MaxCar = fgMaxCar;
  fCfgCMSSWVersion.Resize(MaxCar);
  fCfgCMSSWVersion = "?";
  fCfgCMSSWSubsystem.Resize(MaxCar);
  fCfgCMSSWSubsystem = "?";
  fCfgCMSSWSlc.Resize(MaxCar);
  fCfgCMSSWSlc = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCMSSWParameters and in class attribute fFileForCMSSWParameters

  if ( argFileName == "" )
    {
      string cFileNameForCMSSWParameters = "ECNA/cmssw_parameters";     // config file name
      TString s_file_name = cFileNameForCMSSWParameters.c_str();
      const Text_t *t_file_name = (const Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForCMSSWParameters = s_path_name;
      fFileForCMSSWParameters.Append('/');
      fFileForCMSSWParameters.Append(t_file_name);
    }
  else
    {
      fFileForCMSSWParameters = argFileName.Data();
    }

  //... Reading of the CMSSW version, subsystem and slc name in the file named fFileForCMSSWParameters.Data()
  //

  fFcin_cmssw.open(fFileForCMSSWParameters.Data());
  if(fFcin_cmssw.fail() == kFALSE)
    {
      fFcin_cmssw.clear();

      string xCMSSWVersionFileP;
      fFcin_cmssw >> xCMSSWVersionFileP;
      fCfgCMSSWVersion = xCMSSWVersionFileP.c_str();

      string xCMSSWSubsystemFileP;
      fFcin_cmssw >> xCMSSWSubsystemFileP;
      fCfgCMSSWSubsystem = xCMSSWSubsystemFileP.c_str();

      string xCMSSWSlcFileP;
      fFcin_cmssw >> xCMSSWSlcFileP;
      fCfgCMSSWSlc = xCMSSWSlcFileP.c_str();

      fFcin_cmssw.close();
    }
  else
    {
      fFcin_cmssw.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ************************************************************************************************** " << endl;
      cout << "   !CNA(TEcnaParPaths) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "     " << fFileForCMSSWParameters.Data() << ": file not found. " << endl << endl
	   << "     Please create a subdirectory named ECNA in your HOME directory (if not already done)" << endl
	   << "     and create a file named cmssw_parameters in the subdirectory ECNA." << endl << endl
           << "     The file " << fFileForCMSSWParameters.Data() << " is a configuration file" << endl
	   << "     for the CNA and must contain one line with the following syntax:" << endl << endl
	   << "        CMSSW_VERSION SUBSYSTEM_NAME TEST_SUBDIRECTORY_NAME" << endl
	   << endl << endl
	   << "     Example: CMSSW_2_1_19 CalibCalorimetry slc4_ia32_gcc345" << endl << endl
	   << " ************************************************************************************************** "
	   << fTTBELL << endl;

      fFcin_cmssw.close();
    }
} // ----------- (end of GetCMSSWParameters) --------------------

//=======================================================================================
//
//   P U B L I C    M E T H O D S    T O    R E T U R N    T H E    P A R A M E T E R S 
//
//=======================================================================================
TString TEcnaParPaths::ResultsRootFilePath()       {return fCfgResultsRootFilePath;}
TString TEcnaParPaths::ResultsAsciiFilePath()      {return fCfgResultsAsciiFilePath;}
TString TEcnaParPaths::HistoryRunListFilePath()    {return fCfgHistoryRunListFilePath;}
TString TEcnaParPaths::CMSSWVersion()              {return fCfgCMSSWVersion;}
TString TEcnaParPaths::CMSSWSubsystem()            {return fCfgCMSSWSubsystem;}
TString TEcnaParPaths::CMSSWSlc()                  {return fCfgCMSSWSlc;}

//.....................................................................................
TString TEcnaParPaths::PathModulesData()
{
  // ----- return the path of data subdirectory in package "Modules"
  TString ModulesdataPath = "";
  //...... get HOME directory path, CMSSW version and Subsystem name   
  TString s_path_name = gSystem->Getenv("HOME");
  const Text_t *t_path_name = (const Text_t *)s_path_name.Data();
  ModulesdataPath.Append(t_path_name);
  ModulesdataPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/

  TString s_cmssw = "cmssw";  
  const Text_t *t_cmssw = (const Text_t *)s_cmssw.Data();
  ModulesdataPath.Append(t_cmssw);  
  ModulesdataPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/
      
  const Text_t *t_cmssw_version = (const Text_t *)CMSSWVersion().Data();
  ModulesdataPath.Append(t_cmssw_version);
  ModulesdataPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/

  TString s_src = "src";
  const Text_t *t_src = (const Text_t *)s_src.Data();
  ModulesdataPath.Append(t_src);
  ModulesdataPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src
      
  const Text_t *t_cmssw_subsystem = (const Text_t *)CMSSWSubsystem().Data();
  ModulesdataPath.Append(t_cmssw_subsystem);
  ModulesdataPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/

  TString s_package_data_path = "EcalCorrelatedNoiseAnalysisModules/data";
  const Text_t *t_package_data_path = (const Text_t *)s_package_data_path.Data();
  ModulesdataPath.Append(t_package_data_path);
  ModulesdataPath.Append('/');
  // /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/src/CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/data/

  return ModulesdataPath;
}
//.....................................................................................
TString TEcnaParPaths::PathTestSlc()
{
  // ----- return the path of test/slc... subdirectory in CMSSW_X_Y_Z
  TString TestslcPath = "";
  //...... get HOME directory path, CMSSW version and Subsystem name   
  TString s_path_name = gSystem->Getenv("HOME");
  const Text_t *t_path_name = (const Text_t *)s_path_name.Data();
  TestslcPath.Append(t_path_name);
  TestslcPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/

  TString s_cmssw = "cmssw";  
  const Text_t *t_cmssw = (const Text_t *)s_cmssw.Data();
  TestslcPath.Append(t_cmssw);  
  TestslcPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/
      
  const Text_t *t_cmssw_version = (const Text_t *)CMSSWVersion().Data();
  TestslcPath.Append(t_cmssw_version);
  TestslcPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/

  TString s_test = "test";
  const Text_t *t_test = (const Text_t *)s_test.Data();
  TestslcPath.Append(t_test);
  TestslcPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test
      
  const Text_t *t_cmssw_subsystem = (const Text_t *)CMSSWSlc().Data();
  TestslcPath.Append(t_cmssw_subsystem);
  TestslcPath.Append('/');               //  /afs/cern.ch/user/U/USERNAME/cmssw/CMSSW_X_Y_Z/test/slc4_ia32_gcc345/

  return TestslcPath;
}

//============================================================================= Ancillary methods

void  TEcnaParPaths::SetResultsRootFilePath(const TString ImposedPath) {fCfgResultsRootFilePath  = ImposedPath;}
void  TEcnaParPaths::SetResultsAsciiFilePath(const TString ImposedPath){fCfgResultsAsciiFilePath = ImposedPath;}

void  TEcnaParPaths::TruncateResultsRootFilePath(const Int_t& n1, const Int_t& nbcar) 
{fCfgResultsRootFilePath.Remove(n1,nbcar);}

void  TEcnaParPaths::TruncateResultsAsciiFilePath(const Int_t& n1, const Int_t& nbcar) 
{fCfgResultsAsciiFilePath.Remove(n1,nbcar);}

TString TEcnaParPaths::BeginningOfResultsRootFilePath()
{TString sBegin = "?";
 if( fCfgResultsRootFilePath.BeginsWith("$HOME") ){sBegin = "$HOME";}
 return sBegin;}

TString TEcnaParPaths::BeginningOfResultsAsciiFilePath()
{TString sBegin = "?";
 if( fCfgResultsAsciiFilePath.BeginsWith("$HOME") ){sBegin = "$HOME";}
 return sBegin;}

void TEcnaParPaths::AppendResultsRootFilePath(const Text_t * t_file_nohome)
{fCfgResultsRootFilePath.Append(t_file_nohome);}

void TEcnaParPaths::AppendResultsAsciiFilePath(const Text_t * t_file_nohome)
{fCfgResultsAsciiFilePath.Append(t_file_nohome);}
