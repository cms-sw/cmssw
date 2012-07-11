#ifndef ZTR_TEcnaParPaths
#define ZTR_TEcnaParPaths

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

//------------------------ TEcnaParPaths.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TEcnaParPaths : public TObject {

 private:

  //..... Attributes

  Int_t   fgMaxCar;   // Max nb of caracters for char*

  Int_t   fCnew,        fCdelete;
  Int_t   fCnewRoot,    fCdeleteRoot;

  TString fTTBELL;

  Int_t   fCnaCommand,  fCnaError;


  ifstream fFcin_rr;          // stream for results root files 
  ifstream fFcin_ra;          // stream for results ascii files 
  ifstream fFcin_lor;         // stream for list of runs files
  //  ifstream fFcin_anapar;      // stream for EcnaAnalyzer parameters files
  ifstream fFcin_cmssw;       // stream for cmssw version and subsystem 

  TString fCfgResultsRootFilePath;     // absolute path for the results .root files (/afs/etc...)
  TString fFileForResultsRootFilePath; // name of the file containing the results .root  file path

  TString fCfgResultsAsciiFilePath;     // absolute path for the results .ascii files (/afs/etc...)
  TString fFileForResultsAsciiFilePath; // name of the file containing the results .ascii file path

  TString fCfgHistoryRunListFilePath;       // absolute path for the list-of-runs .ascii files (/afs/etc...)
  TString fFileForHistoryRunListFilePath;   // name of the file containing the list-of-run file path

  //  TString fCfgAnalyzerParametersFilePath;       // absolute path for the analyzer parameters files (/afs/etc...)
  //  TString fFileForAnalyzerParametersFilePath;   // name of the file containing the user's parameters file path

  TString fCfgCMSSWVersion;        // CMSSW version 
  TString fCfgCMSSWSubsystem;      // CMSSW subsystem name
  TString fCfgCMSSWSlc;            // CMSSW slc... name
  TString fFileForCMSSWParameters; // name of the file containing the CMSSW version, subsystem and slc



 public:

  Bool_t fPathForResultsRootFiles;
  Bool_t fPathForResultsAsciiFiles;
  Bool_t fPathForHistoryRunListFiles;

  //..... Methods

           TEcnaParPaths();
  virtual  ~TEcnaParPaths();

  void     Init();

  void GetPathForResultsRootFiles();
  void GetPathForResultsAsciiFiles();
  void GetPathForHistoryRunListFiles();
  //  void GetPathForAnalyzerParametersFiles();
  void GetCMSSWParameters();

  void GetPathForResultsRootFiles(const TString);
  void GetPathForResultsAsciiFiles(const TString);
  void GetPathForHistoryRunListFiles(const TString);
  //  void GetPathForAnalyzerParametersFiles(const TString);
  void GetCMSSWParameters(const TString);

  TString ResultsRootFilePath();
  TString ResultsAsciiFilePath();
  TString HistoryRunListFilePath();
  //TString AnalyzerParametersFilePath();
  TString CMSSWVersion();
  TString CMSSWSubsystem();
  TString CMSSWSlc();

  void SetResultsRootFilePath(const TString);
  void SetResultsAsciiFilePath(const TString);

  void TruncateResultsRootFilePath(const Int_t&, const Int_t&);
  void TruncateResultsAsciiFilePath(const Int_t&, const Int_t&);

  TString BeginningOfResultsRootFilePath();
  TString BeginningOfResultsAsciiFilePath();

  void AppendResultsRootFilePath(const Text_t *);
  void AppendResultsAsciiFilePath(const Text_t *);

  TString PathModulesData();
  TString PathTestSlc();


ClassDef(TEcnaParPaths,1)// Parameter management for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaParameter
