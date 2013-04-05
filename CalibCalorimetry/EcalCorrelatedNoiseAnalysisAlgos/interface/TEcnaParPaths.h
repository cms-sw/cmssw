#ifndef ZTR_TEcnaParPaths
#define ZTR_TEcnaParPaths

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

///-----------------------------------------------------------
///   TEcnaParPaths.h
///   Update: 15/02/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------

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

  TString fCfgCMSSWBase;           // CMSSW base user's directory
  TString fCfgCMSSWSubsystem;      // CMSSW subsystem name
  TString fCfgSCRAMArch;           // SCRAM ARCHITECTURE
  TString fFileForCMSSWParameters; // name of the file containing the CMSSW version, subsystem and slc

 public:

  //..... Methods

           TEcnaParPaths();
           TEcnaParPaths(TEcnaObject*);
  virtual  ~TEcnaParPaths();

  void    Init();

  Bool_t GetPathForResultsRootFiles();
  Bool_t GetPathForResultsAsciiFiles();
  Bool_t GetPathForHistoryRunListFiles();
  void   GetCMSSWParameters();

  Bool_t GetPathForResultsRootFiles(const TString&);
  Bool_t GetPathForResultsAsciiFiles(const TString&);
  Bool_t GetPathForHistoryRunListFiles(const TString&);
  //  Bool_t GetCMSSWParameters(const TString&);

  Bool_t  GetPaths();

  TString ResultsRootFilePath();
  TString ResultsAsciiFilePath();
  TString HistoryRunListFilePath();
  TString CMSSWBase();
  TString CMSSWSubsystem();
  TString SCRAMArch();

  void SetResultsRootFilePath(const TString&);
  void SetResultsAsciiFilePath(const TString&);
  void SetHistoryRunListFilePath(const TString&);

  void TruncateResultsRootFilePath(const Int_t&, const Int_t&);
  void TruncateResultsAsciiFilePath(const Int_t&, const Int_t&);

  TString BeginningOfResultsRootFilePath();
  TString BeginningOfResultsAsciiFilePath();

  void AppendResultsRootFilePath(const Text_t *);
  void AppendResultsAsciiFilePath(const Text_t *);

  TString PathModulesData();
  TString PathTestScramArch();


ClassDef(TEcnaParPaths,1)// Parameter management for ECNA (Ecal Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaParPaths
