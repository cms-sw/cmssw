#ifndef ZTR_TEcnaParPaths
#define ZTR_TEcnaParPaths

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

///-----------------------------------------------------------
///   TEcnaParPaths.h
///   Update: 05/10/2012
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_main_page.htm
///-----------------------------------------------------------

class TEcnaParPaths : public TObject {
private:
  //..... Attributes

  Int_t fCnew, fCdelete;
  Int_t fCnewRoot, fCdeleteRoot;

  Int_t fgMaxCar;  // Max nb of caracters for char*

  TString fTTBELL;

  Int_t fCnaCommand, fCnaError;

  std::ifstream fFcin_rr;   // stream for results root files
  std::ifstream fFcin_ra;   // stream for results ascii files
  std::ifstream fFcin_lor;  // stream for list of runs files
  //  std::ifstream fFcin_anapar;      // stream for EcnaAnalyzer parameters files
  std::ifstream fFcin_cmssw;  // stream for cmssw version and subsystem

  TString fCfgResultsRootFilePath;      // absolute path for the results .root files (/afs/etc...)
  TString fFileForResultsRootFilePath;  // name of the file containing the results .root  file path

  TString fCfgResultsAsciiFilePath;      // absolute path for the results .ascii files (/afs/etc...)
  TString fFileForResultsAsciiFilePath;  // name of the file containing the results .ascii file path

  TString fCfgHistoryRunListFilePath;      // absolute path for the list-of-runs .ascii files (/afs/etc...)
  TString fFileForHistoryRunListFilePath;  // name of the file containing the list-of-run file path

  TString fCfgCMSSWBase;            // CMSSW base user's directory
  TString fCfgCMSSWSubsystem;       // CMSSW subsystem name
  TString fCfgSCRAMArch;            // SCRAM ARCHITECTURE
  TString fFileForCMSSWParameters;  // name of the file containing the CMSSW version, subsystem and slc

public:
  //..... Methods

  TEcnaParPaths();
  TEcnaParPaths(TEcnaObject *);
  ~TEcnaParPaths() override;

  void Init();

  Bool_t GetPathForResultsRootFiles();
  Bool_t GetPathForResultsAsciiFiles();
  Bool_t GetPathForHistoryRunListFiles();
  void GetCMSSWParameters();

  Bool_t GetPathForResultsRootFiles(const TString &);
  Bool_t GetPathForResultsAsciiFiles(const TString &);
  Bool_t GetPathForHistoryRunListFiles(const TString &);
  //  Bool_t GetCMSSWParameters(const TString&);

  Bool_t GetPaths();

  const TString &ResultsRootFilePath() const;
  const TString &ResultsAsciiFilePath() const;
  const TString &HistoryRunListFilePath() const;
  const TString &CMSSWBase() const;
  const TString &CMSSWSubsystem() const;
  const TString &SCRAMArch() const;

  void SetResultsRootFilePath(const TString &);
  void SetResultsAsciiFilePath(const TString &);
  void SetHistoryRunListFilePath(const TString &);

  void TruncateResultsRootFilePath(const Int_t &, const Int_t &);
  void TruncateResultsAsciiFilePath(const Int_t &, const Int_t &);

  TString BeginningOfResultsRootFilePath();
  TString BeginningOfResultsAsciiFilePath();

  void AppendResultsRootFilePath(const Text_t *);
  void AppendResultsAsciiFilePath(const Text_t *);

  TString PathModulesData();
  TString PathTestScramArch();

  ClassDefOverride(TEcnaParPaths, 1)  // Parameter management for ECNA (Ecal Correlated Noises Analysis)
};

#endif  //    ZTR_TEcnaParPaths
