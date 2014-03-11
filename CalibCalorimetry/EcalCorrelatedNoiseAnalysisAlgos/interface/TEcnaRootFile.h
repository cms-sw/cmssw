#ifndef CL_TEcnaRootFile
#define CL_TEcnaRootFile

#include "TObject.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaResultType.h"

///-----------------------------------------------------------
///   TEcnaRootFile.h
///   Update: 30/09/2011
///   Authors:   FX Gentit, B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///

class TEcnaRootFile : public TObject {

protected:

  void      Init();

public:

  TString     fRootFileName;            // Treename of fRootFile
  TString     fRootFileStatus;          // Status of fRootFile

  TFile      *fRootFile;                // Root file for ECNA

  Int_t       fCounterBytesCnaResults;  // Counter of bytes in fCnaResultsTree
  Int_t       fNbEntries;               // Nb of entries in fCnaResultsTree
  TTree      *fCnaResultsTree;    // Tree containing the individual results
                                  // individual result = equivalent
                                  // of one ASCII file

  TBranch         *fCnaResultsBranch;  // Branch of the individual results
  TEcnaResultType *fCnaIndivResult;    // One of the individual results
                                       // in the branch fCnaResultsBranch

  TEcnaRootFile();
  TEcnaRootFile(TEcnaObject*, const Text_t*, const TString&);
  TEcnaRootFile(TEcnaObject*, const Text_t*);

  TEcnaRootFile(const Text_t*);
  TEcnaRootFile(const Text_t*, const TString&);

  ~TEcnaRootFile();

  void   ReStart(const Text_t*);
  void   ReStart(const Text_t*, const TString&);
  void   CloseFile();
  Bool_t OpenR(const Text_t* = "");
  Bool_t OpenW(const Text_t* = "");
  Bool_t ReadElement(Int_t);
  Bool_t ReadElement(CnaResultTyp,Int_t);
  Int_t  ReadElementNextEntryNumber(CnaResultTyp,Int_t);
  ClassDef(TEcnaRootFile,1)  //Root file of CNA
};

R__EXTERN TEcnaRootFile *gCnaRootFile;

#endif
