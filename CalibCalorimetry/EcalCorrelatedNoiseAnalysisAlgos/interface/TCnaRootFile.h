#ifndef CL_TCnaRootFile
#define CL_TCnaRootFile

#include "TObject.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaResultType.h"

class TCnaRootFile : public TObject {

protected:

  void      Init();

public:

  TString     fRootFileName;            // Treename of fRootFile
  TString     fRootFileStatus;          // Status of fRootFile
  TFile      *fRootFile;                // Root file for CNA
  Int_t       fCounterBytesCnaResults;  // Counter of bytes in fCnaResultsTree
  Int_t       fNbEntries;               // Nb of entries in fCnaResultsTree
  TTree      *fCnaResultsTree;    // Tree containing the individual results
                                  // individual result = equivalent
                                  // of one ASCII file

  TBranch         *fCnaResultsBranch;  // Branch of the individual results
  TCnaResultType  *fCnaIndivResult;    // One of the individual results
                                       // in the branch fCnaResultsBranch

  TCnaRootFile()                            { Init();             }
  TCnaRootFile(const Text_t*);
  TCnaRootFile(const Text_t*, TString);
  ~TCnaRootFile();
  void   CloseFile();
  Bool_t OpenR(const Text_t* = "");
  Bool_t OpenW(const Text_t* = "");
  Bool_t ReadElement(Int_t);
  Bool_t ReadElement(CnaResultTyp,Int_t);
  ClassDef(TCnaRootFile,1)  //Root file of CNA
};

R__EXTERN TCnaRootFile *gCnaRootFile;

#endif
