//----------Author's Names: FX Gentit, B.Fabbro  DAPNIA/SPP CEN Saclay
//----------Copyright:Those valid for CEA sofware
//----------Modified:07/06/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRootFile.h"

//#include "Riostream.h"

TCnaRootFile *gCnaRootFile = 0;

ClassImp(TCnaRootFile)
//___________________________________________________________________________
//
//  Reading of the ROOT file written by TCnaRunEB
//
TCnaRootFile::TCnaRootFile(const Text_t *name) {
//constructor
  Init();
  fRootFileName = name;
}

TCnaRootFile::TCnaRootFile(const Text_t *name, TString status) {
//constructor
  Init();
  fRootFileName   = name;
  fRootFileStatus = status;
}

TCnaRootFile::~TCnaRootFile() {
//destructor
}

void TCnaRootFile::Init() {
//Set default values in all variables
  fRootFileName           = "";
  fRootFileStatus         = "";
  fRootFile               = 0;
  fCounterBytesCnaResults = 0;
  fNbEntries              = 0;
  fCnaResultsTree         = 0;
  fCnaResultsBranch       = 0;
  fCnaIndivResult         = 0;
}

void TCnaRootFile::CloseFile() {
//Close the CNA root file for reading
  fRootFile->Close();
  delete fRootFile;
  fRootFile               = 0;
  fCounterBytesCnaResults = 0;
  fCnaResultsTree         = 0;
  fCnaResultsBranch       = 0;
}

Bool_t TCnaRootFile::OpenR(Text_t *name) {
//Open the CNA root file for reading
  Bool_t ok;
  if (name != "") fRootFileName = name;
  fRootFile = new TFile(fRootFileName.Data(),"READ");
  ok = fRootFile->IsOpen();
  if (ok) {
    fCnaResultsTree = (TTree *)fRootFile->Get("CNAResults");
    //cout << "*TCnaRootFile::OpenR(..)> fCnaResultsTree : " << fCnaResultsTree << endl;
    if (fCnaResultsTree) {
      fCnaIndivResult = new TCnaResultType();
      //cout << "*TCnaRootFile::OpenR(..)> fCnaIndivResult : " << fCnaIndivResult << endl;
      fCnaResultsBranch = fCnaResultsTree->GetBranch("Results");
      fCnaResultsBranch->SetAddress(&fCnaIndivResult);
      fNbEntries = (Int_t) fCnaResultsTree->GetEntries();
      //cout << "*TCnaRootFile::OpenR(..)> fNbEntries : " << fNbEntries << endl;
    }
    else ok = kFALSE;
  }
  return ok;
}

Bool_t TCnaRootFile::OpenW(Text_t *name) {
//Open root file for writing
  Bool_t ok = kTRUE;
  if (name != "") fRootFileName = name;

  //cout << "*TCnaRootFile::OpenW(...)> fRootFileName.Data() = "
  //     << fRootFileName.Data() << endl;

  fRootFile = new TFile(fRootFileName.Data(),"RECREATE");
  if (fRootFile) {
    fCnaResultsTree = new TTree("CNAResults","CNAResults");
    fCnaIndivResult = new TCnaResultType();
    fCnaResultsBranch = fCnaResultsTree->
      Branch("Results","TCnaResultType", &fCnaIndivResult, 64000, 0);
  }
  else ok = kFALSE;
  return ok;
}

Bool_t TCnaRootFile::ReadElement(Int_t i) {
//Read element i
  Bool_t ok = kTRUE;
  fCounterBytesCnaResults += fCnaResultsTree->GetEntry(i);
  return ok;
}

Bool_t TCnaRootFile::ReadElement(CnaResultTyp typ, Int_t k) {
//Look for kth element of type typ
  Bool_t ok = kFALSE;
  Int_t i = 0;
  do {
    // cout << "*TCnaRootFile::ReadElement(typ,k)> fIthElement     = "
    // << fCnaIndivResult->fIthElement << endl;
    // cout << "*TCnaRootFile::ReadElement(typ,k)> fTypOfCnaResult = "
    // << fCnaIndivResult->fTypOfCnaResult << endl;

    fCounterBytesCnaResults += fCnaResultsTree->GetEntry(i);
    ok = ( ( fCnaIndivResult->fIthElement == k ) &&
	   ( fCnaIndivResult->fTypOfCnaResult == typ ));
    i++;
  } while ((i<fNbEntries) && (!ok));
  return ok;
}
