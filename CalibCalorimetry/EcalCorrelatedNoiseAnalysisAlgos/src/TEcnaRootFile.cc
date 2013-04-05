//----------Author's Names: FX Gentit, B.Fabbro  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA sofware
//----------Modified:30/09/2011

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaRootFile.h"
#include "Riostream.h"

//--------------------------------------
//  TEcnaRootFile.cc
//  Class creation: 03 Dec 2002
//  Documentation: see TEcnaRootFile.h
//--------------------------------------

TEcnaRootFile *gCnaRootFile = 0;

ClassImp(TEcnaRootFile)
//___________________________________________________________________________
//
//  Reading of the ROOT file written by TEcnaRunEB
//
TEcnaRootFile::TEcnaRootFile() {
//constructor without arguments

  // cout << "[Info Management] CLASS: TEcnaRootFile.      CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaRootFile::TEcnaRootFile(TEcnaObject* pObjectManager, const Text_t *name, const TString& status) {
//constructor

 // cout << "[Info Management] CLASS: TEcnaRootFile.      CREATE OBJECT: this = " << this << endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaRootFile", i_this);

  fRootFileName   = name;
  fRootFileStatus = status;
}

TEcnaRootFile::TEcnaRootFile(TEcnaObject* pObjectManager, const Text_t *name) {
//constructor

 // cout << "[Info Management] CLASS: TEcnaRootFile.      CREATE OBJECT: this = " << this << endl;

  Init();
  Long_t i_this = (Long_t)this;
  pObjectManager->RegisterPointer("TEcnaRootFile", i_this);

  fRootFileName = name;
  fRootFileStatus = "READ";
}

TEcnaRootFile::TEcnaRootFile(const Text_t *name, const TString& status) {
//constructor

 // cout << "[Info Management] CLASS: TEcnaRootFile.      CREATE OBJECT: this = " << this << endl;

  Init();
  fRootFileName   = name;
  fRootFileStatus = status;
}

TEcnaRootFile::TEcnaRootFile(const Text_t *name) {
//constructor

 // cout << "[Info Management] CLASS: TEcnaRootFile.      CREATE OBJECT: this = " << this << endl;

  Init();
  fRootFileName = name;
  fRootFileStatus = "READ";
}

TEcnaRootFile::~TEcnaRootFile() {
//destructor

  //cout << "[Info Management] CLASS: TEcnaRootFile.      DESTROY OBJECT: this = " << this << endl;

  if( fCnaIndivResult != 0 ){delete fCnaIndivResult;}
}

void TEcnaRootFile::Init()
{
//Set default values in all variables

  TString sEmpty = "";
  fRootFileName   = sEmpty.Data();
  fRootFileStatus = sEmpty.Data();

  fRootFile               = 0;
  fCounterBytesCnaResults = 0;
  fNbEntries              = 0;
  fCnaResultsTree         = 0;
  fCnaResultsBranch       = 0;
  fCnaIndivResult         = 0;
}

void TEcnaRootFile::ReStart(const Text_t *name)
{
  // Set default values + fRootFileName + fRootFileStatus

  Init();
  fRootFileName   = name;
  fRootFileStatus = "READ";
}

void TEcnaRootFile::ReStart(const Text_t *name, const TString& status)
{
  // Set default values + fRootFileName + fRootFileStatus

  Init();
  fRootFileName   = name;
  fRootFileStatus = status;
}

void TEcnaRootFile::CloseFile() {
//Close the CNA root file for reading
  if( fRootFile != 0 )
    {
      fRootFile->Close();
      delete fRootFile; fRootFile = 0;
    }
  fCounterBytesCnaResults = 0;
  fCnaResultsTree         = 0;
  fCnaResultsBranch       = 0;
}

Bool_t TEcnaRootFile::OpenR(const Text_t *name) {
//Open the CNA root file for reading
  Bool_t ok;
  TString sEmpty = "";
  if( name != sEmpty ){fRootFileName = name;}

  if( fRootFile == 0 ){fRootFile = new TFile(fRootFileName.Data(),"READ");}

  ok = fRootFile->IsOpen();
  if (ok) {
    fCnaResultsTree = (TTree *)fRootFile->Get("CNAResults");
    if (fCnaResultsTree) {
      if( fCnaIndivResult == 0 ){fCnaIndivResult = new TEcnaResultType();}
      fCnaResultsBranch = fCnaResultsTree->GetBranch("Results");
      fCnaResultsBranch->SetAddress(&fCnaIndivResult);
      fNbEntries = (Int_t)fCnaResultsTree->GetEntries();
    }
    else ok = kFALSE;
  }
  return ok;
}

Bool_t TEcnaRootFile::OpenW(const Text_t *name) {
//Open root file for writing
  Bool_t ok = kTRUE;
  TString sEmpty = "";
  if( name != sEmpty ){fRootFileName = name;}

  if( fRootFile == 0 ){fRootFile = new TFile(fRootFileName.Data(),"RECREATE");}
  if (fRootFile) {
    fCnaResultsTree = new TTree("CNAResults","CNAResults");
    fCnaIndivResult = new TEcnaResultType();
    fCnaResultsBranch = fCnaResultsTree->
      Branch("Results","TEcnaResultType", &fCnaIndivResult, 64000, 0);
  }
  else ok = kFALSE;
  return ok;
}

Bool_t TEcnaRootFile::ReadElement(Int_t i) {
//Read element i
  Bool_t ok = kTRUE;
  fCounterBytesCnaResults += fCnaResultsTree->GetEntry(i);
  return ok;
}

Bool_t TEcnaRootFile::ReadElement(CnaResultTyp typ, Int_t k) {
//Look for kth element of type typ
  Bool_t ok = kFALSE;
  Int_t i = 0;
  do {
    fCounterBytesCnaResults += fCnaResultsTree->GetEntry(i);
    ok = ( ( fCnaIndivResult->fIthElement == k ) &&
	   ( fCnaIndivResult->fTypOfCnaResult == typ ));
    i++;
  } while ((i<fNbEntries) && (!ok));
  return ok;
}

Int_t TEcnaRootFile::ReadElementNextEntryNumber(CnaResultTyp typ, Int_t k) {
//Look for kth element of type typ and return the next entry number
  Bool_t ok = kFALSE;

  Int_t i = 0;
  do {
    fCounterBytesCnaResults += fCnaResultsTree->GetEntry(i);
    ok = ( ( fCnaIndivResult->fIthElement == k ) &&
	   ( fCnaIndivResult->fTypOfCnaResult == typ ));
    i++;
  } while ((i<fNbEntries) && (!ok));

  if( ok == kFALSE ){i = -1;}
  return i;
}
