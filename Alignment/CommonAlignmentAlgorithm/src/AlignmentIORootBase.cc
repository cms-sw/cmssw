// this class's header
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"

#include "TFile.h"
#include "TTree.h"

AlignmentIORootBase::~AlignmentIORootBase()
{
  delete myFile; // tree is deleted automatically with file
}

// ----------------------------------------------------------------------------
// open file/trees for write

int AlignmentIORootBase::openRoot(const char* filename, int iteration, bool write)
{
  bWrite=write;
  int iter;

  edm::LogInfo("AlignmentIORootBase") << "File: " << filename ;

  if (bWrite) { // writing

    int iterfile = testFile(filename,treename);
    if (iterfile == -1) {
      iter=iteration;
	  edm::LogInfo("AlignmentIORootBase") << "Write to new file; first iteration: " << iter ;
	  myFile = TFile::Open(filename,"recreate");
    } else {
      if (iteration == -1) {
        iter=iterfile+1;
		edm::LogInfo("AlignmentIORootBase") 
		  << "Write to existing file; highest iteration: " << iter;
      } else {
        if (iteration<=iterfile) {
          edm::LogError("AlignmentIORootBase") 
            << "Iteration " << iteration 
            <<" invalid or already exists for tree " << treename;
		  return -1;
        }
        iter = iteration;
        edm::LogInfo("AlignmentIORootBase")  << "Write to new iteration: " << iter;
      }
      myFile = TFile::Open(filename,"update");
	  
    }

    // create tree
    myFile->cd();
    edm::LogInfo("AlignmentIORootBase") << "Tree: " << treeName(iter,treename);
    tree = new TTree(treeName(iter,treename),treetxt);
    createBranches();

  } else { // reading

    int iterfile = testFile(filename,treename);
    if ( iterfile == -1 ) {
      edm::LogError("AlignmentIORootBase") << "File does not exist!";
      return -1;
    } else if ( iterfile == -2 ) {
      edm::LogError("AlignmentIORootBase") << "Tree " << treename 
                                           << " does not exist in file " << filename;
      return -1;
    } else {
      if (iteration == -1) {
        iter=iterfile;
        edm::LogInfo("AlignmentIORootBase") << "Read from highest iteration: " << iter;
      } else {
        if (iteration>iterfile) {
        edm::LogError("AlignmentIORootBase")
          << "Iteration " << iteration << " does not exist for tree " << treename;
	return -1;
        }
        iter = iteration;
        edm::LogInfo("AlignmentIORootBase")  << "Read from specified iteration: " << iter;
      }
      myFile = TFile::Open(filename, "read");
    }

    myFile->cd();
    // set trees
    edm::LogInfo("AlignmentIORootBase") <<" Tree: " <<treeName(iter,treename);
    tree = (TTree*)myFile->Get(treeName(iter,treename));
    if (tree==nullptr) {
      edm::LogError("AlignmentIORootBase") <<"Tree does not exist in file!";
      return -1;
    }
    setBranchAddresses();
  }

  return 0;
}

// ----------------------------------------------------------------------------
// write tree and close file

int AlignmentIORootBase::closeRoot(void)
{
  if (bWrite) { //writing
    tree->Write();
  }

  delete myFile;
  myFile = nullptr;
  tree = nullptr; // deleted with file

  return 0;
}

// ----------------------------------------------------------------------------
// returns highest existing iteration in file
// if file does not exist: return -1

int AlignmentIORootBase::testFile(const char* filename, const TString &tname)
{
  FILE* testFILE;
  testFILE = fopen(filename,"r");
  if (testFILE == nullptr) {
    return -1;
  } else {
    fclose(testFILE);
    int ihighest=-2;
    TFile *aFile = TFile::Open(filename,"read");
    for (int iter=0; iter<itermax; iter++) {
      if ((nullptr != (TTree*)aFile->Get(treeName(iter,tname))) 
	  && (iter>ihighest)) ihighest=iter; 
    }
    delete aFile;
    return ihighest;
  }
}

// ----------------------------------------------------------------------------
// create tree name from stub+iteration

TString AlignmentIORootBase::treeName(int iter, const TString &tname)
{
  return TString(tname + Form("_%i",iter));
}
