// this class's header
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"

// ----------------------------------------------------------------------------
// open file/trees for write

int AlignmentIORootBase::openRoot(char* filename, int iteration, bool write)
{
  bWrite=write;
  int iter;

  edm::LogInfo("AlignmentIORootBase") << "File: " << filename ;

  if (bWrite) { // writing

    int iterfile = testFile(filename,treename);
    if (iterfile == -1) {
      iter=iteration;
	  edm::LogInfo("AlignmentIORootBase") << "Write to new file; first iteration: " << iter ;
      IORoot = new TFile(filename,"recreate");
    } else {
      if (iteration == -1) {
        iter=iterfile+1;
		edm::LogInfo("AlignmentIORootBase") 
		  << "Write to existing file; highest iteration: " << iter;
      }
      else {
        if (iteration<=iterfile) {
		  edm::LogError("AlignmentIORootBase") 
			<< "Iteration " << iteration 
			<<" invalid or already exists for tree " << treename;
		  return -1;
        }
        iter = iteration;
        edm::LogInfo("AlignmentIORootBase")  << "Write to new iteration: " << iter;
      }
      IORoot = new TFile(filename,"update");
	  
    }

    IORoot->cd();

    // create tree
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
			<<"Iteration " << iteration << " does not exist for tree " << treename;
		  return -1;
        }
        iter = iteration;
        edm::LogInfo("AlignmentIORootBase")  << "Read from specified iteration: " << iter;
      }
      IORoot = new TFile(filename,"update");
    }

    IORoot->cd();
    // set trees
    edm::LogInfo("AlignmentIORootBase") <<" Tree: " <<treeName(iter,treename);
    tree = (TTree*)IORoot->Get(treeName(iter,treename));
    if (tree==NULL) {
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
    IORoot->cd();
    tree->Write();
    tree->Delete();
    IORoot->Close();
  }
  else { // reading
    IORoot->cd();
    tree->Delete();
    IORoot->Close();
  }

  return 0;
}

// ----------------------------------------------------------------------------
// returns highest existing iteration in file
// if file does not exist: return -1

int AlignmentIORootBase::testFile(char* filename, TString tname)
{
  FILE* testFILE;
  testFILE = fopen(filename,"r");
  if (testFILE == NULL) {
    return -1;
  }
  else {
    fclose(testFILE);
    int ihighest=-2;
    TFile IORoot(filename,"update");
    for (int iter=0; iter<itermax; iter++) {
      if ((0 != (TTree*)IORoot.Get(treeName(iter,tname))) 
         && (iter>ihighest)) ihighest=iter; 
    }
    IORoot.Close();
    return ihighest;
  }
}

// ----------------------------------------------------------------------------
// create tree name from stub+iteration

TString AlignmentIORootBase::treeName(int iter,TString tname)
{
  char iterString[5];
  sprintf(iterString, "%i",iter);
  tname.Append(":");
  tname.Append(iterString);
  return tname;
}
