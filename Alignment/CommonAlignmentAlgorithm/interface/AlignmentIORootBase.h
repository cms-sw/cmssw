#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentIORootBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentIORootBase_h

#include "TFile.h"
#include "TTree.h"
#include "TVector.h"
#include "TString.h"

#include <string>
#include <iostream>
#include <vector>
#include <map>

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// Base class for ROOT-based I/O of Alignment parameters etc. 

class AlignmentIORootBase 
{

  protected:

  /// destructor 
  virtual ~AlignmentIORootBase(){};

  /// open IO 
  int openRoot(char* filename, int iteration, bool writemode);

  /// close IO 
  int closeRoot(void);

  /// create root branches 
  virtual void createBranches(void) = 0;

  /// set root branches 
  virtual void setBranchAddresses(void) = 0;

  /// test if file is existing and if so, what the highest iteration is 
  int testFile(char* filename, TString tname);

  /// compose tree name 
  TString treeName(int iter,TString tname);

  // data members

  TFile* IORoot; // root file
  TTree* tree; // root tree
  TString treename; // tree identifier name
  TString treetxt;  // tree text
  bool bWrite; // if true we are writing, else reading

  const static int nParMax = 6;    // maximal number of Parameters
  const static int itermax = 1000; // max iteration to test for

};

#endif
