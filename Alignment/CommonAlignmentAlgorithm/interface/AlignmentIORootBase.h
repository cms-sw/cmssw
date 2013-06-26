#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentIORootBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentIORootBase_h

#include "TString.h"

class TFile;
class TTree;

/// Base class for ROOT-based I/O of Alignment parameters etc. 

class AlignmentIORootBase 
{

  protected:
  /// constructor
  AlignmentIORootBase() : tree(0), myFile(0) {}
  /// destructor 
  virtual ~AlignmentIORootBase();

  /// open IO 
  int openRoot(const char* filename, int iteration, bool writemode);

  /// close IO 
  int closeRoot(void);

  /// create root branches 
  virtual void createBranches(void) = 0;

  /// set root branches 
  virtual void setBranchAddresses(void) = 0;

  /// test if file is existing and if so, what the highest iteration is 
  int testFile(const char* filename, const TString &tname);

  /// compose tree name 
  TString treeName(int iter, const TString &tname);

  // data members

  TTree* tree; // root tree
  TString treename; // tree identifier name
  TString treetxt;  // tree text
  bool bWrite; // if true we are writing, else reading

  const static int nParMax = 20;   // maximal number of Parameters
  const static int itermax = 1000; // max iteration to test for

 private:
  TFile* myFile; // root file
};

#endif
