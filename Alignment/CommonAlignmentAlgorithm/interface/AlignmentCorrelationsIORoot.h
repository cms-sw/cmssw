#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIORoot_h

#include "TTree.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsIO.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// Concrete class for ROOT based IO of Correlations 

class AlignmentCorrelationsIORoot : public AlignmentIORootBase, public AlignmentCorrelationsIO
{
  friend class AlignmentIORoot;

  private:

  /// stores all alignment correlation informations 
  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;

  /// constructor 
  AlignmentCorrelationsIORoot();

  /// open IO 
  int open(const char* filename, int iteration, bool writemode) {
    return openRoot(filename,iteration,writemode);
  };

  /// close IO 
  int close(void){ return closeRoot(); };

  /// write correlations 
  int write(const Correlations& cor, bool validCheck);

  /// read correlations 
  Correlations read(const std::vector<Alignable*>& alivec, int& ierr);

  void createBranches(void);
  void setBranchAddresses(void);

  // data members

  /// correlation tree 
  unsigned int Ali1Id,Ali2Id;
  int Ali1ObjId,Ali2ObjId,corSize;
  double CorMatrix[nParMax*nParMax];

};

#endif
