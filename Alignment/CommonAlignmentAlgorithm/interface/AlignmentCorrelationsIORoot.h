#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIORoot_h

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsIO.h"

/// Concrete class for ROOT based IO of Correlations 

class AlignmentCorrelationsIORoot : public AlignmentIORootBase, public AlignmentCorrelationsIO
{
  friend class AlignmentIORoot;

  private:

  /// constructor 
  AlignmentCorrelationsIORoot();

  /// open IO 
  int open(const char* filename, int iteration, bool writemode) {
    return openRoot(filename,iteration,writemode);
  };

  /// close IO 
  int close(void){ return closeRoot(); };

  /// write correlations 
  int write(const align::Correlations& cor, bool validCheck);

  /// read correlations 
  align::Correlations read(const align::Alignables& alivec, int& ierr);

  void createBranches(void);
  void setBranchAddresses(void);

  // data members

  /// correlation tree 
  align::ID Ali1Id,Ali2Id;
  align::StructureType Ali1ObjId,Ali2ObjId;
  int corSize;
  double CorMatrix[nParMax*nParMax];

};

#endif
