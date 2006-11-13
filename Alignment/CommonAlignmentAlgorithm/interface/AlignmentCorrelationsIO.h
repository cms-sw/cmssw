#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIO_h

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include <vector>
#include <map>

/// Abstract base class for IO of Correlations 

class AlignmentCorrelationsIO 
{

  protected:

  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;
  typedef std::vector<Alignable*> Alignables;

  /// destructor 
  virtual ~AlignmentCorrelationsIO(){}; 

  /// open IO 
  virtual int open(const char* filename, int iteration, bool writemode) = 0;

  /// close IO 
  virtual int close(void) = 0;

  /// write correlations 
  virtual int write(const Correlations& cor, bool validCheck) = 0;

  /// read correlations 
  virtual Correlations read(const std::vector<Alignable*>& alivec, int& ierr) = 0;

};

#endif
