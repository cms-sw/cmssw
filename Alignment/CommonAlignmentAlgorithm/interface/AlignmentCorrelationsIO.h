#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCorrelationsIO_h

#include "Alignment/CommonAlignment/interface/Utilities.h"

/// Abstract base class for IO of Correlations

class AlignmentCorrelationsIO {
protected:
  /// destructor
  virtual ~AlignmentCorrelationsIO() {}

  /// open IO
  virtual int open(const char* filename, int iteration, bool writemode) = 0;

  /// close IO
  virtual int close(void) = 0;

  /// write correlations
  virtual int write(const align::Correlations& cor, bool validCheck) = 0;

  /// read correlations
  virtual align::Correlations read(const align::Alignables& alivec, int& ierr) = 0;
};

#endif
