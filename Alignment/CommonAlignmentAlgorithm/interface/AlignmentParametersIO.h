#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include<vector>

/// abstract base class for I/O of AlignmentParameters 

class AlignmentParametersIO 
{

  protected:

  virtual  ~AlignmentParametersIO(){};

  /// open IO 
  virtual int open(const char* filename, int iteration, bool writemode) =0;

  /// close IO 
  virtual int close(void) =0;

  /// write AlignmentParameters of one Alignable 
  virtual int writeOne(Alignable* ali) = 0;

  /// read AlignmentParameters of one Alignable 
  virtual AlignmentParameters* readOne(Alignable* ali, int& ierr) = 0;

  /// write AlignmentParameters of many Alignables 
  int write(const std::vector<Alignable*>& alivec, bool validCheck);

  /// read AlignmentParameters of many Alignables 
  std::vector<AlignmentParameters*> read(const std::vector<Alignable*>& alivec, 
    int& ierr);

};

#endif
