#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h

#include "Alignment/CommonAlignment/interface/Utilities.h"

/// \class AlignmentParametersIO
///
/// abstract base class for I/O of AlignmentParameters 
///
///  $Date: 2007/10/08 14:38:15 $
///  $Revision: 1.5 $
/// (last update by $Author: cklae $)

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

  /// write original RigidBodyAlignmentParameters (i.e. 3 shifts and 3 rotation)
  virtual int writeOneOrigRigidBody(Alignable* ali);

  /// read AlignmentParameters of one Alignable 
  virtual AlignmentParameters* readOne(Alignable* ali, int& ierr) = 0;

  /// write AlignmentParameters of many Alignables 
  int write(const align::Alignables& alivec, bool validCheck);

  /// write original RigidBodyAlignmentParameters of many Alignables 
  int writeOrigRigidBody(const align::Alignables& alivec, bool validCheck);

  /// read AlignmentParameters of many Alignables 
  align::Parameters read(const align::Alignables& alivec, int& ierr);

};

#endif
