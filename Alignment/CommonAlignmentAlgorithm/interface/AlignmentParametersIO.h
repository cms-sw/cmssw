#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIO_h


#include<vector>

/// \class AlignmentParametersIO
///
/// abstract base class for I/O of AlignmentParameters 
///
///  $Date: 2006/10/19 14:20:59 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

class Alignable;
class AlignmentParameters;

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
  int write(const std::vector<Alignable*>& alivec, bool validCheck);

  /// write original RigidBodyAlignmentParameters of many Alignables 
  int writeOrigRigidBody(const std::vector<Alignable*>& alivec, bool validCheck);

  /// read AlignmentParameters of many Alignables 
  std::vector<AlignmentParameters*> read(const std::vector<Alignable*>& alivec, 
    int& ierr);

};

#endif
