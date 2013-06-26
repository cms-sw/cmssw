#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h

/// \class AlignmentParametersIORoot
///
/// Concrete class for ROOT-based I/O of AlignmentParameters 
///
///  $Date: 2009/01/23 15:47:42 $
///  $Revision: 1.7 $
/// (last update by $Author: ewidl $)

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIO.h"

class AlignmentParametersIORoot : public AlignmentIORootBase, public AlignmentParametersIO
{
  friend class AlignmentIORoot;

  private:

  /// Constructor 
  AlignmentParametersIORoot(); 

  /// Write AlignmentParameters of one Alignable 
  int writeOne(Alignable* ali);

  /// Read AlignmentParameters of one Alignable 
  AlignmentParameters* readOne(Alignable* ali, int& ierr);

  /// Open IO 
  int open(const char* filename, int iteration, bool writemode)
    {return openRoot(filename,iteration,writemode);};

  /// Close IO 
  int close(void);

  // helper functions

  /// Find entry number corresponding to ID and structure type.
  /// Returns -1 on failure.
  int findEntry(align::ID, align::StructureType);

  /// Create all branches and give names
  void createBranches(void);

  /// Set branch adresses
  void setBranchAddresses(void);

  // Alignment parameter tree 
  int theCovRang, theCovarRang, theHieraLevel, theParamType;
  align::ID theId;
  align::StructureType theObjId;

  double thePar[nParMax],theCov[nParMax*(nParMax+1)/2];

};

#endif
