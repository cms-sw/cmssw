#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h


#include "TTree.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIO.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"


/// Concrete class for ROOT-based I/O of AlignmentParameters 

class AlignmentParametersIORoot : public AlignmentIORootBase,
								  public AlignmentParametersIO
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
  int open(char* filename, int iteration, bool writemode)
    {return openRoot(filename,iteration,writemode);};

  /// Close IO 
  int close(void) {return closeRoot();};

  // helper functions

  /// Find entry number corresponding to Id. Returns -1 on failure.
  int findEntry(unsigned int detId,int comp);

  /// Create all branches and give names
  void createBranches(void);

  /// Set branch adresses
  void setBranchAddresses(void);

  // Alignment parameter tree 
  int theObjId, theCovRang, theCovarRang;
  unsigned int theId;
  double thePar[nParMax],theCov[nParMax*(nParMax+1)/2];

};

#endif
