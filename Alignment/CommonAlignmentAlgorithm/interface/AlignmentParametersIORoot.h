#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParametersIORoot_h

/// \class AlignmentParametersIORoot
///
/// Concrete class for ROOT-based I/O of AlignmentParameters
///
///  $Date: 2008/09/02 15:31:23 $
///  $Revision: 1.6 $
/// (last update by $Author: flucke $)

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIO.h"

class AlignmentParametersIORoot : public AlignmentIORootBase, public AlignmentParametersIO {
  friend class AlignmentIORoot;

private:
  /// Constructor
  AlignmentParametersIORoot();

  /// Write AlignmentParameters of one Alignable
  int writeOne(Alignable* ali) override;

  /// Read AlignmentParameters of one Alignable
  AlignmentParameters* readOne(Alignable* ali, int& ierr) override;

  /// Open IO
  int open(const char* filename, int iteration, bool writemode) override {
    return openRoot(filename, iteration, writemode);
  };

  /// Close IO
  int close(void) override;

  // helper functions

  /// Find entry number corresponding to ID and structure type.
  /// Returns -1 on failure.
  int findEntry(align::ID, align::StructureType);

  /// Create all branches and give names
  void createBranches(void) override;

  /// Set branch adresses
  void setBranchAddresses(void) override;

  // Alignment parameter tree
  int theCovRang, theCovarRang, theHieraLevel, theParamType;
  align::ID theId;
  align::StructureType theObjId;

  double thePar[nParMax], theCov[nParMax * (nParMax + 1) / 2];
};

#endif
