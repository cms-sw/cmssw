#ifndef Alignment_CommonAlignmentAlgorithm_AlignableDataIO_H
#define Alignment_CommonAlignmentAlgorithm_AlignableDataIO_H

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"

class Alignable;

/// Abstract base class for IO of alignable positions/shifts.
/// Derived concrete class must implement raw read/write methods

class AlignableDataIO {
protected:
  enum PosType { Abs, Org, Rel };

  /// Constructor
  AlignableDataIO(PosType p) : thePosType(p){};

  /// Destructor
  virtual ~AlignableDataIO(){};

  /// Open IO handle
  virtual int open(const char* filename, int iteration, bool writemode) = 0;

  /// Close IO handle
  virtual int close(void) = 0;

  /// Write absolute positions of one Alignable
  int writeAbsPos(Alignable* ali, bool validCheck);

  /// Write absolute positions of many Alignables
  int writeAbsPos(const align::Alignables& alivec, bool validCheck);

  /// Read absolute positions of one Alignable
  AlignableAbsData readAbsPos(Alignable* ali, int& ierr);

  /// Read absolute positions of many Alignables
  AlignablePositions readAbsPos(const align::Alignables& alivec, int& ierr);

  /// Write original positions of one Alignable
  int writeOrgPos(Alignable* ali, bool validCheck);

  /// Write original positions of many Alignables
  int writeOrgPos(const align::Alignables& alivec, bool validCheck);

  /// Read original positions of one Alignable
  AlignableAbsData readOrgPos(Alignable* ali, int& ierr);

  /// Read original positions of many Alignables
  AlignablePositions readOrgPos(const align::Alignables& alivec, int& ierr);

  /// Write relative positions of one Alignable
  int writeRelPos(Alignable* ali, bool validCheck);

  /// Write relative positions of many Alignables
  int writeRelPos(const align::Alignables& alivec, bool validCheck);

  /// Read relative positions of one Alignable
  AlignableRelData readRelPos(Alignable* ali, int& ierr);

  /// Read relative positions of many Alignables
  AlignableShifts readRelPos(const align::Alignables& alivec, int& ierr);

  // 'raw' read/write methods
  // must be provided by concrete derived class

  /// Write absolute positions
  virtual int writeAbsRaw(const AlignableAbsData& ad) = 0;
  /// Read absolute positions
  virtual AlignableAbsData readAbsRaw(Alignable* ali, int& ierr) = 0;
  /// Write relative positions
  virtual int writeRelRaw(const AlignableRelData& ad) = 0;
  /// Read relative positions
  virtual AlignableRelData readRelRaw(Alignable* ali, int& ierr) = 0;

  // Data members
  PosType thePosType;  // Defines whether absolute/orig/relative pos. are used
};

#endif
