#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentIO_h

/// \class AlignmentIO
///
/// Abstract base class for input/output of Alignment parameters,
/// Correlations, as well as absolute and relative coordinates of
/// Alignables
///
///  $Date: 2006/11/30 09:56:03 $
///  $Revision: 1.3 $
/// (last update by $Author: flucke $)


#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"


class AlignmentIO
{

  public:
  virtual ~AlignmentIO() = default;
  /// write AlignmentParameters 
  virtual void writeAlignmentParameters (const align::Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read AlignmentParameters 
  virtual align::Parameters readAlignmentParameters (const align::Alignables& alivec, 
    const char* filename, int iter, int& ierr) = 0;

  /// write RigidBodyAlignmentParameters as applied on top of original positions
  virtual void writeOrigRigidBodyAlignmentParameters (const align::Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// write Correlations 
  virtual void writeCorrelations (const align::Correlations& cormap, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Correlations 
  virtual align::Correlations readCorrelations (const align::Alignables& alivec, 
    const char* filename, int iter, int& ierr) = 0;

  /// write Alignable current absolute positions 
  virtual void writeAlignableAbsolutePositions (const align::Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable current absolute positions 
  virtual AlignablePositions readAlignableAbsolutePositions (const align::Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

  /// write Alignable original (before misalignment) absolute positions 
  virtual void writeAlignableOriginalPositions (const align::Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable original (before misalignment) absolute positions 
  virtual AlignablePositions readAlignableOriginalPositions (const align::Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

  /// write Alignable relative positions (shift,rotation) 
  virtual  void writeAlignableRelativePositions (const align::Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable relative positions (shift,rotation) 
  virtual AlignableShifts readAlignableRelativePositions (const align::Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

};

#endif
