#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentIO_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentIO_h

/// \class AlignmentIO
///
/// Abstract base class for input/output of Alignment parameters,
/// Correlations, as well as absolute and relative coordinates of
/// Alignables
///
///  $Date: 2006/10/19 14:20:59 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

#include <map>

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"

class Alignable;
class AlignmentParameters;

class AlignmentIO
{

  public:

  typedef std::vector<AlignmentParameters*> Parameters;
  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;
  typedef std::vector<Alignable*> Alignables;
  typedef std::vector<AlignableAbsData> AlignablePositions;
  typedef std::vector<AlignableRelData> AlignableShifts;

  /// write AlignmentParameters 
  virtual void writeAlignmentParameters (const Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read AlignmentParameters 
  virtual Parameters readAlignmentParameters (const Alignables& alivec, 
    const char* filename, int iter, int& ierr) = 0;

  /// write RigidBodyAlignmentParameters as applied on top of original positions
  virtual void writeOrigRigidBodyAlignmentParameters (const Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// write Correlations 
  virtual void writeCorrelations (const Correlations& cormap, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Correlations 
  virtual Correlations readCorrelations (const Alignables& alivec, 
    const char* filename, int iter, int& ierr) = 0;

  /// write Alignable current absolute positions 
  virtual void writeAlignableAbsolutePositions (const Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable current absolute positions 
  virtual AlignablePositions readAlignableAbsolutePositions (const Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

  /// write Alignable original (before misalignment) absolute positions 
  virtual void writeAlignableOriginalPositions (const Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable original (before misalignment) absolute positions 
  virtual AlignablePositions readAlignableOriginalPositions (const Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

  /// write Alignable relative positions (shift,rotation) 
  virtual  void writeAlignableRelativePositions (const Alignables& alivec, 
    const char* filename, int iter, bool validCheck, int& ierr) = 0;

  /// read Alignable relative positions (shift,rotation) 
  virtual AlignableShifts readAlignableRelativePositions (const Alignables&
    alivec, const char* filename, int iter, int& ierr) = 0;

};

#endif
