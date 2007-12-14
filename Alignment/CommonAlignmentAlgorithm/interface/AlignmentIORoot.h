#ifndef Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h

/// \class AlignmentIORoot
///
/// concrete class for ROOT-based I/O of Alignment parameters, correlations 
///  and Alignable positions. Derived from AlignmentIO
///
///  $Date: 2006/10/19 14:20:59 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIO.h"

class Alignable;
class AlignmentParameters;


class AlignmentIORoot : public AlignmentIO
{

  public:

  /// write AlignmentParameters 
  void writeAlignmentParameters (const Alignables& alivec, 
				 const char* filename, int iter, bool validCheck, int& ierr );

  /// read AlignmentParameters 
  Parameters readAlignmentParameters (const Alignables& alivec, 
				      const char* filename, int iter, int& ierr);

  /// write RigidBodyAlignmentParameters as applied on top of original positions
  void writeOrigRigidBodyAlignmentParameters (const Alignables& alivec, const char* filename,
					      int iter, bool validCheck, int& ierr);

  /// write Correlations 
  void writeCorrelations (const Correlations& cormap, 
			  const char* filename, int iter, bool validCheck, int& ierr);

  /// read Correlations 
  Correlations readCorrelations (const Alignables& alivec, 
				 const char* filename, int iter, int& ierr);
  
  /// write Alignable current absolute positions 
  void writeAlignableAbsolutePositions (const Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable current absolute positions 
  AlignablePositions readAlignableAbsolutePositions (const Alignables& alivec,
						     const char* filename, int iter, int& ierr);
  
  /// write Alignable original (before misalignment) absolute positions 
  void writeAlignableOriginalPositions (const Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable original (before misalignment) absolute positions 
  AlignablePositions readAlignableOriginalPositions (const Alignables& alivec,
						     const char* filename, int iter, int& ierr);
  
  /// write Alignable relative positions (shift,rotation) 
  void writeAlignableRelativePositions (const Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable relative positions (shift,rotation) 
  AlignableShifts readAlignableRelativePositions (const Alignables& alivec,
						  const char* filename, int iter, int& ierr);

};

#endif
