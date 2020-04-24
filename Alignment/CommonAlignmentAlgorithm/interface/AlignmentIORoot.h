#ifndef Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h

/// \class AlignmentIORoot
///
/// concrete class for ROOT-based I/O of Alignment parameters, correlations 
///  and Alignable positions. Derived from AlignmentIO
///
///  $Date: 2006/11/30 09:56:03 $
///  $Revision: 1.3 $
/// (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIO.h"


class AlignmentIORoot : public AlignmentIO
{

  public:
  virtual ~AlignmentIORoot() = default;
  /// write AlignmentParameters 
  void writeAlignmentParameters (const align::Alignables& alivec, 
				 const char* filename, int iter, bool validCheck, int& ierr );

  /// read AlignmentParameters 
  align::Parameters readAlignmentParameters (const align::Alignables& alivec, 
				      const char* filename, int iter, int& ierr);

  /// write RigidBodyAlignmentParameters as applied on top of original positions
  void writeOrigRigidBodyAlignmentParameters (const align::Alignables& alivec, const char* filename,
					      int iter, bool validCheck, int& ierr);

  /// write Correlations 
  void writeCorrelations (const align::Correlations& cormap, 
			  const char* filename, int iter, bool validCheck, int& ierr);

  /// read Correlations 
  align::Correlations readCorrelations (const align::Alignables& alivec, 
				 const char* filename, int iter, int& ierr);
  
  /// write Alignable current absolute positions 
  void writeAlignableAbsolutePositions (const align::Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable current absolute positions 
  AlignablePositions readAlignableAbsolutePositions (const align::Alignables& alivec,
						     const char* filename, int iter, int& ierr);
  
  /// write Alignable original (before misalignment) absolute positions 
  void writeAlignableOriginalPositions (const align::Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable original (before misalignment) absolute positions 
  AlignablePositions readAlignableOriginalPositions (const align::Alignables& alivec,
						     const char* filename, int iter, int& ierr);
  
  /// write Alignable relative positions (shift,rotation) 
  void writeAlignableRelativePositions (const align::Alignables& alivec, 
					const char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable relative positions (shift,rotation) 
  AlignableShifts readAlignableRelativePositions (const align::Alignables& alivec,
						  const char* filename, int iter, int& ierr);

};

#endif
