#ifndef Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignableIORoot_h

#include <map>

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIO.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"

/// concrete class for ROOT-based I/O of Alignment parameters, correlations 
///  and Alignable positions. Derived from AlignmentIO
 

class AlignmentIORoot : public AlignmentIO
{

  public:

  /// write AlignmentParameters 
  void writeAlignmentParameters (const Alignables& alivec, 
								 char* filename, int iter, bool validCheck, int& ierr );

  /// read AlignmentParameters 
  Parameters readAlignmentParameters (const Alignables& alivec, 
									  char* filename, int iter, int& ierr);

  /// write Correlations 
  void writeCorrelations (const Correlations& cormap, 
						  char* filename, int iter, bool validCheck, int& ierr);

  /// read Correlations 
  Correlations readCorrelations (const Alignables& alivec, 
								 char* filename, int iter, int& ierr);
  
  /// write Alignable current absolute positions 
  void writeAlignableAbsolutePositions (const Alignables& alivec, 
										char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable current absolute positions 
  AlignablePositions readAlignableAbsolutePositions (const Alignables& alivec,
													 char* filename, int iter, int& ierr);
  
  /// write Alignable original (before misalignment) absolute positions 
  void writeAlignableOriginalPositions (const Alignables& alivec, 
										char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable original (before misalignment) absolute positions 
  AlignablePositions readAlignableOriginalPositions (const Alignables& alivec,
													 char* filename, int iter, int& ierr);
  
  /// write Alignable relative positions (shift,rotation) 
  void writeAlignableRelativePositions (const Alignables& alivec, 
										char* filename, int iter, bool validCheck, int& ierr);
  
  /// read Alignable relative positions (shift,rotation) 
  AlignableShifts readAlignableRelativePositions (const Alignables& alivec,
												  char* filename, int iter, int& ierr);

};

#endif
