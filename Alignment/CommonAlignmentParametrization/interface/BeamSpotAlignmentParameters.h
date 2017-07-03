#ifndef Alignment_CommonAlignment_BeamSpotAlignmentParameters_h
#define Alignment_CommonAlignment_BeamSpotAlignmentParameters_h

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class BeamSpotAlignmentParameters
///
/// Concrete class for alignment parameters and associated quantities 
/// [derived from AlignmentParameters]. The number of parameters
/// N_PARAM is fixed to 4 (2 translations in x & y, and 2 slopes)
///
///  $Date: 2008/09/02 15:08:12 $
///  $Revision: 1.13 $
/// (last update by $Author: flucke $)

class Alignable;
class AlignableDetOrUnitPtr;
class TrajectoryStateOnSurface;

class BeamSpotAlignmentParameters : public AlignmentParameters 
{

public:

  /// Give parameters a name
  enum AlignmentParameterName 
	{
	  dx = 0, dy,
	  dxslope, dyslope,
	  N_PARAM
	};
  
  /// Constructor with empty parameters/covariance (if calcMis = false) or with parameters
  /// (no covariance) created from current (mis-)placement of alignable (if calcMis = true).
  BeamSpotAlignmentParameters( Alignable* alignable, bool calcMis );

  /// Constructor for full set of parameters
  BeamSpotAlignmentParameters( Alignable* alignable, 
			       const AlgebraicVector& parameters, 
			       const AlgebraicSymMatrix& covMatrix );

  /// Constructor for selection 
  BeamSpotAlignmentParameters( Alignable* alignable, const AlgebraicVector& parameters, 
			       const AlgebraicSymMatrix& covMatrix, 
			       const std::vector<bool>& selection );

  /// Destructor 
  ~BeamSpotAlignmentParameters() override;
  void apply() override;
  int type() const override;

  /// Clone all parameters (for update of parameters)
  BeamSpotAlignmentParameters* clone( const AlgebraicVector& parameters, 
					      const AlgebraicSymMatrix& covMatrix ) const override;
 
  /// Clone selected parameters (for update of parameters)
  BeamSpotAlignmentParameters*
    cloneFromSelected(const AlgebraicVector& parameters, const AlgebraicSymMatrix& covMatrix) const override;
  
  /// Get all derivatives 
  AlgebraicMatrix derivatives( const TrajectoryStateOnSurface& tsos,
				       const AlignableDetOrUnitPtr & ) const override;

  /// Get selected derivatives
  AlgebraicMatrix selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
					       const AlignableDetOrUnitPtr & ) const override;

  /// Get translation parameters
  AlgebraicVector translation(void) const;

  /// Get rotation parameters
  AlgebraicVector rotation(void) const;

  /// calculate and return parameters in global frame 
  AlgebraicVector globalParameters(void) const;

  /// print parameters to screen 
  void print(void) const;

  /// Calculate parameter vector of misplacements (shift+rotation) from alignable.
  /// (If ali=0, return empty AlgebraicVector of proper length.)
  static AlgebraicVector displacementFromAlignable(const Alignable* ali);
};

#endif
