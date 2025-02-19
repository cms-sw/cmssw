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
///  $Date: 2010/09/10 11:16:36 $
///  $Revision: 1.1 $
/// (last update by $Author: mussgill $)

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
  virtual ~BeamSpotAlignmentParameters();
  virtual void apply();
  virtual int type() const;

  /// Clone all parameters (for update of parameters)
  virtual BeamSpotAlignmentParameters* clone( const AlgebraicVector& parameters, 
					      const AlgebraicSymMatrix& covMatrix ) const;
 
  /// Clone selected parameters (for update of parameters)
  virtual BeamSpotAlignmentParameters*
    cloneFromSelected(const AlgebraicVector& parameters, const AlgebraicSymMatrix& covMatrix) const;
  
  /// Get all derivatives 
  virtual AlgebraicMatrix derivatives( const TrajectoryStateOnSurface& tsos,
				       const AlignableDetOrUnitPtr & ) const;

  /// Get selected derivatives
  virtual AlgebraicMatrix selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
					       const AlignableDetOrUnitPtr & ) const;

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
