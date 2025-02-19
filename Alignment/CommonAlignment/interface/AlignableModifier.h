#ifndef Alignment_TrackerAlignment_AlignableModifier_H
#define Alignment_TrackerAlignment_AlignableModifier_H

#include <vector>
#include <string>
#include <utility>

#include "CondFormats/Alignment/interface/Definitions.h"

/// AlignableModifier is a helper class to modify the Alignables.
///
/// Configuration parameters are defined in this class.

class Alignable;

namespace CLHEP { class DRand48Engine; }
namespace edm { class ParameterSet; }

class AlignableModifier 
{

public:

  /// Constructor
  AlignableModifier();

  /// Destructor
  ~AlignableModifier();

  /// Modify given set of alignables according to parameters
  bool modify( Alignable* alignable, const edm::ParameterSet& pSet );

  /// Check if given parameter should be propagated
  bool isPropagated( const std::string& parameterName ) const;

  /// Move alignable in global space according to parameters
  void moveAlignable( Alignable* alignable, bool random, bool gaussian,
		      float sigmaX, float sigmaY, float sigmaZ );

  /// Move alignable in local space according to parameters
  void moveAlignableLocal( Alignable* alignable, bool random, bool gaussian,
			   float sigmaX, float sigmaY, float sigmaZ );

  /// Rotate alignable in global space according to parameters
  void rotateAlignable( Alignable* alignable, bool random, bool gaussian,
			float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ );

  /// Rotate alignable in local space according to parameters
  void rotateAlignableLocal( Alignable* alignable, bool random, bool gaussian,
			     float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ );
  
  /// Add the AlignmentPositionError (in global frame) to Alignable
  void addAlignmentPositionError( Alignable* alignable, 
				  float dx, float dy, float dz );

  /// Add the AlignmentPositionError (in local frame) to Alignable
  void addAlignmentPositionErrorLocal( Alignable* alignable, 
				       float dx, float dy, float dz );

  /// Add alignment position error resulting from rotation in global frame
  void addAlignmentPositionErrorFromRotation( Alignable* alignable, 
					      float phiX, float phiY, float phiZ ); 

  /// Add alignment position error resulting from rotation in local frame
  void addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
						   float phiX, float phiY, float phiZ ); 

  /// Add alignment position error resulting from rotation in global frame
  void addAlignmentPositionErrorFromRotation( Alignable* alignable, 
					      align::RotationType& ); 

  /// Add alignment position error resulting from rotation in local frame
  void addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
						   align::RotationType& ); 

  /// Decodes string and sets distribution accordingly ('fixed', 'flat' or 'gaussian').
  void setDistribution( const std::string& distr );

  /// Resets the generator seed according to the argument.
  void setSeed( long seed );

  /// Return a vector of random numbers (gaussian distribution)
  const std::vector<float> gaussianRandomVector( float sigmaX, float sigmaY, float sigmaZ ) const;
  /// Return a vector of random numbers (flat distribution)
  const std::vector<float> flatRandomVector( float sigmaX, float sigmaY, float sigmaZ ) const;
  /// Randomise all entries in 'rnd': 
  /// - either from gaussian with width rnd[i]
  /// - or from flat distribution between -rnd[i] and rnd[i]
  void randomise(std::vector<double> &rnd, bool gaussian) const;

private:
  typedef std::pair<std::string,std::vector<double> > DeformationMemberType;
  void addDeformation(Alignable *alignable, const DeformationMemberType &deformation,
		      bool random, bool gaussian, double scale);

  /// Unique random number generator
  CLHEP::DRand48Engine* theDRand48Engine;

  /// Initialisation of all parameters
  void init_(); 

  int m_modified; // Indicates if a modification was performed

  // All parameters (see AlignableModifier::init() for definitions)
  std::string distribution_;
  bool   random_, gaussian_, setError_;
  bool   setRotations_,setTranslations_;
  long   seed_;
  double scaleError_,scale_;
  double phiX_, phiY_, phiZ_;
  double phiXlocal_, phiYlocal_, phiZlocal_;
  double dX_, dY_, dZ_;
  double dXlocal_, dYlocal_, dZlocal_;
  DeformationMemberType deformation_;
  double twist_, shear_;

};

#endif //AlignableModifier_H
