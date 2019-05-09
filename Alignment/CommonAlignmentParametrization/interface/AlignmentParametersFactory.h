#ifndef Alignment_CommonAlignmentParametrization_AlignmentParametersFactory_h
#define Alignment_CommonAlignmentParametrization_AlignmentParametersFactory_h

/// \namespace AlignmentParametersFactory
///
/// Factory interface to create AlignmentParameters for the known types,
/// defined by the enum AlignmentParametersFactory::ParametersType.
///
///  $Date: 2010/09/10 11:19:23 $
///  $Revision: 1.4 $
/// (last update by $Author: mussgill $)

#include <string>
#include <vector>

class Alignable;
class AlignmentParameters;

namespace AlignmentParametersFactory {
  /// enums for all available AlignmentParameters
  enum ParametersType {
    kRigidBody = 0,    // RigidBodyAlignmentParameters
    kSurvey,           // SurveyParameters GF: do not belong here, so remove in the long
                       // term...
    kRigidBody4D,      // RigidBodyAlignmentParameters4D
    kBeamSpot,         // BeamSpotAlignmentParameters
    kBowedSurface,     // BowedSurfaceAlignmentParameters
    kTwoBowedSurfaces  // TwoBowedSurfacesAlignmentParameters
  };

  /// convert string to ParametersType - exception if not known
  ParametersType parametersType(const std::string &typeString);
  /// convert int to ParametersType (if same value) - exception if no
  /// corresponding type
  ParametersType parametersType(int typeInt);
  /// convert ParametersType to string understood by parametersType(string
  /// &typeString)
  std::string parametersTypeName(ParametersType parType);

  /// create AlignmentParameters of type 'parType' for Alignable 'ali' with
  /// selection 'sel' for active parameters
  AlignmentParameters *createParameters(Alignable *ali, ParametersType parType, const std::vector<bool> &sel);
}  // namespace AlignmentParametersFactory

#endif
