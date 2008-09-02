#ifndef Alignment_CommonAlignment_AlignmentParametersFactory_h 
#define Alignment_CommonAlignment_AlignmentParametersFactory_h 


/// \namespace AlignmentParametersFactory
///
///  $Date: 2007/10/08 15:56:00 $
///  $Revision: 1.12 $
/// (last update by $Author: cklae $)

#include <vector>
#include <string>

class Alignable;
class AlignmentParameters;

namespace AlignmentParametersFactory {
  /// enums for all available AlignmentParameters
  enum ParametersType {
    kRigidBody = 0, // RigidBodyAlignmentParameters
    kSurvey         // SurveyParameters GF: do not belong here, so remove in the long term...
  };

  /// convert string to ParametersType - exception if not known
  ParametersType parametersType(const std::string &typeString);
  /// convert int to ParametersType (if same value) - exception if no corresponding type
  ParametersType parametersType(int typeInt);
  /// convert ParametersType to string understood by parametersType(string &typeString)
  std::string parametersTypeName(ParametersType parType);

  /// create AlignmentParameters of type 'parType' for Alignable 'ali' with selection
  /// 'sel' for active parameters
  AlignmentParameters* createParameters(Alignable *ali, ParametersType parType,
					const std::vector<bool> &sel);
}

#endif
