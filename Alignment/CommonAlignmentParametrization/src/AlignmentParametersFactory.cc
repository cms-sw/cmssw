///  $Date: 2007/10/08 15:56:00 $
///  $Revision: 1.12 $
/// (last update by $Author: cklae $)

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
//#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <vector>
#include <string>

namespace AlignmentParametersFactory {
  
  //_______________________________________________________________________________________
  ParametersType parametersType(const std::string &typeString)
  {
    if (typeString == "RigidBody") return kRigidBody;
    else if (typeString == "Survey") return kSurvey; //GF: do not belong here, so remove in the long term...
    
    throw cms::Exception("BadConfig") 
      << "AlignmentParametersFactory" << " No AlignmentParameters with name '" << typeString << "'.";
    
    return kRigidBody; // to please compiler...
  }

  //_______________________________________________________________________________________
  ParametersType parametersType(int typeInt)
  {
    if (typeInt == kRigidBody) return kRigidBody;
    if (typeInt == kSurvey) return kSurvey; //GF: do not belong here, so remove in the long term...
    
    throw cms::Exception("BadConfig") 
      << "AlignmentParametersFactory" << " No AlignmentParameters with number " << typeInt << ".";
    
    return kRigidBody; // to please compiler...
  }

  //_______________________________________________________________________________________
  std::string parametersTypeName(ParametersType parType)
  {
    switch(parType) {
    case kRigidBody:
      return "RigiBody";
    case kSurvey: //GF: do not belong here, so remove in the long term...
      return "Survey";
    }

    return "unknown_should_never_reach"; // to please the compiler
  }

  //_______________________________________________________________________________________
  AlignmentParameters* createParameters(Alignable *ali, ParametersType parType,
					const std::vector<bool> &sel)
  {
    switch (parType) {
    case kRigidBody:
      {
	const AlgebraicVector par(RigidBodyAlignmentParameters::N_PARAM, 0);
	const AlgebraicSymMatrix cov(RigidBodyAlignmentParameters::N_PARAM, 0);
	return new RigidBodyAlignmentParameters(ali, par, cov, sel);
      }
      break;
    case kSurvey:
// creates some unwanted dependencies - and does not fit into AlignmentParameters anyway!
      throw cms::Exception("BadConfig") 
	<< "AlignmentParametersFactory cannot create SurveyParameters.";
//       edm::LogWarning("Alignment") << "@SUB=createParameters"
// 				   << "Creating SurveyParameters of length 0!";
//       return new SurveyParameters(ali, AlgebraicVector(), AlgebraicSymMatrix());
      break;
    }
   
    return 0; // unreached (all ParametersType appear in switch), to please the compiler
  }
}


