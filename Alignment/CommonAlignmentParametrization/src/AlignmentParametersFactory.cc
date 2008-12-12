///  $Date: 2008/09/02 15:18:19 $
///  $Revision: 1.1 $
/// (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters4D.h"
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
    else if (typeString == "RigidBody4D") return kRigidBody4D;    
    throw cms::Exception("BadConfig") 
      << "AlignmentParametersFactory" << " No AlignmentParameters with name '" << typeString << "'.";
    
    return kRigidBody; // to please compiler...
  }

  //_______________________________________________________________________________________
  ParametersType parametersType(int typeInt)
  {
    if (typeInt == kRigidBody) return kRigidBody;
    if (typeInt == kSurvey) return kSurvey; //GF: do not belong here, so remove in the long term...
    if (typeInt == kRigidBody4D) return kRigidBody4D;    
    
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
    case kRigidBody4D:
      return "RigiBody4D"; 
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
    case kRigidBody4D:
      {
	const AlgebraicVector par(RigidBodyAlignmentParameters4D::N_PARAM, 0);
	const AlgebraicSymMatrix cov(RigidBodyAlignmentParameters4D::N_PARAM, 0);
	return new RigidBodyAlignmentParameters4D(ali, par, cov, sel);
      }
      break;
    }
   
    return 0; // unreached (all ParametersType appear in switch), to please the compiler
  }
}


