#ifndef Alignment_SurveyAnalysis_SurveyInputDummy_h
#define Alignment_SurveyAnalysis_SurveyInputDummy_h

/** \class SurveyInputDummy
 *
 *  For uploading some random survey values and pseudo-dummy errors to DB.
 *
 *  Usage:
 *    module FPix = SurveyInputDummy
 *    {
 *      VPSet errors =
 *      {
 *        { string level = "DetUnit" double value = 5e-4 },
 *        { string level = "Panel"   double value = 5e-4 }
 *      }
 *    }
 *
 *  The survey value of a structure in a level is randomly selected from a
 *  Gaussian distribution of mean given by the ideal geometry and width =
 *  "value" (e.g. width = 5e-4 for a Panel).
 *  
 *  The covariance matrix for all structures of a level will be diagonal
 *  given by value^2 * identity.
 *
 *  $Date: 2007/06/24 01:46:26 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

class SurveyInputDummy:
  public SurveyInputBase
{
  typedef AlignableObjectId::AlignableObjectIdType StructureType;

  public:

  SurveyInputDummy(
		   const edm::ParameterSet&
		   );

  /// Read ideal tracker geometry from DB
  virtual void beginJob(
			const edm::EventSetup&
			);

  private:

  /// Add survey info to an alignable
  void addSurveyInfo(
		     Alignable*
		     );

  std::map<StructureType, double> theErrors;
};

#endif
