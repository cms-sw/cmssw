#ifndef Alignment_SurveyAnalysis_SurveyInputDummy_h
#define Alignment_SurveyAnalysis_SurveyInputDummy_h

/** \class SurveyInputDummy
 *
 *  For uploading some random survey values and pseudo-dummy errors to DB.
 *
 *  Usage:
 *    module FPix = SurveyInputDummy
 *    {
 *      bool randomizeValue = true
 *
 *      VPSet errors =
 *      {
 *        { string level = "DetUnit" double value = 5e-4 },
 *        { string level = "Panel"   double value = 5e-4 }
 *      }
 *    }
 *
 *  If randomizeValue is true, the survey value of a structure in a level
 *  is randomly selected from a Gaussian distribution of mean given by the
 *  ideal geometry and width = "value" (e.g. width = 5e-4 for a Panel).
 *  
 *  The covariance matrix for all structures of a level will be diagonal
 *  given by value^2 * identity.
 *
 *  $Date: 2010/01/07 14:36:23 $
 *  $Revision: 1.5 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

class SurveyInputDummy:
  public SurveyInputBase
{
  public:

  SurveyInputDummy(
		   const edm::ParameterSet&
		   );

  /// Read ideal tracker geometry from DB
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );

  private:

  /// Add survey info to an alignable
  void addSurveyInfo(
		     Alignable*
		     );

  bool theRandomizeValue; // randomize survey values if true

  std::map<align::StructureType, double> theErrors;
};

#endif
