#ifndef Alignment_SurveyAnalysis_SurveyInputDummy_h
#define Alignment_SurveyAnalysis_SurveyInputDummy_h

/** \class SurveyInputDummy
 *
 *  For uploading some pseudo-dummy survey errors to DB.
 *
 *  $Date: 2007/05/08 22:36:47 $
 *  $Revision: 1.2 $
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
