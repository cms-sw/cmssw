#ifndef Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h
#define Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h

/** \class SurveyInputTrackerFromDB
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: 2007/10/08 16:38:04 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

class SurveyInputTrackerFromDB:
  public SurveyInputBase
{
public:
	
  SurveyInputTrackerFromDB(
			   const edm::ParameterSet&
			   );
	
  /// Read ideal tracker geometry from DB
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );
	
private:
	
  SurveyInputTextReader::MapType uIdMap;

  std::string textFileName;
	
  /// Add survey info to an alignable
  void addSurveyInfo(
		     Alignable*
		     );
};

#endif
