#ifndef Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h
#define Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h

/** \class SurveyInputTrackerFromDB
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: 2007/04/07 01:58:49 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  virtual void beginJob(
		const edm::EventSetup&
		);
	
private:
	
	SurveyInputTextReader::MapType uIdMap;
	//edm::ParameterSet theParameterSet;	
	std::string textFileName;
	
  /// Add survey info to an alignable
  void addSurveyInfo(
		Alignable*
		);
};

#endif
