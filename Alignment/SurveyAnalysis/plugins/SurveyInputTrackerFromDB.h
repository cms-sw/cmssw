#ifndef Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h
#define Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h

/** \class SurveyInputTrackerFromDB
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: 2010/01/07 14:36:23 $
 *  $Revision: 1.3 $
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
  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;
};

#endif
