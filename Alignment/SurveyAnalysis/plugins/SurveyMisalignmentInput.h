#ifndef Alignment_SurveyAnalysis_SurveyMisalignmentInput_h
#define Alignment_SurveyAnalysis_SurveyMisalignmentInput_h

/** \class SurveyMisalignmentInput
 *
 *  Class to misaligned tracker from DB.
 *
 *  $Date: 2010/01/07 14:36:23 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */
// user include files

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"
#include "FWCore/Framework/interface/ESHandle.h"

class AlignableSurface;
class Alignments;

class SurveyMisalignmentInput:
  public SurveyInputBase
{
public:
	
  SurveyMisalignmentInput(
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

  edm::ESHandle<Alignments> alignments;
	
  /// Add survey info to an alignable
  void addSurveyInfo(Alignable*);

  /// Get alignable surface from misalignments.db
  AlignableSurface getAlignableSurface(align::ID);

  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;
};

#endif
