#ifndef Alignment_SurveyAnalysis_CreateSurveyRcds_h
#define Alignment_SurveyAnalysis_CreateSurveyRcds_h

/** \class CreateSurveyRcds
 *
 *  Class to create Survey[Error]Rcd for alignment with survey constraint
 *
 *  $Date: 2010/01/07 14:36:22 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */
// user include files

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"
#include "FWCore/Framework/interface/ESHandle.h"

class AlignableSurface;
class Alignments;

class CreateSurveyRcds:
	public SurveyInputBase
	{
	public:
		
		CreateSurveyRcds(
				 const edm::ParameterSet&
				 );
		
		virtual void analyze(
				     const edm::Event&, 
				     const edm::EventSetup&
				     );
		
	private:
		
		/// module which modifies the geometry
		void setGeometry(Alignable* );
		/// module which creates/inserts the survey errors
		void setSurveyErrors( Alignable* );
		
		/// default values for assembly precision
		AlgebraicVector getStructurePlacements(int ,int );
		
		/// default values for survey uncertainty
		AlgebraicVector getStructureErrors(int ,int );
		
		
		
		std::string m_inputGeom;
		double m_inputSimpleMis;
		bool m_generatedRandom;
		bool m_generatedSimple;
		
		
		SurveyInputTextReader::MapType uIdMap;
		
		std::string textFileName;
		
		edm::ESHandle<Alignments> alignments;
		
	  int m_ROWS_PER_ROC;
	  int m_COLS_PER_ROC;
	  int m_BIG_PIX_PER_ROC_X;
	  int m_BIG_PIX_PER_ROC_Y;
	  int m_ROCS_X;
	  int m_ROCS_Y;
	  bool m_upgradeGeometry;
		
	};

#endif
