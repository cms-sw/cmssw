#ifndef Alignment_SurveyAnalysis_CreateSurveyRcds_h
#define Alignment_SurveyAnalysis_CreateSurveyRcds_h

/** \class CreateSurveyRcds
 *
 *  Class to create Survey[Error]Rcd for alignment with survey constraint
 *
 *  $Date: 2012/06/13 09:22:26 $
 *  $Revision: 1.3 $
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
		
		void analyze(
				     const edm::Event&, 
				     const edm::EventSetup&
				     ) override;
		
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
	};

#endif
