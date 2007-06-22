#ifndef Alignment_SurveyAnalysis_SurveyMisalignmentInput_h
#define Alignment_SurveyAnalysis_SurveyMisalignmentInput_h

/** \class SurveyInputTrackerFromDB
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: 2007/05/08 22:36:45 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

class SurveyMisalignmentInput:
  public SurveyInputBase
{
public:
	
  SurveyMisalignmentInput(
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

	edm::ESHandle<Alignments> alignments;
	
	/// Add survey info to an alignable
  void addSurveyInfo(Alignable*);
	/// Get alignable surface from misalignments.db
	AlignableSurface getAlignableSurface(uint32_t);
	
};

#endif
