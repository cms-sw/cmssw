#ifndef Alignment_SurveyAnalysis_SurveyDBReader_h
#define Alignment_SurveyAnalysis_SurveyDBReader_h

/** \class SurveyDBReader
 *
 * Module that reads survey info from DB and prints them out.
 *
 * Usage:
 *   module reader = SurveyDBReader { string fileName = 'surveyDBDump.root' }
 *   path p = { reader }
 *
 * Only one parameter to set the name of the output ROOT file.
 *
 *  $Date: 2007/06/19 14:50:34 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class SurveyDBReader:
  public edm::EDAnalyzer
{
  public:

  /// Set file name
  SurveyDBReader(
		 const edm::ParameterSet&
		 );

  /// Read from DB and print survey info.
  virtual void beginJob(
			const edm::EventSetup&
			);

  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) {}

  private:

  std::string theFileName;
};

#endif
