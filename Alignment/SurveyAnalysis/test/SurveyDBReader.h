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
 *  $Date: 2010/01/07 14:36:23 $
 *  $Revision: 1.4 $
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
  virtual void beginJob() { theFirstEvent = true; }

  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );

  private:

  std::string theFileName;

  bool theFirstEvent;
};

#endif
