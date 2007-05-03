#ifndef Alignment_SurveyAnalysis_SurveyDBReader_h
#define Alignment_SurveyAnalysis_SurveyDBReader_h

/** \class SurveyDBReader
 *
 * Module that reads survey info from DB and prints them out.
 *
 * Usage:
 *   module reader = SurveyDBReader {}
 *   path p = { reader }
 *
 * No configuration for module is necessary. Just put module in path.
 * However, you should run only 1 event.
 *
 *  $Date: 2007/04/09 01:16:13 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class SurveyDBReader:
  public edm::EDAnalyzer
{
  public:

  /// Do nothing. Required by framework.
  SurveyDBReader(
		 const edm::ParameterSet&
		 ) {}

  /// Read from DB and print survey info.
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );
};

#endif
