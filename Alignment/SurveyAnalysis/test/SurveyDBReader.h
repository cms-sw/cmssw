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
 *
 *  $Date: 2007/05/03 19:20:00 $
 *  $Revision: 1.1 $
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
