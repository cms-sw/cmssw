#ifndef Alignment_SurveyAnalysis_SurveyDBUploader_h
#define Alignment_SurveyAnalysis_SurveyDBUploader_h

/** \class SurveyDBUploader
 *
 * Module for uploading survey info to the DB.
 *
 * Usage:
 *   module uploader = SurveyDBUploader
 *   {
 *     string valueTag = "TrackerSurveyRcd"
 *     string errorTag = "TrackerSurveyErrorExtendedRcd"
 *   }
 *
 *  $Date: 2007/04/09 01:16:13 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class Alignable;
class Alignments;
class AlignTransform;
class SurveyErrors;

class SurveyDBUploader:
  public edm::EDAnalyzer
{
  typedef AlignTransform SurveyValue;
  typedef Alignments     SurveyValues;

  public:

  /// Set value & error tag names for survey records.
  SurveyDBUploader(
		   const edm::ParameterSet&
		   );

  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) {}

  /// Upload to DB
  virtual void endJob();

  private:

  /// Get survey info of an alignable in the detector.
  void getSurveyInfo(
		     const Alignable*
		     );

  std::string theValueRcd; // tag name of survey values record in DB
  std::string theErrorExtendedRcd; // tag name of survey errors record in DB

  SurveyValues* theValues; // survey values for all alignables in detector
  SurveyErrors* theErrors; // survey errors for all alignables in detector
};

#endif
