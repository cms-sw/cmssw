#ifndef Alignment_SurveyAnalysis_SurveyInputBase_h
#define Alignment_SurveyAnalysis_SurveyInputBase_h

/** \class SurveyInputBase
 *
 *  Abstract base class to read survey raw measurements.
 *
 *  $Date: 2007/01/17 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class Alignable;

class SurveyInputBase:
  public edm::EDAnalyzer
{
  public:

  virtual ~SurveyInputBase();

  /// Read data from input
  virtual void beginJob(
			const edm::EventSetup&
			) = 0;

  /// Do nothing for each event
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) {}

  /// Get alignable detector as read from input
  inline static Alignable* detector();

  /// Add a component or sub-system to the detector
  static void addComponent(
			   Alignable*
			   );

  private:

  static Alignable* theDetector; // only one detector
};

Alignable* SurveyInputBase::detector()
{
  return theDetector;
}

#endif
