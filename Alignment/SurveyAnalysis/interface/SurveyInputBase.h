#ifndef Alignment_SurveyAnalysis_SurveyInputBase_h
#define Alignment_SurveyAnalysis_SurveyInputBase_h

/** \class SurveyInputBase
 *
 *  Abstract base class to read survey raw measurements.
 *
 *  $Date: 2010/01/07 14:36:22 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class Alignable;

class SurveyInputBase:
  public edm::EDAnalyzer
{
  public:

  virtual ~SurveyInputBase();

  /// Read data from input.
  virtual void beginJob() { theFirstEvent = true; }

  /// Do nothing for each event.
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) = 0;

  /// Get alignable detector as read from input.
  inline static Alignable* detector();

  /// Add a component or sub-system to the detector.
  /// Class will own component (takes care of deleting the pointer).
  static void addComponent(
			   Alignable*
			   );

  protected:

  bool theFirstEvent;

  private:

  static Alignable* theDetector; // only one detector
};

Alignable* SurveyInputBase::detector()
{
  return theDetector;
}

#endif
