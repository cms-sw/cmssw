#ifndef Alignment_SurveyAnalysis_SurveyInputBase_h
#define Alignment_SurveyAnalysis_SurveyInputBase_h

/** \class SurveyInputBase
 *
 *  Abstract base class to read survey raw measurements.
 *
 *  $Date: 2007/05/15 18:14:04 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

class Alignable;

class SurveyInputBase : public edm::one::EDAnalyzer<> {
public:
  ~SurveyInputBase() override;

  /// Read data from input.
  void beginJob() override { theFirstEvent = true; }

  /// Do nothing for each event.
  void analyze(const edm::Event&, const edm::EventSetup&) override = 0;

  /// Get alignable detector as read from input.
  inline static Alignable* detector();

  /// Add a component or sub-system to the detector.
  /// Class will own component (takes care of deleting the pointer).
  static void addComponent(Alignable*);

protected:
  bool theFirstEvent;

private:
  static Alignable* theDetector;  // only one detector
};

Alignable* SurveyInputBase::detector() { return theDetector; }

#endif
