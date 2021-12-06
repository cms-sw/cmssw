/** \class SurveyAlignmentAlgorithm
 *
 *  Alignment of Silicon Pixel Detector with survey constraint.
 *
 *  $Date: 2010/09/10 11:53:18 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#ifndef Alignment_SurveyAnalysis_SurveyAlignmentAlgorithm_h
#define Alignment_SurveyAnalysis_SurveyAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

class AlignmentParameterStore;
class AlignableMuon;
class AlignableTracker;
class AlignableExtras;

class SurveyAlignmentAlgorithm : public AlignmentAlgorithmBase {
public:
  SurveyAlignmentAlgorithm(const edm::ParameterSet&, const edm::ConsumesCollector&);

  /// call at start of job
  void initialize(
      const edm::EventSetup&, AlignableTracker*, AlignableMuon*, AlignableExtras*, AlignmentParameterStore*) override;

  /// call at end of job
  void terminate(const edm::EventSetup& iSetup) override {}

  /// run for every event
  void run(const edm::EventSetup&, const AlignmentAlgorithmBase::EventInfo&) override {}

private:
  std::string theOutfile;

  unsigned int theIterations;

  std::vector<std::string> theLevels;
};

#endif
