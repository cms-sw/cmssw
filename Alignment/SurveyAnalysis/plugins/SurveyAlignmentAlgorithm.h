/** \class SurveyAlignmentAlgorithm
 *
 *  Alignment of Silicon Pixel Detector with survey constraint.
 *
 *  $Date: 2007/05/03 20:58:59 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#ifndef Alignment_SurveyAnalysis_SurveyAlignmentAlgorithm_h
#define Alignment_SurveyAnalysis_SurveyAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

namespace edm { class ParameterSet; class EventSetup; }

class AlignmentParameterStore;
class AlignableMuon;
class AlignableTracker;

class SurveyAlignmentAlgorithm : public AlignmentAlgorithmBase
{
  public:

  SurveyAlignmentAlgorithm(
			   const edm::ParameterSet&
			   );

  /// call at start of job
  virtual void initialize(
			  const edm::EventSetup&,
			  AlignableTracker*,
			  AlignableMuon*,
			  AlignmentParameterStore*
			  );

  /// call at end of job
  virtual void terminate() {}

  /// run for every event
  virtual void run(
		   const edm::EventSetup&,
		   const AlignmentAlgorithmBase::EventInfo &
		   ) {}


  private:

  std::string theOutfile;

  unsigned int theIterations;

  std::vector<std::string> theLevels;
};

#endif
