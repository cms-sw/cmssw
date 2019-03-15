/*
 * TTAlgoMuon.h
 *
 *  Created on: Oct 19, 2018
 *      Author: kbunkow
 */

#ifndef OMTF_TTALGOMUON_H_
#define OMTF_TTALGOMUON_H_

#include <L1Trigger/L1TMuonBayes/interface/Omtf/AlgoMuon.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include "L1Trigger/L1TMuonBayes/interface/TrackingTriggerTrack.h"

typedef std::vector<TrackingTriggerTrack> TTTracks;

class TTAlgoMuon: public AlgoMuon {
public:
  //move the gpResults content to the this->gpResults, gpResults is empty after that
  TTAlgoMuon(const TrackingTriggerTrack& ttTrack, const GoldenPatternResult& gpResult, GoldenPatternBase* goldenPatern,
      std::vector<std::shared_ptr<GoldenPatternResult> >& gpResults, unsigned int refHitNum, int bx = 0):
    AlgoMuon(gpResult, goldenPatern,  refHitNum, bx),
    ttTrack(ttTrack) {
    this->gpResults.swap(gpResults);
  }

  const TrackingTriggerTrack& getTtTrack() const {
    return ttTrack;
  }

  ///
  void setGpResults(std::vector<std::shared_ptr<GoldenPatternResult> >& gpResults) {
    this->gpResults.swap(gpResults);
    gpResults.clear(); //in case there was something before in the this->gpResults
  }

  std::vector<std::shared_ptr<GoldenPatternResult> >& getGpResults() {
    return gpResults;
  }

  //TODO add getters for phi eta pt at vertex?

private:
  TrackingTriggerTrack ttTrack; //maybe rather should be a pointer?

  ///results for all the reference layers processed for this ttTrack
  ///we cannot use the GoldenPatternResuls stored by the GoldenPatternBase, because in one event the same goldePattern
  ///may be hit many times by different ttTracks
  std::vector<std::shared_ptr<GoldenPatternResult> > gpResults;
};

typedef std::vector<std::shared_ptr<TTAlgoMuon> > TTMuons;


#endif /* OMTF_TTALGOMUON_H_ */
