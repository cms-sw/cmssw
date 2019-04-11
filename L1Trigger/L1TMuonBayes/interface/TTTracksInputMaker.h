/*
 * TTTracksInputMaker.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1TMUONBAYES_TTTRACKSINPUTMAKER_H_
#define L1TMUONBAYES_TTTRACKSINPUTMAKER_H_

#include "L1Trigger/L1TMuonBayes/interface/TrackingTriggerTrack.h"
#include "L1Trigger/L1TMuonBayes/interface/ProcConfigurationBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TTTracksInputMaker {
public:
  enum TTTracksSource {
    NONE,
    SIM_TRACKS,
    TRACKING_PARTICLES,
    L1_TRACKER
  };

  TTTracksInputMaker(const edm::ParameterSet& edmCfg);
  virtual ~TTTracksInputMaker();

  TrackingTriggerTracks loadTTTracks(const edm::Event &event, int bx, const edm::ParameterSet& edmCfg, const ProcConfigurationBase* procConf);

  //, int bxFrom = 0, int bxTo = 0 at lest in the emualtor, the ttTracks are produced only in the BX = 0
/*  virtual const TrackingTriggerTracks buildInputForProcessor(unsigned int iProcessor, l1t::tftype procTyp) {
    return ttTracks;
  }*/

private:
  int l1Tk_nPar = 4;

  unsigned int l1Tk_minNStub = 4;

  TTTracksSource ttTracksSource = SIM_TRACKS;

  void addTTTrack(TrackingTriggerTracks& ttTracks, std::shared_ptr<TrackingTriggerTrack>& ttTrack, const ProcConfigurationBase* procConf);
  ///all ttTracks in a event
  //TrackingTriggerTracks ttTracks;
};

#endif /* INTERFACE_TTTRACKSINPUTMAKER_H_ */
