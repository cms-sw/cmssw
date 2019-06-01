#ifndef FastSimulation_Event_FSimTrackEqual_H
#define FastSimulation_Event_FSimTrackEqual_H

#include "FastSimulation/Event/interface/FSimTrack.h"

class FSimTrackEqual {
public:
  FSimTrackEqual(int index) : trackIndex_(index){};
  FSimTrackEqual(const FSimTrack& myTrack) : trackIndex_(myTrack.trackId()){};
  ~FSimTrackEqual() { ; };

  inline bool operator()(const FSimTrack& track) const { return (track.id() == trackIndex_); }

private:
  int trackIndex_;
};

#endif
