#ifndef AlignmentTrackSelector_h
#define AlignmentTrackSelector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"

class AlignmentTrackSelector
{

 public:

  /// constructor
  AlignmentTrackSelector(const edm::ParameterSet & cfg);

  /// destructor
  ~AlignmentTrackSelector();

  /// select track
  bool operator()(const reco::Track & trk ) const;

 private:

  double ptMin;

};

#endif

