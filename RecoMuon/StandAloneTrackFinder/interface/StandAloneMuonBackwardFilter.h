#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H

/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonFilter outermost state).
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

class MuonServiceProxy;

namespace edm {
  class ParameterSet;
  class EventSetup;
  class Event;
}  // namespace edm

class StandAloneMuonBackwardFilter {
public:
  /// Constructor
  StandAloneMuonBackwardFilter(const edm::ParameterSet& par, const MuonServiceProxy*);

  /// Destructor
  virtual ~StandAloneMuonBackwardFilter(){};

  // Operations

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);

protected:
private:
};
#endif
