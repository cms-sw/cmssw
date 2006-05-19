#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H

/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonRefitter outermost state).
 *
 *  $Date: 2006/05/18 08:37:34 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet; class EventSetup;}

class StandAloneMuonBackwardFilter {
public:
  /// Constructor
  StandAloneMuonBackwardFilter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonBackwardFilter(){};

  // Operations
  void setES(const edm::EventSetup& setup);

protected:

private:

};
#endif


