#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonBackwardFilter_H

/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonRefitter outermost state).
 *
 *  $Date: 2006/03/21 13:27:22 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet;}

class StandAloneMuonBackwardFilter {
public:
  /// Constructor
  StandAloneMuonBackwardFilter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonBackwardFilter(){};

  // Operations

protected:

private:

};
#endif


