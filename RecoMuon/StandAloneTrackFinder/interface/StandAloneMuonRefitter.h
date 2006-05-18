#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/03/21 13:27:22 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet;}

class StandAloneMuonRefitter {
public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonRefitter();

  // Operations

protected:

private:

};
#endif

