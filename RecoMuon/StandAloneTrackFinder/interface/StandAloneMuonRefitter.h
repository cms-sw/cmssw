#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/18 08:37:34 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet;}

class StandAloneMuonRefitter {
public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonRefitter(){};

  // Operations

protected:

private:

};
#endif

