#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H

/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/03/21 13:27:22 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet;}

class StandAloneMuonSmoother {
public:
  /// Constructor
  StandAloneMuonSmoother(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonSmoother();

  // Operations

protected:

private:

};
#endif

