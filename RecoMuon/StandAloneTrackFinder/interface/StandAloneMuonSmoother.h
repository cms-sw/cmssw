#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H

/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/05/18 08:37:34 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet;}

class StandAloneMuonSmoother {
public:
  /// Constructor
  StandAloneMuonSmoother(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonSmoother(){};

  // Operations

protected:

private:

};
#endif

