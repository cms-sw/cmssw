#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H

/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/05/18 09:53:47 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

namespace edm {class ParameterSet; class EventSetup;}

class StandAloneMuonSmoother {
public:
  /// Constructor
  StandAloneMuonSmoother(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonSmoother(){};

  // Operations
  void setES(const edm::EventSetup& setup);

protected:

private:

};
#endif

