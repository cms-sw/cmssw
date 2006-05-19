#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/18 09:53:47 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

// FIXME
// change FreeTrajectoryState in TSOS

namespace edm {class ParameterSet; class EventSetup;}

class StandAloneMuonRefitter {
public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonRefitter(){};

  // Operations
  
  /// Perform the inner-outward fitting
  void refit(FreeTrajectoryState& initialState);
  FreeTrajectoryState lastFTS() const {return theLastFTS;}
  
  void reset();
  void setES(const edm::EventSetup& setup);

  int getTotalChamberUsed() const {return totalChambers;}
  int getDTChamberUsed() const {return dtChambers;}
  int getCSCChamberUsed() const {return cscChambers;}
  int getRPCChamberUsed() const {return rpcChambers;}


protected:

private:
  FreeTrajectoryState theLastFTS;
  FreeTrajectoryState theLastBut1FTS;
  
  int totalChambers;
  int dtChambers;
  int cscChambers;
  int rpcChambers;
};
#endif

