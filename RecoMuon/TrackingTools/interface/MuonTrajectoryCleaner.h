#ifndef RecoMuon_TrackingTools_MuonTrajectoryCleaner_H
#define RecoMuon_TrackingTools_MuonTrajectoryCleaner_H

/** \class MuonTrajectoryCleaner
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

class TrajectoryContainer;

class MuonTrajectoryCleaner {
public:
  /// Constructor
  MuonTrajectoryCleaner(){};

  /// Destructor
  virtual ~MuonTrajectoryCleaner(){};

  // Operations
  void clean(TrajectoryContainer &muonTrajectories){}; //used by reference...

protected:

private:

};
#endif

