#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/07/22 18:20:24 $
 *  $Revision: 1.10 $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

class TrajectorySeed;
class StandAloneMuonRefitter;
class StandAloneMuonBackwardFilter;
class StandAloneMuonSmoother;

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

namespace edm {class ParameterSet;}

class StandAloneMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:
    
  /// Constructor with Parameter set
  StandAloneMuonTrajectoryBuilder(const edm::ParameterSet& par);
          
  /// Destructor
  virtual ~StandAloneMuonTrajectoryBuilder();

  // Returns a vector of the reconstructed trajectories compatible with
  // the given seed.
  TrajectoryContainer trajectories(const TrajectorySeed&);

  // FIXME: not relevant here?
  virtual CandidateContainer trajectories(const reco::TrackRef&) {return CandidateContainer();}


  StandAloneMuonRefitter* refitter() const {return theRefitter;}
  //FIXME
  //  StandAloneMuonBackwardFilter* bwfilter() const {return theBWFilter;}
  StandAloneMuonRefitter* bwfilter() const {return theBWFilter;}
  StandAloneMuonSmoother* smoother() const {return theSmoother;}

  // Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);
  
  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);
  
 protected:
  
 private:
  
  StandAloneMuonRefitter* theRefitter;
  StandAloneMuonRefitter* theBWFilter;
  // FIXME
  //  StandAloneMuonBackwardFilter* theBWFilter;
  StandAloneMuonSmoother* theSmoother;

  bool doBackwardRefit;
  std::string theBWSeedType;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
};
#endif
