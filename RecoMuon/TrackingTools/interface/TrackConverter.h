#ifndef TrackingTools_TrackConverter_H
#define TrackingTools_TrackConverter_H

/**  \class TrackConverter
*
*   Auxillary class to convert a reco::Track into a Trajectory
*
*
*   $Date: 2006/08/28 13:47:13 $
*   $Revision: 1.1 $
*
*   \author   N. Neumeister            Purdue University
*   \author   A. Everett               Purdue University
*
*/

#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

class MagneticField;
class GlobalTrackingGeometry;
class TransientTrackingRecHitBuilder;
class MuonTransientTrackingRecHitBuilder;
class MuonTrackReFitter;

namespace edm {class ParameterSet; class Event; class EventSetup;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

class TrackConverter {

  public:

    typedef TransientTrackingRecHit::ConstRecHitContainer  ConstRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

    /// constructor
    TrackConverter(const edm::ParameterSet&);
  
    /// destructor
    virtual ~TrackConverter();
  
    /// convert a reco::Track into a Trajectory
    std::vector<Trajectory> convert(const reco::Track&) const;
  
    /// convert a reco::TrackRef into a Trajectory
    std::vector<Trajectory> convert(const reco::TrackRef&) const;

    /// get container of transient rechits from a Track
    ConstRecHitContainer getTransientRecHits(const reco::Track&) const;

    /// get container of transient muon rechits from a Track
    ConstMuonRecHitContainer getTransientMuonRecHits(const reco::Track&) const;

    /// set the transient tracking rechit builders
    void setBuilder(TransientTrackingRecHitBuilder*,MuonTransientTrackingRecHitBuilder*);

    /// percolate the Event Setup
    void setES(const edm::EventSetup&);

  private:
  
    edm::ESHandle<MagneticField> theMagField;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    TransientTrackingRecHitBuilder* theTkHitBuilder;
    MuonTransientTrackingRecHitBuilder* theMuHitBuilder;
    MuonTrackReFitter* theRefitter;
  
};

#endif
