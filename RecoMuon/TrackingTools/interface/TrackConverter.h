#ifndef TrackingTools_TrackConverter_H
#define TrackingTools_TrackConverter_H

/**  \class TrackConverter
*
*   Auxillary class to convert a reco::Track into a Trajectory
*
*
*   $Date: $
*   $Revision: $
*
*   \author   N. Neumeister            Purdue University
*/

#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

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
    typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;

    /// constructor
    TrackConverter(const edm::ParameterSet&);
  
    /// destructor
    virtual ~TrackConverter();
  
    /// convert a reco::Track into a Trajectory
    std::vector<Trajectory> convert(const reco::Track&) const;
  
    /// convert a reco::TrackRef into a Trajectory
    std::vector<Trajectory> convert(const reco::TrackRef&) const;

    /// get transient rechits
    ConstRecHitContainer getTransientRecHits(const reco::Track&) const;

    ///
    void setBuilder(TransientTrackingRecHitBuilder*,MuonTransientTrackingRecHitBuilder*);

    ///
    void setES(const edm::EventSetup&);

    ///
    void setEvent(const edm::Event&);
  
  private:
  
    edm::ESHandle<MagneticField> theMagField;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    TransientTrackingRecHitBuilder* theTkHitBuilder;
    MuonTransientTrackingRecHitBuilder* theMuHitBuilder;
    MuonTrackReFitter* theRefitter;
  
};

#endif
