#ifndef RecoMuon_TrackingTools_MuonTrackConverter_H
#define RecoMuon_TrackingTools_MuonTrackConverter_H

/**  \class MuonTrackConverter
*
*   Auxillary class to convert a reco::Track into a Trajectory
*
*
*   $Date: 2006/09/01 15:47:04 $
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

class MuonTrackReFitter;
class MuonServiceProxy;

namespace edm {class ParameterSet; class Event; class EventSetup;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonTrackConverter {

  public:

    typedef TransientTrackingRecHit::ConstRecHitContainer  ConstRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

    /// constructor
    MuonTrackConverter(const edm::ParameterSet&, const MuonServiceProxy*);
  
    /// destructor
    virtual ~MuonTrackConverter();
  
    /// convert a reco::Track into a Trajectory
    std::vector<Trajectory> convert(const reco::Track&) const;
  
    /// convert a reco::TrackRef into a Trajectory
    std::vector<Trajectory> convert(const reco::TrackRef&) const;

    /// get container of transient rechits from a Track
    ConstRecHitContainer getTransientRecHits(const reco::Track&) const;

    /// get container of transient muon rechits from a Track
    ConstMuonRecHitContainer getTransientMuonRecHits(const reco::Track&) const;

    // get the refitter
    const MuonTrackReFitter* refitter() const {return theRefitter;}
  private:

    /// print all RecHits of a trajectory
    void printHits(const ConstRecHitContainer&) const;

  private:
  
    MuonTrackReFitter* theRefitter;

    const MuonServiceProxy *theService;
    std::string theTTRHBuilderName;

};

#endif
