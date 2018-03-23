#ifndef __L1Trigger_VertexFinder_L1fittedTrackBase_h__
#define __L1Trigger_VertexFinder_L1fittedTrackBase_h__


#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

namespace l1tVertexFinder {

  //! Simple wrapper class for TTTrack, to avoid changing other areas of packages immediately
  class L1fittedTrackBase {
  public:
  L1fittedTrackBase(const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>& aTrack) : track_(aTrack) {};
    ~L1fittedTrackBase() {};

    float eta() const { return track_->getMomentum().eta(); };
    float phi0() const { return track_->getMomentum().phi(); };
    float pt() const { return track_->getMomentum().transverse(); };
    float z0() const { return track_->getPOCA().z(); };

    // FIXME: Double check nPar=4 is correct
    float chi2dof() const { return track_->getChi2Red(); };

    unsigned int getNumStubs()  const  {return track_->getStubRefs().size();}
      /*
      // Get best matching tracking particle (=nullptr if none).
      const TP* getMatchedTP() const;
    */

    const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>& getTTTrackPtr() const {
      return track_;
    };

  private:
    edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >> track_;
    /*
    //--- Information about its association (if any) to a truth Tracking Particle.
    const TP*             matchedTP_;
    std::vector<const Stub*>   matchedStubs_;
    unsigned int          nMatchedLayers_;
    unsigned int          numStubs;
    */
  };

} // end ns l1tVertexFinder


#endif
