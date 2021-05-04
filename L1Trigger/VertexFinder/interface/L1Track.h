#ifndef __L1Trigger_VertexFinder_L1Track_h__
#define __L1Trigger_VertexFinder_L1Track_h__

#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

namespace l1tVertexFinder {

  //! Simple wrapper class for TTTrack
  class L1Track {
  public:
    L1Track(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>& aTrack) : track_(aTrack){};
    ~L1Track(){};

    float eta() const { return track_->momentum().eta(); };
    float phi0() const { return track_->momentum().phi(); };
    float pt() const { return track_->momentum().transverse(); };
    float z0() const { return track_->POCA().z(); };

    // FIXME: Double check nPar=4 is correct
    float chi2dof() const { return track_->chi2Red(); };

    unsigned int getNumStubs() const { return track_->getStubRefs().size(); }

    const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>& getTTTrackPtr() const { return track_; };

  private:
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> track_;
  };

}  // namespace l1tVertexFinder

#endif
