#ifndef DATAFORMATS_PHASE2TRACKERRECHIT_PHASE2TRACKERRECHIT1D_H 
#define DATAFORMATS_PHASE2TRACKERRECHIT_PHASE2TRACKERRECHIT1D_H 

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

typedef edm::Ref< edmNew::DetSetVector< Phase2TrackerCluster1D >, Phase2TrackerCluster1D > Phase2ClusterReference;

class Phase2TrackerRecHit1D {

    public:

        Phase2TrackerRecHit1D() { }
        Phase2TrackerRecHit1D(LocalPoint pos, LocalError err, Phase2ClusterReference cluster) : pos_(pos), err_(err), cluster_(cluster) { }

        LocalPoint localPosition() const { return pos_; }
        LocalError localPositionError() const { return err_; }
        Phase2ClusterReference cluster() const { return cluster_; }

    private:

        LocalPoint pos_;
        LocalError err_;
        Phase2ClusterReference cluster_;

};

typedef edmNew::DetSetVector< Phase2TrackerRecHit1D > Phase2TrackerRecHit1DCollectionNew;

#endif
