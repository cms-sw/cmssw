#ifndef __DataFormats_TrackerCommon_ClusterTotals_h__
#define __DataFormats_TrackerCommon_ClusterTotals_h__

namespace reco { namespace utils {
    struct ClusterTotals {
      ClusterTotals();
      int strip; /// number of strip clusters
      int pixel; /// number of pixel clusters
      int stripdets; /// number of strip detectors with at least one cluster
      int pixeldets; /// number of pixel detectors with at least one cluster    
    };
} }

#endif
