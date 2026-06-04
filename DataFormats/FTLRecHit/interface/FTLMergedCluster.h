#ifndef DataFormats_FTLRecHit_FTLMergedCluster_h
#define DataFormats_FTLRecHit_FTLMergedCluster_h

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

class FTLMergedCluster {
public:
  // Default constructor
  constexpr FTLMergedCluster()
      : id_(0), energy_(0.0), time_(0.0), timeError_(0.0), x_(0.0), y_(0.0), xError_(0.0), yError_(0.0), clusterIds_() {}

  // Constructor
  FTLMergedCluster(DetId id,
                   float energy,
                   float time,
                   float timeError,
                   float x,
                   float y,
                   float xError,
                   float yError,
                   const std::vector<DetId>& clusterIds,
                   const std::vector<FTLClusterRef>& clusterRefs)
      : id_(id),
        energy_(energy),
        time_(time),
        timeError_(timeError),
        x_(x),
        y_(y),
        xError_(xError),
        yError_(yError),
        clusterIds_(clusterIds),
        clusterRefs_(clusterRefs) {}

  // Functions to access the data members
  DetId id() const { return id_; }
  float energy() const { return energy_; }
  float time() const { return time_; }
  float timeError() const { return timeError_; }
  float x() const { return x_; }
  float y() const { return y_; }
  float xError() const { return xError_; }
  float yError() const { return yError_; }
  const std::vector<DetId>& clusterIds() const { return clusterIds_; }
  const std::vector<FTLClusterRef>& clusterRefs() const { return clusterRefs_; }
  size_t nClusters() const { return clusterIds_.size(); }

private:
  DetId id_;
  float energy_;
  float time_;
  float timeError_;
  float x_;
  float y_;
  float xError_;
  float yError_;

  std::vector<DetId> clusterIds_;
  std::vector<FTLClusterRef> clusterRefs_;
};

#endif
