#ifndef __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__
#define __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__

#include <memory>
#include <string>

class GBRForest;

namespace pat {
  class Muon;
}

namespace edm {
  class FileInPath;
}

namespace pat {
  class SoftMuonMvaEstimator{
  public:

    SoftMuonMvaEstimator(const edm::FileInPath& weightsfile);

    ~SoftMuonMvaEstimator();

    float computeMva(const pat::Muon& imuon) const;

  private:

    std::unique_ptr<const GBRForest> gbrForest_;

  };
}
#endif
