#ifndef PhysicsTools_PatAlgos_MuonMvaIDEstimator_h
#define PhysicsTools_PatAlgos_MuonMvaIDEstimator_h

#include <memory>
#include <string>
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
//

namespace pat {
  class Muon;
}

namespace edm {
  class FileInPath;
}

namespace pat {
  class MuonMvaIDEstimator {
  public:
    MuonMvaIDEstimator(const edm::FileInPath &weightsfile);
    ~MuonMvaIDEstimator() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions &);
    static void globalEndJob(const cms::Ort::ONNXRuntime *);
    std::vector<float> computeMVAID(const pat::Muon &imuon) const;

  private:
    std::vector<std::string> flav_names_;   // names of the output scores
    std::vector<std::string> input_names_;  // names of each input group - the ordering is important!
    std::unique_ptr<const cms::Ort::ONNXRuntime> randomForest_;
  };
};  // namespace pat
#endif
