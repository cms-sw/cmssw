#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelCPENNReco_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelCPENNReco_h

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplateDefs.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <string>
#include <vector>

class MagneticField;

// NN-based CPE for BPIX. FPIX hits, hits without track angles and hits for which
// the NN inference fails fall back to PixelCPEGeneric.
class PixelCPENNReco : public PixelCPEGeneric {
public:
  PixelCPENNReco(edm::ParameterSet const &conf,
                 const MagneticField *,
                 const TrackerGeometry &,
                 const TrackerTopology &,
                 const SiPixelLorentzAngle *,
                 const SiPixelGenErrorDBObject *,
                 const SiPixelLorentzAngle *,
                 const cms::Ort::ONNXRuntime *);

  ~PixelCPENNReco() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

private:
  struct ClusterParamNN : ClusterParamGeneric {
    explicit ClusterParamNN(const SiPixelCluster &cluster) : ClusterParamGeneric(cluster) {}
    float NNXrec_ = 0.f;
    float NNYrec_ = 0.f;
    float NNSigmaX_ = 0.f;
    float NNSigmaY_ = 0.f;
    bool useGeneric_ = true;
  };
  std::unique_ptr<ClusterParam> createClusterParam(const SiPixelCluster &cl) const override;

  LocalPoint localPosition(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;
  LocalError localError(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

  int PixelPreprocess(const SiPixelCluster &cluster,
                      const PixelTopology &topol,
                      const Topology::LocalTrackPred &loc_trk_pred,
                      float (&Cluster_raw)[TXSIZE][TYSIZE],
                      float (&Cluster_xRaw)[TXSIZE],
                      float (&Cluster_yRaw)[TYSIZE],
                      float (&Cluster)[TXSIZE][TYSIZE],
                      float (&Cluster_x)[TXSIZE],
                      float (&Cluster_y)[TYSIZE],
                      float &Cluster_charge,
                      int &Cluster_size,
                      int &Cluster_sizeX,
                      int &Cluster_sizeY,
                      float &ClusterCenter_x,
                      float &ClusterCenter_y,
                      int &Row_offset,
                      int &Col_offset) const;

  std::string inputTensorName_x, inputTensorName_y, anglesTensorName, cchargeTensorName, modelCategoryName;
  std::string outputTensorName_x, outputTensorName_y;

  const cms::Ort::ONNXRuntime *model;
};

#endif
