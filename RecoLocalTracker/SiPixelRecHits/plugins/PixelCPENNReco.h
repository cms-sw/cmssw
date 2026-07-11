#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPENNReco_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPENNReco_H

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

#include <vector>

class MagneticField;

class PixelCPENNReco : public PixelCPEGenericBase {
public:
  PixelCPENNReco(edm::ParameterSet const &conf,
                 const MagneticField *,
                 const TrackerGeometry &,
                 const TrackerTopology &,
                 const SiPixelLorentzAngle *,
                 const SiPixelGenErrorDBObject *,
                 std::vector<const tensorflow::Session *>,
                 std::vector<const tensorflow::Session *>);

  ~PixelCPENNReco() override;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

private:
  struct ClusterParamNN : ClusterParamGeneric {
    explicit ClusterParamNN(const SiPixelCluster &cluster) : ClusterParamGeneric(cluster) {}
    float NNXrec_ = 0.f;
    float NNYrec_ = 0.f;
    float NNSigmaX_ = 0.f;
    float NNSigmaY_ = 0.f;
    int ierr = 0;
  };
  std::unique_ptr<ClusterParam> createClusterParam(const SiPixelCluster &cl) const override;

  // We only need to implement measurementPosition, since localPosition() from
  // PixelCPEBase will call it and do the transformation
  // Gavril : put it back
  LocalPoint localPosition(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

  // However, we do need to implement localError().
  LocalError localError(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

  // Template storage
  // std::vector<SiPixelTemplateStore> thePixelTemp_;
  //--- DB Error Parametrization object, new light templates
  std::vector<SiPixelGenErrorStore> thePixelGenError_;
  int PixelPreprocess(const SiPixelCluster &cluster,
                      const PixelTopology &topol,
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

  std::string inputTensorName_x, inputTensorName_y, anglesTensorName_x, anglesTensorName_y, cchargeTensorName_x,
      cchargeTensorName_y;
  std::string outputTensorName_x, outputTensorName_y;

  std::vector<const tensorflow::Session *> session_x_vec;
  std::vector<const tensorflow::Session *> session_y_vec;
};

#endif
