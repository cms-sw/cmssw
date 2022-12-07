#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H

#include "PixelCPEBase.h"
#include <vector>

class PixelCPEGenericBase : public PixelCPEBase {
public:
  struct ClusterParamGeneric : ClusterParam {
    ClusterParamGeneric(){};
    ClusterParamGeneric(const SiPixelCluster& cl) : ClusterParam(cl) {}
    // The truncation value pix_maximum is an angle-dependent cutoff on the
    // individual pixel signals. It should be applied to all pixels in the
    // cluster [signal_i = fminf(signal_i, pixmax)] before the column and row
    // sums are made. Morris
    int pixmx{};

    // These are errors predicted by PIXELAV
    float sigmay{};  // CPE Generic y-error for multi-pixel cluster
    float sigmax{};  // CPE Generic x-error for multi-pixel cluster
    float sy1{};     // CPE Generic y-error for single single-pixel
    float sy2{};     // CPE Generic y-error for single double-pixel cluster
    float sx1{};     // CPE Generic x-error for single single-pixel cluster
    float sx2{};     // CPE Generic x-error for single double-pixel cluster

    // These are irradiation bias corrections
    float deltay{};  // CPE Generic y-bias for multi-pixel cluster
    float deltax{};  // CPE Generic x-bias for multi-pixel cluster
    float dy1{};     // CPE Generic y-bias for single single-pixel cluster
    float dy2{};     // CPE Generic y-bias for single double-pixel cluster
    float dx1{};     // CPE Generic x-bias for single single-pixel cluster
    float dx2{};     // CPE Generic x-bias for single double-pixel cluster
  };

  PixelCPEGenericBase(edm::ParameterSet const& conf,
                      const MagneticField* mag,
                      const TrackerGeometry& geom,
                      const TrackerTopology& ttopo,
                      const SiPixelLorentzAngle* lorentzAngle,
                      const SiPixelGenErrorDBObject* genErrorDBObject,
                      const SiPixelLorentzAngle* lorentzAngleWidth);

  ~PixelCPEGenericBase() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

protected:
  std::unique_ptr<ClusterParam> createClusterParam(const SiPixelCluster& cl) const override;

  static void collect_edge_charges(ClusterParam& theClusterParam,  //!< input, the cluster
                                   int& q_f_X,                     //!< output, Q first  in X
                                   int& q_l_X,                     //!< output, Q last   in X
                                   int& q_f_Y,                     //!< output, Q first  in Y
                                   int& q_l_Y,                     //!< output, Q last   in Y
                                   bool truncate);

  void initializeLocalErrorVariables(float& xerr,
                                     float& yerr,
                                     bool& edgex,
                                     bool& edgey,
                                     bool& bigInX,
                                     bool& bigInY,
                                     int& maxPixelCol,
                                     int& maxPixelRow,
                                     int& minPixelCol,
                                     int& minPixelRow,
                                     uint& sizex,
                                     uint& sizey,
                                     DetParam const& theDetParam,
                                     ClusterParamGeneric const& theClusterParam) const;

  void setXYErrors(float& xerr,
                   float& yerr,
                   const bool edgex,
                   const bool edgey,
                   const unsigned int sizex,
                   const unsigned int sizey,
                   const bool bigInX,
                   const bool bigInY,
                   const bool useTemplateErrors,
                   DetParam const& theDetParam,
                   ClusterParamGeneric const& theClusterParam) const;

  const float edgeClusterErrorX_;
  const float edgeClusterErrorY_;
  bool useErrorsFromTemplates_;
  const bool truncatePixelCharge_;

  // Rechit errors in case other, more correct, errors are not vailable
  const std::vector<float> xerr_barrel_l1_, yerr_barrel_l1_, xerr_barrel_ln_;
  const std::vector<float> yerr_barrel_ln_, xerr_endcap_, yerr_endcap_;
  const float xerr_barrel_l1_def_, yerr_barrel_l1_def_, xerr_barrel_ln_def_;
  const float yerr_barrel_ln_def_, xerr_endcap_def_, yerr_endcap_def_;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H
