#ifndef __L1Trigger_L1THGCal_HGCalHistoCluteringConfig_SA_h__
#define __L1Trigger_L1THGCal_HGCalHistoCluteringConfig_SA_h__

#include <vector>

namespace l1thgcfirmware {

  class ClusterAlgoConfig {
  public:
    ClusterAlgoConfig(const double midRadius,
                      const double dr,
                      const std::vector<double>& dr_byLayer_coefficientA,
                      const std::vector<double>& dr_byLayer_coefficientB,
                      const float ptC3dThreshold)
        : midRadius_(midRadius),
          dr_(dr),
          dr_byLayer_coefficientA_(dr_byLayer_coefficientA),
          dr_byLayer_coefficientB_(dr_byLayer_coefficientB),
          ptC3dThreshold_(ptC3dThreshold) {}

    void setParameters(double midRadius,
                       double dr,
                       const std::vector<double>& dr_byLayer_coefficientA,
                       const std::vector<double>& dr_byLayer_coefficientB,
                       float ptC3dThreshold) {
      midRadius_ = midRadius;
      dr_ = dr;
      dr_byLayer_coefficientA_ = dr_byLayer_coefficientA;
      dr_byLayer_coefficientB_ = dr_byLayer_coefficientB;
      ptC3dThreshold_ = ptC3dThreshold;
    }

    void setParameters(const ClusterAlgoConfig& newConfig) {
      setParameters(newConfig.midRadius(),
                    newConfig.dr(),
                    newConfig.dr_byLayer_coefficientA(),
                    newConfig.dr_byLayer_coefficientB(),
                    newConfig.ptC3dThreshold());
    }
    double midRadius() const { return midRadius_; }
    double dr() const { return dr_; }
    const std::vector<double>& dr_byLayer_coefficientA() const { return dr_byLayer_coefficientA_; }
    const std::vector<double>& dr_byLayer_coefficientB() const { return dr_byLayer_coefficientB_; }
    float ptC3dThreshold() const { return ptC3dThreshold_; }

  private:
    double midRadius_;
    double dr_;
    std::vector<double> dr_byLayer_coefficientA_;
    std::vector<double> dr_byLayer_coefficientB_;
    float ptC3dThreshold_;
  };

}  // namespace l1thgcfirmware

#endif
