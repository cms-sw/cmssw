#include "CalibCalorimetry/HcalAlgos/interface/HcalHardcodeParameters.h"

HcalHardcodeParameters::HcalHardcodeParameters(double pedestal,
                                               double pedestalWidth,
                                               const std::vector<double>& gain,
                                               const std::vector<double>& gainWidth,
                                               int zsThreshold,
                                               int qieType,
                                               const std::vector<double>& qieOffset,
                                               const std::vector<double>& qieSlope,
                                               int mcShape,
                                               int recoShape,
                                               double photoelectronsToAnalog,
                                               const std::vector<double>& darkCurrent)
    : pedestal_(pedestal),
      pedestalWidth_(pedestalWidth),
      gain_(gain),
      gainWidth_(gainWidth),
      zsThreshold_(zsThreshold),
      qieType_(qieType),
      qieOffset_(qieOffset),
      qieSlope_(qieSlope),
      mcShape_(mcShape),
      recoShape_(recoShape),
      photoelectronsToAnalog_(photoelectronsToAnalog),
      darkCurrent_(darkCurrent),
      doSipmRadiationDamage_(false) {}

HcalHardcodeParameters::HcalHardcodeParameters(const edm::ParameterSet& p)
    : pedestal_(p.getParameter<double>("pedestal")),
      pedestalWidth_(p.getParameter<double>("pedestalWidth")),
      gain_(p.getParameter<std::vector<double>>("gain")),
      gainWidth_(p.getParameter<std::vector<double>>("gainWidth")),
      zsThreshold_(p.getParameter<int>("zsThreshold")),
      qieType_(p.getParameter<int>("qieType")),
      qieOffset_(p.getParameter<std::vector<double>>("qieOffset")),
      qieSlope_(p.getParameter<std::vector<double>>("qieSlope")),
      mcShape_(p.getParameter<int>("mcShape")),
      recoShape_(p.getParameter<int>("recoShape")),
      photoelectronsToAnalog_(p.getParameter<double>("photoelectronsToAnalog")),
      darkCurrent_(p.getParameter<std::vector<double>>("darkCurrent")),
      doSipmRadiationDamage_(p.getParameter<bool>("doRadiationDamage")) {
  if (doSipmRadiationDamage_)
    sipmRadiationDamage_ = HcalSiPMRadiationDamage(darkCurrent_, p.getParameter<edm::ParameterSet>("radiationDamage"));
}

const double HcalHardcodeParameters::darkCurrent(unsigned index, double intlumi) const {
  if (doSipmRadiationDamage_ and intlumi > 0)
    return sipmRadiationDamage_.getDarkCurrent(intlumi, index);
  return darkCurrent_.at(index);
}
