#include "CalibCalorimetry/HcalAlgos/interface/HcalHardcodeParameters.h"

HcalHardcodeParameters::HcalHardcodeParameters(const double pedestal,
                                               const double pedestalWidth,
                                               const std::vector<double>& gain,
                                               const std::vector<double>& gainWidth,
                                               const int zsThreshold,
                                               const int qieType,
                                               const std::vector<double>& qieOffset,
                                               const std::vector<double>& qieSlope,
                                               const int mcShape,
                                               const int recoShape,
                                               const double photoelectronsToAnalog,
                                               const std::vector<double>& darkCurrent,
                                               const std::vector<double>& noiseCorrelation,
                                               const double noiseTh,
                                               const double seedTh)
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
      noiseCorrelation_(noiseCorrelation),
      doSipmRadiationDamage_(false),
      noiseThreshold_(noiseTh),
      seedThreshold_(seedTh) {}

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
      noiseCorrelation_(p.getParameter<std::vector<double>>("noiseCorrelation")),
      doSipmRadiationDamage_(p.getParameter<bool>("doRadiationDamage")),
      noiseThreshold_(p.getParameter<double>("noiseThreshold")),
      seedThreshold_(p.getParameter<double>("seedThreshold")) {
  if (doSipmRadiationDamage_)
    sipmRadiationDamage_ = HcalSiPMRadiationDamage(darkCurrent_, p.getParameter<edm::ParameterSet>("radiationDamage"));
}

double HcalHardcodeParameters::darkCurrent(unsigned index, double intlumi) const {
  if (doSipmRadiationDamage_ and intlumi > 0)
    return sipmRadiationDamage_.getDarkCurrent(intlumi, index);
  return darkCurrent_.at(index);
}

double HcalHardcodeParameters::noiseCorrelation(unsigned index) const { return noiseCorrelation_.at(index); }
