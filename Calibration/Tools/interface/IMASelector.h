#ifndef UtilAlgos_IMASelector_h
#define UtilAlgos_IMASelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <iostream>

struct IMASelector {
  IMASelector(double ESCOPinMin,
              double ESeedOPoutMin,
              double PinMPoutOPinMin,
              double ESCOPinMax,
              double ESeedOPoutMax,
              double PinMPoutOPinMax,
              double EMPoutMin,
              double EMPoutMax)
      : ESCOPinMin_(ESCOPinMin),
        ESeedOPoutMin_(ESeedOPoutMin),
        PinMPoutOPinMin_(PinMPoutOPinMin),
        ESCOPinMax_(ESCOPinMax),
        ESeedOPoutMax_(ESeedOPoutMax),
        PinMPoutOPinMax_(PinMPoutOPinMax),
        EMPoutMin_(EMPoutMin),
        EMPoutMax_(EMPoutMax) {}

  template <typename T>
  bool operator()(const T& t) const {
    double pin = t.trackMomentumAtVtx().R();
    double poMpiOpi = (pin - t.trackMomentumOut().R()) / pin;
    double ESC = t.energy();
    double pOut = t.trackMomentumOut().R();
    double EseedOPout = t.eSeedClusterOverPout();
    double EoPin = t.eSuperClusterOverP();
    double EoPout = (ESC) / pOut;
    return (poMpiOpi > PinMPoutOPinMin_ && poMpiOpi < PinMPoutOPinMax_ && EseedOPout > ESeedOPoutMin_ &&
            EseedOPout < ESeedOPoutMax_ && EoPin > ESCOPinMin_ && EoPin < ESCOPinMax_ && EoPout > EMPoutMin_ &&
            EoPout < EMPoutMax_);
  }

private:
  double ESCOPinMin_, ESeedOPoutMin_, PinMPoutOPinMin_;
  double ESCOPinMax_, ESeedOPoutMax_, PinMPoutOPinMax_;
  double EMPoutMin_, EMPoutMax_;
};

namespace reco {
  namespace modules {
    template <>
    struct ParameterAdapter<IMASelector> {
      static IMASelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return IMASelector(cfg.getParameter<double>("ESCOPinMin"),
                           cfg.getParameter<double>("ESeedOPoutMin"),
                           cfg.getParameter<double>("PinMPoutOPinMin"),
                           cfg.getParameter<double>("ESCOPinMax"),
                           cfg.getParameter<double>("ESeedOPoutMax"),
                           cfg.getParameter<double>("PinMPoutOPinMax"),
                           cfg.getParameter<double>("EMPoutMin"),
                           cfg.getParameter<double>("EMPoutMax"));
      }
    };
  }  // namespace modules
}  // namespace reco

#endif
