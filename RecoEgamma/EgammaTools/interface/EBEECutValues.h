#ifndef EgammaAnalysis_ElectronTools_EBEECutValues_h
#define EgammaAnalysis_ElectronTools_EBEECutValues_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ptr.h"

template <typename T>
class EBEECutValuesT {
private:
  T barrel_;
  T endcap_;
  //this is currrently used to identify if object is barrel or endcap but may change
  const double barrelCutOff_ = 1.479;

public:
  EBEECutValuesT(const edm::ParameterSet& params, const std::string& name)
      : barrel_(params.getParameter<T>(name + "EB")), endcap_(params.getParameter<T>(name + "EE")) {}
  template <typename CandType>
  T operator()(const edm::Ptr<CandType>& cand) const {
    return isBarrel(cand) ? barrel_ : endcap_;
  }

private:
  template <typename CandType>
  const bool isBarrel(const edm::Ptr<CandType>& cand) const {
    return std::abs(cand->superCluster()->position().eta()) < barrelCutOff_;
  }
};

typedef EBEECutValuesT<double> EBEECutValues;
typedef EBEECutValuesT<int> EBEECutValuesInt;

#endif
