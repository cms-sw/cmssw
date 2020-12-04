#ifndef __SCREGRESSIONCALCULATOR_H__
#define __SCREGRESSIONCALCULATOR_H__

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <vector>
#include <memory>

template <class VarCalc>
class SCRegressionCalculator {
public:
  SCRegressionCalculator(const edm::ParameterSet&);
  void update(const edm::EventSetup&);

  std::unique_ptr<VarCalc>& varCalc() { return var_calc; }

  float getCorrection(const reco::SuperCluster&) const;
  std::pair<float, float> getCorrectionWithErrors(const reco::SuperCluster&) const;

private:
  std::string eb_corr_name, ee_corr_name, eb_err_name, ee_err_name;
  const GBRWrapperRcd* gbr_record;
  edm::ESHandle<GBRForest> eb_corr, ee_corr, eb_err, ee_err;
  std::unique_ptr<VarCalc> var_calc;
};

template <class VarCalc>
SCRegressionCalculator<VarCalc>::SCRegressionCalculator(const edm::ParameterSet& conf) : gbr_record(nullptr) {
  var_calc.reset(new VarCalc());
  eb_corr_name = conf.getParameter<std::string>("regressionKeyEB");
  ee_corr_name = conf.getParameter<std::string>("regressionKeyEE");
  if (conf.existsAs<std::string>("uncertaintyKeyEB"))
    eb_err_name = conf.getParameter<std::string>("uncertaintyKeyEB");
  if (conf.existsAs<std::string>("uncertaintyKeyEE"))
    ee_err_name = conf.getParameter<std::string>("uncertaintyKeyEE");
}

template <class VarCalc>
void SCRegressionCalculator<VarCalc>::update(const edm::EventSetup& es) {
  var_calc->update(es);
  const GBRWrapperRcd& gbrfrom_es = es.get<GBRWrapperRcd>();
  if (!gbr_record || gbrfrom_es.cacheIdentifier() != gbr_record->cacheIdentifier()) {
    gbr_record = &gbrfrom_es;
    gbr_record->get(eb_corr_name.c_str(), eb_corr);
    gbr_record->get(ee_corr_name.c_str(), ee_corr);
    if (eb_err_name.size()) {
      gbr_record->get(eb_err_name.c_str(), eb_err);
    }
    if (ee_err_name.size()) {
      gbr_record->get(ee_err_name.c_str(), ee_err);
    }
  }
}

template <class VarCalc>
float SCRegressionCalculator<VarCalc>::getCorrection(const reco::SuperCluster& sc) const {
  std::vector<float> inputs;
  var_calc->set(sc, inputs);
  switch (sc.seed()->seed().subdetId()) {
    case EcalSubdetector::EcalBarrel:
      return eb_corr->GetResponse(inputs.data());
      break;
    case EcalSubdetector::EcalEndcap:
      return ee_corr->GetResponse(inputs.data());
      break;
  }
  return -1.0f;
}

template <class VarCalc>
std::pair<float, float> SCRegressionCalculator<VarCalc>::getCorrectionWithErrors(const reco::SuperCluster& sc) const {
  std::vector<float> inputs;
  var_calc->set(sc, inputs);
  switch (sc.seed()->seed().subdetId()) {
    case EcalSubdetector::EcalBarrel:
      return std::make_pair(eb_corr->GetResponse(inputs.data()), eb_err->GetResponse(inputs.data()));
      break;
    case EcalSubdetector::EcalEndcap:
      return std::make_pair(ee_corr->GetResponse(inputs.data()), ee_err->GetResponse(inputs.data()));
      break;
  }
  return std::make_pair(-1.0f, -1.0f);
}

#endif
