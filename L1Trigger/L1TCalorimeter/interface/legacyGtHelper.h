// legacyGtHelper.h
// Authors: Alex Barbieri
//
// This is a collection of helper methods to make sure that
// the objects passed to the legacy GT are using the proper
// Et scales and eta coordinates.

#ifndef LEGACYGTHELPER_H
#define LEGACYGTHELPER_H

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include <vector>

namespace l1t {

  void calibrateAndRankJets(CaloParamsHelper *params,
			    const std::vector<l1t::Jet> * input,
			    std::vector<l1t::Jet> *output);

  void calibrateAndRankTaus(CaloParamsHelper *params,
			    const std::vector<l1t::Tau> * input,
			    std::vector<l1t::Tau> *output);

  void JetToGtEtaScales(CaloParamsHelper *params,
			const std::vector<l1t::Jet> * input,
			std::vector<l1t::Jet> *output);

  void JetToGtPtScales(CaloParamsHelper *params,
		       const std::vector<l1t::Jet> * input,
		       std::vector<l1t::Jet> *output);


  void EGammaToGtScales(CaloParamsHelper *params,
			const std::vector<l1t::EGamma> * input,
			std::vector<l1t::EGamma> *output);

  void TauToGtEtaScales(CaloParamsHelper *params,
			const std::vector<l1t::Tau> * input,
			std::vector<l1t::Tau> *output);

  void TauToGtPtScales(CaloParamsHelper *params,
		       const std::vector<l1t::Tau> * input,
		       std::vector<l1t::Tau> *output);

  void EtSumToGtScales(CaloParamsHelper *params,
		       const std::vector<l1t::EtSum> * input,
		       std::vector<l1t::EtSum> *output);

  const unsigned int gtEta(const unsigned int iEta);
}

#endif
