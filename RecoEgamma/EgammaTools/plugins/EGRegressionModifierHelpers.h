#ifndef RecoEgamma_EgammaTools_EGRegressionModifier_H
#define RecoEgamma_EgammaTools_EGRegressionModifier_H

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <string>
#include <vector>

std::vector<const GBRForestD*> retrieveGBRForests(edm::EventSetup const& evs, std::vector<std::string> const& names);

#endif
