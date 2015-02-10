#ifndef RecoLocalTrackerSiStripClusterizerClusterChargeCut_H
#define RecoLocalTrackerSiStripClusterizerClusterChargeCut_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<iostream>

float clusterChargeCut(const edm::ParameterSet& conf, const char * name="clusterChargeCut") {
   return conf.getParameter<edm::ParameterSet>(name).getParameter<double>("value");
}

#endif // RecoLocalTrackerSiStripClusterizerClusterChargeCut_H

