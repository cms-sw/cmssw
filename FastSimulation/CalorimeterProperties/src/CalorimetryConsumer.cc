#include "FastSimulation/CalorimeterProperties/interface/CalorimetryConsumer.h"

CalorimetryConsumer::CalorimetryConsumer(edm::ConsumesCollector&& iC) :
    particleDataTableESToken(iC.esConsumes()),
    caloGeometryESToken(iC.esConsumes()),
    caloTopologyESToken(iC.esConsumes()),
    hcalDDDSimConstantsESToken(iC.esConsumes()),
    hcalSimulationConstantsESToken(iC.esConsumes())
  {}
