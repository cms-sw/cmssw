#include "FastSimulation/CalorimeterProperties/interface/CalorimetryConsumer.h"

CalorimetryConsumer::CalorimetryConsumer(edm::ConsumesCollector&& iC) :
    particleDataTableESToken(iC.esConsumes<edm::Transition::BeginRun>()),
    caloGeometryESToken(iC.esConsumes<edm::Transition::BeginRun>()),
    caloTopologyESToken(iC.esConsumes<edm::Transition::BeginRun>()),
    hcalDDDSimConstantsESToken(iC.esConsumes<edm::Transition::BeginRun>()),
    hcalSimulationConstantsESToken(iC.esConsumes<edm::Transition::BeginRun>())
  {}
