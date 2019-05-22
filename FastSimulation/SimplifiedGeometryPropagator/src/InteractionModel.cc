#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"

std::ostream& fastsim::operator<<(std::ostream& os, const fastsim::InteractionModel& interactionModel) {
  os << std::string("interaction model with name '") << (interactionModel.name_) << std::string("'");
  return os;
}
