#ifndef GeneratorInterface_Herwig7Interface_Herwig7Weights_h
#define GeneratorInterface_Herwig7Interface_Herwig7Weights_h

#include <string>
#include <vector>

#include <ThePEG/Repository/EventGenerator.h>
#include <ThePEG/EventRecord/Event.h>

namespace herwig7 {
  // Names of the shower variations configured in the shower handler (ShowerHandler:AddVariation), in the
  // order in which eventWeights returns them. They are known once the generator is initialised
  std::vector<std::string> showerVariationNames(const ThePEG::EGPtr& eg);

  // The event weight followed by the named shower variation weights. The other optional weights of the
  // event are dropped: the LHE readers copy the LHE weights (stored in the LHEEventProduct already) and
  // per-event LHE information such as clustering scales into them, so their number and order change
  // from event to event
  std::vector<double> eventWeights(const ThePEG::Event& event, const std::vector<std::string>& variationNames);
}  // namespace herwig7

#endif
