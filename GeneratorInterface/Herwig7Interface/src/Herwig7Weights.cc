#include "GeneratorInterface/Herwig7Interface/interface/Herwig7Weights.h"

#include <ThePEG/Handlers/EventHandler.h>
#include <ThePEG/Utilities/DescriptionList.h>
#include <Herwig/Shower/ShowerHandler.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace herwig7 {
  std::vector<std::string> showerVariationNames(const ThePEG::EGPtr& eg) {
    std::vector<std::string> names;
    if (!eg || !eg->eventHandler())
      return names;
    ThePEG::tCascHdlPtr cascadeHandler = eg->eventHandler()->cascadeHandler();
    if (!cascadeHandler)
      return names;
    // Herwig is loaded at run time, so its RTTI is not available for a dynamic_cast. Use ThePEG's class
    // descriptions to check that the cascade handler is a (possibly derived) ShowerHandler instead
    const auto* handlerClass = ThePEG::DescriptionList::find(typeid(*cascadeHandler));
    const auto* showerHandlerClass = ThePEG::DescriptionList::find("Herwig::ShowerHandler");
    if (!handlerClass || !showerHandlerClass || !handlerClass->isA(*showerHandlerClass))
      return names;
    const auto* showerHandler = static_cast<const Herwig::ShowerHandler*>(cascadeHandler.operator->());
    // The variations are stored in a map, so the names are sorted
    for (const auto& variation : showerHandler->showerVariations())
      names.push_back(variation.first);
    return names;
  }

  std::vector<double> eventWeights(const ThePEG::Event& event, const std::vector<std::string>& variationNames) {
    std::vector<double> weights;
    weights.reserve(variationNames.size() + 1);
    weights.push_back(event.weight());
    const auto& optionalWeights = event.optionalWeights();
    for (const auto& name : variationNames) {
      auto weight = optionalWeights.find(name);
      if (weight == optionalWeights.end())
        throw cms::Exception("Herwig7Weights") << "Shower variation weight '" << name << "' missing in the event";
      weights.push_back(weight->second);
    }
    return weights;
  }
}  // namespace herwig7
