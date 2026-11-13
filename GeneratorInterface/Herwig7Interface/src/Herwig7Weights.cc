#include "GeneratorInterface/Herwig7Interface/interface/Herwig7Weights.h"

#include <ThePEG/Handlers/EventHandler.h>
#include <ThePEG/Repository/BaseRepository.h>
#include <ThePEG/Utilities/DescriptionList.h>
#include <Herwig/Shower/ShowerHandler.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace {
  // Herwig is loaded at run time, so its RTTI is not available for a dynamic_cast. Use ThePEG's class
  // descriptions to check that the object is a (possibly derived) ShowerHandler instead
  const Herwig::ShowerHandler* asShowerHandler(ThePEG::tcIBPtr object) {
    if (!object)
      return nullptr;
    const auto* objectClass = ThePEG::DescriptionList::find(typeid(*object));
    const auto* showerHandlerClass = ThePEG::DescriptionList::find("Herwig::ShowerHandler");
    if (!objectClass || !showerHandlerClass || !objectClass->isA(*showerHandlerClass))
      return nullptr;
    return static_cast<const Herwig::ShowerHandler*>(object.operator->());
  }
}  // namespace

namespace herwig7 {
  std::vector<std::string> showerVariationNames(const ThePEG::EGPtr& eg) {
    std::vector<std::string> names;
    if (!eg || !eg->eventHandler())
      return names;
    ThePEG::tCascHdlPtr cascadeHandler = eg->eventHandler()->cascadeHandler();
    const auto* showerHandler = asShowerHandler(cascadeHandler);
    // The variations are stored in a map, so the names are sorted
    if (showerHandler) {
      for (const auto& variation : showerHandler->showerVariations())
        names.push_back(variation.first);
    }

    // Variations given to the default shower handler are silently ignored if another handler showers the
    // events, e.g. the FxFxHandler of merged LHE samples. An unused handler is not kept in the event
    // generator, so look it up in the repository, which is filled if the input was read in this job
    const std::string defaultHandlerName = "/Herwig/Shower/ShowerHandler";
    ThePEG::tcIBPtr defaultObject = eg->getPointer(defaultHandlerName);
    if (!defaultObject)
      defaultObject = ThePEG::BaseRepository::GetPointer(defaultHandlerName);
    const auto* defaultHandler = asShowerHandler(defaultObject);
    if (names.empty() && defaultHandler && defaultHandler != showerHandler &&
        !defaultHandler->showerVariations().empty())
      throw cms::Exception("Configuration")
          << "Shower variations are set for /Herwig/Shower/ShowerHandler, but the events are showered by "
          << (cascadeHandler ? cascadeHandler->fullName() : std::string("no cascade handler"))
          << ", which has none. Add the variations to that handler instead (for FxFxHandler, use "
             "hw_PSWeights_mg_merging_settings from Herwig7PSWeightsSettings_cfi)";
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
