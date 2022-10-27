#include "FWCore/ParameterSet/interface/defaultModuleLabel.h"

#include <algorithm>

namespace edm {
  std::string defaultModuleLabel(std::string label) {
    // remove all colons (module type may contain namespace)
    label.erase(std::remove(label.begin(), label.end(), ':'), label.end());

    // remove everything after first '@' (used for module alternatives)
    label.erase(std::find(label.begin(), label.end(), '@'), label.end());

    // the following code originates from HLTrigger/HLTcore/interface/defaultModuleLabel.h
    // if the label is all uppercase, change it to all lowercase
    // if the label starts with more than one uppercase letter, change n-1 to lowercase
    // otherwise, change the first letter to lowercase
    unsigned int ups = 0;
    for (char c : label)
      if (std::isupper(c))
        ++ups;
      else
        break;
    if (ups > 1 and ups != label.size())
      --ups;
    for (unsigned int i = 0; i < ups; ++i)
      label[i] = std::tolower(label[i]);

    return label;
  }
}  // namespace edm
