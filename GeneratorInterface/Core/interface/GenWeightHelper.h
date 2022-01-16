#ifndef GeneratorInterface_Core_GenWeightHelper_h
#define GeneratorInterface_Core_GenWeightHelper_h

#include <string>
#include <vector>
#include <map>
#include <regex>
#include <fstream>

#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "GeneratorInterface/Core/interface/WeightHelper.h"

#include <tinyxml2.h>

namespace gen {
  class GenWeightHelper : public WeightHelper {
  public:
    GenWeightHelper();
    std::vector<std::unique_ptr<gen::WeightGroupInfo>> parseWeightGroupsFromNames(std::vector<std::string> weightNames,
                                                                                  bool addUnassociatedGroup) const;
  };
}  // namespace gen

#endif
