#include "CondFormats/MTDObjects/interface/BTLElectronicsId.h"
#include "CondFormats/MTDObjects/interface/BTLReadoutMap.h"

namespace CondFormats_MTDObjects {
  struct dictionary {
    BTLElectronicsId electronicsId;
    BTLElectronicsIdPair electronicsIdPair;
    BTLReadoutMap readoutMap;
    std::unordered_map<unsigned int, BTLElectronicsIdPair> detToElec;
  };
}  // namespace CondFormats_MTDObjects
