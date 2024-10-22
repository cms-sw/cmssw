#include "GeneratorInterface/Core/interface/FortranCallback.h"
namespace gen {

  FortranCallback* FortranCallback::fInstance = nullptr;

  FortranCallback* FortranCallback::getInstance() {
    if (fInstance == nullptr)
      fInstance = new gen::FortranCallback;
    return fInstance;
  }
};  // namespace gen
