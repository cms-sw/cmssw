#include "RecoTracker/MkFitCore/interface/MkBuilderWrapper.h"
#include "RecoTracker/MkFitCore/interface/MkBuilder.h"

namespace mkfit {
  MkBuilderWrapper::MkBuilderWrapper(bool silent) : builder_(MkBuilder::make_builder(silent)) {}

  MkBuilderWrapper::~MkBuilderWrapper() {}

  void MkBuilderWrapper::populate() { MkBuilder::populate(); }
}  // namespace mkfit
