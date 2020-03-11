#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"

namespace cms::cuda {

  LogSystem::~LogSystem() { edm::LogSystem(category_) << message_.str(); }

  LogAbsolute::~LogAbsolute() { edm::LogAbsolute(category_) << message_.str(); }

  LogError::~LogError() { edm::LogError(category_) << message_.str(); }

  LogProblem::~LogProblem() { edm::LogProblem(category_) << message_.str(); }

  LogImportant::~LogImportant() { edm::LogImportant(category_) << message_.str(); }

  LogWarning::~LogWarning() { edm::LogWarning(category_) << message_.str(); }

  LogPrint::~LogPrint() { edm::LogPrint(category_) << message_.str(); }

  LogInfo::~LogInfo() { edm::LogInfo(category_) << message_.str(); }

  LogVerbatim::~LogVerbatim() { edm::LogVerbatim(category_) << message_.str(); }

}  // namespace cms::cuda
