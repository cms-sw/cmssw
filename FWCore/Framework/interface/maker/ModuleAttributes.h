#ifndef FWCore_Framework_maker_ModuleAttributes_h
#define FWCore_Framework_maker_ModuleAttributes_h

namespace edm::modules {
  enum class Type { kAnalyzer, kFilter, kProducer, kOutputModule };
  enum class Concurrency { kGlobal, kLimited, kOne, kStream };
}  // namespace edm::modules
#endif
