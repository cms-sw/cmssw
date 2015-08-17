#include "CommonTools/Utils/interface/CutParserManager.h"

using namespace reco::exprEval;

namespace {
  static const std::string colons("::");
}

ParsedCutManagerBase::FunctionMap ParsedCutManagerBase::functions_;

void ParsedCutManagerBase::registerFunction(const edm::TypeID& type, 
                                            const std::string& func) {
}

void ParsedCutManagerBase::registerFunction(const edm::TypeWithDict& twd, 
                                            const std::string& func) {
}

void ParsedCutManagerBase::registerFunction(const edm::ObjectWithDict& owd, 
                                            const std::string& func) {
}

template<typename T> 
reco::CutOnObject<T> const* const 
ParsedCutManagerBase::getFunction(const edm::TypeID& type, 
                                  const std::string& func) {
  return getFunction<T>(type.className(),func);
}

template<typename T> 
reco::CutOnObject<T> const* const 
    ParsedCutManagerBase::getFunction(const edm::TypeWithDict& twd, 
                                      const std::string& func) {
  return getFunction<T>(twd.typeInfo(),func);
}

template<typename T> 
reco::CutOnObject<T> const* const 
  ParsedCutManagerBase::getFunction(const edm::ObjectWithDict& owd, 
                                    const std::string& func) {
  return getFunction<T>(owd.dynamicType(),func);
}
      
template<typename T> 
reco::CutOnObject<T> const* const 
  ParsedCutManagerBase::getFunction(const std::string& type,
                                    const std::string& func) {
  return nullptr;
}
