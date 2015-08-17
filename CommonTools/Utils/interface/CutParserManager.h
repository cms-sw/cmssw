#ifndef __CommonTools_Utils_CutParserManager_h__
#define __CommonTools_Utils_CutParserManager_h__

#include "tbb/concurrent_unordered_map.h"

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

namespace reco {
  namespace exprEval {
    class ParsedCutManagerBase {   
      typedef tbb::concurrent_unordered_map<std::string,void const *> FunctionMap;      
      
    protected:      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func);

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func);

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& type, const std::string& func);
      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const std::string& type, const std::string& func);
      
    private:
      static FunctionMap functions_;
      void registerFunction(const edm::TypeID& type, const std::string& func);      
      void registerFunction(const edm::TypeWithDict& type, const std::string& func);
      void registerFunction(const edm::ObjectWithDict& type, const std::string& func);
    };

    template<typename T, bool lazy>
    class ParsedCutManager : ParsedCutManagerBase {
    public:
      ParsedCutManager() : ParsedCutManagerBase() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) {
        return this->template getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) {
        return this->template getFunction<T>(type,func);
      }
      
    };

    template<typename T>
    class ParsedCutManager<T,true> : ParsedCutManagerBase {
    public:
      ParsedCutManager() : ParsedCutManagerBase() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) {
        return this->template getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) {
        return this->template getFunction<T>(type,func);
      }

      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& type, const std::string& func) {
        return this->template getFunction<T>(type,func);
      }
      
    };
    
  }
}

#endif
