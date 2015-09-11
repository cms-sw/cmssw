#ifndef __CommonTools_Utils_CutParserManager_h__
#define __CommonTools_Utils_CutParserManager_h__

#include "tbb/concurrent_unordered_map.h"

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

namespace reco {
  namespace exprEval {

    namespace {
      static const std::string colons("::");
    };

    template<typename T, bool isLazy> class ParsedCutManager;

    class ParsedCutManagerImpl {   
      typedef tbb::concurrent_unordered_map<std::string,void const*> FunctionMap;      
      
    public:
      template<typename T, bool isLazy> friend class ParsedCutManager;

      ParsedCutManagerImpl();
    
    private:      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        return getFunction<T>(type.className(),func);
      }

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& twd, const std::string& func) const {
        return getFunction<T>(edm::TypeID(twd.typeInfo()),func);
      }

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& owd, const std::string& func) const {
        return getFunction<T>(owd.dynamicType(),func);
      }
      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const std::string& type, const std::string& input) const {
        
        typedef reco::CutOnObject<T> CutType;
        edm::TypeID the_cut_type(typeid(CutType));
        edm::TypeID the_obj_type(typeid(T));
        const std::string& obj_name = the_obj_type.className();
        
        bool justBlanks = true;
        for( auto chr : input ) {
          if( !isspace(chr) ) {
            justBlanks = false;
            break;
          }
        }
        
        const std::string func = ( justBlanks ? "true" : input );

        const std::string key = obj_name + colons + type + colons + func;  
        auto found = functions_.find(key);
        if( found != functions_.cend() ) {
          return static_cast<reco::CutOnObject<T> const*>(found->second);
        }
        
        std::stringstream expr;     
        
        if( obj_name != type ) {
          expr << "bool eval(" << obj_name << " const& input) const override final {\n";
          expr << " const " << type << "& obj = dynamic_cast<const " << type << "&>(input);\n";
        } else {
          expr << "bool eval(" << obj_name << " const& obj) const override final {\n";
        }
        
        expr << " return ( " << func << " );\n";
        expr << "}\n";
        const std::string strexpr = expr.str();
        
        CutType const* the_func = nullptr; 
        
        try {
          reco::ExpressionEvaluator builder("CommonTools/CandUtils",the_cut_type.className().c_str(),strexpr.c_str());
          the_func = builder.expr<CutType>();
        } catch( cms::Exception& e) {
          the_func = nullptr;
          throw edm::Exception(edm::errors::Configuration)
            <<"ExpressionEvaluator compilation error: "<< e.what() << std::endl;
        }
        
        void const* to_add = static_cast<void const*>(the_func);
        auto insert_result = functions_.insert(std::make_pair(key,to_add) );
        return static_cast<reco::CutOnObject<T> const*>(insert_result.first->second);
      }
      
      mutable FunctionMap functions_;
    };

    template<typename T, bool lazy>
    class ParsedCutManager {
    public:
    ParsedCutManager() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        return impl_.template getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) const {
        return impl_.template getFunction<T>(type,func);
      }

      static ParsedCutManager<T,lazy> const* const get() {
        static ParsedCutManager<T,lazy> instance;
        return &instance;
      } 

    private:
      const ParsedCutManagerImpl impl_;
    };

    template<typename T>
    class ParsedCutManager<T,true> {
    public:
    ParsedCutManager() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        return impl_.template getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) const {
        return impl_.template getFunction<T>(type,func);
      }

      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& type, const std::string& func) const {
        return impl_.template getFunction<T>(type,func);
      }

      static ParsedCutManager<T,true> const* const get() {
        static ParsedCutManager<T,true> instance;
        return &instance;
      } 

    private:
      const ParsedCutManagerImpl impl_;      
    };
    
  }
}

#endif
