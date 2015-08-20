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
    class ParsedCutManagerBase {   
      typedef tbb::concurrent_unordered_map<std::string,void const*> FunctionMap;      
      
    public:
      ParsedCutManagerBase();

    protected:      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        std::cout << type << std::endl;
        return getFunction<T>(type.className(),func);
      }

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& twd, const std::string& func) const {
        std::cout <<  twd <<  std::endl;
        return getFunction<T>(edm::TypeID(twd.typeInfo()),func);
      }

      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& owd, const std::string& func) const {
        std::cout << owd << std::endl;
        return getFunction<T>(owd.dynamicType(),func);
      }
      
      template<typename T> 
      reco::CutOnObject<T> const* const getFunction(const std::string& type, const std::string& input) const {

        std::cout << "got to innermost call of getFunction!" << std::endl;

        static const std::string colons("::");
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
        
        expr << "bool eval(" << obj_name << " const& input) const override final {\n";
        if( obj_name != type ) {
          expr << " const " << type << "& obj = dynamic_cast<const " << type << "&>(input);\n";
        } else {
          expr << " const " << obj_name<< "& obj = input;\n";
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
        if( insert_result.second ) std::cout << "inserted a new function!" << std::endl;
        else std::cout << "got a function already compiled!" << std::endl;
        return static_cast<reco::CutOnObject<T> const*>(insert_result.first->second);
      }
      
    private:      
      static FunctionMap functions_;      
    };

    ParsedCutManagerBase::FunctionMap ParsedCutManagerBase::functions_ = ParsedCutManagerBase::FunctionMap();

    template<typename T, bool lazy>
    class ParsedCutManager : public ParsedCutManagerBase {
    public:
    ParsedCutManager() : ParsedCutManagerBase() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        return this->template ParsedCutManagerBase::getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) const {
        return this->template ParsedCutManagerBase::getFunction<T>(type,func);
      }
      
    };

    template<typename T>
    class ParsedCutManager<T,true> : public ParsedCutManagerBase {
    public:
    ParsedCutManager() : ParsedCutManagerBase() {}

      reco::CutOnObject<T> const* const getFunction(const edm::TypeID& type, const std::string& func) const {
        return this->template ParsedCutManagerBase::getFunction<T>(type,func);
      }
      
      reco::CutOnObject<T> const* const getFunction(const edm::TypeWithDict& type, const std::string& func) const {
        return this->template ParsedCutManagerBase::getFunction<T>(type,func);
      }

      reco::CutOnObject<T> const* const getFunction(const edm::ObjectWithDict& type, const std::string& func) const {
        return this->template ParsedCutManagerBase::getFunction<T>(type,func);
      }
      
    };
    
  }
}

#endif
