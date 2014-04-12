#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/ErrorCodes.h"
#include "CommonTools/Utils/interface/Exception.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <cassert>

using namespace std;
using reco::parser::AnyMethodArgument;

//Checks for errors which show we got the correct function be we just can't use it
static bool fatalErrorCondition(int iError)
{
   return (reco::parser::kIsNotPublic==iError ||
      reco::parser::kIsStatic==iError ||
      reco::parser::kIsFunctionAddedByROOT==iError ||
      reco::parser::kIsConstructor==iError ||
      reco::parser::kIsDestructor==iError ||
      reco::parser::kIsOperator==iError);
   
}
namespace reco {
  int checkMethod(const edm::FunctionWithDict & mem, 
                  const edm::TypeWithDict   & type,
                  const std::vector<AnyMethodArgument> &args, std::vector<AnyMethodArgument> &fixuppedArgs) {
    int casts = 0;
    if (mem.isConstructor()) return -1*parser::kIsConstructor;
    if (mem.isDestructor()) return -1*parser::kIsDestructor;
    //if (mem.isOperator()) return -1*parser::kIsOperator;  // no, some operators are allowed, e.g. operator[]
    if (! mem.isPublic()) return -1*parser::kIsNotPublic;
    if (mem.isStatic()) return -1*parser::kIsStatic;
    if ( ! mem.isConst() ) return -1*parser::kIsNotConst;
    if (mem.name().substr(0, 2) == "__") return -1*parser::kIsFunctionAddedByROOT;
    if (mem.declaringType().id() != type.id()) {
        /*std::cerr << "\nMETHOD OVERLOAD " << mem.name() <<
                       " by "   << type.Name(QUALITIED|SCOPED) <<
                       " from " << mem.declaringTy[e().Name(QUALIFIED|SCOPED) << std::endl; */
        return -1*parser::kOverloaded;
    }
    size_t minArgs = mem.functionParameterSize(true), maxArgs = mem.functionParameterSize(false);
    if ((args.size() < minArgs) || (args.size() > maxArgs)) return -1*parser::kWrongNumberOfArguments;
    /*std::cerr << "\nMETHOD " << mem.name() << " of " << mem.declaringType().name() 
        << ", min #args = " << minArgs << ", max #args = " << maxArgs 
        << ", args = " << args.size() << std::endl;*/
    if (!args.empty()) {
        std::vector<AnyMethodArgument> tmpFixups;
        size_t i = 0;
        for (auto const& param : mem) { 
            edm::TypeWithDict parameter(param);
            std::pair<AnyMethodArgument,int> fixup = boost::apply_visitor( reco::parser::AnyMethodArgumentFixup(parameter), args[i] );
            //std::cerr << "\t ARG " << i << " type is " << parameter.name() << " conversion = " << fixup.second << std::endl; 
            if (fixup.second >= 0) { 
                tmpFixups.push_back(fixup.first);
                casts += fixup.second;
            } else { 
                return -1*parser::kWrongArgumentType;
            }
            if(++i == args.size()) {
              break;
            }
        }
        fixuppedArgs.swap(tmpFixups);
    }
    /*std::cerr << "\nMETHOD " << mem.name() << " of " << mem.declaringType().name() 
        << ", min #args = " << minArgs << ", max #args = " << maxArgs 
        << ", args = " << args.size() << " fixupped args = " << fixuppedArgs.size() << "(" << casts << " implicit casts)" << std::endl; */
    return casts;
  }

  typedef pair<int,edm::FunctionWithDict> OK;
  bool nCasts(OK const& a, OK const& b) {
    return a.first < b.first;
  }


  pair<edm::FunctionWithDict, bool> findMethod(const edm::TypeWithDict & t, 
                                const string & name, 
                                const std::vector<AnyMethodArgument> &args, 
                                std::vector<AnyMethodArgument> &fixuppedArgs,
                                const char* iIterator, 
                                int& oError) {
     oError = parser::kNameDoesNotExist;
    edm::TypeWithDict type = t; 
    if (! type)  
      throw parser::Exception(iIterator)
	<< "No dictionary for class \"" << type.name() << "\".";
    while(type.isPointer() || type.isTypedef()) type = type.toType();
    type = edm::TypeWithDict(type, 0L); // strip const, volatile, c++ ref, ..

    pair<edm::FunctionWithDict, bool> mem; mem.second = false;
    int                               err_fatal = 0;

    // suitable members and number of integer->real casts required to get them
    vector<pair<int,edm::FunctionWithDict> > oks;

    // first look in base scope
    edm::TypeFunctionMembers functions(type);
    for(auto const& function : functions) {
      edm::FunctionWithDict m(function);
      if(m.name()==name) {
        int casts = checkMethod(m, type, args, fixuppedArgs);
        if (casts > -1) {
            oks.push_back( make_pair(casts,m) );
        } else {
           oError = -1*casts;
           //is this a show stopper error?
           if(fatalErrorCondition(oError) && err_fatal == 0) {
              err_fatal = oError;
           }
        }
      }
    }
    //std::cout << "At base scope (type " << (type.name()) << ") found " << oks.size() << " methods." << std::endl; 

    if (oks.empty() && err_fatal)
    {
       oError = err_fatal;
       return mem;
    }

    // found at least one method
    if (!oks.empty()) {
        if (oks.size() > 1) {
            // sort by number of conversions needed
            sort(oks.begin(), oks.end(), nCasts);

            if (oks[0].first == oks[1].first) { // two methods with same ambiguity
                throw parser::Exception(iIterator)
                    << "Can't resolve method \"" << name << "\" for class \"" << type.name() << "\", the two candidates " 
                    << oks[0].second.name() << " and " << oks[1].second.name() 
                    << " require the same number of integer->real conversions (" << oks[0].first << ").";        
            }

            // I should fixup again the args, as both good methods have pushed them on fixuppedArgs
            fixuppedArgs.clear();
            checkMethod(oks.front().second, type, args, fixuppedArgs);
        } 
        mem.first = oks.front().second;
    }

    // if nothing was found, look in parent scopes (without checking for cross-scope overloading, as it's not allowed)
    int baseError=parser::kNameDoesNotExist;
    if(! mem.first) {
      edm::TypeBases bases(type);
      for(auto const& base : bases) {
	      if((mem = findMethod(edm::BaseWithDict(base).typeOf(), name, args, fixuppedArgs,iIterator,baseError)).first) break;
	      if(fatalErrorCondition(baseError)) {
            oError = baseError;
            return mem;
	      }
      }
    }

    // otherwise see if this object is just a Ref or Ptr and we should pop it out
    if(!mem.first) {
      // check for edm::Ref or edm::RefToBase or edm::Ptr
      // std::cout << "Mem.first is null, so looking for templates from type " << type.name() << std::endl;
      if(type.isTemplateInstance()) {
         std::string name = type.templateName();
         if(name.compare("edm::Ref") == 0 ||
            name.compare("edm::RefToBase") == 0 ||
            name.compare("edm::Ptr") == 0) {
          // in this case  i think 'get' should be taken with no arguments!
          std::vector<AnyMethodArgument> empty, empty2; 
          int error;
          mem = findMethod(type, "get", empty, empty2,iIterator,error);
          if(!mem.first) {
             throw parser::Exception(iIterator)
                << "No member \"get\" in reference of type \"" << type.name() << "\".";        
          }
          mem.second = true;
         }
      }
    }
    /*
    if(!mem.first) {
      throw edm::Exception(edm::errors::Configuration)
	<< "member \""" << name << "\"" not found in class \""  << type.name() << "\"";        
    }
    */
    if(mem.first) {
       oError = parser::kNoError;
    } else {
       //use error from base check if we never found function in primary class
       if(oError == parser::kNameDoesNotExist) {
          oError = baseError;
       }
    }
    return mem;
  }
}
