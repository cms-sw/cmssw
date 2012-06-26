#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/ErrorCodes.h"
#include "CommonTools/Utils/interface/Exception.h"
#include "Reflex/Base.h"
#include "Reflex/TypeTemplate.h"

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
  int checkMethod(const Reflex::Member & mem, 
                  const Reflex::Type   & type,
                  const std::vector<AnyMethodArgument> &args, std::vector<AnyMethodArgument> &fixuppedArgs) {
    int casts = 0;
    if (mem.IsConstructor()) return -1*parser::kIsConstructor;
    if (mem.IsDestructor()) return -1*parser::kIsDestructor;
    //if (mem.IsOperator()) return -1*parser::kIsOperator;  // no, some operators are allowed, e.g. operator[]
    if (! mem.IsPublic()) return -1*parser::kIsNotPublic;
    if (mem.IsStatic()) return -1*parser::kIsStatic;
    if ( ! mem.TypeOf().IsConst() ) return -1*parser::kIsNotConst;
    if (mem.Name().substr(0, 2) == "__") return -1*parser::kIsFunctionAddedByROOT;
    if (mem.DeclaringType().Id() != type.Id()) {
        /*std::cerr << "\nMETHOD OVERLOAD " << mem.Name() <<
                       " by "   << type.Name(QUALITIED|SCOPED) <<
                       " from " << mem.DeclaringType().Name(QUALIFIED|SCOPED) << std::endl; */
        return -1*parser::kOverloaded;
    }
    size_t minArgs = mem.FunctionParameterSize(true), maxArgs = mem.FunctionParameterSize(false);
    if ((args.size() < minArgs) || (args.size() > maxArgs)) return -1*parser::kWrongNumberOfArguments;
    /*std::cerr << "\nMETHOD " << mem.Name() << " of " << mem.DeclaringType().Name() 
        << ", min #args = " << minArgs << ", max #args = " << maxArgs 
        << ", args = " << args.size() << std::endl;*/
    if (!args.empty()) {
        Reflex::Type t = mem.TypeOf();
        std::vector<AnyMethodArgument> tmpFixups;
        for (size_t i = 0; i < args.size(); ++i) { 
            std::pair<AnyMethodArgument,int> fixup = boost::apply_visitor( reco::parser::AnyMethodArgumentFixup(t.FunctionParameterAt(i)), args[i] );
            //std::cerr << "\t ARG " << i << " type is " << t.FunctionParameterAt(i).Name() << " conversion = " << fixup.second << std::endl; 
            if (fixup.second >= 0) { 
                tmpFixups.push_back(fixup.first);
                casts += fixup.second;
            } else { 
                return -1*parser::kWrongArgumentType;
            }
        }
        fixuppedArgs.swap(tmpFixups);
    }
    /*std::cerr << "\nMETHOD " << mem.Name() << " of " << mem.DeclaringType().Name() 
        << ", min #args = " << minArgs << ", max #args = " << maxArgs 
        << ", args = " << args.size() << " fixupped args = " << fixuppedArgs.size() << "(" << casts << " implicit casts)" << std::endl; */
    return casts;
  }

  pair<Reflex::Member, bool> findMethod(const Reflex::Type & t, 
                                const string & name, 
                                const std::vector<AnyMethodArgument> &args, 
                                std::vector<AnyMethodArgument> &fixuppedArgs,
                                const char* iIterator, 
                                int& oError) {
     oError = parser::kNameDoesNotExist;
    Reflex::Type type = t; 
    if (! type)  
      throw parser::Exception(iIterator)
	<< "No dictionary for class \"" << type.Name() << "\".";
    while(type.IsPointer() || type.IsTypedef()) type = type.ToType();
    type = Reflex::Type(type,0); // strip const, volatile, c++ ref, ..

    pair<Reflex::Member, bool> mem; mem.second = false;

    // suitable members and number of integer->real casts required to get them
    vector<pair<int,Reflex::Member> > oks;

    // first look in base scope
    for(Reflex::Member_Iterator m = type.FunctionMember_Begin(); m != type.FunctionMember_End(); ++m ) {
      if(m->Name()==name) {
        int casts = checkMethod(*m, type, args, fixuppedArgs);
        if (casts > -1) {
            oks.push_back( make_pair(casts,*m) );
        } else {
           oError = -1*casts;
           //is this a show stopper error?
           if(fatalErrorCondition(oError)) {
              return mem;
           }
        }
      }
    }
    //std::cout << "At base scope (type " << (type.Name()) << ") found " << oks.size() << " methods." << std::endl; 
    // found at least one method
    if (!oks.empty()) {
        if (oks.size() > 1) {
            // sort by number of conversiosns needed
            sort(oks.begin(), oks.end());

            if (oks[0].first == oks[1].first) { // two methods with same ambiguity
                throw parser::Exception(iIterator)
                    << "Can't resolve method \"" << name << "\" for class \"" << type.Name() << "\", the two candidates " 
                    << oks[0].second.Name() << " and " << oks[1].second.Name() 
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
      for(Reflex::Base_Iterator b = type.Base_Begin(); b != type.Base_End(); ++ b) {
	      if((mem = findMethod(b->ToType(), name, args, fixuppedArgs,iIterator,baseError)).first) break;
	      if(fatalErrorCondition(baseError)) {
            oError = baseError;
            return mem;
	      }
      }
    }

    // otherwise see if this object is just a Ref or Ptr and we should pop it out
    if(!mem.first) {
      // check for edm::Ref or edm::RefToBase or edm::Ptr
      // std::cout << "Mem.first is null, so looking for templates from type " << type.Name() << std::endl;
      if(type.IsTemplateInstance()) {
         Reflex::TypeTemplate templ = type.TemplateFamily();
         std::string name = templ.Name();
         if(name.compare("Ref") == 0 ||
            name.compare("RefToBase") == 0 ||
            name.compare("Ptr") == 0) {
          // in this case  i think 'get' should be taken with no arguments!
          std::vector<AnyMethodArgument> empty, empty2; 
          int error;
          mem = findMethod(type, "get", empty, empty2,iIterator,error);
          if(!mem.first) {
             throw parser::Exception(iIterator)
                << "No member \"get\" in reference of type \"" << type.Name() << "\".";        
          }
          mem.second = true;
         }
      }
    }
    /*
    if(!mem.first) {
      throw edm::Exception(edm::errors::Configuration)
	<< "member \""" << name << "\"" not found in class \""  << type.Name() << "\"";        
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
