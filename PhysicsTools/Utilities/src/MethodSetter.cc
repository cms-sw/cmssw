#include "PhysicsTools/Utilities/src/MethodSetter.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "PhysicsTools/Utilities/src/findDataMember.h"
#include "PhysicsTools/Utilities/src/ErrorCodes.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
using namespace reco::parser;
using namespace std;
using namespace ROOT::Reflex;

void MethodSetter::operator()(const char * begin, const char * end) const {
  string name(begin, end);
  string::size_type parenthesis = name.find_first_of('(');
  std::vector<AnyMethodArgument> args;
  if(parenthesis != string::npos) {
    name.erase(parenthesis, name.size());
    if(intStack_.size()==0)
      throw edm::Exception(edm::errors::Configuration)
	<< "expected method argument, but integer stack is empty\n";    
    for(vector<AnyMethodArgument>::const_iterator i = intStack_.begin(); i != intStack_.end(); ++i)
      args.push_back(*i);
    intStack_.clear();
  }
  string::size_type endOfExpr = name.find_last_of(' ');
  if(endOfExpr != string::npos)
    name.erase(endOfExpr, name.size());
  push(name, args);
}

void MethodSetter::push(const string & name, const vector<AnyMethodArgument> & args) const {
  Type type = typeStack_.back();
  vector<AnyMethodArgument> fixups;
  int error;
  pair<Member, bool> mem = reco::findMethod(type, name, args, fixups,error);
  if(mem.first) {
     Type retType = reco::returnType(mem.first);
     typeStack_.push_back(retType);
   // check for edm::Ref, edm::RefToBase, edm::Ptr
     if(mem.second) {
        methStack_.push_back(MethodInvoker(mem.first));
        push(name, args); // we have not found the method, so we have not fixupped the arguments
      } else {
         methStack_.push_back(MethodInvoker(mem.first, fixups));
      }
  } else {
     if(error != reco::parser::kNameDoesNotExist) {
        switch(error) {
           case reco::parser::kIsNotPublic:
            throw edm::Exception(edm::errors::Configuration)
              << "method named \"" << name << "\" for type \"" 
              <<type.Name() << "\" is not publically accessible.\n";
            break;
           case reco::parser::kIsStatic:
             throw edm::Exception(edm::errors::Configuration)
               << "method named \"" << name << "\" for type \"" 
               <<type.Name() << "\" is static.\n";
             break;
           case reco::parser::kIsNotConst:
              throw edm::Exception(edm::errors::Configuration)
                << "method named \"" << name << "\" for type \"" 
                <<type.Name() << "\" is not const.\n";
              break;
           case reco::parser::kWrongNumberOfArguments:
               throw edm::Exception(edm::errors::Configuration)
                 << "method named \"" << name << "\" for type \"" 
                 <<type.Name() << "\" was passed the wrong number of arguments.\n";
               break;
           case reco::parser::kWrongArgumentType:
               throw edm::Exception(edm::errors::Configuration)
                     << "method named \"" << name << "\" for type \"" 
                     <<type.Name() << "\" was passed the wrong types of arguments.\n";
               break;
           default:  
            throw edm::Exception(edm::errors::Configuration)
             << "method named \"" << name << "\" for type \"" 
             <<type.Name() << "\" is not usable in this context.\n";
        }
     }
     //see if it is a member data
     int error;
     Member member = reco::findDataMember(type,name,error);
     if(!member) {
        switch(error) {
           case reco::parser::kNameDoesNotExist:
            throw edm::Exception(edm::errors::Configuration)
               << "no method or data member named \"" << name << "\" found for type \"" 
               <<type.Name() << "\"\n";
            break;
           case reco::parser::kIsNotPublic:
            throw edm::Exception(edm::errors::Configuration)
              << "data member named \"" << name << "\" for type \"" 
              <<type.Name() << "\" is not publically accessible.\n";
            break;
           default:
           throw edm::Exception(edm::errors::Configuration)
             << "data member named \"" << name << "\" for type \"" 
             <<type.Name() << "\" is not usable in this context.\n";
           break;
        }
     }
     typeStack_.push_back(member.TypeOf());
     methStack_.push_back(MethodInvoker(member));
  }
}
    
