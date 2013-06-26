#include "CommonTools/Utils/src/MethodSetter.h"
#include "CommonTools/Utils/src/returnType.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/findDataMember.h"
#include "CommonTools/Utils/src/ErrorCodes.h"
#include "CommonTools/Utils/interface/Exception.h"
#include <string>
using namespace reco::parser;
using namespace std;

void MethodSetter::operator()(const char * begin, const char * end) const {
  string name(begin, end);
  string::size_type parenthesis = name.find_first_of('(');
  if (*begin == '[' || *begin == '(') {
    name.insert(0, "operator..");           // operator..[arg];
    parenthesis = 10;                       //           ^--- idx = 10
    name[8] = *begin;                       // operator[.[arg];
    name[9] =  name[name.size()-1];         // operator[][arg];
    name[10] = '(';                         // operator[](arg];
    name[name.size()-1] = ')';              // operator[](arg);    
    // we don't actually need the last two, but just for extra care
    //std::cout << "Transformed {" << string(begin,end) << "} into {"<< name <<"}" << std::endl;
  }
  std::vector<AnyMethodArgument> args;
  if(parenthesis != string::npos) {
    name.erase(parenthesis, name.size());
    if(intStack_.size()==0)
      throw Exception(begin)
	<< "expected method argument, but non given.";    
    for(vector<AnyMethodArgument>::const_iterator i = intStack_.begin(); i != intStack_.end(); ++i)
      args.push_back(*i);
    intStack_.clear();
  }
  string::size_type endOfExpr = name.find_last_of(' ');
  if(endOfExpr != string::npos)
    name.erase(endOfExpr, name.size());
  //std::cerr << "Pushing [" << name << "] with " << args.size() << " args " << (lazy_ ? "(lazy)" : "(immediate)") << std::endl;
  if (lazy_) lazyMethStack_.push_back(LazyInvoker(name, args)); // for lazy parsing we just push method name and arguments
  else push(name, args,begin);  // otherwise we really have to resolve the method
  //std::cerr << "Pushed [" << name << "] with " << args.size() << " args " << (lazy_ ? "(lazy)" : "(immediate)") << std::endl;
}

bool MethodSetter::push(const string & name, const vector<AnyMethodArgument> & args, const char* begin,bool deep) const {
  edm::TypeWithDict type = typeStack_.back();
  vector<AnyMethodArgument> fixups;
  int error;
  pair<edm::FunctionWithDict, bool> mem = reco::findMethod(type, name, args, fixups,begin,error);
  if(mem.first) {
     edm::TypeWithDict retType = reco::returnType(mem.first);
     if(!retType) {
        throw Exception(begin)
     	<< "member \"" << mem.first.name() << "\" return type is invalid:\n" 
        << "  member type: \"" <<  mem.first.typeOf().qualifiedName() << "\"\n"
     	<< "  return type: \"" << mem.first.returnType().qualifiedName() << "\"\n";
        
     }
     typeStack_.push_back(retType);
     // check for edm::Ref, edm::RefToBase, edm::Ptr
     if(mem.second) {
        //std::cout << "Mem.second, so book " << mem.first.name() << " without fixups." << std::endl;
        methStack_.push_back(MethodInvoker(mem.first));
        if (deep) push(name, args,begin); // note: we have not found the method, so we have not fixupped the arguments
        else return false;
      } else {
        //std::cout << "Not mem.second, so book " << mem.first.name() << " with #args = " << fixups.size() << std::endl;
        methStack_.push_back(MethodInvoker(mem.first, fixups));
      }
  } else {
     if(error != reco::parser::kNameDoesNotExist) {
        switch(error) {
           case reco::parser::kIsNotPublic:
            throw Exception(begin)
              << "method named \"" << name << "\" for type \"" 
              <<type.name() << "\" is not publically accessible.";
            break;
           case reco::parser::kIsStatic:
             throw Exception(begin)
               << "method named \"" << name << "\" for type \"" 
               <<type.name() << "\" is static.";
             break;
           case reco::parser::kIsNotConst:
              throw Exception(begin)
                << "method named \"" << name << "\" for type \"" 
                <<type.name() << "\" is not const.";
              break;
           case reco::parser::kWrongNumberOfArguments:
               throw Exception(begin)
                 << "method named \"" << name << "\" for type \"" 
                 <<type.name() << "\" was passed the wrong number of arguments.";
               break;
           case reco::parser::kWrongArgumentType:
               throw Exception(begin)
                     << "method named \"" << name << "\" for type \"" 
                     <<type.name() << "\" was passed the wrong types of arguments.";
               break;
           default:  
            throw Exception(begin)
             << "method named \"" << name << "\" for type \"" 
             <<type.name() << "\" is not usable in this context.";
        }
     }
     //see if it is a member data
     int error;
    edm::MemberWithDict member(reco::findDataMember(type,name,error));
     if(!member) {
        switch(error) {
           case reco::parser::kNameDoesNotExist:
            throw Exception(begin)
               << "no method or data member named \"" << name << "\" found for type \"" 
               <<type.name() << "\"";
            break;
           case reco::parser::kIsNotPublic:
            throw Exception(begin)
              << "data member named \"" << name << "\" for type \"" 
              <<type.name() << "\" is not publically accessible.";
            break;
           default:
           throw Exception(begin)
             << "data member named \"" << name << "\" for type \"" 
             <<type.name() << "\" is not usable in this context.";
           break;
        }
     }
     typeStack_.push_back(member.typeOf());
     methStack_.push_back(MethodInvoker(member));
  }
  return true;
}
    
