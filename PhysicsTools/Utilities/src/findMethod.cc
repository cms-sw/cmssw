#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Base.h"
#include "Reflex/TypeTemplate.h"
using namespace ROOT::Reflex;
using namespace std;

namespace reco {
  pair<Member, bool> findMethod(const Type & t, const string & name) {
    Type type = t; 
    if (! type)  
      throw edm::Exception(edm::errors::Configuration)
	<< "no dictionary for class " << type.Name() << '\n';
    if(type.IsPointer()) type = type.ToType();
    pair<Member, bool> mem = make_pair(type.FunctionMemberByName(name), false);
    if(! mem.first) {
      for(Base_Iterator b = type.Base_Begin(); b != type.Base_End(); ++ b)
	if((mem = findMethod(b->ToType(), name)).first) break;
    }
    if(!mem.first) {
      // check for edm::Ref or edm::RefToBase or edm::Ptr
      if(type.IsTemplateInstance()) {
	TypeTemplate templ = type.TemplateFamily();
	std::string name = templ.Name();
	if(name.compare("Ref") == 0 ||
	   name.compare("RefToBase") == 0 ||
	   name.compare("Ptr") == 0) {
	  mem = findMethod(type, "get");
	  if(!mem.first) {
	    throw edm::Exception(edm::errors::Configuration)
	      << "no member \"get\" in reference of type " << type.Name() << "\n";        
	  }
	  mem.second = true;
	}
      }
    }
    if(!mem.first) {
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << name << " not found in class "  << type.Name() << "\n";        
    }
    checkMethod(type, mem.first);
    return mem;
  }

  void checkMethod(const Type & type, const ROOT::Reflex::Member & mem) {
    if (mem.FunctionParameterSize(true) != 0) 
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << type.Name() << "::" << mem.Name() 
	<< " requires " << mem.FunctionParameterSize(true) 
	<< " arguments, instead of zero\n";
    string name = mem.Name();
    string fullName = type.Name() + "::" + name;
    if (mem.IsConstructor())
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is a constructor\n";    
    if (mem.IsDestructor())
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is the destructor\n";    
    if (mem.IsOperator())
       throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is an operator\n";       
    if (! mem.IsPublic())
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is not public\n";      
    if (mem.IsStatic())
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is static\n";      
    if ( ! mem.TypeOf().IsConst() )
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is a modifier\n";        
    if (name.substr(0, 2) == "__")
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << fullName << " is an internal Reflex implementation\n";       
  }
}
