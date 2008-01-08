#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Reflex/Base.h"
#include "Reflex/TypeTemplate.h"
using namespace ROOT::Reflex;
using namespace std;

namespace reco {
  bool checkMethod(const ROOT::Reflex::Member & mem, size_t args) {
    if (mem.FunctionParameterSize(true) != args) return false;
    if (mem.IsConstructor()) return false;
    if (mem.IsDestructor()) return false;
    if (mem.IsOperator()) return false;
    if (! mem.IsPublic()) return false;
    if (mem.IsStatic()) return false;
    if ( ! mem.TypeOf().IsConst() ) return false;
    if (mem.Name().substr(0, 2) == "__") return false;
    return true;
  }

  pair<Member, bool> findMethod(const Type & t, const string & name, size_t args) {
    Type type = t; 
    if (! type)  
      throw edm::Exception(edm::errors::Configuration)
	<< "no dictionary for class " << type.Name() << '\n';
    if(type.IsPointer()) type = type.ToType();
    bool found = false;
    Member member;
    for(Member_Iterator m = type.FunctionMember_Begin(); m != type.FunctionMember_End(); ++m ) {
      if(m->Name()==name && checkMethod(*m, args)) {
	member = *m;
	found = true;
	break;
      }
    }
    pair<Member, bool> mem = make_pair(member, false);
    if(! mem.first) {
      for(Base_Iterator b = type.Base_Begin(); b != type.Base_End(); ++ b)
	if((mem = findMethod(b->ToType(), name, args)).first) break;
    }
    if(!mem.first) {
      // check for edm::Ref or edm::RefToBase or edm::Ptr
      if(type.IsTemplateInstance()) {
	TypeTemplate templ = type.TemplateFamily();
	std::string name = templ.Name();
	if(name.compare("Ref") == 0 ||
	   name.compare("RefToBase") == 0 ||
	   name.compare("Ptr") == 0) {
	  mem = findMethod(type, "get", args);
	  if(!mem.first) {
	    throw edm::Exception(edm::errors::Configuration)
	      << "no member \"get\" in reference of type " << type.Name() << "\n";        
	  }
	  mem.second = true;
	}
      }
    }
    /*
    if(!mem.first) {
      throw edm::Exception(edm::errors::Configuration)
	<< "member " << name << " not found in class "  << type.Name() << "\n";        
    }
    */
    return mem;
  }
}
