#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Base.h"
#include "Cintex/Cintex.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"


#include "TClass.h"

#include <string>
#include <set>

using namespace std;

namespace edm {
namespace {

  std::string getName(ROOT::Reflex::Type& cc)
  {
    return cc.Name(ROOT::Reflex::SCOPED);
  }

  // ---------------------
  void fillChildren(ROOT::Reflex::Type cc, int rcnt, std::set<std::string>& classes)
  {
    static std::string const fname("LCGReflex/");
    rcnt--;
    FDEBUG(9) << "JBK: parent - " << getName(cc) << endl;

    // ToType strips const, volatile, array, pointer, reference, etc.,
    // and also translates typedefs.
    // To be safe, we do this recursively until we either get a null type
    // or the same type.
    ROOT::Reflex::Type null;
    for (ROOT::Reflex::Type x = cc.ToType(); x != null && x != cc; cc = x, x = cc.ToType()) {}

    if(cc.IsFundamental()) return;
    if(cc.IsEnum()) return;
    if(cc == null) return;

    if(classes.find(getName(cc)) != classes.end()) {
	FDEBUG(6) << "WMT: skipping " << getName(cc) << ", because already processed\n";
	return;
    }

    classes.insert(getName(cc));

    if(rcnt) {
	if(cc.IsTemplateInstance()) {
	    FDEBUG(9) << "JBK: Got template instance " << getName(cc) << endl;
	    int cnt = cc.TemplateArgumentSize();
	    for(int i = 0; i < cnt; ++i) {
		ROOT::Reflex::Type t = cc.TemplateArgumentAt(i);
		fillChildren(t, rcnt, classes);
	    }
	}

        if(getName(cc).find("std::") != 0) {
	  int mcnt = cc.MemberSize();
	  if (mcnt) {
	    FDEBUG(9) << "JBK: declare members " << getName(cc) << endl;
	    for(int i = 0; i < mcnt; ++i) {
	      ROOT::Reflex::Member m = cc.MemberAt(i);
	      if(m.IsTransient() || m.IsStatic()) continue;
	      if(!m.IsDataMember()) continue;
	      ROOT::Reflex::Type t = m.TypeOf();
	      fillChildren(t, rcnt, classes);
	    }
	  }
	  int cnt = cc.BaseSize();
          if (cnt) {
	    FDEBUG(9) << "WMT: declare bases " << getName(cc) << endl;
	    for(int i = 0; i < cnt; ++i) {
	      ROOT::Reflex::Base b = cc.BaseAt(i);

	      ROOT::Reflex::Type t = b.ToType();
	      fillChildren(t, rcnt, classes);
	    }
	  }
	}
    }
	    
    FDEBUG(9) << "JBK: after field loop " << getName(cc) << endl;
    
    try {
      edmplugin::PluginCapabilities::get()->load(fname + getName(cc));
    } catch (cms::Exception const&) {
    }

    TClass* ttest = TClass::GetClass(getName(cc).c_str());

    if(ttest == 0) {
	FDEBUG(1) << "EventStreamImpl: "
		  << "Could not get the TClass for " << getName(cc)
		  << endl;
	return;
    }

    FDEBUG(9) << "JBK: parent complete loadClass - " << getName(cc) << endl;
    if(ttest->GetClassInfo()==0) {
	FDEBUG(8) << "JBK: " << getName(cc) << " has no class info!\n";
    }
    if(ttest->GetStreamerInfo(1)==0) {
	FDEBUG(8) << "JBK: " << getName(cc)
		  << " has no streamer info version 1!\n";
    }

#if 0
    else
      ttest->GetStreamerInfo(1)->ls();

    if(ttest->GetStreamer()==0) {
	FDEBUG(8) << "JBK: " << getName(cc)
		  << " has no streamer!\n";
    }
#endif
  }

}
}

namespace edm {

  void loadCap(const std::string& name)
  {
    std::string const fname("LCGReflex/");

    FDEBUG(1) << "attempting to load cap for: " << name << endl;
	
    try {
      edmplugin::PluginCapabilities::get()->load(fname + name);
      ROOT::Reflex::Type cc = ROOT::Reflex::Type::ByName(name);

      std::set<std::string> classes;

	  // jbk - I'm leaving this out unless we really need it -
	  // its job is to declare each of the types to ROOT
      fillChildren(cc,10000,classes);


      TClass* ttest = TClass::GetClass(getName(cc).c_str());

      if(ttest != 0) 
	    ttest->BuildRealData();
    } 
    catch(...) {
      std::cerr << "Error: could not find Class object for "
		<< name << std::endl;
      return;
    }
  }

  void doBuildRealData(const std::string& name)
  {
  	FDEBUG(3) << "doing BuildRealData for " << name << "\n";
      ROOT::Reflex::Type cc = ROOT::Reflex::Type::ByName(name);
      TClass* ttest = TClass::GetClass(getName(cc).c_str());
      if(ttest != 0) 
	    ttest->BuildRealData();
	  else
	  {
	  	throw cms::Exception("Configuration")
			<< "Could not find TClass for " << name << "\n";
	  }
  }
  // ---------------------

  void loadExtraClasses() {
    static bool done = false;
    if (done == false) {
	loadCap(std::string("edm::ProdPair"));
	loadCap(std::string("std::vector<edm::ProdPair>"));
	loadCap(std::string("edm::SendEvent"));
	loadCap(std::string("std::vector<edm::BranchDescription>"));
	loadCap(std::string("edm::SendJobHeader"));
    }
    ROOT::Cintex::Cintex::Enable();
    done=true;
  }

  namespace {
    ROOT::Reflex::Type const getReflectClass(std::type_info const& ti) {
      ROOT::Reflex::Type const typ = ROOT::Reflex::Type::ByTypeInfo(ti);
      return typ;
    }

    TClass* getRootClass(std::string const& name) {
      TClass* tc = TClass::GetClass(name.c_str());    
      
      // get ROOT TClass for this product
      // CINT::Type* cint_type = CINT::Type::get(typ_ref);
      // tc_ = cint_type->rootClass();
      // TClass* tc = TClass::GetClass(typeid(se));
      // tc_ = TClass::GetClass("edm::SendEvent");
      
      if(tc == 0) {
	throw cms::Exception("Configuration","getRootClass")
	  << "could not find TClass for " << name
	  << "\n";
      }
      
      return tc;
    }
  }

  // ---------------------
  TClass* getTClass(std::type_info const& ti) {
    ROOT::Reflex::Type const typ = getReflectClass(ti);
    return getRootClass(typ.Name(ROOT::Reflex::SCOPED));
  }
}
