#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Common/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Base.h"
#include "PluginManager/PluginCapabilities.h"
#include "StorageSvc/IOODatabaseFactory.h"
#include "StorageSvc/IClassLoader.h"
#include "StorageSvc/DbType.h"


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

  pool::IClassLoader* getClassLoader() {
    pool::IOODatabaseFactory* dbf = pool::IOODatabaseFactory::get();


    //This is an outrageous HACK (AA)
    // New interface needs a second parameter as void* 
    int dummy=0;
    pool::IOODatabase* db=dbf->create(pool::ROOT_StorageType.storageName(), 
                                        (void*) &dummy); 
    // Old interface require only one parameter (AA)
    // does 'db' in the next line need to be cleaned up? (JBK)
    // pool::IOODatabase* db=dbf->create(pool::ROOT_StorageType.storageName());

    if(db == 0) {

        throw cms::Exception("Configuration","EventStreamerImpl")
          << "could not get the IOODatabase from the IOODatabaseFactory\n"
          << "for storageName = " << pool::ROOT_StorageType.storageName()
          << "\n";
    }

    pool::IClassLoader* cl = db->classLoader();
    
    if(cl == 0) {
        throw cms::Exception("Configuration","EventStreamerImpl")
          << "could not get the classloader from the IOODatabase\n";
    }

    return cl;
  }


  // ---------------------
  void fillChildren(pool::IClassLoader* cl, ROOT::Reflex::Type cc, int rcnt, std::set<std::string>& classes)
  {
    rcnt--;
    FDEBUG(9) << "JBK: parent - " << getName(cc) << endl;

    while(cc.IsPointer() == true || cc.IsArray() == true) {
	//seal::Reflex::Pointer rp(*cc);
	cc = cc.ToType();
    }

    if(cc.IsFundamental()) return;

    // this probably need to be corrected also (JBK)
    if(getName(cc).find("std::basic_string<char>")==0 ||
       getName(cc).find("basic_string<char>")==0) {
	static bool has_printed = false;
	if(has_printed == false) {
	    FDEBUG(6) << "JBK: leaving " << getName(cc) << " alone\n";
	    has_printed = true;
	}
	return;
    }

    if(classes.find(getName(cc)) != classes.end()) {
	FDEBUG(6) << "WMT: skipping " << getName(cc) << ", because already processed\n";
	return;
    }

    classes.insert(getName(cc));

    if(rcnt) {
	if(cc.IsTemplateInstance()) {
	    FDEBUG(9) << "JBK: Got template instance " << getName(cc) << endl;
	    int cnt = cc.TemplateArgumentSize();
	    for(int i=0; i<cnt; ++i) {
		ROOT::Reflex::Type t = cc.TemplateArgumentAt(i);
		fillChildren(cl, t, rcnt, classes);
	    }
	}

	FDEBUG(9) << "JBK: declare members " << getName(cc) << endl;
	int mcnt = cc.MemberSize();
	for(int i=0; i<mcnt; ++i) {
	    ROOT::Reflex::Member m = cc.MemberAt(i);
	    if(m.IsTransient() || m.IsStatic()) continue;
	    if(!m.IsDataMember()) continue;

	    ROOT::Reflex::Type t = m.TypeOf();
	    fillChildren(cl, t, rcnt, classes);
	}
      }
	    
    FDEBUG(9) << "JBK: after field loop " << getName(cc) << endl;
    
    if(cl->loadClass(getName(cc)) != pool::DbStatus::SUCCESS) {
	FDEBUG(1) << "Error: could not loadClass for " << getName(cc) << endl;
	return;
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

    if (rcnt) {
        if(getName(cc).find("std::")!=0) {
	    int cnt = cc.BaseSize();
            if (cnt) {
	      FDEBUG(9) << "WMT: declare bases " << getName(cc) << endl;
	      for(int i=0; i<cnt; ++i) {
	        ROOT::Reflex::Base b = cc.BaseAt(i);

	        ROOT::Reflex::Type t = b.ToType();
	        fillChildren(cl, t, rcnt, classes);
	      }
	    }
	}
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
    std::string fname("LCGReflex/");
    fname += name;
    FDEBUG(1) << "attempting to load cap for: " << fname << endl;
    seal::PluginCapabilities::get()->load(fname);
	
    try {
      ROOT::Reflex::Type cc = ROOT::Reflex::Type::ByName(name);

      // next two lines are for explicitly causing every object to get defined
      pool::IClassLoader* cl = getClassLoader();

      std::set<std::string> classes;

	  // jbk - I'm leaving this out unless we really need it -
	  // its job is to declare each of the types to ROOT
      fillChildren(cl,cc,10000,classes);


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
    if(done==false) {
	loadCap(std::string("edm::ProdPair"));
	loadCap(std::string("edm::SendProds"));
	loadCap(std::string("edm::SendEvent"));
	loadCap(std::string("edm::SendDescs"));
	loadCap(std::string("edm::SendJobHeader"));
    }
    ClassFiller();
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
