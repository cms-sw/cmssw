#include "IOPool/StreamerData/interface/StreamedProducts.h"
#include "IOPool/Common/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "PluginManager/PluginCapabilities.h"
#include "StorageSvc/IOODatabaseFactory.h"
#include "StorageSvc/IClassLoader.h"
#include "StorageSvc/DbType.h"


#include "TClass.h"

#include <string>

using namespace std;

namespace edm {
namespace {

  string getName(seal::reflex::Type& cc)
  {
    return cc.name(seal::reflex::SCOPED);
  }

  pool::IClassLoader* getClassLoader() {
    pool::IOODatabaseFactory* dbf = pool::IOODatabaseFactory::get();

    // does 'db' in the next line need to be cleaned up? (JBK)
    pool::IOODatabase* db=dbf->create(pool::ROOT_StorageType.storageName());

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
  void fillChildren(pool::IClassLoader* cl, seal::reflex::Type cc, int rcnt)
  {
    rcnt--;
    FDEBUG(9) << "JBK: parent - " << getName(cc) << endl;

    while(cc.isPointer() == true || cc.isArray() == true)
      {
	//seal::Reflex::Pointer rp(*cc);
	cc = cc.toType();
      }

    if(cc.isFundamental()) return;

    // this probably need to be corrected also (JBK)
    if(getName(cc).find("std::basic_string<char>")==0 ||
       getName(cc).find("basic_string<char>")==0)
      {
	static bool has_printed = false;
	if(has_printed == false)
	  {
	    FDEBUG(6) << "JBK: leaving " << getName(cc) << " alone\n";
	    has_printed = true;
	  }
	return;
      }

    if(rcnt)
      {
	if(cc.isTemplateInstance())
	  {
	    FDEBUG(9) << "JBK: Got template instance " << getName(cc) << endl;
	    int cnt = cc.templateArgumentCount();
	    for(int i=0;i<cnt;++i)
	      {
		seal::reflex::Type t = cc.templateArgument(i);
		fillChildren(cl,t,rcnt);
	      }
	  }

	FDEBUG(9) << "JBK: declare members " << getName(cc) << endl;

	int cnt = cc.memberCount();
	for(int i=0;i<cnt;++i)
	  {
	    seal::reflex::Member m = cc.member(i);
	    if(m.isTransient() || m.isStatic()) continue;
	    if(!m.isDataMember()) continue;

	    seal::reflex::Type t = m.type();
	    fillChildren(cl,t,rcnt);
	  }
      }
	    
    FDEBUG(9) << "JBK: after field loop " << getName(cc) << endl;
    
    if(cl->loadClass(getName(cc)) != pool::DbStatus::SUCCESS)
      {
	FDEBUG(1) << "Error: could not loadClass for " << getName(cc) << endl;
	return;
      }

    TClass* ttest = TClass::GetClass(getName(cc).c_str());

    if(ttest == 0) 
      {
	FDEBUG(1) << "EventStreamImpl: "
		  << "Could not get the TClass for " << getName(cc)
		  << endl;
	return;
      }

    FDEBUG(9) << "JBK: parent complete loadClass - " << getName(cc) << endl;
    if(ttest->GetClassInfo()==0)
      {
	FDEBUG(8) << "JBK: " << getName(cc) << " has no class info!\n";
      }
    if(ttest->GetStreamerInfo(1)==0)
      {
	FDEBUG(8) << "JBK: " << getName(cc)
		  << " has no streamer info version 1!\n";
      }
#if 0
    else
      ttest->GetStreamerInfo(1)->ls();

    if(ttest->GetStreamer()==0)
      {
	FDEBUG(8) << "JBK: " << getName(cc)
		  << " has no streamer!\n";
      }
#endif
  }

}
}

namespace edm {

  void loadCap(const std::string& name,bool do_children)
  {
    std::string fname("LCGReflex/");
    fname += name;
    FDEBUG(1) << "attempting to load cap for: " << fname << endl;
    seal::PluginCapabilities::get()->load(fname);
	
    try {
      seal::reflex::Type cc = seal::reflex::Type::byName(name);

      // next two lines are for explicitly causing every object to get defined
      //pool::IClassLoader* cl = getClassLoader();

	  // jbk - I'm leaving this out unless we really need it -
	  // its job is to declare each of the types to ROOT
      // fillChildren(cl,cc,do_children==true?10000:1);
    } 
    catch(...) {
      std::cerr << "Error: could not find Class object for "
		<< name << std::endl;
      return;
    }
  }

  // ---------------------

  void loadExtraClasses(bool do_children) {
    static bool done = false;
    if(done==false)
      {
	loadCap(std::string("edm::ProdPair"),do_children);
	loadCap(std::string("edm::SendProds"),do_children);
	loadCap(std::string("edm::SendEvent"),do_children);
	loadCap(std::string("edm::SendDescs"),do_children);
	loadCap(std::string("edm::SendJobHeader"),do_children);
      }
    ClassFiller();
    done=true;
  }

  namespace {
    seal::reflex::Type const getReflectClass(std::type_info const& ti) {
      seal::reflex::Type const typ = seal::reflex::Type::byTypeInfo(ti);
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
    seal::reflex::Type const typ = getReflectClass(ti);
    return getRootClass(typ.name(seal::reflex::SCOPED));
  }
}
