
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductDescription.h"

#include "PluginManager/PluginCapabilities.h"
#include "RootStorageSvc/CINTType.h"
#include "StorageSvc/IOODatabaseFactory.h"
#include "StorageSvc/IClassLoader.h"
#include "StorageSvc/DbType.h"

#include "TClass.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <typeinfo>

using namespace std;

namespace edm {

  // ---------------------
  pool::IClassLoader* getClassLoader()
  {
    pool::IOODatabaseFactory* dbf = pool::IOODatabaseFactory::get();
    pool::IOODatabase* db=dbf->create(pool::ROOT_StorageType.storageName());

    if(db==0)
      {
        throw cms::Exception("Configuration","EventStreamerImpl")
          << "could not get the IOODatabase from the IOODatabaseFactory\n"
          << "for storageName = " << pool::ROOT_StorageType.storageName()
          << "\n";
      }

    pool::IClassLoader* cl = db->classLoader();
    
    if(cl==0)
      {
        throw cms::Exception("Configuration","EventStreamerImpl")
          << "could not get the classloader from the IOODatabase\n";
      }

    return cl;
  }

  // ---------------------
  void fillStreamers(ProductRegistry const& reg)
  {
    pool::IClassLoader* cl = getClassLoader();

    typedef ProductRegistry::ProductList Prods;
    const Prods& prods = reg.productList();
    Prods::const_iterator i(prods.begin()),e(prods.end());

    for(;i!=e;++i)
      {
	string name = i->second.fullClassName_;
	
    	string fname("LCGDict/");
    	fname+=name;
    	seal::PluginCapabilities::get()->load(fname);

	if(cl->loadClass(name)!=pool::DbStatus::SUCCESS)
	  {
	    cerr << "EventStreamImpl: "
		 << "Could not loadClass for " << name
		 << endl;
	    continue;
	  }
	
	TClass* ttest = TClass::GetClass(name.c_str());
	
	if(ttest==0)
	  {
	    cerr << "EventStreamImpl: "
		 << "Could not get the TClass for " << name
		 << endl;
	    continue;
	  }
      }
  }

#if 0
  string prodpair_name("LCGDict/edm::ProdPair");
  string vprodpair_name("LCGDict/std::vector<edm::ProdPair>");
  string sendevent_name("LCGDict/edm::SendEvent");
#endif


  namespace
  {
    seal::reflect::Class const * getReflectClass(const std::type_info& ti)
    {
      seal::reflect::Class const * typ = 
	seal::reflect::Class::forTypeinfo(ti);
      
      if(typ==0)
	{
	  throw cms::Exception("Configuration","getReflectClass")
	    << "could not find reflection class for "
	    << ti.name()
	    << "\n";
	}
      
      return typ;
    }

    TClass* getRootClass(const std::string& name)
    {
      TClass* tc = TClass::GetClass(name.c_str());    
      
      // get ROOT TClass for this product
      // CINT::Type* cint_type = CINT::Type::get(typ_ref);
      // tc_ = cint_type->rootClass();
      // TClass* tc = TClass::GetClass(typeid(se));
      // tc_ = TClass::GetClass("edm::SendEvent");
      
      if(tc==0)
	{
	  throw cms::Exception("Configuration","getRootClass")
	    << "could not find TClass for " << name
	    << "\n";
	}
      
      return tc;
    }
  }

  // ---------------------
  TClass* getTClass(const std::type_info& ti)
  {
    seal::reflect::Class const* typ = getReflectClass(ti);
    return getRootClass(typ->fullName());
  }
   
  // ---------------------
  TClass* loadClass(pool::IClassLoader* cl, const std::type_info& ti)
  {
    seal::reflect::Class const * typ = getReflectClass(ti);

    string fname("LCGDict/");
    fname+=typ->fullName();
    seal::PluginCapabilities::get()->load(fname);

    if(cl->loadClass(typ->fullName())!=pool::DbStatus::SUCCESS)
      {
        throw cms::Exception("Configuration","edm::loadClass")
          << "could not do loadClass for " << fname
          << "\n";
      }
    
    TClass* tc = getRootClass(typ->fullName()); 
    // cerr << "TClass name " << tc->GetName() << endl;

    return tc;
  }

  void loadExtraClasses()
  {
    pool::IClassLoader* cl = getClassLoader();
    loadClass(cl,typeid(ProdPair));
    loadClass(cl,typeid(SendProds));
    loadClass(cl,typeid(SendEvent));
    loadClass(cl,typeid(SendDescs));
    loadClass(cl,typeid(SendJobHeader));
    loadClass(cl,typeid(ProductDescription));
    loadClass(cl,typeid(BranchEntryDescription));
  }
  
  
}
