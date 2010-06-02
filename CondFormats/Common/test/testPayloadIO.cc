//
// test Payload I/O
//
// requires a few sed....

#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include "DataSvc/Ref.h"
#include <iostream>

#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondFormats/Common/interface/GenericSummary.h"
#include <vector>


#ifdef ALLCLASSES
#include "CondFormats/THEPACKAGE/src/classes.h"
#else
#include "CondFormats/THEPACKAGE/interface/THEHEADER.h"
#endif

typedef THECLASS Payload;

typedef cond::DataWrapper<Payload> SimplePtr;

namespace{

  bool withWrapper=false;
}

int main(int argc, char** ) {
try{

  // this is the correct container name following cms rules (container name = C++ type name) 
  //  std::string className = cond::classNameForTypeId(typeid(THECLASS));
  // std::string DSW_Name("DSW_"+className);

  // for this test we use the class name THECLASS as typed by the user including space, typedefs etc
  // this makes further mapping query easier at script level....
  std::string className("THECLASS");
  std::string DSW_Name("DSW_THECLASS");


  withWrapper = argc>1;
  
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  unsigned int nobjects=10;
  std::vector<std::string> payTok;
  std::vector<std::string> wrapTok;


  //write....
  {
    cond::DbConnection conn;
    conn.configure( cond::CmsDefaults );
    cond::DbSession session = conn.createSession();
    session.open("sqlite_file:test.db");
    
    cond::DbScopedTransaction tr(session);
    tr.start(false);
    
    unsigned int iw;
    for (iw = 0; iw < nobjects; ++iw )   {
      pool::Ref<Payload> ref = session.storeObject(new Payload,className);
      payTok.push_back(ref.toString());
      if (withWrapper) {
	pool::Ref<cond::PayloadWrapper> refW = 
	  session.storeObject(new SimplePtr(new Payload, new cond::GenericSummary(className)),DSW_Name);
	wrapTok.push_back(refW.toString());
     }
    }
    
    tr.commit();
    if (payTok.size()!=nobjects)
      throw std::string("not all object written!");
    if (withWrapper &&  wrapTok.size()!=nobjects)
      throw std::string("not all wrapped object written!");
 
  }

  //read....
  {
    cond::DbConnection conn;
    conn.configure( cond::CmsDefaults );
    cond::DbSession session = conn.createSession();
    session.open("sqlite_file:test.db");
    
    cond::DbScopedTransaction tr(session);
    tr.start(true);
    
    unsigned int ir;
    for (ir = 0; ir < payTok.size(); ++ir )   {
      pool::Ref<Payload> ref = session.getTypedObject<Payload>(payTok[ir]);
      Payload const & p = *ref;
    }

    if (ir!=nobjects)
      throw std::string("not all object read!");
 

    for (ir = 0; ir < wrapTok.size(); ++ir )   {
      pool::Ref<SimplePtr> ref = session.getTypedObject<SimplePtr>(wrapTok[ir]);
      Payload const & p = ref->data();
    }

    if (withWrapper &&  (ir!=nobjects))
      throw std::string("not all wrapped object read!");
   
    
    tr.commit();
    
  }
 


  //read 



 } catch (const std::exception& e){
  std::cout << "ERROR: " << e.what() << std::endl;
    throw;
 } catch (const std::string& e){
  std::cout << "ERROR: " << e << std::endl;
  throw;
 }

  return 0;
}

