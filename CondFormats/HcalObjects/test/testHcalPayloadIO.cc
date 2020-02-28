//
// test Payload I/O
//
// requires a few sed....

#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include <iostream>

#include <vector>

#ifdef ALLCLASSES
#include "CondFormats/THEPACKAGE/src/classes.h"
#else
#include "CondFormats/THEPACKAGE/interface/THEHEADER.h"
#endif

typedef THECLASS Payload;

int main(int argc, char**) {
  try {
    // this is the correct container name following cms rules (container name = C++ type name)
    //  std::string className = cond::classNameForTypeId(typeid(THECLASS));

    // for this test we use the class name THECLASS as typed by the user including space, typedefs etc
    // this makes further mapping query easier at script level....
    std::string className("THECLASS");

    edmplugin::PluginManager::Config config;
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    unsigned int nobjects = 10;
    std::vector<std::string> payTok;

    //write....
    {
      cond::DbConnection conn;
      conn.configure(cond::CmsDefaults);
      cond::DbSession session = conn.createSession();
      session.open("sqlite_file:test.db", false);

      cond::DbScopedTransaction tr(session);
      tr.start(false);
      session.createDatabase();
      unsigned int iw;
      for (iw = 0; iw < nobjects; ++iw) {
        std::shared_ptr<Payload> payload(new Payload);
        std::string pToken = session.storeObject(payload.get(), className);
        payTok.push_back(pToken);
      }

      tr.commit();
      if (payTok.size() != nobjects)
        throw std::string("not all object written!");
    }

    //read....
    {
      cond::DbConnection conn;
      conn.configure(cond::CmsDefaults);
      cond::DbSession session = conn.createSession();
      session.open("sqlite_file:test.db");
      cond::DbScopedTransaction tr(session);
      tr.start(true);

      unsigned int ir;
      for (ir = 0; ir < payTok.size(); ++ir) {
        std::shared_ptr<Payload> payload = session.getTypedObject<Payload>(payTok[ir]);
        Payload const& p = *payload;
      }

      if (ir != nobjects)
        throw std::string("not all object read!");

      tr.commit();
    }

    //read

  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    throw;
  } catch (const std::string& e) {
    std::cout << "ERROR: " << e << std::endl;
    throw;
  }

  return 0;
}
