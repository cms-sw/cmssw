// test DataProxy
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/Utilities/interface/CondPyInterface.h"

#include "CondCore/DBCommon/interface/Exception.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/ESSources/interface/DataProxy.h"
#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/IOVService/interface/PayloadProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/DBCommon/interface/ClassID.h"

#include "CondCore/IOVService/interface/KeyList.h"
#include "CondCore/IOVService/interface/KeyListProxy.h"

#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>

#include <typeinfo>

// emulate ESSource

namespace {
  std::string buildName(const std::string& iRecordName) { return iRecordName + "@NewProxy"; }

  class CondGetterFromTag : public cond::CondGetter {
  public:
    CondGetterFromTag(const cond::CondDB& db, std::string tag) : m_db(db), m_tag(tag) {}
    virtual ~CondGetterFromTag() {}

    cond::IOVProxy get(std::string name) const override {
      // we do not use the name: still verify that is correct...
      std::cout << "keyed record name " << name << std::endl;
      return m_db.iov(m_tag);
    }

    cond::CondDB m_db;
    std::string m_tag;
  };

  // make compiler happy
  // namespace { const char * sourceRecordName_ = 0;}

}  // namespace

namespace cond {
  class CondDataProxyUtilities : public Utilities {
  public:
    CondDataProxyUtilities();
    ~CondDataProxyUtilities();
    int execute() override;
  };
}  // namespace cond

cond::CondDataProxyUtilities::CondDataProxyUtilities() : Utilities("CondDataProxy_t") {
  this->addConnectOption();
  this->addAuthenticationOptions();
  addOption<bool>("verbose", "v", "verbose");
  addOption<std::string>("tag", "t", "tag");
  addOption<std::string>("keyed", "k", "tag of keyed container");
  addOption<std::string>("record", "r", "record");
  addOption<cond::Time_t>("atTime", "a", "time of event");
}

cond::CondDataProxyUtilities::~CondDataProxyUtilities() {}

int cond::CondDataProxyUtilities::execute() {
  this->initializePluginManager();
  std::string authpath(".");
  if (this->hasOptionValue("authPath"))
    authpath = this->getAuthenticationPathValue();
  std::string connect = this->getConnectValue();
  std::string tag = this->getOptionValue<std::string>("tag");
  std::string keyed = this->getOptionValue<std::string>("keyed");
  std::string record = this->getOptionValue<std::string>("record");

  cond::Time_t time = 0;
  if (this->hasOptionValue("atTime"))
    time = this->getOptionValue<cond::Time_t>("atTime");
  cond::RDBMS rdbms(authpath, this->hasDebug());
  cond::CondDB db = rdbms.getDB(connect);
  cond::DbSession mysession = db.session();

  // here the proxy is constructed:
  cond::DataProxyWrapperBase* pb = cond::ProxyFactory::get()->create(buildName(record));
  pb->lateInit(mysession, db.iovToken(tag), "", connect, tag);

  cond::DataProxyWrapperBase::ProxyP payloadProxy = pb->proxy();

  std::cout << cond::className(typeid(*payloadProxy)) << std::endl;

  CondGetterFromTag getter(db, keyed);
  payloadProxy->loadMore(getter);

  cond::ValidityInterval iov = payloadProxy->setIntervalFor(time);
  payloadProxy->make();
  std::cout << "for " << time << ": since " << iov.first << ", till " << iov.second;
  if (payloadProxy->isValid()) {
  } else
    std::cout << ". No data";
  std::cout << std::endl;

  {
    cond::PayloadProxy<cond::KeyList> const* pp = dynamic_cast<cond::PayloadProxy<cond::KeyList>*>(payloadProxy.get());
    if (pp) {
      const cond::KeyList& keys = (*pp)();
      int n = 0;
      for (int i = 0; i < keys.size(); i++)
        if (keys.elem(i))
          n++;
      std::cout << "found " << n << " valid keyed confs" << std::endl;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  cond::CondDataProxyUtilities utilities;
  return utilities.run(argc, argv);
}
