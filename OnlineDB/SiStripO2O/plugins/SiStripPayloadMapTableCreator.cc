#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/TimeStamp.h"


class SiStripPayloadMapTableCreator : public edm::EDAnalyzer {
public:
  explicit SiStripPayloadMapTableCreator(const edm::ParameterSet& iConfig );
  virtual ~SiStripPayloadMapTableCreator();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_configMapDb;
};


SiStripPayloadMapTableCreator::SiStripPayloadMapTableCreator(const edm::ParameterSet& iConfig):
  m_connectionPool(),
  m_configMapDb( iConfig.getParameter< std::string >("configMapDatabase") ){
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
}

SiStripPayloadMapTableCreator::~SiStripPayloadMapTableCreator() {
}

void SiStripPayloadMapTableCreator::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup ) {
  boost::shared_ptr<coral::ISessionProxy> cmDbSession = m_connectionPool.createCoralSession( m_configMapDb, true );
  coral::TableDescription mapTable;
  mapTable.setName("STRIP_CONFIG_TO_PAYLOAD_MAP");
  mapTable.insertColumn("CONFIG_HASH", coral::AttributeSpecification::typeNameForType<std::string>());
  mapTable.insertColumn("PAYLOAD_HASH", coral::AttributeSpecification::typeNameForType<std::string>());
  mapTable.insertColumn("PAYLOAD_TYPE", coral::AttributeSpecification::typeNameForType<std::string>());
  mapTable.insertColumn("CONFIG_STRING", coral::AttributeSpecification::typeNameForType<std::string>());
  mapTable.insertColumn("INSERTION_TIME", coral::AttributeSpecification::typeNameForType<coral::TimeStamp>());
  mapTable.setPrimaryKey("CONFIG_HASH");
  mapTable.setNotNullConstraint("CONFIG_HASH");
  mapTable.setNotNullConstraint("PAYLOAD_HASH");
  mapTable.setNotNullConstraint("PAYLOAD_TYPE");
  mapTable.setNotNullConstraint("CONFIG_STRING");
  mapTable.setNotNullConstraint("INSERTION_TIME");

  cmDbSession->transaction().start( false );
  cmDbSession->nominalSchema().createTable(mapTable);
  cmDbSession->transaction().commit();
}

void SiStripPayloadMapTableCreator::endJob() {
}

// ------
DEFINE_FWK_MODULE(SiStripPayloadMapTableCreator);
