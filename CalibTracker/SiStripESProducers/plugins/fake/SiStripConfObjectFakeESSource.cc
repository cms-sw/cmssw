// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripConfObjectFakeESSource
//
/**\class SiStripConfObjectFakeESSource SiStripConfObjectFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripConfObjectFakeESSource.cc

 Description: "fake" SiStripConfObject ESProducer - fixed value from configuration for conf object

 Implementation:
     Port of SiStripConfObjectGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripConfObjectFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripConfObjectFakeESSource(const edm::ParameterSet&);
  ~SiStripConfObjectFakeESSource() override;

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity ) override;

  typedef std::unique_ptr<SiStripConfObject> ReturnType;
  ReturnType produce(const SiStripConfObjectRcd&);

private:
  std::vector<edm::ParameterSet> m_parameters;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripConfObjectFakeESSource::SiStripConfObjectFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripConfObjectRcd>();

  m_parameters = iConfig.getParameter<std::vector<edm::ParameterSet>>("Parameters");
}

SiStripConfObjectFakeESSource::~SiStripConfObjectFakeESSource() {}

void SiStripConfObjectFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripConfObjectFakeESSource::ReturnType
SiStripConfObjectFakeESSource::produce(const SiStripConfObjectRcd& iRecord)
{
  using namespace edm::es;

  auto confObject = std::make_unique<SiStripConfObject>();

  for ( const auto& param : m_parameters ) {
    const std::string paramType{param.getParameter<std::string>("ParameterType")};
    const std::string paramName{param.getParameter<std::string>("ParameterName")};
    if( paramType == "int" ) {
      confObject->put(paramName, param.getParameter<int32_t>("ParameterValue"));
    }
    else if( paramType == "double" ) {
      confObject->put(paramName, param.getParameter<double>("ParameterValue"));
    }
    else if( paramType == "string" ) {
      confObject->put(paramName, param.getParameter<std::string>("ParameterValue"));
    }
    else if( paramType == "bool" ) {
      confObject->put(paramName, param.getParameter<bool>("ParameterValue"));
    }
    else if( paramType == "vint32" ) {
      confObject->put(paramName, param.getParameter<std::vector<int> >("ParameterValue"));
    }
    else if( paramType == "vstring" ) {
      confObject->put(paramName, param.getParameter<std::vector<std::string> >("ParameterValue"));
    }
  }

  return confObject;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripConfObjectFakeESSource);
