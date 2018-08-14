#ifndef ECAL_MAPPINGELECTRONICS_HANDLER_H
#define ECAL_MAPPINGELECTRONICS_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <ctime>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"



#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"


#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

class EcalMappingElectronicsHandler : public popcon::PopConSourceHandler<EcalMappingElectronics>
{
 public:
  EcalMappingElectronicsHandler(edm::ParameterSet const & );
  ~EcalMappingElectronicsHandler() override; 
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  
 private:

  const EcalMappingElectronics * myMap;

  std::string txtFileSource_;
  std::string m_name;

  long long since_;
};

#endif

