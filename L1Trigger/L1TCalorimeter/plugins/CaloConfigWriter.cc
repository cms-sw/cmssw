// CaloConfigWriter
//
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloConfigRcd.h"
#include "CondFormats/L1TObjects/interface/CaloConfig.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondTools/L1Trigger/interface/DataWriter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include <iostream>

//
// class declaration
//

class CaloConfigWriter : public edm::EDAnalyzer {
public:
  explicit CaloConfigWriter(const edm::ParameterSet&) {}
  virtual  ~CaloConfigWriter() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;  

};



void CaloConfigWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{
  l1t::DataWriter dataWriter;  
  std::string token = dataWriter.writePayload(evSetup, "L1TCaloConfigRcd@CaloConfig");
  if ( dataWriter.updateIOV("L1TCaloConfigRcd", token, 1, false) ) std::cout << "IOV updated!" << std::endl;
  std::cout << "Payload token = " << token << std::endl;
}

DEFINE_FWK_MODULE(CaloConfigWriter);



