// L1TCaloParamsWriter
//
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

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

class L1TCaloParamsWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1TCaloParamsWriter(const edm::ParameterSet&) {}
  ~L1TCaloParamsWriter() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

void L1TCaloParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  l1t::DataWriter dataWriter;
  std::string token = dataWriter.writePayload(evSetup, "L1TCaloParamsRcd@CaloParams");
  if (dataWriter.updateIOV("L1TCaloParamsRcd", token, 1, false))
    std::cout << "IOV updated!" << std::endl;
  std::cout << "Payload token = " << token << std::endl;
}

DEFINE_FWK_MODULE(L1TCaloParamsWriter);
