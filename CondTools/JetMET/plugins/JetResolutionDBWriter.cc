// Author: Sébastien Brochet

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"

class JetResolutionDBWriter : public edm::one::EDAnalyzer<> {
public:
  JetResolutionDBWriter(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void endJob() override {}
  ~JetResolutionDBWriter() override {}

private:
  std::string m_record;
  std::string m_path;
};

// Constructor
JetResolutionDBWriter::JetResolutionDBWriter(const edm::ParameterSet& pSet) {
  m_record = pSet.getUntrackedParameter<std::string>("record");
  m_path = pSet.getUntrackedParameter<edm::FileInPath>("file").fullPath();
}

// Begin Job
void JetResolutionDBWriter::beginJob() {
  std::cout << "Loading data from '" << m_path << "'" << std::endl;

  const JME::JetResolutionObject jerObject(m_path);

  std::cout << "Opening PoolDBOutputService" << std::endl;

  // now write it into the DB
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    std::cout << "Setting up payload record " << m_record << std::endl;
    cond::Time_t sinceTime = s->isNewTagRequest(m_record) ? s->beginOfTime() : s->currentTime();
    s->writeOneIOV(jerObject, sinceTime, m_record);

    std::cout << "Object saved into the database with the record: " << m_record << std::endl;
  }
}

DEFINE_FWK_MODULE(JetResolutionDBWriter);
