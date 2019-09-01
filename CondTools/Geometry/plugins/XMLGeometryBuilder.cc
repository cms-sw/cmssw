#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

#include <string>
#include <vector>
#include <fstream>

class XMLGeometryBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  XMLGeometryBuilder(const edm::ParameterSet&);

  void beginJob() override;
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  std::string m_fname;
  bool m_zip;
  std::string m_record;
};

XMLGeometryBuilder::XMLGeometryBuilder(const edm::ParameterSet& iConfig) {
  m_fname = iConfig.getUntrackedParameter<std::string>("XMLFileName", "test.xml");
  m_zip = iConfig.getUntrackedParameter<bool>("ZIP", true);
  m_record = iConfig.getUntrackedParameter<std::string>("record", "GeometryFileRcd");
}

void XMLGeometryBuilder::beginJob() {
  edm::LogInfo("XMLGeometryBuilder") << "XMLGeometryBuilder::beginJob";
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("XMLGeometryBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  FileBlob* pgf = new FileBlob(m_fname, m_zip);

  if (mydbservice->isNewTagRequest(m_record)) {
    mydbservice->createNewIOV<FileBlob>(pgf, mydbservice->beginOfTime(), mydbservice->endOfTime(), m_record);
  } else {
    edm::LogError("XMLGeometryBuilder") << "GeometryFileRcd Tag already exist";
  }
}

DEFINE_FWK_MODULE(XMLGeometryBuilder);
