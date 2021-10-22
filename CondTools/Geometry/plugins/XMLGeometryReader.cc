#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

#include <string>
#include <vector>
#include <fstream>

class XMLGeometryReader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  XMLGeometryReader(const edm::ParameterSet&);

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  std::string m_fname;
  std::string m_label;
  edm::ESGetToken<FileBlob, GeometryFileRcd> fileBlobToken_;
};

XMLGeometryReader::XMLGeometryReader(const edm::ParameterSet& iConfig) {
  m_fname = iConfig.getUntrackedParameter<std::string>("XMLFileName", "test.xml");
  m_label = iConfig.getUntrackedParameter<std::string>("geomLabel", "Extended");
  fileBlobToken_ = esConsumes<edm::Transition::BeginRun>();
}

void XMLGeometryReader::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  edm::LogInfo("XMLGeometryReader") << "XMLGeometryReader::beginRun";

  auto geometry = iSetup.getHandle(fileBlobToken_);
  std::unique_ptr<std::vector<unsigned char> > blob((*geometry).getUncompressedBlob());

  std::string outfile1(m_fname);
  std::ofstream output1(outfile1.c_str());
  output1.write((const char*)&(*blob)[0], blob->size());
  output1.close();
}

DEFINE_FWK_MODULE(XMLGeometryReader);
