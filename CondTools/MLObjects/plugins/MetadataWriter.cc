#include <iostream>
#include <fstream>
#include <vector>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/DataRecord/interface/MetadataRcd.h"
#include "CondFormats/MLObjects/interface/Metadata.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class MetadataWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MetadataWriter(const edm::ParameterSet&);
  ~MetadataWriter() override {}

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}

  const edm::ParameterSet model_;
  unsigned long long since_;
};

MetadataWriter::MetadataWriter(const edm::ParameterSet& iConfig)
    : model_(iConfig.getParameter<edm::ParameterSet>("model")),
      since_(iConfig.getParameter<unsigned long long>("since")) {}

void MetadataWriter::beginJob() {
  const auto& ps = model_;
  Metadata metadata;
  int version = ps.getParameter<int>("version");
  std::string model_name = ps.getParameter<std::string>("model_name");
  std::string hash = ps.exists("hash") ? ps.getParameter<std::string>("hash") : "";

  metadata.set_model_name(model_name);
  metadata.set_version(version);
  metadata.set_hash(hash);

  edm::Service<cond::service::PoolDBOutputService> pool;
  if (pool.isAvailable())
    pool->writeOneIOV(metadata, since_, "MetadataRcd");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MetadataWriter);
