#include <iostream>
#include <fstream>
#include <vector>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/DataRecord/interface/MLMetadataRcd.h"
#include "CondFormats/MLObjects/interface/MLMetadata.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class MLMetadataWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MLMetadataWriter(const edm::ParameterSet&);
  ~MLMetadataWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}

  const edm::ParameterSet model_;
  const unsigned long long since_;
};

MLMetadataWriter::MLMetadataWriter(const edm::ParameterSet& iConfig)
    : model_(iConfig.getParameter<edm::ParameterSet>("model")),
      since_(iConfig.getParameter<unsigned long long>("since")) {}

void MLMetadataWriter::beginJob() {
  const auto& ps = model_;
  MLMetadata metadata;
  int version = ps.getParameter<int>("version");
  std::string model_name = ps.getParameter<std::string>("model_name");
  std::string hash = ps.getParameter<std::string>("hash");

  metadata.set_model_name(model_name);
  metadata.set_version(version);
  metadata.set_hash(hash);

  edm::Service<cond::service::PoolDBOutputService> pool;
  if (pool.isAvailable())
    pool->writeOneIOV(metadata, since_, "MLMetadataRcd");
}

void MLMetadataWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription modelDescription;
  modelDescription.add<std::string>("model_name");
  modelDescription.add<int>("version");
  modelDescription.add<std::string>("hash");

  edm::ParameterSetDescription description;
  description.add<edm::ParameterSetDescription>("model", modelDescription);
  description.add<unsigned long long>("since");

  descriptions.add("mlMetadataWriter", description);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MLMetadataWriter);
