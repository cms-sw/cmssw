#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

using namespace lhef;

class LHEWriter : public edm::EDAnalyzer {
public:
  explicit LHEWriter(const edm::ParameterSet &params);
  ~LHEWriter() override;

protected:
  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void endRun(const edm::Run &run, const edm::EventSetup &es) override;
  void analyze(const edm::Event &event, const edm::EventSetup &es) override;

private:
  std::ofstream file;
  std::ofstream file1;

  edm::EDGetTokenT<LHERunInfoProduct> tokenLHERunInfo_;
  edm::EDGetTokenT<LHEEventProduct> tokenLHEEvent_;
};

LHEWriter::LHEWriter(const edm::ParameterSet &params)
    : tokenLHERunInfo_(consumes<LHERunInfoProduct, edm::InRun>(
          params.getUntrackedParameter<edm::InputTag>("moduleLabel", std::string("source")))),
      tokenLHEEvent_(consumes<LHEEventProduct>(
          params.getUntrackedParameter<edm::InputTag>("moduleLabel", std::string("source")))) {}

LHEWriter::~LHEWriter() {}

void LHEWriter::beginRun(const edm::Run &run, const edm::EventSetup &es) {
  file.open("writer.lhe", std::fstream::out | std::fstream::trunc);
  file1.open("writer1.lhe", std::fstream::out | std::fstream::trunc);
}

void LHEWriter::endRun(const edm::Run &run, const edm::EventSetup &es) {
  edm::Handle<LHERunInfoProduct> product;
  //run.getByLabel("source", product);
  run.getByToken(tokenLHERunInfo_, product);

  std::copy(product->begin(), product->end(), std::ostream_iterator<std::string>(file));

  file1 << LHERunInfoProduct::endOfFile();
  file.close();
  file1.close();

  system("cat writer1.lhe >> writer.lhe");
  system("rm -rf writer1.lhe");
}

void LHEWriter::analyze(const edm::Event &event, const edm::EventSetup &es) {
  edm::Handle<LHEEventProduct> product;
  //event.getByLabel("source", product);
  event.getByToken(tokenLHEEvent_, product);

  std::copy(product->begin(), product->end(), std::ostream_iterator<std::string>(file1));
}

DEFINE_FWK_MODULE(LHEWriter);
