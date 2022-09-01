#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/IO_GenEvent.h"

class HepMCEventWriter : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HepMCEventWriter(const edm::ParameterSet &params);
  ~HepMCEventWriter() override = default;

protected:
  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void endRun(const edm::Run &run, const edm::EventSetup &es) override;
  void analyze(const edm::Event &event, const edm::EventSetup &es) override;

private:
  edm::propagate_const<HepMC::IO_GenEvent *> _output;
  const edm::EDGetTokenT<edm::HepMCProduct> hepMCProduct_;
};

HepMCEventWriter::HepMCEventWriter(const edm::ParameterSet &params)
    : hepMCProduct_(consumes<edm::HepMCProduct>(params.getParameter<edm::InputTag>("hepMCProduct"))) {}

void HepMCEventWriter::beginRun(const edm::Run &run, const edm::EventSetup &es) {
  _output = new HepMC::IO_GenEvent("GenEvent_ASCII.dat", std::ios::out);
}

void HepMCEventWriter::endRun(const edm::Run &run, const edm::EventSetup &es) {
  if (_output)
    delete _output.get();
}

void HepMCEventWriter::analyze(const edm::Event &event, const edm::EventSetup &es) {
  const edm::Handle<edm::HepMCProduct> &product = event.getHandle(hepMCProduct_);

  const HepMC::GenEvent *evt = product->GetEvent();

  _output->write_event(evt);
}

DEFINE_FWK_MODULE(HepMCEventWriter);
