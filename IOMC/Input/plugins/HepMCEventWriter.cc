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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/IO_GenEvent.h"


class HepMCEventWriter : public edm::EDAnalyzer {
public:
  explicit HepMCEventWriter(const edm::ParameterSet &params);
  virtual ~HepMCEventWriter();
  
protected:
  virtual void beginRun(const edm::Run &run, const edm::EventSetup &es);
  virtual void endRun(const edm::Run &run, const edm::EventSetup &es);
  virtual void analyze(const edm::Event &event, const edm::EventSetup &es);
  
private:
  HepMC::IO_GenEvent* _output;
  edm::InputTag hepMCProduct_;
};

HepMCEventWriter::HepMCEventWriter(const edm::ParameterSet &params) :
  hepMCProduct_(params.getParameter<edm::InputTag>("hepMCProduct"))
{
}

HepMCEventWriter::~HepMCEventWriter()
{
}

void HepMCEventWriter::beginRun(const edm::Run &run, const edm::EventSetup &es)
{

  _output = new HepMC::IO_GenEvent("GenEvent_ASCII.dat",std::ios::out);

}


void HepMCEventWriter::endRun(const edm::Run &run, const edm::EventSetup &es)
{
  if (_output) delete _output;
}

void HepMCEventWriter::analyze(const edm::Event &event, const edm::EventSetup &es)
{

  edm::Handle<edm::HepMCProduct> product;
  event.getByLabel(hepMCProduct_, product);

  const HepMC::GenEvent* evt = product->GetEvent();

  _output->write_event(evt);

}

DEFINE_FWK_MODULE(HepMCEventWriter);
