/**  
*  See header file for a description of this class.
*
*
*  \author Jo. Weng  - CERN, Ph Division & Uni Karlsruhe
*  \author F.Moortgat - CERN, Ph Division
*/

#include <iostream>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOMC/Input/interface/HepMCFileReader.h"
#include "IOMC/Input/interface/MCFileSource.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

namespace edm {

  //-------------------------------------------------------------------------
  MCFileSource::MCFileSource(const ParameterSet& pset, InputSourceDescription const& desc)
      : ProducerSourceFromFiles(pset, desc, false), reader_(HepMCFileReader::instance()), evt_(nullptr) {
    LogInfo("MCFileSource") << "Reading HepMC file:" << fileNames()[0];
    std::string fileName = fileNames()[0];
    // strip the file:
    if (fileName.find("file:") == 0) {
      fileName.erase(0, 5);
    }

    reader_->initialize(fileName);
    produces<HepMCProduct>("generator");
    produces<GenEventInfoProduct>("generator");
  }

  //-------------------------------------------------------------------------
  MCFileSource::~MCFileSource() {}

  //-------------------------------------------------------------------------
  bool MCFileSource::setRunAndEventInfo(EventID&, TimeValue_t&, EventAuxiliary::ExperimentType&) {
    // Read one HepMC event
    LogInfo("MCFileSource") << "Start Reading";
    evt_ = reader_->fillCurrentEventData();
    return (evt_ != nullptr);
  }

  //-------------------------------------------------------------------------
  void MCFileSource::produce(Event& e) {
    // Store one HepMC event in the Event.

    auto bare_product = std::make_unique<HepMCProduct>();
    bare_product->addHepMCData(evt_);
    e.put(std::move(bare_product), "generator");
    std::unique_ptr<GenEventInfoProduct> info(new GenEventInfoProduct(evt_));
    e.put(std::move(info), "generator");
  }

}  // namespace edm
