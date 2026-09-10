/**
*  See header file for a description of this class.
*/

#include <string>
#include <vector>

#include <HepMC3/GenEvent.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "IOMC/Input/interface/HepMC3FileReader.h"
#include "IOMC/Input/interface/MCFileSource3.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"

namespace edm {

  //-------------------------------------------------------------------------
  MCFileSource3::MCFileSource3(const ParameterSet& pset, InputSourceDescription const& desc)
      : ProducerSourceBase(pset, desc, false),
        evt_(nullptr),
        inputFileCatalog_(pset),
        printEvent_(pset.getUntrackedParameter<bool>("printEvent")) {
    std::vector<std::string> fileNames = inputFileCatalog_.allPFNsFromFirstCatalog();
    for (auto& fileName : fileNames) {
      // strip the file:
      if (fileName.find("file:") == 0) {
        fileName.erase(0, 5);
      }
      LogInfo("MCFileSource3") << "Reading HepMC3 file: " << fileName;
    }

    reader_ = std::make_unique<HepMC3FileReader>(std::move(fileNames));
    produces<HepMC3Product>("generator");
    produces<GenEventInfoProduct3>("generator");
  }

  //-------------------------------------------------------------------------
  MCFileSource3::~MCFileSource3() {}

  //-------------------------------------------------------------------------
  bool MCFileSource3::setRunAndEventInfo(EventID&, TimeValue_t&, EventAuxiliary::ExperimentType&) {
    // Read one HepMC3 event
    evt_ = reader_->fillCurrentEventData();
    if (evt_ != nullptr && printEvent_) {
      reader_->printCurrentEvent();
    }
    return (evt_ != nullptr);
  }

  //-------------------------------------------------------------------------
  void MCFileSource3::produce(Event& e) {
    // Store one HepMC3 event in the Event.

    auto bare_product = std::make_unique<HepMC3Product>(*evt_);
    e.put(std::move(bare_product), "generator");
    auto info = std::make_unique<GenEventInfoProduct3>(evt_);
    e.put(std::move(info), "generator");
  }

  //-------------------------------------------------------------------------
  void MCFileSource3::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("A source which reads HepMC3 files.");
    ProducerSourceBase::fillDescription(desc);
    InputFileCatalog::fillDescription(desc);
    desc.addUntracked<bool>("printEvent", false)->setComment("Print the content of every event which is read.");
    descriptions.add("source", desc);
  }

}  // namespace edm
