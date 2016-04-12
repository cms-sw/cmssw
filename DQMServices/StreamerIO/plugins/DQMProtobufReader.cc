#include "DQMProtobufReader.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// #include "FWCore/Sources/interface/ProducerSourceBase.h"

using namespace dqmservices;

DQMProtobufReader::DQMProtobufReader(edm::ParameterSet const& pset,
                                     edm::InputSourceDescription const& desc)
    : InputSource(pset, desc), fiterator_(pset) {

  flagSkipFirstLumis_ = pset.getUntrackedParameter<bool>("skipFirstLumis");
  flagEndOfRunKills_ = pset.getUntrackedParameter<bool>("endOfRunKills");
  flagDeleteDatFiles_ = pset.getUntrackedParameter<bool>("deleteDatFiles");
  flagLoadFiles_ = pset.getUntrackedParameter<bool>("loadFiles");

  produces<std::string, edm::InLumi>("sourceDataPath");
  produces<std::string, edm::InLumi>("sourceJsonPath");
}

DQMProtobufReader::~DQMProtobufReader() {}

edm::InputSource::ItemType DQMProtobufReader::getNextItemType() {
  typedef DQMFileIterator::State State;
  typedef DQMFileIterator::LumiEntry LumiEntry;

  // fiterator_.logFileAction("getNextItemType");

  for (;;) {
    fiterator_.update_state();

    // check for end of run file and force quit
    if (flagEndOfRunKills_ && (fiterator_.state() != State::OPEN)) {
      return InputSource::IsStop;
    }

    // check for end of run and quit if everything has been processed.
    // this is the clean exit
    if ((!fiterator_.lumiReady()) && (fiterator_.state() == State::EOR)) {
      return InputSource::IsStop;
    }

    // skip to the next file if we have no files openned yet
    if (fiterator_.lumiReady()) {
      return InputSource::IsLumi;
    }

    fiterator_.delay();
    // BUG: for an unknown reason it fails after a certain time if we use
    // IsSynchronize state
    // comment out in order to block at this level
    // the only downside is that we cannot Ctrl+C :)
    // return InputSource::IsSynchronize;
  }

  // this is unreachable
}

std::shared_ptr<edm::RunAuxiliary> DQMProtobufReader::readRunAuxiliary_() {
  // fiterator_.logFileAction("readRunAuxiliary_");

  edm::RunAuxiliary* aux = new edm::RunAuxiliary(
      fiterator_.runNumber(), edm::Timestamp(), edm::Timestamp());
  return std::shared_ptr<edm::RunAuxiliary>(aux);
}

void DQMProtobufReader::readRun_(edm::RunPrincipal& rpCache) {
  // fiterator_.logFileAction("readRun_");
  rpCache.fillRunPrincipal(processHistoryRegistryForUpdate());

  edm::Service<DQMStore> store;
  std::vector<MonitorElement*> allMEs = store->getAllContents("");
  for (auto const& ME : allMEs) {
    ME->Reset();
  }
}

std::shared_ptr<edm::LuminosityBlockAuxiliary>
DQMProtobufReader::readLuminosityBlockAuxiliary_() {
  // fiterator_.logFileAction("readLuminosityBlockAuxiliary_");

  currentLumi_ = fiterator_.open();
  edm::LuminosityBlockAuxiliary* aux = new edm::LuminosityBlockAuxiliary(
      fiterator_.runNumber(), currentLumi_.file_ls, edm::Timestamp(),
      edm::Timestamp());

  return std::shared_ptr<edm::LuminosityBlockAuxiliary>(aux);
}

void DQMProtobufReader::readLuminosityBlock_(
    edm::LuminosityBlockPrincipal& lbCache) {
  // fiterator_.logFileAction("readLuminosityBlock_");

  edm::Service<edm::JobReport> jr;
  jr->reportInputLumiSection(lbCache.id().run(),
                             lbCache.id().luminosityBlock());
  lbCache.fillLuminosityBlockPrincipal(processHistoryRegistryForUpdate());
}

void DQMProtobufReader::beginLuminosityBlock(edm::LuminosityBlock& lb) {
  edm::Service<DQMStore> store;

  // clear the old lumi histograms
  std::vector<MonitorElement*> allMEs = store->getAllContents("");
  for (auto const& ME : allMEs) {
    // We do not want to reset Run Products here!
    if (ME->getLumiFlag()) {
      ME->Reset();
    }
  }

  // load the new file
  std::string path = currentLumi_.get_data_path();
  std::string jspath = currentLumi_.get_json_path();

  std::auto_ptr<std::string> path_product(new std::string(path));
  std::auto_ptr<std::string> json_product(new std::string(jspath));

  lb.put(path_product, "sourceDataPath");
  lb.put(json_product, "sourceJsonPath");

  if (flagLoadFiles_) {
    if (!boost::filesystem::exists(path)) {
      fiterator_.logFileAction("Data file is missing ", path);
      fiterator_.logLumiState(currentLumi_, "error: data file missing");
      return;
    }

    fiterator_.logFileAction("Initiating request to open file ", path);
    fiterator_.logFileAction("Successfully opened file ", path);
    store->load(path);
    fiterator_.logFileAction("Closed file ", path);
    fiterator_.logLumiState(currentLumi_, "close: ok");
  } else {
    fiterator_.logFileAction("Not loading the data file at source level ", path);
    fiterator_.logLumiState(currentLumi_, "close: not loading");
  }
}


void DQMProtobufReader::readEvent_(edm::EventPrincipal&){};

void DQMProtobufReader::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      "Creates runs and lumis and fills the dqmstore from protocol buffer "
      "files.");
  edm::ProducerSourceBase::fillDescription(desc);

  desc.addUntracked<bool>("skipFirstLumis", false)
      ->setComment(
          "Skip (and ignore the minEventsPerLumi parameter) for the files "
          "which have been available at the begining of the processing. "
          "If set to true, the reader will open last available file for "
          "processing.");

  desc.addUntracked<bool>("deleteDatFiles", false)
      ->setComment(
          "Delete data files after they have been closed, in order to "
          "save disk space.");

  desc.addUntracked<bool>("endOfRunKills", false)
      ->setComment(
          "Kill the processing as soon as the end-of-run file appears, even if "
          "there are/will be unprocessed lumisections.");

  desc.addUntracked<bool>("loadFiles", true)
      ->setComment(
          "Tells the source load the data files. If set to false, source will create skeleton lumi transitions.");

  DQMFileIterator::fillDescription(desc);
  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using dqmservices::DQMProtobufReader;
DEFINE_FWK_INPUT_SOURCE(DQMProtobufReader);
