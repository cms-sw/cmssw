#include "DQMProtobufReader.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

DQMProtobufReader::DQMProtobufReader(ParameterSet const& pset,
                                     InputSourceDescription const& desc)
    : InputSource(pset, desc), fiterator_(pset, DQMFileIterator::JS_PROTOBUF) {

  flagSkipFirstLumis_ = pset.getUntrackedParameter<bool>("skipFirstLumis");
  flagEndOfRunKills_ = pset.getUntrackedParameter<bool>("endOfRunKills");
  flagDeleteDatFiles_ = pset.getUntrackedParameter<bool>("deleteDatFiles");
}

DQMProtobufReader::~DQMProtobufReader() {}

InputSource::ItemType DQMProtobufReader::getNextItemType() {
  typedef DQMFileIterator::State State;
  typedef DQMFileIterator::LumiEntry LumiEntry;

  fiterator_.logFileAction("getNextItemType");

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
    return InputSource::IsSynchronize;
  }

  // this is unreachable
}

boost::shared_ptr<edm::RunAuxiliary> DQMProtobufReader::readRunAuxiliary_() {
  fiterator_.logFileAction("readRunAuxiliary_");

  edm::RunAuxiliary* aux = new edm::RunAuxiliary(
      fiterator_.runNumber(), edm::Timestamp(), edm::Timestamp());
  return boost::shared_ptr<edm::RunAuxiliary>(aux);
};

boost::shared_ptr<edm::LuminosityBlockAuxiliary>
DQMProtobufReader::readLuminosityBlockAuxiliary_() {
  fiterator_.logFileAction("readLuminosityBlockAuxiliary_");

  edm::LuminosityBlockAuxiliary* aux = new edm::LuminosityBlockAuxiliary(
      fiterator_.runNumber(), fiterator_.front().ls, edm::Timestamp(),
      edm::Timestamp());

  return boost::shared_ptr<edm::LuminosityBlockAuxiliary>(aux);
};

void DQMProtobufReader::readRun_(edm::RunPrincipal& rpCache) {
  fiterator_.logFileAction("readRun_");
  rpCache.fillRunPrincipal(processHistoryRegistryForUpdate());

  edm::Service<DQMStore> store;
  std::vector<MonitorElement*> allMEs = store->getAllContents("");
  for (auto const& ME : allMEs) {
    ME->Reset();
  }
}

void DQMProtobufReader::readLuminosityBlock_(
    edm::LuminosityBlockPrincipal& lbCache) {

  fiterator_.logFileAction("readLuminosityBlock_");
  edm::Service<DQMStore> store;

  edm::Service<edm::JobReport> jr;
  jr->reportInputLumiSection(lbCache.id().run(),
                             lbCache.id().luminosityBlock());
  lbCache.fillLuminosityBlockPrincipal(processHistoryRegistryForUpdate());

  // clear the old lumi histograms
  std::vector<MonitorElement*> allMEs = store->getAllContents("");
  for (auto const& ME : allMEs) {
    // We do not want to reset Run Products here!
    if (ME->getLumiFlag()) {
      ME->Reset();
    }
  }

  // load the new file
  const DQMFileIterator::LumiEntry& lumi = fiterator_.front();
  std::string p = fiterator_.make_path_data(lumi);
  if (!boost::filesystem::exists(p)) {
    fiterator_.logFileAction("Data file is missing ", p);
    fiterator_.pop();
    return;
  }

  fiterator_.logFileAction("Initiating request to open file ", p);
  fiterator_.logFileAction("Successfully opened file ", p);
  store->load(p);
  fiterator_.logFileAction("Closed file ", p);
  fiterator_.pop();
};

void DQMProtobufReader::readEvent_(edm::EventPrincipal&) {};

void DQMProtobufReader::fillDescriptions(
    ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setComment(
      "Creates runs and lumis and fills the dqmstore from protocol buffer "
      "files.");
  ProducerSourceBase::fillDescription(desc);

  desc.addUntracked<bool>("skipFirstLumis", false)->setComment(
      "Skip (and ignore the minEventsPerLumi parameter) for the files "
      "which have been available at the begining of the processing. "
      "If set to true, the reader will open last available file for "
      "processing.");

  desc.addUntracked<bool>("deleteDatFiles", false)->setComment(
      "Delete data files after they have been closed, in order to "
      "save disk space.");

  desc.addUntracked<bool>("endOfRunKills", false)->setComment(
      "Kill the processing as soon as the end-of-run file appears, even if "
      "there are/will be unprocessed lumisections.");

  DQMFileIterator::fillDescription(desc);
  descriptions.add("source", desc);
}

using edm::DQMProtobufReader;
DEFINE_FWK_INPUT_SOURCE(DQMProtobufReader);
