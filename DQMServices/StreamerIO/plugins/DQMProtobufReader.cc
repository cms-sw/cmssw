#include "DQMProtobufReader.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
// #include "FWCore/Sources/interface/ProducerSourceBase.h"

#include "DQMServices/Core/src/ROOTFilePB.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "TBufferFile.h"

#include <regex>
#include <cstdlib>

using namespace dqmservices;

DQMProtobufReader::DQMProtobufReader(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
    : PuttableSourceBase(pset, desc), fiterator_(pset) {
  flagSkipFirstLumis_ = pset.getUntrackedParameter<bool>("skipFirstLumis");
  flagEndOfRunKills_ = pset.getUntrackedParameter<bool>("endOfRunKills");
  flagDeleteDatFiles_ = pset.getUntrackedParameter<bool>("deleteDatFiles");
  flagLoadFiles_ = pset.getUntrackedParameter<bool>("loadFiles");

  produces<std::string, edm::Transition::BeginLuminosityBlock>("sourceDataPath");
  produces<std::string, edm::Transition::BeginLuminosityBlock>("sourceJsonPath");
  produces<DQMToken, edm::Transition::BeginRun>("DQMGenerationRecoRun");
  produces<DQMToken, edm::Transition::BeginLuminosityBlock>("DQMGenerationRecoLumi");
}

DQMProtobufReader::~DQMProtobufReader() {}

edm::InputSource::ItemType DQMProtobufReader::getNextItemType() {
  typedef DQMFileIterator::State State;
  typedef DQMFileIterator::LumiEntry LumiEntry;

  // fiterator_.logFileAction("getNextItemType");

  for (;;) {
    fiterator_.update_state();

    if (edm::shutdown_flag.load()) {
      fiterator_.logFileAction("Shutdown flag was set, shutting down.");
      return InputSource::IsStop;
    }

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
    //
    // comment out in order to block at this level
    // return InputSource::IsSynchronize;
  }

  // this is unreachable
}

std::shared_ptr<edm::RunAuxiliary> DQMProtobufReader::readRunAuxiliary_() {
  // fiterator_.logFileAction("readRunAuxiliary_");

  edm::RunAuxiliary* aux = new edm::RunAuxiliary(fiterator_.runNumber(), edm::Timestamp(), edm::Timestamp());
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

std::shared_ptr<edm::LuminosityBlockAuxiliary> DQMProtobufReader::readLuminosityBlockAuxiliary_() {
  // fiterator_.logFileAction("readLuminosityBlockAuxiliary_");

  currentLumi_ = fiterator_.open();
  edm::LuminosityBlockAuxiliary* aux = new edm::LuminosityBlockAuxiliary(
      fiterator_.runNumber(), currentLumi_.file_ls, edm::Timestamp(), edm::Timestamp());

  return std::shared_ptr<edm::LuminosityBlockAuxiliary>(aux);
}

void DQMProtobufReader::readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) {
  // fiterator_.logFileAction("readLuminosityBlock_");

  edm::Service<edm::JobReport> jr;
  jr->reportInputLumiSection(lbCache.id().run(), lbCache.id().luminosityBlock());
  lbCache.fillLuminosityBlockPrincipal(processHistoryRegistry().getMapped(lbCache.aux().processHistoryID()));
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

  std::unique_ptr<std::string> path_product(new std::string(path));
  std::unique_ptr<std::string> json_product(new std::string(jspath));

  lb.put(std::move(path_product), "sourceDataPath");
  lb.put(std::move(json_product), "sourceJsonPath");

  if (flagLoadFiles_) {
    if (!boost::filesystem::exists(path)) {
      fiterator_.logFileAction("Data file is missing ", path);
      fiterator_.logLumiState(currentLumi_, "error: data file missing");
      return;
    }

    fiterator_.logFileAction("Initiating request to open file ", path);
    fiterator_.logFileAction("Successfully opened file ", path);
    load(&*store, path);
    fiterator_.logFileAction("Closed file ", path);
    fiterator_.logLumiState(currentLumi_, "close: ok");
  } else {
    fiterator_.logFileAction("Not loading the data file at source level ", path);
    fiterator_.logLumiState(currentLumi_, "close: not loading");
  }
}

void DQMProtobufReader::load(DQMStore* store, std::string filename) {
  using google::protobuf::io::ArrayInputStream;
  using google::protobuf::io::CodedInputStream;
  using google::protobuf::io::FileInputStream;
  using google::protobuf::io::FileOutputStream;
  using google::protobuf::io::GzipInputStream;
  using google::protobuf::io::GzipOutputStream;

  int filedescriptor;
  if ((filedescriptor = ::open(filename.c_str(), O_RDONLY)) == -1) {
    edm::LogError("DQMProtobufReader") << "File " << filename << " does not exist.";
  }

  dqmstorepb::ROOTFilePB dqmstore_message;
  FileInputStream fin(filedescriptor);
  GzipInputStream input(&fin);
  CodedInputStream input_coded(&input);
  input_coded.SetTotalBytesLimit(1024 * 1024 * 1024, -1);
  if (!dqmstore_message.ParseFromCodedStream(&input_coded)) {
    edm::LogError("DQMProtobufReader") << "Fatal parsing file '" << filename << "'";
  }

  ::close(filedescriptor);

  for (int i = 0; i < dqmstore_message.histo_size(); ++i) {
    TObject* obj = nullptr;
    dqmstorepb::ROOTFilePB::Histo const& h = dqmstore_message.histo(i);

    size_t slash = h.full_pathname().rfind('/');
    size_t dirpos = (slash == std::string::npos ? 0 : slash);
    size_t namepos = (slash == std::string::npos ? 0 : slash + 1);
    std::string objname, dirname;
    dirname.assign(h.full_pathname(), 0, dirpos);
    objname.assign(h.full_pathname(), namepos, std::string::npos);
    TBufferFile buf(TBufferFile::kRead, h.size(), (void*)h.streamed_histo().data(), kFALSE);
    buf.Reset();
    if (buf.Length() == buf.BufferSize()) {
      obj = nullptr;
    } else {
      buf.InitMap();
      void* ptr = buf.ReadObjectAny(nullptr);
      obj = reinterpret_cast<TObject*>(ptr);
    }

    if (!obj) {
      edm::LogError("DQMProtobufReader") << "Error reading element:'" << h.full_pathname();
    }

    store->setCurrentFolder(dirname);

    if (h.flags() & DQMNet::DQM_PROP_LUMI) {
      store->setScope(MonitorElementData::Scope::LUMI);
    } else {
      store->setScope(MonitorElementData::Scope::RUN);
    }

    if (obj) {
      int kind = h.flags() & DQMNet::DQM_PROP_TYPE_MASK;
      if (kind == DQMNet::DQM_PROP_TYPE_INT) {
        MonitorElement* me = store->bookInt(objname);
        auto expression = std::string(static_cast<TObjString*>(obj)->String().View());
        std::regex parseint{"<.*>i=(.*)</.*>"};
        std::smatch match;
        bool ok = std::regex_match(expression, match, parseint);
        if (!ok) {
          edm::LogError("DQMProtobufReader") << "Malformed object of type INT: '" << expression << "'";
          continue;
        }
        int value = std::atoi(match[1].str().c_str());
        me->Fill(value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_REAL) {
        MonitorElement* me = store->bookFloat(objname);
        auto expression = std::string(static_cast<TObjString*>(obj)->String().View());
        std::regex parsefloat{"<.*>f=(.*)</.*>"};
        std::smatch match;
        bool ok = std::regex_match(expression, match, parsefloat);
        if (!ok) {
          edm::LogError("DQMProtobufReader") << "Malformed object of type REAL: '" << expression << "'";
          continue;
        }
        double value = std::atof(match[1].str().c_str());
        me->Fill(value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_STRING) {
        auto value = static_cast<TObjString*>(obj)->String();
        store->bookString(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH1F) {
        auto value = static_cast<TH1F*>(obj);
        store->book1D(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH1S) {
        auto value = static_cast<TH1S*>(obj);
        store->book1S(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH1D) {
        auto value = static_cast<TH1D*>(obj);
        store->book1DD(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH2F) {
        auto value = static_cast<TH2F*>(obj);
        store->book2D(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH2S) {
        auto value = static_cast<TH2S*>(obj);
        store->book2S(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH2D) {
        auto value = static_cast<TH2D*>(obj);
        store->book2DD(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TH3F) {
        auto value = static_cast<TH3F*>(obj);
        store->book3D(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TPROF) {
        auto value = static_cast<TProfile*>(obj);
        store->bookProfile(objname, value);
      } else if (kind == DQMNet::DQM_PROP_TYPE_TPROF2D) {
        auto value = static_cast<TProfile2D*>(obj);
        store->bookProfile2D(objname, value);
      } else {
        edm::LogError("DQMProtobufReader") << "Unknown type: " << kind;
      }
      delete obj;
    }
  }
}

void DQMProtobufReader::readEvent_(edm::EventPrincipal&){};

void DQMProtobufReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
