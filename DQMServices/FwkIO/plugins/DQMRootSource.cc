// -*- C++ -*-
//
// Package:     FwkIO
// Class  :     DQMRootSource
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue May  3 11:13:47 CDT 2011
//

// system include files
#include <vector>
#include <string>
#include <map>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"

// user include files
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "format.h"

namespace {
  typedef dqm::harvesting::MonitorElement MonitorElement;

  // TODO: this should probably be moved somewhere else
  class DQMMergeHelper {
  public:
    // Utility function to check the consistency of the axis labels
    // Taken from TH1::CheckBinLabels which is not public
    static bool CheckBinLabels(const TAxis* a1, const TAxis* a2) {
      // Check that axis have same labels
      THashList* l1 = (const_cast<TAxis*>(a1))->GetLabels();
      THashList* l2 = (const_cast<TAxis*>(a2))->GetLabels();

      if (!l1 && !l2)
        return true;
      if (!l1 || !l2) {
        return false;
      }

      // Check now labels sizes  are the same
      if (l1->GetSize() != l2->GetSize()) {
        return false;
      }

      for (int i = 1; i <= a1->GetNbins(); ++i) {
        TString label1 = a1->GetBinLabel(i);
        TString label2 = a2->GetBinLabel(i);
        if (label1 != label2) {
          return false;
        }
      }

      return true;
    }

    // NOTE: the merge logic comes from DataFormats/Histograms/interface/MEtoEDMFormat.h
    static void mergeTogether(TH1* original, TH1* toAdd) {
      if (original->CanExtendAllAxes() && toAdd->CanExtendAllAxes()) {
        TList list;
        list.Add(toAdd);
        if (original->Merge(&list) == -1) {
          edm::LogError("MergeFailure") << "Failed to merge DQM element " << original->GetName();
        }
      } else {
        // TODO: Redo. What's wrong with this implementation?
        if (original->GetNbinsX() == toAdd->GetNbinsX() &&
            original->GetXaxis()->GetXmin() == toAdd->GetXaxis()->GetXmin() &&
            original->GetXaxis()->GetXmax() == toAdd->GetXaxis()->GetXmax() &&
            original->GetNbinsY() == toAdd->GetNbinsY() &&
            original->GetYaxis()->GetXmin() == toAdd->GetYaxis()->GetXmin() &&
            original->GetYaxis()->GetXmax() == toAdd->GetYaxis()->GetXmax() &&
            original->GetNbinsZ() == toAdd->GetNbinsZ() &&
            original->GetZaxis()->GetXmin() == toAdd->GetZaxis()->GetXmin() &&
            original->GetZaxis()->GetXmax() == toAdd->GetZaxis()->GetXmax() &&
            CheckBinLabels(original->GetXaxis(), toAdd->GetXaxis()) &&
            CheckBinLabels(original->GetYaxis(), toAdd->GetYaxis()) &&
            CheckBinLabels(original->GetZaxis(), toAdd->GetZaxis())) {
          original->Add(toAdd);
        } else {
          edm::LogError("MergeFailure") << "Found histograms with different axis limits or different labels '"
                                        << original->GetName() << "' not merged.";
        }
      }
    }
  };

  using MonitorElementsFromFile = std::map<std::tuple<int, int>, std::vector<MonitorElementData*>>;

  // This struct allows to find all MEs belonging to a run-lumi pair
  // All files will be open at once so m_file property indicates the file where data is saved.
  struct FileMetadata {
    unsigned int m_run;
    unsigned int m_lumi;
    ULong64_t m_beginTime;
    ULong64_t m_endTime;
    ULong64_t m_firstIndex;
    ULong64_t m_lastIndex;  // Last is inclusive
    unsigned int m_type;
    TFile* m_file;

    // This will be used when sorting a vector
    bool operator<(const FileMetadata& obj) const {
      if (m_run == obj.m_run)
        return m_lumi < obj.m_lumi;
      else
        return m_run < obj.m_run;
    }

    void describe() {
      std::cout << "read r:" << m_run << " l:" << m_lumi << " bt:" << m_beginTime << " et:" << m_endTime
                << " fi:" << m_firstIndex << " li:" << m_lastIndex << " type:" << m_type << " file: " << m_file
                << std::endl;
    }
  };

  class TreeReaderBase {
  public:
    TreeReaderBase(MonitorElementData::Kind kind) : m_kind(kind) {}
    virtual ~TreeReaderBase() {}

    virtual void read(ULong64_t iIndex, MonitorElementsFromFile& mesFromFile, int run, int lumi) = 0;
    virtual void setTree(TTree* iTree) = 0;

  protected:
    MonitorElementData::Kind m_kind;
    TTree* m_tree;
  };

  template <class T>
  class TreeObjectReader : public TreeReaderBase {
  public:
    TreeObjectReader(MonitorElementData::Kind kind)
        : TreeReaderBase(kind), m_tree(nullptr), m_fullName(nullptr), m_buffer(nullptr), m_tag(0) {
      assert(m_kind != MonitorElementData::Kind::INT);
      assert(m_kind != MonitorElementData::Kind::REAL);
      assert(m_kind != MonitorElementData::Kind::STRING);
    }

    void read(ULong64_t iIndex, MonitorElementsFromFile& mesFromFile, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      MonitorElementData::Key key;
      key.kind_ = m_kind;
      key.path_.set(*m_fullName, MonitorElementData::Path::Type::DIR_AND_NAME);
      key.scope_ = lumi == 0 ? MonitorElementData::Scope::RUN : MonitorElementData::Scope::LUMI;
      // TODO: What should the range be for per run MEs? Now lumi will be 0 and not the max lumi for that run.
      key.coveredrange_ = edm::LuminosityBlockRange(run, lumi, run, lumi);

      std::vector<MonitorElementData*> runLumiMEs = mesFromFile[std::make_tuple(run, lumi)];
      bool merged = false;
      for (MonitorElementData* meData : runLumiMEs) {
        if (meData->key_ == key) {
          // Merge with already existing ME!
          MonitorElementData::Value::Access value(meData->value_);
          DQMMergeHelper::mergeTogether(value.object.get(), m_buffer);
          merged = true;
          break;
        }
      }

      if (!merged) {
        MonitorElementData* meData = new MonitorElementData();
        meData->key_ = key;
        {
          MonitorElementData::Value::Access value(meData->value_);
          value.object = std::unique_ptr<T>((T*)(m_buffer->Clone()));
        }

        mesFromFile[std::make_tuple(run, lumi)].push_back(meData);
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_buffer);
    }

  private:
    TTree* m_tree;
    std::string* m_fullName;
    T* m_buffer;
    uint32_t m_tag;
  };

  class TreeStringReader : public TreeReaderBase {
  public:
    TreeStringReader(MonitorElementData::Kind kind)
        : TreeReaderBase(kind), m_tree(nullptr), m_fullName(nullptr), m_value(nullptr), m_tag(0) {
      assert(m_kind == MonitorElementData::Kind::STRING);
    }

    void read(ULong64_t iIndex, MonitorElementsFromFile& mesFromFile, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      MonitorElementData::Key key;
      key.kind_ = m_kind;
      key.path_.set(*m_fullName, MonitorElementData::Path::Type::DIR_AND_NAME);
      key.scope_ = lumi == 0 ? MonitorElementData::Scope::RUN : MonitorElementData::Scope::LUMI;
      key.coveredrange_ = edm::LuminosityBlockRange(run, lumi, run, lumi);

      std::vector<MonitorElementData*> runLumiMEs = mesFromFile[std::make_tuple(run, lumi)];
      bool merged = false;
      for (MonitorElementData* meData : runLumiMEs) {
        if (meData->key_ == key) {
          // Keep the latest one
          MonitorElementData::Value::Access value(meData->value_);
          value.scalar.str = *m_value;
          merged = true;
          break;
        }
      }

      if (!merged) {
        MonitorElementData* meData = new MonitorElementData();
        meData->key_ = key;
        {
          MonitorElementData::Value::Access value(meData->value_);
          value.scalar.str = *m_value;
        }

        mesFromFile[std::make_tuple(run, lumi)].push_back(std::move(meData));
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_value);
    }

  private:
    TTree* m_tree;
    std::string* m_fullName;
    std::string* m_value;
    uint32_t m_tag;
  };

  template <class T>
  class TreeSimpleReader : public TreeReaderBase {
  public:
    TreeSimpleReader(MonitorElementData::Kind kind)
        : TreeReaderBase(kind), m_tree(nullptr), m_fullName(nullptr), m_buffer(0), m_tag(0) {
      assert(m_kind == MonitorElementData::Kind::INT || m_kind == MonitorElementData::Kind::REAL);
    }

    void read(ULong64_t iIndex, MonitorElementsFromFile& mesFromFile, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      MonitorElementData::Key key;
      key.kind_ = m_kind;
      key.path_.set(*m_fullName, MonitorElementData::Path::Type::DIR_AND_NAME);
      key.scope_ = lumi == 0 ? MonitorElementData::Scope::RUN : MonitorElementData::Scope::LUMI;
      key.coveredrange_ = edm::LuminosityBlockRange(run, lumi, run, lumi);

      std::vector<MonitorElementData*> runLumiMEs = mesFromFile[std::make_tuple(run, lumi)];
      bool merged = false;
      for (MonitorElementData* meData : runLumiMEs) {
        if (meData->key_ == key) {
          // Keep the latest one
          MonitorElementData::Value::Access value(meData->value_);
          if (m_kind == MonitorElementData::Kind::INT)
            value.scalar.num = m_buffer;
          else if (m_kind == MonitorElementData::Kind::REAL)
            value.scalar.real = m_buffer;
          merged = true;
          break;
        }
      }

      if (!merged) {
        MonitorElementData* meData = new MonitorElementData();
        meData->key_ = key;
        {
          MonitorElementData::Value::Access value(meData->value_);
          if (m_kind == MonitorElementData::Kind::INT)
            value.scalar.num = m_buffer;
          else if (m_kind == MonitorElementData::Kind::REAL)
            value.scalar.real = m_buffer;
        }

        mesFromFile[std::make_tuple(run, lumi)].push_back(std::move(meData));
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_buffer);
    }

  private:
    TTree* m_tree;
    std::string* m_fullName;
    T m_buffer;
    uint32_t m_tag;
  };

}  // namespace

class DQMRootSource : public edm::PuttableSourceBase {
public:
  DQMRootSource(edm::ParameterSet const&, const edm::InputSourceDescription&);
  ~DQMRootSource() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  DQMRootSource(const DQMRootSource&) = delete;

  edm::InputSource::ItemType getNextItemType() override;

  std::unique_ptr<edm::FileBlock> readFile_() override;
  std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
  void readRun_(edm::RunPrincipal& rpCache) override;
  void readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) override;
  void readEvent_(edm::EventPrincipal&) override;

  // Read MEs from m_fileMetadatas to m_MEsFromFile till run or lumi transition
  void readElements();
  // True if m_currentIndex points to an element that has a different
  // run or lumi than the previous element (a transition needs to happen).
  // False otherwise.
  bool isRunOrLumiTransition() const;
  void readNextItemType();

  // These methods will be called by the framework.
  // MEs in m_MEsFromFile will be put to products.
  void beginRun(edm::Run& run);
  void beginLuminosityBlock(edm::LuminosityBlock& lumi);

  // If the run matches the filterOnRun configuration parameter, the run
  // (and all its lumis) will be kept.
  // Otherwise, check if a run and a lumi are in the range that needs to be processed.
  // Range is retrieved from lumisToProcess configuration parameter.
  // If at least one lumi of a run needs to be kept, per run MEs of that run will also be kept.
  bool keepIt(edm::RunNumber_t, edm::LuminosityBlockNumber_t) const;
  void logFileAction(char const* msg, char const* fileName) const;

  const DQMRootSource& operator=(const DQMRootSource&) = delete;  // stop default

  // ---------- member data --------------------------------

  // Properties from python config
  bool m_skipBadFiles;
  unsigned int m_filterOnRun;
  edm::InputFileCatalog m_catalog;
  std::vector<edm::LuminosityBlockRange> m_lumisToProcess;

  edm::InputSource::ItemType m_nextItemType;
  // Each ME type gets its own reader
  std::vector<std::shared_ptr<TreeReaderBase>> m_treeReaders;

  // Index of currenlty processed row in m_fileMetadatas
  unsigned int m_currentIndex;
  // All open DQMIO files
  std::vector<TFile*> m_openFiles;
  // MEs read from files and merged if needed. Ready to be put into products
  MonitorElementsFromFile m_MEsFromFile;
  // An item here is a row read from DQMIO indices (metadata) table
  std::vector<FileMetadata> m_fileMetadatas;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

void DQMRootSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("fileNames")->setComment("Names of files to be processed.");
  desc.addUntracked<unsigned int>("filterOnRun", 0)->setComment("Just limit the process to the selected run.");
  desc.addUntracked<bool>("skipBadFiles", false)->setComment("Skip the file if it is not valid");
  desc.addUntracked<std::string>("overrideCatalog", std::string())
      ->setComment("An alternate file catalog to use instead of the standard site one.");
  std::vector<edm::LuminosityBlockRange> defaultLumis;
  desc.addUntracked<std::vector<edm::LuminosityBlockRange>>("lumisToProcess", defaultLumis)
      ->setComment("Skip any lumi inside the specified run:lumi range.");

  descriptions.addDefault(desc);
}

//
// constructors and destructor
//

DQMRootSource::DQMRootSource(edm::ParameterSet const& iPSet, const edm::InputSourceDescription& iDesc)
    : edm::PuttableSourceBase(iPSet, iDesc),
      m_skipBadFiles(iPSet.getUntrackedParameter<bool>("skipBadFiles", false)),
      m_filterOnRun(iPSet.getUntrackedParameter<unsigned int>("filterOnRun", 0)),
      m_catalog(iPSet.getUntrackedParameter<std::vector<std::string>>("fileNames"),
                iPSet.getUntrackedParameter<std::string>("overrideCatalog")),
      m_lumisToProcess(iPSet.getUntrackedParameter<std::vector<edm::LuminosityBlockRange>>(
          "lumisToProcess", std::vector<edm::LuminosityBlockRange>())),
      m_nextItemType(edm::InputSource::IsFile),
      m_treeReaders(kNIndicies, std::shared_ptr<TreeReaderBase>()),
      m_currentIndex(0),
      m_openFiles(std::vector<TFile*>()),
      m_MEsFromFile(MonitorElementsFromFile()),
      m_fileMetadatas(std::vector<FileMetadata>()) {
  edm::sortAndRemoveOverlaps(m_lumisToProcess);

  if (m_catalog.fileNames().size() == 0) {
    m_nextItemType = edm::InputSource::IsStop;
  } else {
    m_treeReaders[kIntIndex].reset(new TreeSimpleReader<Long64_t>(MonitorElementData::Kind::INT));
    m_treeReaders[kFloatIndex].reset(new TreeSimpleReader<double>(MonitorElementData::Kind::REAL));
    m_treeReaders[kStringIndex].reset(new TreeStringReader(MonitorElementData::Kind::STRING));
    m_treeReaders[kTH1FIndex].reset(new TreeObjectReader<TH1F>(MonitorElementData::Kind::TH1F));
    m_treeReaders[kTH1SIndex].reset(new TreeObjectReader<TH1S>(MonitorElementData::Kind::TH1S));
    m_treeReaders[kTH1DIndex].reset(new TreeObjectReader<TH1D>(MonitorElementData::Kind::TH1D));
    m_treeReaders[kTH2FIndex].reset(new TreeObjectReader<TH2F>(MonitorElementData::Kind::TH2F));
    m_treeReaders[kTH2SIndex].reset(new TreeObjectReader<TH2S>(MonitorElementData::Kind::TH2S));
    m_treeReaders[kTH2DIndex].reset(new TreeObjectReader<TH2D>(MonitorElementData::Kind::TH2D));
    m_treeReaders[kTH3FIndex].reset(new TreeObjectReader<TH3F>(MonitorElementData::Kind::TH3F));
    m_treeReaders[kTProfileIndex].reset(new TreeObjectReader<TProfile>(MonitorElementData::Kind::TPROFILE));
    m_treeReaders[kTProfile2DIndex].reset(new TreeObjectReader<TProfile2D>(MonitorElementData::Kind::TPROFILE2D));
  }

  produces<MonitorElementCollection, edm::Transition::BeginRun>("DQMGenerationRecoRun");
  produces<MonitorElementCollection, edm::Transition::BeginLuminosityBlock>("DQMGenerationRecoLumi");
}

DQMRootSource::~DQMRootSource() {
  for (auto& file : m_openFiles) {
    if (file != nullptr && file->IsOpen()) {
      file->Close();
      logFileAction("Closed file", "");
    }
  }
}

//
// member functions
//

edm::InputSource::ItemType DQMRootSource::getNextItemType() { return m_nextItemType; }

// We will read the metadata of all files and fill m_fileMetadatas vector
std::unique_ptr<edm::FileBlock> DQMRootSource::readFile_() {
  const int numFiles = m_catalog.fileNames().size();
  m_openFiles.reserve(numFiles);

  // TODO: add support to fallback files: https://github.com/cms-sw/cmssw/pull/28064/files
  for (auto& filename : m_catalog.fileNames()) {
    TFile* file;

    // Try to open a file
    try {
      file = TFile::Open(filename.c_str());

      // Exception will be trapped so we pull it out ourselves
      std::exception_ptr e = edm::threadLocalException::getException();
      if (e != std::exception_ptr()) {
        edm::threadLocalException::setException(std::exception_ptr());
        std::rethrow_exception(e);
      }

      m_openFiles.insert(m_openFiles.begin(), file);
    } catch (cms::Exception const& e) {
      if (!m_skipBadFiles) {
        edm::Exception ex(edm::errors::FileOpenError, "", e);
        ex.addContext("Opening DQM Root file");
        ex << "\nInput file " << filename << " was not found, could not be opened, or is corrupted.\n";
        throw ex;
      }
    }

    // Check if a file is usable
    if (!file->IsZombie()) {
      logFileAction("Successfully opened file ", filename.c_str());
    } else {
      if (!m_skipBadFiles) {
        edm::Exception ex(edm::errors::FileOpenError);
        ex << "Input file " << filename.c_str() << " could not be opened.\n";
        ex.addContext("Opening DQM Root file");
        throw ex;
      }
    }

    // Check file format version, which is encoded in the Title of the TFile
    if (strcmp(file->GetTitle(), "1") != 0) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "Input file " << filename.c_str() << " does not appear to be a DQM Root file.\n";
    }

    // Read metadata from the file
    TTree* indicesTree = dynamic_cast<TTree*>(file->Get(kIndicesTree));
    assert(indicesTree != nullptr);

    FileMetadata temp;
    // Each line of metadata will be read into the coresponding fields of temp.
    indicesTree->SetBranchAddress(kRunBranch, &temp.m_run);
    indicesTree->SetBranchAddress(kLumiBranch, &temp.m_lumi);
    indicesTree->SetBranchAddress(kBeginTimeBranch, &temp.m_beginTime);
    indicesTree->SetBranchAddress(kEndTimeBranch, &temp.m_endTime);
    indicesTree->SetBranchAddress(kTypeBranch, &temp.m_type);
    indicesTree->SetBranchAddress(kFirstIndex, &temp.m_firstIndex);
    indicesTree->SetBranchAddress(kLastIndex, &temp.m_lastIndex);

    for (Long64_t index = 0; index != indicesTree->GetEntries(); ++index) {
      indicesTree->GetEntry(index);
      temp.m_file = file;

      if (keepIt(temp.m_run, temp.m_lumi)) {
        m_fileMetadatas.push_back(temp);
      }
    }
  }

  // Sort to make sure runs and lumis appear in sequential order
  std::sort(m_fileMetadatas.begin(), m_fileMetadatas.end());

  for (auto& metadata : m_fileMetadatas)
    metadata.describe();

  // Stop if there's nothing to process. Otherwise start the run.
  if (m_fileMetadatas.size() == 0)
    m_nextItemType = edm::InputSource::IsStop;
  else
    m_nextItemType = edm::InputSource::IsRun;

  // We have to return something but not sure why
  return std::unique_ptr<edm::FileBlock>(new edm::FileBlock);
}

std::shared_ptr<edm::RunAuxiliary> DQMRootSource::readRunAuxiliary_() {
  FileMetadata metadata = m_fileMetadatas[m_currentIndex];
  auto runAux =
      edm::RunAuxiliary(metadata.m_run, edm::Timestamp(metadata.m_beginTime), edm::Timestamp(metadata.m_endTime));
  return std::make_shared<edm::RunAuxiliary>(runAux);
}

std::shared_ptr<edm::LuminosityBlockAuxiliary> DQMRootSource::readLuminosityBlockAuxiliary_() {
  FileMetadata metadata = m_fileMetadatas[m_currentIndex];
  auto lumiAux = edm::LuminosityBlockAuxiliary(edm::LuminosityBlockID(metadata.m_run, metadata.m_lumi),
                                               edm::Timestamp(metadata.m_beginTime),
                                               edm::Timestamp(metadata.m_endTime));
  return std::make_shared<edm::LuminosityBlockAuxiliary>(lumiAux);
}

void DQMRootSource::readRun_(edm::RunPrincipal& rpCache) {
  // Read elements of a current run.
  do {
    FileMetadata metadata = m_fileMetadatas[m_currentIndex];
    if (metadata.m_lumi == 0) {
      readElements();
    }
    m_currentIndex++;
  } while (!isRunOrLumiTransition());

  readNextItemType();
}

void DQMRootSource::readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) {
  // Read elements of a current lumi.
  do {
    readElements();
    m_currentIndex++;
  } while (!isRunOrLumiTransition());

  readNextItemType();
}

void DQMRootSource::readEvent_(edm::EventPrincipal&) {}

void DQMRootSource::readElements() {
  FileMetadata metadata = m_fileMetadatas[m_currentIndex];

  if (metadata.m_type != kNoTypesStored) {
    std::shared_ptr<TreeReaderBase> reader = m_treeReaders[metadata.m_type];
    TTree* tree = dynamic_cast<TTree*>(metadata.m_file->Get(kTypeNames[metadata.m_type]));
    reader->setTree(tree);

    ULong64_t index = metadata.m_firstIndex;
    ULong64_t endIndex = metadata.m_lastIndex + 1;

    for (; index != endIndex; ++index) {
      reader->read(index, m_MEsFromFile, metadata.m_run, metadata.m_lumi);
    }
  }
}

bool DQMRootSource::isRunOrLumiTransition() const {
  if (m_currentIndex == 0) {
    return false;
  }

  if (m_currentIndex > m_fileMetadatas.size() - 1) {
    // We reached the end
    return true;
  }

  FileMetadata previousMetadata = m_fileMetadatas[m_currentIndex - 1];
  FileMetadata metadata = m_fileMetadatas[m_currentIndex];

  return previousMetadata.m_run != metadata.m_run || previousMetadata.m_lumi != metadata.m_lumi;
}

void DQMRootSource::readNextItemType() {
  if (m_currentIndex == 0) {
    m_nextItemType = edm::InputSource::IsRun;
  } else if (m_currentIndex > m_fileMetadatas.size() - 1) {
    // We reached the end
    m_nextItemType = edm::InputSource::IsStop;
  } else {
    FileMetadata previousMetadata = m_fileMetadatas[m_currentIndex - 1];
    FileMetadata metadata = m_fileMetadatas[m_currentIndex];

    if (previousMetadata.m_run != metadata.m_run) {
      m_nextItemType = edm::InputSource::IsRun;
    } else if (previousMetadata.m_lumi != metadata.m_lumi) {
      m_nextItemType = edm::InputSource::IsLumi;
    }
  }
}

void DQMRootSource::beginRun(edm::Run& run) {
  TRACE("Begin run: " + std::to_string(run.run()))

  std::unique_ptr<MonitorElementCollection> product = std::make_unique<MonitorElementCollection>();

  auto mes = m_MEsFromFile[std::make_tuple(run.run(), 0)];

  TRACE("Found MEs: " + std::to_string(mes.size()))

  for (MonitorElementData* meData_ptr : mes) {
    product->push_back(meData_ptr);
  }

  run.put(std::move(product), "DQMGenerationRecoRun");

  // Remove already processed MEs
  m_MEsFromFile[std::make_tuple(run.run(), 0)] = std::vector<MonitorElementData*>();
}

void DQMRootSource::beginLuminosityBlock(edm::LuminosityBlock& lumi) {
  TRACE("Begin lumi: " + std::to_string(lumi.luminosityBlock()))

  std::unique_ptr<MonitorElementCollection> product = std::make_unique<MonitorElementCollection>();

  auto mes = m_MEsFromFile[std::make_tuple(lumi.run(), lumi.luminosityBlock())];

  TRACE("Found MEs: " + std::to_string(mes.size()))

  for (MonitorElementData* meData_ptr : mes) {
    assert(meData_ptr != nullptr);
    product->push_back(meData_ptr);
  }

  lumi.put(std::move(product), "DQMGenerationRecoLumi");

  // Remove already processed MEs
  m_MEsFromFile[std::make_tuple(lumi.run(), lumi.luminosityBlock())] = std::vector<MonitorElementData*>();
}

bool DQMRootSource::keepIt(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi) const {
  if (run == m_filterOnRun)
    return true;

  for (edm::LuminosityBlockRange const& lumiToProcess : m_lumisToProcess) {
    if (run >= lumiToProcess.startRun() && run <= lumiToProcess.endRun()) {
      if (lumi >= lumiToProcess.startLumi() && lumi <= lumiToProcess.endLumi()) {
        return true;
      } else if (lumi == 0) {
        return true;
      }
    }
  }
  return false;
}

void DQMRootSource::logFileAction(char const* msg, char const* fileName) const {
  edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << msg << fileName;
  edm::FlushMessageLog();
}

//
// const member functions
//

//
// static member functions
//
DEFINE_FWK_INPUT_SOURCE(DQMRootSource);
