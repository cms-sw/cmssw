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
#include <memory>

#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "format.h"

// class rather than namespace so we can make this a friend of the
// MonitorElement to get access to constructors etc.
struct DQMTTreeIO {
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

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
        std::string_view label1 = a1->GetBinLabel(i);
        std::string_view label2 = a2->GetBinLabel(i);
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
        // TODO: Redo. This is both more strict than what ROOT checks for yet
        // allows cases where ROOT fails with merging.
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
    TreeReaderBase(MonitorElementData::Kind kind, MonitorElementData::Scope rescope)
        : m_kind(kind), m_rescope(rescope) {}
    virtual ~TreeReaderBase() {}

    MonitorElementData::Key makeKey(std::string const& fullname, int run, int lumi) {
      MonitorElementData::Key key;
      key.kind_ = m_kind;
      key.path_.set(fullname, MonitorElementData::Path::Type::DIR_AND_NAME);
      if (m_rescope == MonitorElementData::Scope::LUMI) {
        // no rescoping
        key.scope_ = lumi == 0 ? MonitorElementData::Scope::RUN : MonitorElementData::Scope::LUMI;
        key.id_ = edm::LuminosityBlockID(run, lumi);
      } else if (m_rescope == MonitorElementData::Scope::RUN) {
        // everything becomes run, we'll never see Scope::JOB inside DQMIO files.
        key.scope_ = MonitorElementData::Scope::RUN;
        key.id_ = edm::LuminosityBlockID(run, 0);
      } else if (m_rescope == MonitorElementData::Scope::JOB) {
        // Everything is aggregated over the entire job.
        key.scope_ = MonitorElementData::Scope::JOB;
        key.id_ = edm::LuminosityBlockID(0, 0);
      } else {
        assert(!"Invalid Scope in rescope option.");
      }
      return key;
    }
    virtual void read(ULong64_t iIndex, DQMStore* dqmstore, int run, int lumi) = 0;
    virtual void setTree(TTree* iTree) = 0;

  protected:
    MonitorElementData::Kind m_kind;
    MonitorElementData::Scope m_rescope;
  };

  template <class T>
  class TreeObjectReader : public TreeReaderBase {
  public:
    TreeObjectReader(MonitorElementData::Kind kind, MonitorElementData::Scope rescope) : TreeReaderBase(kind, rescope) {
      assert(m_kind != MonitorElementData::Kind::INT);
      assert(m_kind != MonitorElementData::Kind::REAL);
      assert(m_kind != MonitorElementData::Kind::STRING);
    }

    void read(ULong64_t iIndex, DQMStore* dqmstore, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      auto key = makeKey(*m_fullName, run, lumi);
      auto existing = dqmstore->findOrRecycle(key);
      if (existing) {
        // TODO: make sure there is sufficient locking here.
        DQMMergeHelper::mergeTogether(existing->getTH1(), m_buffer);
      } else {
        // We make our own MEs here, to avoid a round-trip through the booking API.
        MonitorElementData meData;
        meData.key_ = key;
        meData.value_.object_ = std::unique_ptr<T>((T*)(m_buffer->Clone()));
        auto me = new MonitorElement(std::move(meData));
        dqmstore->putME(me);
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_buffer);
    }

  private:
    TTree* m_tree = nullptr;
    std::string* m_fullName = nullptr;
    T* m_buffer = nullptr;
    uint32_t m_tag = 0;
  };

  class TreeStringReader : public TreeReaderBase {
  public:
    TreeStringReader(MonitorElementData::Kind kind, MonitorElementData::Scope rescope) : TreeReaderBase(kind, rescope) {
      assert(m_kind == MonitorElementData::Kind::STRING);
    }

    void read(ULong64_t iIndex, DQMStore* dqmstore, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      auto key = makeKey(*m_fullName, run, lumi);
      auto existing = dqmstore->findOrRecycle(key);

      if (existing) {
        existing->Fill(*m_value);
      } else {
        // We make our own MEs here, to avoid a round-trip through the booking API.
        MonitorElementData meData;
        meData.key_ = key;
        meData.value_.scalar_.str = *m_value;
        auto me = new MonitorElement(std::move(meData));
        dqmstore->putME(me);
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_value);
    }

  private:
    TTree* m_tree = nullptr;
    std::string* m_fullName = nullptr;
    std::string* m_value = nullptr;
    uint32_t m_tag = 0;
  };

  template <class T>
  class TreeSimpleReader : public TreeReaderBase {
  public:
    TreeSimpleReader(MonitorElementData::Kind kind, MonitorElementData::Scope rescope) : TreeReaderBase(kind, rescope) {
      assert(m_kind == MonitorElementData::Kind::INT || m_kind == MonitorElementData::Kind::REAL);
    }

    void read(ULong64_t iIndex, DQMStore* dqmstore, int run, int lumi) override {
      // This will populate the fields as defined in setTree method
      m_tree->GetEntry(iIndex);

      auto key = makeKey(*m_fullName, run, lumi);
      auto existing = dqmstore->findOrRecycle(key);

      if (existing) {
        existing->Fill(m_buffer);
      } else {
        // We make our own MEs here, to avoid a round-trip through the booking API.
        MonitorElementData meData;
        meData.key_ = key;
        if (m_kind == MonitorElementData::Kind::INT)
          meData.value_.scalar_.num = m_buffer;
        else if (m_kind == MonitorElementData::Kind::REAL)
          meData.value_.scalar_.real = m_buffer;
        auto me = new MonitorElement(std::move(meData));
        dqmstore->putME(me);
      }
    }

    void setTree(TTree* iTree) override {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch, &m_fullName);
      m_tree->SetBranchAddress(kFlagBranch, &m_tag);
      m_tree->SetBranchAddress(kValueBranch, &m_buffer);
    }

  private:
    TTree* m_tree = nullptr;
    std::string* m_fullName = nullptr;
    T m_buffer = 0;
    uint32_t m_tag = 0;
  };
};

class DQMRootSource : public edm::PuttableSourceBase, DQMTTreeIO {
public:
  DQMRootSource(edm::ParameterSet const&, const edm::InputSourceDescription&);
  DQMRootSource(const DQMRootSource&) = delete;
  ~DQMRootSource() override;

  // ---------- const member functions ---------------------

  const DQMRootSource& operator=(const DQMRootSource&) = delete;  // stop default

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputSource::ItemType getNextItemType() override;

  std::shared_ptr<edm::FileBlock> readFile_() override;
  std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
  void readRun_(edm::RunPrincipal& rpCache) override;
  void readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) override;
  void readEvent_(edm::EventPrincipal&) override;

  // Read MEs from m_fileMetadatas to DQMStore  till run or lumi transition
  void readElements();
  // True if m_currentIndex points to an element that has a different
  // run or lumi than the previous element (a transition needs to happen).
  // False otherwise.
  bool isRunOrLumiTransition() const;
  void readNextItemType();

  // These methods will be called by the framework.
  // MEs in DQMStore  will be put to products.
  void beginRun(edm::Run& run) override;
  void beginLuminosityBlock(edm::LuminosityBlock& lumi) override;

  // If the run matches the filterOnRun configuration parameter, the run
  // (and all its lumis) will be kept.
  // Otherwise, check if a run and a lumi are in the range that needs to be processed.
  // Range is retrieved from lumisToProcess configuration parameter.
  // If at least one lumi of a run needs to be kept, per run MEs of that run will also be kept.
  bool keepIt(edm::RunNumber_t, edm::LuminosityBlockNumber_t) const;
  void logFileAction(char const* msg, char const* fileName) const;

  // ---------- member data --------------------------------

  // Properties from python config
  bool m_skipBadFiles;
  unsigned int m_filterOnRun;
  edm::InputFileCatalog m_catalog;
  std::vector<edm::LuminosityBlockRange> m_lumisToProcess;
  MonitorElementData::Scope m_rescope;

  edm::InputSource::ItemType m_nextItemType;
  // Each ME type gets its own reader
  std::vector<std::shared_ptr<TreeReaderBase>> m_treeReaders;

  // Index of currenlty processed row in m_fileMetadatas
  unsigned int m_currentIndex;

  // All open DQMIO files
  struct OpenFileInfo {
    OpenFileInfo(TFile* file, edm::JobReport::Token jrToken) : m_file(file), m_jrToken(jrToken) {}
    ~OpenFileInfo() {
      edm::Service<edm::JobReport> jr;
      jr->inputFileClosed(edm::InputType::Primary, m_jrToken);
    }

    OpenFileInfo(OpenFileInfo&&) = default;
    OpenFileInfo& operator=(OpenFileInfo&&) = default;

    std::unique_ptr<TFile> m_file;
    edm::JobReport::Token m_jrToken;
  };
  std::vector<OpenFileInfo> m_openFiles;

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
  desc.addUntracked<std::string>("reScope", "JOB")
      ->setComment(
          "Accumulate histograms more coarsely."
          " Options: \"\": keep unchanged, \"RUN\": turn LUMI histograms into RUN histograms, \"JOB\": turn everything "
          "into JOB histograms.");
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
      m_rescope(std::map<std::string, MonitorElementData::Scope>{
          {"", MonitorElementData::Scope::LUMI},
          {"LUMI", MonitorElementData::Scope::LUMI},
          {"RUN", MonitorElementData::Scope::RUN},
          {"JOB", MonitorElementData::Scope::JOB}}[iPSet.getUntrackedParameter<std::string>("reScope", "JOB")]),
      m_nextItemType(edm::InputSource::IsFile),
      m_treeReaders(kNIndicies, std::shared_ptr<TreeReaderBase>()),
      m_currentIndex(0),
      m_openFiles(std::vector<OpenFileInfo>()),
      m_fileMetadatas(std::vector<FileMetadata>()) {
  edm::sortAndRemoveOverlaps(m_lumisToProcess);

  if (m_catalog.fileNames(0).empty()) {
    m_nextItemType = edm::InputSource::IsStop;
  } else {
    m_treeReaders[kIntIndex].reset(new TreeSimpleReader<Long64_t>(MonitorElementData::Kind::INT, m_rescope));
    m_treeReaders[kFloatIndex].reset(new TreeSimpleReader<double>(MonitorElementData::Kind::REAL, m_rescope));
    m_treeReaders[kStringIndex].reset(new TreeStringReader(MonitorElementData::Kind::STRING, m_rescope));
    m_treeReaders[kTH1FIndex].reset(new TreeObjectReader<TH1F>(MonitorElementData::Kind::TH1F, m_rescope));
    m_treeReaders[kTH1SIndex].reset(new TreeObjectReader<TH1S>(MonitorElementData::Kind::TH1S, m_rescope));
    m_treeReaders[kTH1DIndex].reset(new TreeObjectReader<TH1D>(MonitorElementData::Kind::TH1D, m_rescope));
    m_treeReaders[kTH1IIndex].reset(new TreeObjectReader<TH1I>(MonitorElementData::Kind::TH1I, m_rescope));
    m_treeReaders[kTH2FIndex].reset(new TreeObjectReader<TH2F>(MonitorElementData::Kind::TH2F, m_rescope));
    m_treeReaders[kTH2SIndex].reset(new TreeObjectReader<TH2S>(MonitorElementData::Kind::TH2S, m_rescope));
    m_treeReaders[kTH2DIndex].reset(new TreeObjectReader<TH2D>(MonitorElementData::Kind::TH2D, m_rescope));
    m_treeReaders[kTH2IIndex].reset(new TreeObjectReader<TH2I>(MonitorElementData::Kind::TH2I, m_rescope));
    m_treeReaders[kTH3FIndex].reset(new TreeObjectReader<TH3F>(MonitorElementData::Kind::TH3F, m_rescope));
    m_treeReaders[kTProfileIndex].reset(new TreeObjectReader<TProfile>(MonitorElementData::Kind::TPROFILE, m_rescope));
    m_treeReaders[kTProfile2DIndex].reset(
        new TreeObjectReader<TProfile2D>(MonitorElementData::Kind::TPROFILE2D, m_rescope));
  }

  produces<DQMToken, edm::Transition::BeginRun>("DQMGenerationRecoRun");
  produces<DQMToken, edm::Transition::BeginLuminosityBlock>("DQMGenerationRecoLumi");
}

DQMRootSource::~DQMRootSource() {
  for (auto& file : m_openFiles) {
    if (file.m_file && file.m_file->IsOpen()) {
      logFileAction("Closed file", "");
    }
  }
}

//
// member functions
//

edm::InputSource::ItemType DQMRootSource::getNextItemType() { return m_nextItemType; }

// We will read the metadata of all files and fill m_fileMetadatas vector
std::shared_ptr<edm::FileBlock> DQMRootSource::readFile_() {
  const int numFiles = m_catalog.fileNames(0).size();
  m_openFiles.reserve(numFiles);

  for (auto& fileitem : m_catalog.fileCatalogItems()) {
    TFile* file;
    std::string pfn;
    std::string lfn;
    std::list<std::string> exInfo;
    //loop over names of a file, each of them corresponds to a data catalog
    bool isGoodFile(true);
    //get all names of a file, each of them corresponds to a data catalog
    const std::vector<std::string>& fNames = fileitem.fileNames();
    for (std::vector<std::string>::const_iterator it = fNames.begin(); it != fNames.end(); ++it) {
      // Try to open a file
      try {
        file = TFile::Open(it->c_str());

        // Exception will be trapped so we pull it out ourselves
        std::exception_ptr e = edm::threadLocalException::getException();
        if (e != std::exception_ptr()) {
          edm::threadLocalException::setException(std::exception_ptr());
          std::rethrow_exception(e);
        }

      } catch (cms::Exception const& e) {
        file = nullptr;                       // is there anything we need to free?
        if (std::next(it) == fNames.end()) {  //last name corresponding to the last data catalog to try
          if (!m_skipBadFiles) {
            edm::Exception ex(edm::errors::FileOpenError, "", e);
            ex.addContext("Opening DQM Root file");
            ex << "\nInput file " << *it << " was not found, could not be opened, or is corrupted.\n";
            //report previous exceptions when use other names to open file
            for (auto const& s : exInfo)
              ex.addAdditionalInfo(s);
            throw ex;
          }
          isGoodFile = false;
        }
        // save in case of error when trying next name
        for (auto const& s : e.additionalInfo())
          exInfo.push_back(s);
      }

      // Check if a file is usable
      if (file && !file->IsZombie()) {
        logFileAction("Successfully opened file ", it->c_str());
        pfn = *it;
        lfn = fileitem.logicalFileName();
        break;
      } else {
        if (std::next(it) == fNames.end()) {
          if (!m_skipBadFiles) {
            edm::Exception ex(edm::errors::FileOpenError);
            ex << "Input file " << *it << " could not be opened.\n";
            ex.addContext("Opening DQM Root file");
            //report previous exceptions when use other names to open file
            for (auto const& s : exInfo)
              ex.addAdditionalInfo(s);
            throw ex;
          }
          isGoodFile = false;
        }
        if (file) {
          delete file;
          file = nullptr;
        }
      }
    }  //end loop over names of the file

    if (!isGoodFile && m_skipBadFiles)
      continue;

    std::unique_ptr<std::string> guid{file->Get<std::string>(kCmsGuid)};
    if (not guid) {
      guid = std::make_unique<std::string>(file->GetUUID().AsString());
      std::transform(guid->begin(), guid->end(), guid->begin(), (int (*)(int))std::toupper);
    }

    edm::Service<edm::JobReport> jr;
    auto jrToken = jr->inputFileOpened(
        pfn, lfn, std::string(), std::string(), "DQMRootSource", "source", *guid, std::vector<std::string>());
    m_openFiles.emplace_back(file, jrToken);

    // Check file format version, which is encoded in the Title of the TFile
    if (strcmp(file->GetTitle(), "1") != 0) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "Input file " << fNames[0] << " does not appear to be a DQM Root file.\n";
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

  }  //end loop over files

  // Sort to make sure runs and lumis appear in sequential order
  std::stable_sort(m_fileMetadatas.begin(), m_fileMetadatas.end());

  // If we have lumisections without matching runs, insert dummy runs here.
  unsigned int run = 0;
  auto toadd = std::vector<FileMetadata>();
  for (auto& metadata : m_fileMetadatas) {
    if (run < metadata.m_run && metadata.m_lumi != 0) {
      // run transition and lumi transition at the same time!
      FileMetadata dummy{};  // zero initialize
      dummy.m_run = metadata.m_run;
      dummy.m_lumi = 0;
      dummy.m_type = kNoTypesStored;
      toadd.push_back(dummy);
    }
    run = metadata.m_run;
  }

  if (!toadd.empty()) {
    // rather than trying to insert at the right places, just append and sort again.
    m_fileMetadatas.insert(m_fileMetadatas.end(), toadd.begin(), toadd.end());
    std::stable_sort(m_fileMetadatas.begin(), m_fileMetadatas.end());
  }

  //for (auto& metadata : m_fileMetadatas)
  //  metadata.describe();

  // Stop if there's nothing to process. Otherwise start the run.
  if (m_fileMetadatas.empty())
    m_nextItemType = edm::InputSource::IsStop;
  else
    m_nextItemType = edm::InputSource::IsRun;

  // We have to return something but not sure why
  return std::make_shared<edm::FileBlock>();
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

  edm::Service<edm::JobReport> jr;
  jr->reportInputRunNumber(rpCache.id().run());
  rpCache.fillRunPrincipal(processHistoryRegistryForUpdate());
}

void DQMRootSource::readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) {
  // Read elements of a current lumi.
  do {
    readElements();
    m_currentIndex++;
  } while (!isRunOrLumiTransition());

  readNextItemType();

  edm::Service<edm::JobReport> jr;
  jr->reportInputLumiSection(lbCache.id().run(), lbCache.id().luminosityBlock());
  lbCache.fillLuminosityBlockPrincipal(processHistoryRegistry().getMapped(lbCache.aux().processHistoryID()));
}

void DQMRootSource::readEvent_(edm::EventPrincipal&) {}

void DQMRootSource::readElements() {
  FileMetadata metadata = m_fileMetadatas[m_currentIndex];

  if (metadata.m_type != kNoTypesStored) {
    std::shared_ptr<TreeReaderBase> reader = m_treeReaders[metadata.m_type];
    TTree* tree = dynamic_cast<TTree*>(metadata.m_file->Get(kTypeNames[metadata.m_type]));
    // The Reset() below screws up the tree, so we need to re-read it from file
    // before use here.
    tree->Refresh();

    reader->setTree(tree);

    ULong64_t index = metadata.m_firstIndex;
    ULong64_t endIndex = metadata.m_lastIndex + 1;

    for (; index != endIndex; ++index) {
      reader->read(index, edm::Service<DQMStore>().operator->(), metadata.m_run, metadata.m_lumi);
    }
    // Drop buffers in the TTree. This reduces memory consuption while the tree
    // just sits there and waits for the next block to be read.
    tree->Reset();
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
  std::unique_ptr<DQMToken> product = std::make_unique<DQMToken>();
  run.put(std::move(product), "DQMGenerationRecoRun");
}

void DQMRootSource::beginLuminosityBlock(edm::LuminosityBlock& lumi) {
  std::unique_ptr<DQMToken> product = std::make_unique<DQMToken>();
  lumi.put(std::move(product), "DQMGenerationRecoLumi");
}

bool DQMRootSource::keepIt(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi) const {
  if (m_filterOnRun != 0 && run != m_filterOnRun) {
    return false;
  }

  if (m_lumisToProcess.empty()) {
    return true;
  }

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
