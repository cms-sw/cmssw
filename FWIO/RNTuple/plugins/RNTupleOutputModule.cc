#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"

#include "RNTupleOutputFile.h"

#include <string>
#include <map>
#include <regex>
#include "boost/algorithm/string.hpp"

using namespace ROOT::Experimental;

namespace {
  edm::rntuple::CompressionAlgos convertTo(std::string const& iName) {
    if (iName == "LZMA") {
      return edm::rntuple::CompressionAlgos::kLZMA;
    }
    if (iName == "ZSTD") {
      return edm::rntuple::CompressionAlgos::kZSTD;
    }
    if (iName == "LZ4") {
      return edm::rntuple::CompressionAlgos::kLZ4;
    }
    if (iName == "ZLIB") {
      return edm::rntuple::CompressionAlgos::kZLIB;
    }
    throw cms::Exception("UnknownCompression") << "An unknown compression algorithm was specified: " << iName;
  }

  struct SetStreamerForDataProduct {
    SetStreamerForDataProduct(std::string const& iName, bool iUseStreamer)
        : branch_(convert(iName)), useStreamer_(iUseStreamer) {}
    bool match(std::string const& iName) const;
    std::regex convert(std::string const& iGlobBranchExpression) const;

    std::regex branch_;
    bool useStreamer_;
  };

  inline bool SetStreamerForDataProduct::match(std::string const& iBranchName) const {
    return std::regex_match(iBranchName, branch_);
  }

  std::regex SetStreamerForDataProduct::convert(std::string const& iGlobBranchExpression) const {
    std::string tmp(iGlobBranchExpression);
    boost::replace_all(tmp, "*", ".*");
    boost::replace_all(tmp, "?", ".");
    return std::regex(tmp);
  }

  std::vector<SetStreamerForDataProduct> fromConfig(std::vector<edm::ParameterSet> const& iConfig) {
    std::vector<SetStreamerForDataProduct> returnValue;
    returnValue.reserve(iConfig.size());

    for (auto const& prod : iConfig) {
      returnValue.emplace_back(prod.getUntrackedParameter<std::string>("product"),
                               prod.getUntrackedParameter<bool>("useStreamer"));
    }
    return returnValue;
  }

  std::optional<bool> useStreamer(std::string const& iName, std::vector<SetStreamerForDataProduct> const& iSpecial) {
    auto nameNoDot = iName.substr(0, iName.size() - 1);
    for (auto const& prod : iSpecial) {
      if (prod.match(nameNoDot)) {
        return prod.useStreamer_;
      }
    }
    return {};
  }

}  // namespace

namespace edm {

  class RNTupleOutputModule : public one::OutputModule<> {
  public:
    explicit RNTupleOutputModule(ParameterSet const& pset);
    ~RNTupleOutputModule() final;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) final;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) final;
    void writeRun(RunForOutput const&) final;
    void reallyCloseFile() final;
    void openFile(FileBlock const& fb) final;
    void initialRegistry(edm::ProductRegistry const& iReg) final;
    std::string fileName_;
    std::unique_ptr<edm::ProductRegistry const> reg_;
    std::unique_ptr<RNTupleOutputFile> file_;
    std::vector<SetStreamerForDataProduct> overrideStreamer_;
    std::vector<std::string> noSplitSubFields_;
    rntuple::CompressionAlgos compressionAlgo_;
    unsigned int compressionLevel_;
    unsigned long long approxZippedClusterSize_;
    unsigned long long maxUnzippedClusterSize_;
    unsigned long long initialUnzippedPageSize_;
    unsigned long long maxUnzippedPageSize_;
    unsigned long long pageBufferBudget_;
    bool useBufferedWrite_;
    bool useDirectIO_;
    bool dropMetaData_;
    bool useStreamer_;
  };

  RNTupleOutputModule::RNTupleOutputModule(ParameterSet const& pset)
      : one::OutputModuleBase(pset),
        one::OutputModule<>(pset),
        fileName_(pset.getUntrackedParameter<std::string>("fileName")),
        overrideStreamer_(
            fromConfig(pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("overrideDataProductStreamer"))),
        noSplitSubFields_(pset.getUntrackedParameter<std::vector<std::string>>("noSplitSubFields")),
        compressionAlgo_(convertTo(pset.getUntrackedParameter<std::string>("compressionAlgorithm"))),
        compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel")),
        approxZippedClusterSize_(pset.getUntrackedParameter<unsigned long long>("approxZippedClusterSize")),
        maxUnzippedClusterSize_(pset.getUntrackedParameter<unsigned long long>("maxUnzippedClusterSize")),
        initialUnzippedPageSize_(pset.getUntrackedParameter<unsigned long long>("initialUnzippedPageSize")),
        maxUnzippedPageSize_(pset.getUntrackedParameter<unsigned long long>("maxUnzippedPageSize")),
        pageBufferBudget_(pset.getUntrackedParameter<unsigned long long>("pageBufferBudget")),
        useBufferedWrite_(pset.getUntrackedParameter<bool>("useBufferedWrite")),
        useDirectIO_(pset.getUntrackedParameter<bool>("useDirectIO")),
        dropMetaData_(pset.getUntrackedParameter<bool>("dropPerEventDataProductProvenance")),
        useStreamer_(pset.getUntrackedParameter<bool>("useStreamer")) {}

  void RNTupleOutputModule::openFile(FileBlock const& fb) {
    RNTupleOutputFile::Config conf;
    conf.wantAllEvents = wantAllEvents();
    conf.selectorConfig = selectorConfig();
    conf.compressionAlgo = compressionAlgo_;
    conf.compressionLevel = compressionLevel_;
    conf.approxZippedClusterSize = approxZippedClusterSize_;
    conf.maxUnzippedClusterSize = maxUnzippedClusterSize_;
    conf.initialUnzippedPageSize = initialUnzippedPageSize_;
    conf.maxUnzippedPageSize = maxUnzippedPageSize_;
    conf.pageBufferBudget = pageBufferBudget_;
    conf.useBufferedWrite = useBufferedWrite_;
    conf.useDirectIO = useDirectIO_;
    conf.dropMetaData = dropMetaData_;
    conf.doNotSplitSubFields = noSplitSubFields_;
    if (useStreamer_ and overrideStreamer_.empty()) {
      auto const& prods = keptProducts()[InEvent];
      conf.streamerProduct = std::vector<bool>(prods.size(), true);
    } else if (not overrideStreamer_.empty()) {
      auto const& prods = keptProducts()[InEvent];
      conf.streamerProduct = std::vector<bool>(prods.size(), useStreamer_);
      unsigned int index = 0;
      for (auto const& prod : prods) {
        auto choice = useStreamer(prod.first->branchName(), overrideStreamer_);
        if (choice) {
          if (*choice != useStreamer_) {
            conf.streamerProduct[index] = *choice;
          }
        }
        ++index;
      }
    }
    assert(reg_);
    file_ = std::make_unique<RNTupleOutputFile>(fileName_, fb, keptProducts(), conf, reg_->anyProductProduced());
  }

  void RNTupleOutputModule::initialRegistry(edm::ProductRegistry const& iReg) {
    reg_ = std::make_unique<ProductRegistry>(iReg.productList());
  }

  void RNTupleOutputModule::reallyCloseFile() {
    if (file_) {
      assert(reg_);
      file_->reallyCloseFile(*branchIDLists(), *thinnedAssociationsHelper(), *reg_);
    }
  }

  RNTupleOutputModule::~RNTupleOutputModule() = default;

  void RNTupleOutputModule::write(EventForOutput const& e) { file_->write(e); }

  void RNTupleOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& iLumi) {
    file_->writeLuminosityBlock(iLumi);
  }

  void RNTupleOutputModule::writeRun(RunForOutput const& iRun) { file_->writeRun(iRun); }

  void RNTupleOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Outputs event information into an RNTuple container.");
    desc.addUntracked<std::string>("fileName")->setComment("RNTuple file to read");
    desc.addUntracked<std::string>("compressionAlgorithm", "ZSTD")
        ->setComment(
            "Algorithm used to compress data in the ROOT output file, allowed values are ZLIB, LZMA, LZ4, and ZSTD");
    desc.addUntracked<int>("compressionLevel", 4)->setComment("ROOT compression level of output file.");
    ROOT::Experimental::RNTupleWriteOptions ops;
    desc.addUntracked<unsigned long long>("approxZippedClusterSize", ops.GetApproxZippedClusterSize())
        ->setComment("Approximation of the target compressed cluster size");
    desc.addUntracked<unsigned long long>("maxUnzippedClusterSize", ops.GetMaxUnzippedClusterSize())
        ->setComment("Memory limit for committing a cluster. High compression leads to high IO buffer size.");

    desc.addUntracked<unsigned long long>("initialUnzippedPageSize", ops.GetInitialUnzippedPageSize())
        ->setComment("Initially, columns start with a page of this size (bytes).");
    desc.addUntracked<unsigned long long>("maxUnzippedPageSize", ops.GetMaxUnzippedPageSize())
        ->setComment("Pages can grow only to the given limit (bytes).");
    desc.addUntracked<unsigned long long>("pageBufferBudget", 0)
        ->setComment(
            "The maximum size that the sum of all page buffers used for writing into a persistent sink are allowed to "
            "use."
            " If set to zero, RNTuple will auto-adjust the budget based on the value of 'approxZippedClusterSize'."
            " If set manually, the size needs to be large enough to hold all initial page buffers.");

    desc.addUntracked<bool>("useBufferedWrite", ops.GetUseBufferedWrite())
        ->setComment(
            "Turn on use of buffered writing. This buffers compressed pages in memory, reorders them to keep pages of "
            "the same column adjacent, and coalesces the writes when committing a cluster.");
    desc.addUntracked<bool>("useDirectIO", ops.GetUseDirectIO())
        ->setComment(
            "Set use of direct IO. this introduces alignment requirements that may vary between filesystems and "
            "platforms");

    desc.addUntracked<bool>("dropPerEventDataProductProvenance", false)
        ->setComment(
            "do not store which data products were consumed to create a given data product for a given event.");

    desc.addUntracked<std::vector<std::string>>("noSplitSubFields", {})
        ->setComment(
            "fully qualified subfield names for fields which should not be split. A single value of 'all' means all "
            "possible subfields will be unsplit");
    desc.addUntracked<bool>("useStreamer", false)
        ->setComment("Use streamer storage for top level fields when storing data products");

    {
      ParameterSetDescription specialStreamer;
      specialStreamer.addUntracked<std::string>("product")->setComment(
          "Name of data product needing a special split setting. The name can contain wildcards '*' and '?'");
      specialStreamer.addUntracked<bool>("useStreamer", true)
          ->setComment("Explicitly set if should or should not use streamer (default is to use streamer)");
      desc.addVPSetUntracked("overrideDataProductStreamer", specialStreamer, {});
    }

    OutputModule::fillDescription(desc);

    //Make compatible with PoolOutputModule. Unnecessary ones will be marked obsolete once available
    std::string defaultString;
    desc.addUntracked<std::string>("logicalFileName", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<std::string>("catalog", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<int>("maxSize", 0x7f000000)
        ->setComment(
            "Maximum output file size, in kB.\n"
            "If over maximum, new output file will be started at next input file transition.");
    desc.addObsoleteUntracked<int>("basketSize");
    desc.addObsoleteUntracked<int>("eventAuxiliaryBasketSize");
    desc.addObsoleteUntracked<int>("eventAutoFlushCompressedSize");
    desc.addObsoleteUntracked<int>("splitLevel");
    desc.addObsoleteUntracked<std::string>("sortBaskets");
    desc.addObsoleteUntracked<int>("treeMaxVirtualSize");
    desc.addUntracked<bool>("fastCloning", true)
        ->setComment(
            "True:  Allow fast copying, if possible.\n"
            "False: Disable fast copying.");
    desc.addOptionalUntracked<bool>("mergeJob");
    desc.addObsoleteUntracked<bool>("compactEventAuxiliary");
    desc.addObsoleteUntracked<bool>("overrideInputFileSplitLevels");
    desc.addUntracked<bool>("writeStatusFile", false)
        ->setComment("Write a status file. Intended for use by workflow management.");
    desc.addUntracked<std::string>("dropMetaData", defaultString)
        ->setComment(
            "Determines handling of per product per event metadata.  Options are:\n"
            "'NONE':    Keep all of it.\n"
            "'DROPPED': Keep it for products produced in current process and all kept products. Drop it for dropped "
            "products produced in prior processes.\n"
            "'PRIOR':   Keep it for products produced in current process. Drop it for products produced in prior "
            "processes.\n"
            "'ALL':     Drop all of it.");
    desc.addUntracked<std::string>("overrideGUID", defaultString)
        ->setComment(
            "Allows to override the GUID of the file. Intended to be used only in Tier0 for re-creating files.\n"
            "The GUID needs to be of the proper format. If a new output file is started (see maxSize), the GUID of\n"
            "the first file only is overridden, i.e. the subsequent output files have different, generated GUID.");
    {
      ParameterSetDescription dataSet;
      dataSet.setAllowAnything();
      desc.addUntracked<ParameterSetDescription>("dataset", dataSet)
          ->setComment("PSet is only used by Data Operations and not by this module.");
    }
    { desc.addVPSetObsoleteUntracked("overrideBranchesSplitLevel"); }
    {
      /*
      ParameterSetDescription alias;
      alias.addUntracked<std::string>("branch")->setComment(
          "Name of branch which will get alias. The name can contain wildcards '*' and '?'");
      alias.addUntracked<std::string>("alias")->setComment("The alias to give to the TBranch");
      */
      desc.addVPSetObsoleteUntracked("branchAliases" /*, alias*/);
    }

    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::RNTupleOutputModule;
DEFINE_FWK_MODULE(RNTupleOutputModule);
