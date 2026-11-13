#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TTree.h"

#include <string>
#include <vector>

namespace edmtest::pool {
  // Checks tree pointers and process-tree lookups at both input-file notifications,
  // including the all-null tree view for non-Pool input.
  class FileBlockNotificationTest : public edm::one::OutputModule<edm::WatchInputFiles> {
  public:
    explicit FileBlockNotificationTest(edm::ParameterSet const& pset)
        : edm::one::OutputModuleBase(pset),
          edm::one::OutputModule<edm::WatchInputFiles>(pset),
          nullTrees_(pset.getUntrackedParameter<bool>("nullTrees")),
          expectedProcesses_(pset.getUntrackedParameter<std::vector<std::string>>("expectedProcesses")),
          expectedFiles_(pset.getUntrackedParameter<unsigned int>("expectedFiles")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      edm::one::OutputModule<edm::WatchInputFiles>::fillDescription(desc);
      desc.addUntracked<bool>("nullTrees");
      desc.addUntracked<std::vector<std::string>>("expectedProcesses");
      desc.addUntracked<unsigned int>("expectedFiles");
      descriptions.addDefault(desc);
    }

  private:
    void write(edm::EventForOutput const&) override {}
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void writeRun(edm::RunForOutput const&) override {}

    void check(edm::FileBlock const& fb, char const* transition) const {
      auto fail = [transition](char const* what) {
        throw cms::Exception("TestFailure") << "FileBlock " << transition << ": " << what;
      };
      if (nullTrees_) {
        if (fb.tree() || fb.metaTree() || fb.lumiTree() || fb.lumiMetaTree() || fb.runTree() || fb.runMetaTree() ||
            !fb.processBlockTrees().empty() || !fb.processesWithProcessBlockTrees().empty() ||
            fb.processBlockTree("missing")) {
          fail("expected null trees and empty process lists");
        }
        return;
      }
      auto checkTree = [&](TTree const* tree, std::string const& name) {
        if (!tree || name != tree->GetName()) {
          fail("missing or incorrect tree");
        }
      };
      checkTree(fb.tree(), "Events");
      checkTree(fb.lumiTree(), "LuminosityBlocks");
      checkTree(fb.runTree(), "Runs");
      // The current Pool format has no per-transition metadata trees.
      if (fb.metaTree() || fb.lumiMetaTree() || fb.runMetaTree()) {
        fail("expected null per-transition metadata trees");
      }
      if (fb.processesWithProcessBlockTrees() != expectedProcesses_ ||
          fb.processBlockTrees().size() != expectedProcesses_.size()) {
        fail("incorrect process names or tree count");
      }
      for (unsigned int i = 0; i < expectedProcesses_.size(); ++i) {
        auto const* tree = fb.processBlockTrees()[i];
        checkTree(tree, "ProcessBlocks" + expectedProcesses_[i]);
        auto const* lookup = fb.processBlockTree(expectedProcesses_[i]);
        checkTree(lookup, "ProcessBlocks" + expectedProcesses_[i]);
        if (lookup != tree) {
          fail("process lookup does not match ordered tree");
        }
      }
      if (fb.processBlockTree("missing")) {
        fail("unexpected tree for absent process");
      }
    }

    void respondToOpenInputFile(edm::FileBlock const& fb) override {
      if (opens_ != closes_) {
        throw cms::Exception("TestFailure") << "input opened before previous close";
      }
      check(fb, "open");
      ++opens_;
    }
    void respondToCloseInputFile(edm::FileBlock const& fb) override {
      if (opens_ != closes_ + 1) {
        throw cms::Exception("TestFailure") << "input closed without corresponding open";
      }
      check(fb, "close");
      ++closes_;
    }
    void endJob() override {
      if (opens_ != expectedFiles_ || closes_ != expectedFiles_) {
        throw cms::Exception("TestFailure")
            << "expected " << expectedFiles_ << " opens and closes, got " << opens_ << " and " << closes_;
      }
    }

    bool nullTrees_;
    std::vector<std::string> expectedProcesses_;
    unsigned int expectedFiles_;
    unsigned int opens_ = 0;
    unsigned int closes_ = 0;
  };
}  // namespace edmtest::pool

DEFINE_FWK_MODULE(edmtest::pool::FileBlockNotificationTest);
