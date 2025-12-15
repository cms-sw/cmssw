#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWStorage/StorageFactory/interface/Storage.h"
#include "FWStorage/StorageFactory/interface/StorageProxyMaker.h"

#include <chrono>
#include <regex>
#include <thread>

namespace edm::storage {
  class StorageAddLatencyProxy : public Storage {
  public:
    struct LatencyConfig {
      unsigned int read;
      unsigned int readv;
      unsigned int write;
      unsigned int writev;
    };

    StorageAddLatencyProxy(LatencyConfig latency, std::unique_ptr<Storage> storage)
        : latency_(latency), baseStorage_(std::move(storage)) {}

    IOSize read(void* into, IOSize n) override {
      auto const result = baseStorage_->read(into, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.read));
      return result;
    }

    IOSize read(void* into, IOSize n, IOOffset pos) override {
      auto const result = baseStorage_->read(into, n, pos);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.read));
      return result;
    }

    IOSize readv(IOBuffer* into, IOSize n) override {
      auto const result = baseStorage_->readv(into, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.readv));
      return result;
    }

    IOSize readv(IOPosBuffer* into, IOSize n) override {
      auto const result = baseStorage_->readv(into, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.readv));
      return result;
    }

    IOSize write(const void* from, IOSize n) override {
      auto const result = baseStorage_->write(from, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.write));
      return result;
    }

    IOSize write(const void* from, IOSize n, IOOffset pos) override {
      auto const result = baseStorage_->write(from, n, pos);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.write));
      return result;
    }

    IOSize writev(const IOBuffer* from, IOSize n) override {
      auto const result = baseStorage_->writev(from, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.writev));
      return result;
    }

    IOSize writev(const IOPosBuffer* from, IOSize n) override {
      auto const result = baseStorage_->writev(from, n);
      std::this_thread::sleep_for(std::chrono::microseconds(latency_.writev));
      return result;
    }

    IOOffset position(IOOffset offset, Relative whence) override { return baseStorage_->position(offset, whence); }

    void resize(IOOffset size) override { return baseStorage_->resize(size); }

    void flush() override { return baseStorage_->flush(); }

    void close() override { return baseStorage_->close(); }

    bool prefetch(const IOPosBuffer* what, IOSize n) override { return baseStorage_->prefetch(what, n); }

  private:
    LatencyConfig latency_;
    std::unique_ptr<Storage> baseStorage_;
  };

  class StorageAddLatencyProxyMaker : public StorageProxyMaker {
  public:
    StorageAddLatencyProxyMaker(edm::ParameterSet const& pset)
        : latency_{.read = pset.getUntrackedParameter<unsigned int>("read"),
                   .readv = pset.getUntrackedParameter<unsigned int>("readv"),
                   .write = pset.getUntrackedParameter<unsigned int>("write"),
                   .writev = pset.getUntrackedParameter<unsigned int>("writev")},
          exclude_(vector_transform(pset.getUntrackedParameter<std::vector<std::string>>("exclude"),
                                    [](std::string const& pattern) { return std::regex(pattern); })) {}

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.addUntracked<unsigned int>("read", 0)->setComment(
          "Add this many microseconds of latency to singular reads");
      iDesc.addUntracked<unsigned int>("readv", 0)->setComment("Add this many microseconds of latency to vector reads");
      iDesc.addUntracked<unsigned int>("write", 0)
          ->setComment("Add this many microseconds of latency to singular writes");
      iDesc.addUntracked<unsigned int>("writev", 0)
          ->setComment("Add this many microseconds of latency to vector writes");
      iDesc.addUntracked<std::vector<std::string>>("exclude", {})
          ->setComment(
              "Latency is not added to the operations on the files whose URLs have a part that matches to any of the "
              "regexes in this parameter");
    }

    std::unique_ptr<Storage> wrap(std::string const& url, std::unique_ptr<Storage> storage) const override {
      for (auto const& pattern : exclude_) {
        if (std::regex_search(url, pattern)) {
          return storage;
        }
      }
      return std::make_unique<StorageAddLatencyProxy>(latency_, std::move(storage));
    }

  private:
    StorageAddLatencyProxy::LatencyConfig const latency_;
    std::vector<std::regex> const exclude_;
  };
}  // namespace edm::storage

#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "FWStorage/StorageFactory/interface/StorageProxyMakerFactory.h"
DEFINE_EDM_VALIDATED_PLUGIN(edm::storage::StorageProxyMakerFactory,
                            edm::storage::StorageAddLatencyProxyMaker,
                            "StorageAddLatencyProxy");
