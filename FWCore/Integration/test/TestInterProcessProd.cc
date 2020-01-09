#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include <stdio.h>
#include <iostream>

#include "TBufferFile.h"

#include "boost/interprocess/shared_memory_object.hpp"
#include "boost/interprocess/managed_shared_memory.hpp"
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost::interprocess;

namespace testinter {
  struct Cache {
    mutable int id = 0;
  };

  template <typename T>
  class Deserializer {
  public:
    Deserializer(boost::interprocess::managed_shared_memory& iSM, std::string iBase, int ID)
        : managed_shm_(&iSM),
          name_{unique_name(iBase, ID)},
          resizeName_{unique_name(iBase + "_resize", ID)},
          class_{TClass::GetClass(typeid(T))},
          bufferFile_(TBuffer::kRead) {
      managed_shm_->destroy<char>(name_.c_str());
      managed_shm_->destroy<char>(resizeName_.c_str());

      constexpr unsigned int bufferInitialSize = 5;
      buffer_.first = managed_shm_->construct<char>(name_.c_str())[bufferInitialSize](0);
      buffer_.second = bufferInitialSize;
      assert(buffer_.first);
      buffer_resized_ = managed_shm_->construct<bool>(resizeName_.c_str())(false);
      assert(buffer_resized_);

      bufferFile_.SetBuffer(buffer_.first, buffer_.second, kFALSE);
    }

    ~Deserializer() {
      managed_shm_->destroy<char>(name_.c_str());
      managed_shm_->destroy<bool>(resizeName_.c_str());
    }

    T deserialize() {
      T value;
      if (*buffer_resized_) {
        buffer_ = managed_shm_->find<char>(name_.c_str());
        bufferFile_.SetBuffer(buffer_.first, buffer_.second, kFALSE);
        *buffer_resized_ = false;
      }

      class_->ReadBuffer(bufferFile_, &value);
      bufferFile_.Reset();
      return value;
    }

  private:
    std::string unique_name(std::string const& iBase, int ID) {
      auto pid = getpid();
      return iBase + std::to_string(pid) + "_" + std::to_string(ID);
    }

    boost::interprocess::managed_shared_memory* const managed_shm_;
    const std::string name_;
    const std::string resizeName_;
    std::pair<char*, std::size_t> buffer_;
    bool* buffer_resized_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };
}  // namespace testinter

class TestInterProcessProd : public edm::stream::EDProducer<edm::GlobalCache<testinter::Cache>> {
public:
  TestInterProcessProd(edm::ParameterSet const&, testinter::Cache const*);
  ~TestInterProcessProd();

  void produce(edm::Event&, edm::EventSetup const&);

  static std::unique_ptr<testinter::Cache> initializeGlobalCache(edm::ParameterSet const&);

  static void globalEndJob(testinter::Cache*);

private:
  int id_;
  edm::EDPutTokenT<edmtest::IntProduct> token_;
  FILE* pipe_;

  managed_shared_memory managed_sm_;

  named_mutex named_mtx_;
  named_condition named_cndFromMain_;

  named_condition named_cndToMain_;

  testinter::Deserializer<edmtest::IntProduct> deserializer_;
  bool* stop_;

  std::string unique_name(std::string iBase) {
    auto pid = getpid();
    iBase += std::to_string(pid);
    iBase += "_";
    iBase += std::to_string(id_);

    return iBase;
  }
};

TestInterProcessProd::TestInterProcessProd(edm::ParameterSet const&, testinter::Cache const* cache)
    : id_{++cache->id},
      managed_sm_{open_or_create, unique_name("testProd").c_str(), 1024},
      named_mtx_{open_or_create, unique_name("mtx").c_str()},
      named_cndFromMain_{open_or_create, unique_name("cndFromMain").c_str()},
      named_cndToMain_{open_or_create, unique_name("cndToMain").c_str()},
      deserializer_{managed_sm_, "buffer", id_} {
  token_ = produces<edmtest::IntProduct>();

  managed_sm_.destroy<bool>(unique_name("stop").c_str());
  stop_ = managed_sm_.construct<bool>(unique_name("stop").c_str())(false);
  assert(stop_);

  //make sure output is flushed before popen does any writing
  fflush(stdout);
  fflush(stderr);

  scoped_lock<named_mutex> lock(named_mtx_);
  std::cout << id_ << " starting external process" << std::endl;
  pipe_ = popen(unique_name("cmsInterProcess testProd ").c_str(), "w");

  if (NULL == pipe_) {
    abort();
  }

  {
    auto v = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.gen = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(6), iterations=cms.uint32(10000000))
process.moduleToTest(process.gen)
)_";
    auto length = strlen(v);
    auto nlines = std::to_string(std::count(v, v + length, '\n'));
    auto result = fwrite(nlines.data(), sizeof(char), nlines.size(), pipe_);
    assert(result = nlines.size());
    result = fwrite(v, sizeof(char), strlen(v), pipe_);
    assert(result == strlen(v));
    fflush(pipe_);
  }
  std::cout << id_ << " waiting for external process" << std::endl;
  named_cndToMain_.wait(lock);
  std::cout << id_ << " done waiting for external process" << std::endl;
}

std::unique_ptr<testinter::Cache> TestInterProcessProd::initializeGlobalCache(edm::ParameterSet const&) {
  return std::make_unique<testinter::Cache>();
}

void TestInterProcessProd::globalEndJob(testinter::Cache*) {}

TestInterProcessProd::~TestInterProcessProd() {
  {
    scoped_lock<named_mutex> lock(named_mtx_);
    *stop_ = true;
    named_cndFromMain_.notify_all();
  }

  pclose(pipe_);
  managed_sm_.destroy<bool>(unique_name("stop").c_str());

  named_mutex::remove(unique_name("mtx").c_str());
  named_condition::remove(unique_name("cndFromMain").c_str());
  named_condition::remove(unique_name("cndToMain").c_str());
}

void TestInterProcessProd::produce(edm::Event& iEvent, edm::EventSetup const&) {
  std::cout << id_ << " taking from lock" << std::endl;
  scoped_lock<named_mutex> lock(named_mtx_);
  {
    std::cout << id_ << " notifying" << std::endl;
    named_cndFromMain_.notify_all();
  }

  std::cout << id_ << " waiting" << std::endl;
  named_cndToMain_.wait(lock);

  auto value = deserializer_.deserialize();
  std::cout << id_ << " from shared memory " << value.value << std::endl;

  iEvent.emplace(token_, value);
}

DEFINE_FWK_MODULE(TestInterProcessProd);