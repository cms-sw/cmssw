#include "boost/program_options.hpp"
#include "boost/interprocess/shared_memory_object.hpp"

#include "boost/interprocess/managed_shared_memory.hpp"
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "TClass.h"
#include "TBufferFile.h"

#include <string>
#include <iostream>

#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

//NOTE: Can use TestProcessor as the harness for the worker

namespace {
  std::string unique_name(std::string iBase, std::string_view ID) {
    iBase.append(ID);
    return iBase;
  }

  template <typename T>
  class Serializer {
  public:
    Serializer(boost::interprocess::managed_shared_memory& iSM,
               std::string iSMName,
               std::string const& iBase,
               std::string_view ID)
        : managed_shm_(&iSM),
          smName_{std::move(iSMName)},
          name_{unique_name(iBase, ID)},
          class_{TClass::GetClass(typeid(T))},
          bufferFile_{TBuffer::kWrite} {
      buffer_ = managed_shm_->find<char>(name_.c_str());
      assert(buffer_.first);
      std::pair<bool*, std::size_t> sm_buffer_resized =
          managed_shm_->find<bool>(unique_name(iBase + "_resize", ID).c_str());
      buffer_resized_ = sm_buffer_resized.first;
      assert(buffer_resized_);
    }

    void serialize(T& iValue) {
      bufferFile_.Reset();
      class_->WriteBuffer(bufferFile_, &iValue);

      if (static_cast<unsigned long>(bufferFile_.Length()) > buffer_.second) {
        managed_shm_->destroy<char>(name_.c_str());
        auto diff = bufferFile_.Length() - buffer_.second;
        auto success = managed_shm_->grow(smName_.c_str(), diff);
        assert(success);

        buffer_.first = managed_shm_->construct<char>(name_.c_str())[bufferFile_.Length()](0);
        buffer_.second = bufferFile_.Length();
        *buffer_resized_ = true;
      }
      std::copy(bufferFile_.Buffer(), bufferFile_.Buffer() + bufferFile_.Length(), buffer_.first);
    }

  private:
    boost::interprocess::managed_shared_memory* const managed_shm_;
    std::string smName_;
    std::string name_;
    std::pair<char*, std::size_t> buffer_;
    bool* buffer_resized_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };
}  // namespace

class Harness {
public:
  Harness(std::string const& iConfig) : tester_(edm::test::TestProcessor::Config{iConfig}) {}

  edmtest::IntProduct getNextValue() {
    auto event = tester_.test();
    return *event.get<edmtest::IntProduct>();
  }

private:
  edm::test::TestProcessor tester_;
};

int main(int argc, char* argv[]) {
  std::string descString(argv[0]);
  descString += " [--";
  descString += kMemoryNameOpt;
  descString += "] memory_name";
  boost::program_options::options_description desc(descString);

  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kMemoryNameCommandOpt, boost::program_options::value<std::string>(), "memory name")(
      kUniqueIDCommandOpt, boost::program_options::value<std::string>(), "unique id");

  boost::program_options::positional_options_description p;
  p.add(kMemoryNameOpt, 1);
  p.add(kUniqueIDOpt, 2);

  boost::program_options::options_description all_options("All Options");
  all_options.add(desc);

  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc, argv).options(all_options).positional(p).run(), vm);
    notify(vm);
  } catch (boost::program_options::error const& iException) {
    std::cout << argv[0] << ": Error while trying to process command line arguments:\n"
              << iException.what() << "\nFor usage and an options list, please do 'cmsRun --help'.";
    return 1;
  }

  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count(kMemoryNameOpt)) {
    std::cout << " no argument given" << std::endl;
    return 1;
  }

  if (!vm.count(kUniqueIDOpt)) {
    std::cout << " no second argument given" << std::endl;
    return 1;
  }

  std::string const memoryName(vm[kMemoryNameOpt].as<std::string>());
  std::string const uniqueID(vm[kUniqueIDOpt].as<std::string>());
  {
    using namespace boost::interprocess;
    auto memoryNameUnique = unique_name(memoryName, uniqueID);
    managed_shared_memory managed_shm{open_only, memoryNameUnique.c_str()};

    named_mutex named_mtx{open_or_create, unique_name("mtx", uniqueID).c_str()};
    named_condition named_cndFromMain{open_or_create, unique_name("cndFromMain", uniqueID).c_str()};
    std::pair<bool*, std::size_t> sm_stop = managed_shm.find<bool>(unique_name("stop", uniqueID).c_str());

    named_condition named_cndToMain{open_or_create, unique_name("cndToMain", uniqueID).c_str()};

    int counter = 0;

    scoped_lock<named_mutex> lock(named_mtx);
    std::cerr << uniqueID << " process: initializing " << std::endl;
    int nlines;
    std::cin >> nlines;

    std::string configuration;
    for (int i = 0; i < nlines; ++i) {
      std::string c;
      std::getline(std::cin, c);
      std::cerr << c << "\n";
      configuration += c + "\n";
    }

    Harness harness(configuration);

    Serializer<edmtest::IntProduct> serializer(managed_shm, memoryNameUnique, "buffer", uniqueID);

    std::cerr << uniqueID << " process: done initializing" << std::endl;
    named_cndToMain.notify_all();
    while (true) {
      {
        ++counter;
        std::cerr << uniqueID << " process: waiting " << counter << std::endl;
        named_cndFromMain.wait(lock);
        if (*sm_stop.first) {
          break;
        }
      }

      std::cerr << uniqueID << " process: integrating " << counter << std::endl;
      auto value = harness.getNextValue();

      std::cerr << uniqueID << " process: integrated " << counter << std::endl;

      {
        serializer.serialize(value);
        std::cerr << uniqueID << " process: notifying " << counter << std::endl;
        named_cndToMain.notify_all();
      }
      std::cerr << uniqueID << " process: " << value.value << " "
                << " " << counter << std::endl;
    }
  }
  return 0;
}