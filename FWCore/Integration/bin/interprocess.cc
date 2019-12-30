#include "boost/program_options.hpp"
#include "boost/interprocess/shared_memory_object.hpp"

#include "boost/interprocess/managed_shared_memory.hpp"
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>


#include <string>
#include <iostream>

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

namespace {
  std::string unique_name(std::string iBase, std::string_view ID) {
    iBase.append(ID);
    return iBase;
  }
}

int main(int argc, char* argv[]) {
  
  std::string descString(argv[0]);
  descString += " [--" ;
  descString += kMemoryNameOpt ;
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
    std::cout
        << argv[0] <<": Error while trying to process command line arguments:\n"
        << iException.what() << "\nFor usage and an options list, please do 'cmsRun --help'.";
    return 1;
  }
  
  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }
  
  if (!vm.count(kMemoryNameOpt)) {
    std::cout <<" no argument given"<<std::endl;
    return 1;
  }

  if (!vm.count(kUniqueIDOpt)) {
    std::cout <<" no second argument given"<<std::endl;
    return 1;
  }
  
  std::string const memoryName(vm[kMemoryNameOpt].as<std::string>());
  std::string const uniqueID(vm[kUniqueIDOpt].as<std::string>());
  {
    using namespace boost::interprocess;
    managed_shared_memory managed_shm{open_only, unique_name(memoryName, uniqueID).c_str()};
    std::pair<long double *, std::size_t> sm_sum = managed_shm.find<long double>(unique_name("sum", uniqueID).c_str());
    
    const auto s_pi{std::acos(-1)};
  
    constexpr unsigned int iterations = 100000000;
    
    named_mutex named_mtx{open_or_create, unique_name("mtx", uniqueID).c_str()};
    named_condition named_cndFromMain{open_or_create, unique_name("cndFromMain", uniqueID).c_str()};
    std::pair<bool *, std::size_t> sm_stop = managed_shm.find<bool>(unique_name("stop", uniqueID).c_str());
    
    //named_mutex named_mtxToMain{open_or_create, "mtxToMain"};
    named_condition named_cndToMain{open_or_create, unique_name("cndToMain", uniqueID).c_str()};
  
    int counter = 0;
    scoped_lock<named_mutex> lock(named_mtx);
    while(true) {
      {
        ++counter;
        std::cerr <<uniqueID<<" process: waiting "<<counter<<std::endl;
        named_cndFromMain.wait(lock);
        if(*sm_stop.first) { break;}
      }
  
      std::cerr <<uniqueID <<" process: integrating "<<counter<<std::endl;
      long double sum = 0.;
      const long double stepSize = s_pi / iterations;
      for (unsigned int i = 0; i < iterations; ++i) {
        sum += stepSize * cos(i * stepSize);
      }
      std::cerr <<uniqueID<<" process: integrated "<<counter<<std::endl;

      {
        *sm_sum.first = sum;
        std::cerr <<uniqueID<<" process: notifying "<<counter<<std::endl;
        named_cndToMain.notify_all();
      }
      std::cerr <<uniqueID<<" process: "<<sum<<" "<<counter<<std::endl;
    }
  }  
  return 0;
  
}