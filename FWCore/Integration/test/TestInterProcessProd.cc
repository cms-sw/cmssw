#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <stdio.h>
#include <iostream>

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
}

class TestInterProcessProd : public edm::stream::EDProducer<edm::GlobalCache<testinter::Cache>> {

public:
  TestInterProcessProd( edm::ParameterSet const&, testinter::Cache const* );
  ~TestInterProcessProd();
  
  void produce(edm::Event& , edm::EventSetup const&);

  static std::unique_ptr<testinter::Cache> initializeGlobalCache(edm::ParameterSet const&);

  static void globalEndJob(testinter::Cache*);

private:

  int id_;
  edm::EDPutTokenT<double> token_;
  FILE* pipe_;

  managed_shared_memory managed_sm_;
  
  named_mutex named_mtx_;
  named_condition named_cndFromMain_;

  //named_mutex named_mtxToMain_;
  named_condition named_cndToMain_;
  
  long double* sm_sum_;
  bool* stop_;

  std::string unique_name(std::string iBase) {
    auto pid = getpid();
    iBase += std::to_string(pid);
    iBase +="_";
    iBase +=std::to_string(id_);
    
    return iBase;
  }

};

TestInterProcessProd::TestInterProcessProd( edm::ParameterSet const&, testinter::Cache const* cache ):
id_{ ++cache->id},
managed_sm_{ open_or_create, unique_name("testProd").c_str(), 1024},
named_mtx_{open_or_create, unique_name("mtx").c_str()},
named_cndFromMain_{open_or_create, unique_name("cndFromMain").c_str()},
//named_mtxToMain_{open_or_create, "mtxToMain"},
named_cndToMain_{open_or_create, unique_name("cndToMain").c_str()}

 {
  token_ = produces<double>();

  managed_sm_.destroy<long double>(unique_name("sum").c_str());
  managed_sm_.destroy<bool>(unique_name("stop").c_str());

  sm_sum_ = managed_sm_.construct<long double>(unique_name("sum").c_str())(0);
  stop_ = managed_sm_.construct<bool>(unique_name("stop").c_str())(false);

  //make sure output is flushed before popen does any writing
  fflush(stdout);
  fflush(stderr);
  pipe_ = popen(unique_name("cmsInterProcess testProd ").c_str(), "w");
  
  if(NULL == pipe_) {
    abort();
  }
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
  managed_sm_.destroy<long double>(unique_name("sum").c_str());
  managed_sm_.destroy<bool>(unique_name("stop").c_str());
}


void TestInterProcessProd::produce(edm::Event& iEvent, edm::EventSetup const&) {
  std::cout <<id_<<" taking from lock"<<std::endl;
  scoped_lock<named_mutex> lock(named_mtx_);
  {
    std::cout <<id_ <<" notifying"<<std::endl;
    named_cndFromMain_.notify_all();    
  }
  
  long double value;
  {
    std::cout <<id_<< " waiting"<<std::endl;
    named_cndToMain_.wait(lock);
    value = *sm_sum_;
  }
  std::cout <<id_ << " from shared memory "<<value<<std::endl;
  
  iEvent.emplace(token_, value);
}

DEFINE_FWK_MODULE(TestInterProcessProd);