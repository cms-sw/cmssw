
#include "EventFilter/StorageManager/interface/EPRunner.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "IOPool/Streamer/interface/HLTInfo.h"

#include "boost/bind.hpp"
#include "boost/thread/thread.hpp"

#include <string>
#include <exception>
#include <iostream>

using namespace std;
using namespace edm;

namespace stor
{
  EPRunner::EPRunner(const std::string& config_string,
		     std::auto_ptr<HLTInfo> info):
    info_(info.get()),
    tok_(edm::ServiceRegistry::createContaining(info)),
	ah_(),
    ep_(config_string,tok_,edm::serviceregistry::kOverlapIsError)
  {
  	FDEBUG(4) << "EPRunner ctor body" << endl;
  }

  EPRunner::~EPRunner()
  {
  }

  const edm::ProductRegistry& EPRunner::getRegistry()
  {
    return ep_.getInputSource().productRegistry();
  }

  void EPRunner::start()
  {
    me_.reset( new boost::thread(boost::bind(EPRunner::run,this)));
  }

  void EPRunner::join()
  {
    me_->join();
  }

  void EPRunner::run(EPRunner* t)
  {
    t->dowork();
  }

  void EPRunner::dowork()
  {
    try
      {
	  {
	  	boost::mutex::scoped_lock sl(info_->getExtraLock());
	    ep_.beginJob();
	  }
	ep_.run();
	  {
	  	boost::mutex::scoped_lock sl(info_->getExtraLock());
	    ep_.endJob();
      }
      }
    catch (seal::Error& e)
      {
	std::cerr << "Exception caught EventProcessor" << "\n"
		  << e.explainSelf()
		  << std::endl;
      }
    catch (std::exception& e)
      {
	std::cerr << "Standard library exception caught EventProcessor" << "\n"
		  << e.what()
		  << std::endl;
      }
    catch (...)
      {
	std::cerr << "Unknown exception caught EventProcessor"
		  << std::endl;
      }

  }

}
