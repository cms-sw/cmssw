
/*----------------------------------------------------------------------
$Id: Worker.cc,v 1.2 2005/07/14 22:50:53 wmtan Exp $
----------------------------------------------------------------------*/

#include <iostream>
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm
{
  Worker::~Worker()
  {
  }
  bool Worker::doWork(EventPrincipal& ep, EventSetup const& es) {
    using namespace std;
    bool rc = false;
    try {
      rc = implDoWork(ep,es);
    }
    catch(cms::Exception& e) {
      // should event id be included?
      e << "A cms::Exception is going through "<< workerType()<<":\n"
      << description();
      
      switch(actions_->find(e.rootCause())) {
        case actions::IgnoreCompletely: {
          rc=true;
          cerr << workerType()<<" ignored an exception for event " << ep.id()
            << "\nmessage from exception:\n" << e.what()
            << endl;
          break;
        }
        case actions::FailModule: {
          cerr << workerType()<< "failed due to exception for event " << ep.id()
          << "\nmessage from exception:\n" << e.what()
          << endl;
          break;
        }
        default: throw;
      }
    }
    catch(seal::Error& e) {
      cerr << "A seal::Error is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::exception& e) {
      cerr << "An std::exception is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::string& s) {
      throw cms::Exception("BadExceptionType","std::string") 
      << "string = " << s << "\n"
      << description() ;
    }
    catch(const char* c) {
      throw cms::Exception("BadExceptionType","const char*") 
      << "cstring = " << c << "\n"
      << description() ;
    }
    catch(...) {
      cerr << "An unknown Exception occured in\n" << description();
      throw;
    }
    return rc;
  }

  void Worker::beginJob(EventSetup const& es) {
    using namespace std;
    try {
      implBeginJob(es);
    } catch(cms::Exception& e) {
      // should event id be included?
      e << "A cms::Exception is going through "<< workerType()<<":\n"
      << description();
      throw;
    }
    catch(seal::Error& e) {
      cerr << "A seal::Error is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::exception& e) {
      cerr << "An std::exception is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::string& s) {
      throw cms::Exception("BadExceptionType","std::string") 
      << "string = " << s << "\n"
      << description() ;
    }
    catch(const char* c) {
      throw cms::Exception("BadExceptionType","const char*") 
      << "cstring = " << c << "\n"
      << description() ;
    }
    catch(...) {
      cerr << "An unknown Exception occured in\n" << description();
      throw;
    }
  }
  
  void Worker::endJob() {
    using namespace std;
    try {
      implEndJob();
    } catch(cms::Exception& e) {
      // should event id be included?
      e << "A cms::Exception is going through "<< workerType()<<":\n"
      << description();
      throw;
    }
    catch(seal::Error& e) {
      cerr << "A seal::Error is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::exception& e) {
      cerr << "An std::exception is going through "<< workerType()<<":\n"
      << description()
      << endl;
      throw;
    }
    catch(std::string& s) {
      throw cms::Exception("BadExceptionType","std::string") 
      << "string = " << s << "\n"
      << description() ;
    }
    catch(const char* c) {
      throw cms::Exception("BadExceptionType","const char*") 
      << "cstring = " << c << "\n"
      << description() ;
    }
    catch(...) {
      cerr << "An unknown Exception occured in\n" << description();
      throw;
    }
  }
  
}
