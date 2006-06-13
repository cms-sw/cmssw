// $Id: MLlog4cplus.cc,v 1.1 2006/05/04 14:52:46 meschi Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

#include <iostream>

namespace edm{
  class ELlog4cplus;
}


namespace ML {
  
  // class ELlog4cplus exists

  class MLlog4cplus
  {
  public:
    MLlog4cplus(const edm::ParameterSet&,edm::ActivityRegistry&);
    ~MLlog4cplus();
      
    void postBeginJob();
    void postEndJob();
      
    void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
    void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
    void preModule(const edm::ModuleDescription&);
    void postModule(const edm::ModuleDescription&);
    void setAppl(xdaq::Application *app);
  private:
    edm::EventID curr_event_;
    edm::ELlog4cplus * dest_p;
  };
}

