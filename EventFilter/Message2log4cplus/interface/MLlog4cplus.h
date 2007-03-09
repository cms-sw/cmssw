// $Id: MLlog4cplus.h,v 1.1 2006/06/13 14:35:03 meschi Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

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

