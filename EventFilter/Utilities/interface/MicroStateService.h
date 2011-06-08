#ifndef EvFMicroStateService_H
#define EvFMicroStateService_H 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "boost/thread/thread.hpp"

#include <string>
#include <vector>

namespace evf{



    class MicroStateService
    {
    public:
      MicroStateService(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~MicroStateService();
      
      std::string getMicroState1();
      
      std::string const &getMicroState2();

      void postBeginJob();     

      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preSource();
      void postSource();
      
      
      void preModule(const edm::ModuleDescription&);
      void postModule(const edm::ModuleDescription&);

      void setMicroState(std::string const *);
      
    private:

      std::string microstate1_;
      const std::string init;
      const std::string done;
      const std::string input;
      const std::string fwkovh;
      const std::string *microstate2_;
      boost::mutex lock_;
    };

}

#endif
