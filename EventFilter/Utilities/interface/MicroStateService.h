#ifndef EvFMicroStateService_H
#define EvFMicroStateService_H 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

#include <string>
#include <vector>

namespace evf {

    class MicroStateService
    {
    public:
      MicroStateService(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~MicroStateService();
      
      std::string getMicroState1() const { return microstate1_;}
      std::string getMicroState2() const { return microstate2_;}

      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preSource();
      void postSource();
      
      
      void preModule(const edm::ModuleDescription&);
      void postModule(const edm::ModuleDescription&);
      
    private:

      std::string microstate1_;
      std::string microstate2_;
    };

}

#endif
