#ifndef EvFStepperService_H
#define EvFStepperService_H 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <pthread.h>

#include "EventFilter/Utilities/interface/ServiceWeb.h"

#include <string>
#include <vector>

namespace evf {

  class Stepper : public ServiceWeb 
    {
    public:
      Stepper(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~Stepper();
      
      void defaultWebPage(xgi::Input *in, xgi::Output *out);
      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preSource();
      void postSource();
      
      
      void preModule(const edm::ModuleDescription&);
      void postModule(const edm::ModuleDescription&);
      void publish(xdata::InfoSpace *) {}

    private:

      void wait_on_signal()
	{
	  pthread_mutex_lock(&mutex_);
	  pthread_cond_wait(&cond_,&mutex_);
	  pthread_mutex_unlock(&mutex_);
	}
      std::string epstate_;
      std::string modulelabel_;
      std::string modulename_;
      unsigned int rid_;
      unsigned int eid_;
      std::string original_referrer_;
      pthread_mutex_t mutex_;
      pthread_cond_t cond_;
      bool step_;
    };

}

#endif
