#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h" 


#include <string>
#include <vector>
#include <map>


namespace edm {
  class ActivityRegistry;
  class Event;
  class EventID;
  class EventSetup;
  class Timestamp;
  namespace service {

    class PrescaleService
    {
    public:
      //
      // construction/destruction
      //
      PrescaleService(ParameterSet const&, ActivityRegistry&);
      ~PrescaleService();
      

      //
      // member functions
      //

      void reconfigure(ParameterSet const& ps);

      unsigned int getPrescale(unsigned int lvl1Index,
			       std::string const& prescaledPath);
      unsigned int getPrescale(std::string const& prescaledPath);

      void setIndex(unsigned int lvl1Index){iLvl1IndexDefault_ = lvl1Index;}      
      void postBeginJob();
      void postEndJob() {}
      void preEventProcessing(EventID const&, Timestamp const&) {}
      void postEventProcessing(Event const&, EventSetup const&) {}
      void preModule(ModuleDescription const&) {}
      void postModule(ModuleDescription const&) {}
      

    private:
      //
      // private member functions
      //
      void configure();
      
      //
      // member data
      //
      typedef std::vector<std::string>                         VString_t;
      typedef std::map<std::string, std::vector<unsigned int> > PrescaleTable_t;

      bool	      configured_;
      VString_t       lvl1Labels_; 
      unsigned int    nLvl1Index_;
      unsigned int    iLvl1IndexDefault_;
      std::vector<ParameterSet> vpsetPrescales_;
      PrescaleTable_t prescaleTable_;
    };
  }
}

#endif
