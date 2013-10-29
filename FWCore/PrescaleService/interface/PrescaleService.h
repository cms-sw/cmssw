#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/SaveConfiguration.h"


#include <string>
#include <vector>
#include <map>


namespace edm {
  class ActivityRegistry;
  class Event;
  class EventID;
  class EventSetup;
  class Timestamp;
  class ConfigurationDescriptions;
  class ModuleDescription;

  namespace service {

    class PrescaleService : public edm::serviceregistry::SaveConfiguration
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
      
      typedef std::vector<std::string>                          VString_t;
      typedef std::map<std::string, std::vector<unsigned int> > PrescaleTable_t;
      unsigned int getLvl1IndexDefault() const {return iLvl1IndexDefault_;}
      const VString_t& getLvl1Labels() const {return lvl1Labels_;}
      const PrescaleTable_t& getPrescaleTable() const {return prescaleTable_;}

      static unsigned int findDefaultIndex(std::string const & label, std::vector<std::string> const & labels);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
      //
      // private member functions
      //
      void postBeginJob();

      void configure();
      
      //
      // member data
      //

      bool            configured_;
      bool            forceDefault_;
      VString_t       lvl1Labels_; 
      unsigned int    nLvl1Index_;
      unsigned int    iLvl1IndexDefault_;
      std::vector<ParameterSet> vpsetPrescales_;
      PrescaleTable_t prescaleTable_;
    };
  }
}

#endif
