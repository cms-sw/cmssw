#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/SaveConfiguration.h"

#include <string>
#include <vector>
#include <map>


namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;

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

      unsigned int getPrescale(std::string const& prescaledPath) const;
      unsigned int getPrescale(unsigned int lvl1Index,
                               std::string const& prescaledPath) const;

      typedef std::vector<std::string>                          VString_t;
      typedef std::map<std::string, std::vector<unsigned int> > PrescaleTable_t;
      unsigned int getLvl1IndexDefault() const {return lvl1Default_;}
      const VString_t& getLvl1Labels()   const {return lvl1Labels_;}
      const PrescaleTable_t& getPrescaleTable() const {return prescaleTable_;}

      static unsigned int findDefaultIndex(std::string const & label, std::vector<std::string> const & labels);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
      //
      // private member functions
      //
      void postBeginJob();
      
      //
      // member data
      //
      const bool            forceDefault_;
      const VString_t       lvl1Labels_; 
      const unsigned int    lvl1Default_;
      const std::vector<ParameterSet> vpsetPrescales_;
      PrescaleTable_t prescaleTable_;
    };
  }
}

#endif
