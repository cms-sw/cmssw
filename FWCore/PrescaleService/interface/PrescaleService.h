#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h


#include "DataFormats/Provenance/interface/EventID.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h" 


#include "boost/thread/mutex.hpp"


#include <string>
#include <vector>
#include <map>


namespace edm {
  namespace service {

    class PrescaleService
    {
    public:
      //
      // construction/destruction
      //
      PrescaleService(const ParameterSet&,ActivityRegistry&) throw (cms::Exception);
      ~PrescaleService();
      

      //
      // member functions
      //

      void reconfigure(const ParameterSet &);

      unsigned int getPrescale(unsigned int lvl1Index,
			       const std::string&prescaledPath)throw(cms::Exception);
      unsigned int getPrescale(const std::string&prescaledPath)throw(cms::Exception);
      
      
      void postBeginJob() {;}
      void postEndJob() {;}
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&) {;}
      void postEventProcessing(const edm::Event&, const edm::EventSetup&) {;}
      void preModule(const ModuleDescription&) {;}
      void postModule(const ModuleDescription&) {;}
      

    private:
      //
      // private member functions
      //
      
      
      //
      // member data
      //
      typedef std::vector<std::string>                         VString_t;
      typedef std::map<std::string,std::vector<unsigned int> > PrescaleTable_t;

      boost::mutex    mutex_;
      unsigned int    nLvl1Index_;
      unsigned int    iLvl1IndexDefault_;
      VString_t       lvl1Labels_; 
      PrescaleTable_t prescaleTable_;
    };
  }
}

#endif
