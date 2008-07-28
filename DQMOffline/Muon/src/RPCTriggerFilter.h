// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RPCTriggerFilter : public edm::EDFilter {
   public:
      explicit RPCTriggerFilter(const edm::ParameterSet&);
      ~RPCTriggerFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      bool  enableFilter_;
      
      bool  rpcBarOnly_;
      bool  rpcFwdOnly_ ;
      bool  rpcOnly_;                                                    
      bool  dtOnly_;                                                    
      bool  cscOnly_ ;
      bool  rpcAndDt_;
      bool  rpcAndCsc_;
      bool  dtAndCsc_;
      bool  rpcAndDtAndCsc_;

      edm::InputTag inputTag_;

      int event_, goodEvent_;
};
