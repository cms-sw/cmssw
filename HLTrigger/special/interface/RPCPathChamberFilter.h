// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//
// class declaration
//

class RPCPathChambFilter : public edm::EDFilter {
   public:
      explicit RPCPathChambFilter(const edm::ParameterSet&);
      ~RPCPathChambFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      double rangestrips;
      edm::InputTag rpcRecHitsLabel;
      edm::InputTag rpcDTPointsLabel;
      edm::InputTag rpcCSCPointsLabel;

      
      // ----------member data ---------------------------
};

