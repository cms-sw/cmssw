#ifndef RecoCaloTools_EcalChannelKiller_EcalChannelKiller_HH
#define RecoCaloTools_EcalChannelKiller_EcalChannelKiller_HH
 
/** \class EcalChannelKiller
  *
  *  $Date: 2007/05/03 12:33:39 $
  *  $Revision: 1.1 $
  */
 
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <string>
using namespace cms;
using namespace std;
//
// class decleration
//

class EcalChannelKiller : public edm::EDProducer {
   public:
      explicit EcalChannelKiller(const edm::ParameterSet&);
      ~EcalChannelKiller();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  string hitProducer_;
  string hitCollection_;
  string reducedHitCollection_;
  string DeadChannelFileName_;
  vector<EBDetId> ChannelsDeadID;
};


#endif
