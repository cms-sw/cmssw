#ifndef RecoCaloTools_EcalChannelKiller_EcalChannelKiller_HH
#define RecoCaloTools_EcalChannelKiller_EcalChannelKiller_HH
 
/** \class EcalChannelKiller
  *
  *  $Date: 2010/08/06 20:24:49 $
  *  $Revision: 1.3 $
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
  std::string hitProducer_;
  std::string hitCollection_;
  std::string reducedHitCollection_;
  std::string DeadChannelFileName_;
  std::vector<EBDetId> ChannelsDeadID;
};


#endif
