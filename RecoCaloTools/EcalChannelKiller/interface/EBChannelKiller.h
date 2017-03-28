#ifndef RecoCaloTools_EcalChannelKiller_EBChannelKiller_HH
#define RecoCaloTools_EcalChannelKiller_EBChannelKiller_HH
 
/** \class EBChannelKiller
  *
  *  $Date: 2012/11/21 13:08:40 $
  *  $Revision: 1.0 $
  *  \author Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
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

class EBChannelKiller : public edm::EDProducer {
   public:
      explicit EBChannelKiller(const edm::ParameterSet&);
      ~EBChannelKiller();

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
