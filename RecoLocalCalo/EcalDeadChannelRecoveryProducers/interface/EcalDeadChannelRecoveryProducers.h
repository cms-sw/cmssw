#ifndef RecoLocalCalo_EcalDeadChannelRecoveryProducers_EcalDeadChannelRecoveryProducers_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryProducers_EcalDeadChannelRecoveryProducers_HH
 
/** \class EcalDeadChannelRecoveryProducers
  *
  *  $Date: 2010/08/06 20:24:55 $
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

class EcalDeadChannelRecoveryProducers : public edm::EDProducer {
   public:
      explicit EcalDeadChannelRecoveryProducers(const edm::ParameterSet&);
      ~EcalDeadChannelRecoveryProducers();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  double Sum8GeVThreshold_;
  std::string hitProducer_;
  std::string hitCollection_;
  std::string reducedHitCollection_;
  std::string DeadChannelFileName_;
  std::vector<EBDetId> ChannelsDeadID;
  bool CorrectDeadCells_;
  std::string CorrectionMethod_;

};


#endif
