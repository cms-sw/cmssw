#ifndef RecoLocalCalo_EcalDeadChannelRecoveryProducers_EBDeadChannelRecoveryProducers_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryProducers_EBDeadChannelRecoveryProducers_HH
 
/** \class EBDeadChannelRecoveryProducers
  *
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

class EBDeadChannelRecoveryProducers : public edm::EDProducer {
   public:
      explicit EBDeadChannelRecoveryProducers(const edm::ParameterSet&);
      ~EBDeadChannelRecoveryProducers();

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
