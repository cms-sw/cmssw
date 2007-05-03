#ifndef RecoLocalCalo_EcalDeadChannelRecoveryProducers_EcalDeadChannelRecoveryProducers_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryProducers_EcalDeadChannelRecoveryProducers_HH
 
/** \class EcalDeadChannelRecoveryProducers
  *
  *  $Date: 2006/03/10 08:38:19 $
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

class EcalDeadChannelRecoveryProducers : public edm::EDProducer {
   public:
      explicit EcalDeadChannelRecoveryProducers(const edm::ParameterSet&);
      ~EcalDeadChannelRecoveryProducers();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  double Sum8GeVThreshold_;
  string hitProducer_;
  string hitCollection_;
  string reducedHitCollection_;
  string DeadChannelFileName_;
  vector<EBDetId> ChannelsDeadID;
  bool CorrectDeadCells_;
  string CorrectionMethod_;

};


#endif
