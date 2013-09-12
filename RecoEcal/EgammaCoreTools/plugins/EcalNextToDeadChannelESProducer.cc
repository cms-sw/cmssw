

#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannel.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannelRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

/**
   ESProducer to fill the EcalNextToDeadChannel record
   starting from EcalChannelStatus information
   

   \author Stefano Argiro
   \date 18 May 2011
*/


class EcalNextToDeadChannelESProducer : public edm::ESProducer {
  
public:
  EcalNextToDeadChannelESProducer(const edm::ParameterSet& iConfig);
  
  typedef boost::shared_ptr<EcalNextToDeadChannel> ReturnType;
  
  ReturnType produce(const EcalNextToDeadChannelRcd& iRecord);
  


private:

  void findNextToDeadChannelsCallback(const EcalChannelStatusRcd& chs);
  
  // threshold above which a channel will be considered "dead"
  int statusThreshold_;

  ReturnType returnRcd_;
};

EcalNextToDeadChannelESProducer::EcalNextToDeadChannelESProducer(const edm::ParameterSet& iConfig){
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, 
		  dependsOn (&EcalNextToDeadChannelESProducer::findNextToDeadChannelsCallback));

  statusThreshold_= iConfig.getParameter<int>("channelStatusThresholdForDead");

  returnRcd_=ReturnType(new EcalNextToDeadChannel);
}



EcalNextToDeadChannelESProducer::ReturnType
EcalNextToDeadChannelESProducer::produce(const EcalNextToDeadChannelRcd& iRecord){
  
  return returnRcd_ ;
}


void 
EcalNextToDeadChannelESProducer::
findNextToDeadChannelsCallback(const EcalChannelStatusRcd& chs){

  

  EcalNextToDeadChannel* rcd= new EcalNextToDeadChannel;

  // Find channels next to dead ones and fill corresponding record

  edm::ESHandle <EcalChannelStatus> h;
  chs.get (h);
  
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
	
	EBDetId detid(ieta,iphi);
	
	if (EcalTools::isNextToDeadFromNeighbours(detid,
						  *h,
						  statusThreshold_)){
	  
	  
	  rcd->setValue(detid,1);
	};
      }
    } // for phi
  } // for eta
  
  // endcap

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
  
      if (EEDetId::validDetId(iX,iY,1)) {
	EEDetId detid(iX,iY,1);

	if (EcalTools::isNextToDeadFromNeighbours(detid,
						  *h,
						  statusThreshold_)){
	  rcd->setValue(detid,1);
	};
	
      }

      if (EEDetId::validDetId(iX,iY,-1)) {
	EEDetId detid(iX,iY,-1);

	if (EcalTools::isNextToDeadFromNeighbours(detid,
						  *h,
						  statusThreshold_)){
     
	  rcd->setValue(detid,1);
	};
      }
    } // for iy
  } // for ix
  
  returnRcd_.reset(rcd);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalNextToDeadChannelESProducer);








// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd .. ; scram b"
// End:
