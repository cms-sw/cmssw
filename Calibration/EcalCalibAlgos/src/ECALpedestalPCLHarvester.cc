
// system include files
#include <memory>

// user include files
#include "Calibration/EcalCalibAlgos/interface/ECALpedestalPCLHarvester.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include <iostream>
#include <string>

ECALpedestalPCLHarvester::ECALpedestalPCLHarvester(const edm::ParameterSet& ps):
    currentPedestals_(0),channelStatus_(0){

    chStatusToExclude_= StringToEnumValue<EcalChannelStatusCode::Code>(ps.getParameter<std::vector<std::string> >("ChannelStatusToExclude"));
    minEntries_=ps.getParameter<int>("MinEntries");
    checkAnomalies_  = ps.getParameter<bool>("checkAnomalies"); 
    nSigma_  = ps.getParameter<double>("nSigma");
    thresholdAnomalies_ = ps.getParameter<double>("thresholdAnomalies"); 
    dqmDir_         = ps.getParameter<std::string>("dqmDir");
}

void ECALpedestalPCLHarvester::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {


    // calculate pedestals and fill db record
    EcalPedestals pedestals;


    for (uint16_t i =0; i< EBDetId::kSizeForDenseIndexing; ++i) {
        std::string hname = dqmDir_+"/eb_" + std::to_string(i);
        MonitorElement* ch= igetter_.get(hname);
        double mean = ch->getMean();
        double rms  = ch->getRMS();

        DetId id = EBDetId::detIdFromDenseIndex(i);
        EcalPedestal ped;
        EcalPedestal oldped=* currentPedestals_->find(id.rawId());

        ped.mean_x12=mean;
        ped.rms_x12=rms;

        // if bad channel or low stat skip
        if(ch->getEntries()< minEntries_ || !checkStatusCode(id)){

            ped.mean_x12=oldped.mean_x12;
            ped.rms_x12=oldped.rms_x12;

        }

        ped.mean_x6=oldped.mean_x6;
        ped.rms_x6=oldped.rms_x6;
        ped.mean_x1=oldped.mean_x1;
        ped.rms_x1=oldped.rms_x1;

        pedestals.setValue(id.rawId(),ped);
    }


    for (uint16_t i =0; i< EEDetId::kSizeForDenseIndexing; ++i) {

        std::string hname = dqmDir_+"/ee_" + std::to_string(i);
     
        MonitorElement* ch= igetter_.get(hname);
        double mean = ch->getMean();
        double rms  = ch->getRMS();

        DetId id = EEDetId::detIdFromDenseIndex(i);
        EcalPedestal ped;
        EcalPedestal oldped= *currentPedestals_->find(id.rawId());

        ped.mean_x12=mean;
        ped.rms_x12=rms;

        // if bad channel or low stat skip
        if(ch->getEntries()< minEntries_ || !checkStatusCode(id)){
            ped.mean_x12=oldped.mean_x12;
            ped.rms_x12=oldped.rms_x12;
        }

        ped.mean_x6=oldped.mean_x6;
        ped.rms_x6=oldped.rms_x6;
        ped.mean_x1=oldped.mean_x1;
        ped.rms_x1=oldped.rms_x1;

        pedestals.setValue(id.rawId(),ped);
    }


    // check if there are large variations wrt exisiting pedstals

    if (checkAnomalies_){
        if (checkVariation(*currentPedestals_, pedestals)) {
            edm::LogError("Large Variations found wrt to old pedestals, no file created");
            return;
        }
    }

    // write out pedestal record
    edm::Service<cond::service::PoolDBOutputService> poolDbService;

    if( poolDbService.isAvailable() )
        poolDbService->writeOne( &pedestals, poolDbService->currentTime(),
                                 "EcalPedestalsRcd"  );
    else
        throw std::runtime_error("PoolDBService required.");
}




// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ECALpedestalPCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


void ECALpedestalPCLHarvester::endRun(edm::Run const& run, edm::EventSetup const & isetup){


    edm::ESHandle<EcalChannelStatus> chStatus;
    isetup.get<EcalChannelStatusRcd>().get(chStatus);
    channelStatus_=chStatus.product();

    edm::ESHandle<EcalPedestals> peds;
    isetup.get<EcalPedestalsRcd>().get(peds);
    currentPedestals_=peds.product();

}

bool ECALpedestalPCLHarvester::checkStatusCode(const DetId& id){

    EcalChannelStatusMap::const_iterator dbstatusPtr;
    dbstatusPtr = channelStatus_->getMap().find(id.rawId());
    EcalChannelStatusCode::Code  dbstatus = dbstatusPtr->getStatusCode();

    std::vector<int>::const_iterator res =
        std::find( chStatusToExclude_.begin(), chStatusToExclude_.end(), dbstatus );
    if ( res != chStatusToExclude_.end() ) return false;

    return true;
}


bool ECALpedestalPCLHarvester::checkVariation(const EcalPedestalsMap& oldPedestals, 
                                              const EcalPedestalsMap& newPedestals) {

    uint32_t nAnomaliesEB =0;
    uint32_t nAnomaliesEE =0;

    for (uint16_t i =0; i< EBDetId::kSizeForDenseIndexing; ++i) {
       
        DetId id = EBDetId::detIdFromDenseIndex(i);
        const EcalPedestal& newped=* newPedestals.find(id.rawId());   
        const EcalPedestal& oldped=* oldPedestals.find(id.rawId());

        if  (abs(newped.mean_x12 -oldped.mean_x12)  >  nSigma_ * oldped.rms_x12) nAnomaliesEB++;

    }

    for (uint16_t i =0; i< EEDetId::kSizeForDenseIndexing; ++i) {
       
        DetId id = EEDetId::detIdFromDenseIndex(i);
        const EcalPedestal& newped=* newPedestals.find(id.rawId());   
        const EcalPedestal& oldped=* oldPedestals.find(id.rawId());

        if  (abs(newped.mean_x12 -oldped.mean_x12)  >  nSigma_ * oldped.rms_x12) nAnomaliesEE++;

    }
    
    if (nAnomaliesEB > thresholdAnomalies_ *  EBDetId::kSizeForDenseIndexing || 
        nAnomaliesEE > thresholdAnomalies_ *  EEDetId::kSizeForDenseIndexing)  
        return false;

    return true;


}
