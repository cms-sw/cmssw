
// system include files
#include <memory>

// user include files
#include "Calibration/EcalCalibAlgos/interface/ECALpedestalPCLHarvester.h"
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
    currentPedestals_(nullptr),channelStatus_(nullptr){

    chStatusToExclude_= StringToEnumValue<EcalChannelStatusCode::Code>(ps.getParameter<std::vector<std::string> >("ChannelStatusToExclude"));
    minEntries_=ps.getParameter<int>("MinEntries");
    checkAnomalies_  = ps.getParameter<bool>("checkAnomalies"); 
    nSigma_  = ps.getParameter<double>("nSigma");
    thresholdAnomalies_ = ps.getParameter<double>("thresholdAnomalies"); 
    dqmDir_         = ps.getParameter<std::string>("dqmDir");
    labelG6G1_      = ps.getParameter<std::string>("labelG6G1");
    threshDiffEB_   = ps.getParameter<double>("threshDiffEB");
    threshDiffEE_   = ps.getParameter<double>("threshDiffEE");
}

void ECALpedestalPCLHarvester::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {


    // calculate pedestals and fill db record
    EcalPedestals pedestals;


    for (uint16_t i =0; i< EBDetId::kSizeForDenseIndexing; ++i) {
        std::string hname = dqmDir_+"/EB/"+std::to_string(int(i/100))+"/eb_" + std::to_string(i);
        MonitorElement* ch= igetter_.get(hname);
        double mean = ch->getMean();
        double rms  = ch->getRMS();
        entriesEB_[i] = ch->getEntries();

        DetId id = EBDetId::detIdFromDenseIndex(i);
        EcalPedestal ped;
        EcalPedestal oldped =* currentPedestals_->find(id.rawId());
        EcalPedestal g6g1ped=* g6g1Pedestals_->find(id.rawId());

        ped.mean_x12=mean;
        ped.rms_x12=rms;

        float diff = std::abs(mean-oldped.mean_x12);

        // if bad channel or low stat skip or the difference is too large wrt to previous record
        if(ch->getEntries()< minEntries_ || !checkStatusCode(id) || diff>threshDiffEB_){

            ped.mean_x12=oldped.mean_x12;
            ped.rms_x12=oldped.rms_x12;

        }
        
        // copy g6 and g1 from the corressponding record
        ped.mean_x6= g6g1ped.mean_x6;
        ped.rms_x6 = g6g1ped.rms_x6;
        ped.mean_x1= g6g1ped.mean_x1;
        ped.rms_x1 = g6g1ped.rms_x1;

        pedestals.setValue(id.rawId(),ped);
    }


    for (uint16_t i =0; i< EEDetId::kSizeForDenseIndexing; ++i) {

        std::string hname = dqmDir_+"/EE/"+std::to_string(int(i/100))+"/ee_" + std::to_string(i);
     
        MonitorElement* ch= igetter_.get(hname);
        double mean = ch->getMean();
        double rms  = ch->getRMS();
        entriesEE_[i] = ch->getEntries(); 
 
        DetId id = EEDetId::detIdFromDenseIndex(i);
        EcalPedestal ped;
        EcalPedestal oldped = *currentPedestals_->find(id.rawId());
        EcalPedestal g6g1ped= *g6g1Pedestals_->find(id.rawId());
                
        ped.mean_x12=mean;
        ped.rms_x12=rms;

        float diff = std::abs(mean-oldped.mean_x12);

        // if bad channel or low stat skip or the difference is too large wrt to previous record
        if(ch->getEntries()< minEntries_ || !checkStatusCode(id)|| diff>threshDiffEE_){
            ped.mean_x12=oldped.mean_x12;
            ped.rms_x12=oldped.rms_x12;
        }

        // copy g6 and g1 pedestals from corresponding record
        ped.mean_x6= g6g1ped.mean_x6;
        ped.rms_x6 = g6g1ped.rms_x6;
        ped.mean_x1= g6g1ped.mean_x1;
        ped.rms_x1 = g6g1ped.rms_x1;

        pedestals.setValue(id.rawId(),ped);
    }



    dqmPlots(pedestals, ibooker_);

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

    edm::ESHandle<EcalPedestals> g6g1peds;
    isetup.get<EcalPedestalsRcd>().get(labelG6G1_,g6g1peds);
    g6g1Pedestals_=peds.product();

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

bool ECALpedestalPCLHarvester::isGood(const DetId& id){
    
    EcalChannelStatusMap::const_iterator dbstatusPtr;
    dbstatusPtr = channelStatus_->getMap().find(id.rawId());
    if (dbstatusPtr == channelStatus_->getMap().end())
        edm::LogError("Invalid DetId supplied");
    EcalChannelStatusCode::Code  dbstatus = dbstatusPtr->getStatusCode();
    if (dbstatus ==0 ) return true;
    return false;
}


bool ECALpedestalPCLHarvester::checkVariation(const EcalPedestalsMap& oldPedestals, 
                                              const EcalPedestalsMap& newPedestals) {

    uint32_t nAnomaliesEB =0;
    uint32_t nAnomaliesEE =0;

    for (uint16_t i =0; i< EBDetId::kSizeForDenseIndexing; ++i) {
       
        DetId id = EBDetId::detIdFromDenseIndex(i);
        const EcalPedestal& newped=* newPedestals.find(id.rawId());   
        const EcalPedestal& oldped=* oldPedestals.find(id.rawId());

        if  (std::abs(newped.mean_x12 -oldped.mean_x12)  >  nSigma_ * oldped.rms_x12) nAnomaliesEB++;

    }

    for (uint16_t i =0; i< EEDetId::kSizeForDenseIndexing; ++i) {
       
        DetId id = EEDetId::detIdFromDenseIndex(i);
        const EcalPedestal& newped=* newPedestals.find(id.rawId());   
        const EcalPedestal& oldped=* oldPedestals.find(id.rawId());

        if  (std::abs(newped.mean_x12 -oldped.mean_x12)  >  nSigma_ * oldped.rms_x12) nAnomaliesEE++;

    }
    
    if (nAnomaliesEB > thresholdAnomalies_ *  EBDetId::kSizeForDenseIndexing || 
        nAnomaliesEE > thresholdAnomalies_ *  EEDetId::kSizeForDenseIndexing)  
        return true;

    return false;


}




void  ECALpedestalPCLHarvester::dqmPlots(const EcalPedestals& newpeds, DQMStore::IBooker& ibooker){

     ibooker.cd();
     ibooker.setCurrentFolder(dqmDir_+"/Summary");

     MonitorElement * pmeb = ibooker.book2D("meaneb","Pedestal Means EB",360, 1., 361., 171, -85., 86.);
     MonitorElement * preb = ibooker.book2D("rmseb","Pedestal RMS EB ",360, 1., 361., 171, -85., 86.);

     MonitorElement * pmeep = ibooker.book2D("meaneep","Pedestal Means EEP",100,1,101,100,1,101);
     MonitorElement * preep = ibooker.book2D("rmseep","Pedestal RMS EEP",100,1,101,100,1,101);

     MonitorElement * pmeem = ibooker.book2D("meaneem","Pedestal Means EEM",100,1,101,100,1,101);
     MonitorElement * preem = ibooker.book2D("rmseem","Pedestal RMS EEM",100,1,101,100,1,101);

     MonitorElement * pmebd = ibooker.book2D("meanebdiff","Abs Rel Pedestal Means Diff EB",360,1., 361., 171, -85., 86.);
     MonitorElement * prebd = ibooker.book2D("rmsebdiff","Abs Rel Pedestal RMS Diff E ",360, 1., 361., 171, -85., 86.);

     MonitorElement * pmeepd = ibooker.book2D("meaneepdiff","Abs Rel Pedestal Means Diff EEP",100,1,101,100,1,101);
     MonitorElement * preepd = ibooker.book2D("rmseepdiff","Abs Rel Pedestal RMS Diff EEP",100,1,101,100,1,101);

     MonitorElement * pmeemd = ibooker.book2D("meaneemdiff","Abs Rel Pedestal Means Diff EEM",100,1,101,100,1,101);
     MonitorElement * preemd = ibooker.book2D("rmseemdiff","Abs RelPedestal RMS Diff EEM",100,1,101,100,1,101);

     MonitorElement * poeb = ibooker.book2D("occeb","Occupancy EB",360, 1., 361., 171, -85., 86.);
     MonitorElement * poeep = ibooker.book2D("occeep","Occupancy EEP",100,1,101,100,1,101);
     MonitorElement * poeem = ibooker.book2D("occeem","Occupancy EEM",100,1,101,100,1,101);


     MonitorElement * hdiffeb = ibooker.book1D("diffeb","Pedestal Differences EB",100,-2.5,2.5);
     MonitorElement * hdiffee = ibooker.book1D("diffee","Pedestal Differences EE",100,-2.5,2.5);

     for (int hash =0; hash<EBDetId::kSizeForDenseIndexing;++hash){
         
         EBDetId di= EBDetId::detIdFromDenseIndex(hash); 
         float mean= newpeds[di].mean_x12;
         float rms = newpeds[di].rms_x12;
         
         float cmean = (*currentPedestals_)[di].mean_x12;
         float crms  = (*currentPedestals_)[di].rms_x12;
                  
         if (!isGood(di) ) continue;   // only good channels are plotted
 
         pmeb->Fill(di.iphi(),di.ieta(),mean);
         preb->Fill(di.iphi(),di.ieta(),rms);
         if (cmean) pmebd->Fill(di.iphi(),di.ieta(),std::abs(mean-cmean)/cmean);
         if (crms) prebd->Fill(di.iphi(),di.ieta(),std::abs(rms-crms)/crms);
         poeb->Fill(di.iphi(),di.ieta(),entriesEB_[hash]);
         hdiffeb->Fill(mean-cmean);
     }

     
     for (int hash =0; hash<EEDetId::kSizeForDenseIndexing;++hash){
         
         EEDetId di= EEDetId::detIdFromDenseIndex(hash); 
         float mean= newpeds[di].mean_x12;
         float rms = newpeds[di].rms_x12;
         float cmean = (*currentPedestals_)[di].mean_x12;
         float crms  = (*currentPedestals_)[di].rms_x12;

         if (!isGood(di) ) continue;   // only good channels are plotted
         
         if (di.zside() >0){
             pmeep->Fill(di.ix(),di.iy(),mean);
             preep->Fill(di.ix(),di.iy(),rms);
             poeep->Fill(di.ix(),di.iy(),entriesEE_[hash]);
             if (cmean) pmeepd->Fill(di.ix(),di.iy(),std::abs(mean-cmean)/cmean);
             if (crms)  preepd->Fill(di.ix(),di.iy(),std::abs(rms-crms)/crms);
         } else{
             pmeem->Fill(di.ix(),di.iy(),mean);
             preem->Fill(di.ix(),di.iy(),rms);
             if (cmean) pmeemd->Fill(di.ix(),di.iy(),std::abs(mean-cmean)/cmean);
             poeem->Fill(di.ix(),di.iy(),entriesEE_[hash]); 
             if (crms)preemd->Fill(di.ix(),di.iy(),std::abs(rms-crms)/crms);
         }
         hdiffee->Fill(mean-cmean);
         
     }

     

}
