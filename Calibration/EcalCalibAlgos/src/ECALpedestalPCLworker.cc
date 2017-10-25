
#include "Calibration/EcalCalibAlgos/interface/ECALpedestalPCLworker.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include <iostream>
#include <sstream>



ECALpedestalPCLworker::ECALpedestalPCLworker(const edm::ParameterSet& iConfig)

{
    edm::InputTag digiTagEB= iConfig.getParameter<edm::InputTag>("BarrelDigis");
    edm::InputTag digiTagEE= iConfig.getParameter<edm::InputTag>("EndcapDigis");

    digiTokenEB_ = consumes<EBDigiCollection>(digiTagEB);
    digiTokenEE_ = consumes<EEDigiCollection>(digiTagEE);

    pedestalSamples_ = iConfig.getParameter<uint32_t>("pedestalSamples");
    checkSignal_     = iConfig.getParameter<bool>("checkSignal");
    sThresholdEB_      = iConfig.getParameter<uint32_t>("sThresholdEB");
    sThresholdEE_      = iConfig.getParameter<uint32_t>("sThresholdEE");

    dynamicBooking_ = iConfig.getParameter<bool>("dynamicBooking");
    fixedBookingCenterBin_ = iConfig.getParameter<int>("fixedBookingCenterBin");
    nBins_          = iConfig.getParameter<int>("nBins");
    dqmDir_         = iConfig.getParameter<std::string>("dqmDir");

    edm::InputTag bstRecord= iConfig.getParameter<edm::InputTag>("bstRecord");  
    bstToken_       = consumes<BSTRecord>(bstRecord);
    requireStableBeam_ = iConfig.getParameter<bool>("requireStableBeam");
}



// ------------ method called for each event  ------------
void
ECALpedestalPCLworker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    Handle<EBDigiCollection> pDigiEB;
    iEvent.getByToken(digiTokenEB_,pDigiEB);

    Handle<EEDigiCollection> pDigiEE;
    iEvent.getByToken(digiTokenEE_,pDigiEE);


    // Only Events with stable beam
 
    if (requireStableBeam_){
        edm::Handle<BSTRecord> bstData;           
        iEvent.getByToken(bstToken_,bstData);
        int beamMode = static_cast<int>( bstData->beamMode() );
        if (beamMode != 11 ) return;
    }

    for (EBDigiCollection::const_iterator pDigi=pDigiEB->begin(); pDigi!=pDigiEB->end(); ++pDigi){

        EBDetId id = pDigi->id();
        uint32_t hashedId = id.hashedIndex();

        EBDataFrame digi( *pDigi );
        
        if (checkSignal_){
            uint16_t maxdiff = *std::max_element(digi.frame().begin(), digi.frame().end(), adc_compare )  -
                               *std::min_element(digi.frame().begin(), digi.frame().end(), adc_compare );
            if ( maxdiff> sThresholdEB_ ) continue; // assume there is signal in this frame
        }

        //for (auto& mgpasample : digi.frame()) meEB_[hashedId]->Fill(mgpasample&0xFFF);
        for (edm::DataFrame::iterator mgpasample = digi.frame().begin();
             mgpasample!=digi.frame().begin()+pedestalSamples_;
             ++mgpasample )
            meEB_[hashedId]->Fill(*mgpasample&0xFFF);

    } // eb digis



    for (EEDigiCollection::const_iterator pDigi=pDigiEE->begin(); pDigi!=pDigiEE->end(); ++pDigi){

        EEDetId id = pDigi->id();
        uint32_t hashedId = id.hashedIndex();

        EEDataFrame digi( *pDigi );

        if (checkSignal_){
            uint16_t maxdiff = *std::max_element(digi.frame().begin(), digi.frame().end(), adc_compare )  -
                               *std::min_element(digi.frame().begin(), digi.frame().end(), adc_compare );
            if ( maxdiff> sThresholdEE_ ) continue; // assume there is signal in this frame
        }

        //for (auto& mgpasample : digi.frame()) meEE_[hashedId]->Fill(mgpasample&0xFFF);
        for (edm::DataFrame::iterator mgpasample = digi.frame().begin();
             mgpasample!=digi.frame().begin()+pedestalSamples_;
             ++mgpasample )
            meEE_[hashedId]->Fill(*mgpasample&0xFFF);

    } // ee digis




}


void
ECALpedestalPCLworker::beginJob()
{
}


void
ECALpedestalPCLworker::endJob()
{
}


void
ECALpedestalPCLworker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}


void
ECALpedestalPCLworker::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & run, edm::EventSetup const & es){

    ibooker.cd();
    ibooker.setCurrentFolder(dqmDir_);

    edm::ESHandle<EcalPedestals> peds;
    es.get<EcalPedestalsRcd>().get(peds);
    

    for ( uint32_t i = 0 ; i< EBDetId::kSizeForDenseIndexing; ++i){

        
        ibooker.setCurrentFolder(dqmDir_+"/EB/"+std::to_string(int(i/100)));

        std::string hname = "eb_" + std::to_string(i);
        DetId id = EBDetId::detIdFromDenseIndex(i);
        int centralBin = fixedBookingCenterBin_;
         
        if (dynamicBooking_){
            centralBin =  int ((peds->find(id))->mean_x12) ;
        }

        int min = centralBin - nBins_/2;
        int max = centralBin + nBins_/2;

        meEB_.push_back(ibooker.book1D(hname,hname,nBins_,min,max));
    }

    for ( uint32_t i = 0 ; i< EEDetId::kSizeForDenseIndexing; ++i){

        ibooker.setCurrentFolder(dqmDir_+"/EE/"+std::to_string(int(i/100)));

        std::string hname = "ee_" + std::to_string(i);

        DetId id = EEDetId::detIdFromDenseIndex(i);
        int centralBin = fixedBookingCenterBin_;
         
        if (dynamicBooking_){
            centralBin =  int ((peds->find(id))->mean_x12) ;
        }

        int min = centralBin - nBins_/2;
        int max = centralBin + nBins_/2;

        meEE_.push_back(ibooker.book1D(hname,hname,nBins_,min,max));

    }

}
