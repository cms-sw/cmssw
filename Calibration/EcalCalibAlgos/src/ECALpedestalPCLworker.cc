
#include "Calibration/EcalCalibAlgos/interface/ECALpedestalPCLworker.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>
#include <sstream>



ECALpedestalPCLworker::ECALpedestalPCLworker(const edm::ParameterSet& iConfig)

{
    digiTagEB_= iConfig.getParameter<edm::InputTag>("BarrelDigis");
    digiTagEE_= iConfig.getParameter<edm::InputTag>("EndcapDigis");

    digiTokenEB_ = consumes<EBDigiCollection>(digiTagEB_);
    digiTokenEE_ = consumes<EEDigiCollection>(digiTagEE_);
}



// ------------ method called for each event  ------------
void
ECALpedestalPCLworker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
      
    Handle<EBDigiCollection> pDigiEB;
    iEvent.getByLabel("ecalDigis","ebDigis",pDigiEB);
   
    Handle<EEDigiCollection> pDigiEE;
    iEvent.getByLabel("ecalDigis","eeDigis",pDigiEE);



    for (EBDigiCollection::const_iterator pDigi=pDigiEB->begin(); pDigi!=pDigiEB->end(); ++pDigi){
         
        EBDetId id = pDigi->id();
        uint32_t hashedId = id.hashedIndex();

        EBDataFrame digi( *pDigi );

        uint16_t maxdiff = *std::max_element(digi.frame().begin(), digi.frame().end(), adc_compare )  - 
                           *std::min_element(digi.frame().begin(), digi.frame().end(), adc_compare ); 
        if ( maxdiff> kThreshold ) continue; // assume there is signal in this frame       
        //for (auto& mgpasample : digi.frame()) meEB_[hashedId]->Fill(mgpasample&0xFFF);
        for (edm::DataFrame::iterator mgpasample = digi.frame().begin();  
             mgpasample!=digi.frame().begin()+kPedestalSamples; 
             ++mgpasample )
            meEB_[hashedId]->Fill(*mgpasample&0xFFF);

    } // eb digis
   

    for (EEDigiCollection::const_iterator pDigi=pDigiEE->begin(); pDigi!=pDigiEE->end(); ++pDigi){
         
        EEDetId id = pDigi->id();
        uint32_t hashedId = id.hashedIndex();

        EBDataFrame digi( *pDigi );

        uint16_t maxdiff = *std::max_element(digi.frame().begin(), digi.frame().end(), adc_compare )  - 
                           *std::min_element(digi.frame().begin(), digi.frame().end(), adc_compare ); 
        if ( maxdiff> kThreshold ) continue; // assume there is signal in this frame       
        //for (auto& mgpasample : digi.frame()) meEE_[hashedId]->Fill(mgpasample&0xFFF);
        for (edm::DataFrame::iterator mgpasample = digi.frame().begin();  
             mgpasample!=digi.frame().begin()+kPedestalSamples; 
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
    ibooker.setCurrentFolder("EcalCalibration/EcalPedestalPCL");

    for ( uint32_t i = 0 ; i< EBDetId::kSizeForDenseIndexing; ++i){
        std::stringstream hname;
        hname<<"eb_"<<i;       
        meEB_.push_back(ibooker.book1D(hname.str(),hname.str(),100,150,250));
    }

    for ( uint32_t i = 0 ; i< EEDetId::kSizeForDenseIndexing; ++i){
        std::stringstream hname;
        hname<<"ee_"<<i;
        meEE_.push_back(ibooker.book1D(hname.str(),hname.str(),100,150,250));

    }
       
}


