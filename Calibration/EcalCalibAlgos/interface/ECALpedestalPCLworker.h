// -*- C++ -*-
//
// Package:    Calibration/EcalCalibAlgos
// Class:      ECALpedestalPCLworker
// 
/**\class ECALpedestalPCLworker ECALpedestalPCLworker.cc 

   Description: Fill DQM histograms with pedestals. Intended to be used on laser data from the TestEnablesEcalHcal dataset

 
*/
//
// Original Author:  Stefano Argiro
//         Created:  Wed, 22 Mar 2017 14:46:48 GMT
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

//
// class declaration
//

class ECALpedestalPCLworker : public  DQMEDAnalyzer {
public:
    explicit ECALpedestalPCLworker(const edm::ParameterSet&);
     

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
    virtual void beginJob() ;
    virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);   
    virtual void analyze(const edm::Event&, const edm::EventSetup&) ;
    virtual void endJob() ;

    edm::InputTag digiTagEB_;
    edm::InputTag digiTagEE_;
    edm::EDGetTokenT<EBDigiCollection> digiTokenEB_; 
    edm::EDGetTokenT<EEDigiCollection> digiTokenEE_; 

    std::vector<MonitorElement *> meEB_;
    std::vector<MonitorElement *> meEE_;

    uint32_t pedestalSamples_ ; // number of presamples to be used for pedestal determination    
    bool     checkSignal_;      // avoid frames containing a signal
    uint32_t  sThreshold_ ;     // if checkSignal = true threshold (in adc count) above which we'll assume 
                                // there's signal and not just pedestal

    bool dynamicBooking_;       // use old pedestal to book histograms
    int fixedBookingCenterBin_; // if dynamicBooking_ = false, use this as bin center
    int nBins_ ;                // number of bins per histogram
    std::string dqmDir_;         // DQM directory where histograms are stored
    

    // compare ADC values  
    static bool adc_compare(uint16_t a, uint16_t b) { return ( a&0xFFF )  < (b&0xFFF);}
};
