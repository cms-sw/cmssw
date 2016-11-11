// -*- C++ -*-
//
// Package:    Analyzers
// Class:      testCalibration
// 
/**\class testCalibration testCalibration.cc HGCPFLab/Analyzers/plugins/testCalibration.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Luca Mastrolorenzo
//         Created:  Mon, 07 Dec 2015 16:38:57 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"
#include "TTree.h"

//
// class declaration
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace HGCalTriggerBackend;
using namespace l1t;

class testCalibration : public edm::EDAnalyzer {

public:
 
    explicit testCalibration(const edm::ParameterSet& );
    
    ~testCalibration();
  
private:
 
    edm::Service<TFileService> fs_;  
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
    // ----------member data ---------------------------
    EDGetTokenT<HGCalTriggerCellBxCollection > tokenTrgCell_;
   
    bool debug_;
    int _nEvent=0;

    TH1D *h_photon_pt_;
    TH1D *h_tc_hwPt_;
    TH1D *h_tc_E_;
    TH1D *h_tc_pt_;
    TH1D *h_tc_eta_;
    TH1D *h_tc_phi_;
    TH1D *h_tc_layer_;

    TTree *mytree_;  

    double _TC_pt=0.;
};



// constructors and destructor
testCalibration::testCalibration(const edm::ParameterSet& iConfig) : 
    tokenTrgCell_( consumes< HGCalTriggerCellBxCollection >( iConfig.getParameter<InputTag> ( "triggerCellInputTag" ) ) ),
    debug_( iConfig.getParameter<bool>( "DebugCode" ) )
{
    
//    consumes< BXVector< l1t::HGCalTriggerCell >  > (TrgCells_tag_);    
//    consumes< HGCalTriggerCellBxCollection > (TrgCells_tag_);    
   
    //now do what ever initialization is needed
    //edm::Service<TFileService> fs;
    h_photon_pt_ = fs_->make<TH1D>( "h_photon_pt" ,  "h_photon_pt" , 240 , 0 , 120 );
    h_tc_hwPt_   = fs_->make<TH1D>("h_tc_hwPt" , "h_tc_hwPt" , 100 , 0 , 2 );
    h_tc_E_      = fs_->make<TH1D>("h_tc_E" , "h_tc_E" , 200 , 0 , 50 );
    h_tc_pt_     = fs_->make<TH1D>("h_tc_pt" , "h_tc_pt" , 200 , 0 , 50 );
    h_tc_eta_    = fs_->make<TH1D>("h_tc_eta" , "h_tc_eta" , 100 , -3.1 , 3.1 );
    h_tc_phi_    = fs_->make<TH1D>("h_tc_phi" , "h_tc_phi" , 100 , -3.14 , 3.14 );
    h_tc_layer_  = fs_->make<TH1D>("h_tc_layer" , "h_tc_layer" , 40 , 0 , 40 );
    
    mytree_     = fs_->make <TTree>("tree","tree"); 

    //TTree branch
    mytree_->Branch("event",	&_nEvent,	"event/I");
    mytree_->Branch("TC_pt",	&_TC_pt,	"TC_pt/D");

}


testCalibration::~testCalibration()
{
}


// ------------ method called for each event  ------------
void testCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace HGCalTriggerBackend;
    using namespace l1t;

    Handle<HGCalTriggerCellBxCollection>  trgCell;
    iEvent.getByToken(tokenTrgCell_,  trgCell);
        
//    Handle< >;
//    iEvent.getByLabel(TrgCells_tag_, trgCell);
   
    double E_allTC_endcap0=0.;
    double E_allTC_endcap1=0.;
   
    for(size_t i=0; i<trgCell->size(); ++i){
        HGCalDetId detid( (*trgCell)[i].detId() );
        int tc_layer = detid.layer();

        if((*trgCell)[i].eta()<0) 
            E_allTC_endcap0+=(*trgCell)[i].pt();
        else if((*trgCell)[i].eta()>0) 
            E_allTC_endcap1+=(*trgCell)[i].pt();
        
        h_tc_pt_->Fill((*trgCell)[i].pt());
        h_tc_eta_->Fill((*trgCell)[i].eta());
        h_tc_phi_->Fill((*trgCell)[i].phi());
        h_tc_layer_->Fill(tc_layer);

    }
    h_photon_pt_->Fill(E_allTC_endcap0);
    h_photon_pt_->Fill(E_allTC_endcap0);

    mytree_->Fill();

}

//define this as a plug-in
DEFINE_FWK_MODULE(testCalibration);



