// -*- C++ -*-
//
// Package:    Analyzers
// Class:      testHGCCluster
// 
/**\class testHGCCluster testHGCCluster.cc HGCPFLab/Analyzers/test/hgc_clustering/testHGCCluster.cc

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
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

class testHGCClustering : public edm::EDAnalyzer {

public:
 
    explicit testHGCClustering(const edm::ParameterSet& );
    
    ~testHGCClustering();
  
private:
 
    edm::Service<TFileService> fs_;  
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
    // ----------member data ---------------------------
    //Luca EDGetTokenT<HGCalTriggerCellBxCollection > tokenTrgCell_;
    EDGetTokenT<HGCalClusterBxCollection > tokenCluster_;
   
    bool debug_;
    int nEvent_=0;

    TH1D *h_photon_pt_;
    TH1D *h_cl_hwPt_;
    TH1D *h_cl_E_;
    TH1D *h_cl_pt_;
    TH1D *h_cl_eta_;
    TH1D *h_cl_phi_;
    TH1D *h_cl_layer_;

    TTree *mytree_;  

    double CL_pt_=0.;
};



// constructors and destructor
testHGCClustering::testHGCClustering(const edm::ParameterSet& iConfig) : 
    //Luca tokenTrgCell_( consumes< HGCalTriggerCellBxCollection >( iConfig.getParameter<InputTag> ( "triggerCellInputTag" ) ) )
    tokenCluster_( consumes< HGCalClusterBxCollection >( iConfig.getParameter<InputTag> ( "clusterInputTag" ) ) )
{
    
//    consumes< BXVector< l1t::HGCalTriggerCell >  > (TrgCells_tag_);    
//    consumes< HGCalTriggerCellBxCollection > (TrgCells_tag_);    
   
    //now do what ever initialization is needed
    //edm::Service<TFileService> fs;
    h_photon_pt_ = fs_->make<TH1D>("h_photon_pt" ,  "h_photon_pt" , 250 , 0 , 500 );
    h_cl_hwPt_   = fs_->make<TH1D>("h_cl_hwPt" , "h_tc_hwPt" , 100 , 0 , 2 );
    h_cl_E_      = fs_->make<TH1D>("h_cl_E" , "h_cl_E" , 200 , 0 , 50 );
    h_cl_pt_     = fs_->make<TH1D>("h_cl_pt" , "h_cl_pt" , 200 , 0 , 50 );
    h_cl_eta_    = fs_->make<TH1D>("h_cl_eta" , "h_cl_eta" , 100 , -3.1 , 3.1 );
    h_cl_phi_    = fs_->make<TH1D>("h_cl_phi" , "h_cl_phi" , 100 , -3.14 , 3.14 );
    h_cl_layer_  = fs_->make<TH1D>("h_cl_layer" , "h_cl_layer" , 40 , 0 , 40 );
    //ATTENTION TO THE ETA VALUES CODED IN THE CLUSTERING ALGORITHMS
 
    mytree_     = fs_->make <TTree>("tree","tree"); 

    //TTree branch
    mytree_->Branch("event",	&nEvent_,	"event/I");
    mytree_->Branch("CL_pt",	&CL_pt_,	"CL_pt/D");

}

testHGCClustering::~testHGCClustering()
{
}

// ------------ method called for each event  ------------
void testHGCClustering::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace HGCalTriggerBackend;
    using namespace l1t;

    //Luca    Handle<HGCalTriggerCellBxCollection> trgCell;
    Handle<HGCalClusterBxCollection> trgClu;
    //Luca    iEvent.getByToken(tokenTrgCell_, trgCell);
    iEvent.getByToken(tokenCluster_, trgClu);
   
    //double E_allTC_endcap_PosEta=0.;
    //double E_allTC_endcap_NegEta=0.;
    std::cout << "  size del vettore di cluster: " << trgClu->size() << std::endl;
    for(size_t i=0; i<trgClu->size(); ++i){
        if((*trgClu)[i].pt() >0 ) continue;
        //HGCalDetId detid( (*trgClu)[i].detId() );
        //int cl_layer = detid.layer();

        //if((*trgCell)[i].eta()<0) 
        //    E_allTC_endcap_NegEta += (*trgCell)[i].pt();
        //else if((*trgCell)[i].eta()>0) 
        //    E_allTC_endcap_PosEta += (*trgCell)[i].pt();
        
        h_cl_pt_->Fill((*trgClu)[i].pt());
        std::cout << "cluster pt in the test " << (*trgClu)[i].hwPtEm() << " had "<< (*trgClu)[i].hwPtHad() <<std::endl;
        //h_tc_eta_->Fill((*trgCell)[i].eta());
        //h_tc_phi_->Fill((*trgCell)[i].phi());
        //h_tc_layer_->Fill(tc_layer);

    }
    //if(E_allTC_endcap_PosEta>0){
    //    h_photon_pt_->Fill(E_allTC_endcap_PosEta);
    //}else if(E_allTC_endcap_NegEta>0){
    //    h_photon_pt_->Fill(E_allTC_endcap_NegEta);
    //}
    h_photon_pt_->Fill((*trgClu)[0].pt());

    mytree_->Fill();

}

//define this as a plug-in
DEFINE_FWK_MODULE(testHGCClustering);



