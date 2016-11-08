// -*- C++ -*-
//
// Package:    Analyzers
// Class:      L1TcCalib
// 
/**\class L1TcCalib L1TcCalib.cc HGCPFLab/Analyzers/plugins/L1TcCalib.cc

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
#include "ToInclude.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// We probably don't need all of these
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodecImpl.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalUncalibRecHitRecWeightsAlgo.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalUncalibRecHitRecAbsAlgo.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"


#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1.h" 
#include "TTree.h"
#include "math.h"

#include "TTree.h"
#include "TLorentzVector.h"
#include "TClonesArray.h"
#include "TParticle.h"
#include "TVector3.h"

//
// class declaration
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace HGCalTriggerBackend;
using namespace l1t;

class L1TcCalib : public edm::EDAnalyzer {

public:
 
    explicit L1TcCalib(const edm::ParameterSet& );
    
    ~L1TcCalib();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  
private:
 
    edm::Service<TFileService> fs_;
      
    virtual void beginJob();
    virtual void beginRun(const edm::Run&,  const edm::EventSetup&);
    void init();
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    // override;
    virtual void endJob();
    
    bool debug_;
    edm::InputTag TrgCells_tag_;

    

    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    // ----------member data ---------------------------
    TH1D *h_photon_E;
    TH1D *h_photon_pt;
    TH1D *h_photon_eta;
    TH1D *h_photon_phi;

    TH1D *h_tc_hwPt;
    TH1D *h_tc_E;
    TH1D *h_tc_pt;
    TH1D *h_tc_eta;
    TH1D *h_tc_phi;
    TH1D *h_tc_layer;
//    TH1D *h_tc_x;
//    TH1D *h_tc_y;

    TTree *mytree_;  

    int _nEvent=0;
    double _TC_pt=0.;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TcCalib::L1TcCalib(const edm::ParameterSet& iConfig) : 
    debug_( iConfig.getParameter<bool>( "DebugCode" ) )
{
    TrgCells_tag_ = edm::InputTag("hgcalTriggerPrimitiveDigiProducer:SingleCellClusterAlgo");
    
    consumes< BXVector< l1t::HGCalTriggerCell >  > (TrgCells_tag_);    
   
    //now do what ever initialization is needed
    //edm::Service<TFileService> fs;
    h_photon_E = fs_->make<TH1D>(  "h_photon_E" ,   "h_photon_E" , 240 , 0 , 120 );
    h_photon_pt = fs_->make<TH1D>( "h_photon_pt" ,  "h_photon_pt" , 240 , 0 , 120 );
    h_photon_eta = fs_->make<TH1D>("h_photon_eta" , "h_photon_eta" , 100 , -3.1 , 3.1 );
    h_photon_phi = fs_->make<TH1D>("h_photon_phi" , "h_photon_phi" , 100 , -3.14 , 3.14 );

    h_tc_hwPt = fs_->make<TH1D>("h_tc_hwPt" , "h_tc_hwPt" , 100 , 0 , 2 );
    h_tc_E = fs_->make<TH1D>("h_tc_E" , "h_tc_E" , 200 , 0 , 50 );
    h_tc_pt = fs_->make<TH1D>("h_tc_pt" , "h_tc_pt" , 200 , 0 , 50 );
    h_tc_eta = fs_->make<TH1D>("h_tc_eta" , "h_tc_eta" , 100 , -3.1 , 3.1 );
    h_tc_phi = fs_->make<TH1D>("h_tc_phi" , "h_tc_phi" , 100 , -3.14 , 3.14 );
//    h_tc_x = fs_->make<TH1D>("h_tc_x" , "h_tc_x" , 360 , 180 , 180 );
//    h_tc_y = fs_->make<TH1D>("h_tc_y" , "h_tc_y" , 360 , 180 , 180 );

    h_tc_layer = fs_->make<TH1D>("h_tc_layer" , "h_tc_layer" , 40 , 0 , 40 );
    
    mytree_  = fs_->make <TTree>("tree","tree"); 

    // Global
    mytree_->Branch("event",	&_nEvent,	"event/I");
    mytree_->Branch("TC_pt",	&_TC_pt,	"TC_pt/D");

}


L1TcCalib::~L1TcCalib()
{
 
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TcCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace HGCalTriggerBackend;
    using namespace l1t;

    init();
    std::cout << "\n";
    std::cout << "Event " << iEvent.id() << std::endl;
    
    Handle< BXVector<l1t::HGCalTriggerCell>  > trgCell;
    iEvent.getByLabel(TrgCells_tag_, trgCell);
   
    double totE_0=0.;
    double totE_1=0.;
   
    std::cout << " trigger cell size =  "<< trgCell->size() << std::endl;
    for( size_t i=0; i<trgCell->size(); ++i){
        std::cout << "new implementation of JB --> trgCell Pt = " << (*trgCell)[i].p4().pt() << " energy " << (*trgCell)[i].p4().E() << std::endl;
        HGCalDetId detid( (*trgCell)[i].detId());
        int tc_layer = detid.layer();

        if((*trgCell)[i].p4().Eta()>0)totE_1+=(*trgCell)[i].p4().Pt();
        else totE_0+=(*trgCell)[i].p4().Pt();
        h_tc_pt->Fill((*trgCell)[i].p4().Pt());
        h_tc_eta->Fill((*trgCell)[i].p4().Eta());
        h_tc_phi->Fill((*trgCell)[i].p4().Phi());
        h_tc_layer->Fill(tc_layer);

    }
    h_photon_pt->Fill(totE_0);
    h_photon_pt->Fill(totE_1);

    std::cout << "E tot 0 = " << totE_0 <<  std::endl;
    std::cout << "E tot 1 = " << totE_1 <<  std::endl;

    mytree_->Fill();


#ifdef THIS_IS_AN_EVENT_EXAMPLE
    Handle<ExampleData> pIn;
    iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
    ESHandle<SetupData> pSetup;
    iSetup.get<SetupRecord>().get(pSetup);
#endif
}


void L1TcCalib::init()
{
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TcCalib::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TcCalib::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void L1TcCalib::beginRun(const edm::Run&, const edm::EventSetup& es)
{
}


// ------------ method called when ending the processing of a run  ------------
/*
  void 
  L1TcCalib::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void 
  L1TcCalib::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void 
  L1TcCalib::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TcCalib::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TcCalib);



