// -*- C++ -*-
//
// Package:    DustyDemo
// Class:      DustyDemo
// 
/**\class DustyDemo DustyDemo.cc Demo/DustyDemo/src/DustyDemo.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  dustin stolp
//         Created:  Wed Sep 11 06:35:36 CDT 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
#include "TH1.h"

using namespace std;
//
// class declaration
//

class DustyDemo : public edm::EDAnalyzer {
   public:
      explicit DustyDemo(const edm::ParameterSet&);
      ~DustyDemo();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(edm::Event const&, edm::EventSetup const&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

     // ----------member data ---------------------------
      TH1F * hist_em_et;
      TH1F * hist_em_eta;
      TH1F * hist_jet_et;
      TH1F * hist_jet_eta;
      TH1F * hist_muon_et;
      TH1F * hist_muon_eta;
      edm::InputTag isoEmSource_ ;
      edm::InputTag nonIsoEmSource_ ;
      edm::InputTag cenJetSource_ ;
      edm::InputTag forJetSource_ ;
//      edm::InputTag tauJetSource_ ;
      edm::InputTag muonSource_ ;
      edm::InputTag etMissSource_ ;
//      edm::InputTag htMissSource_ ;
//      edm::InputTag hfRingsSource_ ;
//      edm::InputTag gtReadoutSource_ ;
//      edm::InputTag particleMapSource_ ;
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
DustyDemo::DustyDemo(const edm::ParameterSet& iConfig) :
   isoEmSource_( iConfig.getParameter< edm::InputTag >(
      "isolatedEmSource" ) ),
     nonIsoEmSource_( iConfig.getParameter< edm::InputTag >(
      "nonIsolatedEmSource" ) ),
     cenJetSource_( iConfig.getParameter< edm::InputTag >(
      "centralJetSource" ) ),
     forJetSource_( iConfig.getParameter< edm::InputTag >(
      "forwardJetSource" ) ),
//     tauJetSource_( iConfig.getParameter< edm::InputTag >(
//      "tauJetSource" ) ),
     muonSource_( iConfig.getParameter< edm::InputTag >(
      "muonSource" ) ),
     etMissSource_( iConfig.getParameter< edm::InputTag >(
      "etMissSource" ) )
//     htMissSource_( iConfig.getParameter< edm::InputTag >(
//      "htMissSource" ) ),
//     hfRingsSource_( iConfig.getParameter< edm::InputTag >(
//      "hfRingsSource" ) ),
//     gtReadoutSource_( iConfig.getParameter< edm::InputTag >(
//      "gtReadoutSource" ) ),
//     particleMapSource_( iConfig.getParameter< edm::InputTag >(
//      "particleMapSource" ) )
{
edm::Service<TFileService>fs;
hist_em_et = fs->make<TH1F>("hist_em_et","EM Particle Et;Et (GeV);Events",28,0,70);
hist_em_eta = fs->make<TH1F>("hist_em_eta","EM Particle #eta;#eta occupancy;Events",20,-4,4);
hist_jet_et = fs->make<TH1F>("hist_jet_et","Jet Particle Et;Et (GeV);Events",40,100,300);
hist_jet_eta = fs->make<TH1F>("hist_jet_eta","Jet Particle #eta;#eta occupancy;Events",20,-4,4);
hist_muon_et = fs->make<TH1F>("hist_muon_et","Muon Particle Et;Et (GeV);Events",32,0,160);
hist_muon_eta = fs->make<TH1F>("hist_muon_eta","Muon Particle #eta;#eta occupancy;Events",20,-4,4);

   //now do what ever initialization is needed

}


DustyDemo::~DustyDemo()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
DustyDemo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
/*   std::cout<<"hey hey **********************************************************************************"<<std::endl;
   Handle<EcalTrigPrimDigiCollection> ecal;
   iEvent.getByLabel("simEcalTriggerPrimitiveDigis",ecal);
   EcalTrigPrimDigiCollection ecal_collection;
   if(ecal.isValid()) {ecal_collection = *ecal;}
   else{return;}
//   for(unsigned int i=0;i<ecal_collection.size();i++){
//	if(ecal_collection[i].compressedEt()>0){	
//		hist->Fill(ecal_collection[i].compressedEt());
//		std::cout<<"ecal collection et\t"<<ecal_collection[i].compressedEt()<<std::endl;
//	}
//   }

   Handle<HcalTrigPrimDigiCollection> hcal;
   iEvent.getByLabel("simHcalTriggerPrimitiveDigis",hcal);
   HcalTrigPrimDigiCollection hcal_collection;
   if(hcal.isValid()) {hcal_collection = *hcal;}
   else{return;}
//   for(unsigned int i=0;i<hcal_collection.size();i++){
//	   if(hcal_collection[i].SOI_compressedEt()>0){
//		   std::cout<<"hcal collection et\t"<<hcal_collection[i].SOI_compressedEt()<<std::endl;
//           }
//   }


   Handle<DTLayerIdDTDigiSimLinkMuonDigiCollection> dtdigi;
   iEvent.getByLabel("simMuonDTDigis",dtdigi);
   DTDigiCollection dtdigi_collection;
   if(dtdigi.isValid()) {dtdigi_collection = *dtdigi;}
   else{return;}
   cout<<"dtdigi size "<<endl;
*/
   Handle<l1extra::L1EmParticleCollection> l1em;
   iEvent.getByLabel(nonIsoEmSource_,l1em);
   l1extra::L1EmParticleCollection l1em_collection;
   if(l1em.isValid()) {l1em_collection =*l1em;}
   else{return;}
//   hist_em->Fill(l1em_collection.size());
//   std::cout<<l1em_collection.size()<<std::endl;
   for(unsigned int i=0;i<l1em_collection.size();i++){
//	   if(l1em_collection[i].bx()!=0) continue;
	   hist_em_et->Fill(l1em_collection[i].et());
	   hist_em_eta->Fill(l1em_collection[i].eta());
//	   std::cout<<"l1 em collection bx\t"<<l1em_collection[i].bx()<<std::endl;
   }
/*
   Handle<l1extra::L1EtMissParticleCollection> l1met;
   iEvent.getByLabel(etMissSource_,l1met);
   l1extra::L1EtMissParticleCollection l1met_collection;
   if(l1met.isValid()) {l1met_collection =*l1met;}
   else{return;}
   for(unsigned int i=0;i<l1met_collection.size();i++){
//	   std::cout<<"l1 met collection MET\t"<<l1met_collection[i].etMiss()<<std::endl;
//   }
*/
   Handle<l1extra::L1JetParticleCollection> l1jet;
   iEvent.getByLabel(cenJetSource_,l1jet);
   l1extra::L1JetParticleCollection l1jet_collection;
   if(l1jet.isValid()) {l1jet_collection =*l1jet;}
   else{return;}
   for(unsigned int i=0;i<l1jet_collection.size();i++){
//	   if(l1jet_collection[i].bx()!=0) continue;
	   hist_jet_et->Fill(l1jet_collection[i].et());
	   hist_jet_eta->Fill(l1jet_collection[i].eta());
//   std::cout<<l1jet_collection.size()<<std::endl;
//	   std::cout<<"l1 jet collection bx\t"<<l1jet_collection[i].bx()<<std::endl;
   }

   Handle<l1extra::L1MuonParticleCollection> l1muon;
   iEvent.getByLabel(muonSource_,l1muon);
   l1extra::L1MuonParticleCollection l1muon_collection;
   if(l1muon.isValid()) {l1muon_collection =*l1muon;}
   else{return;}
   std::cout<<l1muon_collection.size()<<std::endl;
   for(unsigned int i=0;i<l1muon_collection.size();i++){
	   if(l1muon_collection[i].bx()!=0) continue;
   	   hist_muon_et->Fill(l1muon_collection[i].et());
	   hist_muon_eta->Fill(l1muon_collection[i].eta());
//	   std::cout<<"l1 muon collection bx\t"<<l1muon_collection[i].bx()<<std::endl;
   }



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
DustyDemo::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DustyDemo::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
DustyDemo::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
DustyDemo::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
DustyDemo::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
DustyDemo::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DustyDemo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DustyDemo);
