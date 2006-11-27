// -*- C++ -*-
//
// Package:    ElectronAnalyzer
// Class:      ElectronAnalyzer
// 
/**\class ElectronAnalyzer ElectronAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alessandro Palma
//         Created:  Thu Sep 21 11:41:35 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include<string>
#include "math.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"

#include "TH1.h"
#include "TFile.h"

//
// constructors and destructor
//
ElectronAnalyzer::ElectronAnalyzer(const edm::ParameterSet& iConfig) :
  minElePt_(iConfig.getParameter<double>("minElePt")), REleCut_(iConfig.getParameter<double>("REleCut")), outputFile_(iConfig.getParameter<std::string>("outputFile")), electronProducer_(iConfig.getParameter<edm::InputTag>("electronProducer")), mcProducer_(iConfig.getParameter<std::string>("mcProducer")), scProducer_(iConfig.getParameter<edm::InputTag>("superClusterProducer"))
{
   //now do what ever initialization is needed
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms
}


ElectronAnalyzer::~ElectronAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete rootFile_;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

    //SUPERCLUSTERS - BEGIN
   
   Handle<reco::SuperClusterCollection> mySC;
   iEvent.getByLabel(scProducer_, mySC); 
   
   for(reco::SuperClusterCollection::const_iterator scIt = mySC->begin();
       scIt != mySC->end(); scIt++){
     for(reco::SuperClusterCollection::const_iterator scIt2 = scIt+1;
	 scIt2 != mySC->end(); scIt2++){
	 }
   }
   //SUPERCLUSTERS - END
   
   
   //GET GENERATOR EVENT - BEGIN

   Handle< HepMCProduct > hepProd ;
   iEvent.getByLabel( mcProducer_.c_str(), hepProd ) ;
   const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();

   //GET GENERATOR EVENT - END
   
   
   //LOOP OVER ELECTRONS - BEGIN
   Handle<reco::ElectronCollection> myEle;
   //   iEvent.getByLabel(electronProducer_.c_str(), myEle); 
   iEvent.getByLabel(electronProducer_, myEle); 
   
   std::vector<reco::Electron> eleVec;
   std::vector<reco::Electron> posVec;
   
   h1_nEleReco_->Fill(myEle->size());
   
   for(reco::ElectronCollection::const_iterator eleIt = myEle->begin();
       eleIt != myEle->end(); eleIt++){

     //try to print ele track exp. error - end
     //std::cout<<eleIt->track()->chi2()<<std::endl;
     //     eleIt->track()->outerStateCovariance();
     //try to print ele track exp. error - end

     h1_recoEleEnergy_->Fill(eleIt->superCluster()->energy());     

     h1_recoElePt_->Fill(eleIt->pt());     
     h1_recoEleEta_->Fill(eleIt->eta());     
     h1_recoElePhi_->Fill(eleIt->phi());     
     
     //FILL an electron vector - begin
     
     if(eleIt->charge()==-1) eleVec.push_back(*eleIt);
     if(eleIt->charge()==+1) posVec.push_back(*eleIt);
     
     //FILL an electron vector - end
     
   }
   //LOOP OVER ELECTRONS - END
   

   //loop over MC electrons and find the closest MC electron in (eta,phi) phace space - begin   
   double REle(0.);
   double REleMin(1000.);
   reco::Electron nearestEleToMC;
   HepMC::GenParticle eleMC;
   
   if(eleVec.size()>=1){
     for(std::vector<reco::Electron>::const_iterator e_it = eleVec.begin(); e_it !=eleVec.end(); e_it++){
       for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	     p != myGenEvent->particles_end(); ++p ) {
	 if ( (*p)->pdg_id() == 11 && (*p)->status()==1 ){
	   eleMC=*(*p);
	   
	   REle = pow(e_it->p4().eta()-eleMC.momentum().eta(),2) + pow(e_it->p4().phi()-eleMC.momentum().phi(),2);
	   REle=sqrt(REle);
	   if(REle<REleMin){
	     REleMin=REle;
	     nearestEleToMC=*e_it;
	   }
	 }
       }
       //   std::cout<<"Rmin tra MCpos e reco pos: "<<REleMin<<std::endl;
       h1_RMin_->Fill(REleMin);
       if( REleMin<= REleCut_ )h1_eleERecoOverEtrue_->Fill( nearestEleToMC.superCluster()->energy() / eleMC.momentum().e());
     }
   }
   //loop over MC electrons and find the closest MC electron in (eta,phi) phace space - end
   
   
   //loop over MC positrons and find the closest MC positron in (eta,phi) phace space - begin   
   double RPos(0.);
   double RPosMin(1000.);
   reco::Electron nearestPosToMC;
   HepMC::GenParticle posMC;
   
   if(posVec.size()>=1){
     for(std::vector<reco::Electron>::const_iterator p_it = posVec.begin(); p_it !=posVec.end(); p_it++){
       for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	     p != myGenEvent->particles_end(); ++p ) {
	 if ( (*p)->pdg_id() == -11 && (*p)->status()==1 ){
	   posMC=*(*p);
	   
	   RPos = pow(p_it->p4().eta()-posMC.momentum().eta(),2) + pow(p_it->p4().phi()-posMC.momentum().phi(),2);
	   RPos=sqrt(RPos);
	   if(RPos<RPosMin){
	     RPosMin=RPos;
	     nearestPosToMC=*p_it;
	   }
	 }
       }
       
       //   std::cout<<"Rmin tra MCpos e reco pos: "<<RPosMin<<std::endl;
       h1_RMin_->Fill(RPosMin);
       if( RPosMin<=REleCut_ ) h1_eleERecoOverEtrue_->Fill( nearestPosToMC.superCluster()->energy() / posMC.momentum().e());
     }
   }
   //loop over MC positrons and find the closest MC positron in (eta,phi) phace space - end
 
   /*
   
   //OTHER (COMMENTED) ANALYSIS - begin

   // SiStrip Electrons
   edm::Handle<reco::SiStripElectronCollection> electronHandle;
   iEvent.getByLabel(electronProducer_, electronCollection_, electronHandle);

   int numberOfElectrons = 0;
   for (reco::SiStripElectronCollection::const_iterator electronIter = electronHandle->begin();
	electronIter != electronHandle->end();
	++electronIter) {
     edm::LogInfo("")  << "about to get stuff from electroncandidate..." << endl;
     edm::LogInfo("")  << "supercluster energy = " << electronIter->superCluster()->energy() << endl;
     edm::LogInfo("")  << "fit results are phi(r) = " << electronIter->phiAtOrigin() << " + " << electronIter->phiVsRSlope() << "*r" << endl;
     edm::LogInfo("")  << "you get the idea..." << endl;

     numberOfElectrons++;
   }
   numCand_->Fill(numberOfElectrons);

   ///////////////////////////////////////////////////////////////////////////////// Now for tracker hits:
   edm::ESHandle<TrackerGeometry> trackerHandle;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);

   edm::Handle<SiStripRecHit2DCollection> rphiHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);

   edm::Handle<SiStripRecHit2DCollection> stereoHitsHandle;
   iEvent.getByLabel(siHitProducer_, siStereoHitCollection_, stereoHitsHandle);

   // Loop over the detector ids
   const std::vector<DetId> ids = stereoHitsHandle->ids();
   for (std::vector<DetId>::const_iterator id = ids.begin();  id != ids.end();  ++id) {

     // Get the hits on this detector id
     SiStripRecHit2DCollection::range hits = stereoHitsHandle->get(*id);

     // Count the number of hits on this detector id
     unsigned int numberOfHits = 0;
     for (SiStripRecHit2DCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
       numberOfHits++;
     }

     // Only take the hits if there aren't too many
     // (Would it be better to loop only once, fill a temporary list,
     // and copy that if numberOfHits <= maxHitsOnDetId_?)
     if (numberOfHits <= 5) {
       for (SiStripRecHit2DCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	 if (trackerHandle->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB  ||
	     trackerHandle->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB    ) {

	   GlobalPoint position = trackerHandle->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
	   edm::LogInfo("")   << "this stereo hit is at " << position.x() << ", " << position.y() << ", " << position.z() << endl;

	 } // end if this is the right subdetector
       } // end loop over hits
     } // end if this detector id doesn't have too many hits on it
   } // loop over stereo hits

   //OTHER (COMMENTED) ANALYSIS - end

   */
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
ElectronAnalyzer::beginJob(const edm::EventSetup&)
{

// go to *OUR* rootfile and book histograms
rootFile_->cd();

h1_nEleReco_ = new TH1F("nEleReco","Number of reco electrons",10,-0.5,10.5);
 h1_nEleReco_->SetXTitle("nEleReco");
 h1_nEleReco_->SetYTitle("events");
h1_recoEleEnergy_ = new TH1F("recoEleEnergy","EleEnergy from SC",400,0.,200.);
 h1_recoEleEnergy_->SetXTitle("eleSCEnergy(GeV)");
 h1_recoEleEnergy_->SetYTitle("events");
h1_recoElePt_ = new TH1F("recoElePt","p_{T} of reco electrons",200,0.,200.);
 h1_recoElePt_->SetXTitle("p_{T}(GeV/c)");
 h1_recoElePt_->SetYTitle("events");
h1_recoEleEta_ = new TH1F("recoEleEta","Eta of reco electrons",100,-4.,4.);
 h1_recoEleEta_->SetXTitle("#eta");
 h1_recoEleEta_->SetYTitle("events");
h1_recoElePhi_ = new TH1F("recoElePhi","Phi of reco electrons",100,-4.,4.);
 h1_recoElePhi_->SetXTitle("#phi");
 h1_recoElePhi_->SetYTitle("events");
h1_RMin_=new TH1F("h1_RMin","Distance of MC electron from nearest RecoElectron",500,0.,10.);
 h1_RMin_->SetXTitle("RMin");
 h1_RMin_->SetYTitle("events");
h1_eleERecoOverEtrue_= new TH1F("eleERecoOverEtrue","Ereco/Etrue for MC matched Z electrons",250,0.,4.);
 h1_eleERecoOverEtrue_->SetXTitle("Ereco/Etrue");
 h1_eleERecoOverEtrue_->SetYTitle("events");
}


// ------------ method called once each job just after ending the event loop  ------------
void 
ElectronAnalyzer::endJob() {

  h1_nEleReco_->Write();
  h1_recoEleEnergy_->Write();
  h1_recoElePt_->Write();
  h1_recoEleEta_->Write();
  h1_recoElePhi_->Write();
  h1_RMin_->Write();
  h1_eleERecoOverEtrue_->Write();

  rootFile_->Close();

}

