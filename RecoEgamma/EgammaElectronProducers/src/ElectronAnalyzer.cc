// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     ElectronAnalyzer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:49:38 EDT 2006
// $Id: ElectronAnalyzer.cc,v 1.7 2006/09/20 12:18:42 rahatlou Exp $
//

// system include files
#include <memory>


// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

// for Si hits
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronAnalyzer::ElectronAnalyzer(const edm::ParameterSet& iConfig) :
minElePt_(iConfig.getParameter<double>("minElectronPt")),
  REleCut_(iConfig.getParameter<double>("drElectronMCMatchCut")),
  electronProducer_(iConfig.getParameter<std::string>("electronProducer")),
  mctruthProducer_(iConfig.getParameter<std::string>("mctruthProducer"))

{
   //now do what ever initialization is needed
   fileName_ = iConfig.getParameter<std::string>("fileName");

   file_ = new TFile(fileName_.c_str(), "RECREATE");
   numCand_ = new TH1F("numCandidates", "Number of candidates found", 10, -0.5, 9.5);

   mctruthProducer_ = iConfig.getParameter<std::string>("mctruthProducer");
   mctruthCollection_ = iConfig.getParameter<std::string>("mctruthCollection");

   superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
   superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");

   electronProducer_ = iConfig.getParameter<std::string>("electronProducer");
   electronCollection_ = iConfig.getParameter<std::string>("electronCollection");

   siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
   siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
   siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");



}

// ElectronAnalyzer::ElectronAnalyzer(const ElectronAnalyzer& rhs)
// {
//    // do actual copying here;
// }

ElectronAnalyzer::~ElectronAnalyzer()
{
}

//
// assignment operators
//
// const ElectronAnalyzer& ElectronAnalyzer::operator=(const ElectronAnalyzer& rhs)
// {
//   //An exception safe implementation is
//   ElectronAnalyzer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;  // so you can say "cout" and "endl"
   using namespace edm; // needed for FW stuff
   edm::Handle<reco::SuperClusterCollection> clusterHandle;
   iEvent.getByLabel(superClusterProducer_, superClusterCollection_, clusterHandle);

   for (reco::SuperClusterCollection::const_iterator clusterIter = clusterHandle->begin();
                                                     clusterIter != clusterHandle->end();
                                                     ++clusterIter) {
      double energy = clusterIter->energy();
      math::XYZPoint position = clusterIter->position();

      edm::LogInfo("")  << "supercluster " << energy << " GeV, position " << position << " cm" << endl;
   }

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


    //SUPERCLUSTERS - BEGIN
   
   Handle<reco::SuperClusterCollection> mySC;
   iEvent.getByLabel("hybridSuperClusters", mySC); 
   
   for(reco::SuperClusterCollection::const_iterator scIt = mySC->begin();
       scIt != mySC->end(); scIt++){
     for(reco::SuperClusterCollection::const_iterator scIt2 = scIt+1;
	 scIt2 != mySC->end(); scIt2++){
	 }
   }
   //SUPERCLUSTERS - END
   
   
   //GET GENERATOR EVENT - BEGIN

   Handle< HepMCProduct > hepProd ;
   iEvent.getByLabel( mctruthProducer_.c_str(), hepProd ) ;
   const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();

   //GET GENERATOR EVENT - END
   
   
   //LOOP OVER ELECTRONS - BEGIN
   Handle<reco::ElectronCollection> myEle;
   iEvent.getByLabel(electronProducer_.c_str(), myEle); 
   
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






} // end of ::analyze()


// ------------ method called once each job just before starting event loop  ------------
void 
ElectronAnalyzer::beginJob(const edm::EventSetup&)
{

// go to *OUR* rootfile and book histograms
file_->cd();

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
void ElectronAnalyzer::endJob() {

  h1_nEleReco_->Write();
  h1_recoEleEnergy_->Write();
  h1_recoElePt_->Write();
  h1_recoEleEta_->Write();
  h1_recoElePhi_->Write();
  h1_RMin_->Write();
  h1_eleERecoOverEtrue_->Write();

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   file_->Write();
   file_->Close();

}

//
// const member functions
//

//
// static member functions
//
