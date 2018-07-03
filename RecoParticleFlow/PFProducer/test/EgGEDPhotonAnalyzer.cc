// -*- C++ -*-
//
// Package:    EgGEDPhotonAnalyzer
// Class:      EgGEDPhotonAnalyzer
// 
/**\class EgGEDPhotonAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Benedetti



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" 
#include "RecoParticleFlow/PFProducer/interface/Utils.h"

#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
//
// class decleration
//

using namespace edm;
using namespace reco;
using namespace std;
class EgGEDPhotonAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EgGEDPhotonAnalyzer(const edm::ParameterSet&);
      ~EgGEDPhotonAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  
  ParameterSet conf_;


  unsigned int ev;
      // ----------member data ---------------------------
  

  TH1F *h_etamc_ele,*h_etaec_ele,*h_etaeg_ele,*h_etaged_ele;
  TH1F *h_ptmc_ele,*h_ptec_ele,*h_pteg_ele,*h_ptged_ele;


  TH2F *etaphi_nonreco;


  TH1F *eg_EEcalEtrue_1,*eg_EEcalEtrue_2,*eg_EEcalEtrue_3,*eg_EEcalEtrue_4,*eg_EEcalEtrue_5;
  TH1F *pf_EEcalEtrue_1,*pf_EEcalEtrue_2,*pf_EEcalEtrue_3,*pf_EEcalEtrue_4,*pf_EEcalEtrue_5;
  TH1F *ged_EEcalEtrue_1,*ged_EEcalEtrue_2,*ged_EEcalEtrue_3,*ged_EEcalEtrue_4,*ged_EEcalEtrue_5;

  TH1F *eg_hoe,*pf_hoe,*ged_hoe; 

  TH1F *eg_sigmaetaeta_eb,*pf_sigmaetaeta_eb,*ged_sigmaetaeta_eb; 
  TH1F *eg_sigmaetaeta_ee,*pf_sigmaetaeta_ee,*ged_sigmaetaeta_ee; 
  
  TH1F *eg_r9_eb,*pf_r9_eb,*ged_r9_eb; 
  TH1F *eg_r9_ee,*pf_r9_ee,*ged_r9_ee; 


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
EgGEDPhotonAnalyzer::EgGEDPhotonAnalyzer(const edm::ParameterSet& iConfig):
  conf_(iConfig)

{

  
  edm::Service<TFileService> fs;

  // Efficiency plots

  h_etamc_ele = fs->make<TH1F>("h_etamc_ele"," ",50,-2.5,2.5);
  h_etaec_ele = fs->make<TH1F>("h_etaec_ele"," ",50,-2.5,2.5);
  h_etaeg_ele = fs->make<TH1F>("h_etaeg_ele"," ",50,-2.5,2.5);
  h_etaged_ele = fs->make<TH1F>("h_etaged_ele"," ",50,-2.5,2.5);




  h_ptmc_ele = fs->make<TH1F>("h_ptmc_ele"," ",50,0,100.);
  h_ptec_ele = fs->make<TH1F>("h_ptec_ele"," ",50,0,100.);
  h_pteg_ele = fs->make<TH1F>("h_pteg_ele"," ",50,0,100.);

  h_ptged_ele = fs->make<TH1F>("h_ptged_ele"," ",50,0,100.);

  pf_hoe = fs->make<TH1F>("pf_hoe"," ",100,0,1.);
  eg_hoe = fs->make<TH1F>("eg_hoe"," ",100,0,1.);
  ged_hoe = fs->make<TH1F>("ged_hoe"," ",100,0,1.);


  pf_sigmaetaeta_eb = fs->make<TH1F>("pf_sigmaetaeta_eb"," ",100,0.,0.03);
  eg_sigmaetaeta_eb = fs->make<TH1F>("eg_sigmaetaeta_eb"," ",100,0.,0.03);
  ged_sigmaetaeta_eb = fs->make<TH1F>("ged_sigmaetaeta_eb"," ",100,0.,0.03);

  pf_sigmaetaeta_ee = fs->make<TH1F>("pf_sigmaetaeta_ee"," ",100,0.,0.1);
  eg_sigmaetaeta_ee = fs->make<TH1F>("eg_sigmaetaeta_ee"," ",100,0.,0.1);
  ged_sigmaetaeta_ee = fs->make<TH1F>("ged_sigmaetaeta_ee"," ",100,0.,0.1);


  pf_r9_eb = fs->make<TH1F>("pf_r9_eb"," ",100,0.,1.0);
  eg_r9_eb = fs->make<TH1F>("eg_r9_eb"," ",100,0.,1.0);
  ged_r9_eb = fs->make<TH1F>("ged_r9_eb"," ",100,0.,1.0);

  pf_r9_ee = fs->make<TH1F>("pf_r9_ee"," ",100,0.,1.0);
  eg_r9_ee = fs->make<TH1F>("eg_r9_ee"," ",100,0.,1.0);
  ged_r9_ee = fs->make<TH1F>("ged_r9_ee"," ",100,0.,1.0);


  // Ecal map
  etaphi_nonreco  = fs->make<TH2F>("etaphi_nonreco"," ",50,-2.5,2.5,50,-3.2,3.2);




  // Energies
  eg_EEcalEtrue_1 =  fs->make<TH1F>("eg_EEcalEtrue_1","  ",50,0.,2.);
  eg_EEcalEtrue_2 =  fs->make<TH1F>("eg_EEcalEtrue_2","  ",50,0.,2.);
  eg_EEcalEtrue_3 =  fs->make<TH1F>("eg_EEcalEtrue_3","  ",50,0.,2.);
  eg_EEcalEtrue_4 =  fs->make<TH1F>("eg_EEcalEtrue_4","  ",50,0.,2.);
  eg_EEcalEtrue_5 =  fs->make<TH1F>("eg_EEcalEtrue_5","  ",50,0.,2.);

  pf_EEcalEtrue_1 =  fs->make<TH1F>("pf_EEcalEtrue_1","  ",50,0.,2.);
  pf_EEcalEtrue_2 =  fs->make<TH1F>("pf_EEcalEtrue_2","  ",50,0.,2.);
  pf_EEcalEtrue_3 =  fs->make<TH1F>("pf_EEcalEtrue_3","  ",50,0.,2.);
  pf_EEcalEtrue_4 =  fs->make<TH1F>("pf_EEcalEtrue_4","  ",50,0.,2.);
  pf_EEcalEtrue_5 =  fs->make<TH1F>("pf_EEcalEtrue_5","  ",50,0.,2.);


  ged_EEcalEtrue_1 =  fs->make<TH1F>("ged_EEcalEtrue_1","  ",50,0.,2.);
  ged_EEcalEtrue_2 =  fs->make<TH1F>("ged_EEcalEtrue_2","  ",50,0.,2.);
  ged_EEcalEtrue_3 =  fs->make<TH1F>("ged_EEcalEtrue_3","  ",50,0.,2.);
  ged_EEcalEtrue_4 =  fs->make<TH1F>("ged_EEcalEtrue_4","  ",50,0.,2.);
  ged_EEcalEtrue_5 =  fs->make<TH1F>("ged_EEcalEtrue_5","  ",50,0.,2.);


}


EgGEDPhotonAnalyzer::~EgGEDPhotonAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EgGEDPhotonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  // Candidate info
  Handle<reco::PFCandidateCollection> collection;
  InputTag label("particleFlow");  // <- Special electron coll. 
  iEvent.getByLabel(label, collection);
  std::vector<reco::PFCandidate> candidates = (*collection.product());


  InputTag recoPhotonLabel(string("photons"));
  Handle<PhotonCollection> theRecoPhotonCollection;
  iEvent.getByLabel(recoPhotonLabel,theRecoPhotonCollection);
  const PhotonCollection theRecoPh = *(theRecoPhotonCollection.product());


  InputTag  MCTruthCollection(string("VtxSmeared"));
  edm::Handle<edm::HepMCProduct> pMCTruth;
  iEvent.getByLabel(MCTruthCollection,pMCTruth);
  const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();



  InputTag gedPhotonLabel(string("gedPhotons"));
  Handle<PhotonCollection> theGedPhotonCollection;
  iEvent.getByLabel(gedPhotonLabel,theGedPhotonCollection);
  const PhotonCollection theGedPh = *(theGedPhotonCollection.product());




  bool debug = true;


  ev++;

  if(debug)
    cout << "************************* New Event:: " << ev << " *************************" << endl;

  // Validation from generator events 

  for(HepMC::GenEvent::particle_const_iterator cP = genEvent->particles_begin(); 
      cP != genEvent->particles_end(); cP++ ) {

    float etamc= (*cP)->momentum().eta();
    float phimc= (*cP)->momentum().phi();
    float ptmc = (*cP)->momentum().perp();
    float Emc = (*cP)->momentum().e();


    if(abs((*cP)->pdg_id())==22 && (*cP)->status()==1 
       && (*cP)->momentum().perp() > 8. && 
       fabs((*cP)->momentum().eta()) < 2.5 ){
   
      h_etamc_ele->Fill(etamc);
      h_ptmc_ele->Fill(ptmc);
      
      float MindR = 1000.;
 


      if(debug)
	cout << " MC particle:  pt " << ptmc << " eta,phi " <<  etamc << ", " << phimc << endl;



      std::vector<reco::PFCandidate>::iterator it;
      for ( it = candidates.begin(); it != candidates.end(); ++it )   {
	reco::PFCandidate::ParticleType type = (*it).particleId();

	if ( type == reco::PFCandidate::gamma )	{ 
	  
	  reco::PhotonRef phref = (*it).photonRef(); 
	  float eta =  (*it).eta();
	  float phi =  (*it).phi();

	  float deta = etamc - eta;
	  float dphi = Utils::mpi_pi(phimc - phi);
	  float dR = sqrt(deta*deta + dphi*dphi);

	  if(dR < 0.1){
	    MindR = dR;

	    if(phref.isNonnull()) {
	      
	      float SCPF = phref->energy();
	      float EpfEtrue = SCPF/Emc;
	      if (fabs(etamc) < 0.5) {
		pf_EEcalEtrue_1->Fill(EpfEtrue);
	      }
	      if (fabs(etamc) >= 0.5 && fabs(etamc) < 1.0) {	
		pf_EEcalEtrue_2->Fill(EpfEtrue);
	      }
	      if (fabs(etamc) >= 1.0 && fabs(etamc) < 1.5) {	
		pf_EEcalEtrue_3->Fill(EpfEtrue);
	      }
	      if (fabs(etamc) >= 1.5 && fabs(etamc) < 2.0) {
		pf_EEcalEtrue_4->Fill(EpfEtrue);
	      }
	      if (fabs(etamc) >= 2.0 && fabs(etamc) < 2.5) {
		pf_EEcalEtrue_5->Fill(EpfEtrue);
	      }

	      pf_hoe->Fill(phref->hadronicOverEm());
	      if(fabs(etamc) < 1.479) {
		pf_r9_eb->Fill(phref->r9());
		pf_sigmaetaeta_eb->Fill(phref->sigmaEtaEta());
	      }
	      else if(fabs(etamc) > 1.479 && fabs(etamc) < 2.5) {
		pf_r9_ee->Fill(phref->r9());
		pf_sigmaetaeta_ee->Fill(phref->sigmaEtaEta());
	      }


	    }

	    if(debug)
	      cout << " PF photon matched:  pt " << (*it).pt()  << " eta,phi " <<  eta << ", " << phi << endl;
	    // all for the moment
	  }
	} // End PFCandidates Electron Selection
      } // End Loop PFCandidates
      
      if (MindR < 0.1) {      

    	h_etaec_ele->Fill(etamc);
	h_ptec_ele->Fill(ptmc);
	



      }
      else {
	etaphi_nonreco->Fill(etamc,phimc);

      }

      float MindREG =1000;
      for (uint j=0; j<theRecoPh.size();j++) {

	float etareco = theRecoPh[j].eta();
	float phireco = theRecoPh[j].phi();



	float deta = etamc - etareco;
	float dphi = Utils::mpi_pi(phimc - phireco);
	float dR = sqrt(deta*deta + dphi*dphi);

	float SCEnergy = (theRecoPh[j]).energy();
	float ErecoEtrue = SCEnergy/Emc;



	if(dR < 0.1){
	  if(debug)
	    cout << " EG ele matched: pt " << theRecoPh[j].pt() << " eta,phi " <<  etareco << ", " << phireco <<  endl;
	

	  if (fabs(etamc) < 0.5) {
	    eg_EEcalEtrue_1->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 0.5 && fabs(etamc) < 1.0) {	
	    eg_EEcalEtrue_2->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 1.0 && fabs(etamc) < 1.5) {	
	    eg_EEcalEtrue_3->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 1.5 && fabs(etamc) < 2.0) {
	    eg_EEcalEtrue_4->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 2.0 && fabs(etamc) < 2.5) {
	    eg_EEcalEtrue_5->Fill(ErecoEtrue);
	  }

	
	  eg_hoe->Fill(theRecoPh[j].hadronicOverEm());  
	  if(fabs(etamc) < 1.479) {
	    eg_r9_eb->Fill(theRecoPh[j].r9());
	    eg_sigmaetaeta_eb->Fill(theRecoPh[j].sigmaEtaEta());
	  }
	  else if(fabs(etamc) > 1.479 && fabs(etamc) < 2.5) {
	    eg_r9_ee->Fill(theRecoPh[j].r9());
	    eg_sigmaetaeta_ee->Fill(theRecoPh[j].sigmaEtaEta());
	  }


	  MindREG = dR;
	}
      }
      if(MindREG < 0.1) {
	h_etaeg_ele->Fill(etamc);
	h_pteg_ele->Fill(ptmc);
      }




      //Add loop on the GED electrons
      
      
      float MindRGedEg =1000;
       
      for (uint j=0; j<theGedPh.size();j++) {
	reco::GsfTrackRef egGsfTrackRef = (theGedPh[j]).gsfTrack();
	float etareco = theGedPh[j].eta();
	float phireco = theGedPh[j].phi();
	
	
	float deta = etamc - etareco;
	float dphi = Utils::mpi_pi(phimc - phireco);
	float dR = sqrt(deta*deta + dphi*dphi);
	
	float SCEnergy = (theGedPh[j]).energy();
	float ErecoEtrue = SCEnergy/Emc;



	if(dR < 0.1){
	  if(debug)
	    cout << " GED ele matched: pt " << theGedPh[j].pt() << " eta,phi " <<  etareco << ", " << phireco << endl;
	  
	  if (fabs(etamc) < 0.5) {
	    ged_EEcalEtrue_1->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 0.5 && fabs(etamc) < 1.0) {	
	    ged_EEcalEtrue_2->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 1.0 && fabs(etamc) < 1.5) {	
	    ged_EEcalEtrue_3->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 1.5 && fabs(etamc) < 2.0) {
	    ged_EEcalEtrue_4->Fill(ErecoEtrue);
	  }
	  if (fabs(etamc) >= 2.0 && fabs(etamc) < 2.5) {
	    ged_EEcalEtrue_5->Fill(ErecoEtrue);
	  }


	  ged_hoe->Fill(theGedPh[j].hadronicOverEm());
	  if(fabs(etamc) < 1.479) {
	    ged_r9_eb->Fill(theGedPh[j].r9());
	    ged_sigmaetaeta_eb->Fill(theGedPh[j].sigmaEtaEta());
	  }
	  else if(fabs(etamc) > 1.479 && fabs(etamc) < 2.5) {
	    ged_r9_ee->Fill(theGedPh[j].r9());
	    ged_sigmaetaeta_ee->Fill(theGedPh[j].sigmaEtaEta());
	  }


	  MindRGedEg = dR;

	}
      }
      if(MindRGedEg < 0.1) {
	h_etaged_ele->Fill(etamc);
	h_ptged_ele->Fill(ptmc);
      }//End Loop Generator Particles  
      
      


    } //End IF Generator Particles 
  
  
  } //End Loop Generator Particles 

}
// ------------ method called once each job just before starting event loop  ------------
void 
EgGEDPhotonAnalyzer::beginJob(const edm::EventSetup&)
{

  ev = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EgGEDPhotonAnalyzer::endJob() {
  cout << " endJob:: #events " << ev << endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(EgGEDPhotonAnalyzer);
