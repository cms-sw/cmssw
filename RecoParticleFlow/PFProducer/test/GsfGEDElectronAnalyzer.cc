// -*- C++ -*-
//
// Package:    GsfGEDElectronAnalyzer
// Class:      GsfGEDElectronAnalyzer
// 
/**\class GsfGEDElectronAnalyzer

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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" 
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


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
class GsfGEDElectronAnalyzer : public edm::EDAnalyzer {
   public:
      explicit GsfGEDElectronAnalyzer(const edm::ParameterSet&);
      ~GsfGEDElectronAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  
  ParameterSet conf_;


  unsigned int ev;
      // ----------member data ---------------------------
  

  TH1F *h_etamc_ele,*h_etaec_ele,*h_etaecTrDr_ele,*h_etaecEcDr_ele,
    *h_etaeg_ele,*h_etaegTrDr_ele,*h_etaegEcDr_ele,
    *h_etaged_ele,*h_etagedTrDr_ele,*h_etagedEcDr_ele;





  TH1F  *h_ptmc_ele,*h_ptec_ele,*h_ptecEcDr_ele,*h_ptecTrDr_ele,
    *h_pteg_ele,*h_ptegTrDr_ele,*h_ptegEcDr_ele,
    *h_ptged_ele,*h_ptgedTrDr_ele,*h_ptgedEcDr_ele;


  TH1F *h_mva_ele;
  TH2F *etaphi_nonreco;


  TH1F *eg_EEcalEtrue_1,*eg_EEcalEtrue_2,*eg_EEcalEtrue_3,*eg_EEcalEtrue_4,*eg_EEcalEtrue_5;
  TH1F *pf_EEcalEtrue_1,*pf_EEcalEtrue_2,*pf_EEcalEtrue_3,*pf_EEcalEtrue_4,*pf_EEcalEtrue_5;
  TH1F *ged_EEcalEtrue_1,*ged_EEcalEtrue_2,*ged_EEcalEtrue_3,*ged_EEcalEtrue_4,*ged_EEcalEtrue_5;

  TH1F *eg_e1x5_all,*eg_e1x5_eb,*eg_e1x5_ee,*ged_e1x5_all,*ged_e1x5_eb,*ged_e1x5_ee,*pf_e1x5_all,*pf_e1x5_eb,*pf_e1x5_ee;

  TH1F *eg_sihih_all,*eg_sihih_eb,*eg_sihih_ee,*ged_sihih_all,*ged_sihih_eb,*ged_sihih_ee,*pf_sihih_all,*pf_sihih_eb,*pf_sihih_ee;

  TH1F *eg_r9_all,*eg_r9_eb,*eg_r9_ee,*ged_r9_all,*ged_r9_eb,*ged_r9_ee,*pf_r9_all,*pf_r9_eb,*pf_r9_ee;

  TH1F *eg_scshh_all,*eg_scshh_eb,*eg_scshh_ee,*ged_scshh_all,*ged_scshh_eb,*ged_scshh_ee,*pf_scshh_all,*pf_scshh_eb,*pf_scshh_ee;

  TH1F *eg_scsff_all,*eg_scsff_eb,*eg_scsff_ee,*ged_scsff_all,*ged_scsff_eb,*ged_scsff_ee,*pf_scsff_all,*pf_scsff_eb,*pf_scsff_ee;


  TH1F *pf_MET_reco ,*pf_MET_rereco;
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
GsfGEDElectronAnalyzer::GsfGEDElectronAnalyzer(const edm::ParameterSet& iConfig):
  conf_(iConfig)

{

  
  edm::Service<TFileService> fs;

  // Efficiency plots

  h_etamc_ele = fs->make<TH1F>("h_etamc_ele"," ",50,-2.5,2.5);
  h_etaec_ele = fs->make<TH1F>("h_etaec_ele"," ",50,-2.5,2.5);
  h_etaecTrDr_ele = fs->make<TH1F>("h_etaecTrDr_ele"," ",50,-2.5,2.5);
  h_etaecEcDr_ele = fs->make<TH1F>("h_etaecEcDr_ele"," ",50,-2.5,2.5);
  h_etaeg_ele = fs->make<TH1F>("h_etaeg_ele"," ",50,-2.5,2.5);
  h_etaegTrDr_ele = fs->make<TH1F>("h_etaegTrDr_ele"," ",50,-2.5,2.5);
  h_etaegEcDr_ele = fs->make<TH1F>("h_etaegEcDr_ele"," ",50,-2.5,2.5);
  h_etaged_ele = fs->make<TH1F>("h_etaged_ele"," ",50,-2.5,2.5);
  h_etagedTrDr_ele = fs->make<TH1F>("h_etagedTrDr_ele"," ",50,-2.5,2.5);
  h_etagedEcDr_ele = fs->make<TH1F>("h_etagedEcDr_ele"," ",50,-2.5,2.5);




  h_ptmc_ele = fs->make<TH1F>("h_ptmc_ele"," ",50,0,100.);
  h_ptec_ele = fs->make<TH1F>("h_ptec_ele"," ",50,0,100.);
  h_ptecTrDr_ele = fs->make<TH1F>("h_ptecTrDr_ele"," ",50,0,100.);
  h_ptecEcDr_ele = fs->make<TH1F>("h_ptecEcDr_ele"," ",50,0,100.);
  h_pteg_ele = fs->make<TH1F>("h_pteg_ele"," ",50,0,100.);
  h_ptegTrDr_ele = fs->make<TH1F>("h_ptegTrDr_ele"," ",50,0,100.);
  h_ptegEcDr_ele = fs->make<TH1F>("h_ptegEcDr_ele"," ",50,0,100.);
  h_ptged_ele = fs->make<TH1F>("h_ptged_ele"," ",50,0,100.);
  h_ptgedTrDr_ele = fs->make<TH1F>("h_ptgedTrDr_ele"," ",50,0,100.);
  h_ptgedEcDr_ele = fs->make<TH1F>("h_ptgedEcDr_ele"," ",50,0,100.);

  
  h_mva_ele  = fs->make<TH1F>("h_mva_ele"," ",50,-1.1,1.1);

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
  
  //shower shapes
  eg_e1x5_all  = fs->make<TH1F>("eg_e1x5_all","  ",60,0.,300.);
  eg_e1x5_eb   = fs->make<TH1F>("eg_e1x5_eb","  ",60,0.,300.);
  eg_e1x5_ee   = fs->make<TH1F>("eg_e1x5_ee","  ",60,0.,300.);

  ged_e1x5_all = fs->make<TH1F>("ged_e1x5_all","  ",60,0.,300.);
  ged_e1x5_eb  = fs->make<TH1F>("ged_e1x5_eb","  ",60,0.,300.);
  ged_e1x5_ee  = fs->make<TH1F>("ged_e1x5_ee","  ",60,0.,300.);

  pf_e1x5_all = fs->make<TH1F>("pf_e1x5_all","  ",60,0.,300.);
  pf_e1x5_eb  = fs->make<TH1F>("pf_e1x5_eb","  ",60,0.,300.);
  pf_e1x5_ee  = fs->make<TH1F>("pf_e1x5_ee","  ",60,0.,300.);

  eg_sihih_all  = fs->make<TH1F>("eg_sihih_all","  ",100,0.0,0.05);
  eg_sihih_eb   = fs->make<TH1F>("eg_sihih_eb","  ",100,0.0,0.05);
  eg_sihih_ee   = fs->make<TH1F>("eg_sihih_ee","  ",100,0.0,0.05);

  ged_sihih_all = fs->make<TH1F>("ged_sihih_all","  ",100,0.0,0.05);
  ged_sihih_eb  = fs->make<TH1F>("ged_sihih_eb","  ",100,0.0,0.05);
  ged_sihih_ee  = fs->make<TH1F>("ged_sihih_ee","  ",100,0.0,0.05);

  pf_sihih_all = fs->make<TH1F>("pf_sihih_all","  ",100,0.0,0.05);
  pf_sihih_eb  = fs->make<TH1F>("pf_sihih_eb","  ",100,0.0,0.05);
  pf_sihih_ee  = fs->make<TH1F>("pf_sihih_ee","  ",100,0.0,0.05);

  eg_r9_all  = fs->make<TH1F>("eg_r9_all","  ",200,0.0,2.0);
  eg_r9_eb   = fs->make<TH1F>("eg_r9_eb","  ",200,0.0,2.0);
  eg_r9_ee   = fs->make<TH1F>("eg_r9_ee","  ",200,0.0,2.0);

  ged_r9_all = fs->make<TH1F>("ged_r9_all","  ",200,0.0,2.0);
  ged_r9_eb  = fs->make<TH1F>("ged_r9_eb","  ",200,0.0,2.0);
  ged_r9_ee  = fs->make<TH1F>("ged_r9_ee","  ",200,0.0,2.0);

  pf_r9_all = fs->make<TH1F>("pf_r9_all","  ",200,0.0,2.0);
  pf_r9_eb  = fs->make<TH1F>("pf_r9_eb","  ",200,0.0,2.0);
  pf_r9_ee  = fs->make<TH1F>("pf_r9_ee","  ",200,0.0,2.0);

  eg_scshh_all  = fs->make<TH1F>("eg_scshh_all","  ",100,0.0,0.06);
  eg_scshh_eb   = fs->make<TH1F>("eg_scshh_eb","  ",100,0.0,0.06);
  eg_scshh_ee   = fs->make<TH1F>("eg_scshh_ee","  ",100,0.0,0.06);

  ged_scshh_all = fs->make<TH1F>("ged_scshh_all","  ",100,0.0,0.06);
  ged_scshh_eb  = fs->make<TH1F>("ged_scshh_eb","  ",100,0.0,0.06);
  ged_scshh_ee  = fs->make<TH1F>("ged_scshh_ee","  ",100,0.0,0.06);

  pf_scshh_all = fs->make<TH1F>("pf_scshh_all","  ",100,0.0,0.06);
  pf_scshh_eb  = fs->make<TH1F>("pf_scshh_eb","  ",100,0.0,0.06);
  pf_scshh_ee  = fs->make<TH1F>("pf_scshh_ee","  ",100,0.0,0.06);

  eg_scsff_all  = fs->make<TH1F>("eg_scsff_all","  ",150,0.0,0.15);
  eg_scsff_eb   = fs->make<TH1F>("eg_scsff_eb","  ",150,0.0,0.15);
  eg_scsff_ee   = fs->make<TH1F>("eg_scsff_ee","  ",150,0.0,0.15);

  ged_scsff_all = fs->make<TH1F>("ged_scsff_all","  ",150,0.0,0.15);
  ged_scsff_eb  = fs->make<TH1F>("ged_scsff_eb","  ",150,0.0,0.15);
  ged_scsff_ee  = fs->make<TH1F>("ged_scsff_ee","  ",150,0.0,0.15);

  pf_scsff_all = fs->make<TH1F>("pf_scsff_all","  ",150,0.0,0.15);
  pf_scsff_eb  = fs->make<TH1F>("pf_scsff_eb","  ",150,0.0,0.15);
  pf_scsff_ee  = fs->make<TH1F>("pf_scsff_ee","  ",150,0.0,0.15);
 
  // MET 
  pf_MET_reco = fs->make<TH1F>("pf_MET_reco","  ",10,0,1000);
  pf_MET_rereco = fs->make<TH1F>("pf_MET_rereco","  ",10,0,1000);
 
}


GsfGEDElectronAnalyzer::~GsfGEDElectronAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
GsfGEDElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  // Candidate info
  Handle<reco::PFCandidateCollection> collection;
  InputTag label("particleFlow::reRECO");  // <- Special electron coll. 
  iEvent.getByLabel(label, collection);
  std::vector<reco::PFCandidate> candidates = (*collection.product());


  InputTag egammaLabel(string("gsfElectrons"));
  Handle<GsfElectronCollection> theGsfEleCollection;
  iEvent.getByLabel(egammaLabel,theGsfEleCollection);
  const GsfElectronCollection theGsfEle = *(theGsfEleCollection.product());


  InputTag  MCTruthCollection(string("genParticles"));
  edm::Handle<std::vector<reco::GenParticle> > pMCTruth;
  iEvent.getByLabel(MCTruthCollection,pMCTruth);


  InputTag gedEleLabel(string("gedGsfElectrons::reRECO"));
  Handle<GsfElectronCollection> theGedEleCollection;
  iEvent.getByLabel(gedEleLabel,theGedEleCollection);
  const GsfElectronCollection theGedEle = *(theGedEleCollection.product());


  //get and plot the reco met
  InputTag pfMETRecoLabel("pfMet::RECO");
  Handle<std::vector<reco::PFMET> > recoMet;
  iEvent.getByLabel(pfMETRecoLabel,recoMet);
  pf_MET_reco->Fill(recoMet->at(0).et());
  
  //get and plot the rereco met
  InputTag pfMETReRecoLabel("pfMet::reRECO");
  Handle<std::vector<reco::PFMET> > rerecoMet;
  iEvent.getByLabel(pfMETReRecoLabel,rerecoMet);
  pf_MET_rereco->Fill(rerecoMet->at(0).et());


  bool debug = true;


  ev++;

  if(debug)
    cout << "************************* New Event:: " << ev << " *************************" << endl;

  // Validation from generator events 

  for(std::vector<reco::GenParticle>::const_iterator cP = pMCTruth->begin(); 
      cP != pMCTruth->end(); cP++ ) {

    float etamc= cP->eta();
    float phimc= cP->phi();
    float ptmc = cP->pt();
    float Emc = cP->energy();


    if(abs(cP->pdgId())==11 && cP->status()==1 
       && cP->pt() > 2. && 
       fabs(cP->eta()) < 2.5 ){
   
      h_etamc_ele->Fill(etamc);
      h_ptmc_ele->Fill(ptmc);
      
      float MindR = 1000.;
      float mvaCutEle = -1.;
      bool isPfTrDr = false;
      bool isPfEcDr = false;



      if(debug)
	cout << " MC particle:  pt " << ptmc << " eta,phi " <<  etamc << ", " << phimc << endl;



      std::vector<reco::PFCandidate>::iterator it;
      for ( it = candidates.begin(); it != candidates.end(); ++it )   {
	reco::PFCandidate::ParticleType type = (*it).particleId();

	if ( type == reco::PFCandidate::e )	{ 
	  
	 
	  float eta =  (*it).eta();
	  float phi =  (*it).phi();
	  float pfmva = (*it).mva_e_pi();


	  reco::GsfTrackRef refGsf = (*it).gsfTrackRef();
	  //ElectronSeedRef seedRef= refGsf->extra()->seedRef().castTo<ElectronSeedRef>();


	  float deta = etamc - eta;
	  float dphi = Utils::mpi_pi(phimc - phi);
	  float dR = sqrt(deta*deta + dphi*dphi);

	  if(dR < 0.05){
	    MindR = dR;
	    mvaCutEle = (*it).mva_e_pi();
	    /*
	    if(seedRef->isEcalDriven())
	      isPfEcDr = true;
	    if(seedRef->isTrackerDriven())
	      isPfTrDr = true;
	    */

	    if(debug)
	      cout << " PF ele matched:  pt " << (*it).pt()  << " (" << (*it).ecalEnergy()/std::cosh(eta) << ") "<< " eta,phi " <<  eta << ", " << phi << " pfmva " << pfmva << endl;
	    // all for the moment
	  }
	} // End PFCandidates Electron Selection
      } // End Loop PFCandidates
      
      if (MindR < 0.05) {      
	h_mva_ele->Fill(mvaCutEle);
    	h_etaec_ele->Fill(etamc);
	h_ptec_ele->Fill(ptmc);

	if(isPfEcDr) {
	  h_etaecEcDr_ele->Fill(etamc);
	  h_ptecEcDr_ele->Fill(ptmc);
	}
	if(isPfTrDr) {
	  h_etaecTrDr_ele->Fill(etamc);
	  h_ptecTrDr_ele->Fill(ptmc);
	}
      }
      else {
	etaphi_nonreco->Fill(etamc,phimc);

      }

      float MindREG =1000;
      bool isEgTrDr = false;
      bool isEgEcDr = false;
      
      for (uint j=0; j<theGsfEle.size();j++) {
	reco::GsfTrackRef egGsfTrackRef = (theGsfEle[j]).gsfTrack();
	float etareco = theGsfEle[j].eta();
	float phireco = theGsfEle[j].phi();

	float pfmva =  theGsfEle[j].mva_e_pi();

	reco::GsfTrackRef refGsf =  theGsfEle[j].gsfTrack();
	//ElectronSeedRef seedRef= refGsf->extra()->seedRef().castTo<ElectronSeedRef>();


	float deta = etamc - etareco;
	float dphi = Utils::mpi_pi(phimc - phireco);
	float dR = sqrt(deta*deta + dphi*dphi);

	float SCEnergy = (theGsfEle[j]).superCluster()->energy();
	float ErecoEtrue = SCEnergy/Emc;



	if(dR < 0.05){
	  if(debug)
	    cout << " EG ele matched: pt " << theGsfEle[j].pt() << " (" << SCEnergy/std::cosh(etareco) << ") " << " eta,phi " <<  etareco << ", " << phireco << " pfmva " <<  pfmva << endl;
	

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

	  if( (theGsfEle[j].parentSuperCluster()).isNonnull()) {
	    float SCPF = (theGsfEle[j]).parentSuperCluster()->rawEnergy();
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
	    const reco::GsfElectron::ShowerShape& pfshapes = 
	      theGsfEle[j].showerShape();
	    
	    pf_e1x5_all->Fill(pfshapes.e1x5);
	    pf_sihih_all->Fill(pfshapes.sigmaIetaIeta);
	    pf_r9_all->Fill(pfshapes.r9);
	    pf_scshh_all->Fill((theGsfEle[j]).parentSuperCluster()->etaWidth());
	    pf_scsff_all->Fill((theGsfEle[j]).parentSuperCluster()->phiWidth());
	    if( std::abs(etareco) < 1.479 ) {
	      pf_e1x5_eb->Fill(pfshapes.e1x5);
	      pf_sihih_eb->Fill(pfshapes.sigmaIetaIeta);
	      pf_r9_eb->Fill(pfshapes.r9);
	      pf_scshh_eb->Fill((theGsfEle[j]).parentSuperCluster()->etaWidth());
	      pf_scsff_eb->Fill((theGsfEle[j]).parentSuperCluster()->phiWidth());
	    }
	    if( std::abs(etareco) >= 1.479 ) {
	      pf_e1x5_ee->Fill(pfshapes.e1x5);
	      pf_sihih_ee->Fill(pfshapes.sigmaIetaIeta);
	      pf_r9_ee->Fill(pfshapes.r9);
	      pf_scshh_ee->Fill((theGsfEle[j]).parentSuperCluster()->etaWidth());
	      pf_scsff_ee->Fill((theGsfEle[j]).parentSuperCluster()->phiWidth());
	    }
	    
	  }

	  eg_e1x5_all->Fill(theGsfEle[j].e1x5());
	  eg_sihih_all->Fill(theGsfEle[j].sigmaIetaIeta());
	  eg_r9_all->Fill(theGsfEle[j].r9());
	  eg_scshh_all->Fill((theGsfEle[j]).superCluster()->etaWidth());
	  eg_scsff_all->Fill((theGsfEle[j]).superCluster()->phiWidth());
	  if( std::abs(etareco) < 1.479 ) {
	    eg_e1x5_eb->Fill(theGsfEle[j].e1x5());
	    eg_sihih_eb->Fill(theGsfEle[j].sigmaIetaIeta());
	    eg_r9_eb->Fill(theGsfEle[j].r9());
	    eg_scshh_eb->Fill((theGsfEle[j]).superCluster()->etaWidth());
	    eg_scsff_eb->Fill((theGsfEle[j]).superCluster()->phiWidth());
	  }
	  if( std::abs(etareco) >= 1.479 ) {
	    eg_e1x5_ee->Fill(theGsfEle[j].e1x5());
	    eg_sihih_ee->Fill(theGsfEle[j].sigmaIetaIeta());
	    eg_r9_ee->Fill(theGsfEle[j].r9());
	    eg_scshh_ee->Fill((theGsfEle[j]).superCluster()->etaWidth());
	    eg_scsff_ee->Fill((theGsfEle[j]).superCluster()->phiWidth());
	  }
  
	  MindREG = dR;
	  /*
	  if(seedRef->isEcalDriven())
	    isEgEcDr = true;
	  if(seedRef->isTrackerDriven())
	    isEgTrDr = true;
	  */
	}
      }
      if(MindREG < 0.05) {
	h_etaeg_ele->Fill(etamc);
	h_pteg_ele->Fill(ptmc);


	// This is to understand the contribution of each seed type. 
	if(isEgEcDr) {
	  h_etaegEcDr_ele->Fill(etamc);
	  h_ptegEcDr_ele->Fill(ptmc);
	}
	if(isEgTrDr) {
	  h_etaegTrDr_ele->Fill(etamc);
	  h_ptegTrDr_ele->Fill(ptmc);
	}
      }




      //Add loop on the GED electrons
      
      
      float MindRGedEg =1000;
      bool isGedEgTrDr = false;
      bool isGedEgEcDr = false;
      
      for (uint j=0; j<theGedEle.size();j++) {
	reco::GsfTrackRef egGsfTrackRef = (theGedEle[j]).gsfTrack();
	float etareco = theGedEle[j].eta();
	float phireco = theGedEle[j].phi();	
	float pfmva =  theGedEle[j].mva_e_pi();
	
	reco::GsfTrackRef refGsf =  theGedEle[j].gsfTrack();
	//ElectronSeedRef seedRef= refGsf->extra()->seedRef().castTo<ElectronSeedRef>();
	
	
	float deta = etamc - etareco;
	float dphi = Utils::mpi_pi(phimc - phireco);
	float dR = sqrt(deta*deta + dphi*dphi);
	
	float SCEnergy = (theGedEle[j]).superCluster()->energy();
	float ErecoEtrue = SCEnergy/Emc;

	

	if(dR < 0.05){	  
	  const reco::PFCandidate* matchPF = NULL;
	  if(debug)
	    cout << " GED ele matched: pt " << theGedEle[j].pt() << " (" << SCEnergy/std::cosh(etareco) << ") "<< " eta,phi " <<  etareco << ", " << phireco << " pfmva " <<  pfmva << endl;
	  
	  for( unsigned k = 0; k < candidates.size(); ++k ) {
	    if( std::abs(candidates[k].pdgId()) == 11 ) {
	      reco::GsfTrackRef gsfEleGsfTrackRef = 
		candidates[k].gsfTrackRef();
	      if( gsfEleGsfTrackRef->ptMode() == egGsfTrackRef->ptMode() ) {
		matchPF = &candidates[k];	     
	      }
	    }
	  }

	  if( debug && matchPF ) std::cout << "GED-PF match!" << std::endl;

	  if( ErecoEtrue < 0.6 ) std::cout << "++bad Ereco/Etrue: " 
					   << ErecoEtrue << std::endl;

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
	    /*
	    if( debug && matchPF && ErecoEtrue > 1.30 ) {
	      cout << " -PF ele matched: pt " 
		   << matchPF->pt()
		 << " eta,phi " <<  matchPF->eta() 
		 << ", " << matchPF->phi() << " pfmva " 
		 <<  matchPF->mva_e_pi() << endl;
	      std::cout << "GED Clusters: " << std::endl;
	      for( const auto& c : theGedEle[j].superCluster()->clusters() ) {
		std::cout << " E/eta/phi : " << c->energy() 
			  << '/' << c->eta() 
			  << '/' << c->phi() << std::endl;		
	      }
	      std::cout << "-PF Clusters: " << std::endl;
	      for( const auto& c : matchPF->superClusterRef()->clusters() ) {
		std::cout << " E/eta/phi : " << c->energy() 
			  << '/' << c->eta() 
			  << '/' << c->phi() << std::endl;
	      }
	      
	    }
	    */
	  }

	  ged_e1x5_all->Fill(theGedEle[j].e1x5());
	  ged_sihih_all->Fill(theGedEle[j].sigmaIetaIeta());
	  ged_r9_all->Fill(theGedEle[j].r9());
	  ged_scshh_all->Fill((theGedEle[j]).superCluster()->etaWidth());
	  ged_scsff_all->Fill((theGedEle[j]).superCluster()->phiWidth());
	  if( std::abs(etareco) < 1.479 ) {
	    ged_e1x5_eb->Fill(theGedEle[j].e1x5());
	    ged_sihih_eb->Fill(theGedEle[j].sigmaIetaIeta());
	    ged_r9_eb->Fill(theGedEle[j].r9());
	    ged_scshh_eb->Fill((theGedEle[j]).superCluster()->etaWidth());
	    ged_scsff_eb->Fill((theGedEle[j]).superCluster()->phiWidth());
	  }
	  if( std::abs(etareco) >= 1.479 ) {
	    ged_e1x5_ee->Fill(theGedEle[j].e1x5());
	    ged_sihih_ee->Fill(theGedEle[j].sigmaIetaIeta());
	    ged_r9_ee->Fill(theGedEle[j].r9());
	    ged_scshh_ee->Fill((theGedEle[j]).superCluster()->etaWidth());
	    ged_scsff_eb->Fill((theGedEle[j]).superCluster()->phiWidth());
	  }

	  MindRGedEg = dR;
	  /*
	  if(seedRef->isEcalDriven())
	    isGedEgEcDr = true;
	  if(seedRef->isTrackerDriven())
	    isGedEgTrDr = true;
	  */
	}
      }
      if(MindRGedEg < 0.05) {
	h_etaged_ele->Fill(etamc);
	h_ptged_ele->Fill(ptmc);
	
	
	// This is to understand the contribution of each seed type. 
	if(isGedEgEcDr) {
	  h_etagedEcDr_ele->Fill(etamc);
	  h_ptgedEcDr_ele->Fill(ptmc);
	}
	if(isGedEgTrDr) {
	  h_etagedTrDr_ele->Fill(etamc);
	  h_ptgedTrDr_ele->Fill(ptmc);
	}
      }//End Loop Generator Particles  
      
      


    } //End IF Generator Particles 
  
  
  } //End Loop Generator Particles 

}
// ------------ method called once each job just before starting event loop  ------------
void 
GsfGEDElectronAnalyzer::beginJob(const edm::EventSetup&)
{

  ev = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GsfGEDElectronAnalyzer::endJob() {
  cout << " endJob:: #events " << ev << endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(GsfGEDElectronAnalyzer);
