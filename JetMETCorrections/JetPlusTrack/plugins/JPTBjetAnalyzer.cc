// Original Author:  Viola Sordini &b Joanna Weng July  2009
// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
// /CMSSW/Calibration/HcalAlCaRecoProducers/src/AlCaIsoTracksProducer.cc  track propagator
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// MC info
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "CLHEP/HepPDT/DefaultConfig.hh"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"
//double dR = deltaR( c1, c2 );
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//jets
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
///electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
//needed for elecID
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "DataFormats/Common/interface/ValueMap.h"
//#include "GeneratorInterface/GenFilters/test/Algos.h"
// muons and tracks
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
// ecal
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
// candidates
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include <vector>

using namespace std;
using namespace reco;

//
// class decleration
//

class JPTBjetAnalyzer : public edm::EDAnalyzer {
public:
  explicit JPTBjetAnalyzer(const edm::ParameterSet&);
  ~JPTBjetAnalyzer();


private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  // output root file
  string fOutputFileName ;
  // names of modules, producing object collections
  // raw calo jets
  string calojetsSrc;
  // calo jets after zsp corrections
  string zspjetsSrc;
  // gen jets
  string genjetsSrc;
  // electrons 
  string recoGsfelectronsSrc;
  // muons 
  string muonsSrc;
  //GenParticles 
  string GenParticlesSrc; 
  double electron_abseta;
  double electron_pt_min;  
  double muon_abseta;
  double muon_pt_min;     
  double jet_abseta;
  double jet_pt_min;    
  double tau_pt_min;
  double tau_abseta;
  //
  // MC jet energy corrections
  //  string JetCorrectionMCJ;
  // ZSP jet energy corrections
  //  string JetCorrectionZSP;
  // Jet+tracks energy corrections
  string JetCorrectionJPT;
  // variables to store in ntpl
  double  EtaGen1, PhiGen1, EtaRaw1, PhiRaw1, EtGen1, EtRaw1, EtMCJ1, EtZSP1, EtJPT1, DRMAXgjet1;
  double drElecFromZjet1, drMuonFromZjet1, drTauFromZjet1;
  double  EtaGen2, PhiGen2, EtaRaw2, PhiRaw2, EtGen2, EtRaw2, EtMCJ2, EtZSP2, EtJPT2, DRMAXgjet2;
  double drElecFromZjet2, drMuonFromZjet2, drTauFromZjet2;
  double hoE;
  int MuonFlagGen1,ElectronFlagGen1, TauFlagGen1;
  int MuonFlagGen2,ElectronFlagGen2, TauFlagGen2;
  int MuonFlagGen1NoLep,ElectronFlagGen1NoLep, TauFlagGen1NoLep;
  int MuonFlagGen2NoLep,ElectronFlagGen2NoLep, TauFlagGen2NoLep;
  int BJet1FlagGen,BJet2FlagGen;
  double elecMomID;
  int nelecs, nmuons, ntaus;
  double elecMom[10], muonMom[10], tauMom[10];
  double elecPt[10], muonPt[10], tauPt[10];

  // output root file and tree
  TFile*      hOutputFile ;
  TTree*      t1;

};


// ------------ method called once each job just before starting event loop  ------------
void 
JPTBjetAnalyzer::beginJob(const edm::EventSetup&)
{
  using namespace edm;
  // creating a simple tree

  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;


  t1 = new TTree("t1","analysis tree");

  //first jet
  t1->Branch("EtaGen1",&EtaGen1,"EtaGen1/D");
  t1->Branch("PhiGen1",&PhiGen1,"PhiGen1/D");
  t1->Branch("ElectronFlagGen1",&ElectronFlagGen1,"ElectronFlagGen1/I");
  t1->Branch("MuonFlagGen1",&MuonFlagGen1,"MuonFlagGen1/I");
  t1->Branch("TauFlagGen1",&TauFlagGen1,"TauFlagGen1/I");
  t1->Branch("ElectronFlagGen1NoLep",&ElectronFlagGen1NoLep,"ElectronFlagGen1NoLep/I");
  t1->Branch("MuonFlagGen1NoLep",&MuonFlagGen1NoLep,"MuonFlagGen1NoLep/I");
  t1->Branch("TauFlagGen1NoLep",&TauFlagGen1NoLep,"TauFlagGen1NoLep/I");
  t1->Branch("BJet1FlagGen",&BJet1FlagGen,"BJet1FlagGen/I");
  t1->Branch("EtaRaw1",&EtaRaw1,"EtaRaw1/D");
  t1->Branch("PhiRaw1",&PhiRaw1,"PhiRaw1/D");
  t1->Branch("EtGen1",&EtGen1,"EtGen1/D");
  t1->Branch("EtRaw1",&EtRaw1,"EtRaw1/D");
  t1->Branch("EtMCJ1",&EtMCJ1,"EtMCJ1/D");
  t1->Branch("EtZSP1",&EtZSP1,"EtZSP1/D");
  t1->Branch("EtJPT1",&EtJPT1,"EtJPT1/D");
  t1->Branch("hoE",&hoE,"hoE/D");
  t1->Branch("DRMAXgjet1",&DRMAXgjet1,"DRMAXgjet1/D");
  t1->Branch("drElecFromZjet1",&drElecFromZjet1,"drElecFromZjet1/D");
  t1->Branch("drMuonFromZjet1",&drMuonFromZjet1,"drMuonFromZjet1/D");
  t1->Branch("drTauFromZjet1",&drTauFromZjet1,"drTauFromZjet1/D");

  //second jet
  t1->Branch("EtaGen2",&EtaGen2,"EtaGen2/D");
  t1->Branch("PhiGen2",&PhiGen2,"PhiGen2/D");
  t1->Branch("ElectronFlagGen2",&ElectronFlagGen2,"ElectronFlagGen2/I");
  t1->Branch("MuonFlagGen2",&MuonFlagGen2,"MuonFlagGen2/I");
  t1->Branch("TauFlagGen2",&TauFlagGen2,"TauFlagGen2/I");
  t1->Branch("ElectronFlagGen2NoLep",&ElectronFlagGen2NoLep,"ElectronFlagGen2NoLep/I");
  t1->Branch("MuonFlagGen2NoLep",&MuonFlagGen2NoLep,"MuonFlagGen2NoLep/I");
  t1->Branch("TauFlagGen2NoLep",&TauFlagGen2NoLep,"TauFlagGen2NoLep/I");
  t1->Branch("BJet2FlagGen",&BJet2FlagGen,"BJet2FlagGen/I");
  t1->Branch("EtaRaw2",&EtaRaw2,"EtaRaw2/D");
  t1->Branch("PhiRaw2",&PhiRaw2,"PhiRaw2/D");
  t1->Branch("EtGen2",&EtGen2,"EtGen2/D");
  t1->Branch("EtRaw2",&EtRaw2,"EtRaw2/D");
  t1->Branch("EtMCJ2",&EtMCJ2,"EtMCJ2/D");
  t1->Branch("EtZSP2",&EtZSP2,"EtZSP2/D");
  t1->Branch("EtJPT2",&EtJPT2,"EtJPT2/D");
  t1->Branch("DRMAXgjet2",&DRMAXgjet2,"DRMAXgjet2/D");
  t1->Branch("drElecFromZjet2",&drElecFromZjet2,"drElecFromZjet2/D");
  t1->Branch("drMuonFromZjet2",&drMuonFromZjet2,"drMuonFromZjet2/D");
  t1->Branch("drTauFromZjet2",&drTauFromZjet2,"drTauFromZjet2/D");

  //leptons inside jets
  t1->Branch("nelecs",&nelecs,"nelecs/I");
  t1->Branch("elecMom",elecMom,"elecMom[nelecs]/D");
  t1->Branch("elecPt",elecPt,"elecPt[nelecs]/D");
  t1->Branch("nmuons",&nmuons,"nmuons/I");
  t1->Branch("muonMom",muonMom,"muonMom[nmuons]/D");
  t1->Branch("muonPt",muonPt,"muonPt[nmuons]/D");
  t1->Branch("ntaus",&nmuons,"ntaus/I");
  t1->Branch("tauMom",tauMom,"tauMom[ntaus]/D");
  t1->Branch("tauPt",tauPt,"tauPt[ntaus]/D");

  return ;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JPTBjetAnalyzer::endJob() {

  hOutputFile->Write() ;
  hOutputFile->Close() ;
  
  return ;
}

//
// constructors and destructor
//
JPTBjetAnalyzer::JPTBjetAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  using namespace edm;
  // 
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
  //
  // get names of input object collections
  // raw calo jets
  calojetsSrc   = iConfig.getParameter< std::string > ("calojets");
  // calo jets after zsp corrections
  zspjetsSrc    = iConfig.getParameter< std::string > ("zspjets");
  genjetsSrc    = iConfig.getParameter< std::string > ("genjets");
  recoGsfelectronsSrc= iConfig.getParameter< std::string > ("electrons");
  muonsSrc= iConfig.getParameter< std::string > ("muons");
  GenParticlesSrc= iConfig.getParameter< std::string > ("genparticles");
  electron_pt_min=iConfig.getParameter<double> ("electron_pt_min");
  electron_abseta= iConfig.getParameter<double> ("electron_abseta"); //
  muon_pt_min=iConfig.getParameter<double> ("muon_pt_min");
  muon_abseta=iConfig.getParameter<double> ("muon_abseta"); // 
  jet_pt_min=iConfig.getParameter<double> ("jet_pt_min");
  jet_abseta=iConfig.getParameter<double> ("jet_abseta"); //
  // MC jet energy corrections
  //  JetCorrectionMCJ = iConfig.getParameter< std::string > ("JetCorrectionMCJ");
  // ZSP jet energy corrections
  //  JetCorrectionZSP = iConfig.getParameter< std::string > ("JetCorrectionZSP");
  // Jet+tracks energy corrections
  JetCorrectionJPT = iConfig.getParameter< std::string > ("JetCorrectionJPT");
}


JPTBjetAnalyzer::~JPTBjetAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
JPTBjetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // initialize vector containing two highest Et gen jets > 20 GeV
  // in this example they are checked not to be leptons from Z->ll decay (DR match)
  vector<CLHEP::HepLorentzVector> gjets;
  gjets.clear();

  // initialize tree variables
  EtaGen1 = 0.;
  PhiGen1 = 0.;
  EtaRaw1 = 0.;
  PhiRaw1 = 0.;
  EtGen1  = 0.;
  EtRaw1  = 0.;
  EtMCJ1  = 0.;
  EtZSP1  = 0.;
  EtJPT1  = 0.;
  DRMAXgjet1 = 1000.;
  ElectronFlagGen1=0;
  MuonFlagGen1=0;
  TauFlagGen1=0;
  ElectronFlagGen1NoLep=0;
  MuonFlagGen1NoLep=0;
  TauFlagGen1NoLep=0;
  BJet1FlagGen=0;
  BJet2FlagGen=0;
  EtaGen2 = 0.;
  PhiGen2 = 0.;
  EtaRaw2 = 0.;
  PhiRaw2 = 0.;
  EtGen2  = 0.;
  EtRaw2  = 0.;
  EtMCJ2  = 0.;
  EtZSP2  = 0.;
  EtJPT2  = 0.;
  DRMAXgjet2 = 1000.;
  ElectronFlagGen2=0;
  MuonFlagGen2=0;
  TauFlagGen2=0;
  ElectronFlagGen2NoLep=0;
  MuonFlagGen2NoLep=0;
  TauFlagGen2NoLep=0;
  hoE=0.;
  nelecs=0;
  nmuons=0;
  ntaus=0;
  for(unsigned i = 0; i < 10; i++){
    elecMom[i]=0;
    muonMom[i]=0;
    tauMom[i]=0;
    elecPt[i]=0;
    muonPt[i]=0;
    tauPt[i]=0;
  }

  
// get MC info
edm::Handle<HepMCProduct> EvtHandle ;
iEvent.getByLabel( "source", EvtHandle ) ;

// get gen jets collection 
Handle<GenJetCollection> genjets;
iEvent.getByLabel(genjetsSrc, genjets);
Handle<reco::GsfElectronCollection>  elecs;
iEvent.getByLabel( recoGsfelectronsSrc, elecs);
Handle<MuonCollection> muons;
iEvent.getByLabel(muonsSrc, muons);
Handle<reco::GenParticleCollection>  genparts;
iEvent.getByLabel(GenParticlesSrc , genparts);
std::vector <const reco::GenParticle*> mcElectrons;
std::vector <const reco::GenParticle*> mcMuons;
std::vector <const reco::GenParticle*> mcTaus;
std::vector <const reco::GenParticle*> mcBquarks;


// Look for prompt e/mu/tau from Z(=23)
for (GenParticleCollection::const_iterator p = genparts->begin(); p != genparts->end(); ++p) {
  if(fabs((&*p)->pdgId())==11 && (&*p)->status()==1 && (&*p)->pt()>electron_pt_min &&  fabs((&*p)->eta()) <electron_abseta && (&*p)->mother()->mother()->pdgId()==23) mcElectrons.push_back((&*p));    
  if(fabs((&*p)->pdgId())==13 && (&*p)->status()==1 && (&*p)->pt()>muon_pt_min &&  fabs((&*p)->eta()) <muon_abseta && (&*p)->mother()->mother()->pdgId()==23 ) mcMuons.push_back((&*p));   
  if(fabs((&*p)->pdgId())==15 && (&*p)->status()==1 && (&*p)->pt()>tau_pt_min &&  fabs((&*p)->eta()) <tau_abseta && (&*p)->mother()->mother()->pdgId()==23 ) mcTaus.push_back((&*p));   
  // Look for bquarks
  if(((&*p)->status()==3) && ((fabs((&*p)->pdgId()) == 5))) mcBquarks.push_back((&*p));    	         
}	        
   
int jg = 0;
for(GenJetCollection::const_iterator gjet = genjets->begin(); gjet != genjets->end(); ++gjet ) {
 
  //Rough preselection,final cut in root macro
  if(gjet->pt() > jet_pt_min && fabs(gjet->eta())<jet_abseta ) {
    CLHEP::HepLorentzVector jet(gjet->px(), gjet->py(), gjet->pz(), gjet->energy());
    double drMINB=999;
    for(unsigned int n=0; n< mcBquarks.size() ; n++ ){
      CLHEP::HepLorentzVector bquark(mcBquarks[n]->px(), mcBquarks[n]->py(), mcBquarks[n]->pz(), mcBquarks[n]->energy());
      double dr =bquark.deltaR(jet);
      if (dr < drMINB) drMINB=dr;
      }
    if(drMINB < 0.4) { //the genjet matches a b quark
    double drMIN=999;
    // Check for electrons 
    hoE = gjet->hadEnergy()/gjet->emEnergy();
    for(unsigned int n=0; n< mcElectrons.size() ; n++ ){
      CLHEP::HepLorentzVector elec(mcElectrons[n]->px(), mcElectrons[n]->py(), mcElectrons[n]->pz(), mcElectrons[n]->energy());
      double dr =elec.deltaR(jet);
      if (dr < drMIN) drMIN=dr;
    }
    if(drMIN > 0.5)
      { // no electrons from Z
	// Check for muons 
	double drMIN2=999;
	for(unsigned int n=0; n< mcMuons.size() ; n++ ){
	  CLHEP::HepLorentzVector mu(mcMuons[n]->px(), mcMuons[n]->py(), mcMuons[n]->pz(), mcMuons[n]->energy());
	  double dr =mu.deltaR(jet);
	  if (dr < drMIN2) drMIN2=dr;
	}
	if(drMIN2 > 0.5)
	  { // no muons from Z
	    // Check for taus
	    double drMIN3=999;
	    for(unsigned int n=0; n< mcTaus.size() ; n++ ){
	      CLHEP::HepLorentzVector tau(mcTaus[n]->px(), mcTaus[n]->py(), mcTaus[n]->pz(), mcTaus[n]->energy());
	      double dr =tau.deltaR(jet);
	      if (dr < drMIN3) drMIN3=dr;
	    }
	    if(drMIN3 > 0.5) 
	      { //no taus from Z
		//Match to bquark
		    jg++;
		    if(jg <= 2) {
		      std::vector <const reco::GenParticle*> mcparts =  gjet->getGenConstituents();
		      for (unsigned i = 0; i < mcparts.size (); i++) {
			const reco::GenParticle* mcpart = mcparts[i];		    
			if (mcpart) {	  
			  if (jg==1){
			    drElecFromZjet1=drMIN;
			    drMuonFromZjet1=drMIN2;
			    drTauFromZjet1=drMIN3;
			    //Electrons
			    if ((fabs(mcpart->pdgId())==11)) { 
			      //				  cout<< "electron with mother " << mcpart->mother()->pdgId() << endl;				 				
			      //vectorial tree
			      elecMom[nelecs]=mcpart->mother()->pdgId();
			      if(abs(elecMom[nelecs])==11) elecMom[nelecs]=mcpart->mother()->mother()->pdgId();
			      elecPt[nelecs]=mcpart->pt();
			      //take away leptons to do a la Anne-Marie
			      ElectronFlagGen1 = 1;
			      if(abs(elecMom[nelecs])>400.)	 ElectronFlagGen1NoLep = 1;
			      nelecs++;
			    }
			    // Muons
			    if ((fabs(mcpart->pdgId())==13)) {		
			      muonMom[nmuons]=mcpart->mother()->pdgId();
			      if(abs(muonMom[nmuons])==13) muonMom[nmuons]=mcpart->mother()->mother()->pdgId();
			      muonPt[nmuons]=mcpart->pt();
			      MuonFlagGen1= 1;
			      if(abs(muonMom[nmuons])>400.) MuonFlagGen1NoLep= 1;	
			      nmuons++;
			    }
			    //Taus
			    if ((fabs(mcpart->pdgId())==15)) {			    
			      tauMom[ntaus]=mcpart->mother()->pdgId();
			      if(abs(tauMom[ntaus])==15) muonMom[ntaus]=mcpart->mother()->mother()->pdgId();
			      tauPt[ntaus]=mcpart->pt();
			      TauFlagGen1= 1;	
			      if(abs(muonMom[ntaus])>400) TauFlagGen1NoLep= 1;	
			      ntaus++;
			      //				  cout<<"ntaus = " << ntaus << endl;
			      // cout<<"tauMom = " << tauMom[ntaus-1] << endl;
			      // cout<<"tauGrandma = " << mcpart->mother()->mother()->pdgId() << endl;
			    }
			  } // first jet if (jg==1)

			  if (jg==2){
			    drElecFromZjet2=drMIN;
			    drMuonFromZjet2=drMIN2;
			    drTauFromZjet2=drMIN3;
			    if ((fabs(mcpart->pdgId())==11)){
			      elecMom[nelecs]=mcpart->mother()->pdgId();
			      if(abs(elecMom[nelecs])==11) elecMom[nelecs]=mcpart->mother()->mother()->pdgId();
			      elecPt[nelecs]=mcpart->pt();
			      ElectronFlagGen2 = 1;
			      if(abs(elecMom[nelecs])>400.)	 ElectronFlagGen2NoLep = 1;
			      nelecs++;
			    }
			    if ((fabs(mcpart->pdgId())==13)){
			      muonMom[nmuons]=mcpart->mother()->pdgId();
			      if(abs(muonMom[nmuons])==13) muonMom[nmuons]=mcpart->mother()->mother()->pdgId();
			      muonPt[nmuons]=mcpart->pt();
			      MuonFlagGen2= 1;	
			      if(abs(muonMom[nmuons])>400.) MuonFlagGen2NoLep= 1;
			      nmuons++;
			    }

			    if ((fabs(mcpart->pdgId())==15)){ 
			      tauMom[ntaus]=mcpart->mother()->pdgId();
			      if(abs(tauMom[ntaus])==15) muonMom[ntaus]=mcpart->mother()->mother()->pdgId();
			      tauPt[ntaus]=mcpart->pt();
			      TauFlagGen2= 1;	
			      if(abs(muonMom[ntaus])>400) TauFlagGen2NoLep= 1;	
			      ntaus++;
			    }
			  } // second jet  if (jg==2) 			     
			} //if mcpart
		      }   //constituents of jet   < mcparts.size (); i++) {
		      gjets.push_back(jet);
		    } // if jg <=2 onnly 2 first jets		  	    
	      } //no taus
	  } //no muons
      } // no electrons
    } //bjets 
  }  //(gjet->pt() > jet_pt_min. && fabs(gjet->eta())<jet_abseta ) {
} // end  for(GenJetCollection::const_iterator gjet = genjets->begin(); 
  //   cout <<" ==> NUMBER OF GOOD GEN JETS " << gjets.size() << endl;


//-------------------------------------------------
// Reco Jet part 
if(gjets.size() > 0) {
  // get calo jet collection
  edm::Handle<CaloJetCollection> calojets;
  iEvent.getByLabel(calojetsSrc, calojets);
  // get calo jet after zsp collection
  edm::Handle<CaloJetCollection> zspjets;
  iEvent.getByLabel(zspjetsSrc, zspjets);
   
  if(calojets->size() > 0) {
    // MC jet energy corrections
    //       const JetCorrector* correctorMCJ = JetCorrector::getJetCorrector (JetCorrectionMCJ, iSetup);
    // ZSP jet energy corrections
    //       const JetCorrector* correctorZSP = JetCorrector::getJetCorrector (JetCorrectionZSP, iSetup);
    // Jet+tracks energy corrections
    const JetCorrector* correctorJPT = JetCorrector::getJetCorrector (JetCorrectionJPT, iSetup);
       
    // loop over jets and do matching with gen jets      
    for( CaloJetCollection::const_iterator cjet = calojets->begin(); 
	 cjet != calojets->end(); ++cjet ){ 
      CLHEP::HepLorentzVector cjetc(cjet->px(), cjet->py(), cjet->pz(), cjet->energy());    
      //Finding ZSP jet that corresponds to calo jet
      CaloJetCollection::const_iterator zspjet;
      for( zspjet = zspjets->begin();  zspjet != zspjets->end(); ++zspjet ){ 
	CLHEP::HepLorentzVector zspjetc(zspjet->px(), zspjet->py(), zspjet->pz(), zspjet->energy());
	double dr = zspjetc.deltaR(cjetc);
	if(dr < 0.001) break;
      }
      // JPT corrections
      double scaleJPT = correctorJPT->correction ((*zspjet),iEvent,iSetup);
      Jet::LorentzVector jetscaleJPT(zspjet->px()*scaleJPT, zspjet->py()*scaleJPT,
				     zspjet->pz()*scaleJPT, zspjet->energy()*scaleJPT);    
      // Building new CaloJet
      CaloJet cjetJPT(jetscaleJPT, cjet->getSpecific(), cjet->getJetConstituents());
	
      //Look for 2 matching jets with dR and fill all Tree quantities
      double DRgjet1 = gjets[0].deltaR(cjetc);
      // Fill 1 jet  
      if(DRgjet1 < DRMAXgjet1) {
	DRMAXgjet1 = DRgjet1;
         
	//electrons in jets  	  
	EtaGen1 = gjets[0].eta();
	PhiGen1 = gjets[0].phi();
	EtGen1  = gjets[0].perp();

	EtaRaw1 = cjet->eta(); 
	PhiRaw1 = cjet->phi();
	EtRaw1  = cjet->pt();
	//	   EtMCJ1  = cjetMCJ.pt(); 
	EtZSP1  = zspjet->pt(); 
	EtJPT1  = cjetJPT.pt(); 
      }
	
      //Fill 2. jet
      if(gjets.size() == 2) {
	double DRgjet2 = gjets[1].deltaR(cjetc);
	if(DRgjet2 < DRMAXgjet2) { 
	  DRMAXgjet2 = DRgjet2;

	  EtaGen2 = gjets[1].eta();
	  PhiGen2 = gjets[1].phi();
	  EtGen2  = gjets[1].perp();

	  EtaRaw2 = cjet->eta(); 
	  PhiRaw2 = cjet->phi();
	  EtRaw2  = cjet->pt();
	  EtZSP2  = zspjet->pt(); 
	  EtJPT2  = cjetJPT.pt(); 
	}
      }  //     if(gjets.size() == 2) {
   
    } // end calo jet loop
    
    /*
      cout <<" best match to 1st gen get = " << DRMAXgjet1
      <<" raw jet pt = " << EtRaw1 <<" eta = " << EtaRaw1 <<" phi " << PhiRaw1 
      <<" mcj pt = " << EtMCJ1 << " zsp pt = " << EtZSP1 <<" jpt = " << EtJPT1 << endl; 
      if(gjets.size() == 2) {
      cout <<" best match to 2st gen get = " << DRMAXgjet2
      <<" raw jet pt = " << EtRaw2 <<" eta = " << EtaRaw2 <<" phi " << PhiRaw2 
      <<" mcj pt = " << EtMCJ2 << " zsp pt = " << EtZSP2 <<" jpt = " << EtJPT2 << endl; 
      }
    */   
  }  //if(calojets->size() > 0)
}   //if(gjets->size() > 0)
// fill tree
t1->Fill();
}


//define this as a plug-in
DEFINE_FWK_MODULE(JPTBjetAnalyzer);
