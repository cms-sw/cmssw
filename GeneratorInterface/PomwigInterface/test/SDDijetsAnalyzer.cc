////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TH1F.h"

class SDDijetsAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  SDDijetsAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~SDDijetsAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  //virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void beginJob();
  virtual void endJob() ;

private:
  // Input from cfg file
  edm::InputTag genParticlesTag_;
  edm::InputTag genJetsTag_;

  // Histograms
  TH1F* hJet1Pt;
  TH1F* hJet1Eta;
  TH1F* hJet1Phi;
  TH1F* hJet2Pt;
  TH1F* hJet2Eta;
  TH1F* hJet2Phi;
  TH1F* hJetDeltaEta;
  TH1F* hJetDeltaPhi;
  TH1F* hJetDeltaPt;
  TH1F* hThrust;
  TH1F* hEnergyvsEta;
  TH1F* hXiGen;
  TH1F* hProtonPt2;	

  int nevents;
  bool debug;
  double Ebeam;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "CommonTools/CandUtils/interface/Thrust.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

struct jetptcomp {
  bool operator() (std::pair<reco::GenJetCollection::const_iterator,double> jet1, std::pair<reco::GenJetCollection::const_iterator,double> jet2){
                return (jet1.second > jet2.second);
        } 
} myjetptcomp;

/// Constructor
SDDijetsAnalyzer::SDDijetsAnalyzer(const edm::ParameterSet& pset)
{
  genParticlesTag_ = pset.getParameter<edm::InputTag>("GenParticleTag");
  genJetsTag_ = pset.getParameter<edm::InputTag>("GenJetTag");
  Ebeam = pset.getParameter<double>("EBeam");
  debug = pset.getUntrackedParameter<bool>("debug",false);
}

/// Destructor
SDDijetsAnalyzer::~SDDijetsAnalyzer(){
}

void SDDijetsAnalyzer::beginJob(){
  edm::Service<TFileService> fs;
  //TH1::SetDefaultSumw2(true);

  hJet1Pt = fs->make<TH1F>("hJet1Pt","hJet1Pt",100,0.,200.);
  hJet1Eta = fs->make<TH1F>("hJet1Eta","hJet1Eta",100,-5.,5.);
  hJet1Phi = fs->make<TH1F>("hJet1Phi","hJet1Phi",100,-3.141592,3.141592);
  hJet2Pt = fs->make<TH1F>("hJet2Pt","hJet2Pt",100,0.,200.);
  hJet2Eta = fs->make<TH1F>("hJet2Eta","hJet2Eta",100,-5.,5.);  
  hJet2Phi = fs->make<TH1F>("hJet2Phi","hJet2Phi",100,-3.141592,3.141592);

  hJetDeltaEta = fs->make<TH1F>("hJetDeltaEta","hJetDeltaEta",100,-5.,5.);
  hJetDeltaPhi = fs->make<TH1F>("hJetDeltaPhi","hJetDeltaPhi",100,-3.141592,3.141592);
  hJetDeltaPt = fs->make<TH1F>("hJetDeltaPt","hJetDeltaPt",100,0.,100.);
  hThrust = fs->make<TH1F>("hThrust","hThrust",100,0.,1.);

  hEnergyvsEta = fs->make<TH1F>("hEnergyvsEta","hEnergyvsEta",100,-15.0,15.0); 		
  hXiGen = fs->make<TH1F>("hXiGen","hXiGen",100,0.,0.21);
  hProtonPt2 = fs->make<TH1F>("hProtonPt2","hProtonPt2",100,0.,3.0);

  nevents = 0;
}

void SDDijetsAnalyzer::endJob(){
  hEnergyvsEta->Scale(1/(float)nevents);	
}

void SDDijetsAnalyzer::analyze(const edm::Event & ev, const edm::EventSetup&){
  nevents++; 

  // Generator Information
  edm::Handle<reco::GenParticleCollection> genParticles;
  ev.getByLabel(genParticlesTag_, genParticles);
  double pz1max = 0.;
  double pz2min = 0.;
  reco::GenParticleCollection::const_iterator proton1 = genParticles->end();
  reco::GenParticleCollection::const_iterator proton2 = genParticles->end();
  //std::vector<reco::GenParticle> myStableParticles;
  //for(size_t i = 0; i < genParticles->size(); ++i) {
  for(reco::GenParticleCollection::const_iterator genpart = genParticles->begin(); genpart != genParticles->end(); ++genpart){
      		//const reco::GenParticle& genpart = (*genParticles)[i];
      		//LogTrace("") << ">>>>>>> pid,status,px,py,px,e= "  << genpart.pdgId() << " , " << genpart.status() << " , " << genpart.px() << " , " << genpart.py() << " , " << genpart.pz() << " , " << genpart.energy();	
		if(genpart->status() != 1) continue;
		//myStableParticles.push_back(*genpart);
		hEnergyvsEta->Fill(genpart->eta(),genpart->energy());	
		
		double pz = genpart->pz();
     		if((genpart->pdgId() == 2212)&&(pz > 0.75*Ebeam)){
			if(pz > pz1max){proton1 = genpart;pz1max=pz;}
		} else if((genpart->pdgId() == 2212)&&(pz < -0.75*Ebeam)){
     			if(pz < pz2min){proton2 = genpart;pz2min=pz;}
     		}
  }

  if(proton1 != genParticles->end()){
		if(debug) std::cout << "Proton 1: " << proton1->pt() << "  " << proton1->eta() << "  " << proton1->phi() << std::endl;
   		double xigen1 = 1 - proton1->pz()/Ebeam;
		hXiGen->Fill(xigen1);
		hProtonPt2->Fill(proton1->pt()*proton1->pt());
  }	

  if(proton2 != genParticles->end()){
		if(debug) std::cout << "Proton 2: " << proton2->pt() << "  " << proton2->eta() << "  " << proton2->phi() << std::endl;	
   		double xigen2 = 1 + proton2->pz()/Ebeam;
        	hXiGen->Fill(xigen2);
		hProtonPt2->Fill(proton2->pt()*proton2->pt());
  }

  edm::Handle<reco::GenJetCollection> genJets;
  ev.getByLabel(genJetsTag_,genJets);
  if(genJets->size() < 2) return;

  /*reco::GenJetCollection::const_iterator jet1 = genJets->end();
  reco::GenJetCollection::const_iterator jet2 = genJets->end();
  double firstpt = -1.;
  double secondpt = -1.;	
  for(reco::GenJetCollection::const_iterator genjet = genJets->begin(); genjet != genJets->end(); ++genjet){
	if(genjet->pt() > firstpt){
		firstpt = genjet->pt();
		jet2 = jet1;
		jet1 = genjet;
	} else if(genjet->pt() > secondpt){
		secondpt = genjet->pt();
		jet2 = genjet;
	}
  }*/
  /*std::vector<std::pair<reco::GenJetCollection::const_iterator,double> > genjetvec;	
  int jetcount = 0;
  for(reco::GenJetCollection::const_iterator genjet = genJets->begin(); genjet != genJets->end(); ++genjet){
	if(debug) std::cout << " Jet " << jetcount++ << " pt: " << genjet->pt() << std::endl;
	genjetvec.push_back(std::make_pair(genjet,genjet->pt()));
  }
  std::sort(genjetvec.begin(),genjetvec.end(),myjetptcomp);
  reco::GenJetCollection::const_iterator jet1 = genjetvec[0].first;
  reco::GenJetCollection::const_iterator jet2 = genjetvec[1].first;
  if(debug){
  	std::cout << ">>> After sorting: " << std::endl;
  	for(size_t k = 0; k < genjetvec.size(); ++k) std::cout << " Jet " << k << " pt: " << genjetvec[k].second << std::endl;
  }*/
	
  const reco::GenJet* jet1 = &(*genJets)[0];
  const reco::GenJet* jet2 = &(*genJets)[1];

  if(jet1&&jet2){
	if(debug) std::cout << ">>> Leading Jet pt,eta: " << jet1->pt() << " , " << jet1->eta() << std::endl;
	hJet1Pt->Fill(jet1->pt());
	hJet1Eta->Fill(jet1->eta());
	hJet1Phi->Fill(jet1->phi());

	if(debug) std::cout << ">>> Second leading Jet pt,eta: " << jet2->pt() << " , " << jet2->eta() << std::endl;
        hJet2Pt->Fill(jet2->pt());
        hJet2Eta->Fill(jet2->eta());
        hJet2Phi->Fill(jet2->phi());

	hJetDeltaEta->Fill(jet1->eta() - jet2->eta());
	hJetDeltaPhi->Fill(jet1->phi() - jet2->phi());
	hJetDeltaPt->Fill(jet1->pt() - jet2->pt());	
  }

  //Calculate Thrust
  edm::Handle<edm::View<reco::Candidate> > genParticlesVisible;
  ev.getByLabel("genParticlesVisible", genParticlesVisible);
  Thrust mythrust(genParticlesVisible->begin(),genParticlesVisible->end());
  double thrustValue = mythrust.thrust();
  if(debug) std::cout << ">>> Event Thrust: " << thrustValue << std::endl;
  hThrust->Fill(thrustValue);
				
}

DEFINE_FWK_MODULE(SDDijetsAnalyzer);
