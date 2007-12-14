#include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"


#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace reco;
using namespace std;

PFJetBenchmark::PFJetBenchmark() {}

PFJetBenchmark::~PFJetBenchmark() {}

void PFJetBenchmark::setup(
               string Filename,
               bool debug, 
			   bool PlotAgainstReco, 
			   double deltaRMax  ) {

  file_ = TFile::Open(Filename.c_str(), "recreate");
  debug_ = debug; 
  PlotAgainstReco_ = PlotAgainstReco; 
  deltaRMax_ = deltaRMax;
// print parameters
  cout<< "PFJetBenchmark Setup parameters =============================================="<<endl;
  cout << "Filename ot write histograms " << Filename<<endl;
  cout << "PFJetBenchmark debug " << debug_<< endl;
  cout << "PlotAgainstReco " << PlotAgainstReco_ << endl;
  cout << "deltaRMax " << deltaRMax << endl;
// Book histograms
// delta et quantities
  hDeltaEt = new TH1F("DeltaEt","DeltaEt",1000,-100,100);
  hDeltaEch = new TH1F("DeltaEch","DeltaE charged",1000,-100,100);
  hDeltaEem = new TH1F("DeltaEem","DeltaE elm",1000,-100,100);
  hDeltaEtvsEt = new TH2F("DeltaEtvsEt","DeltaEtvsEt",1000,0,1000,1000,-100,100);
  hDeltaEtOverEtvsEt = new TH2F("DeltaEtOverEtvsEt","DeltaEtOverEtvsEt",1000,0,1000,100,-1,1);
  hDeltaEtvsEta = new TH2F("DeltaEtvsEta","DeltaEtvsEta",200,-5,5,1000,-100,100);
  hDeltaEtOverEtvsEta = new TH2F("DeltaEtOverEtvsEta","DeltaEtOverEtvsEta",200,-5,5,100,-1,1);

  hDeltaR = new TH1F("DeltaR","DeltaR",100,0,1);
  hDeltaRvsEt = new TH2F("DeltaRvsEt","DeltaRvsEt",1000,0,1000,100,0,1);
  
}

 
void PFJetBenchmark::process(const reco::PFJetCollection& pfJets, const reco::GenJetCollection& genJets) {

    //if(debug_){
    //cout<<"PFJetBenchmark::process---------- Particle Flow Jets: "<<endl;
    //for(unsigned i=0; i<pfJets.size(); i++) {      
    //  cout<<i<<pfJets[i].print()<<endl;
    //}    
    //cout<<endl;
    //cout<<"PFJetBenchmark::process---------- Generated Jets: "<<endl;
    //for(unsigned i=0; i<genJets.size(); i++) {      
    //  cout<<i<<genJets[i].print()<<endl;
    //}
    //}// debug
     // loop over reco  pf  jets
	 if (debug_){ // print highest PT jets
	if(pfJets.size()!=0)cout<<pfJets[0].print()<<endl;
	 //if(genJets.size()!=0)cout<<genJets[0].print()<<endl;
	 //if(pfJets.size()!=0)printPFJet(&pfJets[0]);
	 if(genJets.size()!=0)printGenJet(&genJets[0]);
     }
   deltaEtMax_ = 0.;
   deltaChargedEnergyMax_ = 0.;
   deltaEmEnergyMax_ = 0.; 
   for(unsigned i=0; i<pfJets.size(); i++) {   
   
    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
   const reco::PFJet& pfj = pfJets[i];
   const GenJet *truth = algo_->matchByDeltaR(&pfj,&genJets);
   if(!truth) continue;
    
    // get the quantities to place on the denominator and/or divide by
   double et, eta, phi;
  if (PlotAgainstReco_) { et = pfj.et(); eta = pfj.eta(); phi = pfj.phi(); }
    else { et = truth->et(); eta = truth->eta(); phi = truth->phi(); }
    // get specific quantities
	double true_chargedEnergy;
	double true_EmEnergy;
	gettrue (truth, true_chargedEnergy, true_EmEnergy);
	double rec_chargedEnergy = pfj.chargedHadronEnergy();
	double rec_EmEnergy = pfj.neutralEmEnergy();
	double deltaR = algo_->deltaR(&pfj, truth);
	if(deltaR < deltaRMax_) {//start case deltaR < deltaRMax
	// get other delta quantities
	double deltaChargedEnergy = rec_chargedEnergy- true_chargedEnergy;
	double deltaEmEnergy = rec_EmEnergy- true_EmEnergy;
     double deltaEt = algo_->deltaEt(&pfj, truth);
     
     //double deltaEta = algo_->deltaEta(&pfj, truth);
     //double deltaPhi = algo_->deltaPhi(&pfj, truth);
	 if(abs(deltaEt) > deltaEtMax_) deltaEtMax_ = abs(deltaEt);
	 if(abs(deltaChargedEnergy) > deltaChargedEnergyMax_) deltaChargedEnergyMax_ = abs(deltaChargedEnergy);
	 if(abs(deltaEmEnergy) > deltaEmEnergyMax_) deltaEmEnergyMax_ = abs(deltaEmEnergy);
	 if (debug_) {
	 cout << i <<"  =========PFJet Et "<< pfj.et()
	      << " eta " << pfj.eta()
		  << " phi " << pfj.phi() 
		  << " Charged Energy " << rec_chargedEnergy
		  << " elm Energy " << rec_EmEnergy << endl;
     cout << " matching Gen Jet Et " << truth->et()
	      << " eta " << truth->eta()
		  << " phi " << truth->phi()
		  << " Charged Energy " << true_chargedEnergy
		  << " elm Energy " << true_EmEnergy << endl;
		 
	 cout << "==============deltaR " << deltaR << "  deltaEt " << deltaEt
	      << " deltaChargedEnergy " << deltaChargedEnergy
		  << " deltaEmEnergy " << deltaEmEnergy << endl;
	 }
    
    // fill histograms for delta quantitites of matched jets
		hDeltaEt->Fill(deltaEt);
	   hDeltaEch->Fill(deltaChargedEnergy);
	   hDeltaEem->Fill(deltaEmEnergy);
        hDeltaEtvsEt->Fill(et,deltaEt);
        hDeltaEtOverEtvsEt->Fill(et,deltaEt/et);
        hDeltaEtvsEta->Fill(eta,deltaEt);
      hDeltaEtOverEtvsEta->Fill(eta,deltaEt/et);
    } // end case deltaR < deltaRMax

    hDeltaR->Fill(deltaR);
    hDeltaRvsEt->Fill(et,deltaR);

  } // i loop on pf Jets

}
void PFJetBenchmark::gettrue (const reco::GenJet* truth, double& true_chargedEnergy, double& true_EmEnergy){
  std::vector <const GenParticleCandidate*> mcparts = truth->getConstituents ();
  true_EmEnergy = 0.;
  true_chargedEnergy = 0.;
// for each MC particle in turn  
  for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticleCandidate* mcpart = mcparts[i];
    int PDG = abs( mcpart->pdgId());
	double e = mcpart->energy(); 
    switch(PDG){  // start PDG switch
    case 22: // photon
	true_EmEnergy += e;
	break;
	case 211: // pi
	case 321: // K
	case 2212: // p
    true_chargedEnergy += e;
	break;
	default:
	break;
	}  // end PDG switch
 
  }  // end loop on constituents.

}
//  void PFJetBenchmark::printPFJet(const reco::PFJet* pfj){
//  	std::vector <const PFCandidate> pfCandidates = pfj->getConstituents ();
//  	cout << "PFJet  p/px/py/pz/pt: " << pfj->p() << '/' << pfj->px () 
//  		<< '/' << pfj->py() << '/' << pfj->pz() << '/' << pfj->pt() << endl
//  		<< "    eta/phi: " << pfj->eta () << '/' << pfj->phi () << endl
//  		<< "    # of pfCandidates: " << pfCandidates.size() << endl;
//  	cout << "    pfCandidates:" << endl;
//  	for(unsigned i=0; i<pfCandidates.size(); i++) {
//        cout<<i<<" " << pfCandidates[i]<<endl;
//      }    
//  }
void PFJetBenchmark::printGenJet (const reco::GenJet* truth){
	std::vector <const GenParticleCandidate*> mcparts = truth->getConstituents ();
	cout << "GenJet p/px/py/pz/pt: " << truth->p() << '/' << truth->px () 
		<< '/' << truth->py() << '/' << truth->pz() << '/' << truth->pt() << endl
		<< "    eta/phi: " << truth->eta () << '/' << truth->phi () << endl
		<< "    # of constituents: " << mcparts.size() << endl;
	cout << "    constituents:" << endl;
	for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticleCandidate* mcpart = mcparts[i];
    cout << "      #" << i << "  PDG code:" << mcpart->pdgId() 
		 << ", p/pt/eta/phi: " << mcpart->p() << '/' << mcpart->pt() 
		 << '/' << mcpart->eta() << '/' << mcpart->phi() << endl;

    }    
}

void PFJetBenchmark::write() {

  if (file_){
    cout<<"writing PFBenchmark Histos to "<<file_->GetName()<<endl;
    file_->Write();
	}
}
