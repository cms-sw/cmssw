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
// Jets inclusive  distributions  (Pt > 20 GeV)
  hNjets = new TH1F("hNjets","Jet multiplicity Pt > 20 GeV",50, 0, 50);
  hjetsPt = new TH1F("hjetsPt","Jets Pt Distribution",100, 0, 500);
  hjetsEta = new TH1F("hjetsEta","Jets Eta Distribution",100, -5, 5);

// delta Pt or E quantities for Barrel
  hBRPt = new TH1F("hBRPt","Barrel ResolPt",80,-2,2);
  hBRCHE = new TH1F("hBRCHE","Barrel Resol charged had Energy",80,-2,2);
  hBRNHE = new TH1F("hBRNHE","Barrel Resol neutral had Energy",80,-2,2);
  hBRNEE = new TH1F("hBRNEE","Barrel Resol neutral elm Energy",80,-2,2);
  hBRPtvsPt = new TH2F("hBRPtvsPt","Barrel ResolPt vs Pt",50, 0, 500, 80,-2,2);
  hBRCHEvsPt = new TH2F("hBRCHEvsPt","Barrel Resol charged had Energy vs Pt",50, 0, 500, 80,-2,2);
  hBRNHEvsPt = new TH2F("hBRNHEvsPt","Barrel Resol neutral had Energy vs Pt",50, 0, 500, 80,-2,2);
  hBRNEEvsPt = new TH2F("hBRNEEvsPt","Barrel Resol neutral elm Energyvs Pt",50, 0, 500, 80,-2,2);
  
  // delta Pt or E quantities for Endcap
  hERPt = new TH1F("hERPt","Endcap ResolPt",80,-2,2);
  hERCHE = new TH1F("hERCHE","Endcap Resol charged had Energy",80,-2,2);
  hERNHE = new TH1F("hERNHE","Endcap Resol neutral had Energy",80,-2,2);
  hERNEE = new TH1F("hERNEE","Endcap Resol neutral elm Energy",80,-2,2);
  hERPtvsPt = new TH2F("hERPtvsPt","Endcap ResolPt vs Pt",50, 0, 500, 80,-2,2);
  hERCHEvsPt = new TH2F("hERCHEvsPt","Endcap Resol charged had Energy vs Pt",50, 0, 500, 80,-2,2);
  hERNHEvsPt = new TH2F("hERNHEvsPt","Endcap Resol neutral had Energy vs Pt",50, 0, 500, 80,-2,2);
  hERNEEvsPt = new TH2F("hERNEEvsPt","Endcap Resol neutral elm Energyvs Pt",50, 0, 500, 80,-2,2);

    
}

 
void PFJetBenchmark::process(const reco::PFJetCollection& pfJets, const reco::GenJetCollection& genJets) {
//    	if(debug_){
//	cout<<"PFJetBenchmark::process---------- Particle Flow Jets: "<<endl;
//      for(unsigned i=0; i<pfJets.size(); i++) { 
//	 	  cout<<i<<pfJets[i].print()<<endl;
//          printPFJet(&pfJets[i]);
//      }    
//   cout<<endl;
//   cout<<"PFJetBenchmark::process---------- Generated Jets: "<<endl;
//      for(unsigned i=0; i<genJets.size(); i++) { 
//  	  cout<<i<<genJets[i].print()<<endl;
//	      printGenJet(&genJets[i]);
//      }
//   }// debug
	
     // loop over reco  pf  jets
   resPtMax_ = 0.;
   resChargedHadEnergyMax_ = 0.;
   resNeutralHadEnergyMax_ = 0.;
   resNeutralEmEnergyMax_ = 0.; 
   int NPFJets = 0;

   for(unsigned i=0; i<pfJets.size(); i++) {   
   
   
   const reco::PFJet& pfj = pfJets[i];
   double rec_pt = pfj.pt();
   double rec_eta = pfj.eta();
   double rec_phi = pfj.phi();
   // skip PFjets with pt < 20 GeV
   if (rec_pt<20.) continue;
   NPFJets++;
   
   // fill inclusive PFjet distribution pt > 20 GeV
   hNjets->Fill(NPFJets);
   hjetsPt->Fill(rec_pt);
   hjetsEta->Fill(rec_eta);
   
   // separate Barrel PFJets from Endcap PFJets
   bool Barrel = false;
   bool Endcap = false;
   if (abs(rec_eta) < 1.4 ) Barrel = true;
   if (abs (rec_eta) > 1.6 && abs (rec_eta) < 3. ) Endcap = true;
   // look for the closets gen Jet : truth
   const GenJet *truth = algo_->matchByDeltaR(&pfj,&genJets);
   if(!truth) continue;   
	double deltaR = algo_->deltaR(&pfj, truth);
	// check deltaR is small enough
	if(deltaR < deltaRMax_) {//start case deltaR < deltaRMax
	// generate histograms comparing the reco and truth candidate (truth = closest in delta-R) 
    // get the quantities to place on the denominator and/or divide by
    //double pt_denom;
	double true_pt = truth->pt();
    //if (PlotAgainstReco_) {pt_denom = rec_pt}
    //else {pt_denom = true_pt;}
    // get true specific quantities
	double true_ChargedHadEnergy;
	double true_NeutralHadEnergy;
	double true_NeutralEmEnergy;
	gettrue (truth, true_ChargedHadEnergy, true_NeutralHadEnergy, true_NeutralEmEnergy);
	double rec_ChargedHadEnergy = pfj.chargedHadronEnergy();
	double rec_NeutralHadEnergy = pfj.neutralHadronEnergy();
	double rec_NeutralEmEnergy = pfj.neutralEmEnergy();
    bool plot1 = false;
	bool plot2 = false;
	bool plot3 = false;
	bool plot4 = false;
	double resPt =0.;
	double resChargedHadEnergy= 0.;
	double resNeutralHadEnergy= 0.;
	double resNeutralEmEnergy= 0.;
	// get relative delta quantities (protect against division by zero!)
	if (true_pt > 0.0001){
	resPt = (rec_pt -true_pt)/true_pt ; 
	plot1 = true;}
	if (true_ChargedHadEnergy > 0.0001){
	resChargedHadEnergy = (rec_ChargedHadEnergy- true_ChargedHadEnergy)/true_ChargedHadEnergy;
	plot2 = true;}
	if (true_NeutralHadEnergy > 0.0001){
	resNeutralHadEnergy = (rec_NeutralHadEnergy- true_NeutralHadEnergy)/true_NeutralHadEnergy;
	plot3 = true;}
	else 
	if (rec_NeutralHadEnergy > 0.0001){
	resNeutralHadEnergy = (rec_NeutralHadEnergy- true_NeutralHadEnergy)/rec_NeutralHadEnergy;
	plot3 = true;}
	if (true_NeutralEmEnergy > 0.0001){
	resNeutralEmEnergy = (rec_NeutralEmEnergy- true_NeutralEmEnergy)/true_NeutralEmEnergy;
	plot4 = true;}
     
     
     //double deltaEta = algo_->deltaEta(&pfj, truth);
     //double deltaPhi = algo_->deltaPhi(&pfj, truth);
	 if(abs(resPt) > resPtMax_) resPtMax_ = abs(resPt);
	 if(abs(resChargedHadEnergy) > resChargedHadEnergyMax_) resChargedHadEnergyMax_ = abs(resChargedHadEnergy);
	 if(abs(resNeutralHadEnergy) > resNeutralHadEnergyMax_) resNeutralHadEnergyMax_ = abs(resNeutralHadEnergy);
	 if(abs(resNeutralEmEnergy) > resNeutralEmEnergyMax_) resNeutralEmEnergyMax_ = abs(resNeutralEmEnergy);
	 if (debug_) {
	 cout << i <<"  =========PFJet Pt "<< rec_pt
	      << " eta " << rec_eta
		  << " phi " << rec_phi
		  << " Charged Had Energy " << rec_ChargedHadEnergy
		  << " Neutral Had Energy " << rec_NeutralHadEnergy
		  << " Neutral elm Energy " << rec_NeutralEmEnergy << endl;
     cout << " matching Gen Jet Pt " << true_pt
	      << " eta " << truth->eta()
		  << " phi " << truth->phi()
		  << " Charged Had Energy " << true_ChargedHadEnergy
		  << " Neutral Had Energy " << true_NeutralHadEnergy
		  << " Neutral elm Energy " << true_NeutralEmEnergy << endl;
	 	  printPFJet(&pfj);
	//      cout<<pfj.print()<<endl;
	//	  printGenJet(truth);
	      cout<<truth->print()<<endl;
		 
	 cout << "==============deltaR " << deltaR << "  resPt " << resPt
	      << " resChargedHadEnergy " << resChargedHadEnergy
		  << " resNeutralHadEnergy " << resNeutralHadEnergy

		  << " resNeutralEmEnergy " << resNeutralEmEnergy
	      << endl;
	      	 }
    
    // fill histograms for relative delta quantitites of matched jets
	// delta Pt or E quantities for Barrel
	if (Barrel){
   if(plot1)hBRPt->Fill (resPt);
   if(plot2)hBRCHE->Fill(resChargedHadEnergy);
   if(plot3)hBRNHE->Fill(resNeutralHadEnergy);
   if(plot4)hBRNEE->Fill(resNeutralEmEnergy);
   if(plot1)hBRPtvsPt->Fill(true_pt, resPt);
   if(plot2)hBRCHEvsPt->Fill(true_pt, resChargedHadEnergy);
   if(plot3)hBRNHEvsPt->Fill(true_pt, resNeutralHadEnergy);
   if(plot4)hBRNEEvsPt->Fill(true_pt, resNeutralEmEnergy);
   }
  // delta Pt or E quantities for Endcap
  if (Endcap){
   if(plot1)hERPt->Fill (resPt);
   if(plot2)hERCHE->Fill(resChargedHadEnergy);
   if(plot3)hERNHE->Fill(resNeutralHadEnergy);
   if(plot4)hERNEE->Fill(resNeutralEmEnergy);
   if(plot1)hERPtvsPt->Fill(true_pt, resPt);
   if(plot2)hERCHEvsPt->Fill(true_pt, resChargedHadEnergy);
   if(plot3)hERNHEvsPt->Fill(true_pt, resNeutralHadEnergy);
   if(plot4)hERNEEvsPt->Fill(true_pt, resNeutralEmEnergy);
   }

    
	} // end case deltaR < deltaRMax

  } // i loop on pf Jets

}

void PFJetBenchmark::gettrue (const reco::GenJet* truth, double& true_ChargedHadEnergy, 
  double& true_NeutralHadEnergy, double& true_NeutralEmEnergy){
  std::vector <const GenParticle*> mcparts = truth->getConstituents ();
  true_NeutralEmEnergy = 0.;
  true_ChargedHadEnergy = 0.;
  true_NeutralHadEnergy = 0.;
// for each MC particle in turn  
  for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticle* mcpart = mcparts[i];
    int PDG = abs( mcpart->pdgId());
	double e = mcpart->energy(); 
    switch(PDG){  // start PDG switch
    case 22: // photon
	true_NeutralEmEnergy += e;
	break;
	case 211: // pi
	case 321: // K
	case 2212: // p
	case 11: //electrons (until recognised)
    true_ChargedHadEnergy += e;
	break;
	case 310: // K_S0
	case 130: // K_L0
	case 3122: // Lambda0
	case 2112: // n0
	true_NeutralHadEnergy += e;
	default:
	break;
	}  // end PDG switch
 
  }  // end loop on constituents.

}

   void PFJetBenchmark::printPFJet(const reco::PFJet* pfj){
   cout<<setiosflags(ios::right);
  cout<<setiosflags(ios::fixed);
  cout<<setprecision(3);

   //const reco::Jet* jet = pfj;
  // cout << jet->print();
  	std::vector <const PFCandidate*> pfCandidates = pfj->getConstituents ();
   	cout << "PFJet  p/px/py/pz/pt: " << pfj->p() << '/' << pfj->px () 
   		<< '/' << pfj->py() << '/' << pfj->pz() << '/' << pfj->pt() << endl
   		<< "    eta/phi: " << pfj->eta () << '/' << pfj->phi () << endl   		
        << "    PFJet specific:" << std::endl
        << "      charged/neutral hadrons energy: " << pfj->chargedHadronEnergy () << '/' << pfj->neutralHadronEnergy () << endl
        << "      charged/neutral em energy: " << pfj->chargedEmEnergy () << '/' << pfj->neutralEmEnergy () << endl
        << "      charged muon energy: " << pfj->chargedMuEnergy () << '/' << endl
        << "      charged/neutral multiplicity: " << pfj->chargedMultiplicity () << '/' << pfj->neutralMultiplicity () << endl
		<< "    # of pfCandidates: " << pfCandidates.size() << endl;
   	vector <PFBlockRef> PFBRef;

	//COLIN PFCandidate::block() does not exist anymore
	// 	// print PFCandidates constituents of the jet
	// 	for(unsigned i=0; i<pfCandidates.size(); i++) {
	// 		const PFCandidate* pfCand = pfCandidates[i];
	// 		PFBlockRef blockRef = pfCand->block();
	// 		// store vector of different block refs in the jet	
	// 		bool findref = false;
	// 		for (unsigned k =0; k<PFBRef.size();k++){ // scan blockrefs already stored
	// 		if (PFBRef[k]==blockRef) findref = true;
	// 		}// end loop on k
	// 		if (!findref) PFBRef.push_back(blockRef);
	// 		cout<<i <<" " << *pfCand << endl;
	// 	} // end loop on i (PFCandidates)
	// print blocks associated to the jet


	  cout << "# of blocks in PFJet " <<  PFBRef.size()<< endl;
	  cout << "blocks:"<<endl;
	  // print blocks
	  for (unsigned k =0; k<PFBRef.size();k++){ // for each block in turn
	    // FIXME, JW 
	    //	  cout<< "block id " << PFBRef[k].key() <<  *(PFBRef[k])<<endl;
		}// end loop on k
   cout<<resetiosflags(ios::right|ios::fixed);
   }
void PFJetBenchmark::printGenJet (const reco::GenJet* truth){
	std::vector <const GenParticle*> mcparts = truth->getConstituents ();
	cout << "GenJet p/px/py/pz/pt: " << truth->p() << '/' << truth->px () 
		<< '/' << truth->py() << '/' << truth->pz() << '/' << truth->pt() << endl
		<< "    eta/phi: " << truth->eta () << '/' << truth->phi () << endl
		<< "    # of constituents: " << mcparts.size() << endl;
	cout << "    constituents:" << endl;
	for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticle* mcpart = mcparts[i];
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
