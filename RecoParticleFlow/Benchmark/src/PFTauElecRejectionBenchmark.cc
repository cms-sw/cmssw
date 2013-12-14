#include "RecoParticleFlow/Benchmark/interface/PFTauElecRejectionBenchmark.h"

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  h##name = db_ ? db_->book1D(#name,title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  h##name = db_ ? db_->book2D(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)

// all versions OK
// preprocesor macro for setting axis titles
#define SETAXES(name,xtitle,ytitle) \
  h##name->GetXaxis()->SetTitle(xtitle); h##name->GetYaxis()->SetTitle(ytitle)


using namespace reco;
using namespace std;

class MonitorElement;

PFTauElecRejectionBenchmark::PFTauElecRejectionBenchmark() : file_(0) {}

PFTauElecRejectionBenchmark::~PFTauElecRejectionBenchmark() {
  if(file_) file_->Close();
}

void PFTauElecRejectionBenchmark::write() {
   // Store the DAQ Histograms 
  if (outputFile_.size() != 0) {
    if (db_)
          db_->save(outputFile_.c_str());
    // use bare Root if no DQM (FWLite applications)
    else if (file_) {
       file_->Write(outputFile_.c_str());
       cout << "Benchmark output written to file " << outputFile_.c_str() << endl;
       file_->Close();
       }
  }
  else 
    cout << "No output file specified ("<<outputFile_<<"). Results will not be saved!" << endl;
    
} 

void PFTauElecRejectionBenchmark::setup(
					string Filename,
					string benchmarkLabel,
					double maxDeltaR, 
					double minRecoPt, 
					double maxRecoAbsEta, 
					double minMCPt, 
					double maxMCAbsEta, 
					string sGenMatchObjectLabel,
					bool applyEcalCrackCut,
					DQMStore * db_store) {
  maxDeltaR_ = maxDeltaR;
  benchmarkLabel_ = benchmarkLabel;
  outputFile_=Filename;
  minRecoPt_ = minRecoPt;
  maxRecoAbsEta_= maxRecoAbsEta;
  minMCPt_ = minMCPt;
  maxMCAbsEta_= maxMCAbsEta;
  sGenMatchObjectLabel_ = sGenMatchObjectLabel;
  applyEcalCrackCut_= applyEcalCrackCut;

  file_ = NULL;
  db_ = db_store;

  // print parameters
  cout<< "PFTauElecRejectionBenchmark Setup parameters =============================================="<<endl;
  cout << "Filename to write histograms " << Filename<<endl;
  cout << "Benchmark label name " << benchmarkLabel_<<endl;
  cout << "maxDeltaRMax " << maxDeltaR << endl;
  cout << "minRecoPt " << minRecoPt_ << endl;
  cout << "maxRecoAbsEta " << maxRecoAbsEta_ << endl;
  cout << "minMCPt " << minMCPt_ << endl;
  cout << "maxMCAbsEta " << maxMCAbsEta_ << endl;
  cout << "applyEcalCrackCut " << applyEcalCrackCut_ << endl;
  cout << "sGenMatchObjectLabel " << sGenMatchObjectLabel_ << endl;

  // Book histogram

  // Establish DQM Store
  string path = "PFTask/Benchmarks/"+ benchmarkLabel_ + "/";
  path += "Gen";
  if (db_) {
    db_->setCurrentFolder(path.c_str());
  }
  else {
    file_ = new TFile(outputFile_.c_str(), "recreate");
    cout << "Info: DQM is not available to provide data storage service. Using TFile to save histograms. "<<endl;
  }

  // E/p
  BOOK1D(EoverP,"E/p",100, 0., 4.);
  SETAXES(EoverP, "E/p", "Entries");

  BOOK1D(EoverP_barrel,"E/p barrel",100, 0., 4.);
  SETAXES(EoverP_barrel, "E/p barrel", "Entries");

  BOOK1D(EoverP_endcap,"E/p endcap",100, 0., 4.);
  SETAXES(EoverP_endcap, "E/p endcap", "Entries");

  BOOK1D(EoverP_preid0,"E/p (preid=0)",100, 0., 4.);
  SETAXES(EoverP_preid0, "E/p", "Entries");

  BOOK1D(EoverP_preid1,"E/p (preid=1)",100, 0., 4.);
  SETAXES(EoverP_preid1, "E/p", "Entries");

  // H/p
  BOOK1D(HoverP,"H/p",100, 0., 2.);
  SETAXES(HoverP, "H/p", "Entries");

  BOOK1D(HoverP_barrel,"H/p barrel",100, 0., 2.);
  SETAXES(HoverP_barrel, "H/p barrel", "Entries");

  BOOK1D(HoverP_endcap,"H/p endcap",100, 0., 2.);
  SETAXES(HoverP_endcap, "H/p endcap", "Entries");

  BOOK1D(HoverP_preid0,"H/p (preid=0)",100, 0., 2.);
  SETAXES(HoverP_preid0, "H/p", "Entries");

  BOOK1D(HoverP_preid1,"H/p (preid=1)",100, 0., 2.);
  SETAXES(HoverP_preid1, "H/p", "Entries");

  // emfrac
  BOOK1D(Emfrac,"EM fraction",100, 0., 1.01);
  SETAXES(Emfrac, "em fraction", "Entries");

  BOOK1D(Emfrac_barrel,"EM fraction barrel",100, 0., 1.01);
  SETAXES(Emfrac_barrel, "em fraction barrel", "Entries");

  BOOK1D(Emfrac_endcap,"EM fraction endcap",100, 0., 1.01);
  SETAXES(Emfrac_endcap, "em fraction endcap", "Entries");

  BOOK1D(Emfrac_preid0,"EM fraction (preid=0)",100, 0., 1.01);
  SETAXES(Emfrac_preid0, "em fraction", "Entries");

  BOOK1D(Emfrac_preid1,"EM fraction (preid=1)",100, 0., 1.01);
  SETAXES(Emfrac_preid1, "em fraction", "Entries");

  // PreID
  BOOK1D(ElecPreID,"PFElectron PreID decision",6, 0., 1.01);
  SETAXES(ElecPreID, "PFElectron PreID decision", "Entries");

  // MVA
  BOOK1D(ElecMVA,"PFElectron MVA",100, -1.01, 1.01);
  SETAXES(ElecMVA, "PFElectron MVA", "Entries");

  // Discriminant
  BOOK1D(TauElecDiscriminant,"PFTau-Electron Discriminant",6, 0., 1.01);
  SETAXES(TauElecDiscriminant, "PFTau-Electron Discriminant", "Entries");


  // PFCand clusters
  BOOK1D(pfcand_deltaEta,"PFCand cluster dEta",100, 0., 0.8);
  SETAXES(pfcand_deltaEta, "PFCand cluster #Delta(#eta)", "Entries");

  BOOK1D(pfcand_deltaEta_weightE,"PFCand cluster dEta, energy weighted",100, 0., 0.8);
  SETAXES(pfcand_deltaEta_weightE, "PFCand cluster #Delta(#eta)", "Entries");

  BOOK1D(pfcand_deltaPhiOverQ,"PFCand cluster dPhi/q",100, -0.8, 0.8);
  SETAXES(pfcand_deltaPhiOverQ, "PFCand cluster #Delta(#phi)/q", "Entries");

  BOOK1D(pfcand_deltaPhiOverQ_weightE,"PFCand cluster dEta/q, energy weighted",100, -0.8, 0.8);
  SETAXES(pfcand_deltaPhiOverQ_weightE, "PFCand cluster #Delta(#phi)/q", "Entries");


  // Leading KF track
  BOOK1D(leadTk_pt,"leading KF track pt",100, 0., 80.);
  SETAXES(leadTk_pt, "leading KF track p_{T} (GeV)", "Entries");

  BOOK1D(leadTk_eta,"leading KF track eta",100, -4., 4.);
  SETAXES(leadTk_eta, "leading KF track #eta", "Entries");

  BOOK1D(leadTk_phi,"leading KF track phi",100, -3.2, 3.2);
  SETAXES(leadTk_phi, "leading KF track #phi", "Entries");

  // Leading Gsf track
  BOOK1D(leadGsfTk_pt,"leading Gsf track pt",100, 0., 80.);
  SETAXES(leadGsfTk_pt, "leading Gsf track p_{T} (GeV)", "Entries");

  BOOK1D(leadGsfTk_eta,"leading Gsf track eta",100, -4., 4.);
  SETAXES(leadGsfTk_eta, "leading Gsf track #eta", "Entries");

  BOOK1D(leadGsfTk_phi,"leading Gsf track phi",100, -3.2, 3.2);
  SETAXES(leadGsfTk_phi, "leading Gsf track #phi", "Entries");


  // H/p vs E/p
  BOOK2D(HoPvsEoP,"H/p vs. E/p",100, 0., 2., 100, 0., 2.);
  SETAXES(HoPvsEoP, "E/p", "H/p");

  BOOK2D(HoPvsEoP_preid0,"H/p vs. E/p (preid=0)",100, 0., 2., 100, 0., 2.);
  SETAXES(HoPvsEoP_preid0, "E/p", "H/p");

  BOOK2D(HoPvsEoP_preid1,"H/p vs. E/p (preid=0)",100, 0., 2., 100, 0., 2.);
  SETAXES(HoPvsEoP_preid1, "E/p", "H/p");

  // em fraction vs E/p
  BOOK2D(EmfracvsEoP,"emfrac vs. E/p",100, 0., 2., 100, 0., 1.01);
  SETAXES(EmfracvsEoP, "E/p", "em fraction");

  BOOK2D(EmfracvsEoP_preid0,"emfrac vs. E/p (preid=0)",100, 0., 2., 100, 0., 1.01);
  SETAXES(EmfracvsEoP_preid0, "E/p", "em fraction");

  BOOK2D(EmfracvsEoP_preid1,"emfrac vs. E/p (preid=0)",100, 0., 2., 100, 0., 1.01);
  SETAXES(EmfracvsEoP_preid1, "E/p", "em fraction");


}


void PFTauElecRejectionBenchmark::process(edm::Handle<edm::HepMCProduct> mcevt, edm::Handle<reco::PFTauCollection> pfTaus, 
					  edm::Handle<reco::PFTauDiscriminator> pfTauIsoDiscr, 
					  edm::Handle<reco::PFTauDiscriminator> pfTauElecDiscr) {


  // Find Gen Objects to be matched with
  HepMC::GenEvent * generated_event = new HepMC::GenEvent(*(mcevt->GetEvent()));
  _GenObjects.clear();
  
  TLorentzVector taunet;
  HepMC::GenEvent::particle_iterator p;
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    if(std::abs((*p)->pdg_id()) == 15&&(*p)->status()==2) { 
      bool lept_decay = false;     
      TLorentzVector tau((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
      HepMC::GenVertex::particle_iterator z = (*p)->end_vertex()->particles_begin(HepMC::descendants);
      for(; z != (*p)->end_vertex()->particles_end(HepMC::descendants); z++) {
	if(std::abs((*z)->pdg_id()) == 11 || std::abs((*z)->pdg_id()) == 13) lept_decay=true;
	if(std::abs((*z)->pdg_id()) == 16)
	  taunet.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	
      }
      if(lept_decay==false) {
	TLorentzVector jetMom=tau-taunet;
	if (sGenMatchObjectLabel_=="tau") _GenObjects.push_back(jetMom);
      }
    } else if(std::abs((*p)->pdg_id()) == 11&&(*p)->status()==1) { 
      TLorentzVector elec((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
      if (sGenMatchObjectLabel_=="e") _GenObjects.push_back(elec);
    } 
  }
 
  ////////////

  
  // Loop over all PFTaus
  math::XYZPointF myleadTkEcalPos;
  for (PFTauCollection::size_type iPFTau=0;iPFTau<pfTaus->size();iPFTau++) { 
    PFTauRef thePFTau(pfTaus,iPFTau); 
    if ((*pfTauIsoDiscr)[thePFTau] == 1) {
      if ((*thePFTau).et() > minRecoPt_ && std::abs((*thePFTau).eta()) < maxRecoAbsEta_) {

	// Check if track goes to Ecal crack
	TrackRef myleadTk;
	if(thePFTau->leadPFChargedHadrCand().isNonnull()){
	  myleadTk=thePFTau->leadPFChargedHadrCand()->trackRef();
	  myleadTkEcalPos = thePFTau->leadPFChargedHadrCand()->positionAtECALEntrance();
	  
	  if(myleadTk.isNonnull()){ 
	    if (applyEcalCrackCut_ && isInEcalCrack(std::abs((double)myleadTkEcalPos.eta()))) {
	      continue; // do nothing
	    } else {

	      // Match with gen object
	      for (unsigned int i = 0; i<_GenObjects.size();i++) {
		if (_GenObjects[i].Et() >= minMCPt_ && std::abs(_GenObjects[i].Eta()) < maxMCAbsEta_ ) {
		  TLorentzVector pftau((*thePFTau).px(),(*thePFTau).py(),(*thePFTau).pz(),(*thePFTau).energy());
		  double GenDeltaR = pftau.DeltaR(_GenObjects[i]);
		  if (GenDeltaR<maxDeltaR_) {
		    
		    hleadTk_pt->Fill((float)myleadTk->pt());
		    hleadTk_eta->Fill((float)myleadTk->eta());
		    hleadTk_phi->Fill((float)myleadTk->phi());

		    hEoverP->Fill((*thePFTau).ecalStripSumEOverPLead());
		    hHoverP->Fill((*thePFTau).hcal3x3OverPLead());
		    hEmfrac->Fill((*thePFTau).emFraction());

		    if (std::abs(myleadTk->eta())<1.5) {
		      hEoverP_barrel->Fill((*thePFTau).ecalStripSumEOverPLead());
		      hHoverP_barrel->Fill((*thePFTau).hcal3x3OverPLead());
		      hEmfrac_barrel->Fill((*thePFTau).emFraction());
		    } else if (std::abs(myleadTk->eta())>1.5 && std::abs(myleadTk->eta())<2.5) {
		      hEoverP_endcap->Fill((*thePFTau).ecalStripSumEOverPLead());
		      hHoverP_endcap->Fill((*thePFTau).hcal3x3OverPLead());
		      hEmfrac_endcap->Fill((*thePFTau).emFraction());
		    }

		    // if -999 fill in -1 bin!
		    if ((*thePFTau).electronPreIDOutput()<-1)
		      hElecMVA->Fill(-1);
		    else
		      hElecMVA->Fill((*thePFTau).electronPreIDOutput());

		    hTauElecDiscriminant->Fill((*pfTauElecDiscr)[thePFTau]);

		    hHoPvsEoP->Fill((*thePFTau).ecalStripSumEOverPLead(),(*thePFTau).hcal3x3OverPLead());
		    hEmfracvsEoP->Fill((*thePFTau).emFraction(),(*thePFTau).hcal3x3OverPLead());

		    if ((*thePFTau).electronPreIDDecision()==1) {
		      hEoverP_preid1->Fill((*thePFTau).ecalStripSumEOverPLead());
		      hHoverP_preid1->Fill((*thePFTau).hcal3x3OverPLead());
		      hEmfrac_preid1->Fill((*thePFTau).emFraction());
		      hHoPvsEoP_preid1->Fill((*thePFTau).ecalStripSumEOverPLead(),(*thePFTau).hcal3x3OverPLead());
		      hEmfracvsEoP_preid1->Fill((*thePFTau).emFraction(),(*thePFTau).hcal3x3OverPLead());
		    } else {
		      hEoverP_preid0->Fill((*thePFTau).ecalStripSumEOverPLead());
		      hHoverP_preid0->Fill((*thePFTau).hcal3x3OverPLead());
		      hEmfrac_preid0->Fill((*thePFTau).emFraction());
		      hHoPvsEoP_preid0->Fill((*thePFTau).ecalStripSumEOverPLead(),(*thePFTau).hcal3x3OverPLead());
		      hEmfracvsEoP_preid0->Fill((*thePFTau).emFraction(),(*thePFTau).hcal3x3OverPLead());
		    }


		  }
		}

		// Loop over all PFCands for cluster plots  
		std::vector<PFCandidatePtr> myPFCands=(*thePFTau).pfTauTagInfoRef()->PFCands();
		for(int i=0;i<(int)myPFCands.size();i++){

		  math::XYZPointF candPos;
		  if (myPFCands[i]->particleId()==1 || myPFCands[i]->particleId()==2) // if charged hadron or electron
		    candPos = myPFCands[i]->positionAtECALEntrance();
		  else
		    candPos = math::XYZPointF(myPFCands[i]->px(),myPFCands[i]->py(),myPFCands[i]->pz());

		  //double deltaR   = ROOT::Math::VectorUtil::DeltaR(myleadTkEcalPos,candPos);
		  double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(myleadTkEcalPos,candPos);
		  double deltaEta = std::abs(myleadTkEcalPos.eta()-myPFCands[i]->eta());
		  double deltaPhiOverQ = deltaPhi/(double)myleadTk->charge();
		  
		  hpfcand_deltaEta->Fill(deltaEta);
		  hpfcand_deltaEta_weightE->Fill(deltaEta*myPFCands[i]->ecalEnergy());
		  hpfcand_deltaPhiOverQ->Fill(deltaPhiOverQ);
		  hpfcand_deltaPhiOverQ_weightE->Fill(deltaPhiOverQ*myPFCands[i]->ecalEnergy());	
		  
		}

	      }

	    }
	  }
	}

      }
    }
  }
}

// Ecal crack  map from Egamma POG
bool PFTauElecRejectionBenchmark::isInEcalCrack(double eta) const{  
  return (eta < 0.018 || 
	  (eta>0.423 && eta<0.461) ||
	  (eta>0.770 && eta<0.806) ||
	  (eta>1.127 && eta<1.163) ||
	  (eta>1.460 && eta<1.558));
}
