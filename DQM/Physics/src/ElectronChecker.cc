#include "TLorentzVector.h"
#include "DQM/Physics/interface/ElectronChecker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

ElectronChecker::ElectronChecker(const edm::ParameterSet& cfg, const std::string& directory, const std::string& label)
{
  // dqm storage element
  dqmStore_ = edm::Service<DQMStore>().operator->();
  // set directory structure in the dqm storage object
  dqmStore_->setCurrentFolder( directory+"/Electrons_"+label );
}

ElectronChecker::~ElectronChecker()
{
  // free allocated memory
  delete dqmStore_;
}

void 
ElectronChecker::begin(const edm::EventSetup& setup)
{
  // histogram booking
  hists_["d0"]= dqmStore_->book1D("d0", "Impact parameter d0", 100,-0.2,  0.2);
  hists_["d0"]->setAxisTitle("d_{0} (cm)");
  hists_["d0_vs_phi"]= dqmStore_->book2D("d0_vs_phi", "Impact parameter d0 vs #phi", 100, -0.05, 0.05,  100 ,-3.2 ,  3.2 );
  hists_["d0_vs_phi"]->setAxisTitle("d_{0} (cm)");
  hists_["dr03TkSumPt"]= dqmStore_->book1D("dr03TkSumPt", "dr03TkSumPt", 100, 0.0, 50.0);
  hists_["dr03TkSumPt"]->setAxisTitle("dr03TkSumPt (GeV)");
  hists_["dr03EcalRecHitSumEt"]= dqmStore_->book1D("dr03EcalRecHitSumEt", "dr03EcalRecHitSumEt", 100, 0.0, 50.0);
  hists_["dr03EcalRecHitSumEt"]->setAxisTitle("dr03EcalRecHitSumEt (GeV)");
  hists_["dr03HcalTowerSumEt"]= dqmStore_->book1D("dr03HcalTowerSumEt", "dr03HcalTowerSumEt", 100, 0.0, 30.0);
  hists_["dr03HcalTowerSumEt"]->setAxisTitle("dr03HcalTowerSumEt (GeV)");
  hists_["dr03CombRelIso"]= dqmStore_->book1D("dr03CombRelIso", "Combined Relative Isolation", 100, 0.0,  2.0);
  hists_["dr03CombRelIso"]->setAxisTitle("dr03Combined Relative Isoloation "); 
  hists_["NofElectrons"]= dqmStore_->book1D("NofElectrons", "Number of Electrons",  20,   0,   20);
  hists_["NofElectrons"]->setAxisTitle("Number of Electrons",1);
  hists_["ElectronCharge"]= dqmStore_->book1D("ElectronCharge", "Charge of the Electron",   5,  -2,    3);
  hists_["ElectronCharge"]->setAxisTitle("Charge of Electron",1);
 
  //October exercise quantities added:
  hists_["highestpt_pt"]= dqmStore_->book1D("highestpt_pt", "pt of electron with highest pt", 100,   0.0, 200.0);
  hists_["highestpt_pt" ]->setAxisTitle("p_{T} of electron with highest p_{T}",1);
  hists_["highestpt_eta"]= dqmStore_->book1D("highestpt_eta", "#eta of electron with highest pt", 100,  -3.5,   3.5);
  hists_["highestpt_eta"]->setAxisTitle("#eta of electron with highest p_{T}",1);
  hists_["highestpt_phi"]= dqmStore_->book1D("highestpt_phi", "#phi of electron with highest pt", 100,  -3.2,   3.2);
  hists_["highestpt_phi"]->setAxisTitle("#phi of electron with highest p_{T}",1);
  hists_["highestpt_d0"]= dqmStore_->book1D("highestpt_d0", "d0 of electron with highest pt", 100, -0.05,  0.05);
  hists_["highestpt_d0"]->setAxisTitle("d_{0} of electron with highest p_{T}",1);
  hists_["highestpt_trkIso03"]= dqmStore_->book1D("highestpt_trkIso", "Tracker isolation (#Delta R=0.3) of electron with highest pt", 100, 0.0, 50.0);
  hists_["highestpt_trkIso03"]->setAxisTitle("Tracker Iso of electron with highest p_{T}",1); 
  hists_["highestpt_ecalIso04"]= dqmStore_->book1D("highestpt_ecalIso", "ECAL isolation (#Delta R=0.4) of electron with highest pt", 100, 0.0, 50.0);
  hists_["highestpt_ecalIso04"]->setAxisTitle("ECAL Iso of electron with highest p_{T}",1);
  hists_["highestpt_hcalIso04"]= dqmStore_->book1D("highestpt_hcalIso", "HCAL isolation (#Delta R=0.4) of electron with highest pt", 100, 0.0, 30.0);
  hists_["highestpt_hcalIso04"]->setAxisTitle("HCAL Iso of electron with highest p_{T}",1);
  hists_["highestpt_relIso"]= dqmStore_->book1D("highestpt_relIso", "Relative isolation of electron with highest pt", 100, 0.0, 2.0);
  hists_["highestpt_relIso"]->setAxisTitle("Relative Iso of electron with highest p_{T}",1);
  hists_["ElectronRelIsoOverPt"]= dqmStore_->book1D("ElectronRelIsoOverPt", "Relative isolation the electron", 100,  0,  1);
  hists_["ElectronRelIsoOverPt"]->setAxisTitle("ElectronRelIsoOverPt",1);
  hists_["ElectronTrackerRelIsoOverPt"]= dqmStore_->book1D("ElectronTrackerRelIsoOverPt", "Relative tracker isolation the electron", 100,  0,  1);
  hists_["ElectronTrackerRelIsoOverPt"]->setAxisTitle("ElectronTrackerRelIsOverPto",1);
  hists_["ElectronCaloRelIsoOverPt"]= dqmStore_->book1D("ElectronCaloRelIsoOverPt", "Relative isolation the electron", 100,  0,  1);
  hists_["ElectronCaloRelIsoOverPt"]->setAxisTitle("ElectronCaloRelIsoOverPt",1);
  hists_["ElectronRelIso"]= dqmStore_->book1D("ElectronRelIso", "Relative isolation the electron", 100,0,1);
  hists_["ElectronRelIso"]->setAxisTitle("ElectronRelIso",1);
  hists_["ElectronTrackerRelIso"]= dqmStore_->book1D("ElectronTrackerRelIso", "Relative tracker isolation the electron", 100,  0,  1);
  hists_["ElectronTrackerRelIso"]->setAxisTitle("ElectronTrackerRelIso",1);
  hists_["ElectronCaloRelIso"]= dqmStore_->book1D("ElectronCaloRelIso", "Relative isolation the electron", 100,  0,  1);
  hists_["ElectronCaloRelIso"]->setAxisTitle("ElectronCaloRelIso",1);
  hists_["InvMass"]= dqmStore_->book1D( "InvMass","",50, 0, 200);    
  hists_["InvMass"]->setAxisTitle("dilepton invariant mass GeV/c^{2}"); 
  hists_["MaxElPt"]= dqmStore_->book1D( "MaxElPt", "p_{T} of the highest p_{T} electron",  50,  0, 200);
  hists_["MaxElPt"]->setAxisTitle("p_{T}  GeV/c");
  hists_["SecMaxElPt"]= dqmStore_->book1D("SecMaxElPt", "p_{T} of the second highest p_{T} electron",  50,  0, 200);
  hists_["SecMaxElPt"]->setAxisTitle("p_{T} GeV/c");
  hists_["LeptonPairCharge"]= dqmStore_->book1D( "LeptonPairCharge", "Sum of the charges of the 2 highests leptons p_{T}", 5, -2,  2);
  hists_["LeptonPairCharge"]->setAxisTitle("Sum of charges");
}


void
ElectronChecker::analyze(const std::vector<reco::GsfElectron>& Electrons, const Point& beamSpot)
{
  int NofElectrons(0);
  for(unsigned int electron(0); electron<Electrons.size(); ++electron) {
    if(electron==0) hists_["MaxElPt"]    ->Fill(Electrons.at(electron).pt());  
    if(electron==1) hists_["SecMaxElPt"] ->Fill(Electrons.at(electron).pt()); 
    ++NofElectrons;
    hists_["dr03TkSumPt"]->Fill(Electrons.at(electron).dr03TkSumPt()); 
    hists_["dr03EcalRecHitSumEt"]->Fill(Electrons.at(electron).dr03EcalRecHitSumEt());
    hists_["dr03HcalTowerSumEt"]->Fill(Electrons.at(electron).dr03HcalTowerSumEt()); 
    
    hists_["dr03CombRelIso"]->Fill( (Electrons.at(electron).dr03TkSumPt() +  Electrons.at(electron).dr03EcalRecHitSumEt() + Electrons.at(electron).dr03HcalTowerSumEt()) / Electrons.at(electron).pt() );
    
    hists_["d0"]->Fill(Electrons.at(electron).gsfTrack()->dxy(beamSpot));
    hists_["d0_vs_phi"]->Fill(Electrons.at(electron).gsfTrack()->dxy(beamSpot), Electrons.at(electron).phi());
    
    hists_["ElectronCharge"]->Fill(Electrons.at(electron).charge());  
    
    hists_["ElectronRelIso"]      ->Fill(Electrons.at(electron).pt()/(Electrons.at(electron).pt()+Electrons.at(electron).dr03TkSumPt() + Electrons.at(electron).dr03EcalRecHitSumEt() + Electrons.at(electron).dr03HcalTowerSumEt()));     
    hists_["ElectronTrackerRelIso"] ->Fill(Electrons.at(electron).pt()/(Electrons.at(electron).pt()+ Electrons.at(electron).dr03TkSumPt() )); 
    hists_["ElectronCaloRelIso"]    ->Fill(Electrons.at(electron).pt()/(Electrons.at(electron).pt()+ Electrons.at(electron).dr03EcalRecHitSumEt() + Electrons.at(electron).dr03HcalTowerSumEt())); 
    
    hists_["ElectronRelIsoOverPt"]      ->Fill((Electrons.at(electron).dr03TkSumPt() + Electrons.at(electron).dr03EcalRecHitSumEt() + Electrons.at(electron).dr03HcalTowerSumEt())/Electrons.at(electron).pt()); 
    hists_["ElectronTrackerRelIsoOverPt"] ->Fill((Electrons.at(electron).dr03TkSumPt() )/Electrons.at(electron).pt()); 
    hists_["ElectronCaloRelIsoOverPt"]    ->Fill((Electrons.at(electron).dr03EcalRecHitSumEt() + Electrons.at(electron).dr03HcalTowerSumEt())/Electrons.at(electron).pt()); 
    
    
  }
  
  hists_["NofElectrons"]->Fill(NofElectrons);
  
  //Fill October exercise quantities:
  if (Electrons.size() > 0) {
    hists_["highestpt_pt"]->Fill(Electrons.at(0).pt());
    hists_["highestpt_eta"]->Fill(Electrons.at(0).eta());
    hists_["highestpt_phi"]->Fill(Electrons.at(0).phi());
    hists_["highestpt_d0"]->Fill(Electrons.at(0).gsfTrack()->dxy(beamSpot));
    hists_["highestpt_trkIso03"]->Fill(Electrons.at(0).dr03TkSumPt());
    hists_["highestpt_ecalIso04"]->Fill(Electrons.at(0).dr04EcalRecHitSumEt());
    hists_["highestpt_hcalIso04"]->Fill(Electrons.at(0).dr04HcalTowerSumEt());
    hists_["highestpt_relIso"]->Fill( (Electrons.at(0).dr03TkSumPt() + Electrons.at(0).dr04EcalRecHitSumEt() + Electrons.at(0).dr04HcalTowerSumEt()) / Electrons.at(0).pt() );
  }
    
  if(Electrons.size()>1){
    TLorentzVector v1;
    TLorentzVector v2;
    TLorentzVector v3;
    v1.SetPtEtaPhiE( Electrons[0].pt(), Electrons[0].eta(), Electrons[0].phi(), Electrons[0].energy()  );
    v2.SetPtEtaPhiE( Electrons[1].pt(), Electrons[1].eta(), Electrons[1].phi(), Electrons[1].energy()  );
    v3 = v2+v1;
    double invM = v3.M();
    hists_["InvMass"]->Fill(invM);
    hists_["LeptonPairCharge"]->Fill(Electrons[0].charge() + Electrons[1].charge() );
  }
}

void ElectronChecker::end() 
{
}
