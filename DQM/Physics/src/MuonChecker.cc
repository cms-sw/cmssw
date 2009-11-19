#include "DQM/Physics/interface/MuonChecker.h"

MuonChecker::MuonChecker(const edm::ParameterSet& iConfig, std::string relativePath, std::string label)
{
  dqmStore_     = edm::Service<DQMStore>().operator->();
  relativePath_ = relativePath;
  label_        = label;
}

MuonChecker::~MuonChecker()
{
  delete dqmStore_;
}

void
MuonChecker::analyze(const std::vector<reco::Muon>& muons)
{
  int NbOfGlobalMu = 0;
  for(unsigned int m=0;m<muons.size();m++)
    {
      if(m==0) hists_["MaxMuPt"]    ->Fill(muons[m].pt());    
      if(m==1) hists_["SecMaxMuPt"] ->Fill(muons[m].pt()); 
      if(muons[m].isGlobalMuon())
	{
	  NbOfGlobalMu++;
	  hists_["GlobalMuonInnerTrackNbOfValidHits"]  ->Fill(muons[m].innerTrack()->numberOfValidHits());
	  hists_["GlobalMuonInnerTrackNbOfLostHits"]   ->Fill(muons[m].innerTrack()->numberOfLostHits());
	  hists_["GlobalMuonInnerTrackPt"]   	      ->Fill(muons[m].innerTrack()->pt());
	  
	  hists_["GlobalMuonOuterTrackNbOfValidHits"]  ->Fill(muons[m].outerTrack()->numberOfValidHits());
	  hists_["GlobalMuonOuterTrackNbOfLostHits"]   ->Fill(muons[m].outerTrack()->numberOfLostHits());
	  hists_["GlobalMuonOuterTrackPt"]    	      ->Fill(muons[m].outerTrack()->pt());
	  
	  hists_["GlobalMuonGlobalTrackPt"]	      ->Fill(muons[m].globalTrack()->pt());
	  hists_["GlobalMuonGlobalTrackD0"]	      ->Fill(muons[m].globalTrack()->d0());
	  hists_["GlobalMuonGlobalTrackChi2"]	      ->Fill(muons[m].globalTrack()->normalizedChi2());
	}
      
      hists_["MuonCaloCompatibility"] ->Fill(muons[m].caloCompatibility());
      
      hists_["MuonIsoR03SumPt"] ->Fill(muons[m].isolationR03().sumPt);
      hists_["MuonIsoR03emEt"]  ->Fill(muons[m].isolationR03().emEt);
      hists_["MuonIsoR03hadEt"] ->Fill(muons[m].isolationR03().hadEt);
      hists_["MuonRelIso"]      ->Fill(muons[m].pt()/(muons[m].pt()+muons[m].isolationR03().sumPt + muons[m].isolationR03().emEt + muons[m].isolationR03().hadEt)); 
      hists_["MuonTrackerRelIso"] ->Fill(muons[m].pt()/(muons[m].pt()+ muons[m].isolationR03().sumPt )); 
      hists_["MuonCaloRelIso"]    ->Fill(muons[m].pt()/(muons[m].pt()+ muons[m].isolationR03().emEt + muons[m].isolationR03().hadEt)); 
      
      hists_["MuonRelIsoOverPt"]        ->Fill((muons[m].isolationR03().sumPt + muons[m].isolationR03().emEt + muons[m].isolationR03().hadEt)/muons[m].pt());     
      hists_["MuonTrackerRelIsoOverPt"] ->Fill((muons[m].isolationR03().sumPt )/muons[m].pt()); 
      hists_["MuonCaloRelIsoOverPt"]    ->Fill((muons[m].isolationR03().emEt + muons[m].isolationR03().hadEt)/muons[m].pt()); 
      
      
      //get the veto cone information
      hists_["MuonVetoEm"]  ->Fill(muons[m].calEnergy().em);
      hists_["MuonVetoHad"] ->Fill(muons[m].calEnergy().had);
      
      hists_["MuonCharge"]  ->Fill(muons[m].charge());
    }
  hists_["NofMuons"]  ->Fill(muons.size());
  hists_["NofGlobalMuons"]  ->Fill(NbOfGlobalMu);
  if(muons.size()>1){
    TLorentzVector v1, v2, v3;
    v1.SetPtEtaPhiE( muons[0].pt(), muons[0].eta(), muons[0].phi(), muons[0].energy());
    v2.SetPtEtaPhiE( muons[1].pt(), muons[1].eta(), muons[1].phi(), muons[1].energy());
    v3 = v1+v2;
    double invM = v3.M();
    hists_["InvM"]->Fill(invM);
    hists_["LeptonPairCharge"]->Fill(muons[0].charge() +muons[1].charge() );
  }
  
  
}

void 
MuonChecker::begin(const edm::EventSetup&)
{
  dqmStore_->setCurrentFolder( relativePath_+"/Muons_"+label_ );
    
  hists_["NofMuons"]      		            = dqmStore_->book1D("NofMuons","Number of muons",20,0,20);
  hists_["NofMuons"]->setAxisTitle("Nof muons",1);
  hists_["NofGlobalMuons"]      		            = dqmStore_->book1D("NofGlobalMuons","Number of muons",20,0,20);
  hists_["NofGlobalMuons"]->setAxisTitle("Nof global muons",1);

  hists_["GlobalMuonInnerTrackNbOfValidHits"]      = dqmStore_->book1D("GlobalMuonInnerTrackNbOfValidHits","Number of valid hits in silicon fit for global muons",400,0,20);
  hists_["GlobalMuonInnerTrackNbOfValidHits"]      ->setAxisTitle("Nof valid hits in SiStrip for global muons",1);
  hists_["GlobalMuonInnerTrackNbOfLostHits"]	    = dqmStore_->book1D("GlobalMuonInnerTrackNbOfLostHits","Number of lost hits in silicon fit for global muons",400,0,20);
  hists_["GlobalMuonInnerTrackNbOfLostHits"]       ->setAxisTitle("Nof lost hits in SiStrip for global muons",1);
  hists_["GlobalMuonInnerTrackPt"]	            = dqmStore_->book1D("GlobalMuonInnerTrackPt","Inner track transverse momentum for global muons",400,0,400);
  hists_["GlobalMuonInnerTrackPt"]                 ->setAxisTitle("Pt in SiStrip for global muons",1);

  hists_["GlobalMuonOuterTrackNbOfValidHits"]      = dqmStore_->book1D("GlobalMuonOuterTrackNbOfValidHits","Number of valid hits in silicon fit for global muons",400,0,20);
  hists_["GlobalMuonOuterTrackNbOfValidHits"]      ->setAxisTitle("Nof valid hits in MuonChambers for global muons",1);
  hists_["GlobalMuonOuterTrackNbOfLostHits"]       = dqmStore_->book1D("GlobalMuonOuterTrackNbOfLostHits","Number of lost hits in silicon fit for global muons",400,0,20);
  hists_["GlobalMuonOuterTrackNbOfLostHits"]       ->setAxisTitle("Nof lost hits in MuonChambers for global muons",1);
  hists_["GlobalMuonOuterTrackPt"]		    = dqmStore_->book1D("GlobalMuonOuterTrackPt","Outer track transverse momentum for global muons",400,0,400);
  hists_["GlobalMuonOuterTrackPt"]                  ->setAxisTitle("Pt in MuonChambers for global muons",1);

  hists_["GlobalMuonGlobalTrackPt"]                = dqmStore_->book1D("GlobalMuonGlobalTrackPt","Global track transverse momentum for global muons",400,0,400);
  hists_["GlobalMuonGlobalTrackPt"]                ->setAxisTitle("Pt of global track for global muons",1);
  hists_["GlobalMuonGlobalTrackD0"]    	    = dqmStore_->book1D("GlobalMuonGlobalTrackD0","Global track impact parameter for global muons",400,-0.4,0.4);
  hists_["GlobalMuonGlobalTrackD0"]                ->setAxisTitle("D0 of global track for global muons",1);
  hists_["GlobalMuonGlobalTrackChi2"] 		    = dqmStore_->book1D("GlobalMuonGlobalTrackChi2","Global track normalized #chi^{2} ",400,0,20);
  hists_["GlobalMuonGlobalTrackChi2"]              ->setAxisTitle("Normalized #chi^{2} of global track for global muons",1);
  
  hists_["MuonCaloCompatibility"]                  = dqmStore_->book1D("CaloCompatibility","Value of the LR measuring the probability that the muon is calo-compatible",100,0,1);
  hists_["MuonCaloCompatibility"]                  ->setAxisTitle("MuonCalo Compatibility",1);
  hists_["MuonIsoR03SumPt"]   		            = dqmStore_->book1D("MuonIsoR03SumPt","Sum of the track transverse momenta in a cone of 0.3 around the muon",200,0,20);
  hists_["MuonIsoR03SumPt"]                        ->setAxisTitle("MuonIsoR03 - SumPt",1);
  hists_["MuonIsoR03emEt"]			    = dqmStore_->book1D("MuonIsoR03emEt","Sum of the electromagnetic transverse energy in a cone of 0.3 around the muon",200,0,20);
  hists_["MuonIsoR03emEt"]                         ->setAxisTitle("MuonIsoR03 - emEt",1);
  hists_["MuonIsoR03hadEt"]		            = dqmStore_->book1D("MuonIsoR03hadEt","Sum of the hadronic transverse energy in a cone of 0.3 around the muon",200,0,20);
  hists_["MuonIsoR03hadEt"]                        ->setAxisTitle("MuonIsoR03 - hadEt",1);
  hists_["MuonRelIso"] 			    = dqmStore_->book1D("MuonRelIso","Relative isolation the muon",100,0,1);
  hists_["MuonRelIso"]                             ->setAxisTitle("MuonRelIso",1);
  hists_["MuonTrackerRelIso"] 			    = dqmStore_->book1D("MuonTrackerRelIso","Relative tracker isolation the muon",100,0,1);
  hists_["MuonTrackerRelIso"]                      ->setAxisTitle("MuonTrackerRelIso",1);
  hists_["MuonCaloRelIso"] 			    = dqmStore_->book1D("MuonCaloRelIso","Relative isolation the muon",100,0,1);
  hists_["MuonCaloRelIso"]                         ->setAxisTitle("MuonCaloRelIso",1);
  hists_["MuonVetoEm"] 			    = dqmStore_->book1D("MuonVetoEm","Veto electromagnetic energy deposit in a cone of 0.07",200,0,20);
  hists_["MuonVetoEm"]                             ->setAxisTitle("Veto emDeposit #DeltaR<0.07",1);
  hists_["MuonVetoHad"]			    = dqmStore_->book1D("MuonVetoHad","Veto hadronic energy deposit in a cone of 0.1",200,0,20);
  hists_["MuonVetoHad"]                            ->setAxisTitle("Veto hadDeposit #DeltaR<0.1",1);
  hists_["MuonCharge"]			   	    = dqmStore_->book1D("MuonCharge","Charge of the muon",4,-2,2);
  hists_["MuonCharge"]                             ->setAxisTitle("Charge of muon",1);
  

  hists_["MuonRelIsoOverPt"] 			    = dqmStore_->book1D("MuonRelIsoOverPt","Relative isolation the muon",100,0,1);
  hists_["MuonRelIsoOverPt"]                       ->setAxisTitle("MuonRelIsoOverPt",1);
  hists_["MuonTrackerRelIsoOverPt"] 		    = dqmStore_->book1D("MuonTrackerRelIsoOverPt","Relative tracker isolation the muon",100,0,1);
  hists_["MuonTrackerRelIsoOverPt"]                ->setAxisTitle("MuonTrackerRelIsoOverPt",1);
  hists_["MuonCaloRelIsoOverPt"] 		    = dqmStore_->book1D("MuonCaloRelIsoOverPt","Relative isolation the muon",100,0,1);
  hists_["MuonCaloRelIsoOverPt"]                   ->setAxisTitle("MuonCaloRelIsoOverPt",1);
  
  
  hists_["InvM"]                                   = dqmStore_->book1D("InvM","dimuon invariant mass",50, 0, 200);
  hists_["InvM"]                                   ->setAxisTitle("dimuon invariant mass GeV/c^{2}");
  
  hists_["MaxMuPt"]                                = dqmStore_->book1D("MaxMuPt","p_{T} of the highest p_{T} muon",50, 0, 200);
  hists_["MaxMuPt"]                                ->setAxisTitle("p_{T}  GeV/c");
  
  hists_["SecMaxMuPt"]                             = dqmStore_->book1D("SecMaxMuPt","p_{T} of the second highest p_{T} muon",50, 0, 200);
  hists_["SecMaxMuPt"]                             ->setAxisTitle("p_{T} GeV/c");
  
  
  hists_["isGlobalMuonPromptTight"]                = dqmStore_->book1D("isGlobalMuonPromptTight","isGlobalMuonPromptTight",2, 0, 1);
  
  hists_["LeptonPairCharge"]                       = dqmStore_->book1D("LeptonPairCharge","Sum of the charges of the 2 highests leptons p_{T}",5, -2, 2);
  hists_["LeptonPairCharge"]                       ->setAxisTitle("Sum of charges");
}

void MuonChecker::end() 
{
}
