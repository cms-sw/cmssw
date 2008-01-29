{
   gROOT->SetStyle("Plain");
   gStyle->SetPalette(1);
   gStyle->SetOptStat(1111111);
   gSystem->Load("libRecoMuonMuonIdentification");
   
   TFile f("/tmp/dmytro/muons.root");
   TTree* tree = (TTree*)f.Get("Events");
   TCanvas* c1 = new TCanvas("muons","muons",800,800);
   c1->Divide(2,2);
   TH1F* h1 = new TH1F("h1","global muon",100,0,100);
   TH1F* h2 = new TH1F("h2","tracker muon",100,0,100);
   TH1F* h3 = new TH1F("h3","tracker muon + Loose ID",100,0,100);
   TH1F* h4 = new TH1F("h4","tracker muon + Tight ID",100,0,100);
   
   // create and connect muon collection branch 
   tree->SetBranchStatus("*",0);
   tree->SetBranchStatus("recoMuons*",1);
   std::vector<reco::Muon> muons;
   std::vector<reco::Muon> trackerMuons;
   
   int TMLastStationLoose = 0;
   int TMLastStationTight = 1;
   
   TString branchName1 = tree->GetAlias("muons");
   tree->SetBranchAddress(branchName1,&muons);
   TString branchName2 = tree->GetAlias("trackerMuons");
   tree->SetBranchAddress(branchName2,&trackerMuons);

   for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
      tree->GetEntry(index);
      tree->SetBranchAddress(branchName1,&muons);
      tree->SetBranchAddress(branchName2,&trackerMuons);
      if (index%1000==0) std::cout << "Event " << index << std::endl;
      for(unsigned int i=0; i<muons.size(); i++) h1->Fill(muons[i].pt());
      for(unsigned int i=0; i<trackerMuons.size(); i++) {
	 h2->Fill(trackerMuons[i].pt());
	 if (muonid::isGoodMuon(trackerMuons[i],TMLastStationLoose)) h3->Fill(trackerMuons[i].pt());
	 if (muonid::isGoodMuon(trackerMuons[i],TMLastStationTight)) h4->Fill(trackerMuons[i].pt());
      }
   }
   
   c1->cd(1);
   h1->Draw();
   c1->cd(2);
   h2->Draw();
   c1->cd(3);
   h3->Draw();
   c1->cd(4);
   h4->Draw();
}
