{
   gROOT->SetStyle("Plain");
   gStyle->SetPalette(1);
   gStyle->SetOptStat(1111111);
   gSystem->Load("libRecoMuonMuonIdentification");
   
   f = TFile::Open("dcap://cmsdca3.fnal.gov:24143/pnfs/fnal.gov/usr/cms/WAX/11/store/mc/2007/11/7/RelVal-RelValSingleMuMinusPt10-1194439351/0000/1ACAFBB9-4A8D-DC11-A8EE-001617C3B73A.root");
   TTree* tree = (TTree*)f->Get("Events");
   TCanvas* c1 = new TCanvas("muons","muons",800,800);
   c1->Divide(3,3);
   TH1F* h1 = new TH1F("h1","global muon",100,0,100);
   TH1F* h2 = new TH1F("h2","tracker muon",100,0,100);
   TH1F* h3 = new TH1F("h3","tracker muon + Loose ID",100,0,100);
   TH1F* h4 = new TH1F("h4","tracker muon + Tight ID",100,0,100);
   TH2F* h5 = new TH2F("h5","segment vs calo compatibility",120,-0.1,1.1,120,-0.1,1.1);
   TH1F* h6 = new TH1F("h6","tracker muon + Loose compatibility ID",100,0.,100.);
   TH1F* h7 = new TH1F("h7","tracker muon + Tight compatibility ID",100,0.,100.);
   
   // create and connect muon collection branch 
   tree->SetBranchStatus("*",0);
   tree->SetBranchStatus("recoMuons*",1);
   std::vector<reco::Muon> muons;
   std::vector<reco::Muon> trackerMuons;
   
   int TMLastStationLoose = 0;
   int TMLastStationTight = 1;
   int TM2DCompatibilityLoose      = 2;
   int TM2DCompatibilityTight      = 3;


   TString branchName = tree->GetAlias("muons");
   tree->SetBranchAddress(branchName,&muons);

   for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
      tree->GetEntry(index);
      tree->SetBranchAddress(branchName,&muons);
      if (index%1000==0) std::cout << "Event " << index << std::endl;
      for(unsigned int i=0; i<muons.size(); i++) 
	{
	   if ( muons[i].isGlobalMuon() ) h1->Fill(muons[i].pt());
	   if ( muons[i].isTrackerMuon() ) {
	      h2->Fill(muons[i].pt());
	      if (muon::isGoodMuon(muons[i],TMLastStationLoose)) h3->Fill(muons[i].pt());
	      if (muon::isGoodMuon(muons[i],TMLastStationTight)) h4->Fill(muons[i].pt());
	      h5->Fill(muon::getSegmentCompatibility(muons[i]),muon::getCaloCompatibility(muons[i]));
	      if (muon::isGoodMuon(muons[i],TM2DCompatibilityLoose)) h6->Fill(muons[i].pt());
	      if (muon::isGoodMuon(muons[i],TM2DCompatibilityTight)) h7->Fill(muons[i].pt());
	   }
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
   c1->cd(5);
   h5->Draw();
   c1->cd(6);
   h6->Draw();
   c1->cd(7);
   h7->Draw();
}
