{ 
gSystem->Load("libFWCoreFWLite.so"); 
AutoLibraryLoader::enable();
TFile file_pion_gun("Example-RecoMET.root");

//  book histograms
TH1F* h01 = new TH1F("h01", "GenMET (GeV)",                  50, 0.0, 150.0); //generator-level
TH1F* h02 = new TH1F("h02", "CaloMET (GeV)",                 50, 0.0, 150.0);
TH1F* h04 = new TH1F("h04", "Calo-Gen MET (GeV)",            50, -50.0, 100.0);
TH1F* h07 = new TH1F("h07", "GenSumET (GeV)",                50, 0.0, 1500.0);//generator-level
TH1F* h08 = new TH1F("h08", "CaloSumET (GeV)",               50, 0.0, 1500.0);
TH1F* h10 = new TH1F("h10", "EEMF",                          50, 0.0, 1.0);
TH1F* h11 = new TH1F("h11", "SumET in EE (GeV)",             100, 0.0, 500.0);
TH1F* h12 = new TH1F("h12", "SumET in EB (GeV)",             100, 0.0, 500.0);
TH1F* h13 = new TH1F("h13", "SumET in HF (GeV)",             100, 0.0, 500.0);
TH1F* h14 = new TH1F("h14", "SumET in HE (GeV)",             100, 0.0, 500.0);
TH1F* h15 = new TH1F("h15", "SumET in HB (GeV)",             100, 0.0, 500.0);
TH1F* h16 = new TH1F("h16", "SumET in HO (GeV)",             100, 0.0, 500.0);
TH1F* h17 = new TH1F("h17", "GenMET-phi (rad)",              20, -4.0, 4.0);//generator-level
TH1F* h18 = new TH1F("h18", "CaloMET-phi (rad)",             20, -4.0, 4.0);
TH1F* h20 = new TH1F("h20", "Calo-Gen MET-phi (rad)",        20, -4.0, 4.0);
TH1F* h27 = new TH1F("h27", "Gen  MET-x (GeV)",              50, -100.0, 100.0);
TH1F* h28 = new TH1F("h28", "Calo MET-x (GeV)",              50, -100.0, 100.0);

TCanvas* c = new TCanvas("Name", "Title", 700,700);

//Get event tree and jet & met collection branches
TTree *tree = (TTree*)file_pion_gun.Get("Events");

std::vector<reco::GenMET>  GenMETCollection;
std::vector<reco::CaloMET> CaloMETCollection;

double numevents = 1000.0; 
double xsect =  0.000023855474865; 
xsect *= 100000.0;
double weight = xsect/numevents; 
// Loop over events
cout << tree->GetEntries() << endl;
for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) 
{
  double px[2], py[2], pz[2], E[2];
  tree->GetEntry(index); 
  
  tree->SetBranchAddress("recoGenMETs_genMet__R.obj", &GenMETCollection);
  tree->SetBranchAddress("recoCaloMETs_met__R.obj", &CaloMETCollection);
  
  reco::GenMET*  genMet = (reco::GenMET*)GenMETCollection[0]; 
  reco::CaloMET*    met = (reco::CaloMET*)CaloMETCollection[0]; 
  if( CaloMETCollection.size() > 0 ) 
    {
      h01->Fill(genMet->pt(), weight);
      h02->Fill(   met->pt(), weight);
      h04->Fill(   met->px()-genMet->px(), weight);
      h07->Fill(genMet->sumEt(), weight);
      h08->Fill(   met->sumEt(), weight);
      h10->Fill(met->emEtFraction(), weight);
      h11->Fill(met->emEtInEE(),  weight);
      h12->Fill(met->emEtInEB(),  weight);
      h13->Fill(met->hadEtInHF(), weight);
      h14->Fill(met->hadEtInHE(), weight);
      h15->Fill(met->hadEtInHB(), weight);
      h16->Fill(met->hadEtInHO(), weight);
      h17->Fill(genMet->phi());
      h18->Fill(   met->phi());
      h20->Fill(   met->phi()-genMet->phi());
      h27->Fill(genMet->px(), weight);
      h28->Fill(   met->px(), weight);
    }
}
c->SetLogy(1); 

h01->SetLineColor(2); h01->Draw(); c->Print("h01.ps");
h02->SetLineColor(1); h02->Draw(); c->Print("h02.ps");
h04->SetFillColor(5); h04->Draw(); c->Print("h04.ps");
h07->SetLineColor(2); h07->Draw(); c->Print("h07.ps");
h08->SetLineColor(1); h08->Draw(); c->Print("h08.ps");
c->SetLogy(0); 
h10->SetFillColor(5); h10->Draw(); c->Print("h10.ps");
c->SetLogy(1); 
h11->SetFillColor(5); h11->Draw(); c->Print("h11.ps");
h12->SetFillColor(5); h12->Draw(); c->Print("h12.ps");
h13->SetFillColor(5); h13->Draw(); c->Print("h13.ps");
h14->SetFillColor(5); h14->Draw(); c->Print("h14.ps");
h15->SetFillColor(5); h15->Draw(); c->Print("h15.ps");
h16->SetFillColor(5); h16->Draw(); c->Print("h16.ps");
c->SetLogy(0); 
h17->SetLineColor(2); h17->Draw(); c->Print("h17.ps");
h18->SetLineColor(1); h18->Draw(); c->Print("h18.ps");
h20->SetFillColor(5); h20->Draw(); c->Print("h20.ps");
c->SetLogy(1); 
h27->SetFillColor(5); h27->Draw(); c->Print("h27.ps");
h28->SetFillColor(5); h28->Draw(); c->Print("h28.ps");
}  

//  LocalWords:  TFile
