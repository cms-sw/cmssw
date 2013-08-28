void l1Analyzer(TString dir="", TString fileName="l1Analyzer.root")
{
  // Open the file
  TFile *f = new TFile(fileName);

  // Postscript file
  TPostScript* ps =new TPostScript("l1Analyzer.ps", 112);

  // Canvas
  TCanvas *c1 = new TCanvas("c1","c1",800,800);

  // First the simple histograms
  Plot(f,dir+"/L1Candidates/Et","E_{T} (GeV)","No. of entries"); c1->Update();
  Plot(f,dir+"/L1Candidates/Eta","#eta","No. of entries"); c1->Update();
  Plot(f,dir+"/L1Candidates/Phi","#phi (rad)","No. of entries"); c1->Update();
  
  Plot(f,dir+"/RefCandidates/Et","E_{T} (GeV)","No. of entries"); c1->Update();
  Plot(f,dir+"/RefCandidates/Eta","#eta","No. of entries"); c1->Update();
  Plot(f,dir+"/RefCandidates/Phi","#phi (rad)","No. of entries"); c1->Update();
 
  Plot(f,dir+"/Resolutions/DeltaR","#Delta R (L1,Ref)","No. of entries"); c1->Update();
  Plot(f,dir+"/Resolutions/EtRes","(E_{T,L1}-E_{T,Ref})/E_{T,Ref}","No. of entries"); c1->Update();
  Plot(f,dir+"/Resolutions/EtCor","E_{T, Ref} (GeV)","E_{T, L1} (GeV)"); c1->Update();
  Plot(f,dir+"/Resolutions/EtProf","E_{T, Ref} (GeV)","(E_{T,L1}-E_{T,Ref})/E_{T,Ref}"); c1->Update();

  Plot(f,dir+"/Resolutions/EtaRes","(#eta_{L1}-#eta_{Ref})/#eta_{Ref}","No. of entries"); c1->Update();
  Plot(f,dir+"/Resolutions/EtaCor","#eta_{Ref}","#eta_{L1}"); c1->Update();
  Plot(f,dir+"/Resolutions/EtaProf","#eta_{Ref}","(#eta_{L1}-#eta_{Ref})/#eta_{Ref}"); c1->Update();

  Plot(f,dir+"/Resolutions/PhiRes","(#phi_{L1}-#phi_{Ref})/#phi_{Ref}","No. of entries"); c1->Update();
  Plot(f,dir+"/Resolutions/PhiCor","#phi_{Ref} (rad)","#phi_{L1} (rad)"); c1->Update();
  Plot(f,dir+"/Resolutions/PhiProf","#phi_{Ref} (rad)","(#phi_{L1}-#phi_{Ref})/#phi_{Ref}"); c1->Update();

  Plot(f,dir+"/Efficiencies/EtEff","E_{T} (GeV)","Efficiency","e"); c1->Update();
  Plot(f,dir+"/Efficiencies/EtaEff","#eta","Efficiency","e"); c1->Update();
  Plot(f,dir+"/Efficiencies/PhiEff","#phi (rad)","Efficiency","e"); c1->Update();
  
  ps->Close();

}

void Plot(TFile* f, TString Hist, TString XAxisLabel, TString YAxisLabel="Events", TString Opt="")
{

  // Get the histograms from the files
  TH1D *H   = (TH1D*)f->Get(Hist);

  // Add the X axis label
  H->GetXaxis()->SetTitle(XAxisLabel);
  H->GetYaxis()->SetTitle(YAxisLabel);

  // plot 
  H->Draw(Opt);

}
