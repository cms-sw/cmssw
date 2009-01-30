void plot_validation(){

  bool dojet = true;
  bool compare_releases = false; // if false, assume that we compare full/fast
  bool dofull = true; // only relevant if compare_releases = true

  TString setup = "ideal";
  TString sample = "qcd80";

  TString release1 = "2_2_3";
  TString release2 = "2_2_3";

  if (dojet)  TString trigname = "SingleJet50";
  if (!dojet) TString trigname = "SingleMET35";

  // ---------------------------------------------------------------------------


  TString ff1 = "";
  TString ff2 = "";
  if (!compare_releases) ff2 = "_fastsim";
  if (compare_releases && !dofull){
    ff1 = "_fastsim";
    ff2 = "_fastsim";
  };

  TString shortrelease1 = release1.ReplaceAll("_","");
  TString shortrelease2 = release2.ReplaceAll("_","");
  TString file1 = Form("/uscms_data/d1/cammin/CMSSW/TriggerPerformance/CMSSW_2_2_3/src/run/relval%s_%s/%s%s_output.root",shortrelease1.Data(),setup.Data(),sample.Data(),ff1.Data());
  TString file2 = Form("/uscms_data/d1/cammin/CMSSW/TriggerPerformance/CMSSW_2_2_3/src/run/relval%s_%s/%s%s_output.root",shortrelease2.Data(),setup.Data(),sample.Data(),ff2.Data());


  TString fastfull;

  printf("\nInput file 1: %s\n",file1.Data());
  printf(  "Input file 2: %s\n\n",file2.Data());

  double xmin = 0;
  double xmax = 200;
  if (!dojet) xmax = 250; // for MET

  TCanvas *c = new TCanvas("c", Form("%s",trigname.Data()), 0,0,700,500);
  c->SetTopMargin(0.05);
  c->SetRightMargin(0.05);
  c->SetLeftMargin(0.12);
  c->SetBottomMargin(0.15);
  TH1F *hframe = (TH1F*)c->DrawFrame(xmin,0,xmax,1.1);
  if (dojet)  hframe->GetXaxis()->SetTitle("Leading GenJet p_{T} (GeV)");
  if (!dojet) hframe->GetXaxis()->SetTitle("GenMET (GeV)");
  hframe->GetYaxis()->SetTitle("HLT efficiency");
  hframe->GetXaxis()->SetTitleSize(0.07);
  hframe->GetYaxis()->SetTitleSize(0.07);
  hframe->GetXaxis()->SetTitleOffset(0.9);
  hframe->GetYaxis()->SetTitleOffset(0.7);


  TFile *f1 = new TFile(file1);
  f1->cd(Form("DQMData/HLT/HLTJETMET/%s",trigname.Data()));
  if (dojet){
    TH1F *h1 = (TH1F*)gROOT->FindObject("_meGenJetPtTrg");
    TH1F *h2 = (TH1F*)gROOT->FindObject("_meGenJetPt");
  } else{
    TH1F *h1 = (TH1F*)gROOT->FindObject("_meGenMETTrg");
    TH1F *h2 = (TH1F*)gROOT->FindObject("_meGenMET");
  }

  h1->Sumw2();
  h2->Sumw2();

  h1->Divide(h1,h2,1,1,"B");

  h1->SetFillColor(kYellow);
  h1->Draw("hsame");
  h1->Draw("samee");



  TFile *f2 = new TFile(file2);
  f2->cd(Form("DQMData/HLT/HLTJETMET/%s",trigname.Data()));

  if (dojet){
    TH1F *h3 = (TH1F*)gROOT->FindObject("_meGenJetPtTrg");
    TH1F *h4 = (TH1F*)gROOT->FindObject("_meGenJetPt");
  } else {
    TH1F *h3 = (TH1F*)gROOT->FindObject("_meGenMETTrg");
    TH1F *h4 = (TH1F*)gROOT->FindObject("_meGenMET");
  }
//   h3->Draw();
//   h4->Draw();
  h3->Sumw2();
  h4->Sumw2();

  h3->Divide(h3,h4,1,1,"B");

  //  h3->Draw("histesame");
  h3->SetMarkerStyle(20);
  h3->SetMarkerColor(kRed);
  h3->SetLineColor(kRed);
  h3->Draw("samee");

  TLegend *leg = new TLegend(0.623, 0.2, 0.92, 0.44);
  leg->SetFillColor(kYellow-9);
  leg->SetBorderSize(1);
  if (!compare_releases){
    leg->AddEntry(h1,Form("%s ideal full",release1.Data()),"lfe");
    leg->AddEntry(h3,Form("%s ideal fast",release2.Data()),"ple");
  } else {
    if (dofull)  fastfull = "full";
    if (!dofull) fastfull = "fast";
    leg->AddEntry(h1,Form("%s ideal %s",release1.Data(),fastfull.Data()),"lfe");
    leg->AddEntry(h3,Form("%s ideal %s",release2.Data(),fastfull.Data()),"ple");
  }
  leg->Draw();

  //  TLatex *tex = new TLatex(0.15, 0.9, Form("RelVal TTbar - %s",(trigname.Remove(0,4)).Data()));
  TLatex *tex = new TLatex(0.15, 0.9, Form("RelVal %s - %s",sample.Data(),(trigname).Data()));
  tex->SetNDC();
  tex->Draw();

  c->SetGridx();
  c->RedrawAxis();

  if (!compare_releases){
    c->Print(Form("plots/RelVal_%s_FullFast_%s_%s.eps",sample.Data(),release1.Data(),trigname.Data()));
    c->Print(Form("plots/RelVal_%s_FullFast_%s_%s.png",sample.Data(),release1.Data(),trigname.Data()));
  }
  if (compare_releases){
    c->Print(Form("plots/RelVal_%s_%s_%s--%s_%s.eps",sample.Data(),fastfull.Data(),release1.Data(),release2.Data(),trigname.Data()));
    c->Print(Form("plots/RelVal_%s_%s_%s--%s_%s.png",sample.Data(),fastfull.Data(),release1.Data(),release2.Data(),trigname.Data()));
  }
}
