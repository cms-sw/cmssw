void MakeRatePlots(TString filename = "")
{
   gROOT->SetStyle("Plain");
   gStyle->SetPalette(1);

   if (filename == "")
   {
      cout << "Usage: MakeRatePlots(<my ROOT file name>)" << endl;
      return;
   }

   TFile *f = TFile::Open(filename);
   TCanvas *c1 = new TCanvas("c1","c1",1200,400);
   c1->Divide(3, 1);
   c1->cd(1);
   TH1F *h1 = (TH1F *)f->Get("cumulative");
   h1->SetFillColor(2);
   h1->Draw("hbar3");

   c1->cd(2);
   TH1F *h2 = (TH1F *)f->Get("individual");
   h2->SetFillColor(2);
   h2->Draw("hbar3");

   c1->cd(3);
   TH2F *h3 = (TH2F *)f->Get("overlap");
   h3->SetMinimum(-0.0001);

   h3->Draw("col2z");
}
