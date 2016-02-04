{
//author A.Zabi
//This macro can be used to reproduce the tau benchmark plot
//for tau jet reconstruction studies
// 50 GeV taus desintegrating hadronically have been studied
gROOT->Reset();
TFile *f = new TFile("tauBenchmark.root");
TCanvas* c1 = new TCanvas;

h_deltaETvisible_MCPF->SetStats(1); 
gStyle->SetOptStat("nmeriou");
h_deltaETvisible_MCPF->GetXaxis()->SetTitle("#DeltaE_{T} (GeV)");
h_deltaETvisible_MCPF->SetTitle("Tau Benchmark, E_{T}(rec)-E_{T}(true)");
h_deltaETvisible_MCPF->SetLineColor(2);
h_deltaETvisible_MCPF->Draw();
h_deltaETvisible_MCEHT->Draw("same");

Double_t x_1=0.60; Double_t y_1 = 0.60;
Double_t x_2=0.85; Double_t y_2 = 0.70;

TLegend *leg = new TLegend(x_1,y_1,x_2,y_2,NULL,"brNDC");
leg->SetTextSize(0.035);
leg->SetFillStyle(0);
leg->SetFillColor(0);
leg->SetTextFont(52);
leg->SetTextAlign(32);

leg->AddEntry(h_deltaETvisible_MCPF,"Particle Flow Jets","l");
leg->AddEntry(h_deltaETvisible_MCEHT,"caloTower Jets","l");
leg->Draw();

//c1->Print("tauBenchmark.eps");
c1->Print("tau50_barrel.gif");
//gApplication->Terminate();
}
