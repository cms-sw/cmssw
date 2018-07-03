{
  gROOT->Reset();
  gStyle->SetPalette(1);
  gStyle                  -> SetOptStat(1111);
  
  TCanvas c2h("c2h","2-d options",10,10,800,600);
  c2h.SetFillColor(10);
  gPad->SetDrawOption("e1p");
  TFile f0("coefficients_219_val_barrel_minus_8.9mln.root");  
  
      float x[40]={1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.};
      float y[40]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
       y[0] = h1etacoefmin1l->GetRMS();
       y[1] = h1etacoefmin2l->GetRMS();
       y[2] = h1etacoefmin3l->GetRMS();
       y[3] = h1etacoefmin4l->GetRMS();
       y[4] = h1etacoefmin5l->GetRMS();
       y[5] = h1etacoefmin6l->GetRMS();
       y[6] = h1etacoefmin7l->GetRMS();
       y[7] = h1etacoefmin8l->GetRMS();
       y[8] = h1etacoefmin9l->GetRMS();
       y[9] = h1etacoefmin10l->GetRMS();
       y[10] = h1etacoefmin11l->GetRMS();
       y[11] = h1etacoefmin12l->GetRMS();
       y[12] = h1etacoefmin13l->GetRMS();
       y[13] = h1etacoefmin14l->GetRMS();
       y[14] = h1etacoefmin16_1l->GetRMS();
       
       h1etacoefmin1l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin1l.gif");
       h1etacoefmin2l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin2l.gif");
       h1etacoefmin3l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin3l.gif");
       h1etacoefmin4l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin4l.gif");
       h1etacoefmin5l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin5l.gif");
       h1etacoefmin6l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin6l.gif");
       h1etacoefmin7l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin7l.gif");
       h1etacoefmin8l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin8l.gif");
       h1etacoefmin9l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin9l.gif");
       h1etacoefmin10l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin10l.gif");
       h1etacoefmin11l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin11l.gif");
       h1etacoefmin12l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin12l.gif");
       h1etacoefmin13l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin13l.gif");
       h1etacoefmin14l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin14l.gif");
       h1etacoefmin15l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin15l.gif");
       h1etacoefmin16_1l->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin16_1l.gif");

       TFile f1("coefficients_219_val_endcap_minus_8.9mln.root");     
       h1etacoefmin17a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin17a.gif");
       h1etacoefmin18a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin18a.gif");
       h1etacoefmin19a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin19a.gif");
       h1etacoefmin20a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin20a.gif");
       h1etacoefmin21a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin21a.gif");
       h1etacoefmin22a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin22a.gif");
       h1etacoefmin23a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin23a.gif");
       h1etacoefmin24a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin24a.gif");
       h1etacoefmin25a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin25a.gif");
       h1etacoefmin26a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin26a.gif");
       h1etacoefmin27a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin27a.gif");
       h1etacoefmin28a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin28a.gif");
       h1etacoefmin29a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin29a.gif");
       
       h1etacoefmin17b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin17b.gif");
       h1etacoefmin18b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin18b.gif");
       h1etacoefmin19b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin19b.gif");
       h1etacoefmin20b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin20b.gif");
       h1etacoefmin21b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin21b.gif");
       h1etacoefmin22b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin22b.gif");
       h1etacoefmin23b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin23b.gif");
       h1etacoefmin24b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin24b.gif");
       h1etacoefmin25b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin25b.gif");
       h1etacoefmin26b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin26b.gif");
       h1etacoefmin27b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin27b.gif");
       h1etacoefmin28b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin28b.gif");
       h1etacoefmin29b->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin29b.gif");
       
       
       TFile f2("coefficients_219_val_hf_minus_8.9mln.root");
	 
       y[28] = h1etacoefmin30a->GetRMS();
       y[29] = h1etacoefmin31a->GetRMS();
       y[30] = h1etacoefmin32a->GetRMS();
       h1etacoefmin30a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin30a.gif");
       h1etacoefmin31a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin31a.gif");
       h1etacoefmin32a->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin32a.gif");
       

       /*
       y[15] = h1etacoefmin17_1->GetRMS();
       y[16] = h1etacoefmin18_1->GetRMS();
       y[17] = h1etacoefmin19_1->GetRMS();
       y[18] = h1etacoefmin20_1->GetRMS();
       y[19] = h1etacoefmin21_1->GetRMS();
       y[20] = h1etacoefmin22_1->GetRMS();
       y[21] = h1etacoefmin23_1->GetRMS();
       y[22] = h1etacoefmin24_1->GetRMS();
       y[23] = h1etacoefmin25_1->GetRMS();
       y[24] = h1etacoefmin26_1->GetRMS();
       y[25] = h1etacoefmin27_1->GetRMS();
       y[26] = h1etacoefmin28_1->GetRMS();
       y[27] = h1etacoefmin29_1->GetRMS();
       y[28] = h1etacoefmin30_1->GetRMS();
       y[29] = h1etacoefmin31_1->GetRMS();
       y[30] = h1etacoefmin32_1->GetRMS();
       y[31] = h1etacoefmin33_1->GetRMS();
       y[32] = h1etacoefmin34_1->GetRMS();
       y[33] = h1etacoefmin35_1->GetRMS();
       y[34] = h1etacoefmin36_1->GetRMS();
       y[35] = h1etacoefmin37_1->GetRMS();
       y[36] = h1etacoefmin38_1->GetRMS();
       y[37] = h1etacoefmin39_1->GetRMS();
       y[38] = h1etacoefmin40_1->GetRMS();
       y[39] = h1etacoefmin41_1->GetRMS();
	*/  
	 
  TH2F* h3= new TH2F(" "," ",2,0.,42.,100,0.,0.15);
  h3->GetXaxis()->SetTitle("ieta");
  h3->GetYaxis()->SetTitle("RMS");
  h3->Draw();
  Int_t np=40;
  
  TGraph*  gr1 = new TGraph(np,x,y);
  gr1->Draw("P");
  c2h->Print("GIF/accuracy_minus_8.9mlnl.gif");
  c2h->Print("GIF/accuracy_minus_8.9mlnl.C");
  	   
	   	   
//           h1etacoefmin2D1->Draw("box");c2h->Print("GIF/h1etacoefminD1.gif");
//           h1etacoefmin2D1_noise->Draw("box");c2h->Print("GIF/h1etacoefminD1_noise.gif");
//           h1etacoefmin2D1->Draw("box");c2h->Print("GIF/h1etacoefminD1.gif");
//           h1etacoefmin2D1_noise->Draw("box");c2h->Print("GIF/h1etacoefminD1_noise.gif");
	   
//           h1etacoefmin2D16->Draw("box");c2h->Print("GIF/h1etacoefminD16.gif");
//           h1etacoefmin2D16_noise->Draw("box");c2h->Print("GIF/h1etacoefminD16_noise.gif");
//           h1etacoefmin2D16->Draw("box");c2h->Print("GIF/h1etacoefminD16.gif");
//           h1etacoefmin2D16_noise->Draw("box");c2h->Print("GIF/h1etacoefminD16_noise.gif");
  
}
