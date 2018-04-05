{
  gROOT->Reset();
  gStyle->SetPalette(1);
  gStyle                  -> SetOptStat(1111);
  
  TCanvas c2h("c2h","2-d options",10,10,800,600);
  c2h.SetFillColor(10);
  gPad->SetDrawOption("e1p");
  TFile f0("coefficients_219_val_barrel_minus_8.9mln.root");  
  
      float x[41]={1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.};
      float y[41]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

       y[0] = h1etacoefmin1->GetRMS();
       y[1] = h1etacoefmin2->GetRMS();
       y[2] = h1etacoefmin3->GetRMS();
       y[3] = h1etacoefmin4->GetRMS();
       y[4] = h1etacoefmin5->GetRMS();
       y[5] = h1etacoefmin6->GetRMS();
       y[6] = h1etacoefmin7->GetRMS();
       y[7] = h1etacoefmin8->GetRMS();
       y[8] = h1etacoefmin9->GetRMS();
       y[9] = h1etacoefmin10->GetRMS();
       y[10] = h1etacoefmin11->GetRMS();
       y[11] = h1etacoefmin12->GetRMS();
       y[12] = h1etacoefmin13->GetRMS();
       y[13] = h1etacoefmin14->GetRMS();
       y[14] = h1etacoefmin15->GetRMS();
       y[15] = h1etacoefmin16_1->GetRMS();
       
       h1etacoefmin1->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin1.gif");
       h1etacoefmin2->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin2.gif");
       h1etacoefmin3->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin3.gif");
       h1etacoefmin4->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin4.gif");
       h1etacoefmin5->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin5.gif");
       h1etacoefmin6->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin6.gif");
       h1etacoefmin7->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin7.gif");
       h1etacoefmin8->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin8.gif");
       h1etacoefmin9->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin9.gif");
       h1etacoefmin10->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin10.gif");
       h1etacoefmin11->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin11.gif");
       h1etacoefmin12->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin12.gif");
       h1etacoefmin13->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin13.gif");
       h1etacoefmin14->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin14.gif");
       h1etacoefmin15->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin15.gif");
       h1etacoefmin16_1->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin16_1.gif");
       
       h2etacoefmin1->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin1.gif");
       h2etacoefmin2->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin2.gif");
       h2etacoefmin3->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin3.gif");
       h2etacoefmin4->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin4.gif");
       h2etacoefmin5->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin5.gif");
       h2etacoefmin6->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin6.gif");
       h2etacoefmin7->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin7.gif");
       h2etacoefmin8->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin8.gif");
       h2etacoefmin9->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin9.gif");
       h2etacoefmin10->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin10.gif");
       h2etacoefmin11->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin11.gif");
       h2etacoefmin12->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin12.gif");
       h2etacoefmin13->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin13.gif");
       h2etacoefmin14->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin14.gif");
       h2etacoefmin15->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin15.gif");
       h2etacoefmin16_1->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin16_1.gif");

    TFile f1("coefficients_219_val_endcap_minus_8.9mln.root");     
       
       y[16] = h1etacoefmin17->GetRMS();
       y[17] = h1etacoefmin18->GetRMS();
       y[18] = h1etacoefmin19->GetRMS();
       y[19] = h1etacoefmin20->GetRMS();
       y[20] = h1etacoefmin21->GetRMS();
       y[21] = h1etacoefmin22->GetRMS();
       y[22] = h1etacoefmin23->GetRMS();
       y[23] = h1etacoefmin24->GetRMS();
       y[24] = h1etacoefmin25->GetRMS();
       y[25] = h1etacoefmin26->GetRMS();
       y[26] = h1etacoefmin27->GetRMS();
       y[27] = h1etacoefmin28->GetRMS();
       y[28] = h1etacoefmin29->GetRMS();


       h1etacoefmin17->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin17.gif");
       h1etacoefmin18->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin18.gif");
       h1etacoefmin19->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin19.gif");
       h1etacoefmin20->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin20.gif");
       h1etacoefmin21->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin21.gif");
       h1etacoefmin22->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin22.gif");
       h1etacoefmin23->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin23.gif");
       h1etacoefmin24->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin24.gif");
       h1etacoefmin25->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin25.gif");
       h1etacoefmin26->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin26.gif");
       h1etacoefmin27->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin27.gif");
       h1etacoefmin28->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin28.gif");
       h1etacoefmin29->Draw();c2h->Print("GIF/ALL/8.9mln/h1etacoefmin29.gif");

       h2etacoefmin17->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin17.gif");
       h2etacoefmin18->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin18.gif");
       h2etacoefmin19->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin19.gif");
       h2etacoefmin20->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin20.gif");
       h2etacoefmin21->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin21.gif");
       h2etacoefmin22->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin22.gif");
       h2etacoefmin23->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin23.gif");
       h2etacoefmin24->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin24.gif");
       h2etacoefmin25->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin25.gif");
       h2etacoefmin26->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin26.gif");
       h2etacoefmin27->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin27.gif");
       h2etacoefmin28->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin28.gif");
       h2etacoefmin29->Draw("box");c2h->Print("GIF/ALL/8.9mln/h2etacoefmin29.gif");


    TFile f2("coefficients_219_val_hf_minus_8.9mln.root");

       y[29] = h1etacoefmin30->GetRMS();
       y[30] = h1etacoefmin31->GetRMS();
       y[31] = h1etacoefmin32->GetRMS();
       y[32] = h1etacoefmin33->GetRMS();
       y[33] = h1etacoefmin34->GetRMS();
       y[34] = h1etacoefmin35->GetRMS();
       y[35] = h1etacoefmin36->GetRMS();
       y[36] = h1etacoefmin37->GetRMS();
       y[37] = h1etacoefmin38->GetRMS();
       y[38] = h1etacoefmin39->GetRMS();
       y[39] = h1etacoefmin40->GetRMS();
       y[40] = h1etacoefmin41->GetRMS();
       
       
       h1etacoefmin30->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin30.gif");
       h1etacoefmin31->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin31.gif");
       h1etacoefmin32->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin32.gif");
       h1etacoefmin33->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin33.gif");
       h1etacoefmin34->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin34.gif");
       h1etacoefmin35->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin35.gif");
       h1etacoefmin36->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin36.gif");
       h1etacoefmin37->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin37.gif");
       h1etacoefmin38->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin38.gif");
       h1etacoefmin39->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin39.gif");
       h1etacoefmin40->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin40.gif");
       h1etacoefmin41->Draw(); c2h->Print("GIF/ALL/8.9mln/h1etacoefmin41.gif");
       
       h2etacoefmin30->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin30.gif");
       h2etacoefmin31->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin31.gif");
       h2etacoefmin32->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin32.gif");
       h2etacoefmin33->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin33.gif");
       h2etacoefmin34->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin34.gif");
       h2etacoefmin35->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin35.gif");
       h2etacoefmin36->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin36.gif");
       h2etacoefmin37->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin37.gif");
       h2etacoefmin38->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin38.gif");
       h2etacoefmin39->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin39.gif");
       h2etacoefmin40->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin40.gif");
       h2etacoefmin41->Draw("box"); c2h->Print("GIF/ALL/8.9mln/h2etacoefmin41.gif");
       
       
   cout<<y[27]<<endl;	
	  
  TH2F* h3= new TH2F(" "," ",2,0.,42.,100,0.,0.15);
  h3->GetXaxis()->SetTitle("ieta");
  h3->GetYaxis()->SetTitle("RMS");
  h3->Draw();
  Int_t np=41;
  
  TGraph*  gr1 = new TGraph(np,x,y);
  gr1->Draw("P");
  c2h->Print("GIF/accuracy_minus_8.9mln.gif");
  c2h->Print("GIF/accuracy_minus_8.9mln.C");
  	     
}
