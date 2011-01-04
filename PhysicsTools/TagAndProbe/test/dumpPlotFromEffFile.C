void  dumpPlotFromEffFile(char* dir="GsfElectronToIdToHLT") {

  TFile* f = new TFile("testEfficiency.root");
  f->cd(dir);

  TKey *key;
  TCanvas* c;
  TIter next( gDirectory->GetListOfKeys() ); 
  while ((key = (TKey*) next())) {

    TObject *obj = key->ReadObj();
    char* name = obj->GetName();

    if ( !(obj->IsA()->InheritsFrom( "TDirectory" )) ) continue;
    std::cout << "   ==================================================   " << std::endl;
    //TString dirName = Form("%s/cnt_eff_plots/", name);
    TString dirName = Form("%s/fit_eff_plots/", name);
    gDirectory->cd(dirName);
    c = (TCanvas*) gDirectory->Get("probe_gsfEle_pt_probe_gsfEle_eta_PLOT");
    gStyle->SetPalette(1);
    c->Draw();
    c->SaveAs( TString(name)+TString(".gif")); 


    //now make plot of the fit distributions
    gDirectory->cd("../");
    gDirectory->ls();

    TKey *innerkey;
    TIter innernext( gDirectory->GetListOfKeys() );
    while ((innerkey = (TKey*) innernext())) {
      obj = innerkey->ReadObj();
      char* innername = obj->GetName();
      if(!(obj->IsA()->InheritsFrom( "TDirectory" )) || 
	 !(TString(innername).Contains("_bin")) ) continue;
      gDirectory->cd(innername);
      c = (TCanvas*) gDirectory->Get("fit_canvas");
      c->Draw();
      TString plotname = TString("fit")+TString(name)+TString("_")+
	TString(innername)+TString(".gif");
      plotname.ReplaceAll("probe_gsfEle_", "");
      plotname.ReplaceAll("__pdfSignalPlusBackground", "");
      c->SaveAs(plotname); 
    }


    //get back to the initial directory
    gDirectory->cd("../../");
      std::cout << "   ==================================================   " << std::endl;
  }
  
} 
