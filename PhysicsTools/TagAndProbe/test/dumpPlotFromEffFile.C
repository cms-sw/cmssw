void  dumpPlotFromEffFile(char* dir="GsfElectronToIdToHLT", bool isCutAndCount=true) {

  std::cout << "   ##################################################   " << std::endl;
  if(isCutAndCount) {
    std::cout << "Plotting efficiency from cut & count. No background subtraction performed !" << std::endl;
    std::cout << "If you want to plot MC truth efficiency, please set: isMCTruth = true." << std::endl;
  }
  else std::cout << "Plotting efficiency from simultaneous fit." << std::endl;
  std::cout << "   ##################################################   " << std::endl;

  const bool isMCTruth = false;

  TFile* f = new TFile("testEfficiency_data.root");
  f->cd(dir);

  TKey *key;
  TCanvas* c;
  TIter next( gDirectory->GetListOfKeys() ); 
  while ((key = (TKey*) next())) {

    TObject *obj = key->ReadObj();
    char* name = obj->GetName();

    if ( !(obj->IsA()->InheritsFrom( "TDirectory" )) ) continue;
    if( !isMCTruth && TString(name).Contains("MCtruth_") ) continue;
    if( isMCTruth && !(TString(name).Contains("MCtruth_")) ) continue;

    std::cout << "   ==================================================   " << std::endl;
    TString dirName = Form("%s/cnt_eff_plots/", name);
    if( !isCutAndCount ) dirName = Form("%s/fit_eff_plots/", name);
    gDirectory->cd(dirName);
    char* canvasname = "probe_gsfEle_pt_probe_gsfEle_eta_PLOT";
    if(isMCTruth) canvasname = "probe_gsfEle_pt_probe_gsfEle_eta_PLOT_mcTrue_true";
    c = (TCanvas*) gDirectory->Get( canvasname );
    if(c==0) continue; // do nothing if the canvas doesn't exist
    gStyle->SetPalette(1);
    c->Draw();
    c->SaveAs( TString(name)+TString(".gif")); 
    TH2F* h = (TH2F*) c->FindObject(canvasname );
    makeTable( h, (const char*) (TString(name)+TString(".txt")) );


    if( !isCutAndCount ) {
      
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
      } // end while innerkey
    } // end isCutAndCount


    //get back to the initial directory
    gDirectory->cd("../../");
      std::cout << "   ==================================================   " << std::endl;
  }
  
} 





// generic method to make a efficiency table
void makeTable(TH2F* h, char* tablefilename)
{

  int nX = h->GetNbinsX();
  int nY = h->GetNbinsY();


  FILE *file = fopen(tablefilename,"w+");


  for(int i=1; i<=nX; ++i) {
  
    Double_t pT0 = h->GetXaxis()->GetBinLowEdge(i);
    Double_t pT1 = h->GetXaxis()->GetBinLowEdge(i+1);

    for(int j=1; j<=nY; ++j) {
      Double_t x = h->GetBinContent(i,j);
      Double_t dx = h->GetBinError(i,j);
      Double_t eta0 = h->GetYaxis()->GetBinLowEdge(j);
      Double_t eta1 = h->GetYaxis()->GetBinLowEdge(j+1);

      fprintf( file ,"%4.1f  %4.1f   %+6.4f   %+6.4f  %6.4f   %6.4f \n", 
	       pT0, pT1, eta0, eta1, x, dx);
    }
  }

  fclose(file);
}
