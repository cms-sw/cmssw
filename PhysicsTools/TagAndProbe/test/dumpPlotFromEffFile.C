///////////////////////////////////////////////////////////////////////////////////
////////// Example usages:
//////////  Data: 
//////////    dumpPlotFromEffFile.C("efficiency-data-GsfElectronToId.root", "GsfElectronToId", false, false)
//////////    dumpPlotFromEffFile.C("efficiency-data-SCToGsfElectron.root", "SuperClusterToGsfElectron", false, false)
//////////    dumpPlotFromEffFile.C("efficiency-data-WP90ToHLT.root", "WP90ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-data-WP85ToHLT.root", "WP85ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-data-WP80ToHLT.root", "WP80ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-data-CicLooseToHLT.root", "CicLooseToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-data-CicTightToHLT.root", "CicTightToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-data-CicSuperTightToHLT.root", "CicSuperTightToHLT")
//////////  MC: 
//////////    dumpPlotFromEffFile.C("efficiency-mc-GsfElectronToId.root", "GsfElectronToId")
//////////    dumpPlotFromEffFile.C("efficiency-mc-SCToGsfElectron.root", "SuperClusterToGsfElectron")
//////////    dumpPlotFromEffFile.C("efficiency-mc-WP90ToHLT.root", "WP90ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-mc-WP85ToHLT.root", "WP85ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-mc-WP80ToHLT.root", "WP80ToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-mc-CicLooseToHLT.root", "CicLooseToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-mc-CicTightToHLT.root", "CicTightToHLT")
//////////    dumpPlotFromEffFile.C("efficiency-mc-CicSuperTightToHLT.root", "CicSuperTightToHLT")



void  dumpPlotFromEffFile( char* inputFileName = "testEfficiency_data.root",
			   char* dir="GsfElectronToId", 
			   bool isCutAndCount=true, 
			   bool isMCTruth = false ) {

  std::cout << "   ##################################################   " << std::endl;
  if(isCutAndCount) {
    std::cout << "Plotting efficiency from cut & count. No background subtraction performed !" << std::endl;
    std::cout << "If you want to plot MC truth efficiency, please set: isMCTruth = true." << std::endl;
  }
  else std::cout << "Plotting efficiency from simultaneous fit." << std::endl;
  std::cout << "   ##################################################   " << std::endl;



  TFile* f = new TFile(inputFileName);
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
    cout << "****************************dirName = " << dirName << endl;
    gDirectory->cd(dirName);
    char* canvasname = "probe_sc_et_probe_sc_eta_PLOT";
    if(isMCTruth) canvasname = "probe_sc_et_probe_sc_eta_PLOT_mcTrue_true";
    if(dir=="SuperClusterToGsfElectron") canvasname = "probe_et_probe_eta_PLOT";
    if(dir=="SuperClusterToGsfElectron" && isMCTruth) 
      canvasname = "probe_et_probe_eta_PLOT_mcTrue_true";


    c = (TCanvas*) gDirectory->Get( canvasname );
    if(c==0) continue; // do nothing if the canvas doesn't exist
    gStyle->SetPalette(1);
    gStyle->SetPaintTextFormat(".2f"); 
    TH2F* h = (TH2F*) c->FindObject(canvasname );
    c->Draw();
    c->SaveAs( TString(name)+TString(".png")); 
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
      plotname.ReplaceAll("probe_sc_", "");
      plotname.ReplaceAll("__pdfSignalPlusBackground", "");
      c->SaveAs(plotname); 
      gDirectory->cd("../");
      } // end while innerkey
    } // end isCutAndCount


    //get back to the initial directory
    if( !isCutAndCount ) gDirectory->cd("../");
    else gDirectory->cd("../../");

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
