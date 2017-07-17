
/* ///////////////////////////////////////////////////////////////////////
 Example usages:


 ///////////////////////////////////////////////////////////////////////  
*/



void  dumpScaleFactorTables() {

// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "WP90");
// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "WP85");
// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "WP80");

// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "CicLoose");
// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "CicTight");
// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "CicSuperTight");
// dumpScaleFactorTables("efficiency-data-GsfElectronToId.root", "efficiency-mc-GsfElectronToId.root", "GsfElectronToId", "CicHyperTight1");

// dumpScaleFactorTables("efficiency-data-SCToGsfElectron.root", "efficiency-mc-SCToGsfElectron.root", "SuperClusterToGsfElectron", "efficiency");
 dumpScaleFactorTables("Eff-defaultBin-data/efficiency-data-WP90ToHLT.root", "efficiency-mc-WP90ToHLT.root", "WP90ToHLT", "efficiency");
 dumpScaleFactorTables("Eff-defaultBin-data/efficiency-data-WP85ToHLT.root", "efficiency-mc-WP85ToHLT.root", "WP85ToHLT", "efficiency");
 dumpScaleFactorTables("Eff-defaultBin-data/efficiency-data-WP80ToHLT.root", "efficiency-mc-WP80ToHLT.root", "WP80ToHLT", "efficiency");
 dumpScaleFactorTables("Eff-defaultBin-data/efficiency-data-CicLooseToHLT.root", "efficiency-mc-CicLooseToHLT.root", "CicLooseToHLT", "efficiency");
 dumpScaleFactorTables("Eff-defaultBin-data/efficiency-data-CicTightToHLT.root", "efficiency-mc-CicTightToHLT.root", "CicTightToHLT", "efficiency");
}


void  dumpScaleFactorTables( char* dataFileName,
			     char* mcFileName,
			     char* dir="GsfElectronToId", 
			     char* subdir="WP90") {

  char temp[50];
  char* canvasname = "probe_sc_et_probe_sc_eta_PLOT";
  if(dir=="SuperClusterToGsfElectron") canvasname = "probe_et_probe_eta_PLOT";


  TFile* fData = new TFile(dataFileName);
  TFile* fMC = new TFile(mcFileName);
  TH2F* hData;
  TH2F* hMC;
  TCanvas* c;


  sprintf(temp, "%s/%s/fit_eff_plots/", dir, subdir);
  if(TString(temp).Contains("ToHLT")) 
    sprintf(temp, "%s/%s/cnt_eff_plots/", dir, subdir);
  fData->cd(temp);

  c = (TCanvas*) gDirectory->Get( canvasname );
  if(c==0) continue; // do nothing if the canvas doesn't exist
  TH2F* hData = (TH2F*) c->FindObject(canvasname );
  if( hData==0 ) continue; 



  sprintf(temp, "%s/%s/cnt_eff_plots/", dir, subdir);
  fMC->cd(temp);
  c = (TCanvas*) gDirectory->Get( canvasname );
  if(c==0) continue; // do nothing if the canvas doesn't exist
  TH2F* hMC = (TH2F*) c->FindObject(canvasname );
  if( hMC==0 ) continue; 


  sprintf(temp, "ScaleFactor_%s_%s.txt", dir, subdir);
  hData->Divide(hMC);
  makeTable( hData, temp);
 
  fData->Close();
  fMC->Close();
  delete fData;
  delete fMC;

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
