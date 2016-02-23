
void pushHistogramIntoFile(const char* tag){

   //   const char* tag = "HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0";
   TFile* file1 = new TFile("efficiency.root","read");
   TFile* file2 = new TFile("../data/CentralityTables.root","update");
   file2->cd(tag);
   
   TH1D* h = (TH1D*)file1->Get("hEff");
   h->Write();
   file2->Write();

}

