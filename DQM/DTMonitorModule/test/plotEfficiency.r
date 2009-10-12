
/*
 * Produces 1D plots of chamber efficiency starting from the DQM output files
 * of the DTChamberEfficiency and DTChamberEfficiencyTest modules.
 * 
 * 
 * G. Cerminara 2009
 */

void plotEfficiency(){
  // Load needed macros and files
  gROOT->LoadMacro("macros.C");     // Load service macros

  // Get the style
  TStyle * style = getStyle("tdr");
  style->cd();
  
  // retrieve the last open file
  TFile *file = gROOT->GetListOfFiles()->Last();
  if(file == 0) {
    cout << "No file loaded!Exiting." << endl;
    return;
  }
  cout << "Loading file: " << file->GetName() << endl;
  
  // retrieve the original histos from the file
  TH2F *histoEff_Whm2 = (TH2F *) file->Get("DQMData/Run 109459/DT/Run summary/05-ChamberEff/EfficiencyMap_All_W-2");
  if(histoEff_Whm2 != 0) {
    TCanvas *c = newCanvas(histoEff_Whm2->GetName(), histoEff_Whm2->GetName());
    histoEff_Whm2->Draw("COLZ");
    double sector[12];
    double sectErr[12];    

    double effiency[4][12];
    double efficiencyErr[4][12];

    for(int station = 1; station != 5; ++station) { //loop over stations
      for(int sect = 1; sect != 13; ++sect) { // loop over sectors
	sector[sect-1] = sect;
	effiency[station-1][sect-1] = histoEff_Whm2->GetBinContent(sect,station);
	efficiencyErr[station-1][sect-1] = histoEff_Whm2->GetBinError(sect,station);
	sectErr[sect-1] = 0.5;
      }
    }

    TGraphErrors *mb1Eff_Whm2 = new TGraphErrors(12, sector, effiency[0], sectErr, efficiencyErr[0]);
    TGraphErrors *mb2Eff_Whm2 = new TGraphErrors(12, sector, effiency[1], sectErr, efficiencyErr[1]);
    TGraphErrors *mb3Eff_Whm2 = new TGraphErrors(12, sector, effiency[2], sectErr, efficiencyErr[2]);
    TGraphErrors *mb4Eff_Whm2 = new TGraphErrors(12, sector, effiency[3], sectErr, efficiencyErr[3]);


    TCanvas *c1 = newCanvas("Efficiency_Whm2", "Efficiency_Whm2");
    mb1Eff_Whm2->Draw("AP");
    mb1Eff_Whm2->SetMarkerColor(1);

    mb2Eff_Whm2->Draw("P");
    mb2Eff_Whm2->SetMarkerColor(2);
    mb3Eff_Whm2->Draw("P");
    mb3Eff_Whm2->SetMarkerColor(3);
    mb4Eff_Whm2->Draw("P");
    mb4Eff_Whm2->SetMarkerColor(4);

    
  }

}

