//
// Macro to produce ECAL cosmic plots
//

int Wait() {
     cout << " Continue [<RET>|q]?  "; 
     char x;
     x = getchar();
     if ((x == 'q') || (x == 'Q')) return 1;
     return 0;
}

void DrawCosmicPlots(Char_t* infile = 0, Int_t runNum=0, Bool_t printPics = kTRUE, Char_t* fileType = "png", Char_t* dirName = ".", Bool_t doWait=kFALSE)
{

  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1,0); gStyle->SetOptStat(10);

  if (!infile) {
    cout << " No input file specified !" << endl;
    return;
  }

  cout << "Producing cosmics plots for: " << infile << endl;

  TFile* f = new TFile(infile);

  int runNumber = 0;
  if (runNum==0) {
    runNumber = runNumberHist->GetBinContent(1);
    cout << "Run Number: " << runNumber << endl;
  } else {
    runNumber = runNum;
  }

  char name[500];  

  const int nHists1=25;
  const int nHists2=17;
  const int nHists3=3+15;
  const int nHists4=8+12+8+1+6;
  const int nHists = nHists1+nHists2+nHists3+nHists4;
  //  const int nHists = 9;
  cout << nHists1 << " " << nHists2 << " " << nHists3 << " " << nHists4 << " " << nHists << endl;;

  TCanvas* c[nHists];
  char cname[100]; 

  for (int i=0; i<nHists1; i++) {
    sprintf(cname,"c%i",i);
    int x = (i%3)*600;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    cout << "Hists1 " << i << " : " << x << " , " << y << endl;
  }


//   TCanvas* c[nHists];
//   char cname[100]; 

//   for (int i=0; i<nHists; i++) {
//     sprintf(cname,"c%i",i);
//     int x = (i%3)*600;     //int x = (i%3)*600;
//     int y = (i/3)*300;     //int y = (i/3)*200;
//     c[i] =  new TCanvas(cname,cname,x,y,600,400);
//     cout << "Hists1 " << i << " : " << x << " , " << y << endl;
//   }

  char runChar[50];
  sprintf(runChar,", run %i",runNumber);

  c[0]->cd();
  gStyle->SetOptStat(111110);
  SeedEnergyAllFEDs->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",SeedEnergyAllFEDs->GetTitle()); 
  strcat(mytitle,runChar); SeedEnergyAllFEDs->SetTitle(mytitle);
  c[0]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_SeedEnergyAllFEDs_%i.%s",dirName,runNumber,fileType); c[0]->Print(name); }


  c[1]->cd();
  E2_AllClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",E2_AllClusters->GetTitle()); 
  strcat(mytitle,runChar); E2_AllClusters->SetTitle(mytitle);
  c[1]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_E2_AllClusters_%i.%s",dirName,runNumber,fileType); c[1]->Print(name); }

  c[2]->cd();  
  energy_AllClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",energy_AllClusters->GetTitle()); 
  strcat(mytitle,runChar); energy_AllClusters->SetTitle(mytitle);
  c[2]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_energy_AllClusters_%i.%s",dirName,runNumber,fileType); c[2]->Print(name); }

  c[3]->cd();  
  NumXtalsInClusterAllHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",NumXtalsInClusterAllHist->GetTitle()); 
  strcat(mytitle,runChar); NumXtalsInClusterAllHist->SetTitle(mytitle);
  c[3]->SetLogy(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_NumXtalsInClusterAllHist_%i.%s",dirName,runNumber,fileType); c[3]->Print(name); }

  c[4]->cd();
  energyvsE1_AllClusters->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",energyvsE1_AllClusters->GetTitle()); 
  strcat(mytitle,runChar); energyvsE1_AllClusters->SetTitle(mytitle);
  c[4]->SetLogy(0);
  c[4]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_energyvsE1_AllClusters_%i.%s",dirName,runNumber,fileType); c[4]->Print(name); }

  c[5]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse->SetTitle(mytitle);
  OccupancyAllEventsCoarse->SetMinimum(1);
  OccupancyAllEventsCoarse->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse->GetYaxis()->SetNdivisions(2);
  c[5]->SetLogy(0);
  c[5]->SetLogz(1);
  c[5]->SetGridx(1);
  c[5]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_%i.%s",dirName,runNumber,fileType); c[5]->Print(name); }
  
  c[6]->cd();
  OccupancyAllEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents->SetTitle(mytitle);
  OccupancyAllEvents->SetMinimum(1);  
  //        OccupancyAllEvents->SetMaximum(100);  
  OccupancyAllEvents->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents->GetYaxis()->SetNdivisions(2);
  c[6]->SetLogy(0);
  c[6]->SetLogz(1);
  c[6]->SetGridx(1);
  c[6]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_%i.%s",dirName,runNumber,fileType); c[6]->Print(name); }

  c[7]->cd();
  gStyle->SetOptStat(10);
  TrueOccupancyAllEventsCoarse->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",TrueOccupancyAllEventsCoarse->GetTitle()); 
  strcat(mytitle,runChar); TrueOccupancyAllEventsCoarse->SetTitle(mytitle);
  TrueOccupancyAllEventsCoarse->SetMinimum(1);
  TrueOccupancyAllEventsCoarse->GetXaxis()->SetNdivisions(-18);
  TrueOccupancyAllEventsCoarse->GetYaxis()->SetNdivisions(2);
  c[7]->SetLogy(0);
  c[7]->SetLogz(1);
  //  c[7]->SetGridx(1);
  //  c[7]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_TrueOccupancyAllEventsCoarse_%i.%s",dirName,runNumber,fileType); c[7]->Print(name); }
  
  c[8]->cd();
  TrueOccupancyAllEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",TrueOccupancyAllEvents->GetTitle()); 
  strcat(mytitle,runChar); TrueOccupancyAllEvents->SetTitle(mytitle);
  TrueOccupancyAllEvents->SetMinimum(1);  
  TrueOccupancyAllEvents->GetXaxis()->SetNdivisions(-18);
  TrueOccupancyAllEvents->GetYaxis()->SetNdivisions(2);
  c[8]->SetLogy(0);
  c[8]->SetLogz(1);
  //  c[8]->SetGridx(1);
  //  c[8]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_TrueOccupancyAllEvents_%i.%s",dirName,runNumber,fileType); c[8]->Print(name); }


  c[9]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds->SetTitle(mytitle);
  c[9]->SetLogy(0);
  c[9]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_%i.%s",dirName,runNumber,fileType); c[9]->Print(name); }

  c[10]->cd();
  timeVsAmpAllEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeVsAmpAllEvents->GetTitle()); 
  strcat(mytitle,runChar); timeVsAmpAllEvents->SetTitle(mytitle);
  timeVsAmpAllEvents->SetMinimum(1);
  c[10]->SetLogy(0);
  c[10]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeVsAmpAllFeds_%i.%s",dirName,runNumber,fileType); c[10]->Print(name); }

  c[11]->cd();
  timePhiAllFEDs->ProfileX()->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timePhiAllFEDs->GetTitle()); 
  strcat(mytitle,runChar); timePhiAllFEDs->SetTitle(mytitle);
  timePhiAllFEDs->ProfileX()->GetYaxis()->SetTitle("Relative Time (1 clock = 25ns)");
  c[11]->SetLogy(0);
  c[11]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiAllFeds_%i.%s",dirName,runNumber,fileType); c[11]->Print(name); }

  c[12]->cd();
  timeEBP->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBP->GetTitle()); 
  strcat(mytitle,runChar); timeEBP->SetTitle(mytitle);
  c[12]->SetLogy(0);
  c[12]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBP_%i.%s",dirName,runNumber,fileType); c[12]->Print(name); }

  c[13]->cd();
  timeEBM->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBM->GetTitle()); 
  strcat(mytitle,runChar); timeEBM->SetTitle(mytitle);
  c[13]->SetLogy(0);
  c[13]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBM_%i.%s",dirName,runNumber,fileType); c[13]->Print(name); }

  c[14]->cd();
  timeEBPTop->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBPTop->GetTitle()); 
  strcat(mytitle,runChar); timeEBPTop->SetTitle(mytitle);
  c[14]->SetLogy(0);
  c[14]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBPTop_%i.%s",dirName,runNumber,fileType); c[14]->Print(name); }

  c[15]->cd();
  timeEBMTop->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBMTop->GetTitle()); 
  strcat(mytitle,runChar); timeEBMTop->SetTitle(mytitle);
  c[15]->SetLogy(0);
  c[15]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBMTop_%i.%s",dirName,runNumber,fileType); c[15]->Print(name); }

  c[16]->cd();
  timeEBPBottom->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBPBottom->GetTitle()); 
  strcat(mytitle,runChar); timeEBPBottom->SetTitle(mytitle);
  c[16]->SetLogy(0);
  c[16]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBPBottom_%i.%s",dirName,runNumber,fileType); c[16]->Print(name); }

  c[17]->cd();
  timeEBMBottom->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeEBMBottom->GetTitle()); 
  strcat(mytitle,runChar); timeEBMBottom->SetTitle(mytitle);
  c[17]->SetLogy(0);
  c[17]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBMBottom_%i.%s",dirName,runNumber,fileType); c[17]->Print(name); }

  c[18]->cd();
  FrequencyAllEvent->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEvent->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEvent->SetTitle(mytitle);
  c[18]->SetLogy(0);
  c[18]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEvent_%i.%s",dirName,runNumber,fileType); c[18]->Print(name); }

  c[19]->cd();
  timeVsFreqAllEvent->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeVsFreqAllEvent->GetTitle()); 
  strcat(mytitle,runChar); timeVsFreqAllEvent->SetTitle(mytitle);
  timeVsFreqAllEvent->SetMinimum(1);
  c[19]->SetLogy(0);
  c[19]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeVsFreqAllFeds_%i.%s",dirName,runNumber,fileType); c[19]->Print(name); }

  c[20]->cd();
  numberofCosmicsPerEvent->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",numberofCosmicsPerEvent->GetTitle()); 
  strcat(mytitle,runChar); numberofCosmicsPerEvent->SetTitle(mytitle);
  c[20]->SetLogy(0);
  c[20]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofCosmicsPerEvent_%i.%s",dirName,runNumber,fileType); c[20]->Print(name); }

  c[21]->cd();
  frequencyOfGoodEvents->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",frequencyOfGoodEvents->GetTitle()); 
  strcat(mytitle,runChar); frequencyOfGoodEvents->SetTitle(mytitle);
  c[21]->SetLogy(0);
  c[21]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_frequencyOfGoodEvents_%i.%s",dirName,runNumber,fileType); c[21]->Print(name); }


  c[22]->cd();
  OccupancyAllEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s max forced",OccupancyAllEvents->GetTitle()); 
  //strcat(mytitle,runChar); 
  OccupancyAllEvents->SetTitle(mytitle);
  OccupancyAllEvents->SetMinimum(1);  
  //  OccupancyAllEvents->SetMaximum(100);  
  OccupancyAllEvents->SetMaximum(200);  
  OccupancyAllEvents->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents->GetYaxis()->SetNdivisions(2);
  c[22]->SetLogy(0);
  c[22]->SetLogz(1);
  c[22]->SetGridx(1);
  c[22]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsMax200_%i.%s",dirName,runNumber,fileType); c[22]->Print(name); }

  c[23]->cd();
  OccupancyAllEventsCoarse->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s max forced",OccupancyAllEventsCoarse->GetTitle()); 
  //strcat(mytitle,runChar); 
  OccupancyAllEventsCoarse->SetTitle(mytitle);
  OccupancyAllEventsCoarse->SetMinimum(1);  
  //  OccupancyAllEventsCoarse->SetMaximum(100);  
  OccupancyAllEventsCoarse->SetMaximum(5000);  
  OccupancyAllEventsCoarse->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse->GetYaxis()->SetNdivisions(2);
  c[23]->SetLogy(0);
  c[23]->SetLogz(1);
  c[23]->SetGridx(1);
  c[23]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarseMax5000_%i.%s",dirName,runNumber,fileType); c[23]->Print(name); }

  c[24]->cd();  
  energyHigh_AllClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",energyHigh_AllClusters->GetTitle()); 
  strcat(mytitle,runChar); energyHigh_AllClusters->SetTitle(mytitle);
  c[24]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_energyHigh_AllClusters_%i.%s",dirName,runNumber,fileType); c[24]->Print(name); }

  if (doWait) {
    if (Wait()) return;
  }

  for (int i=nHists1; i<nHists1+nHists2; i++) {
    sprintf(cname,"c%i",i);
    int x = ((i-nHists1)%3)*600;     //int x = (i%3)*600;
    int y = ((i-nHists1)/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    cout << "Hists2 " <<i << " : " << x << " , " << y << endl;
  }


  // triggered

  c[25]->cd();
  triggerHist->GetXaxis()->SetBinLabel(1,"ECAL");
  triggerHist->GetXaxis()->SetBinLabel(2,"HCAL");
  triggerHist->GetXaxis()->SetBinLabel(3,"DT");
  triggerHist->GetXaxis()->SetBinLabel(4,"RPC");
  triggerHist->GetXaxis()->SetBinLabel(5,"CSC");
  triggerHist->GetXaxis()->SetLabelSize(0.07);
  triggerHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",triggerHist->GetTitle()); 
  strcat(mytitle,runChar); triggerHist->SetTitle(mytitle);
  c[25]->SetLogy(0);
  c[25]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_triggerHist_%i.%s",dirName,runNumber,fileType); c[25]->Print(name); }


  c[26]->cd();
  triggerExclusiveHist->GetXaxis()->SetBinLabel(1,"ECAL");
  triggerExclusiveHist->GetXaxis()->SetBinLabel(2,"HCAL");
  triggerExclusiveHist->GetXaxis()->SetBinLabel(3,"DT");
  triggerExclusiveHist->GetXaxis()->SetBinLabel(4,"RPC");
  triggerExclusiveHist->GetXaxis()->SetBinLabel(5,"CSC");
  triggerExclusiveHist->GetXaxis()->SetLabelSize(0.07);
  triggerExclusiveHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",triggerExclusiveHist->GetTitle()); 
  strcat(mytitle,runChar); triggerExclusiveHist->SetTitle(mytitle);
  c[26]->SetLogy(0);
  c[26]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_triggerExclusiveHist_%i.%s",dirName,runNumber,fileType); c[26]->Print(name); }


  // ECAL triggered
  c[27]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ECAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ECAL->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ECAL->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ECAL->SetMaximum(500);
  OccupancyAllEventsCoarse_ECAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ECAL->GetYaxis()->SetNdivisions(2);
  c[27]->SetLogy(0);
  c[27]->SetLogz(1);
  c[27]->SetGridx(1);
  c[27]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ECAL_%i.%s",dirName,runNumber,fileType); c[27]->Print(name); }
  
  c[28]->cd();
  OccupancyAllEvents_ECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ECAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ECAL->SetTitle(mytitle);
    OccupancyAllEvents_ECAL->SetMinimum(1);  
    //    OccupancyAllEvents_ECAL->SetMaximum(500);  
  OccupancyAllEvents_ECAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ECAL->GetYaxis()->SetNdivisions(2);
  c[28]->SetLogy(0);
  c[28]->SetLogz(1);
  c[28]->SetGridx(1);
  c[28]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ECAL_%i.%s",dirName,runNumber,fileType); c[28]->Print(name); }

  c[29]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_ECAL->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_ECAL->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_ECAL->SetTitle(mytitle);
  c[29]->SetLogy(0);
  c[29]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_ECAL_%i.%s",dirName,runNumber,fileType); c[29]->Print(name); }


  // HCAL triggered
  c[30]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_HCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_HCAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_HCAL->SetTitle(mytitle);
  OccupancyAllEventsCoarse_HCAL->SetMinimum(1);
  //  OccupancyAllEventsCoarse_HCAL->SetMaximum(500);
  OccupancyAllEventsCoarse_HCAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_HCAL->GetYaxis()->SetNdivisions(2);
  c[30]->SetLogy(0);
  c[30]->SetLogz(1);
  c[30]->SetGridx(1);
  c[30]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_HCAL_%i.%s",dirName,runNumber,fileType); c[30]->Print(name); }
  
  c[31]->cd();
  OccupancyAllEvents_HCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_HCAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_HCAL->SetTitle(mytitle);
    OccupancyAllEvents_HCAL->SetMinimum(1);  
    //    OccupancyAllEvents_HCAL->SetMaximum(500);  
  OccupancyAllEvents_HCAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_HCAL->GetYaxis()->SetNdivisions(2);
  c[31]->SetLogy(0);
  c[31]->SetLogz(1);
  c[31]->SetGridx(1);
  c[31]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_HCAL_%i.%s",dirName,runNumber,fileType); c[31]->Print(name); }

  c[32]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_HCAL->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_HCAL->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_HCAL->SetTitle(mytitle);
  c[32]->SetLogy(0);
  c[32]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_HCAL_%i.%s",dirName,runNumber,fileType); c[32]->Print(name); }


  // DT triggered
  c[33]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_DT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_DT->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_DT->SetTitle(mytitle);
  OccupancyAllEventsCoarse_DT->SetMinimum(1);
  //  OccupancyAllEventsCoarse_DT->SetMaximum(500);
  OccupancyAllEventsCoarse_DT->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_DT->GetYaxis()->SetNdivisions(2);
  c[33]->SetLogy(0);
  c[33]->SetLogz(1);
  c[33]->SetGridx(1);
  c[33]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_DT_%i.%s",dirName,runNumber,fileType); c[33]->Print(name); }
  
  c[34]->cd();
  OccupancyAllEvents_DT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_DT->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_DT->SetTitle(mytitle);
    OccupancyAllEvents_DT->SetMinimum(1);  
    //    OccupancyAllEvents_DT->SetMaximum(500);  
  OccupancyAllEvents_DT->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_DT->GetYaxis()->SetNdivisions(2);
  c[34]->SetLogy(0);
  c[34]->SetLogz(1);
  c[34]->SetGridx(1);
  c[34]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_DT_%i.%s",dirName,runNumber,fileType); c[34]->Print(name); }

  c[35]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_DT->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_DT->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_DT->SetTitle(mytitle);
  c[35]->SetLogy(0);
  c[35]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_DT_%i.%s",dirName,runNumber,fileType); c[35]->Print(name); }


  // RPC triggered
  c[36]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_RPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_RPC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_RPC->SetTitle(mytitle);
  OccupancyAllEventsCoarse_RPC->SetMinimum(1);
  //  OccupancyAllEventsCoarse_RPC->SetMaximum(500);
  OccupancyAllEventsCoarse_RPC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_RPC->GetYaxis()->SetNdivisions(2);
  c[36]->SetLogy(0);
  c[36]->SetLogz(1);
  c[36]->SetGridx(1);
  c[36]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_RPC_%i.%s",dirName,runNumber,fileType); c[36]->Print(name); }
  
  c[37]->cd();
  OccupancyAllEvents_RPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_RPC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_RPC->SetTitle(mytitle);
    OccupancyAllEvents_RPC->SetMinimum(1);  
    //    OccupancyAllEvents_RPC->SetMaximum(500);  
  OccupancyAllEvents_RPC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_RPC->GetYaxis()->SetNdivisions(2);
  c[37]->SetLogy(0);
  c[37]->SetLogz(1);
  c[37]->SetGridx(1);
  c[37]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_RPC_%i.%s",dirName,runNumber,fileType); c[37]->Print(name); }

  c[38]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_RPC->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_RPC->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_RPC->SetTitle(mytitle);
  c[38]->SetLogy(0);
  c[38]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_RPC_%i.%s",dirName,runNumber,fileType); c[38]->Print(name); }



  // CSC triggered
  c[39]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_CSC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_CSC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_CSC->SetTitle(mytitle);
  OccupancyAllEventsCoarse_CSC->SetMinimum(1);
  //  OccupancyAllEventsCoarse_CSC->SetMaximum(500);
  OccupancyAllEventsCoarse_CSC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_CSC->GetYaxis()->SetNdivisions(2);
  c[39]->SetLogy(0);
  c[39]->SetLogz(1);
  c[39]->SetGridx(1);
  c[39]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_CSC_%i.%s",dirName,runNumber,fileType); c[39]->Print(name); }
  
  c[40]->cd();
  OccupancyAllEvents_CSC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_CSC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_CSC->SetTitle(mytitle);
  OccupancyAllEvents_CSC->SetMinimum(1);  
  //  OccupancyAllEvents_CSC->SetMaximum(500);  
  OccupancyAllEvents_CSC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_CSC->GetYaxis()->SetNdivisions(2);
  c[40]->SetLogy(0);
  c[40]->SetLogz(1);
  c[40]->SetGridx(1);
  c[40]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_CSC_%i.%s",dirName,runNumber,fileType); c[40]->Print(name); }

  c[41]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_CSC->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_CSC->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_CSC->SetTitle(mytitle);
  c[41]->SetLogy(0);
  c[41]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_CSC_%i.%s",dirName,runNumber,fileType); c[41]->Print(name); }
  

  if (doWait) {
    if (Wait()) return;
  }

  for (int i=nHists1+nHists2; i<nHists1+nHists2+nHists3; i++) {
    sprintf(cname,"c%i",i);
    int x = ((i-(nHists1+nHists2))%3)*600;     //int x = (i%3)*600;
    int y = ((i-(nHists1+nHists2))/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    cout << "Hists3 " <<i << " : " << x << " , " << y << endl;
  }

  // ECAL triggered
  c[42]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ExclusiveECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ExclusiveECAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ExclusiveECAL->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ExclusiveECAL->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ExclusiveECAL->SetMaximum(500);
  OccupancyAllEventsCoarse_ExclusiveECAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ExclusiveECAL->GetYaxis()->SetNdivisions(2);
  c[42]->SetLogy(0);
  c[42]->SetLogz(1);
  c[42]->SetGridx(1);
  c[42]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ExclusiveECAL_%i.%s",dirName,runNumber,fileType); c[42]->Print(name); }
  
  c[43]->cd();
  OccupancyAllEvents_ExclusiveECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ExclusiveECAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ExclusiveECAL->SetTitle(mytitle);
    OccupancyAllEvents_ExclusiveECAL->SetMinimum(1);  
    //    OccupancyAllEvents_ExclusiveECAL->SetMaximum(500);  
  OccupancyAllEvents_ExclusiveECAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ExclusiveECAL->GetYaxis()->SetNdivisions(2);
  c[43]->SetLogy(0);
  c[43]->SetLogz(1);
  c[43]->SetGridx(1);
  c[43]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ExclusiveECAL_%i.%s",dirName,runNumber,fileType); c[43]->Print(name); }


  // ExclusiveHCAL triggered
  c[44]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ExclusiveHCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ExclusiveHCAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ExclusiveHCAL->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ExclusiveHCAL->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ExclusiveHCAL->SetMaximum(500);
  OccupancyAllEventsCoarse_ExclusiveHCAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ExclusiveHCAL->GetYaxis()->SetNdivisions(2);
  c[44]->SetLogy(0);
  c[44]->SetLogz(1);
  c[44]->SetGridx(1);
  c[44]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ExclusiveHCAL_%i.%s",dirName,runNumber,fileType); c[44]->Print(name); }
  
  c[45]->cd();
  OccupancyAllEvents_ExclusiveHCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ExclusiveHCAL->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ExclusiveHCAL->SetTitle(mytitle);
    OccupancyAllEvents_ExclusiveHCAL->SetMinimum(1);  
    //    OccupancyAllEvents_ExclusiveHCAL->SetMaximum(500);  
  OccupancyAllEvents_ExclusiveHCAL->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ExclusiveHCAL->GetYaxis()->SetNdivisions(2);
  c[45]->SetLogy(0);
  c[45]->SetLogz(1);
  c[45]->SetGridx(1);
  c[45]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ExclusiveHCAL_%i.%s",dirName,runNumber,fileType); c[45]->Print(name); }


  // ExclusiveDT triggered
  c[46]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ExclusiveDT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ExclusiveDT->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ExclusiveDT->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ExclusiveDT->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ExclusiveDT->SetMaximum(500);
  OccupancyAllEventsCoarse_ExclusiveDT->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ExclusiveDT->GetYaxis()->SetNdivisions(2);
  c[46]->SetLogy(0);
  c[46]->SetLogz(1);
  c[46]->SetGridx(1);
  c[46]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ExclusiveDT_%i.%s",dirName,runNumber,fileType); c[46]->Print(name); }
  
  c[47]->cd();
  OccupancyAllEvents_ExclusiveDT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ExclusiveDT->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ExclusiveDT->SetTitle(mytitle);
    OccupancyAllEvents_ExclusiveDT->SetMinimum(1);  
    //    OccupancyAllEvents_ExclusiveDT->SetMaximum(500);  
  OccupancyAllEvents_ExclusiveDT->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ExclusiveDT->GetYaxis()->SetNdivisions(2);
  c[47]->SetLogy(0);
  c[47]->SetLogz(1);
  c[47]->SetGridx(1);
  c[47]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ExclusiveDT_%i.%s",dirName,runNumber,fileType); c[47]->Print(name); }


  // ExclusiveRPC triggered
  c[48]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ExclusiveRPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ExclusiveRPC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ExclusiveRPC->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ExclusiveRPC->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ExclusiveRPC->SetMaximum(500);
  OccupancyAllEventsCoarse_ExclusiveRPC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ExclusiveRPC->GetYaxis()->SetNdivisions(2);
  c[48]->SetLogy(0);
  c[48]->SetLogz(1);
  c[48]->SetGridx(1);
  c[48]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ExclusiveRPC_%i.%s",dirName,runNumber,fileType); c[48]->Print(name); }
  
  c[49]->cd();
  OccupancyAllEvents_ExclusiveRPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ExclusiveRPC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ExclusiveRPC->SetTitle(mytitle);
    OccupancyAllEvents_ExclusiveRPC->SetMinimum(1);  
    //    OccupancyAllEvents_ExclusiveRPC->SetMaximum(500);  
  OccupancyAllEvents_ExclusiveRPC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ExclusiveRPC->GetYaxis()->SetNdivisions(2);
  c[49]->SetLogy(0);
  c[49]->SetLogz(1);
  c[49]->SetGridx(1);
  c[49]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ExclusiveRPC_%i.%s",dirName,runNumber,fileType); c[49]->Print(name); }


  // ExclusiveCSC triggered
  c[50]->cd();
  gStyle->SetOptStat(10);
  OccupancyAllEventsCoarse_ExclusiveCSC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEventsCoarse_ExclusiveCSC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEventsCoarse_ExclusiveCSC->SetTitle(mytitle);
  OccupancyAllEventsCoarse_ExclusiveCSC->SetMinimum(1);
  //  OccupancyAllEventsCoarse_ExclusiveCSC->SetMaximum(500);
  OccupancyAllEventsCoarse_ExclusiveCSC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEventsCoarse_ExclusiveCSC->GetYaxis()->SetNdivisions(2);
  c[50]->SetLogy(0);
  c[50]->SetLogz(1);
  c[50]->SetGridx(1);
  c[50]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEventsCoarse_ExclusiveCSC_%i.%s",dirName,runNumber,fileType); c[50]->Print(name); }
  
  c[51]->cd();
  OccupancyAllEvents_ExclusiveCSC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyAllEvents_ExclusiveCSC->GetTitle()); 
  strcat(mytitle,runChar); OccupancyAllEvents_ExclusiveCSC->SetTitle(mytitle);
  OccupancyAllEvents_ExclusiveCSC->SetMinimum(1);  
  //  OccupancyAllEvents_ExclusiveCSC->SetMaximum(500);  
  OccupancyAllEvents_ExclusiveCSC->GetXaxis()->SetNdivisions(-18);
  OccupancyAllEvents_ExclusiveCSC->GetYaxis()->SetNdivisions(2);
  c[51]->SetLogy(0);
  c[51]->SetLogz(1);
  c[51]->SetGridx(1);
  c[51]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyAllEvents_ExclusiveCSC_%i.%s",dirName,runNumber,fileType); c[51]->Print(name); }

  c[52]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs->SetTitle(mytitle);
  c[52]->SetLogy(0);
  c[52]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_%i.%s",dirName,runNumber,fileType); c[52]->Print(name); }

  // ECAL
  c[53]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs_ECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_ECAL->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_ECAL->SetTitle(mytitle);
  c[53]->SetLogy(0);
  c[53]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_ECAL_%i.%s",dirName,runNumber,fileType); c[53]->Print(name); }

  // HCAL
  c[54]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs_HCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_HCAL->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_HCAL->SetTitle(mytitle);
  c[54]->SetLogy(0);
  c[54]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_HCAL_%i.%s",dirName,runNumber,fileType); c[54]->Print(name); }

  // DT
  c[55]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs_DT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_DT->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_DT->SetTitle(mytitle);
  c[55]->SetLogy(0);
  c[55]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_DT_%i.%s",dirName,runNumber,fileType); c[55]->Print(name); }

  // RPC
  c[56]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs_RPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_RPC->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_RPC->SetTitle(mytitle);
  c[56]->SetLogy(0);
  c[56]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_RPC_%i.%s",dirName,runNumber,fileType); c[56]->Print(name); }


  // CSC
  c[57]->cd();
  gStyle->SetOptStat(111110);
  timeLMAllFEDs_CSC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_CSC->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_CSC->SetTitle(mytitle);
  c[57]->SetLogy(0);
  c[57]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_CSC_%i.%s",dirName,runNumber,fileType); c[57]->Print(name); }


  c[58]->cd();
  OccupancyHighEnergyEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyHighEnergyEvents->GetTitle()); 
  strcat(mytitle,runChar); OccupancyHighEnergyEvents->SetTitle(mytitle);
  OccupancyHighEnergyEvents->SetMinimum(1);  
  //  OccupancyHighEnergyEvents->SetMaximum(100);  
  //  OccupancyHighEnergyEvents->SetMaximum(500);  
  OccupancyHighEnergyEvents->GetXaxis()->SetNdivisions(-18);
  OccupancyHighEnergyEvents->GetYaxis()->SetNdivisions(2);
  c[58]->SetLogy(0);
  c[58]->SetLogz(0);
  c[58]->SetGridx(1);
  c[58]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyHighEnergyEvents_%i.%s",dirName,runNumber,fileType); c[58]->Print(name); }

  c[59]->cd();
  OccupancyHighEnergyEventsCoarse->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancyHighEnergyEventsCoarse->GetTitle()); 
  strcat(mytitle,runChar); OccupancyHighEnergyEventsCoarse->SetTitle(mytitle);
  OccupancyHighEnergyEventsCoarse->SetMinimum(1);  
  //  OccupancyHighEnergyEventsCoarse->SetMaximum(100);  
  //  OccupancyHighEnergyEventsCoarse->SetMaximum(500);  
  OccupancyHighEnergyEventsCoarse->GetXaxis()->SetNdivisions(-18);
  OccupancyHighEnergyEventsCoarse->GetYaxis()->SetNdivisions(2);
  c[59]->SetLogy(0);
  c[59]->SetLogz(0);
  c[59]->SetGridx(1);
  c[59]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancyHighEnergyEventsCoarse_%i.%s",dirName,runNumber,fileType); c[59]->Print(name); }


  if (doWait) {
    if (Wait()) return;
  }

  for (int i=nHists1+nHists2+nHists3; i<nHists; i++) {
    sprintf(cname,"c%i",i);
    int x = ((i-(nHists1+nHists2+nHists3))%3)*600;     //int x = (i%3)*600;
    int y = ((i-(nHists1+nHists2+nHists3))/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    cout <<"Hists4 " << i << " : " << x << " , " << y << endl;
  }


  c[60]->cd();  
  NumActiveXtalsInClusterAllHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",  NumActiveXtalsInClusterAllHist->GetTitle()); 
  strcat(mytitle,runChar); NumActiveXtalsInClusterAllHist->SetTitle(mytitle);
  c[60]->SetLogy(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_NumActiveXtalsInClusterAllHist_%i.%s",dirName,runNumber,fileType); c[60]->Print(name); }

  c[61]->cd();  
  numberofBCinSC->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",  numberofBCinSC->GetTitle()); 
  strcat(mytitle,runChar); numberofBCinSC->SetTitle(mytitle);
  c[61]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofBCinSC_%i.%s",dirName,runNumber,fileType); c[61]->Print(name); }

  c[62]->cd();  
  //  numberofBCinSCphi->Draw("colz");
  numberofBCinSCphi->ProfileX()->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",  numberofBCinSCphi->GetTitle()); 
  strcat(mytitle,runChar); numberofBCinSCphi->SetTitle(mytitle);
  c[62]->SetLogy(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofBCinSCphi_%i.%s",dirName,runNumber,fileType); c[62]->Print(name); }


  c[63]->cd();
  gStyle->SetOptStat(10);
  BCTrueOccupancyAllEventsCoarse->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",BCTrueOccupancyAllEventsCoarse->GetTitle()); 
  strcat(mytitle,runChar); BCTrueOccupancyAllEventsCoarse->SetTitle(mytitle);
  BCTrueOccupancyAllEventsCoarse->SetMinimum(1);
  BCTrueOccupancyAllEventsCoarse->GetXaxis()->SetNdivisions(-18);
  BCTrueOccupancyAllEventsCoarse->GetYaxis()->SetNdivisions(2);
  c[63]->SetLogy(0);
  c[63]->SetLogz(1);
  //  c[63]->SetGridx(1);
  //  c[63]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_BCTrueOccupancyAllEventsCoarse_%i.%s",dirName,runNumber,fileType); c[63]->Print(name); }
  
  c[64]->cd();
  BCTrueOccupancyAllEvents->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",BCTrueOccupancyAllEvents->GetTitle()); 
  strcat(mytitle,runChar); BCTrueOccupancyAllEvents->SetTitle(mytitle);
  BCTrueOccupancyAllEvents->SetMinimum(1);  
  BCTrueOccupancyAllEvents->GetXaxis()->SetNdivisions(-18);
  BCTrueOccupancyAllEvents->GetYaxis()->SetNdivisions(2);
  c[64]->SetLogy(0);
  c[64]->SetLogz(1);
  //  c[64]->SetGridx(1);
  //  c[64]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_BCTrueOccupancyAllEvents_%i.%s",dirName,runNumber,fileType); c[64]->Print(name); }

  c[65]->cd();
  gStyle->SetOptStat(111110);
  //  NumXtalsVsEnergy->GetYaxis()->SetRange(0,30);
  //NumXtalsVsEnergy->SetMinimum(1);
  //  NumXtalsVsEnergy->Draw("colz");
  NumXtalsVsEnergy->ProfileX()->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",NumXtalsVsEnergy->GetTitle()); 
  strcat(mytitle,runChar); NumXtalsVsEnergy->SetTitle(mytitle);
  c[65]->SetLogy(0);
  c[65]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_NumXtalsVsEnergy_%i.%s",dirName,runNumber,fileType); c[65]->Print(name); }


  c[66]->cd();
  //  NumXtalsVsHighEnergy->GetYaxis()->SetRange(0,30);
  //  NumXtalsVsHighEnergy->SetMinimum(1);
  //  NumXtalsVsHighEnergy->Draw("colz");
  NumXtalsVsHighEnergy->ProfileX()->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",NumXtalsVsHighEnergy->GetTitle()); 
  strcat(mytitle,runChar); NumXtalsVsHighEnergy->SetTitle(mytitle);
  c[66]->SetLogy(0);
  c[66]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_NumXtalsVsHighEnergy_%i.%s",dirName,runNumber,fileType); c[66]->Print(name); }

  c[67]->cd();
  gStyle->SetOptStat(111110);
  timeForAllFeds_EcalMuon->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_EcalMuon->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_EcalMuon->SetTitle(mytitle);
  c[67]->SetLogy(0);
  c[67]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_EcalMuon_%i.%s",dirName,runNumber,fileType); c[67]->Print(name); }


  c[68]->cd();
  gStyle->SetOptStat(111110);
  numberofCosmicsWTrackPerEvent->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",numberofCosmicsWTrackPerEvent->GetTitle()); 
  strcat(mytitle,runChar); numberofCosmicsWTrackPerEvent->SetTitle(mytitle);
  c[68]->SetLogy(0);
  c[68]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofCosmicsWTrackPerEvent_%i.%s",dirName,runNumber,fileType); c[68]->Print(name); }

  c[69]->cd();
  gStyle->SetOptStat(111110);
  numberofCosmicsTopBottomPerEvent->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",numberofCosmicsTopBottomPerEvent->GetTitle()); 
  strcat(mytitle,runChar); numberofCosmicsTopBottomPerEvent->SetTitle(mytitle);
  c[69]->SetLogy(0);
  c[69]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofCosmicsTopBottomPerEvent_%i.%s",dirName,runNumber,fileType); c[69]->Print(name); }


  c[70]->cd();
  gStyle->SetOptStat(111110);
  numberofCrossedEcalCosmicsPerEvent->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",numberofCrossedEcalCosmicsPerEvent->GetTitle()); 
  strcat(mytitle,runChar); numberofCrossedEcalCosmicsPerEvent->SetTitle(mytitle);
  c[70]->SetLogy(0);
  c[70]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofCrossedEcalCosmicsPerEvent_%i.%s",dirName,runNumber,fileType); c[70]->Print(name); }


  c[71]->cd();
  gStyle->SetOptStat(111110);
  deltaRHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",deltaRHist->GetTitle()); 
  strcat(mytitle,runChar); deltaRHist->SetTitle(mytitle);
  c[71]->SetLogy(0);
  c[71]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_deltaRHist_%i.%s",dirName,runNumber,fileType); c[71]->Print(name); }


  c[72]->cd();
  gStyle->SetOptStat(111110);
  deltaIEtaHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",deltaIEtaHist->GetTitle()); 
  strcat(mytitle,runChar); deltaIEtaHist->SetTitle(mytitle);
  c[72]->SetLogy(0);
  c[72]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_deltaIEtaHist_%i.%s",dirName,runNumber,fileType); c[72]->Print(name); }

  c[73]->cd();
  gStyle->SetOptStat(111110);
  deltaIPhiHist->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",deltaIPhiHist->GetTitle()); 
  strcat(mytitle,runChar); deltaIPhiHist->SetTitle(mytitle);
  c[73]->SetLogy(0);
  c[73]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_deltaIPhiHist_%i.%s",dirName,runNumber,fileType); c[73]->Print(name); }

  c[74]->cd();
  gStyle->SetOptStat(111110);
  ratioAssocClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",ratioAssocClusters->GetTitle()); 
  strcat(mytitle,runChar); ratioAssocClusters->SetTitle(mytitle);
  c[74]->SetLogy(0);
  c[74]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_ratioAssocClusters_%i.%s",dirName,runNumber,fileType); c[74]->Print(name); }


  c[75]->cd();
  gStyle->SetOptStat(111110);
  ratioAssocTracks->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",ratioAssocTracks->GetTitle()); 
  strcat(mytitle,runChar); ratioAssocTracks->SetTitle(mytitle);
  c[75]->SetLogy(0);
  c[75]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_ratioAssocTracks_%i.%s",dirName,runNumber,fileType); c[75]->Print(name); }


  c[76]->cd();
  gStyle->SetOptStat(111110);
  deltaEtaDeltaPhi->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",deltaEtaDeltaPhi->GetTitle()); 
  strcat(mytitle,runChar); deltaEtaDeltaPhi->SetTitle(mytitle);
  c[76]->SetLogy(0);
  c[76]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_deltaEtaDeltaPhi_%i.%s",dirName,runNumber,fileType); c[76]->Print(name); }


  c[77]->cd();
  gStyle->SetOptStat(111110);
  seedTrackPhi->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",seedTrackPhi->GetTitle()); 
  strcat(mytitle,runChar); seedTrackPhi->SetTitle(mytitle);
  c[77]->SetLogy(0);
  c[77]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_seedTrackPhi_%i.%s",dirName,runNumber,fileType); c[77]->Print(name); }


  c[78]->cd();
  gStyle->SetOptStat(111110);
  seedTrackEta->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",seedTrackEta->GetTitle()); 
  strcat(mytitle,runChar); seedTrackEta->SetTitle(mytitle);
  c[78]->SetLogy(0);
  c[78]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_seedTrackEta_%i.%s",dirName,runNumber,fileType); c[78]->Print(name); }


  c[79]->cd();
  gStyle->SetOptStat(10);
  trackAssoc_muonsEcal->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",trackAssoc_muonsEcal->GetTitle()); 
  strcat(mytitle,runChar); trackAssoc_muonsEcal->SetTitle(mytitle);
  c[79]->SetLogy(0);
  c[79]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_trackAssoc_muonsEcal_%i.%s",dirName,runNumber,fileType); c[79]->Print(name); }

  c[80]->cd();  
  energyHigh_HighEnergyClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",energyHigh_HighEnergyClusters->GetTitle()); 
  strcat(mytitle,runChar); energyHigh_HighEnergyClusters->SetTitle(mytitle);
  c[80]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_energyHigh_HighEnergyClusters_%i.%s",dirName,runNumber,fileType); c[80]->Print(name); }

  c[81]->cd();  
  energy_SingleXtalClusters->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",energy_SingleXtalClusters->GetTitle()); 
  strcat(mytitle,runChar); energy_SingleXtalClusters->SetTitle(mytitle);
  c[81]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_energy_SingleXtalClusters_%i.%s",dirName,runNumber,fileType); c[81]->Print(name); }

  c[82]->cd();
  gStyle->SetOptStat(10);
  OccupancySingleXtal->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",OccupancySingleXtal->GetTitle()); 
  strcat(mytitle,runChar); OccupancySingleXtal->SetTitle(mytitle);
  OccupancySingleXtal->SetMinimum(1);  
  OccupancySingleXtal->GetXaxis()->SetNdivisions(-18);
  OccupancySingleXtal->GetYaxis()->SetNdivisions(2);
  c[82]->SetLogy(0);
  c[82]->SetLogz(0);
  c[82]->SetGridx(1);
  c[82]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_OccupancySingleXtal_%i.%s",dirName,runNumber,fileType); c[82]->Print(name); }

  f->cd("EventTiming");

  c[83]->cd();
  FrequencyAllEventsInTime->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEventsInTime->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEventsInTime->SetTitle(mytitle);
  c[83]->SetLogy(0);
  c[83]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEventsInTime_%i.%s",dirName,runNumber,fileType); c[83]->Print(name); }

  c[84]->cd();
  FrequencyAllEventsInTimeVsPhi->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEventsInTimeVsPhi->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEventsInTimeVsPhi->SetTitle(mytitle);
  c[84]->SetLogy(0);
  c[84]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEventsInTimeVsPhi_%i.%s",dirName,runNumber,fileType); c[84]->Print(name); }

  c[85]->cd();
  FrequencyAllEventsInTimeVsTTPhi->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEventsInTimeVsTTPhi->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEventsInTimeVsTTPhi->SetTitle(mytitle);
  c[85]->SetLogy(0);
  c[85]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEventsInTimeVsTTPhi_%i.%s",dirName,runNumber,fileType); c[85]->Print(name); }

  c[86]->cd();
  FrequencyAllEventsInTimeVsEta->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEventsInTimeVsEta->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEventsInTimeVsEta->SetTitle(mytitle);
  c[86]->SetLogy(0);
  c[86]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEventsInTimeVsEta_%i.%s",dirName,runNumber,fileType); c[86]->Print(name); }

  c[87]->cd();
  FrequencyAllEventsInTimeVsTTEta->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",FrequencyAllEventsInTimeVsTTEta->GetTitle()); 
  strcat(mytitle,runChar); FrequencyAllEventsInTimeVsTTEta->SetTitle(mytitle);
  c[87]->SetLogy(0);
  c[87]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_FrequencyAllEventsInTimeVsTTEta_%i.%s",dirName,runNumber,fileType); c[87]->Print(name); }

  f->cd();


  // BX plots
  c[88]->cd();
  dccBXErrorByFED->GetXaxis()->SetTitle("FED Number");
  dccBXErrorByFED->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",dccBXErrorByFED->GetTitle()); 
  strcat(mytitle,runChar); dccBXErrorByFED->SetTitle(mytitle);
  c[88]->SetLogy(0);
  //c[88]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccBXErrorByFED_%i.%s",dirName,runNumber,fileType); c[88]->Print(name); }

  c[89]->cd();
  dccOrbitErrorByFED->GetXaxis()->SetTitle("FED Number");
  dccOrbitErrorByFED->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",dccOrbitErrorByFED->GetTitle()); 
  strcat(mytitle,runChar); dccOrbitErrorByFED->SetTitle(mytitle);
  c[89]->SetLogy(0);
  //c[89]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccOrbitErrorByFED_%i.%s",dirName,runNumber,fileType); c[89]->Print(name); }
  
  c[90]->cd();
  dccRuntypeErrorByFED->GetXaxis()->SetTitle("FED Number");
  dccRuntypeErrorByFED->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",dccRuntypeErrorByFED->GetTitle()); 
  strcat(mytitle,runChar); dccRuntypeErrorByFED->SetTitle(mytitle);
  c[90]->SetLogy(0);
  //c[90]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccRuntypeErrorByFED_%i.%s",dirName,runNumber,fileType); c[90]->Print(name); }

  
  c[91]->cd();
  gStyle->SetOptStat(10);
  dccEventVsBx->GetYaxis()->SetBinLabel(1,"COSMIC");
  dccEventVsBx->GetYaxis()->SetBinLabel(2,"BEAMH4");
  dccEventVsBx->GetYaxis()->SetBinLabel(3,"BEAMH2");
  dccEventVsBx->GetYaxis()->SetBinLabel(4,"MTCC");
  dccEventVsBx->GetYaxis()->SetBinLabel(5,"LASER_STD");
  dccEventVsBx->GetYaxis()->SetBinLabel(6,"LASER_POWER_SCAN");
  dccEventVsBx->GetYaxis()->SetBinLabel(7,"LASER_DELAY_SCAN");
  dccEventVsBx->GetYaxis()->SetBinLabel(8,"TESTPULSE_SCAN_MEM");
  dccEventVsBx->GetYaxis()->SetBinLabel(9,"TESTPULSE_MGPA");
  dccEventVsBx->GetYaxis()->SetBinLabel(10,"PEDESTAL_STD");
  dccEventVsBx->GetYaxis()->SetBinLabel(11,"PEDESTAL_OFFSET_SCAN");
  dccEventVsBx->GetYaxis()->SetBinLabel(12,"PEDESTAL_25NS_SCAN");
  dccEventVsBx->GetYaxis()->SetBinLabel(13,"LED_STD");
  dccEventVsBx->GetYaxis()->SetBinLabel(14,"PHYSICS_GLOBAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(15,"COSMICS_GLOBAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(16,"HALO_GLOBAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(17,"LASER_GAP");
  dccEventVsBx->GetYaxis()->SetBinLabel(18,"TESTPULSE_GAP");
  dccEventVsBx->GetYaxis()->SetBinLabel(19,"PEDESTAL_GAP");
  dccEventVsBx->GetYaxis()->SetBinLabel(20,"LED_GAP");
  dccEventVsBx->GetYaxis()->SetBinLabel(21,"PHYSICS_LOCAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(22,"COSMICS_LOCAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(23,"HALO_LOCAL");
  dccEventVsBx->GetYaxis()->SetBinLabel(24,"CALIB_LOCAL");
  dccEventVsBx->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",dccEventVsBx->GetTitle()); 
  strcat(mytitle,runChar); dccEventVsBx->SetTitle(mytitle);
  //c[91]->SetLogy(0);
  c[91]->SetLogz(1);
  c[91]->SetCanvasSize(1200,500);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccEventVsBx_%i.%s",dirName,runNumber,fileType); c[91]->Print(name); }

  c[92]->cd();
  gStyle->SetOptStat(10);
  dccErrorVsBX->GetYaxis()->SetBinLabel(1,"BX");
  dccErrorVsBX->GetYaxis()->SetBinLabel(2,"Orbit");
  dccErrorVsBX->GetYaxis()->SetBinLabel(3,"Runtype");
  dccErrorVsBX->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",dccErrorVsBX->GetTitle()); 
  strcat(mytitle,runChar); dccErrorVsBX->SetTitle(mytitle);
  //c[92]->SetLogy(0);
  c[92]->SetLogz(1);
  c[92]->SetCanvasSize(1200,500);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccErrorVsBx_%i.%s",dirName,runNumber,fileType); c[92]->Print(name); }

  c[93]->cd();
  dccRuntype->GetXaxis()->SetBinLabel(1,"COSMIC");
  dccRuntype->GetXaxis()->SetBinLabel(2,"BEAMH4");
  dccRuntype->GetXaxis()->SetBinLabel(3,"BEAMH2");
  dccRuntype->GetXaxis()->SetBinLabel(4,"MTCC");
  dccRuntype->GetXaxis()->SetBinLabel(5,"LASER_STD");
  dccRuntype->GetXaxis()->SetBinLabel(6,"LASER_POWER_SCAN");
  dccRuntype->GetXaxis()->SetBinLabel(7,"LASER_DELAY_SCAN");
  dccRuntype->GetXaxis()->SetBinLabel(8,"TESTPULSE_SCAN_MEM");
  dccRuntype->GetXaxis()->SetBinLabel(9,"TESTPULSE_MGPA");
  dccRuntype->GetXaxis()->SetBinLabel(10,"PEDESTAL_STD");
  dccRuntype->GetXaxis()->SetBinLabel(11,"PEDESTAL_OFFSET_SCAN");
  dccRuntype->GetXaxis()->SetBinLabel(12,"PEDESTAL_25NS_SCAN");
  dccRuntype->GetXaxis()->SetBinLabel(13,"LED_STD");
  dccRuntype->GetXaxis()->SetBinLabel(14,"PHYSICS_GLOBAL");
  dccRuntype->GetXaxis()->SetBinLabel(15,"COSMICS_GLOBAL");
  dccRuntype->GetXaxis()->SetBinLabel(16,"HALO_GLOBAL");
  dccRuntype->GetXaxis()->SetBinLabel(17,"LASER_GAP");
  dccRuntype->GetXaxis()->SetBinLabel(18,"TESTPULSE_GAP");
  dccRuntype->GetXaxis()->SetBinLabel(19,"PEDESTAL_GAP");
  dccRuntype->GetXaxis()->SetBinLabel(20,"LED_GAP");
  dccRuntype->GetXaxis()->SetBinLabel(21,"PHYSICS_LOCAL");
  dccRuntype->GetXaxis()->SetBinLabel(22,"COSMICS_LOCAL");
  dccRuntype->GetXaxis()->SetBinLabel(23,"HALO_LOCAL");
  dccRuntype->GetXaxis()->SetBinLabel(24,"CALIB_LOCAL");
  dccRuntype->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",dccRuntype->GetTitle()); 
  strcat(mytitle,runChar); dccRuntype->SetTitle(mytitle);
  c[93]->SetLogy(1);
  //c[93]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_dccRuntype_%i.%s",dirName,runNumber,fileType); c[93]->Print(name); }

  
  c[20]->cd();
  numberofCosmicsPerEvent_EB->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",numberofCosmicsPerEvent_EB->GetTitle()); 
  strcat(mytitle,runChar); numberofCosmicsPerEvent_EB->SetTitle(mytitle);
  c[20]->SetLogy(0);
  c[20]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_numberofCosmicsPerEventEB_%i.%s",dirName,runNumber,fileType); c[20]->Print(name); }

  if (doWait) {
    if (Wait()) return;
  }

  // fancy timing plots
  //
  gStyle->SetOptStat(10);

  cTiming = new TCanvas("cTiming","(phi,eta,timing)",900,600);
  cTiming->cd(1);
  cTiming->SetGridx();
  cTiming->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {
      //printf("(%i,%i) -> %0.3f %0.3f\n",i,j,ayx->GetBinContent(i,j),ayx->GetBinError(i,j));
      
      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);
      //cout << "         (x,y) nbin nentries: (" << xcorr << "," << ycorr;
      //cout << ")  " << nBin  << " " << nBinEntries << endl;
      //sprintf(tempErr,"%0.2f",ayx->GetBinContent(i,j));

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_%i.%s",dirName,runNumber,fileType); cTiming->Print(name); }


  //  ( TT binning ) timeTTAllFEDs

  gStyle->SetOptStat(10);

  cTiming_TT = new TCanvas("cTiming_TT TT bin","(phi,eta,timing)",900,600);
  cTiming_TT->cd(1);
  cTiming_TT->SetGridx();
  cTiming_TT->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

//   int nxb = ayx->GetNbinsX();
//   int nyb = ayx->GetNbinsY();

//   char tempErr[200];
//   for (int i=1; i<=nxb; i++ ) {
//     for (int j=1; j<=nyb; j++ ) {
//       double xcorr = ayx->GetXaxis()->GetBinCenter(i);
//       double ycorr = ayx->GetYaxis()->GetBinCenter(j);
//       sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
//       int nBin = ayx->GetBin(i,j,0);
//       int nBinEntries = ayx->GetBinEntries(nBin);

//       // print RMS of mean
//       if (nBinEntries!=0) {
// 	tex = new TLatex(xcorr,ycorr,tempErr);
// 	tex->SetTextAlign(23);
// 	tex->SetTextFont(42);
// 	tex->SetTextSize(0.025);
// 	tex->SetLineWidth(2);
// 	tex->Draw();
//       }
      
//       // print number of bin entries
//       sprintf(tempErr,"%i",nBinEntries);
//       if (nBinEntries!=0) {
// 	tex = new TLatex(xcorr,ycorr,tempErr);
// 	tex->SetTextAlign(21);
// 	tex->SetTextFont(42);
// 	tex->SetTextSize(0.025);
// 	tex->SetLineWidth(2);
// 	tex->Draw();
//       }
//     }
//   }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_%i.%s",dirName,runNumber,fileType); cTiming_TT->Print(name); }



  // fancy timing plots (ECAL)
  //
  gStyle->SetOptStat(10);

  cTiming_ECAL = new TCanvas("cTiming_ECAL","(phi,eta,timing)",900,600);
  cTiming_ECAL->cd(1);
  cTiming_ECAL->SetGridx();
  cTiming_ECAL->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs_ECAL");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {      

      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_ECAL_%i.%s",dirName,runNumber,fileType); cTiming_ECAL->Print(name); }


  //  ( TT binning ) timeTTAllFEDs_ECAL

  gStyle->SetOptStat(10);

  cTiming_TT_ECAL = new TCanvas("cTiming_TT_ECAL TT bin","(phi,eta,timing)",900,600);
  cTiming_TT_ECAL->cd(1);
  cTiming_TT_ECAL->SetGridx();
  cTiming_TT_ECAL->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs_ECAL");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_ECAL_%i.%s",dirName,runNumber,fileType); cTiming_TT_ECAL->Print(name); }



  // fancy timing plots (HCAL)
  //
  gStyle->SetOptStat(10);

  cTiming_HCAL = new TCanvas("cTiming_HCAL","(phi,eta,timing)",900,600);
  cTiming_HCAL->cd(1);
  cTiming_HCAL->SetGridx();
  cTiming_HCAL->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs_HCAL");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {
      
      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_HCAL_%i.%s",dirName,runNumber,fileType); cTiming_HCAL->Print(name); }


  //  ( TT binning ) timeTTAllFEDs_HCAL

  gStyle->SetOptStat(10);

  cTiming_TT_HCAL = new TCanvas("cTiming_TT_HCAL TT bin","(phi,eta,timing)",900,600);
  cTiming_TT_HCAL->cd(1);
  cTiming_TT_HCAL->SetGridx();
  cTiming_TT_HCAL->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs_HCAL");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_HCAL_%i.%s",dirName,runNumber,fileType); cTiming_TT_HCAL->Print(name); }



  // fancy timing plots (DT)
  //
  gStyle->SetOptStat(10);

  cTiming_DT = new TCanvas("cTiming_DT","(phi,eta,timing)",900,600);
  cTiming_DT->cd(1);
  cTiming_DT->SetGridx();
  cTiming_DT->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs_DT");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {
      
      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_DT_%i.%s",dirName,runNumber,fileType); cTiming_DT->Print(name); }


  //  ( TT binning ) timeTTAllFEDs_DT

  gStyle->SetOptStat(10);

  cTiming_TT_DT = new TCanvas("cTiming_TT_DT TT bin","(phi,eta,timing)",900,600);
  cTiming_TT_DT->cd(1);
  cTiming_TT_DT->SetGridx();
  cTiming_TT_DT->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs_DT");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_DT_%i.%s",dirName,runNumber,fileType); cTiming_TT_DT->Print(name); }


  // fancy timing plots (Exclusive)
  //
  gStyle->SetOptStat(10);

  cTiming_RPC = new TCanvas("cTiming_RPC","(phi,eta,timing)",900,600);
  cTiming_RPC->cd(1);
  cTiming_RPC->SetGridx();
  cTiming_RPC->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs_RPC");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {
      
      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_RPC_%i.%s",dirName,runNumber,fileType); cTiming_RPC->Print(name); }


  //  ( TT binning ) timeTTAllFEDs_RPC

  gStyle->SetOptStat(10);

  cTiming_TT_RPC = new TCanvas("cTiming_TT_RPC TT bin","(phi,eta,timing)",900,600);
  cTiming_TT_RPC->cd(1);
  cTiming_TT_RPC->SetGridx();
  cTiming_TT_RPC->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs_RPC");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_RPC_%i.%s",dirName,runNumber,fileType); cTiming_TT_RPC->Print(name); }


  // fancy timing plots (CSC)
  //
  gStyle->SetOptStat(10);

  cTiming_CSC = new TCanvas("cTiming_CSC","(phi,eta,timing)",900,600);
  cTiming_CSC->cd(1);
  cTiming_CSC->SetGridx();
  cTiming_CSC->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timePhiEtaAllFEDs_CSC");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

   char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
   strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  int nxb = ayx->GetNbinsX();
  int nyb = ayx->GetNbinsY();

  char tempErr[200];
  for (int i=1; i<=nxb; i++ ) {
    for (int j=1; j<=nyb; j++ ) {
      
      double xcorr = ayx->GetXaxis()->GetBinCenter(i);
      double ycorr = ayx->GetYaxis()->GetBinCenter(j);
      sprintf(tempErr,"%0.2f",ayx->GetBinError(i,j));
      int nBin = ayx->GetBin(i,j,0);
      int nBinEntries = ayx->GetBinEntries(nBin);

      // print RMS of mean
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(23);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
      
      // print number of bin entries
      sprintf(tempErr,"%i",nBinEntries);
      if (nBinEntries!=0) {
	tex = new TLatex(xcorr,ycorr,tempErr);
	tex->SetTextAlign(21);
	tex->SetTextFont(42);
	tex->SetTextSize(0.025);
	tex->SetLineWidth(2);
	tex->Draw();
      }
    }
  }
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timePhiEtaAllFEDs_CSC_%i.%s",dirName,runNumber,fileType); cTiming_CSC->Print(name); }


  //  ( TT binning ) timeTTAllFEDs_CSC

  gStyle->SetOptStat(10);

  cTiming_TT_CSC = new TCanvas("cTiming_TT_CSC TT bin","(phi,eta,timing)",900,600);
  cTiming_TT_CSC->cd(1);
  cTiming_TT_CSC->SetGridx();
  cTiming_TT_CSC->SetGridy();

  TH3F* h1 = (TH3F*)f->Get("timeTTAllFEDs_CSC");
  TProfile2D* ayx = (TProfile2D*) h1->Project3DProfile("yx");
  ayx->Draw("colz");
  ayx->GetXaxis()->SetTitle("i#phi");
  ayx->GetYaxis()->SetTitle("i#eta");

  ayx->GetXaxis()->SetNdivisions(-18);
  ayx->GetYaxis()->SetNdivisions(2);

  char mytitle[100]; sprintf(mytitle,"%s",ayx->GetTitle()); 
  strcat(mytitle,runChar); ayx->SetTitle(mytitle);

  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeTTAllFEDs_CSC_%i.%s",dirName,runNumber,fileType); cTiming_TT_CSC->Print(name); }


  // SM by SM timing histos

  gStyle->SetOptStat(1110);

  TCanvas* cTiming_EBM = new TCanvas("cTiming_EBM","timing SM EB-",1500,600);
  cTiming_EBM->Divide(6,3);

  cTiming_EBM->cd(1);  
  TH1F* hT = (TH1F*)f->Get("EB-01/TimeFED610");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(2);  
  TH1F* hT = (TH1F*)f->Get("EB-02/TimeFED611");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(3);  
  TH1F* hT = (TH1F*)f->Get("EB-03/TimeFED612");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(4);  
  TH1F* hT = (TH1F*)f->Get("EB-04/TimeFED613");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(5);  
  TH1F* hT = (TH1F*)f->Get("EB-05/TimeFED614");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(6);  
  TH1F* hT = (TH1F*)f->Get("EB-06/TimeFED615");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(7);  
  TH1F* hT = (TH1F*)f->Get("EB-07/TimeFED616");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(8);  
  TH1F* hT = (TH1F*)f->Get("EB-08/TimeFED617");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(9); 
  TH1F* hT = (TH1F*)f->Get("EB-09/TimeFED618");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(10);  
  TH1F* hT = (TH1F*)f->Get("EB-10/TimeFED619");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(11);  
  TH1F* hT = (TH1F*)f->Get("EB-11/TimeFED620");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(12);  
  TH1F* hT = (TH1F*)f->Get("EB-12/TimeFED621");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(13);  
  TH1F* hT = (TH1F*)f->Get("EB-13/TimeFED622");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(14);  
  TH1F* hT = (TH1F*)f->Get("EB-14/TimeFED623");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(15);  
  TH1F* hT = (TH1F*)f->Get("EB-15/TimeFED624");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(16);  
  TH1F* hT = (TH1F*)f->Get("EB-16/TimeFED625");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(17);  
  TH1F* hT = (TH1F*)f->Get("EB-17/TimeFED626");  if ( hT ) { hT->Draw();}
  cTiming_EBM->cd(18);  
  TH1F* hT = (TH1F*)f->Get("EB-18/TimeFED627");  if ( hT ) { hT->Draw();}

  
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBMFEDbyFED_%i.%s",dirName,runNumber,fileType); cTiming_EBM->Print(name); }
  cout << name << endl;


  TCanvas* cTiming_EBP = new TCanvas("cTiming_EBP","timing SM EB+",1500,600);
  cTiming_EBP->Divide(6,3);

  cTiming_EBP->cd(1);  
  TH1F* hT = (TH1F*)f->Get("EB+01/TimeFED628");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(2);  
  TH1F* hT = (TH1F*)f->Get("EB+02/TimeFED629");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(3);  
  TH1F* hT = (TH1F*)f->Get("EB+03/TimeFED630");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(4);  
  TH1F* hT = (TH1F*)f->Get("EB+04/TimeFED631");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(5);  
  TH1F* hT = (TH1F*)f->Get("EB+05/TimeFED632");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(6);  
  TH1F* hT = (TH1F*)f->Get("EB+06/TimeFED633");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(7);  
  TH1F* hT = (TH1F*)f->Get("EB+07/TimeFED634");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(8);  
  TH1F* hT = (TH1F*)f->Get("EB+08/TimeFED635");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(9); 
  TH1F* hT = (TH1F*)f->Get("EB+09/TimeFED636");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(10);  
  TH1F* hT = (TH1F*)f->Get("EB+10/TimeFED637");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(11);  
  TH1F* hT = (TH1F*)f->Get("EB+11/TimeFED638");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(12);  
  TH1F* hT = (TH1F*)f->Get("EB+12/TimeFED639");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(13);  
  TH1F* hT = (TH1F*)f->Get("EB+13/TimeFED640");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(14);  
  TH1F* hT = (TH1F*)f->Get("EB+14/TimeFED641");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(15);  
  TH1F* hT = (TH1F*)f->Get("EB+15/TimeFED642");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(16);  
  TH1F* hT = (TH1F*)f->Get("EB+16/TimeFED643");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(17);  
  TH1F* hT = (TH1F*)f->Get("EB+17/TimeFED644");  if ( hT ) { hT->Draw();}
  cTiming_EBP->cd(18);  
  TH1F* hT = (TH1F*)f->Get("EB+18/TimeFED645");  if ( hT ) { hT->Draw();}
  
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeEBPFEDbyFED_%i.%s",dirName,runNumber,fileType); cTiming_EBP->Print(name); }  


int EBFEDID[36] = {
627, //"EB-18"
626, //"EB-17"
625, //"EB-16"
624, //"EB-15"
623, //"EB-14"
622, //"EB-13"
621, //"EB-12"
620, //"EB-11"
619, //"EB-10"
618, //"EB-09"
617, //"EB-08"
616, //"EB-07"
615, //"EB-06"
614, //"EB-05"
613, //"EB-04"
612, //"EB-03"
611, //"EB-02"
610, //"EB-01"
628, //"EB+01"
629, //"EB+02"
630, //"EB+03"
631, //"EB+04"
632, //"EB+05"
633, //"EB+06"
634, //"EB+07"
635, //"EB+08"
636, //"EB+09"
637, //"EB+10"
638, //"EB+11"
639, //"EB+12"
640, //"EB+13"
641, //"EB+14"
642, //"EB+15"
643, //"EB+16"
644, //"EB+17"
645, //"EB+18"
};

int EEFEDID[18] = {
603, //"EE-09"
602, //"EE-08"
601, //"EE-07"
609, //"EE-06"
608, //"EE-05"
607, //"EE-04"
606, //"EE-03"
605, //"EE-02"
604, //"EE-01"
649, //"EE+01"
650, //"EE+02"
651, //"EE+03"
652, //"EE+04"
653, //"EE+05"
654, //"EE+06"
646, //"EE+07"
647, //"EE+08"
648, //"EE+09"
};

  //Average Amplitudes & DCC comparisons
  const int mnHistsEB=36;
  TCanvas* dccCanvasEB[mnHistsEB];
  char apname[100];
  for (int i=0; i<mnHistsEB; i++) {
    sprintf(apname,"can%i",i);
    int x = (i%3)*600;     
    int y = (i/3)*100;     
    dccCanvasEB[i] =  new TCanvas(apname,apname,x,y,600,400);
    cout << "Hists " << i << " : " << x << " , " << y << endl;
  }
  
  int iEBFED = 0;
  for (int iEB=-18; iEB <19; iEB++) {

    if (iEB==0) iEB++;
    char SMstr[100];    
    if (iEB < 0) {
      if (abs(iEB)<10) {
	sprintf(SMstr,"-0%d",abs(iEB));
      } else {
	sprintf(SMstr,"%d",iEB);
      }
    } else {
      if (abs(iEB)<10) {
	sprintf(SMstr,"+0%d",iEB);
      } else {
	sprintf(SMstr,"+%d",iEB);
      }
    }

    sprintf(SMstr,"EB%s",SMstr);

    //    cout << SMstr << " " << iEBFED << " " << EBFEDID[iEBFED] << endl;
    char fname[100];
    sprintf(fname,"%s/DCCRuntypeVsBxFED_%i",SMstr,EBFEDID[iEBFED]);
    TH1F *dccHist  =  (TH1F*) f->Get(fname) ; 
    //    cout << fname << " " << dccHist << endl;

    if (dccHist){
      dccCanvasEB[iEBFED]->cd();

      dccHist->GetYaxis()->SetBinLabel(1,"BX");
      dccHist->GetYaxis()->SetBinLabel(2,"Orbit");
      dccHist->GetYaxis()->SetBinLabel(3,"Runtype");
      dccHist->Draw("colz");
      char mytitle[100]; sprintf(mytitle,"%s",dccHist->GetTitle()); 
      strcat(mytitle,runChar); dccErrorVsBX->SetTitle(mytitle);

      dccCanvasEB[iEBFED]->SetLogz(1);
      dccCanvasEB[iEBFED]->SetCanvasSize(1200,500);
      
      //      cout << dirName << " " << SMstr << " " << runNumber << " " << fileType << endl;
      sprintf(name,"%s/cosmicsAnalysis_DCCRuntypeVsBxFED_%s_%i.%s", dirName,SMstr,runNumber,fileType);
      if (printPics) { sprintf(name,"%s/cosmicsAnalysis_DCCRuntypeVsBxFED_%s_%i.%s", dirName,SMstr,runNumber,fileType); dccCanvasEB[iEBFED]->Print(name); }
    }     
    //    cout << fname << " " <<  name << endl;

    iEBFED++;

  }


  const int mnHistsEE=18;
  TCanvas* dccCanvasEE[mnHistsEE];
  char apname[100];
  for (int i=0; i<mnHistsEE; i++) {
    sprintf(apname,"can%i",i);
    int x = (i%3)*600;     
    int y = (i/3)*100;     
    dccCanvasEE[i] =  new TCanvas(apname,apname,x,y,600,400);
    cout << "Hists " << i << " : " << x << " , " << y << endl;
  }

  int iEEFED = 0;
  for (int iEE=-9; iEE <10; iEE++) {

    if (iEE==0) iEE++;
    char SMstr[100];    
    if (iEE < 0) {
      if (abs(iEE)<10) {
	sprintf(SMstr,"-0%d",abs(iEE));
      } else {
	sprintf(SMstr,"%d",iEE);
      }
    } else {
      if (abs(iEE)<10) {
	sprintf(SMstr,"+0%d",iEE);
      } else {
	sprintf(SMstr,"+%d",iEE);
      }
    }

    sprintf(SMstr,"EE%s",SMstr);
    cout << SMstr << " " << iEEFED << " " << EEFEDID[iEEFED] << endl;

    char fname[100];
    sprintf(fname,"%s/DCCRuntypeVsBxFED_%i",SMstr,EEFEDID[iEEFED]);
    TH1F *dccHist  =  (TH1F*) f->Get(fname) ; 
    cout << fname << " " << dccHist << endl;

    if (dccHist){
      cout << "found histo in EE" << endl;
      dccCanvasEE[iEEFED]->cd();

      dccHist->GetYaxis()->SetBinLabel(1,"BX");
      dccHist->GetYaxis()->SetBinLabel(2,"Orbit");
      dccHist->GetYaxis()->SetBinLabel(3,"Runtype");
      dccHist->Draw("colz");
      char mytitle[100]; sprintf(mytitle,"%s",dccHist->GetTitle()); 
      strcat(mytitle,runChar); dccErrorVsBX->SetTitle(mytitle);

      dccCanvasEE[iEEFED]->SetLogz(1);
      dccCanvasEE[iEEFED]->SetCanvasSize(1200,500);
      
      cout << dirName << " " << SMstr << " " << runNumber << " " << fileType << endl;
      sprintf(name,"%s/cosmicsAnalysis_DCCRuntypeVsBxFED_%s_%i.%s", dirName,SMstr,runNumber,fileType);
      if (printPics) { sprintf(name,"%s/cosmicsAnalysis_DCCRuntypeVsBxFED_%s_%i.%s", dirName,SMstr,runNumber,fileType); dccCanvasEE[iEEFED]->Print(name); }
    }     
    //    cout << fname << " " <<  name << endl;

    iEEFED++;

  }



  return;

}



