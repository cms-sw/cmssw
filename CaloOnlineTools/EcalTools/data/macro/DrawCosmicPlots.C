//
// Macro to produce ECAL cosmic plots
//
void DrawCosmicPlots(Char_t* infile = 0, Int_t runNum=0, Bool_t printPics = kTRUE, Char_t* fileType = "png", Char_t* dirName = ".")
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

  char name[100];  

  const int nHists1=25;
  const int nHists2=17;
  const int nHists3=3+15;
  const int nHists = nHists1+nHists2+nHists3;
  cout << nHists1 << " " << nHists2 << " " << nHists3 << " " << nHists << endl;;

  TCanvas* c[nHists];
  char cname[100]; 

  for (int i=0; i<nHists1; i++) {
    sprintf(cname,"c%i",i);
    int x = (i%3)*600;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    //    cout << i << " : " << x << " , " << y << endl;
  }

  for (int i=nHists1; i<nHists2; i++) {
    sprintf(cname,"c%i",i);
    int x = ((i-nHists1)%3)*600;     //int x = (i%3)*600;
    int y = ((i-nHists1)/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    //    cout << i << " : " << x << " , " << y << endl;
  }


  for (int i=nHists2; i<nHists; i++) {
    sprintf(cname,"c%i",i);
    int x = ((i-nHists2)%3)*600;     //int x = (i%3)*600;
    int y = ((i-nHists2)/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,600,400);
    //    cout << i << " : " << x << " , " << y << endl;
  }


  char runChar[50];
  sprintf(runChar,", run %i",runNumber);

  c[0]->cd();
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
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
  gStyle->SetOptStat(1110);
  timeForAllFeds_CSC->Draw();
  char mytitle[100]; sprintf(mytitle,"%s",timeForAllFeds_CSC->GetTitle()); 
  strcat(mytitle,runChar); timeForAllFeds_CSC->SetTitle(mytitle);
  c[41]->SetLogy(0);
  c[41]->SetLogz(0);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeForAllFeds_CSC_%i.%s",dirName,runNumber,fileType); c[41]->Print(name); }
  


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
  gStyle->SetOptStat(1110);
  timeLMAllFEDs->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs->SetTitle(mytitle);
  c[52]->SetLogy(0);
  c[52]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_%i.%s",dirName,runNumber,fileType); c[52]->Print(name); }

  // ECAL
  c[53]->cd();
  gStyle->SetOptStat(1110);
  timeLMAllFEDs_ECAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_ECAL->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_ECAL->SetTitle(mytitle);
  c[53]->SetLogy(0);
  c[53]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_ECAL_%i.%s",dirName,runNumber,fileType); c[53]->Print(name); }

  // HCAL
  c[54]->cd();
  gStyle->SetOptStat(1110);
  timeLMAllFEDs_HCAL->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_HCAL->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_HCAL->SetTitle(mytitle);
  c[54]->SetLogy(0);
  c[54]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_HCAL_%i.%s",dirName,runNumber,fileType); c[54]->Print(name); }

  // DT
  c[55]->cd();
  gStyle->SetOptStat(1110);
  timeLMAllFEDs_DT->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_DT->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_DT->SetTitle(mytitle);
  c[55]->SetLogy(0);
  c[55]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_DT_%i.%s",dirName,runNumber,fileType); c[55]->Print(name); }

  // RPC
  c[56]->cd();
  gStyle->SetOptStat(1110);
  timeLMAllFEDs_RPC->Draw("colz");
  char mytitle[100]; sprintf(mytitle,"%s",timeLMAllFEDs_RPC->GetTitle()); 
  strcat(mytitle,runChar); timeLMAllFEDs_RPC->SetTitle(mytitle);
  c[56]->SetLogy(0);
  c[56]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/cosmicsAnalysis_timeLMAllFEDs_RPC_%i.%s",dirName,runNumber,fileType); c[56]->Print(name); }


  // CSC
  c[57]->cd();
  gStyle->SetOptStat(1110);
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


  return;

}
