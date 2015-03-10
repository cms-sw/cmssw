int Wait() {
     cout << " Continue [<RET>|q]?  "; 
     char x;
     x = getchar();
     if ((x == 'q') || (x == 'Q')) return 1;
     return 0;
}

void DrawLaserPlots(Char_t* infile = 0, Int_t runNum=0, Bool_t printPics = kTRUE, Char_t* fileType = "png", Char_t* dirName = ".", Bool_t doWait=kFALSE, Char_t* mType = "Laser")
{

  gROOT->SetStyle("Plain");
  gStyle->SetNumberContours(99);
  gStyle->SetPalette(1,0); gStyle->SetOptStat(10);

  if (!infile) {
    cout << " No input file specified !" << endl;
    return;
  }

  cout << "Producing Laser plots for: " << infile << endl;

  TFile* f = new TFile(infile);
  f->cd(); //added by jason for completeness

  int runNumber = 0;
  runNumber = runNum;


  char name[100];  
  char mytitle[200];

  const int nHists1=80;
  const int nHists = nHists1;
  //  const int nHists = 9;
  cout << nHists1 << " " << nHists << endl;;

  TCanvas* c[nHists];
  char cname[100]; 

  for (int i=0; i<nHists1; i++) {
    sprintf(cname,"c%i",i);
    int x = (i%3)*600;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,900,600);
    cout << "Hists1 " << i << " : " << x << " , " << y << endl;
  }

  char runChar[50];
  sprintf(runChar,"Run %i ",runNumber);
  
  //TTree helpers
  int tbins = 52;
  double tbinsL = -26.;
  double tbinsH = 26.;
  
//First thing is do print the profiles

  //Timing by FED/SM  
  c[0]->cd();
  gStyle->SetOptStat(10);
  TProfile *SM_timing = (TProfile*) f->Get("SM_timing");
  customizeTProfile(SM_timing);
  SM_timing->Draw();
   sprintf(mytitle,"%s %s",runChar,SM_timing->GetTitle()); 
  SM_timing->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_SM_timing_%i.%s",dirName,mType,runNumber,fileType); c[0]->Print(name); }

  c[1]->cd();
  gStyle->SetOptStat(10);
  TH1F *SM_timingh = CorrectProfToHist(SM_timing,"SM_timingh",-5,25.0);
  customizeTHist(SM_timingh);
  SM_timingh->Draw("p");
   sprintf(mytitle,"%s %s to optimal;FED;Time (ns)",runChar,SM_timing->GetTitle()); 
  SM_timingh->SetMinimum(-30.);
  SM_timingh->SetMaximum(50.);
  SM_timingh->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_SM_timingCorrected_%i.%s",dirName,mType,runNumber,fileType); c[1]->Print(name); }

  //Timing by LM
  c[2]->cd();
  gStyle->SetOptStat(10);
  TProfile *LM_timing = (TProfile*) f->Get("LM_timing");
  customizeTProfile(LM_timing);
  LM_timing->Draw();
   sprintf(mytitle,"%s %s",runChar,LM_timing->GetTitle()); 
  LM_timing->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_LM_timing_%i.%s",dirName,mType,runNumber,fileType); c[2]->Print(name); }

  c[3]->cd();
  gStyle->SetOptStat(10);
  TH1F *LM_timingh = CorrectProfToHist(LM_timing,"LM_timingh",-5,25.0);
  customizeTHist(LM_timingh);
  LM_timingh->Draw("p");
   sprintf(mytitle,"%s %s to optimal;FED;Time (ns)",runChar,LM_timing->GetTitle()); 
  LM_timingh->SetMinimum(-30.);
  LM_timingh->SetMaximum(50.);
  LM_timingh->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_LM_timingCorrected_%i.%s",dirName,mType,runNumber,fileType); c[3]->Print(name); }
 

  //Timing within the towers
  c[4]->cd();
  gStyle->SetOptStat(1111);
  TProfile *Inside_TT_timing = (TProfile*) f->Get("Inside_TT_timing");
  Inside_TT_timing->Draw();
   sprintf(mytitle,"%s %s",runChar,Inside_TT_timing->GetTitle()); 
  Inside_TT_timing->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_Inside_TT_timing_%i.%s",dirName,mType,runNumber,fileType); c[4]->Print(name); }
 
  //Eta Profiles by TT
  c[5]->cd();
  gStyle->SetOptStat(1111);
  TProfile *timeTTAllFEDsEta = (TProfile*) f->Get("timeTTAllFEDsEta");
  timeTTAllFEDsEta->Draw();
  //timeTTAllFEDsEta->SetMinimum(4.95);
  //timeTTAllFEDsEta->SetMaximum(5.05);
   sprintf(mytitle,"%s %s",runChar,timeTTAllFEDsEta->GetTitle()); 
  timeTTAllFEDsEta->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEta_%i.%s",dirName,mType,runNumber,fileType); c[5]->Print(name); }
  
  c[6]->cd();
  gStyle->SetOptStat(1111);
  TProfile *timeTTAllFEDsEtaEEP = (TProfile*) f->Get("timeTTAllFEDsEtaEEP");
  timeTTAllFEDsEtaEEP->Draw();
  //timeTTAllFEDsEtaEEP->SetMinimum(4.5);
  //timeTTAllFEDsEtaEEP->SetMaximum(5.5);
   sprintf(mytitle,"%s %s",runChar,timeTTAllFEDsEtaEEP->GetTitle()); 
  timeTTAllFEDsEtaEEP->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEtaEEP_%i.%s",dirName,mType,runNumber,fileType); c[6]->Print(name); }
    
  c[7]->cd();
  gStyle->SetOptStat(1111);
  TProfile *timeTTAllFEDsEtaEEM = (TProfile*) f->Get("timeTTAllFEDsEtaEEM");
  timeTTAllFEDsEtaEEM->Draw();
  //timeTTAllFEDsEtaEEM->SetMinimum(4.5);
  //timeTTAllFEDsEtaEEM->SetMaximum(5.5);
   sprintf(mytitle,"%s %s",runChar,timeTTAllFEDsEtaEEM->GetTitle()); 
  timeTTAllFEDsEtaEEM->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEtaEEM_%i.%s",dirName,mType,runNumber,fileType); c[7]->Print(name); }
  
  //Eta profile by Ch
  c[8]->cd();
  gStyle->SetOptStat(1111);
  TProfile *timeCHAllFEDsEta = (TProfile*) f->Get("timeCHAllFEDsEta");
  timeCHAllFEDsEta->Draw();
  //timeCHAllFEDsEta->SetMinimum(4.8);
  //timeCHAllFEDsEta->SetMaximum(5.2);
   sprintf(mytitle,"%s %s",runChar,timeCHAllFEDsEta->GetTitle()); 
  timeCHAllFEDsEta->SetTitle(mytitle);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeCHAllFEDsEta_%i.%s",dirName,mType,runNumber,fileType); c[8]->Print(name); }
  
//1-D Histograms
  c[9]->cd();
  gStyle->SetOptStat(111110);
  TH1F *Rel_TimingSigma = (TH1F*) f->Get("Rel_TimingSigma");
  Rel_TimingSigma->Draw();
   sprintf(mytitle,"%s %s",runChar,Rel_TimingSigma->GetTitle()); 
  Rel_TimingSigma->SetTitle(mytitle);
  c[9]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_Rel_TimingSigma_%i.%s",dirName,mType,runNumber,fileType); c[9]->Print(name); }
  
  c[10]->cd();
  gStyle->SetOptStat(111110);
  TH1F *XtalsPerEvt = (TH1F*) f->Get("XtalsPerEvt");
  XtalsPerEvt->Draw();
  //XtalsPerEvt->GetXaxis()->SetRangeUser(0,100);
   sprintf(mytitle,"%s %s",runChar,XtalsPerEvt->GetTitle()); 
  XtalsPerEvt->SetTitle(mytitle);
  //c[10]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_XtalsPerEvt_%i.%s",dirName,mType,runNumber,fileType); c[10]->Print(name); }
  
  c[11]->cd();
  gStyle->SetOptStat(111110);
  TH1F *laserShift = (TH1F*) f->Get("laserShift");
  laserShift->Draw();
   sprintf(mytitle,"%s %s",runChar,laserShift->GetTitle()); 
  laserShift->SetTitle(mytitle);
  c[11]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_laserShift_%i.%s",dirName,mType,runNumber,fileType); c[11]->Print(name); }
  
//2-D Histogram
  c[12]->cd();
  gStyle->SetOptStat(111110);
  TH2F *RelRMS_vs_AbsTime = (TH2F*) f->Get("RelRMS_vs_AbsTime");
  RelRMS_vs_AbsTime->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,RelRMS_vs_AbsTime->GetTitle()); 
  RelRMS_vs_AbsTime->SetTitle(mytitle);
  c[12]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_RelRMS_vs_AbsTime_%i.%s",dirName,mType,runNumber,fileType); c[12]->Print(name); }
  
//1-D TGraphs  
  c[13]->cd();
  gStyle->SetOptStat(111110);
  TGraph *TTMeanWithRMS_All_FEDS = (TGraph*) f->Get("TTMeanWithRMS_All_FEDS");
   sprintf(mytitle,"%s %s",runChar,TTMeanWithRMS_All_FEDS->GetTitle()); 
  TTMeanWithRMS_All_FEDS->SetTitle(mytitle);
  TTMeanWithRMS_All_FEDS->GetYaxis()->SetLimits(5.,6.);
  //TTMeanWithRMS_All_FEDS->GetYaxis()->SetLimits(4.,6.);
  TTMeanWithRMS_All_FEDS->GetYaxis()->UnZoom();
  TTMeanWithRMS_All_FEDS->Draw("AP*");
  //c[13]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_TTMeanWithRMS_All_FEDS_%i.%s",dirName,mType,runNumber,fileType); c[13]->Print(name); }
  
  c[14]->cd();
  gStyle->SetOptStat(111110);
  TGraph *TTMeanWithRMS_All_FEDS_CHANGED = (TGraph*) f->Get("TTMeanWithRMS_All_FEDS_CHANGED");
   sprintf(mytitle,"%s %s",runChar,TTMeanWithRMS_All_FEDS_CHANGED->GetTitle()); 
  TTMeanWithRMS_All_FEDS_CHANGED->SetTitle(mytitle);
  TTMeanWithRMS_All_FEDS_CHANGED->GetYaxis()->SetLimits(-5.,5.);
  //TTMeanWithRMS_All_FEDS_CHANGED->GetYaxis()->SetLimits(4.,6.);
  TTMeanWithRMS_All_FEDS_CHANGED->GetYaxis()->UnZoom();
  TTMeanWithRMS_All_FEDS_CHANGED->Draw("AP*");
  //c[13]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_TTMeanWithRMS_All_FEDS_Corrected_%i.%s",dirName,mType,runNumber,fileType); c[14]->Print(name); }
  
//2-D TGraphs/Profiles 
  //Ch by Ch timing profile
  c[15]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeCHProfile = (TProfile2D*) f->Get("timeCHProfile");
  timeCHProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,timeCHProfile->GetTitle()); 
  timeCHProfile->SetTitle(mytitle);
  timeCHProfile->SetMinimum(4.0);
  //timeCHProfile->SetMinimum(4.5);
  //timeCHProfile->SetMaximum(5.5);
  timeCHProfile->GetXaxis()->SetNdivisions(-18);
  timeCHProfile->GetYaxis()->SetNdivisions(2);
  c[15]->SetLogy(0);
  c[15]->SetLogz(0);
  c[15]->SetGridx(1);
  c[15]->SetGridy(1);
  //c[15]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[15]->Print(name); }
  
   c[15]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeCHProfilep = TProfToRelProf2D(timeCHProfile,"timeCHProfilep", -5., 25.);
  timeCHProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",timeCHProfile->GetTitle()); 
  timeCHProfilep->SetTitle(mytitle);
  timeCHProfilep->SetMinimum(-10.);
  timeCHProfilep->SetMaximum(30.);
  //timeTTProfile->SetMinimum(5.85);
  //timeTTProfile->SetMinimum(4.8);
  //timeTTProfile->SetMaximum(6.45);
  timeCHProfilep->GetXaxis()->SetNdivisions(-18);
  timeCHProfilep->GetYaxis()->SetNdivisions(2);
  c[15]->SetLogy(0);
  c[15]->SetLogz(0);
  c[15]->SetGridx(1);
  c[15]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeCHProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[15]->Print(name); }
  //TT by TT timing profile
  
  c[39]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeCHProfileO = TProfile2DOccupancyFromProf2D(timeCHProfile,"timeCHProfileO");
  timeCHProfileO->Draw("colz");
   sprintf(mytitle,"CH occupancy"); 
  timeCHProfileO->SetTitle(mytitle);
  timeCHProfileO->SetMinimum(1.);
  timeCHProfileO->GetXaxis()->SetNdivisions(-18);
  timeCHProfileO->GetYaxis()->SetNdivisions(2);
  c[39]->SetLogy(0);
  c[39]->SetLogz(1);
  c[39]->SetGridx(1);
  c[39]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_OccuCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[39]->Print(name); }
  
  c[16]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeTTProfile = (TProfile2D*) f->Get("timeTTProfile");
  timeTTProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,timeTTProfile->GetTitle()); 
  timeTTProfile->SetTitle(mytitle);
  timeTTProfile->SetMinimum(4.0);
  //timeTTProfile->SetMinimum(5.85);
  //timeTTProfile->SetMinimum(4.8);
  //timeTTProfile->SetMaximum(6.45);
  timeTTProfile->GetXaxis()->SetNdivisions(-18);
  timeTTProfile->GetYaxis()->SetNdivisions(2);
  c[16]->SetLogy(0);
  c[16]->SetLogz(1);
  c[16]->SetGridx(1);
  c[16]->SetGridy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[16]->Print(name); }
 
  c[16]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeTTProfilep = TProfToRelProf2D(timeTTProfile,"timeTTProfilep", -5., 25.);
  timeTTProfilep->SetContour(10);
  timeTTProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",timeTTProfile->GetTitle()); 
  timeTTProfilep->SetTitle(mytitle);
  timeTTProfilep->SetMinimum(6.);
  timeTTProfilep->SetMaximum(16.);


  //timeTTProfile->SetMinimum(5.85);
  //timeTTProfile->SetMinimum(4.8);
  //timeTTProfile->SetMaximum(6.45);
  timeTTProfilep->GetXaxis()->SetNdivisions(-18);
  timeTTProfilep->GetYaxis()->SetNdivisions(2);
  c[16]->SetLogy(0);
  c[16]->SetLogz(0);
  c[16]->SetGridx(1);
  c[16]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[16]->Print(name); }

  c[40]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *timeTTProfileO = TProfile2DOccupancyFromProf2D(timeTTProfile,"timeTTProfileO");
  timeTTProfileO->Draw("colz");
   sprintf(mytitle,"TT occupancy"); 
  timeTTProfileO->SetTitle(mytitle);
  timeTTProfileO->SetMinimum(1.);
  timeTTProfileO->GetXaxis()->SetNdivisions(-18);
  timeTTProfileO->GetYaxis()->SetNdivisions(2);
  c[40]->SetLogy(0);
  c[40]->SetLogz(1);
  c[40]->SetGridx(1);
  c[40]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_OccuTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[40]->Print(name); }
  //Ch by Ch timing profile EE+
  c[17]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeCHProfile = (TProfile2D*) f->Get("EEPtimeCHProfile");
  EEPtimeCHProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,EEPtimeCHProfile->GetTitle()); 
  EEPtimeCHProfile->SetTitle(mytitle);
  EEPtimeCHProfile->SetMinimum(4.0);
  //EEPtimeCHProfile->SetMinimum(4.5);
  //EEPtimeCHProfile->SetMaximum(5.5);
  EEPtimeCHProfile->GetXaxis()->SetNdivisions(18);
  c[17]->SetLogy(0);
  c[17]->SetLogz(0);
  drawEELines();
  //c[15]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPtimeCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[17]->Print(name); }
  
  c[17]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeCHProfilep = TProfToRelProf2D(EEPtimeCHProfile,"EEPtimeCHProfilep", -5., 25.);
  EEPtimeCHProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",EEPtimeCHProfilep->GetTitle()); 
  EEPtimeCHProfilep->SetTitle(mytitle);
  EEPtimeCHProfilep->SetMinimum(-30.);
  EEPtimeCHProfilep->SetMaximum(50.);
  //EEPtimeCHProfile->SetMinimum(4.5);
  //EEPtimeCHProfile->SetMaximum(5.5);
  EEPtimeCHProfilep->GetXaxis()->SetNdivisions(18);
  c[17]->SetLogy(0);
  c[17]->SetLogz(0);
  drawEELines();
  //c[15]->SetLogz(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPtimeCHProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[17]->Print(name); }
 
  c[41]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeCHProfileO = TProfile2DOccupancyFromProf2D(EEPtimeCHProfile,"EEPtimeCHProfileO");
  EEPtimeCHProfileO->Draw("colz");
   sprintf(mytitle,"CH occupancy"); 
  EEPtimeCHProfileO->SetTitle(mytitle);
  EEPtimeCHProfileO->SetMinimum(1.);
  c[41]->SetLogy(0);
  c[41]->SetLogz(1);
  drawEELines();
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPOccuCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[41]->Print(name); }
  //Ch by Ch timing profile EE+
  c[18]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeCHProfile = (TProfile2D*) f->Get("EEMtimeCHProfile");
  EEMtimeCHProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,EEMtimeCHProfile->GetTitle()); 
  EEMtimeCHProfile->SetTitle(mytitle);
  EEMtimeCHProfile->SetMinimum(4.);
  //EEMtimeCHProfile->SetMinimum(4.5);
  //EEMtimeCHProfile->SetMaximum(5.5);
  EEMtimeCHProfile->GetXaxis()->SetNdivisions(18);
  c[18]->SetLogy(0);
  c[18]->SetLogz(1);
  drawEELines();
  //c[15]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMtimeCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[18]->Print(name); }
 
 c[18]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeCHProfilep = TProfToRelProf2D(EEMtimeCHProfile,"EEMtimeCHProfilep", -5., 25.);
  EEMtimeCHProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",EEMtimeCHProfilep->GetTitle()); 
  EEMtimeCHProfilep->SetTitle(mytitle);
  EEMtimeCHProfilep->SetMinimum(-30.);
  EEMtimeCHProfilep->SetMaximum(50.);
  //EEPtimeCHProfile->SetMinimum(4.5);
  //EEPtimeCHProfile->SetMaximum(5.5);
  EEMtimeCHProfilep->GetXaxis()->SetNdivisions(18);
  c[18]->SetLogy(0);
  c[18]->SetLogz(0);
  drawEELines();
  //c[15]->SetLogz(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMtimeCHProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[18]->Print(name); } 
  
  c[42]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeCHProfileO = TProfile2DOccupancyFromProf2D(EEMtimeCHProfile,"EEMtimeCHProfileO");
  EEMtimeCHProfileO->Draw("colz");
   sprintf(mytitle,"CH occupancy"); 
  EEMtimeCHProfileO->SetTitle(mytitle);
  EEMtimeCHProfileO->SetMinimum(1.);
  c[42]->SetLogy(0);
  c[42]->SetLogz(1);
  drawEELines();
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMOccuCHProfile_%i.%s",dirName,mType,runNumber,fileType); c[42]->Print(name); }
  //TT by TT timing profile EE+
  c[19]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeTTProfile = (TProfile2D*) f->Get("EEPtimeTTProfile");
  EEPtimeTTProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,EEPtimeTTProfile->GetTitle()); 
  EEPtimeTTProfile->SetTitle(mytitle);
  EEPtimeTTProfile->SetMinimum(4.);
  //EEPtimeTTProfile->SetMinimum(4.5);
  //EEPtimeTTProfile->SetMaximum(5.5);
  EEPtimeTTProfile->GetXaxis()->SetNdivisions(18);
  c[19]->SetLogy(0);
  c[19]->SetLogz(1);
  drawEELines();
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPtimeTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[19]->Print(name); }
 

  c[19]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeTTProfilep = TProfToRelProf2D(EEPtimeTTProfile,"EEPtimeTTProfilep", -5., 25.);
  EEPtimeTTProfilep->SetContour(10);
  EEPtimeTTProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",EEPtimeTTProfile->GetTitle()); 
  EEPtimeTTProfilep->SetTitle(mytitle);
  EEPtimeTTProfilep->SetMinimum(6.);
  EEPtimeTTProfilep->SetMaximum(16.);

  //timeTTProfile->SetMinimum(5.85);
  //timeTTProfile->SetMinimum(4.8);
  //timeTTProfile->SetMaximum(6.45);
  EEPtimeTTProfilep->GetXaxis()->SetNdivisions(18);
  //EEPtimeTTProfilep->GetYaxis()->SetNdivisions(2);
  drawEELines();
  c[19]->SetLogy(0);
  c[19]->SetLogz(0);
  // c[19]->SetGridx(1);
  //c[19]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPtimeTTProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[19]->Print(name); }
  c[44]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEPtimeTTProfileO = TProfile2DOccupancyFromProf2D(EEPtimeTTProfile,"EEPtimeTTProfileO");
  EEPtimeTTProfileO->Draw("colz");
   sprintf(mytitle,"TT occupancy"); 
  EEPtimeTTProfileO->SetTitle(mytitle);
  EEPtimeTTProfileO->SetMinimum(1.);
  c[44]->SetLogy(0);
  c[44]->SetLogz(1);
  drawEELines();
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPOccuTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[44]->Print(name); }
 
  //TT by TT timing profile EE-
  c[20]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeTTProfile = (TProfile2D*) f->Get("EEMtimeTTProfile");
  EEMtimeTTProfile->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,EEMtimeTTProfile->GetTitle()); 
  EEMtimeTTProfile->SetTitle(mytitle);
  EEMtimeTTProfile->SetMinimum(4.);
  //EEMtimeTTProfile->SetMinimum(4.5);
  //EEMtimeTTProfile->SetMaximum(5.5);
  EEMtimeTTProfile->GetXaxis()->SetNdivisions(18);
  c[20]->SetLogy(0);
  c[20]->SetLogz(1);
  drawEELines();
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMtimeTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[20]->Print(name); }
  
  c[20]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeTTProfilep = TProfToRelProf2D(EEMtimeTTProfile,"EEMtimeTTProfilep", -5., 25.);
  
  EEMtimeTTProfilep->Draw("colz");
   sprintf(mytitle,"%s in ns",EEMtimeTTProfile->GetTitle()); 
  EEMtimeTTProfilep->SetTitle(mytitle);
  EEMtimeTTProfilep->SetContour(10);
  EEMtimeTTProfilep->SetMinimum(6.);
  EEMtimeTTProfilep->SetMaximum(16.);

  //timeTTProfile->SetMinimum(5.85);
  //timeTTProfile->SetMinimum(4.8);
  //timeTTProfile->SetMaximum(6.45);
  EEMtimeTTProfilep->GetXaxis()->SetNdivisions(18);
  //EEMtimeTTProfilep->GetYaxis()->SetNdivisions(2);
  c[20]->SetLogy(0);
  c[20]->SetLogz(0);
  drawEELines();
  //c[20]->SetGridx(1);
  //c[20]->SetGridy(1);
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMtimeTTProfileRel_%i.%s",dirName,mType,runNumber,fileType); c[20]->Print(name); }
  c[43]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *EEMtimeTTProfileO = TProfile2DOccupancyFromProf2D(EEMtimeTTProfile,"EEMtimeTTProfileO");
  EEMtimeTTProfileO->Draw("colz");
   sprintf(mytitle,"TT occupancy"); 
  EEMtimeTTProfileO->SetTitle(mytitle);
  EEMtimeTTProfileO->SetMinimum(1.);
  c[43]->SetLogy(0);
  c[43]->SetLogz(1);
  drawEELines();
  gStyle->SetOptStat(0);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMOccuTTProfile_%i.%s",dirName,mType,runNumber,fileType); c[43]->Print(name); }

  //Amplitude Profiles
  c[21]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *fullAmpProfileEB = (TProfile2D*) f->Get("fullAmpProfileEB");
  fullAmpProfileEB->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,fullAmpProfileEB->GetTitle()); 
  fullAmpProfileEB->SetTitle(mytitle);
  if (fullAmpProfileEB->GetMaximum() > 0 ) {
     fullAmpProfileEB->SetMinimum(0.1);
     c[21]->SetLogy(0);
     c[21]->SetLogz(1);
  }
  if (printPics) { sprintf(name,"%s/%sAnalysis_fullAmpProfileEB_%i.%s",dirName,mType,runNumber,fileType); c[21]->Print(name); }
  
  c[22]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *fullAmpProfileEEP = (TProfile2D*) f->Get("fullAmpProfileEEP");
  fullAmpProfileEEP->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,fullAmpProfileEEP->GetTitle()); 
  fullAmpProfileEEP->SetTitle(mytitle);
  if (fullAmpProfileEEP->GetMaximum() > 0 ) {
     fullAmpProfileEEP->SetMinimum(0.1);
     c[22]->SetLogy(0);
     c[22]->SetLogz(1);
  }
  drawEELines();
  if (printPics) { sprintf(name,"%s/%sAnalysis_fullAmpProfileEEP_%i.%s",dirName,mType,runNumber,fileType); c[22]->Print(name); }
  
  c[23]->cd();
  gStyle->SetOptStat(10);
  TProfile2D *fullAmpProfileEEM = (TProfile2D*) f->Get("fullAmpProfileEEM");
  fullAmpProfileEEM->Draw("colz");
   sprintf(mytitle,"%s %s",runChar,fullAmpProfileEEM->GetTitle()); 
  fullAmpProfileEEM->SetTitle(mytitle);
  if (fullAmpProfileEEM->GetMaximum() > 0 ) {
     fullAmpProfileEEM->SetMinimum(0.1);
     c[23]->SetLogy(0);
     c[23]->SetLogz(1);
  }
  drawEELines();
  if (printPics) { sprintf(name,"%s/%sAnalysis_fullAmpProfileEEM_%i.%s",dirName,mType,runNumber,fileType); c[23]->Print(name); }
   
   
  //Eta Profiles by TT Normalized
  c[24]->cd();
  TProfile *timeTTAllFEDsEtap =  TProfToRelProf(timeTTAllFEDsEta,"timeTTAllFEDsEtap",-5,25.);
  customizeTProfile(timeTTAllFEDsEtap);
  timeTTAllFEDsEtap->Draw("p");
  timeTTAllFEDsEtap->SetMinimum(-30.);
  timeTTAllFEDsEtap->SetMaximum(50.);
   sprintf(mytitle,"%s to optimal;i#eta;Time (ns)",timeTTAllFEDsEta->GetTitle()); 
  timeTTAllFEDsEtap->SetTitle(mytitle);
  gStyle->SetOptStat(100);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEtaRel_%i.%s",dirName,mType,runNumber,fileType); c[24]->Print(name); }
  
  c[25]->cd();
  TProfile *timeTTAllFEDsEtaEEPp =  TProfToRelProf(timeTTAllFEDsEtaEEP,"timeTTAllFEDsEtaEEPp",-5,25.);
  customizeTProfile(timeTTAllFEDsEtaEEPp);
  timeTTAllFEDsEtaEEPp->Draw("p");
  timeTTAllFEDsEtaEEPp->SetMinimum(-30.);
  timeTTAllFEDsEtaEEPp->SetMaximum(50.);
   sprintf(mytitle,"%s to optimal;i#eta;Time (ns)",timeTTAllFEDsEtaEEP->GetTitle()); 
  timeTTAllFEDsEtaEEPp->SetTitle(mytitle);
  gStyle->SetOptStat(100);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEtaEEPRel_%i.%s",dirName,mType,runNumber,fileType); c[25]->Print(name); }
  
  c[26]->cd();
  TProfile *timeTTAllFEDsEtaEEMp =  TProfToRelProf(timeTTAllFEDsEtaEEM,"timeTTAllFEDsEtaEEMp",-5,25.);
  customizeTProfile(timeTTAllFEDsEtaEEMp);
  timeTTAllFEDsEtaEEMp->Draw("p");
  timeTTAllFEDsEtaEEMp->SetMinimum(-30.);
  timeTTAllFEDsEtaEEMp->SetMaximum(50.);
   sprintf(mytitle,"%s to optimal;i#eta;Time (ns)",timeTTAllFEDsEtaEEM->GetTitle()); 
  timeTTAllFEDsEtaEEMp->SetTitle(mytitle);
  gStyle->SetOptStat(100);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeTTAllFEDsEtaEEMRel_%i.%s",dirName,mType,runNumber,fileType); c[26]->Print(name); }
  
  //Eta profile by Ch
  c[27]->cd();
  gStyle->SetOptStat(1111);
  TProfile *timeCHAllFEDsEta = (TProfile*) f->Get("timeCHAllFEDsEta");
  timeCHAllFEDsEta->Draw();
  TProfile *timeCHAllFEDsEtap =  TProfToRelProf(timeCHAllFEDsEta,"timeCHAllFEDsEtap",-5,25.);
  customizeTProfile(timeCHAllFEDsEtap);
  timeCHAllFEDsEtap->Draw("p");
  timeCHAllFEDsEtap->SetMinimum(-30.);
  timeCHAllFEDsEtap->SetMaximum(50.);
   sprintf(mytitle,"%s to optimal;i#eta;Time (ns)",timeCHAllFEDsEta->GetTitle()); 
  timeCHAllFEDsEtap->SetTitle(mytitle);
  gStyle->SetOptStat(100);
  if (printPics) { sprintf(name,"%s/%sAnalysis_timeCHAllFEDsEtaRel_%i.%s",dirName,mType,runNumber,fileType); c[27]->Print(name); }
 
  //+_+_+_+_+_+_+_+_+__-----------------------------+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_
  //Now it is time to see if the timing tree is there and use the individual ntuple information

  char ebhashfilter[500];
  char eehashfilter[3000];
  sprintf(ebhashfilter,"crystalHashedIndicesEB != 25822 && crystalHashedIndicesEB != 32705 && crystalHashedIndicesEB != 56473");
  sprintf(eehashfilter,"crystalHashedIndicesEE != 11658 && crystalHashedIndicesEE != 11742 && crystalHashedIndicesEE != 10224 && crystalHashedIndicesEE != 10225 && crystalHashedIndicesEE != 10226 && crystalHashedIndicesEE != 10310 && crystalHashedIndicesEE != 10311 && crystalHashedIndicesEE != 10394 && crystalHashedIndicesEE != 10395 && crystalHashedIndicesEE != 10875 && crystalHashedIndicesEE != 11316 && crystalHashedIndicesEE != 11659 && crystalHashedIndicesEE != 11660 && crystalHashedIndicesEE != 11661 && crystalHashedIndicesEE != 11743  && crystalHashedIndicesEE != 11744 && crystalHashedIndicesEE != 11744 && crystalHashedIndicesEE != 11745 && crystalHashedIndicesEE != 11932 && crystalHashedIndicesEE != 11746 && crystalHashedIndicesEE != 12702 && crystalHashedIndicesEE != 4252 && crystalHashedIndicesEE != 4335 && crystalHashedIndicesEE != 4337 && crystalHashedIndicesEE != 4419 && crystalHashedIndicesEE != 4423 && crystalHashedIndicesEE != 4785 && crystalHashedIndicesEE != 6181 && crystalHashedIndicesEE != 14613 && crystalHashedIndicesEE != 13726 && crystalHashedIndicesEE != 13727 && crystalHashedIndicesEE != 7717 && crystalHashedIndicesEE != 7778 && crystalHashedIndicesEE != 4420 && crystalHashedIndicesEE != 4421 && crystalHashedIndicesEE != 4423 && crystalHashedIndicesEE != 2946 && crystalHashedIndicesEE != 2900 && crystalHashedIndicesEE != 2902 && crystalHashedIndicesEE != 2901 && crystalHashedIndicesEE != 2903 && crystalHashedIndicesEE != 2904 && crystalHashedIndicesEE != 2905 && crystalHashedIndicesEE != 2986 && crystalHashedIndicesEE != 2987 && crystalHashedIndicesEE != 2988 && crystalHashedIndicesEE != 2989 && crystalHashedIndicesEE != 3070 && crystalHashedIndicesEE != 3071 && crystalHashedIndicesEE != 4252 && crystalHashedIndicesEE != 4253 && crystalHashedIndicesEE != 4254 && crystalHashedIndicesEE != 4255 && crystalHashedIndicesEE != 4256");

  char ebtimefilter[100];
  char eetimefilter[100];
  sprintf(ebtimefilter,"(crystalTimeErrorsEB)*25. < 5.0");
  sprintf(eetimefilter,"(crystalTimeErrorsEE)*25. < 5.0");

  char ebfilter[500];
  char eefilter[3100];
  sprintf(ebfilter,"(%s) && (%s)",ebtimefilter,ebhashfilter);
  sprintf(eefilter,"(%s) && (%s)",eetimefilter,eehashfilter);

  char eepfilter[3200];
  char eemfilter[3200];
  sprintf(eepfilter,"(%s) && (crystalHashedIndicesEE > 7341)",eefilter);
  sprintf(eemfilter,"(%s) && (crystalHashedIndicesEE < 7342)",eefilter);

  TTree* eventTimingInfoTree = ((TTree*) f->Get("eventTimingInfoTree"));
  if (!eventTimingInfoTree) { std::cout << " No TTree in the event, probalby expected" << std::endl; cout << name << endl; return;} 
  //Now, we will only do the below if there is a TTree in the event.
  c[28]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEB-5.)*25. >> hctEB(tbins,tbinsL,tbinsH)",ebhashfilter);
   sprintf(mytitle,"%s EB Crystal Times;Time (ns);Number of Crystals",runChar); 
  hctEB->SetTitle(mytitle);
  hctEB->GetXaxis()->SetNdivisions(512);
  hctEB->Fit("gaus");
  gStyle->SetOptFit(111);
  c[28]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMES_%i.%s",dirName,mType,runNumber,fileType); c[28]->Print(name); }

  eventTimingInfoTree->Draw("(crystalTimesEB-5.)*25. >> hctEBf(tbins,tbinsL,tbinsH)",ebfilter);
   sprintf(mytitle,"%s EB Crystal Times (Error Filtered);Time (ns);Number of Crystals",runChar); 
  hctEBf->SetTitle(mytitle);
  hctEBf->GetXaxis()->SetNdivisions(512);
  hctEBf->Fit("gaus");
  gStyle->SetOptFit(111);
  c[28]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMESFILT_%i.%s",dirName,mType,runNumber,fileType); c[28]->Print(name); }
  
  c[29]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEE-5.)*25. >> hctEE(tbins,tbinsL,tbinsH)",eehashfilter);
   sprintf(mytitle,"%s EE Crystal Times;Time (ns);Number of Crystals",runChar); 
  hctEE->SetTitle(mytitle);
  c[29]->SetLogy(1);
  hctEE->GetXaxis()->SetNdivisions(512);
  hctEE->Fit("gaus");
  gStyle->SetOptFit(111);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMES_%i.%s",dirName,mType,runNumber,fileType); c[29]->Print(name); }
  eventTimingInfoTree->Draw("(crystalTimesEE-5.)*25. >> hctEEf(tbins,tbinsL,tbinsH)",eefilter);
   sprintf(mytitle,"%s EE Crystal Times (Error Filtered);Time (ns);Number of Crystals",runChar); 
  hctEEf->SetTitle(mytitle);
  c[29]->SetLogy(1);
  hctEEf->GetXaxis()->SetNdivisions(512);
  hctEEf->Fit("gaus");
  gStyle->SetOptFit(111);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMESFILT_%i.%s",dirName,mType,runNumber,fileType); c[29]->Print(name); }

  c[54]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEE-5.)*25. >> hctEEp(tbins,tbinsL,tbinsH)",eepfilter);
   sprintf(mytitle,"%s EE+ Crystal Times;Time (ns);Number of Crystals",runChar); 
  hctEEp->SetTitle(mytitle);
  c[54]->SetLogy(1);
  hctEEp->GetXaxis()->SetNdivisions(512);
  hctEEp->Fit("gaus");
  gStyle->SetOptFit(111);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPTIMES_%i.%s",dirName,mType,runNumber,fileType); c[54]->Print(name); }

  c[55]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEE-5.)*25. >> hctEEm(tbins,tbinsL,tbinsH)",eemfilter);
   sprintf(mytitle,"%s EE- Crystal Times;Time (ns);Number of Crystals",runChar); 
  hctEEm->SetTitle(mytitle);
  c[55]->SetLogy(1);
  hctEEm->GetXaxis()->SetNdivisions(512);
  hctEEm->Fit("gaus");
  gStyle->SetOptFit(111);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMTIMES_%i.%s",dirName,mType,runNumber,fileType); c[55]->Print(name); }

  
  //Time to average event time
  c[30]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(correctionToSampleEB-5.)*25.:(crystalTimesEE-5.)*25. >> hctEEtoAve(tbins,tbinsL,tbinsH,tbins,tbinsL,tbinsH)", eehashfilter, "COLZ");
   sprintf(mytitle,"%s EE Crystal Times to Average Time;Crystal Time (ns);Average EB Event Time (ns)",runChar); 
  //hctEEtoAve->Draw;
  hctEEtoAve->SetTitle(mytitle);
  hctEEtoAve->GetXaxis()->SetNdivisions(512);
  hctEEtoAve->GetYaxis()->SetNdivisions(512);
  c[30]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMEStoAverage_%i.%s",dirName,mType,runNumber,fileType); c[30]->Print(name); }
  c[31]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(correctionToSampleEB-5.)*25.:(crystalTimesEB-5.)*25. >> hctEBtoAve(tbins,tbinsL,tbinsH,tbins,tbinsL,tbinsH)", ebhashfilter, "COLZ");
   sprintf(mytitle,"%s EB Crystal Times to Average Time;Crystal Time (ns);Average EB Event Time (ns)",runChar); 
  //hctEEtoAve->Draw;
  hctEBtoAve->SetTitle(mytitle);
  hctEBtoAve->GetXaxis()->SetNdivisions(512);
  hctEBtoAve->GetYaxis()->SetNdivisions(512);
  c[31]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMEStoAverage_%i.%s",dirName,mType,runNumber,fileType); c[31]->Print(name); }
  
  //Time to Time error
  c[32]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimeErrorsEB)*25.:(crystalTimesEB-5.)*25. >> hctEBtoTerr(tbins,tbinsL,tbinsH,26,0.,tbinsH)",ebhashfilter,"COLZ");
   sprintf(mytitle,"%s EB Crystal Times to Time Error;Crystal Time (ns);Crystal Time Error (ns)",runChar); 
  hctEBtoTerr->SetTitle(mytitle);
  hctEBtoTerr->GetXaxis()->SetNdivisions(512);
  hctEBtoTerr->GetYaxis()->SetNdivisions(507);
  c[32]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMEStoTERR_%i.%s",dirName,mType,runNumber,fileType); c[32]->Print(name); }
  c[33]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimeErrorsEE)*25.:(crystalTimesEE-5.)*25. >> hctEEtoTerr(tbins,tbinsL,tbinsH,26,0.,tbinsH)",eehashfilter,"COLZ");
   sprintf(mytitle,"%s EE Crystal Times to Time Error;Crystal Time (ns);Crystal Time Error (ns)",runChar); 
  hctEEtoTerr->SetTitle(mytitle);
  hctEEtoTerr->GetXaxis()->SetNdivisions(512);
  hctEEtoTerr->GetYaxis()->SetNdivisions(507);
  c[33]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMEStoTERR_%i.%s",dirName,mType,runNumber,fileType); c[33]->Print(name); }
  
  //Amplitude to time
  c[34]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalAmplitudesEE:(crystalTimesEE-5.)*25. >> hctEEtoAmp(tbins,tbinsL,tbinsH,30,0.,30.)",eehashfilter,"COLZ");
   sprintf(mytitle,"%s EE Crystal Times to Amplitdue;Crystal Time (ns);Crystal Amplitude (GeV)",runChar); 
  hctEEtoAmp->SetTitle(mytitle);
  hctEEtoAmp->GetXaxis()->SetNdivisions(512);
  hctEEtoAmp->GetYaxis()->SetNdivisions(507);
  c[34]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMEStoAMP_%i.%s",dirName,mType,runNumber,fileType); c[34]->Print(name); } 
  c[35]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalAmplitudesEB:(crystalTimesEB-5.)*25. >> hctEBtoAmp(tbins,tbinsL,tbinsH,30,0.,30.)",ebhashfilter,"COLZ");
   sprintf(mytitle,"%s EB Crystal Times to Amplitdue;Crystal Time (ns);Crystal Amplitude (GeV)",runChar); 
  hctEBtoAmp->SetTitle(mytitle);
  hctEBtoAmp->GetXaxis()->SetNdivisions(512);
  hctEBtoAmp->GetYaxis()->SetNdivisions(507);
  c[35]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMEStoAMP_%i.%s",dirName,mType,runNumber,fileType); c[35]->Print(name); } 
  
  //Amplitdue to ave event time
  c[36]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalAmplitudesEB:(correctionToSampleEB-5.0)*25. >> hctEBtoAmpEvt(tbins,tbinsL,tbinsH,30,0.,30.)",ebhashfilter,"COLZ");
   sprintf(mytitle,"%s EB Event Time to Crystal Amplitudes;Average EB Event Time (ns);Crystal Amplitude (GeV)",runChar); 
  hctEBtoAmpEvt->SetTitle(mytitle);
  hctEBtoAmpEvt->GetXaxis()->SetNdivisions(512);
  hctEBtoAmpEvt->GetYaxis()->SetNdivisions(507);
  c[36]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBEvtTIMEStoAMP_%i.%s",dirName,mType,runNumber,fileType); c[36]->Print(name); } 
  c[37]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalAmplitudesEE:(correctionToSampleEB-5.0)*25. >> hctEEtoAmpEvt(tbins,tbinsL,tbinsH,30,0.,30.)",eehashfilter,"COLZ");
   sprintf(mytitle,"%s EE Event Time to Crystal Amplitudes;Average EB Event Time (ns);Crystal Amplitude (GeV)",runChar); 
  hctEEtoAmpEvt->SetTitle(mytitle);
  hctEEtoAmpEvt->GetXaxis()->SetNdivisions(512);
  hctEEtoAmpEvt->GetYaxis()->SetNdivisions(507);
  c[37]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEEvtTIMEStoAMP_%i.%s",dirName,mType,runNumber,fileType); c[37]->Print(name); } 
  
  //Amplitude to time error.
  c[38]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimeErrorsEE)*25.:crystalAmplitudesEE >> hctEEtoAmpErr(30,0.,30.,26,0.,26.)",eehashfilter,"COLZ");
   sprintf(mytitle,"%s EE Time Error to Crystal Amplitudes;Crystal Amplitude (GeV);Time Error (ns)",runChar); 
  hctEEtoAmpErr->SetTitle(mytitle);
  hctEEtoAmpErr->GetXaxis()->SetNdivisions(512);
  hctEEtoAmpErr->GetYaxis()->SetNdivisions(507);
  c[38]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EETIMESErrtoAMP_%i.%s",dirName,mType,runNumber,fileType); c[38]->Print(name); } 
  c[39]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimeErrorsEB)*25.:crystalAmplitudesEB >> hctEBtoAmpErr(30,0.,30.,26,0.,26.)",ebhashfilter,"COLZ");
   sprintf(mytitle,"%s EB Time Error to Crystal Amplitudes;Crystal Amplitude (GeV);Time Error (ns)",runChar); 
  hctEBtoAmpErr->SetTitle(mytitle);
  hctEBtoAmpErr->GetXaxis()->SetNdivisions(512);
  hctEBtoAmpErr->GetYaxis()->SetNdivisions(507);
  c[39]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTIMESErrtoAMP_%i.%s",dirName,mType,runNumber,fileType); c[39]->Print(name); } 

  //Hashed Index's
  c[50]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalHashedIndicesEB >> hctEBHashed(62000,0.,62000.)",ebhashfilter);
   sprintf(mytitle,"%s EB Hashed Index Occupancy;Hashed Index",runChar); 
  hctEBHashed->SetTitle(mytitle);
  hctEBHashed->GetXaxis()->SetNdivisions(512);
  c[50]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBHashed_%i.%s",dirName,mType,runNumber,fileType); c[50]->Print(name); } 
  
  c[51]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("crystalHashedIndicesEE >> hctEEHashed(15000,0.,15000.)",eehashfilter);
   sprintf(mytitle,"%s EE Hashed Index Occupancy;Hashed Index",runChar); 
  hctEEHashed->SetTitle(mytitle);
  c[51]->SetLogy(1);
  hctEEHashed->GetXaxis()->SetNdivisions(512);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEHashed_%i.%s",dirName,mType,runNumber,fileType); c[51]->Print(name); } 
  
  
  //Time to Hashed Index
  c[52]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEB-5.)*25.:crystalHashedIndicesEB >> hctEBtoHashed(2000,0.,62000.,52,-40.,100.)",ebhashfilter,"COLZ");
   sprintf(mytitle,"%s EB Hashed Index to Time;Hashed Index;Time(ns)",runChar); 
  hctEBtoHashed->SetTitle(mytitle);
  hctEBtoHashed->GetXaxis()->SetNdivisions(512);
  hctEBtoHashed->GetYaxis()->SetNdivisions(507);
  hctEBtoHashed->SetMinimum(1);
  c[52]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBHashedToTime_%i.%s",dirName,mType,runNumber,fileType); c[52]->Print(name); } 

  c[53]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("(crystalTimesEE-5.)*25.:crystalHashedIndicesEE >> hctEEtoHashed(500,0.,15000.,52,-40.,100.)",eehashfilter,"COLZ");
   sprintf(mytitle,"%s EE Hashed Index to Time;Hashed Index;Time(ns)",runChar); 
  hctEEtoHashed->SetTitle(mytitle);
  hctEEtoHashed->GetXaxis()->SetNdivisions(512);
  hctEEtoHashed->GetYaxis()->SetNdivisions(507);
  hctEEtoHashed->SetMinimum(1);
  c[53]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEHashedToTime_%i.%s",dirName,mType,runNumber,fileType); c[53]->Print(name); } 

  //1-D Number of crystal distributions
  c[56]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("numberOfEBcrys >> hctEBCry(25,0.,25.)","");
   sprintf(mytitle,"%s EB Number of Crystals;Number of EB crystals",runChar); 
  hctEBCry->SetTitle(mytitle);
  hctEBCry->GetXaxis()->SetNdivisions(512);
  c[56]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBCrys_%i.%s",dirName,mType,runNumber,fileType); c[56]->Print(name); } 
  
  c[57]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("numberOfEEcrys >> hctEECry(25,0.,25.)","");
   sprintf(mytitle,"%s EB Number of Crystals;Number of EE crystals",runChar); 
  hctEECry->SetTitle(mytitle);
  hctEECry->GetXaxis()->SetNdivisions(512);
  c[57]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EECrys_%i.%s",dirName,mType,runNumber,fileType); c[57]->Print(name); } 


  //2-D crystal plots 
  c[58]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("numberOfEBcrys:(correctionToSampleEB-5.0)*25. >> hctEBCryT(tbins,tbinsL,tbinsH,25,0.,25.)","numberOfEBcrys>0","colz");
   sprintf(mytitle,"%s EB Number of Crystals to EB average time;EB average time (ns);Number of EB crystals",runChar); 
  hctEBCryT->SetTitle(mytitle);
  hctEBCryT->GetXaxis()->SetNdivisions(512);
  hctEBCryT->GetYaxis()->SetNdivisions(507);
  hctEBCryT->SetMinimum(1);
  c[58]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBCrysToTime_%i.%s",dirName,mType,runNumber,fileType); c[58]->Print(name); } 
    
  c[59]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("numberOfEEcrys:(correctionToSampleEEP-5.0)*25. >> hctEEPCryT(tbins,tbinsL,tbinsH,25,0.,25.)","numberOfEEcrys>0 && correctionToSampleEEP>0","colz");
   sprintf(mytitle,"%s EE Number of Crystals to EE+ average time;EE+ average time (ns);Number of EE crystals",runChar); 
  hctEEPCryT->SetTitle(mytitle);
  hctEEPCryT->GetXaxis()->SetNdivisions(512);
  hctEEPCryT->GetYaxis()->SetNdivisions(507);
  hctEEPCryT->SetMinimum(1);
  c[59]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPCrysToTime_%i.%s",dirName,mType,runNumber,fileType); c[59]->Print(name); } 
    
  c[60]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("numberOfEEcrys:(correctionToSampleEEM-5.0)*25. >> hctEEMCryT(tbins,tbinsL,tbinsH,25,0.,25.)","numberOfEEcrys>0 && correctionToSampleEEM>0","colz");
   sprintf(mytitle,"%s EE Number of Crystals to EE- average time;EE- average time (ns);Number of EE crystals",runChar); 
  hctEEMCryT->SetTitle(mytitle);
  hctEEMCryT->GetXaxis()->SetNdivisions(512);
  hctEEMCryT->GetYaxis()->SetNdivisions(507);
  hctEEMCryT->SetMinimum(1);
  c[60]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMCrysToTime_%i.%s",dirName,mType,runNumber,fileType); c[60]->Print(name); } 
  
  c[61]->cd();
  gStyle->SetOptStat(111110);
  eventTimingInfoTree->Draw("(correctionToSampleEEM-5.0)*25.:(correctionToSampleEEP-5.0)*25. >> hctEEMEEP(tbins,tbinsL,tbinsH,tbins,tbinsL,tbinsH)","correctionToSampleEEP>0 && correctionToSampleEEM>0","colz");
   sprintf(mytitle,"%s EE+ average time to EE- average time;EE- average time (ns);EE+ average time (ns)",runChar); 
  hctEEMEEP->SetTitle(mytitle);
  hctEEMEEP->GetXaxis()->SetNdivisions(512);
  hctEEMEEP->GetYaxis()->SetNdivisions(507);
  hctEEMEEP->SetMinimum(1);
  c[61]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPTimeToEEMTime_%i.%s",dirName,mType,runNumber,fileType); c[61]->Print(name); } 

  c[63]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("numberOfEEcrys:(correctionToSampleEEM-5.0)*25.-(correctionToSampleEEP-5.0)*25. >> hctEEMDEEPcry(25,-40,100,25,0,25)","correctionToSampleEEP>0 && correctionToSampleEEM>0","colz");
  sprintf(mytitle,"%s EE- minus EE+ average time vs EE crystals;(EEM - EEP) average time (ns);Number EE crystals",runChar); 
  hctEEMDEEPcry->SetTitle(mytitle);
  hctEEMDEEPcry->GetXaxis()->SetNdivisions(512);
  hctEEMDEEPcry->SetMinimum(1);
  c[63]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPDiffEEMTimeCrys_%i.%s",dirName,mType,runNumber,fileType); c[63]->Print(name); }
  
  c[62]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  eventTimingInfoTree->Draw("(correctionToSampleEEM-5.0)*25.-(correctionToSampleEEP-5.0)*25. >> hctEEMDEEP(tbins,tbinsL,tbinsH)","correctionToSampleEEP>0 && correctionToSampleEEM>0");
   sprintf(mytitle,"%s EE- minus EE+ average time;(EEM - EEP) average time (ns)",runChar); 
  hctEEMDEEP->SetTitle(mytitle);
  hctEEMDEEP->GetXaxis()->SetNdivisions(512);
  cout << "mean is " << hctEEMDEEP->GetMean() << endl;
  if (hctEEMDEEP->GetMean() != 0) hctEEMDEEP->Fit("gaus");
  //hctEEMEEP->SetMinimum(.5);
  c[62]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPDiffEEMTime_%i.%s",dirName,mType,runNumber,fileType); c[62]->Print(name); } 
  
  //Number of crystals vs amplitude
  c[64]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("numberOfEEcrys:crystalAmplitudesEE >> hctEEcryamp(30,0,30,25,0,25)","numberOfEEcrys>0","colz");
  sprintf(mytitle,"%s EE amplitudes vs number of crystals;Crystal Amp (GeV);Number EE crystals",runChar); 
  hctEEcryamp->SetTitle(mytitle);
  hctEEcryamp->GetXaxis()->SetNdivisions(512);
  hctEEcryamp->SetMinimum(1);
  c[64]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EECrysAmp_%i.%s",dirName,mType,runNumber,fileType); c[64]->Print(name); }

  c[65]->cd();
  gStyle->SetOptStat(1110);
  eventTimingInfoTree->Draw("numberOfEBcrys:crystalAmplitudesEB >> hctEBcryamp(30,0,30,25,0,25)","numberOfEBcrys>0","colz");
  sprintf(mytitle,"%s EB amplitudes vs number of crystals;Crystal Amp (GeV);Number EB crystals",runChar); 
  hctEBcryamp->SetTitle(mytitle);
  hctEBcryamp->GetXaxis()->SetNdivisions(512);
  hctEBcryamp->SetMinimum(1);
  c[65]->SetLogz(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBCrysAmp_%i.%s",dirName,mType,runNumber,fileType); c[65]->Print(name); }

  //FINAL 1D timing number, by crystal by TT
  c[66]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *tthistEB = HistFromTProfile2D(timeTTProfile,"tthistEB",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EB TT Timing;TT time average (ns)",runChar); 
  tthistEB->SetTitle(mytitle);
  tthistEB->GetXaxis()->SetNdivisions(512);
  if (tthistEB->GetMean() != 0) tthistEB->Fit("gaus");
  c[66]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBTTTIME_%i.%s",dirName,mType,runNumber,fileType); c[66]->Print(name); }

  c[67]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *chhistEB = HistFromTProfile2D(timeCHProfile,"chhistEB",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EB CH Timing;CH time average (ns)",runChar); 
  chhistEB->SetTitle(mytitle);
  chhistEB->GetXaxis()->SetNdivisions(512);
  if (chhistEB->GetMean() != 0) chhistEB->Fit("gaus");
  c[67]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EBCHTIME_%i.%s",dirName,mType,runNumber,fileType); c[67]->Print(name); }

  c[68]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *tthistEEP = HistFromTProfile2D(EEPtimeTTProfile,"tthistEEP",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EE+ TT Timing;TT time average (ns)",runChar); 
  tthistEEP->SetTitle(mytitle);
  tthistEEP->GetXaxis()->SetNdivisions(512);
  if (tthistEEP->GetMean() != 0) tthistEEP->Fit("gaus");
  c[68]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPTTTIME_%i.%s",dirName,mType,runNumber,fileType); c[68]->Print(name); }

  c[69]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *chhistEEP = HistFromTProfile2D(EEPtimeCHProfile,"chhistEEP",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EE+ CH Timing;CH time average (ns)",runChar); 
  chhistEEP->SetTitle(mytitle);
  chhistEEP->GetXaxis()->SetNdivisions(512);
  if (chhistEEP->GetMean() != 0) chhistEEP->Fit("gaus");
  c[69]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEPCHTIME_%i.%s",dirName,mType,runNumber,fileType); c[69]->Print(name); }

  c[70]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *tthistEEM = HistFromTProfile2D(EEMtimeTTProfile,"tthistEEM",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EE- TT Timing;TT time average (ns)",runChar); 
  tthistEEM->SetTitle(mytitle);
  tthistEEM->GetXaxis()->SetNdivisions(512);
  if (tthistEEM->GetMean() != 0) tthistEEM->Fit("gaus");
  c[70]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMTTTIME_%i.%s",dirName,mType,runNumber,fileType); c[70]->Print(name); }

  c[71]->cd();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(111);
  TH1F *chhistEEM = HistFromTProfile2D(EEMtimeCHProfile,"chhistEEM",200, -30., 40.,-5.,25.);
  sprintf(mytitle,"%s EE- CH Timing;CH time average (ns)",runChar); 
  chhistEEM->SetTitle(mytitle);
  chhistEEM->GetXaxis()->SetNdivisions(512);
  if (chhistEEM->GetMean() != 0) chhistEEM->Fit("gaus");
  c[71]->SetLogy(1);
  if (printPics) { sprintf(name,"%s/%sAnalysis_EEMCHTIME_%i.%s",dirName,mType,runNumber,fileType); c[71]->Print(name); }
  
  cout << name << endl;

  return;

}

void drawEELines() {

  int ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};
 
  int iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};


 for ( int i=0; i<202; i++) {
   ixSectorsEE[i] += 1;
   iySectorsEE[i] += 1;
//   std::cout << i << " " << ixSectorsEE[i] << " " << iySectorsEE[i] << std::endl;
 }

 TLine l;
 l.SetLineWidth(1);
 for ( int i=0; i<201; i=i+1) {
   if ( (ixSectorsEE[i]!=1 || iySectorsEE[i]!=1) && 
	(ixSectorsEE[i+1]!=1 || iySectorsEE[i+1]!=1) ) {
     l.DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		ixSectorsEE[i+1], iySectorsEE[i+1]);
   }
 }

}

void customizeTProfile (TProfile* myTProfile) {
  if (myTProfile) {
    myTProfile->SetLineWidth(2);
    myTProfile->SetMarkerStyle(kFullCircle);
    myTProfile->SetMarkerSize(0.7);
  }
}

void customizeTHist (TH1F* myTHist) {
  if (myTHist) {
    myTHist->SetLineWidth(2);
    myTHist->SetMarkerStyle(kFullCircle);
    myTHist->SetMarkerSize(0.7);
  }
}


TH1F* CorrectProfToHist(TProfile *prof, const char * histname, double numb=0, double myScale = 1.0 )
{
  TH1F *temphist = new TH1F(histname,histname,prof->GetNbinsX(),prof->GetXaxis()->GetXmin(),prof->GetXaxis()->GetXmax());
  for (int i = 1; i < prof->GetNbinsX()+1; ++i)
  {
    //std::cout << " bin " << i << " is " << temphist->GetBinContent(i) << std::endl;
    if (prof->GetBinEntries(i) > 0 )
        {
          temphist->SetBinContent(i,prof->GetBinContent(i)+numb);
          temphist->SetBinError(i,prof->GetBinError(i));
        }
	else {temphist->SetBinContent(i,-100.);}
        //std::cout << " bin " << i  << " bin content before " << prof->GetBinContent(i) << " bin entries " << prof->GetBinEntries(i)  << " new bi\ncontent " << temphist->GetBinContent(i) << std::endl;
  }
  temphist->Sumw2();
  temphist->Scale(myScale);
  return temphist;
}

void ScaleTProfile2D(TProfile2D* myprof, Double_t myfac, Double_t myscale)
{
int nxbins = myprof->GetNbinsX();
int nybins = myprof->GetNbinsY();

for (int i=0; i<=(nxbins+2)*(nybins+2); i++ ) {   
       Double_t oldcont = myprof->GetBinContent(i);
       Double_t binents = myprof->GetBinEntries(i);
       if (binents == 0 ) {binents =1.;myprof->SetBinEntries(i,1); }
       myprof->SetBinContent(i,myscale*(oldcont+myfac)*binents);
}
}




TProfile2D* TProfToRelProf2D(TProfile2D *prof, const char * histname, double numb=0, double myScale = 1.0)
{
TProfile2D *myprof = prof->Clone(histname);
ScaleTProfile2D(myprof,numb,myScale);

return myprof;
}

TProfile* TProfToRelProf(TProfile *prof, const char * histname, double numb=0, double myScale = 1.0)
{
TProfile *myprof = prof->Clone(histname);
ScaleTProfile(myprof,numb,myScale);

return myprof;
}

void ScaleTProfile(TProfile* myprof, Double_t myfac, Double_t myscale)
{
int nxbins = myprof->GetNbinsX();

for (int i=1; i<(nxbins+1); i++ ) {   
       Double_t oldcont = myprof->GetBinContent(i);
       Double_t binents = myprof->GetBinEntries(i);
       Double_t binerrr = myprof->GetBinError(i);
       
       if (binents == 0 ) { continue; /*binents =1.;myprof->SetBinEntries(i,1);*/ }
       myprof->SetBinContent(i,myscale*(oldcont+myfac)*binents);
	   Double_t newentries = myprof->GetBinEntries(i);
	   Double_t newcont = myprof->GetBinContent(i);
	   //cout << " cont " << oldcont << " ent " << binents << " err " << binerrr << " new err " << myprof->GetBinError(i);
	   if ( newentries == 1) { myprof->SetBinError(i,5+fabs(myprof->GetBinContent(i)-myprof->GetBinContent(i)/2.5));}
           //else {myprof->SetBinError(i,binerrr*myscale+1.0);}
	   //cout << " newnew " << myprof->GetBinError(i) << std::endl;
	   if (newentries != binents) {std::cout << "NONONO" << std::endl;}
}
}

void EntryProfileFromTProfile2D(TProfile2D* myprof)
{
int nxbins = myprof->GetNbinsX();
int nybins = myprof->GetNbinsY();

for (int i=0; i<=(nxbins+2)*(nybins+2); i++ ) {   
       Double_t oldcont = myprof->GetBinContent(i);
       Double_t binents = myprof->GetBinEntries(i);
       if (binents == 0 ) { continue; }
       myprof->SetBinContent(i,binents*binents);
}
}


TProfile2D* TProfile2DOccupancyFromProf2D(TProfile2D *prof, const char * histname)
{
TProfile2D *myprof = prof->Clone(histname);
EntryProfileFromTProfile2D(myprof);
return myprof;
}

TH1F* HistFromTProfile2D(TProfile2D *prof, const char * histname, Int_t xbins, Double_t xmin, Double_t xmax, Double_t myfac, Double_t myscale)
{
int nxbins = prof->GetNbinsX();
int nybins = prof->GetNbinsY();

TH1F *temphist = new TH1F(histname,histname,xbins,xmin,xmax);

for (int i=0; i<=(nxbins+2)*(nybins+2); i++ ) {   
       Double_t oldcont = prof->GetBinContent(i);
       Double_t binents = prof->GetBinEntries(i);
       if (binents == 0 ) { continue; }
       temphist->Fill((oldcont+myfac)*myscale);
}
return temphist;
}




