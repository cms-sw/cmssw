
int Wait() {
     cout << " Continue [<RET>|q]?  "; 
     char x;
     x = getchar();
     if ((x == 'q') || (x == 'Q')) return 1;
     return 0;
}

void DrawCalibPlots(Char_t* infile = 0, Int_t runNum=0, Bool_t printPics = kTRUE, Char_t* fileType = "png", Char_t* dirName = ".", Bool_t doWait=kFALSE)
{

  gROOT->SetStyle("Plain");
  //gStyle->SetPalette(1,0); 
  gStyle->SetOptStat(10);

  if (!infile) {
    cout << " No input file specified !" << endl;
    return;
  }

  cout << "Producing Calib plots for: " << infile << endl;

  TFile* f = new TFile(infile);
  f->cd(); //added by jason for completeness
  f->cd("ecalCalibrationAnalyzer");
  f->ls();
  
  int runNumber = 0;
  runNumber = runNum;


  char name[100];  

  const int nHists1=14;
  const int nHists = nHists1;
  //  const int nHists = 9;
  cout << nHists1 << " " << nHists << endl;;

  TCanvas* c[nHists]; 
  char cname[100]; 

  for (int i=0; i<nHists1; i++) {
    sprintf(cname,"c%i",i);
    int x = (i%3)*600;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    c[i] =  new TCanvas(cname,cname,x,y,1200,800);
    cout << "Hists1 " << i << " : " << x << " , " << y << endl;
  }

  char runChar[50];
  sprintf(runChar,"Run %i ",runNumber);
  
//First thing is to print the 2-D histograms
// For these I will ignore the possibility of the "-1" value for now (UPDATE THIS LATER)
// 
  Int_t colors[]={kGreen,kWhite,kRed,kCyan+2,kBlack};
  Double_t levels[]={-1.5,-0.5,0.5,1.5,2.5,3.5}
  gStyle->SetNumberContours(5);
  gStyle->SetPalette(5,colors);
  //Simple Data Run Type 
  c[0]->cd();
  gStyle->SetOptStat(10);
  TH2F *dataHeadRunType_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DdataHeadRunType_"); //NEEDED as I stupidly started the histo names with a number
  if (dataHeadRunType_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dataHeadRunType_->GetTitle());
     dataHeadRunType_->SetFillColor(0);
     dataHeadRunType_->SetTitle(mytitle);
     dataHeadRunType_->SetMinimum(-1.5);
     dataHeadRunType_->SetMaximum(3.5);
     dataHeadRunType_->SetContour(5, levels);
     dataHeadRunType_->Draw("colz");
     dataHeadRunType_->GetZaxis()->Set(5,-1.5,3.5);
     dataHeadRunType_->GetZaxis()->SetBinLabel(1,"Unknown");
     dataHeadRunType_->GetZaxis()->SetBinLabel(2,"Empty");
     dataHeadRunType_->GetZaxis()->SetBinLabel(3,"Laser");
     dataHeadRunType_->GetZaxis()->SetBinLabel(4,"Testpulse");
     dataHeadRunType_->GetZaxis()->SetBinLabel(5,"Ped");
     dataHeadRunType_->GetZaxis()->SetLabelColor(kBlack);
     dataHeadRunType_->GetZaxis()->SetLabelSize(0.06);
     dataHeadRunType_->GetZaxis()->SetLabelOffset(-0.02);
     dataHeadRunType_->GetZaxis()->SetTickLength(0);
  
     dataHeadRunType_->GetYaxis()->SetTitleOffset(1.37);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_simpleDataRunType_%i.%s",dirName,runNumber,fileType); c[0]->Print(name); }
  }
  
  c[1]->cd();
  gStyle->SetOptStat(10);
  TH2F *simpleHeadRunType_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DsimpleHeadRunType_"); //NEEDED as I stupidly started the histo names with a number
  if (simpleHeadRunType_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,simpleHeadRunType_->GetTitle());
     simpleHeadRunType_->SetFillColor(0);
     simpleHeadRunType_->SetTitle(mytitle);
     simpleHeadRunType_->SetMinimum(-1.5);
     simpleHeadRunType_->SetMaximum(3.5);
     simpleHeadRunType_->SetContour(5, levels);
     simpleHeadRunType_->Draw("colz");
     simpleHeadRunType_->GetZaxis()->Set(5,-1.5,3.5);
     simpleHeadRunType_->GetZaxis()->SetBinLabel(1,"Unknown");
     simpleHeadRunType_->GetZaxis()->SetBinLabel(2,"Empty");
     simpleHeadRunType_->GetZaxis()->SetBinLabel(3,"Laser");
     simpleHeadRunType_->GetZaxis()->SetBinLabel(4,"Testpulse");
     simpleHeadRunType_->GetZaxis()->SetBinLabel(5,"Ped");
     simpleHeadRunType_->GetZaxis()->SetLabelColor(kBlack);
     simpleHeadRunType_->GetZaxis()->SetLabelSize(0.06);
     simpleHeadRunType_->GetZaxis()->SetLabelOffset(-0.02);
     simpleHeadRunType_->GetZaxis()->SetTickLength(0);
  
     simpleHeadRunType_->GetYaxis()->SetTitleOffset(1.37);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_simpleHeadRunType_%i.%s",dirName,runNumber,fileType); c[1]->Print(name); }
  }
  
  gStyle->SetNumberContours(20);
  gStyle->SetPalette(0,0); 

  c[2]->cd();
  gStyle->SetOptStat(10);
  TH2F *dccHeadRunType_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DdccHeadRunType_"); //NEEDED as I stupidly started the histo names with a number
  if (dccHeadRunType_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dccHeadRunType_->GetTitle());
     dccHeadRunType_->SetFillColor(0);
     dccHeadRunType_->SetTitle(mytitle);
     dccHeadRunType_->Draw("colz");
     dccHeadRunType_->GetZaxis()->SetTickLength(0.03);
  
     dccHeadRunType_->GetYaxis()->SetTitleOffset(1.37);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_dccHeadRunType_%i.%s",dirName,runNumber,fileType); c[2]->Print(name); }
  }
  gStyle->SetNumberContours(99);  
  gStyle->SetPalette(1,0); 

  c[3]->cd();
  gStyle->SetOptStat(10);
  TH2F *RunTypeByCycle_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DRunTypeByCycle_"); //NEEDED as I stupidly started the histo names with a number
  if (RunTypeByCycle_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,RunTypeByCycle_->GetTitle());
     RunTypeByCycle_->SetFillColor(0);
     RunTypeByCycle_->SetTitle(mytitle);
     RunTypeByCycle_->Draw("colztext");
     RunTypeByCycle_->GetXaxis()->SetTickLength(0.03);
     RunTypeByCycle_->GetXaxis()->SetBinLabel(1,"Unknown");
     RunTypeByCycle_->GetXaxis()->SetBinLabel(2,"Empty");
     RunTypeByCycle_->GetXaxis()->SetBinLabel(3,"Laser");
     RunTypeByCycle_->GetXaxis()->SetBinLabel(4,"Testpulse");
     RunTypeByCycle_->GetXaxis()->SetBinLabel(5,"Ped");
     RunTypeByCycle_->GetXaxis()->SetLabelSize(0.06);
     RunTypeByCycle_->SetMinimum(1.0);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_RunTypeByCycle_%i.%s",dirName,runNumber,fileType); c[3]->Print(name); }
  }
  
  c[4]->cd();
  gStyle->SetOptStat(10);
  TH2F *LaserFEDCycle_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DLaserFEDCycle_"); //NEEDED as I stupidly started the histo names with a number
  if (LaserFEDCycle_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,LaserFEDCycle_->GetTitle());
     LaserFEDCycle_->SetFillColor(0);
     LaserFEDCycle_->SetTitle(mytitle);
     LaserFEDCycle_->Draw("colztext");
     LaserFEDCycle_->GetZaxis()->SetTickLength(0.03);
     LaserFEDCycle_->SetMinimum(1.0);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_LaserFEDCycle_%i.%s",dirName,runNumber,fileType); c[4]->Print(name); }
  }
  
  c[5]->cd();
  gStyle->SetOptStat(10);
  TH2F *PedFEDCycle_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DPedFEDCycle_"); //NEEDED as I stupidly started the histo names with a number
  if (PedFEDCycle_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,PedFEDCycle_->GetTitle());
     PedFEDCycle_->SetFillColor(0);
     PedFEDCycle_->SetTitle(mytitle);
     PedFEDCycle_->Draw("colztext");
     PedFEDCycle_->GetZaxis()->SetTickLength(0.03);
     PedFEDCycle_->SetMinimum(1.0);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_PedFEDCycle_%i.%s",dirName,runNumber,fileType); c[5]->Print(name); }
  }
  
  c[6]->cd();
  gStyle->SetOptStat(10);
  TH2F *TestPulseFEDCycle_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DTestPulseFEDCycle_"); //NEEDED as I stupidly started the histo names with a number
  if (TestPulseFEDCycle_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,TestPulseFEDCycle_->GetTitle());
     TestPulseFEDCycle_->SetFillColor(0);
     TestPulseFEDCycle_->SetTitle(mytitle);
     TestPulseFEDCycle_->Draw("colztext");
     TestPulseFEDCycle_->GetZaxis()->SetTickLength(0.03);
     TestPulseFEDCycle_->SetMinimum(1.0);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_TestPulseFEDCycle_%i.%s",dirName,runNumber,fileType); c[6]->Print(name); }
  }
  
  c[7]->cd();
  gStyle->SetOptStat(10);
  TH2F *UnknownFEDCycle_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/2DUnknownFEDCycle_"); //NEEDED as I stupidly started the histo names with a number
  if (UnknownFEDCycle_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,UnknownFEDCycle_->GetTitle());
     UnknownFEDCycle_->SetFillColor(0);
     UnknownFEDCycle_->SetTitle(mytitle);
     UnknownFEDCycle_->Draw("colztext");
     UnknownFEDCycle_->GetZaxis()->SetTickLength(0.03);
     UnknownFEDCycle_->SetMinimum(1.0);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_UnknownFEDCycle_%i.%s",dirName,runNumber,fileType); c[7]->Print(name); }
  }
  
//Average Amplitudes & DCC comparisons
  const int mnHists=54;
  TCanvas* ap[mnHists];
  char apname[100];
  for (int i=0; i<mnHists; i++) {
    sprintf(apname,"ap%i",i);
    int x = (i%3)*600;     //int x = (i%3)*600;
    int y = (i/3)*100;     //int y = (i/3)*200;
    ap[i] =  new TCanvas(apname,apname,x,y,900,600);
    cout << "Hists " << i << " : " << x << " , " << y << endl;
  }
  int ampdccanvas = 0;
  bool inEE = false;

  for (int SM=-18; SM < 19; ++SM)  {
     if (SM==0) SM++;
     char SMstr[100];
     
     ( SM < 0) ? ( sprintf(SMstr,"%d",SM) ) : (sprintf(SMstr,"+%d",SM) ) ;
     if (((!inEE) && SM > -10) && ((!inEE) && SM < 10))  {inEE = true; sprintf(SMstr,"EE%s",SMstr); SM--;} else {inEE = false;sprintf(SMstr,"EB%s",SMstr) ;}
     gStyle->SetOptStat(10);
     TH1F *aveampHist  =  (TH1F*) f->Get(Form("ecalCalibrationAnalyzer/avgAmp_%s",SMstr)) ; //NEEDED as I stupidly started the histo names with a number
     if (aveampHist){
        ap[ampdccanvas]->cd();
        char mytitle[100]; sprintf(mytitle,"%s %s; Event Number; Average Amplitude (ADC)",runChar,aveampHist->GetTitle());
        aveampHist->SetFillColor(0);
        aveampHist->SetTitle(mytitle);
        aveampHist->Draw("");
        ap[ampdccanvas]->SetLogy(1);
        if (printPics) { sprintf(name,"%s/CalibAnalysis_aveampHist_%s_%i.%s", dirName,SMstr,runNumber,fileType); ap[ampdccanvas]->Print(name); }
     } 
     
     TCanvas *dccCan  =  (TCanvas*) f->Get(Form("ecalCalibrationAnalyzer/dccAndDataRunTypes_%s",SMstr)) ; //NEEDED as I stupidly started the histo names with a number
     if (dccCan){
        dccCan->cd();
        char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dccCan->GetTitle());
        dccCan->SetFillColor(0);
        dccCan->SetTitle(mytitle);
        dccCan->Draw("");
	

        if (printPics) { sprintf(name,"%s/CalibAnalysis_DccandDataHist_%s_%i.%s", dirName,SMstr,runNumber,fileType);  dccCan->Print(name); }
     } 
      
     ampdccanvas++;
       
  } 


// DCC Header Errors 
  c[8]->cd();
  gStyle->SetOptStat(10);
  TH1F *dccHeaderErrors  = (TH1F*) f->Get("ecalCalibrationAnalyzer/dccHeaderErrors"); //NEEDED as I stupidly started the histo names with a number
  if (dccHeaderErrors){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dccHeaderErrors->GetTitle());
     dccHeaderErrors->SetFillColor(0);
	 dccHeaderErrors->SetMaximum(1.);
     dccHeaderErrors->SetTitle(mytitle);
     dccHeaderErrors->Draw("");
     dccHeaderErrors->GetXaxis()->SetBinLabel(1,"Data-to-DCC RunType Conflicts");
     dccHeaderErrors->GetXaxis()->SetBinLabel(2,"DCC RunType Conflicts");
     dccHeaderErrors->GetXaxis()->SetBinLabel(3,"RTHalf Conflict");
     c[8]->SetLogy(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_dccHeaderErrors%i.%s",dirName,runNumber,fileType); c[8]->Print(name); }
  }
  
  c[9]->cd();
  gStyle->SetOptStat(10);
  TH2F *dataRunTypeVsBX_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/TwoDdataRunTypeVsBX_"); 
  if (dataRunTypeVsBX_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dataRunTypeVsBX_->GetTitle());
     dataRunTypeVsBX_->SetFillColor(0);
     dataRunTypeVsBX_->SetTitle(mytitle);
     dataRunTypeVsBX_->Draw("colztext");
     dataRunTypeVsBX_->GetZaxis()->SetTickLength(0.03);
     dataRunTypeVsBX_->SetMinimum(1.0);
	 dataRunTypeVsBX_->GetYaxis()->SetLabelOffset(-0.01);
     c[9]->SetLogz(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_dataRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[9]->Print(name); }
     dataRunTypeVsBX_->GetXaxis()->SetRangeUser(3490.,3490.);
     dataRunTypeVsBX_->GetXaxis()->SetNdivisions(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_zoomdataRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[9]->Print(name); }
  }
  
  c[10]->cd();
  gStyle->SetOptStat(10);
  TH2F *dccRunTypeVsBX_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/TwoDdccRunTypeVsBX_"); 
  if (dccRunTypeVsBX_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dccRunTypeVsBX_->GetTitle());
     dccRunTypeVsBX_->SetFillColor(0);
     dccRunTypeVsBX_->SetTitle(mytitle);
     dccRunTypeVsBX_->Draw("colztext");
     dccRunTypeVsBX_->GetZaxis()->SetTickLength(0.03);
     dccRunTypeVsBX_->SetMinimum(1.0);
	 dccRunTypeVsBX_->GetYaxis()->SetLabelOffset(-0.045);
     c[10]->SetLogz(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_dccRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[10]->Print(name); }
     dccRunTypeVsBX_->GetXaxis()->SetRangeUser(3490.,3490.);
     dccRunTypeVsBX_->GetXaxis()->SetNdivisions(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_zoomdccRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[10]->Print(name); }
  }
  
  c[11]->cd();
  gStyle->SetOptStat(10);
  TH2F *simpledccRunTypeVsBX_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/TwoDsimpledccRunTypeVsBX_"); 
  if (simpledccRunTypeVsBX_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,simpledccRunTypeVsBX_->GetTitle());
     simpledccRunTypeVsBX_->SetFillColor(0);
     simpledccRunTypeVsBX_->SetTitle(mytitle);
     simpledccRunTypeVsBX_->Draw("colztext");
     simpledccRunTypeVsBX_->GetZaxis()->SetTickLength(0.03);
     simpledccRunTypeVsBX_->SetMinimum(1.0);
	 simpledccRunTypeVsBX_->GetYaxis()->SetLabelOffset(-0.01);
     c[11]->SetLogz(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_simpledccRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[11]->Print(name); }
     simpledccRunTypeVsBX_->GetXaxis()->SetRangeUser(3490.,3490.);
     simpledccRunTypeVsBX_->GetXaxis()->SetNdivisions(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_zoomsimpledccRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[11]->Print(name); }
  }
  
  c[12]->cd();
  gStyle->SetOptStat(10);
  TH2F *dccInTCCRunTypeVsBX_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/TwoDdccInTCCRunTypeVsBX_"); 
  if (dccInTCCRunTypeVsBX_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,dccInTCCRunTypeVsBX_->GetTitle());
     dccInTCCRunTypeVsBX_->SetFillColor(0);
     dccInTCCRunTypeVsBX_->SetTitle(mytitle);
     dccInTCCRunTypeVsBX_->Draw("colztext");
     dccInTCCRunTypeVsBX_->GetZaxis()->SetTickLength(0.03);
     dccInTCCRunTypeVsBX_->SetMinimum(1.0);
	 dccInTCCRunTypeVsBX_->GetYaxis()->SetLabelOffset(-0.045);
     c[12]->SetLogz(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_dccInTCCRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[12]->Print(name); }
     dccInTCCRunTypeVsBX_->GetXaxis()->SetRangeUser(3490.,3490.);
     dccInTCCRunTypeVsBX_->GetXaxis()->SetNdivisions(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_zoomdccInTCCRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[12]->Print(name); }
  }
  
  c[13]->cd();
  gStyle->SetOptStat(10);
  TH2F *simpledccInTCCRunTypeVsBX_  = (TH2F*) f->Get("ecalCalibrationAnalyzer/TwoDsimpledccInTCCRunTypeVsBX_"); 
  if (simpledccInTCCRunTypeVsBX_){
     char mytitle[100]; sprintf(mytitle,"%s %s",runChar,simpledccInTCCRunTypeVsBX_->GetTitle());
     simpledccInTCCRunTypeVsBX_->SetFillColor(0);
     simpledccInTCCRunTypeVsBX_->SetTitle(mytitle);
     simpledccInTCCRunTypeVsBX_->Draw("colztext");
     simpledccInTCCRunTypeVsBX_->GetZaxis()->SetTickLength(0.03);
     simpledccInTCCRunTypeVsBX_->SetMinimum(1.0);
	 simpledccInTCCRunTypeVsBX_->GetYaxis()->SetLabelOffset(-0.01);
     c[13]->SetLogz(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_simpledccInTCCRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[13]->Print(name); }
     simpledccInTCCRunTypeVsBX_->GetXaxis()->SetRangeUser(3490.,3490.);
     simpledccInTCCRunTypeVsBX_->GetXaxis()->SetNdivisions(1);
     if (printPics) { sprintf(name,"%s/CalibAnalysis_zoomsimpledccInTCCRunTypeVsBX_%i.%s",dirName,runNumber,fileType); c[13]->Print(name); }
  }
  
  

cout << name << endl;



  return;

}





