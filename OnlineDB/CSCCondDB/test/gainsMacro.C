void gainsMacro(){
//proccess in batch mode. this takes a VERY long time out of batch mode
gROOT->SetBatch();
//contains several functions used in other macros as well
gROOT->ProcessLine(".L GenFuncMacro.C");

//get myFile. this is for batch processecing 
//copies the name of any file beginning with "c" from the /tmp/csccalib directory. 
char *myFileName;  //new file name for directory name
char *myFilePath; //for accessing file by root 
void *dirp = gSystem->OpenDirectory("/tmp/csccalib"); //read tmp directory
char *afile; //temp file name
while(afile = gSystem->GetDirEntry(dirp)) { //parse directory
   char *bfile[0] = afile[0]; //new temp file name
   if (bfile[0]=='c') { //if file begins with c
     printf("file: %s\n",afile); //check name
     myFileName = afile; //set for out of scope processing
   }
   myFilePath = Form("/tmp/csccalib/%s", myFileName);
}

//set global parameters
int nDDU  = 1;
int nCham =  1;
int nLayer = 6;

//style-ize all canvases
gStyle->SetCanvasColor(0);
gStyle->SetPadColor(0);
gStyle->SetPadBorderMode(0);
gStyle->SetCanvasBorderMode(0);
gStyle->SetFrameBorderMode(0);
gStyle->SetStatH(0.2);
gStyle->SetStatW(0.3);

directoryCheck();

//open file generated from analyzer
std::cout << "opening: " << myFileName << std::endl; 
TFile *myFile = TFile::Open(myFilePath);

Calibration->Draw("cham");
int nCham = htemp->GetXaxis()->GetXmax();

//system is in folder "test". check:  
directoryCheck();

//this is one big section of linux directory processing
//if this is edited, be careful! it's easy to mess up. 
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");
makeDirectory("images");
gSystem->cd("images");
 //create subdirectory Gains
makeDirectory("Gains");
gSystem->cd("Gains/");
//create subdirectory for run
makeDirectory(myFileName);
gSystem->cd(myFileName);
gSystem->cd("../../../");

SlopeFlagGraphs(myFileName);
InterceptFlagGraphs(myFileName);
gStyle->SetOptStat(0);
gStyle->SetMarkerStyle(20);
gStyle->SetMarkerSize(0.6);
SlopeVsStrip(myFileName, nDDU,nCham,nLayer);
InterceptVsStrip(myFileName, nDDU,nCham,nLayer);
myFile->Close();
gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_1_1/src/OnlineDB/CSCCondDB/test");
myFilePath = Form("/tmp/csccalib/%s", myFileName); 
std::cout << "just before PulseGraphs" << std::endl;
std::cout << " fileName: " << myFileName << std::endl;
std::cout << " filePath: " << myFilePath << std::endl;
GetADCCharge (myFileName, myFilePath);
gStyle->SetOptStat(1);
gROOT->ProcessLine(".q");
} 
 
//this creates an array which can then be accessed
//proper usage: 
//  int idArray[9];
//  GetChamberIDs(idArray);
//  std::cout << "id for chamber 0: " << idArray[0] << std::endl;
void GetChamberIDs(int IDArray[9]){
TCanvas *IDcanv = new TCanvas ("idGraph", "idGraph");
IDcanv->cd();
TH1F *idDummy = new TH1F("idDummy", "idDummy", 480, 111000, 250000);
idDummy->Draw();
for (int chamber=0; chamber<9; ++chamber){
  TString idCut = Form ("cham==%d", chamber);
  Calibration->Project("idDummy", "id", idCut);
  Int_t idNum = idDummy->GetMean();
  IDArray[chamber]=idNum;
}
}

void GetADCCharge(TString myFileName, TString myFilePath){
TFile *myFile = TFile::Open(myFilePath);

TH1F *ADCCharge;  
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");
TCanvas *ADCChargeCanvas = new TCanvas("ADC_Charge", "ADC_Charge",1100,700);
ADCCharge = (TH1F*)myFile->Get("adcCharge");  
ADCChargeCanvas->cd(); 
ADCCharge->GetYaxis()->SetTitle("ADC Count"); 
ADCCharge->GetXaxis()->SetTitle("Charge"); 
ADCCharge->Draw();
ADCChargeCanvas->Update();  

gSystem->cd("images/Gains/");
gSystem->cd(myFileName);
PrintAsGif(ADCChargeCanvas, "ADC_Charge");
gSystem->cd("../../../");
directoryCheck(); 
myFile->Close();
}

void SlopeVsStrip(TString myFileName, int nDDU, int nCham, int nLayer){
gSystem->cd("images/Gains/");
gSystem->cd(myFileName);
makeDirectory("SlopeVsStripGraphs");
gSystem->cd("SlopeVsStripGraphs");

TH2F *slopeGraph;
TCanvas *slopeCanv;
for (int i=0; i<nDDU; ++i){
  int idArray[9];
  GetChamberIDs(idArray);
  for (int j=0; j<nCham; ++j){
    TString slopeCanvName = Form("Slope_per_strip_Chamber_%d_DDU_%d",idArray[j],i);
    slopeCanv = new TCanvas(slopeCanvName, slopeCanvName,200, 20, 1100,700);
    slopeCanv->Divide(3,2);
    for (int k=0; k<nLayer; ++k){
      TString slopeGraphName = Form("Layer_%d_Slope",k);
      slopeGraph = new TH2F(slopeGraphName,slopeGraphName,80,0,80,20,5,10); 
      slopeCanv->cd(k+1); 
      TString slopeGraphCut = Form("cham==%d&&layer==%d",j,k); 
      slopeGraph->SetName(slopeGraphName);
      slopeGraph->SetTitle(slopeGraphName);
      slopeGraph->GetXaxis()->SetTitle("strip");
      slopeGraph->GetXaxis()->SetTitleSize(0.06);
      slopeGraph->GetXaxis()->SetTitleOffset(0.7);
      slopeGraph->GetYaxis()->SetTitleSize(0.06);
      slopeGraph->GetYaxis()->SetTitleOffset(0.7);
      slopeGraph->GetYaxis()->SetTitle("slope");
      slopeGraph->Draw();
      Calibration->Project(slopeGraphName, "slope:strip", slopeGraphCut);      
      slopeCanv->Update(); 
    }//nLayer
    PrintAsGif(slopeCanv, slopeCanvName);
  }//nCham
}//nDDU
gSystem->cd("../../../../");
directoryCheck();
}//SlopeVsStrip

void InterceptVsStrip(TString myFileName, int nDDU, int nCham, int nLayer){
gSystem->cd("images/Gains/");
gSystem->cd(myFileName);
makeDirectory("InterceptVsStripGraphs");
gSystem->cd("InterceptVsStripGraphs");

TH2F *interceptGraph;
TCanvas *interceptCanv;
for (int i=0; i<nDDU; ++i){
  int idArray[9];
  GetChamberIDs(idArray);
  for (int j=0; j<nCham; ++j){
    TString interceptCanvName = Form("Intercept_per_strip_Chamber_%d_DDU_%d",idArray[j],i);
    interceptCanv = new TCanvas(interceptCanvName, interceptCanvName,200, 20, 1100,700);
    interceptCanv->Divide(3,2);
    for (int k=0; k<nLayer; ++k){
      TString interceptGraphName = Form("Layer_%d_Intercept",k);
      interceptGraph = new TH2F(interceptGraphName,interceptGraphName,80,0,80,20,-50,50); 
      interceptCanv->cd(k+1); 
      TString interceptGraphCut = Form("cham==%d&&layer==%d",j,k); 
      interceptGraph->SetName(interceptGraphName);
      interceptGraph->SetTitle(interceptGraphName);
      interceptGraph->GetXaxis()->SetTitle("strip");
      interceptGraph->GetXaxis()->SetTitleSize(0.06);
      interceptGraph->GetXaxis()->SetTitleOffset(0.7);
      interceptGraph->GetYaxis()->SetTitleSize(0.06);
      interceptGraph->GetYaxis()->SetTitleOffset(0.7);
      interceptGraph->GetYaxis()->SetTitle("intercept");
      interceptGraph->Draw();
      Calibration->Project(interceptGraphName, "intercept:strip", interceptGraphCut);      
      interceptCanv->Update(); 
    }//nLayer
    PrintAsGif(interceptCanv, interceptCanvName);
  }//nCham
}//nDDU
gSystem->cd("../../../../");
directoryCheck();
}//InterceptVsStrip

void SlopeFlagGraphs(TString myFileName){
gSystem->cd("images/Gains/");
gSystem->cd(myFileName);

TCanvas *flagGainCanv = new TCanvas("flagGainCanv", "flagGainCanv", 200,10,800,800);
flagGainCanv->SetCanvasSize(1200,800);
flagGainGraph_1D = new TH1F("flagGainGraph_1D", "flagGainGraph_1D", 5, 0, 5);
flagGainGraph_1D->GetYaxis()->SetTitle("Number of flag of each type");
flagGainGraph_1D->GetYaxis()->SetLabelSize(0.035);

TLegend *LegGain = new TLegend(0.85,0.8,0.98,0.98);
LegGain->SetHeader("Slope Flags Definitions");
LegGain->SetFillColor(0);
LegGain->SetTextSize(0);

Calibration->UseCurrentStyle(); 
Calibration->Draw("flagGain>>flagGainGraph_1D");

LegGain->AddEntry("", "1: Good");
LegGain->AddEntry("", "2: Low slope, fit fails");
LegGain->AddEntry("", "3: High slope., fit fails");
LegGain->Draw("same");

flagGainCanv->Update();

/////////////////////////////////       2D GRAPHS            ////////////////////////////////
//no statistics shown
gStyle->SetOptStat(0);

////////// Strip Flag Graphs///////////////
//canvas
TCanvas *flagGainStripCanv = new TCanvas("flagGainStripCanv", "flagGainStripCanv", 200,10,800,800);
flagGainStripCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainStrip = new TLegend(0.85,0.8,0.98,0.98);
LegGainStrip->SetHeader("Slope Flags Definitions");
LegGainStrip->SetFillColor(0);
LegGainStrip->SetTextSize(0);

//final histogram for display
flagGainGraph_2D_Strip = new TH2F("flagGainGraph_2D_Strip", "flagGainGraph_2D_Strip", 80, 0, 80, 4, 0, 4);
///dummy histo to get bin maximum
flagGainGraph_2D_Strip0 = new TH2F("flagGainGraph_2D_Strip0", "flagGainGraph_2D_Strip0", 80, 0, 80, 4, 0, 4);
//one histo for each flag value. 
flagGainGraph_2D_Strip1 = new TH2F("flagGainGraph_2D_Strip1", "flagGainGraph_2D_Strip1", 80, 0, 80, 4, 0, 4);
flagGainGraph_2D_Strip2 = new TH2F("flagGainGraph_2D_Strip2", "flagGainGraph_2D_Strip2", 80, 0, 80, 4, 0, 4);
flagGainGraph_2D_Strip3 = new TH2F("flagGainGraph_2D_Strip3", "flagGainGraph_2D_Strip3", 80, 0, 80, 4, 0, 4);
flagGainGraph_2D_Strip4 = new TH2F("flagGainGraph_2D_Strip4", "flagGainGraph_2D_Strip4", 80, 0, 80, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagGainGraph_2D_Strip0","flagGain:strip"); 
Double_t binMaxValStrip = flagGainGraph_2D_Strip0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagGainGraph_2D_Strip->SetMaximum(binMaxValStrip);
//fill each "bin"
Calibration->Project("flagGainGraph_2D_Strip1","flagGain:strip", "flagGain==1","box"); 
Calibration->Project("flagGainGraph_2D_Strip2","flagGain:strip", "flagGain==2","box"); 
Calibration->Project("flagGainGraph_2D_Strip3","flagGain:strip", "flagGain==3","box"); 

//set appropriate colors
flagGainGraph_2D_Strip1->SetFillColor(1);//Black for eveything is OK
flagGainGraph_2D_Strip2->SetFillColor(2);//red for VERY BAD
flagGainGraph_2D_Strip3->SetFillColor(3);//Green for pretty good

flagGainStripCanv->cd(); 
flagGainGraph_2D_Strip->GetYaxis()->SetTitle("Flag");
flagGainGraph_2D_Strip->GetXaxis()->SetTitle("Strip");
  
//draw original histogram, empty
flagGainGraph_2D_Strip->Draw("box");
//overlay the individual "bin" graphs
flagGainGraph_2D_Strip1->Draw("samebox");
flagGainGraph_2D_Strip2->Draw("samebox");
flagGainGraph_2D_Strip3->Draw("samebox");
flagGainGraph_2D_Strip4->Draw("samebox");

//set legend entries appropriately
LegGainStrip->AddEntry(flagGainGraph_2D_Strip1, "Good", "f");
LegGainStrip->AddEntry(flagGainGraph_2D_Strip2, "Low slope, fit fails", "f");
LegGainStrip->AddEntry(flagGainGraph_2D_Strip3, "High slope, fit fails", "f");
LegGainStrip->Draw("same");

//print as gif
PrintAsGif(flagGainStripCanv, "flagGainStrip");

///////// CHAMBER flag Graph ///////////////////////
TCanvas *flagGainChamberCanv = new TCanvas("flagGainChamberCanv", "flagGainChamberCanv", 200,10,800,800);
flagGainChamberCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainChamber = new TLegend(0.85,0.8,0.98,0.98);
LegGainChamber->SetHeader("Slope Flags Definitions");
LegGainChamber->SetFillColor(0);
LegGainChamber->SetTextSize(0);

//final histogram for display
flagGainGraph_2D_Chamber = new TH2F("flagGainGraph_2D_Chamber", "flagGainGraph_2D_Chamber", 9, 0, 9, 4, 0, 4);
///dummy histo to get bin maximum
flagGainGraph_2D_Chamber0 = new TH2F("flagGainGraph_2D_Chamber0", "flagGainGraph_2D_Chamber0", 9, 0, 9, 4, 0, 4);
//one histo for each flag value. 
flagGainGraph_2D_Chamber1 = new TH2F("flagGainGraph_2D_Chamber1", "flagGainGraph_2D_Chamber1", 9, 0, 9, 4, 0, 4);
flagGainGraph_2D_Chamber2 = new TH2F("flagGainGraph_2D_Chamber2", "flagGainGraph_2D_Chamber2", 9, 0, 9, 4, 0, 4);
flagGainGraph_2D_Chamber3 = new TH2F("flagGainGraph_2D_Chamber3", "flagGainGraph_2D_Chamber3", 9, 0, 9, 4, 0, 4);
flagGainGraph_2D_Chamber4 = new TH2F("flagGainGraph_2D_Chamber4", "flagGainGraph_2D_Chamber4", 9, 0, 9, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagGainGraph_2D_Chamber0","flagGain:cham"); 
Double_t binMaxValCham = flagGainGraph_2D_Chamber0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagGainGraph_2D_Chamber->SetMaximum(binMaxValCham);

//fill each "bin"
Calibration->Project("flagGainGraph_2D_Chamber1","flagGain:cham", "flagGain==1", "box"); 
Calibration->Project("flagGainGraph_2D_Chamber2","flagGain:cham", "flagGain==2", "box"); 
Calibration->Project("flagGainGraph_2D_Chamber3","flagGain:cham", "flagGain==3", "box"); 

//set appropriate colors
flagGainGraph_2D_Chamber1->SetFillColor(1);//Black for eveything is OK
flagGainGraph_2D_Chamber2->SetFillColor(2);//red for VERY BAD
flagGainGraph_2D_Chamber3->SetFillColor(3);//Green for pretty good

int idArray[9];
GetChamberIDs(idArray);
for (int chamNum = 0; chamNum<9; ++chamNum){
int chamNumPlus = chamNum + 1; //for bin access

Int_t chamber_id_int = idArray[chamNum]; //set individual id as int
std::stringstream chamber_id_stream;     //define variable in intermediate format
chamber_id_stream << chamber_id_int;     //convert from int to intermediate "stringstream" format
TString chamber_id_str = chamber_id_stream.str();  //convert from stream into string
 if (chamber_id_str.BeginsWith("220")==0 ){  //binary check, i.e. if the string doesn't begin with 220
   chamber_id_str=0;                         //clean out; set to 0.
 }else{
   chamber_id_str.Remove(8,8);               //remove 0 at end
   chamber_id_str.Remove(0,3);               //remove 220 at beginning 
 }
flagGainGraph_2D_Chamber->GetXaxis()->SetBinLabel(chamNumPlus,chamber_id_str); //set bins to have chamber names
}

flagGainGraph_2D_Chamber->GetYaxis()->SetTitle("Flag");
flagGainGraph_2D_Chamber->GetXaxis()->SetTitle("Chamber");

flagGainChamberCanv->cd();  
//draw original histogram, empty
flagGainGraph_2D_Chamber->Draw("box");
//overlay the individual "bin" graphs
flagGainGraph_2D_Chamber1->Draw("samebox");
flagGainGraph_2D_Chamber2->Draw("samebox");
flagGainGraph_2D_Chamber3->Draw("samebox");
flagGainGraph_2D_Chamber4->Draw("samebox");

//set legend entries appropriately
LegGainChamber->AddEntry(flagGainGraph_2D_Chamber1, "Good", "f");
LegGainChamber->AddEntry(flagGainGraph_2D_Chamber2, "Low slope, fit fails", "f");
LegGainChamber->AddEntry(flagGainGraph_2D_Chamber3, "High slope, fit fails", "f");
LegGainChamber->Draw("same");

  //print as gif
PrintAsGif(flagGainChamberCanv, "flagGainChamber");

/////////////////////// Layer Graphs //////////////

TCanvas *flagGainLayerCanv = new TCanvas("flagGainLayerCanv", "flagGainLayerCanv", 200,10,800,800);
flagGainLayerCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainLayer = new TLegend(0.85,0.8,0.98,0.98);
LegGainLayer->SetHeader("Slope Flags Definitions");
LegGainLayer->SetFillColor(0);
LegGainLayer->SetTextSize(0);

//final histogram for display
flagGainGraph_2D_Layer = new TH2F("flagGainGraph_2D_Layer", "flagGainGraph_2D_Layer", 6, 0, 6, 4, 0, 4);
///dummy histo to get bin maximum
flagGainGraph_2D_Layer0 = new TH2F("flagGainGraph_2D_Layer0", "flagGainGraph_2D_Layer0", 6, 0, 6, 4, 0, 4);
//one histo for each flag value. 
flagGainGraph_2D_Layer1 = new TH2F("flagGainGraph_2D_Layer1", "flagGainGraph_2D_Layer1", 6, 0, 6, 4, 0, 4);
flagGainGraph_2D_Layer2 = new TH2F("flagGainGraph_2D_Layer2", "flagGainGraph_2D_Layer2", 6, 0, 6, 4, 0, 4);
flagGainGraph_2D_Layer3 = new TH2F("flagGainGraph_2D_Layer3", "flagGainGraph_2D_Layer3", 6, 0, 6, 4, 0, 4);
flagGainGraph_2D_Layer4 = new TH2F("flagGainGraph_2D_Layer4", "flagGainGraph_2D_Layer4", 6, 0, 6, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagGainGraph_2D_Layer0","flagGain:layer"); 
Double_t binMaxValLayer = flagGainGraph_2D_Layer0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagGainGraph_2D_Layer->SetMaximum(binMaxValLayer);

//fill each "bin"
Calibration->Project("flagGainGraph_2D_Layer1","flagGain:layer", "flagGain==1", "box"); 
Calibration->Project("flagGainGraph_2D_Layer2","flagGain:layer", "flagGain==2", "box"); 
Calibration->Project("flagGainGraph_2D_Layer3","flagGain:layer", "flagGain==3", "box"); 

//set appropriate colors
flagGainGraph_2D_Layer1->SetFillColor(1);//Black for eveything is OK
flagGainGraph_2D_Layer2->SetFillColor(2);//red for VERY BAD
flagGainGraph_2D_Layer3->SetFillColor(3);//Green for pretty good

flagGainLayerCanv->cd(); 
flagGainGraph_2D_Layer->GetYaxis()->SetTitle("Flag");
flagGainGraph_2D_Layer->GetXaxis()->SetTitle("Layer");
  
//draw original histogram, empty
flagGainGraph_2D_Layer->Draw("box");
//overlay the individual "bin" graphs
flagGainGraph_2D_Layer1->Draw("samebox");
flagGainGraph_2D_Layer2->Draw("samebox");
flagGainGraph_2D_Layer3->Draw("samebox");
flagGainGraph_2D_Layer4->Draw("samebox");

//set legend entries appropriately
LegGainLayer->AddEntry(flagGainGraph_2D_Layer1, "Good", "f");
LegGainLayer->AddEntry(flagGainGraph_2D_Layer2, "Low slope, fit fails", "f");
LegGainLayer->AddEntry(flagGainGraph_2D_Layer3, "High slope, fit fails", "f");
LegGainLayer->Draw("same");

//print as gif
PrintAsGif(flagGainLayerCanv, "flagGainLayer");
gSystem->cd("../../../");
directoryCheck();
gStyle->SetOptStat(1); 
}

void InterceptFlagGraphs(TString myFileName){
gSystem->cd("images/Gains/");
gSystem->cd(myFileName);

TCanvas *flagInterceptCanv = new TCanvas("flagInterceptCanv", "flagInterceptCanv", 200,10,800,800);
flagInterceptCanv->SetCanvasSize(1200,800);
flagInterceptGraph_1D = new TH1F("flagInterceptGraph_1D", "flagInterceptGraph_1D", 5, 0, 5);
flagInterceptGraph_1D->GetYaxis()->SetTitle("Number of flag of each type");
flagInterceptGraph_1D->GetYaxis()->SetLabelSize(0.035);

TLegend *LegGain = new TLegend(0.85,0.8,0.98,0.98);
LegGain->SetHeader("Intercept Flags Definitions");
LegGain->SetFillColor(0);
LegGain->SetTextSize(0);

Calibration->UseCurrentStyle(); 
Calibration->Draw("flagIntercept>>flagInterceptGraph_1D");

LegGain->AddEntry("", "1: Good");
LegGain->AddEntry("", "2: Low intercept, fit fails");
LegGain->AddEntry("", "3: High intercept., fit fails");
LegGain->Draw("same");

flagInterceptCanv->Update();

/////////////////////////////////       2D GRAPHS            ////////////////////////////////
//no statistics shown
gStyle->SetOptStat(0);

////////// Strip Flag Graphs///////////////
//canvas
TCanvas *flagInterceptStripCanv = new TCanvas("flagInterceptStripCanv", "flagInterceptStripCanv", 200,10,800,800);
flagInterceptStripCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainStrip = new TLegend(0.85,0.8,0.98,0.98);
LegGainStrip->SetHeader("Intercept Flags Definitions");
LegGainStrip->SetFillColor(0);
LegGainStrip->SetTextSize(0);

//final histogram for display
flagInterceptGraph_2D_Strip = new TH2F("flagInterceptGraph_2D_Strip", "flagInterceptGraph_2D_Strip", 80, 0, 80, 4, 0, 4);
///dummy histo to get bin maximum
flagInterceptGraph_2D_Strip0 = new TH2F("flagInterceptGraph_2D_Strip0", "flagInterceptGraph_2D_Strip0", 80, 0, 80, 4, 0, 4);
//one histo for each flag value. 
flagInterceptGraph_2D_Strip1 = new TH2F("flagInterceptGraph_2D_Strip1", "flagInterceptGraph_2D_Strip1", 80, 0, 80, 4, 0, 4);
flagInterceptGraph_2D_Strip2 = new TH2F("flagInterceptGraph_2D_Strip2", "flagInterceptGraph_2D_Strip2", 80, 0, 80, 4, 0, 4);
flagInterceptGraph_2D_Strip3 = new TH2F("flagInterceptGraph_2D_Strip3", "flagInterceptGraph_2D_Strip3", 80, 0, 80, 4, 0, 4);
flagInterceptGraph_2D_Strip4 = new TH2F("flagInterceptGraph_2D_Strip4", "flagInterceptGraph_2D_Strip4", 80, 0, 80, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagInterceptGraph_2D_Strip0","flagIntercept:strip"); 
Double_t binMaxValStrip = flagInterceptGraph_2D_Strip0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagInterceptGraph_2D_Strip->SetMaximum(binMaxValStrip);
//fill each "bin"
Calibration->Project("flagInterceptGraph_2D_Strip1","flagIntercept:strip", "flagIntercept==1","box"); 
Calibration->Project("flagInterceptGraph_2D_Strip2","flagIntercept:strip", "flagIntercept==2","box"); 
Calibration->Project("flagInterceptGraph_2D_Strip3","flagIntercept:strip", "flagIntercept==3","box"); 

//set appropriate colors
flagInterceptGraph_2D_Strip1->SetFillColor(1);//Black for eveything is OK
flagInterceptGraph_2D_Strip2->SetFillColor(2);//red for VERY BAD
flagInterceptGraph_2D_Strip3->SetFillColor(3);//Green for pretty good

flagInterceptStripCanv->cd(); 
flagInterceptGraph_2D_Strip->GetYaxis()->SetTitle("Flag");
flagInterceptGraph_2D_Strip->GetXaxis()->SetTitle("Strip");
  
//draw original histogram, empty
flagInterceptGraph_2D_Strip->Draw("box");
//overlay the individual "bin" graphs
flagInterceptGraph_2D_Strip1->Draw("samebox");
flagInterceptGraph_2D_Strip2->Draw("samebox");
flagInterceptGraph_2D_Strip3->Draw("samebox");
flagInterceptGraph_2D_Strip4->Draw("samebox");

//set legend entries appropriately
LegGainStrip->AddEntry(flagInterceptGraph_2D_Strip1, "Good", "f");
LegGainStrip->AddEntry(flagInterceptGraph_2D_Strip2, "Low intercept, fit fails", "f");
LegGainStrip->AddEntry(flagInterceptGraph_2D_Strip3, "High intercept, fit fails", "f");
LegGainStrip->Draw("same");

//print as gif
PrintAsGif(flagInterceptStripCanv, "flagInterceptStrip");

///////// CHAMBER flag Graph ///////////////////////
TCanvas *flagInterceptChamberCanv = new TCanvas("flagInterceptChamberCanv", "flagInterceptChamberCanv", 200,10,800,800);
flagInterceptChamberCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainChamber = new TLegend(0.85,0.8,0.98,0.98);
LegGainChamber->SetHeader("Intercept Flags Definitions");
LegGainChamber->SetFillColor(0);
LegGainChamber->SetTextSize(0);

//final histogram for display
flagInterceptGraph_2D_Chamber = new TH2F("flagInterceptGraph_2D_Chamber", "flagInterceptGraph_2D_Chamber", 9, 0, 9, 4, 0, 4);
///dummy histo to get bin maximum
flagInterceptGraph_2D_Chamber0 = new TH2F("flagInterceptGraph_2D_Chamber0", "flagInterceptGraph_2D_Chamber0", 9, 0, 9, 4, 0, 4);
//one histo for each flag value. 
flagInterceptGraph_2D_Chamber1 = new TH2F("flagInterceptGraph_2D_Chamber1", "flagInterceptGraph_2D_Chamber1", 9, 0, 9, 4, 0, 4);
flagInterceptGraph_2D_Chamber2 = new TH2F("flagInterceptGraph_2D_Chamber2", "flagInterceptGraph_2D_Chamber2", 9, 0, 9, 4, 0, 4);
flagInterceptGraph_2D_Chamber3 = new TH2F("flagInterceptGraph_2D_Chamber3", "flagInterceptGraph_2D_Chamber3", 9, 0, 9, 4, 0, 4);
flagInterceptGraph_2D_Chamber4 = new TH2F("flagInterceptGraph_2D_Chamber4", "flagInterceptGraph_2D_Chamber4", 9, 0, 9, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagInterceptGraph_2D_Chamber0","flagIntercept:cham"); 
Double_t binMaxValCham = flagInterceptGraph_2D_Chamber0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagInterceptGraph_2D_Chamber->SetMaximum(binMaxValCham);

//fill each "bin"
Calibration->Project("flagInterceptGraph_2D_Chamber1","flagIntercept:cham", "flagIntercept==1", "box"); 
Calibration->Project("flagInterceptGraph_2D_Chamber2","flagIntercept:cham", "flagIntercept==2", "box"); 
Calibration->Project("flagInterceptGraph_2D_Chamber3","flagIntercept:cham", "flagIntercept==3", "box"); 

//set appropriate colors
flagInterceptGraph_2D_Chamber1->SetFillColor(1);//Black for eveything is OK
flagInterceptGraph_2D_Chamber2->SetFillColor(2);//red for VERY BAD
flagInterceptGraph_2D_Chamber3->SetFillColor(3);//Green for pretty good

int idArray[9];
GetChamberIDs(idArray);
for (int chamNum = 0; chamNum<9; ++chamNum){
int chamNumPlus = chamNum + 1; //for bin access

Int_t chamber_id_int = idArray[chamNum]; //set individual id as int
std::stringstream chamber_id_stream;     //define variable in intermediate format
chamber_id_stream << chamber_id_int;     //convert from int to intermediate "stringstream" format
TString chamber_id_str = chamber_id_stream.str();  //convert from stream into string
 if (chamber_id_str.BeginsWith("220")==0 ){  //binary check, i.e. if the string doesn't begin with 220
   chamber_id_str=0;                         //clean out; set to 0.
 }else{
   chamber_id_str.Remove(8,8);               //remove 0 at end
   chamber_id_str.Remove(0,3);               //remove 220 at beginning 
 }
flagInterceptGraph_2D_Chamber->GetXaxis()->SetBinLabel(chamNumPlus,chamber_id_str); //set bins to have chamber names
}

flagInterceptGraph_2D_Chamber->GetYaxis()->SetTitle("Flag");
flagInterceptGraph_2D_Chamber->GetXaxis()->SetTitle("Chamber");

flagInterceptChamberCanv->cd(); 
 //draw original histogram, empty
flagInterceptGraph_2D_Chamber->Draw("box");
//overlay the individual "bin" graphs
flagInterceptGraph_2D_Chamber1->Draw("samebox");
flagInterceptGraph_2D_Chamber2->Draw("samebox");
flagInterceptGraph_2D_Chamber3->Draw("samebox");
flagInterceptGraph_2D_Chamber4->Draw("samebox");

//set legend entries appropriately
LegGainChamber->AddEntry(flagInterceptGraph_2D_Chamber1, "Good", "f");
LegGainChamber->AddEntry(flagInterceptGraph_2D_Chamber2, "Low intercept, fit fails", "f");
LegGainChamber->AddEntry(flagInterceptGraph_2D_Chamber3, "High intercept, fit fails", "f");
LegGainChamber->Draw("same");

  //print as gif
PrintAsGif(flagInterceptChamberCanv, "flagInterceptChamber");

/////////////////////// Layer Graphs //////////////

TCanvas *flagInterceptLayerCanv = new TCanvas("flagInterceptLayerCanv", "flagInterceptLayerCanv", 200,10,800,800);
flagInterceptLayerCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegGainLayer = new TLegend(0.85,0.8,0.98,0.98);
LegGainLayer->SetHeader("Intercept Flags Definitions");
LegGainLayer->SetFillColor(0);
LegGainLayer->SetTextSize(0);

//final histogram for display
flagInterceptGraph_2D_Layer = new TH2F("flagInterceptGraph_2D_Layer", "flagInterceptGraph_2D_Layer", 6, 0, 6, 4, 0, 4);
///dummy histo to get bin maximum
flagInterceptGraph_2D_Layer0 = new TH2F("flagInterceptGraph_2D_Layer0", "flagInterceptGraph_2D_Layer0", 6, 0, 6, 4, 0, 4);
//one histo for each flag value. 
flagInterceptGraph_2D_Layer1 = new TH2F("flagInterceptGraph_2D_Layer1", "flagInterceptGraph_2D_Layer1", 6, 0, 6, 4, 0, 4);
flagInterceptGraph_2D_Layer2 = new TH2F("flagInterceptGraph_2D_Layer2", "flagInterceptGraph_2D_Layer2", 6, 0, 6, 4, 0, 4);
flagInterceptGraph_2D_Layer3 = new TH2F("flagInterceptGraph_2D_Layer3", "flagInterceptGraph_2D_Layer3", 6, 0, 6, 4, 0, 4);
flagInterceptGraph_2D_Layer4 = new TH2F("flagInterceptGraph_2D_Layer4", "flagInterceptGraph_2D_Layer4", 6, 0, 6, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagInterceptGraph_2D_Layer0","flagIntercept:layer"); 
Double_t binMaxValLayer = flagInterceptGraph_2D_Layer0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagInterceptGraph_2D_Layer->SetMaximum(binMaxValLayer);

//fill each "bin"
Calibration->Project("flagInterceptGraph_2D_Layer1","flagIntercept:layer", "flagIntercept==1", "box"); 
Calibration->Project("flagInterceptGraph_2D_Layer2","flagIntercept:layer", "flagIntercept==2", "box"); 
Calibration->Project("flagInterceptGraph_2D_Layer3","flagIntercept:layer", "flagIntercept==3", "box"); 

//set appropriate colors
flagInterceptGraph_2D_Layer1->SetFillColor(1);//Black for eveything is OK
flagInterceptGraph_2D_Layer2->SetFillColor(2);//red for VERY BAD
flagInterceptGraph_2D_Layer3->SetFillColor(3);//Green for pretty good

flagInterceptLayerCanv->cd(); 
flagInterceptGraph_2D_Layer->GetYaxis()->SetTitle("Flag");
flagInterceptGraph_2D_Layer->GetXaxis()->SetTitle("Layer");
  
//draw original histogram, empty
flagInterceptGraph_2D_Layer->Draw("box");
//overlay the individual "bin" graphs
flagInterceptGraph_2D_Layer1->Draw("samebox");
flagInterceptGraph_2D_Layer2->Draw("samebox");
flagInterceptGraph_2D_Layer3->Draw("samebox");
flagInterceptGraph_2D_Layer4->Draw("samebox");

//set legend entries appropriately
LegGainLayer->AddEntry(flagInterceptGraph_2D_Layer1, "Good", "f");
LegGainLayer->AddEntry(flagInterceptGraph_2D_Layer2, "Low intercept, fit fails", "f");
LegGainLayer->AddEntry(flagInterceptGraph_2D_Layer3, "High intercept, fit fails", "f");
LegGainLayer->Draw("same");

//print as gif
PrintAsGif(flagInterceptLayerCanv, "flagInterceptLayer");
gStyle->SetOptStat(1);
gSystem->cd("../../../");
directoryCheck();
}
