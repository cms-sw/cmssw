void saturationMacro(){
gROOT->SetBatch();
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

std::cout << "opening: " << myFileName << std::endl; 
TFile *myFile = TFile::Open(myFilePath);
//get number of chambers per DDU, don't save graph. reatains value for NEntires. 
Calibration->Draw("cham");
int nCham = htemp->GetXaxis()->GetXmax();

//this is one big section of linux directory processing
//if this is edited, be careful! it's easy to mess up. 
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");

//system is in folder "test". check:  
directoryCheck();
//create "images" folder. 
makeDirectory("images");
gSystem->cd("images");
 //create subdirectory Gains
makeDirectory("Saturation");
gSystem->cd("Saturation");
//create subdirectory for run
makeDirectory(fileName);
gSystem->cd(fileName);
gSystem->cd("../../../");

//be in test before processing
directoryCheck(); 

myFile->Close();

gSytle->SetOptStat(0);
GetSaturation(myFileName, myFilePath); 
GetSaturationGraphs(myFileName, myFilePath, nDDU, nCham); 
gSytle->SetOptStat(1);

directoryCheck();
gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_1_1/src/OnlineDB/CSCCondDB/test");
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
TH1F *idDummy = new TH1F("idDummy", "idDummy", 10, 220000000, 221000000);
idDummy->Draw();
for (int chamber=0; chamber<9; ++chamber){
  TString idCut = Form ("cham==%d", chamber);
  Calibration->Project("idDummy", "id", idCut);
  Int_t idNum = idDummy->GetMean();
  IDArray[chamber]=idNum;
}
}

GetSaturation(TString myFileName, TString myFilePath){
gSystem->cd("images/Saturation");
gSystem->cd(fileName);
TFile *myFile = TFile::Open(myFilePath);

TH1F *Saturation;
TCanvas *SaturationCanvas =  new TCanvas("SaturationVsCharge", "SaturationVsCharge",1100,700);
Saturation = (TH1F*)myFile->Get("Saturation");  
SaturationCanvas->cd(); 
Saturation->GetXaxis()->SetTitle("Charge"); 
Saturation->GetYaxis()->SetTitle("ADC"); 
Saturation->Draw();
SaturationCanvas->Update();  
PrintAsGif(SaturationCanvas, "SaturationVsCharge");
gSystem->cd("../../../");
directoryCheck();
myFile->Close();
}

GetSaturationGraphs(TString myFileName, TString myFilePath, int nDDU, int nCham){
gSystem->cd("images/Saturation");
gSystem->cd(fileName);
makeDirectory("ChamberGraphs");
gSystem->cd("ChamberGraphs");
TFile *myFile = TFile::Open(myFilePath);
TH1F *SaturationGraph;
TCanvas *SaturationGraphsCanvas;
for (int i=0; i<nDDU; ++i){
  int idArray[9];
  GetChamberIDs(idArray);
  for (int j=0; j<nCham; ++j){
    TString SaturationGraphsCanvasName = Form("SaturationVSCharge_Chamber_%d",idArray[j]);
    SaturationGraphsCanvas = new TCanvas(SaturationGraphsCanvasName, SaturationGraphsCanvasName, 1100,700);
    SaturationGraphsCanvas->Divide(2,3);
    for (int k=1; k<6; ++k){
      TString SatGraphName = Form ("Saturation%d%d",j,k);
      SaturationGraph = (TH1F*)myFile->Get(SatGraphName);  
      SaturationGraphsCanvas->cd(k); 
      SaturationGraph->GetXaxis()->SetTitle("Charge"); 
      SaturationGraph->GetYaxis()->SetTitle("ADC"); 
      SaturationGraph->GetXaxis()->SetTitleSize(0.06);
      SaturationGraph->GetXaxis()->SetTitleOffset(0.7);
      SaturationGraph->GetYaxis()->SetTitleSize(0.06);
      SaturationGraph->GetYaxis()->SetTitleOffset(0.7);
      SaturationGraph->GetXaxis()->SetLabelSize(0.06);
      SaturationGraph->GetYaxis()->SetLabelSize(0.06);
      SaturationGraph->SetMarkerStyle(20);
      SaturationGraph->SetMarkerSize(0.8);
      SaturationGraph->Draw();
    }//"strips" loop
    SaturationGraphsCanvas->Update(); 
    PrintAsGif(SaturationGraphsCanvas, SaturationGraphsCanvasName);
  }//chamber loop  
}//DDU loop
myFile->Close();
gSystem->cd("../../../../");
directoryCheck();
}//GetSaturationGraphs()  
