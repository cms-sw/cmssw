void afebMacro(){
gROOT->SetBatch();
//set global parameters
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

int nDDU  = 1;
int nCham =  5;
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

//this is one big section of linux directory processing
//if this is edited, be careful! it's easy to mess up. 
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");

//create "images" folder. 
makeDirectory("images");
gSystem->cd("images");
 //create subdirectory Gains
makeDirectory("AFEBAnalysis");
gSystem->cd("AFEBAnalysis");
//create subdirectory for run
//TString rmline = Form("rm -r %s", myFile->GetName() ) ;
//std::cout << "executing: " << rmline << std::endl;
//gSystem->Exec(rmline);
Int_t bool_dir_bin = gSystem->mkdir(myFileName);//0 if there is no directory of this name, -1 if there is 
if (bool_dir_bin == -1){
   std::cout << "directory " << myFileName << " already exists. remove directory and try again." << std::endl;
   std::cout << "please wait while ROOT aborts and exits." << std::endl;
   gSystem->Abort();
 }

makeDirectory(myFileName);   
gSystem->cd(myFileName);  

//make graph subdirectories 
std::vector<TString> graphID_str(0);
TIter next (myFile->GetListOfKeys() );
TKey *key;
while ( (key=(TKey*)next() ) ) {
  TString name = key->GetName();
  TString nClass = key->GetClassName();
  //  std::cout << name << "  " << nClass << std::endl;
  TString chamID = name.Remove(5);
  //std::cout << chamID << std::endl;
  TString chamID_direc = chamID + "_AfebDacGraphs";
  Int_t bool_dir_check = gSystem->mkdir(chamID_direc); 
  if (bool_dir_check == 0){
    graphID_str.push_back(chamID); 
  }
  makeDirectory(chamID_direc); 
  }//while  

gSystem->cd("../../../"); 
GetAFEBDACGraphs(myFileName, myFilePath, graphID_str);
GetOtherGraphs(myFileName, myFilePath, graphID_str);

myFile->Close();
TFile *myFile2 = new TFile(file2, "read"); 
TString fileName2 = myFile2->GetName();

directoryCheck();
gSystem->cd("images/AFEBAnalysis");
makeDirectory(fileName2);
gSystem->cd(fileName2);
//make graph subdirectories 
std::vector<TString> graphID_str2(0);
TIter next (myFile2->GetListOfKeys() );
TKey *key;
while ( (key=(TKey*)next() ) ) {
  TString name = key->GetName();
  TString nClass = key->GetClassName();
  // std::cout << name << "  " << nClass << std::endl;
  TString chamID = name.Remove(5);
  std::cout << chamID << std::endl;
  TString chamID_direc = chamID + "_ConnectivityGraphs";
  Int_t bool_dir_check = gSystem->mkdir(chamID_direc); 
  if (bool_dir_check == 0 && chamID != "Layer"){
    graphID_str2.push_back(chamID); 
  }
  makeDirectory(chamID_direc); 
  }//while  
gSystem->cd("../../../"); 
GetConnectGraphs(myFile2,fileName2, graphID_str2);
directoryCheck();
}

void GetAFEBDACGraphs(TFile *myFile, TString fileName, std::vector<TString> chID){
gSystem->cd("images/AFEBAnalysis");
gSystem->cd(fileName);

for (int CHndx=0; CHndx<chID.size(); ++CHndx){
TString chamID = chID.at(CHndx);
//std::cout << "chambers: " << chamID << std::endl;
TString direcName =  chamID + "_AfebDacGraphs"; 
TH2F *AFEBDACGraph; 
TCanvas *AFEBDacCanvas =  new TCanvas("AFEBDacCanvas","AFEBDacCanvas", 1100,700); 
//AFEBDacCanvas->Divide(3,6);
gSystem->cd(direcName);
  for (int indx1=1;indx1<10; ++indx1){ 
    TString GraphName = Form("0%d_Anode_AfebDac",indx1); 
    TString GraphName = chamID+ "0" + indx1 + "_Anode_AfebDac"; 
    AFEBDACGraph = (TH2F*)myFile->Get(GraphName);  
    AFEBDACGraph->GetYaxis()->SetLimits(0,0.5);
    AFEBDACGraph->GetXaxis()->SetLimits(0,20);
    AFEBDACGraph->GetXaxis()->SetTitleSize(0.06);
    AFEBDACGraph->GetXaxis()->SetTitleOffset(0.7);
    AFEBDACGraph->GetYaxis()->SetTitleSize(0.06);
    AFEBDACGraph->GetYaxis()->SetTitleOffset(0.7);
    AFEBDacCanvas->cd(indx1); 
    AFEBDACGraph->Draw(); 
    PrintAsGif(AFEBDacCanvas, GraphName);
  } 
  for (int indx2=0;indx2<9; ++indx2){  
    TString GraphName = Form("1%d_Anode_AfebDac",indx2);  
    TString GraphName = chamID+ "1" + indx2 + "_Anode_AfebDac"; 
    AFEBDACGraph = (TH2F*)myFile->Get(GraphName); 
    AFEBDACGraph->GetYaxis()->SetLimits(0,0.5);
    AFEBDACGraph->GetXaxis()->SetLimits(0,20);
    AFEBDACGraph->GetXaxis()->SetTitleSize(0.06);
    AFEBDACGraph->GetXaxis()->SetTitleOffset(0.7);
    AFEBDACGraph->GetYaxis()->SetTitleSize(0.06);
    AFEBDACGraph->GetYaxis()->SetTitleOffset(0.7);
    AFEBDacCanvas->cd(10+indx2);  
    AFEBDACGraph->Draw(); 
    PrintAsGif(AFEBDacCanvas, GraphName);
  }  
  gSystem->cd("../");
  
}//Chamber Loop
gSystem->cd("../../../");
directoryCheck(); 
}

void GetOtherGraphs(TFile *myFile, TString fileName, std::vector<TString> chID){
gSystem->cd("images/AFEBAnalysis");
gSystem->cd(fileName);

TH2F *chi2perNDF;
TH2F *NDF;
TH2F *NoisePar;
TH2F *ThreshPar;
TH2F *FirstTime;
TH1F *ChanEff;

TCanvas *NDFCanv;
TCanvas *ParCanv;
TCanvas *EffCanv;

for (int CHndx=0; CHndx<chID.size(); ++CHndx){
TString chamID = chID.at(CHndx);
//std::cout << "chamber: " << chamID << std::endl;

//NDF's
TString direcName =  chamID + "_SpecificsGraphs";
makeDirectory(direcName);
gSystem->cd(direcName);

TString NDFGraphName = chamID + "_Anode_AfebNDF" ;
TString NDFchi2GraphName = chamID + "_Anode_AfebChi2perNDF" ;
TString NoiseParGraphName = chamID + "_Anode_AfebNoisePar" ;
TString ThreshParGraphName = chamID + "_Anode_AfebThrPar" ;
TString FirstTimeGraphName = chamID + "_Anode_First_Time" ;
TString ChanEffGraphName = chamID + "_Anode_Chan_Eff" ;

NDFCanv = new TCanvas ("NDFChamCanv", "NDFChamCanv", 1100, 700);
NDFCanv->Divide(1,2);
NDFCanv->cd(1);
NDF = (TH2F*)myFile->Get(NDFGraphName);  
NDF->Draw();
NDFCanv->cd(2);
chi2perNDF = (TH2F*)myFile->Get(NDFchi2GraphName); 
chi2perNDF->GetYaxis()->SetLimits(0,0.5);
chi2perNDF->GetXaxis()->SetLimits(0,20);
chi2perNDF->GetXaxis()->SetTitleSize(0.06);
chi2perNDF->GetXaxis()->SetTitleOffset(0.7);
chi2perNDF->GetYaxis()->SetTitleSize(0.06);
chi2perNDF->GetYaxis()->SetTitleOffset(0.7); 
chi2perNDF->Draw();

ParCanv = new TCanvas ("ParChamCanv", "ParChamCanv", 1100, 700);
ParCanv->Divide(1,2);
ParCanv->cd(1);

NoisePar = (TH2F*)myFile->Get(NoiseParGraphName);  
NoisePar->GetYaxis()->SetLimits(0,0.5);
NoisePar->GetXaxis()->SetLimits(0,20);
NoisePar->GetXaxis()->SetTitleSize(0.06);
NoisePar->GetXaxis()->SetTitleOffset(0.7);
NoisePar->GetYaxis()->SetTitleSize(0.06);
NoisePar->GetYaxis()->SetTitleOffset(0.7); 
NoisePar->Draw();

ParCanv->cd(2);
ThreshPar = (TH2F*)myFile->Get(ThreshParGraphName);  
ThreshPar->GetYaxis()->SetLimits(0,0.5);
ThreshPar->GetXaxis()->SetLimits(0,20);
ThreshPar->GetXaxis()->SetTitleSize(0.06);
ThreshPar->GetXaxis()->SetTitleOffset(0.7);
ThreshPar->GetYaxis()->SetTitleSize(0.06);
ThreshPar->GetYaxis()->SetTitleOffset(0.7); 
ThreshPar->Draw();

EffCanv = new TCanvas ("EffChamCanv", "EffChamCanv", 1100, 700);
EffCanv->Divide(1,2);
EffCanv->cd(1);
ChanEff = (TH1F*)myFile->Get(ChanEffGraphName);  
ChanEff->GetYaxis()->SetLimits(0,0.5);
ChanEff->GetXaxis()->SetLimits(0,20);
ChanEff->GetXaxis()->SetTitleSize(0.06);
ChanEff->GetXaxis()->SetTitleOffset(0.7);
ChanEff->GetYaxis()->SetTitleSize(0.06);
ChanEff->GetYaxis()->SetTitleOffset(0.7); 
ChanEff->Draw();

EffCanv->cd(2);
FirstTime = (TH2F*)myFile->Get(FirstTimeGraphName);  
FirstTime->GetYaxis()->SetLimits(0,0.5);
FirstTime->GetXaxis()->SetLimits(0,20);
FirstTime->GetXaxis()->SetTitleSize(0.06);
FirstTime->GetXaxis()->SetTitleOffset(0.7);
FirstTime->GetYaxis()->SetTitleSize(0.06);
FirstTime->GetYaxis()->SetTitleOffset(0.7); 
FirstTime->Draw();

PrintAsGif(ParCanv, "ParChamCanv");
PrintAsGif(NDFCanv, "NDFChamCanv");
PrintAsGif(EffCanv, "EffChamCanv");
gSystem->cd("../");
}//chamber loop

gSystem->cd("../../../");
directoryCheck();
}

void GetConnectGraphs(TFile *myFile2, TString fileName2, std::vector<TString> chID2){
gSystem->cd("images/AFEBAnalysis");
gSystem->cd(fileName2);

TH1F *LayNmbPulseGraph;
TCanvas *LayNmbPulseCanv;

LayNmbPulseCanv = new TCanvas ("LayNmbPulseChamCanv", "LayNmbPulseChamCanv", 1100, 700);
LayNmbPulseCanv->cd();
LayNmbPulseGraph = (TH1F*)myFile2->Get("Layer_Nmb_Pulses");  
LayNmbPulseGraph->Draw();

PrintAsGif(LayNmbPulseCanv, "LayNmbPulseChamCanv");

TH2F *AnodeFirstTime;

TH1F *AnodeEff;
TH1F *AnodeLayerNonPair;
TH1F *AnodeLayerPair;

TH1F *AnodeWireEff;
TH1F *AnodeWireNonPair;
TH1F *AnodeWirePair;

TCanvas *FirstTimeCanv;
TCanvas *LayerCanv;
TCanvas *WireCanv;

for (int CHndx=0; CHndx<chID2.size(); ++CHndx){
TString chamID = chID2.at(CHndx);
TString chamID_direc = chamID + "_ConnectivityGraphs";
gSystem->cd(chamID_direc);
FirstTimeCanv = new TCanvas ("FirstTime", "FirstTime", 1100, 700);
TString AnodeFirstTimeGraphName = chamID + "_Anode_First_Time" ;

LayerCanv = new TCanvas ("LayerEff_Crosstalk", "LayerEff_Crosstalk", 1100, 700);
LayerCanv->Divide(1,3);
TString AnodeEffGraphName = chamID + "_Anode_Eff" ; 
TString AnodeLayerNonPairGraphName = chamID + "_Anode_NonPair_Layer_Crosstalk" ; 
TString AnodeLayerPairGraphName = chamID + "_Anode_Pair_Layer_Crosstalk" ; 

WireCanv = new TCanvas ("WireEff_Crosstalk", "WireEff_Crosstalk", 1100, 700);
WireCanv->Divide(1,3);
TString AnodeWireEffGraphName = chamID + "_Anode_Wire_Eff" ; 
TString AnodeWireNonPairGraphName = chamID + "_Anode_Wire_NonPair_Crosstalk" ; 
TString AnodeWirePairGraphName = chamID + "_Anode_Wire__Pair_Crosstalk" ; 

AnodeFirstTime= (TH2F*)myFile2->Get(AnodeFirstTimeGraphName);   
FirstTimeCanv->cd();
AnodeFirstTime->GetYaxis()->SetLimits(0,0.5);
AnodeFirstTime->GetXaxis()->SetLimits(0,20);
AnodeFirstTime->GetXaxis()->SetTitleSize(0.06);
AnodeFirstTime->GetXaxis()->SetTitleOffset(0.7);
AnodeFirstTime->GetYaxis()->SetTitleSize(0.06);
AnodeFirstTime->GetYaxis()->SetTitleOffset(0.7); 
AnodeFirstTime->Draw();
PrintAsGif(FirstTimeCanv, "FirstTime");

AnodeEff = (TH1F*)myFile2->Get(AnodeEffGraphName);   
LayerCanv->cd(1);
AnodeEff->GetYaxis()->SetLimits(0,0.5);
AnodeEff->GetXaxis()->SetLimits(0,20);
AnodeEff->GetXaxis()->SetTitleSize(0.06);
AnodeEff->GetXaxis()->SetTitleOffset(0.7);
AnodeEff->GetYaxis()->SetTitleSize(0.06);
AnodeEff->GetYaxis()->SetTitleOffset(0.7); 
AnodeEff->Draw();

AnodeLayerNonPair = (TH1F*)myFile2->Get(AnodeLayerNonPairGraphName);   
LayerCanv->cd(2);
AnodeLayerNonPair->GetYaxis()->SetLimits(0,0.5);
AnodeLayerNonPair->GetXaxis()->SetLimits(0,20);
AnodeLayerNonPair->GetXaxis()->SetTitleSize(0.06);
AnodeLayerNonPair->GetXaxis()->SetTitleOffset(0.7);
AnodeLayerNonPair->GetYaxis()->SetTitleSize(0.06);
AnodeLayerNonPair->GetYaxis()->SetTitleOffset(0.7); 
AnodeLayerNonPair->Draw();

AnodeLayerPair = (TH1F*)myFile2->Get(AnodeLayerPairGraphName);   
LayerCanv->cd(3);
AnodeLayerPair->GetYaxis()->SetLimits(0,0.5);
AnodeLayerPair->GetXaxis()->SetLimits(0,20);
AnodeLayerPair->GetXaxis()->SetTitleSize(0.06);
AnodeLayerPair->GetXaxis()->SetTitleOffset(0.7);
AnodeLayerPair->GetYaxis()->SetTitleSize(0.06);
AnodeLayerPair->GetYaxis()->SetTitleOffset(0.7); 
AnodeLayerPair->Draw();
PrintAsGif(LayerCanv, "LayerEff_Crosstalk");

AnodeWireEff = (TH1F*)myFile2->Get(AnodeWireEffGraphName);    
WireCanv->cd(1);
AnodeWireEff->GetYaxis()->SetLimits(0,0.5);
AnodeWireEff->GetXaxis()->SetLimits(0,20);
AnodeWireEff->GetXaxis()->SetTitleSize(0.06);
AnodeWireEff->GetXaxis()->SetTitleOffset(0.7);
AnodeWireEff->GetYaxis()->SetTitleSize(0.06);
AnodeWireEff->GetYaxis()->SetTitleOffset(0.7); 
AnodeWireEff->Draw();

//AnodeWireNonPair = (TH1F*)myFile2->Get(AnodeWireNonPairGraphName);   
//WireCanv->cd(2);
//AnodeWireNonPair->GetYaxis()->SetLimits(0,0.5);
//AnodeWireNonPair->GetXaxis()->SetLimits(0,20);
//AnodeWireNonPair->GetXaxis()->SetTitleSize(0.06);
//AnodeWireNonPair->GetXaxis()->SetTitleOffset(0.7);
//AnodeWireNonPair->GetYaxis()->SetTitleSize(0.06);
//AnodeWireNonPair->GetYaxis()->SetTitleOffset(0.7); 
//AnodeWireNonPair->Draw();

//AnodeWirePair = (TH1F*)myFile2->Get(AnodeWirePairGraphName);  
//WireCanv->cd(3);
//AnodeWirePair->GetYaxis()->SetLimits(0,0.5);
//AnodeWirePair->GetXaxis()->SetLimits(0,20);
//AnodeWirePair->GetXaxis()->SetTitleSize(0.06);
//AnodeWirePair->GetXaxis()->SetTitleOffset(0.7);
//AnodeWirePair->GetYaxis()->SetTitleSize(0.06);
//AnodeWirePair->GetYaxis()->SetTitleOffset(0.7); 
//AnodeWirePair->Draw();
PrintAsGif(WireCanv, "WireEff_Crosstalk"); 
gSystem->cd("../");
}

gSystem->cd("../../../"); 
directoryCheck(); 
}
