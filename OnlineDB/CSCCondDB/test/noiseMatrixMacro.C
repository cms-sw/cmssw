void noiseMatrixMacro(){
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

//create "images" folder. 
makeDirectory("images");
gSystem->cd("images");
 //create subdirectory Gains
makeDirectory("NoiseMatrix");
gSystem->cd("NoiseMatrix");
//create subdirectory for run
makeDirectory(myFileName);
gSystem->cd(myFileName);
gSystem->cd("../../../");

//be in test before processing
directoryCheck();

flagNoiseMatrixGraphs(myFileName);  
NoiseMatrixChamberGraphs(myFileName, nDDU, nCham);
NoiseMatrixLayerGraphs(myFileName, nDDU, nCham, nLayer);

gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_1_1/src/OnlineDB/CSCCondDB/test");
gROOT->ProcessLine(".q"); 
}

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

//void GetNoiseBounds(int nChamber, int minNoise, int maxNoise){
//void GetNoiseBounds(Float_t boundsMinArray[108], Float_t boundsMaxArray[108], Float_t boundsChamMin[9], Float_t boundsChamMax[9]){
void GetNoiseBounds(Float_t boundsChamMin[9], Float_t boundsChamMax[9], int nCham){
  TCanvas *NoiseBoundCanv = new TCanvas ("NoiseBoundCanv", "NoiseBoundCanv");
  NoiseBoundCanv->Divide(2,1);
  for (int chamber=0; chamber<nCham; ++chamber){
    NoiseBoundCanv->cd(1);
    TString NoiseCut = Form ("cham==%d", chamber);
    Calibration->Draw("MaxNoise", NoiseCut);
    Float_t NoiseBoundMax =  htemp->GetXaxis()->GetXmax();
    boundsChamMax[chamber]=NoiseBoundMax;
    std::cout << NoiseBoundMax << std::endl;
    NoiseBoundCanv->cd(2);
    Calibration->Draw("MinNoise", NoiseCut);
    Float_t NoiseBoundMin = htemp->GetXaxis()->GetXmin();
    boundsChamMin[chamber]=NoiseBoundMin;
    std::cout << NoiseBoundMin << std::endl;
  }
}

void NoiseMatrixChamberGraphs(TString myFileName, int nDDU, int nCham){
gSystem->cd("images/NoiseMatrix");
gSystem->cd(myFileName);
makeDirectory("ElementChamberGraphs");
gSystem->cd("ElementChamberGraphs");

int nElements = 12;
TH2F *noiseGraph;
TCanvas *noiseCanv;
 for (int i=0; i< nDDU; ++i){
   int idArray[9];
   GetChamberIDs(idArray);
   Float_t BoundsChamMin[9];
   Float_t BoundsChamMax[9];
   //GetNoiseBounds(BoundsMinArray, BoundsMaxArray, BoundsChamMin, BoundsChamMax);
   GetNoiseBounds(BoundsChamMin, BoundsChamMax, nCham);
   for (int j=0; j<nCham; ++j){
     TString NoiseCanvName = Form("Noise_Matrix_Elements_Chamber_%d",idArray[j]);
     noiseCanv = new TCanvas(NoiseCanvName, NoiseCanvName,200,10,1200,800);
     noiseCanv->Divide(4,3);
     noiseCanv->Draw();
     for (int k=0; k<nElements; ++k){ 
       TString ElementGraphName = Form("Element_%d",k);
       noiseGraph = new TH2F(ElementGraphName,ElementGraphName,80,0,80,20,BoundsChamMin[j],BoundsChamMax[j]); 
       noiseCanv->cd(k+1);
       noiseGraph->Draw();
       TString GraphForm = Form("elem[%d]:strip", k);
       TString GraphCut = Form("cham==%d", j);
       Calibration->Project(ElementGraphName, GraphForm, GraphCut);
     }//nElements
     noiseCanv->Update();    
     PrintAsGif(noiseCanv, NoiseCanvName);
   }//nCham
 }//nDDU
 //gSystem->cd("../../../../");
gSystem->cd("../");
directoryCheck();
}//NoiseMatrixGraphs

void NoiseMatrixLayerGraphs(TString myFileName, int nDDU, int nCham, int nLayer){
  //gSystem->cd("images/NoiseMatrix");
  //gSystem->cd(myFileName);

int nElements = 12;
TH2F *noiseGraph;
TCanvas *noiseCanv;
 for (int i=0; i< nDDU; ++i){
   int idArray[9];
   GetChamberIDs(idArray);
   //   Float_t BoundsMinArray[108];
   //   Float_t BoundsMaxArray[108];
   Float_t BoundsChamMin[9];
   Float_t BoundsChamMax[9];
   //GetNoiseBounds(BoundsMinArray, BoundsMaxArray, BoundsChamMin, BoundsChamMax);
   GetNoiseBounds(BoundsChamMin, BoundsChamMax, nCham);
   for (int j=0; j<nCham; ++j){
     TString ChamDirecName = Form("Chamber_%d_Layer_Graphs", idArray[j]);
     makeDirectory(ChamDirecName); 
     gSystem->cd(ChamDirecName); 
     for (int l=0; l<nLayer; ++l){ 
       TString NoiseCanvName = Form ("Noise_Matrix_Elements_Chamber_%d_Layer_%d",idArray[j],l); 
       noiseCanv = new TCanvas(NoiseCanvName, NoiseCanvName,200,10,1200,800); 
       noiseCanv->Divide(4,3); 
       noiseCanv->Draw(); 
       for (int k=0; k<nElements; ++k){  
	 TString ElementGraphName = Form("Element_%d",k); 
	 noiseGraph = new TH2F(ElementGraphName,ElementGraphName,80,0,80,20,BoundsChamMin[j],BoundsChamMax[j]);  
	 noiseCanv->cd(k+1); 
	 noiseGraph->Draw(); 
	 TString GraphForm = Form("elem[%d]:strip", k); 
	 TString GraphCut = Form("cham==%d&&layer==%d", j, l); 
	 Calibration->Project(ElementGraphName, GraphForm, GraphCut); 
       }//nElements 
       noiseCanv->Update();    
       PrintAsGif(noiseCanv, NoiseCanvName); 
     }//nLayer
     gSystem->cd("../");
   }//nCham 
 }//nDDU 
gSystem->cd("../../../"); 
directoryCheck(); 
}//NoiseMatrixGraphs 
 
void flagNoiseMatrixGraphs(TString myFileName){  
gSystem->cd("images/NoiseMatrix");
gSystem->cd(myFileName);

TCanvas *flagNoiseMatrixCanv = new TCanvas("flagNoiseMatrixCanv", "flagNoiseMatrixCanv", 200,10,800,800);
flagNoiseMatrixCanv->SetCanvasSize(1200,800);
flagNoiseMatrixCanv->cd();
flagNoiseMatrixGraph_1D = new TH1F("flagNoiseMatrixGraph_1D", "flagNoiseMatrixGraph_1D", 5, 0, 5);
flagNoiseMatrixGraph_1D->GetYaxis()->SetTitle("Number of flag of each type");
flagNoiseMatrixGraph_1D->GetYaxis()->SetLabelSize(0.035);

TLegend *LegNoiseMatrix = new TLegend(0.7,0.5,0.89,0.7);
LegNoiseMatrix->SetHeader("Noise Matrix Flags Definitions");
LegNoiseMatrix->SetFillColor(0);
LegNoiseMatrix->SetTextSize(0);

Calibration->UseCurrentStyle(); 
Calibration->Draw("flagMatrix>>flagNoiseMatrixGraph_1D"); 

LegNoiseMatrix->AddEntry("", "1: Good");
LegNoiseMatrix->AddEntry("", "2: High Noise");
LegNoiseMatrix->AddEntry("", "3: Low Noise" );
LegNoiseMatrix->Draw("same");

flagNoiseMatrixCanv->Update();

PrintAsGif(flagNoiseMatrixCanv, "flagNoiseMatrixGraph_1D");

///////// CHAMBER flag Graph ///////////////////////
gStyle->SetOptStat(0);
TCanvas *flagNoiseMatrixChamberCanv = new TCanvas("flagNoiseMatrixChamberCanv", "flagNoiseMatrixChamberCanv", 200,10,800,800);
flagNoiseMatrixChamberCanv->SetCanvasSize(1200,800);

//create legend 
TLegend *LegNoiseMatrixChamber = new TLegend(0.85,0.8,0.98,0.98);
LegNoiseMatrixChamber->SetHeader("Noise Matrix Flags Definitions");
LegNoiseMatrixChamber->SetFillColor(0);
LegNoiseMatrixChamber->SetTextSize(0);

//final histogram for display
flagNoiseMatrixGraph_2D_Chamber = new TH2F("flagNoiseMatrixGraph_2D_Chamber", "flagNoiseMatrixGraph_2D_Chamber", 9, 0, 9, 4, 0, 4);
///dummy histo to get bin maximum
flagNoiseMatrixGraph_2D_Chamber0 = new TH2F("flagNoiseMatrixGraph_2D_Chamber0", "flagNoiseMatrixGraph_2D_Chamber0", 9, 0, 9, 4, 0, 4);
//one histo for each flag value. 
flagNoiseMatrixGraph_2D_Chamber1 = new TH2F("flagNoiseMatrixGraph_2D_Chamber1", "flagNoiseMatrixGraph_2D_Chamber1", 9, 0, 9, 4, 0, 4);
flagNoiseMatrixGraph_2D_Chamber2 = new TH2F("flagNoiseMatrixGraph_2D_Chamber2", "flagNoiseMatrixGraph_2D_Chamber2", 9, 0, 9, 4, 0, 4);
flagNoiseMatrixGraph_2D_Chamber3 = new TH2F("flagNoiseMatrixGraph_2D_Chamber3", "flagNoiseMatrixGraph_2D_Chamber3", 9, 0, 9, 4, 0, 4);
flagNoiseMatrixGraph_2D_Chamber4 = new TH2F("flagNoiseMatrixGraph_2D_Chamber4", "flagNoiseMatrixGraph_2D_Chamber4", 9, 0, 9, 4, 0, 4);

//fill completley, get bin maximum, set it for overall graph
Calibration->Project("flagNoiseMatrixGraph_2D_Chamber0", "flagMatrix:cham"); 
Double_t binMaxValCham = flagNoiseMatrixGraph_2D_Chamber0->GetMaximum();
//normalize each box appropriately, with respect to the most filled box
flagNoiseMatrixGraph_2D_Chamber->SetMaximum(binMaxValCham);

//fill each "bin"
Calibration->Project("flagNoiseMatrixGraph_2D_Chamber1","flagMatrix:cham", "flagMatrix==1", "box"); 
Calibration->Project("flagNoiseMatrixGraph_2D_Chamber2","flagMatrix:cham", "flagMatrix==2", "box"); 
Calibration->Project("flagNoiseMatrixGraph_2D_Chamber3","flagMatrix:cham", "flagMatrix==3", "box"); 

//set appropriate colors
flagNoiseMatrixGraph_2D_Chamber1->SetFillColor(1);//Black for eveything is OK
flagNoiseMatrixGraph_2D_Chamber2->SetFillColor(2);//red for VERY BAD
flagNoiseMatrixGraph_2D_Chamber3->SetFillColor(3);//Green for pretty good

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
flagNoiseMatrixGraph_2D_Chamber->GetXaxis()->SetBinLabel(chamNumPlus,chamber_id_str); //set bins to have chamber names
}

flagNoiseMatrixGraph_2D_Chamber->GetYaxis()->SetTitle("Flag");
flagNoiseMatrixGraph_2D_Chamber->GetXaxis()->SetTitle("Chamber");

flagNoiseMatrixChamberCanv->cd();  
//draw original histogram, empty
flagNoiseMatrixGraph_2D_Chamber->Draw("box");
//overlay the individual "bin" graphs
flagNoiseMatrixGraph_2D_Chamber1->Draw("samebox");
flagNoiseMatrixGraph_2D_Chamber2->Draw("samebox");
flagNoiseMatrixGraph_2D_Chamber3->Draw("samebox");
flagNoiseMatrixGraph_2D_Chamber4->Draw("samebox");

//set legend entries appropriately
LegNoiseMatrixChamber->AddEntry(flagNoiseMatrixGraph_2D_Chamber1, "Good", "f");
LegNoiseMatrixChamber->AddEntry(flagNoiseMatrixGraph_2D_Chamber2, "High Noise", "f");
LegNoiseMatrixChamber->AddEntry(flagNoiseMatrixGraph_2D_Chamber3, "Low noise", "f");
LegNoiseMatrixChamber->Draw("same");

//print as gif
PrintAsGif(flagNoiseMatrixChamberCanv, "flagNoiseMatrixChamber");
gStyle->SetOptStat(0);gSystem->cd("../../../");
directoryCheck();
}
