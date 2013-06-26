void cfebConnectMacro(){
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

int nDDU  = 1;
int nCham =  1;
int nLayer = 6;

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

gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");
makeDirectory("images");
gSystem->cd("images");
makeDirectory("CFEBConnectivity");
gSystem->cd("CFEBConnectivity");

makeDirectory(myFileName);

directoryCheck();

DifferenceGraphs(nDDU, nCham, nLayer, myFileName);
gROOT->ProcessLine(".q");
gSystem->cd("../../");
}

void GetChamberIDs(int IDArray[9], int nCham){
TCanvas *IDcanv = new TCanvas ("idGraph", "idGraph");
IDcanv->cd();
TH1F *idDummy = new TH1F("idDummy", "idDummy", 10, 220000000, 221000000);
idDummy->Draw();
for (int chamber=0; chamber<nCham; ++chamber){
  TString idCut = Form ("cham==%d", chamber);
  Calibration->Project("idDummy", "id", idCut);
  Int_t idNum = idDummy->GetMean();
  IDArray[chamber]=idNum;
}
}

DifferenceGraphs(int nDDU, int nCham, int nLayer, TString fileName) {
directoryCheck();

gSystem->cd("CFEBConnectivity");
gSystem->cd(fileName);

TH1F *diffGraph;
TCanvas *diffCanv;

for(int i=0; i<nDDU; ++i){ 
  // int idArray[9];
  // GetChamberIDs(idArray, nCham);
  for (int j=0; j<nCham; ++j){
    TString canvName = Form ("%d_DiffCanv", j);
    diffCanv = new TCanvas (canvName, canvName, 1000, 700);
    diffCanv->Divide(3,2);
      for (int k=0; k<nLayer; ++k){
	TString diffCut = Form ("cham==%d&&layer==%d",j,k);
	diffGraph = new TH1F ("diffGraph", "diffGraph", 100,800,1000);
	diffCanv->cd(k+1);
	diffGraph->Draw();	
	Calibration->Project("diffGraph", "diff", diffCut);
	diffGraph->Draw();	
      }
      diffCanv->Update();
      PrintAsGif(diffCanv, canvName);
  }
}

directoryCheck();
}

