void xTalkMacro(){
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
int nCham =  1; //set default to one chamber, recalculates later
 int nLayer = 6; //may as well be hardcoded 

//style-ize all canvases 
gStyle->SetCanvasColor(0); 
gStyle->SetPadColor(0); 
gStyle->SetPadBorderMode(0);
gStyle->SetCanvasBorderMode(0);
gStyle->SetFrameBorderMode(0);
gStyle->SetStatH(0.2);
gStyle->SetStatW(0.3);

//see GenFuncMacro.C
directoryCheck();

std::cout << "opening: " << myFileName << std::endl; 
TFile *myFile = TFile::Open(myFilePath);

//get number of chambers per DDU, don't save graph. reatains value for NEntires. 
TCanvas *ChamberEntryCanvas = new TCanvas("ChamberDummyCanvas", "ChamberDummyCanvas");
TH1F *ChamberEntryGraph = new TH1F ("ChamberGraph", "ChamberGraph",480,0,480);
ChamberEntryCanvas->cd();
Calibration->Draw("cham>>ChamberGraph");
int NEntries = ChamberEntryGraph->GetEntries();

//calculate the number of chambers
nCham = (NEntries/480);
std::cout << "number of chambers is: " << nCham << std::endl;
//make sure it doesn't crash if the file is empty
 if (nCham == 0){
   std::cout << "no chambers, quitting." << std::endl;
   gROOT->ProcessLine(".q");
 }
 /*
//check that file has the right number of events. if not, don't process it!!!
TCanvas *canv_pulseCheck = new TCanvas("EventCheckCanv","EventCheckCanv",1100,700); ; 
TH2F *pulseCheck = (TH2F*)myFile->Get("pulse01"); 
canv_pulseCheck->cd();
pulseCheck->Draw(); 
int NPulses = pulseCheck->GetEntries();
//this "9" might have to be changed (at some point) to the varialbe nCham, if the graphs are filled differently
int NEvents = (NPulses/9);
std::cout << "number of events is: " << NEvents << std::endl;
 if (NEvents != 320) {
   std::cout << "wrong number of events! file " << myFileName << " is not being processed." << std::endl; 
   gROOT->ProcessLine(".q"); 
 }
 */

//this is one big section of linux directory processing
//if this is edited, be careful! it's easy to mess up. 
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");
//see GenFuncMacro.C
makeDirectory("images");
 
gSystem->cd("images");
 //create subdirectory Crosstalk
makeDirectory("Crosstalk");
//open file generated from analyzer
gSystem->cd("../");

 //create folder for this run
gSystem->cd("images/Crosstalk/");
makeDirectory(myFileName);
  
gSystem->cd(myFileName);

 ///this is a bit ridiculous, but if this graph is not drawn, chamberID's cannot
 ///cannot be accessed. The graph does not need to be kept. 
TCanvas *Maxdummy = new TCanvas("Maxdummy", "Maxdummy");
Maxdummy->Divide(1,2);
Maxdummy->cd(1); 
Calibration->Draw("MaxPed");  
Maxdummy->cd(2); 
Calibration->Draw("MaxRMS");

gSystem->cd("../../../");

//create pedestal  flag graphs 
pedRMSFlagGraph(myFileName, myFilePath); 
//create all pedestal RMS
gStyle->SetOptStat(0); 
pedRMSGraphs(nDDU,nCham,nLayer, myFileName, myFilePath);
//create all pedestal MEAN graphs 
pedMeanGraphs(nDDU,nCham,nLayer, myFileName, myFilePath); 
gStyle->SetOptStat(1);
//create noise graphs 
pedNoiseFlagGraph(myFileName, myFilePath); 
//create CrossTalk pulse graphs

gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_5_1/src/OnlineDB/CSCCondDB/test");
std::cout << "now in : " << gSystem->pwd() << std::endl;

//PeakGraphs(myFileName, myFilePath, nCham);

myFile->Close();
myFilePath = Form("/tmp/csccalib/%s", myFileName); 
std::cout << "just before PulseGraphs" << std::endl;
std::cout << " fileName: " << myFileName << std::endl;
std::cout << " filePath: " << myFilePath << std::endl;
PulseGraphs(myFileName, myFilePath); 
gROOT->ProcessLine(".q"); 
} 

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
  idDummy->Draw();
}

void pedRMSGraphs(int nDDU, int nCham, int nLayer, TString myFileName, TString myFilePath){

gSystem->cd("images/Crosstalk");
gSystem->cd(myFileName);
makeDirectory("pedRMSGraphs");
gSystem->cd("pedRMSGraphs");
makeDirectory("pedRMSStrips");
makeDirectory("pedRMSLayerMean");

//create rms canvas arrays
 TObjArray CANV_PEDRMS_STRIP_ARRAY(nCham*nDDU);
 TObjArray CANV_PEDRMS_MEAN_ARRAY(nCham*nDDU);

//create rms lines for comparison to standards
//warning levels are guesses based on other work
TLine *horiz_low_pedrms = new TLine(0,0,80,0);
TLine *horiz_high_pedrms = new TLine(0,5,80,5);
TLine *vert_low_pedrms = new TLine(5, 0, 5, 200);
TLine *vert_high_pedrms = new TLine(0, 0, 0, 200);

horiz_low_pedrms->SetLineStyle(3);//dotted
horiz_high_pedrms->SetLineStyle(3);
vert_low_pedrms->SetLineStyle(3); 
vert_high_pedrms->SetLineStyle(3);

horiz_low_pedrms->SetLineColor(6);//red
horiz_low_pedrms->SetLineWidth(2);//visible but not overwhelming width
horiz_high_pedrms->SetLineColor(6);
horiz_high_pedrms->SetLineWidth(2);

vert_low_pedrms->SetLineColor(6);
vert_low_pedrms->SetLineWidth(2);
vert_high_pedrms->SetLineColor(6);
vert_high_pedrms->SetLineWidth(2);

//Ideally, cha would take nDDU*nCham as it's argument,and lay would take nDDU*nCham*nLay. 
//But ROOT doesn't like having a varialbe there, so it is created at it's largest useful value. 
TH1F *cha[480]; 
TH2F *lay[2880]; 
 //loop over all DDU's
  for(int i=0; i<nDDU; ++i){
    int idArray[9];
    GetChamberIDs(idArray);
    //loop over each chamber in a given DDU
    for (int j=0; j<nCham; ++j){
      
      gStyle->SetMarkerStyle(20);
      gStyle->SetMarkerSize(0.8);
      
      //get the chamber ID number
      Int_t chamber_id_int = idArray[j];
      //get the max RMS value for the chamber
      Float_t chamber_maxRMSval = Calibration->GetLeaf("MaxRMS")->GetValue(j);      
            
      Float_t chamber_axisMax = chamber_maxRMSval + (.1)*chamber_maxRMSval; 
      if (chamber_axisMax < 5.5){
	chamber_axisMax = 5.5;
      }

      //RMS mean Canvases
      TString CANV_PEDRMS_MEAN_NAME = Form("ChamberID_%d_RMS", chamber_id_int);
      TCanvas* canv_rms1 ;
      canv_rms1 =  new TCanvas(CANV_PEDRMS_MEAN_NAME,CANV_PEDRMS_MEAN_NAME, 200, 10, 1200, 800); 
      canv_rms1->SetCanvasSize(1200,800);
      
      //add to this array, for out-of-scope acess
      CANV_PEDRMS_MEAN_ARRAY.Add(canv_rms1);

      //RMS strip Canvases
      TString CANV_PEDRMS_STRIP_NAME = Form("ChamberID_%d_RMS_per_Layer", chamber_id_int);

      TCanvas* canv_rms2;
      canv_rms2 = new TCanvas(CANV_PEDRMS_STRIP_NAME,CANV_PEDRMS_STRIP_NAME, 200, 10, 1200, 800);
      canv_rms2->SetCanvasSize(1200,800);
      canv_rms2->Divide(2,3);
      CANV_PEDRMS_STRIP_ARRAY.Add(canv_rms2);

      //create, fill and draw overall RMS graphs
             
      int chamber_access_int =( (nCham*i)+ j);
      
      //this access one of the (*TH1F's) initialized outside the DDU loop
      TString hist_name= Form("cha[%d]", chamber_access_int);
      cha[j] = new TH1F(hist_name, "rms mean plot", 100, -0.5, chamber_axisMax);

      TCanvas *thisCanv_mean = CANV_PEDRMS_MEAN_ARRAY[chamber_access_int]; 
      thisCanv_mean->cd();

      TString xAxisTitle = Form("Mean RMS for Chamber %d",chamber_id_int);
      cha[j]->GetXaxis()->SetTitle(xAxisTitle);

      const char val[7];
      //cuts based on each chamber
      sprintf(val, "cham==%d", j);
      const char *val_ = val;

      Calibration->UseCurrentStyle();
      //this draws an empty graph called cha[j] with the chosen options
      //into the proper canvas, thisCanv_mean
      cha[j]->Draw();
      //this pipes histo pedRMS from Tree Calibration into graph cha[j] with cut val_
      Calibration->Project(hist_name, "pedRMS", val_);
      
      //retrieve the max, so the scale of the graph and vertical lines can be redrawn appropriately
      Double_t yMaxVal = cha[j]->GetMaximum();
      Double_t yMaxRan = yMaxVal + (yMaxVal*.1);

      //reset the line range
      vert_low_pedrms->SetY2(yMaxRan);
      vert_high_pedrms->SetY2(yMaxRan);

      //reset the axis range
      cha[j]->SetAxisRange(0,yMaxRan,"Y");
      vert_low_pedrms->Draw();  
      vert_high_pedrms->Draw();
    
      thisCanv_mean->Update();
      gSystem->cd("pedRMSLayerMean");
      TString canvName = CANV_PEDRMS_MEAN_ARRAY[chamber_access_int]->GetName(); 
      PrintAsGif(thisCanv_mean, canvName);  
      gSystem->cd("../");
      
      int NStrips = 0;
      int xStripMax = 1;
      //dummy graph to retrieve number of strips, to set graph accordingly
      TCanvas *StripEntryCanvas = new TCanvas("StripDummyCanvas", "StripDummyCanvas");
      TH1F *StripEntryDummy = new TH1F("StripEntryDummy", "StripEntryDummy",10,0,100);
      TString chamVar = Form("cham==%d",j);
      StripEntryCanvas->cd();
      Calibration->Draw("strip>>StripEntryDummy", chamVar);
      //number of strips in a given chamber. should be 480 for 5 CFEBs or 384 for 4 CFEBs
      NStrips= StripEntryDummy->GetEntries();
      
      if (NStrips==480){
	xStripMax=80;
      }if (NStrips==384){
	xStripMax=64;
     }
     
      for (int k=0; k<nLayer; ++k){
	int layer_access_int = ( (nCham*nLayer*i) + (nLayer*j) + k );
	//RMS: create, fill and draw strip graphs
	TString hist_name_= Form("lay[%d]",layer_access_int);
	
	lay[layer_access_int] = new TH2F(hist_name_,"RMS strip plot", xStripMax, 0, xStripMax, 10, -0.5, chamber_axisMax);
	//	lay[layer_access_int] = new TH2F(hist_name_,"RMS strip plot", xStripMax, 0, xStripMax, 10, 1.5, 3);
	
	TCanvas *thisCanv_rms_strip = CANV_PEDRMS_STRIP_ARRAY[chamber_access_int];
	thisCanv_rms_strip -> cd(k+1);
	
	lay[layer_access_int]->SetMarkerStyle(20);
	TString new_title2 = Form("Layer_%d", k+1);
	lay[layer_access_int]->SetTitle(new_title2);

	TString newTitle = Form("strip RMS in chamber %d, per layer",chamber_id_int,k);
	//TString newTitle = Form("strip RMS in chamber %d, per layer",j+1,k);
    	thisCanv_rms_strip->SetTitle(newTitle);
	lay[layer_access_int]->GetXaxis()->SetTitle("Strip");
	lay[layer_access_int]->GetYaxis()->SetTitle("RMS");

	lay[layer_access_int]->GetXaxis()->SetTitleSize(0.070);
	lay[layer_access_int]->GetXaxis()->SetTitleOffset(0.5);
	lay[layer_access_int]->GetYaxis()->SetTitleSize(0.070);
	lay[layer_access_int]->GetXaxis()->SetTitleOffset(0.5);

	lay[layer_access_int]->GetXaxis()->SetLabelSize(0.07);
	lay[layer_access_int]->GetYaxis()->SetLabelSize(0.07);

  	horiz_low_pedrms->SetX2(xStripMax);
	horiz_high_pedrms->SetX2(xStripMax);

	TString graph_name_TH2F = Form("lay[%d]" ,layer_access_int);
	
	const char valu[17];
	sprintf(valu, "cham==%d&&layer==%d",j,k);
	const char *valu_ = valu;

	Calibration->UseCurrentStyle();
	lay[layer_access_int]->Draw();
	Calibration->Project(hist_name_, "pedRMS:strip", valu_);
	horiz_low_pedrms->Draw(); 
	horiz_high_pedrms->Draw();

	thisCanv_rms_strip->Update();

      }//layer loop 
      //      process at end of looping
      gSystem->cd("pedRMSStrips");
      TString canvName = CANV_PEDRMS_STRIP_ARRAY[chamber_access_int]->GetName();
      PrintAsGif(thisCanv_rms_strip, canvName); 
      gSystem->cd("../");    
 
    }//chamber loop
  }//DDU loop
  gSystem->cd("../../../../");
  directoryCheck();
}//pedRMSGraph()

void pedMeanGraphs(int nDDU, int nCham, int nLayer, TString myFileName, TString myFilePath){
 gSystem->cd("images/Crosstalk");
 gSystem->cd(myFileName);
 makeDirectory("pedMeanGraphs");
 gSystem->cd("pedMeanGraphs");
 makeDirectory("pedMeanStrips");
 makeDirectory("pedMeanLayerMean");

//create mean canvas arrays
 TObjArray CANV_PEDMEAN_STRIP_ARRAY(nCham*nDDU);     
 TObjArray CANV_PEDMEAN_MEAN_ARRAY(nCham*nDDU); 

//create mean lines 
TLine *horiz_low_pedmean = new TLine(0,400,80,400);  
TLine *horiz_high_pedmean = new TLine(0,800,80,800);  
TLine *vert_low_pedmean = new TLine(400, 0, 400, 200);  
TLine *vert_high_pedmean = new TLine(800, 0, 800, 200);  
 
horiz_low_pedmean->SetLineStyle(3); 
horiz_high_pedmean->SetLineStyle(3);  
vert_low_pedmean->SetLineStyle(3);   
vert_high_pedmean->SetLineStyle(3); 

horiz_low_pedmean->SetLineColor(6); 
horiz_low_pedmean->SetLineWidth(2); 
horiz_high_pedmean->SetLineColor(6);
horiz_high_pedmean->SetLineWidth(2);

vert_low_pedmean->SetLineColor(6); 
vert_low_pedmean->SetLineWidth(2); 
vert_high_pedmean->SetLineColor(6); 
vert_high_pedmean->SetLineWidth(2); 

 TH1F *chamber[480]; 
 TH2F *layer[2880];
  for(int i=0; i<nDDU; ++i){
    int idArray[9];
    GetChamberIDs(idArray);
    for (int j=0; j<nCham; ++j){

      gStyle->SetMarkerStyle(20);
      gStyle->SetMarkerSize(1.0);

      Int_t chamber_id_int = idArray[j];
      //get the max RMS value for the chamber
      Float_t chamber_maxPedval = Calibration->GetLeaf("MaxPed")->GetValue(j);      
      Float_t chamber_axisMax = chamber_maxPedval + (.1)*chamber_maxPedval; 

      if (chamber_axisMax<900.0){
	chamber_axisMax = 900.0;
	  }

      //Mean mean Cavases
      TString CANV_PEDMEAN_MEAN_NAME = Form("ChamberID_%d_ped_Mean",chamber_id_int);
      TCanvas* canv_mean1;
      canv_mean1 =  new TCanvas(CANV_PEDMEAN_MEAN_NAME,CANV_PEDMEAN_MEAN_NAME, 200, 10, 1200, 800);
      CANV_PEDMEAN_MEAN_ARRAY.Add(canv_mean1);
      //Mean strip Canvases
      TString CANV_PEDMEAN_STRIP_NAME = Form("ChamberID_%d_ped_Mean_per_Layer",chamber_id_int);
      TCanvas* canv_mean2;
      canv_mean2= new TCanvas(CANV_PEDMEAN_STRIP_NAME,CANV_PEDMEAN_STRIP_NAME, 200, 10, 1200, 800);
      canv_mean2->SetCanvasSize(1200,800);
      canv_mean2->Divide(2,3);
      CANV_PEDMEAN_STRIP_ARRAY.Add(canv_mean2);
      
      //create, fill and draw overall MEAN graphs
      int chamber_access_int =( (nCham*i)+ j);
      TString hist_name= Form("chamber[%d]",chamber_access_int);
      chamber[j] = new TH1F(hist_name, "mean mean plot", 100, 300, chamber_axisMax);

      TCanvas *thisCanv_meanMean = CANV_PEDMEAN_MEAN_ARRAY[chamber_access_int];
      thisCanv_meanMean->SetCanvasSize(1200,800);
      thisCanv_meanMean->cd();
        
      const char value[7];
      sprintf(value, "cham==%d", j);
      const char *value_ = value;
 
      TString new_title1_mean = Form("Mean_ChamberID_%d_DDU_%d", chamber_id_int, i+1);
      chamber[j]->SetTitle(new_title1_mean);

      TString new_title1_mean = Form("ChamberID_%d_DDU_%d_Mean", chamber_id_int, i+1);
      TString xAxisTitle = Form("Mean Pedestal for Chamber %d",chamber_id_int);

      chamber[j]->SetTitle(new_title1_mean);
      chamber[j]->GetXaxis()->SetTitle(xAxisTitle);

      Calibration->UseCurrentStyle();
      chamber[j]->Draw();
      Calibration->Project(hist_name, "pedMean", value_);

      Double_t yMaxVal = chamber[j]->GetMaximum();
      Double_t yMaxRan = yMaxVal + (yMaxVal*.1);

      vert_low_pedmean->SetY2(yMaxRan);
      vert_high_pedmean->SetY2(yMaxRan);

      chamber[j]->SetAxisRange(0,yMaxRan,"Y");
      
      vert_low_pedmean->Draw(); 
      vert_high_pedmean->Draw(); 

      thisCanv_meanMean->Update();

      gSystem->cd("pedMeanLayerMean");
      TString canvName = CANV_PEDMEAN_MEAN_ARRAY[chamber_access_int]->GetName();
      PrintAsGif(thisCanv_meanMean, canvName); 
      gSystem->cd("../");

      int NStrips = 0;
      int xStripMax = 1;
      //dummy graph to retrieve number of strips, to set graph accordingly
      TCanvas *StripEntryCanvas = new TCanvas("StripDummyCanvas", "StripDummyCanvas");
      TH1F *StripEntryDummy = new TH1F("StripEntryDummy", "StripEntryDummy",10,0,100);
      TString chamVar = Form("cham==%d",j);
      StripEntryCanvas->cd();
      Calibration->Draw("strip>>StripEntryDummy", chamVar);
      //number of strips in a given chamber. should be 480 for 5 CFEBs or 384 for 4 CFEBs
      NStrips=StripEntryDummy->GetEntries();
      
      if (NStrips==480){
	xStripMax=80;
      }if (NStrips==384){
	xStripMax=64;
     }

      for (int k=0; k<nLayer; ++k){
    	gStyle->SetMarkerStyle(20);
	gStyle->SetMarkerSize(1.0); 
	int layer_access_int = ( (nCham*nLayer*i) + (nLayer*j) + k );
	//MEAN: create, fill and draw strip graphs
	TString hist_name_mean= Form("layer[%d]",layer_access_int);
	layer[layer_access_int] = new TH2F(hist_name_mean,"MEAN strip plot", xStripMax, 0, xStripMax, 100, 300, chamber_axisMax);
      
	TCanvas *thisCanv_mean_strip=CANV_PEDMEAN_STRIP_ARRAY[chamber_access_int];
	thisCanv_mean_strip-> cd(k+1);

	layer[layer_access_int]->GetXaxis()->SetTitle("Strip");
	layer[layer_access_int]->GetXaxis()->SetTitleSize(0.070);
	layer[layer_access_int]->GetXaxis()->SetTitleOffset(0.5);

	layer[layer_access_int]->GetYaxis()->SetTitle("Pedestal Mean");
	layer[layer_access_int]->GetYaxis()->SetTitleSize(0.070);
	layer[layer_access_int]->GetYaxis()->SetTitleOffset(0.5);
	
	const char value2[17];
	sprintf(value2, "cham==%d&&layer==%d",j,k);
	const char *value2_ = value2;

	layer[layer_access_int]->SetMarkerStyle(20);

	horiz_low_pedmean->SetX2(xStripMax);
	horiz_high_pedmean->SetX2(xStripMax);

       	Calibration->UseCurrentStyle();
	layer[layer_access_int]->Draw();
	Calibration->Project(hist_name_mean, "pedMean:strip", value2_);
	horiz_low_pedmean->Draw("same"); 
	horiz_high_pedmean->Draw("same"); 
	
      }//layer

      gSystem->cd("pedMeanStrips");
      TString canvName = CANV_PEDMEAN_STRIP_ARRAY[chamber_access_int]->GetName();
      PrintAsGif(thisCanv_mean_strip, canvName); 
      gSystem->cd("../");
    }//chamber
  }//DDU
  
  gSystem->cd("../../../../");
  directoryCheck();
}//pedMeanGraphs()

void pedRMSFlagGraph(TString myFileName, TString myFilePath){ 
  gStyle->SetOptStat(0); 
  gSystem->cd("images/Crosstalk");
  gSystem->cd(myFileName);

  //1D Graph, straight from Calibration Tree
  TCanvas *flagRMSCanv = new TCanvas("flagRMSCanv", "flagRMSCanv", 200,10,800,800);
  flagRMSCanv->SetCanvasSize(1200,800);
  flagRMSGraph_1D = new TH1F("flagRMSGraph_1D", "flagRMSGraph_1D", 5, 0, 5);
  flagRMSGraph_1D->GetYaxis()->SetTitle("Number");
  flagRMSGraph_1D->GetYaxis()->SetLabelSize(0.035);
  flagRMSGraph_1D->GetXaxis()->SetTitle("Flag");
  flagRMSGraph_1D->GetXaxis()->SetLabelSize(0.035);

  TLegend *LegRMS = new TLegend(0.85,0.8,0.98,0.98);
  LegRMS->SetHeader("RMS Flags Definitions");
  LegRMS->SetFillColor(0);
  LegRMS->SetTextSize(0);

  Calibration->UseCurrentStyle(); 
  Calibration->Draw("flagRMS>>flagRMSGraph_1D");

  LegRMS->AddEntry("", "1: Good");
  LegRMS->AddEntry("", "2: High CFEB Noise");
  LegRMS->AddEntry("", "3: Too low Noise");
  LegRMS->AddEntry("", "4: Too high Noise");
  LegRMS->Draw("same");

  flagRMSCanv->Update();

  //print as gif

  PrintAsGif(flagRMSCanv, "flagRMS");

  /////////////////////////////////       2D GRAPHS            ////////////////////////////////
  //no statistics shown
  gStyle->SetOptStat(0);

  ////////// Strip Flag Graphs///////////////
  //canvas
  TCanvas *flagRMSStripCanv = new TCanvas("flagRMSStripCanv", "flagRMSStripCanv", 200,10,800,800);
  flagRMSStripCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegRMSStrip = new TLegend(0.85,0.8,0.98,0.98);
  LegRMSStrip->SetHeader("RMS Flags Definitions");
  LegRMSStrip->SetFillColor(0);
  LegRMSStrip->SetTextSize(0);

  //final histogram for display
  flagRMSGraph_2D_Strip = new TH2F("flagRMSGraph_2D_Strip", "flagRMSGraph_2D_Strip", 80, 0, 80, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagRMSGraph_2D_Strip0 = new TH2F("flagRMSGraph_2D_Strip0", "flagRMSGraph_2D_Strip0", 80, 0, 80, 6, 0, 6);
  //one histo for each flag value. 
  flagRMSGraph_2D_Strip1 = new TH2F("flagRMSGraph_2D_Strip1", "flagRMSGraph_2D_Strip1", 80, 0, 80, 6, 0, 6);
  flagRMSGraph_2D_Strip2 = new TH2F("flagRMSGraph_2D_Strip2", "flagRMSGraph_2D_Strip2", 80, 0, 80, 6, 0, 6);
  flagRMSGraph_2D_Strip3 = new TH2F("flagRMSGraph_2D_Strip3", "flagRMSGraph_2D_Strip3", 80, 0, 80, 6, 0, 6);
  flagRMSGraph_2D_Strip4 = new TH2F("flagRMSGraph_2D_Strip4", "flagRMSGraph_2D_Strip4", 80, 0, 80, 6, 0, 6);
  flagRMSGraph_2D_Strip5 = new TH2F("flagRMSGraph_2D_Strip5", "flagRMSGraph_2D_Strip5", 80, 0, 80, 6, 0, 6);
  flagRMSGraph_2D_Strip6 = new TH2F("flagRMSGraph_2D_Strip6", "flagRMSGraph_2D_Strip6", 80, 0, 80, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagRMSGraph_2D_Strip0","flagRMS:strip"); 
  Double_t binMaxValStrip = flagRMSGraph_2D_Strip0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagRMSGraph_2D_Strip->SetMaximum(binMaxValStrip);

  //fill each "bin"
  Calibration->Project("flagRMSGraph_2D_Strip1","flagRMS:strip", "flagRMS==1","box"); 
  Calibration->Project("flagRMSGraph_2D_Strip2","flagRMS:strip", "flagRMS==2","box"); 
  Calibration->Project("flagRMSGraph_2D_Strip3","flagRMS:strip", "flagRMS==3","box"); 
  Calibration->Project("flagRMSGraph_2D_Strip4","flagRMS:strip", "flagRMS==4","box"); 
  Calibration->Project("flagRMSGraph_2D_Strip5","flagRMS:strip", "flagRMS==5","box"); 

  //set appropriate colors
  flagRMSGraph_2D_Strip1->SetFillColor(1);//Black for eveything is OK
  flagRMSGraph_2D_Strip2->SetFillColor(2);//red for VERY BAD
  flagRMSGraph_2D_Strip3->SetFillColor(3);//Green for pretty good
  flagRMSGraph_2D_Strip4->SetFillColor(4);//blue for less good

  flagRMSStripCanv->cd(); 
  flagRMSGraph_2D_Strip->GetYaxis()->SetTitle("Flag");
  flagRMSGraph_2D_Strip->GetXaxis()->SetTitle("Strip, all Chambers");
  
  //draw original histogram, empty
  flagRMSGraph_2D_Strip->Draw("box");
  //overlay the individual "bin" graphs
  flagRMSGraph_2D_Strip1->Draw("samebox");
  flagRMSGraph_2D_Strip2->Draw("samebox");
  flagRMSGraph_2D_Strip3->Draw("samebox");
  flagRMSGraph_2D_Strip4->Draw("samebox");
  flagRMSGraph_2D_Strip5->Draw("samebox");
  flagRMSGraph_2D_Strip6->Draw("samebox");

  //set legend entries appropriately
  LegRMSStrip->AddEntry(flagRMSGraph_2D_Strip1, "Good", "f");
  LegRMSStrip->AddEntry(flagRMSGraph_2D_Strip2, "High CFEB Noise", "f");
  LegRMSStrip->AddEntry(flagRMSGraph_2D_Strip3, "Too low Noise", "f");
  LegRMSStrip->AddEntry(flagRMSGraph_2D_Strip4, "Too high Noise", "f");
  LegRMSStrip->Draw("same");
  //print as gif
  PrintAsGif(flagRMSStripCanv, "flagRMSStrip");

  ///////// CHAMBER flag Graph ///////////////////////
  TCanvas *flagRMSChamberCanv = new TCanvas("flagRMSChamberCanv", "flagRMSChamberCanv", 200,10,800,800);
  flagRMSChamberCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegRMSChamber = new TLegend(0.85,0.8,0.98,0.98);
  LegRMSChamber->SetHeader("RMS Flags Definitions");
  LegRMSChamber->SetFillColor(0);
  LegRMSChamber->SetTextSize(0);

  //final histogram for display
  flagRMSGraph_2D_Chamber = new TH2F("flagRMSGraph_2D_Chamber", "flagRMSGraph_2D_Chamber", 9, 0, 9, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagRMSGraph_2D_Chamber0 = new TH2F("flagRMSGraph_2D_Chamber0", "flagRMSGraph_2D_Chamber0", 9, 0, 9, 6, 0, 6);
  //one histo for each flag value. 
  flagRMSGraph_2D_Chamber1 = new TH2F("flagRMSGraph_2D_Chamber1", "flagRMSGraph_2D_Chamber1", 9, 0, 9, 6, 0, 6);
  flagRMSGraph_2D_Chamber2 = new TH2F("flagRMSGraph_2D_Chamber2", "flagRMSGraph_2D_Chamber2", 9, 0, 9, 6, 0, 6);
  flagRMSGraph_2D_Chamber3 = new TH2F("flagRMSGraph_2D_Chamber3", "flagRMSGraph_2D_Chamber3", 9, 0, 9, 6, 0, 6);
  flagRMSGraph_2D_Chamber4 = new TH2F("flagRMSGraph_2D_Chamber4", "flagRMSGraph_2D_Chamber4", 9, 0, 9, 6, 0, 6);
  flagRMSGraph_2D_Chamber5 = new TH2F("flagRMSGraph_2D_Chamber5", "flagRMSGraph_2D_Chamber5", 9, 0, 9, 6, 0, 6);
  flagRMSGraph_2D_Chamber6 = new TH2F("flagRMSGraph_2D_Chamber6", "flagRMSGraph_2D_Chamber6", 9, 0, 9, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagRMSGraph_2D_Chamber0","flagRMS:cham"); 
  Double_t binMaxValCham = flagRMSGraph_2D_Chamber0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagRMSGraph_2D_Chamber->SetMaximum(binMaxValCham);

  //fill each "bin"
  Calibration->Project("flagRMSGraph_2D_Chamber1","flagRMS:cham", "flagRMS==1", "box"); 
  Calibration->Project("flagRMSGraph_2D_Chamber2","flagRMS:cham", "flagRMS==2", "box"); 
  Calibration->Project("flagRMSGraph_2D_Chamber3","flagRMS:cham", "flagRMS==3", "box"); 
  Calibration->Project("flagRMSGraph_2D_Chamber4","flagRMS:cham", "flagRMS==4", "box"); 
  Calibration->Project("flagRMSGraph_2D_Chamber5","flagRMS:cham", "flagRMS==5", "box"); 

  //set appropriate colors
  flagRMSGraph_2D_Chamber1->SetFillColor(1);//Black for eveything is OK
  flagRMSGraph_2D_Chamber2->SetFillColor(2);//red for VERY BAD
  flagRMSGraph_2D_Chamber3->SetFillColor(3);//Green for pretty good
  flagRMSGraph_2D_Chamber4->SetFillColor(4);//blue for less good

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
    flagRMSGraph_2D_Chamber->GetXaxis()->SetBinLabel(chamNumPlus,chamber_id_str); //set bins to have chamber names
  }
  
  flagRMSChamberCanv->cd(); 
  flagRMSGraph_2D_Chamber->GetYaxis()->SetTitle("Flag");
  flagRMSGraph_2D_Chamber->GetXaxis()->SetTitle("Chamber");

  //draw original histogram, empty
  flagRMSGraph_2D_Chamber->Draw("box");
  //overlay the individual "bin" graphs
  flagRMSGraph_2D_Chamber1->Draw("samebox");
  flagRMSGraph_2D_Chamber2->Draw("samebox");
  flagRMSGraph_2D_Chamber3->Draw("samebox");
  flagRMSGraph_2D_Chamber4->Draw("samebox");
  flagRMSGraph_2D_Chamber5->Draw("samebox");
  flagRMSGraph_2D_Chamber6->Draw("samebox");

  //set legend entries appropriately
  LegRMSChamber->AddEntry(flagRMSGraph_2D_Chamber1, "Good", "f");
  LegRMSChamber->AddEntry(flagRMSGraph_2D_Chamber2, "High CFEB Noise", "f");
  LegRMSChamber->AddEntry(flagRMSGraph_2D_Chamber3, "Too low Noise", "f");
  LegRMSChamber->AddEntry(flagRMSGraph_2D_Chamber4, "Too high Noise", "f");
  LegRMSChamber->Draw("same");

  //print as gif
  PrintAsGif(flagRMSChamberCanv, "flagRMSChamber");

/////////////////////// Layer Graphs //////////////

  TCanvas *flagRMSLayerCanv = new TCanvas("flagRMSLayerCanv", "flagRMSLayerCanv", 200,10,800,800);
  flagRMSLayerCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegRMSLayer = new TLegend(0.85,0.8,0.98,0.98);
  LegRMSLayer->SetHeader("RMS Flags Definitions");
  LegRMSLayer->SetFillColor(0);
  LegRMSLayer->SetTextSize(0);

  //final histogram for display
  flagRMSGraph_2D_Layer = new TH2F("flagRMSGraph_2D_Layer", "flagRMSGraph_2D_Layer", 6, 0, 6, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagRMSGraph_2D_Layer0 = new TH2F("flagRMSGraph_2D_Layer0", "flagRMSGraph_2D_Layer0", 6, 0, 6, 6, 0, 6);
  //one histo for each flag value. 
  flagRMSGraph_2D_Layer1 = new TH2F("flagRMSGraph_2D_Layer1", "flagRMSGraph_2D_Layer1", 6, 0, 6, 6, 0, 6);
  flagRMSGraph_2D_Layer2 = new TH2F("flagRMSGraph_2D_Layer2", "flagRMSGraph_2D_Layer2", 6, 0, 6, 6, 0, 6);
  flagRMSGraph_2D_Layer3 = new TH2F("flagRMSGraph_2D_Layer3", "flagRMSGraph_2D_Layer3", 6, 0, 6, 6, 0, 6);
  flagRMSGraph_2D_Layer4 = new TH2F("flagRMSGraph_2D_Layer4", "flagRMSGraph_2D_Layer4", 6, 0, 6, 6, 0, 6);
  flagRMSGraph_2D_Layer5 = new TH2F("flagRMSGraph_2D_Layer5", "flagRMSGraph_2D_Layer5", 6, 0, 6, 6, 0, 6);
  flagRMSGraph_2D_Layer6 = new TH2F("flagRMSGraph_2D_Layer6", "flagRMSGraph_2D_Layer6", 6, 0, 6, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagRMSGraph_2D_Layer0","flagRMS:layer"); 
  Double_t binMaxValLayer = flagRMSGraph_2D_Layer0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagRMSGraph_2D_Layer->SetMaximum(binMaxValLayer);

  //fill each "bin"
  Calibration->Project("flagRMSGraph_2D_Layer1","flagRMS:layer", "flagRMS==1", "box"); 
  Calibration->Project("flagRMSGraph_2D_Layer2","flagRMS:layer", "flagRMS==2", "box"); 
  Calibration->Project("flagRMSGraph_2D_Layer3","flagRMS:layer", "flagRMS==3", "box"); 
  Calibration->Project("flagRMSGraph_2D_Layer4","flagRMS:layer", "flagRMS==4", "box"); 
  Calibration->Project("flagRMSGraph_2D_Layer5","flagRMS:layer", "flagRMS==5", "box"); 

  //set appropriate colors
  flagRMSGraph_2D_Layer1->SetFillColor(1);//Black for eveything is OK
  flagRMSGraph_2D_Layer2->SetFillColor(2);//red for VERY BAD
  flagRMSGraph_2D_Layer3->SetFillColor(3);//Green for pretty good
  flagRMSGraph_2D_Layer4->SetFillColor(4);//blue for less good
  

  flagRMSLayerCanv->cd(); 
  flagRMSGraph_2D_Layer->GetYaxis()->SetTitle("Flag");
  flagRMSGraph_2D_Layer->GetXaxis()->SetTitle("Layer, all Chambers");
  
  //draw original histogram, empty
  flagRMSGraph_2D_Layer->Draw("box");
  //overlay the individual "bin" graphs
  flagRMSGraph_2D_Layer1->Draw("samebox");
  flagRMSGraph_2D_Layer2->Draw("samebox");
  flagRMSGraph_2D_Layer3->Draw("samebox");
  flagRMSGraph_2D_Layer4->Draw("samebox");
  flagRMSGraph_2D_Layer5->Draw("samebox");
  flagRMSGraph_2D_Layer6->Draw("samebox");

  //set legend entries appropriately
  LegRMSLayer->AddEntry(flagRMSGraph_2D_Layer1, "Good", "f");
  LegRMSLayer->AddEntry(flagRMSGraph_2D_Layer2, "High CFEB Noise", "f");
  LegRMSLayer->AddEntry(flagRMSGraph_2D_Layer3, "Too low Noise", "f");
  LegRMSLayer->AddEntry(flagRMSGraph_2D_Layer4, "Too high Noise", "f");
  LegRMSLayer->Draw("same");
  //print as gif
  PrintAsGif(flagRMSLayerCanv, "flagRMSLayer");

  gSystem->cd("../../../");
  gStyle->SetOptStat(1); 
} 

void pedNoiseFlagGraph(TString myFileName, TString myFilePath){
  //  TFile *myFile = new TFile(myFilePath, "read");
  gStyle->SetOptStat(0); 
  gSystem->cd("images/Crosstalk");
  gSystem->cd(myFileName);

    //1D Graph, straight from Calibration Tree
  TCanvas *flagMeanCanv = new TCanvas("flagMeanCanv", "flagMeanCanv", 200,10,800,800);
  flagMeanCanv->SetCanvasSize(1200,800);
  flagMeanGraph_1D = new TH1F("flagMeanGraph_1D", "flagMeanGraph_1D", 5, 0, 5);
  flagMeanGraph_1D->GetYaxis()->SetTitle("Number");
  flagMeanGraph_1D->GetYaxis()->SetLabelSize(0.035);
  flagMeanGraph_1D->GetXaxis()->SetTitle("Flag");
  flagMeanGraph_1D->GetXaxis()->SetLabelSize(0.035);

  TLegend *LegMean = new TLegend(0.85,0.8,0.98,0.98);
  LegMean->SetHeader("Mean Flags Definitions");
  LegMean->SetFillColor(0);
  LegMean->SetTextSize(0);

  Calibration->UseCurrentStyle(); 
  Calibration->Draw("flagNoise>>flagMeanGraph_1D");

  LegMean->AddEntry(flagMeanGraph_1D, "1: Normal Pedestal", "f");
  LegMean->AddEntry(flagMeanGraph_1D, "2: Low Failure", "f");
  LegMean->AddEntry(flagMeanGraph_1D, "3: Low Warning", "f");
  LegMean->AddEntry(flagMeanGraph_1D, "4: High Warning", "f");
  LegMean->AddEntry(flagMeanGraph_1D, "5: High Failure", "f");

  LegMean->Draw("same");

  flagMeanCanv->Update();

  //print as gif
  PrintAsGif(flagMeanCanv, "flagMean");

  /////////////////////////////////       2D GRAPHS            ////////////////////////////////
  //no statistics shown
  gStyle->SetOptStat(0);

  ////////// Strip Flag Graphs///////////////
  //canvas
  TCanvas *flagMeanStripCanv = new TCanvas("flagMeanStripCanv", "flagMeanStripCanv", 200,10,800,800);
  flagMeanStripCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegMeanStrip = new TLegend(0.85,0.8,0.98,0.98);
  LegMeanStrip->SetHeader("Mean Flags Definitions");
  LegMeanStrip->SetFillColor(0);
  LegMeanStrip->SetTextSize(0);

  //final histogram for display
  flagMeanGraph_2D_Strip = new TH2F("flagMeanGraph_2D_Strip", "flagMeanGraph_2D_Strip", 80, 0, 80, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagMeanGraph_2D_Strip0 = new TH2F("flagMeanGraph_2D_Strip0", "flagMeanGraph_2D_Strip0", 80, 0, 80, 6, 0, 6);
  //one histo for each flag value. 
  flagMeanGraph_2D_Strip1 = new TH2F("flagMeanGraph_2D_Strip1", "flagMeanGraph_2D_Strip1", 80, 0, 80, 6, 0, 6);
  flagMeanGraph_2D_Strip2 = new TH2F("flagMeanGraph_2D_Strip2", "flagMeanGraph_2D_Strip2", 80, 0, 80, 6, 0, 6);
  flagMeanGraph_2D_Strip3 = new TH2F("flagMeanGraph_2D_Strip3", "flagMeanGraph_2D_Strip3", 80, 0, 80, 6, 0, 6);
  flagMeanGraph_2D_Strip4 = new TH2F("flagMeanGraph_2D_Strip4", "flagMeanGraph_2D_Strip4", 80, 0, 80, 6, 0, 6);
  flagMeanGraph_2D_Strip5 = new TH2F("flagMeanGraph_2D_Strip5", "flagMeanGraph_2D_Strip5", 80, 0, 80, 6, 0, 6);
  flagMeanGraph_2D_Strip6 = new TH2F("flagMeanGraph_2D_Strip6", "flagMeanGraph_2D_Strip6", 80, 0, 80, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagMeanGraph_2D_Strip0","flagNoise:strip"); 
  Double_t binMaxValStrip = flagMeanGraph_2D_Strip0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagMeanGraph_2D_Strip->SetMaximum(binMaxValStrip);

  //fill each "bin"
  Calibration->Project("flagMeanGraph_2D_Strip1","flagNoise:strip", "flagNoise==1","box"); 
  Calibration->Project("flagMeanGraph_2D_Strip2","flagNoise:strip", "flagNoise==2","box"); 
  Calibration->Project("flagMeanGraph_2D_Strip3","flagNoise:strip", "flagNoise==3","box"); 
  Calibration->Project("flagMeanGraph_2D_Strip4","flagNoise:strip", "flagNoise==4","box"); 
  Calibration->Project("flagMeanGraph_2D_Strip5","flagNoise:strip", "flagNoise==5","box"); 

  //set appropriate colors
  flagMeanGraph_2D_Strip1->SetFillColor(1);//Black for eveything is OK
  flagMeanGraph_2D_Strip2->SetFillColor(2);//red for VERY BAD
  flagMeanGraph_2D_Strip3->SetFillColor(3);//Green for pretty good
  flagMeanGraph_2D_Strip4->SetFillColor(4);//blue for less good
  flagMeanGraph_2D_Strip5->SetFillColor(5);//red for VERY BAD


  flagMeanStripCanv->cd(); 
  flagMeanGraph_2D_Strip->GetYaxis()->SetTitle("Flag");
  flagMeanGraph_2D_Strip->GetXaxis()->SetTitle("Strip, all Chambers");
  
  //draw original histogram, empty
  flagMeanGraph_2D_Strip->Draw("box");
  //overlay the individual "bin" graphs
  flagMeanGraph_2D_Strip1->Draw("samebox");
  flagMeanGraph_2D_Strip2->Draw("samebox");
  flagMeanGraph_2D_Strip3->Draw("samebox");
  flagMeanGraph_2D_Strip4->Draw("samebox");
  flagMeanGraph_2D_Strip5->Draw("samebox");
  flagMeanGraph_2D_Strip6->Draw("samebox");

  //set legend entries appropriately
  LegMeanStrip->AddEntry(flagMeanGraph_2D_Strip1, "1: Normal Pedestal", "f");
  LegMeanStrip->AddEntry(flagMeanGraph_2D_Strip2, "2: Low Failure", "f");
  LegMeanStrip->AddEntry(flagMeanGraph_2D_Strip3, "3: Low Warning", "f");
  LegMeanStrip->AddEntry(flagMeanGraph_2D_Strip4, "4: High Warning", "f");
  LegMeanStrip->AddEntry(flagMeanGraph_2D_Strip5, "5: High Failure", "f");
  LegMeanStrip->Draw("same");

  //print as gif
  PrintAsGif(flagMeanStripCanv, "flagMeanStrip");

  ///////// CHAMBER flag Graph ///////////////////////
  TCanvas *flagMeanChamberCanv = new TCanvas("flagMeanChamberCanv", "flagMeanChamberCanv", 200,10,800,800);
  flagMeanChamberCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegMeanChamber = new TLegend(0.85,0.8,0.98,0.98);
  LegMeanChamber->SetHeader("Mean Flags Definitions");
  LegMeanChamber->SetFillColor(0);
  LegMeanChamber->SetTextSize(0);

  //final histogram for display
  flagMeanGraph_2D_Chamber = new TH2F("flagMeanGraph_2D_Chamber", "flagMeanGraph_2D_Chamber", 9, 0, 9, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagMeanGraph_2D_Chamber0 = new TH2F("flagMeanGraph_2D_Chamber0", "flagMeanGraph_2D_Chamber0", 9, 0, 9, 6, 0, 6);
  //one histo for each flag value. 
  flagMeanGraph_2D_Chamber1 = new TH2F("flagMeanGraph_2D_Chamber1", "flagMeanGraph_2D_Chamber1", 9, 0, 9, 6, 0, 6);
  flagMeanGraph_2D_Chamber2 = new TH2F("flagMeanGraph_2D_Chamber2", "flagMeanGraph_2D_Chamber2", 9, 0, 9, 6, 0, 6);
  flagMeanGraph_2D_Chamber3 = new TH2F("flagMeanGraph_2D_Chamber3", "flagMeanGraph_2D_Chamber3", 9, 0, 9, 6, 0, 6);
  flagMeanGraph_2D_Chamber4 = new TH2F("flagMeanGraph_2D_Chamber4", "flagMeanGraph_2D_Chamber4", 9, 0, 9, 6, 0, 6);
  flagMeanGraph_2D_Chamber5 = new TH2F("flagMeanGraph_2D_Chamber5", "flagMeanGraph_2D_Chamber5", 9, 0, 9, 6, 0, 6);
  flagMeanGraph_2D_Chamber6 = new TH2F("flagMeanGraph_2D_Chamber6", "flagMeanGraph_2D_Chamber6", 9, 0, 9, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagMeanGraph_2D_Chamber0","flagNoise:cham"); 
  Double_t binMaxValCham = flagMeanGraph_2D_Chamber0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagMeanGraph_2D_Chamber->SetMaximum(binMaxValCham);

  //fill each "bin"
  Calibration->Project("flagMeanGraph_2D_Chamber1","flagNoise:cham", "flagNoise==1", "box"); 
  Calibration->Project("flagMeanGraph_2D_Chamber2","flagNoise:cham", "flagNoise==2", "box"); 
  Calibration->Project("flagMeanGraph_2D_Chamber3","flagNoise:cham", "flagNoise==3", "box"); 
  Calibration->Project("flagMeanGraph_2D_Chamber4","flagNoise:cham", "flagNoise==4", "box"); 
  Calibration->Project("flagMeanGraph_2D_Chamber5","flagNoise:cham", "flagNoise==5", "box"); 

  //set appropriate colors
  flagMeanGraph_2D_Chamber1->SetFillColor(1);//Black for eveything is OK
  flagMeanGraph_2D_Chamber2->SetFillColor(2);//red for VERY BAD
  flagMeanGraph_2D_Chamber3->SetFillColor(3);//Green for pretty good
  flagMeanGraph_2D_Chamber4->SetFillColor(4);//blue for less good
  flagMeanGraph_2D_Chamber5->SetFillColor(5);//blue for less good

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
    flagMeanGraph_2D_Chamber->GetXaxis()->SetBinLabel(chamNumPlus,chamber_id_str); //set bins to have chamber names
  }
  
  flagMeanChamberCanv->cd(); 
  flagMeanGraph_2D_Chamber->GetYaxis()->SetTitle("Flag");
  flagMeanGraph_2D_Chamber->GetXaxis()->SetTitle("Chamber");
  //draw original histogram, empty
  flagMeanGraph_2D_Chamber->Draw("box");
  //overlay the individual "bin" graphs
  flagMeanGraph_2D_Chamber1->Draw("samebox");
  flagMeanGraph_2D_Chamber2->Draw("samebox");
  flagMeanGraph_2D_Chamber3->Draw("samebox");
  flagMeanGraph_2D_Chamber4->Draw("samebox");
  flagMeanGraph_2D_Chamber5->Draw("samebox");
  flagMeanGraph_2D_Chamber6->Draw("samebox");

  //set legend entries appropriately
  LegMeanChamber->AddEntry(flagMeanGraph_2D_Chamber1, "1: Normal Pedestal", "f");
  LegMeanChamber->AddEntry(flagMeanGraph_2D_Chamber2, "2: Low Failure", "f");
  LegMeanChamber->AddEntry(flagMeanGraph_2D_Chamber3, "3: Low Warning", "f");
  LegMeanChamber->AddEntry(flagMeanGraph_2D_Chamber4, "4: High Warning", "f");
  LegMeanChamber->AddEntry(flagMeanGraph_2D_Chamber5, "5: High Failure", "f");
  LegMeanChamber->Draw("same");

  //print as gif
  PrintAsGif(flagMeanChamberCanv, "flagMeanChamber");

/////////////////////// Layer Graphs //////////////

  TCanvas *flagMeanLayerCanv = new TCanvas("flagMeanLayerCanv", "flagMeanLayerCanv", 200,10,800,800);
  flagMeanLayerCanv->SetCanvasSize(1200,800);

  //create legend 
  TLegend *LegMeanLayer = new TLegend(0.85,0.8,0.98,0.98);
  LegMeanLayer->SetHeader("Mean Flags Definitions");
  LegMeanLayer->SetFillColor(0);
  LegMeanLayer->SetTextSize(0);

  //final histogram for display
  flagMeanGraph_2D_Layer = new TH2F("flagMeanGraph_2D_Layer", "flagMeanGraph_2D_Layer", 6, 0, 6, 6, 0, 6);
  ///dummy histo to get bin maximum
  flagMeanGraph_2D_Layer0 = new TH2F("flagMeanGraph_2D_Layer0", "flagMeanGraph_2D_Layer0", 6, 0, 6, 6, 0, 6);
  //one histo for each flag value. 
  flagMeanGraph_2D_Layer1 = new TH2F("flagMeanGraph_2D_Layer1", "flagMeanGraph_2D_Layer1", 6, 0, 6, 6, 0, 6);
  flagMeanGraph_2D_Layer2 = new TH2F("flagMeanGraph_2D_Layer2", "flagMeanGraph_2D_Layer2", 6, 0, 6, 6, 0, 6);
  flagMeanGraph_2D_Layer3 = new TH2F("flagMeanGraph_2D_Layer3", "flagMeanGraph_2D_Layer3", 6, 0, 6, 6, 0, 6);
  flagMeanGraph_2D_Layer4 = new TH2F("flagMeanGraph_2D_Layer4", "flagMeanGraph_2D_Layer4", 6, 0, 6, 6, 0, 6);
  flagMeanGraph_2D_Layer5 = new TH2F("flagMeanGraph_2D_Layer5", "flagMeanGraph_2D_Layer5", 6, 0, 6, 6, 0, 6);
  flagMeanGraph_2D_Layer6 = new TH2F("flagMeanGraph_2D_Layer6", "flagMeanGraph_2D_Layer6", 6, 0, 6, 6, 0, 6);

  //fill completley, get bin maximum, set it for overall graph
  Calibration->Project("flagMeanGraph_2D_Layer0","flagNoise:layer"); 
  Double_t binMaxValLayer = flagMeanGraph_2D_Layer0->GetMaximum();
  //normalize each box appropriately, with respect to the most filled box
  flagMeanGraph_2D_Layer->SetMaximum(binMaxValLayer);

  //fill each "bin"
  Calibration->Project("flagMeanGraph_2D_Layer1","flagNoise:layer", "flagNoise==1", "box"); 
  Calibration->Project("flagMeanGraph_2D_Layer2","flagNoise:layer", "flagNoise==2", "box"); 
  Calibration->Project("flagMeanGraph_2D_Layer3","flagNoise:layer", "flagNoise==3", "box"); 
  Calibration->Project("flagMeanGraph_2D_Layer4","flagNoise:layer", "flagNoise==4", "box"); 
  Calibration->Project("flagMeanGraph_2D_Layer5","flagNoise:layer", "flagNoise==5", "box"); 

  //set appropriate colors
  flagMeanGraph_2D_Layer1->SetFillColor(1);//Black for eveything is OK
  flagMeanGraph_2D_Layer2->SetFillColor(2);//red for VERY BAD
  flagMeanGraph_2D_Layer3->SetFillColor(3);//Green for pretty good
  flagMeanGraph_2D_Layer4->SetFillColor(4);//blue for less good
  flagMeanGraph_2D_Layer5->SetFillColor(5);

  flagMeanLayerCanv->cd(); 
  flagMeanGraph_2D_Layer->GetYaxis()->SetTitle("Flag");
  flagMeanGraph_2D_Layer->GetXaxis()->SetTitle("Layer, per Chamber");
  
  //draw original histogram, empty
  flagMeanGraph_2D_Layer->Draw("box");
  //overlay the individual "bin" graphs
  flagMeanGraph_2D_Layer1->Draw("samebox");
  flagMeanGraph_2D_Layer2->Draw("samebox");
  flagMeanGraph_2D_Layer3->Draw("samebox");
  flagMeanGraph_2D_Layer4->Draw("samebox");
  flagMeanGraph_2D_Layer5->Draw("samebox");
  flagMeanGraph_2D_Layer6->Draw("samebox");

  //set legend entries appropriately
  LegMeanLayer->AddEntry(flagMeanGraph_2D_Layer1, "1: Normal Pedestal", "f");
  LegMeanLayer->AddEntry(flagMeanGraph_2D_Layer2, "2: Low Failure", "f");
  LegMeanLayer->AddEntry(flagMeanGraph_2D_Layer3, "3: Low Warning", "f");
  LegMeanLayer->AddEntry(flagMeanGraph_2D_Layer4, "4: High Warning", "f");
  LegMeanLayer->AddEntry(flagMeanGraph_2D_Layer5, "5: High Failure", "f");
  LegMeanChamber->Draw("same");

  PrintAsGif(flagMeanLayerCanv, "flagMeanLayer");

  gSystem->cd("../../../");
  gStyle->SetOptStat(1); 
}

void PulseGraphs(TString myFileName, TString myFilePath){  
  TFile *myfile = TFile::Open(myFilePath);
  std::cout << "begginng of PulseGraphs, now in : " << gSystem->pwd() << std::endl;
  std::cout << "opening file: " << myFilePath << std::endl; 

  gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");
  gSystem->cd("images/Crosstalk");  
  gSystem->cd(myFileName);  
  makeDirectory("PulseGraphs"); 

  TCanvas *canv_pulse; 
  TH2F *gPulse; 
  TString cname="Pulse_Graph_all_chambers";
  TString ctitle="Pulse Graph over all Chambers";
  canv_pulse =  new TCanvas(cname,ctitle,1100,700);
  gPulse = (TH2F*)myfile->Get("pulse"); 
  gPulse->GetXaxis()->SetLabelSize(0.07);
  gPulse->GetYaxis()->SetLabelSize(0.07);
  gPulse->GetXaxis()->SetTitleSize(0.07);
  gPulse->GetXaxis()->SetTitleOffset(0.7);
  gPulse->GetXaxis()->SetTitle("Time (ns)");
  gPulse->GetYaxis()->SetTitle("ADC Count");
  canv_pulse->cd();
  gPulse->Draw();   
  /*
  int idArray[9];
  GetChamberIDs(idArray);
   for (int cham_num=0; cham_num<nCham; ++cham_num){
     int NStrips = 0;
     int nCFEB = 0;
     //dummy graph to retrieve number of strips, to set graph accordingly
     TCanvas *StripEntryCanvas = new TCanvas("StripDummyCanvas", "StripDummyCanvas");
     TH1F *StripEntryDummy = new TH1F("StripEntryDummy", "StripEntryDummy",10,0,100);
     TString chamVar = Form("cham==%d",cham_num);
     StripEntryCanvas->cd();
     Calibration->Draw("strip>>StripEntryDummy", chamVar);
     //number of strips in a given chamber. should be 480 for 5 CFEBs or 384 for 4 CFEBs
     NStrips= StripEntryDummy->GetEntries();
     
     TString cname  = Form("Chamber_%d_Pulse_Graphs",idArray[cham_num]); 
     //TString cname  = Form("Chamber_%d_Pulse_Graphs",cham_num); 
     TString ctitle = Form("Pulse for Chamber %d",idArray[cham_num]);
     //TString ctitle = Form("Pulse for Chamber %d",cham_num);
     canv_pulse =  new TCanvas(cname,ctitle,1100,700); 
     

     if (NStrips==480){
       nCFEB=6;
       canv_pulse->Divide(2,3);
     }if (NStrips==384){
       nCFEB=5; 
       canv_pulse->Divide(2,2);
     }
     
     for (int CFEB_num=1; CFEB_num<nCFEB; ++CFEB_num){ 
       TString graphname= Form("pulse%d%d",cham_num,CFEB_num); 
       canv_pulse->cd(CFEB_num); 
       pulse = (TH2F*)myfile->Get(graphname); 
       pulse->GetXaxis()->SetLabelSize(0.07);
       pulse->GetYaxis()->SetLabelSize(0.07);
       pulse->GetXaxis()->SetTitleSize(0.07);
       pulse->GetXaxis()->SetTitleOffset(0.7);
       pulse->GetXaxis()->SetTitle("Time (ns)");
       pulse->GetYaxis()->SetTitle("ADC Count");
       pulse->Draw();   
     }//for CFEB 
}//for cham    
  */ 
  gSystem->cd("PulseGraphs");   
  TString PulseGraphCanvName = canv_pulse->GetName();
  PrintAsGif(canv_pulse,PulseGraphCanvName);
  
  gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_5_1/src/OnlineDB/CSCCondDB/test");
  directoryCheck();   
  myfile->Close(); 
}//PulseGraphs() 

void PeakGraphs(TString myFileName, TString myFilePath, int nCham){ 
std::cout << "begginng of PeakGraphs, now in : " << gSystem->pwd() << std::endl;
std::cout << "opening file: " << myFilePath << std::endl; 
gSystem->cd("/afs/cern.ch/cms/CSC/html/csccalib/");

gSystem->cd("images/Crosstalk");
gSystem->cd(myFileName);
makeDirectory("PeakADCTimingGraphs");

TLine *vert_min_Time = new TLine(250, 0, 250, 200);
TLine *vert_max_Time = new TLine(360, 0, 360, 200);
TLine *vert_min_ADC = new TLine(700, 0, 700, 200);
TLine *vert_max_ADC = new TLine(1200, 0, 1200, 200);

vert_min_Time->SetLineStyle(3);//dotted
vert_min_Time->SetLineColor(6);//red
vert_min_Time->SetLineWidth(2);//visible but not overwhelming width

vert_max_Time->SetLineStyle(3);//dotted
vert_max_Time->SetLineColor(6);//red
vert_max_Time->SetLineWidth(2);//visible but not overwhelming width

vert_min_ADC->SetLineStyle(3);//dotted
vert_min_ADC->SetLineColor(6);//red
vert_min_ADC->SetLineWidth(2);//visible but not overwhelming width

vert_max_ADC->SetLineStyle(3);//dotted
vert_max_ADC->SetLineColor(6);//red
vert_max_ADC->SetLineWidth(2);//visible but not overwhelming width

//Time and ADC Graphs for everything 
TCanvas *PeakCanv = new TCanvas("Peak Timing", "Peak Time and ADC for all Chambers", 200, 10, 1200, 800);
PeakCanv->Divide(2,1); 
PeakCanv->SetCanvasSize(1200,800);

PeakCanv->cd(1);
TH1F *PeakADC = new TH1F("PeakADC", "Peak ADC over all Chambers", 100, 600, 1400);
PeakADC->Draw(); 
PeakADC->GetXaxis()->SetTitle("ADC Count at Peak Time");
PeakADC->GetXaxis()->SetTitleSize(0.070);
PeakADC->GetXaxis()->SetTitleOffset(0.5);
Calibration->Project("PeakADC", "maxADC");   

Double_t yMaxValADC = PeakADC->GetMaximum();
Double_t yMaxRanADC = yMaxValADC + (yMaxValADC*.1);
vert_max_ADC->SetY2(yMaxRanADC);
vert_min_ADC->SetY2(yMaxRanADC);
PeakADC->SetAxisRange(0,yMaxRanADC,"Y");
vert_max_ADC->Draw(); 
vert_min_ADC->Draw();

PeakCanv->cd(2);
TH1F *PeakTime = new TH1F("PeakTime", "Peak Time over all Chambers", 100, 200, 400);
PeakTime->Draw(); 
PeakTime->GetXaxis()->SetTitle("Peak Time (ns)");
PeakTime->GetXaxis()->SetTitleSize(0.070);
PeakTime->GetXaxis()->SetTitleOffset(0.5);
Calibration->Project("PeakTime", "peakTime");   

Double_t yMaxValTime = PeakTime->GetMaximum();
Double_t yMaxRanTime = yMaxValTime + (yMaxValTime*.1);
vert_max_Time->SetY2(yMaxRanTime);
vert_min_Time->SetY2(yMaxRanTime);
PeakTime->SetAxisRange(0,yMaxRanTime,"Y");
vert_max_Time->Draw(); 
vert_min_Time->Draw(); 

//take the new line into account 
PeakCanv->Update();
PrintAsGif(PeakCanv, "PeakTimingADC");

///Time and ADC Graphs per chamber
TObjArray PeakGraphCanvArray(nCham);

TH1F *PeakADCGraphs[9];
TH1F *PeakTimeGraphs[9];

int idArray[9];
GetChamberIDs(idArray);
for (int cham_num=0; cham_num < nCham; ++cham_num){
gSystem->cd("PeakADCTimingGraphs");
Int_t chamber_id_int = idArray[cham_num];
 
Float_t MaxPeakTimeVal = Calibration->GetLeaf("MaxPeakTime")->GetValue(cham_num);
Float_t MinPeakTimeVal = Calibration->GetLeaf("MinPeakTime")->GetValue(cham_num);
Float_t MaxPeakADCVal = Calibration->GetLeaf("MaxPeakADC")->GetValue(cham_num);
Float_t MinPeakADCVal = Calibration->GetLeaf("MinPeakADC")->GetValue(cham_num);

Float_t MaxPeakTimeValue = MaxPeakTimeVal + (.1)*MaxPeakTimeVal; 
if (MaxPeakTimeValue < 400.0){
   MaxPeakTimeValue = 400.0;
 }
Float_t MinPeakTimeValue = MinPeakTimeVal + (.1)*MinPeakTimeVal; 
if (MinPeakTimeValue > 200.0){
   MinPeakTimeValue = 200.0;
}
Float_t MaxPeakADCValue = MaxPeakADCVal + (.1)*MaxPeakADCVal; 
if (MaxPeakADCValue < 1300.0){
   MaxPeakADCValue = 1300.0;
 }
Float_t MinPeakADCValue = MinPeakADCVal + (.1)*MinPeakADCVal; 
if (MinPeakADCValue < 600.0){
  MinPeakADCValue = 600.0;
}

TString PeakGraphCanvName = Form ("Chamber_%d_Peak_Graphs", chamber_id_int);
TCanvas *canv_peak =  new TCanvas(PeakGraphCanvName,PeakGraphCanvName, 200, 10, 1200, 800); 
canv_peak->Divide(2,1);
canv_peak->SetCanvasSize(1200,800);

canv_peak->cd(1);
TString hist_name = Form("PeakADCGraph_Chamber_%d", chamber_id_int);
PeakADCGraphs[cham_num] = new TH1F(hist_name, "peakADC Plot", 50, MinPeakADCValue, MaxPeakADCValue);
PeakADCGraphs[cham_num]->Draw();

TString xTitleName = Form("ADC Count at Peak Time for chamber %d", chamber_id_int);
PeakADCGraphs[cham_num]->GetXaxis()->SetTitle(xTitleName);
PeakADCGraphs[cham_num]->GetXaxis()->SetTitleSize(0.050);
PeakADCGraphs[cham_num]->GetXaxis()->SetTitleOffset(0.5);

const char value[7];
sprintf(value, "cham==%d", cham_num);
const char *value_ = value;
Calibration->Project(hist_name, "maxADC", value_); 

Double_t yMaxValADC = PeakADCGraphs[cham_num]->GetMaximum();
Double_t yMaxRanADC = yMaxValADC + (yMaxValADC*.1);
vert_max_ADC->SetY2(yMaxRanADC); 
vert_min_ADC->SetY2(yMaxRanADC); 
PeakADCGraphs[cham_num]->SetAxisRange(0,yMaxRanADC,"Y"); 
vert_max_ADC->Draw(); 
vert_min_ADC->Draw(); 

canv_peak->cd(2);
TString hist_name_ = Form("PeakTimeGraph_Chamber_%d", chamber_id_int);
PeakTimeGraphs[cham_num] = new TH1F(hist_name_, "peakTime Plot", 50, MinPeakTimeValue, MaxPeakTimeValue);
PeakTimeGraphs[cham_num]->Draw();

PeakTimeGraphs[cham_num]->GetXaxis()->SetTitle("Peak Time in nano seconds");
PeakTimeGraphs[cham_num]->GetXaxis()->SetTitleSize(0.070);
PeakTimeGraphs[cham_num]->GetXaxis()->SetTitleOffset(0.5);

const char valu[7];
sprintf(valu, "cham==%d", cham_num);
const char *valu_ = valu;
Calibration->Project(hist_name_, "peakTime", valu_); 

Double_t yMaxValTime = PeakTimeGraphs[cham_num]->GetMaximum();
Double_t yMaxRanTime = yMaxValTime + (yMaxValTime*.1);
vert_max_Time->SetY2(yMaxRanTime);
vert_min_Time->SetY2(yMaxRanTime);
PeakTimeGraphs[cham_num]->SetAxisRange(0,yMaxRanTime,"Y");
vert_max_Time->Draw(); 
vert_min_Time->Draw(); 
 
canv_peak->Update();

PeakGraphCanvArray.Add(canv_peak);

PrintAsGif(canv_peak, PeakGraphCanvName); 
}//for chamber
gSystem->cd("/afs/cern.ch/user/c/csccalib/scratch0/CMSSW_1_5_1/src/OnlineDB/CSCCondDB/test");
directoryCheck(); 
}//PeakGraphs 
