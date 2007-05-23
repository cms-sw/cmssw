/* 2007-4-25
   Wade Fisher
   Hcal DQM individual channel plotter
*/

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>

///This variable must be set to a valid electronics map!!
const char* gl_emap = "hfplusmap.txt";

/*
Usage:

1) First enter a valid electronics map (above) which corresponds to the
mapping used to create DQM output files.  Choose a valid HCAL DQM output file ("DQM_Hcal_xxRUNxxNUMBERxx.root")

2) Open ROOT: "> root -l"

3) Load this file: "root [0] .L hcalDQMchannelPlotter.C"

4a) To plot a channel indexed by detector geometry:
   root [1] plotGeometric("DQM_Hcal_000000274.root")

4b) To plot a channel indexed by detector electronics:
   root [1] plotElectronic("DQM_Hcal_000000274.root")

The software will test for valid channels based on the electronics
map.  The only per-channel histos currently implemented are pedestal
and LED histos.

*/

vector<int> inputEMAP(vector<int> compVar, vector<int> compIdx){
  
  vector<string> lines;
  std::string line;
  lines.clear();
  std::ifstream infile(gl_emap, std::ios_base::in);
  if (!infile) {
    cout<<"Cannot open file "<<gl_emap<<endl;
    return false;
  }
  
  while(getline(infile, line, '\n')) {
    
    /*
    string dummy;     //0
    string crate;     //1
    string slot;      //2
    string tb;        //3
    string dcc;       //4
    string spigot;    //5
    string fiber;     //6
    string fiberchan; //7
    string subdet;    //8
    string ieta;      //9
    string iphi;      //10
    string depth;     //11
    */

    TString conv = TString(line.c_str());
    TObjArray* aline = conv.Tokenize(" ");
    bool foundIt = true;
    for(int idx=0; idx<compVar.size() && foundIt; idx++){
      TString s = ((TObjString*)(aline->At(compIdx[idx])))->GetString();
      if( compVar[idx] != atoi(s.Data())) foundIt = false;
    }
    if(foundIt){
      vector<int> found;
      for(int l=0; l<aline->GetEntriesFast(); l++){
	TString s = ((TObjString*)(aline->At(l)))->GetString();
	if(l!=3 && l!=8) found.push_back(atoi(s.Data()));
	else{
	  if(l==3){
	    if(s==TString("t")) found.push_back(0);
	    else if(s==TString("b")) found.push_back(1);
	  }
	  if(l==8){
	    if(s==TString("HB")) found.push_back(0);
	    else if(s==TString("HE")) found.push_back(1);
	    else if(s==TString("HF")) found.push_back(2);
	    else if(s==TString("HO")) found.push_back(3);
	  }
	}
      }
      return found;
    }
  }
  vector<int> notFound;
  return notFound;
}

vector<int> inputEMAP2(vector<int> compVar, vector<int> compIdx){
  
  vector<string> lines;
  std::string line;
  lines.clear();
  std::ifstream infile(gl_emap, std::ios_base::in);
  if (!infile) {
    cout<<"Cannot open file "<<gl_emap<<endl;
    return false;
  }
  
  while(getline(infile, line, '\n')) {

    TString conv = TString(line.c_str());
    TObjArray* aline = conv.Tokenize(" ");
    bool foundIt = true;
    for(int idx=0; idx<compVar.size() && foundIt; idx++){
      TString s = ((TObjString*)(aline->At(compIdx[idx])))->GetString();
      if( compVar[idx] != atoi(s.Data())) foundIt = false;
    }
    if(foundIt){
      vector<int> found;
      for(int l=0; l<aline->GetEntriesFast(); l++){
	TString s = ((TObjString*)(aline->At(l)))->GetString();
	if(l!=3 && l!=8) found.push_back(atoi(s.Data()));
	else{
	  if(l==3){
	    if(s==TString("t")) found.push_back(0);
	    else if(s==TString("b")) found.push_back(1);
	  }
	  if(l==8){
	    if(s==TString("HB")) found.push_back(0);
	    else if(s==TString("HE")) found.push_back(1);
	    else if(s==TString("HF")) found.push_back(2);
	    else if(s==TString("HO")) found.push_back(3);
	  }
	}
      }
      printf("Found: line\n");
    }
  }
  vector<int> notFound;
  return notFound;
}

void makePedestalHistos(vector<int> inLine, const char* histFile){
  TFile* infile = new TFile(histFile);
  if(!infile){
    cout<<"Error opening "<<histFile<<endl; 
    return;
  }

  string type = "HB";
  if(inLine[8]==1) type="HE";
  else if(inLine[8]==2) type="HF";
  else if(inLine[8]==3) type="HO";
  char name[256];

  TH1F* pedADC[4];
  TH1F* pedSUB[4];

  for(int i=0; i<4; i++){
    sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d",type.c_str(),type.c_str(),inLine[9],inLine[10],inLine[11],i);
    pedADC[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value (Subtracted) ieta=%d iphi=%d depth=%d CAPID=%d",type.c_str(),type.c_str(),inLine[9],inLine[10],inLine[11],i);
    pedSUB[i] = (TH1F*)infile->Get(name);
  }

  if(!pedADC[0] || !pedSUB[0]){
    cout << "Pedestal Histograms do not exist for this channel!" << endl;
    return;
  }

  sprintf(name,"ADC Peds ieta=%d iphi=%d depth=%d",inLine[9],inLine[10],inLine[11]);
  TCanvas* c1 = new TCanvas(name,name);
  c1->Divide(2,2);
  for(int i=0; i<4; i++){
    c1->cd(i+1);
    pedADC[i]->Draw();
  }

  sprintf(name,"Subtracted Peds ieta=%d iphi=%d depth=%d",inLine[9],inLine[10],inLine[11]);
  TCanvas* c1 = new TCanvas(name,name);
  c1->Divide(2,2);
  for(int i=0; i<4; i++){
    c1->cd(i+1);
    pedSUB[i]->Draw();
  }

  return;
}

void makeLEDHistos(vector<int> inLine, const char* histFile){
  TFile* infile = new TFile(histFile);
  if(!infile){
    cout<<"Error opening "<<histFile<<endl; 
    return;
  }

  string type = "HB";
  if(inLine[8]==1) type="HE";
  else if(inLine[8]==2) type="HF";
  else if(inLine[8]==3) type="HO";
  char name[256];

  TH1F* ledHist[3];

  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",type.c_str(),type.c_str(),inLine[9],inLine[10],inLine[11]);
  ledHist[0] = (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",type.c_str(),type.c_str(),inLine[9],inLine[10],inLine[11]);
  ledHist[1] = (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",type.c_str(),type.c_str(),inLine[9],inLine[10],inLine[11]);
  ledHist[2] = (TH1F*)infile->Get(name);


  if(!ledHist[0]){
    cout << "LED Histograms do not exist for this channel!" << endl;
    return;
  }

  sprintf(name,"LED Histos ieta=%d iphi=%d depth=%d",inLine[9],inLine[10],inLine[11]);
  TCanvas* c1 = new TCanvas(name,name);
  c1->Divide(2,2);
  for(int i=0; i<3; i++){
    c1->cd(i+1);
    ledHist[i]->Draw();
  }

  return;
}

vector<int> findLine_Geo(int ieta, int iphi,int depth){  
  vector<int> testVar; vector<int> testIdx;
  testVar.push_back(ieta); testIdx.push_back(9);
  testVar.push_back(iphi); testIdx.push_back(10);
  testVar.push_back(depth); testIdx.push_back(11);
  return inputEMAP(testVar, testIdx);
}

vector<int> findLine_Elec(int dcc,int spigot,int fib,int fchan){
  vector<int> testVar; vector<int> testIdx;
  testVar.push_back(dcc); testIdx.push_back(4);
  testVar.push_back(spigot); testIdx.push_back(5);
  testVar.push_back(fib); testIdx.push_back(6);
  testVar.push_back(fchan); testIdx.push_back(7);
  return inputEMAP(testVar, testIdx);
}

vector<int> findLine_Elec2(int dcc,int crate,int slot){
  vector<int> testVar; vector<int> testIdx;
  testVar.push_back(dcc); testIdx.push_back(4);
  testVar.push_back(crate); testIdx.push_back(1);
  testVar.push_back(slot); testIdx.push_back(2);
  return inputEMAP2(testVar, testIdx);
}

void plotGeometric(const char* histFile){

  printf("\n\n*****************************************\n");
  printf("***        HCAL Channel Plotter       ***\n");
  printf("***                                   ***\n");
  printf("***        Plotting by Geometry       ***\n");
  printf("*****************************************\n");

  int ieta, iphi, depth;
  printf("Enter iEta: ");
  scanf("%d",&ieta);  
  printf("Enter iPhi: ");
  scanf("%d",&iphi);  
  printf("Enter Depth: ");
  scanf("%d",&depth);  
  printf("You have entered %d, %d, %d\n",ieta,iphi,depth);

  vector<int> cLine = findLine_Geo(ieta,iphi,depth);
  if(cLine.size()==0){
     cout<<"Electronics map did not contain specified point!"<<endl;
     return;
  }
  
  makePedestalHistos(cLine,histFile);
  makeLEDHistos(cLine,histFile);  

  return;
}

void plotElectronic(const char* histFile){

  printf("\n\n*****************************************\n");
  printf("***        HCAL Channel Plotter       ***\n");
  printf("***                                   ***\n");
  printf("***       Plotting by Electronics     ***\n");
  printf("*****************************************\n");

  int dcc, spigot, fib, fchan;
  printf("Enter dcc-ID: ");
  scanf("%d",&dcc);  
  printf("Enter spigot: ");
  scanf("%d",&spigot);  
  printf("Enter fiber: ");
  scanf("%d",&fib);
  printf("Enter fiberChannel: ");
  scanf("%d",&fchan);
  
  printf("You have entered %d, %d, %d, %d\n",dcc,spigot,fib,fchan);

  vector<int> cLine = findLine_Elec(dcc,spigot,fib,fchan);
  if(cLine.size()==0){
     cout<<"Electronics map did not contain specified point!"<<endl;
     return;
  }
  
  makePedestalHistos(cLine,histFile);
  makeLEDHistos(cLine,histFile);  

  return;
}

void plotElectronic2(const char* histFile){

  printf("\n\n*****************************************\n");
  printf("***        HCAL Channel Plotter       ***\n");
  printf("***                                   ***\n");
  printf("***     Plotting by Electronics(#2)   ***\n");
  printf("*****************************************\n");

  int dcc, crate, slot;
  printf("Enter dcc-ID: ");
  scanf("%d",&dcc);  
  printf("Enter crate: ");
  scanf("%d",&crate);  
  printf("Enter slot: ");
  scanf("%d",&slot);
  printf("You have entered %d, %d, %d\n",dcc,crate,slot);

  vector<int> cLine = findLine_Elec2(dcc,crate,slot);

  return;
}
