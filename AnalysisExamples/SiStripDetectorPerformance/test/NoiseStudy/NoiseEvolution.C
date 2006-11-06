/*
* $Date: $
* $Revision:  $
*
* \author: D. Giordano, domenico.giordano@cern.ch
*/


#include <iostream>
#include <stdio.h>
#include <string.h>
#include <pair>
#include <map>
#include <vector>

gROOT->Reset();

//&&&&&&&&&&&&&&&&&&&&
//Global variables
//&&&&&&&&&&&&&&&&&&&&

//---------------------------------
//User Defined Variables

char outFile[128]="NoiseEvolution";
char path[128]="/castor/cern.ch/user/g/giordano/MTCC/Display/Display_PedNoise_RunNb";
//char path[128]="/data/giordano/Display/Display_PedNoise_RunNb";
int histoBins=60;
float histoMin=-1.;
float histoMax= 1.;

//End User Defined Variables
//---------------------------------

float profileAlpha= 0.2;

char inFile[128], refFile[128];
TRFIOFile *inFile_, *refFile_;
TFile *outFile_;

TH1F *inH, *refH, *outH;

TCanvas * C;

int iov[11]={2354,2371,2440,2459,2475,2500,2515,2554,2601,2644,10000};
int iovDim = 11;

char *SubDet[4]={"TIB","TID","TOB","TEC"};

std::vector<string> vHistoNames;
std::vector<int> vHistoNBinsX;
std::vector<string> vLayerName;

TObjArray Hlist(0);

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void book(){
  cout << "\n[book]\n" << endl;

  char filename[128];
  sprintf(filename,"%s.root",outFile);
  outFile_ = new TFile(filename,"RECREATE");
  
  for (int i=1;i<iovDim;i++){
    char dirName[128];
    sprintf(dirName,"IOV_%d",iov[i]);
    outFile_->mkdir(dirName);
    outFile_->cd(dirName);
    
    gDirectory->mkdir("DetId");
    gDirectory->mkdir("Layer");
    gDirectory->mkdir("SubDet");

    gDirectory->cd("DetId");
    //Make Histos for each detector
    for (int j=0;j<vHistoNames.size();j++){
      //      cout << vHistoNames[j] << endl;
      char newName[128];
      sprintf(newName,"NoiseVariationProfile_%s",&((vHistoNames[j].c_str())[23]));
      //cout << newName << endl;
      //cout << "h " << h << endl;
      Hlist.Add(new TH1F(newName,newName,vHistoNBinsX[j],-0.5,vHistoNBinsX[j]-0.5));
      sprintf(newName,"NoiseVariation_%s",&vHistoNames[j][23]);
      Hlist.Add(new TH1F(newName,newName,histoBins,histoMin,histoMax));      

    }

    gDirectory->cd("../Layer");
    //Make Histos for each layer
    for (int j=0;j<vLayerName.size();j++){
      char newName[128];
      sprintf(newName,"NoiseVariation_%s",vLayerName[j].c_str());
      Hlist.Add(new TH1F(newName,newName,histoBins,histoMin,histoMax));      
      sprintf(newName,"NoiseComparison_%s",vLayerName[j].c_str());
      Hlist.Add(new TH2F(newName,newName,histoBins,0,10,histoBins,0,10));      
    }

    gDirectory->cd("../SubDet");
    //Make Histos hor each SubDet
    Hlist.Add(new TH1F("NoiseVariation_TIB","NoiseVariation_TIB",histoBins,histoMin,histoMax));      
    Hlist.Add(new TH1F("NoiseVariation_TOB","NoiseVariation_TOB",histoBins,histoMin,histoMax));      
    Hlist.Add(new TH1F("NoiseVariation_TEC","NoiseVariation_TEC",histoBins,histoMin,histoMax));      
    Hlist.Add(new TH1F("NoiseVariation_TID","NoiseVariation_TID",histoBins,histoMin,histoMax));      
  }
  
  gDirectory->cd("../..");
  
  for (int j=0;j<vLayerName.size();j++){
    char newName[128];
    sprintf(newName,"pNoiseVariation_%s",vLayerName[j].c_str());
    Hlist.Add(new TProfile(newName,newName,100,iov[1],2800,histoMin,histoMax));
  }
  
  Hlist.Add(new TProfile("pNoiseVariation_TIB","pNoiseVariation_TIB",10,iov[1],2800,histoMin,histoMax));
  Hlist.Add(new TProfile("pNoiseVariation_TID","pNoiseVariation_TID",10,iov[1],2800,histoMin,histoMax));
  Hlist.Add(new TProfile("pNoiseVariation_TOB","pNoiseVariation_TOB",10,iov[1],2800,histoMin,histoMax));
  Hlist.Add(new TProfile("pNoiseVariation_TEC","pNoiseVariation_TEC",10,iov[1],2800,histoMin,histoMax));
  
  outFile_->cd();
}     

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void LayerName(char* input,char* output){

  char *pch;
  
  for(int i=0;i<4;i++){
    pch=strstr(input,SubDet[i]);
    if (pch != NULL){
      sprintf(output,"%s_",SubDet[i]);
      char *qch = strstr(input,"Rphi");
      if ( qch != NULL){
	if (SubDet[i]!="TEC")
	  strncat(output,qch,5);
	else
	  strncat(output,--qch,5);
      }else{
	qch = strstr(input,"Ster");
	if (qch != NULL ){
	  if (SubDet[i]!="TEC")
	    strncat(output,qch,5);
	  else
	    strncat(output,--qch,5);
	} else {
	  cout << "ERROR: module isn't Rphi or Stereo" << endl;
	}
      }
      break;
    }
  }
}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void save(){
  cout << "\n[save]\n" << endl;

  //gDirectory->cd();
  
  std::cout << "........ saving file "<< outFile_->GetName() << std::endl;
  outFile_->Write();
  std::cout << "........ done"<< std::endl;
  
  char filename[128];
  sprintf(filename,"%s.ps",outFile);
  std::cout << "........ opening file "<< filename << std::endl;
  TPostScript ps(filename,121);
  
  //outFile_->cd("IOV_2371/DetId");
  //gDirectory->ls();
  TCanvas myCanvas;
  myCanvas.Draw();

  for (int i=1;i<iovDim;i++){
    //Make Histos for each layer
    for (int j=0;j<vLayerName.size();j++){
      char newName[128];
      myCanvas.SetLogy(1);;
      sprintf(newName,"IOV_%d/Layer/NoiseVariation_%s",iov[i],vLayerName[j].c_str());
      if (((TH1F*) outFile_->Get(newName))->GetEntries()){
	((TH1F*) outFile_->Get(newName))->Draw();
	myCanvas.Update();
	ps.NewPage();
      }
      
      myCanvas.SetLogy(0);
      sprintf(newName,"IOV_%d/Layer/NoiseComparison_%s",iov[i],vLayerName[j].c_str());
      if (((TH2F*) outFile_->Get(newName))->GetEntries()){
	((TH2F*) outFile_->Get(newName))->Draw();
	myCanvas.Update();
	ps.NewPage();
      }
    }

    myCanvas.SetLogy(1);
    for (int j=0;j<4;j++){
      sprintf(newName,"IOV_%d/SubDet/NoiseVariation_%s",iov[i],SubDet[j]);
      if (((TH1F*) outFile_->Get(newName))->GetEntries()){
	((TH1F*) outFile_->Get(newName))->Draw();
	myCanvas.Update();
	ps.NewPage();
      }
    }
    myCanvas.SetLogy(0);
  }

  for (int j=0;j<vLayerName.size();j++){
    char newName[128];
    sprintf(newName,"pNoiseVariation_%s",vLayerName[j].c_str());
    float Max = ((TProfile*) outFile_->Get(newName))->GetMaximum();
    float Min = ((TProfile*) outFile_->Get(newName))->GetMinimum();
    ((TProfile*) outFile_->Get(newName))->SetMaximum((1+profileAlpha)*Max-profileAlpha*Min);
    ((TProfile*) outFile_->Get(newName))->SetMinimum((1+profileAlpha)*Min-profileAlpha*Max);
    ((TProfile*) outFile_->Get(newName))->Draw();
    myCanvas.Update();
    ps.NewPage();
  }
  
  for (int j=0;j<4;j++){
    char newName[128];
    sprintf(newName,"pNoiseVariation_%s",SubDet[j]);
    float Max = ((TProfile*) outFile_->Get(newName))->GetMaximum();
    float Min = ((TProfile*) outFile_->Get(newName))->GetMinimum();
    ((TProfile*) outFile_->Get(newName))->SetMaximum((1+profileAlpha)*Max-profileAlpha*Min);
    ((TProfile*) outFile_->Get(newName))->SetMinimum((1+profileAlpha)*Min-profileAlpha*Max);
    ((TProfile*) outFile_->Get(newName))->Draw();
    myCanvas.Update();
    ps.NewPage();
  }
  
  //  for (int ih=0; ih<Hlist.GetEntries();ih++){
    //    Hlist.At(ih)->Draw();
    //myCanvas.Update();
    //ps.NewPage();
  //}
  ps.Close();
  std::cout << "........ closed"<< std::endl;
  std::cout << "to see the file please do \n gv " << filename << std::endl;
  
  outFile_->Close();  
}

void variation(TH1F* in, TH1F* ref, int iov){

  char inName[128];
  char outName[128];
  TH1F* outH1[4];
  TH2F* outH2;
  TProfile* outP[2];

  strcpy(inName,in->GetTitle());
  sprintf(outName,"IOV_%d/DetId/NoiseVariationProfile_%s",iov,&(inName[23]));
  outH1[0] = (TH1F*)  outFile_->Get(outName);

  sprintf(outName,"IOV_%d/DetId/NoiseVariation_%s",iov,&(inName[23]));
  outH1[1] = (TH1F*)  outFile_->Get(outName);

  char tmp[128];
  LayerName(inName,tmp);
  sprintf(outName,"IOV_%d/Layer/NoiseVariation_%s",iov,tmp);
  outH1[2] = (TH1F*)  outFile_->Get(outName);

  sprintf(outName,"IOV_%d/Layer/NoiseComparison_%s",iov,tmp);
  outH2 = (TH2F*)  outFile_->Get(outName);

  sprintf(outName,"pNoiseVariation_%s",tmp);
  outP[0] = (TProfile*) outFile_->Get(outName);

  char det[4];
  det[3]='\0';
  strncpy(det,tmp,3);

  sprintf(outName,"IOV_%d/SubDet/NoiseVariation_%s",iov,det);
  //  cout << outName << endl;
  outH1[3] = (TH1F*)  outFile_->Get(outName);
  
  sprintf(outName,"pNoiseVariation_%s",det);
  outP[1] = (TProfile*) outFile_->Get(outName);
  
  float delta;
  for (int i=1;i<=in->GetNbinsX();i++){
    //    cout << " i = " << i << " " << in->GetBinContent(i) << endl;
    
    if (ref->GetBinContent(i))
      delta=  in->GetBinContent(i)/ref->GetBinContent(i) - 1 ;
    else
      -9999;

    outH1[0]->SetBinContent(i,delta);
    
    for (int j=1;j<4;j++)
      outH1[j]->Fill(delta);

    outH2->Fill(ref->GetBinContent(i),in->GetBinContent(i));

    int iov_= iov==10000 ? 2700 : iov;
    for (int j=0;j<2;j++)
      outP[j]->Fill(iov_,delta);

  }
}

void loop(int iov){

  //Loop on detectors
  for (int j=0;j<vHistoNames.size();j++){
      
    //cout << vHistoNames[j] << endl;
    char hTitle[128];
    sprintf(hTitle,"Noises/%s",vHistoNames[j].c_str());
    //cout << hTitle << endl;
    inH  = (TH1F*) inFile_->Get(hTitle);
    refH = (TH1F*) refFile_->Get(hTitle);
    
    variation(inH,refH,iov);
  }  
}

NoiseEvolution(){

  //Open Reference File
  sprintf(refFile,"%s_%d.root",path,iov[0]);
  cout << "\nReference File " << refFile << endl;
  refFile_= new TRFIOFile(refFile);

  //Get Histo Names
  refFile_->cd("Noises");
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey *key;
  while (key = (TKey*)nextkey()) {    
    const char * title;
    title=key->GetTitle();
    if (strncmp(title,"Noises_Field",12)==0){
      vHistoNames.push_back(string(title));
      vHistoNBinsX.push_back(((TH1F*)key->ReadObj())->GetNbinsX());
      
      char tmp[128];
      LayerName(title,tmp);
      string tmp1(tmp);
      int i=0;
      while (i<vLayerName.size() && tmp1!=vLayerName[i]){i++;}
      if (i==vLayerName.size())
	vLayerName.push_back(tmp1);      
    }
  }

  book();

  
  for (int i=1;i<iovDim;i++){

    sprintf(inFile,"%s_%d.root",path,iov[i]);
    cout << "\nAnalyzing File " << inFile << endl;
    inFile_= new TRFIOFile(inFile);
  
    loop(iov[i]);
  }
  
  save();
}
