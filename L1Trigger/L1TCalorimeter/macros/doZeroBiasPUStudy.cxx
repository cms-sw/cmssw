#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TChain.h"
#include <TCanvas.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "L1Trigger/L1TNtuples/interface/L1AnalysisEventDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoVertexDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisCaloTPDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1CaloTowerDataFormat.h"


void fitProfile(TH1D* prof){
 
  TCanvas* profCanvas = new TCanvas("profCanvas","",600,600);
 
  TH1D *prof_fit2 = (TH1D*)prof->Clone("prof_fit2");
  TH1D *prof_fit3 = (TH1D*)prof->Clone("prof_fit3");
  TH1D *prof_fit4 = (TH1D*)prof->Clone("prof_fit4");
  prof->SetMarkerStyle(7);
  prof_fit2->SetMarkerStyle(7);
  prof_fit3->SetMarkerStyle(7);
  prof_fit4->SetMarkerStyle(7);
  
  TF1 *fit1 = new TF1("fit1","[0]*x+[1]");
  fit1->SetParameter(0,1.0); 
  fit1->SetParameter(1,20);   
  fit1->SetRange(0,50);     
  prof->Fit("fit1","R");
  TF1 *fit2 = new TF1("fit2","[0]*x+[1]");
  fit2->SetParameter(0,0.8); 
  fit2->SetParameter(1,60);   
  fit2->SetRange(50,100);
  prof_fit2->Fit("fit2","R");
  TF1 *fit3 = new TF1("fit3","[0]*x+[1]");
  fit3->SetParameter(0,0.5);
  fit3->SetParameter(1,4);  
  fit3->SetRange(110,140);
  prof_fit3->Fit("fit3","R");   
  TF1 *fit4 = new TF1("fit4","[0]*x+[1]");
  fit4->SetParameter(0,0.5);
  fit4->SetParameter(1,4);  
  fit4->SetRange(140,160);   
  prof_fit4->Fit("fit4","R");
  prof->Draw();  
  prof_fit2->Draw("sames");  
  prof_fit3->Draw("sames");  
  prof_fit4->Draw("sames");  
  gPad->Update();
  TPaveStats *st1 = (TPaveStats*)prof->FindObject("stats");
  TPaveStats *st2 = (TPaveStats*)prof_fit2->FindObject("stats");
  TPaveStats *st3 = (TPaveStats*)prof_fit3->FindObject("stats");
  TPaveStats *st4 = (TPaveStats*)prof_fit4->FindObject("stats");
  st1->SetY1NDC(0.75); st1->SetY2NDC(0.95); 
  st2->SetY1NDC(0.55); st2->SetY2NDC(0.75);
  st3->SetY1NDC(0.35); st3->SetY2NDC(0.55);  
  st4->SetY1NDC(0.15); st4->SetY2NDC(0.35);  
  fit1->Draw("sames"); fit2->Draw("sames"); //fit3->Draw("sames"); fit4->Draw("sames");
  prof->Write();
  profCanvas->SaveAs("ProfNVtxNTowemu4.pdf");

}

void makeLUT(TFile* file, int puBins){

  vector<vector<TH1D*> > hTowEtPU(puBins, vector<TH1D*> (41));

  for(uint pu=0;pu<puBins;pu++){
    for(uint eta=0; eta<41; ++eta){
      stringstream tempPUEt;
      tempPUEt  << "hTowEt_"     << pu << "_" << eta+1;
      hTowEtPU[pu][eta] = (TH1D*)file->Get(tempPUEt.str().c_str());
    }
  }  

  ofstream one;
  ofstream p3;
  ofstream p5;

  one.open("one.txt");
  p3.open("p3.txt");
  p5.open("p5.txt");
      
  stringstream intro;
  intro << "# address to et sum tower Et threshold LUT\n" \
    "# maps 11 bits to 9 bits\n" \
    "# 11 bits = (compressedPileupEstimate << 6) | abs(ieta)\n" \
    "# compressedPileupEstimate is unsigned 5 bits, abs(ieta) is unsigned 6 bits\n" \
    "# data: tower energy threshold returned has 9 bits \n" \
    "# anything after # is ignored with the exception of the header\n" \
    "# the header is first valid line starting with #<header> versionStr nrBitsAddress nrBitsData </header>\n" \
    "#<header> v1 11 9 </header>\n"; 

  one.write(intro.str().c_str(), intro.str().length() );
  p3.write(intro.str().c_str(), intro.str().length() ); 
  p5.write(intro.str().c_str(), intro.str().length() );
  
  int addr = 0;

  int lastFilled = 0;

  for(uint pu=0;pu<puBins;pu++){
    for(int eta=-1; eta<64; ++eta){	
	
      if(eta==28) continue;

      if(eta==-1){
	one << addr  << " 0             # nTT4 = " << pu*5 << "-" << pu*5+5 << " ieta = 0\n";
	p3 << addr   << " 0             # nTT4 = " << pu*5 << "-" << pu*5+5 << " ieta = 0\n";
	p5 << addr   << " 0             # nTT4 = " << pu*5 << "-" << pu*5+5 << " ieta = 0\n";
	++addr;
	continue;
      }

      if(pu < 14 && eta<41){
	one << addr  << " " << 0  << "             # ieta = " << eta+1 << "\n";
	p3 << addr   << " " << 0  << "             # ieta = " << eta+1 << "\n";
	p5 << addr   << " " << 0  << "             # ieta = " << eta+1 << "\n";
	++addr;  
	continue;
      }
	
      if(eta>40){
	one << addr  << " 0 #dummy\n";
	p3 << addr   << " 0 #dummy\n";
	p5 << addr   << " 0 #dummy\n";
	++addr;  
	continue;
      }
	
      double pass = 0;
      double d1(999.),dp3(999.),dp5(999.);
      double t1(999.),tp3(999.),tp5(999.);

      int nBins = hTowEtPU[pu][eta]->GetNbinsX();
      
      for(uint t=0; t<nBins; t++){
	int puHist = pu;
	if(hTowEtPU[pu][eta]->Integral(1,nBins)==0){
	  if(lastFilled == 0){
	    t1=tp3=tp5=0;
	    break;
	  }else puHist = lastFilled;
	}else{
	    lastFilled = pu;
	}

	pass = hTowEtPU[puHist][eta]->Integral(t+1,nBins)/hTowEtPU[puHist][eta]->Integral(1,nBins);
	if( abs(pass-0.01) < d1  ){
	  t1  = t;
	  d1 = pass - 0.01;
	}
	if( abs(pass-0.003) < dp3  ){
	  tp3  = t;
	  dp3 = pass - 0.003;
	}
	if( abs(pass-0.005) < dp5  ){
	  tp5  = t;
	  dp5 = pass - 0.005;
	}
      }

      //int sat = 32;

      // double rai = 1.2;
      // double div = 40;
      // double off = 0.2;

      // t1   = round( t1  *((pow(float(pu),rai)/div)+off) );
      // tp3   = round( tp3  *((pow(float(pu),rai)/div)+off) );
      // tp5  = round( tp5 *((pow(float(pu),rai)/div)+off) );

      //if(eta==27) cout << pu << "   " << ((pow(float(pu),rai)/div)+off) << endl; 

      // if(eta<15){
      // 	t1 = round(t1*(((double)eta)/14));
      // 	tp3 = round(tp3*(((double)eta)/14));
      // 	tp5 = round(tp5*(((double)eta)/14));
      // }
      
      //if(t1>sat)  t1 =sat;
      //if(tp3>sat)  tp3 =sat;
      //if(tp5>sat)   tp5=sat;


      one  << addr  << " "  << t1   << "             # ieta = " << eta+1 << "\n";
      p3   << addr  << " "  << tp3  << "             # ieta = " << eta+1 << "\n";
      p5   << addr  << " "  << tp5  << "             # ieta = " << eta+1 << "\n";          
      	  
      ++addr;

    }   
  }

  one.close();
  p3.close();
  p5.close();

}


// 1d formatter
void formatPlot1D(TH1D* plot1d, int colour){

  plot1d->GetXaxis()->SetTitleOffset(1.2);
  plot1d->GetYaxis()->SetTitleOffset(1.2);
  plot1d->SetMarkerStyle(7);
  plot1d->SetMarkerColor(colour);
  plot1d->SetLineColor(colour);
  plot1d->SetLineWidth(2);
  //plot1d->SetStats(false);

}

//2d formatter
void formatPlot2D(TH2D* plot2d){

  plot2d->GetXaxis()->SetTitleOffset(1.4);
  plot2d->GetYaxis()->SetTitleOffset(1.4);
  //plot2d->SetStats(false);
  
}


//main plotting function
void doZeroBiasPUStudy(bool doTow, bool doLUT, bool doFit){

  gStyle->SetStatW(0.1);
  gStyle->SetOptStat("ne");
  gStyle->SetOptFit(0001);

  //output filename
  string outFilename = "zb2018_metA.root";
  vector<int> puBinBs   = {0,0,0,0,0,0,0,0,1,5,10,14,18,23,27,32,36,41,45,50,56,62,68,74,80,86,93,99,105,111,117,123,999};

  int puBins = puBinBs.size()-1;

  TFile* file = TFile::Open( outFilename.c_str() );

  //check file exists
  if(doLUT || doFit){
    if (file==0){
      cout << "TERMINATE: input file does not exist: " << outFilename << endl;
      return;
    }
    if(doFit){
      TH1D* prof = (TH1D*)file->Get("hProfNVtxNTowemu4");
      fitProfile(prof);
      return;
    }
    if(doLUT){
      makeLUT(file, puBins);
      return;
    }  
  }

  //input ntuple
  string  inputFile01 = "root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/bundocka/ZeroBias/zbLUT_2018/180508_101338/0000/L1Ntuple_*.root";
  
  //check file doesn't exist
  if (file!=0){
    cout << "TERMINATE: not going to overwrite file " << outFilename << endl;
    return;
  }

  cout << "Loading up the TChain..." << endl;
  TChain * eventTree = new TChain("l1EventTree/L1EventTree");
  eventTree->Add(inputFile01.c_str());
  TChain * vtxTree = new TChain("l1RecoTree/RecoTree");
  vtxTree->Add(inputFile01.c_str());
  TChain * treeL1Towemu = new TChain("l1CaloTowerEmuTree/L1CaloTowerTree");
  treeL1Towemu->Add(inputFile01.c_str());
  TChain * treeTPhw = new TChain("l1CaloTowerTree/L1CaloTowerTree");
  treeTPhw->Add(inputFile01.c_str());

  L1Analysis::L1AnalysisEventDataFormat           *event_ = new L1Analysis::L1AnalysisEventDataFormat();
  eventTree->SetBranchAddress("Event", &event_);
  L1Analysis::L1AnalysisRecoVertexDataFormat        *vtx_ = new L1Analysis::L1AnalysisRecoVertexDataFormat();
  vtxTree->SetBranchAddress("Vertex", &vtx_);
  L1Analysis::L1AnalysisL1CaloTowerDataFormat   *l1Towemu_ = new L1Analysis::L1AnalysisL1CaloTowerDataFormat();
  treeL1Towemu->SetBranchAddress("L1CaloTower", &l1Towemu_);
  L1Analysis::L1AnalysisCaloTPDataFormat   *l1TPhw_ = new L1Analysis::L1AnalysisCaloTPDataFormat();
  treeTPhw->SetBranchAddress("CaloTP", &l1TPhw_);


  //initialise histograms
  TH1D* hNTow_emu = new TH1D("nTower_emu", ";nTowers; # Events",  150, -0.5, 1499.5);  
  TH1D* hNTow4_emu = new TH1D("nTower_emu4", ";nTowers; # Events",  100, -0.5, 499.5);  
  TH1D* hNVtx = new TH1D("nVtx_reco", ";nVtx; # Events", 120, -0.5, 119.5);
  
  TH2D* hNTowemuVsNVtx = new TH2D("nTowemu_vs_nVtx", ";nRecoVtx; nL1Tow", 100, -0.5, 99.5, 125, -0.5, 1499.5);
  TH2D* hNTowemuVsNVtx4 = new TH2D("nTowemu_vs_nVtx4", ";nRecoVtx; nL1Tow |i#eta|<5", 70, -0.5, 69.5, 50, -0.5, 199.5);  
  
  TProfile* hProfNVtxNTowemu = new TProfile("hProfNVtxNTowemu",";nVtx; <nTT>", 60,-0.5,119.5);
  TProfile* hProfNVtxNTowemu4 = new TProfile("hProfNVtxNTowemu4",";nVtx; <nTT> abs(i#eta)<5", 128,-0.5,127.5);
  TProfile* hProfNTowemuNVtx4 = new TProfile("hProfNTowemuNVtx4",";<nTT> abs(i#eta)<5;nVtx", 40,0,200);

  TH1D* hAllTowEtemu = new TH1D("towerEtEmu", ";Tower E_{T}; # Towers", 40, -0.5, 39.5);
  TH1D* hAllTowEtaemu = new TH1D("towerEtaEmu", ";Tower E_{T}; # Towers", 100, -50.5, 49.5);

  TProfile* hProfTowEtEta = new TProfile("hProfTowEtEta",";Tower i#eta;<Tower E_{T}> (GeV)", 100, -50.5, 49.5);
  vector<TProfile*>  hProfTowEtEtaPU(puBins);

  TProfile* hProfHCALTPEtEta = new TProfile("hProfHCALTPEtEta",";TP i#eta;<TP E_{T}> (GeV)", 100, -50.5, 49.5);
  vector<TProfile*>  hProfHCALTPEtEtaPU(puBins);

  TProfile* hProfECALTPEtEta = new TProfile("hProfECALTPEtEta",";TP i#eta;<TP E_{T}> (GeV)", 100, -50.5, 49.5);
  vector<TProfile*>  hProfECALTPEtEtaPU(puBins);

  TH1D* hTowEt = new TH1D("hTowEt",";Tower E_{T}; #Towers", 512, 0, 256);
  vector<vector<TH1D*> > hTowEtPU(puBins, vector<TH1D*> (41));

  TH1D* hECALTPEt = new TH1D("hECALTPEt",";ECAL TP E_{T}; #TPs", 256, 0, 128);
  vector<vector<TH1D*> > hECALTPEtPU(puBins, vector<TH1D*> (28));

  TH1D* hHCALTPEt = new TH1D("hHCALTPEt",";HCAL TP E_{T}; #TPs", 256, 0, 128);
  vector<vector<TH1D*> > hHCALTPEtPU(puBins, vector<TH1D*> (41));

  TH2D* hTowEtaPhi = new TH2D("hTowEtaPhi","; eta; phi", 100, -50, 50, 72, 0, 72);
  
  for(uint pu=0; pu<puBins;++pu){
    stringstream tempPU;
    stringstream tempPUE;
    stringstream tempPUH;
    stringstream tempPUEt;
    stringstream tempPUEEt;
    stringstream tempPUHEt;
    
    tempPU << "hProfTowEtEta_" << pu;
    tempPUE << "hProfECALTPEtEta_" << pu;
    tempPUH << "hProfHCALTPEtEta_" << pu;

    hProfTowEtEtaPU[pu] = new TProfile(tempPU.str().c_str(),";Tower i#eta;<Tower E_{T}> (GeV)", 100, -50.5, 49.5);
    hProfECALTPEtEtaPU[pu] = new TProfile(tempPUE.str().c_str(),";ECAL TP i#eta;<TP E_{T}> (GeV)", 100, -50.5, 49.5);
    hProfHCALTPEtEtaPU[pu] = new TProfile(tempPUH.str().c_str(),";HCAL TP i#eta;<TP E_{T}> (GeV)", 100, -50.5, 49.5);
    
    for(uint eta=0; eta<41; eta++){ 
      stringstream tempPUEt;
      stringstream tempPUEEt;
      stringstream tempPUHEt;
      tempPUEt  << "hTowEt_"     << pu << "_" << eta+1;
      tempPUEEt << "hECALTPEt_" << pu << "_" << eta+1;
      tempPUHEt << "hHCALTPEt_" << pu << "_" << eta+1;

      hTowEtPU[pu][eta] = new TH1D(tempPUEt.str().c_str(),";Tower E_{T} (GeV); #Towers", 512, 0, 256);
      if(eta<28) hECALTPEtPU[pu][eta] = new TH1D(tempPUEEt.str().c_str(),";ECAL TP E_{T} (GeV); #TPs", 256, 0, 128);
      hHCALTPEtPU[pu][eta] = new TH1D(tempPUHEt.str().c_str(),";HCAL TP E_{T} (GeV); #TPs", 256, 0, 128);
    }
  }
      
  
  //initialise some variables
  int nTowemu(-1), nVtx(-1), nTowemu4(-1);
  int nVtxBin(-1);
  int nHCALTP(-1), nECALTP(-1);
  float towEt(-1);
  vector<int> towOcc(41);
  

  //main loop

  // get number of entries
  Long64_t nentries;
  nentries = treeL1Towemu->GetEntries();

  // Number of events to run over
  int nEvents = nentries; // lol

  for (Long64_t jentry=0; jentry<nEvents; jentry++){
  
    nTowemu = -1;
    nVtx = -1; 
    nTowemu4 = -1;
    nVtxBin = -1;
    nHCALTP = -1;
    nECALTP = -1;
    towEt = -1;
    fill(towOcc.begin(), towOcc.end(), 0);
 
    //counter
    if((jentry%10000)==0) cout << "Done " << jentry  << " events of " << nEvents << endl;
    if((jentry%1000)==0) cout << "." << flush;

    eventTree->GetEntry(jentry);
    vtxTree->GetEntry(jentry);
    nVtx = vtx_->nVtx;

    hNVtx->Fill(nVtx);

    if(nVtx==0) continue;

    treeL1Towemu->GetEntry(jentry);
    nTowemu = l1Towemu_->nTower;

    if(!doTow){
      //treeTPhw->GetEntry(jentry); 
      //nHCALTP = l1TPhw_->nHCALTP;
      //nECALTP = l1TPhw_->nECALTP;    
      for(uint puIt=0; puIt<puBins;puIt++){
	if(nVtx >= puBinBs[puIt] && nVtx < puBinBs[puIt+1]){
	  nVtxBin = puIt;
	  break;
	}
      }
    }

    
    for(uint towIt=0; towIt<nTowemu; ++towIt){

      hTowEtaPhi->Fill(l1Towemu_->ieta[towIt], l1Towemu_->iphi[towIt]);
      if(abs(l1Towemu_->ieta[towIt])<5) nTowemu4++;
      if(!doTow){ 
	towOcc[abs(l1Towemu_->ieta[towIt])-1] +=1;
	towEt = l1Towemu_->iet[towIt];
	hTowEt->Fill(towEt/2);
	if(towEt/2 > 120) cout << "Tow Et = " << towEt << endl;
	hTowEtPU[nVtxBin][abs(l1Towemu_->ieta[towIt])-1]->Fill(towEt/2);
	hProfTowEtEta->Fill(l1Towemu_->ieta[towIt],l1Towemu_->iet[towIt]);
	hProfTowEtEtaPU[nVtxBin]->Fill(l1Towemu_->ieta[towIt],l1Towemu_->iet[towIt]);
      }
    }

    if(!doTow){ 
      for(uint eta=0;eta<41;eta++){
	hTowEtPU[nVtxBin][eta]->Fill(0.,(144-towOcc[eta]));
      }
    
      
      // //fill ECAL TP histos
      // for(uint tpIt=0; tpIt<nECALTP; ++tpIt){
      //   hECALTPEt->Fill(l1TPhw_->ecalTPet[tpIt]);
      //   hECALTPEtPU[nVtxBin][abs(l1TPhw_->ecalTPieta[tpIt])-1]->Fill(l1TPhw_->ecalTPet[tpIt]);
      //   hProfECALTPEtEta->Fill(l1TPhw_->ecalTPieta[tpIt],l1TPhw_->ecalTPet[tpIt]);
      //   hProfECALTPEtEtaPU[nVtxBin]->Fill(l1TPhw_->ecalTPieta[tpIt],l1TPhw_->ecalTPet[tpIt]);
      // }
      // //fill HCAL TP histos
      // for(uint tpIt=0; tpIt<nHCALTP; ++tpIt){
      //   hHCALTPEt->Fill(l1TPhw_->hcalTPet[tpIt]);
      //   hHCALTPEtPU[nVtxBin][abs(l1TPhw_->hcalTPieta[tpIt])-1]->Fill(l1TPhw_->hcalTPet[tpIt]);
      //   hProfHCALTPEtEta->Fill(l1TPhw_->hcalTPieta[tpIt],l1TPhw_->hcalTPet[tpIt]);
      //   hProfHCALTPEtEtaPU[nVtxBin]->Fill(l1TPhw_->hcalTPieta[tpIt],l1TPhw_->hcalTPet[tpIt]);
      // }

    }

    //fill emulated tower histos
    hNTowemuVsNVtx->Fill(nVtx,nTowemu);
    hProfNVtxNTowemu->Fill(nVtx,nTowemu);
    hNTow_emu->Fill(nTowemu);
    hNTow4_emu->Fill(nTowemu4);
    hNTowemuVsNVtx4->Fill(nVtx,nTowemu4);
    hProfNVtxNTowemu4->Fill(nVtx,nTowemu4);
    hProfNTowemuNVtx4->Fill(nTowemu4,nVtx);

  }
     
  //end event loop, now plot histos  
  TFile outFile( outFilename.c_str() , "new");

  TCanvas* canvas = new TCanvas("canvas","",600,600);
  TLegend* legend = new TLegend(0.5,0.6,0.7,0.88);
  
  legend->SetLineColor(0);

  stringstream saveName("");
 
  formatPlot1D(hNTow_emu,4); 
  formatPlot1D(hNTow4_emu,4); 
  formatPlot1D(hNVtx,4);
  formatPlot2D(hTowEtaPhi);

  hTowEtaPhi->Draw("colz");
  hTowEtaPhi->Write();
  canvas->SaveAs("towEtaPhi.pdf");

  formatPlot2D(hNTowemuVsNVtx);
 
  hNTow_emu->Draw();
  hNTow_emu->Write();
  canvas->SaveAs("nTowemu.pdf");

  hNTow4_emu->Draw();
  hNTow4_emu->Write();
  canvas->SaveAs("nTowemu4.pdf");
  
  hNVtx->Draw();
  hNVtx->Write();
  canvas->SaveAs("nVtx.pdf");

  hNTowemuVsNVtx->Draw("colz");
  hNTowemuVsNVtx->Write();
  canvas->SaveAs("nTowemuVsNVtx.pdf");

  hNTowemuVsNVtx4->Draw("colz");
  hNTowemuVsNVtx4->Write();
  canvas->SaveAs("nTowemuVsNVtx4.pdf");

  hProfNVtxNTowemu->SetMarkerStyle(7);
  hProfNVtxNTowemu->Draw("");
  hProfNVtxNTowemu->Write();
  canvas->SaveAs("ProfNVtxNTowemu.pdf");

  hProfNVtxNTowemu4->SetMarkerStyle(7);  
  hProfNVtxNTowemu4->Draw("");
  hProfNVtxNTowemu4->Write();
  canvas->SaveAs("ProfNVtxNTowemu4.pdf");

  fitProfile(hProfNTowemuNVtx4);

  if(!doTow){ 

    formatPlot1D(hTowEt,4);
    //formatPlot1D(hECALTPEt,4);
    //formatPlot1D(hHCALTPEt,4);
    
    hProfTowEtEta->SetMarkerStyle(7);
    hProfTowEtEta->SetMaximum(4.0);
    hProfTowEtEta->Scale(0.5);
    hProfTowEtEta->Draw("");
    hProfTowEtEta->Write();
    canvas->SaveAs("ProfTowEtEta.pdf");
      

    for(uint i=0;i<hProfTowEtEtaPU.size();i++){
      hProfTowEtEtaPU[i]->SetMarkerStyle(7);
      hProfTowEtEtaPU[i]->SetMaximum(4.0);
      hProfTowEtEtaPU[i]->Scale(0.5);
      hProfTowEtEtaPU[i]->Draw("");
      hProfTowEtEtaPU[i]->Write();
      stringstream fn("");
      fn << "ProfTowEtEtaPU_" << i << ".pdf";
      //canvas->SaveAs(fn.str().c_str());
    }
    hProfECALTPEtEta->SetMarkerStyle(7);
    hProfECALTPEtEta->SetMaximum(4.0);
    hProfECALTPEtEta->Draw("");
    hProfECALTPEtEta->Write();
    //canvas->SaveAs("ProfECALTPEtEta.pdf");

    for(uint i=0;i<hProfECALTPEtEtaPU.size();i++){
      hProfECALTPEtEtaPU[i]->SetMarkerStyle(7);
      hProfECALTPEtEtaPU[i]->SetMaximum(4.0);
      hProfECALTPEtEtaPU[i]->Draw("");
      hProfECALTPEtEtaPU[i]->Write();
      stringstream fn("");
      fn << "ProfECALTPEtEtaPU_" << i << ".pdf";
      //canvas->SaveAs(fn.str().c_str());
    }

    hProfHCALTPEtEta->SetMarkerStyle(7);
    hProfHCALTPEtEta->SetMaximum(4.0);
    hProfHCALTPEtEta->Draw("");
    hProfHCALTPEtEta->Write();
    //canvas->SaveAs("ProfHCALTPEtEta.pdf");

    for(uint i=0;i<hProfHCALTPEtEtaPU.size();i++){
      hProfHCALTPEtEtaPU[i]->SetMarkerStyle(7);
      hProfHCALTPEtEtaPU[i]->SetMaximum(4.0);
      hProfHCALTPEtEtaPU[i]->Draw("");
      hProfHCALTPEtEtaPU[i]->Write();
      stringstream fn("");
      fn << "ProfHCALTPEtEtaPU_" << i << ".pdf";
      //canvas->SaveAs(fn.str().c_str());
    }
  
    canvas->SetLogy(1);

    hTowEt->Draw("");
    hTowEt->Write("");
    canvas->SaveAs("TowEt.pdf");

    hECALTPEt->Draw("");
    hECALTPEt->Write("");
    //canvas->SaveAs("ECALTPEt.pdf");

    hHCALTPEt->Draw("");
    hHCALTPEt->Write("");
    //canvas->SaveAs("HCALTPEt.pdf");

  
    for(uint pu=0;pu<puBins;pu++){
      for(uint eta=0; eta<41; ++eta){
      
    	stringstream etaS("");
    	etaS << "_Eta" << eta+1;
	  
	formatPlot1D(hTowEtPU[pu][eta],4);
	hTowEtPU[pu][eta]->GetXaxis()->SetLimits(0.,50.);
	hTowEtPU[pu][eta]->Draw("");
	hTowEtPU[pu][eta]->Write();
	stringstream fn("");
	fn << "TowEtPU_" << pu << etaS.str() << ".pdf";
	//canvas->SaveAs(fn.str().c_str());

	// for(uint pu=0;pu<hECALTPEtPU.size();pu++){
	//   formatPlot1D(hECALTPEtPU[pu][eta],4);
	//   hECALTPEtPU[pu][eta]->Draw("");
	//   hECALTPEtPU[pu][eta]->Write();
	//   stringstream fn("");
	//   fn << "ECALTPEtPU_" << pu << etaS.str() << ".pdf";
	//   //canvas->SaveAs(fn.str().c_str());
	// }
     
	// for(uint pu=0;pu<hHCALTPEtPU.size();pu++){
	//   formatPlot1D(hHCALTPEtPU[pu][eta],4);
	//   hHCALTPEtPU[pu][eta]->Draw("");
	// hHCALTPEtPU[pu][eta]->Write();
	// stringstream fn("");
	// fn << "HCALTPEtPU_" << pu << etaS.str() << ".pdf";
	// //canvas->SaveAs(fn.str().c_str());
	// }
      }
    }
    
    
    
    
    canvas->Close();
    outFile.Close();
    
  }
}
