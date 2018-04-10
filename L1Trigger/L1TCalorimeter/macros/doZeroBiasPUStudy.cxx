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


void makeLUT(string filename, int puBins){

  //check file doesn't exist
  TFile* file = TFile::Open(filename.c_str());
  if (file==0){
    cout << "TERMINATE: input file does not exist: " << filename << endl;
    return;
  }

  vector<vector<TH1D*> > hTowEtPU(puBins, vector<TH1D*> (41));

  for(uint pu=0;pu<puBins;pu++){
    for(uint eta=0; eta<41; ++eta){
      stringstream tempPUEt;
      tempPUEt  << "hTowEt_"     << pu << "_" << eta+1;
      hTowEtPU[pu][eta] = (TH1D*)file->Get(tempPUEt.str().c_str());
    }
  }  

  ofstream one;
  ofstream three;
  ofstream five;
  ofstream ten;

  one.open ("one.txt");
  three.open ("three.txt");
  five.open ("five.txt");
  ten.open ("ten.txt");
    
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
  three.write(intro.str().c_str(), intro.str().length() ); 
  five.write(intro.str().c_str(), intro.str().length() );
  ten.write(intro.str().c_str(), intro.str().length() );

  int addr = 0;

  for(uint pu=0;pu<puBins;pu++){
    for(int eta=-1; eta<64; ++eta){	
	
      if(eta==28) continue;

      if(eta==-1){
	one   << addr  << " 0\t# nTT4 = " << pu*5 << "-" << pu*5+5 << "  ieta = 0\n";
	three << addr  << " 0\t# nTT4 = " << pu*5 << "-" << pu*5+5 << "  ieta = 0\n";
	five  << addr  << " 0\t# nTT4 = " << pu*5 << "-" << pu*5+5 << "  ieta = 0\n";
	ten   << addr  << " 0\t# nTT4 = " << pu*5 << "-" << pu*5+5 << "  ieta = 0\n";
	++addr;
	continue;
      }

      if((eta<13 || pu < 6) && eta<41){
	one   << addr  << " " << 0  << "\t# ieta = " << eta+1 << "\n";
	three << addr  << " " << 0  << "\t# ieta = " << eta+1 << "\n";
	five  << addr  << " " << 0  << "\t# ieta = " << eta+1 << "\n";
	ten   << addr  << " " << 0  << "\t# ieta = " << eta+1 << "\n";
	++addr;  
	continue;
      }
	
      if(eta>40){
	one   << addr  << " 0\t# dummy\n";
	three << addr  << " 0\t# dummy\n";
	five  << addr  << " 0\t# dummy\n";
	ten   << addr  << " 0\t# dummy\n";
	++addr;  
	continue;
      }
	
      double pass = 0;
      double d1(999.),d3(999.),d5(999.),d10(999.);
      double t1(999.),t3(999.),t5(999.),t10(999.);      

      for(uint t=0; t<hTowEtPU[pu][eta]->GetNbinsX(); t++){
	if(hTowEtPU[pu][eta]->Integral(0,512)==0){
	  t1=t3=t5=t10=0;
	  break;
	}
	pass = hTowEtPU[pu][eta]->Integral(t,512)/hTowEtPU[pu][eta]->Integral(0,512);
	if( abs(pass-0.01) < d1  ){
	  t1  = t;
	  d1 = pass - 0.01;
	}
	if( abs(pass-0.03) < d3  ){
	  t3  = t;
	  d3 = pass - 0.03;
	}
	if( abs(pass-0.05) < d5  ){
	  t5  = t;
	  d5 = pass - 0.05;
	}
	if( abs(pass-0.1) < d10  ){
	  t10  = t;
	  d10 = pass - 0.1;
	}
      }

      t1  = round( t1 *(pow(float(pu),1.2)/64) );
      t3  = round( t3 *(pow(float(pu),1.2)/64) );
      t5  = round( t5 *(pow(float(pu),1.2)/64) );
      t10 = round( t10*(pow(float(pu),1.2)/64) );    
    
      one   << addr  << " " << t1  << "\t# ieta = " << eta+1 << "\n";
      three << addr  << " " << t3  << "\t# ieta = " << eta+1 << "\n";
      five  << addr  << " " << t5  << "\t# ieta = " << eta+1 << "\n";
      ten   << addr  << " " << t10 << "\t# ieta = " << eta+1 << "\n";
	  
      ++addr;

      //cout << std::fixed << setprecision(1) << "PU = " << pu << "\t eta = " << eta+1 << "\t 1% = " << t1/2 << "\t 3% = " << t3/2 << "\t 5% = " << t5/2 << "\t 10% = " << t10/2 << endl;
    }   
  }

  one.close();
  three.close();
  five.close();
  ten.close();

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
void doZeroBiasPUStudy(bool doTow, bool doLUT){

  //output filename
  string outFilename = "zbPUStudy.root";

  vector<int> puBinBs = {0,5,7,10,12,14,16,18,21,23,26,29,32,34,36,38,
			 40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,999};
 
  int puBins = puBinBs.size()-1;

  if(doLUT){
    makeLUT(outFilename, puBins);
    return;
  }  

  //input ntuple
  string  inputFile01 = "root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/bundocka/ZeroBias/zbLUT/180305_142059/0000/L1Ntuple_*.root";
  
  //check file doesn't exist
  TFile* fileCheck = TFile::Open( outFilename.c_str() );
  if (fileCheck!=0){
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

  gStyle->SetStatW(0.1);
  gStyle->SetOptStat("ne");
  gStyle->SetOptFit(0001);

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
  int nEvents = 5000;//nentries; // lol

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
      treeTPhw->GetEntry(jentry); 
      nHCALTP = l1TPhw_->nHCALTP;
      nECALTP = l1TPhw_->nECALTP;    
      for(uint puIt=0; puIt<puBins;puIt++){
	if(nVtx >= puBinBs[puIt] && nVtx <= puBinBs[puIt+1]){
	  nVtxBin = puIt;
	  break;
	}
      }
    }

    
    for(uint towIt=0; towIt<nTowemu; ++towIt){
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

  
  TH1D *hProfNTowemuNVtx4_fit2 = (TH1D*)hProfNTowemuNVtx4->Clone("hProfNTowemuNVtx4_fit2");
  TH1D *hProfNTowemuNVtx4_fit3 = (TH1D*)hProfNTowemuNVtx4->Clone("hProfNTowemuNVtx4_fit3");
  hProfNTowemuNVtx4->SetMarkerStyle(7);
  hProfNTowemuNVtx4_fit2->SetMarkerStyle(7);
  hProfNTowemuNVtx4_fit3->SetMarkerStyle(7);

  TF1 *fit1 = new TF1("fit1","[0]*x+[1]");
  fit1->SetParameter(0,0.5); 
  fit1->SetParameter(1,4);   
  fit1->SetRange(10,40);     
  hProfNTowemuNVtx4->Fit("fit1","R");
  TF1 *fit2 = new TF1("fit2","[0]*x+[1]");
  fit2->SetParameter(0,1.5); 
  fit2->SetParameter(1,0);   
  fit2->SetRange(40,60);
  hProfNTowemuNVtx4_fit2->Fit("fit2","R");
  TF1 *fit3 = new TF1("fit3","[0]*x+[1]");
  fit3->SetParameter(0,0.5);
  fit3->SetParameter(1,4);  
  fit3->SetRange(60,160);   
  hProfNTowemuNVtx4_fit3->Fit("fit3","R");
  hProfNTowemuNVtx4->Draw();  
  hProfNTowemuNVtx4_fit2->Draw("sames");  
  hProfNTowemuNVtx4_fit3->Draw("sames");  
  gPad->Update();
  TPaveStats *st1 = (TPaveStats*)hProfNTowemuNVtx4->FindObject("stats");
  TPaveStats *st2 = (TPaveStats*)hProfNTowemuNVtx4_fit2->FindObject("stats");
  TPaveStats *st3 = (TPaveStats*)hProfNTowemuNVtx4_fit3->FindObject("stats");
  st1->SetY1NDC(0.75);
  st1->SetY2NDC(0.95);
  st2->SetY1NDC(0.55);
  st2->SetY2NDC(0.75);
  st3->SetY1NDC(0.35);
  st3->SetY2NDC(0.55);  
  fit1->Draw("sames");
  fit2->Draw("sames");
  fit3->Draw("sames");
  hProfNTowemuNVtx4->Write();
  canvas->SaveAs("ProfNTowemuNVtx4.pdf");


  if(!doTow){ 

    formatPlot1D(hTowEt,4);
    formatPlot1D(hECALTPEt,4);
    formatPlot1D(hHCALTPEt,4);
    
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
      canvas->SaveAs(fn.str().c_str());
    }
    hProfECALTPEtEta->SetMarkerStyle(7);
    hProfECALTPEtEta->SetMaximum(4.0);
    hProfECALTPEtEta->Draw("");
    hProfECALTPEtEta->Write();
    canvas->SaveAs("ProfECALTPEtEta.pdf");

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
    canvas->SaveAs("ProfHCALTPEtEta.pdf");

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

    // hECALTPEt->Draw("");
    // hECALTPEt->Write("");
    // canvas->SaveAs("ECALTPEt.pdf");

    // hHCALTPEt->Draw("");
    // hHCALTPEt->Write("");
    // canvas->SaveAs("HCALTPEt.pdf");

  
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

	// if(eta<28){
	//   for(uint pu=0;pu<hECALTPEtPU.size();pu++){
	// 	formatPlot1D(hECALTPEtPU[pu][eta],4);
	// 	hECALTPEtPU[pu][eta]->Draw("");
	// 	hECALTPEtPU[pu][eta]->Write();
	// 	stringstream fn("");
	// 	fn << "ECALTPEtPU_" << pu << etaS.str() << ".pdf";
	// 	canvas->SaveAs(fn.str().c_str());
	//   }
	// }

	// for(uint pu=0;pu<hHCALTPEtPU.size();pu++){
	//   formatPlot1D(hHCALTPEtPU[pu][eta],4);
	//   hHCALTPEtPU[pu][eta]->Draw("");
	//   hHCALTPEtPU[pu][eta]->Write();
	//   stringstream fn("");
	//   fn << "HCALTPEtPU_" << pu << etaS.str() << ".pdf";
	//   canvas->SaveAs(fn.str().c_str());
	// }
      }
    }


    
    canvas->Close();
    outFile.Close();

    makeLUT(outFilename, puBins);
    
  }
}
