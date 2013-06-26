#ifndef UTIL
#define UTIL
#include <utility>
#endif

#ifndef MAP
#define MAP
#include <map>
#endif

#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

template <class T>
inline std::string to_string (const T& t)
{
  std::stringstream ss; 
  ss << t;
  return ss.str();
}


void getEfficiency(float efficientEvents, float allEvents, std::vector<float> &efficiencyResult);
void getType(int iE, int iS, int iR, float & verticalScale);

void computeEfficiencies(){
  //TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  //cout << tmp[0] << endl;
  char *file_name = "cscHists_bigger.root";
  TFile *f1=
    (TFile*)gROOT->GetListOfFiles()->FindObject(file_name);
  if (!f1){
    TFile *f1 = new TFile(file_name);
  }
  // All files in a vector
  std::vector< TFile * > DataFiles;
  DataFiles.push_back(f1);
  
  string mytitle;
  char *myTitle;
  
  const Int_t nx = 36;
  const Int_t ny = 16;
  
  char *chambers[nx]  = {"01","02","03","04","05","06","07","08","09","10",
			 "11","12","13","14","15","16","17","18","19","20",
			 "21","22","23","24","25","26","27","28","29","30",
			 "31","32","33","34","35","36"};
  char *types[ny] = {"ME-41","ME-32","ME-31","ME-22","ME-21","ME-13","ME-12","ME-11",
		     "ME+11","ME+12","ME+13","ME+21","ME+22","ME+31","ME+32","ME+41"};
  
  
  TH2F *h_alctEfficiency = new TH2F("h_alctEfficiency","ALCT efficiency; chamber number  ",
				    nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_clctEfficiency = new TH2F("h_clctEfficiency","CLCT efficiency; chamber number  ",
				    nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_corrlctEfficiency = new TH2F("h_corrlctEfficiency","CorrLCT efficiency; chamber number  ",
				       nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_rhEfficiency = new TH2F("h_rhEfficiency","RecHit efficiency; chamber number ",
				  nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_simrhEfficiency = new TH2F("h_simrhEfficiency","RecHit efficiency (simhit based); chamber number ",
				  nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_segEfficiency = new TH2F("h_segEfficiency","Segment efficiency; chamber number  ",
				   nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  
  TH2F *h_stripEfficiency = new TH2F("h_stripEfficiency","Strip efficiency; chamber number  ",
				     nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_wireEfficiency = new TH2F("h_wireEfficiency","Wire group efficiency; chamber number  ",
				    nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);

  for(int i=0;i<nx;++i){
    h_alctEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_clctEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_corrlctEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_rhEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_simrhEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_segEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_stripEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
    h_wireEfficiency->GetXaxis()->SetBinLabel(i+1,chambers[i]);
  }
  for(int i=0;i<ny;++i){
    h_alctEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_clctEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_corrlctEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_rhEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_simrhEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_segEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_stripEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
    h_wireEfficiency->GetYaxis()->SetBinLabel(i+1,types[i]);
  }
  
  //TH1F *h_alct_theta_Efficiency = new TH1F("alct_theta_Efficiency",
  //				   "ALCT efficiency vs local theta;local theta;efficiency",
  //				   );
  
  TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  TCanvas *c1 = new TCanvas("c1", "canvas 1",16,30,700,500);
  gStyle->SetOptStat(0);
  gStyle->SetErrorX(0);
  gPad->SetFillColor(0);
  gStyle->SetPalette(1);
  
  TH1F *data_p1;
  TH1F *data_p2;
  TH1F *data_p3;
  TH1F *data_p4;

  TH1F * h_Chi2_ndf;
  TH1F * h_hitsInSegm;
  TH1F * h_Residual_Segm;



  TH1F * hV_All_alct_dydz[ny];
  TH1F * hV_Eff_alct_dydz[ny];
  TH1F * hV_All_clct_dxdz[ny];
  TH1F * hV_Eff_clct_dxdz[ny];

  TH1F * h_All_alct_dydz;
  TH1F * h_Eff_alct_dydz;

  TH1F * h_attachment_AllHits;
  TH1F * h_attachment_EffHits;

  bool nextLoop[ny];
  for(int i=0;i<ny;++i){
    nextLoop[i] = false;
  }
  //std::cout<<" ny = "<<ny<<" size = "<<  nextLoop.size()<<std::endl;

  string dirName;
  char * charName;
  char * charName_2;
  char * charName_3;
  char * charName_4;
  char * charName_5;
  string histo_1, histo_2, histo_3, histo_4, histo_5;
  int firstSt_it = 0;
  //TH1F *sum_histo;
  int nEfficient = 0;
  int nAll = 0;
  std::vector<float> efficiencyResult(2);
  float verticalScale;
  int iterations = 0;
  for(int iE=1;iE<3;++iE){
    for(int iS=1;iS<5;++iS){
      dirName = Form("Stations__E%d_S%d",iE,iS);

      histo_1 = dirName + "/" + Form("segmentChi2_ndf_St%d",iS);
      histo_2 = dirName + "/" + Form("hitsInSegment_St%d",iS);
      histo_3 = dirName + "/" + Form("ResidualSegments_St%d",iS);
      charName = histo_1.c_str();
      charName_2 = histo_2.c_str();
      charName_3 = histo_3.c_str();
      data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
      data_p2 =(TH1F*)(DataFiles[0]->Get(charName_2));
      data_p3 =(TH1F*)(DataFiles[0]->Get(charName_3));

      if(!firstSt_it){
	h_Chi2_ndf = (TH1F *)data_p1->Clone();
	
	h_hitsInSegm = (TH1F *)data_p2->Clone();
	h_Residual_Segm = (TH1F *)data_p3->Clone();
	//sum_histo = (TH1F *)data_p1->Clone();
	
      }
      else{
	
	h_Chi2_ndf->Add(data_p1);
	h_hitsInSegm->Add(data_p2);
	h_Residual_Segm->Add(data_p3);
	//sum_histo->Add(data_p1);
	
      }
      
      ++firstSt_it;
      for(int iR=1;iR<4;++iR){
	if(1!=iS && iR>2){
	  continue;
	}
	else if(2==iR && 4==iS){
	  continue;
	}
	getType(iE,iS,iR, verticalScale);
	//std::cout<<" iE = "<<iE<< " iS = "<<iS<<" iR = "<<iR<<" verticalScale = "<<verticalScale<<std::endl;
	for(int iC=1;iC<37;++iC){
	  if(1!=iS && 1==iR && iC >18){
	    continue;
	  }
	  // Attachment 
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/AllSingleHits_Ch%d",iE,iS,iR,iC,iC);
	  histo_2 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/InefficientSingleHits_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
	  charName_2 = histo_2.c_str();
	  data_p2 =(TH1F*)(DataFiles[0]->Get(charName_2));
	  if(!iterations){
	    h_attachment_AllHits=(TH1F*)data_p1->Clone();
	    h_attachment_EffHits=(TH1F*)data_p1->Clone();
	    h_attachment_EffHits->Add(data_p2, -1.);
	  }
	  else{
	    h_attachment_AllHits->Add(data_p1);
	    h_attachment_EffHits->Add(data_p1);
	    h_attachment_EffHits->Add(data_p2, -1.);
	  }
	  // RecHits
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientRechits_good_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
	  std::cout<<" chamber name : "<<charName<<std::endl;
	  nEfficient = 0;
	  for(int iL=1;iL<7;++iL){
	    nEfficient +=data_p1->GetBinContent(iL+1);
	  }
	  nAll = 6 * data_p1->GetBinContent(8+1);
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  std::cout<<" Rechit eff = "<<efficiencyResult[0]<<" +-"<<efficiencyResult[1]<<std::endl;
	  h_rhEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_rhEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]); 
          //(simhit based)
          histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/Sim_Simhits_Ch%d",iE,iS,iR,iC,iC);
          histo_2 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/Sim_Rechits_Ch%d",iE,iS,iR,iC,iC);
          charName = histo_1.c_str();
          charName_2 = histo_2.c_str();
          data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
          data_p2 =(TH1F*)(DataFiles[0]->Get(charName_2));
	  std::cout<<" chamber name : "<<charName<<std::endl;
          nEfficient = 0;
          nAll =0;
          for(int iL=1;iL<7;++iL){
            nAll += data_p1->GetBinContent(iL+1);
            nEfficient += data_p2->GetBinContent(iL+1);
          }
          getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
          h_simrhEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
          h_simrhEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);


	  // Segments
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/digiAppearanceCount_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
	  nEfficient = data_p1->GetBinContent(2);
	  nAll = data_p1->GetBinContent(1) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_segEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_segEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  // LCT
	  nEfficient = data_p1->GetBinContent(4);
	  nAll = data_p1->GetBinContent(3) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_alctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_alctEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  //
	  nEfficient = data_p1->GetBinContent(6);
	  nAll = data_p1->GetBinContent(5) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_clctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_clctEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  //
	  nEfficient = data_p1->GetBinContent(8);
	  nAll = data_p1->GetBinContent(7) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_corrlctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_corrlctEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  //
	  
	  // Strips
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientStrips_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
	  nEfficient = 0;
	  
	  for(int iBin = 2;iBin<8;++iBin){
	    nEfficient +=  data_p1->GetBinContent(iBin);
	  }
	  nAll = 6*data_p1->GetBinContent(10);	  
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_stripEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_stripEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  // Wires
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientWireGroups_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName));
	  nEfficient = 0;
	  for(int iBin = 2;iBin<8;++iBin){
	    nEfficient +=  data_p1->GetBinContent(iBin);
	  }
	  nAll = 6*data_p1->GetBinContent(10);
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_wireEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  h_wireEfficiency->SetBinError(iC,int(verticalScale+0.5), efficiencyResult[1]);
	  
	  //ALCT
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientALCT_dydz_Ch%d",iE,iS,iR,iC,iC);
	  charName_2 = histo_1.c_str();
	  histo_2 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/InefficientALCT_dydz_Ch%d",iE,iS,iR,iC,iC);
	  charName_3 = histo_2.c_str();
	  data_p1 =(TH1F*)(DataFiles[0]->Get(charName_2));
	  data_p2 =(TH1F*)(DataFiles[0]->Get(charName_3));

	  histo_4 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientCLCT_dxdz_Ch%d",iE,iS,iR,iC,iC);
	  charName_4 = histo_4.c_str();
	  histo_5 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/InefficientCLCT_dxdz_Ch%d",iE,iS,iR,iC,iC);
	  charName_5 = histo_5.c_str();
	  data_p3 =(TH1F*)(DataFiles[0]->Get(charName_4));
	  data_p4 =(TH1F*)(DataFiles[0]->Get(charName_5));

	  int type = int(verticalScale+0.1);
	  if(!nextLoop[type]){
	    nextLoop[type] = true;
	    hV_All_alct_dydz[int(verticalScale+0.5)-1]=(TH1F*)data_p1->Clone();
	    hV_All_alct_dydz[int(verticalScale+0.5)-1]->Add(data_p2);
	    hV_Eff_alct_dydz[int(verticalScale+0.5)-1]=(TH1F*)data_p1->Clone();

	    hV_All_clct_dxdz[int(verticalScale+0.5)-1]=(TH1F*)data_p3->Clone();
	    hV_All_clct_dxdz[int(verticalScale+0.5)-1]->Add(data_p4);
	    hV_Eff_clct_dxdz[int(verticalScale+0.5)-1]=(TH1F*)data_p3->Clone();

	  }
	  else{
	    hV_All_alct_dydz[int(verticalScale+0.5)-1]->Add(data_p1);
	    hV_All_alct_dydz[int(verticalScale+0.5)-1]->Add(data_p2);
	    hV_Eff_alct_dydz[int(verticalScale+0.5)-1]->Add(data_p1);

	    hV_All_clct_dxdz[int(verticalScale+0.5)-1]->Add(data_p3);
	    hV_All_clct_dxdz[int(verticalScale+0.5)-1]->Add(data_p4);
	    hV_Eff_clct_dxdz[int(verticalScale+0.5)-1]->Add(data_p3);

	  }

	  ++iterations;
	}
      }
    }
  }

  TGraphAsymmErrors* h_attachment_efficiency =  new TGraphAsymmErrors(h_attachment_AllHits->GetNbinsX());
  h_attachment_efficiency->BayesDivide(h_attachment_EffHits,h_attachment_AllHits);
  mytitle = "Attachment efficiency (all chambers); layer number; efficiency ";
  myTitle = mytitle.c_str();
  h_attachment_efficiency->SetTitle(myTitle);
  mytitle = "h_attachment_efficiency";
  myTitle = mytitle.c_str();
  h_attachment_efficiency->SetName(myTitle);
  h_attachment_efficiency->Write();
  h_attachment_EffHits->Delete();
  h_attachment_AllHits->Delete();

  TGraphAsymmErrors* gV_effALCT_dydz[ny];
  TGraphAsymmErrors* gV_effCLCT_dxdz[ny];

  for(int i = 0;i<ny;++i){
    if(!i){
      h_All_alct_dydz=(TH1F*)hV_All_alct_dydz[i]->Clone();
      h_Eff_alct_dydz=(TH1F*)hV_Eff_alct_dydz[i]->Clone();
    }
    else{
      h_All_alct_dydz->Add(hV_All_alct_dydz[i]);
      h_Eff_alct_dydz->Add(hV_Eff_alct_dydz[i]);
    }
    gV_effALCT_dydz[i] = new TGraphAsymmErrors(hV_Eff_alct_dydz[i]->GetNbinsX());
    gV_effALCT_dydz[i]->BayesDivide(hV_Eff_alct_dydz[i],hV_All_alct_dydz[i]);
    mytitle = "Efficiency - ALCT vs dydz : " + to_string(types[i]);
    mytitle += "; local dydz (ME 3 and 4 flipped); efficiency ";
    myTitle = mytitle.c_str();
    gV_effALCT_dydz[i]->SetTitle(myTitle);
    mytitle = "g_effALCT_dydz_" + to_string(types[i]);
    myTitle = mytitle.c_str();
    gV_effALCT_dydz[i]->SetName(myTitle);
    hV_Eff_alct_dydz[i]->Delete();
    hV_All_alct_dydz[i]->Delete();
    gV_effALCT_dydz[i]->Write();


    gV_effCLCT_dxdz[i] = new TGraphAsymmErrors(hV_Eff_clct_dxdz[i]->GetNbinsX());
    gV_effCLCT_dxdz[i]->BayesDivide(hV_Eff_clct_dxdz[i],hV_All_clct_dxdz[i]);
    mytitle = "Efficiency - CLCT vs dxdz : " + to_string(types[i]);
    mytitle += "; dxdz (local); efficiency ";
    myTitle = mytitle.c_str();
    gV_effCLCT_dxdz[i]->SetTitle(myTitle);
    mytitle = "g_effCLCT_dxdz_" + to_string(types[i]);
    myTitle = mytitle.c_str();
    gV_effCLCT_dxdz[i]->SetName(myTitle);
    hV_Eff_clct_dxdz[i]->Delete();
    hV_All_clct_dxdz[i]->Delete();
    gV_effCLCT_dxdz[i]->Write();
  }

  TGraphAsymmErrors* g_effALCT_dydz;
  g_effALCT_dydz = new TGraphAsymmErrors(h_Eff_alct_dydz->GetNbinsX());
  g_effALCT_dydz->BayesDivide(h_Eff_alct_dydz,h_All_alct_dydz);
  mytitle = "Efficiency - ALCT vs dydz : All chambers";
  mytitle += "; local dydz (ME 3 and 4 flipped); efficiency ";
  myTitle = mytitle.c_str();
  g_effALCT_dydz->SetTitle(myTitle);
  mytitle = "g_effALCT_dydz_AllCh";
  myTitle = mytitle.c_str();
  g_effALCT_dydz->SetName(myTitle);
  h_Eff_alct_dydz->Delete();
  h_All_alct_dydz->Delete();
  g_effALCT_dydz->Write();
  
  
  //sum_All_alct_theta->Delete();
  //sum_Eff_alct_theta->Delete();
  h_rhEfficiency->Draw("colz");
  
  //f1->Close();
  //TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  //sum_histo->Write();
  //h_Chi2_ndf->Write();
  //h_hitsInSegm->Write();
  //h_Residual_Segm->Write();
  mytitle = "Chi2/ndf (segments) : All chambers; chi2/ndf; entries";
  myTitle = mytitle.c_str();
  h_Chi2_ndf->SetName(myTitle);
  mytitle = "h_Chi2_ndf_AllCh";
  myTitle = mytitle.c_str();
  h_Chi2_ndf->SetName(myTitle);

  mytitle = "Rechits in a segment : All chambers;number of rechits;entries";
  myTitle = mytitle.c_str();
  h_hitsInSegm->SetTitle(myTitle);
  mytitle = "h_hitsInSegm_AllCh";
  myTitle = mytitle.c_str();
  h_hitsInSegm->SetName(myTitle);

  mytitle = "Residuals (track to segment) : All chambers; resirual, cm;entries";
  myTitle = mytitle.c_str();
  h_Residual_Segm->SetTitle(myTitle);
  mytitle = "h_ResidInSegm_AllCh";
  myTitle = mytitle.c_str();
  h_Residual_Segm->SetName(myTitle);

  h_alctEfficiency->Write();
  h_clctEfficiency->Write();
  h_corrlctEfficiency->Write();

  h_rhEfficiency->Write();
  h_simrhEfficiency->Write();
  h_segEfficiency->Write();

  h_stripEfficiency->Write();
  h_wireEfficiency->Write();

  anaFile->Write();
  anaFile->Close();
  std::cout<<" Efficiency calculations done. Open the file efficiency.root to look at the efficiency histograms."<<std::endl;
  //f1->Close();
}

void getEfficiency(float efficientEvents, float allEvents, std::vector<float> &efficiencyResult){
  //---- Efficiency with binomial error
  float efficiency = 0.;
  float effError = 0.;
  if(fabs(allEvents)>0.000000001){
    efficiency = efficientEvents/allEvents;
    if(efficientEvents<allEvents){
      effError = sqrt( (1.-efficiency)*efficiency/allEvents );
    }
    else{
      double effTemp = (allEvents -1)/allEvents;
      if(allEvents<=1){
	effError = 1;
      }
      else{
	effError = sqrt( (1.-effTemp)*effTemp/allEvents );
      }
    }
  }
  efficiencyResult.clear();
  efficiencyResult.push_back(efficiency);
  efficiencyResult.push_back(effError);
}

void getType(int iE, int iS, int iR, float & verticalScale){
  if (1==iS){
    if(4==iR){
      //verticalScale = 0.5;
    }
    else if(1==iR){
      verticalScale = 0.5;
    }
    else if(2==iR){
      verticalScale = 1.5;
    }
    else if(3==iR){
      verticalScale = 2.5;
    }
  }
  else if (2==iS){
    if(1==iR){
      verticalScale = 3.5;
    }
    else if(2==iR){
      verticalScale = 4.5;
    }
  }
  else if(3==iS){
    if(1==iR){
      verticalScale = 5.5;
    }
    else if(2==iR){
      verticalScale = 6.5;
    }
  }
  else if( 4==iS){
    if(1==iR){
      verticalScale = 7.5;
    }
  }
  if(2==iE){
    verticalScale = - verticalScale;
  } 
  verticalScale +=8.5;
  //std:cout<<" IN: iE"<<iE<<" iS = "<<iS<<" iR = "<<iR<<" verticalScale = "<<verticalScale<<std::endl;
}
