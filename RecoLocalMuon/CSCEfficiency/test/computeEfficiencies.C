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

void getEfficiency(float efficientEvents, float allEvents, std::vector<float> &efficiencyResult);
void getType(int iE, int iS, int iR, float & verticalScale);

void monHists(){
  //TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  //cout << tmp[0] << endl;
  char *file_name = "cscHists.root";
  TFile *f1=
    (TFile*)gROOT->GetListOfFiles()->FindObject(file_name);
  if (!f1){
    TFile *f1 = new TFile(file_name);
  }
  // All files in a vector
  std::vector< TFile * > DataFiles;
  DataFiles.push_back(f1);

  const Int_t nx = 36;
  const Int_t ny = 16;
  
  char *chambers[nx]  = {"01","02","03","04","05","06","07","08","09","10",
			 "11","12","13","14","15","16","17","18","19","20",
			 "21","22","23","24","25","26","27","28","29","30",
			 "31","32","33","34","35","36"};
  char *types[ny] = {"ME-41","ME-32","ME-31","ME-22","ME-21","ME-13","ME-12","ME-11",
		     "ME+11","ME+12","ME+13","ME+21","ME+22","ME+31","ME+32","ME+41"};


  TH2F *h_alctEfficiency = new TH2F("h_alctEfficiency","ALCT efficiency (in %)",
				   nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_clctEfficiency = new TH2F("h_clctEfficiency","CLCT efficiency (in %)",
				   nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_corrlctEfficiency = new TH2F("h_corrlctEfficiency","CorrLCT efficiency (in %)",
				   nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_rhEfficiency = new TH2F("h_rhEfficiency","RecHit efficiency (in %)",
				  nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);
  TH2F *h_segEfficiency = new TH2F("h_segEfficiency","Segment efficiency (in %)",
				   nx,0. + 0.5,float(nx) + 0.5, ny,0. + 0.5,float(ny) + 0.5);



  TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  TCanvas *c1 = new TCanvas("c1", "canvas 1",16,30,700,500);
  gStyle->SetOptStat(1110);
  gStyle->SetErrorX(0);
  gPad->SetFillColor(0);
  gStyle->SetPalette(1);

  TH1F *data_p1;
  string dirName;
  char * charName;
  string histo_1, histo_2, histo_3;
  int firstSt_it = 0;
  //TH1F *sum_histo;
  int nEfficient = 0;
  int nAll = 0;
  std::vector<float> efficiencyResult(2);
  float verticalScale;
  for(int iE=1;iE<3;++iE){
    for(int iS=1;iS<5;++iS){
      dirName = Form("Stations__E%d_S%d",iE,iS);
      histo_1 = dirName + "/" + Form("AllSegments_eta_St%d",iS);
      charName = histo_1.c_str();
      std::cout<<" charName = "<<charName<<std::endl;
      data_p1 =(TH1F*)(*(DataFiles[0]))->Get(charName);
      std::cout<<"entries = "<<data_p1->GetEntries() <<std::endl;
      if(!firstSt_it){
	//sum_histo = (TH1F *)data_p1->Clone();
      }
      else{
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
	std::cout<<" iE = "<<iE<< " iS = "<<iS<<" iR = "<<iR<<" verticalScale = "<<verticalScale<<std::endl;
	
	for(int iC=1;iC<37;++iC){
	  if(1!=iS && 1==iR && iC >18){
	    continue;
	  }
	  // RecHits
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/EfficientRechits_good_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  //std::cout<<" chamber charName = "<<charName<<std::endl;
	  data_p1 =(TH1F*)(*(DataFiles[0]))->Get(charName);
	  //if(5==iC){
	  std::cout<<" chamber charName = "<<charName<<std::endl;
	  nEfficient = 0;
	  for(int iL=1;iL<7;++iL){
	    nEfficient +=data_p1->GetBinContent(iL+1);
	    std::cout<<" iL = "<<iL<<" cont = "<<data_p1->GetBinContent(iL+1)<<std::endl;
	  }
	  nAll = 6 * data_p1->GetBinContent(8+1);
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  std::cout<<" eff = "<<efficiencyResult[0]<<" +-"<<efficiencyResult[1]<<std::endl;
	  std::cout<<" int(verticalScale+0.5) = "<<int(verticalScale+0.5)<<std::endl;
	  h_rhEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  //h_rhEfficiency->SetBinContent(chambers[iC],types[int(verticalScale+0.5)],efficiencyResult[0]);
	  //}

	  // Segments
	  histo_1 = Form("Chambers__E%d_S%d_R%d_Chamber_%d/digiAppearanceCount_Ch%d",iE,iS,iR,iC,iC);
	  charName = histo_1.c_str();
	  data_p1 =(TH1F*)(*(DataFiles[0]))->Get(charName);
	  //std::cout<<" bin 1 = "<<data_p1->GetBinContent(1) <<" bin 2 = "<<data_p1->GetBinContent(2)<<
	  //"bin 8 = "<<data_p1->GetBinContent(8)<<" bin 9 = "<<data_p1->GetBinContent(9)<<std::endl;
	  nEfficient = data_p1->GetBinContent(2);
	  nAll = data_p1->GetBinContent(1) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_segEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  // LCT
	  nEfficient = data_p1->GetBinContent(4);
	  nAll = data_p1->GetBinContent(3) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_alctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  //
	  nEfficient = data_p1->GetBinContent(6);
	  nAll = data_p1->GetBinContent(5) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_clctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  //
	  nEfficient = data_p1->GetBinContent(8);
	  nAll = data_p1->GetBinContent(7) + nEfficient;
	  getEfficiency(float(nEfficient), float(nAll), efficiencyResult);
	  h_corrlctEfficiency->SetBinContent(iC,int(verticalScale+0.5), efficiencyResult[0]);
	  //h_corrlctEfficiency->Fill(chambers[iC],types[int(verticalScale+0.5)],efficiencyResult[0]);
	  //h2->Fill(chambers[ibinX-1],types[ibinY-1],newCont);

	}
      }
    }
  }
  h_rhEfficiency->Draw("colz");
  
  //f1->Close();
  //TFile* anaFile = TFile::Open("efficiencies.root", "RECREATE"); // my output file
  //sum_histo->Write();
  h_alctEfficiency->Write();
  h_clctEfficiency->Write();
  h_corrlctEfficiency->Write();

  h_rhEfficiency->Write();
  h_segEfficiency->Write();

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
 std:cout<<" IN: iE"<<iE<<" iS = "<<iS<<" iR = "<<iR<<" verticalScale = "<<verticalScale<<std::endl;
}
