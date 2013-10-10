#include <vector>
#include <fstream>
#include <sstream>
#include "TTree.h"
#include "TFile.h"
#include "TMath.h"
#include "TH1F.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraphAsymmErrors.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TEfficiency.h"

#include "TStyle.h"

void makePlots(string name1, string name2, string name3){

  TFile * f1 = TFile::Open(name1.c_str());
  f1->cd();

  TH1F * pTRec_NoGEM = (TH1F*)gDirectory->Get("pTRec");
  TH1F * pTSim_NoGEM = (TH1F*)gDirectory->Get("pTSim");
  TH1F * pTRes_NoGEM = (TH1F*)gDirectory->Get("pTRes");
  TH1F * invPTRes_NoGEM = (TH1F*)gDirectory->Get("invPTRes");
  TH1F * pTDiff_NoGEM = (TH1F*)gDirectory->Get("pTDiff");
  TH1F * PSimEta_NoGEM = (TH1F*)gDirectory->Get("PSimEta");
  TH1F * PRecEta_NoGEM = (TH1F*)gDirectory->Get("PRecEta");
  TH1F * PDeltaEta_NoGEM = (TH1F*)gDirectory->Get("PDeltaEta");
  TH1F * PSimPhi_NoGEM = (TH1F*)gDirectory->Get("PSimPhi");
  TH1F * PRecPhi_NoGEM = (TH1F*)gDirectory->Get("PRecPhi");
  TH1F * NumSimTracks_NoGEM = (TH1F*)gDirectory->Get("NumSimTracks");
  TH1F * NumMuonSimTracks_NoGEM = (TH1F*)gDirectory->Get("NumMuonSimTracks");
  TH1F * NumRecTracks_NoGEM = (TH1F*)gDirectory->Get("NumRecTracks");
  TH2F * PtResVsPt_NoGEM = (TH2F*)gDirectory->Get("PtResVsPt");
  TH2F * InvPtResVsPt_NoGEM = (TH2F*)gDirectory->Get("InvPtResVsPt");
  TH2F * PtResVsEta_NoGEM = (TH2F*)gDirectory->Get("PtResVsEta");
  TH2F * InvPtResVsEta_NoGEM = (TH2F*)gDirectory->Get("InvPtResVsEta");
  TH2F * DPhiVsPt_NoGEM = (TH2F*)gDirectory->Get("DPhiVsPt");
  TH1F * DenPt_NoGEM = (TH1F*)gDirectory->Get("DenPt");
  TH1F * DenEta_NoGEM = (TH1F*)gDirectory->Get("DenEta");
  TH1F * NumPt_NoGEM = (TH1F*)gDirectory->Get("NumPt");
  TH1F * NumEta_NoGEM = (TH1F*)gDirectory->Get("NumEta");
  TH1F * PullGEM_NoGEM = (TH1F*)gDirectory->Get("PullGEMx");
  TH1F * PullCSC_NoGEM = (TH1F*)gDirectory->Get("PullCSC");
  TH1F * GEMRecHitEta_NoGEM = (TH1F*)gDirectory->Get("GEMRecHitEta");
  TH2F * DeltaCharge_NoGEM = (TH2F*)gDirectory->Get("DeltaCharge");
  TH2F * RecoPtVsSimPt_NoGEM = (TH2F*)gDirectory->Get("RecoPtVsSimPt");

  TH1F * NumPt_NoGEM2 = (TH1F*)NumPt_NoGEM->Clone();
  TH1F * NumEta_NoGEM2 = (TH1F*)NumEta_NoGEM->Clone();

  TH2F * PtResVsPt_NoGEM2 = (TH2F*)PtResVsPt_NoGEM->Clone();
  TH2F * InvPtResVsPt_NoGEM2 = (TH2F*)InvPtResVsPt_NoGEM->Clone();
  TH2F * PtResVsEta_NoGEM2 = (TH2F*)PtResVsEta_NoGEM->Clone();
  TH2F * InvPtResVsEta_NoGEM2 = (TH2F*)InvPtResVsEta_NoGEM->Clone();
  TH2F * DPhiVsPt_NoGEM2 = (TH2F*)DPhiVsPt_NoGEM->Clone();

  NumPt_NoGEM2->Divide(DenPt_NoGEM);
  NumEta_NoGEM2->Divide(DenEta_NoGEM);

  TProfile * prof1_NoGEM = PtResVsPt_NoGEM2->ProfileX();
  TProfile * prof2_NoGEM = InvPtResVsPt_NoGEM2->ProfileX();
  TProfile * prof1_2_NoGEM = PtResVsEta_NoGEM2->ProfileX();
  TProfile * prof2_2_NoGEM = InvPtResVsEta_NoGEM2->ProfileX();
  TProfile * prof2bis_NoGEM = InvPtResVsPt_NoGEM2->ProfileX("profile",-1,-1,"s");
  TProfile * prof3_NoGEM = DPhiVsPt_NoGEM2->ProfileX();

  TProfile * prof4_NoGEM = RecoPtVsSimPt_NoGEM->ProfileX();

  std::vector<TH1F*> vecResNoGEM;
  for(int i=1; i<=PtResVsPt_NoGEM->GetNbinsX(); i++){

  	TH1F * temp = (TH1F*)pTRes_NoGEM->Clone();
	int col = i;
	if(i==10) col=41;
	if(i==5) col=46;
	temp->SetLineColor(col);
	temp->SetLineWidth(2);

	for(int j=1; j<=PtResVsPt_NoGEM->GetNbinsY(); j++){

		float bin = PtResVsPt_NoGEM->GetBinContent(i,j);

		temp->SetBinContent(bin,i);
		//std::cout<<bin<<std::endl;

	}

	vecResNoGEM.push_back(temp);

  }

  TH1F * DeltaChargePercentage_NoGEM = new TH1F("DeltaChargePercentage_NoGEM","Frac. Wrong Charge (SIM-RECO)",10,0,1000);

  TH1F * numNoGem = (TH1F*)DeltaChargePercentage_NoGEM->Clone();
  TH1F * denNoGem = (TH1F*)DeltaChargePercentage_NoGEM->Clone();

  for(int i=1; i<=DeltaCharge_NoGEM->GetNbinsX(); i++){

	int num1 = DeltaCharge_NoGEM->GetBinContent(i,2);
	int num2 = DeltaCharge_NoGEM->GetBinContent(i,4); //zero
	int num3 = DeltaCharge_NoGEM->GetBinContent(i,6);

	numNoGem->SetBinContent(i,num1+num3);
	denNoGem->SetBinContent(i,num1+num2+num3);

	//cout<<num1<<" "<<num2<<" "<<num3<<endl;

  }

  TEfficiency* pEffCharge_NoGEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*numNoGem,*denNoGem))
  {
    	pEffCharge_NoGEM = new TEfficiency(*numNoGem,*denNoGem);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  /////////////////////////////////////////////////////////////////////////////////////

  TFile * f2 = TFile::Open(name2.c_str());
  f2->cd();
  TH1F * pTRec_GEM = (TH1F*)gDirectory->Get("pTRec");
  TH1F * pTSim_GEM = (TH1F*)gDirectory->Get("pTSim");
  TH1F * pTRes_GEM = (TH1F*)gDirectory->Get("pTRes");
  TH1F * invPTRes_GEM = (TH1F*)gDirectory->Get("invPTRes");
  TH1F * pTDiff_GEM = (TH1F*)gDirectory->Get("pTDiff");
  TH1F * PSimEta_GEM = (TH1F*)gDirectory->Get("PSimEta");
  TH1F * PRecEta_GEM = (TH1F*)gDirectory->Get("PRecEta");
  TH1F * PDeltaEta_GEM = (TH1F*)gDirectory->Get("PDeltaEta");
  TH1F * PSimPhi_GEM = (TH1F*)gDirectory->Get("PSimPhi");
  TH1F * PRecPhi_GEM = (TH1F*)gDirectory->Get("PRecPhi");
  TH1F * NumSimTracks_GEM = (TH1F*)gDirectory->Get("NumSimTracks");
  TH1F * NumMuonSimTracks_GEM = (TH1F*)gDirectory->Get("NumMuonSimTracks");
  TH1F * NumRecTracks_GEM = (TH1F*)gDirectory->Get("NumRecTracks");
  TH2F * PtResVsPt_GEM = (TH2F*)gDirectory->Get("PtResVsPt");
  TH2F * InvPtResVsPt_GEM = (TH2F*)gDirectory->Get("InvPtResVsPt");
  TH2F * PtResVsEta_GEM = (TH2F*)gDirectory->Get("PtResVsEta");
  TH2F * InvPtResVsEta_GEM = (TH2F*)gDirectory->Get("InvPtResVsEta");
  TH2F * DPhiVsPt_GEM = (TH2F*)gDirectory->Get("DPhiVsPt");
  TH1F * DenPt_GEM = (TH1F*)gDirectory->Get("DenPt");
  TH1F * DenEta_GEM = (TH1F*)gDirectory->Get("DenEta");
  TH1F * DenPhi_GEM = (TH1F*)gDirectory->Get("DenPhi");
  TH1F * DenPhiPlus_GEM = (TH1F*)gDirectory->Get("DenPhiPlus");
  TH1F * DenPhiMinus_GEM = (TH1F*)gDirectory->Get("DenPhiMinus");
  TH1F * NumPt_GEM = (TH1F*)gDirectory->Get("NumPt");
  TH1F * NumEta_GEM = (TH1F*)gDirectory->Get("NumEta");
  TH1F * NumPhi_GEM = (TH1F*)gDirectory->Get("NumPhi");
  TH1F * NumPhiPlus_GEM = (TH1F*)gDirectory->Get("NumPhiPlus");
  TH1F * NumPhiMinus_GEM = (TH1F*)gDirectory->Get("NumPhiMinus");

  TH1F * DenSimPt_GEM = (TH1F*)gDirectory->Get("DenSimPt");
  TH1F * DenSimEta_GEM = (TH1F*)gDirectory->Get("DenSimEta");
  TH1F * DenSimPhiPlus_GEM = (TH1F*)gDirectory->Get("DenSimPhiPlus");
  TH1F * DenSimPhiMinus_GEM = (TH1F*)gDirectory->Get("DenSimPhiMinus");
  TH1F * NumSimPt_GEM = (TH1F*)gDirectory->Get("NumSimPt");
  TH1F * NumSimEta_GEM = (TH1F*)gDirectory->Get("NumSimEta");
  TH1F * NumSimPhiPlus_GEM = (TH1F*)gDirectory->Get("NumSimPhiPlus");
  TH1F * NumSimPhiMinus_GEM = (TH1F*)gDirectory->Get("NumSimPhiMinus");

  TH1F * PullGEM_GEM = (TH1F*)gDirectory->Get("PullGEMx");
  TH1F * PullCSC_GEM = (TH1F*)gDirectory->Get("PullCSC");
  TH1F * GEMRecHitEta_GEM = (TH1F*)gDirectory->Get("GEMRecHitEta");
  TH1F * GEMRecHitPhi_GEM = (TH1F*)gDirectory->Get("GEMRecHitPhi");
  TH2F * DeltaCharge_GEM = (TH2F*)gDirectory->Get("DeltaCharge");
  TH2F * RecPhi2DPlusLayer1_GEM = (TH2F*)gDirectory->Get("RecHitPhi2DPlusLayer1");
  TH2F * RecPhi2DMinusLayer1_GEM = (TH2F*)gDirectory->Get("RecHitPhi2DMinusLayer1");
  TH2F * RecPhi2DPlusLayer2_GEM = (TH2F*)gDirectory->Get("RecHitPhi2DPlusLayer2");
  TH2F * RecPhi2DMinusLayer2_GEM = (TH2F*)gDirectory->Get("RecHitPhi2DMinusLayer2");

  TH1F * NumPt_GEM2 = (TH1F*)NumPt_GEM->Clone();
  TH1F * NumEta_GEM2 = (TH1F*)NumEta_GEM->Clone();

  TH2F * PtResVsPt_GEM2 = (TH2F*)PtResVsPt_GEM->Clone();
  TH2F * InvPtResVsPt_GEM2 = (TH2F*)InvPtResVsPt_GEM->Clone();
  TH2F * PtResVsEta_GEM2 = (TH2F*)PtResVsEta_GEM->Clone();
  TH2F * InvPtResVsEta_GEM2 = (TH2F*)InvPtResVsEta_GEM->Clone();
  TH2F * DPhiVsPt_GEM2 = (TH2F*)DPhiVsPt_GEM->Clone();
  TH2F * RecoPtVsSimPt_GEM = (TH2F*)gDirectory->Get("RecoPtVsSimPt");

  NumPt_GEM2->Divide(DenPt_GEM);
  NumEta_GEM2->Divide(DenEta_GEM);

  TProfile * prof1_GEM = PtResVsPt_GEM2->ProfileX();
  TProfile * prof2_GEM = InvPtResVsPt_GEM2->ProfileX();
  TProfile * prof1_2_GEM = PtResVsEta_GEM2->ProfileX();
  TProfile * prof2_2_GEM = InvPtResVsEta_GEM2->ProfileX();
  TProfile * prof2bis_GEM = InvPtResVsPt_GEM2->ProfileX("profile",-1,-1,"s");
  TProfile * prof3_GEM = DPhiVsPt_GEM2->ProfileX();

  TProfile * prof4_GEM = RecoPtVsSimPt_GEM->ProfileX();

  TEfficiency* pEffPt_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumPt_GEM,*DenPt_GEM))
  {
    	pEffPt_GEM = new TEfficiency(*NumPt_GEM,*DenPt_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffEta_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumEta_GEM,*DenEta_GEM))
  {
    	pEffEta_GEM = new TEfficiency(*NumEta_GEM,*DenEta_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffEta_GEM->Draw();
  }

  TEfficiency* pEffPhi_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumPhi_GEM,*DenPhi_GEM))
  {
    	pEffPhi_GEM = new TEfficiency(*NumPhi_GEM,*DenPhi_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffPhiPlus_GEM = 0;
 
  NumPhiPlus_GEM->Rebin(10);
  DenPhiPlus_GEM->Rebin(10);
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumPhiPlus_GEM,*DenPhiPlus_GEM))
  {
    	pEffPhiPlus_GEM = new TEfficiency(*NumPhiPlus_GEM,*DenPhiPlus_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffPhiMinus_GEM = 0;
 
  NumPhiMinus_GEM->Rebin(10);
  DenPhiMinus_GEM->Rebin(10);
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumPhiMinus_GEM,*DenPhiMinus_GEM))
  {
    	pEffPhiMinus_GEM = new TEfficiency(*NumPhiMinus_GEM,*DenPhiMinus_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffSimPt_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumSimPt_GEM,*DenSimPt_GEM))
  {
    	pEffSimPt_GEM = new TEfficiency(*NumSimPt_GEM,*DenSimPt_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffSimEta_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumSimEta_GEM,*DenSimEta_GEM))
  {
    	pEffSimEta_GEM = new TEfficiency(*NumSimEta_GEM,*DenSimEta_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffEta_GEM->Draw();
  }

  TEfficiency* pEffSimPhiPlus_GEM = 0;
 
  NumSimPhiPlus_GEM->Rebin(10);
  DenSimPhiPlus_GEM->Rebin(10);
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumSimPhiPlus_GEM,*DenSimPhiPlus_GEM))
  {
    	pEffSimPhiPlus_GEM = new TEfficiency(*NumSimPhiPlus_GEM,*DenSimPhiPlus_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  TEfficiency* pEffSimPhiMinus_GEM = 0;
 
  NumSimPhiMinus_GEM->Rebin(10);
  DenSimPhiMinus_GEM->Rebin(10);
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*NumSimPhiMinus_GEM,*DenSimPhiMinus_GEM))
  {
    	pEffSimPhiMinus_GEM = new TEfficiency(*NumSimPhiMinus_GEM,*DenSimPhiMinus_GEM);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }

  std::vector<TH1F*> vecResGEM;
  for(int i=1; i<=PtResVsPt_GEM->GetNbinsX(); i++){

	TH1F * temp = (TH1F*)pTRes_GEM->Clone();
	int col = i;
	if(i==10) col=41;
	if(i==5) col=46;
	temp->SetLineColor(col);
	temp->SetLineWidth(2);

	for(int j=1; j<=PtResVsPt_GEM->GetNbinsY(); j++){

		float bin = PtResVsPt_GEM->GetBinContent(i,j);

		temp->SetBinContent(bin,i);

	}

	vecResGEM.push_back(temp);

  }

  //std::cout<<vecResNoGEM.size()<<" "<<vecResGEM.size()<<std::endl;

  TH1F * DeltaChargePercentage_GEM = new TH1F("DeltaChargePercentage_GEM","Frac. Wrong Charge (SIM-RECO)",10,0,1000);
  TH1F * numGem = (TH1F*)DeltaChargePercentage_GEM->Clone();
  TH1F * denGem = (TH1F*)DeltaChargePercentage_GEM->Clone();
  for(int i=1; i<=DeltaCharge_GEM->GetNbinsX(); i++){

	int num1 = DeltaCharge_GEM->GetBinContent(i,2);
	int num2 = DeltaCharge_GEM->GetBinContent(i,4); //zero
	int num3 = DeltaCharge_GEM->GetBinContent(i,6);

	numGem->SetBinContent(i,num1+num3);
	denGem->SetBinContent(i,num1+num2+num3);

  }

  TEfficiency* pEffCharge_GEM = 0;
 
  //h_pass and h_total are valid and consistent histograms
  if(TEfficiency::CheckConsistency(*numGem,*denGem))
  {
    	pEffCharge_GEM = new TEfficiency(*numGem,*denGem);
    	// this will write the TEfficiency object to "myfile.root"
    	// AND pEff will be attached to the current directory
   	//pEffPt_GEM->Draw();
  }


  /////////////////////////////////////////////////////////////////////////////////////

  TFile * fileOut = new TFile(name3.c_str(), "RECREATE");

  pTRes_NoGEM->Write("pTRes_NoGEM");
  pTRes_GEM->Write("pTRes_GEM");

  invPTRes_NoGEM->Write("invPTRes_NoGEM");
  invPTRes_GEM->Write("invPTRes_GEM");

  pTDiff_GEM->Write("pTDiff_GEM");
  pTDiff_NoGEM->Write("pTDiff_NoGEM");

  prof1_NoGEM->Write("prof1_NoGEM");
  prof1_GEM->Write("prof1_GEM");
  PtResVsPt_NoGEM->Write("PtResVsPt_NoGEM");
  PtResVsPt_GEM->Write("PtResVsPt_GEM");
  PtResVsEta_NoGEM->Write("PtResVsEta_NoGEM");
  PtResVsEta_GEM->Write("PtResVsEta_GEM");

  prof2_NoGEM->Write("prof2_NoGEM");
  prof2_GEM->Write("prof2_GEM");
  InvPtResVsPt_NoGEM->Write("InvPtResVsPt_NoGEM");
  InvPtResVsPt_GEM->Write("InvPtResVsPt_GEM");
  InvPtResVsEta_NoGEM->Write("InvPtResVsEta_NoGEM");
  InvPtResVsEta_GEM->Write("InvPtResVsEta_GEM");

  prof1_2_NoGEM->Write("prof1_2_NoGEM");
  prof1_2_GEM->Write("prof1_2_GEM");

  prof2_2_NoGEM->Write("prof2_2_NoGEM");
  prof2_2_GEM->Write("prof2_2_GEM");

  pEffPt_GEM->Write("pEffPt_GEM");

  pEffEta_GEM->Write("pEffEta_GEM");

  pEffSimPt_GEM->Write("pEffSimPt_GEM");

  pEffSimEta_GEM->Write("pEffSimEta_GEM");

  pEffCharge_NoGEM->Write("pEffCharge_NoGEM");
  pEffCharge_GEM->Write("pEffCharge_GEM");

  pEffPhiPlus_GEM->Write("pEffPhiPlus_GEM");
  pEffPhiMinus_GEM->Write("pEffPhiMinus_GEM");
  pEffSimPhiPlus_GEM->Write("pEffSimPhiPlus_GEM");
  pEffSimPhiMinus_GEM->Write("pEffSimPhiMinus_GEM");

  prof4_NoGEM->Write("prof4_NoGEM");
  prof4_GEM->Write("prof4_GEM");

  fileOut->Write();

}

void makeAll(){

	makePlots("GLBMuonAnalyzer_5GeV.root","GLBMuonAnalyzerWithGEMs_5GeV.root","plots_5GeV.root");
	makePlots("GLBMuonAnalyzer_10GeV.root","GLBMuonAnalyzerWithGEMs_10GeV.root","plots_10GeV.root");
	makePlots("GLBMuonAnalyzer_50GeV.root","GLBMuonAnalyzerWithGEMs_50GeV.root","plots_50GeV.root");
	makePlots("GLBMuonAnalyzer_100GeV.root","GLBMuonAnalyzerWithGEMs_100GeV.root","plots_100GeV.root");
	makePlots("GLBMuonAnalyzer_200GeV.root","GLBMuonAnalyzerWithGEMs_200GeV.root","plots_200GeV.root");
	makePlots("GLBMuonAnalyzer_500GeV.root","GLBMuonAnalyzerWithGEMs_500GeV.root","plots_500GeV.root");
	makePlots("GLBMuonAnalyzer_1000GeV.root","GLBMuonAnalyzerWithGEMs_1000GeV.root","plots_1000GeV.root");

}

