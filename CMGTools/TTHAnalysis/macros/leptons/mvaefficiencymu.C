#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"

void mvaefficiencymu()
{

  //gROOT->SetStyle("Plain");
  
  TFile * mc = new TFile("DYJetsM50_tnp2.root");
  TTree * mc_tree = mc->Get("tpTree/fitter_tree") ;
  TFile * data = new TFile("DoubleMu_tnp2.root");
  TTree * data_tree = data->Get("tpTree/fitter_tree") ;

  double pt_binsL[11] = {5,10,15,20,25,30,38,45,60,80,100}; 
  double pt_binsT[10] = {10,15,20,25,30,38,45,60,80,100}; 
  double eta_bins[11] = {-2.4,-2.1,-1.5,-0.9,-0.3,0,0.3,0.9,1.5,2.1,2.4}; 
  
  TH1::SetDefaultSumw2();
  TH2::SetDefaultSumw2();

  TFile *fOut = TFile::Open("MVAandTigthChargeSF_mu.root", "RECREATE");    
  fOut->cd();

  //Loose MVA working point vs Good lepton selection
  TH1F *hLMmcptb = new TH1F("LepMVALooseEff_mcptb","LepMVALooseEff_mcptb",10,pt_binsL);
  TH1F *hLMmcpte = new TH1F("LepMVALooseEff_mcpte","LepMVALooseEff_mcpte",10,pt_binsL);
  TH1F *hLMdataptb = new TH1F("LepMVALooseEff_dataptb","LepMVALooseEff_dataptb",10,pt_binsL);
  TH1F *hLMdatapte = new TH1F("LepMVALooseEff_datapte","LepMVALooseEff_datapte",10,pt_binsL);
  TH1F *hLMmceta = new TH1F("LepMVALooseEff_mceta","LepMVALooseEff_mceta",10,eta_bins);
  TH1F *hLMdataeta = new TH1F("LepMVALooseEff_dataeta","LepMVALooseEff_dataeta",10,eta_bins);
  TH2F *hLMmc2D = new TH2F("LepMVALooseEff_mc2D","LepMVALooseEff_mc2D",10,pt_binsL,10,eta_bins);
  TH2F *hLMdata2D = new TH2F("LepMVALooseEff_data2D","LepMVALooseEff_data2D",10,pt_binsL,10,eta_bins);

  TH1F *hLMmcptb_den = new TH1F("LepMVALooseEff_mcptb_den","LepMVALooseEff_mcptb_den",10,pt_binsL);
  TH1F *hLMmcpte_den = new TH1F("LepMVALooseEff_mcpte_den","LepMVALooseEff_mcpte_den",10,pt_binsL);
  TH1F *hLMdataptb_den = new TH1F("LepMVALooseEff_dataptb_den","LepMVALooseEff_dataptb_den",10,pt_binsL);
  TH1F *hLMdatapte_den = new TH1F("LepMVALooseEff_datapte_den","LepMVALooseEff_datapte_den",10,pt_binsL);
  TH1F *hLMmceta_den = new TH1F("LepMVALooseEff_mceta_den","LepMVALooseEff_mceta_den",10,eta_bins);
  TH1F *hLMdataeta_den = new TH1F("LepMVALooseEff_dataeta_den","LepMVALooseEff_dataeta_den",10,eta_bins);
  TH2F *hLMmc2D_den = new TH2F("LepMVALooseEff_mc2D_den","LepMVALooseEff_mc2D_den",10,pt_binsL,10,eta_bins);
  TH2F *hLMdata2D_den = new TH2F("LepMVALooseEff_data2D_den","LepMVALooseEff_data2D_den",10,pt_binsL,10,eta_bins);

  TH1F *hLMmcptb_num = new TH1F("LepMVALooseEff_mcptb_num","LepMVALooseEff_mcptb_num",10,pt_binsL);
  TH1F *hLMmcpte_num = new TH1F("LepMVALooseEff_mcpte_num","LepMVALooseEff_mcpte_num",10,pt_binsL);
  TH1F *hLMdataptb_num = new TH1F("LepMVALooseEff_dataptb_num","LepMVALooseEff_dataptb_num",10,pt_binsL);
  TH1F *hLMdatapte_num = new TH1F("LepMVALooseEff_datapte_num","LepMVALooseEff_datapte_num",10,pt_binsL);
  TH1F *hLMmceta_num = new TH1F("LepMVALooseEff_mceta_num","LepMVALooseEff_mceta_num",10,eta_bins);
  TH1F *hLMdataeta_num = new TH1F("LepMVALooseEff_dataeta_num","LepMVALooseEff_dataeta_num",10,eta_bins);
  TH2F *hLMmc2D_num = new TH2F("LepMVALooseEff_mc2D_num","LepMVALooseEff_mc2D_num",10,pt_binsL,10,eta_bins);
  TH2F *hLMdata2D_num = new TH2F("LepMVALooseEff_data2D_num","LepMVALooseEff_data2D_num",10,pt_binsL,10,eta_bins);
 

  TH2F *hLMSF2D = new TH2F("LepMVALooseSF","LepMVALooseSF",10,pt_binsL,10,eta_bins);

  //Tight MVA working point vs Good lepton selection
  TH1F *hTMmcptb = new TH1F("LepMVATightEff_mcptb","LepMVATightEff_mcpte",9,pt_binsT);
  TH1F *hTMmcpte = new TH1F("LepMVATightEff_mcpte","LepMVATightEff_mcpte",9,pt_binsT);
  TH1F *hTMdataptb = new TH1F("LepMVATightEff_dataptb","LepMVATightEff_dataptb",9,pt_binsT);
  TH1F *hTMdatapte = new TH1F("LepMVATightEff_datapte","LepMVATightEff_datapte",9,pt_binsT);
  TH1F *hTMmceta = new TH1F("LepMVATightEff_mceta","LepMVATightEff_mceta",10,eta_bins);
  TH1F *hTMdataeta = new TH1F("LepMVATightEff_dataeta","LepMVATightEff_dataeta",10,eta_bins);
  TH2F *hTMmc2D = new TH2F("LepMVATightEff_mc2D","LepMVATightEff_mc2D",9,pt_binsT,10,eta_bins);
  TH2F *hTMdata2D = new TH2F("LepMVATightEff_data2D","LepMVATightEff_data2D",9,pt_binsT,10,eta_bins);
 
  TH1F *hTMmcptb_den = new TH1F("LepMVATightEff_mcptb_den","LepMVATightEff_mcpte_den",9,pt_binsT);
  TH1F *hTMmcpte_den = new TH1F("LepMVATightEff_mcpte_den","LepMVATightEff_mcpte_den",9,pt_binsT);
  TH1F *hTMdataptb_den = new TH1F("LepMVATightEff_dataptb_den","LepMVATightEff_dataptb_den",9,pt_binsT);
  TH1F *hTMdatapte_den = new TH1F("LepMVATightEff_datapte_den","LepMVATightEff_datapte_den",9,pt_binsT);
  TH1F *hTMmceta_den = new TH1F("LepMVATightEff_mceta_den","LepMVATightEff_mceta_den",10,eta_bins);
  TH1F *hTMdataeta_den = new TH1F("LepMVATightEff_dataeta_den","LepMVATightEff_dataeta_den",10,eta_bins);
  TH2F *hTMmc2D_den = new TH2F("LepMVATightEff_mc2D_den","LepMVATightEff_mc2D_den",9,pt_binsT,10,eta_bins);
  TH2F *hTMdata2D_den = new TH2F("LepMVATightEff_data2D_den","LepMVATightEff_data2D_den",9,pt_binsT,10,eta_bins);

  TH1F *hTMmcptb_num = new TH1F("LepMVATightEff_mcptb_num","LepMVATightEff_mcpte_num",9,pt_binsT);
  TH1F *hTMmcpte_num = new TH1F("LepMVATightEff_mcpte_num","LepMVATightEff_mcpte_num",9,pt_binsT);
  TH1F *hTMdataptb_num = new TH1F("LepMVATightEff_dataptb_num","LepMVATightEff_dataptb_num",9,pt_binsT);
  TH1F *hTMdatapte_num = new TH1F("LepMVATightEff_datapte_num","LepMVATightEff_datapte_num",9,pt_binsT);
  TH1F *hTMmceta_num = new TH1F("LepMVATightEff_mceta_num","LepMVATightEff_mceta_num",10,eta_bins);
  TH1F *hTMdataeta_num = new TH1F("LepMVATightEff_dataeta_num","LepMVATightEff_dataeta_num",10,eta_bins);
  TH2F *hTMmc2D_num = new TH2F("LepMVATightEff_mc2D_num","LepMVATightEff_mc2D_num",9,pt_binsT,10,eta_bins);
  TH2F *hTMdata2D_num = new TH2F("LepMVATightEff_data2D_num","LepMVATightEff_data2D_num",9,pt_binsT,10,eta_bins);

  TH2F *hTMSF2D = new TH2F("LepMVATightSF2D","LepMVATightSF2D",9,pt_binsT,10,eta_bins);
  
  //Tight charge vs Tight MVA working point
  TH1F *hTCmcptb = new TH1F("TightChargeEff_mcptb","TightChargeEff_mcptb",9,pt_binsT);
  TH1F *hTCmcpte = new TH1F("TightChargeEff_mcpte","TightChargeEff_mcpte",9,pt_binsT);
  TH1F *hTCdataptb = new TH1F("TightChargeEff_dataptb","TightChargeEff_dataptb",9,pt_binsT);
  TH1F *hTCdatapte = new TH1F("TightChargeEff_datapte","TightChargeEff_datapte",9,pt_binsT);
  TH1F *hTCmceta = new TH1F("TightChargeEff_mceta","TightChargeEff_mceta",10,eta_bins);
  TH1F *hTCdataeta = new TH1F("TightChargeEff_dataeta","TightChargeEff_dataeta",10,eta_bins);
  TH2F *hTCmc2D = new TH2F("TightChargeEff_mc2D","TightChargeEff_mc2D",9,pt_binsT,10,eta_bins);
  TH2F *hTCdata2D = new TH2F("TightChargeEff_data2D","TightChargeEff_data2D",9,pt_binsT,10,eta_bins);


  TH1F *hTCmcptb_den = new TH1F("TightChargeEff_mcptb_den","TightChargeEff_mcptb_den",9,pt_binsT);
  TH1F *hTCmcpte_den = new TH1F("TightChargeEff_mcpte_den","TightChargeEff_mcpte_den",9,pt_binsT);
  TH1F *hTCdataptb_den = new TH1F("TightChargeEff_dataptb_den","TightChargeEff_dataptb_den",9,pt_binsT);
  TH1F *hTCdatapte_den = new TH1F("TightChargeEff_datapte_den","TightChargeEff_datapte_den",9,pt_binsT);
  TH1F *hTCmceta_den = new TH1F("TightChargeEff_mceta_den","TightChargeEff_mceta_den",10,eta_bins);
  TH1F *hTCdataeta_den = new TH1F("TightChargeEff_dataeta_den","TightChargeEff_dataeta_den",10,eta_bins);
  TH2F *hTCmc2D_den = new TH2F("TightChargeEff_mc2D_den","TightChargeEff_mc2D_den",9,pt_binsT,10,eta_bins);
  TH2F *hTCdata2D_den = new TH2F("TightChargeEff_data2D_den","TightChargeEff_data2D_den",9,pt_binsT,10,eta_bins);

  
  TH1F *hTCmcptb_num = new TH1F("TightChargeEff_mcptb_num","TightChargeEff_mcptb_num",9,pt_binsT);
  TH1F *hTCmcpte_num = new TH1F("TightChargeEff_mcpte_num","TightChargeEff_mcpte_num",9,pt_binsT);
  TH1F *hTCdataptb_num = new TH1F("TightChargeEff_dataptb_num","TightChargeEff_dataptb_num",9,pt_binsT);
  TH1F *hTCdatapte_num = new TH1F("TightChargeEff_datapte_num","TightChargeEff_datapte_num",9,pt_binsT);
  TH1F *hTCmceta_num = new TH1F("TightChargeEff_mceta_num","TightChargeEff_mceta_num",10,eta_bins);
  TH1F *hTCdataeta_num = new TH1F("TightChargeEff_dataeta_num","TightChargeEff_dataeta_num",10,eta_bins);
  TH2F *hTCmc2D_num = new TH2F("TightChargeEff_mc2D_num","TightChargeEff_mc2D_num",9,pt_binsT,10,eta_bins);
  TH2F *hTCdata2D_num = new TH2F("TightChargeEff_data2D_num","TightChargeEff_data2D_num",9,pt_binsT,10,eta_bins);

  TH2F *hTCSF2D = new TH2F("TightChargeSF2D","TightChargeSF2D",9,pt_binsT,10,eta_bins);
  


  

  
  fOut->cd();

  
  // Loose MVA Efficiency 
  

  //barrel
  mc_tree->Draw("pt>>+LepMVALooseEff_mcptb_den","abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+LepMVALooseEff_mcptb_num","mva>0.35 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hLMmcptb->Divide(hLMmcptb_num,hLMmcptb_den,1,1,"B");
  hLMmcptb->Write();
  hLMmcptb->Draw();
  data_tree->Draw("pt>>+LepMVALooseEff_dataptb_den","abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+LepMVALooseEff_dataptb_num","mva>0.35 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hLMdataptb->Divide(hLMdataptb_num,hLMdataptb_den,1,1,"B");
  hLMdataptb->Write();

  //endcap
  mc_tree->Draw("pt>>+LepMVALooseEff_mcpte_den","abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+LepMVALooseEff_mcpte_num","mva>0.35 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hLMmcpte->Divide(hLMmcpte_num,hLMmcpte_den,1,1,"B");
  hLMmcpte->Write();
  data_tree->Draw("pt>>+LepMVALooseEff_datapte_den","abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+LepMVALooseEff_datapte_num","mva>0.35 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hLMdatapte->Divide(hLMdatapte_num,hLMdatapte_den,1,1,"B");
  hLMdatapte->Write();
  
  //eta
  mc_tree->Draw("eta>>+LepMVALooseEff_mceta_den","abs(mass-90)<10 && pt>5 && tag_pt>25");
  mc_tree->Draw("eta>>+LepMVALooseEff_mceta_num","mva>0.35 && abs(mass-90)<10 && pt>5 && tag_pt>25");
  hLMmceta->Divide(hLMmceta_num,hLMmceta_den,1,1,"B");
  hLMmceta->Write();
  data_tree->Draw("eta>>+LepMVALooseEff_dataeta_den","abs(mass-90)<10 && pt>5 && tag_pt>25");
  data_tree->Draw("eta>>+LepMVALooseEff_dataeta_num","mva>0.35 && abs(mass-90)<10 && pt > 5 && tag_pt>25");
  hLMdataeta->Divide(hLMdataeta_num,hLMdataeta_den,1,1,"B");
  hLMdataeta->Write();

  //2D
  mc_tree->Draw("eta:pt>>+LepMVALooseEff_mc2D_den","abs(mass-90)<10 && tag_pt>25");
  mc_tree->Draw("eta:pt>>+LepMVALooseEff_mc2D_num","mva>0.35 && abs(mass-90)<10  && tag_pt>25");
  hLMmc2D->Divide(hLMmc2D_num,hLMmc2D_den,1,1,"B");
  hLMmc2D->Write();
  data_tree->Draw("eta:pt>>+LepMVALooseEff_data2D_den","abs(mass-90)<10 && tag_pt>25");
  data_tree->Draw("eta:pt>>+LepMVALooseEff_data2D_num","mva>0.35 && abs(mass-90)<10 && tag_pt>25");
  hLMdata2D->Divide(hLMdata2D_num,hLMdata2D_den,1,1,"B");
  hLMdata2D->Write();

  hLMSF2D=(TH2F*)hLMdata2D->Clone("LepMVALooseSF2D");
  hLMSF2D->Divide(hLMmc2D);
  hLMSF2D->Write();

  // Tight MVA Efficiency 
  

  //barrel
  mc_tree->Draw("pt>>+LepMVATightEff_mcptb_den","abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+LepMVATightEff_mcptb_num","mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hTMmcptb->Divide(hTMmcptb_num,hTMmcptb_den,1,1,"B");
  hTMmcptb->Write();
  data_tree->Draw("pt>>+LepMVATightEff_dataptb_den","abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+LepMVATightEff_dataptb_num","mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hTMdataptb->Divide(hTMdataptb_num,hTMdataptb_den,1,1,"B");
  hTMdataptb->Write();

  //endcap
  mc_tree->Draw("pt>>+LepMVATightEff_mcpte_den","abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+LepMVATightEff_mcpte_num","mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hTMmcpte->Divide(hTMmcpte_num,hTMmcpte_den,1,1,"B");
  hTMmcpte->Write();
  data_tree->Draw("pt>>+LepMVATightEff_datapte_den","abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+LepMVATightEff_datapte_num","mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hTMdatapte->Divide(hTMdatapte_num,hTMdatapte_den,1,1,"B");
  hTMdatapte->Write();
  
  //eta
  mc_tree->Draw("eta>>+LepMVATightEff_mceta_den","abs(mass-90)<10 && pt>5 && tag_pt>25");
  mc_tree->Draw("eta>>+LepMVATightEff_mceta_num","mva>0.7 && abs(mass-90)<10 && pt>10 && tag_pt>25");
  hTMmceta->Divide(hTMmceta_num,hTMmceta_den,1,1,"B");
  hTMmceta->Write();
  data_tree->Draw("eta>>+LepMVATightEff_dataeta_den","abs(mass-90)<10 && pt>5 && tag_pt>25");
  data_tree->Draw("eta>>+LepMVATightEff_dataeta_num","mva>0.7 && abs(mass-90)<10 && pt > 10 && tag_pt>25");
  hTMdataeta->Divide(hTMdataeta_num,hTMdataeta_den,1,1,"B");
  hTMdataeta->Write();

  //2D
  mc_tree->Draw("eta:pt>>+LepMVATightEff_mc2D_den","abs(mass-90)<10 && tag_pt>25");
  mc_tree->Draw("eta:pt>>+LepMVATightEff_mc2D_num","mva>0.7 && abs(mass-90)<10  && tag_pt>25");
  hTMmc2D->Divide(hTMmc2D_num,hTMmc2D_den,1,1,"B");
  hTMmc2D->Write();
  data_tree->Draw("eta:pt>>+LepMVATightEff_data2D_den","abs(mass-90)<10 && tag_pt>25");
  data_tree->Draw("eta:pt>>+LepMVATightEff_data2D_num","mva>0.7 && abs(mass-90)<10 && tag_pt>25");
  hTMdata2D->Divide(hTMdata2D_num,hTMdata2D_den,1,1,"B");
  hTMdata2D->Write();


  hTMSF2D=(TH2F*)hTMdata2D->Clone("LepMVATightSF2D");
  hTMSF2D->Divide(hTMmc2D);
  hTMSF2D->Write();


  // Tight Charge vs Tight MVA  Efficiency 

  
  //barrel
  mc_tree->Draw("pt>>+TightChargeEff_mcptb_den","mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+TightChargeEff_mcptb_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hTCmcptb->Divide(hTCmcptb_num,hTCmcptb_den,1,1,"B");
  hTCmcptb->Write();
  data_tree->Draw("pt>>+TightChargeEff_dataptb_den","mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+TightChargeEff_dataptb_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && abseta < 1.5 && tag_pt>25");
  hTCdataptb->Divide(hTCdataptb_num,hTCdataptb_den,1,1,"B");
  hTCdataptb->Write();

  //endcap
  mc_tree->Draw("pt>>+TightChargeEff_mcpte_den","mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  mc_tree->Draw("pt>>+TightChargeEff_mcpte_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hTCmcpte->Divide(hTCmcpte_num,hTCmcpte_den,1,1,"B");
  hTCmcpte->Write();
  data_tree->Draw("pt>>+TightChargeEff_datapte_den","mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  data_tree->Draw("pt>>+TightChargeEff_datapte_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && abseta > 1.5 && tag_pt>25");
  hTCdatapte->Divide(hTCdatapte_num,hTCdatapte_den,1,1,"B");
  hTCdatapte->Write();
  
  //eta
  mc_tree->Draw("eta>>+TightChargeEff_mceta_den","mva>0.7 && abs(mass-90)<10 && pt>5 && tag_pt>25");
  mc_tree->Draw("eta>>+TightChargeEff_mceta_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && pt>10 && tag_pt>25");
  hTCmceta->Divide(hTCmceta_num,hTCmceta_den,1,1,"B");
  hTCmceta->Write();
  data_tree->Draw("eta>>+TightChargeEff_dataeta_den","mva>0.7 && abs(mass-90)<10 && pt>5 && tag_pt>25");
  data_tree->Draw("eta>>+TightChargeEff_dataeta_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && pt > 10 && tag_pt>25");
  hTCdataeta->Divide(hTCdataeta_num,hTCdataeta_den,1,1,"B");
  hTCdataeta->Write();

  //2D
  mc_tree->Draw("eta:pt>>+TightChargeEff_mc2D_den","mva>0.7 && abs(mass-90)<10 && tag_pt>25");
  mc_tree->Draw("eta:pt>>+TightChargeEff_mc2D_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10  && tag_pt>25");
  hTCmc2D->Divide(hTCmc2D_num,hTCmc2D_den,1,1,"B");
  hTCmc2D->Write();
  data_tree->Draw("eta:pt>>+TightChargeEff_data2D_den","mva>0.7 && abs(mass-90)<10 && tag_pt>25");
  data_tree->Draw("eta:pt>>+TightChargeEff_data2D_num","tightCharge > 0 && mva>0.7 && abs(mass-90)<10 && tag_pt>25");
  hTCdata2D->Divide(hTCdata2D_num,hTCdata2D_den,1,1,"B");
  hTCdata2D->Write();

  hTCSF2D=(TH2F*)hTCdata2D->Clone("TightChargeSF2D");
  hTCSF2D->Divide(hTCmc2D);
  hTCSF2D->Write();

  fOut->Close();

}
