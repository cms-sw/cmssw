#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
//#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
//#include "CondCore/HcalPlugins/plugins/HcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalGains.h" //or Gain.h???

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>

namespace {

  //TODO: Check these
  enum{ HBmaxAbsEta = 19, maxPhi = 72 };

  /******************************************
     2d plot of ECAL GainRatios of 1 IOV
  ******************************************/
  class HcalGainsPlot : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsPlot() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      //TODO: Shift binning by 0.5
      

      auto iov = iovs.front();
      std::shared_ptr<HcalGains> payload = fetchPayload( std::get<1>(iov) );
      //unsigned int run = std::get<0>(iov);
      //HcalTopology hcalTopo = HcalTopology(HcalTopologyMode::LHC,7,7,HcalTopologyMode::TriggerMode_2018);
      //payload->setTopo(&hcalTopo);


      gStyle->SetPalette(1);

      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map",1680,1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);

      if( payload.get() ){
        std::vector<DetId> channels = (*payload).getAllChannels();
        std::vector< std::pair< std::string, std::vector<HcalGain> >> containers = (*payload).getAllContainers();

        for( std::pair< std::string, std::vector<HcalGain> > cont : containers){  
            //std::cout << "Container for " << std::get<0>(cont) <<  "has content " << std::to_string(std::get<1>(cont).getValues()) << std::endl;
            std::vector<HcalGain> gainsVec = std::get<1>(cont);
            for(HcalGain gain : gainsVec){
              HcalDetId detId = HcalDetId(gain.rawId());
              int iphi = detId.iphi();
              int ieta = detId.ieta();
              int depth = detId.depth();
              float gainVal = (*gain.getValues());
              std::cout << "detId Vals: (det, eta, phi, depth, val) = (" << std::get<0>(cont) << ", " << std::to_string(ieta) << ", " << std::to_string(iphi) << ", " << std::to_string(depth) << ", " << std::to_string((*gain.getValues())) << ")" << std::endl;
             // std::cout << "Container for " << std::get<0>(cont) << " at raw id " << std::to_string(gain.rawId()) <<  "has content " << std::to_string((*gain.getValues())) << std::endl;
              //if(depth==1) hHB_d1->Fill(ieta, iphi, gainVal);
            }
        }

        //for( HcalDetId channelId : channels) {
        //    if(channelId.subdetId()==HcalBarrel){
        //      std::cout << "Channel content is:" << (*payload).(channelId) << std::endl;
        //      
        //      float gainVal = payload->getValues(channelId, true)->getValue(0);
        //      hHB_d1->Fill(channelId.ieta(), channelId.iphi(), gainVal);
        //    }
        //}
          //canvas->cd()
          //hHB_d1->Draw();
        }
      else{
        //t1.DrawLatex(0.5, 0.96, Form("COULDN'T FIND PAYLOAD %i", -1));
      } 
     
      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL GainRatios difference between 2 IOVs
  **********************************************************/
  class HcalGainsDiff : public cond::payloadInspector::PlotImage<HcalGains> {

  public:
    HcalGainsDiff() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalGains> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalGains> payload2 = fetchPayload( std::get<1>(iov2) );
      unsigned int run1 = std::get<0>(iov1);
      unsigned int run2 = std::get<0>(iov2);

    //here you can change z-ranges for different
	float mr = 0.2; // z-range amplitude
	float r_a = 1 - 1 * mr, r_b = 1 + 1 * mr;//hb
	float r3_a = 1 - 1 * mr, r3_b = 1 + 1 * mr;//he
	float r4_a = 1 - 1 * mr, r4_b = 1 + 1 * mr;//ho
	float r2_a = 1 - 1 * mr, r2_b = 1 + 1 * mr;//hf

	float outR = 0.2;  // z-range amplitude for oputliers
	float  OUTLdown = 1 - outR, OUTLup = 1 + outR;


  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  gStyle->SetLabelFont(42);
  gStyle->SetLabelFont(42);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleFont(42);
  gStyle->SetMarkerSize(0);
  gStyle->SetTitleOffset(1.3,"Y");
  gStyle->SetTitleOffset(1.0,"X");
  gStyle->SetNdivisions(510);
  gStyle->SetStatH(0.11);
  gStyle->SetStatW(0.33);
  gStyle->SetTitleW(0.4);
  gStyle->SetTitleX(0.13);
  //gROOT->ForceStyle();
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
   


 
  TH2F *hHB_d1 = new TH2F("HB_d1", "          HB_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHB_d2 = new TH2F("HB_d2", "          HB_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHO_d1 = new TH2F("HO_d1", "          HO_d4", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d1 = new TH2F("HE_d1", "          HE_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d2 = new TH2F("HE_d2", "          HE_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d3 = new TH2F("HE_d3", "          HE_d3", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d4 = new TH2F("HE_d4", "          HE_d4", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d5 = new TH2F("HE_d5", "          HE_d5", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d6 = new TH2F("HE_d6", "          HE_d6", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d7 = new TH2F("HE_d7", "          HE_d7", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d1 = new TH2F("HF_d1", "          HF_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d2 = new TH2F("HF_d2", "          HF_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d3 = new TH2F("HF_d3", "          HF_d3", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d4 = new TH2F("HF_d4", "          HF_d4", 83, -42, 42, 71, 1, 72);

  TH2F *hHB_d1b = new TH2F("HB_d1b", "          HB_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHB_d2b = new TH2F("HB_d2b", "          HB_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHO_d1b = new TH2F("HO_d1b", "          HO_d4", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d1b = new TH2F("HE_d1b", "          HE_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d2b = new TH2F("HE_d2b", "          HE_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d3b = new TH2F("HE_d3b", "          HE_d3", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d4b = new TH2F("HE_d4b", "          HE_d4", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d5b = new TH2F("HE_d5b", "          HE_d5", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d6b = new TH2F("HE_d6b", "          HE_d6", 83, -42, 42, 71, 1, 72);
  TH2F *hHE_d7b = new TH2F("HE_d7b", "          HE_d7", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d1b = new TH2F("HF_d1b", "          HF_d1", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d2b = new TH2F("HF_d2b", "          HF_d2", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d3b = new TH2F("HF_d3b", "          HF_d3", 83, -42, 42, 71, 1, 72);
  TH2F *hHF_d4b = new TH2F("HF_d4b", "          HF_d4", 83, -42, 42, 71, 1, 72);
    
    

      if( payload1.get() ){
        std::vector<DetId> channels1 = (*payload1).getAllChannels();
        std::vector< std::pair< std::string, std::vector<HcalGain> >> containers1 = (*payload1).getAllContainers();
        int iphi1;
        int ieta1;
        int depth1;
        float gainVal1;
        HcalDetId detId1;
        for( std::pair< std::string, std::vector<HcalGain> > cont : containers1){  
            //std::cout << "Container for " << std::get<0>(cont) <<  "has content " << std::to_string(std::get<1>(cont).getValues()) << std::endl;
            std::vector<HcalGain> gainsVals1 = std::get<1>(cont);
            for(HcalGain gain : gainsVals1){
              detId1 = HcalDetId(gain.rawId());
              iphi1 = detId1.iphi();
              ieta1 = detId1.ieta();
              depth1 = detId1.depth();
              gainVal1 = gain.getValue(0) + gain.getValue(1) + gain.getValue(2) + gain.getValue(3);

	      if (std::get<0>(cont)[1] == 'B') {
	              if (depth1 == 1) hHB_d1->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 2) hHB_d2->Fill(ieta1, iphi1, gainVal1);
	      }
	      if (std::get<0>(cont)[1] == 'O') {
	              if (depth1>0)hHO_d1->Fill(ieta1, iphi1, gainVal1);
	      }
	 
	      if (std::get<0>(cont)[1] == 'E') {
	              if (depth1 == 1)hHE_d1->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 2)hHE_d2->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 3)hHE_d3->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 4)hHE_d4->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 5)hHE_d5->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 6)hHE_d6->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 7)hHE_d7->Fill(ieta1, iphi1, gainVal1);
	      }
	      if (std::get<0>(cont)[1] == 'F') {
	              if (depth1 == 1)hHF_d1->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 2)hHF_d2->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 3)hHF_d3->Fill(ieta1, iphi1, gainVal1);
	              if (depth1 == 4)hHF_d4->Fill(ieta1, iphi1, gainVal1);
	      }
            }
        }
     }


      if( payload2.get() ){
        std::vector<DetId> channels2 = (*payload2).getAllChannels();
        std::vector< std::pair< std::string, std::vector<HcalGain> >> containers2 = (*payload2).getAllContainers();
        int iphi2;
        int ieta2;
        int depth2;
        float gainVal2;
        HcalDetId detId2;
        for( std::pair< std::string, std::vector<HcalGain> > cont : containers2){  
            //std::cout << "Container for " << std::get<0>(cont) <<  "has content " << std::to_string(std::get<1>(cont).getValues()) << std::endl;
            std::vector<HcalGain> gainsVals2 = std::get<1>(cont);
            for(HcalGain gain : gainsVals2){
              detId2 = HcalDetId(gain.rawId());
              iphi2 = detId2.iphi();
              ieta2 = detId2.ieta();
              depth2 = detId2.depth();
              gainVal2 = gain.getValue(0) + gain.getValue(1) + gain.getValue(2) + gain.getValue(3);

	      if (std::get<0>(cont)[1] == 'B') {
	              if (depth2 == 1) hHB_d1b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 2) hHB_d2b->Fill(ieta2, iphi2, gainVal2);
	      }
	      if (std::get<0>(cont)[1] == 'O') {
	              if (depth2>0)hHO_d1b->Fill(ieta2, iphi2, gainVal2);
	      }
	 
	      if (std::get<0>(cont)[1] == 'E') {
	              if (depth2 == 1)hHE_d1b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 2)hHE_d2b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 3)hHE_d3b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 4)hHE_d4b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 5)hHE_d5b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 6)hHE_d6b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 7)hHE_d7b->Fill(ieta2, iphi2, gainVal2);
	      }
	      if (std::get<0>(cont)[1] == 'F') {
	              if (depth2 == 1)hHF_d1b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 2)hHF_d2b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 3)hHF_d3b->Fill(ieta2, iphi2, gainVal2);
	              if (depth2 == 4)hHF_d4b->Fill(ieta2, iphi2, gainVal2);
	      }
            }
        }
     }

  
  hHB_d1->Divide(hHB_d1, hHB_d1b);
  hHB_d2->Divide(hHB_d2, hHB_d2b);
  hHO_d1->Divide(hHO_d1, hHO_d1b);
  hHE_d1->Divide(hHE_d1, hHE_d1b);
  hHE_d2->Divide(hHE_d2, hHE_d2b);
  hHE_d3->Divide(hHE_d3, hHE_d3b);
  hHE_d4->Divide(hHE_d4, hHE_d4b);
  hHE_d5->Divide(hHE_d5, hHE_d5b);
  hHE_d6->Divide(hHE_d6, hHE_d6b);
  hHE_d7->Divide(hHE_d7, hHE_d7b);
  hHF_d1->Divide(hHF_d1, hHF_d1b);
  hHF_d2->Divide(hHF_d2, hHF_d2b);
  hHF_d3->Divide(hHF_d3, hHF_d3b);
  hHF_d4->Divide(hHF_d4, hHF_d4b);


  for (int i = 1; i <= 83; i++) {
	  for (int k = 1; k <= 72; k++) {
		  if(hHB_d1->GetBinContent(i, k) == 0) hHB_d1->SetBinContent(i, k, -999);
		  if(hHB_d2->GetBinContent(i, k) == 0) hHB_d2->SetBinContent(i, k, -999);
		  if(hHO_d1->GetBinContent(i, k) == 0) hHO_d1->SetBinContent(i, k, -999);
		  if(hHE_d1->GetBinContent(i, k) == 0)hHE_d1->SetBinContent(i, k, -999);
		  if(hHE_d2->GetBinContent(i, k) == 0)hHE_d2->SetBinContent(i, k, -999);
		  if(hHE_d3->GetBinContent(i, k) == 0)hHE_d3->SetBinContent(i, k, -999);
		  if(hHE_d4->GetBinContent(i, k) == 0)hHE_d4->SetBinContent(i, k, -999);
		  if(hHE_d5->GetBinContent(i, k) == 0)hHE_d5->SetBinContent(i, k, -999);
		  if(hHE_d6->GetBinContent(i, k) == 0)hHE_d6->SetBinContent(i, k, -999);
		  if(hHE_d7->GetBinContent(i, k) == 0)hHE_d7->SetBinContent(i, k, -999);
		  if(hHF_d1->GetBinContent(i, k) == 0)hHF_d1->SetBinContent(i, k, -999);
		  if(hHF_d2->GetBinContent(i, k) == 0)hHF_d2->SetBinContent(i, k, -999);
		  if(hHF_d3->GetBinContent(i, k) == 0)hHF_d3->SetBinContent(i, k, -999);
		  if(hHF_d4->GetBinContent(i, k) == 0)hHF_d4->SetBinContent(i, k, -999);
	  }
  }
  
  TCanvas *cv1 = new TCanvas("cv1", "HBHO", 1000, 1000);
  cv1->Divide(2, 2);

  cv1->cd(1);
  cv1->GetPad(1)->SetGridx(1);
  cv1->GetPad(1)->SetGridy(1);
  hHB_d1->Draw("colz");
  hHB_d1->SetContour(100);
  hHB_d1->GetXaxis()->SetTitle("ieta");
  hHB_d1->GetYaxis()->SetTitle("iphi");
  hHB_d1->GetXaxis()->CenterTitle();
  hHB_d1->GetYaxis()->CenterTitle();
  hHB_d1->GetZaxis()->SetRangeUser(r_a, r_b);
  hHB_d1->GetYaxis()->SetTitleSize(0.06);
  hHB_d1->GetYaxis()->SetTitleOffset(0.80);
  hHB_d1->GetXaxis()->SetTitleSize(0.06);
  hHB_d1->GetXaxis()->SetTitleOffset(0.80);
  hHB_d1->GetYaxis()->SetLabelSize(0.055);
  hHB_d1->GetXaxis()->SetLabelSize(0.055);

  cv1->cd(2);
  cv1->GetPad(2)->SetGridx(1);
  cv1->GetPad(2)->SetGridy(1);
  hHB_d2->Draw("colz");
  hHB_d2->SetContour(100);
  hHB_d2->GetXaxis()->SetTitle("ieta");
  hHB_d2->GetYaxis()->SetTitle("iphi");
  hHB_d2->GetXaxis()->CenterTitle();
  hHB_d2->GetYaxis()->CenterTitle();
  hHB_d2->GetZaxis()->SetRangeUser(r_a, r_b);
  hHB_d2->GetYaxis()->SetTitleSize(0.06);
  hHB_d2->GetYaxis()->SetTitleOffset(0.80);
  hHB_d2->GetXaxis()->SetTitleSize(0.06);
  hHB_d2->GetXaxis()->SetTitleOffset(0.80);
  hHB_d2->GetYaxis()->SetLabelSize(0.055);
  hHB_d2->GetXaxis()->SetLabelSize(0.055);
  hHB_d1->SetContour(100);

  cv1->cd(3);
  cv1->GetPad(3)->SetGridx(1);
  cv1->GetPad(3)->SetGridy(1);
  hHO_d1->Draw("colz");
  hHO_d1->SetContour(100);
  hHO_d1->GetXaxis()->SetTitle("ieta");
  hHO_d1->GetYaxis()->SetTitle("iphi");
  hHO_d1->GetXaxis()->CenterTitle();
  hHO_d1->GetYaxis()->CenterTitle();
  hHO_d1->GetZaxis()->SetRangeUser(r4_a, r4_b);
  hHO_d1->GetYaxis()->SetTitleSize(0.06);
  hHO_d1->GetYaxis()->SetTitleOffset(0.80);
  hHO_d1->GetXaxis()->SetTitleSize(0.06);
  hHO_d1->GetXaxis()->SetTitleOffset(0.80);
  hHO_d1->GetYaxis()->SetLabelSize(0.055);
  hHO_d1->GetXaxis()->SetLabelSize(0.055);


  TCanvas *cv2 = new TCanvas("cv2", "HE", 1000, 1000);
  cv2->Divide(3, 3);

  cv2->cd(1);
  cv2->GetPad(1)->SetGridx(1);
  cv2->GetPad(1)->SetGridy(1);
  hHE_d1->SetContour(100);
  hHE_d1->Draw("colz");
  hHE_d1->GetXaxis()->SetTitle("ieta");
  hHE_d1->GetYaxis()->SetTitle("iphi");
  hHE_d1->GetXaxis()->CenterTitle();
  hHE_d1->GetYaxis()->CenterTitle();
  hHE_d1->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d1->GetYaxis()->SetTitleSize(0.06);
  hHE_d1->GetYaxis()->SetTitleOffset(0.80);
  hHE_d1->GetXaxis()->SetTitleSize(0.06);
  hHE_d1->GetXaxis()->SetTitleOffset(0.80);
  hHE_d1->GetYaxis()->SetLabelSize(0.055);
  hHE_d1->GetXaxis()->SetLabelSize(0.055);

  cv2->cd(2);
  cv2->GetPad(2)->SetGridx(1);
  cv2->GetPad(2)->SetGridy(1);
  hHE_d2->Draw("colz");
  hHE_d2->SetContour(100);
  hHE_d2->GetXaxis()->SetTitle("ieta");
  hHE_d2->GetYaxis()->SetTitle("iphi");
  hHE_d2->GetXaxis()->CenterTitle();
  hHE_d2->GetYaxis()->CenterTitle();
  hHE_d2->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d2->GetYaxis()->SetTitleSize(0.06);
  hHE_d2->GetYaxis()->SetTitleOffset(0.80);
  hHE_d2->GetXaxis()->SetTitleSize(0.06);
  hHE_d2->GetXaxis()->SetTitleOffset(0.80);
  hHE_d2->GetYaxis()->SetLabelSize(0.055);
  hHE_d2->GetXaxis()->SetLabelSize(0.055);

  cv2->cd(3);
  cv2->GetPad(3)->SetGridx(1);
  cv2->GetPad(3)->SetGridy(1);
  hHE_d3->Draw("colz");
  hHE_d3->SetContour(100);
  hHE_d3->GetXaxis()->SetTitle("ieta");
  hHE_d3->GetYaxis()->SetTitle("iphi");
  hHE_d3->GetXaxis()->CenterTitle();
  hHE_d3->GetYaxis()->CenterTitle();
  hHE_d3->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d3->GetYaxis()->SetTitleSize(0.06);
  hHE_d3->GetYaxis()->SetTitleOffset(0.80);
  hHE_d3->GetXaxis()->SetTitleSize(0.06);
  hHE_d3->GetXaxis()->SetTitleOffset(0.80);
  hHE_d3->GetYaxis()->SetLabelSize(0.055);
  hHE_d3->GetXaxis()->SetLabelSize(0.055);

  cv2->cd(4);
  cv2->GetPad(4)->SetGridx(1);
  cv2->GetPad(4)->SetGridy(1);
  hHE_d4->Draw("colz");
  hHE_d4->SetContour(100);
  hHE_d4->GetXaxis()->SetTitle("ieta");
  hHE_d4->GetYaxis()->SetTitle("iphi");
  hHE_d4->GetXaxis()->CenterTitle();
  hHE_d4->GetYaxis()->CenterTitle();
  hHE_d4->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d4->GetYaxis()->SetTitleSize(0.06);
  hHE_d4->GetYaxis()->SetTitleOffset(0.80);
  hHE_d4->GetXaxis()->SetTitleSize(0.06);
  hHE_d4->GetXaxis()->SetTitleOffset(0.80);
  hHE_d4->GetYaxis()->SetLabelSize(0.055);
  hHE_d4->GetXaxis()->SetLabelSize(0.055);

  cv2->cd(5);
  cv2->GetPad(5)->SetGridx(1);
  cv2->GetPad(5)->SetGridy(1);
  hHE_d5->Draw("colz");
  hHE_d5->SetContour(100);
  hHE_d5->GetXaxis()->SetTitle("ieta");
  hHE_d5->GetYaxis()->SetTitle("iphi");
  hHE_d5->GetXaxis()->CenterTitle();
  hHE_d5->GetYaxis()->CenterTitle();
  hHE_d5->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d5->GetYaxis()->SetTitleSize(0.06);
  hHE_d5->GetYaxis()->SetTitleOffset(0.80);
  hHE_d5->GetXaxis()->SetTitleSize(0.06);
  hHE_d5->GetXaxis()->SetTitleOffset(0.80);
  hHE_d5->GetYaxis()->SetLabelSize(0.055);
  hHE_d5->GetXaxis()->SetLabelSize(0.055);

  cv2->cd(6);
  cv2->GetPad(6)->SetGridx(1);
  cv2->GetPad(6)->SetGridy(1);
  hHE_d6->Draw("colz");
  hHE_d6->SetContour(100);
  hHE_d6->GetXaxis()->SetTitle("ieta");
  hHE_d6->GetYaxis()->SetTitle("iphi");
  hHE_d6->GetXaxis()->CenterTitle();
  hHE_d6->GetYaxis()->CenterTitle();
  hHE_d6->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d6->GetYaxis()->SetTitleSize(0.06);
  hHE_d6->GetYaxis()->SetTitleOffset(0.80);
  hHE_d6->GetXaxis()->SetTitleSize(0.06);
  hHE_d6->GetXaxis()->SetTitleOffset(0.80);
  hHE_d6->GetYaxis()->SetLabelSize(0.055);
  hHE_d6->GetXaxis()->SetLabelSize(0.055);


  cv2->cd(7);
  cv2->GetPad(7)->SetGridx(1);
  cv2->GetPad(7)->SetGridy(1);
  hHE_d7->Draw("colz");
  hHE_d7->SetContour(100);
  hHE_d7->GetXaxis()->SetTitle("ieta");
  hHE_d7->GetYaxis()->SetTitle("iphi");
  hHE_d7->GetXaxis()->CenterTitle();
  hHE_d7->GetYaxis()->CenterTitle();
  hHE_d7->GetZaxis()->SetRangeUser(r3_a, r3_b);
  hHE_d7->GetYaxis()->SetTitleSize(0.06);
  hHE_d7->GetYaxis()->SetTitleOffset(0.80);
  hHE_d7->GetXaxis()->SetTitleSize(0.06);
  hHE_d7->GetXaxis()->SetTitleOffset(0.80);
  hHE_d7->GetYaxis()->SetLabelSize(0.055);
  hHE_d7->GetXaxis()->SetLabelSize(0.055);



  TCanvas *cv3 = new TCanvas("cv3", "HF", 1000, 1000);
  cv3->Divide(2, 2);

  cv3->cd(1);
  cv3->GetPad(1)->SetGridx(1);
  cv3->GetPad(1)->SetGridy(1);
  hHF_d1->Draw("colz");
  hHF_d1->SetContour(100);
  hHF_d1->GetXaxis()->SetTitle("ieta");
  hHF_d1->GetYaxis()->SetTitle("iphi");
  hHF_d1->GetXaxis()->CenterTitle();
  hHF_d1->GetYaxis()->CenterTitle();
  hHF_d1->GetZaxis()->SetRangeUser(r2_a, r2_b);
  hHF_d1->GetYaxis()->SetTitleSize(0.06);
  hHF_d1->GetYaxis()->SetTitleOffset(0.80);
  hHF_d1->GetXaxis()->SetTitleSize(0.06);
  hHF_d1->GetXaxis()->SetTitleOffset(0.80);
  hHF_d1->GetYaxis()->SetLabelSize(0.055);
  hHF_d1->GetXaxis()->SetLabelSize(0.055);

  cv3->cd(2);
  cv3->GetPad(2)->SetGridx(1);
  cv3->GetPad(2)->SetGridy(1);
  hHF_d2->Draw("colz");
  hHF_d2->GetXaxis()->SetTitle("ieta");
  hHF_d2->GetYaxis()->SetTitle("iphi");
  hHF_d2->GetXaxis()->CenterTitle();
  hHF_d2->GetYaxis()->CenterTitle();
  hHF_d2->GetZaxis()->SetRangeUser(r2_a, r2_b);
  hHF_d2->GetYaxis()->SetTitleSize(0.06);
  hHF_d2->GetYaxis()->SetTitleOffset(0.80);
  hHF_d2->GetXaxis()->SetTitleSize(0.06);
  hHF_d2->GetXaxis()->SetTitleOffset(0.80);
  hHF_d2->GetYaxis()->SetLabelSize(0.055);
  hHF_d2->GetXaxis()->SetLabelSize(0.055);
  hHF_d2->SetContour(100);

  cv3->cd(3);
  cv3->GetPad(3)->SetGridx(1);
  cv3->GetPad(3)->SetGridy(1);
  hHF_d3->Draw("colz");
  hHF_d3->SetContour(100);
  hHF_d3->GetXaxis()->SetTitle("ieta");
  hHF_d3->GetYaxis()->SetTitle("iphi");
  hHF_d3->GetXaxis()->CenterTitle();
  hHF_d3->GetYaxis()->CenterTitle();
  hHF_d3->GetZaxis()->SetRangeUser(r2_a, r2_b);
  hHF_d3->GetYaxis()->SetTitleSize(0.06);
  hHF_d3->GetYaxis()->SetTitleOffset(0.80);
  hHF_d3->GetXaxis()->SetTitleSize(0.06);
  hHF_d3->GetXaxis()->SetTitleOffset(0.80);
  hHF_d3->GetYaxis()->SetLabelSize(0.055);
  hHF_d3->GetXaxis()->SetLabelSize(0.055);

  cv3->cd(4);
  cv3->GetPad(4)->SetGridx(1);
  cv3->GetPad(4)->SetGridy(1);
  hHF_d4->Draw("colz");
  hHF_d4->GetXaxis()->SetTitle("ieta");
  hHF_d4->GetYaxis()->SetTitle("iphi");
  hHF_d4->GetXaxis()->CenterTitle();
  hHF_d4->GetYaxis()->CenterTitle();
  hHF_d4->GetZaxis()->SetRangeUser(r2_a, r2_b);
  hHF_d4->GetYaxis()->SetTitleSize(0.06);
  hHF_d4->GetYaxis()->SetTitleOffset(0.80);
  hHF_d4->GetXaxis()->SetTitleSize(0.06);
  hHF_d4->GetXaxis()->SetTitleOffset(0.80);
  hHF_d4->GetYaxis()->SetLabelSize(0.055);
  hHF_d4->GetXaxis()->SetLabelSize(0.055);
  hHF_d4->SetContour(100);




      std::string ImageName(m_imageFileName);
      cv3->SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalGains){
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsDiff);
}
