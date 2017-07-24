#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>

namespace {
  enum {kEBChannels = 61200, kEEChannels = 14648, kGains = 3};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};         // endcaps lower and upper bounds on x and y

  /*************************************************
     1d plot of ECAL pedestal of 1 IOV
  *************************************************/
  class EcalPedestalsHist : public cond::payloadInspector::PlotImage<EcalPedestals> {

  public:
    EcalPedestalsHist() : cond::payloadInspector::PlotImage<EcalPedestals>( "ECAL pedestal map") {
      setSingleIov(true);
    }
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH1F** barrel_m = new TH1F*[kGains];
      TH1F** endcap_m = new TH1F*[kGains];
      TH1F** barrel_r = new TH1F*[kGains];
      TH1F** endcap_r = new TH1F*[kGains];
      float bmin[kGains] ={0.7, 0.5, 0.4};
      float bmax[kGains] ={2.2, 1.3, 0.7};
      float emin[kGains] ={1.5, 0.8, 0.4};
      float emax[kGains] ={2.5, 1.5, 0.8};
      for (int gainId = 0; gainId < kGains; gainId++) {
	barrel_m[gainId] = new TH1F(Form("EBm%i", gainId), Form("mean %i EB", gainValues[gainId]), 100, 150., 250.);
	endcap_m[gainId] = new TH1F(Form("EEm%i", gainId), Form("mean %i EE", gainValues[gainId]), 100, 150., 250.);
	barrel_r[gainId] = new TH1F(Form("EBr%i", gainId), Form("rms %i EB",  gainValues[gainId]), 100, bmin[gainId], bmax[gainId]);
	endcap_r[gainId] = new TH1F(Form("EEr%i", gainId), Form("rms %i EE",  gainValues[gainId]), 100, emin[gainId], emax[gainId]);
      }
      auto iov = iovs.front();
      std::shared_ptr<EcalPedestals> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	// looping over the EB channels, via the dense-index, mapped into EBDetId's
	if (!payload->barrelItems().size()) return false;
	for(int cellid = EBDetId::MIN_HASH;
	    cellid < EBDetId::kSizeForDenseIndexing;
	    ++cellid) {
	  uint32_t rawid = EBDetId::unhashIndex(cellid);  
	  if (payload->find(rawid) == payload->end()) continue;
	  barrel_m[0]->Fill((*payload)[rawid].mean_x12);
	  barrel_r[0]->Fill((*payload)[rawid].rms_x12);
	  barrel_m[1]->Fill((*payload)[rawid].mean_x6);
	  barrel_r[1]->Fill((*payload)[rawid].rms_x6);
	  barrel_m[2]->Fill((*payload)[rawid].mean_x1);
	  barrel_r[2]->Fill((*payload)[rawid].rms_x1);
	}  // loop over cellid
	if (!payload->endcapItems().size()) return false;

	// looping over the EE channels
	for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	  for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	    for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
	      if(EEDetId::validDetId(ix, iy, iz)) {
		EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		uint32_t rawid = myEEId.rawId();
		if (payload->find(rawid) == payload->end()) continue;
		endcap_m[0]->Fill((*payload)[rawid].mean_x12);
		endcap_r[0]->Fill((*payload)[rawid].rms_x12);
		endcap_m[1]->Fill((*payload)[rawid].mean_x6);
		endcap_r[1]->Fill((*payload)[rawid].rms_x6);
		endcap_m[2]->Fill((*payload)[rawid].mean_x1);
		endcap_r[2]->Fill((*payload)[rawid].rms_x1);
	      }  // validDetId 
      }   // if payload.get()
      else return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(111110);      
      TCanvas canvas("CC map","CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i", run));

      float xmi[2] = {0.0 ,  0.50};
      float xma[2] = {0.50,  1.00};
      TPad*** pad = new TPad**[6];
      for (int gId = 0; gId < 6; gId++) {
	pad[gId] = new TPad*[2];
	for (int obj = 0; obj < 2; obj++) {
	  //	  float yma = 1.- (0.17 * gId);
	  //	  float ymi = yma - 0.15;
	  float yma = 0.94- (0.16 * gId);
	  float ymi = yma - 0.14;
	  pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
				   xmi[obj], ymi, xma[obj], yma);
	  pad[gId][obj]->Draw();
	}
      }
     for (int gId = 0; gId < kGains; gId++) {
	pad[gId][0]->cd();
	barrel_m[gId]->Draw();
	pad[gId + kGains][0]->cd();
	barrel_r[gId]->Draw();
	pad[gId][1]->cd();
	endcap_m[gId]->Draw();
	pad[gId + kGains][1]->cd();
	endcap_r[gId]->Draw();
      }
      canvas.SaveAs("ecalpedhist.png");
      return true;
    }   // fill method
  };   //   class EcalPedestalsHist

  /*************************************************
     2d plot of ECAL pedestal of 1 IOV
  *************************************************/

  class EcalPedestalsPlot : public cond::payloadInspector::PlotImage<EcalPedestals> {

  public:
    EcalPedestalsPlot() : cond::payloadInspector::PlotImage<EcalPedestals>( "ECAL pedestal map") {
      setSingleIov(true);
    }
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      TH2F** barrel_r = new TH2F*[kGains];
      TH2F** endc_p_r = new TH2F*[kGains];
      TH2F** endc_m_r = new TH2F*[kGains];
      for (int gainId = 0; gainId < kGains; gainId++) {
	barrel_m[gainId] = new TH2F(Form("EBm%i", gainId),Form("mean %i EB", gainValues[gainId]), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p_m[gainId] = new TH2F(Form("EE+m%i",gainId),Form("mean %i EE+",gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m_m[gainId] = new TH2F(Form("EE-m%i",gainId),Form("mean %i EE-",gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	barrel_r[gainId] = new TH2F(Form("EBr%i", gainId),Form("rms %i EB",  gainValues[gainId]), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p_r[gainId] = new TH2F(Form("EE+r%i",gainId),Form("rms %i EE+", gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m_r[gainId] = new TH2F(Form("EE-r%i",gainId),Form("rms %i EE-", gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      }
      auto iov = iovs.front();
      std::shared_ptr<EcalPedestals> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	// looping over the EB channels, via the dense-index, mapped into EBDetId's
	if (!payload->barrelItems().size()) return false;
	for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {    // loop on EE cells
	  uint32_t rawid = EBDetId::unhashIndex(cellid);  
	  if (payload->find(rawid) == payload->end()) continue;
	  Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
	  Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
	  if(eta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;         //  -84.5 to -0.5
	  barrel_m[0]->Fill(phi, eta, (*payload)[rawid].mean_x12);
	  barrel_r[0]->Fill(phi, eta, (*payload)[rawid].rms_x12);
	  barrel_m[1]->Fill(phi, eta, (*payload)[rawid].mean_x6);
	  barrel_r[1]->Fill(phi, eta, (*payload)[rawid].rms_x6);
	  barrel_m[2]->Fill(phi, eta, (*payload)[rawid].mean_x1);
	  barrel_r[2]->Fill(phi, eta, (*payload)[rawid].rms_x1);
	}  // loop over cellid
	if (!payload->endcapItems().size()) return false;

	// looping over the EE channels
	for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	  for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	    for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
	      if(EEDetId::validDetId(ix, iy, iz)) {
		EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		uint32_t rawid = myEEId.rawId();
		if (payload->find(rawid) == payload->end()) continue;
		if (iz == 1) {
		  endc_p_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12);
		  endc_p_r[0]->Fill(ix, iy, (*payload)[rawid].rms_x12);
		  endc_p_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6);
		  endc_p_r[1]->Fill(ix, iy, (*payload)[rawid].rms_x6);
		  endc_p_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1);
		  endc_p_r[2]->Fill(ix, iy, (*payload)[rawid].rms_x1);
		}
		else { 
		  endc_m_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12);
		  endc_m_r[0]->Fill(ix, iy, (*payload)[rawid].rms_x12);
		  endc_m_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6);
		  endc_m_r[1]->Fill(ix, iy, (*payload)[rawid].rms_x6);
		  endc_m_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1);
		  endc_m_r[2]->Fill(ix, iy, (*payload)[rawid].rms_x1);
		}
	      }  // validDetId 
      }   // if payload.get()
      else return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i", run));

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[6];
      for (int gId = 0; gId < 6; gId++) {
	pad[gId] = new TPad*[3];
	for (int obj = 0; obj < 3; obj++) {
	  float yma = 0.94- (0.16 * gId);
	  float ymi = yma - 0.14;
	  pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
				   xmi[obj], ymi, xma[obj], yma);
	  pad[gId][obj]->Draw();
	}
      }
      float bmin[kGains] ={0.7, 0.5, 0.4};
      float bmax[kGains] ={2.2, 1.3, 0.7};
      float emin[kGains] ={1.5, 0.8, 0.4};
      float emax[kGains] ={2.5, 1.5, 0.8};
      //      TLine* l = new TLine(0., 0., 0., 0.);
      //      l->SetLineWidth(1);
      for (int gId = 0; gId < kGains; gId++) {
	pad[gId][0]->cd();
	DrawEE(endc_m_m[gId], 175., 225.);
	pad[gId + kGains][0]->cd();
	DrawEE(endc_m_r[gId], emin[gId], emax[gId]);
	pad[gId][1]->cd();
	DrawEB(barrel_m[gId], 175., 225.);
	pad[gId + kGains][1]->cd();
	DrawEB(barrel_r[gId], bmin[gId], bmax[gId]);
	pad[gId][2]->cd();
	DrawEE(endc_p_m[gId], 175., 225.);
	pad[gId + kGains][2]->cd();
	DrawEE(endc_p_r[gId], emin[gId], emax[gId]);
      }
      canvas.SaveAs("ecalped.png");
      return true;
    }// fill method

  };   // class EcalPedestalsPlot

  /************************************************************
     2d plot of ECAL pedestals difference between 2 IOVs
  *************************************************************/
  class EcalPedestalsDiff : public cond::payloadInspector::PlotImage<EcalPedestals> {

  public:
    EcalPedestalsDiff() : cond::payloadInspector::PlotImage<EcalPedestals>("ECAL Barrel channel status difference") {
      setSingleIov(false);
    }
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      TH2F** barrel_r = new TH2F*[kGains];
      TH2F** endc_p_r = new TH2F*[kGains];
      TH2F** endc_m_r = new TH2F*[kGains];
      for (int gainId = 0; gainId < kGains; gainId++) {
	barrel_m[gainId] = new TH2F(Form("EBm%i", gainId),Form("mean %i EB", gainValues[gainId]), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p_m[gainId] = new TH2F(Form("EE+m%i",gainId),Form("mean %i EE+",gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m_m[gainId] = new TH2F(Form("EE-m%i",gainId),Form("mean %i EE-",gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	barrel_r[gainId] = new TH2F(Form("EBr%i", gainId),Form("rms %i EB",  gainValues[gainId]), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p_r[gainId] = new TH2F(Form("EE+r%i",gainId),Form("rms %i EE+", gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m_r[gainId] = new TH2F(Form("EE-r%i",gainId),Form("rms %i EE-", gainValues[gainId]), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      }
      unsigned int run[2], irun = 0;
      //      unsigned int irun = 0;
      double meanEB[kGains][kEBChannels], rmsEB[kGains][kEBChannels], meanEE[kGains][kEEChannels], rmsEE[kGains][kEEChannels];
      for ( auto const & iov: iovs) {
	std::shared_ptr<EcalPedestals> payload = fetchPayload( std::get<1>(iov) );
	run[irun] = std::get<0>(iov);
	//	std::cout << "run " << irun << " : " << run[irun] << std::endl;
	if( payload.get() ){
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);  
	    if (payload->find(rawid) == payload->end()) continue;

	    if(irun == 0) {
	      meanEB[0][cellid] = (*payload)[rawid].mean_x12;
	      rmsEB[0][cellid] =  (*payload)[rawid].rms_x12;
	      meanEB[1][cellid] =  (*payload)[rawid].mean_x6;
	      rmsEB[1][cellid] =   (*payload)[rawid].rms_x6;
	      meanEB[2][cellid] =  (*payload)[rawid].mean_x1;
	      rmsEB[2][cellid] = (*payload)[rawid].rms_x1;
	    }
	    else {
	      Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
	      Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
	      if(eta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	      else eta  = eta + 0.5;         //  -84.5 to -0.5
	      barrel_m[0]->Fill(phi, eta, meanEB[0][cellid] - (*payload)[rawid].mean_x12);
	      barrel_r[0]->Fill(phi, eta, rmsEB[0][cellid] - (*payload)[rawid].rms_x12);
	      barrel_m[1]->Fill(phi, eta, meanEB[1][cellid] - (*payload)[rawid].mean_x6);
	      barrel_r[1]->Fill(phi, eta, rmsEB[1][cellid] - (*payload)[rawid].rms_x6);
	      barrel_m[2]->Fill(phi, eta, meanEB[2][cellid] - (*payload)[rawid].mean_x1);
	      barrel_r[2]->Fill(phi, eta, rmsEB[2][cellid] - (*payload)[rawid].rms_x1);
	    }
	  }  // loop over cellid

	  if (!payload->endcapItems().size()) return false;
	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2) {   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++) {
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++) {
		if(EEDetId::validDetId(ix, iy, iz)) {
		  EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		  uint32_t rawid = myEEId.rawId();
		  uint32_t index = myEEId.hashedIndex();
		  if (payload->find(rawid) == payload->end()) continue;
		  if(irun == 0) {
		    meanEE[0][index] = (*payload)[rawid].mean_x12;
		    rmsEE[0][index] =  (*payload)[rawid].rms_x12;
		    meanEE[1][index] =  (*payload)[rawid].mean_x6;
		    rmsEE[1][index] =   (*payload)[rawid].rms_x6;
		    meanEE[2][index] =  (*payload)[rawid].mean_x1;
		    rmsEE[2][index] = (*payload)[rawid].rms_x1;
		  }
		  else {
		    if (iz == 1) {
		      endc_p_m[0]->Fill(ix, iy, meanEE[0][index] - (*payload)[rawid].mean_x12);
		      endc_p_r[0]->Fill(ix, iy, rmsEE[0][index] - (*payload)[rawid].rms_x12);
		      endc_p_m[1]->Fill(ix, iy, meanEE[1][index] - (*payload)[rawid].mean_x6);
		      endc_p_r[1]->Fill(ix, iy, rmsEE[1][index] - (*payload)[rawid].rms_x6);
		      endc_p_m[2]->Fill(ix, iy, meanEE[2][index] - (*payload)[rawid].mean_x1);
		      endc_p_r[2]->Fill(ix, iy, rmsEE[2][index] - (*payload)[rawid].rms_x1);
		    }
		    else { 
		      endc_m_m[0]->Fill(ix, iy, meanEE[0][index] - (*payload)[rawid].mean_x12);
		      endc_m_r[0]->Fill(ix, iy, rmsEE[0][index] - (*payload)[rawid].rms_x12);
		      endc_m_m[1]->Fill(ix, iy, meanEE[1][index] - (*payload)[rawid].mean_x6);
		      endc_m_r[1]->Fill(ix, iy, rmsEE[1][index] - (*payload)[rawid].rms_x6);
		      endc_m_m[2]->Fill(ix, iy, meanEE[2][index] - (*payload)[rawid].mean_x1);
		      endc_m_r[2]->Fill(ix, iy, rmsEE[2][index] - (*payload)[rawid].rms_x1);
		    }
		  }
		} // validDetId
	      }   //   loop over ix
	    }    //  loop over iy
	  }     //  loop over iz
	}  //  if payload.get()
	else return false;
	irun++;
      }      // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i vs %i", run[1], run[0]));

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[6];
      for (int gId = 0; gId < 6; gId++) {
	pad[gId] = new TPad*[3];
	for (int obj = 0; obj < 3; obj++) {
	  float yma = 0.94- (0.16 * gId);
	  float ymi = yma - 0.14;
	  pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
				   xmi[obj], ymi, xma[obj], yma);
	  pad[gId][obj]->Draw();
	}
      }
      float bmin[kGains] ={-0.1, -0.1, -0.1};
      float bmax[kGains] ={ 0.1,  0.1,  0.1};
      float emin[kGains] ={-0.1, -0.1, -0.1};
      float emax[kGains] ={ 0.1,  0.1,  0.1};
      for (int gId = 0; gId < kGains; gId++) {
	pad[gId][0]->cd();
	DrawEE(endc_m_m[gId], -2., 2.);
	pad[gId + kGains][0]->cd();
	DrawEE(endc_m_r[gId], emin[gId], emax[gId]);
	pad[gId][1]->cd();
	DrawEB(barrel_m[gId], -2., 2.);
	pad[gId + kGains][1]->cd();
	DrawEB(barrel_r[gId], bmin[gId], bmax[gId]);
	pad[gId][2]->cd();
	DrawEE(endc_p_m[gId], -2., 2.);
	pad[gId + kGains][2]->cd();
	DrawEE(endc_p_r[gId], emin[gId], emax[gId]);
      }
      canvas.SaveAs("ecalpeddiff.png");
      return true;
    }// fill method
  };   // class EcalPedestalsDiff


  /*************************************************  
     2d histogram of ECAL barrel pedestal of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram2D
  class EcalPedestalsEBMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBMean12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain12 - map",
										     "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // there's no ieta==0 in the EB numbering
	    //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
	    // fill the Histogram2D here
	    //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x12;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEBMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBMean6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain6 - map",
										    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -MAX_IETA, MAX_IETA+1){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x6;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
  	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEBMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
   EcalPedestalsEBMean1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain1 - map",
										    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -MAX_IETA, MAX_IETA+1) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x1;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEEMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEMean12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain12 - map", 
										"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x12;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEEMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEMean6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain6 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
       Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x6;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);
		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEEMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEMean1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain1 - map",
										    "ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x1;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);
		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEBRMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBRMS12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain12 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {								      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // there's no ieta==0 in the EB numbering
	    //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
	    // fill the Histogram2D here
	    //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x12;
	    if(valrms > 2.2) valrms = 2.2;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEBRMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBRMS6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain6 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {										Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x6;
	    if(valrms > 1.5) valrms = 1.5;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEBRMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
   EcalPedestalsEBRMS1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain1 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {								      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x1;
	    if(valrms > 1.0) valrms = 1.0;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEERMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEERMS12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain12 - map", 
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x12;
		    if(valrms > 3.5) valrms = 3.5;
		    if(iz == -1)
		      fillWithValue(ix, iy, valrms);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valrms);

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEERMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
   EcalPedestalsEERMS6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain6 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x6;
		    if(valrms > 2.0) valrms = 2.0;
		    if(iz == -1)
		      fillWithValue( ix, iy, valrms);
		    else
		      fillWithValue( ix + IX_MAX + 20, iy, valrms);
		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEERMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEERMS1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain1 - map",
										   "ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x1;
		    if(valrms > 1.5) valrms = 1.5;
		    if(iz == -1)
		      fillWithValue( ix, iy, valrms);
		    else
		      fillWithValue( ix + IX_MAX + 20, iy, valrms);
		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalPedestals){
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsHist);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsPlot);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsDiff);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS1Map );
}
