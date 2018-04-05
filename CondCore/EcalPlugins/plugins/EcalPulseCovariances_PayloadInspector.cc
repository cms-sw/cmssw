#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum {kEBChannels = 61200, kEEChannels = 14648, kSides = 2, kRMS = 5, TEMPLATESAMPLES = 12};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};           // endcaps lower and upper bounds on x and y

  /*****************************************************
     2d plot of ECAL PulseCovariances of 1 IOV
  *****************************************************/
  class EcalPulseCovariancesPlot : public cond::payloadInspector::PlotImage<EcalPulseCovariances> {

  public:
    EcalPulseCovariancesPlot() : cond::payloadInspector::PlotImage<EcalPulseCovariances>("ECAL PulseCovariances - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F** barrel = new TH2F*[TEMPLATESAMPLES];
      TH2F** endc_p = new TH2F*[TEMPLATESAMPLES];
      TH2F** endc_m = new TH2F*[TEMPLATESAMPLES];
      //      long double EBmean[TEMPLATESAMPLES], EBrms[TEMPLATESAMPLES], EEmean[TEMPLATESAMPLES], EErms[TEMPLATESAMPLES];
      //      int EBtot[TEMPLATESAMPLES], EEtot[TEMPLATESAMPLES];
      double pEBmin[TEMPLATESAMPLES], pEBmax[TEMPLATESAMPLES], pEEmin[TEMPLATESAMPLES], pEEmax[TEMPLATESAMPLES];
      for(int s = 0; s < TEMPLATESAMPLES; ++s) {
	barrel[s] = new TH2F(Form("EBs%i",s),Form("sample %i EB",s), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p[s] = new TH2F(Form("EE+s%i",s),Form("sample %i EE+",s), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m[s] = new TH2F(Form("EE-s%i",s),Form("sample %i EE-",s), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	/*	EBmean[s] = 0.;
	EBrms[s] = 0.;
	EEmean[s] = 0.;
	EErms[s] = 0.;
	EBtot[s] = 0;
	EEtot[s] = 0;*/
	pEBmin[s] = 10.;
	pEBmax[s] = -10.;
	pEEmin[s] = 10.;
	pEEmax[s] = -10.;
      }

      auto iov = iovs.front();
      std::shared_ptr<EcalPulseCovariances> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	//	int chan = 0;
	for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
	  Double_t eta = (Double_t)ieta;
	  if(ieta == 0) continue;
	  else if(ieta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;          //  -84.5 to -0.5
	  for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
	    Double_t phi = (Double_t)iphi - 0.5;
	    EBDetId id(ieta, iphi);
	    /*
	      chan++;
	    if(chan < 20) std::cout << " channel " << chan << " hash " << id.hashedIndex() << " raw " << id.rawId() << std::endl;
	    int rawId = id.rawId();
	    EcalPulseCovariance pulse = (*payload)[rawId];
	    if(chan < 20) {
	      for(int i = 0; i < TEMPLATESAMPLES; ++i) {
		std::cout << std::setw(2) << i;
		for(int j = 0; j < TEMPLATESAMPLES; ++j) {
		  double val = pulse.covval[i][j];
		  std::cout << " " << std::setw(10) << val ;
		}
		std::cout << std::endl;
	      }
	    }
	    */
	    for(int s = 0; s < TEMPLATESAMPLES; ++s) {
	      double val = (*payload)[id.rawId()].covval[0][s];
	      barrel[s]->Fill(phi, eta, val);
	      //	      EBmean[s] = EBmean[s] + val;
	      //	      EBrms[s] = EBrms[s] + val * val;
	      //	      EBtot[s]++;
	      if(val < pEBmin[s]) pEBmin[s] = val;
	      if(val > pEBmax[s]) pEBmax[s] = val;
	    }
	  }
	}

	for (int sign = 0; sign < kSides; sign++) {
	  int thesign = sign==1 ? 1:-1;
	  for (int ix = 1; ix <= IX_MAX; ix++) {
	    for (int iy = 1; iy <= IY_MAX; iy++) {
	      if (! EEDetId::validDetId(ix, iy, thesign)) continue;
	      EEDetId id(ix, iy, thesign);
	      for(int s = 0; s < TEMPLATESAMPLES; ++s) {
		double val = (*payload)[id.rawId()].covval[0][s];
		//		EEmean[s] = EEmean[s] + val;
		//		EErms[s] = EErms[s] + val * val;
		//		EEtot[s]++;
		if(val < pEEmin[s]) pEEmin[s] = val;
		if(val > pEEmax[s]) pEEmax[s] = val;
		if (thesign==1) 
		  endc_p[s]->Fill(ix, iy, val);
		else
		  endc_m[s]->Fill(ix, iy, val);
	      }
	    }// iy
	  } // ix
	}  // side
      }  // payload
      /*
      for(int s = 0; s < TEMPLATESAMPLES; ++s) {
	std::cout << "EB sample " << s << " mean " << EBmean[s] << " rms " << EBrms[s] << " entries " << EBtot[s] 
		  << " EE mean " << EEmean[s] << " rms " << EErms[s] << " entries " << EEtot[s]  << std::endl;
	double vt =(double)EBtot[s];
	EBmean[s] = EBmean[s] / vt;
	EBrms[s] = EBrms[s] / vt - (EBmean[s] * EBmean[s]);
	if(EBrms[s] > 0) EBrms[s] = sqrt(EBrms[s]);
	else EBrms[s] = 1.e-06;
	pEBmin[s] = EBmean[s] - kRMS * EBrms[s];
	pEBmax[s] = EBmean[s] + kRMS * EBrms[s];
	std::cout << "EB sample " << s << " mean " << EBmean[s] << " rms " << EBrms[s] << " entries " << EBtot[s] << " min " << pEBmin[s] << " max " << pEBmax[s] << std::endl;
	//	if(pEBmin[s] <= 0.) pEBmin[s] = 1.e-06;
	vt =(double)EEtot[s];
	EEmean[s] = EEmean[s] / vt;
	EErms[s] = EErms[s] / vt - (EEmean[s] * EEmean[s]);
	if(EErms[s] > 0) EErms[s] = sqrt(EErms[s]);
	else EErms[s] = 1.e-06;
	pEEmin[s] = EEmean[s] - kRMS * EErms[s];
	pEEmax[s] = EEmean[s] + kRMS * EErms[s];
	std::cout << "EE sample " << s  << " mean " << EEmean[s] << " rms " << EErms[s] << " entries " << EEtot[s] << " min " << pEEmin[s] << " max " << pEEmax[s] << std::endl;
	//	if(pEEmin[s] <= 0.) pEEmin[s] = 1.e-06;
      }
      */
      //      for(int s = 0; s < TEMPLATESAMPLES; ++s)
      //	std::cout << " sample " << s << " EB min " << pEBmin[s] << " max " << pEBmax[s] << " EE min " << pEEmin[s] << " max " << pEEmax[s] << std::endl;
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 2800);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal PulseCovariances, IOV %i", run));
 
      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[6];
      for (int s = 0; s < 6; s++) {
	pad[s] = new TPad*[3];
	for (int obj = 0; obj < 3; obj++) {
	  float yma = 0.94 - (0.16 * s);
	  float ymi = yma - 0.14;
	  pad[s][obj] = new TPad(Form("p_%i_%i", obj, s),Form("p_%i_%i", obj, s),
				 xmi[obj], ymi, xma[obj], yma);
	  pad[s][obj]->Draw();
	}
      }

      int ipad=0;
      for(int s = 0; s<7; ++s) {      // plot only the measured ones, not the extrapolated
	// for(int s = 7; s<12; ++s) {	// plot only the extrapolated ones
	if(s == 2) continue;	// do not plot the maximum sample, which is 0 by default
	pad[ipad][0]->cd();
	if(pEBmin[s] == pEBmax[s]) {   // same values everywhere!..
	  pEBmin[s] = pEBmin[s] - 1.e-06;
	  pEBmax[s] = pEBmax[s] + 1.e-06;
	}
	if(pEEmin[s] == pEEmax[s]) {
	  pEEmin[s] = pEEmin[s] - 1.e-06;
	  pEEmax[s] = pEEmax[s] + 1.e-06;
	}
	DrawEE(endc_m[s], pEEmin[s], pEEmax[s]);
	//	pad[ipad][0]->SetLogz(1);
	pad[ipad][1]->cd();
	DrawEB(barrel[s], pEBmin[s], pEBmax[s]);
	//	pad[ipad][1]->SetLogz(1);
	pad[ipad][2]->cd();
	DrawEE(endc_p[s], pEEmin[s], pEEmax[s]);
	//	pad[ipad][2]->SetLogz(1);
	ipad++;
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /*****************************************************
     2d plot of ECAL PulseCovariances matrix of 1 IOV
  *****************************************************/
  class EcalPulseCovariancesMatrix : public cond::payloadInspector::PlotImage<EcalPulseCovariances> {

  public:
    EcalPulseCovariancesMatrix() : cond::payloadInspector::PlotImage<EcalPulseCovariances>("ECAL PulseCovariances - matrix ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
     double EBmean[TEMPLATESAMPLES][TEMPLATESAMPLES], EBrms[TEMPLATESAMPLES][TEMPLATESAMPLES], 
       EEmean[TEMPLATESAMPLES][TEMPLATESAMPLES], EErms[TEMPLATESAMPLES][TEMPLATESAMPLES];
     for(int i = 0; i < TEMPLATESAMPLES; i++) {
       for(int j = 0; j < TEMPLATESAMPLES; j++) {
	 EBmean[i][j] = 0.;
	 EBrms[i][j] = 0.;
	 EEmean[i][j] = 0.;
	 EErms[i][j] = 0.;
       }
     }

      auto iov = iovs.front();
      std::shared_ptr<EcalPulseCovariances> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	//	int chan = 0;
	for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
	  Double_t eta = (Double_t)ieta;
	  if(ieta == 0) continue;
	  else if(ieta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;          //  -84.5 to -0.5
	  for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
	    //Double_t phi = (Double_t)iphi - 0.5;
	    EBDetId id(ieta, iphi);
	    for(int i = 0; i < TEMPLATESAMPLES; ++i) {
	      for(int j = 0; j < TEMPLATESAMPLES; ++j) {
		double val = (*payload)[id.rawId()].covval[i][j];
		EBmean[i][j] = EBmean[i][j] + val;
		EBrms[i][j] = EBrms[i][j] + val * val;
	      }
	    }
	  }
	}

	for (int sign = 0; sign < kSides; sign++) {
	  int thesign = sign==1 ? 1:-1;
	  for (int ix = 1; ix <= IX_MAX; ix++) {
	    for (int iy = 1; iy <= IY_MAX; iy++) {
	      if (! EEDetId::validDetId(ix, iy, thesign)) continue;
	      EEDetId id(ix, iy, thesign);
	      for(int i = 0; i < TEMPLATESAMPLES; i++) {
		for(int j = 0; j < TEMPLATESAMPLES; j++) {
		  double val = (*payload)[id.rawId()].covval[i][j];
		  EEmean[i][j] = EEmean[i][j] + val;
		  EErms[i][j] = EErms[i][j] + val * val;
		}
	      }
	    }// iy
	  } // ix
	}  // side
      }  // payload

      TH2F* barrel_m = new TH2F("EBm", "EB mean", TEMPLATESAMPLES, 0, TEMPLATESAMPLES, TEMPLATESAMPLES, 0., TEMPLATESAMPLES);
      TH2F* barrel_r = new TH2F("EBr", "EB rms",  TEMPLATESAMPLES, 0, TEMPLATESAMPLES, TEMPLATESAMPLES, 0., TEMPLATESAMPLES);
      TH2F* endcap_m = new TH2F("EEm", "EE mean", TEMPLATESAMPLES, 0, TEMPLATESAMPLES, TEMPLATESAMPLES, 0., TEMPLATESAMPLES);
      TH2F* endcap_r = new TH2F("EEr", "EE rms",  TEMPLATESAMPLES, 0, TEMPLATESAMPLES, TEMPLATESAMPLES, 0., TEMPLATESAMPLES);
      for(int i = 0; i < TEMPLATESAMPLES; i++) {
	//	std::cout << "EB sample " << i << std::endl
	//	std::cout << "EB sample " << i;
	for(int j = 0; j < TEMPLATESAMPLES; j++) {
	  double vt =(double)kEBChannels;
	  EBmean[i][j] = EBmean[i][j] / vt;
	  barrel_m->Fill(i, j, EBmean[i][j]);
	  EBrms[i][j] = EBrms[i][j] / vt - (EBmean[i][j] * EBmean[i][j]);
	  if(EBrms[i][j] >= 0) EBrms[i][j] = sqrt(EBrms[i][j]);
	  else EBrms[i][j] = 0.;
	  barrel_r->Fill(i, j, EBrms[i][j]);
	  //	  std::cout << "EB sample " << j << " mean " << EBmean[i][j] << " rms " << EBrms[i][j] << std::endl;
	  //	  std::cout << " " << std::setw(10) << EBrms[i][j];
	  vt =(double)kEEChannels;
	  EEmean[i][j] = EEmean[i][j] / vt;
	  endcap_m->Fill(i, j, EEmean[i][j]);
	  EErms[i][j] = EErms[i][j] / vt - (EEmean[i][j] * EEmean[i][j]);
	  if(EErms[i][j] >= 0) EErms[i][j] = sqrt(EErms[i][j]);
	  else EErms[i][j] = 0.;
	  endcap_r->Fill(i, j, EErms[i][j]);
	  //	  std::cout << "EE sample " << j  << " mean " << EEmean[i][j] << " rms " << EErms[i][j] << std::endl;
	}
	//	std::cout << std::endl;
      }

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1440, 1500);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal PulseCovariances, IOV %i", run));
 
      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad*** pad = new TPad**[2];
      for (int s = 0; s < 2; s++) {
	pad[s] = new TPad*[2];
	for (int obj = 0; obj < 2; obj++) {
	  float yma = 0.94- (0.47 * s);
	  float ymi = yma - 0.45;
	  pad[s][obj] = new TPad(Form("p_%i_%i", obj, s),Form("p_%i_%i", obj, s),
				 xmi[obj], ymi, xma[obj], yma);
	  pad[s][obj]->Draw();
	}
      }

      pad[0][0]->cd();
      //      barrel_m->Draw("COLZ1");
      grid(barrel_m);
      pad[0][1]->cd();
      //      endcap_m->Draw("COLZ1");
      grid(endcap_m);
      pad[1][0]->cd();
      //      barrel_r->Draw("COLZ1");
      grid(barrel_r);
      pad[1][1]->cd();
      //      endcap_r->Draw("COLZ1");
      grid(endcap_r);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method

    void grid(TH2F* matrix) {
      matrix->Draw("COLZ1");
      TLine* lh = new TLine(1., 0., 1., 1.);
      lh->SetLineWidth(0.2);
      TLine* lv = new TLine(1., 0., 1., 1.);
      lv->SetLineWidth(0.2);
      //      double max = (double)TEMPLATESAMPLES;
      for(int i = 1; i < TEMPLATESAMPLES; i++) {
	//	double x = (double)i;
	lv = new TLine(i, 0., i, TEMPLATESAMPLES);
	lv->Draw();
	lh = new TLine(0., i, TEMPLATESAMPLES, i);
	lh->Draw();
      }
    }
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalPulseCovariances){
  PAYLOAD_INSPECTOR_CLASS(EcalPulseCovariancesPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalPulseCovariancesMatrix);
}
