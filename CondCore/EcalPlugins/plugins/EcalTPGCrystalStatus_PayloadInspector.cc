#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum {kEBChannels = 61200, kEEChannels = 14648, kSides = 2};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};           // endcaps lower and upper bounds on x and y

  /***********************************************
     2d plot of ECAL TPGCrystalStatus of 1 IOV
  ************************************************/
  class EcalTPGCrystalStatusPlot : public cond::payloadInspector::PlotImage<EcalTPGCrystalStatus> {

  public:
    EcalTPGCrystalStatusPlot() : cond::payloadInspector::PlotImage<EcalTPGCrystalStatus>("ECAL TPGCrystalStatus - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB","EB TPG Crystal Status", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+","EE+ TPG Crystal Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-","EE- TPG Crystal Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EBstat = 0, EEstat[2] = {0, 0};

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGCrystalStatus> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
	  Double_t eta = (Double_t)ieta;
	  if(ieta == 0) continue;
	  else if(ieta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;          //  -84.5 to -0.5
	  for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
	    Double_t phi = (Double_t)iphi - 0.5;
	    EBDetId id(ieta, iphi);
	    double val = (*payload)[id.rawId()].getStatusCode();
	    barrel->Fill(phi, eta, val);
	    if(val > 0) EBstat++;
	  }
	}

	for (int sign = 0; sign < kSides; sign++) {
	  int thesign = sign==1 ? 1:-1;
	  for (int ix = 1; ix <= IX_MAX; ix++) {
	    for (int iy = 1; iy <= IY_MAX; iy++) {
	      if (! EEDetId::validDetId(ix, iy, thesign)) continue;
	      EEDetId id(ix, iy, thesign);
	      double val = (*payload)[id.rawId()].getStatusCode();
	      if (thesign==1) {
		endc_p->Fill(ix, iy, val);
		if(val > 0) EEstat[1]++;
	      }
	      else {
		endc_m->Fill(ix, iy, val);
		if(val > 0) EEstat[0]++;
	      }
	    }// iy
	  } // ix
	}  // side
      }  // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      //      TCanvas canvas("CC map","CC map", 1600, 450);
      Double_t w = 1200;
      Double_t h = 1400;
      TCanvas canvas("c", "c", w, h);
      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGCrystalStatus, IOV %i", run));
 
      //      float xmi[3] = {0.0 , 0.24, 0.76};
      //      float xma[3] = {0.24, 0.76, 1.00};
      float xmi[3] = {0.0, 0.0, 0.5};
      float xma[3] = {1.0, 0.5, 1.0};
      float ymi[3] = {0.47, 0.0, 0.0};
      float yma[3] = {0.94, 0.47, 0.47};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
	pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], ymi[obj], xma[obj], yma[obj]);
	pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEB(barrel, 0., 1.);
      t1.DrawLatex(0.2, 0.94, Form("%i crystals", EBstat));
      pad[1]->cd();
      DrawEE(endc_m, 0., 1.);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEstat[0]));
      pad[2]->cd();
      DrawEE(endc_p, 0., 1.);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEstat[1]));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /************************************************************************
     2d plot of ECAL TPGCrystalStatus difference between 2 IOVs
  ************************************************************************/
  class EcalTPGCrystalStatusDiff : public cond::payloadInspector::PlotImage<EcalTPGCrystalStatus> {

  public:
    EcalTPGCrystalStatusDiff() : cond::payloadInspector::PlotImage<EcalTPGCrystalStatus>("ECAL TPGCrystalStatus difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB","EB difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+","EE+ difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-","EE- difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EBstat = 0, EEstat[2] = {0, 0};

      unsigned int run[2] = {0, 0}, irun = 0;
      float vEB[kEBChannels], vEE[kEEChannels];
      for ( auto const & iov: iovs) {
	std::shared_ptr<EcalTPGCrystalStatus> payload = fetchPayload( std::get<1>(iov) );
	run[irun] = std::get<0>(iov);
	if( payload.get() ){
	  for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
	    Double_t eta = (Double_t)ieta;
	    if(ieta == 0) continue;
	    else if(ieta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	    else eta  = eta + 0.5;          //  -84.5 to -0.5
	    for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
		Double_t phi = (Double_t)iphi - 0.5;
		EBDetId id(ieta, iphi);
		int channel = id.hashedIndex();
		double val = (*payload)[id.rawId()].getStatusCode();
		if(irun == 0) vEB[channel] = val;
		else {
		  double diff = val - vEB[channel];
		  barrel->Fill(phi, eta, diff);
		  if(diff != 0) EBstat++;
		  //		  std::cout << " entry " << EBtot << " mean " << EBmean << " rms " << EBrms << std::endl;
		}
	      }
	    }

	  for (int sign = 0; sign < kSides; sign++) {
	    int thesign = sign==1 ? 1:-1;
	    for (int ix = 1; ix <= IX_MAX; ix++) {
	      for (int iy = 1; iy <= IY_MAX; iy++) {
		if (! EEDetId::validDetId(ix, iy, thesign)) continue;
		EEDetId id(ix, iy, thesign);
		int channel = id.hashedIndex();
		double val = (*payload)[id.rawId()].getStatusCode();
		if(irun == 0) vEE[channel] = val;
		else {
		  double diff = val - vEE[channel];
		  if (thesign==1) {
		    endc_p->Fill(ix, iy, diff);
		    if(diff != 0) EEstat[1]++;
		  }
		  else {
		    endc_m->Fill(ix, iy, diff);
		    if(diff != 0) EEstat[0]++;
		  }
		}
	      }// iy
	    } // ix
	  }  // side
	}  // payload
	else return false;
	irun++;
      }   // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      Double_t w = 1200;
      Double_t h = 1400;
      TCanvas canvas("c", "c", w, h);
      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGCrystalStatus, IOV %i - %i", run[1], run[0]));
 
      //      float xmi[3] = {0.0 , 0.24, 0.76};
      //      float xma[3] = {0.24, 0.76, 1.00};
      float xmi[3] = {0.0, 0.0, 0.5};
      float xma[3] = {1.0, 0.5, 1.0};
      float ymi[3] = {0.47, 0.0, 0.0};
      float yma[3] = {0.94, 0.47, 0.47};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
	pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], ymi[obj], xma[obj], yma[obj]);
	pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEB(barrel, -1., 1.);
      t1.DrawLatex(0.2, 0.94, Form("%i differences", EBstat));
      pad[1]->cd();
      DrawEE(endc_m, -1., 1.);
      t1.DrawLatex(0.15, 0.92, Form("%i differences", EEstat[0]));
      pad[2]->cd();
      DrawEE(endc_p, -1., 1.);
      t1.DrawLatex(0.15, 0.92, Form("%i differences", EEstat[1]));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGCrystalStatus){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusDiff);
}
