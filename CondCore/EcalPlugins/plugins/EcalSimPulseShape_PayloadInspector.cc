#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalSimPulseShape.h"

#include "TProfile.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"
#include "TMarker.h"

#include <string>

namespace {
  /********************************************
      profile of ECAL SimPulseShape for 1 IOV
  ********************************************/
  class EcalSimPulseShapeProfile : public cond::payloadInspector::PlotImage<EcalSimPulseShape> {
  public:
    EcalSimPulseShapeProfile() : cond::payloadInspector::PlotImage<EcalSimPulseShape>("ECAL SimPulseShape - Profile ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<EcalSimPulseShape> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      TProfile *barrel, *endcap, *apd;
      int EBnbin = 0, EEnbin = 0, APDnbin = 0;
      double EBxmax, EExmax, APDxmax, EBth, EEth, APDth;
      if (payload.get()) {
        EBth = (*payload).barrel_thresh;
        EEth = (*payload).endcap_thresh;
        APDth = (*payload).apd_thresh;
        double time = (*payload).time_interval;
        std::vector<double> EBshape = (*payload).barrel_shape;
        std::vector<double> EEshape = (*payload).endcap_shape;
        std::vector<double> APDshape = (*payload).apd_shape;
        EBnbin = EBshape.size();
        EBxmax = EBnbin * time;
        EEnbin = EBshape.size();
        EExmax = EEnbin * time;
        APDnbin = APDshape.size();
        APDxmax = APDnbin * time;

        /*	std::cout << "_thresh barrel " << EBth << " endcap " << EEth << " apd " << APDth << std::endl
		  << " time interval " << time << std::endl
		  << " shape size barrel " << EBnbin << " endcap " << EEnbin << " apd " << APDnbin
		  << std::endl; */
        barrel = new TProfile("EBshape", "", EBnbin, 0, EBxmax);
        endcap = new TProfile("EEshape", "", EEnbin, 0, EExmax);
        apd = new TProfile("APDshape", "", APDnbin, 0, APDxmax);
        for (int s = 0; s < EBnbin; s++) {
          double val = EBshape[s];
          barrel->Fill(s, val);
        }
        for (int s = 0; s < EEnbin; s++) {
          double val = EEshape[s];
          endcap->Fill(s, val);
        }
        for (int s = 0; s < APDnbin; s++) {
          double val = APDshape[s];
          apd->Fill(s, val);
        }
      }  // if payload.get()
      else
        return false;

      //      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("ESPS", "ESPS", 1000, 500);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Sim Pulse Shape, IOV %i", run));

      if (EBnbin == EEnbin && EBnbin == APDnbin) {
        TPad *pad = new TPad("p_0", "p_0", 0.0, 0.0, 1.0, 0.95);
        pad->Draw();
        pad->cd();
        barrel->SetXTitle("time (ns)");
        barrel->SetYTitle("normalized amplitude (ADC#)");
        barrel->SetMarkerColor(kBlack);
        barrel->SetMarkerStyle(24);
        //	barrel->SetMarkerSize(0.5);
        barrel->Draw("P");
        TMarker *EBMarker = new TMarker(0.58, 0.85, 24);
        EBMarker->SetNDC();
        EBMarker->SetMarkerSize(1.0);
        EBMarker->SetMarkerColor(kBlack);
        EBMarker->Draw();
        t1.SetTextAlign(12);
        t1.DrawLatex(0.59, 0.85, Form("EB pulse, threshold %f", EBth));

        endcap->SetMarkerColor(kRed);
        endcap->SetMarkerStyle(24);
        //	endcap->SetMarkerSize(0.5);
        endcap->Draw("PSAME");
        TMarker *EEMarker = new TMarker(0.58, 0.78, 24);
        EEMarker->SetNDC();
        EEMarker->SetMarkerSize(1.0);
        EEMarker->SetMarkerColor(kRed);
        EEMarker->Draw();
        t1.SetTextColor(kRed);
        t1.DrawLatex(0.59, 0.78, Form("EE pulse, threshold %f", EEth));

        apd->SetMarkerColor(kBlue);
        apd->SetMarkerStyle(24);
        //	apd->SetMarkerSize(0.5);
        apd->Draw("PSAME");
        TMarker *APDMarker = new TMarker(0.58, 0.71, 24);
        APDMarker->SetNDC();
        APDMarker->SetMarkerSize(1.0);
        APDMarker->SetMarkerColor(kBlue);
        APDMarker->Draw();
        t1.SetTextColor(kBlue);
        t1.DrawLatex(0.59, 0.71, Form("APD pulse, threshold %f", APDth));
      } else {
        canvas.SetCanvasSize(1000, 1000);
        TPad **pad = new TPad *[3];
        for (int s = 0; s < 3; s++) {
          float yma = 0.94 - (0.31 * s);
          float ymi = yma - 0.29;
          pad[s] = new TPad(Form("p_%i", s), Form("p_%i", s), 0.0, ymi, 1.0, yma);
          pad[s]->Draw();
        }
        pad[0]->cd();
        barrel->Draw("P");
        barrel->SetXTitle("time (ns)");
        barrel->SetYTitle("normalized amplitude (ADC#)");
        barrel->SetMarkerColor(kBlack);
        barrel->SetMarkerStyle(24);
        TMarker *EBMarker = new TMarker(0.58, 0.80, 24);
        EBMarker->SetNDC();
        EBMarker->Draw();
        EBMarker->SetMarkerSize(1.0);
        EBMarker->SetMarkerColor(kBlack);
        t1.SetTextAlign(12);
        t1.DrawLatex(0.59, 0.80, Form("EB pulse, threshold %f", EBth));

        pad[1]->cd();
        endcap->SetMarkerStyle(24);
        endcap->SetMarkerColor(kRed);
        endcap->Draw("P");
        endcap->SetXTitle("time (ns)");
        endcap->SetYTitle("normalized amplitude (ADC#)");
        TMarker *EEMarker = new TMarker(0.58, 0.8, 24);
        EEMarker->SetNDC();
        EEMarker->Draw();
        EEMarker->SetMarkerSize(1.0);
        EEMarker->SetMarkerColor(kRed);
        t1.SetTextColor(kRed);
        t1.DrawLatex(0.59, 0.80, Form("EE pulse, threshold %f", EEth));

        pad[2]->cd();
        apd->SetMarkerStyle(24);
        apd->Draw("P");
        apd->SetMarkerColor(kBlue);
        apd->SetXTitle("time (ns)");
        apd->SetYTitle("normalized amplitude (ADC#)");
        TMarker *APDMarker = new TMarker(0.58, 0.8, 24);
        APDMarker->SetNDC();
        APDMarker->Draw();
        APDMarker->SetMarkerSize(1.0);
        APDMarker->SetMarkerColor(kBlue);
        t1.SetTextColor(kBlue);
        t1.DrawLatex(0.59, 0.80, Form("APD pulse, threshold %f", APDth));
      }
      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalSimPulseShape) { PAYLOAD_INSPECTOR_CLASS(EcalSimPulseShapeProfile); }
