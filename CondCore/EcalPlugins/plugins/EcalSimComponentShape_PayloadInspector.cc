#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalSimComponentShape.h"

#include "TProfile.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"
#include "TMarker.h"

#include <string>

namespace {
  /********************************************
      profile of ECAL SimComponentShape for 1 IOV
  ********************************************/
  class EcalSimComponentShapeProfile : public cond::payloadInspector::PlotImage<EcalSimComponentShape> {
  public:
    EcalSimComponentShapeProfile()
        : cond::payloadInspector::PlotImage<EcalSimComponentShape>("ECAL SimComponentShape - Profile ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<EcalSimComponentShape> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      std::vector<TProfile *> profiles;
      std::vector<int> EBnbins;
      std::vector<double> EBxmaxs;
      double EBth;
      int iShape = 0;
      if (payload.get()) {
        EBth = (*payload).barrel_thresh;
        double time = (*payload).time_interval;
        std::vector<std::vector<float> > EBshapes = (*payload).barrel_shapes;
        char nameBuffer[50];
        for (auto EBshape : EBshapes) {
          EBnbins.push_back(EBshape.size());
          EBxmaxs.push_back(EBnbins[iShape] * time);
          sprintf(nameBuffer, "EBComponentShape_%d", iShape);
          profiles.push_back(new TProfile(nameBuffer, "", EBnbins[iShape], 0, EBxmaxs[iShape]));
          for (int s = 0; s < EBnbins[iShape]; s++) {
            double val = EBshape[s];
            profiles[iShape]->Fill(s, val);
          }
          ++iShape;
        }
      }  // if payload.get()
      else
        return false;

      //      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      gStyle->SetPalette(kThermometer);
      TCanvas canvas("ESPS", "ESPS", 1000, 500);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Sim Component Shapes, IOV %i", run));

      TPad *pad = new TPad("p_0", "p_0", 0.0, 0.0, 1.0, 0.95);
      pad->Draw();
      pad->cd();
      iShape = 0;
      for (auto profile : profiles) {
        if (iShape == 0) {
          profile->SetXTitle("time (ns)");
          profile->SetYTitle("normalized amplitude (ADC#)");
          profile->GetXaxis()->SetRangeUser(0., 275.);
          profile->GetYaxis()->SetRangeUser(0., 1.25);
        }
        profile->SetMarkerColor(TColor::GetPalette().At(10 * iShape));
        profile->SetLineColor(TColor::GetPalette().At(10 * iShape));
        profile->SetMarkerStyle(20);
        profile->SetMarkerSize(.25);
        if (iShape == 0) {
          profile->Draw("");
        } else {
          profile->Draw("SAME");
        }
        ++iShape;
      }
      pad->BuildLegend(.7, .225, .85, .8);
      t1.SetTextAlign(12);
      t1.DrawLatex(0.4, 0.85, Form("EB component shapes, threshold %f", EBth));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalSimComponentShape) { PAYLOAD_INSPECTOR_CLASS(EcalSimComponentShapeProfile); }
