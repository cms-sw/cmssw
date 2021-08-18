#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
#include "CondCore/EcalPlugins/plugins/EcalBadCrystalsCount.h"
// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648, kSides = 2 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /***********************************************
     2d plot of ECAL TPGCrystalStatus of 1 IOV
  ************************************************/
  class EcalTPGCrystalStatusPlot : public cond::payloadInspector::PlotImage<EcalTPGCrystalStatus> {
  public:
    EcalTPGCrystalStatusPlot()
        : cond::payloadInspector::PlotImage<EcalTPGCrystalStatus>("ECAL TPGCrystalStatus - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* barrel = new TH2F("EB", "EB TPG Crystal Status", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "EE+ TPG Crystal Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE- TPG Crystal Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EBstat = 0, EEstat[2] = {0, 0};

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGCrystalStatus> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
          Double_t eta = (Double_t)ieta;
          if (ieta == 0)
            continue;
          else if (ieta > 0.)
            eta = eta - 0.5;  //   0.5 to 84.5
          else
            eta = eta + 0.5;  //  -84.5 to -0.5
          for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
            Double_t phi = (Double_t)iphi - 0.5;
            EBDetId id(ieta, iphi);
            double val = (*payload)[id.rawId()].getStatusCode();
            barrel->Fill(phi, eta, val);
            if (val > 0)
              EBstat++;
          }
        }

        for (int sign = 0; sign < kSides; sign++) {
          int thesign = sign == 1 ? 1 : -1;
          for (int ix = 1; ix <= IX_MAX; ix++) {
            for (int iy = 1; iy <= IY_MAX; iy++) {
              if (!EEDetId::validDetId(ix, iy, thesign))
                continue;
              EEDetId id(ix, iy, thesign);
              double val = (*payload)[id.rawId()].getStatusCode();
              if (thesign == 1) {
                endc_p->Fill(ix, iy, val);
                if (val > 0)
                  EEstat[1]++;
              } else {
                endc_m->Fill(ix, iy, val);
                if (val > 0)
                  EEstat[0]++;
              }
            }  // iy
          }    // ix
        }      // side
      }        // payload

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
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], ymi[obj], xma[obj], yma[obj]);
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
    }  // fill method
  };

  /************************************************************************
     2d plot of ECAL TPGCrystalStatus difference between 2 IOVs
  ************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalTPGCrystalStatusDiffBase : public cond::payloadInspector::PlotImage<EcalTPGCrystalStatus, nIOVs, ntags> {
  public:
    EcalTPGCrystalStatusDiffBase()
        : cond::payloadInspector::PlotImage<EcalTPGCrystalStatus, nIOVs, ntags>("ECAL TPGCrystalStatus difference") {}

    bool fill() override {
      TH2F* barrel = new TH2F("EB", "EB difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "EE+ difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE- difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EBstat = 0, EEstat[2] = {0, 0};

      unsigned int run[2] = {0, 0};
      float vEB[kEBChannels], vEE[kEEChannels];
      std::string l_tagname[2];
      auto iovs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      l_tagname[0] = cond::payloadInspector::PlotBase::getTag<0>().name;
      auto firstiov = iovs.front();
      run[0] = std::get<0>(firstiov);
      std::tuple<cond::Time_t, cond::Hash> lastiov;
      if (ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname[1] = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = iovs.back();
        l_tagname[1] = l_tagname[0];
      }
      run[1] = std::get<0>(lastiov);
      for (int irun = 0; irun < nIOVs; irun++) {
        std::shared_ptr<EcalTPGCrystalStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
            Double_t eta = (Double_t)ieta;
            if (ieta == 0)
              continue;
            else if (ieta > 0.)
              eta = eta - 0.5;  //   0.5 to 84.5
            else
              eta = eta + 0.5;  //  -84.5 to -0.5
            for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
              Double_t phi = (Double_t)iphi - 0.5;
              EBDetId id(ieta, iphi);
              int channel = id.hashedIndex();
              double val = (*payload)[id.rawId()].getStatusCode();
              if (irun == 0)
                vEB[channel] = val;
              else {
                double diff = val - vEB[channel];
                barrel->Fill(phi, eta, diff);
                if (diff != 0)
                  EBstat++;
                //      std::cout << " entry " << EBtot << " mean " << EBmean << " rms " << EBrms << std::endl;
              }
            }
          }

          for (int sign = 0; sign < kSides; sign++) {
            int thesign = sign == 1 ? 1 : -1;
            for (int ix = 1; ix <= IX_MAX; ix++) {
              for (int iy = 1; iy <= IY_MAX; iy++) {
                if (!EEDetId::validDetId(ix, iy, thesign))
                  continue;
                EEDetId id(ix, iy, thesign);
                int channel = id.hashedIndex();
                double val = (*payload)[id.rawId()].getStatusCode();
                if (irun == 0)
                  vEE[channel] = val;
                else {
                  double diff = val - vEE[channel];
                  if (thesign == 1) {
                    endc_p->Fill(ix, iy, diff);
                    if (diff != 0)
                      EEstat[1]++;
                  } else {
                    endc_m->Fill(ix, iy, diff);
                    if (diff != 0)
                      EEstat[0]++;
                  }
                }
              }  // iy
            }    // ix
          }      // side
        }        // payload
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      Double_t w = 1200;
      Double_t h = 1400;
      TCanvas canvas("c", "c", w, h);
      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2) {
        if (len < 80) {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("%s %i - %s %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.05);
          t1.DrawLatex(0.5, 0.96, Form("Ecal TPGCrystalStatus, IOV %i - %i", run[1], run[0]));
        }
      } else {
        t1.SetTextSize(0.03);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }

      //      float xmi[3] = {0.0 , 0.24, 0.76};
      //      float xma[3] = {0.24, 0.76, 1.00};
      float xmi[3] = {0.0, 0.0, 0.5};
      float xma[3] = {1.0, 0.5, 1.0};
      float ymi[3] = {0.47, 0.0, 0.0};
      float yma[3] = {0.94, 0.47, 0.47};
      std::vector<TPad*> pad;
      for (int obj = 0; obj < 3; obj++) {
        pad.push_back(new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], ymi[obj], xma[obj], yma[obj]));
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

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalTPGCrystalStatusDiffBase
  using EcalTPGCrystalStatusDiffOneTag = EcalTPGCrystalStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalTPGCrystalStatusDiffTwoTags = EcalTPGCrystalStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

  /*********************************************************
    2d plot of EcalTPGCrystalStatus Error Summary of 1 IOV
   *********************************************************/
  class EcalTPGCrystalStatusSummaryPlot : public cond::payloadInspector::PlotImage<EcalTPGCrystalStatus> {
  public:
    EcalTPGCrystalStatusSummaryPlot()
        : cond::payloadInspector::PlotImage<EcalTPGCrystalStatus>("Ecal TPGCrystal Status Error Summary - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<EcalTPGCrystalStatus> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F* align;                          //pointer to align which is a 2D histogram

      int NbRows = 3;
      int NbColumns = 3;

      if (payload.get()) {  //payload is an iov retrieved from payload using hash.

        align = new TH2F("Ecal TPGCrystal Status Error Summary",
                         "EB/EE-/EE+            ErrorCount            Total Number",
                         NbColumns,
                         0,
                         NbColumns,
                         NbRows,
                         0,
                         NbRows);

        long unsigned int ebErrorCount = 0;
        long unsigned int ee1ErrorCount = 0;
        long unsigned int ee2ErrorCount = 0;

        long unsigned int ebTotal = (payload->barrelItems()).size();
        long unsigned int ee1Total = 0;
        long unsigned int ee2Total = 0;

        getBarrelErrorSummary<EcalTPGCrystalStatusCode>(payload->barrelItems(), ebErrorCount);
        getEndCapErrorSummary<EcalTPGCrystalStatusCode>(
            payload->endcapItems(), ee1ErrorCount, ee2ErrorCount, ee1Total, ee2Total);

        double row = NbRows - 0.5;

        //EB summary values
        align->Fill(0.5, row, 1);
        align->Fill(1.5, row, ebErrorCount);
        align->Fill(2.5, row, ebTotal);

        row--;
        //EE- summary values
        align->Fill(0.5, row, 2);
        align->Fill(1.5, row, ee1ErrorCount);
        align->Fill(2.5, row, ee1Total);

        row--;

        //EE+ summary values
        align->Fill(0.5, row, 3);
        align->Fill(1.5, row, ee2ErrorCount);
        align->Fill(2.5, row, ee2Total);

      }  // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1000, 1000);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.04);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("EcalTPGCrystalStatus Error Summary, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, NbColumns);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGCrystalStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGCrystalStatusSummaryPlot);
}
