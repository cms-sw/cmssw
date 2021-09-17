#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {

  enum { kEBTotalTowers = 2448, kEETotalTowers = 1584 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 17, MAX_IPHI = 72 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 20, IY_MAX = 20 };          // endcaps lower and upper bounds on x and y
  /***********************************************
    2d plot of EcalDQMTowerStatus of 1 IOV
************************************************/
  class EcalDQMTowerStatusPlot : public cond::payloadInspector::PlotImage<EcalDQMTowerStatus> {
  public:
    EcalDQMTowerStatusPlot() : cond::payloadInspector::PlotImage<EcalDQMTowerStatus>("EcalDQMTowerStatus - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* barrel = new TH2F("EB", "EB DQM Tower Status", 72, 0, 72, 34, -17, 17);
      TH2F* endc_p = new TH2F("EE+", "EE+ DQM Tower Status", 22, 0, 22, 22, 0, 22);
      TH2F* endc_m = new TH2F("EE-", "EE- DQM Tower Status", 22, 0, 22, 22, 0, 22);

      auto iov = iovs.front();
      std::shared_ptr<EcalDQMTowerStatus> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      double maxEB = 0, maxEE = 0;

      if (payload.get()) {
        for (uint cellid = 0; cellid < EcalTrigTowerDetId::kEBTotalTowers; ++cellid) {
          if (payload->barrelItems().empty())
            break;
          EcalTrigTowerDetId rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
          if ((*payload).find(rawid) == (*payload).end())
            continue;

          int ieta = rawid.ieta();
          if (ieta > 0)
            ieta--;                     // 1 to 17
          int iphi = rawid.iphi() - 1;  // 0 to 71
          barrel->Fill(iphi, ieta, (*payload)[rawid].getStatusCode());

          if (maxEB < (*payload)[rawid].getStatusCode())
            maxEB = (*payload)[rawid].getStatusCode();
        }

        if (payload->endcapItems().empty())
          return false;

        for (uint cellid = 0; cellid < EcalTrigTowerDetId::kEETotalTowers; ++cellid) {
          if (EcalScDetId::validHashIndex(cellid)) {
            EcalScDetId rawid = EcalScDetId::unhashIndex(cellid);
            if ((*payload).find(rawid) == (*payload).end())
              continue;
            int ix = rawid.ix();  // 1 to 20
            int iy = rawid.iy();  // 1 to 20
            int side = rawid.zside();
            if (side == -1)
              endc_m->Fill(ix, iy, (*payload)[rawid].getStatusCode());
            else
              endc_p->Fill(ix, iy, (*payload)[rawid].getStatusCode());

            if (maxEE < (*payload)[rawid].getStatusCode())
              maxEE = (*payload)[rawid].getStatusCode();
          }
        }

      }  // payload

      TCanvas canvas("CC map", "CC map", 800, 800);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal DQM Tower Status, IOV %i", run));

      //TPad* padb = new TPad("padb","padb", 0., 0.55, 1., 1.);
      TPad* padb = new TPad("padb", "padb", 0., 0.45, 1., 0.9);
      padb->Draw();
      TPad* padem = new TPad("padem", "padem", 0., 0., 0.45, 0.45);
      padem->Draw();
      TPad* padep = new TPad("padep", "padep", 0.55, 0., 1., 0.45);
      padep->Draw();

      TLine* l = new TLine(0., 0., 0., 0.);
      l->SetLineWidth(1);
      padb->cd();
      barrel->SetStats(false);
      barrel->SetMaximum(maxEB);
      barrel->SetMinimum(0);
      barrel->Draw("colz");
      //barrel->Draw("col");
      for (int i = 0; i < 17; i++) {
        Double_t x = 4. + (i * 4);
        l = new TLine(x, -17., x, 17.);
        l->Draw();
      }

      l = new TLine(0., 0., 72., 0.);
      l->Draw();

      padem->cd();
      DrawEE_Tower(endc_m, l, 0, maxEE);

      padep->cd();
      DrawEE_Tower(endc_p, l, 0, maxEE);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /***************************************************
    2d plot of EcalDQMTowerStatus Diff between 2 IOV
   ****************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalDQMTowerStatusDiffBase : public cond::payloadInspector::PlotImage<EcalDQMTowerStatus, nIOVs, ntags> {
  public:
    EcalDQMTowerStatusDiffBase()
        : cond::payloadInspector::PlotImage<EcalDQMTowerStatus, nIOVs, ntags>("EcalDQMTowerStatusDiff - map ") {}
    bool fill() override {
      TH2F* barrel = new TH2F("EB", "EB DQM Tower Status", 72, 0, 72, 34, -17, 17);
      TH2F* endc_p = new TH2F("EE+", "EE+ DQM Tower Status", 22, 0, 22, 22, 0, 22);
      TH2F* endc_m = new TH2F("EE-", "EE- DQM Tower Status", 22, 0, 22, 22, 0, 22);

      unsigned int run[2];
      float pEB[kEBTotalTowers], pEE[kEETotalTowers];
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
        std::shared_ptr<EcalDQMTowerStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          for (uint cellid = 0; cellid < EcalTrigTowerDetId::kEBTotalTowers; ++cellid) {
            if (payload->barrelItems().empty())
              break;

            EcalTrigTowerDetId rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
            if ((*payload).find(rawid) == (*payload).end())
              continue;

            float weight = (*payload)[rawid].getStatusCode();

            if (irun == 0) {
              pEB[cellid] = weight;
            } else {
              int ieta = rawid.ieta();
              if (ieta > 0)
                ieta--;                     // 1 to 17
              int iphi = rawid.iphi() - 1;  // 0 to 71
              unsigned int new_status = (*payload)[rawid].getStatusCode();
              if (new_status != pEB[cellid]) {
                int tmp3 = 0;

                if (new_status > pEB[cellid])
                  tmp3 = 1;
                else
                  tmp3 = -1;

                barrel->Fill(iphi, ieta, 0.05 + 0.95 * (tmp3 > 0));
              }
            }
          }

          if (payload->endcapItems().empty())
            return false;

          for (uint cellid = 0; cellid < EcalTrigTowerDetId::kEETotalTowers; ++cellid) {
            if (EcalScDetId::validHashIndex(cellid)) {
              EcalScDetId rawid = EcalScDetId::unhashIndex(cellid);
              if ((*payload).find(rawid) == (*payload).end())
                continue;

              float weight = (*payload)[rawid].getStatusCode();

              if (irun == 0) {
                pEE[cellid] = weight;
              } else {
                int ix = rawid.ix();  // 1 to 20
                int iy = rawid.iy();  // 1 to 20
                int side = rawid.zside();

                unsigned int new_status = (*payload)[rawid].getStatusCode();
                if (new_status != pEE[cellid]) {
                  int tmp3 = 0;

                  if (new_status > pEE[cellid])
                    tmp3 = 1;
                  else
                    tmp3 = -1;

                  if (side == -1)
                    endc_m->Fill(ix, iy, 0.05 + 0.95 * (tmp3 > 0));
                  else
                    endc_p->Fill(ix, iy, 0.05 + 0.95 * (tmp3 > 0));
                }
              }
            }
          }
        }  // payload
      }

      TCanvas canvas("CC map", "CC map", 800, 800);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2 && len < 58) {
        t1.SetTextSize(0.025);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.04);
        t1.DrawLatex(0.5, 0.96, Form("Ecal DQM Tower Status (Diff), IOV %i vs %i", run[0], run[1]));
      }
      TPad* padb = new TPad("padb", "padb", 0., 0.45, 1., 0.9);
      padb->Draw();
      TPad* padem = new TPad("padem", "padem", 0., 0., 0.45, 0.45);
      padem->Draw();
      TPad* padep = new TPad("padep", "padep", 0.55, 0., 1., 0.45);
      padep->Draw();

      TLine* l = new TLine(0., 0., 0., 0.);
      l->SetLineWidth(1);
      padb->cd();
      barrel->SetStats(false);
      barrel->SetMaximum(1.15);
      barrel->SetMinimum(0);
      barrel->Draw("colz");
      //barrel->Draw("col");
      for (int i = 0; i < 17; i++) {
        Double_t x = 4. + (i * 4);
        l = new TLine(x, -17., x, 17.);
        l->Draw();
      }

      l = new TLine(0., 0., 72., 0.);
      l->Draw();

      padem->cd();
      DrawEE_Tower(endc_m, l, 0, 1.15);

      padep->cd();
      DrawEE_Tower(endc_p, l, 0, 1.15);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalDQMTowerStatusDiffBase
  using EcalDQMTowerStatusDiffOneTag = EcalDQMTowerStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalDQMTowerStatusDiffTwoTags = EcalDQMTowerStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalDQMTowerStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalDQMTowerStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMTowerStatusDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMTowerStatusDiffTwoTags);
}
