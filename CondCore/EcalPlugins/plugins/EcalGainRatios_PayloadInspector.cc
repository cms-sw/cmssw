#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

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
  enum { kEBChannels = 61200, kEEChannels = 14648 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /******************************************
     2d plot of ECAL GainRatios of 1 IOV
  ******************************************/
  class EcalGainRatiosPlot : public cond::payloadInspector::PlotImage<EcalGainRatios> {
  public:
    EcalGainRatiosPlot() : cond::payloadInspector::PlotImage<EcalGainRatios>("ECAL Gain Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* barrel_12O6 = new TH2F("EB_12O6", "EB gain 12/6", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p_12O6 = new TH2F("EE+_12O6", "EE+ gain 12/6", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m_12O6 = new TH2F("EE-_12O6", "EE- gain 12/6", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* barrel_6O1 = new TH2F("EB_6O1", "EB gain 6/1", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p_6O1 = new TH2F("EE+_6O1", "EE+ gain 6/1", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m_6O1 = new TH2F("EE-_6O1", "EE- gain 6/1", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH1F* b_12O6 = new TH1F("b_12O6", "EB gain 12/6", 50, 1.8, 2.1);
      TH1F* e_12O6 = new TH1F("e_12O6", "EE gain 12/6", 50, 1.8, 2.1);
      TH1F* b_6O1 = new TH1F("b_6O1", "EB gain 6/1", 100, 5.3, 6.3);
      TH1F* e_6O1 = new TH1F("e_6O1", "EE gain 6/1", 100, 5.3, 6.3);

      auto iov = iovs.front();
      std::shared_ptr<EcalGainRatios> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        // looping over the EB channels, via the dense-index, mapped into EBDetId's
        for (int cellid = 0; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {  // loop on EB cells
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
          Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
          if (eta > 0.)
            eta = eta - 0.5;  //   0.5 to 84.5
          else
            eta = eta + 0.5;  //  -84.5 to -0.5
          barrel_12O6->Fill(phi, eta, (*payload)[rawid].gain12Over6());
          barrel_6O1->Fill(phi, eta, (*payload)[rawid].gain6Over1());
          b_12O6->Fill((*payload)[rawid].gain12Over6());
          b_6O1->Fill((*payload)[rawid].gain6Over1());
        }  // loop over cellid

        // looping over the EE channels
        for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
          if (!EEDetId::validHashIndex(cellid))
            continue;
          uint32_t rawid = EEDetId::unhashIndex(cellid);
          EEDetId myEEId(rawid);
          if (myEEId.zside() == 1) {
            endc_p_12O6->Fill(myEEId.ix(), myEEId.iy(), (*payload)[rawid].gain12Over6());
            endc_p_6O1->Fill(myEEId.ix(), myEEId.iy(), (*payload)[rawid].gain6Over1());
          } else {
            endc_m_12O6->Fill(myEEId.ix(), myEEId.iy(), (*payload)[rawid].gain12Over6());
            endc_m_6O1->Fill(myEEId.ix(), myEEId.iy(), (*payload)[rawid].gain6Over1());
          }
          e_12O6->Fill((*payload)[rawid].gain12Over6());
          e_6O1->Fill((*payload)[rawid].gain6Over1());
        }  // validDetId
      }    // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1680, 1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Gain Ratios, IOV %i", run));

      float xmi[3] = {0.0, 0.22, 0.78};
      float xma[3] = {0.22, 0.78, 1.00};
      TPad*** pad = new TPad**[2];
      for (int gId = 0; gId < 2; gId++) {
        pad[gId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * gId);
          float ymi = yma - 0.28;
          pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId), Form("p_%i_%i", obj, gId), xmi[obj], ymi, xma[obj], yma);
          pad[gId][obj]->Draw();
        }
      }
      TPad** pad1 = new TPad*[4];
      for (int obj = 0; obj < 4; obj++) {
        float xmi = 0.26 * obj;
        float xma = xmi + 0.22;
        pad1[obj] = new TPad(Form("p1_%i", obj), Form("p1_%i", obj), xmi, 0.0, xma, 0.32);
        pad1[obj]->Draw();
      }

      float min12O6 = 1.8, max12O6 = 2.1, min6O1 = 5.3, max6O1 = 6.3;
      pad[0][0]->cd();
      DrawEE(endc_m_12O6, min12O6, max12O6);
      endc_m_12O6->SetStats(false);
      pad[0][1]->cd();
      DrawEB(barrel_12O6, min12O6, max12O6);
      barrel_12O6->SetStats(false);
      pad[0][2]->cd();
      DrawEE(endc_p_12O6, min12O6, max12O6);
      endc_p_12O6->SetStats(false);
      pad[1][0]->cd();
      DrawEE(endc_m_6O1, min6O1, max6O1);
      endc_m_6O1->SetStats(false);
      pad[1][1]->cd();
      DrawEB(barrel_6O1, min6O1, max6O1);
      barrel_6O1->SetStats(false);
      pad[1][2]->cd();
      DrawEE(endc_p_6O1, min6O1, max6O1);
      endc_p_6O1->SetStats(false);

      gStyle->SetOptStat(111110);
      pad1[0]->cd();
      b_12O6->Draw();
      pad1[0]->Update();
      TPaveStats* st = (TPaveStats*)b_12O6->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[1]->cd();
      e_12O6->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)e_12O6->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[2]->cd();
      b_6O1->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)b_6O1->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[3]->cd();
      e_6O1->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)e_6O1->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     2d plot of ECAL GainRatios difference between 2 IOVs
  **********************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalGainRatiosDiffBase : public cond::payloadInspector::PlotImage<EcalGainRatios, nIOVs, ntags> {
  public:
    EcalGainRatiosDiffBase()
        : cond::payloadInspector::PlotImage<EcalGainRatios, nIOVs, ntags>("ECAL Gain Ratios difference") {}

    bool fill() override {
      TH2F* barrel_12O6 =
          new TH2F("EB_12O6", "EB gain 12/6 difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p_12O6 =
          new TH2F("EE+_12O6", "EE+ gain 12/6 difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m_12O6 =
          new TH2F("EE-_12O6", "EE- gain 12/6 difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* barrel_6O1 =
          new TH2F("EB_6O1", "EB gain 6/1 difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p_6O1 =
          new TH2F("EE+_6O1", "EE+ gain 6/1 difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m_6O1 =
          new TH2F("EE-_6O1", "EE- gain 6/1 difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH1F* b_12O6 = new TH1F("b_12O6", "EB gain 12/6 difference", 50, -0.1, 0.1);
      TH1F* e_12O6 = new TH1F("e_12O6", "EE gain 12/6 difference", 50, -0.1, 0.1);
      TH1F* b_6O1 = new TH1F("b_6O1", "EB gain 6/1 difference", 100, -0.1, 0.1);
      TH1F* e_6O1 = new TH1F("e_6O1", "EE gain 6/1 difference", 100, -0.1, 0.1);

      unsigned int run[2];
      float gEB[3][kEBChannels], gEE[3][kEEChannels];
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
        std::shared_ptr<EcalGainRatios> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          for (int cellid = 0; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {  // loop on EB cells
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            if (irun == 0) {
              gEB[0][cellid] = (*payload)[rawid].gain12Over6();
              gEB[1][cellid] = (*payload)[rawid].gain6Over1();
            } else {
              Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
              Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
              if (eta > 0.)
                eta = eta - 0.5;  //   0.5 to 84.5
              else
                eta = eta + 0.5;  //  -84.5 to -0.5
              float diff = gEB[0][cellid] - (*payload)[rawid].gain12Over6();
              barrel_12O6->Fill(phi, eta, diff);
              b_12O6->Fill(diff);
              diff = gEB[1][cellid] - (*payload)[rawid].gain6Over1();
              barrel_6O1->Fill(phi, eta, diff);
              b_6O1->Fill(diff);
            }
          }  // loop over cellid

          // looping over the EE channels
          for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
            if (!EEDetId::validHashIndex(cellid))
              continue;
            uint32_t rawid = EEDetId::unhashIndex(cellid);
            EEDetId myEEId(rawid);
            if (irun == 0) {
              gEE[0][cellid] = (*payload)[rawid].gain12Over6();
              gEE[1][cellid] = (*payload)[rawid].gain6Over1();
            } else {
              float diff1 = gEE[0][cellid] - (*payload)[rawid].gain12Over6();
              float diff2 = gEE[1][cellid] - (*payload)[rawid].gain6Over1();
              if (myEEId.zside() == 1) {
                endc_p_12O6->Fill(myEEId.ix(), myEEId.iy(), diff1);
                endc_p_6O1->Fill(myEEId.ix(), myEEId.iy(), diff2);
              } else {
                endc_m_12O6->Fill(myEEId.ix(), myEEId.iy(), diff1);
                endc_m_6O1->Fill(myEEId.ix(), myEEId.iy(), diff2);
              }
              e_12O6->Fill(diff1);
              e_6O1->Fill(diff2);
            }
          }  // loop over cellid
        }    //  if payload.get()
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1680, 1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2 && len < 70) {
        t1.SetTextSize(0.03);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }

      float xmi[3] = {0.0, 0.22, 0.78};
      float xma[3] = {0.22, 0.78, 1.00};
      TPad*** pad = new TPad**[2];
      for (int gId = 0; gId < 2; gId++) {
        pad[gId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * gId);
          float ymi = yma - 0.28;
          pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId), Form("p_%i_%i", obj, gId), xmi[obj], ymi, xma[obj], yma);
          pad[gId][obj]->Draw();
        }
      }
      TPad** pad1 = new TPad*[4];
      for (int obj = 0; obj < 4; obj++) {
        float xmi = 0.26 * obj;
        float xma = xmi + 0.22;
        pad1[obj] = new TPad(Form("p1_%i", obj), Form("p1_%i", obj), xmi, 0.0, xma, 0.32);
        pad1[obj]->Draw();
      }

      float min12O6 = -0.1, max12O6 = 0.1, min6O1 = -0.1, max6O1 = 0.1;
      pad[0][0]->cd();
      DrawEE(endc_m_12O6, min12O6, max12O6);
      endc_m_12O6->SetStats(false);
      pad[0][1]->cd();
      DrawEB(barrel_12O6, min12O6, max12O6);
      barrel_12O6->SetStats(false);
      pad[0][2]->cd();
      DrawEE(endc_p_12O6, min12O6, max12O6);
      endc_p_12O6->SetStats(false);
      pad[1][0]->cd();
      DrawEE(endc_m_6O1, min6O1, max6O1);
      endc_m_6O1->SetStats(false);
      pad[1][1]->cd();
      DrawEB(barrel_6O1, min6O1, max6O1);
      barrel_6O1->SetStats(false);
      pad[1][2]->cd();
      DrawEE(endc_p_6O1, min6O1, max6O1);
      endc_p_6O1->SetStats(false);

      gStyle->SetOptStat(111110);
      pad1[0]->cd();
      b_12O6->Draw();
      pad1[0]->Update();
      TPaveStats* st = (TPaveStats*)b_12O6->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[1]->cd();
      e_12O6->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)e_12O6->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[2]->cd();
      b_6O1->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)b_6O1->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position
      pad1[3]->cd();
      e_6O1->Draw();
      pad1[0]->Update();
      st = (TPaveStats*)e_6O1->FindObject("stats");
      st->SetX1NDC(0.6);   //new x start position
      st->SetY1NDC(0.75);  //new y start position

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalGainRatiosDiffBase
  using EcalGainRatiosDiffOneTag = EcalGainRatiosDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalGainRatiosDiffTwoTags = EcalGainRatiosDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalGainRatios) {
  PAYLOAD_INSPECTOR_CLASS(EcalGainRatiosPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalGainRatiosDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalGainRatiosDiffTwoTags);
}
