#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"

#include <string>

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648, kSides = 2, kValues = 3, kRMS = 5 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /*************************************************
     2d plot of ECAL LinearCorrections of 1 IOV
  *************************************************/
  class EcalLinearCorrectionsPlot : public cond::payloadInspector::PlotImage<EcalLinearCorrections> {
  public:
    EcalLinearCorrectionsPlot()
        : cond::payloadInspector::PlotImage<EcalLinearCorrections>("ECAL LinearCorrections - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F** barrel = new TH2F*[kValues];
      TH2F** endc_p = new TH2F*[kValues];
      TH2F** endc_m = new TH2F*[kValues];
      double EBmean[kValues], EBrms[kValues], EEmean[kValues], EErms[kValues], pEBmin[kValues], pEBmax[kValues],
          pEEmin[kValues], pEEmax[kValues];
      int EBtot[kValues], EEtot[kValues];
      for (int valId = 0; valId < kValues; valId++) {
        barrel[valId] = new TH2F(
            Form("EBp%i", valId), Form("EBp%i", valId), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
        endc_p[valId] = new TH2F(
            Form("EE+p%i", valId), Form("EE+p%i", valId), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        endc_m[valId] = new TH2F(
            Form("EE-p%i", valId), Form("EE-p%i", valId), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        EBmean[valId] = 0.;
        EBrms[valId] = 0.;
        EEmean[valId] = 0.;
        EErms[valId] = 0.;
        EBtot[valId] = 0;
        EEtot[valId] = 0;
      }

      auto iov = iovs.front();
      std::shared_ptr<EcalLinearCorrections> payload = fetchPayload(std::get<1>(iov));
      unsigned long IOV = std::get<0>(iov);
      int run = 0;
      if (IOV < 4294967296)
        run = std::get<0>(iov);
      else {  // time type IOV
        run = IOV >> 32;
        //	std::cout << " IOV " << std::hex << IOV << " run " << run << std::endl;
        //	std::cout << " IOV " << std::dec << IOV << " run " << run << std::endl;
      }
      if (payload.get()) {
        for (int ieta = -MAX_IETA; ieta <= MAX_IETA; ieta++) {
          Double_t eta = (Double_t)ieta;
          if (ieta == 0)
            continue;
          else if (ieta > 0)
            eta = eta - 0.5;  //   0.5 to 84.5
          else
            eta = eta + 0.5;  //  -84.5 to -0.5
          for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
            Double_t phi = (Double_t)iphi - 0.5;
            EBDetId id(ieta, iphi);
            Double_t val = (*payload).getValueMap()[id.rawId()].p1;
            barrel[0]->Fill(phi, eta, val);
            EBmean[0] = EBmean[0] + val;
            EBrms[0] = EBrms[0] + val * val;
            EBtot[0]++;
            val = (*payload).getValueMap()[id.rawId()].p2;
            barrel[1]->Fill(phi, eta, val);
            EBmean[1] = EBmean[1] + val;
            EBrms[1] = EBrms[1] + val * val;
            EBtot[1]++;
            val = (*payload).getValueMap()[id.rawId()].p3;
            barrel[2]->Fill(phi, eta, val);
            EBmean[2] = EBmean[2] + val;
            EBrms[2] = EBrms[2] + val * val;
            EBtot[2]++;
          }  // iphi
        }    // ieta

        for (int sign = 0; sign < kSides; sign++) {
          int thesign = sign == 1 ? 1 : -1;
          for (int ix = 1; ix <= IX_MAX; ix++) {
            for (int iy = 1; iy <= IY_MAX; iy++) {
              if (!EEDetId::validDetId(ix, iy, thesign))
                continue;
              EEDetId id(ix, iy, thesign);
              Double_t val = (*payload).getValueMap()[id.rawId()].p1;
              EEmean[0] = EEmean[0] + val;
              EErms[0] = EErms[0] + val * val;
              EEtot[0]++;
              if (thesign == 1)
                endc_p[0]->Fill(ix, iy, val);
              else
                endc_m[0]->Fill(ix, iy, val);
              val = (*payload).getValueMap()[id.rawId()].p1;
              EEmean[1] = EEmean[1] + val;
              EErms[1] = EErms[1] + val * val;
              EEtot[1]++;
              if (thesign == 1)
                endc_p[1]->Fill(ix, iy, val);
              else
                endc_m[1]->Fill(ix, iy, val);
              val = (*payload).getValueMap()[id.rawId()].p2;
              EEmean[2] = EEmean[2] + val;
              EErms[2] = EErms[2] + val * val;
              EEtot[2]++;
              if (thesign == 1)
                endc_p[2]->Fill(ix, iy, val);
              else
                endc_m[2]->Fill(ix, iy, val);
            }  // iy
          }    // ix
        }      // side
      }        // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 2800, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      //      t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %lu", run));
      if (IOV < 4294967296)
        t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %i", run));
      else {  // time type IOV
        time_t t = run;
        char buf[256];
        struct tm lt;
        localtime_r(&t, &lt);
        strftime(buf, sizeof(buf), "%F %R:%S", &lt);
        buf[sizeof(buf) - 1] = 0;
        t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %s", buf));
      }

      float xmi[3] = {0.0, 0.26, 0.74};
      float xma[3] = {0.26, 0.74, 1.00};
      TPad*** pad = new TPad**[3];
      //      std::cout << " entries " << EBtot[0] << " mean " << EBmean[0] << " rms " << EBrms[0] << std::endl;
      for (int valId = 0; valId < kValues; valId++) {
        pad[valId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * valId);
          float ymi = yma - 0.28;
          pad[valId][obj] =
              new TPad(Form("p_%i_%i", obj, valId), Form("p_%i_%i", obj, valId), xmi[obj], ymi, xma[obj], yma);
          pad[valId][obj]->Draw();
        }
        double vt = (double)EBtot[valId];
        EBmean[valId] = EBmean[valId] / vt;
        EBrms[valId] = (EBrms[valId] / vt) - (EBmean[valId] * EBmean[valId]);
        EBrms[valId] = sqrt(EBrms[valId]);
        if (EBrms[valId] == 0.)
          EBrms[valId] = 0.001;
        pEBmin[valId] = EBmean[valId] - kRMS * EBrms[valId];
        pEBmax[valId] = EBmean[valId] + kRMS * EBrms[valId];
        //	std::cout << " mean " << EBmean[valId] << " rms " << EBrms[valId] << " entries " << EBtot[valId] << " min " << pEBmin[valId]
        //		  << " max " << pEBmax[valId] << std::endl;
        vt = (double)EEtot[valId];
        EEmean[valId] = EEmean[valId] / vt;
        EErms[valId] = (EErms[valId] / vt) - (EEmean[valId] * EEmean[valId]);
        EErms[valId] = sqrt(EErms[valId]);
        if (EErms[valId] == 0.)
          EErms[valId] = 0.001;
        pEEmin[valId] = EEmean[valId] - kRMS * EErms[valId];
        pEEmax[valId] = EEmean[valId] + kRMS * EErms[valId];
        //	std::cout << " mean " << EEmean[valId] << " rms " << EErms[valId] << " entries " << EEtot[valId] << " min " << pEEmin[valId]
        //		  << " max " << pEEmax[valId] << std::endl;
      }

      for (int valId = 0; valId < 3; valId++) {
        pad[valId][0]->cd();
        DrawEE(endc_m[valId], pEEmin[valId], pEEmax[valId]);
        pad[valId][1]->cd();
        DrawEB(barrel[valId], pEBmin[valId], pEBmax[valId]);
        barrel[valId]->SetStats(false);
        pad[valId][2]->cd();
        DrawEE(endc_p[valId], pEEmin[valId], pEEmax[valId]);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalLinearCorrectionsPlot

  /****************************************************************
     2d plot of ECAL LinearCorrections difference between 2 IOVs
  ****************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalLinearCorrectionsDiffBase : public cond::payloadInspector::PlotImage<EcalLinearCorrections, nIOVs, ntags> {
  public:
    EcalLinearCorrectionsDiffBase()
        : cond::payloadInspector::PlotImage<EcalLinearCorrections, nIOVs, ntags>("ECAL LinearCorrections - map ") {}

    bool fill() override {
      TH2F** barrel = new TH2F*[kValues];
      TH2F** endc_p = new TH2F*[kValues];
      TH2F** endc_m = new TH2F*[kValues];
      double EBmean[kValues], EBrms[kValues], EEmean[kValues], EErms[kValues], pEBmin[kValues], pEBmax[kValues],
          pEEmin[kValues], pEEmax[kValues];
      int EBtot[kValues], EEtot[kValues];
      for (int valId = 0; valId < kValues; valId++) {
        barrel[valId] = new TH2F(
            Form("EBp%i", valId), Form("EBp%i", valId), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
        endc_p[valId] = new TH2F(
            Form("EE+p%i", valId), Form("EE+p%i", valId), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        endc_m[valId] = new TH2F(
            Form("EE-p%i", valId), Form("EE-p%i", valId), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        EBmean[valId] = 0.;
        EBrms[valId] = 0.;
        EEmean[valId] = 0.;
        EErms[valId] = 0.;
        EBtot[valId] = 0;
        EEtot[valId] = 0;
      }
      float vEB[kValues][kEBChannels], vEE[kValues][kEEChannels];

      unsigned int run[2] = {0, 0};
      unsigned long IOV = 0;
      std::string l_tagname[2];
      auto iovs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      l_tagname[0] = cond::payloadInspector::PlotBase::getTag<0>().name;
      auto firstiov = iovs.front();
      IOV = std::get<0>(firstiov);
      if (IOV < 4294967296)
        run[0] = IOV;
      else {  // time type IOV
        run[0] = IOV >> 32;
      }
      std::tuple<cond::Time_t, cond::Hash> lastiov;
      if (ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname[1] = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = iovs.back();
        l_tagname[1] = l_tagname[0];
      }
      IOV = std::get<0>(lastiov);
      if (IOV < 4294967296)
        run[1] = IOV;
      else {  // time type IOV
        run[1] = IOV >> 32;
      }
      for (int irun = 0; irun < nIOVs; irun++) {
        std::shared_ptr<EcalLinearCorrections> payload;
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
            else if (ieta > 0)
              eta = eta - 0.5;  //   0.5 to 84.5
            else
              eta = eta + 0.5;  //  -84.5 to -0.5
            for (int iphi = 1; iphi <= MAX_IPHI; iphi++) {
              Double_t phi = (Double_t)iphi - 0.5;
              EBDetId id(ieta, iphi);
              Double_t val = (*payload).getValueMap()[id.rawId()].p1;
              int channel = id.hashedIndex();
              if (irun == 0)
                vEB[0][channel] = val;
              else {
                double diff = val - vEB[0][channel];
                barrel[0]->Fill(phi, eta, diff);
                EBmean[0] = EBmean[0] + diff;
                EBrms[0] = EBrms[0] + diff * diff;
                EBtot[0]++;
              }
              val = (*payload).getValueMap()[id.rawId()].p2;
              if (irun == 0)
                vEB[1][channel] = val;
              else {
                double diff = val - vEB[1][channel];
                barrel[1]->Fill(phi, eta, diff);
                EBmean[1] = EBmean[1] + diff;
                EBrms[1] = EBrms[1] + diff * diff;
                EBtot[1]++;
              }
              val = (*payload).getValueMap()[id.rawId()].p3;
              if (irun == 0)
                vEB[2][channel] = val;
              else {
                double diff = val - vEB[2][channel];
                barrel[2]->Fill(phi, eta, diff);
                EBmean[2] = EBmean[2] + diff;
                EBrms[2] = EBrms[2] + diff * diff;
                EBtot[2]++;
              }
            }  // iphi
          }    // ieta

          for (int sign = 0; sign < kSides; sign++) {
            int thesign = sign == 1 ? 1 : -1;
            for (int ix = 1; ix <= IX_MAX; ix++) {
              for (int iy = 1; iy <= IY_MAX; iy++) {
                if (!EEDetId::validDetId(ix, iy, thesign))
                  continue;
                EEDetId id(ix, iy, thesign);
                Double_t val = (*payload).getValueMap()[id.rawId()].p1;
                int channel = id.hashedIndex();
                if (irun == 0)
                  vEE[0][channel] = val;
                else {
                  double diff = val - vEE[0][channel];
                  EEmean[0] = EEmean[0] + diff;
                  EErms[0] = EErms[0] + diff * diff;
                  EEtot[0]++;
                  if (thesign == 1)
                    endc_p[0]->Fill(ix, iy, diff);
                  else
                    endc_m[0]->Fill(ix, iy, diff);
                }
                val = (*payload).getValueMap()[id.rawId()].p1;
                if (irun == 0)
                  vEE[1][channel] = val;
                else {
                  double diff = val - vEE[1][channel];
                  EEmean[1] = EEmean[1] + diff;
                  EErms[1] = EErms[1] + diff * diff;
                  EEtot[1]++;
                  if (thesign == 1)
                    endc_p[1]->Fill(ix, iy, diff);
                  else
                    endc_m[1]->Fill(ix, iy, diff);
                }
                val = (*payload).getValueMap()[id.rawId()].p2;
                if (irun == 0)
                  vEE[2][channel] = val;
                else {
                  double diff = val - vEE[2][channel];
                  EEmean[2] = EEmean[2] + diff;
                  EErms[2] = EErms[2] + diff * diff;
                  EEtot[2]++;
                  if (thesign == 1)
                    endc_p[2]->Fill(ix, iy, diff);
                  else
                    endc_m[2]->Fill(ix, iy, diff);
                }
              }  // iy
            }    // ix
          }      // side
        }        // if payload.get()
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 2800, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      //      t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %lu", run));
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (IOV < 4294967296) {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %i - %i", run[1], run[0]));
      } else {  // time type IOV
        time_t t = run[0];
        char buf0[256], buf1[256];
        struct tm lt;
        localtime_r(&t, &lt);
        strftime(buf0, sizeof(buf0), "%F %R:%S", &lt);
        buf0[sizeof(buf0) - 1] = 0;
        t = run[1];
        localtime_r(&t, &lt);
        strftime(buf1, sizeof(buf1), "%F %R:%S", &lt);
        buf1[sizeof(buf1) - 1] = 0;
        if (ntags == 2) {
          if (len < 80) {
            t1.SetTextSize(0.02);
            t1.DrawLatex(0.5, 0.96, Form("%s %s - %s %s", l_tagname[1].c_str(), buf1, l_tagname[0].c_str(), buf0));
          } else {
            t1.SetTextSize(0.03);
            t1.DrawLatex(0.5, 0.96, Form("Ecal Linear Corrections, IOV %s - %s", buf1, buf0));
          }
        } else {
          t1.SetTextSize(0.015);
          t1.DrawLatex(0.5, 0.96, Form("%s, IOV %s - %s", l_tagname[0].c_str(), buf1, buf0));
        }
      }

      float xmi[3] = {0.0, 0.26, 0.74};
      float xma[3] = {0.26, 0.74, 1.00};
      TPad*** pad = new TPad**[3];
      //      std::cout << " entries " << EBtot[0] << " mean " << EBmean[0] << " rms " << EBrms[0] << std::endl;
      for (int valId = 0; valId < kValues; valId++) {
        pad[valId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * valId);
          float ymi = yma - 0.28;
          pad[valId][obj] =
              new TPad(Form("p_%i_%i", obj, valId), Form("p_%i_%i", obj, valId), xmi[obj], ymi, xma[obj], yma);
          pad[valId][obj]->Draw();
        }
        double vt = (double)EBtot[valId];
        EBmean[valId] = EBmean[valId] / vt;
        EBrms[valId] = (EBrms[valId] / vt) - (EBmean[valId] * EBmean[valId]);
        EBrms[valId] = sqrt(EBrms[valId]);
        if (EBrms[valId] == 0.)
          EBrms[valId] = 0.001;
        pEBmin[valId] = EBmean[valId] - kRMS * EBrms[valId];
        pEBmax[valId] = EBmean[valId] + kRMS * EBrms[valId];
        //	std::cout << " mean " << EBmean[valId] << " rms " << EBrms[valId] << " entries " << EBtot[valId] << " min " << pEBmin[valId]
        //		  << " max " << pEBmax[valId] << std::endl;
        vt = (double)EEtot[valId];
        EEmean[valId] = EEmean[valId] / vt;
        EErms[valId] = (EErms[valId] / vt) - (EEmean[valId] * EEmean[valId]);
        EErms[valId] = sqrt(EErms[valId]);
        if (EErms[valId] == 0.)
          EErms[valId] = 0.001;
        pEEmin[valId] = EEmean[valId] - kRMS * EErms[valId];
        pEEmax[valId] = EEmean[valId] + kRMS * EErms[valId];
        //	std::cout << " mean " << EEmean[valId] << " rms " << EErms[valId] << " entries " << EEtot[valId] << " min " << pEEmin[valId]
        //		  << " max " << pEEmax[valId] << std::endl;
      }

      for (int valId = 0; valId < 3; valId++) {
        pad[valId][0]->cd();
        DrawEE(endc_m[valId], pEEmin[valId], pEEmax[valId]);
        pad[valId][1]->cd();
        DrawEB(barrel[valId], pEBmin[valId], pEBmax[valId]);
        barrel[valId]->SetStats(false);
        pad[valId][2]->cd();
        DrawEE(endc_p[valId], pEEmin[valId], pEEmax[valId]);
      }

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalLinearCorrectionsDiffBase
  using EcalLinearCorrectionsDiffOneTag = EcalLinearCorrectionsDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalLinearCorrectionsDiffTwoTags = EcalLinearCorrectionsDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalLinearCorrections) {
  PAYLOAD_INSPECTOR_CLASS(EcalLinearCorrectionsPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalLinearCorrectionsDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLinearCorrectionsDiffTwoTags);
}
