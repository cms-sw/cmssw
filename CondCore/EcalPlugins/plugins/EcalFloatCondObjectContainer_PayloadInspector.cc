#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648, kSides = 2, kRMS = 5 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /*****************************************************
     2d plot of ECAL FloatCondObjectContainer of 1 IOV
  *****************************************************/
  class EcalFloatCondObjectContainerPlot : public cond::payloadInspector::PlotImage<EcalFloatCondObjectContainer> {
  public:
    EcalFloatCondObjectContainerPlot()
        : cond::payloadInspector::PlotImage<EcalFloatCondObjectContainer>("ECAL FloatCondObjectContainer - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* barrel = new TH2F("EB", "EB", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "EE+", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE-", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      double EBmean = 0., EBrms = 0., EEmean = 0., EErms = 0.;
      int EBtot = 0, EEtot = 0;

      auto iov = iovs.front();
      std::shared_ptr<EcalFloatCondObjectContainer> payload = fetchPayload(std::get<1>(iov));
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
            double val = (*payload)[id.rawId()];
            barrel->Fill(phi, eta, val);
            EBmean = EBmean + val;
            EBrms = EBrms + val * val;
            EBtot++;
          }
        }

        for (int sign = 0; sign < kSides; sign++) {
          int thesign = sign == 1 ? 1 : -1;
          for (int ix = 1; ix <= IX_MAX; ix++) {
            for (int iy = 1; iy <= IY_MAX; iy++) {
              if (!EEDetId::validDetId(ix, iy, thesign))
                continue;
              EEDetId id(ix, iy, thesign);
              double val = (*payload)[id.rawId()];
              EEmean = EEmean + val;
              EErms = EErms + val * val;
              EEtot++;
              if (thesign == 1)
                endc_p->Fill(ix, iy, val);
              else
                endc_m->Fill(ix, iy, val);
            }  // iy
          }    // ix
        }      // side
      }        // payload
      double vt = (double)EBtot;
      EBmean = EBmean / vt;
      EBrms = EBrms / vt - (EBmean * EBmean);
      EBrms = sqrt(EBrms);
      if (EBrms == 0.)
        EBrms = 0.001;
      double pEBmin = EBmean - kRMS * EBrms;
      double pEBmax = EBmean + kRMS * EBrms;
      //      std::cout << " mean " << EBmean << " rms " << EBrms << " entries " << EBtot << " min " << pEBmin << " max " << pEBmax << std::endl;
      vt = (double)EEtot;
      EEmean = EEmean / vt;
      EErms = EErms / vt - (EEmean * EEmean);
      EErms = sqrt(EErms);
      if (EErms == 0.)
        EErms = 0.001;
      double pEEmin = EEmean - kRMS * EErms;
      double pEEmax = EEmean + kRMS * EErms;
      //      std::cout << " mean " << EEmean << " rms " << EErms << " entries " << EEtot << " min " << pEEmin << " max " << pEEmax << std::endl;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1600, 450);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal FloatCondObjectContainer, IOV %i", run));

      float xmi[3] = {0.0, 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEE(endc_m, pEEmin, pEEmax);
      pad[1]->cd();
      DrawEB(barrel, pEBmin, pEBmax);
      pad[2]->cd();
      DrawEE(endc_p, pEEmin, pEEmax);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /************************************************************************
     2d plot of ECAL FloatCondObjectContainer difference between 2 IOVs
  ************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalFloatCondObjectContainerDiffBase
      : public cond::payloadInspector::PlotImage<EcalFloatCondObjectContainer, nIOVs, ntags> {
  public:
    EcalFloatCondObjectContainerDiffBase()
        : cond::payloadInspector::PlotImage<EcalFloatCondObjectContainer, nIOVs, ntags>(
              "ECAL FloatCondObjectContainer difference") {}

    bool fill() override {
      TH2F* barrel = new TH2F("EB", "EB difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "EE+ difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE- difference", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      double EBmean = 0., EBrms = 0., EEmean = 0., EErms = 0.;
      int EBtot = 0, EEtot = 0;

      unsigned int run[2];
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
        std::shared_ptr<EcalFloatCondObjectContainer> payload;
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
              double val = (*payload)[id.rawId()];
              if (irun == 0)
                vEB[channel] = val;
              else {
                double diff = val - vEB[channel];
                barrel->Fill(phi, eta, diff);
                EBmean = EBmean + diff;
                EBrms = EBrms + diff * diff;
                EBtot++;
                //		  std::cout << " entry " << EBtot << " mean " << EBmean << " rms " << EBrms << std::endl;
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
                double val = (*payload)[id.rawId()];
                if (irun == 0)
                  vEE[channel] = val;
                else {
                  double diff = val - vEE[channel];
                  EEmean = EEmean + diff;
                  EErms = EErms + diff * diff;
                  EEtot++;
                  if (thesign == 1)
                    endc_p->Fill(ix, iy, diff);
                  else
                    endc_m->Fill(ix, iy, diff);
                }
              }  // iy
            }    // ix
          }      // side
        }        // payload
        else
          return false;
      }  // loop over IOVs

      double vt = (double)EBtot;
      EBmean = EBmean / vt;
      EBrms = EBrms / vt - (EBmean * EBmean);
      EBrms = sqrt(EBrms);
      if (EBrms == 0.)
        EBrms = 0.001;
      double pEBmin = EBmean - kRMS * EBrms;
      double pEBmax = EBmean + kRMS * EBrms;
      //      std::cout << " mean " << EBmean << " rms " << EBrms << " entries " << EBtot << " min " << pEBmin << " max " << pEBmax << std::endl;
      vt = (double)EEtot;
      EEmean = EEmean / vt;
      EErms = EErms / vt - (EEmean * EEmean);
      EErms = sqrt(EErms);
      if (EErms == 0.)
        EErms = 0.001;
      double pEEmin = EEmean - kRMS * EErms;
      double pEEmax = EEmean + kRMS * EErms;
      //      std::cout << " mean " << EEmean << " rms " << EErms << " entries " << EEtot << " min " << pEEmin << " max " << pEEmax << std::endl;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1600, 450);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2 && len < 150) {
        t1.SetTextSize(0.04);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }
      float xmi[3] = {0.0, 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEE(endc_m, pEEmin, pEEmax);
      pad[1]->cd();
      DrawEB(barrel, pEBmin, pEBmax);
      pad[2]->cd();
      DrawEE(endc_p, pEEmin, pEEmax);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalFloatCondObjectContainerDiffBase
  using EcalFloatCondObjectContainerDiffOneTag =
      EcalFloatCondObjectContainerDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalFloatCondObjectContainerDiffTwoTags =
      EcalFloatCondObjectContainerDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalFloatCondObjectContainer) {
  PAYLOAD_INSPECTOR_CLASS(EcalFloatCondObjectContainerPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalFloatCondObjectContainerDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalFloatCondObjectContainerDiffTwoTags);
}
