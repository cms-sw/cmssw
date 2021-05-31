#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"

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
  enum { kEBChannels = 61200, kEEChannels = 14648, kGains = 3, kSides = 2 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y
  int gainValues[kGains] = {12, 6, 1};

  /**************************************************
     2d plot of ECAL TPGPedestals of 1 IOV
  **************************************************/
  class EcalTPGPedestalsPlot : public cond::payloadInspector::PlotImage<EcalTPGPedestals> {
  public:
    EcalTPGPedestalsPlot() : cond::payloadInspector::PlotImage<EcalTPGPedestals>("ECAL Gain Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      float mEBmin[kGains], mEEmin[kGains], mEBmax[kGains], mEEmax[kGains];
      for (int gainId = 0; gainId < kGains; gainId++) {
        barrel_m[gainId] = new TH2F(Form("EBm%i", gainId),
                                    Form("EB mean_x%i ", gainValues[gainId]),
                                    MAX_IPHI,
                                    0,
                                    MAX_IPHI,
                                    2 * MAX_IETA,
                                    -MAX_IETA,
                                    MAX_IETA);
        endc_p_m[gainId] = new TH2F(Form("EE+m%i", gainId),
                                    Form("EE+ mean_x%i", gainValues[gainId]),
                                    IX_MAX,
                                    IX_MIN,
                                    IX_MAX + 1,
                                    IY_MAX,
                                    IY_MIN,
                                    IY_MAX + 1);
        endc_m_m[gainId] = new TH2F(Form("EE-m%i", gainId),
                                    Form("EE- mean_x%i", gainValues[gainId]),
                                    IX_MAX,
                                    IX_MIN,
                                    IX_MAX + 1,
                                    IY_MAX,
                                    IY_MIN,
                                    IY_MAX + 1);
        mEBmin[gainId] = 10.;
        mEEmin[gainId] = 10.;
        mEBmax[gainId] = -10.;
        mEEmax[gainId] = -10.;
      }

      //      std::ofstream fout;
      //      fout.open("./bid.txt");
      auto iov = iovs.front();
      std::shared_ptr<EcalTPGPedestals> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        for (int sign = 0; sign < kSides; sign++) {
          int thesign = sign == 1 ? 1 : -1;

          for (int ieta = 0; ieta < MAX_IETA; ieta++) {
            for (int iphi = 0; iphi < MAX_IPHI; iphi++) {
              EBDetId id((ieta + 1) * thesign, iphi + 1);
              float y = -1 - ieta;
              if (sign == 1)
                y = ieta;
              float val = (*payload)[id.rawId()].mean_x12;
              barrel_m[0]->Fill(iphi, y, val);
              if (val < mEBmin[0])
                mEBmin[0] = val;
              if (val > mEBmax[0])
                mEBmax[0] = val;
              val = (*payload)[id.rawId()].mean_x6;
              barrel_m[1]->Fill(iphi, y, val);
              if (val < mEBmin[1])
                mEBmin[1] = val;
              if (val > mEBmax[1])
                mEBmax[1] = val;
              val = (*payload)[id.rawId()].mean_x1;
              barrel_m[2]->Fill(iphi, y, val);
              if (val < mEBmin[2])
                mEBmin[2] = val;
              if (val > mEBmax[2])
                mEBmax[2] = val;
            }  // iphi
          }    // ieta

          for (int ix = 0; ix < IX_MAX; ix++) {
            for (int iy = 0; iy < IY_MAX; iy++) {
              if (!EEDetId::validDetId(ix + 1, iy + 1, thesign))
                continue;
              EEDetId id(ix + 1, iy + 1, thesign);
              float val = (*payload)[id.rawId()].mean_x12;
              if (thesign == 1)
                endc_p_m[0]->Fill(ix + 1, iy + 1, val);
              else
                endc_m_m[0]->Fill(ix + 1, iy + 1, val);
              if (val < mEEmin[0])
                mEEmin[0] = val;
              if (val > mEEmax[0])
                mEEmax[0] = val;
              val = (*payload)[id.rawId()].mean_x6;
              if (thesign == 1)
                endc_p_m[1]->Fill(ix + 1, iy + 1, val);
              else
                endc_m_m[1]->Fill(ix + 1, iy + 1, val);
              if (val < mEEmin[1])
                mEEmin[1] = val;
              if (val > mEEmax[1])
                mEEmax[1] = val;
              val = (*payload)[id.rawId()].mean_x1;
              if (thesign == 1)
                endc_p_m[2]->Fill(ix + 1, iy + 1, val);
              else
                endc_m_m[2]->Fill(ix + 1, iy + 1, val);
              if (val < mEEmin[2])
                mEEmin[2] = val;
              if (val > mEEmax[2])
                mEEmax[2] = val;
              //	      fout << " x " << ix << " y " << " val " << val << std::endl;
            }  // iy
          }    // ix
        }      // side
      }        // if payload.get()
      else
        return false;
      //      std::cout << " min " << rEEmin[2] << " max " << rEEmax[2] << std::endl;
      //      fout.close();

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1200, 900);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Gain TPGPedestals, IOV %i", run));

      float xmi[3] = {0.0, 0.22, 0.78};
      float xma[3] = {0.22, 0.78, 1.00};
      TPad*** pad = new TPad**[kGains];
      for (int gId = 0; gId < kGains; gId++) {
        pad[gId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * gId);
          float ymi = yma - 0.30;
          pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId), Form("p_%i_%i", obj, gId), xmi[obj], ymi, xma[obj], yma);
          pad[gId][obj]->Draw();
        }
      }

      for (int gId = 0; gId < kGains; gId++) {
        pad[gId][0]->cd();
        DrawEE(endc_m_m[gId], mEEmin[gId], mEEmax[gId]);
        pad[gId][1]->cd();
        DrawEB(barrel_m[gId], mEBmin[gId], mEBmax[gId]);
        pad[gId][2]->cd();
        DrawEE(endc_p_m[gId], mEEmin[gId], mEEmax[gId]);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /******************************************************************
     2d plot of ECAL TPGPedestals difference between 2 IOVs
  ******************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags, int method>
  class EcalTPGPedestalsBase : public cond::payloadInspector::PlotImage<EcalTPGPedestals, nIOVs, ntags> {
  public:
    EcalTPGPedestalsBase()
        : cond::payloadInspector::PlotImage<EcalTPGPedestals, nIOVs, ntags>("ECAL Gain Ratios comparison") {}
    bool fill() override {
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      float mEBmin[kGains], mEEmin[kGains], mEBmax[kGains], mEEmax[kGains];
      float mEB[kGains][kEBChannels], mEE[kGains][kEEChannels];
      for (int gainId = 0; gainId < kGains; gainId++) {
        barrel_m[gainId] = new TH2F(Form("EBm%i", gainId),
                                    Form("EB mean_x%i ", gainValues[gainId]),
                                    MAX_IPHI,
                                    0,
                                    MAX_IPHI,
                                    2 * MAX_IETA,
                                    -MAX_IETA,
                                    MAX_IETA);
        endc_p_m[gainId] = new TH2F(Form("EE+m%i", gainId),
                                    Form("EE+ mean_x%i", gainValues[gainId]),
                                    IX_MAX,
                                    IX_MIN,
                                    IX_MAX + 1,
                                    IY_MAX,
                                    IY_MIN,
                                    IY_MAX + 1);
        endc_m_m[gainId] = new TH2F(Form("EE-m%i", gainId),
                                    Form("EE- mean_x%i", gainValues[gainId]),
                                    IX_MAX,
                                    IX_MIN,
                                    IX_MAX + 1,
                                    IY_MAX,
                                    IY_MIN,
                                    IY_MAX + 1);
        mEBmin[gainId] = 10.;
        mEEmin[gainId] = 10.;
        mEBmax[gainId] = -10.;
        mEEmax[gainId] = -10.;
      }

      unsigned int run[2];
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
        std::shared_ptr<EcalTPGPedestals> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          float dr;
          for (int sign = 0; sign < kSides; sign++) {
            int thesign = sign == 1 ? 1 : -1;

            for (int ieta = 0; ieta < MAX_IETA; ieta++) {
              for (int iphi = 0; iphi < MAX_IPHI; iphi++) {
                EBDetId id((ieta + 1) * thesign, iphi + 1);
                int hashindex = id.hashedIndex();
                float y = -1 - ieta;
                if (sign == 1)
                  y = ieta;
                float val = (*payload)[id.rawId()].mean_x12;
                if (irun == 0) {
                  mEB[0][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEB[0][hashindex];
                  else {
                    if (mEB[0][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEB[0][hashindex];
                  }
                  barrel_m[0]->Fill(iphi, y, dr);
                  if (dr < mEBmin[0])
                    mEBmin[0] = dr;
                  if (dr > mEBmax[0])
                    mEBmax[0] = dr;
                }
                val = (*payload)[id.rawId()].mean_x6;
                if (irun == 0) {
                  mEB[1][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEB[1][hashindex];
                  else {
                    if (mEB[1][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEB[1][hashindex];
                  }
                  barrel_m[1]->Fill(iphi, y, dr);
                  if (dr < mEBmin[1])
                    mEBmin[1] = dr;
                  if (dr > mEBmax[1])
                    mEBmax[1] = dr;
                }
                val = (*payload)[id.rawId()].mean_x1;
                if (irun == 0) {
                  mEB[2][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEB[2][hashindex];
                  else {
                    if (mEB[2][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEB[2][hashindex];
                  }
                  barrel_m[2]->Fill(iphi, y, dr);
                  if (dr < mEBmin[2])
                    mEBmin[2] = dr;
                  if (dr > mEBmax[2])
                    mEBmax[2] = dr;
                }
              }  // iphi
            }    // ieta

            for (int ix = 0; ix < IX_MAX; ix++) {
              for (int iy = 0; iy < IY_MAX; iy++) {
                if (!EEDetId::validDetId(ix + 1, iy + 1, thesign))
                  continue;
                EEDetId id(ix + 1, iy + 1, thesign);
                int hashindex = id.hashedIndex();
                float val = (*payload)[id.rawId()].mean_x12;
                if (irun == 0) {
                  mEE[0][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEE[0][hashindex];
                  else {
                    if (mEE[0][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEE[0][hashindex];
                  }
                  if (thesign == 1)
                    endc_p_m[0]->Fill(ix + 1, iy + 1, dr);
                  else
                    endc_m_m[0]->Fill(ix + 1, iy + 1, dr);
                  if (dr < mEEmin[0])
                    mEEmin[0] = dr;
                  if (dr > mEEmax[0])
                    mEEmax[0] = dr;
                }
                val = (*payload)[id.rawId()].mean_x6;
                if (irun == 0) {
                  mEE[1][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEE[1][hashindex];
                  else {
                    if (mEE[1][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEE[1][hashindex];
                  }
                  if (thesign == 1)
                    endc_p_m[1]->Fill(ix + 1, iy + 1, dr);
                  else
                    endc_m_m[1]->Fill(ix + 1, iy + 1, dr);
                  if (dr < mEEmin[1])
                    mEEmin[1] = dr;
                  if (dr > mEEmax[1])
                    mEEmax[1] = dr;
                }
                val = (*payload)[id.rawId()].mean_x1;
                if (irun == 0) {
                  mEE[2][hashindex] = val;
                } else {
                  if (method == 0)
                    dr = val - mEE[2][hashindex];
                  else {
                    if (mEE[2][hashindex] == 0.) {
                      if (val == 0.)
                        dr = 1.;
                      else
                        dr = 9999.;
                    } else
                      dr = val / mEE[2][hashindex];
                  }
                  if (thesign == 1)
                    endc_p_m[2]->Fill(ix + 1, iy + 1, dr);
                  else
                    endc_m_m[2]->Fill(ix + 1, iy + 1, dr);
                  if (dr < mEEmin[2])
                    mEEmin[2] = dr;
                  if (dr > mEEmax[2])
                    mEEmax[2] = dr;
                }
                //	      fout << " x " << ix << " y " << " dr " << dr << std::endl;
              }  // iy
            }    // ix
          }      // side
        }        //  if payload.get()
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1200, 900);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      std::string dr[2] = {"-", "/"};
      if (ntags == 2) {
        if (len < 150) {
          t1.SetTextSize(0.03);
          t1.DrawLatex(
              0.5,
              0.96,
              Form("%s %i %s %s %i", l_tagname[1].c_str(), run[1], dr[method].c_str(), l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("Ecal TPGPedestals, IOV %i %s %i", run[1], dr[method].c_str(), run[0]));
        }
      } else {
        t1.SetTextSize(0.03);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i %s %i", l_tagname[0].c_str(), run[1], dr[method].c_str(), run[0]));
      }

      float xmi[3] = {0.0, 0.22, 0.78};
      float xma[3] = {0.22, 0.78, 1.00};
      TPad*** pad = new TPad**[kGains];
      for (int gId = 0; gId < kGains; gId++) {
        pad[gId] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * gId);
          float ymi = yma - 0.30;
          pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId), Form("p_%i_%i", obj, gId), xmi[obj], ymi, xma[obj], yma);
          pad[gId][obj]->Draw();
        }
      }

      for (int gId = 0; gId < kGains; gId++) {
        pad[gId][0]->cd();
        DrawEE(endc_m_m[gId], mEEmin[gId], mEEmax[gId]);
        pad[gId][1]->cd();
        DrawEB(barrel_m[gId], mEBmin[gId], mEBmax[gId]);
        pad[gId][2]->cd();
        DrawEE(endc_p_m[gId], mEEmin[gId], mEEmax[gId]);
      }

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalTPGPedestalsDiffBase
  using EcalTPGPedestalsDiffOneTag = EcalTPGPedestalsBase<cond::payloadInspector::SINGLE_IOV, 1, 0>;
  using EcalTPGPedestalsDiffTwoTags = EcalTPGPedestalsBase<cond::payloadInspector::SINGLE_IOV, 2, 0>;
  using EcalTPGPedestalsRatioOneTag = EcalTPGPedestalsBase<cond::payloadInspector::SINGLE_IOV, 1, 1>;
  using EcalTPGPedestalsRatioTwoTags = EcalTPGPedestalsBase<cond::payloadInspector::SINGLE_IOV, 2, 1>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGPedestals) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGPedestalsPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGPedestalsDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGPedestalsDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGPedestalsRatioOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGPedestalsRatioTwoTags);
}
