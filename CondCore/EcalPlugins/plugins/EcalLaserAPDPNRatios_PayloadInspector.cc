#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/RunInfoProxy.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TText.h"
#include "TGraph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648 };
  enum {
    MIN_IETA = 1,
    MIN_IPHI = 1,
    MAX_IETA = 85,
    MAX_IPHI = 360,
    EBhistEtaMax = 171
  };  // barrel lower and upper bounds on eta and phi
  enum {
    IX_MIN = 1,
    IY_MIN = 1,
    IX_MAX = 100,
    IY_MAX = 100,
    EEhistXMax = 220
  };  // endcaps lower and upper bounds on x and y

  /*******************************************************
   
     2d histogram of ECAL barrel APDPNRatios of 1 IOV 

  *******************************************************/

  // inherit from one of the predefined plot class: Histogram2D
  class EcalLaserAPDPNRatiosEBMap : public cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios> {
  public:
    EcalLaserAPDPNRatiosEBMap()
        : cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios>("ECAL Barrel APDPNRatios - map ",
                                                                    "iphi",
                                                                    MAX_IPHI,
                                                                    MIN_IPHI,
                                                                    MAX_IPHI + 1,
                                                                    "ieta",
                                                                    EBhistEtaMax,
                                                                    -MAX_IETA,
                                                                    MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalLaserAPDPNRatios> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // set to 1 for ieta 0 (no crystal)
          for (int iphi = 1; iphi < 361; iphi++)
            fillWithValue(iphi, 0, 1);

          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            float p2 = (payload->getLaserMap())[rawid].p2;
            // fill the Histogram2D here
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), p2);
          }  // loop over cellid
        }  // if payload.get()
      }  // loop over IOV's (1 in this case)

      return true;
    }  // fill method
  };

  class EcalLaserAPDPNRatiosEEMap : public cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios> {
  private:
    int EEhistSplit = 20;

  public:
    EcalLaserAPDPNRatiosEEMap()
        : cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios>("ECAL Endcap APDPNRatios - map ",
                                                                    "ix",
                                                                    EEhistXMax,
                                                                    IX_MIN,
                                                                    EEhistXMax + 1,
                                                                    "iy",
                                                                    IY_MAX,
                                                                    IY_MIN,
                                                                    IY_MAX + 1) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalLaserAPDPNRatios> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // set to 0 everywhwere
          for (int ix = IX_MIN; ix < EEhistXMax + 1; ix++)
            for (int iy = IY_MIN; iy < IY_MAX + 1; iy++)
              fillWithValue(ix, iy, 0);

          for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
            if (!EEDetId::validHashIndex(cellid))
              continue;
            uint32_t rawid = EEDetId::unhashIndex(cellid);
            float p2 = (payload->getLaserMap())[rawid].p2;
            EEDetId myEEId(rawid);
            if (myEEId.zside() == -1)
              fillWithValue(myEEId.ix(), myEEId.iy(), p2);
            else
              fillWithValue(myEEId.ix() + IX_MAX + EEhistSplit, myEEId.iy(), p2);
          }  // loop over cellid
        }  // payload
      }  // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  /*************************************************
     2d plot of ECAL IntercalibConstants of 1 IOV
  *************************************************/
  class EcalLaserAPDPNRatiosPlot : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios> {
  public:
    EcalLaserAPDPNRatiosPlot()
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios>("ECAL Laser APDPNRatios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      TH2F** barrel = new TH2F*[3];
      TH2F** endc_p = new TH2F*[3];
      TH2F** endc_m = new TH2F*[3];
      float pEBmin[3], pEEmin[3];

      for (int i = 0; i < 3; i++) {
        barrel[i] =
            new TH2F(Form("EBp%i", i), Form("EB p%i", i + 1), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
        endc_p[i] =
            new TH2F(Form("EE+p%i", i), Form("EE+ p%i", i + 1), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        endc_m[i] =
            new TH2F(Form("EE-p%i", i), Form("EE- p%i", i + 1), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        pEBmin[i] = 10.;
        pEEmin[i] = 10.;
      }

      auto iov = iovs.front();
      std::shared_ptr<EcalLaserAPDPNRatios> payload = fetchPayload(std::get<1>(iov));
      unsigned long IOV = std::get<0>(iov);
      int run = 0;
      if (IOV < 4294967296)
        run = std::get<0>(iov);
      else  // time type IOV
        run = IOV >> 32;
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
          float p1 = (payload->getLaserMap())[rawid].p1;
          if (p1 < pEBmin[0])
            pEBmin[0] = p1;
          float p2 = (payload->getLaserMap())[rawid].p2;
          if (p2 < pEBmin[1])
            pEBmin[1] = p2;
          float p3 = (payload->getLaserMap())[rawid].p3;
          if (p3 < pEBmin[2])
            pEBmin[2] = p3;
          barrel[0]->Fill(phi, eta, p1);
          barrel[1]->Fill(phi, eta, p2);
          barrel[2]->Fill(phi, eta, p3);
        }  // loop over cellid

        // looping over the EE channels
        for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
          if (!EEDetId::validHashIndex(cellid))
            continue;
          uint32_t rawid = EEDetId::unhashIndex(cellid);
          EEDetId myEEId(rawid);
          float p1 = (payload->getLaserMap())[rawid].p1;
          if (p1 < pEEmin[0])
            pEEmin[0] = p1;
          float p2 = (payload->getLaserMap())[rawid].p2;
          if (p2 < pEEmin[1])
            pEEmin[1] = p2;
          float p3 = (payload->getLaserMap())[rawid].p3;
          if (p3 < pEEmin[2])
            pEEmin[2] = p3;
          if (myEEId.zside() == 1) {
            endc_p[0]->Fill(myEEId.ix(), myEEId.iy(), p1);
            endc_p[1]->Fill(myEEId.ix(), myEEId.iy(), p2);
            endc_p[2]->Fill(myEEId.ix(), myEEId.iy(), p3);
          } else {
            endc_m[0]->Fill(myEEId.ix(), myEEId.iy(), p1);
            endc_m[1]->Fill(myEEId.ix(), myEEId.iy(), p2);
            endc_m[2]->Fill(myEEId.ix(), myEEId.iy(), p3);
          }
        }  // validDetId
      }  // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 2800, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      if (IOV < 4294967296)
        t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %i", run));
      else {  // time type IOV
        time_t t = run;
        char buf[256];
        struct tm lt;
        localtime_r(&t, &lt);
        strftime(buf, sizeof(buf), "%F %R:%S", &lt);
        buf[sizeof(buf) - 1] = 0;
        t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %s", buf));
      }

      float xmi[3] = {0.0, 0.26, 0.74};
      float xma[3] = {0.26, 0.74, 1.00};
      TPad*** pad = new TPad**[3];
      for (int i = 0; i < 3; i++) {
        pad[i] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * i);
          float ymi = yma - 0.28;
          pad[i][obj] = new TPad(Form("p_%i_%i", obj, i), Form("p_%i_%i", obj, i), xmi[obj], ymi, xma[obj], yma);
          pad[i][obj]->Draw();
        }
      }

      for (int i = 0; i < 3; i++) {
        // compute histo limits with some rounding
        //	std::cout << " before " << pEBmin[i];
        float xmin = pEBmin[i] * 10.;
        int min = (int)xmin;
        pEBmin[i] = (float)min / 10.;
        //	std::cout << " after " << pEBmin[i] << std::endl << " before " << pEEmin[i];
        xmin = pEEmin[i] * 10.;
        min = (int)xmin;
        pEEmin[i] = (float)min / 10.;
        //	std::cout << " after " << pEEmin[i] << std::endl;
        pad[i][0]->cd();
        DrawEE(endc_m[i], pEEmin[i], 1.1);
        pad[i][1]->cd();
        DrawEB(barrel[i], pEBmin[i], 1.1);
        pad[i][2]->cd();
        DrawEE(endc_p[i], pEEmin[i], 1.1);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /*****************************************************************
     2d plot of ECAL IntercalibConstants difference between 2 IOVs
  ******************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags, int method>
  class EcalLaserAPDPNRatiosBase : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags> {
  public:
    EcalLaserAPDPNRatiosBase()
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags>("ECAL Laser APDPNRatios difference") {}

    bool fill() override {
      TH2F** barrel = new TH2F*[3];
      TH2F** endc_p = new TH2F*[3];
      TH2F** endc_m = new TH2F*[3];
      float pEBmin[3], pEEmin[3], pEBmax[3], pEEmax[3];
      for (int i = 0; i < 3; i++) {
        barrel[i] =
            new TH2F(Form("EBp%i", i), Form("EB p%i", i + 1), MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
        endc_p[i] =
            new TH2F(Form("EE+p%i", i), Form("EE+ p%i", i + 1), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        endc_m[i] =
            new TH2F(Form("EE-p%i", i), Form("EE- p%i", i + 1), IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
        pEBmin[i] = 10.;
        pEEmin[i] = 10.;
        pEBmax[i] = -10.;
        pEEmax[i] = -10.;
      }
      unsigned int run[2] = {0, 0};
      std::string l_tagname[2];
      unsigned long IOV = 0;
      float pEB[3][kEBChannels], pEE[3][kEEChannels];
      auto iovs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      l_tagname[0] = cond::payloadInspector::PlotBase::getTag<0>().name;
      auto firstiov = iovs.front();
      IOV = std::get<0>(firstiov);
      if (IOV < 4294967296)
        run[0] = std::get<0>(firstiov);
      else  // time type IOV
        run[0] = IOV >> 32;
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
        run[1] = std::get<0>(lastiov);
      else  // time type IOV
        run[1] = IOV >> 32;

      for (int irun = 0; irun < nIOVs; irun++) {
        std::shared_ptr<EcalLaserAPDPNRatios> payload;
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
              pEB[0][cellid] = (payload->getLaserMap())[rawid].p1;
              pEB[1][cellid] = (payload->getLaserMap())[rawid].p2;
              pEB[2][cellid] = (payload->getLaserMap())[rawid].p3;
            } else {
              Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
              Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
              if (eta > 0.)
                eta = eta - 0.5;  //   0.5 to 84.5
              else
                eta = eta + 0.5;  //  -84.5 to -0.5
              double dr;
              if (method == 0)  // difference
                dr = (payload->getLaserMap())[rawid].p1 - pEB[0][cellid];
              else {  // ratio
                if (pEB[0][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p1 == 0.)
                    dr = 1.;
                  else
                    dr = 9999.;  //use a large value
                } else
                  dr = (payload->getLaserMap())[rawid].p1 / pEB[0][cellid];
              }
              if (dr < pEBmin[0])
                pEBmin[0] = dr;
              if (dr > pEBmax[0])
                pEBmax[0] = dr;
              barrel[0]->Fill(phi, eta, dr);
              if (method == 0)  // difference
                dr = (payload->getLaserMap())[rawid].p2 - pEB[1][cellid];
              else {  // ratio
                if (pEB[1][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p2 == 0.)
                    dr = 1.;
                  else
                    dr = 9999.;  //use a large value
                } else
                  dr = (payload->getLaserMap())[rawid].p2 / pEB[1][cellid];
              }
              if (dr < pEBmin[1])
                pEBmin[1] = dr;
              if (dr > pEBmax[1])
                pEBmax[1] = dr;
              barrel[1]->Fill(phi, eta, dr);
              if (method == 0)  // difference
                dr = (payload->getLaserMap())[rawid].p3 - pEB[2][cellid];
              else {  // ratio
                if (pEB[2][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p3 == 0.)
                    dr = 1.;
                  else
                    dr = 9999.;  //use a large value
                } else
                  dr = (payload->getLaserMap())[rawid].p3 / pEB[2][cellid];
              }
              if (dr < pEBmin[2])
                pEBmin[2] = dr;
              if (dr > pEBmax[2])
                pEBmax[2] = dr;
              barrel[2]->Fill(phi, eta, dr);
            }
          }  // loop over cellid

          // looping over the EE channels
          for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
            if (!EEDetId::validHashIndex(cellid))
              continue;
            uint32_t rawid = EEDetId::unhashIndex(cellid);
            EEDetId myEEId(rawid);
            if (irun == 0) {
              pEE[0][cellid] = (payload->getLaserMap())[rawid].p1;
              pEE[1][cellid] = (payload->getLaserMap())[rawid].p2;
              pEE[2][cellid] = (payload->getLaserMap())[rawid].p3;
            } else {
              double dr1, dr2, dr3;
              if (method == 0)  // difference
                dr1 = (payload->getLaserMap())[rawid].p1 - pEE[0][cellid];
              else {  // ratio
                if (pEE[0][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p1 == 0.)
                    dr1 = 1.;
                  else
                    dr1 = 9999.;  //use a large value
                } else
                  dr1 = (payload->getLaserMap())[rawid].p1 / pEE[0][cellid];
              }
              if (dr1 < pEEmin[0])
                pEEmin[0] = dr1;
              if (dr1 > pEEmax[0])
                pEEmax[0] = dr1;
              if (method == 0)  // difference
                dr2 = (payload->getLaserMap())[rawid].p2 - pEE[1][cellid];
              else {  // ratio
                if (pEE[1][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p2 == 0.)
                    dr2 = 1.;
                  else
                    dr2 = 9999.;  //use a large value
                } else
                  dr2 = (payload->getLaserMap())[rawid].p2 / pEE[1][cellid];
              }
              if (dr2 < pEEmin[1])
                pEEmin[1] = dr2;
              if (dr2 > pEEmax[1])
                pEEmax[1] = dr2;
              if (method == 0)  // difference
                dr3 = (payload->getLaserMap())[rawid].p3 - pEE[2][cellid];
              else {  // ratio
                if (pEE[0][cellid] == 0.) {
                  if ((payload->getLaserMap())[rawid].p3 == 0.)
                    dr3 = 1.;
                  else
                    dr3 = 9999.;  //use a large value
                } else
                  dr3 = (payload->getLaserMap())[rawid].p3 / pEE[2][cellid];
              }
              if (dr3 < pEEmin[2])
                pEEmin[2] = dr3;
              if (dr3 > pEEmax[2])
                pEEmax[2] = dr3;
              if (myEEId.zside() == 1) {
                endc_p[0]->Fill(myEEId.ix(), myEEId.iy(), dr1);
                endc_p[1]->Fill(myEEId.ix(), myEEId.iy(), dr2);
                endc_p[2]->Fill(myEEId.ix(), myEEId.iy(), dr3);
              } else {
                endc_m[0]->Fill(myEEId.ix(), myEEId.iy(), dr1);
                endc_m[1]->Fill(myEEId.ix(), myEEId.iy(), dr2);
                endc_m[2]->Fill(myEEId.ix(), myEEId.iy(), dr3);
              }
            }
          }  // loop over cellid
        }  //  if payload.get()
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 2800, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      std::string dr[2] = {"-", "/"};
      if (IOV < 4294967296) {
        if (ntags == 2) {
          if (len < 80) {
            t1.SetTextSize(0.02);
            t1.DrawLatex(0.5,
                         0.96,
                         Form("%s IOV %i %s %s  IOV %i",
                              l_tagname[1].c_str(),
                              run[1],
                              dr[method].c_str(),
                              l_tagname[0].c_str(),
                              run[0]));
          } else {
            t1.SetTextSize(0.03);
            t1.DrawLatex(0.5, 0.96, Form("Ecal LaserAPDPNRatios, IOV %i %s %i", run[1], dr[method].c_str(), run[0]));
          }
        } else {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i %s %i", l_tagname[0].c_str(), run[1], dr[method].c_str(), run[0]));
        }
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
            t1.DrawLatex(0.5,
                         0.96,
                         Form("%s IOV %i %s %s  IOV %i",
                              l_tagname[1].c_str(),
                              run[1],
                              dr[method].c_str(),
                              l_tagname[0].c_str(),
                              run[0]));
          } else {
            t1.SetTextSize(0.03);
            t1.DrawLatex(0.5, 0.96, Form("Ecal LaserAPDPNRatios, IOV %i %s %i", run[1], dr[method].c_str(), run[0]));
          }
        } else {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i %s %i", l_tagname[0].c_str(), run[1], dr[method].c_str(), run[0]));
        }
      }
      float xmi[3] = {0.0, 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[3];
      for (int i = 0; i < 3; i++) {
        pad[i] = new TPad*[3];
        for (int obj = 0; obj < 3; obj++) {
          float yma = 0.94 - (0.32 * i);
          float ymi = yma - 0.28;
          pad[i][obj] = new TPad(Form("p_%i_%i", obj, i), Form("p_%i_%i", obj, i), xmi[obj], ymi, xma[obj], yma);
          pad[i][obj]->Draw();
        }
      }

      for (int i = 0; i < 3; i++) {
        // compute histo limits with some rounding
        //       std::cout << " before min " << pEBmin[i] << " max " << pEBmax[i];
        float xmin = (pEBmin[i] - 0.009) * 100.;
        int min = (int)xmin;
        pEBmin[i] = (float)min / 100.;
        float xmax = (pEBmax[i] + 0.009) * 100.;
        int max = (int)xmax;
        pEBmax[i] = (float)max / 100.;
        //       std::cout << " after min " << pEBmin[i] << " max " << pEBmax[i] << std::endl << " before min " << pEEmin[i] << " max " << pEEmax[i];
        xmin = (pEEmin[i] + 0.009) * 100.;
        min = (int)xmin;
        pEEmin[i] = (float)min / 100.;
        xmax = (pEEmax[i] + 0.009) * 100.;
        max = (int)xmax;
        pEEmax[i] = (float)max / 100.;
        //       std::cout << " after min " << pEEmin[i]  << " max " << pEEmax[i]<< std::endl;
        pad[i][0]->cd();
        DrawEE(endc_m[i], pEEmin[i], pEEmax[i]);
        endc_m[i]->GetZaxis()->SetLabelSize(0.02);
        pad[i][1]->cd();
        DrawEB(barrel[i], pEBmin[i], pEBmax[i]);
        pad[i][2]->cd();
        DrawEE(endc_p[i], pEEmin[i], pEEmax[i]);
        endc_p[i]->GetZaxis()->SetLabelSize(0.02);
      }

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosDiffBase
  using EcalLaserAPDPNRatiosDiffOneTag = EcalLaserAPDPNRatiosBase<cond::payloadInspector::MULTI_IOV, 1, 0>;
  using EcalLaserAPDPNRatiosDiffTwoTags = EcalLaserAPDPNRatiosBase<cond::payloadInspector::SINGLE_IOV, 2, 0>;
  using EcalLaserAPDPNRatiosRatioOneTag = EcalLaserAPDPNRatiosBase<cond::payloadInspector::MULTI_IOV, 1, 1>;
  using EcalLaserAPDPNRatiosRatioTwoTags = EcalLaserAPDPNRatiosBase<cond::payloadInspector::SINGLE_IOV, 2, 1>;

  // human readable label of an IOV since: run, run:lumi or date for time type IOVs
  inline std::string laserIOVLabel(cond::Time_t since) {
    if (since < 4294967296)  // run type IOV
      return std::to_string(since);
    cond::Time_t high = since >> 32;
    if (high < 100000000)  // lumi type IOV, the upper 32 bits are a run number, not a unix time
      return std::to_string(high) + ":" + std::to_string(since & 0xFFFFFFFF);
    time_t t = high;  // time type IOV
    char buf[256];
    struct tm lt;
    localtime_r(&t, &lt);
    strftime(buf, sizeof(buf), "%F %R:%S", &lt);
    buf[sizeof(buf) - 1] = 0;
    return std::string(buf);
  }

  /************************************************************************
     CMS publication style helpers, shared by the distribution comparison
     and the history plots. The text sizes are in pixels (precision 3 fonts)
  *************************************************************************/
  // colors of the two tags or IOVs, from the CMS recommended (Petroff) palette
  inline int laserTagColor(int itag) { return (itag == 0) ? TColor::GetColor("#5790fc") : TColor::GetColor("#e42536"); }

  // value of an input parameter of the plot, def if it is not set
  inline std::string laserInputParam(const std::map<std::string, std::string>& values,
                                     const std::string& name,
                                     const std::string& def) {
    auto it = values.find(name);
    return (it == values.end()) ? def : it->second;
  }

  // "CMS <cmsLabel>" left aligned at xLeft and rightLabel right aligned at xRight, baseline at y (NDC of the current pad).
  // No CMS label if cmsLabel is empty, "CMS" alone if cmsLabel is "CMS"
  inline void drawLaserCMSLabel(
      double xLeft, double xRight, double y, const std::string& cmsLabel, const std::string& rightLabel) {
    if (!cmsLabel.empty()) {
      TLatex cms;
      cms.SetNDC();
      cms.SetTextAlign(11);
      cms.SetTextFont(63);
      cms.SetTextSize(62);
      cms.DrawLatex(xLeft, y, "CMS");
      if (cmsLabel != "CMS") {
        TLatex width(0., 0., "CMS ");
        width.SetTextFont(63);
        width.SetTextSize(62);
        TLatex extra;
        extra.SetNDC();
        extra.SetTextAlign(11);
        extra.SetTextFont(53);
        extra.SetTextSize(48);
        extra.DrawLatex(xLeft + width.GetXsize(), y, cmsLabel.c_str());
      }
    }
    TLatex right;
    right.SetNDC();
    right.SetTextAlign(31);
    right.SetTextFont(43);
    right.SetTextSize(48);
    right.DrawLatex(xRight, y, rightLabel.c_str());
  }

  // one line per tag, left aligned at x from yTop down, TText instead of TLatex to keep the underscores
  inline void drawLaserTagLines(double x, double yTop, double step, int size, const std::vector<std::string>& lines) {
    TText text;
    text.SetNDC();
    text.SetTextAlign(13);
    text.SetTextFont(43);
    text.SetTextSize(size);
    for (size_t i = 0; i < lines.size(); i++) {
      text.SetTextColor((lines.size() == 1) ? int(kBlack) : laserTagColor(i));
      text.DrawText(x, yTop - step * i, lines[i].c_str());
    }
  }

  // default labels, they can be changed with the plot input parameters cmsLabel and rightLabel:
  // the CMS label is drawn only on request, e.g. cmsLabel=Preliminary for "CMS Preliminary"
  constexpr const char* kLaserCMSLabel = "";
  constexpr const char* kLaserRightLabel = "";

  /************************************************************************
     1d distributions of ECAL Laser APDPNRatios p1, p2 and p3 in EE-, EB
     and EE+, superimposed for 2 IOVs of one tag or 2 IOVs of two tags
  *************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalLaserAPDPNRatiosHistCompareBase
      : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags> {
  public:
    EcalLaserAPDPNRatiosHistCompareBase()
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags>(
              "ECAL Laser APDPNRatios distributions comparison") {
      // text after "CMS" (no CMS label if not set, "CMS" alone if set to CMS) and on the top right of the plot
      cond::payloadInspector::PlotBase::addInputParam("cmsLabel");
      cond::payloadInspector::PlotBase::addInputParam("rightLabel");
    }

    bool fill() override {
      constexpr int kParams = 3;  // p1, p2, p3
      constexpr int kParts = 3;   // EE-, EB, EE+
      constexpr int kBins = 100;
      const std::array<const char*, kParts> partName = {{"EE-", "EB", "EE+"}};

      auto tag1 = cond::payloadInspector::PlotBase::getTag<0>();
      std::array<std::string, 2> l_tagname = {{tag1.name, tag1.name}};
      std::array<std::tuple<cond::Time_t, cond::Hash>, 2> theIOVs = {{tag1.iovs.front(), tag1.iovs.back()}};
      if constexpr (ntags == 2) {
        auto tag2 = cond::payloadInspector::PlotBase::getTag<1>();
        l_tagname[1] = tag2.name;
        theIOVs[1] = tag2.iovs.front();
      }

      // values[iov][param][partition]
      std::vector<float> values[2][kParams][kParts];
      for (int irun = 0; irun < 2; irun++) {
        std::shared_ptr<EcalLaserAPDPNRatios> payload = this->fetchPayload(std::get<1>(theIOVs[irun]));
        if (!payload.get())
          return false;
        const auto& laserMap = payload->getLaserMap();

        // looping over the EB channels, via the dense-index, mapped into EBDetId's
        for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          const auto& apdpn = laserMap[rawid];
          values[irun][0][1].push_back(apdpn.p1);
          values[irun][1][1].push_back(apdpn.p2);
          values[irun][2][1].push_back(apdpn.p3);
        }  // loop over cellid

        // looping over the EE channels
        for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
          if (!EEDetId::validHashIndex(cellid))
            continue;
          uint32_t rawid = EEDetId::unhashIndex(cellid);
          int part = (EEDetId(rawid).zside() == 1) ? 2 : 0;
          const auto& apdpn = laserMap[rawid];
          values[irun][0][part].push_back(apdpn.p1);
          values[irun][1][part].push_back(apdpn.p2);
          values[irun][2][part].push_back(apdpn.p3);
        }  // loop over cellid
      }  // loop over IOVs

      // book the histograms with a common range for the two IOVs, covering all the values
      std::unique_ptr<TH1F> hist[2][kParams][kParts];
      for (int i = 0; i < kParams; i++) {
        for (int part = 0; part < kParts; part++) {
          for (int irun = 0; irun < 2; irun++) {
            hist[irun][i][part] = std::make_unique<TH1F>(
                Form("%s_p%i_%i", partName[part], i + 1, irun), Form(";p%i;Crystals", i + 1), kBins, 0., 1.01);
            hist[irun][i][part]->SetDirectory(nullptr);
            for (float v : values[irun][i][part])
              hist[irun][i][part]->Fill(v);
            hist[irun][i][part]->SetLineColor(laserTagColor(irun));
            hist[irun][i][part]->SetLineWidth(3);
          }
          // (1) filled, (2) as a line on top of it
          hist[0][i][part]->SetFillColorAlpha(laserTagColor(0), 0.35);
        }
      }

      // CMS style: text sizes in pixels (precision 3 fonts), one pad per parameter (rows) and partition (columns),
      // below the tag names and the CMS label
      constexpr double kGridTop = 0.914;
      constexpr double kLeft = 0.17, kRight = 0.04, kTop = 0.04, kBottom = 0.15;  // pad margins
      gStyle->SetOptStat(0);
      TCanvas canvas("CC hist", "CC hist", 2400, 2400);
      canvas.Divide(kParts, kParams);
      std::vector<std::unique_ptr<TObject>> keep;  // ROOT objects drawn on the canvas
      for (int i = 0; i < kParams; i++) {
        for (int part = 0; part < kParts; part++) {
          const double yhigh = kGridTop * (1. - double(i) / kParams), ylow = kGridTop * (1. - double(i + 1) / kParams);
          TVirtualPad* pad = canvas.cd(i * kParts + part + 1);
          pad->SetPad(part / 3., ylow, (part + 1) / 3., yhigh);
          pad->SetLeftMargin(kLeft);
          pad->SetRightMargin(kRight);
          pad->SetTopMargin(kTop);
          pad->SetBottomMargin(kBottom);
          pad->SetTickx(1);
          pad->SetTicky(1);

          TH1F* h1 = hist[0][i][part].get();
          TH1F* h2 = hist[1][i][part].get();
          h1->SetMaximum(1.5 * std::max(h1->GetMaximum(), h2->GetMaximum()));  // leave room for the legend
          h1->SetMinimum(0.);
          for (auto* axis : {h1->GetXaxis(), h1->GetYaxis()}) {
            axis->SetLabelFont(43);
            axis->SetTitleFont(43);
            axis->SetLabelSize(32);
            axis->SetTitleSize(36);
          }
          h1->GetXaxis()->SetNdivisions(505);
          h1->GetYaxis()->SetNdivisions(505);
          h1->GetYaxis()->SetMaxDigits(4);
          h1->GetXaxis()->SetTitleOffset(1.2);
          h1->GetYaxis()->SetTitleOffset(1.9);
          h1->Draw("HIST");
          h2->Draw("HIST SAME");
          h1->Draw("AXIS SAME");

          auto label = std::make_unique<TLatex>(kLeft + 0.04, 1. - kTop - 0.04, Form("%s", partName[part]));
          label->SetNDC();
          label->SetTextAlign(13);
          label->SetTextFont(63);
          label->SetTextSize(40);
          label->Draw();
          auto legend = std::make_unique<TLegend>(0.36, 0.78, 1. - kRight - 0.02, 1. - kTop - 0.03);
          legend->SetBorderSize(0);
          legend->SetFillStyle(0);
          legend->SetTextFont(43);
          legend->SetTextSize(28);
          for (int irun = 0; irun < 2; irun++)
            legend->AddEntry(hist[irun][i][part].get(),
                             Form("(%i) mean %.4f, RMS %.4f",
                                  irun + 1,
                                  hist[irun][i][part]->GetMean(),
                                  hist[irun][i][part]->GetRMS()),
                             (irun == 0) ? "f" : "l");
          legend->Draw();
          keep.push_back(std::move(label));
          keep.push_back(std::move(legend));
        }
      }

      // CMS label on top of the pads, tag names and IOVs above it
      canvas.cd();
      const auto& params = cond::payloadInspector::PlotBase::inputParamValues();
      drawLaserCMSLabel(kLeft / kParts,
                        1. - kRight / kParts,
                        kGridTop + 0.006,
                        laserInputParam(params, "cmsLabel", kLaserCMSLabel),
                        laserInputParam(params, "rightLabel", kLaserRightLabel));
      std::vector<std::string> lines;
      lines.reserve(2);
      for (int irun = 0; irun < 2; irun++)
        lines.push_back(Form(
            "(%i) %s, IOV %s", irun + 1, l_tagname[irun].c_str(), laserIOVLabel(std::get<0>(theIOVs[irun])).c_str()));
      drawLaserTagLines(kLeft / kParts, 0.99, 0.018, 34, lines);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosHistCompareBase
  using EcalLaserAPDPNRatiosHistCompareOneTag =
      EcalLaserAPDPNRatiosHistCompareBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalLaserAPDPNRatiosHistCompareTwoTags =
      EcalLaserAPDPNRatiosHistCompareBase<cond::payloadInspector::SINGLE_IOV, 2>;

  /************************************************************************
     helpers for the history plots: partitions are indexed as in the
     plots above, 0 = EE-, 1 = EB, 2 = EE+, parameters as 1 = p1, 2 = p2, 3 = p3
  *************************************************************************/
  enum EcalLaserPartition { kEEMinus = 0, kEB = 1, kEEPlus = 2 };
  constexpr int kNLaserPartitions = 3;
  enum EcalLaserStatistic { kMean = 0, kRMS = 1, kMedian = 2 };

  // order of the partitions in the pads of the history plots, from top to bottom
  constexpr std::array<int, kNLaserPartitions> kLaserPadOrder = {{kEB, kEEPlus, kEEMinus}};

  inline const char* laserPartitionName(int part) {
    constexpr std::array<const char*, kNLaserPartitions> names = {{"EE-", "EB", "EE+"}};
    return names[part];
  }

  inline const char* laserStatisticName(int stat) {
    constexpr std::array<const char*, 3> names = {{"mean", "RMS", "median"}};
    return names[stat];
  }

  template <int param>
  float laserParam(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair& apdpn) {
    static_assert(param >= 1 && param <= 3, "the APD/PN parameter must be p1, p2 or p3");
    if constexpr (param == 1)
      return apdpn.p1;
    else if constexpr (param == 2)
      return apdpn.p2;
    else
      return apdpn.p3;
  }

  // calls f(partition, value) for the parameter p<param> of every crystal, always in the same crystal order
  template <int param, typename F>
  void forEachLaserCrystal(const EcalLaserAPDPNRatios& payload, F&& f) {
    const auto& laserMap = payload.getLaserMap();
    for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid)
      f(kEB, laserParam<param>(laserMap[EBDetId::unhashIndex(cellid).rawId()]));
    for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
      if (!EEDetId::validHashIndex(cellid))
        continue;
      EEDetId myEEId = EEDetId::unhashIndex(cellid);
      f((myEEId.zside() == 1) ? kEEPlus : kEEMinus, laserParam<param>(laserMap[myEEId.rawId()]));
    }
  }

  // mean, RMS (standard deviation) and median of a set of values, the median is set by laserPartitionStats
  struct EcalLaserPartitionStats {
    double sum = 0.;
    double sum2 = 0.;
    unsigned int n = 0;
    double median = 0.;

    void add(double value) {
      sum += value;
      sum2 += value * value;
      n++;
    }
    double mean() const { return (n > 0) ? sum / n : 0.; }
    double rms() const { return (n > 0) ? std::sqrt(std::max(sum2 / n - mean() * mean(), 0.)) : 0.; }
    double get(int stat) const {
      switch (stat) {
        case kMean:
          return mean();
        case kRMS:
          return rms();
        default:
          return median;
      }
    }
  };
  typedef std::array<EcalLaserPartitionStats, kNLaserPartitions> EcalLaserStats;
  typedef std::array<std::vector<float>, kNLaserPartitions> EcalLaserValues;

  // mean, RMS and median over the crystals of each partition
  template <int param>
  EcalLaserStats laserPartitionStats(const EcalLaserAPDPNRatios& payload) {
    EcalLaserStats stats;
    EcalLaserValues values;
    forEachLaserCrystal<param>(payload, [&stats, &values](int part, float value) {
      stats[part].add(value);
      values[part].push_back(value);
    });
    for (int part = 0; part < kNLaserPartitions; part++) {
      auto& v = values[part];
      if (v.empty())
        continue;
      auto nth = v.begin() + static_cast<long>(0.5 * (v.size() - 1) + 0.5);
      std::nth_element(v.begin(), nth, v.end());
      stats[part].median = *nth;
    }
    return stats;
  }

  // values of the crystals of each partition
  template <int param>
  EcalLaserValues laserPartitionValues(const EcalLaserAPDPNRatios& payload) {
    EcalLaserValues values;
    forEachLaserCrystal<param>(payload, [&values](int part, float value) { values[part].push_back(value); });
    return values;
  }

  /************************************************************************
     common base of the history plots of p1, p2 or p3: one or two tags, one
     pad per partition (EB, EE+ and EE- from top to bottom).
     Two tags can have different IOV types: run and lumi based IOVs are put
     on a common run number axis (the lumi sections are squeezed in the
     [run, run+1) slot), time based IOVs can only be compared with other
     time based IOVs.
     With the input parameter xAxis=time the x axis is the IOV start time
     for all the tags: run and lumi based IOVs are placed at the start time
     of their run, from the RunInfo of the conditions DB, plus the lumi
     sections; then time based tags can be compared with run or lumi based ones.
  *************************************************************************/
  template <int param, int ntags>
  class EcalLaserAPDPNRatiosHistoryBase
      : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, ntags> {
  public:
    explicit EcalLaserAPDPNRatiosHistoryBase(const std::string& title)
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, ntags>(title) {
      // text after "CMS" (no CMS label if not set, "CMS" alone if set to CMS) and on the top right of the plot
      cond::payloadInspector::PlotBase::addInputParam("cmsLabel");
      cond::payloadInspector::PlotBase::addInputParam("rightLabel");
      // "run" (default) or "time"
      cond::payloadInspector::PlotBase::addInputParam("xAxis");
    }

  protected:
    static_assert(ntags == 1 || ntags == 2, "the history plots support one or two tags");

    // CMS style: canvas size in pixels, text sizes in pixels (precision 3 fonts), layout in canvas fractions
    static constexpr int kCanvasWidth = 2000;
    static constexpr int kCanvasHeight = 2400;
    static constexpr double kLeftMargin = 0.11;
    static constexpr double kRightMargin = 0.04;
    static constexpr double kBottomArea = 0.065;  // below the bottom frame, for the x axis labels and title
    static constexpr double kPadGap = 0.016;      // between two frames
    static constexpr double kTickLength = 0.012;
    static constexpr int kLabelSize = 40;
    static constexpr int kTitleSize = 44;
    static constexpr int kLegendSize = 38;
    static constexpr int kHeaderSize = 34;

    std::array<std::string, ntags> m_tagName;
    std::array<std::string, ntags> m_iovRange;  // human readable first - last IOV of each tag
    // one entry per IOV, x is the start of the IOV on the common axis
    std::array<std::vector<double>, ntags> m_x;
    std::array<std::vector<cond::Hash>, ntags> m_hash;
    std::array<std::vector<EcalLaserStats>, ntags> m_stats;  // filled by computeStats()
    bool m_timeAxis = false;

    // puts the IOVs of the tags on the common axis, returns false if one of them has no IOV in range
    bool collect() {
      auto tagRef = [this](int itag) {
        return (itag == 0) ? cond::payloadInspector::PlotBase::getTag<0>()
                           : cond::payloadInspector::PlotBase::getTag<1>();
      };
      const auto& params = cond::payloadInspector::PlotBase::inputParamValues();
      const std::string xAxis = laserInputParam(params, "xAxis", "run");
      if (xAxis != "run" && xAxis != "time")
        cond::throwException("the input parameter xAxis must be run or time, not " + xAxis,
                             "EcalLaserAPDPNRatiosHistoryBase::collect");
      const bool timeRequested = (xAxis == "time");

      std::array<cond::TimeType, ntags> l_timeType;
      for (int itag = 0; itag < ntags; itag++) {
        auto tag = tagRef(itag);
        m_tagName[itag] = tag.name;
        l_timeType[itag] = this->getTagInfo(tag.name).timeType;
        if (tag.iovs.empty())
          return false;
        const cond::Time_t first = std::max(tag.boundary.first, std::get<0>(tag.iovs.front()));
        const cond::Time_t last = std::get<0>(tag.iovs.back());
        m_iovRange[itag] = laserIOVLabel(first) + " - " + laserIOVLabel(last);

        // position of an IOV since on the axis, none if a run or lumi based IOV starts after the last run in RunInfo
        cond::persistency::RunInfoProxy runInfo;
        const bool fromRunInfo = timeRequested && l_timeType[itag] != cond::timestamp;
        if (fromRunInfo) {
          const bool lumi = (l_timeType[itag] == cond::lumiid);
          runInfo = this->dbSession().getRunInfo(lumi ? first >> 32 : first, lumi ? last >> 32 : last);
        }
        auto toAxis = [&](cond::Time_t since) -> std::optional<double> {
          return fromRunInfo ? iovToTime(runInfo, since, l_timeType[itag]) : iovToAxis(since, l_timeType[itag]);
        };

        // the first IOV can start well before the requested range (e.g. since = 1), clamp it
        const double xstart = toAxis(tag.boundary.first).value_or(std::numeric_limits<double>::lowest());
        for (const auto& iov : tag.iovs) {
          const auto x = toAxis(std::get<0>(iov));
          if (!x)
            continue;
          m_x[itag].push_back(std::max(xstart, *x));
          m_hash[itag].push_back(std::get<1>(iov));
        }
        if (m_x[itag].empty())
          cond::throwException("no run of the IOVs of " + tag.name + " found in RunInfo",
                               "EcalLaserAPDPNRatiosHistoryBase::collect");
      }
      m_timeAxis = timeRequested || (l_timeType[0] == cond::timestamp);
      if (!timeRequested && (l_timeType[ntags - 1] == cond::timestamp) != m_timeAxis) {
        cond::throwException(
            "cannot compare a time based tag with a run or lumi based tag on a run number axis, use xAxis=time",
            "EcalLaserAPDPNRatiosHistoryBase::collect");
      }
      return true;
    }

    // fetches the payloads and computes the crystal mean and RMS of every IOV of the tags
    bool computeStats() {
      for (int itag = 0; itag < ntags; itag++) {
        // payloads can be repeated in a tag, compute the statistics only once per hash
        std::map<cond::Hash, EcalLaserStats> cache;
        for (const auto& hash : m_hash[itag]) {
          auto it = cache.find(hash);
          if (it == cache.end()) {
            std::shared_ptr<EcalLaserAPDPNRatios> payload = this->fetchPayload(hash);
            if (!payload.get())
              return false;
            it = cache.emplace(hash, laserPartitionStats<param>(*payload)).first;
          }
          m_stats[itag].push_back(it->second);
        }
      }
      return true;
    }

    // common x range of the tags, the last IOV of each tag is extended up to the end of the plot
    std::pair<double, double> xRange(double xfirst) const {
      double xmax = m_x[0].back();
      for (int itag = 1; itag < ntags; itag++)
        xmax = std::max(xmax, m_x[itag].back());
      const double xmargin = (xmax > xfirst) ? 0.02 * (xmax - xfirst) : 1.;
      return {xfirst - xmargin, xmax + xmargin};
    }

    // top of the pads, above there are the tag names and the CMS label
    static constexpr double padsTop() { return 0.985 - 0.018 * ntags - 0.035; }

    // one pad per partition, the frames have the same height and share the x axis of the bottom one
    void makePads(TCanvas& canvas) const {
      const double frameHeight = (padsTop() - kBottomArea - 2.5 * kPadGap) / kNLaserPartitions;
      canvas.Divide(1, kNLaserPartitions);
      double frameTop = padsTop() - 0.5 * kPadGap;
      for (int ipad = 0; ipad < kNLaserPartitions; ipad++) {
        const double frameBottom = frameTop - frameHeight;
        const double below = (ipad == kNLaserPartitions - 1) ? kBottomArea : 0.5 * kPadGap;
        const double low = frameBottom - below, high = frameTop + 0.5 * kPadGap;
        TVirtualPad* pad = canvas.cd(ipad + 1);
        pad->SetPad(0., std::max(low, 0.), 1., high);
        pad->SetLeftMargin(kLeftMargin);
        pad->SetRightMargin(kRightMargin);
        pad->SetTopMargin(0.5 * kPadGap / (high - low));
        pad->SetBottomMargin(below / (high - low));
        pad->SetTickx(1);
        pad->SetTicky(1);
        pad->SetFillStyle(4000);
        frameTop = frameBottom - kPadGap;
      }
    }

    // empty histogram setting the axes of the pad of one partition, drawn in the current pad,
    // only the bottom pad shows the labels and the title of the x axis
    TH1F* drawFrame(std::vector<std::unique_ptr<TObject>>& keep,
                    int part,
                    bool bottom,
                    const std::string& yTitle,
                    double xmin,
                    double xmax,
                    double ymin,
                    double ymax) const {
      auto frame = std::make_unique<TH1F>(
          Form("frame_%s", laserPartitionName(part)),
          Form(";%s;%s", m_timeAxis ? "IOV start time (UTC)" : "IOV (run number)", yTitle.c_str()),
          1000,
          xmin,
          xmax);
      frame->SetDirectory(nullptr);
      frame->SetMinimum(ymin);
      frame->SetMaximum(ymax);
      TAxis* xaxis = frame->GetXaxis();
      TAxis* yaxis = frame->GetYaxis();
      for (auto* axis : {xaxis, yaxis}) {
        axis->SetLabelFont(43);
        axis->SetTitleFont(43);
        axis->SetLabelSize(kLabelSize);
        axis->SetTitleSize(kTitleSize);
      }
      if (!bottom) {
        xaxis->SetLabelSize(0);
        xaxis->SetTitleSize(0);
      }
      // the tick lengths are fractions of the pad size, keep them the same in all the pads
      xaxis->SetTickLength(kTickLength / gPad->GetHNDC());
      yaxis->SetTickLength(kTickLength * kCanvasHeight / kCanvasWidth);
      xaxis->SetTitleOffset(1.1);
      yaxis->SetTitleOffset(2.1);
      xaxis->SetLabelOffset(0.008 / gPad->GetHNDC());
      yaxis->SetNdivisions(505);
      if (m_timeAxis) {
        xaxis->SetTimeDisplay(1);
        xaxis->SetTimeFormat("%d/%m/%y%F1970-01-01 00:00:00");
        xaxis->SetNdivisions(510);
      } else {
        xaxis->SetNoExponent();
        xaxis->SetMaxDigits(7);
        xaxis->SetNdivisions(510);
      }
      frame->Draw("AXIS");
      // partition name in the top left corner of the frame
      auto label =
          std::make_unique<TLatex>(kLeftMargin + 0.02, 1. - gPad->GetTopMargin() - 0.05, laserPartitionName(part));
      label->SetNDC();
      label->SetTextAlign(13);
      label->SetTextFont(63);
      label->SetTextSize(48);
      label->Draw();
      keep.push_back(std::move(label));
      keep.push_back(std::move(frame));
      return static_cast<TH1F*>(keep.back().get());
    }

    // redraws the axes on top of the drawn objects
    static void redrawAxes(TH1F* frame) { frame->Draw("AXIS SAME"); }

    // legend at the top of the current frame, right of the partition name
    TLegend* drawLegend(std::vector<std::unique_ptr<TObject>>& keep, int ncolumns = 2) const {
      const double frameTop = 1. - gPad->GetTopMargin();
      const double frameHeight = frameTop - gPad->GetBottomMargin();
      auto legend = std::make_unique<TLegend>(
          kLeftMargin + 0.11, frameTop - 0.25 * frameHeight, 1. - kRightMargin - 0.02, frameTop - 0.03 * frameHeight);
      legend->SetNColumns(ncolumns);
      legend->SetBorderSize(0);
      legend->SetFillStyle(0);
      legend->SetTextFont(43);
      legend->SetTextSize(kLegendSize);
      keep.push_back(std::move(legend));
      return static_cast<TLegend*>(keep.back().get());
    }

    // CMS label on top of the frames, tag names above it
    void drawHeader(TCanvas& canvas) const {
      canvas.cd();
      const auto& params = cond::payloadInspector::PlotBase::inputParamValues();
      drawLaserCMSLabel(kLeftMargin,
                        1. - kRightMargin,
                        padsTop() + 0.006,
                        laserInputParam(params, "cmsLabel", kLaserCMSLabel),
                        laserInputParam(params, "rightLabel", kLaserRightLabel));
      std::vector<std::string> lines;
      lines.reserve(ntags);
      for (int itag = 0; itag < ntags; itag++)
        lines.push_back((ntags == 1)
                            ? Form("%s, IOVs %s", m_tagName[itag].c_str(), m_iovRange[itag].c_str())
                            : Form("(%i) %s, IOVs %s", itag + 1, m_tagName[itag].c_str(), m_iovRange[itag].c_str()));
      drawLaserTagLines(kLeftMargin, 0.99, 0.018, kHeaderSize, lines);
    }

    // appends to the graph the steps of y(x): the value holds until the next point, the last one until xend
    static void addSteps(TGraph& graph, const std::vector<double>& xs, const std::vector<double>& ys, double xend) {
      for (size_t i = 0; i < xs.size(); i++) {
        graph.AddPoint(xs[i], ys[i]);
        graph.AddPoint((i + 1 < xs.size()) ? xs[i + 1] : xend, ys[i]);
      }
    }

  private:
    // the lumi sections of a run are squeezed in the [run, run+1) slot of the run number axis
    static constexpr double kLumisPerRunSlot = 10000.;
    // duration of a lumi section, 2^18 LHC orbits
    static constexpr double kSecondsPerLumi = 23.31;

    // unix time of the start of a run or lumi based IOV: start time of the run from RunInfo, plus the lumi sections.
    // An IOV starting on a run not in RunInfo starts with the following run, none if there is no following run
    static std::optional<double> iovToTime(const cond::persistency::RunInfoProxy& runInfo,
                                           cond::Time_t since,
                                           cond::TimeType timeType) {
      const bool lumi = (timeType == cond::lumiid);
      const cond::Time_t run = lumi ? since >> 32 : since;
      auto it = runInfo.find(run);
      if (it == runInfo.end())
        return std::nullopt;
      const cond::RunInfo_t runInfoEntry = *it;
      double time = double(cond::time::from_boost(runInfoEntry.start) >> 32);
      if (lumi && runInfoEntry.run == run)
        time += kSecondsPerLumi * std::max(double(since & 0xFFFFFFFF) - 1., 0.);
      return time;
    }

    // position of an IOV since on the common axis: run number for run and lumi based IOVs, unix time otherwise
    static double iovToAxis(cond::Time_t since, cond::TimeType timeType) {
      switch (timeType) {
        case cond::lumiid: {
          double lumi = std::min<double>(since & 0xFFFFFFFF, kLumisPerRunSlot - 1);
          return double(since >> 32) + std::max(lumi - 1., 0.) / kLumisPerRunSlot;
        }
        case cond::timestamp:
          return double(since >> 32);
        default:
          return double(since);
      }
    }
  };  // class EcalLaserAPDPNRatiosHistoryBase

  /************************************************************************
     history of the crystal mean, RMS or median of p1, p2 or p3 in EB, EE+
     and EE- for one tag, or overlaid for two tags. The median is not affected
     by the few crystals with broken or very small values, unlike the mean
  *************************************************************************/
  template <int param, int stat, int ntags>
  class EcalLaserAPDPNRatiosStatHistory : public EcalLaserAPDPNRatiosHistoryBase<param, ntags> {
  public:
    typedef EcalLaserAPDPNRatiosHistoryBase<param, ntags> Base;
    EcalLaserAPDPNRatiosStatHistory()
        : Base(std::string("ECAL Laser APDPNRatios crystal ") + laserStatisticName(stat) + " history") {}

    bool fill() override {
      if (!this->collect() || !this->computeStats())
        return false;
      const std::array<int, 2> marker = {{kFullCircle, kFullSquare}};
      const auto& x = this->m_x;
      double xfirst = x[0].front();
      for (int itag = 1; itag < ntags; itag++)
        xfirst = std::min(xfirst, x[itag].front());
      const auto [xmin, xmax] = this->xRange(xfirst);

      gStyle->SetOptStat(0);
      std::vector<std::unique_ptr<TObject>> keep;  // ROOT objects drawn on the canvas
      TCanvas canvas("CC history", "CC history", Base::kCanvasWidth, Base::kCanvasHeight);
      this->makePads(canvas);
      for (int ipad = 0; ipad < kNLaserPartitions; ipad++) {
        const int part = kLaserPadOrder[ipad];
        std::array<std::vector<double>, ntags> y;
        double ymin = std::numeric_limits<double>::max(), ymax = std::numeric_limits<double>::lowest();
        for (int itag = 0; itag < ntags; itag++) {
          for (const auto& stats : this->m_stats[itag])
            y[itag].push_back(stats[part].get(stat));
          ymin = std::min(ymin, *std::min_element(y[itag].begin(), y[itag].end()));
          ymax = std::max(ymax, *std::max_element(y[itag].begin(), y[itag].end()));
        }
        const double ymargin = (ymax > ymin) ? 0.1 * (ymax - ymin) : 0.01;

        canvas.cd(ipad + 1);
        TH1F* frame = this->drawFrame(keep,
                                      part,
                                      ipad == kNLaserPartitions - 1,
                                      Form("Crystal %s p%i", laserStatisticName(stat), param),
                                      xmin,
                                      xmax,
                                      ymin - ymargin,
                                      ymax + 4 * ymargin);  // leave room for the legend
        TLegend* legend = this->drawLegend(keep);
        for (int itag = 0; itag < ntags; itag++) {
          const int color = laserTagColor(itag);
          const double average = std::accumulate(y[itag].begin(), y[itag].end(), 0.) / y[itag].size();
          auto averageLine = std::make_unique<TLine>(xmin, average, xmax, average);
          averageLine->SetLineColor(color);
          averageLine->SetLineStyle(kDashed);
          averageLine->SetLineWidth(2);
          averageLine->Draw();
          // step graph (the value holds until the next IOV) and markers at the start of each IOV
          auto steps = std::make_unique<TGraph>();
          Base::addSteps(*steps, x[itag], y[itag], xmax);
          auto points = std::make_unique<TGraph>(int(x[itag].size()), x[itag].data(), y[itag].data());
          for (auto* g : {steps.get(), points.get()}) {
            g->SetLineColor(color);
            g->SetMarkerColor(color);
          }
          steps->SetLineWidth(2);
          points->SetMarkerStyle(marker[itag]);
          points->SetMarkerSize(1.1);
          steps->Draw("L SAME");
          points->Draw("P SAME");
          const std::string prefix = (ntags == 1) ? "" : Form("(%i) ", itag + 1);
          legend->AddEntry(points.get(), Form("%s%zu IOVs", prefix.c_str(), x[itag].size()), "lp");
          legend->AddEntry(averageLine.get(), Form("%sAverage %.4f", prefix.c_str(), average), "l");
          keep.push_back(std::move(averageLine));
          keep.push_back(std::move(steps));
          keep.push_back(std::move(points));
        }
        legend->Draw();
        Base::redrawAxes(frame);
      }

      this->drawHeader(canvas);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosStatHistory
  using EcalLaserAPDPNRatiosP1MeanHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<1, kMean, 1>;
  using EcalLaserAPDPNRatiosP2MeanHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<2, kMean, 1>;
  using EcalLaserAPDPNRatiosP3MeanHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<3, kMean, 1>;
  using EcalLaserAPDPNRatiosP1RMSHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<1, kRMS, 1>;
  using EcalLaserAPDPNRatiosP2RMSHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<2, kRMS, 1>;
  using EcalLaserAPDPNRatiosP3RMSHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<3, kRMS, 1>;
  using EcalLaserAPDPNRatiosP1MeanHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<1, kMean, 2>;
  using EcalLaserAPDPNRatiosP2MeanHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<2, kMean, 2>;
  using EcalLaserAPDPNRatiosP3MeanHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<3, kMean, 2>;
  using EcalLaserAPDPNRatiosP1RMSHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<1, kRMS, 2>;
  using EcalLaserAPDPNRatiosP2RMSHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<2, kRMS, 2>;
  using EcalLaserAPDPNRatiosP3RMSHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<3, kRMS, 2>;
  using EcalLaserAPDPNRatiosP1MedianHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<1, kMedian, 1>;
  using EcalLaserAPDPNRatiosP2MedianHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<2, kMedian, 1>;
  using EcalLaserAPDPNRatiosP3MedianHistoryOneTag = EcalLaserAPDPNRatiosStatHistory<3, kMedian, 1>;
  using EcalLaserAPDPNRatiosP1MedianHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<1, kMedian, 2>;
  using EcalLaserAPDPNRatiosP2MedianHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<2, kMedian, 2>;
  using EcalLaserAPDPNRatiosP3MedianHistoryTwoTags = EcalLaserAPDPNRatiosStatHistory<3, kMedian, 2>;

  // median and central 68% interval (16th - 84th percentile, +-1 sigma for a gaussian) of a set of values,
  // insensitive to the few crystals with broken or very small values
  struct EcalLaserRobustSummary {
    double median = 0.;
    double low = 0.;
    double high = 0.;

    // the values are reordered
    static EcalLaserRobustSummary compute(std::vector<double>& values) {
      EcalLaserRobustSummary summary;
      if (values.empty())
        return summary;
      auto quantile = [&values](double q) {
        auto nth = values.begin() + static_cast<long>(q * (values.size() - 1) + 0.5);
        std::nth_element(values.begin(), nth, values.end());
        return *nth;
      };
      summary.median = quantile(0.5);
      summary.low = quantile(0.16);
      summary.high = quantile(0.84);
      return summary;
    }
  };

  /************************************************************************
     history of the crystal-by-crystal difference of p1, p2 or p3 in EB,
     EE+ and EE- between two tags, (2) - (1). The difference is evaluated on
     the union of the IOV boundaries of the two tags, for every point the
     line is the median over the crystals of the difference and the band
     its central 68% interval: both account for the two tags measuring the
     same crystals and are not biased by a few outlier crystals.
     With percent = true the crystal-by-crystal difference is relative to
     the first tag, 100 * ((2) - (1)) / (1), crystals with (1) = 0 are skipped
  *************************************************************************/
  template <int param, bool percent>
  class EcalLaserAPDPNRatiosDiffHistory : public EcalLaserAPDPNRatiosHistoryBase<param, 2> {
  public:
    typedef EcalLaserAPDPNRatiosHistoryBase<param, 2> Base;
    EcalLaserAPDPNRatiosDiffHistory()
        : Base(percent ? "ECAL Laser APDPNRatios crystal relative difference history"
                       : "ECAL Laser APDPNRatios crystal difference history") {}

    bool fill() override {
      if (!this->collect())
        return false;
      const auto& x = this->m_x;
      const auto& hashes = this->m_hash;

      // the difference is defined from the first point where both tags have a value,
      // and changes at every IOV boundary of either of the two tags
      const double xfirst = std::max(x[0].front(), x[1].front());
      std::vector<double> xd;
      for (int itag = 0; itag < 2; itag++)
        std::copy_if(
            x[itag].begin(), x[itag].end(), std::back_inserter(xd), [xfirst](double v) { return v >= xfirst; });
      xd.push_back(xfirst);
      std::sort(xd.begin(), xd.end());
      xd.erase(std::unique(xd.begin(), xd.end()), xd.end());

      // walk along the points keeping in memory only the crystal values of the current payload of each tag,
      // the statistics of the crystal-by-crystal difference are recomputed only when one of the payloads changes
      std::array<std::vector<EcalLaserRobustSummary>, kNLaserPartitions> summary;
      std::array<size_t, 2> index = {{0, 0}};  // last IOV of each tag starting before the current point
      std::array<cond::Hash, 2> currentHash;
      std::array<EcalLaserValues, 2> values;
      std::array<EcalLaserRobustSummary, kNLaserPartitions> current;
      std::vector<double> diffs;
      for (double xp : xd) {
        bool changed = false;
        for (int itag = 0; itag < 2; itag++) {
          while (index[itag] + 1 < x[itag].size() && x[itag][index[itag] + 1] <= xp)
            index[itag]++;
          const cond::Hash& hash = hashes[itag][index[itag]];
          if (hash == currentHash[itag])
            continue;
          std::shared_ptr<EcalLaserAPDPNRatios> payload = this->fetchPayload(hash);
          if (!payload.get())
            return false;
          values[itag] = laserPartitionValues<param>(*payload);
          currentHash[itag] = hash;
          changed = true;
        }
        if (changed) {
          for (int part = 0; part < kNLaserPartitions; part++) {
            diffs.clear();
            for (size_t i = 0; i < values[0][part].size(); i++) {
              const double v1 = values[0][part][i], v2 = values[1][part][i];
              if constexpr (percent) {
                if (v1 != 0.)
                  diffs.push_back(100. * (v2 - v1) / v1);
              } else {
                diffs.push_back(v2 - v1);
              }
            }
            current[part] = EcalLaserRobustSummary::compute(diffs);
          }
        }
        for (int part = 0; part < kNLaserPartitions; part++)
          summary[part].push_back(current[part]);
      }

      const auto [xmin, xmax] = this->xRange(xd.front());

      gStyle->SetOptStat(0);
      std::vector<std::unique_ptr<TObject>> keep;  // ROOT objects drawn on the canvas
      TCanvas canvas("CC history", "CC history", Base::kCanvasWidth, Base::kCanvasHeight);
      this->makePads(canvas);
      for (int ipad = 0; ipad < kNLaserPartitions; ipad++) {
        const int part = kLaserPadOrder[ipad];
        std::vector<double> median, up, down;
        for (const auto& point : summary[part]) {
          median.push_back(point.median);
          up.push_back(point.high);
          down.push_back(point.low);
        }
        const double ymin = std::min(*std::min_element(down.begin(), down.end()), 0.);
        const double ymax = std::max(*std::max_element(up.begin(), up.end()), 0.);
        const double ymargin = (ymax > ymin) ? 0.1 * (ymax - ymin) : 0.01;

        canvas.cd(ipad + 1);
        TH1F* frame = this->drawFrame(keep,
                                      part,
                                      ipad == kNLaserPartitions - 1,
                                      percent ? Form("(p%i^{(2)} - p%i^{(1)}) / p%i^{(1)} [%%]", param, param, param)
                                              : Form("p%i^{(2)} - p%i^{(1)}", param, param),
                                      xmin,
                                      xmax,
                                      ymin - ymargin,
                                      ymax + 4 * ymargin);  // leave room for the legend

        // the band is a variable bin histogram, one bin per step, drawn as error boxes from low to high
        std::vector<double> edges(xd);
        edges.push_back(xmax);
        auto band = std::make_unique<TH1D>(Form("band_%s", laserPartitionName(part)), "", int(xd.size()), edges.data());
        band->SetDirectory(nullptr);
        for (size_t i = 0; i < xd.size(); i++) {
          band->SetBinContent(i + 1, 0.5 * (up[i] + down[i]));
          band->SetBinError(i + 1, 0.5 * (up[i] - down[i]));
        }
        band->SetFillColorAlpha(laserTagColor(0), 0.45);
        band->SetLineWidth(0);
        band->SetMarkerSize(0);
        band->Draw("E2 SAME");
        auto zero = std::make_unique<TLine>(xmin, 0., xmax, 0.);
        zero->SetLineColor(kGray + 2);
        zero->SetLineStyle(kDashed);
        zero->SetLineWidth(2);
        zero->Draw();
        auto line = std::make_unique<TGraph>();
        Base::addSteps(*line, xd, median, xmax);
        line->SetLineColor(kBlack);
        line->SetLineWidth(2);
        line->Draw("L SAME");

        const double averageMedian = std::accumulate(median.begin(), median.end(), 0.) / xd.size();
        double averageWidth = 0.;
        for (size_t i = 0; i < xd.size(); i++)
          averageWidth += 0.5 * (up[i] - down[i]) / xd.size();
        TLegend* legend = this->drawLegend(keep, 1);
        if constexpr (percent) {
          legend->AddEntry(line.get(), Form("Median over crystals, average %.3f%%", averageMedian), "l");
          legend->AddEntry(
              band.get(), Form("16-84%% quantiles of crystals, average half width %.3f%%", averageWidth), "f");
        } else {
          legend->AddEntry(line.get(), Form("Median over crystals, average %.4f", averageMedian), "l");
          legend->AddEntry(
              band.get(), Form("16-84%% quantiles of crystals, average half width %.4f", averageWidth), "f");
        }
        legend->Draw();
        keep.push_back(std::move(band));
        keep.push_back(std::move(zero));
        keep.push_back(std::move(line));
        Base::redrawAxes(frame);
      }

      this->drawHeader(canvas);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosDiffHistory
  using EcalLaserAPDPNRatiosP1DiffHistory = EcalLaserAPDPNRatiosDiffHistory<1, false>;
  using EcalLaserAPDPNRatiosP2DiffHistory = EcalLaserAPDPNRatiosDiffHistory<2, false>;
  using EcalLaserAPDPNRatiosP3DiffHistory = EcalLaserAPDPNRatiosDiffHistory<3, false>;
  using EcalLaserAPDPNRatiosP1PercentDiffHistory = EcalLaserAPDPNRatiosDiffHistory<1, true>;
  using EcalLaserAPDPNRatiosP2PercentDiffHistory = EcalLaserAPDPNRatiosDiffHistory<2, true>;
  using EcalLaserAPDPNRatiosP3PercentDiffHistory = EcalLaserAPDPNRatiosDiffHistory<3, true>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalLaserAPDPNRatios) {
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBMap);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMap);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosRatioOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosRatioTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosHistCompareOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosHistCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1MeanHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2MeanHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3MeanHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1RMSHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2RMSHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3RMSHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1MeanHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2MeanHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3MeanHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1RMSHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2RMSHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3RMSHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1MedianHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2MedianHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3MedianHistoryOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1MedianHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2MedianHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3MedianHistoryTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1DiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2DiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3DiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP1PercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP2PercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosP3PercentDiffHistory);
}
