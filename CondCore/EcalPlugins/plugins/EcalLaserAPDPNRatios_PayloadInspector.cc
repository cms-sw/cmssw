#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
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
#include <map>
#include <memory>
#include <numeric>
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
     1d distributions of ECAL Laser APDPNRatios p1, p2 and p3 in EE-, EB
     and EE+, superimposed for 2 IOVs of one tag or 2 IOVs of two tags
  *************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalLaserAPDPNRatiosHistCompareBase
      : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags> {
  public:
    EcalLaserAPDPNRatiosHistCompareBase()
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, nIOVs, ntags>(
              "ECAL Laser APDPNRatios distributions comparison") {}

    bool fill() override {
      constexpr int kParams = 3;  // p1, p2, p3
      constexpr int kParts = 3;   // EE-, EB, EE+
      constexpr int kBins = 100;
      const std::array<const char*, kParts> partName = {{"EE-", "EB", "EE+"}};
      const std::array<int, 2> color = {{kRed, kBlue}};

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
            hist[irun][i][part] = std::make_unique<TH1F>(Form("%s_p%i_%i", partName[part], i + 1, irun),
                                                         Form("%s p%i;p%i;crystals", partName[part], i + 1, i + 1),
                                                         kBins,
                                                         0.,
                                                         1.01);
            hist[irun][i][part]->SetDirectory(nullptr);
            for (float v : values[irun][i][part])
              hist[irun][i][part]->Fill(v);
            hist[irun][i][part]->SetLineColor(color[irun]);
            hist[irun][i][part]->SetLineWidth(2);
          }
        }
      }

      gStyle->SetOptStat(0);
      TCanvas canvas("CC hist", "CC hist", 2400, 2400);
      // rows: p1, p2, p3 ; columns: EE-, EB, EE+
      canvas.Divide(kParts, kParams);
      std::vector<std::unique_ptr<TLegend>> legends;
      for (int i = 0; i < kParams; i++) {
        for (int part = 0; part < kParts; part++) {
          float yma = 0.93 - (0.31 * i);
          float ymi = yma - 0.31;
          TVirtualPad* pad = canvas.cd(i * kParts + part + 1);
          pad->SetPad(part / 3., ymi, (part + 1) / 3., yma);  // leave room for the header
          pad->SetLeftMargin(0.13);
          pad->SetRightMargin(0.04);

          double ymax = std::max(hist[0][i][part]->GetMaximum(), hist[1][i][part]->GetMaximum());
	  if(part==0 || part==2){
	    hist[0][i][part]->SetMaximum(500);
	  } else {
	    hist[0][i][part]->SetMaximum(4500);
	  }
          hist[0][i][part]->Draw("HIST");
          hist[1][i][part]->Draw("HIST SAME");

          legends.push_back(std::make_unique<TLegend>(0.15, 0.78, 0.95, 0.89));
          legends.back()->SetTextSize(0.035);
          legends.back()->SetBorderSize(0);
          legends.back()->SetFillStyle(0);
          for (int irun = 0; irun < 2; irun++)
            legends.back()->AddEntry(hist[irun][i][part].get(),
                                     Form("(%i) mean %.4f, rms %.4f",
                                          irun + 1,
                                          hist[irun][i][part]->GetMean(),
                                          hist[irun][i][part]->GetRMS()),
                                     "l");
          legends.back()->Draw();
        }
      }

      // header, TText instead of TLatex to keep the underscores of the tag names
      canvas.cd();
      TText t1;
      t1.SetNDC();
      t1.SetTextAlign(22);
      t1.SetTextSize(0.018);
      for (int irun = 0; irun < 2; irun++) {
        t1.SetTextColor(color[irun]);
        t1.DrawText(0.5,
                    0.975 - 0.025 * irun,
                    Form("(%i) %s, IOV %s",
                         irun + 1,
                         l_tagname[irun].c_str(),
                         laserIOVLabel(std::get<0>(theIOVs[irun])).c_str()));
      }

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
  enum EcalLaserStatistic { kMean = 0, kRMS = 1 };

  inline const char* laserPartitionName(int part) {
    constexpr std::array<const char*, 3> names = {{"EE-", "EB", "EE+"}};
    return names[part];
  }

  inline const char* laserStatisticName(int stat) { return (stat == kMean) ? "mean" : "RMS"; }

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

  // calls f(value) for the parameter p<param> of every crystal of one partition, always in the same crystal order
  template <int param, int part, typename F>
  void forEachLaserCrystal(const EcalLaserAPDPNRatios& payload, F&& f) {
    static_assert(part >= kEEMinus && part <= kEEPlus, "the partition must be EE-, EB or EE+");
    const auto& laserMap = payload.getLaserMap();
    if constexpr (part == kEB) {
      for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid)
        f(laserParam<param>(laserMap[EBDetId::unhashIndex(cellid).rawId()]));
    } else {
      constexpr int zside = (part == kEEPlus) ? 1 : -1;
      for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
        if (!EEDetId::validHashIndex(cellid))
          continue;
        EEDetId myEEId = EEDetId::unhashIndex(cellid);
        if (myEEId.zside() == zside)
          f(laserParam<param>(laserMap[myEEId.rawId()]));
      }
    }
  }

  // mean and RMS (standard deviation) of a set of values
  struct EcalLaserPartitionStats {
    double mean = 0.;
    double rms = 0.;
    double get(int stat) const { return (stat == kMean) ? mean : rms; }

    template <typename Values>
    static EcalLaserPartitionStats compute(const Values& values) {
      double sum = 0., sum2 = 0.;
      unsigned int n = 0;
      values([&](double value) {
        sum += value;
        sum2 += value * value;
        n++;
      });
      EcalLaserPartitionStats stats;
      if (n > 0) {
        stats.mean = sum / n;
        stats.rms = std::sqrt(std::max(sum2 / n - stats.mean * stats.mean, 0.));
      }
      return stats;
    }
  };

  // mean and RMS over the crystals of one partition
  template <int param, int part>
  EcalLaserPartitionStats laserPartitionStats(const EcalLaserAPDPNRatios& payload) {
    return EcalLaserPartitionStats::compute([&payload](auto&& add) { forEachLaserCrystal<param, part>(payload, add); });
  }

  // values of the parameter p<param> of the crystals of one partition
  template <int param, int part>
  std::vector<float> laserPartitionValues(const EcalLaserAPDPNRatios& payload) {
    std::vector<float> values;
    forEachLaserCrystal<param, part>(payload, [&values](float value) { values.push_back(value); });
    return values;
  }

  /************************************************************************
     history of the crystal mean or RMS of p1, p2 or p3 in EE-, EB or EE+
     (one tag)
  *************************************************************************/
  template <int param, int part, int stat>
  class EcalLaserAPDPNRatiosStatHistory : public cond::payloadInspector::HistoryPlot<EcalLaserAPDPNRatios, float> {
  public:
    EcalLaserAPDPNRatiosStatHistory()
        : cond::payloadInspector::HistoryPlot<EcalLaserAPDPNRatios, float>(
              std::string(laserPartitionName(part)) + " crystal " + laserStatisticName(stat) + " history 'p" +
                  std::to_string(param) + "'",
              std::string(laserStatisticName(stat)) + " p" + std::to_string(param)) {}

    float getFromPayload(EcalLaserAPDPNRatios& payload) override {
      return laserPartitionStats<param, part>(payload).get(stat);
    }
  };
  using EcalLaserAPDPNRatiosEEMinusP1MeanHistory = EcalLaserAPDPNRatiosStatHistory<1, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP2MeanHistory = EcalLaserAPDPNRatiosStatHistory<2, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP3MeanHistory = EcalLaserAPDPNRatiosStatHistory<3, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEBP1MeanHistory = EcalLaserAPDPNRatiosStatHistory<1, kEB, kMean>;
  using EcalLaserAPDPNRatiosEBP2MeanHistory = EcalLaserAPDPNRatiosStatHistory<2, kEB, kMean>;
  using EcalLaserAPDPNRatiosEBP3MeanHistory = EcalLaserAPDPNRatiosStatHistory<3, kEB, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP1MeanHistory = EcalLaserAPDPNRatiosStatHistory<1, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP2MeanHistory = EcalLaserAPDPNRatiosStatHistory<2, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP3MeanHistory = EcalLaserAPDPNRatiosStatHistory<3, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP1RMSHistory = EcalLaserAPDPNRatiosStatHistory<1, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEEMinusP2RMSHistory = EcalLaserAPDPNRatiosStatHistory<2, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEEMinusP3RMSHistory = EcalLaserAPDPNRatiosStatHistory<3, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEBP1RMSHistory = EcalLaserAPDPNRatiosStatHistory<1, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEBP2RMSHistory = EcalLaserAPDPNRatiosStatHistory<2, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEBP3RMSHistory = EcalLaserAPDPNRatiosStatHistory<3, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP1RMSHistory = EcalLaserAPDPNRatiosStatHistory<1, kEEPlus, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP2RMSHistory = EcalLaserAPDPNRatiosStatHistory<2, kEEPlus, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP3RMSHistory = EcalLaserAPDPNRatiosStatHistory<3, kEEPlus, kRMS>;

  /************************************************************************
     common base of the two tags history plots: collects the crystal mean and
     RMS of p1, p2 or p3 in EE-, EB or EE+ for every IOV of the two tags.
     The two tags can have different IOV types: run and lumi based IOVs are
     put on a common run number axis (the lumi sections are squeezed in the
     [run, run+1) slot), time based IOVs can only be compared with other
     time based IOVs.
  *************************************************************************/
  template <int param, int part>
  class EcalLaserAPDPNRatiosHistoryCompareBase
      : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, 2> {
  public:
    explicit EcalLaserAPDPNRatiosHistoryCompareBase(const std::string& title)
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, 2>(title) {}

  protected:
    static constexpr std::array<int, 2> kColor = {{kRed, kBlue}};

    std::array<std::string, 2> m_tagName;
    std::array<std::string, 2> m_iovRange;  // human readable first - last IOV of each tag
    // one entry per IOV, x is the start of the IOV on the common axis
    std::array<std::vector<double>, 2> m_x;
    std::array<std::vector<cond::Hash>, 2> m_hash;
    std::array<std::vector<EcalLaserPartitionStats>, 2> m_stats;  // filled by computeStats()
    bool m_timeAxis = false;

    // puts the IOVs of the two tags on the common axis, returns false if one of them has no IOV in range
    bool collect() {
      auto tag1 = cond::payloadInspector::PlotBase::getTag<0>();
      auto tag2 = cond::payloadInspector::PlotBase::getTag<1>();
      m_tagName = {{tag1.name, tag2.name}};
      std::array<std::vector<std::tuple<cond::Time_t, cond::Hash>>, 2> l_iovs = {{tag1.iovs, tag2.iovs}};
      std::array<cond::Time_t, 2> l_start = {{tag1.boundary.first, tag2.boundary.first}};
      std::array<cond::TimeType, 2> l_timeType;
      for (int itag = 0; itag < 2; itag++)
        l_timeType[itag] = this->getTagInfo(m_tagName[itag]).timeType;

      const bool isTime1 = (l_timeType[0] == cond::timestamp), isTime2 = (l_timeType[1] == cond::timestamp);
      if (isTime1 != isTime2) {
        cond::throwException("cannot compare a time based tag with a run or lumi based tag",
                             "EcalLaserAPDPNRatiosHistoryCompareBase::collect");
      }
      m_timeAxis = isTime1;

      for (int itag = 0; itag < 2; itag++) {
        if (l_iovs[itag].empty())
          return false;
        // the first IOV can start well before the requested range (e.g. since = 1), clamp it
        const double xstart = iovToAxis(l_start[itag], l_timeType[itag]);
        for (const auto& iov : l_iovs[itag]) {
          m_x[itag].push_back(std::max(xstart, iovToAxis(std::get<0>(iov), l_timeType[itag])));
          m_hash[itag].push_back(std::get<1>(iov));
        }
        m_iovRange[itag] = laserIOVLabel(std::max(l_start[itag], std::get<0>(l_iovs[itag].front()))) + " - " +
                           laserIOVLabel(std::get<0>(l_iovs[itag].back()));
      }
      return true;
    }

    // fetches the payloads and computes the crystal mean and RMS of every IOV of the two tags
    bool computeStats() {
      for (int itag = 0; itag < 2; itag++) {
        // payloads can be repeated in a tag, compute the statistics only once per hash
        std::map<cond::Hash, EcalLaserPartitionStats> cache;
        for (const auto& hash : m_hash[itag]) {
          auto it = cache.find(hash);
          if (it == cache.end()) {
            std::shared_ptr<EcalLaserAPDPNRatios> payload = this->fetchPayload(hash);
            if (!payload.get())
              return false;
            it = cache.emplace(hash, laserPartitionStats<param, part>(*payload)).first;
          }
          m_stats[itag].push_back(it->second);
        }
      }
      return true;
    }

    // empty histogram setting the axes of the plot
    std::unique_ptr<TH1F> makeFrame(const std::string& yTitle, double xmin, double xmax) const {
      auto frame = std::make_unique<TH1F>(
          "frame",
          Form(";%s;%s", m_timeAxis ? "IOV start time (UTC)" : "IOV (run number)", yTitle.c_str()),
          1000,
          xmin,
          xmax);
      frame->SetDirectory(nullptr);
      frame->GetYaxis()->SetTitleOffset(1.2);
      if (m_timeAxis) {
        frame->GetXaxis()->SetTimeDisplay(1);
        frame->GetXaxis()->SetTimeFormat("%d/%m/%y%F1970-01-01 00:00:00");
        frame->GetXaxis()->SetNdivisions(510);
      } else {
        frame->GetXaxis()->SetNoExponent();
        frame->GetXaxis()->SetMaxDigits(7);
      }
      return frame;
    }

    // header, TText instead of TLatex to keep the underscores of the tag names
    void drawHeader(const std::string& title) const {
      TText t1;
      t1.SetNDC();
      t1.SetTextAlign(22);
      t1.SetTextSize(0.03);
      t1.DrawText(0.5, 0.965, title.c_str());
      t1.SetTextSize(0.022);
      for (int itag = 0; itag < 2; itag++) {
        t1.SetTextColor(kColor[itag]);
        t1.DrawText(0.5,
                    0.925 - 0.03 * itag,
                    Form("(%i) %s, IOVs %s", itag + 1, m_tagName[itag].c_str(), m_iovRange[itag].c_str()));
      }
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
  };  // class EcalLaserAPDPNRatiosHistoryCompareBase

  /************************************************************************
     history of the crystal mean or RMS of p1, p2 or p3 in EE-, EB or EE+,
     overlaid for two tags
  *************************************************************************/
  template <int param, int part, int stat>
  class EcalLaserAPDPNRatiosStatHistoryCompare : public EcalLaserAPDPNRatiosHistoryCompareBase<param, part> {
  public:
    typedef EcalLaserAPDPNRatiosHistoryCompareBase<param, part> Base;
    EcalLaserAPDPNRatiosStatHistoryCompare()
        : Base(std::string("ECAL Laser APDPNRatios crystal ") + laserStatisticName(stat) + " history comparison") {}

    bool fill() override {
      if (!this->collect() || !this->computeStats())
        return false;
      const std::array<int, 2> marker = {{kFullCircle, kFullSquare}};
      const auto& x = this->m_x;
      std::array<std::vector<double>, 2> y;
      for (int itag = 0; itag < 2; itag++)
        for (const auto& stats : this->m_stats[itag])
          y[itag].push_back(stats.get(stat));

      // common range, the last IOV of each tag is extended up to the end of the plot
      double xmin = std::min(x[0].front(), x[1].front());
      double xmax = std::max(x[0].back(), x[1].back());
      double ymin = std::min(*std::min_element(y[0].begin(), y[0].end()), *std::min_element(y[1].begin(), y[1].end()));
      double ymax = std::max(*std::max_element(y[0].begin(), y[0].end()), *std::max_element(y[1].begin(), y[1].end()));
      const double xmargin = (xmax > xmin) ? 0.02 * (xmax - xmin) : 1.;
      const double ymargin = (ymax > ymin) ? 0.1 * (ymax - ymin) : 0.01;
      xmin -= xmargin;
      xmax += xmargin;

      // step graphs (the value holds until the next IOV) and markers at the start of each IOV
      std::array<std::unique_ptr<TGraph>, 2> steps, points;
      std::array<double, 2> average;
      for (int itag = 0; itag < 2; itag++) {
        steps[itag] = std::make_unique<TGraph>();
        Base::addSteps(*steps[itag], x[itag], y[itag], xmax);
        points[itag] = std::make_unique<TGraph>(int(x[itag].size()), x[itag].data(), y[itag].data());
        average[itag] = std::accumulate(y[itag].begin(), y[itag].end(), 0.) / y[itag].size();
        for (auto* g : {steps[itag].get(), points[itag].get()}) {
          g->SetLineColor(Base::kColor[itag]);
          g->SetMarkerColor(Base::kColor[itag]);
        }
        steps[itag]->SetLineWidth(2);
        points[itag]->SetMarkerStyle(marker[itag]);
        points[itag]->SetMarkerSize(0.8);
      }

      gStyle->SetOptStat(0);
      TCanvas canvas("CC history", "CC history", 2000, 1200);
      canvas.SetLeftMargin(0.09);
      canvas.SetRightMargin(0.03);
      canvas.SetTopMargin(0.14);
      canvas.SetGrid();

      auto frame = this->makeFrame(
          Form("%s crystal %s p%i", laserPartitionName(part), laserStatisticName(stat), param), xmin, xmax);
      frame->SetMinimum(ymin - ymargin);
      frame->SetMaximum(ymax + 2 * ymargin);  // leave room for the legend
      frame->Draw();

      std::array<std::unique_ptr<TLine>, 2> averageLine;
      TLegend legend(0.10, 0.75, 0.96, 0.85);
      legend.SetNColumns(2);
      legend.SetBorderSize(0);
      legend.SetFillStyle(0);
      legend.SetTextSize(0.025);
      for (int itag = 0; itag < 2; itag++) {
        averageLine[itag] = std::make_unique<TLine>(xmin, average[itag], xmax, average[itag]);
        averageLine[itag]->SetLineColor(Base::kColor[itag]);
        averageLine[itag]->SetLineStyle(kDashed);
        averageLine[itag]->Draw();
        steps[itag]->Draw("L SAME");
        points[itag]->Draw("P SAME");
        legend.AddEntry(points[itag].get(), Form("(%i) %zu IOVs", itag + 1, x[itag].size()), "lp");
        legend.AddEntry(averageLine[itag].get(), Form("(%i) average %.4f", itag + 1, average[itag]), "l");
      }
      legend.Draw();

      this->drawHeader(Form("%s crystal %s history 'p%i'", laserPartitionName(part), laserStatisticName(stat), param));

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosStatHistoryCompare
  using EcalLaserAPDPNRatiosEEMinusP1MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP2MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP3MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEEMinus, kMean>;
  using EcalLaserAPDPNRatiosEBP1MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEB, kMean>;
  using EcalLaserAPDPNRatiosEBP2MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEB, kMean>;
  using EcalLaserAPDPNRatiosEBP3MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEB, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP1MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP2MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEPlusP3MeanHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEEPlus, kMean>;
  using EcalLaserAPDPNRatiosEEMinusP1RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEEMinusP2RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEEMinusP3RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEEMinus, kRMS>;
  using EcalLaserAPDPNRatiosEBP1RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEBP2RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEBP3RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEB, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP1RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<1, kEEPlus, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP2RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<2, kEEPlus, kRMS>;
  using EcalLaserAPDPNRatiosEEPlusP3RMSHistoryCompare = EcalLaserAPDPNRatiosStatHistoryCompare<3, kEEPlus, kRMS>;

  /************************************************************************
     history of the difference of the crystal mean of p1, p2 or p3 in EE-,
     EB or EE+ between two tags, (2) - (1). The difference is evaluated on
     the union of the IOV boundaries of the two tags, the band is the RMS
     of the crystal-by-crystal difference, which accounts for the two tags
     measuring the same crystals.
     With percent = true the crystal-by-crystal difference is relative to
     the first tag, 100 * ((2) - (1)) / (1), crystals with (1) = 0 are skipped
  *************************************************************************/
  template <int param, int part, bool percent>
  class EcalLaserAPDPNRatiosMeanDiffHistory : public EcalLaserAPDPNRatiosHistoryCompareBase<param, part> {
  public:
    typedef EcalLaserAPDPNRatiosHistoryCompareBase<param, part> Base;
    EcalLaserAPDPNRatiosMeanDiffHistory()
        : Base(percent ? "ECAL Laser APDPNRatios crystal mean relative difference history"
                       : "ECAL Laser APDPNRatios crystal mean difference history") {}

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
      std::vector<double> diff, err, up, down;
      std::array<size_t, 2> index = {{0, 0}};  // last IOV of each tag starting before the current point
      std::array<cond::Hash, 2> currentHash;
      std::array<std::vector<float>, 2> values;
      EcalLaserPartitionStats diffStats;
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
          values[itag] = laserPartitionValues<param, part>(*payload);
          currentHash[itag] = hash;
          changed = true;
        }
        if (changed) {
          diffStats = EcalLaserPartitionStats::compute([&values](auto&& add) {
            for (size_t i = 0; i < values[0].size(); i++) {
              const double v1 = values[0][i], v2 = values[1][i];
              if constexpr (percent) {
                if (v1 != 0.)
                  add(100. * (v2 - v1) / v1);
              } else {
                add(v2 - v1);
              }
            }
          });
        }
        diff.push_back(diffStats.mean);
        err.push_back(diffStats.rms);
        up.push_back(diff.back() + err.back());
        down.push_back(diff.back() - err.back());
      }

      double xmin = xd.front();
      double xmax = std::max(x[0].back(), x[1].back());
      const double xmargin = (xmax > xmin) ? 0.02 * (xmax - xmin) : 1.;
      xmin -= xmargin;
      xmax += xmargin;
      double ymin = std::min(*std::min_element(down.begin(), down.end()), 0.);
      double ymax = std::max(*std::max_element(up.begin(), up.end()), 0.);
      const double ymargin = (ymax > ymin) ? 0.1 * (ymax - ymin) : 0.01;

      // the band is a closed polygon: upper edge forward, lower edge backward
      TGraph band, line;
      Base::addSteps(band, xd, up, xmax);
      TGraph lower;
      Base::addSteps(lower, xd, down, xmax);
      for (int i = lower.GetN() - 1; i >= 0; i--)
        band.AddPoint(lower.GetX()[i], lower.GetY()[i]);
      band.SetFillColorAlpha(kAzure + 1, 0.35);
      band.SetLineColor(kAzure + 1);
      Base::addSteps(line, xd, diff, xmax);
      line.SetLineColor(kBlack);
      line.SetLineWidth(2);

      const double averageDiff = std::accumulate(diff.begin(), diff.end(), 0.) / diff.size();
      const double averageErr = std::accumulate(err.begin(), err.end(), 0.) / err.size();

      gStyle->SetOptStat(0);
      TCanvas canvas("CC history", "CC history", 2000, 1200);
      canvas.SetLeftMargin(0.09);
      canvas.SetRightMargin(0.03);
      canvas.SetTopMargin(0.14);
      canvas.SetGrid();

      auto frame = this->makeFrame(
          percent ? Form("%s #Delta crystal mean p%i, ((2) - (1)) / (1) [%%]", laserPartitionName(part), param)
                  : Form("%s #Delta crystal mean p%i, (2) - (1)", laserPartitionName(part), param),
          xmin,
          xmax);
      frame->SetMinimum(ymin - ymargin);
      frame->SetMaximum(ymax + 2 * ymargin);  // leave room for the legend
      frame->Draw();

      band.Draw("F SAME");
      TLine zero(xmin, 0., xmax, 0.);
      zero.SetLineStyle(kDashed);
      zero.Draw();
      line.Draw("L SAME");

      TLegend legend(0.10, 0.75, 0.96, 0.85);
      legend.SetNColumns(2);
      legend.SetBorderSize(0);
      legend.SetFillStyle(0);
      legend.SetTextSize(0.025);
      if constexpr (percent) {
        legend.AddEntry(&line, Form("mean of ((2) - (1)) / (1), average %.3f%%", averageDiff), "l");
        legend.AddEntry(
            &band, Form("#pm RMS of the crystal-by-crystal relative difference, average %.3f%%", averageErr), "f");
      } else {
        legend.AddEntry(&line, Form("mean (2) - mean (1), average %.4f", averageDiff), "l");
        legend.AddEntry(&band, Form("#pm RMS of the crystal-by-crystal difference, average %.4f", averageErr), "f");
      }
      legend.Draw();

      this->drawHeader(Form(
          "%s crystal mean %sdifference history 'p%i'", laserPartitionName(part), percent ? "relative " : "", param));

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };  // class EcalLaserAPDPNRatiosMeanDiffHistory
  using EcalLaserAPDPNRatiosEEMinusP1MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEEMinus, false>;
  using EcalLaserAPDPNRatiosEEMinusP2MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEEMinus, false>;
  using EcalLaserAPDPNRatiosEEMinusP3MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEEMinus, false>;
  using EcalLaserAPDPNRatiosEBP1MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEB, false>;
  using EcalLaserAPDPNRatiosEBP2MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEB, false>;
  using EcalLaserAPDPNRatiosEBP3MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEB, false>;
  using EcalLaserAPDPNRatiosEEPlusP1MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEEPlus, false>;
  using EcalLaserAPDPNRatiosEEPlusP2MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEEPlus, false>;
  using EcalLaserAPDPNRatiosEEPlusP3MeanDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEEPlus, false>;
  using EcalLaserAPDPNRatiosEEMinusP1MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEEMinus, true>;
  using EcalLaserAPDPNRatiosEEMinusP2MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEEMinus, true>;
  using EcalLaserAPDPNRatiosEEMinusP3MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEEMinus, true>;
  using EcalLaserAPDPNRatiosEBP1MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEB, true>;
  using EcalLaserAPDPNRatiosEBP2MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEB, true>;
  using EcalLaserAPDPNRatiosEBP3MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEB, true>;
  using EcalLaserAPDPNRatiosEEPlusP1MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<1, kEEPlus, true>;
  using EcalLaserAPDPNRatiosEEPlusP2MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<2, kEEPlus, true>;
  using EcalLaserAPDPNRatiosEEPlusP3MeanPercentDiffHistory = EcalLaserAPDPNRatiosMeanDiffHistory<3, kEEPlus, true>;

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
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3MeanHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3RMSHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3MeanHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3RMSHistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3MeanDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2MeanPercentDiffHistory);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3MeanPercentDiffHistory);
}
