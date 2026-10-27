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
#include <ctime>
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

  inline const char* laserPartitionName(int part) {
    constexpr std::array<const char*, 3> names = {{"EE-", "EB", "EE+"}};
    return names[part];
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

  // mean over the crystals of one partition of the parameter p<param>
  template <int param, int part>
  float laserPartitionMean(const EcalLaserAPDPNRatios& payload) {
    static_assert(part >= kEEMinus && part <= kEEPlus, "the partition must be EE-, EB or EE+");
    const auto& laserMap = payload.getLaserMap();
    double sum = 0.;
    unsigned int n = 0;
    if constexpr (part == kEB) {
      for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
        sum += laserParam<param>(laserMap[EBDetId::unhashIndex(cellid).rawId()]);
        n++;
      }
    } else {
      constexpr int zside = (part == kEEPlus) ? 1 : -1;
      for (int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
        if (!EEDetId::validHashIndex(cellid))
          continue;
        EEDetId myEEId = EEDetId::unhashIndex(cellid);
        if (myEEId.zside() != zside)
          continue;
        sum += laserParam<param>(laserMap[myEEId.rawId()]);
        n++;
      }
    }
    return n > 0 ? sum / n : 0.;
  }

  /************************************************************************
     history of the crystal mean of p1, p2 or p3 in EE-, EB or EE+ (one tag)
  *************************************************************************/
  template <int param, int part>
  class EcalLaserAPDPNRatiosMeanHistory : public cond::payloadInspector::HistoryPlot<EcalLaserAPDPNRatios, float> {
  public:
    EcalLaserAPDPNRatiosMeanHistory()
        : cond::payloadInspector::HistoryPlot<EcalLaserAPDPNRatios, float>(
              std::string(laserPartitionName(part)) + " crystal mean history 'p" + std::to_string(param) + "'",
              "mean p" + std::to_string(param)) {}

    float getFromPayload(EcalLaserAPDPNRatios& payload) override { return laserPartitionMean<param, part>(payload); }
  };
  using EcalLaserAPDPNRatiosEEMinusP1History = EcalLaserAPDPNRatiosMeanHistory<1, kEEMinus>;
  using EcalLaserAPDPNRatiosEEMinusP2History = EcalLaserAPDPNRatiosMeanHistory<2, kEEMinus>;
  using EcalLaserAPDPNRatiosEEMinusP3History = EcalLaserAPDPNRatiosMeanHistory<3, kEEMinus>;
  using EcalLaserAPDPNRatiosEBP1History = EcalLaserAPDPNRatiosMeanHistory<1, kEB>;
  using EcalLaserAPDPNRatiosEBP2History = EcalLaserAPDPNRatiosMeanHistory<2, kEB>;
  using EcalLaserAPDPNRatiosEBP3History = EcalLaserAPDPNRatiosMeanHistory<3, kEB>;
  using EcalLaserAPDPNRatiosEEPlusP1History = EcalLaserAPDPNRatiosMeanHistory<1, kEEPlus>;
  using EcalLaserAPDPNRatiosEEPlusP2History = EcalLaserAPDPNRatiosMeanHistory<2, kEEPlus>;
  using EcalLaserAPDPNRatiosEEPlusP3History = EcalLaserAPDPNRatiosMeanHistory<3, kEEPlus>;

  /************************************************************************
     history of the crystal mean of p1, p2 or p3 in EE-, EB or EE+,
     overlaid for two tags. The two tags can have different IOV types:
     run and lumi based IOVs are put on a common run number axis (the lumi
     sections are squeezed in the [run, run+1) slot), time based IOVs can
     only be compared with other time based IOVs.
  *************************************************************************/
  template <int param, int part>
  class EcalLaserAPDPNRatiosMeanHistoryCompare
      : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, 2> {
  public:
    EcalLaserAPDPNRatiosMeanHistoryCompare()
        : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios, cond::payloadInspector::MULTI_IOV, 2>(
              "ECAL Laser APDPNRatios crystal mean history comparison") {}

    bool fill() override {
      const std::array<int, 2> color = {{kRed, kBlue}};
      const std::array<int, 2> marker = {{kFullCircle, kFullSquare}};

      auto tag1 = cond::payloadInspector::PlotBase::getTag<0>();
      auto tag2 = cond::payloadInspector::PlotBase::getTag<1>();
      std::array<std::string, 2> l_tagname = {{tag1.name, tag2.name}};
      std::array<std::vector<std::tuple<cond::Time_t, cond::Hash>>, 2> l_iovs = {{tag1.iovs, tag2.iovs}};
      std::array<cond::Time_t, 2> l_start = {{tag1.boundary.first, tag2.boundary.first}};
      std::array<cond::TimeType, 2> l_timeType;
      for (int itag = 0; itag < 2; itag++)
        l_timeType[itag] = this->getTagInfo(l_tagname[itag]).timeType;

      const bool isTime1 = (l_timeType[0] == cond::timestamp), isTime2 = (l_timeType[1] == cond::timestamp);
      if (isTime1 != isTime2) {
        cond::throwException("cannot overlay a time based tag with a run or lumi based tag",
                             "EcalLaserAPDPNRatiosMeanHistoryCompare::fill");
      }
      const bool timeAxis = isTime1;

      // x[tag], y[tag]: one point per IOV, x is the start of the IOV on the common axis
      std::array<std::vector<double>, 2> x, y;
      for (int itag = 0; itag < 2; itag++) {
        if (l_iovs[itag].empty())
          return false;
        // the first IOV can start well before the requested range (e.g. since = 1), clamp it
        const double xstart = iovToAxis(l_start[itag], l_timeType[itag]);
        // payloads can be repeated in a tag, compute the mean only once per hash
        std::map<cond::Hash, float> cache;
        for (const auto& iov : l_iovs[itag]) {
          const cond::Hash& hash = std::get<1>(iov);
          auto it = cache.find(hash);
          if (it == cache.end()) {
            std::shared_ptr<EcalLaserAPDPNRatios> payload = this->fetchPayload(hash);
            if (!payload.get())
              continue;
            it = cache.emplace(hash, laserPartitionMean<param, part>(*payload)).first;
          }
          x[itag].push_back(std::max(xstart, iovToAxis(std::get<0>(iov), l_timeType[itag])));
          y[itag].push_back(it->second);
        }
        if (x[itag].empty())
          return false;
      }

      // common range, the last IOV of each tag is extended up to the end of the plot
      double xmin = std::min(x[0].front(), x[1].front());
      double xmax = std::max(x[0].back(), x[1].back());
      double ymin = std::min(*std::min_element(y[0].begin(), y[0].end()), *std::min_element(y[1].begin(), y[1].end()));
      double ymax = std::max(*std::max_element(y[0].begin(), y[0].end()), *std::max_element(y[1].begin(), y[1].end()));
      const double xmargin = (xmax > xmin) ? 0.02 * (xmax - xmin) : 1.;
      const double ymargin = (ymax > ymin) ? 0.1 * (ymax - ymin) : 0.01;
      xmax += xmargin;

      // step graphs (the value holds until the next IOV) and markers at the start of each IOV
      std::array<std::unique_ptr<TGraph>, 2> steps, points;
      std::array<double, 2> mean;
      for (int itag = 0; itag < 2; itag++) {
        const auto& xs = x[itag];
        const auto& ys = y[itag];
        steps[itag] = std::make_unique<TGraph>();
        for (size_t i = 0; i < xs.size(); i++) {
          double xend = (i + 1 < xs.size()) ? xs[i + 1] : xmax;
          steps[itag]->AddPoint(xs[i], ys[i]);
          steps[itag]->AddPoint(xend, ys[i]);
        }
        points[itag] = std::make_unique<TGraph>(xs.size(), xs.data(), ys.data());
        mean[itag] = std::accumulate(ys.begin(), ys.end(), 0.) / ys.size();
        for (auto* g : {steps[itag].get(), points[itag].get()}) {
          g->SetLineColor(color[itag]);
          g->SetMarkerColor(color[itag]);
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

      TH1F frame("frame",
                 Form(";%s;%s crystal mean p%i",
                      timeAxis ? "IOV start time (UTC)" : "IOV (run number)",
                      laserPartitionName(part),
                      param),
                 1000,
                 xmin - xmargin,
                 xmax);
      frame.SetDirectory(nullptr);
      frame.SetMinimum(ymin - ymargin);
      frame.SetMaximum(ymax + 2 * ymargin);  // leave room for the legend
      frame.GetYaxis()->SetTitleOffset(1.2);
      if (timeAxis) {
        frame.GetXaxis()->SetTimeDisplay(1);
        frame.GetXaxis()->SetTimeFormat("%d/%m/%y%F1970-01-01 00:00:00");
        frame.GetXaxis()->SetNdivisions(510);
      } else {
        frame.GetXaxis()->SetNoExponent();
        frame.GetXaxis()->SetMaxDigits(7);
      }
      frame.Draw();

      std::array<std::unique_ptr<TLine>, 2> meanLine;
      TLegend legend(0.10, 0.75, 0.96, 0.85);
      legend.SetNColumns(2);
      legend.SetBorderSize(0);
      legend.SetFillStyle(0);
      legend.SetTextSize(0.025);
      for (int itag = 0; itag < 2; itag++) {
        meanLine[itag] = std::make_unique<TLine>(xmin - xmargin, mean[itag], xmax, mean[itag]);
        meanLine[itag]->SetLineColor(color[itag]);
        meanLine[itag]->SetLineStyle(kDashed);
        meanLine[itag]->Draw();
        steps[itag]->Draw("L SAME");
        points[itag]->Draw("P SAME");
        legend.AddEntry(points[itag].get(), Form("(%i) %zu IOVs", itag + 1, x[itag].size()), "lp");
        legend.AddEntry(meanLine[itag].get(), Form("(%i) mean %.4f", itag + 1, mean[itag]), "l");
      }
      legend.Draw();

      // header, TText instead of TLatex to keep the underscores of the tag names
      TText t1;
      t1.SetNDC();
      t1.SetTextAlign(22);
      t1.SetTextSize(0.03);
      t1.DrawText(0.5, 0.965, Form("%s crystal mean history 'p%i'", laserPartitionName(part), param));
      t1.SetTextSize(0.022);
      for (int itag = 0; itag < 2; itag++) {
        t1.SetTextColor(color[itag]);
        t1.DrawText(0.5,
                    0.925 - 0.03 * itag,
                    Form("(%i) %s, IOVs %s - %s",
                         itag + 1,
                         l_tagname[itag].c_str(),
                         laserIOVLabel(std::max(l_start[itag], std::get<0>(l_iovs[itag].front()))).c_str(),
                         laserIOVLabel(std::get<0>(l_iovs[itag].back())).c_str()));
      }

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method

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
  };  // class EcalLaserAPDPNRatiosMeanHistoryCompare
  using EcalLaserAPDPNRatiosEEMinusP1HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<1, kEEMinus>;
  using EcalLaserAPDPNRatiosEEMinusP2HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<2, kEEMinus>;
  using EcalLaserAPDPNRatiosEEMinusP3HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<3, kEEMinus>;
  using EcalLaserAPDPNRatiosEBP1HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<1, kEB>;
  using EcalLaserAPDPNRatiosEBP2HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<2, kEB>;
  using EcalLaserAPDPNRatiosEBP3HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<3, kEB>;
  using EcalLaserAPDPNRatiosEEPlusP1HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<1, kEEPlus>;
  using EcalLaserAPDPNRatiosEEPlusP2HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<2, kEEPlus>;
  using EcalLaserAPDPNRatiosEEPlusP3HistoryCompare = EcalLaserAPDPNRatiosMeanHistoryCompare<3, kEEPlus>;

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
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3History);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP1HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP2HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMinusP3HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP1HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP2HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBP3HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP1HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP2HistoryCompare);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEPlusP3HistoryCompare);
}
