#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <memory>
#include <sstream>

#include "TStyle.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648, NRGBs = 5, NCont = 255 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /*******************************************************
     2d plot of ECAL barrel DQM channel status of 1 IOV
  *******************************************************/
  class EcalDQMChannelStatusEBMap : public cond::payloadInspector::PlotImage<EcalDQMChannelStatus> {
  public:
    EcalDQMChannelStatusEBMap()
        : cond::payloadInspector::PlotImage<EcalDQMChannelStatus>("ECAL Barrel DQM channel status") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      TH2F *ebmap = new TH2F(
          "ebmap", "", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);  //drawing the whole map (barels)
      TH2F *ebmap_coarse = new TH2F(
          "ebmap_coarse", "", MAX_IPHI / 20, 0, MAX_IPHI, 2, -MAX_IETA, MAX_IETA);  //drawing the halves (18 by 2)
      Int_t ebcount = 0;
      unsigned int run = 0;
      //      for ( auto const & iov: iovs) {
      auto iov = iovs.front();
      std::shared_ptr<EcalDQMChannelStatus> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);

      if (payload.get()) {
        // looping over the EB channels, via the dense-index, mapped into EBDetId's
        if (payload->barrelItems().empty())
          return false;
        for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          // check the existence of ECAL channel status, for a given ECAL barrel channel
          if (payload->find(rawid) == payload->end())
            continue;
          //if (!(*payload)[rawid].getEncodedStatusCode()) continue;
          Double_t weight = (Double_t)(*payload)[rawid].getStatusCode();
          Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
          Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
          if (eta > 0.)
            eta = eta - 0.5;  //   0.5 to 84.5
          else
            eta = eta + 0.5;  //  -84.5 to -0.5

          ebmap->Fill(phi, eta, weight);
          if (weight > 0) {
            ebcount++;
            ebmap_coarse->Fill(phi, eta);
          }
        }  // loop over cellid
      }    // if payload.get()
      else
        return false;

      gStyle->SetOptStat(0);
      //set the background color to white
      gStyle->SetFillColor(10);
      gStyle->SetFrameFillColor(10);
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetTitleFillColor(0);
      gStyle->SetStatColor(10);
      //dont put a colored frame around the plots
      gStyle->SetFrameBorderMode(0);
      gStyle->SetCanvasBorderMode(0);
      gStyle->SetPadBorderMode(0);
      //use the primary color palette
      gStyle->SetPalette(1);

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);

      TCanvas c1("c1", "c1", 1200, 700);
      c1.SetGridx(1);
      c1.SetGridy(1);

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.06);

      ebmap->SetXTitle("i#phi");
      ebmap->SetYTitle("i#eta");
      ebmap->GetXaxis()->SetNdivisions(-418, kFALSE);
      ebmap->GetYaxis()->SetNdivisions(-1702, kFALSE);
      ebmap->GetXaxis()->SetLabelSize(0.03);
      ebmap->GetYaxis()->SetLabelSize(0.03);
      ebmap->GetXaxis()->SetTickLength(0.01);
      ebmap->GetYaxis()->SetTickLength(0.01);
      ebmap->SetMaximum(15);

      c1.cd();
      ebmap->Draw("colz");

      ebmap_coarse->SetMarkerSize(1.3);
      ebmap_coarse->Draw("text,same");

      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("EB DQM Channel Status, IOV %i", run));

      char txt[80];
      Double_t prop = (Double_t)ebcount / kEBChannels * 100.;
      sprintf(txt, "%d/61200 (%4.3f%%)", ebcount, prop);
      t1.SetTextColor(2);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.91, txt);

      std::string ImageName(m_imageFileName);
      c1.SaveAs(ImageName.c_str());
      return true;
    }
  };

  /*******************************************************
     2d plot of ECAL Endcap DQM channel status of 1 IOV
   *******************************************************/
  class EcalDQMChannelStatusEEMap : public cond::payloadInspector::PlotImage<EcalDQMChannelStatus> {
  public:
    EcalDQMChannelStatusEEMap()
        : cond::payloadInspector::PlotImage<EcalDQMChannelStatus>("ECAL EndCaps DQM channel status") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      //TH2F *eemap = new TH2F("eemap","", 2*IX_MAX, IX_MIN, 2*IX_MAX+1, IY_MAX, IY_MIN, IY_MAX+IY_MIN);
      TH2F *eemap = new TH2F("eemap", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      TH2F *eemap_coarse = new TH2F("eemap_coarse", "", 2, 0, 2 * IX_MAX, 1, 0, IY_MAX);
      TH2F *eetemp = new TH2F("eetemp", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);

      Int_t eecount = 0;
      unsigned int run = 0;
      auto iov = iovs.front();
      std::shared_ptr<EcalDQMChannelStatus> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);  //Time_t parameter

      if (payload.get()) {
        if (payload->endcapItems().empty())
          return false;

        // looping over the EE channels
        for (int iz = -1; iz < 2; iz = iz + 2)  // -1 or +1
          for (int iy = IY_MIN; iy < IY_MAX + IY_MIN; iy++)
            for (int ix = IX_MIN; ix < IX_MAX + IX_MIN; ix++)

              if (EEDetId::validDetId(ix, iy, iz)) {
                EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                uint32_t rawid = myEEId.rawId();
                // check the existence of ECAL channel status, for a given ECAL endcap channel
                if (payload->find(rawid) == payload->end())
                  continue;
                //        if (!(*payload)[rawid].getEncodedStatusCode()) continue;
                float weight = (float)(*payload)[rawid].getStatusCode();
                if (iz == -1) {
                  //          eemap->Fill(ix, iy, weight);
                  eemap->Fill(ix - 1, iy - 1, weight);
                  if (weight > 0) {
                    eecount++;
                    eemap_coarse->Fill(ix - 1, iy - 1);
                  }
                } else {
                  //          eemap->Fill(ix+IX_MAX, iy, weight);
                  eemap->Fill(ix + IX_MAX - 1, iy - 1, weight);
                  if (weight > 0) {
                    eecount++;
                    eemap_coarse->Fill(ix + IX_MAX - 1, iy - 1);
                  }
                }
              }  // validDetId

      }  // payload

      gStyle->SetOptStat(0);
      //set the background color to white
      gStyle->SetFillColor(10);
      gStyle->SetFrameFillColor(10);
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetTitleFillColor(0);
      gStyle->SetStatColor(10);
      //dont put a colored frame around the plots
      gStyle->SetFrameBorderMode(0);
      gStyle->SetCanvasBorderMode(0);
      gStyle->SetPadBorderMode(0);
      //use the primary color palette
      gStyle->SetPalette(1);

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);

      // set the EE contours
      for (Int_t i = 1; i <= IX_MAX; i++) {
        for (Int_t j = 1; j <= IY_MAX; j++) {
          if (EEDetId::validDetId(i, j, 1)) {
            //      eetemp->SetBinContent(i + 1, j + 1, 2);
            //      eetemp->SetBinContent(i + IX_MAX + 1, j +1, 2);
            eetemp->SetBinContent(i, j, 2);
            eetemp->SetBinContent(i + IX_MAX, j, 2);
          }
        }
      }

      eetemp->SetFillColor(920);
      TCanvas c1("c1", "c1", 1200, 600);
      c1.SetGridx(1);
      c1.SetGridy(1);

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.06);

      eetemp->GetXaxis()->SetNdivisions(40, kFALSE);
      eetemp->GetYaxis()->SetNdivisions(20, kFALSE);
      eetemp->GetXaxis()->SetLabelSize(0.00);
      eetemp->GetYaxis()->SetLabelSize(0.00);
      eetemp->GetXaxis()->SetTickLength(0.01);
      eetemp->GetYaxis()->SetTickLength(0.01);
      eetemp->SetMaximum(1.15);

      eemap->GetXaxis()->SetNdivisions(40, kFALSE);
      eemap->GetYaxis()->SetNdivisions(20, kFALSE);
      eemap->GetXaxis()->SetLabelSize(0.00);
      eemap->GetYaxis()->SetLabelSize(0.00);
      eemap->GetXaxis()->SetTickLength(0.01);
      eemap->GetYaxis()->SetTickLength(0.01);
      eemap->SetMaximum(15);

      eetemp->Draw("box");
      eemap->Draw("same,colz");

      eemap_coarse->SetMarkerSize(2);
      eemap_coarse->Draw("same,text");

      t1.SetTextColor(1);
      t1.SetTextSize(0.055);
      t1.DrawLatex(0.5, 0.96, Form("EE DQM Channel Status, IOV %i", run));

      char txt[80];
      Double_t prop = (Double_t)eecount / kEEChannels * 100.;
      sprintf(txt, "%d/14648 (%4.3f%%)", eecount, prop);
      t1.SetTextColor(2);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.91, txt);
      t1.SetTextColor(1);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.14, 0.84, "EE-");
      t1.DrawLatex(0.86, 0.84, "EE+");

      std::string ImageName(m_imageFileName);
      c1.SaveAs(ImageName.c_str());
      return true;
    }
  };

  /**********************************************************************
     2d plot of ECAL barrel DQM channel status difference between 2 IOVs
  ***********************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalDQMChannelStatusEBDiffBase : public cond::payloadInspector::PlotImage<EcalDQMChannelStatus, nIOVs, ntags> {
  public:
    EcalDQMChannelStatusEBDiffBase()
        : cond::payloadInspector::PlotImage<EcalDQMChannelStatus, nIOVs, ntags>(
              "ECAL Barrel DQM channel status difference") {}
    bool fill() override {
      TH2F *ebmap = new TH2F("ebmap", "", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F *ebmap_coarse = new TH2F("ebmap_coarse", "", MAX_IPHI / 20, 0, MAX_IPHI, 2, -MAX_IETA, MAX_IETA);
      Int_t ebcount = 0;
      unsigned int run[2], status[kEBChannels];
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
        std::shared_ptr<EcalDQMChannelStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            // check the existence of ECAL channel status, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;

            if (irun == 0) {
              status[cellid] = (*payload)[rawid].getStatusCode();
            } else {
              unsigned int new_status = (*payload)[rawid].getStatusCode();
              if (new_status != status[cellid]) {
                int tmp3 = 0;

                if (new_status > status[cellid])
                  tmp3 = 1;
                else
                  tmp3 = -1;

                Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
                Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
                if (eta > 0.)
                  eta = eta - 0.5;  //   0.5 to 84.5
                else
                  eta = eta + 0.5;  //  -84.5 to -0.5

                ebmap->Fill(phi, eta, 0.05 + 0.95 * (tmp3 > 0));
                ebcount++;
                ebmap_coarse->Fill(phi, eta, tmp3);
              }
            }
          }  // loop over cellid
        }    // if payload.get()
        else
          return false;
      }  // loop over IOV's

      gStyle->SetOptStat(0);
      //set the background color to white
      gStyle->SetFillColor(10);
      gStyle->SetFrameFillColor(10);
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetTitleFillColor(0);
      gStyle->SetStatColor(10);
      //dont put a colored frame around the plots
      gStyle->SetFrameBorderMode(0);
      gStyle->SetCanvasBorderMode(0);
      gStyle->SetPadBorderMode(0);
      //use the primary color palette
      gStyle->SetPalette(1);

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);

      TCanvas c1("c1", "c1", 1200, 700);
      c1.SetGridx(1);
      c1.SetGridy(1);

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.06);

      ebmap->SetXTitle("i#phi");
      ebmap->SetYTitle("i#eta");
      ebmap->GetXaxis()->SetNdivisions(-418, kFALSE);
      ebmap->GetYaxis()->SetNdivisions(-1702, kFALSE);
      ebmap->GetXaxis()->SetLabelSize(0.03);
      ebmap->GetYaxis()->SetLabelSize(0.03);
      ebmap->GetXaxis()->SetTickLength(0.01);
      ebmap->GetYaxis()->SetTickLength(0.01);
      ebmap->SetMaximum(1.15);

      c1.cd();
      ebmap->Draw("colz");

      ebmap_coarse->SetMarkerSize(1.3);
      ebmap_coarse->Draw("text,same");

      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2 && len < 58) {
        t1.SetTextSize(0.025);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("EB DQM Channel Status (Diff), IOV: %i vs %i", run[0], run[1]));
      }
      char txt[80];
      sprintf(txt, "Net difference: %d channel(s)", ebcount);
      t1.SetTextColor(2);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.91, txt);

      std::string ImageName(this->m_imageFileName);
      c1.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalDQMChannelStatusEBDiffBase
  using EcalDQMChannelStatusEBDiffOneTag = EcalDQMChannelStatusEBDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalDQMChannelStatusEBDiffTwoTags = EcalDQMChannelStatusEBDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

  /************************************************************************
      2d plot of ECAL endcaps DQM channel status difference between 2 IOVs
  ************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalDQMChannelStatusEEDiffBase : public cond::payloadInspector::PlotImage<EcalDQMChannelStatus, nIOVs, ntags> {
  public:
    EcalDQMChannelStatusEEDiffBase()
        : cond::payloadInspector::PlotImage<EcalDQMChannelStatus, nIOVs, ntags>(
              "ECAL Endcaps DQM channel status difference") {}

    bool fill() override {
      TH2F *eemap = new TH2F("eemap", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      TH2F *eemap_coarse = new TH2F("eemap_coarse", "", 2, 0, 2 * IX_MAX, 1, 0, IY_MAX);
      TH2F *eetemp = new TH2F("eetemp", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      Int_t eecount = 0;
      unsigned int run[2], status[kEEChannels];
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
        std::shared_ptr<EcalDQMChannelStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          if (payload->endcapItems().empty())
            return false;

          // looping over the EE channels
          for (int iz = -1; iz < 2; iz = iz + 2)  // -1 or +1
            for (int iy = IY_MIN; iy < IY_MAX + IY_MIN; iy++)
              for (int ix = IX_MIN; ix < IX_MAX + IX_MIN; ix++)
                if (EEDetId::validDetId(ix, iy, iz)) {
                  EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                  uint32_t rawid = myEEId.rawId();
                  int channel = myEEId.hashedIndex();
                  // check the existence of ECAL channel status, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;

                  if (irun == 0) {
                    status[channel] = (*payload)[rawid].getStatusCode();
                  } else {
                    unsigned int new_status = (*payload)[rawid].getStatusCode();
                    if (new_status != status[channel]) {
                      int tmp3 = 0;
                      if (new_status > status[channel])
                        tmp3 = 1;
                      else
                        tmp3 = -1;

                      if (iz == -1) {
                        eemap->Fill(ix - 1, iy - 1, 0.05 + 0.95 * (tmp3 > 0));
                        eecount++;
                        eemap_coarse->Fill(ix - 1, iy - 1, tmp3);
                      } else {
                        eemap->Fill(ix + IX_MAX - 1, iy - 1, 0.05 + 0.95 * (tmp3 > 0));
                        eecount++;
                        eemap_coarse->Fill(ix + IX_MAX - 1, iy - 1, tmp3);
                      }  // z side
                    }    //  any difference ?
                  }      //   2nd IOV, fill the plots
                }        //    validDetId
        }                //     get the payload
      }                  //      loop over payloads

      gStyle->SetOptStat(0);
      //set the background color to white
      gStyle->SetFillColor(10);
      gStyle->SetFrameFillColor(10);
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetTitleFillColor(0);
      gStyle->SetStatColor(10);
      //dont put a colored frame around the plots
      gStyle->SetFrameBorderMode(0);
      gStyle->SetCanvasBorderMode(0);
      gStyle->SetPadBorderMode(0);
      //use the primary color palette
      gStyle->SetPalette(1);

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);

      // set the EE contours
      for (Int_t i = 1; i <= IX_MAX; i++) {
        for (Int_t j = 1; j <= IY_MAX; j++) {
          if (EEDetId::validDetId(i, j, 1)) {
            eetemp->SetBinContent(i, j, 2);
            eetemp->SetBinContent(i + IX_MAX, j, 2);
          }
        }
      }

      eetemp->SetFillColor(920);
      TCanvas c1("c1", "c1", 1200, 600);
      c1.SetGridx(1);
      c1.SetGridy(1);

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.06);

      eetemp->GetXaxis()->SetNdivisions(40, kFALSE);
      eetemp->GetYaxis()->SetNdivisions(20, kFALSE);
      eetemp->GetXaxis()->SetLabelSize(0.00);
      eetemp->GetYaxis()->SetLabelSize(0.00);
      eetemp->GetXaxis()->SetTickLength(0.01);
      eetemp->GetYaxis()->SetTickLength(0.01);
      eetemp->SetMaximum(1.15);

      eemap->GetXaxis()->SetNdivisions(40, kFALSE);
      eemap->GetYaxis()->SetNdivisions(20, kFALSE);
      eemap->GetXaxis()->SetLabelSize(0.00);
      eemap->GetYaxis()->SetLabelSize(0.00);
      eemap->GetXaxis()->SetTickLength(0.01);
      eemap->GetYaxis()->SetTickLength(0.01);
      eemap->SetMaximum(1.15);

      eetemp->Draw("box");
      eemap->Draw("same,colz");

      eemap_coarse->SetMarkerSize(2);
      eemap_coarse->Draw("same,text");

      t1.SetTextColor(1);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2 && len < 58) {
        t1.SetTextSize(0.025);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.055);
        t1.DrawLatex(0.5, 0.96, Form("EE DQM Channel Status (Diff), IOV %i vs %i", run[0], run[1]));
      }
      char txt[80];
      sprintf(txt, "Net difference: %d channel(s)", eecount);
      t1.SetTextColor(2);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.91, txt);
      t1.SetTextColor(1);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.14, 0.84, "EE-");
      t1.DrawLatex(0.86, 0.84, "EE+");

      std::string ImageName(this->m_imageFileName);
      c1.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalDQMChannelStatusEEDiffBase
  using EcalDQMChannelStatusEEDiffOneTag = EcalDQMChannelStatusEEDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalDQMChannelStatusEEDiffTwoTags = EcalDQMChannelStatusEEDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalDQMChannelStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEBMap);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEEMap);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEBDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEBDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEEDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalDQMChannelStatusEEDiffTwoTags);
}
