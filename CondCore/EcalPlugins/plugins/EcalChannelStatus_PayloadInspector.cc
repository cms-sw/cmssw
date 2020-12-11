#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
#include "CondCore/EcalPlugins/plugins/EcalBadCrystalsCount.h"
// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

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
        2d plot of ECAL barrel channel status of 1 IOV
  *******************************************************/
  class EcalChannelStatusEBMap : public cond::payloadInspector::PlotImage<EcalChannelStatus> {
  public:
    EcalChannelStatusEBMap() : cond::payloadInspector::PlotImage<EcalChannelStatus>("ECAL Barrel channel status") {
      setSingleIov(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      TH2F *ebmap = new TH2F("ebmap", "", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F *ebmap_coarse = new TH2F("ebmap_coarse", "", MAX_IPHI / 20, 0, MAX_IPHI, 2, -MAX_IETA, MAX_IETA);
      Int_t ebcount = 0;
      unsigned int run = 0;
      //      for ( auto const & iov: iovs) {
      auto iov = iovs.front();
      std::shared_ptr<EcalChannelStatus> payload = fetchPayload(std::get<1>(iov));
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
          //	    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
          Double_t weight = (Double_t)(*payload)[rawid].getEncodedStatusCode();
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
      t1.DrawLatex(0.5, 0.96, Form("EB Channel Status Masks, IOV %i", run));

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

  /*********************************************************
       2d plot of ECAL endcaps channel status of 1 IOV
  *********************************************************/
  class EcalChannelStatusEEMap : public cond::payloadInspector::PlotImage<EcalChannelStatus> {
  public:
    EcalChannelStatusEEMap() : cond::payloadInspector::PlotImage<EcalChannelStatus>("ECAL Barrel channel status") {
      setSingleIov(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      //      TH2F *eemap = new TH2F("eemap","", 2*IX_MAX, IX_MIN, 2*IX_MAX+1, IY_MAX, IY_MIN, IY_MAX+IY_MIN);
      TH2F *eemap = new TH2F("eemap", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      TH2F *eemap_coarse = new TH2F("eemap_coarse", "", 2, 0, 2 * IX_MAX, 1, 0, IY_MAX);
      TH2F *eetemp = new TH2F("eetemp", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      Int_t eecount = 0;
      unsigned int run = 0;
      auto iov = iovs.front();
      std::shared_ptr<EcalChannelStatus> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);
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
                //		    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
                float weight = (float)(*payload)[rawid].getEncodedStatusCode();
                if (iz == -1) {
                  //		      eemap->Fill(ix, iy, weight);
                  eemap->Fill(ix - 1, iy - 1, weight);
                  if (weight > 0) {
                    eecount++;
                    eemap_coarse->Fill(ix - 1, iy - 1);
                  }
                } else {
                  //		      eemap->Fill(ix+IX_MAX, iy, weight);
                  eemap->Fill(ix + IX_MAX - 1, iy - 1, weight);
                  if (weight > 0) {
                    eecount++;
                    eemap_coarse->Fill(ix + IX_MAX - 1, iy - 1);
                  }
                }
              }  // validDetId
      }          // payload

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
            //	    eetemp->SetBinContent(i + 1, j + 1, 2);
            //	    eetemp->SetBinContent(i + IX_MAX + 1, j +1, 2);
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
      t1.DrawLatex(0.5, 0.96, Form("EE Channel Status Masks, IOV %i", run));

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
    }  // fill method
  };

  /**********************************************************************
     2d plot of ECAL barrel channel status difference between 2 IOVs
  ***********************************************************************/
  class EcalChannelStatusEBDiff : public cond::payloadInspector::PlotImage<EcalChannelStatus> {
  public:
    EcalChannelStatusEBDiff()
        : cond::payloadInspector::PlotImage<EcalChannelStatus>("ECAL Barrel channel status difference") {
      setSingleIov(false);
      setTwoTags(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      TH2F *ebmap = new TH2F("ebmap", "", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F *ebmap_coarse = new TH2F("ebmap_coarse", "", MAX_IPHI / 20, 0, MAX_IPHI, 2, -MAX_IETA, MAX_IETA);
      Int_t ebcount = 0;
      unsigned int run[2], irun = 0, status[kEBChannels];
      for (auto const &iov : iovs) {
        std::shared_ptr<EcalChannelStatus> payload = fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          run[irun] = std::get<0>(iov);
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            // check the existence of ECAL channel status, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            //	    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
            if (irun == 0) {
              status[cellid] = (*payload)[rawid].getEncodedStatusCode();
            } else {
              unsigned int new_status = (*payload)[rawid].getEncodedStatusCode();
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
          irun++;
        }  // if payload.get()
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

      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("EB Channel Status Masks (Diff), IOV: %i vs %i", run[0], run[1]));

      char txt[80];
      sprintf(txt, "Net difference: %d channel(s)", ebcount);
      t1.SetTextColor(2);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.91, txt);

      std::string ImageName(m_imageFileName);
      c1.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /************************************************************************
       2d plot of ECAL endcaps channel status difference between 2 IOVs
  ************************************************************************/
  class EcalChannelStatusEEDiff : public cond::payloadInspector::PlotImage<EcalChannelStatus> {
  public:
    EcalChannelStatusEEDiff()
        : cond::payloadInspector::PlotImage<EcalChannelStatus>("ECAL Endcaps channel status difference") {
      setSingleIov(true);
      setTwoTags(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      TH2F *eemap = new TH2F("eemap", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      TH2F *eemap_coarse = new TH2F("eemap_coarse", "", 2, 0, 2 * IX_MAX, 1, 0, IY_MAX);
      TH2F *eetemp = new TH2F("eetemp", "", 2 * IX_MAX, 0, 2 * IX_MAX, IY_MAX, 0, IY_MAX);
      Int_t eecount = 0;
      unsigned int run[2], irun = 0, status[kEEChannels];
      for (auto const &iov : iovs) {
        std::shared_ptr<EcalChannelStatus> payload = fetchPayload(std::get<1>(iov));
        run[irun] = std::get<0>(iov);
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
                    status[channel] = (*payload)[rawid].getEncodedStatusCode();
                  } else {
                    unsigned int new_status = (*payload)[rawid].getEncodedStatusCode();
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
          irun++;
        }  //     get the payload
      }    //      loop over payloads

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
      t1.SetTextSize(0.055);
      t1.DrawLatex(0.5, 0.96, Form("EE Channel Status Masks (Diff), IOV %i vs %i", run[0], run[1]));

      char txt[80];
      sprintf(txt, "Net difference: %d channel(s)", eecount);
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
    }  // fill method
  };

  /*****************************************
 2d plot of EcalChannelStatus Error Summary of 1 IOV
 ******************************************/
  class EcalChannelStatusSummaryPlot : public cond::payloadInspector::PlotImage<EcalChannelStatus> {
  public:
    EcalChannelStatusSummaryPlot()
        : cond::payloadInspector::PlotImage<EcalChannelStatus>("Ecal Channel Status Error Summary - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<EcalChannelStatus> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F *align;                          //pointer to align which is a 2D histogram

      int NbRows = 3;
      int NbColumns = 3;

      if (payload.get()) {  //payload is an iov retrieved from payload using hash.

        align = new TH2F("Ecal Channel Status Error Summary",
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

        getBarrelErrorSummary<EcalChannelStatusCode>(payload->barrelItems(), ebErrorCount);
        getEndCapErrorSummary<EcalChannelStatusCode>(
            payload->endcapItems(), ee1ErrorCount, ee2ErrorCount, ee1Total, ee2Total);

        double row = NbRows - 0.5;

        //EB summary values
        align->Fill(0.5, row, 1);
        align->Fill(1.5, row, ebErrorCount);
        align->Fill(2.5, row, ebTotal);

        row--;

        align->Fill(0.5, row, 2);
        align->Fill(1.5, row, ee1ErrorCount);
        align->Fill(2.5, row, ee1Total);

        row--;

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
      t1.SetTextSize(0.045);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("EcalChannelStatus Error Summary, IOV %i", run));

      TPad *pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
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
PAYLOAD_INSPECTOR_MODULE(EcalChannelStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalChannelStatusEBMap);
  PAYLOAD_INSPECTOR_CLASS(EcalChannelStatusEEMap);
  PAYLOAD_INSPECTOR_CLASS(EcalChannelStatusEBDiff);
  PAYLOAD_INSPECTOR_CLASS(EcalChannelStatusEEDiff);
  PAYLOAD_INSPECTOR_CLASS(EcalChannelStatusSummaryPlot);
}
