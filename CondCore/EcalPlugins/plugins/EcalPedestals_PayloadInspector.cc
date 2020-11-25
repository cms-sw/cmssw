#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>

namespace {
  enum { kEBChannels = 61200, kEEChannels = 14648, kGains = 3, kRMS = 5 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360 };  // barrel lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };         // endcaps lower and upper bounds on x and y

  /*************************************************
     1d plot of ECAL pedestal of 1 IOV
  *************************************************/
  class EcalPedestalsHist : public cond::payloadInspector::PlotImage<EcalPedestals> {
  public:
    EcalPedestalsHist() : cond::payloadInspector::PlotImage<EcalPedestals>("ECAL pedestal map") { setSingleIov(true); }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH1F** barrel_m = new TH1F*[kGains];
      TH1F** endcap_m = new TH1F*[kGains];
      TH1F** barrel_r = new TH1F*[kGains];
      TH1F** endcap_r = new TH1F*[kGains];
      float bmin[kGains] = {0.7, 0.5, 0.4};
      float bmax[kGains] = {2.2, 1.3, 0.7};
      float emin[kGains] = {1.5, 0.8, 0.4};
      float emax[kGains] = {2.5, 1.5, 0.8};
      for (int gainId = 0; gainId < kGains; gainId++) {
        barrel_m[gainId] = new TH1F(Form("EBm%i", gainId), Form("mean %i EB", gainValues[gainId]), 100, 150., 250.);
        endcap_m[gainId] = new TH1F(Form("EEm%i", gainId), Form("mean %i EE", gainValues[gainId]), 100, 150., 250.);
        barrel_r[gainId] =
            new TH1F(Form("EBr%i", gainId), Form("rms %i EB", gainValues[gainId]), 100, bmin[gainId], bmax[gainId]);
        endcap_r[gainId] =
            new TH1F(Form("EEr%i", gainId), Form("rms %i EE", gainValues[gainId]), 100, emin[gainId], emax[gainId]);
      }
      auto iov = iovs.front();
      std::shared_ptr<EcalPedestals> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        // looping over the EB channels, via the dense-index, mapped into EBDetId's
        if (payload->barrelItems().empty())
          return false;
        for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          if (payload->find(rawid) == payload->end())
            continue;
          barrel_m[0]->Fill((*payload)[rawid].mean_x12);
          barrel_r[0]->Fill((*payload)[rawid].rms_x12);
          barrel_m[1]->Fill((*payload)[rawid].mean_x6);
          barrel_r[1]->Fill((*payload)[rawid].rms_x6);
          barrel_m[2]->Fill((*payload)[rawid].mean_x1);
          barrel_r[2]->Fill((*payload)[rawid].rms_x1);
        }  // loop over cellid
        if (payload->endcapItems().empty())
          return false;

        // looping over the EE channels
        for (int iz = -1; iz < 2; iz = iz + 2)  // -1 or +1
          for (int iy = IY_MIN; iy < IY_MAX + IY_MIN; iy++)
            for (int ix = IX_MIN; ix < IX_MAX + IX_MIN; ix++)
              if (EEDetId::validDetId(ix, iy, iz)) {
                EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                uint32_t rawid = myEEId.rawId();
                if (payload->find(rawid) == payload->end())
                  continue;
                endcap_m[0]->Fill((*payload)[rawid].mean_x12);
                endcap_r[0]->Fill((*payload)[rawid].rms_x12);
                endcap_m[1]->Fill((*payload)[rawid].mean_x6);
                endcap_r[1]->Fill((*payload)[rawid].rms_x6);
                endcap_m[2]->Fill((*payload)[rawid].mean_x1);
                endcap_r[2]->Fill((*payload)[rawid].rms_x1);
              }  // validDetId
      }          // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(111110);
      TCanvas canvas("CC map", "CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i", run));

      float xmi[2] = {0.0, 0.50};
      float xma[2] = {0.50, 1.00};
      TPad*** pad = new TPad**[6];
      for (int gId = 0; gId < 6; gId++) {
        pad[gId] = new TPad*[2];
        for (int obj = 0; obj < 2; obj++) {
          //	  float yma = 1.- (0.17 * gId);
          //	  float ymi = yma - 0.15;
          float yma = 0.94 - (0.16 * gId);
          float ymi = yma - 0.14;
          pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId), Form("p_%i_%i", obj, gId), xmi[obj], ymi, xma[obj], yma);
          pad[gId][obj]->Draw();
        }
      }
      for (int gId = 0; gId < kGains; gId++) {
        pad[gId][0]->cd();
        barrel_m[gId]->Draw();
        pad[gId + kGains][0]->cd();
        barrel_r[gId]->Draw();
        pad[gId][1]->cd();
        endcap_m[gId]->Draw();
        pad[gId + kGains][1]->cd();
        endcap_r[gId]->Draw();
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   //   class EcalPedestalsHist

  /*************************************************
     2d plot of ECAL pedestal of 1 IOV
  *************************************************/

  class EcalPedestalsPlot : public cond::payloadInspector::PlotImage<EcalPedestals> {
  public:
    EcalPedestalsPlot() : cond::payloadInspector::PlotImage<EcalPedestals>("ECAL pedestal map") { setSingleIov(true); }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      TH2F** barrel_r = new TH2F*[kGains];
      TH2F** endc_p_r = new TH2F*[kGains];
      TH2F** endc_m_r = new TH2F*[kGains];
      double EBmean[kGains], EBrms[kGains], EEmean[kGains], EErms[kGains], pEBmin[kGains], pEBmax[kGains],
          pEEmin[kGains], pEEmax[kGains];
      int EBtot[kGains], EEtot[kGains];
      for (int gId = 0; gId < kGains; gId++) {
        barrel_m[gId] = new TH2F(Form("EBm%i", gId),
                                 Form("mean %i EB", gainValues[gId]),
                                 MAX_IPHI,
                                 0,
                                 MAX_IPHI,
                                 2 * MAX_IETA,
                                 -MAX_IETA,
                                 MAX_IETA);
        endc_p_m[gId] = new TH2F(Form("EE+m%i", gId),
                                 Form("mean %i EE+", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        endc_m_m[gId] = new TH2F(Form("EE-m%i", gId),
                                 Form("mean %i EE-", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        barrel_r[gId] = new TH2F(Form("EBr%i", gId),
                                 Form("rms %i EB", gainValues[gId]),
                                 MAX_IPHI,
                                 0,
                                 MAX_IPHI,
                                 2 * MAX_IETA,
                                 -MAX_IETA,
                                 MAX_IETA);
        endc_p_r[gId] = new TH2F(Form("EE+r%i", gId),
                                 Form("rms %i EE+", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        endc_m_r[gId] = new TH2F(Form("EE-r%i", gId),
                                 Form("rms %i EE-", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        EBmean[gId] = 0.;
        EBrms[gId] = 0.;
        EEmean[gId] = 0.;
        EErms[gId] = 0.;
        EBtot[gId] = 0;
        EEtot[gId] = 0;
      }
      auto iov = iovs.front();
      std::shared_ptr<EcalPedestals> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        // looping over the EB channels, via the dense-index, mapped into EBDetId's
        if (payload->barrelItems().empty())
          return false;
        for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {  // loop on EE cells
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          if (payload->find(rawid) == payload->end())
            continue;
          Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
          Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
          if (eta > 0.)
            eta = eta - 0.5;  //   0.5 to 84.5
          else
            eta = eta + 0.5;  //  -84.5 to -0.5
          barrel_m[0]->Fill(phi, eta, (*payload)[rawid].mean_x12);
          Double_t val = (*payload)[rawid].rms_x12;
          barrel_r[0]->Fill(phi, eta, val);
          if (val < 10) {
            EBmean[0] = EBmean[0] + val;
            EBrms[0] = EBrms[0] + val * val;
            EBtot[0]++;
          }
          //	  else std::cout << " gain 12 chan " << cellid << " val " << val << std::endl;
          barrel_m[1]->Fill(phi, eta, (*payload)[rawid].mean_x6);
          val = (*payload)[rawid].rms_x6;
          barrel_r[1]->Fill(phi, eta, val);
          if (val < 10) {
            EBmean[1] = EBmean[1] + val;
            EBrms[1] = EBrms[1] + val * val;
            EBtot[1]++;
          }
          //	  else std::cout << " gain 6 chan " << cellid << " val " << val << std::endl;
          barrel_m[2]->Fill(phi, eta, (*payload)[rawid].mean_x1);
          val = (*payload)[rawid].rms_x1;
          barrel_r[2]->Fill(phi, eta, val);
          if (val < 10) {
            EBmean[2] = EBmean[2] + val;
            EBrms[2] = EBrms[2] + val * val;
            EBtot[2]++;
          }
          //	  else std::cout << " gain 1 chan " << cellid << " val " << val << std::endl;
        }  // loop over cellid
        if (payload->endcapItems().empty())
          return false;

        // looping over the EE channels
        for (int iz = -1; iz < 2; iz = iz + 2)  // -1 or +1
          for (int iy = IY_MIN; iy < IY_MAX + IY_MIN; iy++)
            for (int ix = IX_MIN; ix < IX_MAX + IX_MIN; ix++)
              if (EEDetId::validDetId(ix, iy, iz)) {
                EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                uint32_t rawid = myEEId.rawId();
                if (payload->find(rawid) == payload->end())
                  continue;
                if (iz == 1) {
                  endc_p_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12);
                  Double_t val = (*payload)[rawid].rms_x12;
                  endc_p_r[0]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[0] = EEmean[0] + val;
                    EErms[0] = EErms[0] + val * val;
                    EEtot[0]++;
                  }
                  endc_p_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6);
                  val = (*payload)[rawid].rms_x6;
                  endc_p_r[1]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[1] = EEmean[1] + val;
                    EErms[1] = EErms[1] + val * val;
                    EEtot[1]++;
                  }
                  endc_p_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1);
                  val = (*payload)[rawid].rms_x1;
                  endc_p_r[2]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[2] = EEmean[2] + val;
                    EErms[2] = EErms[2] + val * val;
                    EEtot[2]++;
                  }
                } else {
                  endc_m_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12);
                  Double_t val = (*payload)[rawid].rms_x12;
                  endc_m_r[0]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[0] = EEmean[0] + val;
                    EErms[0] = EErms[0] + val * val;
                    EEtot[0]++;
                  }
                  endc_m_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6);
                  val = (*payload)[rawid].rms_x6;
                  endc_m_r[1]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[1] = EEmean[1] + val;
                    EErms[1] = EErms[1] + val * val;
                    EEtot[1]++;
                  }
                  endc_m_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1);
                  val = (*payload)[rawid].rms_x1;
                  endc_m_r[2]->Fill(ix, iy, val);
                  if (val < 10) {
                    EEmean[2] = EEmean[2] + val;
                    EErms[2] = EErms[2] + val * val;
                    EEtot[2]++;
                  }
                }
              }  // validDetId
      }          // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i", run));

      float xmi[3] = {0.0, 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[6];
      int view = 0;
      for (int gId = 0; gId < kGains; gId++) {
        for (int val = 0; val < 2; val++) {  //  mean and sigma
          pad[view] = new TPad*[3];
          for (int obj = 0; obj < 3; obj++) {
            float yma = 0.94 - (0.16 * view);
            float ymi = yma - 0.14;
            pad[view][obj] =
                new TPad(Form("p_%i_%i", obj, view), Form("p_%i_%i", obj, view), xmi[obj], ymi, xma[obj], yma);
            pad[view][obj]->Draw();
          }
          view++;
        }
        double vt = (double)EBtot[gId];
        EBmean[gId] = EBmean[gId] / vt;
        EBrms[gId] = (EBrms[gId] / vt) - (EBmean[gId] * EBmean[gId]);
        EBrms[gId] = sqrt(EBrms[gId]);
        if (EBrms[gId] == 0.)
          EBrms[gId] = 0.001;
        pEBmin[gId] = EBmean[gId] - kRMS * EBrms[gId];
        pEBmax[gId] = EBmean[gId] + kRMS * EBrms[gId];
        //	std::cout << " mean " << EBmean[gId] << " rms " << EBrms[gId] << " entries " << EBtot[gId] << " min " << pEBmin[gId]
        //		  << " max " << pEBmax[gId] << std::endl;
        if (pEBmin[gId] < 0.)
          pEBmin[gId] = 0.;
        vt = (double)EEtot[gId];
        EEmean[gId] = EEmean[gId] / vt;
        EErms[gId] = (EErms[gId] / vt) - (EEmean[gId] * EEmean[gId]);
        EErms[gId] = sqrt(EErms[gId]);
        if (EErms[gId] == 0.)
          EErms[gId] = 0.001;
        pEEmin[gId] = EEmean[gId] - kRMS * EErms[gId];
        pEEmax[gId] = EEmean[gId] + kRMS * EErms[gId];
        //	std::cout << " mean " << EEmean[gId] << " rms " << EErms[gId] << " entries " << EEtot[gId] << " min " << pEEmin[gId]
        //		  << " max " << pEEmax[gId] << std::endl;
        if (pEEmin[gId] < 0.)
          pEEmin[gId] = 0.;
      }
      //float bmin[kGains] ={0.7, 0.5, 0.4};
      //float bmax[kGains] ={2.2, 1.3, 0.7};
      //float emin[kGains] ={1.5, 0.8, 0.4};
      //float emax[kGains] ={2.5, 1.5, 0.8};
      //      TLine* l = new TLine(0., 0., 0., 0.);
      //      l->SetLineWidth(1);
      for (int gId = 0; gId < kGains; gId++) {
        pad[gId][0]->cd();
        DrawEE(endc_m_m[gId], 175., 225.);
        pad[gId + kGains][0]->cd();
        DrawEE(endc_m_r[gId], pEEmin[gId], pEEmax[gId]);
        pad[gId][1]->cd();
        DrawEB(barrel_m[gId], 175., 225.);
        pad[gId + kGains][1]->cd();
        DrawEB(barrel_r[gId], pEBmin[gId], pEBmax[gId]);
        pad[gId][2]->cd();
        DrawEE(endc_p_m[gId], 175., 225.);
        pad[gId + kGains][2]->cd();
        DrawEE(endc_p_r[gId], pEEmin[gId], pEEmax[gId]);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method

  };  // class EcalPedestalsPlot

  /************************************************************
     2d plot of ECAL pedestals difference between 2 IOVs
  *************************************************************/
  class EcalPedestalsDiff : public cond::payloadInspector::PlotImage<EcalPedestals> {
  public:
    EcalPedestalsDiff() : cond::payloadInspector::PlotImage<EcalPedestals>("ECAL Barrel channel status difference") {
      setSingleIov(false);
      setTwoTags(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      uint32_t gainValues[kGains] = {12, 6, 1};
      TH2F** barrel_m = new TH2F*[kGains];
      TH2F** endc_p_m = new TH2F*[kGains];
      TH2F** endc_m_m = new TH2F*[kGains];
      TH2F** barrel_r = new TH2F*[kGains];
      TH2F** endc_p_r = new TH2F*[kGains];
      TH2F** endc_m_r = new TH2F*[kGains];
      double EBmean[kGains], EBrms[kGains], EEmean[kGains], EErms[kGains], pEBmin[kGains], pEBmax[kGains],
          pEEmin[kGains], pEEmax[kGains];
      int EBtot[kGains], EEtot[kGains];
      for (int gId = 0; gId < kGains; gId++) {
        barrel_m[gId] = new TH2F(Form("EBm%i", gId),
                                 Form("mean %i EB", gainValues[gId]),
                                 MAX_IPHI,
                                 0,
                                 MAX_IPHI,
                                 2 * MAX_IETA,
                                 -MAX_IETA,
                                 MAX_IETA);
        endc_p_m[gId] = new TH2F(Form("EE+m%i", gId),
                                 Form("mean %i EE+", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        endc_m_m[gId] = new TH2F(Form("EE-m%i", gId),
                                 Form("mean %i EE-", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        barrel_r[gId] = new TH2F(Form("EBr%i", gId),
                                 Form("rms %i EB", gainValues[gId]),
                                 MAX_IPHI,
                                 0,
                                 MAX_IPHI,
                                 2 * MAX_IETA,
                                 -MAX_IETA,
                                 MAX_IETA);
        endc_p_r[gId] = new TH2F(Form("EE+r%i", gId),
                                 Form("rms %i EE+", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        endc_m_r[gId] = new TH2F(Form("EE-r%i", gId),
                                 Form("rms %i EE-", gainValues[gId]),
                                 IX_MAX,
                                 IX_MIN,
                                 IX_MAX + 1,
                                 IY_MAX,
                                 IY_MIN,
                                 IY_MAX + 1);
        EBmean[gId] = 0.;
        EBrms[gId] = 0.;
        EEmean[gId] = 0.;
        EErms[gId] = 0.;
        EBtot[gId] = 0;
        EEtot[gId] = 0;
      }
      unsigned int run[2], irun = 0;
      //      unsigned int irun = 0;
      double meanEB[kGains][kEBChannels], rmsEB[kGains][kEBChannels], meanEE[kGains][kEEChannels],
          rmsEE[kGains][kEEChannels];
      for (auto const& iov : iovs) {
        std::shared_ptr<EcalPedestals> payload = fetchPayload(std::get<1>(iov));
        run[irun] = std::get<0>(iov);
        //	std::cout << "run " << irun << " : " << run[irun] << std::endl;
        if (payload.get()) {
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            if (payload->find(rawid) == payload->end())
              continue;

            if (irun == 0) {
              meanEB[0][cellid] = (*payload)[rawid].mean_x12;
              rmsEB[0][cellid] = (*payload)[rawid].rms_x12;
              meanEB[1][cellid] = (*payload)[rawid].mean_x6;
              rmsEB[1][cellid] = (*payload)[rawid].rms_x6;
              meanEB[2][cellid] = (*payload)[rawid].mean_x1;
              rmsEB[2][cellid] = (*payload)[rawid].rms_x1;
            } else {
              Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
              Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
              if (eta > 0.)
                eta = eta - 0.5;  //   0.5 to 84.5
              else
                eta = eta + 0.5;  //  -84.5 to -0.5
              barrel_m[0]->Fill(phi, eta, (*payload)[rawid].mean_x12 - meanEB[0][cellid]);
              double diff = (*payload)[rawid].rms_x12 - rmsEB[0][cellid];
              barrel_r[0]->Fill(phi, eta, diff);
              if (std::abs(diff) < 1.) {
                EBmean[0] = EBmean[0] + diff;
                EBrms[0] = EBrms[0] + diff * diff;
                EBtot[0]++;
              }
              //	      else std::cout << " gain 12 chan " << cellid << " diff " << diff << std::endl;
              barrel_m[1]->Fill(phi, eta, (*payload)[rawid].mean_x6 - meanEB[1][cellid]);
              diff = (*payload)[rawid].rms_x6 - rmsEB[1][cellid];
              barrel_r[1]->Fill(phi, eta, diff);
              if (std::abs(diff) < 1.) {
                EBmean[1] = EBmean[1] + diff;
                EBrms[1] = EBrms[1] + diff * diff;
                EBtot[1]++;
              }
              //	      else std::cout << " gain 6 chan " << cellid << " diff " << diff << std::endl;
              barrel_m[2]->Fill(phi, eta, (*payload)[rawid].mean_x1 - meanEB[2][cellid]);
              diff = (*payload)[rawid].rms_x1 - rmsEB[2][cellid];
              barrel_r[2]->Fill(phi, eta, diff);
              if (std::abs(diff) < 1.) {
                EBmean[2] = EBmean[2] + diff;
                EBrms[2] = EBrms[2] + diff * diff;
                EBtot[2]++;
              }
              //	      else std::cout << " gain 1 chan " << cellid << " diff " << diff << std::endl;
            }
          }  // loop over cellid

          if (payload->endcapItems().empty())
            return false;
          // looping over the EE channels
          for (int iz = -1; iz < 2; iz = iz + 2) {  // -1 or +1
            for (int iy = IY_MIN; iy < IY_MAX + IY_MIN; iy++) {
              for (int ix = IX_MIN; ix < IX_MAX + IX_MIN; ix++) {
                if (EEDetId::validDetId(ix, iy, iz)) {
                  EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                  uint32_t rawid = myEEId.rawId();
                  uint32_t index = myEEId.hashedIndex();
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (irun == 0) {
                    meanEE[0][index] = (*payload)[rawid].mean_x12;
                    rmsEE[0][index] = (*payload)[rawid].rms_x12;
                    meanEE[1][index] = (*payload)[rawid].mean_x6;
                    rmsEE[1][index] = (*payload)[rawid].rms_x6;
                    meanEE[2][index] = (*payload)[rawid].mean_x1;
                    rmsEE[2][index] = (*payload)[rawid].rms_x1;
                  }  // fist run
                  else {
                    if (iz == 1) {
                      endc_p_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12 - meanEE[0][index]);
                      double diff = (*payload)[rawid].rms_x12 - rmsEE[0][index];
                      endc_p_r[0]->Fill(ix, iy, rmsEE[0][index] - (*payload)[rawid].rms_x12);
                      if (std::abs(diff) < 1.) {
                        EEmean[0] = EEmean[0] + diff;
                        EErms[0] = EErms[0] + diff * diff;
                        EEtot[0]++;
                      }
                      //else std::cout << " gain 12 chan " << index << " diff " << diff << std::endl;
                      endc_p_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6 - meanEE[1][index]);
                      diff = (*payload)[rawid].rms_x6 - rmsEE[1][index];
                      endc_p_r[1]->Fill(ix, iy, diff);
                      if (std::abs(diff) < 1.) {
                        EEmean[1] = EEmean[1] + diff;
                        EErms[1] = EErms[1] + diff * diff;
                        EEtot[1]++;
                      }
                      //else std::cout << " gain 6 chan " << index << " diff " << diff << std::endl;
                      endc_p_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1 - meanEE[2][index]);
                      diff = (*payload)[rawid].rms_x1 - rmsEE[2][index];
                      endc_p_r[2]->Fill(ix, iy, diff);
                      if (std::abs(diff) < 1.) {
                        EEmean[2] = EEmean[2] + diff;
                        EErms[2] = EErms[2] + diff * diff;
                        EEtot[2]++;
                      }
                      //else std::cout << " gain 1 chan " << index << " diff " << diff << std::endl;
                    }  // EE+
                    else {
                      endc_m_m[0]->Fill(ix, iy, (*payload)[rawid].mean_x12 - meanEE[0][index]);
                      double diff = (*payload)[rawid].rms_x12 - rmsEE[0][index];
                      endc_m_r[0]->Fill(ix, iy, rmsEE[0][index] - (*payload)[rawid].rms_x12);
                      if (std::abs(diff) < 1.) {
                        EEmean[0] = EEmean[0] + diff;
                        EErms[0] = EErms[0] + diff * diff;
                        EEtot[0]++;
                      }
                      //else std::cout << " gain 12 chan " << index << " diff " << diff << std::endl;
                      endc_m_m[1]->Fill(ix, iy, (*payload)[rawid].mean_x6 - meanEE[1][index]);
                      diff = (*payload)[rawid].rms_x6 - rmsEE[1][index];
                      endc_m_r[1]->Fill(ix, iy, diff);
                      if (std::abs(diff) < 1.) {
                        EEmean[1] = EEmean[1] + diff;
                        EErms[1] = EErms[1] + diff * diff;
                        EEtot[1]++;
                      }
                      //else std::cout << " gain 6 chan " << index << " diff " << diff << std::endl;
                      endc_m_m[2]->Fill(ix, iy, (*payload)[rawid].mean_x1 - meanEE[2][index]);
                      diff = (*payload)[rawid].rms_x1 - rmsEE[2][index];
                      endc_m_r[2]->Fill(ix, iy, diff);
                      if (std::abs(diff) < 1.) {
                        EEmean[2] = EEmean[2] + diff;
                        EErms[2] = EErms[2] + diff * diff;
                        EEtot[2]++;
                      }
                      //else std::cout << " gain 1 chan " << index << " diff " << diff << std::endl;
                    }  // EE-
                  }    // second run
                }      // validDetId
              }        //   loop over ix
            }          //  loop over iy
          }            //  loop over iz
        }              //  if payload.get()
        else
          return false;
        irun++;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1600, 2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals, IOV %i - %i", run[1], run[0]));

      float xmi[3] = {0.0, 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[6];
      int view = 0;
      for (int gId = 0; gId < kGains; gId++) {
        for (int val = 0; val < 2; val++) {  //  mean and sigma
          pad[view] = new TPad*[3];
          for (int obj = 0; obj < 3; obj++) {
            float yma = 0.94 - (0.16 * view);
            float ymi = yma - 0.14;
            pad[view][obj] =
                new TPad(Form("p_%i_%i", obj, view), Form("p_%i_%i", obj, view), xmi[obj], ymi, xma[obj], yma);
            pad[view][obj]->Draw();
          }
          view++;
        }
        double vt = (double)EBtot[gId];
        EBmean[gId] = EBmean[gId] / vt;
        EBrms[gId] = (EBrms[gId] / vt) - (EBmean[gId] * EBmean[gId]);
        EBrms[gId] = sqrt(EBrms[gId]);
        if (EBrms[gId] == 0.)
          EBrms[gId] = 0.001;
        pEBmin[gId] = EBmean[gId] - kRMS * EBrms[gId];
        pEBmax[gId] = EBmean[gId] + kRMS * EBrms[gId];
        //	std::cout << " mean " << EBmean[gId] << " rms " << EBrms[gId] << " entries " << EBtot[gId] << " min " << pEBmin[gId]
        //		  << " max " << pEBmax[gId] << std::endl;
        vt = (double)EEtot[gId];
        EEmean[gId] = EEmean[gId] / vt;
        EErms[gId] = (EErms[gId] / vt) - (EEmean[gId] * EEmean[gId]);
        EErms[gId] = sqrt(EErms[gId]);
        if (EErms[gId] == 0.)
          EErms[gId] = 0.001;
        pEEmin[gId] = EEmean[gId] - kRMS * EErms[gId];
        pEEmax[gId] = EEmean[gId] + kRMS * EErms[gId];
        //	std::cout << " mean " << EEmean[gId] << " rms " << EErms[gId] << " entries " << EEtot[gId] << " min " << pEEmin[gId]
        //		  << " max " << pEEmax[gId] << std::endl;
      }
      for (int gId = 0; gId < kGains; gId++) {
        pad[gId][0]->cd();
        DrawEE(endc_m_m[gId], -2., 2.);
        pad[gId + kGains][0]->cd();
        DrawEE(endc_m_r[gId], pEEmin[gId], pEEmax[gId]);
        pad[gId][1]->cd();
        DrawEB(barrel_m[gId], -2., 2.);
        pad[gId + kGains][1]->cd();
        DrawEB(barrel_r[gId], pEBmin[gId], pEBmax[gId]);
        pad[gId][2]->cd();
        DrawEE(endc_p_m[gId], -2., 2.);
        pad[gId + kGains][2]->cd();
        DrawEE(endc_p_r[gId], pEEmin[gId], pEEmax[gId]);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalPedestalsDiff

  /*************************************************  
     2d histogram of ECAL barrel pedestal of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram2D
  class EcalPedestalsEBMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBMean12Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel pedestal gain12 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -1 * MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12)
              continue;

            // there's no ieta==0 in the EB numbering
            //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
            // fill the Histogram2D here
            //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
            // set min and max on 2d plots
            float valped = (*payload)[rawid].mean_x12;
            if (valped > 250.)
              valped = 250.;
            //	    if(valped < 150.) valped = 150.;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEBMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBMean6Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel pedestal gain6 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6)
              continue;

            // set min and max on 2d plots
            float valped = (*payload)[rawid].mean_x6;
            if (valped > 250.)
              valped = 250.;
            //	    if(valped < 150.) valped = 150.;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEBMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBMean1Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel pedestal gain1 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1)
              continue;

            // set min and max on 2d plots
            float valped = (*payload)[rawid].mean_x1;
            if (valped > 250.)
              valped = 250.;
            //	    if(valped < 150.) valped = 150.;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEEMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEEMean12Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap pedestal gain12 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12)
                    continue;
                  // set min and max on 2d plots
                  float valped = (*payload)[rawid].mean_x12;
                  if (valped > 250.)
                    valped = 250.;
                  //		    if(valped < 150.) valped = 150.;
                  if (iz == -1)
                    fillWithValue(ix, iy, valped);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valped);

                }  // validDetId
        }          // payload
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEEMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEEMean6Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap pedestal gain6 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6)
                    continue;
                  // set min and max on 2d plots
                  float valped = (*payload)[rawid].mean_x6;
                  if (valped > 250.)
                    valped = 250.;
                  //		    if(valped < 150.) valped = 150.;
                  if (iz == -1)
                    fillWithValue(ix, iy, valped);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valped);
                }  // validDetId
        }          // payload
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEEMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEEMean1Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap pedestal gain1 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12)
                    continue;
                  // set min and max on 2d plots
                  float valped = (*payload)[rawid].mean_x1;
                  if (valped > 250.)
                    valped = 250.;
                  //		    if(valped < 150.) valped = 150.;
                  if (iz == -1)
                    fillWithValue(ix, iy, valped);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valped);
                }  // validDetId
        }          // if payload.get()
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEBRMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBRMS12Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel noise gain12 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -1 * MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12)
              continue;

            // there's no ieta==0 in the EB numbering
            //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
            // fill the Histogram2D here
            //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
            // set max on noise 2d plots
            float valrms = (*payload)[rawid].rms_x12;
            if (valrms > 2.2)
              valrms = 2.2;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEBRMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBRMS6Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel noise gain6 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -1 * MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6)
              continue;

            // set max on noise 2d plots
            float valrms = (*payload)[rawid].rms_x6;
            if (valrms > 1.5)
              valrms = 1.5;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEBRMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEBRMS1Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Barrel noise gain1 - map",
                                                             "iphi",
                                                             MAX_IPHI,
                                                             MIN_IPHI,
                                                             MAX_IPHI + MIN_IPHI,
                                                             "ieta",
                                                             2 * MAX_IETA + 1,
                                                             -1 * MAX_IETA,
                                                             MAX_IETA + 1) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty())
            return false;
          for (int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL pedestal, for a given ECAL barrel channel
            if (payload->find(rawid) == payload->end())
              continue;
            if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1)
              continue;

            // set max on noise 2d plots
            float valrms = (*payload)[rawid].rms_x1;
            if (valrms > 1.0)
              valrms = 1.0;
            fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
          }  // loop over cellid
        }    // if payload.get()
      }      // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEERMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEERMS12Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap noise gain12 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12)
                    continue;
                  // set max on noise 2d plots
                  float valrms = (*payload)[rawid].rms_x12;
                  if (valrms > 3.5)
                    valrms = 3.5;
                  if (iz == -1)
                    fillWithValue(ix, iy, valrms);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valrms);

                }  // validDetId
        }          // payload
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEERMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEERMS6Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap noise gain6 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6)
                    continue;
                  // set max on noise 2d plots
                  float valrms = (*payload)[rawid].rms_x6;
                  if (valrms > 2.0)
                    valrms = 2.0;
                  if (iz == -1)
                    fillWithValue(ix, iy, valrms);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valrms);
                }  // validDetId
        }          // payload
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  class EcalPedestalsEERMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {
  public:
    EcalPedestalsEERMS1Map()
        : cond::payloadInspector::Histogram2D<EcalPedestals>("ECAL Endcap noise gain1 - map",
                                                             "ix",
                                                             2.2 * IX_MAX,
                                                             IX_MIN,
                                                             2.2 * IX_MAX + 1,
                                                             "iy",
                                                             IY_MAX,
                                                             IY_MIN,
                                                             IY_MAX + IY_MIN) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<EcalPedestals> payload = Base::fetchPayload(std::get<1>(iov));
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
                  // check the existence of ECAL pedestal, for a given ECAL endcap channel
                  if (payload->find(rawid) == payload->end())
                    continue;
                  if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12)
                    continue;
                  // set max on noise 2d plots
                  float valrms = (*payload)[rawid].rms_x1;
                  if (valrms > 1.5)
                    valrms = 1.5;
                  if (iz == -1)
                    fillWithValue(ix, iy, valrms);
                  else
                    fillWithValue(ix + IX_MAX + 20, iy, valrms);
                }  // validDetId
        }          // if payload.get()
      }            // loop over IOV's (1 in this case)
      return true;
    }  // fill method
  };

  /*****************************************
 2d plot of Ecal Pedestals Summary of 1 IOV
 ******************************************/
  class EcalPedestalsSummaryPlot : public cond::payloadInspector::PlotImage<EcalPedestals> {
  public:
    EcalPedestalsSummaryPlot() : cond::payloadInspector::PlotImage<EcalPedestals>("Ecal Pedestals Summary - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<EcalPedestals> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F* align;                          //pointer to align which is a 2D histogram

      int NbRows = 6;
      int NbColumns = 5;

      if (payload.get()) {  //payload is an iov retrieved from payload using hash.

        align = new TH2F("Ecal Pedestals Summary",
                         "EB/EE      Gain                   Mean               RMS            Total Items",
                         NbColumns,
                         0,
                         NbColumns,
                         NbRows,
                         0,
                         NbRows);

        float ebVals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float eeVals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        long unsigned int ebTotal = (payload->barrelItems()).size();
        long unsigned int eeTotal = (payload->endcapItems()).size();

        pedestalsSummary(payload->barrelItems(), ebVals, ebTotal);
        pedestalsSummary(payload->endcapItems(), eeVals, eeTotal);

        double row = NbRows - 0.5;

        //EB summary values
        align->Fill(1.5, row, 12);         //Gain
        align->Fill(2.5, row, ebVals[0]);  //mean12
        align->Fill(3.5, row, ebVals[1]);  //rms12

        row--;

        align->Fill(0.5, row, 1);
        align->Fill(1.5, row, 6);          //Gain
        align->Fill(2.5, row, ebVals[2]);  //mean6
        align->Fill(3.5, row, ebVals[3]);  //rms6
        align->Fill(4.5, row, ebTotal);

        row--;

        align->Fill(1.5, row, 1);          //Gain
        align->Fill(2.5, row, ebVals[4]);  //mean1
        align->Fill(3.5, row, ebVals[5]);  //rms1

        row--;

        //EE summary values
        align->Fill(1.5, row, 12);         //Gain
        align->Fill(2.5, row, eeVals[0]);  //mean12
        align->Fill(3.5, row, eeVals[1]);  //rms12

        row--;

        align->Fill(0.5, row, 2);
        align->Fill(1.5, row, 6);          //Gain
        align->Fill(2.5, row, eeVals[2]);  //mean6
        align->Fill(3.5, row, eeVals[3]);  //rms6
        align->Fill(4.5, row, eeTotal);

        row--;

        align->Fill(1.5, row, 1);          //Gain
        align->Fill(2.5, row, eeVals[4]);  //mean1
        align->Fill(3.5, row, eeVals[5]);  //rms1

      }  // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1000, 1000);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Pedestals Summary, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");
      TLine* l = new TLine;
      l->SetLineWidth(1);

      for (int i = 1; i < NbRows; i++) {
        double y = (double)i;
        if (i == NbRows / 2)
          l = new TLine(0., y, NbColumns, y);
        else
          l = new TLine(1., y, NbColumns - 1, y);
        l->Draw();
      }

      for (int i = 1; i < NbColumns; i++) {
        double x = (double)i;
        double y = (double)NbRows;
        l = new TLine(x, 0., x, y);
        l->Draw();
      }

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method

    void pedestalsSummary(std::vector<EcalPedestal> vItems, float vals[], long unsigned int& total) {
      for (std::vector<EcalPedestal>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems) {
        //vals[0]=100;
        vals[0] += iItems->mean(1);  //G12
        vals[1] += iItems->rms(1);
        vals[2] += iItems->mean(2);  //G6
        vals[3] += iItems->rms(2);
        vals[4] += iItems->mean(3);  //G1
        vals[5] += iItems->rms(3);
      }

      vals[0] = vals[0] / total;
      vals[1] = vals[1] / total;
      vals[2] = vals[2] / total;
      vals[3] = vals[3] / total;
      vals[4] = vals[4] / total;
      vals[5] = vals[5] / total;
    }
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalPedestals) {
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsHist);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsDiff);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBMean12Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBMean6Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBMean1Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEEMean12Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEEMean6Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEEMean1Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBRMS12Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBRMS6Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEBRMS1Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEERMS12Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEERMS6Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsEERMS1Map);
  PAYLOAD_INSPECTOR_CLASS(EcalPedestalsSummaryPlot);
}
