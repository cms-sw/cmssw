#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum { kEBTotalTowers = 2448 };
  enum { MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 17, MAX_IPHI = 72 };  // barrel lower and upper bounds on eta and phi

  /***********************************************
     2d plot of ECAL TPGTowerStatus of 1 IOV
  ************************************************/
  class EcalTPGTowerStatusPlot : public cond::payloadInspector::PlotImage<EcalTPGTowerStatus> {
  public:
    EcalTPGTowerStatusPlot() : cond::payloadInspector::PlotImage<EcalTPGTowerStatus>("ECAL TPGTowerStatus - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* barrel = new TH2F("EB", "EB TPG Tower Status", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      int EBstat = 0;

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGTowerStatus> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        const EcalTPGTowerStatusMap& towerMap = (*payload).getMap();
        //	std::cout << " tower map size " << towerMap.size() << std::endl;
        EcalTPGTowerStatusMapIterator it;
        for (it = towerMap.begin(); it != towerMap.end(); ++it) {
          if ((*it).second > 0) {
            EcalTrigTowerDetId ttId((*it).first);
            int ieta = ttId.ieta();
            //	    if(ieta < 0) ieta--;   // -1 to -17
            int iphi = ttId.iphi() - 1;  // 0 to 71
            //	    std::cout << " sub det " << ttId.subDet() << " phi " << iphi << " eta " << ieta << std::endl;
            // ieta goes from -18 to -2 and 1 to 17. Change it to -17/-1 and 0/16
            ieta--;
            if (ttId.subDet() == 1) {  // barrel
              barrel->Fill(iphi, ieta, (*it).second);
              EBstat++;
            }
          }
        }
      }  // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      //      TCanvas canvas("CC map","CC map", 1600, 450);
      Double_t w = 1400;
      Double_t h = 1200;
      TCanvas canvas("c", "c", w, h);
      //      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGTowerStatus, IOV %i", run));

      TPad** pad = new TPad*;
      for (int obj = 0; obj < 1; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), 0.0, 0.04, 1.0, 0.94);
        pad[obj]->Draw();
      }
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.2, 0.88, Form("%i towers", EBstat));
      //      canvas.cd();
      pad[0]->cd();
      //      barrel->SetStats(false);
      barrel->Draw("col");
      TLine* l = new TLine(0., 0., 0., 0.);
      l->SetLineWidth(1);
      for (int i = 0; i < MAX_IETA; i++) {
        Double_t x = 4. + (i * 4);
        l = new TLine(x, -MAX_IETA, x, MAX_IETA);
        l->Draw();
      }
      l = new TLine(0., 0., 72., 0.);
      l->Draw();

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /************************************************************************
     2d plot of ECAL TPGTowerStatus difference between 2 IOVs
  ************************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalTPGTowerStatusDiffBase : public cond::payloadInspector::PlotImage<EcalTPGTowerStatus, nIOVs, ntags> {
  public:
    EcalTPGTowerStatusDiffBase()
        : cond::payloadInspector::PlotImage<EcalTPGTowerStatus, nIOVs, ntags>("ECAL TPGTowerStatus difference") {}

    bool fill() override {
      TH2F* barrel = new TH2F("EB", "EB difference", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      //      int EBstat[2] = {0, 0};
      int EBstat = 0;

      unsigned int run[2] = {0, 0}, vEB[kEBTotalTowers];
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
        std::shared_ptr<EcalTPGTowerStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          const EcalTPGTowerStatusMap& towerMap = (*payload).getMap();
          //	  std::cout << " tower map size " << towerMap.size() << std::endl;
          EcalTPGTowerStatusMapIterator it;
          for (it = towerMap.begin(); it != towerMap.end(); ++it) {
            EcalTrigTowerDetId ttId((*it).first);
            int ieta = ttId.ieta();
            if (ieta < 0)
              ieta--;                    // 1 to 17
            int iphi = ttId.iphi() - 1;  // 0 to 71
            int towerId = ttId.hashedIndex();
            int stat = (*it).second;
            if (irun == 0) {
              if (ttId.subDet() == 1) {  // barrel
                vEB[towerId] = stat;
                if (stat > 0) {  // bad tower
                  if (towerId >= kEBTotalTowers)
                    std::cout << " strange tower Id " << towerId << std::endl;
                }
              }                          // barrel
            }                            // 1st run
            else {                       // 2nd run
              if (ttId.subDet() == 1) {  // barrel
                if (stat > 0) {          // bad tower
                  if (towerId >= kEBTotalTowers)
                    std::cout << " strange tower Id " << towerId << std::endl;
                  //		  else std::cout << " phi " << iphi << " eta " << ieta << std::endl;
                }  //  bad tower
                int diff = stat - vEB[towerId];
                // ieta goes from -18 to -2 and 1 to 17. Change it to -17/-1 and 0/16
                ieta--;
                if (diff != 0)
                  barrel->Fill(iphi, ieta, diff);
                //		  vEB[towerId] = 0;
                EBstat += diff;
              }  // barrel
            }    // 2nd run
          }      // loop over towers
        }        // payload
        else
          return false;
      }  // loop over IOVs
      /*
      // now check if towers have disappered
      for(int it = 0; it < kEBTotalTowers; it++) {
	if(vEB[it] != 0) {
	  std::cout << " tower " << vEB[it] << " not found in run 2, plot it" << std::endl;
	  EcalTrigTowerDetId ttId = EBId[it];
	  int ieta = ttId.ieta();
	  // ieta goes from -18 to -2 and 1 to 17. Change it to -17/-1 and 0/16
	  ieta--;
	  int iphi = ttId.iphi() - 1;  // 0 to 71
	  barrel->Fill(iphi, ieta, vEB[it]);
	}
      }
      */
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      Double_t w = 1400;
      Double_t h = 1200;
      TCanvas canvas("c", "c", w, h);
      //      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2) {
        if (len < 80) {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("%s %i - %s %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.05);
          t1.DrawLatex(0.5, 0.96, Form("Ecal TPGTowerStatus, IOV %i - %i", run[1], run[0]));
        }
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }
      t1.SetTextSize(0.05);

      TPad** pad = new TPad*;
      for (int obj = 0; obj < 1; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), 0.0, 0.04, 1.0, 0.94);
        pad[obj]->Draw();
      }
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.2, 0.88, Form("%i tower(s)", EBstat));

      pad[0]->cd();
      //      barrel->SetStats(false);
      barrel->Draw("colz");
      TLine* l = new TLine(0., 0., 0., 0.);
      l->SetLineWidth(1);
      for (int i = 0; i < MAX_IETA; i++) {
        Double_t x = 4. + (i * 4);
        l = new TLine(x, -MAX_IETA, x, MAX_IETA);
        l->Draw();
      }
      l = new TLine(0., 0., 72., 0.);
      l->Draw();

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalTPGTowerStatusDiffBase
  using EcalTPGTowerStatusDiffOneTag = EcalTPGTowerStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalTPGTowerStatusDiffTwoTags = EcalTPGTowerStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

  /*******************************************************
    2d plot of EcalTPGTowerStatus Error Summary of 1 IOV
  ********************************************************/
  class EcalTPGTowerStatusSummaryPlot : public cond::payloadInspector::PlotImage<EcalTPGTowerStatus> {
  public:
    EcalTPGTowerStatusSummaryPlot()
        : cond::payloadInspector::PlotImage<EcalTPGTowerStatus>("Ecal TPGTower Status Summary - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<EcalTPGTowerStatus> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F* align;                          //pointer to align which is a 2D histogram

      int NbRows = 1;
      int NbColumns = 2;

      if (payload.get()) {  //payload is an iov retrieved from payload using hash.
        const EcalTPGTowerStatusMap& towerMap = (*payload).getMap();

        align = new TH2F("Ecal TPGTower Status Summary",
                         "Total                NumberOfMasked",
                         NbColumns,
                         0,
                         NbColumns,
                         NbRows,
                         0,
                         NbRows);

        int NbMaskedTT = 0;

        for (EcalTPGTowerStatusMapIterator it = towerMap.begin(); it != towerMap.end(); ++it)
          if ((*it).second > 0)
            NbMaskedTT++;

        align->Fill(0.5, 0.5, towerMap.size());
        align->Fill(1.5, 0.5, NbMaskedTT);

      }  // if payload.get()
      else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1000, 1000);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.04);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("Barrel masked Trigger Towers, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
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
PAYLOAD_INSPECTOR_MODULE(EcalTPGTowerStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGTowerStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGTowerStatusDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGTowerStatusDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGTowerStatusSummaryPlot);
}
