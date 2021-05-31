#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum { NTCC = 108, NTower = 28, NStrip = 5, NXtal = 5 };
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };  // endcaps lower and upper bounds on x and y

  /***********************************************
     2d plot of ECAL TPGStripStatus of 1 IOV
  ************************************************/
  class EcalTPGStripStatusPlot : public cond::payloadInspector::PlotImage<EcalTPGStripStatus> {
  public:
    EcalTPGStripStatusPlot() : cond::payloadInspector::PlotImage<EcalTPGStripStatus>("ECAL TPGStripStatus - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F* endc_p = new TH2F("EE+", "EE+ TPG Strip Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE- TPG Strip Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EEstat[2] = {0, 0};

      std::string mappingFile = "Geometry/EcalMapping/data/EEMap.txt";
      std::ifstream f(edm::FileInPath(mappingFile).fullPath().c_str());
      if (!f.good()) {
        std::cout << "EcalTPGStripStatus File EEMap.txt not found" << std::endl;
        throw cms::Exception("FileNotFound");
      }

      uint32_t rawEE[NTCC][NTower][NStrip][NXtal];
      int NbrawEE[NTCC][NTower][NStrip];
      for (int TCC = 0; TCC < NTCC; TCC++)
        for (int TT = 0; TT < NTower; TT++)
          for (int ST = 0; ST < NStrip; ST++)
            NbrawEE[TCC][TT][ST] = 0;
      while (!f.eof()) {
        int ix, iy, iz, CL;
        int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
        int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
        f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip >> tccid >> tower >>
            pseudostrip_in_TCC >> pseudostrip_in_TT;
        EEDetId detid(ix, iy, iz, EEDetId::XYMODE);
        uint32_t rawId = detid.denseIndex();
        if (tccid > NTCC || tower > NTower || pseudostrip_in_TT > NStrip || xtal_in_pseudostrip > NXtal)
          std::cout << " tccid " << tccid << " tower " << tower << " pseudostrip_in_TT " << pseudostrip_in_TT
                    << " xtal_in_pseudostrip " << xtal_in_pseudostrip << std::endl;
        else {
          rawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1][xtal_in_pseudostrip - 1] = rawId;
          NbrawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1]++;
        }
      }  // read EEMap file
      f.close();
      double wei[2] = {0., 0.};

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGStripStatus> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      if (payload.get()) {
        const EcalTPGStripStatusMap& stripMap = (*payload).getMap();
        //	std::cout << " tower map size " << stripMap.size() << std::endl;
        EcalTPGStripStatusMapIterator itSt;
        for (itSt = stripMap.begin(); itSt != stripMap.end(); ++itSt) {
          if (itSt->second > 0) {
            // let's decode the ID
            int strip = itSt->first / 8;
            int pseudostrip = strip & 0x7;
            strip /= 8;
            int tt = strip & 0x7F;
            strip /= 128;
            int tccid = strip & 0x7F;
            int NbXtalInStrip = NbrawEE[tccid - 1][tt - 1][pseudostrip - 1];
            if (NbXtalInStrip != NXtal)
              std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip << " Nx Xtals "
                        << NbXtalInStrip << std::endl;
            //	    std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip
            //						 << " Nx Xtals " << NbXtalInStrip << std::endl;
            for (int Xtal = 0; Xtal < NbXtalInStrip; Xtal++) {
              uint32_t rawId = rawEE[tccid - 1][tt - 1][pseudostrip - 1][Xtal];
              //	std::cout << " rawid " << rawId << std::endl;
              EEDetId detid = EEDetId::detIdFromDenseIndex(rawId);
              float x = (float)detid.ix();
              float y = (float)detid.iy();
              int iz = detid.zside();
              if (iz == -1)
                iz++;
              if (Xtal == 0)
                wei[iz] += 1.;
              if (iz == 0) {
                endc_m->Fill(x + 0.5, y + 0.5, wei[iz]);
                EEstat[0]++;
              } else {
                endc_p->Fill(x + 0.5, y + 0.5, wei[iz]);
                EEstat[1]++;
              }
              //	      std::cout << " x " << x << " y " << y << " z " << iz << std::endl;
            }
          }
        }
      }  // payload
      //      std::cout << " nb strip EE- " << wei[0] << " EE+ " << wei[1] << std::endl;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      const Int_t NRGBs = 5;
      const Int_t NCont = 255;

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);
      //      TCanvas canvas("CC map","CC map", 1600, 450);
      Double_t w = 1200;
      Double_t h = 650;
      TCanvas canvas("c", "c", w, h);
      //      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGStripStatus, IOV %i", run));

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 2; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEE(endc_m, 0., wei[0]);
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEstat[0]));
      pad[1]->cd();
      DrawEE(endc_p, 0., wei[1]);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEstat[1]));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /***************************************************************
     2d plot of ECAL TPGStripStatus difference between 2 IOVs
  ****************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalTPGStripStatusDiffBase : public cond::payloadInspector::PlotImage<EcalTPGStripStatus, nIOVs, ntags> {
  public:
    EcalTPGStripStatusDiffBase()
        : cond::payloadInspector::PlotImage<EcalTPGStripStatus, nIOVs, ntags>("ECAL TPGStripStatus difference") {}

    bool fill() override {
      TH2F* endc_p = new TH2F("EE+", "EE+ TPG Strip Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "EE- TPG Strip Status", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EEstat[2][2] = {{0, 0}, {0, 0}};

      std::string mappingFile = "Geometry/EcalMapping/data/EEMap.txt";
      std::ifstream f(edm::FileInPath(mappingFile).fullPath().c_str());
      if (!f.good()) {
        std::cout << "EcalTPGStripStatus File EEMap.txt not found" << std::endl;
        throw cms::Exception("FileNotFound");
      }

      uint32_t rawEE[NTCC][NTower][NStrip][NXtal];
      int NbrawEE[NTCC][NTower][NStrip];
      for (int TCC = 0; TCC < NTCC; TCC++)
        for (int TT = 0; TT < NTower; TT++)
          for (int ST = 0; ST < NStrip; ST++)
            NbrawEE[TCC][TT][ST] = 0;
      while (!f.eof()) {
        int ix, iy, iz, CL;
        int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
        int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
        f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip >> tccid >> tower >>
            pseudostrip_in_TCC >> pseudostrip_in_TT;
        EEDetId detid(ix, iy, iz, EEDetId::XYMODE);
        uint32_t rawId = detid.denseIndex();
        if (tccid > NTCC || tower > NTower || pseudostrip_in_TT > NStrip || xtal_in_pseudostrip > NXtal)
          std::cout << " tccid " << tccid << " tower " << tower << " pseudostrip_in_TT " << pseudostrip_in_TT
                    << " xtal_in_pseudostrip " << xtal_in_pseudostrip << std::endl;
        else {
          rawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1][xtal_in_pseudostrip - 1] = rawId;
          NbrawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1]++;
        }
      }  // read EEMap file
      f.close();

      unsigned int run[2] = {0, 0};
      int vEE[100];
      int istat = 0;
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
        std::shared_ptr<EcalTPGStripStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          const EcalTPGStripStatusMap& stripMap = (*payload).getMap();
          //	  std::cout << " tower map size " << stripMap.size() << std::endl;
          EcalTPGStripStatusMapIterator itSt;
          for (itSt = stripMap.begin(); itSt != stripMap.end(); ++itSt) {
            if (itSt->second > 0) {
              int ID = itSt->first / 8;
              if (irun == 0 && istat < 100) {
                vEE[istat] = ID;
                //		std::cout << " strip " << ID << " found in run 1" << std::endl;
                istat++;
                if (istat == 100)
                  std::cout << " limit on number of strips reached, stop keeping others" << std::endl;
              } else {
                bool found = false;
                for (int is = 0; is < istat; is++) {
                  //		  std::cout << " checking " << ID << " against " << vEE[is] << std::endl;
                  if (vEE[is] == ID) {
                    //		    std::cout << " strip " << ID << " already in run 1" << std::endl;
                    found = true;
                    vEE[is] = -1;
                    break;
                  }
                }
                if (!found) {
                  //		  std::cout << " strip " << ID << " new, plot it" << std::endl;
                  // let's decode the ID
                  int strip = ID;
                  int pseudostrip = strip & 0x7;
                  strip /= 8;
                  int tt = strip & 0x7F;
                  strip /= 128;
                  int tccid = strip & 0x7F;
                  int NbXtalInStrip = NbrawEE[tccid - 1][tt - 1][pseudostrip - 1];
                  if (NbXtalInStrip != NXtal)
                    std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip << " Nx Xtals "
                              << NbXtalInStrip << std::endl;
                  for (int Xtal = 0; Xtal < NbXtalInStrip; Xtal++) {
                    uint32_t rawId = rawEE[tccid - 1][tt - 1][pseudostrip - 1][Xtal];
                    //	std::cout << " rawid " << rawId << std::endl;
                    EEDetId detid = EEDetId::detIdFromDenseIndex(rawId);
                    float x = (float)detid.ix();
                    float y = (float)detid.iy();
                    int iz = detid.zside();
                    if (iz == -1)
                      iz++;
                    if (iz == 0) {
                      endc_m->Fill(x + 0.5, y + 0.5, 1.);
                      EEstat[0][0]++;
                    } else {
                      endc_p->Fill(x + 0.5, y + 0.5, 1.);
                      EEstat[1][0]++;
                    }
                    //	      std::cout << " x " << x << " y " << y << " z " << iz << std::endl;
                  }  // loop over crystals in strip
                }    // new strip
              }      // second run
            }
          }  // loop over strips
        }    // payload
        else
          return false;
        //	std::cout << " nb of strips " << istat << std::endl;
      }  // loop over IOVs

      // now check if strips have disappered
      for (int is = 0; is < istat; is++) {
        if (vEE[is] != -1) {
          //	  std::cout << " strip " << vEE[is] << " not found in run 2, plot it" << std::endl;
          // let's decode the ID
          int strip = vEE[is];
          int pseudostrip = strip & 0x7;
          strip /= 8;
          int tt = strip & 0x7F;
          strip /= 128;
          int tccid = strip & 0x7F;
          int NbXtalInStrip = NbrawEE[tccid - 1][tt - 1][pseudostrip - 1];
          if (NbXtalInStrip != NXtal)
            std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip << " Nx Xtals "
                      << NbXtalInStrip << std::endl;
          for (int Xtal = 0; Xtal < NbXtalInStrip; Xtal++) {
            uint32_t rawId = rawEE[tccid - 1][tt - 1][pseudostrip - 1][Xtal];
            //	std::cout << " rawid " << rawId << std::endl;
            EEDetId detid = EEDetId::detIdFromDenseIndex(rawId);
            float x = (float)detid.ix();
            float y = (float)detid.iy();
            int iz = detid.zside();
            if (iz == -1)
              iz++;
            if (iz == 0) {
              endc_m->Fill(x + 0.5, y + 0.5, -1.);
              EEstat[0][1]++;
            } else {
              endc_p->Fill(x + 0.5, y + 0.5, -1.);
              EEstat[1][1]++;
            }
            //	      std::cout << " x " << x << " y " << y << " z " << iz << std::endl;
          }  // loop over crystals in strip
        }    // new strip
      }      // loop over run 1 strips

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      const Int_t NRGBs = 5;
      const Int_t NCont = 255;

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);
      //      TCanvas canvas("CC map","CC map", 1600, 450);
      Double_t w = 1200;
      Double_t h = 650;
      TCanvas canvas("c", "c", w, h);
      //      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2) {
        if (len < 180) {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("%s %i - %s %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.05);
          t1.DrawLatex(0.5, 0.96, Form("Ecal TPGStripStatus, IOV %i - %i", run[1], run[0]));
        }
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 2; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj), Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEE(endc_m, -1.0, 1.0);
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.15, 0.92, Form("new %i old %i", EEstat[0][0], EEstat[0][1]));
      pad[1]->cd();
      DrawEE(endc_p, -1.0, 1.0);
      t1.DrawLatex(0.15, 0.92, Form("new %i old %i", EEstat[1][0], EEstat[1][1]));

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalTPGStripStatusDiffBase
  using EcalTPGStripStatusDiffOneTag = EcalTPGStripStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalTPGStripStatusDiffTwoTags = EcalTPGStripStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

  /*****************************************
 2d plot of EcalTPGStripStatus Error Summary of 1 IOV
 ******************************************/
  class EcalTPGStripStatusSummaryPlot : public cond::payloadInspector::PlotImage<EcalTPGStripStatus> {
  public:
    EcalTPGStripStatusSummaryPlot()
        : cond::payloadInspector::PlotImage<EcalTPGStripStatus>("Ecal TPGStrip Status Summary - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<EcalTPGStripStatus> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F* align;                          //pointer to align which is a 2D histogram

      int NbRows = 1;
      int NbColumns = 2;

      if (payload.get()) {  //payload is an iov retrieved from payload using hash.
        const EcalTPGStripStatusMap& stripMap = (*payload).getMap();

        align = new TH2F("Ecal TPGStrip Status Summary",
                         "Total                NumberOfMasked",
                         NbColumns,
                         0,
                         NbColumns,
                         NbRows,
                         0,
                         NbRows);

        int NbMaskedTT = 0;

        for (EcalTPGStripStatusMapIterator it = stripMap.begin(); it != stripMap.end(); ++it)
          if ((*it).second > 0)
            NbMaskedTT++;

        align->Fill(0.5, 0.5, stripMap.size());
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
      t1.DrawLatex(0.5, 0.96, Form("Endcap:Number of masked Trigger Strips, IOV %i", run));

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
PAYLOAD_INSPECTOR_MODULE(EcalTPGStripStatus) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGStripStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGStripStatusDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGStripStatusDiffTwoTags);
  PAYLOAD_INSPECTOR_CLASS(EcalTPGStripStatusSummaryPlot);
}
