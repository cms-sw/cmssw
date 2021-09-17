#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

#include "TH2F.h"  // a 2-D histogram with four bytes per cell (float)
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"  //write mathematical equations.
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>

namespace {

  /*******************************************************
 2d plot of Ecal Time Offset Constant of 1 IOV
 *******************************************************/
  class EcalTimeOffsetConstantPlot : public cond::payloadInspector::PlotImage<EcalTimeOffsetConstant> {
  public:
    EcalTimeOffsetConstantPlot()
        : cond::payloadInspector::PlotImage<EcalTimeOffsetConstant>("Ecal Time Offset Constant - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<EcalTimeOffsetConstant> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      TH2F* align;
      int NbRows;

      if (payload.get()) {
        NbRows = 1;
        align = new TH2F("Time Offset Constant [ns]", "EB          EE", 2, 0, 2, NbRows, 0, NbRows);
        EcalTimeOffsetConstant it = (*payload);

        double row = NbRows - 0.5;

        align->Fill(0.5, row, it.getEBValue());
        align->Fill(1.5, row, it.getEEValue());
      } else
        return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1000, 1000);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Time Offset Constant, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, 2);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }
  };

  /*****************************************************************
     2d plot of Ecal Time Offset Constant difference between 2 IOVs
   *****************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalTimeOffsetConstantDiffBase
      : public cond::payloadInspector::PlotImage<EcalTimeOffsetConstant, nIOVs, ntags> {
  public:
    EcalTimeOffsetConstantDiffBase()
        : cond::payloadInspector::PlotImage<EcalTimeOffsetConstant, nIOVs, ntags>(
              "Ecal Time Offset Constant difference") {}

    bool fill() override {
      unsigned int run[2], NbRows = 0;
      float val[2] = {};
      TH2F* align = new TH2F("", "", 1, 0., 1., 1, 0., 1.);  // pseudo creation
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
        std::shared_ptr<EcalTimeOffsetConstant> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          NbRows = 1;

          if (irun == 1)
            align = new TH2F("Ecal Time Offset Constant [ns]", "EB          EE", 2, 0, 2, NbRows, 0, NbRows);

          EcalTimeOffsetConstant it = (*payload);

          if (irun == 0) {
            val[0] = it.getEBValue();
            val[1] = it.getEEValue();

          } else {
            double row = NbRows - 0.5;
            align->Fill(0.5, row, it.getEBValue() - val[0]);
            align->Fill(1.5, row, it.getEEValue() - val[1]);
          }

        }  //  if payload.get()
        else
          return false;
      }  // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      TCanvas canvas("CC map", "CC map", 1000, 1000);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextColor(2);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2) {
        if (len < 80) {
          t1.SetTextSize(0.02);
          t1.DrawLatex(0.5, 0.96, Form("%s %i - %s %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.03);
          t1.DrawLatex(0.5, 0.96, Form("Ecal Time Offset Constant, IOV %i - %i", run[1], run[0]));
        }
      } else {
        t1.SetTextSize(0.03);
        t1.DrawLatex(0.5, 0.96, Form("%s, IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, 2);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }
  };  // class EcalTimeOffsetConstantDiffBase
  using EcalTimeOffsetConstantDiffOneTag = EcalTimeOffsetConstantDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalTimeOffsetConstantDiffTwoTags = EcalTimeOffsetConstantDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTimeOffsetConstant) {
  PAYLOAD_INSPECTOR_CLASS(EcalTimeOffsetConstantPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTimeOffsetConstantDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTimeOffsetConstantDiffTwoTags);
}
