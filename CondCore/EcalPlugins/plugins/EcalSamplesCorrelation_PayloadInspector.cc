#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"

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
 2d plot of Ecal Samples Correlation of 1 IOV
 *******************************************************/
  class EcalSamplesCorrelationPlot : public cond::payloadInspector::PlotImage<EcalSamplesCorrelation> {
  public:
    //void fill_align(const std::vector<double>& vect,TH2F* align, const int column, int row);

    EcalSamplesCorrelationPlot()
        : cond::payloadInspector::PlotImage<EcalSamplesCorrelation>("Ecal Samples Correlation - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<EcalSamplesCorrelation> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      TH2F* align;
      int NbRows;

      if (payload.get()) {
        EcalSamplesCorrelation it = (*payload);
        NbRows = it.EBG12SamplesCorrelation.size();
        if (NbRows == 0)
          return false;

        align = new TH2F("Ecal Samples Correlation",
                         "EBG12          EBG06          EBG01          EEG12          EEG06          EEG01",
                         6,
                         0,
                         6,
                         NbRows,
                         0,
                         NbRows);

        double row = NbRows - 0.5;

        fill_align(it.EBG12SamplesCorrelation, align, 0.5, row);
        fill_align(it.EBG6SamplesCorrelation, align, 1.5, row);
        fill_align(it.EBG1SamplesCorrelation, align, 2.5, row);

        fill_align(it.EEG12SamplesCorrelation, align, 3.5, row);
        fill_align(it.EEG6SamplesCorrelation, align, 4.5, row);
        fill_align(it.EEG1SamplesCorrelation, align, 5.5, row);

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
      t1.DrawLatex(0.5, 0.96, Form("Ecal Samples Correlation, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, 6);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }

    void fill_align(const std::vector<double>& vect, TH2F* align, const float column, double row) {
      for (std::vector<double>::const_iterator i = vect.begin(); i != vect.end(); ++i) {
        align->Fill(column, row, *i);
        row = row - 1;
      }
    }
  };

  /*******************************************************
 2d plot of Ecal Samples Correlation difference between 2 IOVs
*******************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalSamplesCorrelationDiffBase
      : public cond::payloadInspector::PlotImage<EcalSamplesCorrelation, nIOVs, ntags> {
  public:
    EcalSamplesCorrelationDiffBase()
        : cond::payloadInspector::PlotImage<EcalSamplesCorrelation, nIOVs, ntags>(
              "Ecal Samples Correlation difference") {}
    bool fill() override {
      unsigned int run[2];
      float val[6][36];
      TH2F* align = new TH2F("", "", 1, 0., 1., 1, 0., 1.);  // pseudo creation
      int NbRows = 0;
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
        std::shared_ptr<EcalSamplesCorrelation> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }

        if (payload.get()) {
          EcalSamplesCorrelation it = (*payload);
          NbRows = it.EBG12SamplesCorrelation.size();

          if (irun == 1) {
            align = new TH2F("Ecal Samples Correlation",
                             "EBG12          EBG06          EBG01          EEG12          EEG06          EEG01",
                             6,
                             0,
                             6,
                             NbRows,
                             0,
                             NbRows);
          }

          double row = NbRows - 0.5;

          fill_align(it.EBG12SamplesCorrelation, align, val[0], 0.5, row, irun);
          fill_align(it.EBG6SamplesCorrelation, align, val[1], 1.5, row, irun);
          fill_align(it.EBG1SamplesCorrelation, align, val[2], 2.5, row, irun);

          fill_align(it.EEG12SamplesCorrelation, align, val[3], 3.5, row, irun);
          fill_align(it.EEG6SamplesCorrelation, align, val[4], 4.5, row, irun);
          fill_align(it.EEG1SamplesCorrelation, align, val[5], 5.5, row, irun);

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
      t1.SetTextSize(0.05);
      t1.SetTextColor(2);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Samples Correlation, IOV %i - %i", run[1], run[0]));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");
      TLine* l = new TLine;
      l->SetLineWidth(1);

      drawTable(NbRows, 6);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }

    void fill_align(
        const std::vector<double>& vect, TH2F* align, float val[], const float column, double row, unsigned irun) {
      int irow = 0;

      for (std::vector<double>::const_iterator i = vect.begin(); i != vect.end(); ++i) {
        if (irun == 0) {
          val[irow] = (*i);
        } else {
          align->Fill(column, row, (*i) - val[irow]);
          row--;
        }
        irow++;
      }
    }
  };  // class EcalSamplesCorrelationDiffBase
  using EcalSamplesCorrelationDiffOneTag = EcalSamplesCorrelationDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalSamplesCorrelationDiffTwoTags = EcalSamplesCorrelationDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalSamplesCorrelation) {
  PAYLOAD_INSPECTOR_CLASS(EcalSamplesCorrelationPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalSamplesCorrelationDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalSamplesCorrelationDiffTwoTags);
}
