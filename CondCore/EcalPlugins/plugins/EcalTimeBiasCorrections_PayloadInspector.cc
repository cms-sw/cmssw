#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"

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

  /****************************************************
      2d plot of Ecal TimeBias Corrections of 1 IOV
   ****************************************************/
  class EcalTimeBiasCorrectionsPlot : public cond::payloadInspector::PlotImage<EcalTimeBiasCorrections> {
  public:
    //void fillPlot_align(std::vector<float>& vect,TH2F* align, float column, double row);

    EcalTimeBiasCorrectionsPlot()
        : cond::payloadInspector::PlotImage<EcalTimeBiasCorrections>("Ecal Time Bias Corrections - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<EcalTimeBiasCorrections> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      TH2F* align;
      int NbRows;

      if (payload.get()) {
        EcalTimeBiasCorrections it = (*payload);
        /*
std::ostream stream(nullptr);
std::stringbuf str;
stream.rdbuf(&str);
it.print(stream);
std::cout<<str.str()<<std::endl;
*/

        NbRows = it.EBTimeCorrAmplitudeBins.size();
        /*
      std::cout<<"....."<<std::endl<<it.EBTimeCorrAmplitudeBins.size()<<std::endl;
      std::cout<<it.EBTimeCorrShiftBins.size()<<std::endl;
      std::cout<<it.EETimeCorrAmplitudeBins.size()<<std::endl;
      std::cout<<it.EETimeCorrShiftBins.size()<<std::endl;
*/
        if (NbRows == 0)
          return false;

        align = new TH2F("Ecal Time Bias Corrections",
                         "EBTimeCorrAmplitudeBins  EBTimeCorrShiftBins  EETimeCorrAmplitudeBins  EETimeCorrShiftBins",
                         4,
                         0,
                         4,
                         NbRows,
                         0,
                         NbRows);

        double row = NbRows - 0.5;

        fillPlot_align(it.EBTimeCorrAmplitudeBins, align, 0.5, row);
        fillPlot_align(it.EBTimeCorrShiftBins, align, 1.5, row);
        fillPlot_align(it.EETimeCorrAmplitudeBins, align, 2.5, row);
        fillPlot_align(it.EETimeCorrShiftBins, align, 3.5, row);

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
      t1.DrawLatex(0.5, 0.96, Form("Ecal Time Bias Corrections, IOV %i", run));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, 4);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }

    void fillPlot_align(std::vector<float>& vect, TH2F* align, float column, double& row) {
      for (std::vector<float>::const_iterator i = vect.begin(); i != vect.end(); i++) {
        align->Fill(column, row, *i);
        row = row - 1;
      }
    }
  };

  /********************************************************************
     2d plot of Ecal Time Bias Corrections difference between 2 IOVs
   ********************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalTimeBiasCorrectionsDiffBase
      : public cond::payloadInspector::PlotImage<EcalTimeBiasCorrections, nIOVs, ntags> {
  public:
    EcalTimeBiasCorrectionsDiffBase()
        : cond::payloadInspector::PlotImage<EcalTimeBiasCorrections, nIOVs, ntags>(
              "Ecal Time Bias Corrections difference") {}
    bool fill() override {
      unsigned int run[2], NbRows = 0;
      float val[4][36];
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
        std::shared_ptr<EcalTimeBiasCorrections> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        if (payload.get()) {
          EcalTimeBiasCorrections it = (*payload);

          NbRows = it.EBTimeCorrAmplitudeBins.size();
          if (irun == 1) {
            align =
                new TH2F("Ecal Time Bias Corrections",
                         "EBTimeCorrAmplitudeBins  EBTimeCorrShiftBins  EETimeCorrAmplitudeBins  EETimeCorrShiftBins",
                         4,
                         0,
                         4,
                         NbRows,
                         0,
                         NbRows);
          }

          double row = NbRows - 0.5;

          fillDiff_align(it.EBTimeCorrAmplitudeBins, align, val[0], 0.5, row, irun);
          fillDiff_align(it.EBTimeCorrShiftBins, align, val[1], 1.5, row, irun);
          fillDiff_align(it.EETimeCorrAmplitudeBins, align, val[2], 2.5, row, irun);
          fillDiff_align(it.EETimeCorrShiftBins, align, val[3], 3.5, row, irun);
          ;

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
      t1.DrawLatex(0.5, 0.96, Form("Ecal Time Bias Corrections, IOV %i - %i", run[1], run[0]));

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");

      drawTable(NbRows, 4);

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }

    void fillDiff_align(
        const std::vector<float>& vect, TH2F* align, float val[], const float column, double row, unsigned irun) {
      int irow = 0;

      for (std::vector<float>::const_iterator i = vect.begin(); i != vect.end(); ++i) {
        if (irun == 0) {
          val[irow] = (*i);
        } else {
          align->Fill(column, row, (*i) - val[irow]);
          row--;
        }
        irow++;
      }
    }
  };  // class EcalTimeBiasCorrectionsDiffBase
  using EcalTimeBiasCorrectionsDiffOneTag = EcalTimeBiasCorrectionsDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalTimeBiasCorrectionsDiffTwoTags = EcalTimeBiasCorrectionsDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTimeBiasCorrections) {
  PAYLOAD_INSPECTOR_CLASS(EcalTimeBiasCorrectionsPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTimeBiasCorrectionsDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalTimeBiasCorrectionsDiffTwoTags);
}
