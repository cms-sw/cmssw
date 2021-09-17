#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/Alignments.h"

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
  enum { kEBChannels = 61200, kEEChannels = 14648 };
  enum {
    MIN_IETA = 1,
    MIN_IPHI = 1,
    MAX_IETA = 85,
    MAX_IPHI = 360
  };                                                            // barrel (EB) lower and upper bounds on eta and phi
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100 };  // endcaps (EE) lower and upper bounds on x and y

  /*****************************************
 2d plot of ECAL Alignment of 1 IOV
 ******************************************/
  class EcalAlignmentPlot : public cond::payloadInspector::PlotImage<Alignments> {
  public:
    EcalAlignmentPlot() : cond::payloadInspector::PlotImage<Alignments>("ECAL Alignment - map ") { setSingleIov(true); }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();  //get reference to 1st element in the vector iovs
      std::shared_ptr<Alignments> payload =
          fetchPayload(std::get<1>(iov));   //std::get<1>(iov) refers to the Hash in the tuple iov
      unsigned int run = std::get<0>(iov);  //referes to Time_t in iov.
      TH2F* align;                          //pointer to align which is a 2D histogram
      std::string subdet;
      int NbRows;
      if (payload.get()) {  //payload is an iov retrieved from payload using hash.
        NbRows = (*payload).m_align.size();
        if (NbRows == 36)
          subdet = "EB";
        else if (NbRows == 4)
          subdet = "EE";
        else if (NbRows == 8)
          subdet = "ES";
        else
          subdet = "unknown";
        //  align = new TH2F("Align",Form("Alignment %s", subdet.c_str()),6, 0, 6, NbRows, 0, NbRows);
        align = new TH2F("Align",
                         "x           y            z               Phi         Theta         Psi",
                         6,
                         0,
                         6,
                         NbRows,
                         0,
                         NbRows);

        double row = NbRows - 0.5;
        for (std::vector<AlignTransform>::const_iterator it = (*payload).m_align.begin();
             it != (*payload).m_align.end();
             it++) {
          align->Fill(0.5, row, (*it).translation().x());
          align->Fill(1.5, row, (*it).translation().y());
          align->Fill(2.5, row, (*it).translation().z());
          align->Fill(3.5, row, (*it).rotation().getPhi());
          align->Fill(4.5, row, (*it).rotation().getTheta());
          align->Fill(5.5, row, (*it).rotation().getPsi());
          row = row - 1.;
        }
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
      t1.DrawLatex(0.5, 0.96, Form("Ecal %s Alignment, IOV %i", subdet.c_str(), run));
      //      t1.SetTextSize(0.03);
      //      t1.DrawLatex(0.3, 0.94, "x          y           z           Phi          Theta          Psi");

      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");
      TLine* l = new TLine;
      l->SetLineWidth(1);
      for (int i = 1; i < NbRows; i++) {
        double y = (double)i;
        l = new TLine(0., y, 6., y);
        l->Draw();
      }

      for (int i = 1; i < 6; i++) {
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
  };

  /*********************************************************
      2d plot of ECAL Alignment difference between 2 IOVs
  **********************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class EcalAlignmentDiffBase : public cond::payloadInspector::PlotImage<Alignments, nIOVs, ntags> {
  public:
    EcalAlignmentDiffBase()
        : cond::payloadInspector::PlotImage<Alignments, nIOVs, ntags>("ECAL Alignment difference") {}

    bool fill() override {
      unsigned int run[2];
      float val[6][36];
      TH2F* align = new TH2F("", "", 1, 0., 1., 1, 0., 1.);  // pseudo creation
      std::string subdet;
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
        std::shared_ptr<Alignments> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }

        if (payload.get()) {
          NbRows = (*payload).m_align.size();
          if (irun == 1) {
            if (NbRows == 36)
              subdet = "EB";
            else if (NbRows == 4)
              subdet = "EE";
            else if (NbRows == 8)
              subdet = "ES";
            else
              subdet = "unknown";
            delete align;
            align = new TH2F("Align",
                             "x           y            z               Phi         Theta         Psi",
                             6,
                             0,
                             6,
                             NbRows,
                             0,
                             NbRows);
          }

          double row = NbRows - 0.5;
          int irow = 0;
          for (std::vector<AlignTransform>::const_iterator it = (*payload).m_align.begin();
               it != (*payload).m_align.end();
               it++) {
            if (irun == 0) {
              val[0][irow] = (*it).translation().x();
              val[1][irow] = (*it).translation().y();
              val[2][irow] = (*it).translation().z();
              val[3][irow] = (*it).rotation().getPhi();
              val[4][irow] = (*it).rotation().getTheta();
              val[5][irow] = (*it).rotation().getPsi();
            } else {
              align->Fill(0.5, row, (*it).translation().x() - val[0][irow]);
              align->Fill(1.5, row, (*it).translation().y() - val[1][irow]);
              align->Fill(2.5, row, (*it).translation().z() - val[2][irow]);
              align->Fill(3.5, row, (*it).rotation().getPhi() - val[3][irow]);
              align->Fill(4.5, row, (*it).rotation().getTheta() - val[3][irow]);
              align->Fill(5.5, row, (*it).rotation().getPsi() - val[5][irow]);
              row = row - 1.;
            }
            irow++;
          }  // loop over alignment rows

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
      if (ntags == 2 && len < 58) {
        t1.SetTextSize(0.025);
        t1.DrawLatex(
            0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("Ecal %s Alignment, IOV %i - %i", subdet.c_str(), run[1], run[0]));
      }
      TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
      pad->Draw();
      pad->cd();
      align->Draw("TEXT");
      TLine* l = new TLine;
      l->SetLineWidth(1);

      for (int i = 1; i < NbRows; i++) {
        double y = (double)i;
        l = new TLine(0., y, 6., y);
        l->Draw();
      }

      for (int i = 1; i < 6; i++) {
        double x = (double)i;
        double y = (double)NbRows;
        l = new TLine(x, 0., x, y);
        l->Draw();
      }

      align->GetXaxis()->SetTickLength(0.);
      align->GetXaxis()->SetLabelSize(0.);
      align->GetYaxis()->SetTickLength(0.);
      align->GetYaxis()->SetLabelSize(0.);

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };   // class EcalAlignmentDiffBase
  using EcalAlignmentDiffOneTag = EcalAlignmentDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using EcalAlignmentDiffTwoTags = EcalAlignmentDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalAlignment) {
  PAYLOAD_INSPECTOR_CLASS(EcalAlignmentPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalAlignmentDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(EcalAlignmentDiffTwoTags);
}
