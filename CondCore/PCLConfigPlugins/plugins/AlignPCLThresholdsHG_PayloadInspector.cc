/*!
  \file AlignPCLThresholdsHG_PayloadInspector
  \Payload Inspector Plugin for High Granularity Alignment PCL thresholds
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/10/19 12:51:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"

// for the PI Helper
#include "CondCore/PCLConfigPlugins/interface/SiPixelAliPCLThresholdsPayloadInspectorHelper.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <functional>
#include <fmt/printf.h>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLine.h"
#include "TStyle.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    Display of AlignPCLThresholdsHG
  *************************************************/
  class AlignPCLThresholdsHG_Display : public PlotImage<AlignPCLThresholdsHG, SINGLE_IOV> {
  public:
    AlignPCLThresholdsHG_Display()
        : PlotImage<AlignPCLThresholdsHG, SINGLE_IOV>("Display of threshold parameters for SiPixelAli PCL") {}

    bool fill() override {
      // to be able to see zeros
      gStyle->SetHistMinimumZero(kTRUE);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<AlignPCLThresholdsHG> payload = fetchPayload(std::get<1>(iov));
      auto alignables = payload->getAlignableList();

      TCanvas canvas("Alignment PCL thresholds summary", "Alignment PCL thresholds summary", 1500, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.20);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.05);
      canvas.Modified();
      canvas.SetGrid();

      auto Thresholds = std::make_unique<TH2F>("Thresholds", "", alignables.size(), 0, alignables.size(), 30, 0, 30);
      Thresholds->SetStats(false);
      Thresholds->GetXaxis()->SetLabelSize(0.028);

      std::function<float(PCLThresholdsPI::types, std::string, AlignPCLThresholdsHG::coordType)> cutFunctor =
          [&payload](PCLThresholdsPI::types my_type, std::string alignable, AlignPCLThresholdsHG::coordType coord) {
            float ret(-999.);
            switch (my_type) {
              case PCLThresholdsPI::DELTA:
                return payload->getCut(alignable, coord);
              case PCLThresholdsPI::SIG:
                return payload->getSigCut(alignable, coord);
              case PCLThresholdsPI::MAXMOVE:
                return payload->getMaxMoveCut(alignable, coord);
              case PCLThresholdsPI::MAXERR:
                return payload->getMaxErrorCut(alignable, coord);
              case PCLThresholdsPI::FRACTION_CUT: {
                const AlignPCLThresholdsHG::param_map& floatMap = payload->getFloatMap();
                if (floatMap.find(alignable) != floatMap.end()) {
                  return payload->getFractionCut(alignable, coord);
                } else {
                  return 0.f;
                }
              }
              case PCLThresholdsPI::END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

      unsigned int xBin = 0;
      for (const auto& alignable : alignables) {
        xBin++;

        auto xLabel =
            PCLThresholdsPI::replaceAll(PCLThresholdsPI::replaceAll(alignable, "minus", "(-)"), "plus", "(+)");
        Thresholds->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        unsigned int yBin = 30;
        for (int foo = AlignPCLThresholdsHG::X; foo != AlignPCLThresholdsHG::extra_DOF; foo++) {
          AlignPCLThresholdsHG::coordType coord = static_cast<AlignPCLThresholdsHG::coordType>(foo);
          for (int bar = PCLThresholdsPI::DELTA; bar != PCLThresholdsPI::END_OF_TYPES; bar++) {
            PCLThresholdsPI::types type = static_cast<PCLThresholdsPI::types>(bar);
            std::string theLabel =
                PCLThresholdsPI::getStringFromTypeEnum(type) + PCLThresholdsPI::getStringFromCoordEnum(coord);
            if (xBin == 1) {
              Thresholds->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
            }

            Thresholds->SetBinContent(xBin, yBin, cutFunctor(type, alignable, coord));

            yBin--;
          }  // loop on types
        }    // loop on coordinates
      }      // loop on alignables

      Thresholds->GetXaxis()->LabelsOption("v");
      Thresholds->Draw("TEXT");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.047);
      ltx.SetTextAlign(11);
      std::string ltxText =
          fmt::sprintf("#color[4]{%s} IOV: #color[4]{%s}", tag.name, std::to_string(std::get<0>(iov)));
      ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, ltxText.c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    Compare AlignPCLThresholdsHG mapping
  *************************************************/
  template <IOVMultiplicity nIOVs, int ntags>
  class AlignPCLThresholdsHG_CompareBase : public PlotImage<AlignPCLThresholdsHG, nIOVs, ntags> {
  public:
    AlignPCLThresholdsHG_CompareBase()
        : PlotImage<AlignPCLThresholdsHG, nIOVs, ntags>("Table of AlignPCLThresholdsHG comparison") {}

    bool fill() override {
      gStyle->SetPalette(kTemperatureMap);

      // to be able to see zeros
      gStyle->SetHistMinimumZero(kTRUE);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto f_tagname = PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        l_tagname = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<AlignPCLThresholdsHG> l_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignPCLThresholdsHG> f_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      const auto& alignables = l_payload->getAlignableList();
      const auto& alignables2 = f_payload->getAlignableList();

      if (!isEqual(alignables, alignables2)) {
        edm::LogError("AlignPCLThresholdsHG_PayloadInspector")
            << "Cannot compare the two AlignPCLThresholdsHG objects, as the list of alignables differs";
        return false;
      }

      TCanvas canvas("Alignment PCL thresholds summary", "Alignment PCL thresholds summary", 1500, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.20);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.12);
      canvas.Modified();
      canvas.SetGrid();

      auto Thresholds = std::make_unique<TH2F>("Thresholds", "", alignables.size(), 0, alignables.size(), 30, 0, 30);
      Thresholds->SetStats(false);
      Thresholds->GetXaxis()->SetLabelSize(0.028);

      auto ThresholdsColor =
          std::make_unique<TH2F>("ThresholdsC", "", alignables.size(), 0, alignables.size(), 30, 0, 30);
      ThresholdsColor->SetStats(false);
      ThresholdsColor->GetXaxis()->SetLabelSize(0.028);

      std::function<float(PCLThresholdsPI::types, std::string, AlignPCLThresholdsHG::coordType)> cutFunctor =
          [&f_payload, &l_payload](
              PCLThresholdsPI::types my_type, std::string alignable, AlignPCLThresholdsHG::coordType coord) {
            float ret(-999.);
            switch (my_type) {
              case PCLThresholdsPI::DELTA:
                return l_payload->getCut(alignable, coord) - f_payload->getCut(alignable, coord);
              case PCLThresholdsPI::SIG:
                return l_payload->getSigCut(alignable, coord) - f_payload->getSigCut(alignable, coord);
              case PCLThresholdsPI::MAXMOVE:
                return l_payload->getMaxMoveCut(alignable, coord) - f_payload->getMaxMoveCut(alignable, coord);
              case PCLThresholdsPI::MAXERR:
                return l_payload->getMaxErrorCut(alignable, coord) - f_payload->getMaxErrorCut(alignable, coord);
              case PCLThresholdsPI::FRACTION_CUT: {
                const AlignPCLThresholdsHG::param_map& f_floatMap = f_payload->getFloatMap();
                const AlignPCLThresholdsHG::param_map& l_floatMap = l_payload->getFloatMap();
                if (f_floatMap.find(alignable) != f_floatMap.end() && l_floatMap.find(alignable) != l_floatMap.end()) {
                  return l_payload->getFractionCut(alignable, coord) - f_payload->getFractionCut(alignable, coord);
                } else {
                  return +999.f;
                }
              }
              case PCLThresholdsPI::END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

      unsigned int xBin = 0;
      for (const auto& alignable : alignables) {
        xBin++;

        auto xLabel =
            PCLThresholdsPI::replaceAll(PCLThresholdsPI::replaceAll(alignable, "minus", "(-)"), "plus", "(+)");
        Thresholds->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        ThresholdsColor->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        unsigned int yBin = 30;
        for (int foo = AlignPCLThresholdsHG::X; foo != AlignPCLThresholdsHG::extra_DOF; foo++) {
          AlignPCLThresholdsHG::coordType coord = static_cast<AlignPCLThresholdsHG::coordType>(foo);
          for (int bar = PCLThresholdsPI::DELTA; bar != PCLThresholdsPI::END_OF_TYPES; bar++) {
            PCLThresholdsPI::types type = static_cast<PCLThresholdsPI::types>(bar);
            std::string theLabel =
                PCLThresholdsPI::getStringFromTypeEnum(type) + PCLThresholdsPI::getStringFromCoordEnum(coord);
            if (xBin == 1) {
              Thresholds->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
              ThresholdsColor->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
            }

            const auto& value = cutFunctor(type, alignable, coord);
            Thresholds->SetBinContent(xBin, yBin, value);
            ThresholdsColor->SetBinContent(xBin, yBin, value);

            yBin--;
          }  // loop on types
        }    // loop on coordinates
      }      // loop on alignables

      ThresholdsColor->Draw("COLZ0");
      ThresholdsColor->GetXaxis()->LabelsOption("v");
      Thresholds->Draw("TEXTsame");
      Thresholds->GetXaxis()->LabelsOption("v");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.047);
      ltx.SetTextAlign(11);
      std::string ltxText;
      if (this->m_plotAnnotations.ntags == 2) {
        ltxText = fmt::sprintf("#color[2]{%s, %s} vs #color[4]{%s, %s}",
                               f_tagname,
                               std::to_string(std::get<0>(firstiov)),
                               l_tagname,
                               std::to_string(std::get<0>(lastiov)));
      } else {
        ltxText = fmt::sprintf("%s IOV: #color[2]{%s} vs IOV: #color[4]{%s}",
                               f_tagname,
                               std::to_string(std::get<0>(firstiov)),
                               std::to_string(std::get<0>(lastiov)));
      }
      ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, ltxText.c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    template <typename T>
    bool isEqual(std::vector<T> const& v1, std::vector<T> const& v2) {
      return (v1.size() == v2.size() && std::equal(v1.begin(), v1.end(), v2.begin()));
    }
  };

  using AlignPCLThresholdsHG_Compare = AlignPCLThresholdsHG_CompareBase<MULTI_IOV, 1>;
  using AlignPCLThresholdsHG_CompareTwoTags = AlignPCLThresholdsHG_CompareBase<SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(AlignPCLThresholdsHG) {
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_Display);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_Compare);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_CompareTwoTags);
}
