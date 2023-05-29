#ifndef CONDCORE_PCLCONFIGPLUGINS_SIPIXELALIPCLTHRESHOLDSPAYLOADINSPECTORHELPER_H
#define CONDCORE_PCLCONFIGPLUGINS_SIPIXELALIPCLTHRESHOLDSPAYLOADINSPECTORHELPER_H

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"

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

namespace AlignPCLThresholdPlotHelper {

  enum types { DELTA = 0, SIG = 1, MAXMOVE = 2, MAXERR = 3, FRACTION_CUT = 4, END_OF_TYPES = 5 };

  /************************************************/
  inline const std::string getStringFromCoordEnum(const AlignPCLThresholds::coordType& coord) {
    switch (coord) {
      case AlignPCLThresholds::X:
        return "X";
      case AlignPCLThresholds::Y:
        return "Y";
      case AlignPCLThresholds::Z:
        return "Z";
      case AlignPCLThresholds::theta_X:
        return "#theta_{X}";
      case AlignPCLThresholds::theta_Y:
        return "#theta_{Y}";
      case AlignPCLThresholds::theta_Z:
        return "#theta_{Z}";
      default:
        return "should never be here";
    }
  }

  /************************************************/
  inline const std::string getStringFromTypeEnum(const types& type) {
    switch (type) {
      case types::DELTA:
        return "#Delta";
      case types::SIG:
        return "#Delta/#sigma ";
      case types::MAXMOVE:
        return "max. move ";
      case types::MAXERR:
        return "max. err ";
      case types::FRACTION_CUT:
        return "fraction cut ";
      default:
        return "should never be here";
    }
  }

  /************************************************/
  inline std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) {
    std::string out(str);

    if (from.empty())
      return out;
    size_t start_pos = 0;
    while ((start_pos = out.find(from, start_pos)) != std::string::npos) {
      out.replace(start_pos, from.length(), to);
      start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
    return out;
  }

  /************************************************
    Display of AlignPCLThreshold
  *************************************************/
  template <class PayloadType>
  class AlignPCLThresholds_DisplayBase
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    AlignPCLThresholds_DisplayBase()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "Display of threshold parameters for SiPixelAli PCL") {
      if constexpr (std::is_same_v<PayloadType, AlignPCLThresholdsHG>) {
        isHighGranularity_ = true;
        label_ = "AlignPCLThresholdsHG_PayloadInspector";
      } else {
        isHighGranularity_ = false;
        label_ = "AlignPCLThresholds_PayloadInspector";
      }
    }

    bool fill() override {
      // to be able to see zeros
      gStyle->SetHistMinimumZero(kTRUE);

      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));
      auto alignables = payload->getAlignableList();

      TCanvas canvas("Alignment PCL thresholds summary", "Alignment PCL thresholds summary", 1500, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(isHighGranularity_ ? 0.20 : 0.06);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.05);
      canvas.Modified();
      canvas.SetGrid();

      // needed for the internal loop
      const auto& local_end_of_types = isHighGranularity_ ? END_OF_TYPES : FRACTION_CUT;
      const int N_Y_BINS = AlignPCLThresholds::extra_DOF * local_end_of_types;

      auto Thresholds =
          std::make_unique<TH2F>("Thresholds", "", alignables.size(), 0, alignables.size(), N_Y_BINS, 0, N_Y_BINS);
      Thresholds->SetStats(false);
      Thresholds->GetXaxis()->SetLabelSize(0.028);

      std::function<float(types, std::string, AlignPCLThresholds::coordType)> cutFunctor =
          [&payload](types my_type, std::string alignable, AlignPCLThresholds::coordType coord) {
            float ret(-999.);
            switch (my_type) {
              case DELTA:
                return payload->getCut(alignable, coord);
              case SIG:
                return payload->getSigCut(alignable, coord);
              case MAXMOVE:
                return payload->getMaxMoveCut(alignable, coord);
              case MAXERR:
                return payload->getMaxErrorCut(alignable, coord);
              case FRACTION_CUT: {
                if constexpr (std::is_same_v<PayloadType, AlignPCLThresholdsHG>) {
                  const AlignPCLThresholdsHG::param_map& floatMap = payload->getFloatMap();
                  if (floatMap.find(alignable) != floatMap.end()) {
                    return payload->getFractionCut(alignable, coord);
                  } else {
                    return 0.f;
                  }
                } else {
                  assert(false); /* cannot be here */
                }
              }
              case END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

      unsigned int xBin = 0;
      for (const auto& alignable : alignables) {
        xBin++;
        auto xLabel = replaceAll(replaceAll(alignable, "minus", "(-)"), "plus", "(+)");
        Thresholds->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        unsigned int yBin = N_Y_BINS;
        for (int foo = PayloadType::X; foo != PayloadType::extra_DOF; foo++) {
          AlignPCLThresholds::coordType coord = static_cast<AlignPCLThresholds::coordType>(foo);
          for (int bar = DELTA; bar != local_end_of_types; bar++) {
            types type = static_cast<types>(bar);
            std::string theLabel = getStringFromTypeEnum(type) + getStringFromCoordEnum(coord);
            if (xBin == 1) {
              Thresholds->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
            }

            Thresholds->SetBinContent(xBin, yBin, cutFunctor(type, alignable, coord));

            yBin--;
          }  // loop on types
        }    // loop on coordinates
      }      // loop on alignables

      Thresholds->GetXaxis()->LabelsOption(isHighGranularity_ ? "v" : "h");
      Thresholds->Draw("TEXT");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.047);
      ltx.SetTextAlign(11);
      std::string ltxText =
          fmt::sprintf("#color[4]{%s} IOV: #color[4]{%s}", tag.name, std::to_string(std::get<0>(iov)));
      ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, ltxText.c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    bool isHighGranularity_;
    std::string label_;
  };

  /************************************************
    Compare AlignPCLThresholdsHG mapping
  *************************************************/
  template <class PayloadType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class AlignPCLThresholds_CompareBase : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    AlignPCLThresholds_CompareBase()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>("Table of AlignPCLThresholdsHG comparison") {
      if constexpr (std::is_same_v<PayloadType, AlignPCLThresholdsHG>) {
        isHighGranularity_ = true;
        label_ = "AlignPCLThresholdsHG_PayloadInspector";
      } else {
        isHighGranularity_ = false;
        label_ = "AlignPCLThresholds_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetPalette(kTemperatureMap);

      // to be able to see zeros
      gStyle->SetHistMinimumZero(kTRUE);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<PayloadType> l_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<PayloadType> f_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      const auto& alignables = l_payload->getAlignableList();
      const auto& alignables2 = f_payload->getAlignableList();

      std::vector<std::string> v_intersection;

      if (!isEqual(alignables, alignables2)) {
        edm::LogWarning(label_)
            << "Cannot compare directly the two AlignPCLThresholds objects, as the list of alignables differs";
        std::set_intersection(alignables.begin(),
                              alignables.end(),
                              alignables2.begin(),
                              alignables2.end(),
                              std::back_inserter(v_intersection));

        std::vector<std::string> not_in_first_keys, not_in_last_keys;

        // find the elements not in common
        std::set_difference(alignables.begin(),
                            alignables.end(),
                            alignables2.begin(),
                            alignables2.end(),
                            std::inserter(not_in_last_keys, not_in_last_keys.begin()));

        std::stringstream ss;
        ss << "the following keys are not in the last IoV: ";
        for (const auto& key : not_in_last_keys) {
          ss << key << ",";
        }
        ss << std::endl;
        edm::LogWarning(label_) << ss.str();
        ss.str(std::string()); /* clear the stream */

        std::set_difference(alignables2.begin(),
                            alignables2.end(),
                            alignables.begin(),
                            alignables.end(),
                            std::inserter(not_in_first_keys, not_in_first_keys.begin()));

        ss << "the following keys are not in the first IoV: ";
        for (const auto& key : not_in_first_keys) {
          ss << key << ",";
        }
        ss << std::endl;
        edm::LogWarning(label_) << ss.str();
      } else {
        // vectors are the same, just copy one
        v_intersection = alignables;
      }

      TCanvas canvas("Alignment PCL thresholds summary", "Alignment PCL thresholds summary", 1500, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(isHighGranularity_ ? 0.20 : 0.06);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.12);
      canvas.Modified();
      canvas.SetGrid();

      // needed for the internal loop
      const auto& local_end_of_types = isHighGranularity_ ? END_OF_TYPES : FRACTION_CUT;
      const int N_Y_BINS = AlignPCLThresholds::extra_DOF * local_end_of_types;

      auto Thresholds = std::make_unique<TH2F>(
          "Thresholds", "", v_intersection.size(), 0, v_intersection.size(), N_Y_BINS, 0, N_Y_BINS);
      Thresholds->SetStats(false);
      Thresholds->GetXaxis()->SetLabelSize(0.028);

      auto ThresholdsColor = std::make_unique<TH2F>(
          "ThresholdsC", "", v_intersection.size(), 0, v_intersection.size(), N_Y_BINS, 0, N_Y_BINS);
      ThresholdsColor->SetStats(false);
      ThresholdsColor->GetXaxis()->SetLabelSize(0.028);

      std::function<float(types, std::string, AlignPCLThresholds::coordType)> cutFunctor =
          [&f_payload, &l_payload](types my_type, std::string alignable, AlignPCLThresholds::coordType coord) {
            float ret(-999.);
            switch (my_type) {
              case DELTA:
                return l_payload->getCut(alignable, coord) - f_payload->getCut(alignable, coord);
              case SIG:
                return l_payload->getSigCut(alignable, coord) - f_payload->getSigCut(alignable, coord);
              case MAXMOVE:
                return l_payload->getMaxMoveCut(alignable, coord) - f_payload->getMaxMoveCut(alignable, coord);
              case MAXERR:
                return l_payload->getMaxErrorCut(alignable, coord) - f_payload->getMaxErrorCut(alignable, coord);
              case FRACTION_CUT: {
                if constexpr (std::is_same_v<PayloadType, AlignPCLThresholdsHG>) {
                  const AlignPCLThresholdsHG::param_map& f_floatMap = f_payload->getFloatMap();
                  const AlignPCLThresholdsHG::param_map& l_floatMap = l_payload->getFloatMap();
                  if (f_floatMap.find(alignable) != f_floatMap.end() &&
                      l_floatMap.find(alignable) != l_floatMap.end()) {
                    return l_payload->getFractionCut(alignable, coord) - f_payload->getFractionCut(alignable, coord);
                  } else {
                    return +999.f;
                  }
                } else {
                  assert(false); /* cannot be here */
                }
              }
              case END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

      unsigned int xBin = 0;
      for (const auto& alignable : v_intersection) {
        xBin++;
        auto xLabel = replaceAll(replaceAll(alignable, "minus", "(-)"), "plus", "(+)");
        Thresholds->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        ThresholdsColor->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        unsigned int yBin = N_Y_BINS;
        for (int foo = PayloadType::X; foo != PayloadType::extra_DOF; foo++) {
          AlignPCLThresholds::coordType coord = static_cast<AlignPCLThresholds::coordType>(foo);
          for (int bar = DELTA; bar != local_end_of_types; bar++) {
            types type = static_cast<types>(bar);
            std::string theLabel = getStringFromTypeEnum(type) + getStringFromCoordEnum(coord);
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
      ThresholdsColor->GetXaxis()->LabelsOption(isHighGranularity_ ? "v" : "h");
      Thresholds->Draw("TEXTsame");
      Thresholds->GetXaxis()->LabelsOption(isHighGranularity_ ? "v" : "h");

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
    bool isHighGranularity_;
    std::string label_;
  };
}  // namespace AlignPCLThresholdPlotHelper

#endif
