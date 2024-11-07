#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"

#include "TCanvas.h"
#include "TLatex.h"
#include "TLine.h"
#include <fmt/printf.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <numeric>
#include <sstream>

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    Display AlCaRecoTriggerBits mapping
  *************************************************/
  class AlCaRecoTriggerBits_Display : public PlotImage<AlCaRecoTriggerBits, SINGLE_IOV> {
  public:
    AlCaRecoTriggerBits_Display() : PlotImage<AlCaRecoTriggerBits, SINGLE_IOV>("Table of AlCaRecoTriggerBits") {}

    using TriggerMap = std::map<std::string, std::string>;

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::string IOVsince = std::to_string(std::get<0>(iov));
      auto tagname = tag.name;
      std::shared_ptr<AlCaRecoTriggerBits> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        const TriggerMap &triggerMap = payload->m_alcarecoToTrig;

        // pre-compute how many time we break the line
        const int totalCarriageReturns = calculateTotalCarriageReturns(payload);
        LogDebug("AlCaRecoTriggerBits_Display") << "Total number of carriage returns: " << totalCarriageReturns;

        // Dynamically calculate the pitch and canvas height
        float pitch = 1.0 / (totalCarriageReturns + 2.0);  // Adjusted pitch for better spacing

        float y = 1.0;
        float x1 = 0.02, x2 = x1 + 0.25;
        std::vector<float> y_x1, y_x2, y_line;
        std::vector<std::string> s_x1, s_x2;

        // Header row setup
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back("#scale[1.2]{Key}");
        y_x2.push_back(y);
        s_x2.push_back("#scale[1.2]{tag: " + tagname + " in IOV: " + IOVsince + "}");

        y -= pitch / 2.0;
        y_line.push_back(y);

        // Populate rows with data from the trigger map
        for (const auto &element : triggerMap) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back(element.first);
          std::vector<std::string> output;

          std::string toAppend = "";
          const std::vector<std::string> paths = payload->decompose(element.second);
          for (unsigned int iPath = 0; iPath < paths.size(); ++iPath) {
            if ((toAppend + paths[iPath]).length() < 80) {  // Wider lines
              toAppend += paths[iPath] + ";";
            } else {
              output.push_back(toAppend);
              toAppend.clear();
              toAppend += paths[iPath] + ";";
            }
            if (iPath == paths.size() - 1)
              output.push_back(toAppend);
          }

          for (unsigned int br = 0; br < output.size(); br++) {
            y_x2.push_back(y);
            s_x2.push_back("#color[2]{" + output[br] + "}");
            if (br != output.size() - 1)
              y -= pitch;
          }

          y_line.push_back(y - (pitch / 2.0));
        }

        // Dynamically calculate the pitch and canvas height
        float canvasHeight = std::max(800.0f, totalCarriageReturns * 30.0f);  // Adjust canvas height based on entries
        TCanvas canvas("AlCaRecoTriggerBits", "AlCaRecoTriggerBits", 2000, static_cast<int>(canvasHeight));

        TLatex l;
        l.SetTextAlign(12);
        float textSize = std::clamp(pitch, 0.015f, 0.035f);
        l.SetTextSize(textSize);

        // Draw the columns
        int totalPitches = 0;
        canvas.cd();
        for (unsigned int i = 0; i < y_x1.size(); i++) {
          l.DrawLatexNDC(x1, y_x1[i], s_x1[i].c_str());
          if (i != 0) {
            LogDebug("AlCaRecoTriggerBits_Display")
                << "x1:" << x1 << " y_x1[" << std::setw(2) << i << "]: " << y_x1[i]
                << " Delta = " << std::ceil((y_x1[i - 1] - y_x1[i]) / pitch) << " pitches " << s_x1[i].c_str();
            totalPitches += std::ceil((y_x1[i - 1] - y_x1[i]) / pitch);
          }
        }

        LogDebug("AlCaRecoTriggerBits_Display") << "We've gone down by " << totalPitches << "pitches ";

        for (unsigned int i = 0; i < y_x2.size(); i++) {
          l.DrawLatexNDC(x2, y_x2[i], s_x2[i].c_str());
        }

        // Draw lines for row separation
        TLine lines[y_line.size()];
        for (unsigned int i = 0; i < y_line.size(); i++) {
          lines[i] = TLine(gPad->GetUxmin(), y_line[i], gPad->GetUxmax(), y_line[i]);
          lines[i].SetLineWidth(1);
          lines[i].SetLineStyle(9);
          lines[i].SetLineColor(2);
          lines[i].Draw("same");
        }

        canvas.SaveAs(m_imageFileName.c_str());
      }
      return true;
    }

  private:
    int calculateTotalCarriageReturns(std::shared_ptr<AlCaRecoTriggerBits> payload) {
      int totalCarriageReturns = 0;
      const TriggerMap &triggerMap = payload->m_alcarecoToTrig;

      for (const auto &element : triggerMap) {
        const auto &paths = payload->decompose(element.second);
        int lineLength = 0;

        for (const auto &path : paths) {
          lineLength += path.length() + 1;  // +1 for the semicolon
          if (lineLength >= 80) {
            totalCarriageReturns++;
            lineLength = path.length() + 1;  // Reset for the next line segment
          }
        }
        totalCarriageReturns++;  // Count the initial line for each element
      }
      return totalCarriageReturns;
    }
  };

  /************************************************
    Compare AlCaRecoTriggerBits mapping
  *************************************************/
  template <IOVMultiplicity nIOVs, int ntags>
  class AlCaRecoTriggerBits_CompareBase : public PlotImage<AlCaRecoTriggerBits, nIOVs, ntags> {
  public:
    AlCaRecoTriggerBits_CompareBase()
        : PlotImage<AlCaRecoTriggerBits, nIOVs, ntags>("Table of AlCaRecoTriggerBits comparison") {}

    bool fill() override {
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

      std::shared_ptr<AlCaRecoTriggerBits> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlCaRecoTriggerBits> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      // Get map of strings to concatenated list of names of HLT paths:
      typedef std::map<std::string, std::string> TriggerMap;
      const TriggerMap &first_triggerMap = first_payload->m_alcarecoToTrig;
      const TriggerMap &last_triggerMap = last_payload->m_alcarecoToTrig;

      std::vector<std::string> first_keys, not_in_first_keys;
      std::vector<std::string> last_keys, not_in_last_keys;

      // fill the vector of first keys
      for (const auto &element : first_triggerMap) {
        first_keys.push_back(element.first);
      }

      // fill the vector of last keys
      for (const auto &element : last_triggerMap) {
        last_keys.push_back(element.first);
      }

      // find the elements not in common
      std::set_difference(first_keys.begin(),
                          first_keys.end(),
                          last_keys.begin(),
                          last_keys.end(),
                          std::inserter(not_in_last_keys, not_in_last_keys.begin()));

      std::set_difference(last_keys.begin(),
                          last_keys.end(),
                          first_keys.begin(),
                          first_keys.end(),
                          std::inserter(not_in_first_keys, not_in_first_keys.begin()));

      float pitch = 0.013;
      float y, x1, x2, x3;

      std::vector<float> y_x1, y_x2, y_x3, y_line;
      std::vector<std::string> s_x1, s_x2, s_x3;

      y = 1.0;
      x1 = 0.02;
      x2 = x1 + 0.20;
      x3 = x2 + 0.30;

      y -= pitch;
      y_x1.push_back(y);
      s_x1.push_back("#scale[1.2]{Key}");
      y_x2.push_back(y);
      s_x2.push_back(fmt::sprintf("#scale[1.2]{%s in IOV: %s}", f_tagname, firstIOVsince));
      y_x3.push_back(y);
      s_x3.push_back(fmt::sprintf("#scale[1.2]{%s in IOV: %s}", l_tagname, lastIOVsince));
      y -= pitch / 3;

      // print the ones missing in the last key
      for (const auto &key : not_in_last_keys) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(key);

        const std::vector<std::string> missing_in_last_paths = first_payload->decompose(first_triggerMap.at(key));

        std::vector<std::string> output;
        std::string toAppend = "";
        for (unsigned int iPath = 0; iPath < missing_in_last_paths.size(); ++iPath) {
          // if the line to be added has less than 60 chars append to current
          if ((toAppend + missing_in_last_paths[iPath]).length() < 60) {
            toAppend += missing_in_last_paths[iPath] + ";";
          } else {
            // else if the line exceeds 60 chars, dump in the vector and resume from scratch
            output.push_back(toAppend);
            toAppend.clear();
            toAppend += missing_in_last_paths[iPath] + ";";
          }
          // if it's the last, dump it
          if (iPath == missing_in_last_paths.size() - 1)
            output.push_back(toAppend);
        }

        for (unsigned int br = 0; br < output.size(); br++) {
          y_x2.push_back(y);
          s_x2.push_back("#color[2]{" + output[br] + "}");
          if (br != output.size() - 1)
            y -= pitch;
        }
        y_line.push_back(y - 0.008);
      }

      // print the ones missing in the first key
      for (const auto &key : not_in_first_keys) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(key);
        const std::vector<std::string> missing_in_first_paths = last_payload->decompose(last_triggerMap.at(key));

        std::vector<std::string> output;
        std::string toAppend = "";
        for (unsigned int iPath = 0; iPath < missing_in_first_paths.size(); ++iPath) {
          // if the line to be added has less than 60 chars append to current
          if ((toAppend + missing_in_first_paths[iPath]).length() < 60) {
            toAppend += missing_in_first_paths[iPath] + ";";
          } else {
            // else if the line exceeds 60 chars, dump in the vector and resume from scratch
            output.push_back(toAppend);
            toAppend.clear();
            toAppend += missing_in_first_paths[iPath] + ";";
          }
          // if it's the last, dump it
          if (iPath == missing_in_first_paths.size() - 1)
            output.push_back(toAppend);
        }

        for (unsigned int br = 0; br < output.size(); br++) {
          y_x3.push_back(y);
          s_x3.push_back("#color[4]{" + output[br] + "}");
          if (br != output.size() - 1)
            y -= pitch;
        }
        y_line.push_back(y - 0.008);
      }

      for (const auto &element : first_triggerMap) {
        if (last_triggerMap.find(element.first) != last_triggerMap.end()) {
          auto lastElement = last_triggerMap.find(element.first);

          std::string output;
          const std::vector<std::string> first_paths = first_payload->decompose(element.second);
          const std::vector<std::string> last_paths = last_payload->decompose(lastElement->second);

          std::vector<std::string> not_in_first;
          std::vector<std::string> not_in_last;

          std::set_difference(first_paths.begin(),
                              first_paths.end(),
                              last_paths.begin(),
                              last_paths.end(),
                              std::inserter(not_in_last, not_in_last.begin()));

          std::set_difference(last_paths.begin(),
                              last_paths.end(),
                              first_paths.begin(),
                              first_paths.end(),
                              std::inserter(not_in_first, not_in_first.begin()));

          if (!not_in_last.empty() || !not_in_first.empty()) {
            y -= pitch;
            y_x1.push_back(y);
            s_x1.push_back(element.first);

            std::vector<std::string> output;
            std::string toAppend = "";
            for (unsigned int iPath = 0; iPath < not_in_last.size(); ++iPath) {
              // if the line to be added has less than 60 chars append to current
              if ((toAppend + not_in_last[iPath]).length() < 60) {
                toAppend += not_in_last[iPath] + ";";
              } else {
                // else if the line exceeds 60 chars, dump in the vector and resume from scratch
                output.push_back(toAppend);
                toAppend.clear();
                toAppend += not_in_last[iPath] + ";";
              }
              // if it's the last and not empty, dump it
              if (toAppend.length() > 0 && iPath == not_in_last.size() - 1)
                output.push_back(toAppend);
            }

            unsigned int count = output.size();

            for (unsigned int br = 0; br < count; br++) {
              y_x2.push_back(y - (br * pitch));
              s_x2.push_back("#color[6]{" + output[br] + "}");
            }

            // clear vector and string
            toAppend.clear();
            output.clear();
            for (unsigned int jPath = 0; jPath < not_in_first.size(); ++jPath) {
              // if the line to be added has less than 60 chars append to current
              if ((toAppend + not_in_first[jPath]).length() < 60) {
                toAppend += not_in_first[jPath] + ";";
              } else {
                // else if the line exceeds 60 chars, dump in the vector and resume from scratch
                output.push_back(toAppend);
                toAppend.clear();
                toAppend += not_in_first[jPath] + ";";
              }
              // if it's the last and not empty, dump it
              if (toAppend.length() > 0 && jPath == not_in_first.size() - 1)
                output.push_back(toAppend);
            }

            unsigned int count1 = output.size();

            for (unsigned int br = 0; br < count1; br++) {
              y_x3.push_back(y - (br * pitch));
              s_x3.push_back("#color[8]{" + output[br] + "}");
            }

            // decrease the y position to the maximum of the two lists
            y -= (std::max(count, count1) - 1) * pitch;
            //y-=count*pitch;
            y_line.push_back(y - 0.008);

          }  // close if there is at least a difference
        }  // if there is a common key
      }  //loop on the keys

      TCanvas canvas("AlCaRecoTriggerBits", "AlCaRecoTriggerBits", 2500., std::max(y_x1.size(), y_x2.size()) * 40);

      TLatex l;
      // Draw the columns titles
      l.SetTextAlign(12);

      // rescale the width of the table row to fit into the canvas
      float newpitch = 1 / (std::max(y_x1.size(), y_x2.size()) * 1.65);
      float factor = newpitch / pitch;
      l.SetTextSize(newpitch - 0.002);
      canvas.cd();
      for (unsigned int i = 0; i < y_x1.size(); i++) {
        l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]) * factor, s_x1[i].c_str());
      }

      for (unsigned int i = 0; i < y_x2.size(); i++) {
        l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]) * factor, s_x2[i].c_str());
      }

      for (unsigned int i = 0; i < y_x3.size(); i++) {
        l.DrawLatexNDC(x3, 1 - (1 - y_x3[i]) * factor, s_x3[i].c_str());
      }

      canvas.cd();
      canvas.Update();

      TLine lines[y_line.size()];
      unsigned int iL = 0;
      for (const auto &line : y_line) {
        lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
        lines[iL].SetLineWidth(1);
        lines[iL].SetLineStyle(9);
        lines[iL].SetLineColor(2);
        lines[iL].Draw("same");
        iL++;
      }

      //canvas.SetCanvasSize(2000,(1-y)*1000);
      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

  using AlCaRecoTriggerBits_Compare = AlCaRecoTriggerBits_CompareBase<MULTI_IOV, 1>;
  using AlCaRecoTriggerBits_CompareTwoTags = AlCaRecoTriggerBits_CompareBase<SINGLE_IOV, 2>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(AlCaRecoTriggerBits) {
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_Display);
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_Compare);
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_CompareTwoTags);
}
