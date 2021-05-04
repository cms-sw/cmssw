#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"

#include <memory>
#include <sstream>
#include <iostream>
#include "TCanvas.h"
#include "TLatex.h"
#include "TLine.h"

namespace {

  /************************************************
    Display AlCaRecoTriggerBits mapping
  *************************************************/
  class AlCaRecoTriggerBits_Display : public cond::payloadInspector::PlotImage<AlCaRecoTriggerBits> {
  public:
    AlCaRecoTriggerBits_Display()
        : cond::payloadInspector::PlotImage<AlCaRecoTriggerBits>("Table of AlCaRecoTriggerBits") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<AlCaRecoTriggerBits> payload = fetchPayload(std::get<1>(iov));

      std::string IOVsince = std::to_string(std::get<0>(iov));

      // Get map of strings to concatenated list of names of HLT paths:
      typedef std::map<std::string, std::string> TriggerMap;
      const TriggerMap &triggerMap = payload->m_alcarecoToTrig;

      unsigned int mapsize = triggerMap.size();
      float pitch = 1. / (mapsize * 1.1);

      float y, x1, x2;
      std::vector<float> y_x1, y_x2, y_line;
      std::vector<std::string> s_x1, s_x2, s_x3;

      // starting table at y=1.0 (top of the canvas)
      // first column is at 0.02, second column at 0.32 NDC
      y = 1.0;
      x1 = 0.02;
      x2 = x1 + 0.30;

      y -= pitch;
      y_x1.push_back(y);
      s_x1.push_back("#scale[1.2]{Key}");
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.2]{in IOV: " + IOVsince + "}");

      y -= pitch / 2.;
      y_line.push_back(y);

      for (const auto &element : triggerMap) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(element.first);

        std::vector<std::string> output;
        std::string toAppend = "";
        const std::vector<std::string> paths = payload->decompose(element.second);
        for (unsigned int iPath = 0; iPath < paths.size(); ++iPath) {
          // if the line to be added has less than 60 chars append to current
          if ((toAppend + paths[iPath]).length() < 60) {
            toAppend += paths[iPath] + ";";
          } else {
            // else if the line exceeds 60 chars, dump in the vector and resume from scratch
            output.push_back(toAppend);
            toAppend.clear();
            toAppend += paths[iPath] + ";";
          }
          // if it's the last, dump it
          if (iPath == paths.size() - 1)
            output.push_back(toAppend);
        }

        for (unsigned int br = 0; br < output.size(); br++) {
          y_x2.push_back(y);
          s_x2.push_back("#color[2]{" + output[br] + "}");
          if (br != output.size() - 1)
            y -= pitch;
        }

        y_line.push_back(y - (pitch / 2.));
      }

      TCanvas canvas("AlCaRecoTriggerBits", "AlCaRecoTriggerBits", 2000, std::max(y_x1.size(), y_x2.size()) * 40);
      TLatex l;
      // Draw the columns titles
      l.SetTextAlign(12);

      float newpitch = 1 / (std::max(y_x1.size(), y_x2.size()) * 1.1);
      float factor = newpitch / pitch;
      l.SetTextSize(newpitch - 0.002);
      canvas.cd();
      for (unsigned int i = 0; i < y_x1.size(); i++) {
        l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]) * factor, s_x1[i].c_str());
      }

      for (unsigned int i = 0; i < y_x2.size(); i++) {
        l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]) * factor, s_x2[i].c_str());
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

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

  /************************************************
    Compare AlCaRecoTriggerBits mapping
  *************************************************/
  class AlCaRecoTriggerBits_CompareBase : public cond::payloadInspector::PlotImage<AlCaRecoTriggerBits> {
  public:
    AlCaRecoTriggerBits_CompareBase()
        : cond::payloadInspector::PlotImage<AlCaRecoTriggerBits>("Table of AlCaRecoTriggerBits comparison") {}

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      std::vector<std::tuple<cond::Time_t, cond::Hash> > sorted_iovs = iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<AlCaRecoTriggerBits> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlCaRecoTriggerBits> first_payload = fetchPayload(std::get<1>(firstiov));

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
      s_x2.push_back("#scale[1.2]{in IOV: " + firstIOVsince + "}");
      y_x3.push_back(y);
      s_x3.push_back("#scale[1.2]{in IOV: " + lastIOVsince + "}");
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
        }    // if there is a common key
      }      //loop on the keys

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
      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

  class AlCaRecoTriggerBits_Compare : public AlCaRecoTriggerBits_CompareBase {
  public:
    AlCaRecoTriggerBits_Compare() : AlCaRecoTriggerBits_CompareBase() { this->setSingleIov(false); }
  };

  class AlCaRecoTriggerBits_CompareTwoTags : public AlCaRecoTriggerBits_CompareBase {
  public:
    AlCaRecoTriggerBits_CompareTwoTags() : AlCaRecoTriggerBits_CompareBase() { this->setTwoTags(true); }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(AlCaRecoTriggerBits) {
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_Display);
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_Compare);
  PAYLOAD_INSPECTOR_CLASS(AlCaRecoTriggerBits_CompareTwoTags);
}
