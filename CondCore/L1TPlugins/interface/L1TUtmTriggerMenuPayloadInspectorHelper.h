#ifndef CondCore_L1TPlugins_L1TUtmTriggerMenuPayloadInspectorHelper_H
#define CondCore_L1TPlugins_L1TUtmTriggerMenuPayloadInspectorHelper_H

#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLine.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

namespace L1TUtmTriggerMenuInspectorHelper {

  using l1tUtmAlgoMap = std::map<std::string, L1TUtmAlgorithm>;
  using l1tUtmConditionMap = std::map<std::string, L1TUtmCondition>;

  class L1UtmTriggerMenuInfo {
  public:
    // constructor
    L1UtmTriggerMenuInfo(const L1TUtmTriggerMenu* l1utmMenu) {
      m_algoMap = l1utmMenu->getAlgorithmMap();
      m_condMap = l1utmMenu->getConditionMap();
    }

    // destructor
    ~L1UtmTriggerMenuInfo() = default;

  public:
    //___________________________________________________________________
    const std::vector<std::string> listOfAlgos() const {
      std::vector<std::string> output;
      std::transform(m_algoMap.begin(),
                     m_algoMap.end(),
                     std::back_inserter(output),
                     [](const std::pair<std::string, L1TUtmAlgorithm>& pair) {
                       return pair.first;  // Extracting the string key using lambda
                     });
      return output;
    }

    //___________________________________________________________________
    const std::vector<std::string> listOfConditions() const {
      std::vector<std::string> output;
      std::transform(m_condMap.begin(),
                     m_condMap.end(),
                     std::back_inserter(output),
                     [](const std::pair<std::string, L1TUtmCondition>& pair) {
                       return pair.first;  // Extracting the string key using lambda
                     });
      return output;
    }

    //___________________________________________________________________
    template <typename T>
    const std::vector<std::string> listOfCommonKeys(const L1TUtmTriggerMenu* other) const {
      const auto& otherMap = getOtherMap<T>(other);
      const auto& thisMap = getThisMap<T>();

      std::vector<std::string> commonKeys;

      // Lambda function to find common keys and store them in commonKeys vector
      std::for_each(thisMap.begin(), thisMap.end(), [&commonKeys, &otherMap](const std::pair<std::string, T>& pair) {
        const std::string& key = pair.first;

        // Check if the key exists in map2
        if (otherMap.find(key) != otherMap.end()) {
          commonKeys.push_back(key);
        }
      });
      return commonKeys;
    }

    //___________________________________________________________________
    template <typename T>
    const std::vector<std::string> onlyInThis(const L1TUtmTriggerMenu* other) const {
      const auto& otherMap = getOtherMap<T>(other);
      const auto& thisMap = getThisMap<T>();

      std::vector<std::string> stringsOnlyInFirstMap;

      // Lambda function to extract only the strings present in thisMap but not in otherMap
      std::for_each(
          thisMap.begin(), thisMap.end(), [&stringsOnlyInFirstMap, &otherMap](const std::pair<std::string, T>& pair) {
            const std::string& key = pair.first;
            // Check if the key exists in otherMap
            if (otherMap.find(key) == otherMap.end()) {
              stringsOnlyInFirstMap.push_back(key);  // Add key to the vector
            }
          });

      return stringsOnlyInFirstMap;
    }

    //___________________________________________________________________
    template <typename T>
    const std::vector<std::string> onlyInOther(const L1TUtmTriggerMenu* other) const {
      const auto& otherMap = getOtherMap<T>(other);
      const auto& thisMap = getThisMap<T>();

      std::vector<std::string> stringsOnlyInSecondMap;

      // Lambda function capturing 'this' to access the member variable 'thisMap'
      std::for_each(
          otherMap.begin(), otherMap.end(), [thisMap, &stringsOnlyInSecondMap](const std::pair<std::string, T>& pair) {
            const std::string& key = pair.first;

            // Check if the key exists in thisMap
            if (thisMap.find(key) == thisMap.end()) {
              stringsOnlyInSecondMap.push_back(key);  // Add key to the vector
            }
          });

      return stringsOnlyInSecondMap;
    }

  private:
    l1tUtmAlgoMap m_algoMap;
    l1tUtmConditionMap m_condMap;

    // Helper function to get otherMap based on T
    template <typename T>
    decltype(auto) getOtherMap(const L1TUtmTriggerMenu* other) const {
      if constexpr (std::is_same<T, L1TUtmCondition>::value) {
        return other->getConditionMap();
      } else {
        return other->getAlgorithmMap();
      }
    }

    // Helper function to get this Map based on T
    template <typename T>
    decltype(auto) getThisMap() const {
      if constexpr (std::is_same<T, L1TUtmCondition>::value) {
        return m_condMap;
      } else {
        return m_algoMap;
      }
    }
  };

  template <typename T>
  class L1TUtmTriggerMenuDisplay {
  public:
    L1TUtmTriggerMenuDisplay(const L1TUtmTriggerMenu* thisMenu, std::string theTag, std::string theIOV)
        : m_info(thisMenu), m_tagName(theTag), m_IOVsinceDisplay(theIOV) {}
    ~L1TUtmTriggerMenuDisplay() = default;

    void setImageFileName(const std::string& theFileName) {
      m_imageFileName = theFileName;
      return;
    }

    // Function to set label based on the type T
    std::string getLabel() const;

    //___________________________________________________________________
    void plotDiffWithOtherMenu(const L1TUtmTriggerMenu* other, std::string theRefTag, std::string theRefIOV) {
      const auto& vec_only_in_this = m_info.template onlyInThis<T>(other);
      const auto& vec_only_in_other = m_info.template onlyInOther<T>(other);

      // preparations for plotting
      // starting table at y=1.0 (top of the canvas)
      // first column is at 0.03, second column at 0.22 NDC
      unsigned int mapsize = vec_only_in_this.size() + vec_only_in_other.size();
      float pitch = 1. / (mapsize * 1.1);
      float y, x1, x2;
      std::vector<float> y_x1, y_x2, y_line;
      std::vector<std::string> s_x1, s_x2, s_x3;
      y = 1.0;
      x1 = 0.02;
      x2 = x1 + 0.37;
      y -= pitch;

      // title for plot
      y_x1.push_back(y);
      s_x1.push_back(getLabel());
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.1]{Target tag / IOV: #color[2]{" + m_tagName + "} / " + m_IOVsinceDisplay + "}");

      y -= pitch;
      y_x1.push_back(y);
      s_x1.push_back("");
      y_x2.push_back(y);
      s_x2.push_back("#scale[1.1]{Refer  tag / IOV: #color[4]{" + theRefTag + "} / " + theRefIOV + "}");

      y -= pitch / 2.;
      y_line.push_back(y);

      // First, check if there are records in reference which are not in target
      for (const auto& ref : vec_only_in_other) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back("#scale[0.7]{" + ref + "}");
        y_x2.push_back(y);
        s_x2.push_back("#color[4]{#bf{Only in reference, not in target.}}");
        y_line.push_back(y - (pitch / 2.));
      }

      // Second, check if there are records in target which are not in reference
      for (const auto& tar : vec_only_in_this) {
        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back("#scale[0.7]{" + tar + "}");
        y_x2.push_back(y);
        s_x2.push_back("#color[2]{#bf{Only in target, not in reference.}}");
        y_line.push_back(y - (pitch / 2.));
      }

      // Finally, print text to TCanvas
      TCanvas canvas("L1TUtmMenuData", "L1TUtmMenuData", 2000, std::max(y_x1.size(), y_x2.size()) * 40);
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

      // Draw horizontal lines separating records
      TLine lines[y_line.size()];
      unsigned int iL = 0;
      for (const auto& line : y_line) {
        lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
        lines[iL].SetLineWidth(1);
        lines[iL].SetLineStyle(9);
        lines[iL].SetLineColor(2);
        lines[iL].Draw("same");
        iL++;
      }

      std::string fileName("L1UtmMenuData_Compare.png");
      if (!m_imageFileName.empty())
        fileName = m_imageFileName;
      canvas.SaveAs(fileName.c_str());
    }

  private:
    L1UtmTriggerMenuInfo m_info;    //!< map of the record / metadata associations
    std::string m_tagName;          //!< tag name
    std::string m_IOVsinceDisplay;  //!< iov since
    std::string m_imageFileName;    //!< image file name
  };

  // Explicit specialization outside the class
  template <>
  std::string L1TUtmTriggerMenuDisplay<L1TUtmCondition>::getLabel() const {
    return "#scale[1.1]{Condition Name}";
  }

  template <>
  std::string L1TUtmTriggerMenuDisplay<L1TUtmAlgorithm>::getLabel() const {
    return "#scale[1.1]{Algo Name}";
  }
}  // namespace L1TUtmTriggerMenuInspectorHelper

#endif
