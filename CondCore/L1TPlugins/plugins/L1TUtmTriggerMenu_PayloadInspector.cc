/*!
  \file L1UtmTriggerMenu_PayloadInspector
  \Payload Inspector Plugin for L1UtmTriggerMenu payloads
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2023/11/15 14:49:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondCore/L1TPlugins/interface/L1TUtmTriggerMenuPayloadInspectorHelper.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  class L1TUtmTriggerMenuDisplayAlgos : public PlotImage<L1TUtmTriggerMenu, SINGLE_IOV> {
  public:
    L1TUtmTriggerMenuDisplayAlgos() : PlotImage<L1TUtmTriggerMenu, SINGLE_IOV>("L1TUtmTriggerMenu plot") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::string IOVsince = std::to_string(std::get<0>(iov));
      auto tagname = tag.name;
      std::shared_ptr<L1TUtmTriggerMenu> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        /// Create a canvas
        const auto& theMap = payload->getAlgorithmMap();

        unsigned int mapsize = theMap.size();
        float pitch = 1. / (mapsize);

        float y, x1, x2;
        std::vector<float> y_x1, y_x2, y_line;
        std::vector<std::string> s_x1, s_x2, s_x3;

        // starting table at y=1.0 (top of the canvas)
        // first column is at 0.02, second column at 0.32 NDC
        y = 1.0;
        x1 = 0.02;
        x2 = x1 + 0.15;

        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back("#scale[1.2]{Algo Name}");
        y_x2.push_back(y);
        s_x2.push_back("#scale[1.2]{tag: " + tag.name + " in IOV: " + IOVsince + "}");

        y -= pitch / 2.;
        y_line.push_back(y);

        for (const auto& [name, algo] : theMap) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back("''");

          y_x2.push_back(y);
          s_x2.push_back("#color[2]{" + name + "}");
          y_line.push_back(y - (pitch / 2.));
        }

        TCanvas canvas("L1TriggerAlgos", "L1TriggerAlgos", 2000, mapsize * 40);
        TLatex l;
        // Draw the columns titles
        l.SetTextAlign(12);
        l.SetTextSize(pitch * 10);
        canvas.cd();
        for (unsigned int i = 0; i < y_x1.size(); i++) {
          l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]), s_x1[i].c_str());
        }

        for (unsigned int i = 0; i < y_x2.size(); i++) {
          l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]), s_x2[i].c_str());
        }

        canvas.cd();
        canvas.Update();

        TLine lines[y_line.size()];
        unsigned int iL = 0;
        for (const auto& line : y_line) {
          lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line), gPad->GetUxmax(), 1 - (1 - line));
          lines[iL].SetLineWidth(1);
          lines[iL].SetLineStyle(9);
          lines[iL].SetLineColor(2);
          lines[iL].Draw("same");
          iL++;
        }

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }  // payload
      return true;
    }  // fill
  };

  template <IOVMultiplicity nIOVs, int ntags>
  class L1TUtmTriggerMenu_CompareAlgosBase : public PlotImage<L1TUtmTriggerMenu, nIOVs, ntags> {
  public:
    L1TUtmTriggerMenu_CompareAlgosBase()
        : PlotImage<L1TUtmTriggerMenu, nIOVs, ntags>("L1TUtmTriggerMenu comparison of contents") {}

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

      std::shared_ptr<L1TUtmTriggerMenu> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<L1TUtmTriggerMenu> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      // In case of only one tag, use f_tagname for both target and reference
      std::string tmpTagName = l_tagname;
      if (tmpTagName.empty())
        tmpTagName = f_tagname;

      L1TUtmTriggerMenuInspectorHelper::L1TUtmTriggerMenuDisplay thePlot(last_payload.get(), tmpTagName, lastIOVsince);
      thePlot.setImageFileName(this->m_imageFileName);
      thePlot.plotDiffWithOtherMenu(first_payload.get(), f_tagname, firstIOVsince);

      return true;
    }
  };

  using L1TUtmTriggerMenu_CompareAlgos = L1TUtmTriggerMenu_CompareAlgosBase<MULTI_IOV, 1>;
  using L1TUtmTriggerMenu_CompareAlgosTwoTags = L1TUtmTriggerMenu_CompareAlgosBase<SINGLE_IOV, 2>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(L1TUtmTriggerMenu) {
  PAYLOAD_INSPECTOR_CLASS(L1TUtmTriggerMenuDisplayAlgos);
  PAYLOAD_INSPECTOR_CLASS(L1TUtmTriggerMenu_CompareAlgos);
  PAYLOAD_INSPECTOR_CLASS(L1TUtmTriggerMenu_CompareAlgosTwoTags);
}
