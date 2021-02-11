#ifndef CONDCORE_SIPIXELPLUGINS_SIPIXELTEMPLATEHELPER_H
#define CONDCORE_SIPIXELPLUGINS_SIPIXELTEMPLATEHELPER_H

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"

#include <type_traits>
#include <memory>
#include <sstream>
#include <fmt/printf.h>

// include ROOT
#include "TH2F.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace templateHelper {

  //************************************************
  // Display of Template Titles
  // *************************************************/
  template <class PayloadType, class StoreType, class TransientType>
  class SiPixelTitles_Display
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelTitles_Display()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>("Table of Template titles") {
      if constexpr (std::is_same_v<PayloadType, SiPixelTemplateDBObject>) {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      } else {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
       }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::vector<StoreType> thePixelTemp_;
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      std::string IOVsince = std::to_string(std::get<0>(iov));

      if (payload.get()) {
        if (!TransientType::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception(label_)
              << "\nERROR: Templates not filled correctly. Check the conditions. Using "
	      << (isTemplate_ ? "SiPixelTemplateDBObject" : "SiPixelGenErrorDBObject")
	      << " version "
              << payload->version() << "\n\n";
        }

        unsigned int mapsize = thePixelTemp_.size();
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
        s_x1.push_back(Form("#scale[1.2]{%s}",( isTemplate_ ? "Template ID" : "GenError ID" ) ));
        y_x2.push_back(y);
        s_x2.push_back(Form("#scale[1.2]{#color[4]{%s} in IOV: #color[4]{%s}}",tagname.c_str(),IOVsince.c_str()));

        y -= pitch / 2.;
        y_line.push_back(y);

        for (const auto& element : thePixelTemp_) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back(std::to_string(element.head.ID));

          y_x2.push_back(y);
          s_x2.push_back(Form("#color[2]{%s}", element.head.title));

          y_line.push_back(y - (pitch / 2.));
        }

        TCanvas canvas("Template Titles", "Template Titles", 2000, std::max(y_x1.size(), y_x2.size()) * 40);
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
        for (const auto& line : y_line) {
          lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
          lines[iL].SetLineWidth(1);
          lines[iL].SetLineStyle(9);
          lines[iL].SetLineColor(2);
          lines[iL].Draw("same");
          iL++;
        }

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

      }  // if paylaod.get()
      return true;
    }

  protected:
    bool isTemplate_;
    std::string label_;
  };
  
}  // namespace templatebHelper

#endif
