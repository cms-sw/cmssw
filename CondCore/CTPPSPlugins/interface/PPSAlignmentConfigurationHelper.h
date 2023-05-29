/****************************************************************************
 *
 * This is a part of PPS PI software.
 *
 ****************************************************************************/

#ifndef CONDCORE_CTPPSPLUGINS_PPSALIGNMENTCONFIGURATIONHELPER_H
#define CONDCORE_CTPPSPLUGINS_PPSALIGNMENTCONFIGURATIONHELPER_H

// User includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"

// system includes
#include <memory>
#include <sstream>
#include <fstream>

// ROOT includes
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TGraph.h"

template <class PayloadType>
class AlignmentPayloadInfo : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
public:
  AlignmentPayloadInfo()
      : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
            "PPSAlignmentConfiguration payload information") {}

  bool fill() override {
    auto tag = cond::payloadInspector::PlotBase::getTag<0>();
    auto tagname = tag.name;
    auto iov = tag.iovs.back();
    auto m_payload = this->fetchPayload(std::get<1>(iov));

    if (m_payload != nullptr) {
      std::string line;
      std::vector<std::string> lines;
      std::stringstream ss;
      ss << *m_payload;
      while (getline(ss, line)) {
        lines.push_back(line);
      }

      TCanvas canvas(
          "PPSAlignmentConfiguration payload information", "PPSAlignmentConfiguration payload information", 1000, 1400);
      canvas.cd(1);
      TLatex t;
      t.SetTextSize(0.018);

      int index = 0;
      for (float y = 0.98; index < int(lines.size()); y -= 0.02) {
        if (index < int(lines.size() / 2) + 3)
          t.DrawLatex(0.02, y, lines[index++].c_str());
        else if (index == int(lines.size() / 2) + 3) {
          y = 0.98;
          t.DrawLatex(0.5, y, lines[index++].c_str());
        } else
          t.DrawLatex(0.5, y, lines[index++].c_str());
      }
      t.Draw();

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    } else {
      return false;
    }
  }
};

#endif
