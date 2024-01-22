#ifndef CONDCORE_CTPPSPLUGINS_PPSDAQMAPPINGPAYLOADINSPECTORHELPER_H
#define CONDCORE_CTPPSPLUGINS_PPSDAQMAPPINGPAYLOADINSPECTORHELPER_H

// User includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

// system includes
#include <memory>
#include <sstream>

// ROOT includes
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TGraph.h"

namespace DAQMappingPI {
  inline std::string resolveDetIDForDAQMapping(int detIDNumber) {
    static const std::map<int, std::string> mapping = {{CTPPSDetId::SubDetector::sdTrackingStrip, "Strip"},
                                                       {CTPPSDetId::SubDetector::sdTrackingPixel, "Pixel"},
                                                       {CTPPSDetId::SubDetector::sdTimingDiamond, "Diamond"},
                                                       {CTPPSDetId::SubDetector::sdTimingFastSilicon, "FastSilicon"},
                                                       {CTPPSDetId::SubDetector::sdTotemT2, "TotemT2"}};

    auto it = mapping.find(detIDNumber);
    if (it != mapping.end()) {
      return it->second;
    } else {
      return "not defined";
    }
  }
}  // namespace DAQMappingPI

template <class PayloadType>
class DAQMappingPayloadInfo
    : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
public:
  DAQMappingPayloadInfo()
      : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
            "DAQMappingPayloadInfo text") {}

  bool fill() override {
    auto tag = cond::payloadInspector::PlotBase::getTag<0>();
    auto tagname = tag.name;
    auto iov = tag.iovs.back();
    auto m_payload = this->fetchPayload(std::get<1>(iov));

    if (m_payload != nullptr) {
      std::stringstream payloadInfo, lineCountStream;
      int subDet = CTPPSDetId(m_payload->VFATMapping.begin()->second.symbolicID.symbolicID).subdetId();
      payloadInfo << "TAG: " << tagname << ", the mapping for: " << DAQMappingPI::resolveDetIDForDAQMapping(subDet)
                  << std::endl;
      payloadInfo << *m_payload;
      lineCountStream << *m_payload;
      std::string line;

      //created to dynamically set canvas height
      int lineCounter = 0;
      while (std::getline(lineCountStream, line)) {
        lineCounter++;
      }

      TCanvas canvas("canvas", "Canvas", 800, 20 * lineCounter);

      TLatex latex;
      latex.SetNDC();
      latex.SetTextSize(0.015);
      double yPos = 0.95;

      while (std::getline(payloadInfo, line)) {
        yPos -= 0.015;
        latex.DrawLatex(0.1, yPos, line.c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    } else {
      return false;
    }
  }
};

#endif
