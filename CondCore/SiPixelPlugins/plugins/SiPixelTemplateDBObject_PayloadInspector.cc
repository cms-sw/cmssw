/*!
  \file SiPixelTemplateDBObject_PayloadInspector
  \Payload Inspector Plugin for SiPixelTemplateDBObject
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/16 18:00:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  /************************************************
    test class
  *************************************************/

  class SiPixelTemplateDBObjectTest : public cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject> {
  public:
    SiPixelTemplateDBObjectTest()
        : cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject>(
              "SiPixelTemplateDBObject test", "SiPixelTemplateDBObject test", 10, 0.0, 100.) {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto const& iov : iovs) {
        std::vector<SiPixelTemplateStore> thePixelTemp_;

        std::shared_ptr<SiPixelTemplateDBObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!SiPixelTemplate::pushfile(*payload, thePixelTemp_)) {
            throw cms::Exception("") << "\nERROR: Templates not filled correctly. Check the sqlite file. Using "
                                        "SiPixelTemplateDBObject version "
                                     << payload->version() << "\n\n";
          }

          SiPixelTemplate templ(thePixelTemp_);

          std::map<unsigned int, short> templMap = payload->getTemplateIDs();
          for (auto const& entry : templMap) {
            std::cout << "DetID: " << entry.first << " template ID: " << entry.second << std::endl;
            templ.interpolate(entry.second, 0.f, 0.f, 1.f, 1.f);

            std::cout << "\t lorywidth  " << templ.lorywidth() << " lorxwidth: " << templ.lorxwidth() << " lorybias "
                      << templ.lorybias() << " lorxbias: " << templ.lorxbias() << "\n"
                      << std::endl;
          }

          fillWithValue(1.);

        }  // payload
      }    // iovs
      return true;
    }  // fill
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelTemplateDBObject) { PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateDBObjectTest); }
