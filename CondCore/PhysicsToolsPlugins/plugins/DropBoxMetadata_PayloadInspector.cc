/*!
  \file DropBoxMetadata_PayloadInspector
  \Payload Inspector Plugin for DropBoxMetadata
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/03/18 10:01:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Common/interface/DropBoxMetadata.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system includes
#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TProfile.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TPaletteAxis.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
     DropBoxMetadata Payload Inspector of 1 IOV 
  *************************************************/
  class DropBoxMetadataTest : public Histogram1D<DropBoxMetadata, SINGLE_IOV> {
  public:
    DropBoxMetadataTest()
        : Histogram1D<DropBoxMetadata, SINGLE_IOV>("Test DropBoxMetadata", "Test DropBoxMetadata", 1, 0.0, 1.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<DropBoxMetadata> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<std::string> records = payload->getAllRecords();
          for (const auto& record : records) {
            std::cout << "record: " << record << std::endl;
            const auto& parameters = payload->getRecordParameters(record);
            const auto& recordParams = parameters.getParameterMap();
            for (const auto& [key, val] : recordParams) {
              std::cout << key << " : "
                        << " value: " << val << std::endl;
            }
          }
        }
      }
      return true;
    }
  };
}  // namespace

PAYLOAD_INSPECTOR_MODULE(DropBoxMetadata) { PAYLOAD_INSPECTOR_CLASS(DropBoxMetadataTest); }
