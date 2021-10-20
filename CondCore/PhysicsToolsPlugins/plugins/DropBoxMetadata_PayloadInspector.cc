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

// helper classes
#include "CondCore/PhysicsToolsPlugins/interface/DropBoxMetaDataPayloadInspectorHelper.h"

// system includes
#include <memory>
#include <sstream>
#include <iostream>
#include <boost/algorithm/string/replace.hpp>

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
            edm::LogPrint("DropBoxMetadata_PayloadInspector") << "record: " << record << std::endl;
            const auto& parameters = payload->getRecordParameters(record);
            const auto& recordParams = parameters.getParameterMap();
            for (const auto& [key, val] : recordParams) {
              if (val.find("&quot;") != std::string::npos) {
                const auto& replaced = replaceAll(val, std::string("&quot;"), std::string("'"));
                edm::LogPrint("DropBoxMetadata_PayloadInspector") << key << " : " << replaced << std::endl;
              } else {
                edm::LogPrint("DropBoxMetadata_PayloadInspector") << key << " : " << val << std::endl;
              }
            }
          }
        }
      }
      return true;
    }

  private:
    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Handles case where 'to' is a substring of 'from'
      }
      return str;
    }
  };

  /************************************************
     DropBoxMetadata Payload Inspector of 1 IOV 
  *************************************************/
  class DropBoxMetadata_Display : public PlotImage<DropBoxMetadata, SINGLE_IOV> {
  public:
    DropBoxMetadata_Display() : PlotImage<DropBoxMetadata, SINGLE_IOV>("DropBoxMetadata Display of contents") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<DropBoxMetadata> payload = fetchPayload(std::get<1>(iov));

      std::vector<std::string> records = payload->getAllRecords();
      TCanvas canvas("Canv", "Canv", 1200, 100 * records.size());

      DPMetaDataHelper::recordMap theRecordMap;
      for (const auto& record : records) {
        edm::LogPrint("DropBoxMetadata_PayloadInspector") << "record: " << record << std::endl;
        const auto& parameters = payload->getRecordParameters(record);
        theRecordMap.insert(std::make_pair(record, DPMetaDataHelper::RecordMetaDataInfo(parameters)));
      }

      DPMetaDataHelper::DBMetaDataTableDisplay theDisplay(theRecordMap);
      theDisplay.printMetaDatas();

      //const auto& recordParams = parameters.getParameterMap();
      // for (const auto& [key, val] : recordParams) {
      //   if (val.find("&quot;") != std::string::npos) {
      //     const auto& replaced = replaceAll(val, std::string("&quot;"), std::string("'"));
      //     edm::LogPrint("DropBoxMetadata_PayloadInspector") << key << " : " << replaced << std::endl;
      //   } else {
      //     edm::LogPrint("DropBoxMetadata_PayloadInspector") << key << " : " << val << std::endl;
      //   }
      // }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }

  private:
  };
}  // namespace

PAYLOAD_INSPECTOR_MODULE(DropBoxMetadata) {
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadataTest);
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadata_Display);
}
