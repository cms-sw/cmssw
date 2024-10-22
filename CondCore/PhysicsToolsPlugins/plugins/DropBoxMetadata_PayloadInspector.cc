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

      DBoxMetadataHelper::recordMap theRecordMap;
      for (const auto& record : records) {
        edm::LogPrint("DropBoxMetadata_PayloadInspector") << "record: " << record << std::endl;
        const auto& parameters = payload->getRecordParameters(record);
        theRecordMap.insert(std::make_pair(record, DBoxMetadataHelper::RecordMetaDataInfo(parameters)));
      }

      DBoxMetadataHelper::DBMetaDataTableDisplay theDisplay(theRecordMap);
      theDisplay.printMetaDatas();

      DBoxMetadataHelper::DBMetaDataPlotDisplay thePlot(theRecordMap, tag.name, std::to_string(std::get<0>(iov)));
      thePlot.setImageFileName(this->m_imageFileName);
      thePlot.plotMetaDatas();

      return true;
    }
  };

  /************************************************
     DropBoxMetadata Payload Comparator of 2 IOVs 
   *************************************************/
  template <IOVMultiplicity nIOVs, int ntags>
  class DropBoxMetadata_CompareBase : public PlotImage<DropBoxMetadata, nIOVs, ntags> {
  public:
    DropBoxMetadata_CompareBase()
        : PlotImage<DropBoxMetadata, nIOVs, ntags>("DropBoxMetadata comparison of contents") {}

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

      std::shared_ptr<DropBoxMetadata> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<DropBoxMetadata> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      // first payload
      std::vector<std::string> f_records = first_payload->getAllRecords();
      DBoxMetadataHelper::recordMap f_theRecordMap;
      for (const auto& record : f_records) {
        //edm::LogPrint("DropBoxMetadata_PayloadInspector") << "record: " << record << std::endl;
        const auto& parameters = first_payload->getRecordParameters(record);
        f_theRecordMap.insert(std::make_pair(record, DBoxMetadataHelper::RecordMetaDataInfo(parameters)));
      }

      DBoxMetadataHelper::DBMetaDataTableDisplay f_theDisplay(f_theRecordMap);
      //f_theDisplay.printMetaDatas();

      // last payload
      std::vector<std::string> l_records = last_payload->getAllRecords();
      DBoxMetadataHelper::recordMap l_theRecordMap;
      for (const auto& record : l_records) {
        //edm::LogPrint("DropBoxMetadata_PayloadInspector") << "record: " << record << std::endl;
        const auto& parameters = last_payload->getRecordParameters(record);
        l_theRecordMap.insert(std::make_pair(record, DBoxMetadataHelper::RecordMetaDataInfo(parameters)));
      }

      DBoxMetadataHelper::DBMetaDataTableDisplay l_theDisplay(l_theRecordMap);
      //l_theDisplay.printMetaDatas();

      l_theDisplay.printDiffWithMetadata(f_theRecordMap);

      // In case of only one tag, use f_tagname for both target and reference
      std::string tmpTagName = l_tagname;
      if (tmpTagName.empty())
        tmpTagName = f_tagname;
      DBoxMetadataHelper::DBMetaDataPlotDisplay thePlot(l_theRecordMap, tmpTagName, lastIOVsince);
      thePlot.setImageFileName(this->m_imageFileName);
      thePlot.plotDiffWithMetadata(f_theRecordMap, f_tagname, firstIOVsince);

      return true;
    }
  };

  using DropBoxMetadata_Compare = DropBoxMetadata_CompareBase<MULTI_IOV, 1>;
  using DropBoxMetadata_CompareTwoTags = DropBoxMetadata_CompareBase<SINGLE_IOV, 2>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(DropBoxMetadata) {
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadataTest);
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadata_Display);
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadata_Compare);
  PAYLOAD_INSPECTOR_CLASS(DropBoxMetadata_CompareTwoTags);
}
