/*!
  \file SiStripFedCabling_PayloadInspector
  \Payload Inspector Plugin for SiStrip Fed Cabling
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/11/02 17:05:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <fmt/format.h>
#include <memory>
#include <sstream>

#include "TCanvas.h"
#include "TH2D.h"
#include "TLatex.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    TrackerMap of SiStrip FED Cabling
  *************************************************/
  class SiStripFedCabling_TrackerMap : public PlotImage<SiStripFedCabling, SINGLE_IOV> {
  public:
    SiStripFedCabling_TrackerMap() : PlotImage<SiStripFedCabling, SINGLE_IOV>("Tracker Map SiStrip Fed Cabling") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripFedCabling> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripFedCabling");
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of SiStrip Fed Cabling per module, IOV : " + std::to_string(std::get<0>(iov));
      tmap->setTitle(titleMap);

      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
          edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());
      std::unique_ptr<SiStripDetCabling> detCabling_ = std::make_unique<SiStripDetCabling>(*(payload.get()), &tTopo);

      std::vector<uint32_t> activeDetIds;
      detCabling_->addActiveDetectorsRawIds(activeDetIds);

      for (const auto& detId : activeDetIds) {
        int32_t n_conn = 0;
        for (uint32_t connDet_i = 0; connDet_i < detCabling_->getConnections(detId).size(); connDet_i++) {
          if (detCabling_->getConnections(detId)[connDet_i] != nullptr &&
              detCabling_->getConnections(detId)[connDet_i]->isConnected() != 0)
            n_conn++;
        }
        if (n_conn != 0) {
          tmap->fill(detId, n_conn * 2);
        }
      }

      std::string fileName(m_imageFileName);
      tmap->save(true, 0., 6., fileName, 4500, 2400);  // max 6 APVs per module

      return true;
    }
  };

  /************************************************
    TrackerMap of uncabled modules
  *************************************************/
  class SiStripUncabledChannels_TrackerMap : public PlotImage<SiStripFedCabling, SINGLE_IOV> {
  public:
    SiStripUncabledChannels_TrackerMap()
        : PlotImage<SiStripFedCabling, SINGLE_IOV>("Tracker Map SiStrip Fed Cabling") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripFedCabling> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripFedCabling");
      tmap->setPalette(1);
      std::string titleMap =
          "TrackerMap of SiStrip Fraction of uncabled channels per module, IOV : " + std::to_string(std::get<0>(iov));
      tmap->setTitle(titleMap);

      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
          edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());
      std::unique_ptr<SiStripDetCabling> detCabling_ = std::make_unique<SiStripDetCabling>(*(payload.get()), &tTopo);

      std::vector<uint32_t> activeDetIds;
      detCabling_->addActiveDetectorsRawIds(activeDetIds);

      const auto detInfo =
          SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
      std::vector<uint32_t> all_detids = detInfo.getAllDetIds();

      // first add the fully unconnected modules
      for (const auto& detId : all_detids) {
        if (!detCabling_->IsConnected(detId)) {
          tmap->fill(detId, 1);
        }
      }

      // then add the partially unconnected ones
      for (const auto& detId : activeDetIds) {
        float frac = calculateConnectedFraction(detCabling_.get(), detId);
        if (frac != 1.f) {
          tmap->fill(detId, frac);
        }
      }

      std::string fileName(m_imageFileName);
      tmap->save(true, 0., 1., fileName, 4500, 2400);

      return true;
    }

  private:
    // Function to calculate the number of connections for a given detId
    int32_t calculateConnectedFraction(const SiStripDetCabling* detCabling, const uint32_t detId) {
      float totAPVs = detCabling->nApvPairs(detId);
      float n_conn{0};
      for (uint32_t connDet_i = 0; connDet_i < detCabling->getConnections(detId).size(); connDet_i++) {
        if (detCabling->getConnections(detId)[connDet_i] != nullptr &&
            detCabling->getConnections(detId)[connDet_i]->isConnected() != 0) {
          n_conn++;
        }
      }
      return n_conn / totAPVs;
    }
  };

  /************************************************
    TrackerMap of SiStrip FED Cabling difference between 2 payloads
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripFedCablingComparisonTrackerMapBase : public PlotImage<SiStripFedCabling, nIOVs, ntags> {
  public:
    SiStripFedCablingComparisonTrackerMapBase()
        : PlotImage<SiStripFedCabling, nIOVs, ntags>("Tracker Map SiStrip Fed Cabling difference") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripFedCabling> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripFedCabling> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripFedCabling Difference");
      tmap->setPalette(1);

      std::string titleMap{};
      std::string commonPart = "SiStrip Fed Cabling Map: #Delta connections per module";
      if (this->m_plotAnnotations.ntags == 2) {
        titleMap = fmt::format("{}: {} - {}", commonPart, tagname2, tagname1);
      } else {
        titleMap = fmt::format("{}: IOV : {} - {}", commonPart, std::get<0>(lastiov), std::get<0>(firstiov));
      }
      tmap->setTitle(titleMap);

      const std::string k_TrackerParams =
          edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath();
      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(k_TrackerParams);

      std::unique_ptr<SiStripDetCabling> l_detCabling =
          std::make_unique<SiStripDetCabling>(*(last_payload.get()), &tTopo);
      std::unique_ptr<SiStripDetCabling> f_detCabling =
          std::make_unique<SiStripDetCabling>(*(first_payload.get()), &tTopo);

      std::vector<uint32_t> f_activeDetIds;
      f_detCabling->addActiveDetectorsRawIds(f_activeDetIds);

      std::vector<uint32_t> l_activeDetIds;
      l_detCabling->addActiveDetectorsRawIds(l_activeDetIds);

      const auto& setsToPlot = prepareSets(f_activeDetIds, l_activeDetIds);

      edm::LogPrint("SiStripFedCablingComparisonTrackerMapBase")
          << "Common Detids: " << setsToPlot.commonElements.size()
          << " | only in last payload: " << setsToPlot.lExclusiveElements.size()
          << " | only in first payload: " << setsToPlot.fExclusiveElements.size() << std::endl;

      // Process common elements
      for (const auto& detId : setsToPlot.commonElements) {
        int32_t f_n_conn = calculateConnections(f_detCabling.get(), detId);
        int32_t l_n_conn = calculateConnections(l_detCabling.get(), detId);

        if (l_n_conn != f_n_conn) {
          tmap->fill(detId, (l_n_conn - f_n_conn) * 2);  // 2 APVs per channel
        }
      }

      // Process elements only in the last
      for (const auto& detId : setsToPlot.lExclusiveElements) {
        int32_t l_n_conn = calculateConnections(l_detCabling.get(), detId);

        if (l_n_conn != 0) {
          tmap->fill(detId, l_n_conn * 2);  // 2 APVs per channel
        }
      }

      // Process elements only in the first
      for (const auto& detId : setsToPlot.fExclusiveElements) {
        int32_t f_n_conn = calculateConnections(f_detCabling.get(), detId);

        if (f_n_conn != 0) {
          tmap->fill(detId, -f_n_conn * 2);  // 2 APVs per channel
        }
      }

      std::string fileName(this->m_imageFileName);
      tmap->save(true, -6., 6., fileName, 4500, 2400);  // max 6 APVs per module

      return true;
    }

  private:
    struct setsOfDetids {
      std::vector<uint32_t> commonElements;
      std::vector<uint32_t> fExclusiveElements;
      std::vector<uint32_t> lExclusiveElements;
    };

    // Function to calculate the number of connections for a given detId
    int32_t calculateConnections(const SiStripDetCabling* detCabling, const uint32_t detId) {
      int32_t n_conn = 0;
      for (uint32_t connDet_i = 0; connDet_i < detCabling->getConnections(detId).size(); connDet_i++) {
        if (detCabling->getConnections(detId)[connDet_i] != nullptr &&
            detCabling->getConnections(detId)[connDet_i]->isConnected() != 0) {
          n_conn++;
        }
      }
      return n_conn;
    }

    // Function to calculate the interesection and exclusive elements out of two vectors of DetIds
    setsOfDetids prepareSets(std::vector<uint32_t> f_activeDetIds, std::vector<uint32_t> l_activeDetIds) {
      setsOfDetids output;
      // Sort the input vectors if they are not already sorted
      std::sort(f_activeDetIds.begin(), f_activeDetIds.end());
      std::sort(l_activeDetIds.begin(), l_activeDetIds.end());

      // Common elements
      output.commonElements.reserve(std::min(f_activeDetIds.size(), l_activeDetIds.size()));
      std::set_intersection(f_activeDetIds.begin(),
                            f_activeDetIds.end(),
                            l_activeDetIds.begin(),
                            l_activeDetIds.end(),
                            std::back_inserter(output.commonElements));

      // Elements only in f_activeDetIds
      output.fExclusiveElements.reserve(f_activeDetIds.size() - output.commonElements.size());
      std::set_difference(f_activeDetIds.begin(),
                          f_activeDetIds.end(),
                          l_activeDetIds.begin(),
                          l_activeDetIds.end(),
                          std::back_inserter(output.fExclusiveElements));

      // Elements only in l_activeDetIds
      output.lExclusiveElements.reserve(l_activeDetIds.size() - output.commonElements.size());
      std::set_difference(l_activeDetIds.begin(),
                          l_activeDetIds.end(),
                          f_activeDetIds.begin(),
                          f_activeDetIds.end(),
                          std::back_inserter(output.lExclusiveElements));

      return output;
    }
  };

  using SiStripFedCablingComparisonTrackerMapSingleTag = SiStripFedCablingComparisonTrackerMapBase<1, MULTI_IOV>;
  using SiStripFedCablingComparisonTrackerMapTwoTags = SiStripFedCablingComparisonTrackerMapBase<2, SINGLE_IOV>;

  /************************************************
    Summary Plot of SiStrip FED Cabling
  *************************************************/
  class SiStripFedCabling_Summary : public PlotImage<SiStripFedCabling, SINGLE_IOV> {
  public:
    SiStripFedCabling_Summary() : PlotImage<SiStripFedCabling, SINGLE_IOV>("SiStrip Fed Cabling Summary") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripFedCabling> payload = fetchPayload(std::get<1>(iov));
      int IOV = std::get<0>(iov);
      std::vector<uint32_t> activeDetIds;

      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
          edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());
      std::unique_ptr<SiStripDetCabling> detCabling_ = std::make_unique<SiStripDetCabling>(*(payload.get()), &tTopo);

      detCabling_->addActiveDetectorsRawIds(activeDetIds);

      containers myCont;
      containers allCounts;

      const auto detInfo =
          SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
      for (const auto& it : detInfo.getAllData()) {
        // check if det id is correct and if it is actually cabled in the detector
        if (it.first == 0 || it.first == 0xFFFFFFFF) {
          edm::LogError("DetIdNotGood") << "@SUB=analyze"
                                        << "Wrong det id: " << it.first << "  ... neglecting!" << std::endl;
          continue;
        }
        updateCounters(it.first, allCounts, tTopo);
      }

      for (const auto& detId : activeDetIds) {
        updateCounters(detId, myCont, tTopo);
      }

      TH2D* ME = new TH2D("SummaryOfCabling", "SummaryOfCabling", 6, 0.5, 6.5, 9, 0.5, 9.5);
      ME->GetXaxis()->SetTitle("Sub Det");
      ME->GetYaxis()->SetTitle("Layer");

      ME->SetTitle("");

      ME->GetXaxis()->SetBinLabel(1, "TIB");
      ME->GetXaxis()->SetBinLabel(2, "TID F");
      ME->GetXaxis()->SetBinLabel(3, "TID B");
      ME->GetXaxis()->SetBinLabel(4, "TOB");
      ME->GetXaxis()->SetBinLabel(5, "TEC F");
      ME->GetXaxis()->SetBinLabel(6, "TEC B");

      for (int i = 0; i < 4; i++) {
        ME->Fill(1, i + 1, float(myCont.counterTIB[i]) / allCounts.counterTIB[i]);
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          ME->Fill(i + 2, j + 1, float(myCont.counterTID[i][j]) / allCounts.counterTID[i][j]);
        }
      }

      for (int i = 0; i < 6; i++) {
        ME->Fill(4, i + 1, float(myCont.counterTOB[i]) / allCounts.counterTOB[i]);
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 9; j++) {
          ME->Fill(i + 5, j + 1, float(myCont.counterTEC[i][j]) / allCounts.counterTEC[i][j]);
        }
      }

      TCanvas c1("SiStrip FED cabling summary", "SiStrip FED cabling summary", 800, 600);
      c1.SetTopMargin(0.07);
      c1.SetBottomMargin(0.10);
      c1.SetLeftMargin(0.07);
      c1.SetRightMargin(0.10);

      ME->Draw("colz");
      ME->Draw("TEXTsame");
      ME->SetStats(kFALSE);

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("SiStrip FedCabling, IOV %i", IOV));

      std::string fileName(m_imageFileName);
      c1.SaveAs(fileName.c_str());

      return true;
    }

  private:
    struct containers {
    public:
      int counterTIB[4] = {0};
      int counterTID[2][3] = {{0}};
      int counterTOB[6] = {0};
      int counterTEC[2][9] = {{0}};
    };

    void updateCounters(int detId, containers& cont, const TrackerTopology& tTopo) {
      StripSubdetector subdet(detId);

      switch (subdet.subdetId()) {
        case StripSubdetector::TIB: {
          int i = tTopo.tibLayer(detId) - 1;
          cont.counterTIB[i]++;
          break;
        }
        case StripSubdetector::TID: {
          int j = tTopo.tidWheel(detId) - 1;
          int side = tTopo.tidSide(detId);
          if (side == 2) {
            cont.counterTID[0][j]++;
          } else if (side == 1) {
            cont.counterTID[1][j]++;
          }
          break;
        }
        case StripSubdetector::TOB: {
          int i = tTopo.tobLayer(detId) - 1;
          cont.counterTOB[i]++;
          break;
        }
        case StripSubdetector::TEC: {
          int j = tTopo.tecWheel(detId) - 1;
          int side = tTopo.tecSide(detId);
          if (side == 2) {
            cont.counterTEC[0][j]++;
          } else if (side == 1) {
            cont.counterTEC[1][j]++;
          }
          break;
        }
      }
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripFedCabling) {
  PAYLOAD_INSPECTOR_CLASS(SiStripFedCabling_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripUncabledChannels_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripFedCablingComparisonTrackerMapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripFedCablingComparisonTrackerMapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripFedCabling_Summary);
}
