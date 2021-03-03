/*!
  \file SiPixelFEDChannelContainer_PayloadInspector
  \Payload Inspector Plugin for SiPixelFEDChannelContainer
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/02/22 10:00:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <fmt/printf.h>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
  1d histogram of SiPixelFEDChannelContainer of 1 IOV 
  *************************************************/

  class SiPixelFEDChannelContainerTest : public PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV> {
  public:
    SiPixelFEDChannelContainerTest()
        : PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV>("SiPixelFEDChannelContainer scenarios count"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
      // for inputs
      PlotBase::addInputParam("Scenarios");

      // hardcoded connection to the MC cabling tag, though luck
      m_condDbCabling = "frontier://FrontierProd/CMS_CONDITIONS";
      m_CablingTagName = "SiPixelFedCablingMap_phase1_v7";

      m_connectionPool.setParameters(m_connectionPset);
      m_connectionPool.configure();
    }

    bool fill() override {
      std::vector<std::string> the_scenarios = {};

      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("Scenarios");
      if (ip != paramValues.end()) {
        auto input = boost::lexical_cast<std::string>(ip->second);
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        boost::char_separator<char> sep{","};
        tokenizer tok{input, sep};
        for (const auto& t : tok) {
          the_scenarios.push_back(t);
        }
      } else {
        edm::LogWarning("SiPixelFEDChannelContainerTest")
            << "\n WARNING!!!! \n The needed parameter Scenarios has not been passed. Will use all the scenarios in "
               "the file!"
            << "\n Buckle your seatbelts... this might take a while... \n\n";
        the_scenarios.push_back("all");
      }

      int nlad_list[n_layers] = {6, 14, 22, 32};
      int divide_roc = 1;

      // ---------------------    BOOK HISTOGRAMS
      std::array<TH2D*, n_layers> h_bpix_occ;
      std::array<TH2D*, n_rings> h_fpix_occ;

      // barrel
      for (unsigned int lay = 1; lay <= 4; lay++) {
        int nlad = nlad_list[lay - 1];

        std::string name = "occ_Layer_" + std::to_string(lay);
        std::string title = "; Module # ; Ladder #";
        h_bpix_occ[lay - 1] = new TH2D(name.c_str(),
                                       title.c_str(),
                                       72 * divide_roc,
                                       -4.5,
                                       4.5,
                                       (nlad * 4 + 2) * divide_roc,
                                       -nlad - 0.5,
                                       nlad + 0.5);
      }

      // endcaps
      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        int n = ring == 1 ? 92 : 140;
        float y = ring == 1 ? 11.5 : 17.5;
        std::string name = "occ_ring_" + std::to_string(ring);
        std::string title = "; Disk # ; Blade/Panel #";

        h_fpix_occ[ring - 1] = new TH2D(name.c_str(), title.c_str(), 56 * divide_roc, -3.5, 3.5, n * divide_roc, -y, y);
      }

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      // open db session for the cabling map
      edm::LogPrint("SiPixelFEDChannelContainerTest") << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
                                                      << "Query the condition database " << m_condDbCabling;

      cond::persistency::Session condDbSession = m_connectionPool.createSession(m_condDbCabling);
      condDbSession.transaction().start(true);

      // query the database
      edm::LogPrint("SiPixelFEDChannelContainerTest") << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
                                                      << "Reading IOVs from tag " << m_CablingTagName;

      const auto MIN_VAL = cond::timeTypeSpecs[cond::runnumber].beginValue;
      const auto MAX_VAL = cond::timeTypeSpecs[cond::runnumber].endValue;

      // get the list of payloads for the Cabling Map
      std::vector<std::tuple<cond::Time_t, cond::Hash>> m_cabling_iovs;
      condDbSession.readIov(m_CablingTagName).selectRange(MIN_VAL, MAX_VAL, m_cabling_iovs);

      std::vector<unsigned int> listOfCablingIOVs;
      std::transform(m_cabling_iovs.begin(),
                     m_cabling_iovs.end(),
                     std::back_inserter(listOfCablingIOVs),
                     [](std::tuple<cond::Time_t, cond::Hash> myIOV2) -> unsigned int { return std::get<0>(myIOV2); });

      edm::LogPrint("SiPixelFEDChannelContainerTest")
          << " Number of SiPixelFedCablngMap payloads: " << listOfCablingIOVs.size() << std::endl;

      auto it = std::find(
          listOfCablingIOVs.begin(), listOfCablingIOVs.end(), closest_from_below(listOfCablingIOVs, std::get<0>(iov)));
      int index = std::distance(listOfCablingIOVs.begin(), it);

      edm::LogPrint("SiPixelFEDChannelContainerTest")
          << " using the SiPixelFedCablingMap with hash: " << std::get<1>(m_cabling_iovs.at(index)) << std::endl;

      auto theCablingMap = condDbSession.fetchPayload<SiPixelFedCablingMap>(std::get<1>(m_cabling_iovs.at(index)));
      theCablingMap->initializeRocs();
      // auto theCablingTree = (*theCablingMap).cablingTree();

      //auto map = theCablingMap->det2fedMap();
      //for (const auto &element : map){
      //	std::cout << element.first << " " << element.second << std::endl;
      //}

      std::shared_ptr<SiPixelFEDChannelContainer> payload = fetchPayload(std::get<1>(iov));
      const auto& scenarioMap = payload->getScenarioMap();

      auto pIndexConverter = PixelIndices(numColumns, numRows);

      for (const auto& scenario : scenarioMap) {
        std::string scenName = scenario.first;

        if (std::find_if(the_scenarios.begin(), the_scenarios.end(), compareKeys(scenName)) != the_scenarios.end()) {
          edm::LogPrint("SiPixelFEDChannelContainerTest") << "\t Found Scenario: " << scenName << " ==> dumping it";
        } else {
          continue;
        }

        //if (strcmp(scenName.c_str(),"320824_103") != 0) continue;

        const auto& theDetSetBadPixelFedChannels = payload->getDetSetBadPixelFedChannels(scenName);
        for (const auto& disabledChannels : *theDetSetBadPixelFedChannels) {
          const auto t_detid = disabledChannels.detId();
          int subid = DetId(t_detid).subdetId();
          LogDebug("SiPixelFEDChannelContainerTest") << fmt::sprintf("DetId : %i \n", t_detid) << std::endl;

          std::bitset<16> badRocsFromFEDChannels;

          for (const auto& ch : disabledChannels) {
            std::string toOut_ = fmt::sprintf("fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                                              ch.fed,
                                              ch.link,
                                              ch.roc_first,
                                              ch.roc_last);

            LogDebug("SiPixelFEDChannelContainerTest") << toOut_ << std::endl;
            const std::vector<sipixelobjects::CablingPathToDetUnit>& path =
                theCablingMap->pathToDetUnit(disabledChannels.detId());
            for (unsigned int i_roc = ch.roc_first; i_roc <= ch.roc_last; ++i_roc) {
              for (const auto p : path) {
                const sipixelobjects::PixelROC* myroc = theCablingMap->findItem(p);
                if (myroc->idInDetUnit() == static_cast<unsigned int>(i_roc)) {
                  sipixelobjects::LocalPixel::RocRowCol local = {39, 25};  //corresponding to center of ROC row,col
                  sipixelobjects::GlobalPixel global = myroc->toGlobal(sipixelobjects::LocalPixel(local));
                  int chipIndex(0), colROC(0), rowROC(0);

                  pIndexConverter.transformToROC(global.col, global.row, chipIndex, colROC, rowROC);

                  LogDebug("SiPixelFEDChannelContainerTest")
                      << " => i_roc:" << i_roc << "  " << global.col << "-" << global.row << " | => " << chipIndex
                      << " : (" << colROC << "," << rowROC << ")" << std::endl;

                  badRocsFromFEDChannels[chipIndex] = true;
                }
              }
            }
          }

          LogDebug("SiPixelFEDChannelContainerTest") << badRocsFromFEDChannels << std::endl;

          auto myDetId = DetId(t_detid);

          if (subid == PixelSubdetector::PixelBarrel) {
            auto layer = m_trackerTopo.pxbLayer(myDetId);
            auto s_ladder = SiPixelPI::signed_ladder(myDetId, m_trackerTopo, true);
            auto s_module = SiPixelPI::signed_module(myDetId, m_trackerTopo, true);

            bool isFlipped = SiPixelPI::isBPixOuterLadder(myDetId, m_trackerTopo, false);
            if ((layer > 1 && s_module < 0))
              isFlipped = !isFlipped;

            auto ladder = m_trackerTopo.pxbLadder(myDetId);
            auto module = m_trackerTopo.pxbModule(myDetId);
            LogDebug("SiPixelFEDChannelContainerTest")
                << "layer:" << layer << " ladder:" << ladder << " module:" << module << " signed ladder: " << s_ladder
                << " signed module: " << s_module << std::endl;

            auto rocsToMask =
                SiPixelPI::maskedBarrelRocsToBins(layer, s_ladder, s_module, badRocsFromFEDChannels, isFlipped);
            for (const auto& bin : rocsToMask) {
              double x = h_bpix_occ[layer - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
              double y = h_bpix_occ[layer - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
              h_bpix_occ[layer - 1]->Fill(x, y, 1);
            }
          }  // if it's barrel
          else if (subid == PixelSubdetector::PixelEndcap) {
            auto ring = SiPixelPI::ring(myDetId, m_trackerTopo, true);
            auto s_blade = SiPixelPI::signed_blade(myDetId, m_trackerTopo, true);
            auto s_disk = SiPixelPI::signed_disk(myDetId, m_trackerTopo, true);
            auto s_blade_panel = SiPixelPI::signed_blade_panel(myDetId, m_trackerTopo, true);
            auto panel = m_trackerTopo.pxfPanel(t_detid);

            //bool isFlipped = (s_disk > 0) ? (std::abs(s_blade)%2==0) : (std::abs(s_blade)%2==1);
            bool isFlipped = (s_disk > 0) ? (panel == 1) : (panel == 2);

            LogDebug("SiPixelFEDChannelContainerTest")
                << "ring:" << ring << " blade: " << s_blade << " panel: " << panel
                << " signed blade/panel: " << s_blade_panel << " disk: " << s_disk << std::endl;

            auto rocsToMask =
                SiPixelPI::maskedForwardRocsToBins(ring, s_blade, panel, s_disk, badRocsFromFEDChannels, isFlipped);
            for (const auto& bin : rocsToMask) {
              double x = h_fpix_occ[ring - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
              double y = h_fpix_occ[ring - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
              h_fpix_occ[ring - 1]->SetBinContent(x, y, 1);
            }
          }  // if it's endcap
        }    // loop on the channels
      }      // loop on the scenarios

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 1600);
      canvas.Divide(2, 3);
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      // dress the plots
      for (unsigned int lay = 1; lay <= n_layers; lay++) {
        SiPixelPI::dress_occup_plot(canvas, h_bpix_occ[lay - 1], lay, 0, 1);
      }

      canvas.Update();
      canvas.Modified();
      canvas.cd();

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        SiPixelPI::dress_occup_plot(canvas, h_fpix_occ[ring - 1], 0, n_layers + ring, 1);
      }

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int lay = 1; lay <= n_layers; lay++) {
        canvas.cd(lay);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.055);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        canvas.cd(n_layers + ring);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.050);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      // close the DB session
      condDbSession.transaction().commit();

      return true;
    }

  public:
    inline unsigned int closest_from_above(std::vector<unsigned int> const& vec, unsigned int value) {
      auto const it = std::lower_bound(vec.begin(), vec.end(), value);
      return vec.at(it - vec.begin() - 1);
    }

    inline unsigned int closest_from_below(std::vector<unsigned int> const& vec, unsigned int value) {
      auto const it = std::upper_bound(vec.begin(), vec.end(), value);
      return vec.at(it - vec.begin() - 1);
    }

    // auxilliary check
    struct compareKeys {
      std::string key;
      compareKeys(std::string const& i) : key(i) {}

      bool operator()(std::string const& i) { return (key == i); }
    };

  private:
    // tough luck, we can only do phase-1...
    static constexpr int numColumns = 416;
    static constexpr int numRows = 160;
    static constexpr int n_rings = 2;
    static constexpr int n_layers = 4;

    TrackerTopology m_trackerTopo;
    edm::ParameterSet m_connectionPset;
    cond::persistency::ConnectionPool m_connectionPool;
    std::string m_CablingTagName;
    std::string m_condDbCabling;
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelFEDChannelContainer) { PAYLOAD_INSPECTOR_CLASS(SiPixelFEDChannelContainerTest); }
