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
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"  // to display aggregate probability
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"

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

  template <SiPixelPI::DetType myType>
  class SiPixelFEDChannelContainerMap : public PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV> {
  public:
    SiPixelFEDChannelContainerMap()
        : PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV>(
              "SiPixelFEDChannelContainer Pixel Track Map of one (or more scenarios"),
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
        auto input = ip->second;
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        boost::char_separator<char> sep{","};
        tokenizer tok{input, sep};
        for (const auto& t : tok) {
          the_scenarios.push_back(t);
        }
      } else {
        edm::LogWarning(k_ClassName)
            << "\n WARNING!!!! \n The needed parameter Scenarios has not been passed. Will use all the scenarios in "
               "the file!"
            << "\n Buckle your seatbelts... this might take a while... \n\n";
        the_scenarios.push_back("all");
      }

      Phase1PixelROCMaps theROCMap("");

      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      // open db session for the cabling map
      edm::LogPrint(k_ClassName) << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
                                 << "Query the condition database " << m_condDbCabling;

      cond::persistency::Session condDbSession = m_connectionPool.createSession(m_condDbCabling);
      condDbSession.transaction().start(true);

      // query the database
      edm::LogPrint(k_ClassName) << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
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

      edm::LogPrint(k_ClassName) << " Number of SiPixelFedCablngMap payloads: " << listOfCablingIOVs.size()
                                 << std::endl;

      auto it = std::find(
          listOfCablingIOVs.begin(), listOfCablingIOVs.end(), closest_from_below(listOfCablingIOVs, std::get<0>(iov)));
      int index = std::distance(listOfCablingIOVs.begin(), it);

      edm::LogPrint(k_ClassName) << " using the SiPixelFedCablingMap with hash: "
                                 << std::get<1>(m_cabling_iovs.at(index)) << std::endl;

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

        if (std::find_if(the_scenarios.begin(), the_scenarios.end(), compareKeys(scenName)) != the_scenarios.end() ||
            the_scenarios[0] == "all") {
          edm::LogPrint(k_ClassName) << "\t Found Scenario: " << scenName << " ==> dumping it";
        } else {
          continue;
        }

        //if (strcmp(scenName.c_str(),"320824_103") != 0) continue;

        const auto& theDetSetBadPixelFedChannels = payload->getDetSetBadPixelFedChannels(scenName);
        for (const auto& disabledChannels : *theDetSetBadPixelFedChannels) {
          const auto t_detid = disabledChannels.detId();
          int subid = DetId(t_detid).subdetId();
          LogDebug(k_ClassName) << fmt::sprintf("DetId : %i \n", t_detid) << std::endl;

          std::bitset<16> badRocsFromFEDChannels;

          for (const auto& ch : disabledChannels) {
            std::string toOut_ = fmt::sprintf("fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                                              ch.fed,
                                              ch.link,
                                              ch.roc_first,
                                              ch.roc_last);

            LogDebug(k_ClassName) << toOut_ << std::endl;
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

                  LogDebug(k_ClassName) << " => i_roc:" << i_roc << "  " << global.col << "-" << global.row << " | => "
                                        << chipIndex << " : (" << colROC << "," << rowROC << ")" << std::endl;

                  badRocsFromFEDChannels[chipIndex] = true;
                }
              }
            }
          }

          LogDebug(k_ClassName) << badRocsFromFEDChannels << std::endl;

          auto myDetId = DetId(t_detid);

          if (subid == PixelSubdetector::PixelBarrel) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, 1.);
          }  // if it's barrel
          else if (subid == PixelSubdetector::PixelEndcap) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, 1.);
          }  // if it's endcap
          else {
            throw cms::Exception("LogicError") << "Unknown Pixel SubDet ID " << std::endl;
          }  // else nonsense
        }    // loop on the channels
      }      // loop on the scenarios

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      std::string IOVstring = (unpacked.first == 0)
                                  ? std::to_string(unpacked.second)
                                  : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second));

      const auto headerText = fmt::sprintf("#color[4]{%s},  IOV: #color[4]{%s}", tagname, IOVstring);

      switch (myType) {
        case SiPixelPI::t_barrel:
          theROCMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          theROCMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          theROCMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("LogicError") << "\nERROR: unrecognized Pixel Detector part " << std::endl;
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

    // graphics
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    static constexpr const char* k_ClassName = "SiPixelFEDChannelContainerMap";

    TrackerTopology m_trackerTopo;
    edm::ParameterSet m_connectionPset;
    cond::persistency::ConnectionPool m_connectionPool;
    std::string m_CablingTagName;
    std::string m_condDbCabling;
  };

  /************************************************
  1d histogram of SiPixelFEDChannelContainer of 1 IOV
  *************************************************/

  template <SiPixelPI::DetType myType>
  class SiPixelFEDChannelContainerMapSimple : public PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV> {
  public:
    SiPixelFEDChannelContainerMapSimple()
        : PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV>(
              "SiPixelFEDChannelContainer Pixel Track Map of one (or more scenarios)") {
      // for inputs
      PlotBase::addInputParam("Scenarios");
    }

    bool fill() override {
      std::vector<std::string> the_scenarios = {};

      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("Scenarios");
      if (ip != paramValues.end()) {
        auto input = ip->second;
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        boost::char_separator<char> sep{","};
        tokenizer tok{input, sep};
        for (const auto& t : tok) {
          the_scenarios.push_back(t);
        }
      } else {
        edm::LogWarning(k_ClassName)
            << "\n WARNING!!!! \n The needed parameter Scenarios has not been passed. Will use all the scenarios in "
               "the file!"
            << "\n Buckle your seatbelts... this might take a while... \n\n";
        the_scenarios.push_back("all");
      }

      Phase1PixelROCMaps theROCMap("");

      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      std::shared_ptr<SiPixelFEDChannelContainer> payload = fetchPayload(std::get<1>(iov));
      const auto& scenarioMap = payload->getScenarioMap();

      for (const auto& scenario : scenarioMap) {
        std::string scenName = scenario.first;

        if (std::find_if(the_scenarios.begin(), the_scenarios.end(), compareKeys(scenName)) != the_scenarios.end() ||
            the_scenarios[0] == "all") {
          edm::LogPrint(k_ClassName) << "\t Found Scenario: " << scenName << " ==> dumping it";
        } else {
          continue;
        }

        const auto& theDetSetBadPixelFedChannels = payload->getDetSetBadPixelFedChannels(scenName);
        for (const auto& disabledChannels : *theDetSetBadPixelFedChannels) {
          const auto t_detid = disabledChannels.detId();
          int subid = DetId(t_detid).subdetId();
          LogDebug(k_ClassName) << fmt::sprintf("DetId : %i \n", t_detid) << std::endl;

          std::bitset<16> badRocsFromFEDChannels;

          for (const auto& ch : disabledChannels) {
            std::string toOut_ = fmt::sprintf("fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                                              ch.fed,
                                              ch.link,
                                              ch.roc_first,
                                              ch.roc_last);

            LogDebug(k_ClassName) << toOut_ << std::endl;
            for (unsigned int i_roc = ch.roc_first; i_roc <= ch.roc_last; ++i_roc) {
              badRocsFromFEDChannels.set(i_roc);
            }
          }

          LogDebug(k_ClassName) << badRocsFromFEDChannels << std::endl;

          const auto& myDetId = DetId(t_detid);

          if (subid == PixelSubdetector::PixelBarrel) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, 1.);
          }  // if it's barrel
          else if (subid == PixelSubdetector::PixelEndcap) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, 1.);
          }  // if it's endcap
          else {
            throw cms::Exception("LogicError") << "Unknown Pixel SubDet ID " << std::endl;
          }  // else nonsense
        }    // loop on the channels
      }      // loop on the scenarios

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      std::string IOVstring = (unpacked.first == 0)
                                  ? std::to_string(unpacked.second)
                                  : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second));

      const auto headerText = fmt::sprintf("#color[4]{%s},  IOV: #color[4]{%s}", tagname, IOVstring);

      switch (myType) {
        case SiPixelPI::t_barrel:
          theROCMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          theROCMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          theROCMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("LogicError") << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      // add list of scenarios watermark
      canvas.cd();
      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kMagenta);
      ltx.SetTextSize(0.023);
      ltx.DrawLatexNDC(
          gPad->GetLeftMargin() - 0.09,
          gPad->GetBottomMargin() - 0.09,
          ("scenarios: #color[4]{" +
           std::accumulate(the_scenarios.begin(),
                           the_scenarios.end(),
                           std::string(),
                           [](const std::string& acc, const std::string& str) { return acc + " " + str; }) +
           "}")
              .c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }

  public:
    // auxilliary check
    struct compareKeys {
      std::string key;
      compareKeys(std::string const& i) : key(i) {}

      bool operator()(std::string const& i) { return (key == i); }
    };

  private:
    // graphics
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    static constexpr const char* k_ClassName = "SiPixelFEDChannelContainerMapSimple";
  };

  using SiPixelBPixFEDChannelContainerMap = SiPixelFEDChannelContainerMapSimple<SiPixelPI::t_barrel>;
  using SiPixelFPixFEDChannelContainerMap = SiPixelFEDChannelContainerMapSimple<SiPixelPI::t_forward>;
  using SiPixelFullFEDChannelContainerMap = SiPixelFEDChannelContainerMapSimple<SiPixelPI::t_all>;

  /*
    Produces an aggregate map of the masked components for all scenarios,
    weighted on the probability per PU unit from SiPixelQualityProbabilities
    assuming a flat PU profile in the range encoded in  SiPixelQualityProbabilities
    The SiPixelQualityProbabilities tag comes from user input
  */
  template <SiPixelPI::DetType myType>
  class SiPixelFEDChannelContainerMapWeigthed : public PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV> {
  public:
    SiPixelFEDChannelContainerMapWeigthed()
        : PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV>(
              "SiPixelFEDChannelContainer Pixel Track Map of one (or more scenarios)") {
      // for inputs
      PlotBase::addInputParam("SiPixelQualityProbabilitiesTag");

      // hardcoded connection to the SiPixelQualityProbability tag, though luck
      m_condSiPixelProb = "frontier://FrontierProd/CMS_CONDITIONS";
      m_connectionPool.setParameters(m_connectionPset);
      m_connectionPool.configure();
    }

    bool fill() override {
      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("SiPixelQualityProbabilitiesTag");
      if (ip != paramValues.end()) {
        m_SiPixelProbTagName = ip->second;
      } else {
        edm::LogWarning(k_ClassName) << "\n WARNING!!!! \n The needed parameter SiPixelQualityProbabilitiesTag was not "
                                        "inputed from the user \n Display will be aborted \n\n";
        return false;
      }

      Phase1PixelROCMaps theROCMap("", "Masking Probability [%]");

      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      // open db session for the cabling map
      edm::LogPrint(k_ClassName) << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
                                 << "Query the condition database " << m_condSiPixelProb;

      cond::persistency::Session condDbSession = m_connectionPool.createSession(m_condSiPixelProb);
      condDbSession.transaction().start(true);

      // query the database
      edm::LogPrint(k_ClassName) << "[SiPixelFEDChannelContainerTest::" << __func__ << "] "
                                 << "Reading IOVs from tag " << m_SiPixelProbTagName;

      const auto MIN_VAL = cond::timeTypeSpecs[cond::runnumber].beginValue;
      const auto MAX_VAL = cond::timeTypeSpecs[cond::runnumber].endValue;

      // get the list of payloads for the Cabling Map
      std::vector<std::tuple<cond::Time_t, cond::Hash>> m_pixelProb_iovs;
      condDbSession.readIov(m_SiPixelProbTagName).selectRange(MIN_VAL, MAX_VAL, m_pixelProb_iovs);

      // in MC there should be only 1 IOV, oh well...
      edm::LogPrint(k_ClassName) << " using the SiPixelQualityProbabilities with hash: "
                                 << std::get<1>(m_pixelProb_iovs.front()) << std::endl;

      auto probabilitiesPayload =
          condDbSession.fetchPayload<SiPixelQualityProbabilities>(std::get<1>(m_pixelProb_iovs.front()));

      const auto& PUbins = probabilitiesPayload->getPileUpBins();

      SiPixelQualityProbabilities::probabilityMap m_probabilities = probabilitiesPayload->getProbability_Map();

      // find the PU-averaged (assuming flat PU in the range) probabilities for each scenario
      std::map<std::string, float> puAvgedProbabilities;
      for (const auto& [PUbin, ProbMap] : m_probabilities) {
        float totProbInPUbin = 0.f;
        for (const auto& [scenName, prob] : ProbMap) {
          totProbInPUbin += prob;
          if (puAvgedProbabilities.find(scenName) == puAvgedProbabilities.end()) {
            puAvgedProbabilities[scenName] += prob;
          } else {
            puAvgedProbabilities.insert({scenName, prob});
          }
        }
        LogDebug(k_ClassName) << "PU bin: " << PUbin << " tot probability " << totProbInPUbin << std::endl;
      }

      std::shared_ptr<SiPixelFEDChannelContainer> payload = fetchPayload(std::get<1>(iov));
      const auto& scenarioMap = payload->getScenarioMap();

      float totProb{0.f};
      for (const auto& [scenName, prob] : puAvgedProbabilities) {
        // only sum up the scenarios that are in the SiPixelFEDChannelContainer payload!
        if (scenarioMap.find(scenName) != scenarioMap.end()) {
          LogDebug(k_ClassName) << scenName << " : " << prob << std::endl;
          totProb += prob;
        }
      }

      LogDebug(k_ClassName) << "Total probability to normalize to: " << totProb << std::endl;

      //normalize the probabilities per scenario to the toal probability
      for (auto& pair : puAvgedProbabilities) {
        pair.second /= totProb;
      }

      for (const auto& scenario : scenarioMap) {
        std::string scenName = scenario.first;
        LogDebug(k_ClassName) << "\t Found Scenario: " << scenName << " ==> dumping it";

        // calculate the weight
        float w_frac = 0.f;
        if (puAvgedProbabilities.find(scenName) != puAvgedProbabilities.end()) {
          w_frac = puAvgedProbabilities[scenName];
        }

        // if scenario is not in the probability payload, continue
        if (w_frac == 0.f)
          continue;

        LogDebug(k_ClassName) << "scen: " << scenName << " weight: " << w_frac << " log(weight):" << log10(w_frac)
                              << std::endl;

        const auto& theDetSetBadPixelFedChannels = payload->getDetSetBadPixelFedChannels(scenName);
        for (const auto& disabledChannels : *theDetSetBadPixelFedChannels) {
          const auto t_detid = disabledChannels.detId();
          int subid = DetId(t_detid).subdetId();
          LogDebug(k_ClassName) << fmt::sprintf("DetId : %i \n", t_detid) << std::endl;

          std::bitset<16> badRocsFromFEDChannels;

          for (const auto& ch : disabledChannels) {
            std::string toOut_ = fmt::sprintf("fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                                              ch.fed,
                                              ch.link,
                                              ch.roc_first,
                                              ch.roc_last);

            LogDebug(k_ClassName) << toOut_ << std::endl;
            for (unsigned int i_roc = ch.roc_first; i_roc <= ch.roc_last; ++i_roc) {
              badRocsFromFEDChannels.set(i_roc);
            }
          }

          LogDebug(k_ClassName) << badRocsFromFEDChannels << std::endl;

          const auto& myDetId = DetId(t_detid);

          if (subid == PixelSubdetector::PixelBarrel) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, w_frac * 100);
          }  // if it's barrel
          else if (subid == PixelSubdetector::PixelEndcap) {
            theROCMap.fillSelectedRocs(myDetId, badRocsFromFEDChannels, w_frac * 100);
          }  // if it's endcap
          else {
            throw cms::Exception("LogicError") << "Unknown Pixel SubDet ID " << std::endl;
          }  // else nonsense
        }    // loop on the channels
      }      // loop on the scenarios

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      std::string IOVstring = (unpacked.first == 0)
                                  ? std::to_string(unpacked.second)
                                  : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second));

      const auto headerText =
          fmt::sprintf("#bf{#scale[0.6]{#color[2]{%s}, #color[4]{%s}}}", tagname, m_SiPixelProbTagName);

      switch (myType) {
        case SiPixelPI::t_barrel:
          theROCMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          theROCMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          theROCMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("LogicError") << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      // add list of scenarios watermark
      canvas.cd();
      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kMagenta);
      ltx.SetTextSize(0.023);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() - 0.09, gPad->GetBottomMargin() - 0.09, "");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }

  private:
    // graphics
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    static constexpr const char* k_ClassName = "SiPixelFEDChannelContainerMapWeigthed";

    // parameters for auxilliary DB connection
    edm::ParameterSet m_connectionPset;
    cond::persistency::ConnectionPool m_connectionPool;
    std::string m_SiPixelProbTagName;
    std::string m_condSiPixelProb;
  };

  using SiPixelBPixFEDChannelContainerWeightedMap = SiPixelFEDChannelContainerMapWeigthed<SiPixelPI::t_barrel>;
  using SiPixelFPixFEDChannelContainerWeightedMap = SiPixelFEDChannelContainerMapWeigthed<SiPixelPI::t_forward>;
  using SiPixelFullFEDChannelContainerWeightedMap = SiPixelFEDChannelContainerMapWeigthed<SiPixelPI::t_all>;

  /************************************************
  1d histogram of number of SiPixelFEDChannelContainer scenarios
  *************************************************/

  class SiPixelFEDChannelContainerScenarios : public PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV> {
  public:
    SiPixelFEDChannelContainerScenarios()
        : PlotImage<SiPixelFEDChannelContainer, SINGLE_IOV>("SiPixelFEDChannelContainer scenarios count") {}
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      TGaxis::SetMaxDigits(3);

      std::shared_ptr<SiPixelFEDChannelContainer> payload = fetchPayload(std::get<1>(iov));
      std::vector<std::string> scenarios = payload->getScenarioList();
      sort(scenarios.begin(), scenarios.end());

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      canvas.SetGrid();
      auto h1 = std::make_unique<TH1F>("Count",
                                       "SiPixelFEDChannelContainer Bad Roc count;Scenario index;n. of bad ROCs",
                                       scenarios.size(),
                                       1,
                                       scenarios.size());
      h1->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      int scenarioIndex = 0;
      for (const auto& scenario : scenarios) {
        scenarioIndex++;
        int badRocCount = 0;
        LogDebug("SiPixelFEDChannelContainerScenarios") << scenario << std::endl;
        auto badChannelCollection = payload->getDetSetBadPixelFedChannels(scenario);
        for (const auto& disabledChannels : *badChannelCollection) {
          for (const auto& ch : disabledChannels) {
            int local_bad_rocs = ch.roc_last - ch.roc_first;
            badRocCount += local_bad_rocs;
          }  // loop on the channels
        }    // loop on the DetSetVector

        h1->SetBinContent(scenarioIndex, badRocCount);
      }  // loop on scenarios

      TGaxis::SetExponentOffset(-0.1, 0.01, "y");    // Y offset
      TGaxis::SetExponentOffset(-0.03, -0.10, "x");  // Y and Y offset for X axis

      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("bar2");

      SiPixelPI::makeNicePlotStyle(h1.get());

      canvas.Update();

      TLegend legend = TLegend(0.30, 0.88, 0.95, 0.94);
      //legend.SetHeader(("#splitline{Payload hash: #bf{" + (std::get<1>(iov)) + "}}{Total Scenarios:"+std::to_string(scenarioIndex)+"}").c_str(),"C");  // option "C" allows to center the header

      legend.SetHeader(fmt::sprintf("Payload hash: #bf{%s}", std::get<1>(iov)).c_str(), "C");
      legend.AddEntry(h1.get(), fmt::sprintf("total scenarios: #bf{%s}", std::to_string(scenarioIndex)).c_str(), "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      //ltx.SetTextAlign(11);
      ltx.SetTextSize(0.040);
      ltx.DrawLatexNDC(
          gPad->GetLeftMargin(),
          1 - gPad->GetTopMargin() + 0.01,
          fmt::sprintf("#color[4]{%s} IOV: #color[4]{%s}", tagname, std::to_string(std::get<0>(iov))).c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;

    }  // fill
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelFEDChannelContainer) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixFEDChannelContainerMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixFEDChannelContainerMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullFEDChannelContainerMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixFEDChannelContainerWeightedMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixFEDChannelContainerWeightedMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullFEDChannelContainerWeightedMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFEDChannelContainerScenarios);
}
