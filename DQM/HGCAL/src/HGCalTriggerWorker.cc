#include "DQM/HGCAL/interface/HGCalTriggerWorker.h"
#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <TDirectory.h>
#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace hgcal {
  namespace dqm {

    using MonitoredElementKey_t = HGCalDQMGeometry::MonitoredElementKey_t;
    using TriggerMonitoredElement_t = HGCalDQMGeometry::TriggerMonitoredElement_t;

    HGCalTriggerWorker::HGCalTriggerWorker(std::string folderRoot, EcontErrorSummarizer& econtErrorSummarizer)
        : folderRoot_(std::move(folderRoot)), econt_error_summarizer_(econtErrorSummarizer) {}

    // Books the trigger hex plots and the ECON-T quality summaries.
    void HGCalTriggerWorker::book(DQMStore::IBooker& ibooker, HGCalDQMGeometry const& geom) {
      auto const& unique_directionallayers = geom.uniqueDirectionalLayers();
      auto const& cassettesPerLayer = geom.cassettesPerLayer();
      auto const& corners_layer = geom.cornersLayer();
      auto const& corners_cassette = geom.cornersCassette();
      auto const& TriggerModuleMap = geom.triggerModuleMap();

      const std::string TriggerFolder = folderRoot_ + "/Trigger";

      for (const std::string& BX : BXlist_) {
        for (int layer : unique_directionallayers) {
          float xmin = corners_layer.at(layer)[0];
          float xmax = corners_layer.at(layer)[1];
          float ymin = corners_layer.at(layer)[2];
          float ymax = corners_layer.at(layer)[3];

          std::string endcapFolder = layer > 0 ? "/Endcap_Plus" : "/Endcap_Minus";
          std::string layerFolder = "/Layer_" + std::to_string(std::abs(layer));

          ibooker.setCurrentFolder(TriggerFolder + endcapFolder + layerFolder);

          hexTriggerLayer_[layer][BX + "_energy"] =
              ibooker.book2DPoly(BX + "_energy",
                                 BX + "_energy_layer_" + std::to_string(std::abs(layer)) + ";x[cm];y[cm];Count",
                                 xmin,
                                 xmax,
                                 ymin,
                                 ymax);
          hexTriggerLayer_[layer][BX + "_rms_energy"] =
              ibooker.book2DPoly(BX + "_rms_energy",
                                 BX + "_rms_energy_layer_" + std::to_string(std::abs(layer)) + ";x[cm];y[cm];Count",
                                 xmin,
                                 xmax,
                                 ymin,
                                 ymax);
          hexTriggerLayer_[layer][BX + "_occupancy"] =
              ibooker.book2DPoly(BX + "_occupancy",
                                 BX + "_occupancy_layer_" + std::to_string(std::abs(layer)) + ";x[cm];y[cm];Count",
                                 xmin,
                                 xmax,
                                 ymin,
                                 ymax);

          for (int cassette : cassettesPerLayer.at(layer)) {
            float xmin_ = (std::abs(layer) == 44) ? -30 : corners_cassette.at(layer).at(cassette)[0];
            float xmax_ = (std::abs(layer) == 44) ? 30 : corners_cassette.at(layer).at(cassette)[1];
            float ymin_ = (std::abs(layer) == 44) ? 90 : corners_cassette.at(layer).at(cassette)[2];
            float ymax_ = (std::abs(layer) == 44) ? 160 : corners_cassette.at(layer).at(cassette)[3];

            std::string cassetteFolder = "/Cassette_" + std::to_string(cassette);
            ibooker.setCurrentFolder(TriggerFolder + endcapFolder + layerFolder + cassetteFolder);

            hexTriggerCassette_[layer][cassette][BX + "_energy"] =
                ibooker.book2DPoly(BX + "_energy",
                                   BX + "_energy_cassette_" + std::to_string(cassette) + ";x[cm];y[cm];Count",
                                   xmin_,
                                   xmax_,
                                   ymin_,
                                   ymax_);
            hexTriggerCassette_[layer][cassette][BX + "_rms_energy"] =
                ibooker.book2DPoly(BX + "_rms_energy",
                                   BX + "_rms_energy_cassette_" + std::to_string(cassette) + ";x[cm];y[cm];Count",
                                   xmin_,
                                   xmax_,
                                   ymin_,
                                   ymax_);
            hexTriggerCassette_[layer][cassette][BX + "_occupancy"] =
                ibooker.book2DPoly(BX + "_occupancy",
                                   BX + "_occupancy_cassette_" + std::to_string(cassette) + ";x[cm];y[cm];Count",
                                   xmin_,
                                   xmax_,
                                   ymin_,
                                   ymax_);
          }
        }

        for (auto const& [key, trModule] : TriggerModuleMap) {
          std::string uvStr = "(u" + std::to_string(trModule.i1) + "-v" + std::to_string(trModule.i2) + ") ";
          std::string endcapFolder = trModule.zside ? "/Endcap_Plus" : "/Endcap_Minus";
          std::string layerFolder = "/Layer_" + std::to_string(trModule.plane);
          std::string cassetteFolder = "/Cassette_" + std::to_string(trModule.cassette);
          std::string moduleFolder = "/" + uvStr + trModule.typecode;

          ibooker.setCurrentFolder(TriggerFolder + endcapFolder + layerFolder + cassetteFolder + moduleFolder);

          hexTriggerPlots_[trModule.typecode][BX + "_energy"] = ibooker.book2DPoly(
              BX + "_energy", BX + "_energy_" + trModule.typecode + ";x[cm];y[cm];Counts", -14, 14, -14, 14);
          hexTriggerPlots_[trModule.typecode][BX + "_rms_energy"] = ibooker.book2DPoly(
              BX + "_rms_energy", BX + "_rms_energy_" + trModule.typecode + ";x[cm];y[cm];Counts", -14, 14, -14, 14);
          hexTriggerPlots_[trModule.typecode][BX + "_occupancy"] = ibooker.book2DPoly(
              BX + "_occupancy", BX + "_occupancy_" + trModule.typecode + ";x[cm];y[cm];Counts", -14, 14, -14, 14);
        }
      }

      // ECON-T quality summaries.
      const std::vector<std::string> categoryNames = {"Unpacking Errors", "Header/Trailer"};
      const int nCategories = static_cast<int>(EconTErrorCategory::NUM_CATEGORIES);
      const size_t nLayers = geom.nLayers();
      auto const& TrigHGCALMap = geom.trigHgcalMap();

      if (nLayers > 0) {
        ibooker.setCurrentFolder(TriggerFolder);
        me_econt_quality_summary_ =
            ibooker.book2D("econtQuality", ";Layer;;", nLayers, 0, nLayers, nCategories, 0, nCategories);
        addBinLabels(const_cast<std::vector<std::string>&>(categoryNames), me_econt_quality_summary_, 2);
      }

      for (auto const& [endcap, layerMap] : TrigHGCALMap) {
        std::string endCapString = endCapKey_[endcap];
        for (auto const& [layer, cassetteMap] : layerMap) {
          std::string layerStr = "Layer_" + std::to_string(std::abs(layer));
          std::string layerFolder = TriggerFolder + "/Endcap_" + endCapString + "/" + layerStr;
          int nCassettes = cassetteMap.size();

          ibooker.setCurrentFolder(layerFolder);
          auto* me = ibooker.book2D(
              "econtQuality" + layerStr, ";Cassette;", nCassettes, 0, nCassettes, nCategories, 0, nCategories);
          addBinLabels(const_cast<std::vector<std::string>&>(categoryNames), me, 2);
          econtQualityLayer_[endcap][layer] = me;
        }
      }
    }

    void HGCalTriggerWorker::endRun(DQMStore::IBooker& /*ibooker*/,
                                    DQMStore::IGetter& igetter,
                                    HGCalDQMGeometry& geom) {
      fillHexaPlots(igetter, geom);
      fillEcontQuality(igetter, geom);
    }

    // Fills the trigger hex plots from the HGCalTPGDQM MEs under HGCAL/Trigger/.
    void HGCalTriggerWorker::fillHexaPlots(DQMStore::IGetter& igetter, HGCalDQMGeometry& geom) {
      auto const& TrigHGCALMap = geom.trigHgcalMap();
      auto const& TriggerModuleMap = geom.triggerModuleMap();

      for (auto const& [endcap, layerMap] : TrigHGCALMap) {
        for (auto const& [plane, cassetteMap] : layerMap) {
          int layerBinIdx(0);
          for (auto const& [cassette, moduleSet] : cassetteMap) {
            int cassetteBinOffset(0);
            for (MonitoredElementKey_t key : moduleSet) {
              TriggerMonitoredElement_t const& trModule = TriggerModuleMap.at(key);
              int layer = trModule.layer;
              std::string typecode = trModule.typecode;

              std::string uvStr = "(u" + std::to_string(trModule.i1) + "-v" + std::to_string(trModule.i2) + ") ";
              std::ostringstream oss;
              oss << "HGCAL/Trigger/Endcap_" << (trModule.zside ? "Plus" : "Minus") << "/Layer_" << plane
                  << "/Cassette_" << cassette << "/" << uvStr << typecode;
              std::string plotFolder(oss.str());

              const MonitorElement* energy_me = igetter.get(plotFolder + "/BX_energy");
              const MonitorElement* energy2_me = igetter.get(plotFolder + "/BX_energy2");
              const MonitorElement* occ_me = igetter.get(plotFolder + "/occupancy");
              const MonitorElement* algo_me = igetter.get(plotFolder + "/algo");

              if (!energy_me || !energy2_me || !occ_me || !algo_me) {
                edm::LogWarning("HGCalTriggerWorker")
                    << "Missing client ME under " << plotFolder << " (BX_energy=" << (energy_me != nullptr)
                    << ", BX_energy2=" << (energy2_me != nullptr) << ", occupancy=" << (occ_me != nullptr)
                    << ", algo=" << (algo_me != nullptr) << ") skipping this module";
                continue;
              }

              std::map<std::string, float> value_module;
              for (const std::string& BX : BXlist_) {
                value_module.emplace(BX + "_location", 0.0f);
                value_module.emplace(BX + "_energy", 0.0f);
              }

              TFile* fwafer = nullptr;
              try {
                fwafer = geom.trigTemplateFile(typecode, trModule.isSiPM);
              } catch (const cms::Exception& e) {
                edm::LogError("HGCalTriggerWorker") << "Didn't find TC wafer map for " << typecode << ": " << e.what();
                continue;
              }

              TKey* tkey;
              TIter nextkey(fwafer->GetDirectory("hex")->GetListOfKeys());
              uint32_t chIdx(0);
              uint32_t binIdx(0);

              bool isML5 = (typecode[1] == 'L') && (typecode[3] == '5');
              bool isMHB = (typecode[1] == 'H') && (typecode[3] == 'B');

              while ((tkey = (TKey*)nextkey())) {
                TObject* obj = tkey->ReadObj();
                if (!obj->InheritsFrom("TGraph"))
                  continue;
                TGraph* TCBin = (TGraph*)obj;

                bool isNC = (isML5 && (chIdx >= 24 && chIdx < 32)) || (isMHB && (chIdx == 12 || chIdx == 14));
                if (isNC) {
                  chIdx++;
                  continue;
                }

                double angleOffset = -3 * M_PI / 6.;
                // Tileboard trigger-cell templates are drawn in detector coordinates; silicon ones are module-local.
                bool const inGlobalCoordinates = trModule.isSiPM;
                if (!inGlobalCoordinates) {
                  HGCalDQMGeometry::rotateShape(TCBin, trModule.irot, angleOffset);
                }

                uint32_t i_BX(0);
                for (const std::string& BX : BXlist_) {
                  uint32_t occupancy = occ_me->getBinContent(i_BX + 1, chIdx + 1);
                  double energy_count = energy_me->getBinContent(i_BX + 1, chIdx + 1);
                  double rms_energy_count = energy2_me->getBinContent(i_BX + 1, chIdx + 1);
                  i_BX++;

                  if (occupancy != 0) {
                    energy_count /= occupancy;
                    rms_energy_count = std::sqrt(rms_energy_count / occupancy);
                  } else {
                    energy_count = 0;
                    rms_energy_count = 0;
                  }

                  value_module[BX + "_energy"] += energy_count;
                  value_module[BX + "_rms_energy"] += rms_energy_count;
                  value_module[BX + "_occupancy"] += occupancy;

                  hexTriggerPlots_[typecode][BX + "_energy"]->addBin(TCBin);
                  hexTriggerPlots_[typecode][BX + "_energy"]->setBinContent(binIdx + 1, energy_count);

                  hexTriggerPlots_[typecode][BX + "_rms_energy"]->addBin(TCBin);
                  hexTriggerPlots_[typecode][BX + "_rms_energy"]->setBinContent(binIdx + 1, rms_energy_count);

                  hexTriggerPlots_[typecode][BX + "_occupancy"]->addBin(TCBin);
                  hexTriggerPlots_[typecode][BX + "_occupancy"]->setBinContent(binIdx + 1, occupancy);

                  TGraph* TCBinCassette = new TGraph(*static_cast<TGraph*>(TCBin));
                  double x0 = trModule.x0y0[0];
                  double y0 = trModule.x0y0[1];
                  if (!inGlobalCoordinates) {
                    HGCalDQMGeometry::translateBin(TCBinCassette, x0, y0);
                  }

                  hexTriggerCassette_[layer][cassette][BX + "_energy"]->addBin(TCBinCassette);
                  hexTriggerCassette_[layer][cassette][BX + "_energy"]->setBinContent(cassetteBinOffset + binIdx + 1,
                                                                                      energy_count);

                  hexTriggerCassette_[layer][cassette][BX + "_rms_energy"]->addBin(TCBinCassette);
                  hexTriggerCassette_[layer][cassette][BX + "_rms_energy"]->setBinContent(
                      cassetteBinOffset + binIdx + 1, rms_energy_count);

                  hexTriggerCassette_[layer][cassette][BX + "_occupancy"]->addBin(TCBinCassette);
                  hexTriggerCassette_[layer][cassette][BX + "_occupancy"]->setBinContent(cassetteBinOffset + binIdx + 1,
                                                                                         occupancy);
                }
                binIdx++;
                chIdx++;
              }

              cassetteBinOffset += binIdx;
              for (auto& [k, v] : value_module)
                v /= binIdx;

              TGraph* trigBin = geom.triggerModuleBin(trModule.dqmIndex);

              for (const std::string& BX : BXlist_) {
                hexTriggerLayer_[layer][BX + "_energy"]->addBin(trigBin);
                hexTriggerLayer_[layer][BX + "_energy"]->setBinContent(layerBinIdx + 1, value_module[BX + "_energy"]);

                hexTriggerLayer_[layer][BX + "_rms_energy"]->addBin(trigBin);
                hexTriggerLayer_[layer][BX + "_rms_energy"]->setBinContent(layerBinIdx + 1,
                                                                           value_module[BX + "_rms_energy"]);

                hexTriggerLayer_[layer][BX + "_occupancy"]->addBin(trigBin);
                hexTriggerLayer_[layer][BX + "_occupancy"]->setBinContent(layerBinIdx + 1,
                                                                          value_module[BX + "_occupancy"]);
              }
              layerBinIdx++;
            }
          }
        }
      }
    }

    // Aggregates TPGDQM's per-cassette econtQualityCassette_<c> MEs into
    // folderRoot_/Trigger/{Endcap_X/Layer_N/econtQualityLayer_N, econtQuality}.
    void HGCalTriggerWorker::fillEcontQuality(DQMStore::IGetter& igetter, HGCalDQMGeometry const& geom) {
      auto const& TrigHGCALMap = geom.trigHgcalMap();
      int trig_layer_idx = 0;

      for (auto const& [endcap, layerMap] : TrigHGCALMap) {
        for (auto const& [layer, cassetteMap] : layerMap) {
          auto layerIt = econtQualityLayer_[endcap].find(layer);
          if (layerIt == econtQualityLayer_[endcap].end() || layerIt->second == nullptr)
            continue;
          MonitorElement* layerMe = layerIt->second;

          int cassette_idx = 0;
          for (auto const& [cassette, _keys] : cassetteMap) {
            std::ostringstream path;
            path << "HGCAL/Trigger/Endcap_" << endCapKey_[endcap] << "/Layer_" << std::abs(layer) << "/Cassette_"
                 << cassette << "/econtQualityCassette_" << cassette;
            const MonitorElement* quality_me = igetter.get(path.str());
            if (!quality_me) {
              edm::LogError("HGCalTriggerWorker")
                  << "Missing " << path.str() << " (skipping cassette " << cassette << ")";
              ++cassette_idx;
              continue;
            }

            std::string binlabel = "Cassette" + std::to_string(cassette);
            layerMe->setBinLabel(cassette_idx + 1, binlabel, 1);
            econt_error_summarizer_.processAndFill(quality_me, layerMe, cassette_idx, ProcessMode::STAT_TO_GRADE);
            ++cassette_idx;
          }

          if (me_econt_quality_summary_) {
            int directionalLayer = layer * endcap;
            me_econt_quality_summary_->setBinLabel(trig_layer_idx + 1, std::to_string(directionalLayer), 1);
            econt_error_summarizer_.processAndFill(
                layerMe, me_econt_quality_summary_, trig_layer_idx, ProcessMode::GRADE_TO_GRADE);
          }
          ++trig_layer_idx;
        }
      }
    }

  }  // namespace dqm
}  // namespace hgcal
