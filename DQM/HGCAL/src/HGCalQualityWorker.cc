#include "DQM/HGCAL/interface/HGCalQualityWorker.h"

#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace hgcal {
  namespace dqm {

    HGCalQualityWorker::HGCalQualityWorker(std::string folderRoot, EcondErrorSummarizer& econdErrorSummarizer)
        : folderRoot_(std::move(folderRoot)), econd_error_summarizer_(econdErrorSummarizer) {}

    /**
* @brief Find bin index by label with caching
* @param me MonitorElement containing the histogram
* @param label The label to search for
* @return Bin index (1-based) or -1 if not found
*/
    int HGCalQualityWorker::findBinByLabel(MonitorElement* me, const std::string& label) const {
      auto& labelMap = label_to_bin_cache_[me];

      // Lazy initialization: build the mapping if empty
      if (labelMap.empty()) {
        const TH2F* hist = me->getTH2F();
        if (!hist)
          return -1;

        labelMap.reserve(hist->GetNbinsX());  // Pre-allocate for efficiency
        for (int xbin = 1; xbin <= hist->GetNbinsX(); ++xbin) {
          std::string binLabel = hist->GetXaxis()->GetBinLabel(xbin);
          if (!binLabel.empty()) {
            labelMap[binLabel] = xbin;
          }
        }
      }

      auto it = labelMap.find(label);
      return (it != labelMap.end()) ? it->second : -1;
    }

    // Books the ECON-D quality/payload summaries and their bin labels (ECON-T: HGCalTriggerWorker).
    void HGCalQualityWorker::book(DQMStore::IBooker& ibooker, HGCalDQMGeometry const& geom) {
      auto const& HGCALMap = geom.hgcalMap();
      auto const& corners_layer = geom.cornersLayer();

      int nQualityCategories = static_cast<int>(ErrorCategory::NUM_CATEGORIES);

      auto getPlotTitle = [](const std::string& plotKey) -> std::string {
        if (plotKey == "econdQuality")
          return std::string("Quality Grade");
        return std::string("Occupancy");
      };

      const size_t nLayers = geom.nLayers();

      if (nLayers > 0) {
        ibooker.setCurrentFolder(folderRoot_);

        // ECON-D Quality in TH2F at summary level
        me_econd_quality_summary_ =
            ibooker.book2D("econdQuality", ";Layer;;", nLayers, 0, nLayers, nQualityCategories, 0, nQualityCategories);
        hgcal::dqm::addBinLabels(hgcal::dqm::qualityCategoryNames, me_econd_quality_summary_, 2);
      }

      // ECON-D Quality at layer level
      int layer_idx = 0;
      for (const auto& endcapPair : HGCALMap) {
        int endcap = endcapPair.first;
        std::string endCapString = endCapKey_[endcap];
        std::string endcapFolder = folderRoot_ + "/EndCap_" + endCapString + "/";
        const auto& layerMap = endcapPair.second;

        for (const auto& layerPair : layerMap) {
          int layer = layerPair.first;  // directional layer
          const auto& cassetteMap = layerPair.second;
          std::string layerStr = "Layer_" + std::to_string(std::abs(layer));
          std::string layerFolder = endcapFolder + layerStr + "/";
          int nCassettes = cassetteMap.size();

          // set current folder
          ibooker.setCurrentFolder(layerFolder);

          // register layer-level TH2Poly
          for (const std::string& plotKey : hexPlotsFastStreamKey_) {
            float xmin = corners_layer.at(layer)[0];
            float xmax = corners_layer.at(layer)[1];
            float ymin = corners_layer.at(layer)[2];
            float ymax = corners_layer.at(layer)[3];
            hexPlotsFastStream_[plotKey][layer] =
                ibooker.book2DPoly("hex_" + plotKey + layerStr,
                                   layerStr + "; x[cm]; y[cm];" + getPlotTitle(plotKey),
                                   xmin,
                                   xmax,
                                   ymin,
                                   ymax);
          }
          // Module outlines are added once; endLumi only updates the bin contents, in the same module order.
          for (const auto& cassettePair : cassetteMap)
            for (const auto& econdPair : cassettePair.second)
              for (const std::string& plotKey : hexPlotsFastStreamKey_)
                hexPlotsFastStream_[plotKey][layer]->addBin(geom.moduleBin(econdPair.second.dqmIndex));

          // register layer-level summaries of the ECON-D
          econdQualityLayer_[endcap][layer] = ibooker.book2D("econdQuality" + layerStr,
                                                             ";Cassette;",
                                                             nCassettes,
                                                             0,
                                                             nCassettes,
                                                             nQualityCategories,
                                                             0,
                                                             nQualityCategories);
          hgcal::dqm::addBinLabels(hgcal::dqm::qualityCategoryNames, econdQualityLayer_[endcap][layer], 2);
          econdPayloadLayer_[endcap][layer] =
              ibooker.book1D("econdPayload" + layerStr, ";Payload (32-bit words);", 480, 0, 480);

          int cassette_idx = 0;
          for (const auto& cassettePair : cassetteMap) {
            int cassette = cassettePair.first;

            // set bin labels
            std::string binlabel = "Cassette" + std::to_string(cassette);
            econdQualityLayer_[endcap][layer]->setBinLabel(cassette_idx + 1, binlabel, 1);

            cassette_idx++;
          }

          // set bin labels
          me_econd_quality_summary_->setBinLabel(layer_idx + 1, std::to_string(layer), 1);
          layer_idx++;
        }
      }
    }

    void HGCalQualityWorker::endLumi(DQMStore::IBooker& /*ibooker*/,
                                     DQMStore::IGetter& igetter,
                                     HGCalDQMGeometry& geom,
                                     edm::LuminosityBlock const& /*iLumi*/) {
      auto const& HGCALMap = geom.hgcalMap();

      // fills layer level histograms
      int layer_idx = 0;
      for (const auto& endcapPair : HGCALMap) {
        int endcap = endcapPair.first;
        std::string endCapString = endCapKey_[endcap];
        std::string endcapFolder = folderRoot_ + "/EndCap_" + endCapString + "/";
        const auto& layerMap = endcapPair.second;

        for (const auto& layerPair : layerMap) {
          int layer = layerPair.first;  // directional layer
          const auto& cassetteMap = layerPair.second;
          std::string layerStr = "Layer_" + std::to_string(std::abs(layer));
          std::string layerFolder = endcapFolder + layerStr + "/";

          MonitorElement *econdQuality_, *econdPayload_;
          econdQuality_ = econdQualityLayer_[endcap][layer];
          econdPayload_ = econdPayloadLayer_[endcap][layer];
          // The cassette payloads accumulate over the run: rebuild the layer sum instead of adding to it.
          econdPayload_->Reset();

          int cassette_idx = 0;
          int module_bin_index = 0;  // TH2Poly bin index for current layer
          for (const auto& cassettePair : cassetteMap) {
            int cassette = cassettePair.first;
            std::string cassetteFolder = layerFolder + "Cassette_" + std::to_string(cassette) + "/";

            // load cassette-level ECON-D TH2F
            std::string cassetteQualityPath = cassetteFolder + "econdQualityCassette_" + std::to_string(cassette);
            std::string cassettePayloadPath = cassetteFolder + "econdPayloadCassette_" + std::to_string(cassette);
            MonitorElement* quality_me = igetter.get(cassetteQualityPath);
            MonitorElement* payload_me = igetter.get(cassettePayloadPath);
            if (!quality_me) {
              edm::LogError("HGCalQualityWorker") << "Did not find quality_me in " << cassetteQualityPath;
            }
            if (!payload_me) {
              edm::LogError("HGCalQualityWorker") << "Did not find payload_me in " << cassettePayloadPath;
            }

            // fill layer-level ECON-D TH2F
            econd_error_summarizer_.processAndFill(quality_me, econdQuality_, cassette_idx, ProcessMode::STAT_TO_GRADE);
            econdPayload_->getTH1()->Add(payload_me->getTH1());

            // collect statistics from modules for layer-level TH2Poly
            int nYbins = quality_me->getNbinsY();
            const auto& econdMap = cassettePair.second;
            for (const auto& econdPair : econdMap) {
              const auto typecode = econdPair.first;

              int maxGrade = 0;
              auto meanPayload = payload_me->getMean();
              auto stdPayload = payload_me->getRMS();

              // derive grade for overall module quality
              int xbin = findBinByLabel(
                  quality_me,
                  typecode);  // quality and payload follow the same typecode-xbin mapping (as set in HGCalFastStreamDQM)
              for (int ybin = 1; ybin <= nYbins; ++ybin) {
                double content = quality_me->getBinContent(xbin, ybin);
                int error_type_index = ybin - 1;
                int grade = econd_error_summarizer_.calculateGrade(error_type_index, static_cast<int>(content));
                if (grade > maxGrade) {
                  maxGrade = grade;
                }

                // sanity check
                if (first_run_) {
                  if (!econd_error_summarizer_.validateHistogramLabels(quality_me)) {
                    throw cms::Exception("HGCalQualityWorker") << "ECON-D quality labels of " << cassetteQualityPath
                                                               << " do not match the dqmQualityThreshold configuration";
                  }
                  first_run_ = false;
                }
              }

              // collect results
              std::map<std::string, int> hexModuleEntries;
              hexModuleEntries["econdQuality"] = maxGrade;
              hexModuleEntries["avgPayload"] = meanPayload;
              hexModuleEntries["stdPayload"] = stdPayload;

              // fill polygonal histograms (bins added in book())
              for (const std::string& plotKey : hexPlotsFastStreamKey_) {
                int errorCode = hexModuleEntries[plotKey];
                hexPlotsFastStream_[plotKey][layer]->setBinContent(module_bin_index + 1, errorCode);
              }
              module_bin_index++;
            }  // end of module loop
            cassette_idx++;
          }  // end of cassette loop

          // fill summary-level ECON-D TH2F
          econd_error_summarizer_.processAndFill(
              econdQuality_, me_econd_quality_summary_, layer_idx, ProcessMode::GRADE_TO_GRADE);
          layer_idx++;
        }  // end of layer loop
      }  // end of endcap loop
    }

  }  // namespace dqm
}  // namespace hgcal
