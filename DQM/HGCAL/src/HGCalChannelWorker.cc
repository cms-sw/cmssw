#include "DQM/HGCAL/interface/HGCalChannelWorker.h"

#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"

#include <cmath>
#include <algorithm>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TFile.h>
#include <TGraph.h>
#include <TH2Poly.h>
#include <TKey.h>
#include <TPolyMarker.h>

namespace hgcal {
  namespace dqm {

    namespace {
      const std::map<std::string, std::string> SummaryLabel = {
          {"occupancy", "Occupancy"},
          {"avgcm", getLabelForSummaryIndex(SummaryIndices_t::CMAVG)},
          {"avgadc", getLabelForSummaryIndex(SummaryIndices_t::PEDESTAL)},
          {"stdadc", getLabelForSummaryIndex(SummaryIndices_t::NOISE)},
          {"deltaadc", getLabelForSummaryIndex(SummaryIndices_t::DELTAPEDESTAL)},
          {"avgtoa", getLabelForSummaryIndex(SummaryIndices_t::TOAAVG)},
          {"avgtot", getLabelForSummaryIndex(SummaryIndices_t::TOTAVG)},
          {"n_vacant_channels", "Occupancy"},
          {"avgmips", getLabelForSummaryIndex(SummaryIndices_t::NMIPSAVG)},
          {"stdmips", getLabelForSummaryIndex(SummaryIndices_t::NMIPSSTD)},
          {"toaoccupancy", ""},
          {"totoccupancy", ""},
          {"noisy", "Number of Noisy Channels"},
          {"stuck", "Number of Stuck Channels"},
          {"saturated", "Number of Saturated Channels"}};
    }  // namespace

    HGCalChannelWorker::HGCalChannelWorker(std::string folderRoot,
                                           EcondErrorSummarizer& econdErrorSummarizer,
                                           float overflowThreshold,
                                           float saturatedAdcThreshold,
                                           bool enableOverflowMarkers)
        : folderRoot_(std::move(folderRoot)),
          econd_error_summarizer_(econdErrorSummarizer),
          overflow_threshold_(overflowThreshold),
          saturated_adc_threshold_(saturatedAdcThreshold),
          enable_overflow_markers_(enableOverflowMarkers) {}

    void HGCalChannelWorker::book(DQMStore::IBooker& ibooker, HGCalDQMGeometry const& geom) {
      auto const& HGCALMap = geom.hgcalMap();

      // Per-layer summaries.
      ibooker.setCurrentFolder(folderRoot_);
      for (auto const& variable : variables_) {
        Layer_[variable] = ibooker.book1D("layer_summary_" + variable,
                                          ";Layer;" + variable + ";" + SummaryLabel.at(variable),
                                          geom.nLayers(),
                                          0.5,
                                          geom.nLayers() + 0.5);
      }

      // Count modules belonging to each FED.
      std::map<uint32_t, int> fed_module_count;
      for (auto const& endcapPair : HGCALMap) {
        auto const& layerMap = endcapPair.second;
        for (auto const& layerPair : layerMap) {
          auto const& cassetteMap = layerPair.second;
          for (auto const& cassettePair : cassetteMap) {
            auto const& econdMap = cassettePair.second;
            for (auto const& econdPair : econdMap) {
              auto const& ele = econdPair.second;
              ++fed_module_count[ele.fedid];
            }
          }
        }
      }

      // Per-FED module summaries.
      for (auto const& fedPair : fed_module_count) {
        uint32_t const fedid = fedPair.first;
        int const nModules = fedPair.second;
        ibooker.setCurrentFolder(folderRoot_ + "/FED/FED_" + std::to_string(fedid) + "/");
        summary_ME_perFED_[fedid] = ibooker.book2D(
            "summaryPerModule_FED" + std::to_string(fedid), "; ; Counters;", nModules, -0.5, nModules - 0.5, 8, 0, 8);
      }

      // Layer-, cassette-, and module-level channel plots.
      auto const& corners_layer = geom.cornersLayer();
      auto const& corners_cassette = geom.cornersCassette();

      for (auto const& endcapPair : HGCALMap) {
        int const endcap = endcapPair.first;
        auto const& layerMap = endcapPair.second;

        std::string const endcapFolder = folderRoot_ + "/EndCap_" + endCapKey_.at(endcap) + "/";

        for (auto const& layerPair : layerMap) {
          int const layer = layerPair.first;
          auto const& cassetteMap = layerPair.second;

          std::string const layerTag = "_layer_" + std::to_string(std::abs(layer));
          std::string const layerStr = "Layer_" + std::to_string(std::abs(layer));
          std::string const layerFolder = endcapFolder + layerStr + "/";

          // Whole-layer module summaries.
          ibooker.setCurrentFolder(layerFolder);

          std::vector<std::string> layerVariables = variables_;
          layerVariables.insert(layerVariables.end(), {"noisy", "stuck", "saturated"});

          auto const& cornersL = corners_layer.at(layer);

          for (auto const& variable : layerVariables) {
            hexLayer_[layer][variable] = ibooker.book2DPoly("module_" + variable + layerTag,
                                                            layerStr + "; x[cm]; y[cm];" + SummaryLabel.at(variable),
                                                            cornersL[0],
                                                            cornersL[1],
                                                            cornersL[2],
                                                            cornersL[3]);
          }

          // Cassette and module-level channel plots.
          for (auto const& cassettePair : cassetteMap) {
            int const cassette = cassettePair.first;
            auto const& econdMap = cassettePair.second;
            std::string const cassetteFolder = layerFolder + "Cassette_" + std::to_string(cassette) + "/";

            ibooker.setCurrentFolder(cassetteFolder);

            auto const& cornersC = corners_cassette.at(layer).at(cassette);

            for (size_t vi = 0; vi < variables_.size(); ++vi) {
              std::string const& variable = variables_[vi];
              hexCassette_[layer][cassette][vi] =
                  ibooker.book2DPoly("hex_" + variable + layerTag,
                                     layerStr + "; x[cm]; y[cm];" + SummaryLabel.at(variable),
                                     cornersC[0],
                                     cornersC[1],
                                     cornersC[2],
                                     cornersC[3]);
            }

            for (auto const& econdPair : econdMap) {
              std::string const& typecode = econdPair.first;
              auto const& ele = econdPair.second;
              std::string const tag = "_module_" + std::to_string(ele.dqmIndex);
              std::string const uvStr = "(u" + std::to_string(ele.i1) + "-v" + std::to_string(ele.i2) + ") ";
              std::string const plotFolder = cassetteFolder + uvStr + typecode;
              ibooker.setCurrentFolder(plotFolder);

              auto const& scope = geom.moduleChannelScope(ele);
              float const xmin = scope.xmin;
              float const xmax = scope.xmax;
              float const ymin = scope.ymin;
              float const ymax = scope.ymax;

              size_t const nch = ele.nErx * 37;
              stdadc_me_[ele.dqmIndex] = ibooker.bookProfile(
                  "stdadc" + tag,
                  typecode + "; Channel; STD ADC;" + getLabelForSummaryIndex(SummaryIndices_t::NOISE),
                  nch,
                  -0.5,
                  nch - 0.5,
                  100,
                  0,
                  1024,
                  "s");

              auto& plots = hexPlots_[ele.dqmIndex];

              plots[enumIDX_avgcm] =
                  ibooker.book2DPoly("hex_avgcm" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::CMAVG),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_avgadc] =
                  ibooker.book2DPoly("hex_avgadc" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::PEDESTAL),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_stdadc] =
                  ibooker.book2DPoly("hex_stdadc" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::NOISE),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_deltaadc] = ibooker.book2DPoly(
                  "hex_deltaadc" + tag,
                  typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::DELTAPEDESTAL),
                  xmin,
                  xmax,
                  ymin,
                  ymax);

              plots[enumIDX_avgtoa] =
                  ibooker.book2DPoly("hex_avgtoa" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::TOAAVG),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_avgtot] =
                  ibooker.book2DPoly("hex_avgtot" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::TOTAVG),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_occupancy] = ibooker.book2DPoly(
                  "hex_occupancy" + tag, typecode + "; x[cm]; y[cm];Occupancy", xmin, xmax, ymin, ymax);

              plots[enumIDX_avgmips] =
                  ibooker.book2DPoly("hex_avgmips" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::NMIPSAVG),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_stdmips] =
                  ibooker.book2DPoly("hex_stdmips" + tag,
                                     typecode + "; x[cm]; y[cm];" + getLabelForSummaryIndex(SummaryIndices_t::NMIPSSTD),
                                     xmin,
                                     xmax,
                                     ymin,
                                     ymax);

              plots[enumIDX_toaoccupancy] = ibooker.book2DPoly(
                  "hex_toaoccupancy" + tag, typecode + "; x[cm]; y[cm];TOA Occupancy", xmin, xmax, ymin, ymax);

              plots[enumIDX_totoccupancy] = ibooker.book2DPoly(
                  "hex_totoccupancy" + tag, typecode + "; x[cm]; y[cm];TOT Occupancy", xmin, xmax, ymin, ymax);
            }
          }
        }
      }
    }

    void HGCalChannelWorker::endRun(DQMStore::IBooker& /*ibooker*/,
                                    DQMStore::IGetter& igetter,
                                    HGCalDQMGeometry& geom) {
      auto const& HGCALMap = geom.hgcalMap();

      // Known issue: fed_module_bin is keyed by a global module ordinal but looked up by
      // layer_idx below, so modules of the same layer share one bin of summaryPerModule_FED<id>.
      std::map<uint32_t, int> fed_module_count;
      std::map<uint32_t, std::vector<std::string>> fed_modules_typecode_map;
      std::map<uint32_t, std::map<uint32_t, int>> fed_module_bin;
      uint32_t globalModuleIndex = 0;

      for (auto const& endcapPair : HGCALMap) {
        auto const& layerMap = endcapPair.second;
        for (auto const& layerPair : layerMap) {
          auto const& cassetteMap = layerPair.second;
          for (auto const& cassettePair : cassetteMap) {
            auto const& econdMap = cassettePair.second;
            for (auto const& econdPair : econdMap) {
              auto const& ele = econdPair.second;
              ++fed_module_count[ele.fedid];
              fed_module_bin[ele.fedid][globalModuleIndex] = globalModuleIndex;
              ++globalModuleIndex;
              fed_modules_typecode_map[ele.fedid].push_back(ele.typecode);
            }
          }
        }
      }

      int layer_idx = 0;

      for (auto const& endcapPair : HGCALMap) {
        int const endcap = endcapPair.first;
        auto const& layerMap = endcapPair.second;
        std::string const endcapFolder = folderRoot_ + "/EndCap_" + endCapKey_.at(endcap) + "/";

        for (auto const& layerPair : layerMap) {
          int const layer = layerPair.first;
          auto const& cassetteMap = layerPair.second;
          std::string const layerStr = "Layer_" + std::to_string(std::abs(layer));
          std::string const layerFolder = endcapFolder + layerStr + "/";

          int module_count = 0;
          float value_layer[N_VARS] = {};

          for (auto const& cassettePair : cassetteMap) {
            int const cassette = cassettePair.first;
            auto const& econdMap = cassettePair.second;
            std::string const cassetteFolder = layerFolder + "Cassette_" + std::to_string(cassette) + "/";

            uint32_t iMod = 0;

            for (auto const& econdPair : econdMap) {
              std::string const& typecode = econdPair.first;
              auto const& ele = econdPair.second;
              std::string const uvStr = "(u" + std::to_string(ele.i1) + "-v" + std::to_string(ele.i2) + ") ";
              std::string const plotFolder = cassetteFolder + uvStr + typecode;

              auto const* avgadc_me = igetter.get(plotFolder + "/avgadc");
              auto const* avgtot_me = igetter.get(plotFolder + "/avgtot");
              auto const* avgtoa_me = igetter.get(plotFolder + "/avgtoa");
              auto const* avgcm_me = igetter.get(plotFolder + "/avgcm2");
              auto const* avgdeltaadc_me = igetter.get(plotFolder + "/avgdeltaadc");
              auto const* avgmips_me = igetter.get(plotFolder + "/avgrechit_nmips");

              // avgrechit_nmips is optional: it is only booked when HGCalRecHitDQM runs.
              if (!avgadc_me || !avgtot_me || !avgtoa_me || !avgcm_me || !avgdeltaadc_me) {
                edm::LogWarning("HGCalChannelWorker")
                    << "Skipping module " << typecode << " because required client MonitorElements are missing in "
                    << plotFolder;
                continue;
              }

              TFile* fgeo = geom.moduleTemplateFile(typecode, ele.isSiPM);
              if (!fgeo) {
                edm::LogWarning("HGCalChannelWorker")
                    << "Skipping module " << typecode << " because its geometry template file is unavailable.";
                continue;
              }

              int const v_coordinate = ele.i2;

              double const rot_sipm = M_PI / 36. - M_PI / 2. + static_cast<double>(v_coordinate) * M_PI / 18.;
              double const angleOffset = ele.isSiPM ? rot_sipm : -3. * M_PI / 6.;

              char const irot = ele.irot;
              double const x0 = ele.x0y0[0];
              double const y0 = ele.x0y0[1];

              float value_module[N_VARS] = {};
              float saturated_channels = 0.f;

              TKey* key = nullptr;
              TIter nextkey(fgeo->GetDirectory(nullptr)->GetListOfKeys());
              uint32_t iobj = 0;
              uint32_t deltaIdx = 0;

              auto* overflow_markers_module = new TPolyMarker();
              overflow_markers_module->SetMarkerStyle(5);
              overflow_markers_module->SetMarkerColor(kWhite);
              overflow_markers_module->SetBit(kCanDelete);

              auto* overflow_markers_cassette = new TPolyMarker();
              overflow_markers_cassette->SetMarkerStyle(5);
              overflow_markers_cassette->SetMarkerColor(kWhite);
              overflow_markers_cassette->SetBit(kCanDelete);

              float frac_adc_module = 0.f;
              float frac_tot_module = 0.f;
              float frac_toa_module = 0.f;
              float frac_occupancy_module = 0.f;
              float frac_stuckc_module = 0.f;
              float frac_noisyc_module = 0.f;
              float frac_saturatedc_module = 0.f;
              float frac_mpips_module = 0.f;

              auto& modulePlots = hexPlots_.at(ele.dqmIndex);

              // Loop over cells/channels from the cached module template.
              while ((key = static_cast<TKey*>(nextkey()))) {
                TObject* obj = key->ReadObj();
                if (!obj->InheritsFrom("TGraph")) {
                  continue;
                }

                auto* gr = static_cast<TGraph*>(obj);

                // Skip common-mode and known non-connected channels.
                bool const isCM = (iobj % 39 == 37) || (iobj % 39 == 38);

                bool isNC = false;
                int const r = iobj % 39;
                char const c4 = (typecode.length() >= 4) ? typecode[3] : '\0';

                if (typecode.length() >= 4 && typecode[0] == 'M' && typecode[1] == 'L' && typecode[2] == '_') {
                  if (c4 == 'F') {
                    isNC = (r == 8 || r == 17 || r == 19 || r == 28);
                  } else if (c4 == 'B') {
                    isNC = (iobj == 8 || iobj == 17 || iobj == 19 || iobj == 28 || iobj == 47 || iobj == 56 ||
                            iobj == 58 || iobj == 67 || iobj == 106);
                  } else if (c4 == 'L' || c4 == 'R') {
                    isNC = (iobj == 8 || iobj == 17 || iobj == 47 || iobj == 56 || iobj == 58 || iobj == 67 ||
                            iobj == 86 || iobj == 95);
                  }
                }

                bool const isSiPMNC = ele.isSiPM && ((iobj % 37 == 8) || (iobj % 37 == 17) || (iobj % 37 == 18) ||
                                                     (iobj % 37 == 19) || (iobj % 37 == 28));

                if (isCM && !ele.isSiPM) {
                  ++iobj;
                  continue;
                }

                if (isNC)
                  ++deltaIdx;
                if (isSiPMNC)
                  ++deltaIdx;

                HGCalDQMGeometry::rotateShape(gr, irot, angleOffset);

                unsigned const eRx = static_cast<unsigned>(iobj / 39);
                unsigned chIdx = iobj - eRx * 2;
                if (ele.isSiPM)
                  chIdx = iobj;

                int const bin = static_cast<int>(chIdx) + 1;

                float const nHits = avgadc_me->getBinEntries(bin) + avgtot_me->getBinEntries(bin);

                modulePlots[enumIDX_occupancy]->addBin(gr);
                modulePlots[enumIDX_occupancy]->setBinContent(bin, nHits);
                value_module[enumIDX_occupancy] += nHits;

                float const nHitsToa = avgtoa_me->getBinEntries(bin);
                modulePlots[enumIDX_toaoccupancy]->addBin(gr);
                modulePlots[enumIDX_toaoccupancy]->setBinContent(bin, nHitsToa);
                value_module[enumIDX_toaoccupancy] += nHitsToa;

                float const nHitsTot = avgtot_me->getBinEntries(bin);
                modulePlots[enumIDX_totoccupancy]->addBin(gr);
                modulePlots[enumIDX_totoccupancy]->setBinContent(bin, nHitsTot);
                value_module[enumIDX_totoccupancy] += nHitsTot;

                float const avgcm = avgcm_me->getBinContent(bin);
                modulePlots[enumIDX_avgcm]->addBin(gr);
                modulePlots[enumIDX_avgcm]->setBinContent(bin, avgcm);
                value_module[enumIDX_avgcm] += avgcm;

                float const avgadc = avgadc_me->getBinContent(bin);
                modulePlots[enumIDX_avgadc]->addBin(gr);
                modulePlots[enumIDX_avgadc]->setBinContent(bin, avgadc);
                value_module[enumIDX_avgadc] += avgadc;

                if (avgadc > saturated_adc_threshold_)
                  saturated_channels += 1.f;

                float const stdadc = avgadc_me->getBinError(bin);
                modulePlots[enumIDX_stdadc]->addBin(gr);
                modulePlots[enumIDX_stdadc]->setBinContent(bin, stdadc);
                value_module[enumIDX_stdadc] += stdadc;

                if (avgadc_me->getBinEntries(bin) > 0)
                  stdadc_me_.at(ele.dqmIndex)->Fill(bin - 1, stdadc);

                float const deltaadc = avgdeltaadc_me->getBinContent(bin);
                modulePlots[enumIDX_deltaadc]->addBin(gr);
                modulePlots[enumIDX_deltaadc]->setBinContent(bin, deltaadc);
                value_module[enumIDX_deltaadc] += deltaadc;

                float const avgtoa = avgtoa_me->getBinContent(bin);
                modulePlots[enumIDX_avgtoa]->addBin(gr);
                modulePlots[enumIDX_avgtoa]->setBinContent(bin, avgtoa);
                value_module[enumIDX_avgtoa] += avgtoa;

                float const avgtot = avgtot_me->getBinContent(bin);
                modulePlots[enumIDX_avgtot]->addBin(gr);
                modulePlots[enumIDX_avgtot]->setBinContent(bin, avgtot);
                value_module[enumIDX_avgtot] += avgtot;

                float avgmips = 0.f;
                float stdmips = 0.f;
                if (avgmips_me) {
                  avgmips = avgmips_me->getBinContent(bin);
                  stdmips = avgmips_me->getBinError(bin);
                }

                modulePlots[enumIDX_avgmips]->addBin(gr);
                modulePlots[enumIDX_avgmips]->setBinContent(bin, avgmips);
                value_module[enumIDX_avgmips] += avgmips;

                modulePlots[enumIDX_stdmips]->addBin(gr);
                modulePlots[enumIDX_stdmips]->setBinContent(bin, stdmips);
                value_module[enumIDX_stdmips] += stdmips;

                frac_adc_module += (avgadc > 0.f) ? 1.f : 0.f;
                frac_tot_module += (avgtot > 0.f) ? 1.f : 0.f;
                frac_toa_module += (avgtoa > 0.f) ? 1.f : 0.f;
                frac_occupancy_module += (nHits > 0.f) ? 1.f : 0.f;
                frac_stuckc_module += (stdadc > 0.f) ? 1.f : 0.f;
                frac_noisyc_module += (deltaadc > 0.f) ? 1.f : 0.f;
                frac_saturatedc_module += (avgadc > saturated_adc_threshold_) ? 1.f : 0.f;
                frac_mpips_module += (avgmips > 0.f) ? 1.f : 0.f;

                if (!isNC && !isSiPMNC) {
                  auto* grLayer = new TGraph(*gr);

                  if (!ele.isSiPM)
                    HGCalDQMGeometry::translateBin(grLayer, x0, y0);

                  int const hexbin = static_cast<int>(iMod + chIdx - deltaIdx + 1);

                  hexCassette_[layer][cassette][enumIDX_avgcm]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_avgcm]->setBinContent(hexbin, avgcm);

                  hexCassette_[layer][cassette][enumIDX_avgadc]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_avgadc]->setBinContent(hexbin, avgadc);

                  hexCassette_[layer][cassette][enumIDX_n_vacant_channels]->addBin(grLayer);
                  if (avgadc == 0.f) {
                    value_module[enumIDX_n_vacant_channels] += 1.f;
                    hexCassette_[layer][cassette][enumIDX_n_vacant_channels]->setBinContent(hexbin, 1.f);
                  }

                  hexCassette_[layer][cassette][enumIDX_stdadc]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_stdadc]->setBinContent(hexbin, stdadc);

                  if (enable_overflow_markers_ && (stdadc > overflow_threshold_)) {
                    double xmin = 0., xmax = 0.;
                    double ymin = 0., ymax = 0.;
                    gr->ComputeRange(xmin, ymin, xmax, ymax);

                    double const xCenter = (xmin + xmax) / 2.;
                    double const yCenter = (ymin + ymax) / 2.;

                    overflow_markers_module->SetNextPoint(xCenter, yCenter);
                    overflow_markers_cassette->SetNextPoint(xCenter + x0, yCenter + y0);
                  }

                  hexCassette_[layer][cassette][enumIDX_deltaadc]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_deltaadc]->setBinContent(hexbin, deltaadc);

                  hexCassette_[layer][cassette][enumIDX_avgtoa]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_avgtoa]->setBinContent(hexbin, avgtoa);

                  hexCassette_[layer][cassette][enumIDX_avgtot]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_avgtot]->setBinContent(hexbin, avgtot);

                  hexCassette_[layer][cassette][enumIDX_occupancy]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_occupancy]->setBinContent(hexbin, nHits);

                  hexCassette_[layer][cassette][enumIDX_avgmips]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_avgmips]->setBinContent(hexbin, avgmips);

                  hexCassette_[layer][cassette][enumIDX_stdmips]->addBin(grLayer);
                  hexCassette_[layer][cassette][enumIDX_stdmips]->setBinContent(hexbin, stdmips);
                }

                ++iobj;
              }

              if (overflow_markers_module->GetN() > 0) {
                auto* p = static_cast<TH2Poly*>(modulePlots[enumIDX_stdadc]->getTH2Poly());
                p->GetListOfFunctions()->Add(overflow_markers_module);
              } else {
                delete overflow_markers_module;
              }

              if (overflow_markers_cassette->GetN() > 0) {
                auto* p = static_cast<TH2Poly*>(hexCassette_[layer][cassette][enumIDX_stdadc]->getTH2Poly());
                p->GetListOfFunctions()->Add(overflow_markers_cassette);
              } else {
                delete overflow_markers_cassette;
              }

              int nCMSkipped = ele.isSiPM ? 0 : 2 * static_cast<int>(iobj / 39);
              iMod += iobj - nCMSkipped - deltaIdx;

              int const n_cells = 37 * ele.nErx;
              int const norm_cells = std::max(n_cells, 1);

              frac_adc_module /= norm_cells;
              frac_tot_module /= norm_cells;
              frac_toa_module /= norm_cells;
              frac_occupancy_module /= norm_cells;
              frac_stuckc_module /= norm_cells;
              frac_noisyc_module /= norm_cells;
              frac_saturatedc_module /= norm_cells;
              frac_mpips_module /= norm_cells;

              // Per-FED module summary.
              uint32_t const fedid = ele.fedid;
              int const fedBin = fed_module_bin[fedid][layer_idx] + 1;

              auto* fedSummary = summary_ME_perFED_.at(fedid);
              fedSummary->setBinContent(fedBin, 1, frac_adc_module);
              fedSummary->setBinContent(fedBin, 2, frac_tot_module);
              fedSummary->setBinContent(fedBin, 3, frac_toa_module);
              fedSummary->setBinContent(fedBin, 4, frac_occupancy_module);
              fedSummary->setBinContent(fedBin, 5, frac_stuckc_module);
              fedSummary->setBinContent(fedBin, 6, frac_noisyc_module);
              fedSummary->setBinContent(fedBin, 7, frac_saturatedc_module);
              fedSummary->setBinContent(fedBin, 8, frac_mpips_module);

              // Whole-layer module summaries.
              TGraph* grModule = geom.moduleBin(ele.dqmIndex);

              if (!grModule) {
                edm::LogError("HGCalChannelWorker") << "Could not obtain module bin for dqmIndex " << ele.dqmIndex;
              } else {
                grModule->SetName(typecode.c_str());

                for (size_t vi = 0; vi < variables_.size(); ++vi) {
                  std::string const& variable = variables_[vi];

                  if (vi != enumIDX_occupancy && vi != enumIDX_n_vacant_channels) {
                    value_module[vi] /= norm_cells;
                  }

                  hexLayer_[layer][variable]->addBin(grModule);

                  if (vi == enumIDX_stdadc) {
                    hexLayer_[layer][variable]->setBinContent(module_count + 1, value_module[vi]);

                    auto const stats = econd_error_summarizer_.analyzeChannelQuality(modulePlots[enumIDX_stdadc],
                                                                                     "channel_noise_threshold");

                    int const noisy_channels = stats.at("noisy");
                    int const stuck_channels = stats.at("stuck");

                    hexLayer_[layer]["noisy"]->addBin(grModule);
                    hexLayer_[layer]["noisy"]->setBinContent(module_count + 1, noisy_channels);

                    hexLayer_[layer]["stuck"]->addBin(grModule);
                    hexLayer_[layer]["stuck"]->setBinContent(module_count + 1, stuck_channels);

                    hexLayer_[layer]["saturated"]->addBin(grModule);
                    hexLayer_[layer]["saturated"]->setBinContent(module_count + 1, saturated_channels);
                  } else {
                    hexLayer_[layer][variable]->setBinContent(module_count + 1, value_module[vi]);
                  }

                  value_layer[vi] += value_module[vi];
                }
              }

              ++module_count;
            }
          }

          // Per-layer 1D summaries.
          for (size_t vi = 0; vi < variables_.size(); ++vi) {
            std::string const& variable = variables_[vi];
            Layer_[variable]->setBinLabel(layer_idx + 1, std::to_string(layer));
            Layer_[variable]->setBinContent(layer_idx + 1, value_layer[vi] / std::max(module_count, 1));
          }
          ++layer_idx;
        }
      }

      // Final labels and display settings for per-FED summaries.
      for (auto const& fedPair : fed_module_count) {
        uint32_t const fedid = fedPair.first;
        int const nModules = fedPair.second;

        auto* summary = summary_ME_perFED_.at(fedid);
        auto* hist = summary->getTH2F();

        hist->SetStats(kFALSE);
        hist->SetOption("textcolz");
        hist->GetXaxis()->SetNdivisions(-nModules, kFALSE);

        for (int i = 1; i <= nModules; ++i) {
          hist->GetXaxis()->SetBinLabel(i, fed_modules_typecode_map.at(fedid).at(i - 1).c_str());
        }

        hist->GetXaxis()->SetLabelSize(0.05);
        hist->GetXaxis()->CenterLabels(kTRUE);
        hist->GetXaxis()->SetDecimals(kFALSE);

        hist->GetYaxis()->SetBinLabel(1, "ADC Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(2, "TOT Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(3, "TOA Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(4, "Occupancy Ch. fr.");
        hist->GetYaxis()->SetBinLabel(5, "Stuck Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(6, "Noisy Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(7, "Saturated Ch. fr. ");
        hist->GetYaxis()->SetBinLabel(8, "MIPs Ch. fr.");

        hist->GetYaxis()->SetLabelSize(0.05);
        hist->GetYaxis()->SetTickLength(0.0);
      }
    }

  }  // namespace dqm
}  // namespace hgcal
