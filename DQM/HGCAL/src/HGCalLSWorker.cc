#include "DQM/HGCAL/interface/HGCalLSWorker.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"

namespace hgcal {
  namespace dqm {

    HGCalLSWorker::HGCalLSWorker(std::string folderRoot,
                                 EcondErrorSummarizer& econdErrorSummarizer,
                                 EcontErrorSummarizer& econtErrorSummarizer,
                                 bool skipTriggerDQM)
        : folderRoot_(std::move(folderRoot)),
          econd_error_summarizer_(econdErrorSummarizer),
          econt_error_summarizer_(econtErrorSummarizer),
          skipTriggerDQM_(skipTriggerDQM) {}

    void HGCalLSWorker::book(DQMStore::IBooker& ibooker, HGCalDQMGeometry const& geom) {
      ibooker.setCurrentFolder(folderRoot_ + "/LSSummary");

      size_t necondWithCBflags = econdWithCBflags.size();
      size_t necondTFlags = econdTFlags.size();
      size_t nLayers = geom.nLayers();

      // ECON-D vs LS
      me_econd_finequality_LS_ = ibooker.book2D(
          "econdFineQuality", ";LS;Header quality;", kMaxLS, 0, kMaxLS, necondWithCBflags, 0, necondWithCBflags);
      me_econd_quality_LS_ = ibooker.book2D(
          "econdQuality", ";LS;;", kMaxLS, 0, kMaxLS, qualityCategoryNames.size(), 0, qualityCategoryNames.size());
      me_econd_layer_LS_ =
          ibooker.book2D("econdLayerQuality", ";LS;Layers;", kMaxLS, 0.5, kMaxLS + 0.5, nLayers, 0, nLayers);

      addBinLabels(qualityCategoryNames, me_econd_quality_LS_, 2);
      addBinLabels(econdWithCBflags, me_econd_finequality_LS_, 2);

      me_econd_quality_LS_->getTH2F()->SetMinimum(1.0);
      me_econd_quality_LS_->getTH2F()->SetMaximum(5.0);
      me_econd_layer_LS_->getTH2F()->SetMinimum(1.0);
      me_econd_layer_LS_->getTH2F()->SetMaximum(5.0);

      // layer labels
      std::vector<std::string> layer_labels;
      for (auto i : geom.uniqueDirectionalLayers())
        layer_labels.push_back(std::to_string(i));
      addBinLabels(layer_labels, me_econd_layer_LS_, 2);

      // ECON-T vs LS
      if (!skipTriggerDQM_) {
        me_econt_finequality_LS_ =
            ibooker.book2D("econtFineQuality", ";LS;Exception;", kMaxLS, 0, kMaxLS, necondTFlags, 0, necondTFlags);
        me_econt_quality_LS_ = ibooker.book2D("econtQuality",
                                              ";LS;;",
                                              kMaxLS,
                                              0,
                                              kMaxLS,
                                              econtQualityCategoryNames.size(),
                                              0,
                                              econtQualityCategoryNames.size());
        me_econt_layer_LS_ = ibooker.book2D("econtLayerQuality", ";LS;Layers;", kMaxLS, 0, kMaxLS, nLayers, 0, nLayers);

        addBinLabels(econdTFlags, me_econt_finequality_LS_, 2);
        addBinLabels(econtQualityCategoryNames, me_econt_quality_LS_, 2);
        addBinLabels(layer_labels, me_econt_layer_LS_, 2);

        me_econt_quality_LS_->getTH2F()->SetMinimum(1.0);
        me_econt_quality_LS_->getTH2F()->SetMaximum(5.0);
        me_econt_layer_LS_->getTH2F()->SetMinimum(1.0);
        me_econt_layer_LS_->getTH2F()->SetMaximum(5.0);
      }
    }

    void HGCalLSWorker::endLumi(DQMStore::IBooker& ibooker,
                                DQMStore::IGetter& igetter,
                                HGCalDQMGeometry& geom,
                                edm::LuminosityBlock const& iLumi) {
      edm::LuminosityBlockNumber_t lumi = iLumi.luminosityBlock();
      int binNumber = lumi;

      // ECON-D fill
      me_econd_quality_layer_ = igetter.get(folderRoot_ + "/econd_lastLS");
      if (me_econd_quality_layer_) {
        econd_error_summarizer_.processAndFillLS(
            me_econd_quality_layer_, binNumber, me_econd_quality_LS_, me_econd_finequality_LS_, me_econd_layer_LS_);
      }

      // ECON-T fill
      if (!skipTriggerDQM_) {
        me_econt_quality_layer_ = igetter.get(folderRoot_ + "/econt_lastLS");
        if (me_econt_quality_layer_ && me_econt_quality_LS_ && me_econt_finequality_LS_ && me_econt_layer_LS_) {
          econt_error_summarizer_.processAndFill(
              me_econt_quality_layer_, me_econt_finequality_LS_, binNumber, ProcessMode::STAT_TO_STAT);
          econt_error_summarizer_.processAndFill(
              me_econt_quality_layer_, me_econt_quality_LS_, binNumber, ProcessMode::STAT_TO_GRADE);
          econt_error_summarizer_.processAndFill(
              me_econt_quality_layer_, me_econt_layer_LS_, binNumber, ProcessMode::STAT_TO_GRADE_X);
        }
      }

      // Zoom x-axis to show only filled lumisections
      double xmax = static_cast<double>(lumi + 1);
      me_econd_quality_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
      me_econd_finequality_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
      me_econd_layer_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
      if (!skipTriggerDQM_ && me_econt_quality_LS_ && me_econt_finequality_LS_ && me_econt_layer_LS_) {
        me_econt_quality_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
        me_econt_finequality_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
        me_econt_layer_LS_->getTH2F()->GetXaxis()->SetRangeUser(0, xmax);
      }
    }

  }  // namespace dqm
}  // namespace hgcal
