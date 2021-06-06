/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "RecoPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSProtonProducer(const edm::ParameterSet &);
  ~CTPPSProtonProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

  bool pixelDiscardBXShiftedTracks_;

  std::string lhcInfoLabel_;
  std::string opticsLabel_;

  unsigned int verbosity_;

  bool doSingleRPReconstruction_;
  bool doMultiRPReconstruction_;

  std::string singleRPReconstructionLabel_;
  std::string multiRPReconstructionLabel_;

  double localAngleXMin_, localAngleXMax_, localAngleYMin_, localAngleYMax_;

  struct AssociationCuts {
    bool x_cut_apply;
    double x_cut_mean, x_cut_value;
    bool y_cut_apply;
    double y_cut_mean, y_cut_value;
    bool xi_cut_apply;
    double xi_cut_mean, xi_cut_value;
    bool th_y_cut_apply;
    double th_y_cut_mean, th_y_cut_value;

    double ti_tr_min;
    double ti_tr_max;

    void load(const edm::ParameterSet &ps) {
      x_cut_apply = ps.getParameter<bool>("x_cut_apply");
      x_cut_mean = ps.getParameter<double>("x_cut_mean");
      x_cut_value = ps.getParameter<double>("x_cut_value");

      y_cut_apply = ps.getParameter<bool>("y_cut_apply");
      y_cut_mean = ps.getParameter<double>("y_cut_mean");
      y_cut_value = ps.getParameter<double>("y_cut_value");

      xi_cut_apply = ps.getParameter<bool>("xi_cut_apply");
      xi_cut_mean = ps.getParameter<double>("xi_cut_mean");
      xi_cut_value = ps.getParameter<double>("xi_cut_value");

      th_y_cut_apply = ps.getParameter<bool>("th_y_cut_apply");
      th_y_cut_mean = ps.getParameter<double>("th_y_cut_mean");
      th_y_cut_value = ps.getParameter<double>("th_y_cut_value");

      ti_tr_min = ps.getParameter<double>("ti_tr_min");
      ti_tr_max = ps.getParameter<double>("ti_tr_max");
    }

    static edm::ParameterSetDescription getDefaultParameters() {
      edm::ParameterSetDescription desc;

      desc.add<bool>("x_cut_apply", false)->setComment("whether to apply track-association cut in x");
      desc.add<double>("x_cut_mean", 0E-6)->setComment("mean of track-association cut in x, mm");
      desc.add<double>("x_cut_value", 800E-6)->setComment("threshold of track-association cut in x, mm");

      desc.add<bool>("y_cut_apply", false)->setComment("whether to apply track-association cut in y");
      desc.add<double>("y_cut_mean", 0E-6)->setComment("mean of track-association cut in y, mm");
      desc.add<double>("y_cut_value", 600E-6)->setComment("threshold of track-association cut in y, mm");

      desc.add<bool>("xi_cut_apply", true)->setComment("whether to apply track-association cut in xi");
      desc.add<double>("xi_cut_mean", 0.)->setComment("mean of track-association cut in xi");
      desc.add<double>("xi_cut_value", 0.013)->setComment("threshold of track-association cut in xi");

      desc.add<bool>("th_y_cut_apply", true)->setComment("whether to apply track-association cut in th_y");
      desc.add<double>("th_y_cut_mean", 0E-6)->setComment("mean of track-association cut in th_y, rad");
      desc.add<double>("th_y_cut_value", 20E-6)->setComment("threshold of track-association cut in th_y, rad");

      desc.add<double>("ti_tr_min", -1.)->setComment("minimum value for timing-tracking association cut");
      desc.add<double>("ti_tr_max", +1.)->setComment("maximum value for timing-tracking association cut");

      return desc;
    }
  };

  std::map<unsigned int, AssociationCuts> association_cuts_;  // map: arm -> AssociationCuts

  unsigned int max_n_timing_tracks_;
  double default_time_;

  ProtonReconstructionAlgorithm algorithm_;

  bool opticsValid_;
  edm::ESWatcher<CTPPSInterpolatedOpticsRcd> opticsWatcher_;

  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> opticalFunctionsToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geometryToken_;
};

//----------------------------------------------------------------------------------------------------

CTPPSProtonProducer::CTPPSProtonProducer(const edm::ParameterSet &iConfig)
    : tracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagLocalTrackLite"))),

      pixelDiscardBXShiftedTracks_(iConfig.getParameter<bool>("pixelDiscardBXShiftedTracks")),

      lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),
      opticsLabel_(iConfig.getParameter<std::string>("opticsLabel")),
      verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
      doSingleRPReconstruction_(iConfig.getParameter<bool>("doSingleRPReconstruction")),
      doMultiRPReconstruction_(iConfig.getParameter<bool>("doMultiRPReconstruction")),
      singleRPReconstructionLabel_(iConfig.getParameter<std::string>("singleRPReconstructionLabel")),
      multiRPReconstructionLabel_(iConfig.getParameter<std::string>("multiRPReconstructionLabel")),

      localAngleXMin_(iConfig.getParameter<double>("localAngleXMin")),
      localAngleXMax_(iConfig.getParameter<double>("localAngleXMax")),
      localAngleYMin_(iConfig.getParameter<double>("localAngleYMin")),
      localAngleYMax_(iConfig.getParameter<double>("localAngleYMax")),

      max_n_timing_tracks_(iConfig.getParameter<unsigned int>("max_n_timing_tracks")),
      default_time_(iConfig.getParameter<double>("default_time")),

      algorithm_(iConfig.getParameter<bool>("fitVtxY"),
                 iConfig.getParameter<bool>("useImprovedInitialEstimate"),
                 iConfig.getParameter<std::string>("multiRPAlgorithm"),
                 verbosity_),
      opticsValid_(false),
      lhcInfoToken_(esConsumes<LHCInfo, LHCInfoRcd>(edm::ESInputTag("", lhcInfoLabel_))),
      opticalFunctionsToken_(esConsumes<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd>(
          edm::ESInputTag("", opticsLabel_))),
      geometryToken_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord>()) {
  for (const std::string &sector : {"45", "56"}) {
    const unsigned int arm = (sector == "45") ? 0 : 1;
    association_cuts_[arm].load(iConfig.getParameterSet("association_cuts_" + sector));
  }

  if (doSingleRPReconstruction_)
    produces<reco::ForwardProtonCollection>(singleRPReconstructionLabel_);

  if (doMultiRPReconstruction_)
    produces<reco::ForwardProtonCollection>(multiRPReconstructionLabel_);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tagLocalTrackLite", edm::InputTag("ctppsLocalTrackLiteProducer"))
      ->setComment("specification of the input lite-track collection");

  desc.add<bool>("pixelDiscardBXShiftedTracks", false)
      ->setComment("whether to discard pixel tracks built from BX-shifted planes");

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("opticsLabel", "")->setComment("label of the optics record");

  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");

  desc.add<bool>("doSingleRPReconstruction", true)
      ->setComment("flag whether to apply single-RP reconstruction strategy");

  desc.add<bool>("doMultiRPReconstruction", true)->setComment("flag whether to apply multi-RP reconstruction strategy");

  desc.add<std::string>("singleRPReconstructionLabel", "singleRP")
      ->setComment("output label for single-RP reconstruction products");

  desc.add<std::string>("multiRPReconstructionLabel", "multiRP")
      ->setComment("output label for multi-RP reconstruction products");

  desc.add<double>("localAngleXMin", -0.03)->setComment("minimal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleXMax", +0.03)->setComment("maximal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleYMin", -0.04)->setComment("minimal accepted value of local vertical angle (rad)");
  desc.add<double>("localAngleYMax", +0.04)->setComment("maximal accepted value of local vertical angle (rad)");

  for (const std::string &sector : {"45", "56"}) {
    desc.add<edm::ParameterSetDescription>("association_cuts_" + sector, AssociationCuts::getDefaultParameters())
        ->setComment("track-association cuts for sector " + sector);
  }

  std::vector<edm::ParameterSet> config;

  desc.add<unsigned int>("max_n_timing_tracks", 5)->setComment("maximum number of timing tracks per RP");

  desc.add<double>("default_time", 0.)->setComment("proton time to be used when no timing information available");

  desc.add<bool>("fitVtxY", true)
      ->setComment("for multi-RP reconstruction, flag whether y* should be free fit parameter");

  desc.add<bool>("useImprovedInitialEstimate", true)
      ->setComment(
          "for multi-RP reconstruction, flag whether a quadratic estimate of the initial point should be used");

  desc.add<std::string>("multiRPAlgorithm", "chi2")
      ->setComment("algorithm for multi-RP reco, options include chi2, newton, anal-iter");

  descriptions.add("ctppsProtons", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  iEvent.getByToken(tracksToken_, hTracks);

  // book output
  std::unique_ptr<reco::ForwardProtonCollection> pOutSingleRP(new reco::ForwardProtonCollection);
  std::unique_ptr<reco::ForwardProtonCollection> pOutMultiRP(new reco::ForwardProtonCollection);

  // continue only if there is something to process
  // NB: this avoids loading (possibly non-existing) conditions in workflows without proton data
  if (!hTracks->empty()) {
    // get conditions
    edm::ESHandle<LHCInfo> hLHCInfo = iSetup.getHandle(lhcInfoToken_);

    edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> hOpticalFunctions =
        iSetup.getHandle(opticalFunctionsToken_);

    edm::ESHandle<CTPPSGeometry> hGeometry = iSetup.getHandle(geometryToken_);

    // re-initialise algorithm upon crossing-angle change
    if (opticsWatcher_.check(iSetup)) {
      if (hOpticalFunctions->empty()) {
        edm::LogInfo("CTPPSProtonProducer") << "No optical functions available, reconstruction disabled.";
        algorithm_.release();
        opticsValid_ = false;
      } else {
        algorithm_.init(*hOpticalFunctions);
        opticsValid_ = true;
      }
    }

    // do reconstruction only if optics is valid
    if (opticsValid_) {
      // prepare log
      std::ostringstream ssLog;
      if (verbosity_)
        ssLog << "* input tracks:" << std::endl;

      // select tracks with small local angles, split them by LHC sector and tracker/timing RPs
      std::map<unsigned int, std::vector<unsigned int>> trackingSelection, timingSelection;

      for (unsigned int idx = 0; idx < hTracks->size(); ++idx) {
        const auto &tr = hTracks->at(idx);

        if (tr.tx() < localAngleXMin_ || tr.tx() > localAngleXMax_ || tr.ty() < localAngleYMin_ ||
            tr.ty() > localAngleYMax_)
          continue;

        if (pixelDiscardBXShiftedTracks_) {
          if (tr.pixelTrackRecoInfo() == CTPPSpixelLocalTrackReconstructionInfo::allShiftedPlanes ||
              tr.pixelTrackRecoInfo() == CTPPSpixelLocalTrackReconstructionInfo::mixedPlanes)
            continue;
        }

        const CTPPSDetId rpId(tr.rpId());

        if (verbosity_)
          ssLog << "\t"
                << "[" << idx << "] " << tr.rpId() << " (" << (rpId.arm() * 100 + rpId.station() * 10 + rpId.rp())
                << "): "
                << "x=" << tr.x() << " +- " << tr.xUnc() << " mm, "
                << "y=" << tr.y() << " +- " << tr.yUnc() << " mm" << std::endl;

        const bool trackerRP =
            (rpId.subdetId() == CTPPSDetId::sdTrackingStrip || rpId.subdetId() == CTPPSDetId::sdTrackingPixel);

        if (trackerRP)
          trackingSelection[rpId.arm()].push_back(idx);
        else
          timingSelection[rpId.arm()].push_back(idx);
      }

      // process each arm
      for (const auto &arm_it : trackingSelection) {
        const auto &indices = arm_it.second;

        const auto &ac = association_cuts_[arm_it.first];

        // do single-RP reco if needed
        std::map<unsigned int, reco::ForwardProton> singleRPResultsIndexed;
        if (doSingleRPReconstruction_ || ac.xi_cut_apply || ac.th_y_cut_apply) {
          for (const auto &idx : indices) {
            if (verbosity_)
              ssLog << std::endl << "* reconstruction from track " << idx << std::endl;

            singleRPResultsIndexed[idx] =
                algorithm_.reconstructFromSingleRP(CTPPSLocalTrackLiteRef(hTracks, idx), *hLHCInfo, ssLog);
          }
        }

        // check that exactly two tracking RPs are involved
        //    - 1 is insufficient for multi-RP reconstruction
        //    - PPS did not use more than 2 tracking RPs per arm -> algorithms are tuned to this
        std::set<unsigned int> rpIds;
        for (const auto &idx : indices)
          rpIds.insert(hTracks->at(idx).rpId());

        // do multi-RP reco if chosen
        if (doMultiRPReconstruction_ && rpIds.size() == 2) {
          // find matching track pairs from different tracking RPs, ordered: i=near, j=far RP
          std::vector<std::pair<unsigned int, unsigned int>> idx_pairs;
          std::map<unsigned int, unsigned int> idx_pair_multiplicity;
          for (const auto &i : indices) {
            for (const auto &j : indices) {
              const auto &tr_i = hTracks->at(i);
              const auto &tr_j = hTracks->at(j);

              const double z_i = hGeometry->rpTranslation(tr_i.rpId()).z();
              const double z_j = hGeometry->rpTranslation(tr_j.rpId()).z();

              const auto &pr_i = singleRPResultsIndexed[i];
              const auto &pr_j = singleRPResultsIndexed[j];

              if (tr_i.rpId() == tr_j.rpId())
                continue;

              if (std::abs(z_i) >= std::abs(z_j))
                continue;

              bool matching = true;

              if (ac.x_cut_apply && std::abs(tr_i.x() - tr_j.x() - ac.x_cut_mean) > ac.x_cut_value)
                matching = false;
              else if (ac.y_cut_apply && std::abs(tr_i.y() - tr_j.y() - ac.y_cut_mean) > ac.y_cut_value)
                matching = false;
              else if (ac.xi_cut_apply && std::abs(pr_i.xi() - pr_j.xi() - ac.xi_cut_mean) > ac.xi_cut_value)
                matching = false;
              else if (ac.th_y_cut_apply &&
                       std::abs(pr_i.thetaY() - pr_j.thetaY() - ac.th_y_cut_mean) > ac.th_y_cut_value)
                matching = false;

              if (!matching)
                continue;

              idx_pairs.emplace_back(i, j);
              idx_pair_multiplicity[i]++;
              idx_pair_multiplicity[j]++;
            }
          }

          // evaluate track multiplicity in each timing RP
          std::map<unsigned int, unsigned int> timing_RP_track_multiplicity;
          for (const auto &ti : timingSelection[arm_it.first]) {
            const auto &tr = hTracks->at(ti);
            timing_RP_track_multiplicity[tr.rpId()]++;
          }

          // associate tracking-RP pairs with timing-RP tracks
          std::map<unsigned int, std::vector<unsigned int>> matched_timing_track_indices;
          std::map<unsigned int, unsigned int> matched_timing_track_multiplicity;
          for (unsigned int pr_idx = 0; pr_idx < idx_pairs.size(); ++pr_idx) {
            const auto &i = idx_pairs[pr_idx].first;
            const auto &j = idx_pairs[pr_idx].second;

            // skip non-unique associations
            if (idx_pair_multiplicity[i] > 1 || idx_pair_multiplicity[j] > 1)
              continue;

            const auto &tr_i = hTracks->at(i);
            const auto &tr_j = hTracks->at(j);

            const double z_i = hGeometry->rpTranslation(tr_i.rpId()).z();
            const double z_j = hGeometry->rpTranslation(tr_j.rpId()).z();

            for (const auto &ti : timingSelection[arm_it.first]) {
              const auto &tr_ti = hTracks->at(ti);

              // skip if timing RP saturated (high track multiplicity)
              if (timing_RP_track_multiplicity[tr_ti.rpId()] > max_n_timing_tracks_)
                continue;

              // interpolation from tracking RPs
              const double z_ti = hGeometry->rpTranslation(tr_ti.rpId()).z();
              const double f_i = (z_ti - z_j) / (z_i - z_j), f_j = (z_i - z_ti) / (z_i - z_j);
              const double x_inter = f_i * tr_i.x() + f_j * tr_j.x();
              const double x_inter_unc_sq =
                  f_i * f_i * tr_i.xUnc() * tr_i.xUnc() + f_j * f_j * tr_j.xUnc() * tr_j.xUnc();

              const double de_x = tr_ti.x() - x_inter;
              const double de_x_unc = sqrt(tr_ti.xUnc() * tr_ti.xUnc() + x_inter_unc_sq);
              const double r = (de_x_unc > 0.) ? de_x / de_x_unc : 1E100;

              const bool matching = (ac.ti_tr_min < r && r < ac.ti_tr_max);

              if (verbosity_)
                ssLog << "ti=" << ti << ", i=" << i << ", j=" << j << " | z_ti=" << z_ti << ", z_i=" << z_i
                      << ", z_j=" << z_j << " | x_ti=" << tr_ti.x() << ", x_inter=" << x_inter << ", de_x=" << de_x
                      << ", de_x_unc=" << de_x_unc << ", matching=" << matching << std::endl;

              if (!matching)
                continue;

              matched_timing_track_indices[pr_idx].push_back(ti);
              matched_timing_track_multiplicity[ti]++;
            }
          }

          // process associated tracks
          for (unsigned int pr_idx = 0; pr_idx < idx_pairs.size(); ++pr_idx) {
            const auto &i = idx_pairs[pr_idx].first;
            const auto &j = idx_pairs[pr_idx].second;

            // skip non-unique associations of tracking-RP tracks
            if (idx_pair_multiplicity[i] > 1 || idx_pair_multiplicity[j] > 1)
              continue;

            if (verbosity_)
              ssLog << std::endl
                    << "* reconstruction from tracking-RP tracks: " << i << ", " << j << " and timing-RP tracks: ";

            // buffer contributing tracks
            CTPPSLocalTrackLiteRefVector sel_tracks;
            sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, i));
            sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, j));

            CTPPSLocalTrackLiteRefVector sel_track_for_kin_reco = sel_tracks;

            // process timing-RP data
            double sw = 0., swt = 0.;
            for (const auto &ti : matched_timing_track_indices[pr_idx]) {
              // skip non-unique associations of timing-RP tracks
              if (matched_timing_track_multiplicity[ti] > 1)
                continue;

              sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, ti));

              if (verbosity_)
                ssLog << ti << ", ";

              const auto &tr = hTracks->at(ti);
              const double t_unc = tr.timeUnc();
              const double w = (t_unc > 0.) ? 1. / t_unc / t_unc : 1.;
              sw += w;
              swt += w * tr.time();
            }

            float time = default_time_, time_unc = 0.;
            if (sw > 0.) {
              time = swt / sw;
              time_unc = 1. / sqrt(sw);
            }

            if (verbosity_)
              ssLog << std::endl << "    time = " << time << " +- " << time_unc << std::endl;

            // process tracking-RP data
            reco::ForwardProton proton = algorithm_.reconstructFromMultiRP(sel_track_for_kin_reco, *hLHCInfo, ssLog);

            // save combined output
            proton.setContributingLocalTracks(sel_tracks);
            proton.setTime(time);
            proton.setTimeError(time_unc);

            pOutMultiRP->emplace_back(proton);
          }
        }

        // save single-RP results (un-indexed)
        for (const auto &p : singleRPResultsIndexed)
          pOutSingleRP->emplace_back(p.second);
      }

      // dump log
      if (verbosity_)
        edm::LogInfo("CTPPSProtonProducer") << ssLog.str();
    }
  }

  // save output
  if (doSingleRPReconstruction_)
    iEvent.put(std::move(pOutSingleRP), singleRPReconstructionLabel_);

  if (doMultiRPReconstruction_)
    iEvent.put(std::move(pOutMultiRP), multiRPReconstructionLabel_);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonProducer);
