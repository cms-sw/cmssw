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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonProducer(const edm::ParameterSet&);
    ~CTPPSProtonProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

    std::string lhcInfoLabel_;

    unsigned int verbosity_;

    bool doSingleRPReconstruction_;
    bool doMultiRPReconstruction_;

    std::string singleRPReconstructionLabel_;
    std::string multiRPReconstructionLabel_;

    double localAngleXMin_, localAngleXMax_, localAngleYMin_, localAngleYMax_;

    struct AssociationCuts
    {
      bool x_cut_apply;
      double x_cut_value;
      bool y_cut_apply;
      double y_cut_value;
      bool xi_cut_apply;
      double xi_cut_value;
      bool th_y_cut_apply;
      double th_y_cut_value;

      void load(const edm::ParameterSet &ps)
      {
        x_cut_apply    = ps.getParameter<bool>  ("x_cut_apply");
        x_cut_value    = ps.getParameter<double>("x_cut_value");
        y_cut_apply    = ps.getParameter<bool>  ("y_cut_apply");
        y_cut_value    = ps.getParameter<double>("y_cut_value");
        xi_cut_apply   = ps.getParameter<bool>  ("xi_cut_apply");
        xi_cut_value   = ps.getParameter<double>("xi_cut_value");
        th_y_cut_apply = ps.getParameter<bool>  ("th_y_cut_apply");
        th_y_cut_value = ps.getParameter<double>("th_y_cut_value");
      }

      static edm::ParameterSetDescription getDefaultParameters()
      {
        edm::ParameterSetDescription desc;

        desc.add<bool>("x_cut_apply", false)->setComment("whether to apply track-association cut in x");
        desc.add<double>("x_cut_value", 800E-6)->setComment("threshold of track-association cut in x, mm");
        desc.add<bool>("y_cut_apply", false)->setComment("whether to apply track-association cut in y");
        desc.add<double>("y_cut_value", 600E-6)->setComment("threshold of track-association cut in y, mm");
        desc.add<bool>("xi_cut_apply", true)->setComment("whether to apply track-association cut in xi");
        desc.add<double>("xi_cut_value", 0.013)->setComment("threshold of track-association cut in xi");
        desc.add<bool>("th_y_cut_apply", true)->setComment("whether to apply track-association cut in th_y");
        desc.add<double>("th_y_cut_value", 20E-6)->setComment("threshold of track-association cut in th_y, rad");

        return desc;
      }
    };

    std::map<unsigned int, AssociationCuts> association_cuts_;  // map: arm -> AssociationCuts

    unsigned int max_n_timing_tracks_;

    ProtonReconstructionAlgorithm algorithm_;

    bool opticsValid_;
    edm::ESWatcher<CTPPSInterpolatedOpticsRcd> opticsWatcher_;
};

//----------------------------------------------------------------------------------------------------

CTPPSProtonProducer::CTPPSProtonProducer(const edm::ParameterSet& iConfig) :
  tracksToken_                (consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagLocalTrackLite"))),
  lhcInfoLabel_               (iConfig.getParameter<std::string>("lhcInfoLabel")),
  verbosity_                  (iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
  doSingleRPReconstruction_   (iConfig.getParameter<bool>("doSingleRPReconstruction")),
  doMultiRPReconstruction_    (iConfig.getParameter<bool>("doMultiRPReconstruction")),
  singleRPReconstructionLabel_(iConfig.getParameter<std::string>("singleRPReconstructionLabel")),
  multiRPReconstructionLabel_ (iConfig.getParameter<std::string>("multiRPReconstructionLabel")),

  localAngleXMin_             (iConfig.getParameter<double>("localAngleXMin")),
  localAngleXMax_             (iConfig.getParameter<double>("localAngleXMax")),
  localAngleYMin_             (iConfig.getParameter<double>("localAngleYMin")),
  localAngleYMax_             (iConfig.getParameter<double>("localAngleYMax")),

  max_n_timing_tracks_        (iConfig.getParameter<unsigned int>("max_n_timing_tracks")),

  algorithm_                  (iConfig.getParameter<bool>("fitVtxY"), iConfig.getParameter<bool>("useImprovedInitialEstimate"), verbosity_),
  opticsValid_(false)
{
  for (const std::string &sector : { "45", "56" })
  {
    const unsigned int arm = (sector == "45") ? 0 : 1;
    association_cuts_[arm].load(iConfig.getParameterSet("association_cuts_" + sector));
  }

  if (doSingleRPReconstruction_)
    produces<reco::ForwardProtonCollection>(singleRPReconstructionLabel_);

  if (doMultiRPReconstruction_)
    produces<reco::ForwardProtonCollection>(multiRPReconstructionLabel_);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tagLocalTrackLite", edm::InputTag("ctppsLocalTrackLiteProducer"))
    ->setComment("specification of the input lite-track collection");

  desc.add<std::string>("lhcInfoLabel", "")
    ->setComment("label of the LHCInfo record");

  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");

  desc.add<bool>("doSingleRPReconstruction", true)
    ->setComment("flag whether to apply single-RP reconstruction strategy");

  desc.add<bool>("doMultiRPReconstruction", true)
    ->setComment("flag whether to apply multi-RP reconstruction strategy");

  desc.add<std::string>("singleRPReconstructionLabel", "singleRP")
    ->setComment("output label for single-RP reconstruction products");

  desc.add<std::string>("multiRPReconstructionLabel", "multiRP")
    ->setComment("output label for multi-RP reconstruction products");

  desc.add<double>("localAngleXMin", -0.03)->setComment("minimal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleXMax", +0.03)->setComment("maximal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleYMin", -0.04)->setComment("minimal accepted value of local vertical angle (rad)");
  desc.add<double>("localAngleYMax", +0.04)->setComment("maximal accepted value of local vertical angle (rad)");

  for (const std::string &sector : { "45", "56" })
  {
    desc.add<edm::ParameterSetDescription>("association_cuts_" + sector, AssociationCuts::getDefaultParameters())
      ->setComment("track-association cuts for sector " + sector);
  }

  std::vector<edm::ParameterSet> config;

  desc.add<unsigned int>("max_n_timing_tracks", 5)->setComment("maximum number of timing tracks per RP");

  desc.add<bool>("fitVtxY", true)
    ->setComment("for multi-RP reconstruction, flag whether y* should be free fit parameter");

  desc.add<bool>("useImprovedInitialEstimate", true)
    ->setComment("for multi-RP reconstruction, flag whether a quadratic estimate of the initial point should be used");

  descriptions.add("ctppsProtons", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  iEvent.getByToken(tracksToken_, hTracks);

  // book output
  std::unique_ptr<reco::ForwardProtonCollection> pOutSingleRP(new reco::ForwardProtonCollection);
  std::unique_ptr<reco::ForwardProtonCollection> pOutMultiRP(new reco::ForwardProtonCollection);

  // continue only if there is something to process
  // NB: this avoids loading (possibly non-existing) conditions in workflows without proton data
  if (!hTracks->empty())
  {
    // get conditions
    edm::ESHandle<LHCInfo> hLHCInfo;
    iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, hLHCInfo);

    edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> hOpticalFunctions;
    iSetup.get<CTPPSInterpolatedOpticsRcd>().get(hOpticalFunctions);

    edm::ESHandle<CTPPSGeometry> hGeometry;
    iSetup.get<VeryForwardRealGeometryRecord>().get(hGeometry);

    // re-initialise algorithm upon crossing-angle change
    if (opticsWatcher_.check(iSetup))
    {
      if (hOpticalFunctions->empty()) {
        edm::LogInfo("CTPPSProtonProducer") << "No optical functions available, reconstruction disabled.";
        algorithm_.release();
        opticsValid_ = false;
      }
      else {
        algorithm_.init(*hOpticalFunctions);
        opticsValid_ = true;
      }
    }

    // do reconstruction only if optics is valid
    if (opticsValid_)
    {
      // prepare log
      std::ostringstream ssLog;
      if (verbosity_)
        ssLog << "* input tracks:" << std::endl;

      // select tracks with small local angles, split them by LHC sector and tracker/timing RPs
      std::map<unsigned int, std::vector<unsigned int>> trackingSelection, timingSelection;

      for (unsigned int idx = 0; idx < hTracks->size(); ++idx)
      {
        const auto& tr = hTracks->at(idx);

        if (tr.getTx() < localAngleXMin_ || tr.getTx() > localAngleXMax_
            || tr.getTy() < localAngleYMin_ || tr.getTy() > localAngleYMax_)
          continue;

        const CTPPSDetId rpId(tr.getRPId());

        if (verbosity_)
          ssLog << "\t"
            << "[" << idx << "] "
            << tr.getRPId() << " (" << (rpId.arm()*100 + rpId.station()*10 + rpId.rp()) << "): "
            << "x=" << tr.getX() << " +- " << tr.getXUnc() << " mm, "
            << "y=" << tr.getY() << " +- " << tr.getYUnc() << " mm" << std::endl;

        const bool trackerRP = (rpId.subdetId() == CTPPSDetId::sdTrackingStrip || rpId.subdetId() == CTPPSDetId::sdTrackingPixel);

        if (trackerRP)
          trackingSelection[rpId.arm()].push_back(idx);
        else
          timingSelection[rpId.arm()].push_back(idx);
      }

      // process each arm
      for (const auto &arm_it : trackingSelection)
      {
        const auto &indices = arm_it.second;

        const auto &ac = association_cuts_[arm_it.first];

        // do single-RP reco if needed
        std::map<unsigned int, reco::ForwardProton> singleRPResultsIndexed;
        if (doSingleRPReconstruction_ || ac.xi_cut_apply || ac.th_y_cut_apply)
        {
          for (const auto &idx : indices)
          {
            if (verbosity_)
              ssLog << std::endl << "* reconstruction from track " << idx << std::endl;

            singleRPResultsIndexed[idx] = algorithm_.reconstructFromSingleRP(CTPPSLocalTrackLiteRef(hTracks, idx), *hLHCInfo, ssLog);
          }
        }

        // check that exactly two tracking RPs are involved
        //    - 1 is insufficient for multi-RP reconstruction
        //    - PPS did not use more than 2 tracking RPs per arm -> algorithms are tuned to this
        std::set<unsigned int> rpIds;
        for (const auto &idx : indices)
          rpIds.insert(hTracks->at(idx).getRPId());

        // do multi-RP reco if chosen
        if (doMultiRPReconstruction_ && rpIds.size() == 2)
        {
          // find matching track pairs from different tracking RPs
          std::vector<std::pair<unsigned int, unsigned int>> idx_pairs;
          std::map<unsigned int, unsigned int> idx_pair_multiplicity;
          for (const auto &i : indices)
          {
            for (const auto &j : indices)
            {
              if (j <= i)
                continue;

              const auto &tr_i = hTracks->at(i);
              const auto &tr_j = hTracks->at(j);

              const auto &pr_i = singleRPResultsIndexed[i];
              const auto &pr_j = singleRPResultsIndexed[j];

              if (tr_i.getRPId() == tr_j.getRPId())
                continue;

              bool matching = true;

              if (ac.x_cut_apply && std::abs(tr_i.getX() - tr_j.getX()) > ac.x_cut_value)
                matching = false;
              if (ac.y_cut_apply && std::abs(tr_i.getY() - tr_j.getY()) > ac.y_cut_value)
                matching = false;
              if (ac.xi_cut_apply && std::abs(pr_i.xi() - pr_j.xi()) > ac.xi_cut_value)
                matching = false;
              if (ac.th_y_cut_apply && std::abs(pr_i.thetaY() - pr_j.thetaY()) > ac.th_y_cut_value)
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
          for (const auto &ti : timingSelection[arm_it.first])
          {
            const auto &tr = hTracks->at(ti);
            timing_RP_track_multiplicity[tr.getRPId()]++;
          }

          // associate tracking-RP pairs with timing-RP tracks
          std::map<unsigned int, std::vector<unsigned int>> matched_timing_track_indices;
          std::map<unsigned int, unsigned int> matched_timing_track_multiplicity;
          for (unsigned int pr_idx = 0; pr_idx < idx_pairs.size(); ++pr_idx)
          {
            const auto &i = idx_pairs[pr_idx].first;
            const auto &j = idx_pairs[pr_idx].second;

            // skip non-unique associations
            if (idx_pair_multiplicity[i] > 1 || idx_pair_multiplicity[j] > 1)
              continue;

            const auto &tr_i = hTracks->at(i);
            const auto &tr_j = hTracks->at(j);

            const double z_i = hGeometry->getRPTranslation(tr_i.getRPId()).z();
            const double z_j = hGeometry->getRPTranslation(tr_j.getRPId()).z();

            for (const auto &ti : timingSelection[arm_it.first])
            {
              const auto &tr_ti = hTracks->at(ti);

              // skip if timing RP saturated (high track multiplicity)
              if (timing_RP_track_multiplicity[tr_ti.getRPId()] > max_n_timing_tracks_)
                continue;

              // interpolation from tracking RPs
              const double z_ti = - hGeometry->getRPTranslation(tr_ti.getRPId()).z(); // the minus sign fixes a bug in the diamond geometry
              const double f_i = (z_ti - z_j) / (z_i - z_j), f_j = (z_i - z_ti) / (z_i - z_j);
              const double x_inter = f_i * tr_i.getX() + f_j * tr_j.getX();
              const double x_inter_unc_sq = f_i*f_i * tr_i.getXUnc()*tr_i.getXUnc() + f_j*f_j * tr_j.getXUnc()*tr_j.getXUnc();

              const double de_x = tr_ti.getX() - x_inter;
              const double de_x_unc = sqrt(tr_ti.getXUnc()*tr_ti.getXUnc() + x_inter_unc_sq);

              const bool matching = (std::abs(de_x) <= de_x_unc);

              if (verbosity_)
                ssLog << "ti=" << ti << ", i=" << i << ", j=" << j
                  << " | z_ti=" << z_ti << ", z_i=" << z_i << ", z_j=" << z_j
                  << " | x_ti=" << tr_ti.getX() << ", x_inter=" << x_inter << ", de_x=" << de_x << ", de_x_unc=" << de_x_unc
                  << ", matching=" << matching << std::endl;

              if (!matching)
                continue;

              matched_timing_track_indices[pr_idx].push_back(ti);
              matched_timing_track_multiplicity[ti]++;
            }
          }

          // process associated tracks
          for (unsigned int pr_idx = 0; pr_idx < idx_pairs.size(); ++pr_idx)
          {
            const auto &i = idx_pairs[pr_idx].first;
            const auto &j = idx_pairs[pr_idx].second;

            // skip non-unique associations of tracking-RP tracks
            if (idx_pair_multiplicity[i] > 1 || idx_pair_multiplicity[j] > 1)
              continue;

            if (verbosity_)
              ssLog << std::endl << "* reconstruction from tracking-RP tracks: " << i << ", " << j << " and timing-RP tracks: ";

            // process tracking-RP data
            CTPPSLocalTrackLiteRefVector sel_tracks;
            sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, i));
            sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, j));
            reco::ForwardProton proton = algorithm_.reconstructFromMultiRP(sel_tracks, *hLHCInfo, ssLog);

            // process timing-RP data
            double sw=0., swt=0.;
            for (const auto &ti : matched_timing_track_indices[pr_idx])
            {
              // skip non-unique associations of timing-RP tracks
              if (matched_timing_track_multiplicity[ti] > 1)
                continue;

              sel_tracks.push_back(CTPPSLocalTrackLiteRef(hTracks, ti));

              if (verbosity_)
                ssLog << ti << ", ";

              const auto &tr = hTracks->at(ti);
              const double t_unc = tr.getTimeUnc();
              const double w = (t_unc > 0.) ? 1./t_unc/t_unc : 1.;
              sw += w;
              swt += w * tr.getTime();
            }

            float time = 0., time_unc = 0.;
            if (sw > 0.)
            {
              time = swt / sw;
              time_unc = 1. / sqrt(sw);
            }

            if (verbosity_)
              ssLog << std::endl << "    time = " << time << " +- " << time_unc << std::endl;

            // save combined output
            proton.setContributingLocalTracks(sel_tracks);
            proton.setTime(time);
            proton.setTimeError(time_unc);

            pOutMultiRP->emplace_back(proton);
          }
        }

        // save single-RP results (un-indexed)
        for (const auto &p : singleRPResultsIndexed)
          pOutSingleRP->emplace_back(std::move(p.second));
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

