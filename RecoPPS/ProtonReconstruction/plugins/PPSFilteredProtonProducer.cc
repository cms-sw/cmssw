/****************************************************************************
 * Authors:
 *   Jan Kaspar 
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include <ostream>

//----------------------------------------------------------------------------------------------------

/// Module to apply Proton POG quality criteria.
class PPSFilteredProtonProducer : public edm::stream::EDProducer<> {
public:
  explicit PPSFilteredProtonProducer(const edm::ParameterSet &);
  ~PPSFilteredProtonProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

  bool verbosity_;

  double tracks_all_local_angle_x_max_, tracks_all_local_angle_y_max_;

  std::vector<unsigned int> tracks_pixel_forbidden_RecoInfo_values_;
  unsigned int tracks_pixel_number_of_hits_min_;
  double tracks_pixel_normalised_chi_sq_max_;

  bool protons_single_rp_include_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> protons_single_rp_input_token_;
  std::string protons_single_rp_output_label_;

  bool protons_multi_rp_include_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> protons_multi_rp_input_token_;
  std::string protons_multi_rp_output_label_;

  bool protons_multi_rp_check_valid_fit_;
  double protons_multi_rp_chi_sq_max_;
  double protons_multi_rp_normalised_chi_sq_max_;

  /// counters
  unsigned int n_protons_single_rp_all, n_protons_single_rp_kept;
  unsigned int n_protons_multi_rp_all, n_protons_multi_rp_kept;

  /// check one track
  bool IsTrackOK(const CTPPSLocalTrackLite &tr, unsigned int idx, std::ostringstream &log);
};

//----------------------------------------------------------------------------------------------------

PPSFilteredProtonProducer::PPSFilteredProtonProducer(const edm::ParameterSet &iConfig)
  : verbosity_(iConfig.getUntrackedParameter<bool>("verbosity", 0)),
    n_protons_single_rp_all(0),
    n_protons_single_rp_kept(0),
    n_protons_multi_rp_all(0),
    n_protons_multi_rp_kept(0) {
  const auto &tracks_all = iConfig.getParameterSet("tracks_all");
  tracks_all_local_angle_x_max_ = tracks_all.getParameter<double>("local_angle_x_max");
  tracks_all_local_angle_y_max_ = tracks_all.getParameter<double>("local_angle_y_max");

  const auto &tracks_pixel = iConfig.getParameterSet("tracks_pixel");
  tracks_pixel_forbidden_RecoInfo_values_ =
    tracks_pixel.getParameter<std::vector<unsigned int>>("forbidden_RecoInfo_values");
  tracks_pixel_number_of_hits_min_ = tracks_pixel.getParameter<unsigned int>("number_of_hits_min");
  tracks_pixel_normalised_chi_sq_max_ = tracks_pixel.getParameter<double>("normalised_chi_sq_max");

  const auto &protons_single_rp = iConfig.getParameterSet("protons_single_rp");
  protons_single_rp_include_ = protons_single_rp.getParameter<bool>("include");
  protons_single_rp_input_token_ =
    consumes<reco::ForwardProtonCollection>(protons_single_rp.getParameter<edm::InputTag>("input_tag"));
  protons_single_rp_output_label_ = protons_single_rp.getParameter<std::string>("output_label");

  const auto &protons_multi_rp = iConfig.getParameterSet("protons_multi_rp");
  protons_multi_rp_include_ = protons_multi_rp.getParameter<bool>("include");
  protons_multi_rp_input_token_ =
    consumes<reco::ForwardProtonCollection>(protons_multi_rp.getParameter<edm::InputTag>("input_tag"));
  protons_multi_rp_output_label_ = protons_multi_rp.getParameter<std::string>("output_label");
  protons_multi_rp_check_valid_fit_ = protons_multi_rp.getParameter<bool>("check_valid_fit");
  protons_multi_rp_chi_sq_max_ = protons_multi_rp.getParameter<double>("chi_sq_max");
  protons_multi_rp_normalised_chi_sq_max_ = protons_multi_rp.getParameter<double>("normalised_chi_sq_max");

  if (protons_single_rp_include_)
    produces<reco::ForwardProtonCollection>(protons_single_rp_output_label_);

  if (protons_multi_rp_include_)
    produces<reco::ForwardProtonCollection>(protons_multi_rp_output_label_);
}

//----------------------------------------------------------------------------------------------------

void PPSFilteredProtonProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<bool>("verbosity", 0)->setComment("verbosity level");

  edm::ParameterSetDescription tracks_all;
  tracks_all.add<double>("local_angle_x_max", 0.020)
    ->setComment("maximum absolute value of local horizontal angle, in rad");
  tracks_all.add<double>("local_angle_y_max", 0.020)
    ->setComment("maximum absolute value of local horizontal angle, in rad");
  desc.add<edm::ParameterSetDescription>("tracks_all", tracks_all)->setComment("settings for all tracks");

  edm::ParameterSetDescription tracks_pixel;
  const std::vector<unsigned int> def_for_RecoInfo_vals = {
    (unsigned int)CTPPSpixelLocalTrackReconstructionInfo::allShiftedPlanes,
    (unsigned int)CTPPSpixelLocalTrackReconstructionInfo::mixedPlanes};
  tracks_pixel.add<std::vector<unsigned int>>("forbidden_RecoInfo_values", def_for_RecoInfo_vals)
    ->setComment("list of forbidden RecoInfo values");
  tracks_pixel.add<unsigned int>("number_of_hits_min", 0)->setComment("minimum required number of hits");
  tracks_pixel.add<double>("normalised_chi_sq_max", 1E100)->setComment("maximum tolerated chi square / ndof");
  desc.add<edm::ParameterSetDescription>("tracks_pixel", tracks_pixel)
    ->setComment("specific settings for pixel-RP tracks");

  edm::ParameterSetDescription protons_single_rp;
  protons_single_rp.add<bool>("include", true)->setComment("flag whether single-RP protons should be processed");
  protons_single_rp.add<edm::InputTag>("input_tag", edm::InputTag("ctppsProtons", "singleRP"))->setComment("input tag");
  protons_single_rp.add<std::string>("output_label", "singleRP")->setComment("output label");
  desc.add<edm::ParameterSetDescription>("protons_single_rp", protons_single_rp)
      ->setComment("settings for single-RP protons");

  edm::ParameterSetDescription protons_multi_rp;
  protons_multi_rp.add<bool>("include", true)->setComment("flag whether multi-RP protons should be processed");
  protons_multi_rp.add<edm::InputTag>("input_tag", edm::InputTag("ctppsProtons", "multiRP"))->setComment("input tag");
  protons_multi_rp.add<std::string>("output_label", "multiRP")->setComment("output label");
  protons_multi_rp.add<bool>("check_valid_fit", true)->setComment("flag whether validFit should be checked");
  protons_multi_rp.add<double>("chi_sq_max", 1E-4)->setComment("maximum tolerated value of chi square");
  protons_multi_rp.add<double>("normalised_chi_sq_max", 1E100)
      ->setComment("maximum tolerated value of chi square / ndof, applied only if ndof > 0");
  desc.add<edm::ParameterSetDescription>("protons_multi_rp", protons_multi_rp)
      ->setComment("settings for multi-RP protons");

  descriptions.add("ppsFilteredProtonProducer", desc);
}

//----------------------------------------------------------------------------------------------------

bool PPSFilteredProtonProducer::IsTrackOK(const CTPPSLocalTrackLite &tr, unsigned int idx, std::ostringstream &log) {
  bool ok = true;

  // checks for all tracks
  ok &= (std::abs(tr.tx()) < tracks_all_local_angle_x_max_);
  ok &= (std::abs(tr.ty()) < tracks_all_local_angle_y_max_);

  // pixel checks
  const CTPPSDetId rpId(tr.rpId());
  if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel) {
    ok &= (find(tracks_pixel_forbidden_RecoInfo_values_.begin(),
                tracks_pixel_forbidden_RecoInfo_values_.end(),
                (unsigned int)tr.pixelTrackRecoInfo()) == tracks_pixel_forbidden_RecoInfo_values_.end());
    ok &= (tr.numberOfPointsUsedForFit() >= tracks_pixel_number_of_hits_min_);
    ok &= (tr.chiSquaredOverNDF() <= tracks_pixel_normalised_chi_sq_max_);
  }

  if (!ok && verbosity_)
    log << "track idx=" << idx << " does not fulfil criteria." << std::endl;

  return ok;
}

//----------------------------------------------------------------------------------------------------

void PPSFilteredProtonProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::ostringstream ssLog;

  // process single-RP protons
  if (protons_single_rp_include_) {
    edm::Handle<reco::ForwardProtonCollection> hInputProtons;
    iEvent.getByToken(protons_single_rp_input_token_, hInputProtons);

    std::unique_ptr<reco::ForwardProtonCollection> pOutputProtons(new reco::ForwardProtonCollection);

    for (unsigned int pr_idx = 0; pr_idx < hInputProtons->size(); ++pr_idx) {
      const auto &proton = hInputProtons->at(pr_idx);

      bool keep = true;

      // no specific checks for single-RP protons

      // test contributing tracks
      for (const auto &tr_ref : proton.contributingLocalTracks()) {
        if (!keep)
          break;

        keep &= IsTrackOK(*tr_ref, tr_ref.key(), ssLog);
      }

      n_protons_single_rp_all++;

      if (keep) {
        n_protons_single_rp_kept++;
        pOutputProtons->push_back(proton);
      } else {
        if (verbosity_)
          ssLog << "single-RP proton idx=" << pr_idx << " excluded." << std::endl;
      }
    }

    iEvent.put(std::move(pOutputProtons), protons_single_rp_output_label_);
  }

  // process multi-RP protons
  if (protons_multi_rp_include_) {
    edm::Handle<reco::ForwardProtonCollection> hInputProtons;
    iEvent.getByToken(protons_multi_rp_input_token_, hInputProtons);

    std::unique_ptr<reco::ForwardProtonCollection> pOutputProtons(new reco::ForwardProtonCollection);

    for (unsigned int pr_idx = 0; pr_idx < hInputProtons->size(); ++pr_idx) {
      const auto &proton = hInputProtons->at(pr_idx);

      bool keep = true;

      // multi-RP proton checks
      if (protons_multi_rp_check_valid_fit_)
        keep &= proton.validFit();

      keep &= (proton.chi2() <= protons_multi_rp_chi_sq_max_);

      if (proton.ndof() > 0)
        keep &= (proton.normalizedChi2() <= protons_multi_rp_normalised_chi_sq_max_);

      // test contributing tracks
      for (const auto &tr_ref : proton.contributingLocalTracks()) {
        if (!keep)
          break;

        keep &= IsTrackOK(*tr_ref, tr_ref.key(), ssLog);
      }

      n_protons_multi_rp_all++;

      if (keep) {
        n_protons_multi_rp_kept++;
        pOutputProtons->push_back(proton);
      } else {
        if (verbosity_)
          ssLog << "multi-RP proton idx=" << pr_idx << " excluded." << std::endl;
      }
    }

    iEvent.put(std::move(pOutputProtons), protons_multi_rp_output_label_);
  }

  if (verbosity_ && !ssLog.str().empty())
    edm::LogInfo("PPS") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

void PPSFilteredProtonProducer::endStream() {
  edm::LogInfo("PPS")
    << "single-RP protons: total=" << n_protons_single_rp_all << ", kept=" << n_protons_single_rp_kept
    << " --> keep rate="
    << ((n_protons_single_rp_all > 0) ? double(n_protons_single_rp_kept) / n_protons_single_rp_all * 100. : 0.)
    << "%\n"
    << "multi-RP protons: total=" << n_protons_multi_rp_all << ", kept=" << n_protons_multi_rp_kept
    << " --> keep rate="
    << ((n_protons_multi_rp_all > 0) ? double(n_protons_multi_rp_kept) / n_protons_multi_rp_all * 100. : 0.) << "%";
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSFilteredProtonProducer);
