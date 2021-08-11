#include "RecoEgamma/EgammaTools/interface/LowPtConversion.h"

////////////////////////////////////////////////////////////////////////////////
// Matched to any conversion (without selections)
//
bool LowPtConversion::wpOpen() const { return matched_; }

////////////////////////////////////////////////////////////////////////////////
// Nancy's baseline selections for conversions
// Based on: https://github.com/CMSBParking/BParkingNANO/blob/b2664ed/BParkingNano/plugins/ConversionSelector.cc#L253-L300
bool LowPtConversion::wpLoose() const {
  return (wpOpen() && ntracks_ == 2 && valid_ && quality_high_purity_ && chi2prob_ > 0.0005);
}

////////////////////////////////////////////////////////////////////////////////
// Nancy's selection for analysis of conversions
// Based on: slide 20 of https://indico.cern.ch/event/814402/contributions/3401312/
bool LowPtConversion::wpTight() const {
  return (wpLoose() && sum_nhits_before_vtx_ <= 1 && l_xy_ > 0. && mass_from_conv_ > 0. &&  // sanity check
          mass_from_conv_ < 0.05);
}

////////////////////////////////////////////////////////////////////////////////
// adds minimal set of flags to electron userData
void LowPtConversion::addUserVars(pat::Electron& ele) const {
  ele.addUserInt("convOpen", matched_ ? 1 : 0);
  ele.addUserInt("convLoose", wpLoose() ? 1 : 0);
  ele.addUserInt("convTight", wpTight() ? 1 : 0);
  ele.addUserInt("convLead", matched_lead_.isNonnull() ? 1 : 0);
  ele.addUserInt("convTrail", matched_trail_.isNonnull() ? 1 : 0);
  if (ele.hasUserInt("convExtra") == false) {
    ele.addUserInt("convExtra", 0);
  }
}

////////////////////////////////////////////////////////////////////////////////
// adds all variables to electron userData
void LowPtConversion::addExtraUserVars(pat::Electron& ele) const {
  // Flag that indicates if extra variables are added to electron userData
  ele.addUserInt("convExtra", 1, true);  // overwrite

  // quality
  ele.addUserInt("convValid", valid_ ? 1 : 0);
  ele.addUserFloat("convChi2Prob", chi2prob_);
  ele.addUserInt("convQualityHighPurity", quality_high_purity_ ? 1 : 0);
  ele.addUserInt("convQualityHighEff", quality_high_efficiency_ ? 1 : 0);

  // tracks
  ele.addUserInt("convTracksN", ntracks_);
  ele.addUserFloat("convMinTrkPt", min_trk_pt_);
  ele.addUserInt("convLeadIdx", ilead_);
  ele.addUserInt("convTrailIdx", itrail_);

  // displacement
  ele.addUserFloat("convLxy", l_xy_);
  ele.addUserFloat("convVtxRadius", vtx_radius_);

  // invariant mass
  ele.addUserFloat("convMass", mass_from_conv_);
  ele.addUserFloat("convMassFromPin", mass_from_Pin_);
  ele.addUserFloat("convMassBeforeFit", mass_before_fit_);
  ele.addUserFloat("convMassAfterFit", mass_after_fit_);

  // hits before vertex
  ele.addUserInt("convLeadNHitsBeforeVtx", lead_nhits_before_vtx_);
  ele.addUserInt("convTrailNHitsBeforeVtx", trail_nhits_before_vtx_);
  ele.addUserInt("convMaxNHitsBeforeVtx", max_nhits_before_vtx_);
  ele.addUserInt("convSumNHitsBeforeVtx", sum_nhits_before_vtx_);
  ele.addUserInt("convDeltaExpectedNHitsInner", delta_expected_nhits_inner_);

  // opening angle
  ele.addUserFloat("convDeltaCotFromPin", delta_cot_from_Pin_);
}

////////////////////////////////////////////////////////////////////////////////
//
bool LowPtConversion::match(const reco::BeamSpot& beamSpot,
                            const reco::ConversionCollection& conversions,
                            const pat::Electron& ele) {
  // Iterate through conversions and calculate quantities (requirement from Nancy)
  for (const auto& conv : conversions) {
    // Filter
    if (conv.tracks().size() != 2) {
      continue;
    }

    // Quality
    valid_ = conv.conversionVertex().isValid();                                                         // (=true)
    chi2prob_ = ChiSquaredProbability(conv.conversionVertex().chi2(), conv.conversionVertex().ndof());  // (<0.005)
    quality_high_purity_ = conv.quality(reco::Conversion::highPurity);                                  // (=true)
    quality_high_efficiency_ = conv.quality(reco::Conversion::highEfficiency);                          // (none)

    // Tracks
    ntracks_ = conv.tracks().size();  // (=2)
    min_trk_pt_ = -1.;                // (>0.5)
    for (const auto& trk : conv.tracks()) {
      if (trk.isNonnull() && trk.isAvailable() && (min_trk_pt_ < 0. || trk->pt() < min_trk_pt_)) {
        min_trk_pt_ = trk->pt();
      }
    }
    ilead_ = -1;
    itrail_ = -1;
    if (conv.tracks().size() == 2) {
      const edm::RefToBase<reco::Track>& trk1 = conv.tracks().front();
      const edm::RefToBase<reco::Track>& trk2 = conv.tracks().back();
      if (trk1.isNonnull() && trk1.isAvailable() && trk2.isNonnull() && trk2.isAvailable()) {
        if (trk1->pt() > trk2->pt()) {
          ilead_ = 0;
          itrail_ = 1;
        } else {
          ilead_ = 1;
          itrail_ = 0;
        }
      }
    }

    // Transverse displacement (with respect to beamspot) and vertex radius
    math::XYZVectorF p_refitted = conv.refittedPairMomentum();
    float dx = conv.conversionVertex().x() - beamSpot.x0();
    float dy = conv.conversionVertex().y() - beamSpot.y0();
    l_xy_ = (p_refitted.x() * dx + p_refitted.y() * dy) / p_refitted.rho();
    vtx_radius_ = sqrt(conv.conversionVertex().position().perp2());  // (1.5<r<4.)

    // invariant mass from track pair from conversion
    mass_from_conv_ = conv.pairInvariantMass();

    // Invariant mass from Pin before fit to common vertex
    if (conv.tracksPin().size() >= 2 && ilead_ > -1 && itrail_ > -1) {
      math::XYZVectorF lead_Pin = conv.tracksPin().at(ilead_);
      math::XYZVectorF trail_Pin = conv.tracksPin().at(itrail_);
      mass_from_Pin_ = mee(lead_Pin.x(), lead_Pin.y(), lead_Pin.z(), trail_Pin.x(), trail_Pin.y(), trail_Pin.z());
      // Opening angle
      delta_cot_from_Pin_ = 1. / tan(trail_Pin.theta()) - 1. / tan(lead_Pin.theta());
    }

    // Invariant mass before fit to common vertex
    if (conv.tracks().size() >= 2 && ilead_ > -1 && itrail_ > -1) {
      auto lead_before_vtx_fit = conv.tracks().at(ilead_)->momentum();
      auto trail_before_vtx_fit = conv.tracks().at(itrail_)->momentum();
      mass_before_fit_ = mee(lead_before_vtx_fit.x(),
                             lead_before_vtx_fit.y(),
                             lead_before_vtx_fit.z(),
                             trail_before_vtx_fit.x(),
                             trail_before_vtx_fit.y(),
                             trail_before_vtx_fit.z());
    }

    // Invariant mass after the fit to common vertex
    if (conv.conversionVertex().refittedTracks().size() >= 2 && ilead_ > -1 && itrail_ > -1) {
      auto const& lead_after_vtx_fit = conv.conversionVertex().refittedTracks().at(ilead_);
      auto const& trail_after_vtx_fit = conv.conversionVertex().refittedTracks().at(itrail_);
      mass_after_fit_ = mee(lead_after_vtx_fit.px(),
                            lead_after_vtx_fit.py(),
                            lead_after_vtx_fit.pz(),
                            trail_after_vtx_fit.px(),
                            trail_after_vtx_fit.py(),
                            trail_after_vtx_fit.pz());
      // Difference in expeted hits
      delta_expected_nhits_inner_ =
          lead_after_vtx_fit.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) -
          trail_after_vtx_fit.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    }

    // Hits prior to vertex
    if (ilead_ > -1 && itrail_ > -1) {
      auto const& nHits = conv.nHitsBeforeVtx();
      bool enoughTracks = nHits.size() > 1;
      lead_nhits_before_vtx_ = enoughTracks ? nHits.at(ilead_) : 0;
      trail_nhits_before_vtx_ = enoughTracks ? nHits.at(itrail_) : 0;
      max_nhits_before_vtx_ = enoughTracks ? std::max(nHits[0], nHits[1]) : 0;
      sum_nhits_before_vtx_ = enoughTracks ? nHits[0] + nHits[1] : 0;
    }

    // Attempt to match conversion track to electron
    for (uint itrk = 0; itrk < conv.tracks().size(); ++itrk) {
      const edm::RefToBase<reco::Track> trk = conv.tracks()[itrk];
      if (trk.isNull()) {
        continue;
      }
      reco::GsfTrackRef ref = ele.core()->gsfTrack();
      reco::GsfTrackRef gsf = ele.gsfTrack();
      if (gsf.isNull()) {
        continue;
      }
      if (ref.id() == trk.id() && ref.key() == trk.key()) {
        matched_ = true;
        if (static_cast<int>(itrk) == ilead_) {
          matched_lead_ = trk;
        }
        if (static_cast<int>(itrk) == itrail_) {
          matched_trail_ = trk;
        }
      }
    }  // track loop
  }    // conversions loop

  return matched_;
}

////////////////////////////////////////////////////////////////////////////////
//
float LowPtConversion::mee(float px1, float py1, float pz1, float px2, float py2, float pz2) {
  const float m = 0.000511;
  const float px = px1 + px2;
  const float py = py1 + py2;
  const float pz = pz1 + pz2;
  const float p1 = px1 * px1 + py1 * py1 + pz1 * pz1;
  const float p2 = px2 * px2 + py2 * py2 + pz2 * pz2;
  const float e = sqrt(p1 + m * m) + sqrt(p2 + m * m);
  const float mass = (e * e - px * px - py * py - pz * pz);
  return mass > 0. ? sqrt(mass) : -1.;
}
