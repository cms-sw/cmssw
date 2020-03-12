#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;

const double PFCandConnector::pion_mass2 = 0.0194;

const reco::PFCandidate::Flags PFCandConnector::fT_TO_DISP_ = PFCandidate::T_TO_DISP;
const reco::PFCandidate::Flags PFCandConnector::fT_FROM_DISP_ = PFCandidate::T_FROM_DISP;

void PFCandConnector::setParameters(bool bCorrect,
                                    bool bCalibPrimary,
                                    double dptRel_PrimaryTrack,
                                    double dptRel_MergedTrack,
                                    double ptErrorSecondary,
                                    const std::vector<double>& nuclCalibFactors) {
  bCorrect_ = bCorrect;
  bCalibPrimary_ = bCalibPrimary;
  dptRel_PrimaryTrack_ = dptRel_PrimaryTrack;
  dptRel_MergedTrack_ = dptRel_MergedTrack;
  ptErrorSecondary_ = ptErrorSecondary;

  if (nuclCalibFactors.size() == 5) {
    fConst_[0] = nuclCalibFactors[0];
    fConst_[1] = nuclCalibFactors[1];

    fNorm_[0] = nuclCalibFactors[2];
    fNorm_[1] = nuclCalibFactors[3];

    fExp_[0] = nuclCalibFactors[4];
  } else {
    edm::LogWarning("PFCandConnector")
        << "Wrong calibration factors for nuclear interactions. The calibration procedure would not be applyed."
        << std::endl;
    bCalibPrimary_ = false;
  }

  std::string sCorrect = bCorrect_ ? "On" : "Off";
  edm::LogInfo("PFCandConnector") << " ====================== The PFCandConnector is switched " << sCorrect.c_str()
                                  << " ==================== " << std::endl;
  std::string sCalibPrimary = bCalibPrimary_ ? "used for calibration" : "not used for calibration";
  if (bCorrect_)
    edm::LogInfo("PFCandConnector") << "Primary Tracks are " << sCalibPrimary.c_str() << std::endl;
  if (bCorrect_ && bCalibPrimary_)
    edm::LogInfo("PFCandConnector") << "Under the condition that the precision on the Primary track is better than "
                                    << dptRel_PrimaryTrack_ << " % " << std::endl;
  if (bCorrect_ && bCalibPrimary_)
    edm::LogInfo("PFCandConnector") << "      and on merged tracks better than " << dptRel_MergedTrack_ << " % "
                                    << std::endl;
  if (bCorrect_ && bCalibPrimary_)
    edm::LogInfo("PFCandConnector") << "      and secondary tracks in some cases more precise than "
                                    << ptErrorSecondary_ << " GeV" << std::endl;
  if (bCorrect_ && bCalibPrimary_)
    edm::LogInfo("PFCandConnector") << "factor = (" << fConst_[0] << " + " << fConst_[1] << "*cFrac) - (" << fNorm_[0]
                                    << " - " << fNorm_[1] << "cFrac)*exp( " << -1 * fExp_[0] << "*pT )" << std::endl;
  edm::LogInfo("PFCandConnector") << " =========================================================== " << std::endl;
}

reco::PFCandidateCollection PFCandConnector::connect(PFCandidateCollection& pfCand) const {
  /// Collection of primary PFCandidates to be transmitted to the Event
  PFCandidateCollection pfC{};
  /// A mask to define the candidates which shall not be transmitted
  std::vector<bool> bMask;
  bMask.resize(pfCand.size(), false);

  // loop on primary
  if (bCorrect_) {
    LogTrace("PFCandConnector|connect") << "pfCand.size()=" << pfCand.size() << "bCalibPrimary_=" << bCalibPrimary_;

    for (unsigned int ce1 = 0; ce1 < pfCand.size(); ++ce1) {
      if (isPrimaryNucl(pfCand.at(ce1))) {
        LogTrace("PFCandConnector|connect")
            << "" << endl
            << "Nuclear Interaction w Primary Candidate " << ce1 << " " << pfCand.at(ce1) << endl
            << " based on the Track " << pfCand.at(ce1).trackRef().key()
            << " w pT = " << pfCand.at(ce1).trackRef()->pt() << " #pm "
            << pfCand.at(ce1).trackRef()->ptError() / pfCand.at(ce1).trackRef()->pt() * 100 << " %"
            << " ECAL = " << pfCand.at(ce1).ecalEnergy() << " HCAL = " << pfCand.at(ce1).hcalEnergy() << endl;

#ifdef EDM_ML_DEBUG
        (pfCand.at(ce1)).displacedVertexRef(fT_TO_DISP_)->Dump();
#endif

        analyseNuclearWPrim(pfCand, bMask, ce1);

#ifdef EDM_ML_DEBUG
        LogTrace("PFCandConnector|connect")
            << "After Connection the candidate " << ce1 << " is " << pfCand.at(ce1) << endl
            << endl;

        PFCandidate::ElementsInBlocks elementsInBlocks = pfCand.at(ce1).elementsInBlocks();
        for (unsigned blockElem = 0; blockElem < elementsInBlocks.size(); blockElem++) {
          if (blockElem == 0)
            LogTrace("PFCandConnector|connect") << *(elementsInBlocks[blockElem].first) << endl;
          LogTrace("PFCandConnector|connect") << " position " << elementsInBlocks[blockElem].second;
        }
#endif
      }
    }

    for (unsigned int ce1 = 0; ce1 < pfCand.size(); ++ce1) {
      if (!bMask[ce1] && isSecondaryNucl(pfCand.at(ce1))) {
        LogTrace("PFCandConnector|connect")
            << "" << endl
            << "Nuclear Interaction w no Primary Candidate " << ce1 << " " << pfCand.at(ce1) << endl
            << " based on the Track " << pfCand.at(ce1).trackRef().key()
            << " w pT = " << pfCand.at(ce1).trackRef()->pt() << " #pm " << pfCand.at(ce1).trackRef()->ptError() << " %"
            << " ECAL = " << pfCand.at(ce1).ecalEnergy() << " HCAL = " << pfCand.at(ce1).hcalEnergy()
            << " dE(Trk-CALO) = "
            << pfCand.at(ce1).trackRef()->p() - pfCand.at(ce1).ecalEnergy() - pfCand.at(ce1).hcalEnergy()
            << " Nmissing hits = "
            << pfCand.at(ce1).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) << endl;

#ifdef EDM_ML_DEBUG
        (pfCand.at(ce1)).displacedVertexRef(fT_FROM_DISP_)->Dump();
#endif

        analyseNuclearWSec(pfCand, bMask, ce1);

#ifdef EDM_ML_DEBUG
        LogTrace("PFCandConnector|connect") << "After Connection the candidate " << ce1 << " is " << pfCand.at(ce1)
                                            << " and elements connected to it are: " << endl;

        PFCandidate::ElementsInBlocks elementsInBlocks = pfCand.at(ce1).elementsInBlocks();
        for (unsigned blockElem = 0; blockElem < elementsInBlocks.size(); blockElem++) {
          if (blockElem == 0)
            LogTrace("PFCandConnector|connect") << *(elementsInBlocks[blockElem].first) << endl;
          LogTrace("PFCandConnector|connect") << " position " << elementsInBlocks[blockElem].second;
        }
#endif
      }
    }
  }

  for (unsigned int ce1 = 0; ce1 < pfCand.size(); ++ce1)
    if (!bMask[ce1])
      pfC.push_back(pfCand.at(ce1));

  LogTrace("PFCandConnector|connect") << "end of function";

  return pfC;
}

void PFCandConnector::analyseNuclearWPrim(PFCandidateCollection& pfCand,
                                          std::vector<bool>& bMask,
                                          unsigned int ce1) const {
  PFDisplacedVertexRef ref1, ref2, ref1_bis;

  PFCandidate primaryCand = pfCand.at(ce1);

  // ------- look for the little friends -------- //

  const math::XYZTLorentzVectorD& momentumPrim = primaryCand.p4();

  math::XYZTLorentzVectorD momentumSec;

  momentumSec = momentumPrim / momentumPrim.E() * (primaryCand.ecalEnergy() + primaryCand.hcalEnergy());

  map<double, math::XYZTLorentzVectorD> candidatesWithTrackExcess;
  map<double, math::XYZTLorentzVectorD> candidatesWithoutCalo;

  ref1 = primaryCand.displacedVertexRef(fT_TO_DISP_);

  for (unsigned int ce2 = 0; ce2 < pfCand.size(); ++ce2) {
    if (ce2 != ce1 && isSecondaryNucl(pfCand.at(ce2))) {
      ref2 = (pfCand.at(ce2)).displacedVertexRef(fT_FROM_DISP_);

      if (ref1 == ref2) {
        LogTrace("PFCandConnector|analyseNuclearWPrim")
            << "\t here is a Secondary Candidate " << ce2 << " " << pfCand.at(ce2) << endl
            << "\t based on the Track " << pfCand.at(ce2).trackRef().key()
            << " w p = " << pfCand.at(ce2).trackRef()->p() << " w pT = " << pfCand.at(ce2).trackRef()->pt() << " #pm "
            << pfCand.at(ce2).trackRef()->ptError() << " %"
            << " ECAL = " << pfCand.at(ce2).ecalEnergy() << " HCAL = " << pfCand.at(ce2).hcalEnergy()
            << " dE(Trk-CALO) = "
            << pfCand.at(ce2).trackRef()->p() - pfCand.at(ce2).ecalEnergy() - pfCand.at(ce2).hcalEnergy()
            << " Nmissing hits = "
            << pfCand.at(ce2).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) << endl;

        if (isPrimaryNucl(pfCand.at(ce2))) {
          LogTrace("PFCandConnector|analyseNuclearWPrim") << "\t\t but it is also a Primary Candidate " << ce2 << endl;

          ref1_bis = (pfCand.at(ce2)).displacedVertexRef(fT_TO_DISP_);
          if (ref1_bis.isNonnull())
            analyseNuclearWPrim(pfCand, bMask, ce2);
        }

        // Take now the parameters of the secondary track that are relevant and use them to construct the NI candidate

        PFCandidate::ElementsInBlocks elementsInBlocks = pfCand.at(ce2).elementsInBlocks();
        PFCandidate::ElementsInBlocks elementsAlreadyInBlocks = pfCand.at(ce1).elementsInBlocks();
        for (unsigned blockElem = 0; blockElem < elementsInBlocks.size(); blockElem++) {
          bool isAlreadyHere = false;
          for (unsigned alreadyBlock = 0; alreadyBlock < elementsAlreadyInBlocks.size(); alreadyBlock++) {
            if (elementsAlreadyInBlocks[alreadyBlock].second == elementsInBlocks[blockElem].second)
              isAlreadyHere = true;
          }
          if (!isAlreadyHere)
            pfCand.at(ce1).addElementInBlock(elementsInBlocks[blockElem].first, elementsInBlocks[blockElem].second);
        }

        double caloEn = pfCand.at(ce2).ecalEnergy() + pfCand.at(ce2).hcalEnergy();
        double deltaEn = pfCand.at(ce2).p4().E() - caloEn;
        int nMissOuterHits =
            pfCand.at(ce2).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);

        // Check if the difference Track Calo is not too large and if we can trust the track, ie it doesn't miss too much hits.
        if (deltaEn > 1 && nMissOuterHits > 1) {
          math::XYZTLorentzVectorD momentumToAdd = pfCand.at(ce2).p4() * caloEn / pfCand.at(ce2).p4().E();
          momentumSec += momentumToAdd;
          LogTrace("PFCandConnector|analyseNuclearWPrim")
              << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may "
                 "have happened. Let's trust the calo energy"
              << endl
              << "add " << momentumToAdd << endl;

        } else {
          // Check if the difference Track Calo is not too large and if we can trust the track, ie it doesn't miss too much hits.
          if (caloEn > 0.01 && deltaEn > 1 && nMissOuterHits > 0) {
            math::XYZTLorentzVectorD momentumExcess = pfCand.at(ce2).p4() * deltaEn / pfCand.at(ce2).p4().E();
            candidatesWithTrackExcess[pfCand.at(ce2).trackRef()->pt() / pfCand.at(ce2).trackRef()->ptError()] =
                momentumExcess;
          } else if (caloEn < 0.01)
            candidatesWithoutCalo[pfCand.at(ce2).trackRef()->pt() / pfCand.at(ce2).trackRef()->ptError()] =
                pfCand.at(ce2).p4();
          momentumSec += (pfCand.at(ce2)).p4();
        }

        bMask[ce2] = true;
      }
    }
  }

  // We have more primary energy than secondary: reject all secondary tracks which have no calo energy attached.

  if (momentumPrim.E() < momentumSec.E()) {
    LogTrace("PFCandConnector|analyseNuclearWPrim")
        << "Size of 0 calo Energy secondary candidates" << candidatesWithoutCalo.size() << endl;
    for (map<double, math::XYZTLorentzVectorD>::iterator iter = candidatesWithoutCalo.begin();
         iter != candidatesWithoutCalo.end() && momentumPrim.E() < momentumSec.E();
         iter++)
      if (momentumSec.E() > iter->second.E() + 0.1) {
        momentumSec -= iter->second;

        LogTrace("PFCandConnector|analyseNuclearWPrim")
            << "\t Remove a SecondaryCandidate with 0 calo energy " << iter->second << endl;
        LogTrace("PFCandConnector|analyseNuclearWPrim")
            << "momentumPrim.E() = " << momentumPrim.E() << " and momentumSec.E() = " << momentumSec.E() << endl;
      }
  }

  if (momentumPrim.E() < momentumSec.E()) {
    LogTrace("PFCandConnector|analyseNuclearWPrim")
        << "0 Calo Energy rejected but still not sufficient. Size of not enough calo Energy secondary candidates"
        << candidatesWithTrackExcess.size() << endl;
    for (map<double, math::XYZTLorentzVectorD>::iterator iter = candidatesWithTrackExcess.begin();
         iter != candidatesWithTrackExcess.end() && momentumPrim.E() < momentumSec.E();
         iter++)
      if (momentumSec.E() > iter->second.E() + 0.1)
        momentumSec -= iter->second;
  }

  double dpt = pfCand.at(ce1).trackRef()->ptError() / pfCand.at(ce1).trackRef()->pt() * 100;

  if (momentumSec.E() < 0.1) {
    bMask[ce1] = true;
    return;
  }

  // Rescale the secondary candidates to account for the loss of energy, but only if we can trust the primary track:
  // if it has more energy than secondaries and is precise enough and secondary exist and was not eaten or rejected during the PFAlgo step.

  if (((ref1->isTherePrimaryTracks() && dpt < dptRel_PrimaryTrack_) ||
       (ref1->isThereMergedTracks() && dpt < dptRel_MergedTrack_)) &&
      momentumPrim.E() > momentumSec.E() && momentumSec.E() > 0.1) {
    if (bCalibPrimary_) {
      double factor = rescaleFactor(momentumPrim.Pt(), momentumSec.E() / momentumPrim.E());
      LogTrace("PFCandConnector|analyseNuclearWPrim") << "factor = " << factor << endl;
      if (factor * momentumPrim.Pt() < momentumSec.Pt())
        momentumSec = momentumPrim;
      else
        momentumSec += (1 - factor) * momentumPrim;
    }

    double px = momentumPrim.Px() * momentumSec.P() / momentumPrim.P();
    double py = momentumPrim.Py() * momentumSec.P() / momentumPrim.P();
    double pz = momentumPrim.Pz() * momentumSec.P() / momentumPrim.P();
    double E = sqrt(px * px + py * py + pz * pz + pion_mass2);
    math::XYZTLorentzVectorD momentum(px, py, pz, E);
    pfCand.at(ce1).setP4(momentum);

    return;

  } else {
    math::XYZVector primDir = ref1->primaryDirection();

    if (primDir.Mag2() < 0.1) {
      // It might be 0 but this situation should never happend. Throw a warning if it happens.
      edm::LogWarning("PFCandConnector") << "A Nuclear Interaction do not have primary direction" << std::endl;
      pfCand.at(ce1).setP4(momentumSec);
      return;
    } else {
      // rescale the primary direction to the optimal momentum. But take care of the factthat it shall not be completly 0 to avoid a warning if Jet Area.
      double momentumS = momentumSec.P();
      if (momentumS < 1e-4)
        momentumS = 1e-4;
      double px = momentumS * primDir.x();
      double py = momentumS * primDir.y();
      double pz = momentumS * primDir.z();
      double E = sqrt(px * px + py * py + pz * pz + pion_mass2);

      math::XYZTLorentzVectorD momentum(px, py, pz, E);
      pfCand.at(ce1).setP4(momentum);
      return;
    }
  }
}

void PFCandConnector::analyseNuclearWSec(PFCandidateCollection& pfCand,
                                         std::vector<bool>& bMask,
                                         unsigned int ce1) const {
  PFDisplacedVertexRef ref1, ref2;

  // Check if the track excess was not too large and track may miss some outer hits. This may point to a secondary NI.

  double caloEn = pfCand.at(ce1).ecalEnergy() + pfCand.at(ce1).hcalEnergy();
  double deltaEn = pfCand.at(ce1).p4().E() - caloEn;
  int nMissOuterHits = pfCand.at(ce1).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);

  ref1 = pfCand.at(ce1).displacedVertexRef(fT_FROM_DISP_);

  // ------- check if an electron or a muon vas spotted as incoming track -------- //
  // ------- this mean probably that the NI was fake thus we do not correct it -------- /

  if (ref1->isTherePrimaryTracks() || ref1->isThereMergedTracks()) {
    std::vector<reco::Track> refittedTracks = ref1->refittedTracks();
    for (unsigned it = 0; it < refittedTracks.size(); it++) {
      reco::TrackBaseRef primaryBaseRef = ref1->originalTrack(refittedTracks[it]);
      if (ref1->isIncomingTrack(primaryBaseRef))
        LogTrace("PFCandConnector|analyseNuclearWSec")
            << "There is a Primary track ref with pt = " << primaryBaseRef->pt() << endl;

      for (unsigned int ce = 0; ce < pfCand.size(); ++ce) {
        //	  cout << "PFCand Id = " << (pfCand.at(ce)).particleId() << endl;
        if ((pfCand.at(ce)).particleId() == reco::PFCandidate::e ||
            (pfCand.at(ce)).particleId() == reco::PFCandidate::mu) {
          LogTrace("PFCandConnector|analyseNuclearWSec")
              << " It is an electron and it has a ref to a track " << (pfCand.at(ce)).trackRef().isNonnull() << endl;

          if ((pfCand.at(ce)).trackRef().isNonnull()) {
            reco::TrackRef tRef = (pfCand.at(ce)).trackRef();
            reco::TrackBaseRef bRef(tRef);
            LogTrace("PFCandConnector|analyseNuclearWSec")
                << "With Track Ref pt = " << (pfCand.at(ce)).trackRef()->pt() << endl;

            if (bRef == primaryBaseRef) {
              if ((pfCand.at(ce)).particleId() == reco::PFCandidate::e)
                LogTrace("PFCandConnector|analyseNuclearWSec")
                    << "It is a NI from electron. NI Discarded. Just release the candidate." << endl;
              if ((pfCand.at(ce)).particleId() == reco::PFCandidate::mu)
                LogTrace("PFCandConnector|analyseNuclearWSec")
                    << "It is a NI from muon. NI Discarded. Just release the candidate" << endl;

              // release the track but take care of not overcounting bad tracks. In fact those tracks was protected against destruction in
              // PFAlgo. Now we treat them as if they was treated in PFAlgo

              if (caloEn < 0.1 && pfCand.at(ce1).trackRef()->ptError() > ptErrorSecondary_) {
                edm::LogInfo("PFCandConnector|analyseNuclearWSec")
                    << "discarded track since no calo energy and ill measured" << endl;
                bMask[ce1] = true;
              }
              if (caloEn > 0.1 && deltaEn > ptErrorSecondary_ &&
                  pfCand.at(ce1).trackRef()->ptError() > ptErrorSecondary_) {
                edm::LogInfo("PFCandConnector|analyseNuclearWSec")
                    << "rescaled momentum of the track since no calo energy and ill measured" << endl;

                double factor = caloEn / pfCand.at(ce1).p4().E();
                pfCand.at(ce1).rescaleMomentum(factor);
              }

              return;
            }
          }
        }
      }
    }
  }

  PFCandidate secondaryCand = pfCand.at(ce1);

  math::XYZTLorentzVectorD momentumSec = secondaryCand.p4();

  if (deltaEn > ptErrorSecondary_ && nMissOuterHits > 1) {
    math::XYZTLorentzVectorD momentumToAdd = pfCand.at(ce1).p4() * caloEn / pfCand.at(ce1).p4().E();
    momentumSec = momentumToAdd;
    LogTrace("PFCandConnector|analyseNuclearWSec")
        << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may have "
           "happened. Let's trust the calo energy"
        << endl
        << "add " << momentumToAdd << endl;
  }

  // ------- look for the little friends -------- //
  for (unsigned int ce2 = ce1 + 1; ce2 < pfCand.size(); ++ce2) {
    if (isSecondaryNucl(pfCand.at(ce2))) {
      ref2 = (pfCand.at(ce2)).displacedVertexRef(fT_FROM_DISP_);

      if (ref1 == ref2) {
        LogTrace("PFCandConnector|analyseNuclearWSec")
            << "\t here is a Secondary Candidate " << ce2 << " " << pfCand.at(ce2) << endl
            << "\t based on the Track " << pfCand.at(ce2).trackRef().key()
            << " w pT = " << pfCand.at(ce2).trackRef()->pt() << " #pm " << pfCand.at(ce2).trackRef()->ptError() << " %"
            << " ECAL = " << pfCand.at(ce2).ecalEnergy() << " HCAL = " << pfCand.at(ce2).hcalEnergy()
            << " dE(Trk-CALO) = "
            << pfCand.at(ce2).trackRef()->p() - pfCand.at(ce2).ecalEnergy() - pfCand.at(ce2).hcalEnergy()
            << " Nmissing hits = "
            << pfCand.at(ce2).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) << endl;

        // Take now the parameters of the secondary track that are relevant and use them to construct the NI candidate
        PFCandidate::ElementsInBlocks elementsInBlocks = pfCand.at(ce2).elementsInBlocks();
        PFCandidate::ElementsInBlocks elementsAlreadyInBlocks = pfCand.at(ce1).elementsInBlocks();
        for (unsigned blockElem = 0; blockElem < elementsInBlocks.size(); blockElem++) {
          bool isAlreadyHere = false;
          for (unsigned alreadyBlock = 0; alreadyBlock < elementsAlreadyInBlocks.size(); alreadyBlock++) {
            if (elementsAlreadyInBlocks[alreadyBlock].second == elementsInBlocks[blockElem].second)
              isAlreadyHere = true;
          }
          if (!isAlreadyHere)
            pfCand.at(ce1).addElementInBlock(elementsInBlocks[blockElem].first, elementsInBlocks[blockElem].second);
        }

        double caloEn = pfCand.at(ce2).ecalEnergy() + pfCand.at(ce2).hcalEnergy();
        double deltaEn = pfCand.at(ce2).p4().E() - caloEn;
        int nMissOuterHits =
            pfCand.at(ce2).trackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
        if (deltaEn > ptErrorSecondary_ && nMissOuterHits > 1) {
          math::XYZTLorentzVectorD momentumToAdd = pfCand.at(ce2).p4() * caloEn / pfCand.at(ce2).p4().E();
          momentumSec += momentumToAdd;
          LogTrace("PFCandConnector|analyseNuclearWSec")
              << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may "
                 "have happened. Let's trust the calo energy"
              << endl
              << "add " << momentumToAdd << endl;
        } else {
          momentumSec += (pfCand.at(ce2)).p4();
        }

        bMask[ce2] = true;
      }
    }
  }

  math::XYZVector primDir = ref1->primaryDirection();

  if (primDir.Mag2() < 0.1) {
    // It might be 0 but this situation should never happend. Throw a warning if it happens.
    pfCand.at(ce1).setP4(momentumSec);
    edm::LogWarning("PFCandConnector") << "A Nuclear Interaction do not have primary direction" << std::endl;
    return;
  } else {
    // rescale the primary direction to the optimal momentum. But take care of the factthat it shall not be completly 0 to avoid a warning if Jet Area.
    double momentumS = momentumSec.P();
    if (momentumS < 1e-4)
      momentumS = 1e-4;
    double px = momentumS * primDir.x();
    double py = momentumS * primDir.y();
    double pz = momentumS * primDir.z();
    double E = sqrt(px * px + py * py + pz * pz + pion_mass2);

    math::XYZTLorentzVectorD momentum(px, py, pz, E);

    pfCand.at(ce1).setP4(momentum);
    return;
  }
}

bool PFCandConnector::isSecondaryNucl(const PFCandidate& pf) const {
  PFDisplacedVertexRef ref1;
  // nuclear
  if (pf.flag(fT_FROM_DISP_)) {
    ref1 = pf.displacedVertexRef(fT_FROM_DISP_);
    //    ref1->Dump();
    if (!ref1.isNonnull())
      return false;
    else if (ref1->isNucl() || ref1->isNucl_Loose() || ref1->isNucl_Kink())
      return true;
  }

  return false;
}

bool PFCandConnector::isPrimaryNucl(const PFCandidate& pf) const {
  PFDisplacedVertexRef ref1;

  // nuclear
  if (pf.flag(fT_TO_DISP_)) {
    ref1 = pf.displacedVertexRef(fT_TO_DISP_);
    //ref1->Dump();

    if (!ref1.isNonnull())
      return false;
    else if (ref1->isNucl() || ref1->isNucl_Loose() || ref1->isNucl_Kink())
      return true;
  }

  return false;
}

double PFCandConnector::rescaleFactor(const double pt, const double cFrac) const {
  /*
    LOG NORMAL FIT
 FCN=35.8181 FROM MIGRAD    STATUS=CONVERGED     257 CALLS         258 TOTAL
 EDM=8.85763e-09    STRATEGY= 1      ERROR MATRIX ACCURATE
  EXT PARAMETER                                   STEP         FIRST
  NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
   1  p0           7.99434e-01   2.77264e-02   6.59108e-06   9.80247e-03
   2  p1           1.51303e-01   2.89981e-02   1.16775e-05   6.99035e-03
   3  p2          -5.03829e-01   2.87929e-02   1.90070e-05   1.37015e-03
   4  p3           4.54043e-01   5.00908e-02   3.17625e-05   3.86622e-03
   5  p4          -4.61736e-02   8.07940e-03   3.25775e-06  -1.37247e-02
  */

  /*
    FCN=34.4051 FROM MIGRAD    STATUS=CONVERGED     221 CALLS         222 TOTAL
    EDM=1.02201e-09    STRATEGY= 1  ERROR MATRIX UNCERTAINTY   2.3 per cent

   fConst
   1  p0           7.99518e-01   2.23519e-02   1.41523e-06   4.05975e-04
   2  p1           1.44619e-01   2.39398e-02  -7.68117e-07  -2.55775e-03

   fNorm
   3  p2          -5.16571e-01   3.12362e-02   5.74932e-07   3.42292e-03
   4  p3           4.69055e-01   5.09665e-02   1.94353e-07   1.69031e-03

   fExp
   5  p4          -5.18044e-02   8.13458e-03   4.29815e-07  -1.07624e-02
  */

  double fConst, fNorm, fExp;

  fConst = fConst_[0] + fConst_[1] * cFrac;
  fNorm = fNorm_[0] - fNorm_[1] * cFrac;
  fExp = fExp_[0];

  double factor = fConst - fNorm * exp(-fExp * pt);

  return factor;
}

void PFCandConnector::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  iDesc.add<bool>("bCorrect", true);
  iDesc.add<bool>("bCalibPrimary", true);
  iDesc.add<double>("dptRel_PrimaryTrack", 10.0);
  iDesc.add<double>("dptRel_MergedTrack", 5.0);
  iDesc.add<double>("ptErrorSecondary", 1.0);
  iDesc.add<std::vector<double>>("nuclCalibFactors", {0.8, 0.15, 0.5, 0.5, 0.05});
}
