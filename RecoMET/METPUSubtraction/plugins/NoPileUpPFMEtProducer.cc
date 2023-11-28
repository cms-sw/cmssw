#include "RecoMET/METPUSubtraction/plugins/NoPileUpPFMEtProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

//#include "DataFormats/METReco/interface/PFMEtSignCovMatrix.h" //never used so far
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "DataFormats/METReco/interface/SigInputObj.h"

#include <cmath>

const double defaultPFMEtResolutionX = 10.;
const double defaultPFMEtResolutionY = 10.;

const double epsilon = 1.e-9;

NoPileUpPFMEtProducer::NoPileUpPFMEtProducer(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
  srcMEt_ = consumes<reco::PFMETCollection>(cfg.getParameter<edm::InputTag>("srcMEt"));
  srcMEtCov_ = edm::InputTag();  //MM, disabled for the moment until we really need it
  //( cfg.exists("srcMEtCov") ) ?
  //  consumes<edm::Handle<> >(cfg.getParameter<edm::InputTag>("srcMEtCov")) : edm::InputTag();
  srcJetInfo_ = consumes<reco::PUSubMETCandInfoCollection>(cfg.getParameter<edm::InputTag>("srcPUSubMETDataJet"));
  srcJetInfoLeptonMatch_ =
      consumes<reco::PUSubMETCandInfoCollection>(cfg.getParameter<edm::InputTag>("srcPUSubMETDataJetLeptonMatch"));
  srcPFCandInfo_ =
      consumes<reco::PUSubMETCandInfoCollection>(cfg.getParameter<edm::InputTag>("srcPUSubMETDataPFCands"));
  srcPFCandInfoLeptonMatch_ =
      consumes<reco::PUSubMETCandInfoCollection>(cfg.getParameter<edm::InputTag>("srcPUSubMETDataPFCandsLeptonMatch"));
  vInputTag srcLeptonsTags = cfg.getParameter<vInputTag>("srcLeptons");
  for (vInputTag::const_iterator it = srcLeptonsTags.begin(); it != srcLeptonsTags.end(); it++) {
    srcLeptons_.push_back(consumes<edm::View<reco::Candidate> >(*it));
  }

  srcType0Correction_ = consumes<CorrMETData>(cfg.getParameter<edm::InputTag>("srcType0Correction"));

  sfNoPUjets_ = cfg.getParameter<double>("sfNoPUjets");
  sfNoPUjetOffsetEnCorr_ = cfg.getParameter<double>("sfNoPUjetOffsetEnCorr");
  sfPUjets_ = cfg.getParameter<double>("sfPUjets");
  sfNoPUunclChargedCands_ = cfg.getParameter<double>("sfNoPUunclChargedCands");
  sfPUunclChargedCands_ = cfg.getParameter<double>("sfPUunclChargedCands");
  sfUnclNeutralCands_ = cfg.getParameter<double>("sfUnclNeutralCands");
  sfType0Correction_ = cfg.getParameter<double>("sfType0Correction");
  sfLeptonIsoCones_ = cfg.getParameter<double>("sfLeptonIsoCones");

  pfMEtSignInterface_ = new PFMEtSignInterfaceBase(cfg.getParameter<edm::ParameterSet>("resolution"));
  sfMEtCovMin_ = cfg.getParameter<double>("sfMEtCovMin");
  sfMEtCovMax_ = cfg.getParameter<double>("sfMEtCovMax");

  saveInputs_ = (cfg.exists("saveInputs")) ? cfg.getParameter<bool>("saveInputs") : false;

  verbosity_ = (cfg.exists("verbosity")) ? cfg.getParameter<int>("verbosity") : 0;

  produces<reco::PFMETCollection>();

  sfLeptonsName_ = "sumLeptons";
  sfNoPUjetsName_ = "sumNoPUjets";
  sfNoPUjetOffsetEnCorrName_ = "sumNoPUjetOffsetEnCorr";
  sfPUjetsName_ = "sumPUjets";
  sfNoPUunclChargedCandsName_ = "sumNoPUunclChargedCands";
  sfPUunclChargedCandsName_ = "sumPUunclChargedCands";
  sfUnclNeutralCandsName_ = "sumUnclNeutralCands";
  sfType0CorrectionName_ = "type0Correction";
  sfLeptonIsoConesName_ = "sumLeptonIsoCones";

  if (saveInputs_) {
    produces<CommonMETData>(sfLeptonsName_);
    produces<CommonMETData>(sfNoPUjetsName_);
    produces<CommonMETData>(sfNoPUjetOffsetEnCorrName_);
    produces<CommonMETData>(sfPUjetsName_);
    produces<CommonMETData>(sfNoPUunclChargedCandsName_);
    produces<CommonMETData>(sfPUunclChargedCandsName_);
    produces<CommonMETData>(sfUnclNeutralCandsName_);
    produces<CommonMETData>(sfType0CorrectionName_);
    produces<CommonMETData>(sfLeptonIsoConesName_);
  }
  produces<double>("sfNoPU");
}

NoPileUpPFMEtProducer::~NoPileUpPFMEtProducer() { delete pfMEtSignInterface_; }

void initializeCommonMETData(CommonMETData& metData) {
  metData.met = 0.;
  metData.mex = 0.;
  metData.mey = 0.;
  metData.mez = 0.;
  metData.sumet = 0.;
  metData.phi = 0.;
}

void addToCommonMETData(CommonMETData& metData, const reco::Candidate::LorentzVector& p4) {
  metData.mex += p4.px();
  metData.mey += p4.py();
  metData.mez += p4.pz();
  metData.sumet += p4.pt();
}

void finalizeCommonMETData(CommonMETData& metData) {
  metData.met = sqrt(metData.mex * metData.mex + metData.mey * metData.mey);
  metData.phi = atan2(metData.mey, metData.mex);
}

int findBestMatchingLepton(const std::vector<reco::Candidate::LorentzVector>& leptons,
                           const reco::Candidate::LorentzVector& p4_ref) {
  int leptonIdx_dR2min = -1;
  double dR2min = 1.e+3;
  int leptonIdx = 0;
  for (std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin(); lepton != leptons.end();
       ++lepton) {
    double dR2 = deltaR2(*lepton, p4_ref);
    if (leptonIdx_dR2min == -1 || dR2 < dR2min) {
      leptonIdx_dR2min = leptonIdx;
      dR2min = dR2;
    }
    ++leptonIdx;
  }
  assert(leptonIdx_dR2min >= 0 && leptonIdx_dR2min < (int)leptons.size());
  return leptonIdx_dR2min;
}

void scaleAndAddPFMEtSignObjects(std::vector<metsig::SigInputObj>& metSignObjects_scaled,
                                 const std::vector<metsig::SigInputObj>& metSignObjects,
                                 double sf,
                                 double sfMin,
                                 double sfMax) {
  double sf_value = sf;
  if (sf_value > sfMax)
    sf_value = sfMax;
  if (sf_value < sfMin)
    sf_value = sfMin;
  for (std::vector<metsig::SigInputObj>::const_iterator metSignObject = metSignObjects.begin();
       metSignObject != metSignObjects.end();
       ++metSignObject) {
    metsig::SigInputObj metSignObject_scaled;
    metSignObject_scaled.set(metSignObject->get_type(),
                             sf_value * metSignObject->get_energy(),
                             metSignObject->get_phi(),
                             sf_value * metSignObject->get_sigma_e(),
                             metSignObject->get_sigma_tan());
    metSignObjects_scaled.push_back(metSignObject_scaled);
  }
}

reco::METCovMatrix computePFMEtSignificance(const std::vector<metsig::SigInputObj>& metSignObjects) {
  reco::METCovMatrix pfMEtCov;
  if (metSignObjects.size() >= 2) {
    metsig::significanceAlgo pfMEtSignAlgorithm;
    pfMEtSignAlgorithm.addObjects(metSignObjects);
    pfMEtCov = pfMEtSignAlgorithm.getSignifMatrix();
  }

  double det = 0;
  pfMEtCov.Det(det);
  if (std::abs(det) < epsilon) {
    edm::LogWarning("computePFMEtSignificance") << "Inversion of PFMEt covariance matrix failed, det = " << det
                                                << " --> replacing covariance matrix by resolution defaults !!";
    pfMEtCov(0, 0) = defaultPFMEtResolutionX * defaultPFMEtResolutionX;
    pfMEtCov(0, 1) = 0.;
    pfMEtCov(1, 0) = 0.;
    pfMEtCov(1, 1) = defaultPFMEtResolutionY * defaultPFMEtResolutionY;
  }

  return pfMEtCov;
}

void printP4(const std::string& label_part1, int idx, const std::string& label_part2, const reco::Candidate& candidate) {
  std::cout << label_part1 << " #" << idx << label_part2 << ": Pt = " << candidate.pt() << ", eta = " << candidate.eta()
            << ", phi = " << candidate.phi() << " (charge = " << candidate.charge() << ")" << std::endl;
}

void printCommonMETData(const std::string& label, const CommonMETData& metData) {
  std::cout << label << ": Px = " << metData.mex << ", Py = " << metData.mey << ", sumEt = " << metData.sumet
            << std::endl;
}

void printMVAMEtJetInfo(const std::string& label, int idx, const reco::PUSubMETCandInfo& jet) {
  std::cout << label << " #" << idx << " (";
  if (jet.type() == reco::PUSubMETCandInfo::kHS)
    std::cout << "no-PU";
  else if (jet.type() == reco::PUSubMETCandInfo::kPU)
    std::cout << "PU";
  std::cout << "): Pt = " << jet.p4().pt() << ", eta = " << jet.p4().eta() << ", phi = " << jet.p4().phi();
  std::cout << " id. flags: anti-noise = " << jet.passesLooseJetId() << std::endl;
  std::cout << std::endl;
}

void printMVAMEtPFCandInfo(const std::string& label, int idx, const reco::PUSubMETCandInfo& pfCand) {
  std::cout << label << " #" << idx << " (";
  if (pfCand.type() == reco::PUSubMETCandInfo::kChHS)
    std::cout << "no-PU charged";
  else if (pfCand.type() == reco::PUSubMETCandInfo::kChPU)
    std::cout << "PU charged";
  else if (pfCand.type() == reco::PUSubMETCandInfo::kNeutral)
    std::cout << "neutral";
  std::cout << "): Pt = " << pfCand.p4().pt() << ", eta = " << pfCand.p4().eta() << ", phi = " << pfCand.p4().phi();
  std::string isWithinJet_string;
  if (pfCand.isWithinJet())
    isWithinJet_string = "true";
  else
    isWithinJet_string = "false";
  std::cout << " (isWithinJet = " << isWithinJet_string << ")";
  if (pfCand.isWithinJet())
    std::cout << " Jet id. flags: anti-noise = " << pfCand.passesLooseJetId() << std::endl;
  std::cout << std::endl;
}

void NoPileUpPFMEtProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  LogDebug("produce") << " moduleLabel = " << moduleLabel_ << std::endl;

  // get original MET
  edm::Handle<reco::PFMETCollection> pfMETs;
  evt.getByToken(srcMEt_, pfMETs);
  if (!(pfMETs->size() == 1))
    throw cms::Exception("NoPileUpPFMEtProducer::produce") << "Failed to find unique MET object !!\n";
  const reco::PFMET& pfMEt_original = pfMETs->front();

  // get MET covariance matrix
  reco::METCovMatrix pfMEtCov;
  if (!srcMEtCov_.label().empty()) {
    //MM manual bypass to pfMET as this case has neer been presented
    // edm::Handle<PFMEtSignCovMatrix> pfMEtCovHandle;
    // evt.getByToken(srcMEtCov_, pfMEtCovHandle);
    // pfMEtCov = (*pfMEtCovHandle);
    pfMEtCov = pfMEt_original.getSignificanceMatrix();
  } else {
    pfMEtCov = pfMEt_original.getSignificanceMatrix();
  }

  // get lepton momenta
  std::vector<reco::Candidate::LorentzVector> leptons;
  std::vector<metsig::SigInputObj> metSignObjectsLeptons;
  reco::Candidate::LorentzVector sumLeptonP4s;
  for (std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator srcLeptons_i = srcLeptons_.begin();
       srcLeptons_i != srcLeptons_.end();
       ++srcLeptons_i) {
    edm::Handle<reco::CandidateView> leptons_i;
    evt.getByToken(*srcLeptons_i, leptons_i);
    for (reco::CandidateView::const_iterator lepton = leptons_i->begin(); lepton != leptons_i->end(); ++lepton) {
      leptons.push_back(lepton->p4());
      metSignObjectsLeptons.push_back(pfMEtSignInterface_->compResolution(&(*lepton)));
      sumLeptonP4s += lepton->p4();
    }
  }
  LogDebug("produce") << " sum(leptons): Pt = " << sumLeptonP4s.pt() << ", eta = " << sumLeptonP4s.eta()
                      << ", phi = " << sumLeptonP4s.phi() << ","
                      << " mass = " << sumLeptonP4s.mass() << std::endl;

  // get jet and PFCandidate information
  edm::Handle<reco::PUSubMETCandInfoCollection> jets;
  evt.getByToken(srcJetInfo_, jets);
  edm::Handle<reco::PUSubMETCandInfoCollection> jetsLeptonMatch;
  evt.getByToken(srcJetInfoLeptonMatch_, jetsLeptonMatch);
  edm::Handle<reco::PUSubMETCandInfoCollection> pfCandidates;
  evt.getByToken(srcPFCandInfo_, pfCandidates);
  edm::Handle<reco::PUSubMETCandInfoCollection> pfCandidatesLeptonMatch;
  evt.getByToken(srcPFCandInfoLeptonMatch_, pfCandidatesLeptonMatch);

  reco::PUSubMETCandInfoCollection jets_leptons = utils_.cleanJets(*jetsLeptonMatch, leptons, 0.5, true);
  reco::PUSubMETCandInfoCollection pfCandidates_leptons =
      utils_.cleanPFCandidates(*pfCandidatesLeptonMatch, leptons, 0.3, true);
  std::vector<CommonMETData> sumJetsPlusPFCandidates_leptons(leptons.size());
  for (std::vector<CommonMETData>::iterator sumJetsPlusPFCandidates = sumJetsPlusPFCandidates_leptons.begin();
       sumJetsPlusPFCandidates != sumJetsPlusPFCandidates_leptons.end();
       ++sumJetsPlusPFCandidates) {
    initializeCommonMETData(*sumJetsPlusPFCandidates);
  }
  for (reco::PUSubMETCandInfoCollection::const_iterator jet = jets_leptons.begin(); jet != jets_leptons.end(); ++jet) {
    int leptonIdx_dRmin = findBestMatchingLepton(leptons, jet->p4());
    assert(leptonIdx_dRmin >= 0 && leptonIdx_dRmin < (int)sumJetsPlusPFCandidates_leptons.size());

    LogDebug("produce") << "jet-to-lepton match:"
                        << " jetPt = " << jet->p4().pt() << ", jetEta = " << jet->p4().eta()
                        << ", jetPhi = " << jet->p4().phi() << " leptonPt = " << leptons[leptonIdx_dRmin].pt()
                        << ", leptonEta = " << leptons[leptonIdx_dRmin].eta()
                        << ", leptonPhi = " << leptons[leptonIdx_dRmin].phi() << std::endl;

    sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].mex += jet->p4().px();
    sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].mey += jet->p4().py();
    sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].sumet += jet->p4().pt();
  }
  for (reco::PUSubMETCandInfoCollection::const_iterator pfCandidate = pfCandidates_leptons.begin();
       pfCandidate != pfCandidates_leptons.end();
       ++pfCandidate) {
    bool isWithinJet_lepton = false;
    if (pfCandidate->isWithinJet()) {
      for (reco::PUSubMETCandInfoCollection::const_iterator jet = jets_leptons.begin(); jet != jets_leptons.end();
           ++jet) {
        double dR2 = deltaR2(pfCandidate->p4(), jet->p4());
        if (dR2 < 0.5 * 0.5)
          isWithinJet_lepton = true;
      }
    }
    if (!isWithinJet_lepton) {
      int leptonIdx_dRmin = findBestMatchingLepton(leptons, pfCandidate->p4());
      assert(leptonIdx_dRmin >= 0 && leptonIdx_dRmin < (int)sumJetsPlusPFCandidates_leptons.size());
      LogDebug("produce") << "pfCandidate-to-lepton match:"
                          << " pfCandidatePt = " << pfCandidate->p4().pt()
                          << ", pfCandidateEta = " << pfCandidate->p4().eta()
                          << ", pfCandidatePhi = " << pfCandidate->p4().phi()
                          << " leptonPt = " << leptons[leptonIdx_dRmin].pt()
                          << ", leptonEta = " << leptons[leptonIdx_dRmin].eta()
                          << ", leptonPhi = " << leptons[leptonIdx_dRmin].phi() << std::endl;

      sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].mex += pfCandidate->p4().px();
      sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].mey += pfCandidate->p4().py();
      sumJetsPlusPFCandidates_leptons[leptonIdx_dRmin].sumet += pfCandidate->p4().pt();
    } else {
      LogDebug("produce") << " pfCandidate is within jet --> skipping." << std::endl;
    }
  }
  auto sumLeptons = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumLeptons);
  auto sumLeptonIsoCones = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumLeptonIsoCones);
  int leptonIdx = 0;
  for (std::vector<CommonMETData>::iterator sumJetsPlusPFCandidates = sumJetsPlusPFCandidates_leptons.begin();
       sumJetsPlusPFCandidates != sumJetsPlusPFCandidates_leptons.end();
       ++sumJetsPlusPFCandidates) {
    if (sumJetsPlusPFCandidates->sumet > leptons[leptonIdx].pt()) {
      double leptonEnFrac = leptons[leptonIdx].pt() / sumJetsPlusPFCandidates->sumet;
      assert(leptonEnFrac >= 0.0 && leptonEnFrac <= 1.0);
      sumLeptons->mex += (leptonEnFrac * sumJetsPlusPFCandidates->mex);
      sumLeptons->mey += (leptonEnFrac * sumJetsPlusPFCandidates->mey);
      sumLeptons->sumet += (leptonEnFrac * sumJetsPlusPFCandidates->sumet);
      double leptonIsoConeEnFrac = 1.0 - leptonEnFrac;
      assert(leptonIsoConeEnFrac >= 0.0 && leptonIsoConeEnFrac <= 1.0);
      sumLeptonIsoCones->mex += (leptonIsoConeEnFrac * sumJetsPlusPFCandidates->mex);
      sumLeptonIsoCones->mey += (leptonIsoConeEnFrac * sumJetsPlusPFCandidates->mey);
      sumLeptonIsoCones->sumet += (leptonIsoConeEnFrac * sumJetsPlusPFCandidates->sumet);
    } else {
      sumLeptons->mex += sumJetsPlusPFCandidates->mex;
      sumLeptons->mey += sumJetsPlusPFCandidates->mey;
      sumLeptons->sumet += sumJetsPlusPFCandidates->sumet;
    }
    ++leptonIdx;
  }

  reco::PUSubMETCandInfoCollection jets_cleaned = utils_.cleanJets(*jets, leptons, 0.5, false);
  reco::PUSubMETCandInfoCollection pfCandidates_cleaned = utils_.cleanPFCandidates(*pfCandidates, leptons, 0.3, false);

  auto sumNoPUjets = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumNoPUjets);
  std::vector<metsig::SigInputObj> metSignObjectsNoPUjets;
  auto sumNoPUjetOffsetEnCorr = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumNoPUjetOffsetEnCorr);
  std::vector<metsig::SigInputObj> metSignObjectsNoPUjetOffsetEnCorr;
  auto sumPUjets = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumPUjets);
  std::vector<metsig::SigInputObj> metSignObjectsPUjets;
  for (reco::PUSubMETCandInfoCollection::const_iterator jet = jets_cleaned.begin(); jet != jets_cleaned.end(); ++jet) {
    if (jet->passesLooseJetId()) {
      if (jet->type() == reco::PUSubMETCandInfo::kHS) {
        addToCommonMETData(*sumNoPUjets, jet->p4());
        metSignObjectsNoPUjets.push_back(jet->metSignObj());
        float jetp = jet->p4().P();
        float jetcorr = jet->offsetEnCorr();
        sumNoPUjetOffsetEnCorr->mex += jetcorr * jet->p4().px() / jetp;
        sumNoPUjetOffsetEnCorr->mey += jetcorr * jet->p4().py() / jetp;
        sumNoPUjetOffsetEnCorr->mez += jetcorr * jet->p4().pz() / jetp;
        sumNoPUjetOffsetEnCorr->sumet += jetcorr * jet->p4().pt() / jetp;
        metsig::SigInputObj pfMEtSignObjectOffsetEnCorr(
            jet->metSignObj().get_type(),
            jet->offsetEnCorr(),
            jet->metSignObj().get_phi(),
            (jet->offsetEnCorr() / jet->p4().E()) * jet->metSignObj().get_sigma_e(),
            jet->metSignObj().get_sigma_tan());
        metSignObjectsNoPUjetOffsetEnCorr.push_back(pfMEtSignObjectOffsetEnCorr);
      } else {
        addToCommonMETData(*sumPUjets, jet->p4());
        metSignObjectsPUjets.push_back(jet->metSignObj());
      }
    }
  }

  auto sumNoPUunclChargedCands = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumNoPUunclChargedCands);
  std::vector<metsig::SigInputObj> metSignObjectsNoPUunclChargedCands;
  auto sumPUunclChargedCands = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumPUunclChargedCands);
  std::vector<metsig::SigInputObj> metSignObjectsPUunclChargedCands;
  auto sumUnclNeutralCands = std::make_unique<CommonMETData>();
  initializeCommonMETData(*sumUnclNeutralCands);
  std::vector<metsig::SigInputObj> metSignObjectsUnclNeutralCands;
  for (reco::PUSubMETCandInfoCollection::const_iterator pfCandidate = pfCandidates_cleaned.begin();
       pfCandidate != pfCandidates_cleaned.end();
       ++pfCandidate) {
    if (pfCandidate->passesLooseJetId()) {
      if (!pfCandidate->isWithinJet()) {
        if (pfCandidate->type() == reco::PUSubMETCandInfo::kChHS) {
          addToCommonMETData(*sumNoPUunclChargedCands, pfCandidate->p4());
          metSignObjectsNoPUunclChargedCands.push_back(pfCandidate->metSignObj());
        } else if (pfCandidate->type() == reco::PUSubMETCandInfo::kChPU) {
          addToCommonMETData(*sumPUunclChargedCands, pfCandidate->p4());
          metSignObjectsPUunclChargedCands.push_back(pfCandidate->metSignObj());
        } else if (pfCandidate->type() == reco::PUSubMETCandInfo::kNeutral) {
          addToCommonMETData(*sumUnclNeutralCands, pfCandidate->p4());
          metSignObjectsUnclNeutralCands.push_back(pfCandidate->metSignObj());
        }
      }
    }
  }

  edm::Handle<CorrMETData> type0Correction_input;
  evt.getByToken(srcType0Correction_, type0Correction_input);
  auto type0Correction_output = std::make_unique<CommonMETData>();
  initializeCommonMETData(*type0Correction_output);
  type0Correction_output->mex = type0Correction_input->mex;
  type0Correction_output->mey = type0Correction_input->mey;

  finalizeCommonMETData(*sumLeptons);
  finalizeCommonMETData(*sumNoPUjetOffsetEnCorr);
  finalizeCommonMETData(*sumNoPUjets);
  finalizeCommonMETData(*sumPUjets);
  finalizeCommonMETData(*sumNoPUunclChargedCands);
  finalizeCommonMETData(*sumPUunclChargedCands);
  finalizeCommonMETData(*sumUnclNeutralCands);
  finalizeCommonMETData(*type0Correction_output);
  finalizeCommonMETData(*sumLeptonIsoCones);

  double noPileUpScaleFactor =
      (sumPUunclChargedCands->sumet > 0.)
          ? (sumPUunclChargedCands->sumet / (sumNoPUunclChargedCands->sumet + sumPUunclChargedCands->sumet))
          : 1.;
  LogDebug("produce") << "noPileUpScaleFactor = " << noPileUpScaleFactor << std::endl;

  double noPileUpMEtPx =
      -(sumLeptons->mex + sumNoPUjets->mex + sumNoPUunclChargedCands->mex +
        noPileUpScaleFactor *
            (sfNoPUjetOffsetEnCorr_ * sumNoPUjetOffsetEnCorr->mex + sfUnclNeutralCands_ * sumUnclNeutralCands->mex +
             sfPUunclChargedCands_ * sumPUunclChargedCands->mex + sfPUjets_ * sumPUjets->mex)) +
      noPileUpScaleFactor * sfType0Correction_ * type0Correction_output->mex;
  if (sfLeptonIsoCones_ >= 0.)
    noPileUpMEtPx -= (noPileUpScaleFactor * sfLeptonIsoCones_ * sumLeptonIsoCones->mex);
  else
    noPileUpMEtPx -= (std::abs(sfLeptonIsoCones_) * sumLeptonIsoCones->mex);
  double noPileUpMEtPy =
      -(sumLeptons->mey + sumNoPUjets->mey + sumNoPUunclChargedCands->mey +
        noPileUpScaleFactor *
            (sfNoPUjetOffsetEnCorr_ * sumNoPUjetOffsetEnCorr->mey + sfUnclNeutralCands_ * sumUnclNeutralCands->mey +
             sfPUunclChargedCands_ * sumPUunclChargedCands->mey + sfPUjets_ * sumPUjets->mey)) +
      noPileUpScaleFactor * sfType0Correction_ * type0Correction_output->mey;
  if (sfLeptonIsoCones_ >= 0.)
    noPileUpMEtPy -= (noPileUpScaleFactor * sfLeptonIsoCones_ * sumLeptonIsoCones->mey);
  else
    noPileUpMEtPy -= (std::abs(sfLeptonIsoCones_) * sumLeptonIsoCones->mey);
  double noPileUpMEtPt = sqrt(noPileUpMEtPx * noPileUpMEtPx + noPileUpMEtPy * noPileUpMEtPy);
  reco::Candidate::LorentzVector noPileUpMEtP4(noPileUpMEtPx, noPileUpMEtPy, 0., noPileUpMEtPt);

  reco::PFMET noPileUpMEt(pfMEt_original);
  noPileUpMEt.setP4(noPileUpMEtP4);
  //noPileUpMEt.setSignificanceMatrix(pfMEtCov);

  std::vector<metsig::SigInputObj> metSignObjects_scaled;
  scaleAndAddPFMEtSignObjects(metSignObjects_scaled, metSignObjectsLeptons, 1.0, sfMEtCovMin_, sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(
      metSignObjects_scaled, metSignObjectsNoPUjetOffsetEnCorr, sfNoPUjetOffsetEnCorr_, sfMEtCovMin_, sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(metSignObjects_scaled, metSignObjectsNoPUjets, sfNoPUjets_, sfMEtCovMin_, sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(
      metSignObjects_scaled, metSignObjectsPUjets, noPileUpScaleFactor * sfPUjets_, sfMEtCovMin_, sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(
      metSignObjects_scaled, metSignObjectsNoPUunclChargedCands, sfNoPUunclChargedCands_, sfMEtCovMin_, sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(metSignObjects_scaled,
                              metSignObjectsPUunclChargedCands,
                              noPileUpScaleFactor * sfPUunclChargedCands_,
                              sfMEtCovMin_,
                              sfMEtCovMax_);
  scaleAndAddPFMEtSignObjects(metSignObjects_scaled,
                              metSignObjectsUnclNeutralCands,
                              noPileUpScaleFactor * sfUnclNeutralCands_,
                              sfMEtCovMin_,
                              sfMEtCovMax_);
  reco::METCovMatrix pfMEtCov_recomputed = computePFMEtSignificance(metSignObjects_scaled);
  noPileUpMEt.setSignificanceMatrix(pfMEtCov_recomputed);

  LogDebug("produce") << "<NoPileUpPFMEtProducer::produce>:" << std::endl
                      << " moduleLabel = " << moduleLabel_ << std::endl
                      << " PFMET: Pt = " << pfMEt_original.pt() << ", phi = " << pfMEt_original.phi() << " "
                      << "(Px = " << pfMEt_original.px() << ", Py = " << pfMEt_original.py() << ")" << std::endl
                      << " Cov:" << std::endl
                      << " " << pfMEtCov(0, 0) << "  " << pfMEtCov(0, 1) << "\n " << pfMEtCov(1, 0) << "  "
                      << pfMEtCov(1, 1) << std::endl
                      << " no-PU MET: Pt = " << noPileUpMEt.pt() << ", phi = " << noPileUpMEt.phi() << " "
                      << "(Px = " << noPileUpMEt.px() << ", Py = " << noPileUpMEt.py() << ")" << std::endl
                      << " Cov:" << std::endl
                      << " " << (noPileUpMEt.getSignificanceMatrix())(0, 0) << "  "
                      << (noPileUpMEt.getSignificanceMatrix())(0, 1) << std::endl
                      << (noPileUpMEt.getSignificanceMatrix())(1, 0) << "  "
                      << (noPileUpMEt.getSignificanceMatrix())(1, 1) << std::endl;

  // add no-PU MET object to the event
  auto noPileUpMEtCollection = std::make_unique<reco::PFMETCollection>();
  noPileUpMEtCollection->push_back(noPileUpMEt);

  evt.put(std::move(noPileUpMEtCollection));
  if (saveInputs_) {
    evt.put(std::move(sumLeptons), sfLeptonsName_);
    evt.put(std::move(sumNoPUjetOffsetEnCorr), sfNoPUjetOffsetEnCorrName_);
    evt.put(std::move(sumNoPUjets), sfNoPUjetsName_);
    evt.put(std::move(sumPUjets), sfPUjetsName_);
    evt.put(std::move(sumNoPUunclChargedCands), sfNoPUunclChargedCandsName_);
    evt.put(std::move(sumPUunclChargedCands), sfPUunclChargedCandsName_);
    evt.put(std::move(sumUnclNeutralCands), sfUnclNeutralCandsName_);
    evt.put(std::move(type0Correction_output), sfType0CorrectionName_);
    evt.put(std::move(sumLeptonIsoCones), sfLeptonIsoConesName_);
  }

  evt.put(std::make_unique<double>(noPileUpScaleFactor), "sfNoPU");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(NoPileUpPFMEtProducer);
