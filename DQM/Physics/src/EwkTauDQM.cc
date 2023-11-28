#include "DQM/Physics/src/EwkTauDQM.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

const std::string dqmSeparator = "/";

std::string dqmDirectoryName(const std::string& dqmRootDirectory, const std::string& dqmSubDirectory) {
  //--- concatenate names of dqmRootDirectory and dqmSubDirectory;
  //    add "/" separator inbetween if necessary
  std::string dirName = dqmRootDirectory;
  if (!dirName.empty() && dirName.find_last_of(dqmSeparator) != (dirName.length() - 1))
    dirName.append(dqmSeparator);
  dirName.append(dqmSubDirectory);
  return dirName;
}

EwkTauDQM::EwkTauDQM(const edm::ParameterSet& cfg) : dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")) {
  maxNumWarnings_ = cfg.exists("maxNumWarnings") ? cfg.getParameter<int>("maxNumWarnings") : 1;

  edm::ParameterSet cfgChannels = cfg.getParameter<edm::ParameterSet>("channels");

  edm::ParameterSet cfgElecTauChannel = cfgChannels.getParameter<edm::ParameterSet>("elecTauChannel");
  std::string dqmSubDirectoryElecTauChannel = cfgElecTauChannel.getParameter<std::string>("dqmSubDirectory");
  cfgElecTauChannel.addParameter<std::string>("dqmDirectory",
                                              dqmDirectoryName(dqmDirectory_, dqmSubDirectoryElecTauChannel));
  cfgElecTauChannel.addParameter<int>("maxNumWarnings", maxNumWarnings_);
  elecTauHistManager_ = new EwkElecTauHistManager(cfgElecTauChannel);

  edm::ParameterSet cfgMuTauChannel = cfgChannels.getParameter<edm::ParameterSet>("muTauChannel");
  std::string dqmSubDirectoryMuTauChannel = cfgMuTauChannel.getParameter<std::string>("dqmSubDirectory");
  cfgMuTauChannel.addParameter<std::string>("dqmDirectory",
                                            dqmDirectoryName(dqmDirectory_, dqmSubDirectoryMuTauChannel));
  cfgMuTauChannel.addParameter<int>("maxNumWarnings", maxNumWarnings_);
  muTauHistManager_ = new EwkMuTauHistManager(cfgMuTauChannel);
}

EwkTauDQM::~EwkTauDQM() {
  delete elecTauHistManager_;
  delete muTauHistManager_;
}

void EwkTauDQM::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  elecTauHistManager_->bookHistograms(iBooker);
  muTauHistManager_->bookHistograms(iBooker);
}

void EwkTauDQM::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  elecTauHistManager_->fillHistograms(evt, es);
  muTauHistManager_->fillHistograms(evt, es);
}

void EwkTauDQM::dqmEndRun(const edm::Run&, const edm::EventSetup&) {
  elecTauHistManager_->finalizeHistograms();
  muTauHistManager_->finalizeHistograms();
}

//-------------------------------------------------------------------------------
// code specific to Z --> e + tau-jet channel
//-------------------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TMath.h"

#include <iostream>
#include <iomanip>

EwkElecTauHistManager::EwkElecTauHistManager(const edm::ParameterSet& cfg)
    : dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")),
      numEventsAnalyzed_(0),
      numEventsSelected_(0),
      cfgError_(0),
      numWarningsTriggerResults_(0),
      numWarningsHLTpath_(0),
      numWarningsVertex_(0),
      numWarningsBeamSpot_(0),
      numWarningsElectron_(0),
      numWarningsTauJet_(0),
      numWarningsTauDiscrByLeadTrackFinding_(0),
      numWarningsTauDiscrByLeadTrackPtCut_(0),
      numWarningsTauDiscrByTrackIso_(0),
      numWarningsTauDiscrByEcalIso_(0),
      numWarningsTauDiscrAgainstElectrons_(0),
      numWarningsTauDiscrAgainstMuons_(0),
      numWarningsCaloMEt_(0),
      numWarningsPFMEt_(0) {
  triggerResultsSource_ = cfg.getParameter<edm::InputTag>("triggerResultsSource");
  vertexSource_ = cfg.getParameter<edm::InputTag>("vertexSource");
  beamSpotSource_ = cfg.getParameter<edm::InputTag>("beamSpotSource");
  electronSource_ = cfg.getParameter<edm::InputTag>("electronSource");
  tauJetSource_ = cfg.getParameter<edm::InputTag>("tauJetSource");
  caloMEtSource_ = cfg.getParameter<edm::InputTag>("caloMEtSource");
  pfMEtSource_ = cfg.getParameter<edm::InputTag>("pfMEtSource");

  tauDiscrByLeadTrackFinding_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackFinding");
  tauDiscrByLeadTrackPtCut_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackPtCut");
  tauDiscrByTrackIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByTrackIso");
  tauDiscrByEcalIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByEcalIso");
  tauDiscrAgainstElectrons_ = cfg.getParameter<edm::InputTag>("tauDiscrAgainstElectrons");
  tauDiscrAgainstMuons_ = cfg.getParameter<edm::InputTag>("tauDiscrAgainstMuons");

  hltPaths_ = cfg.getParameter<vstring>("hltPaths");

  electronEtaCut_ = cfg.getParameter<double>("electronEtaCut");
  electronPtCut_ = cfg.getParameter<double>("electronPtCut");
  electronTrackIsoCut_ = cfg.getParameter<double>("electronTrackIsoCut");
  electronEcalIsoCut_ = cfg.getParameter<double>("electronEcalIsoCut");
  std::string electronIsoMode_string = cfg.getParameter<std::string>("electronIsoMode");
  electronIsoMode_ = getIsoMode(electronIsoMode_string, cfgError_);

  tauJetEtaCut_ = cfg.getParameter<double>("tauJetEtaCut");
  tauJetPtCut_ = cfg.getParameter<double>("tauJetPtCut");

  visMassCut_ = cfg.getParameter<double>("visMassCut");

  maxNumWarnings_ = cfg.exists("maxNumWarnings") ? cfg.getParameter<int>("maxNumWarnings") : 1;
}

void EwkElecTauHistManager::bookHistograms(DQMStore::IBooker& iBooker) {
  iBooker.setCurrentFolder(dqmDirectory_);
  hElectronPt_ = iBooker.book1D("ElectronPt", "P_{T}^{e}", 20, 0., 100.);
  hElectronEta_ = iBooker.book1D("ElectronEta", "#eta_{e}", 20, -4.0, +4.0);
  hElectronPhi_ = iBooker.book1D("ElectronPhi", "#phi_{e}", 20, -TMath::Pi(), +TMath::Pi());
  hElectronTrackIsoPt_ = iBooker.book1D("ElectronTrackIsoPt", "Electron Track Iso.", 20, -0.01, 0.5);
  hElectronEcalIsoPt_ = iBooker.book1D("ElectronEcalIsoPt", "Electron Ecal Iso.", 20, -0.01, 0.5);
  hTauJetPt_ = iBooker.book1D("TauJetPt", "P_{T}^{#tau-Jet}", 20, 0., 100.);
  hTauJetEta_ = iBooker.book1D("TauJetEta", "#eta_{#tau-Jet}", 20, -4.0, +4.0);
  hVisMass_ = iBooker.book1D("VisMass", "e + #tau-Jet visible Mass", 20, 20., 120.);
  hMtElecPFMEt_ = iBooker.book1D("MtElecPFMEt", "e + E_{T}^{miss} (PF) transverse Mass", 20, 20., 120.);
  hElecTauAcoplanarity_ =
      iBooker.book1D("ElecTauAcoplanarity", "#Delta #phi_{e #tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
  hElecTauCharge_ = iBooker.book1D("ElecTauCharge", "Q_{e * #tau-Jet}", 5, -2.5, +2.5);
  hVertexZ_ = iBooker.book1D("VertexZ", "Event Vertex z-Position", 20, -25., +25.);
  hCaloMEtPt_ = iBooker.book1D("CaloMEtPt", "E_{T}^{miss} (Calo)", 20, 0., 100.);
  hPFMEtPt_ = iBooker.book1D("PFMEtPt", "E_{T}^{miss} (PF)", 20, 0., 100.);
  hCutFlowSummary_ = iBooker.book1D("CutFlowSummary", "Cut-flow Summary", 11, 0.5, 11.5);
  hCutFlowSummary_->setBinLabel(kPassedPreselection, "Preselection");
  hCutFlowSummary_->setBinLabel(kPassedTrigger, "HLT");
  hCutFlowSummary_->setBinLabel(kPassedElectronId, "e ID");
  hCutFlowSummary_->setBinLabel(kPassedElectronTrackIso, "e Trk Iso.");
  hCutFlowSummary_->setBinLabel(kPassedElectronEcalIso, "e Ecal Iso.");
  hCutFlowSummary_->setBinLabel(kPassedTauLeadTrack, "#tau lead. Track");
  hCutFlowSummary_->setBinLabel(kPassedTauLeadTrackPt, "#tau lead. Track P_{T}");
  hCutFlowSummary_->setBinLabel(kPassedTauDiscrAgainstElectrons, "#tau anti-e Discr.");
  hCutFlowSummary_->setBinLabel(kPassedTauDiscrAgainstMuons, "#tau anti-#mu Discr.");
  hCutFlowSummary_->setBinLabel(kPassedTauTrackIso, "#tau Track Iso.");
  hCutFlowSummary_->setBinLabel(kPassedTauEcalIso, "#tau Ecal Iso.");
}

void EwkElecTauHistManager::fillHistograms(const edm::Event& evt, const edm::EventSetup& es) {
  if (cfgError_)
    return;

  //-----------------------------------------------------------------------------
  // access event-level information
  //-----------------------------------------------------------------------------

  bool readError = false;

  //--- get decision of high-level trigger for the event
  edm::Handle<edm::TriggerResults> hltDecision;
  readEventData(evt,
                triggerResultsSource_,
                hltDecision,
                numWarningsTriggerResults_,
                maxNumWarnings_,
                readError,
                "Failed to access Trigger results");
  if (readError)
    return;

  const edm::TriggerNames& triggerNames = evt.triggerNames(*hltDecision);

  bool isTriggered = false;
  for (vstring::const_iterator hltPath = hltPaths_.begin(); hltPath != hltPaths_.end(); ++hltPath) {
    unsigned int index = triggerNames.triggerIndex(*hltPath);
    if (index < triggerNames.size()) {
      if (hltDecision->accept(index))
        isTriggered = true;
    } else {
      if (numWarningsHLTpath_ < maxNumWarnings_ || maxNumWarnings_ == -1)
        edm::LogWarning("EwkElecTauHistManager") << " Undefined HLT path = " << (*hltPath) << " !!";
      ++numWarningsHLTpath_;
      continue;
    }
  }

  //--- get reconstructed primary event vertex of the event
  //   (take as "the" primary event vertex the first entry in the collection
  //    of vertex objects, corresponding to the vertex associated to the highest
  // Pt sum of tracks)
  edm::Handle<reco::VertexCollection> vertexCollection;
  readEventData(evt,
                vertexSource_,
                vertexCollection,
                numWarningsVertex_,
                maxNumWarnings_,
                readError,
                "Failed to access Vertex collection");
  if (readError)
    return;

  const reco::Vertex* theEventVertex = (!vertexCollection->empty()) ? &(vertexCollection->at(0)) : nullptr;

  //--- get beam-spot (expected vertex position) for the event
  edm::Handle<reco::BeamSpot> beamSpot;
  readEventData(
      evt, beamSpotSource_, beamSpot, numWarningsBeamSpot_, maxNumWarnings_, readError, "Failed to access Beam-spot");
  if (readError)
    return;

  //--- get collections of reconstructed electrons from the event
  edm::Handle<reco::GsfElectronCollection> electrons;
  readEventData(evt,
                electronSource_,
                electrons,
                numWarningsElectron_,
                maxNumWarnings_,
                readError,
                "Failed to access Electron collection");
  if (readError)
    return;

  const reco::GsfElectron* theElectron = getTheElectron(*electrons, electronEtaCut_, electronPtCut_);

  double theElectronTrackIsoPt = 1.e+3;
  double theElectronEcalIsoPt = 1.e+3;
  if (theElectron) {
    theElectronTrackIsoPt = theElectron->dr03TkSumPt();
    theElectronEcalIsoPt = theElectron->dr03EcalRecHitSumEt();

    if (electronIsoMode_ == kRelativeIso && theElectron->pt() > 0.) {
      theElectronTrackIsoPt /= theElectron->pt();
      theElectronEcalIsoPt /= theElectron->pt();
    }
  }

  //--- get collections of reconstructed tau-jets from the event
  edm::Handle<reco::PFTauCollection> tauJets;
  readEventData(evt,
                tauJetSource_,
                tauJets,
                numWarningsTauJet_,
                maxNumWarnings_,
                readError,
                "Failed to access Tau-jet collection");
  if (readError)
    return;

  //--- get collections of tau-jet discriminators for those tau-jets
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackFinding;
  readEventData(evt,
                tauDiscrByLeadTrackFinding_,
                tauDiscrByLeadTrackFinding,
                numWarningsTauDiscrByLeadTrackFinding_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "leading Track finding");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackPtCut;
  readEventData(evt,
                tauDiscrByLeadTrackPtCut_,
                tauDiscrByLeadTrackPtCut,
                numWarningsTauDiscrByLeadTrackPtCut_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "leading Track Pt cut");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByTrackIso;
  readEventData(evt,
                tauDiscrByTrackIso_,
                tauDiscrByTrackIso,
                numWarningsTauDiscrByTrackIso_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "Track isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByEcalIso;
  readEventData(evt,
                tauDiscrByTrackIso_,
                tauDiscrByEcalIso,
                numWarningsTauDiscrByEcalIso_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by ECAL "
                "isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrAgainstElectrons;
  readEventData(evt,
                tauDiscrAgainstElectrons_,
                tauDiscrAgainstElectrons,
                numWarningsTauDiscrAgainstElectrons_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators against "
                "Electrons");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrAgainstMuons;
  readEventData(evt,
                tauDiscrAgainstMuons_,
                tauDiscrAgainstMuons,
                numWarningsTauDiscrAgainstMuons_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators against Muons");
  if (readError)
    return;

  int theTauJetIndex = -1;
  const reco::PFTau* theTauJet = getTheTauJet(*tauJets, tauJetEtaCut_, tauJetPtCut_, theTauJetIndex);

  double theTauDiscrByLeadTrackFinding = -1.;
  double theTauDiscrByLeadTrackPtCut = -1.;
  double theTauDiscrByTrackIso = -1.;
  double theTauDiscrByEcalIso = -1.;
  double theTauDiscrAgainstElectrons = -1.;
  double theTauDiscrAgainstMuons = -1.;
  if (theTauJetIndex != -1) {
    reco::PFTauRef theTauJetRef(tauJets, theTauJetIndex);
    theTauDiscrByLeadTrackFinding = (*tauDiscrByLeadTrackFinding)[theTauJetRef];
    theTauDiscrByLeadTrackPtCut = (*tauDiscrByLeadTrackPtCut)[theTauJetRef];
    theTauDiscrByTrackIso = (*tauDiscrByTrackIso)[theTauJetRef];
    theTauDiscrByEcalIso = (*tauDiscrByEcalIso)[theTauJetRef];
    theTauDiscrAgainstElectrons = (*tauDiscrAgainstElectrons)[theTauJetRef];
    theTauDiscrAgainstMuons = (*tauDiscrAgainstMuons)[theTauJetRef];
  }

  //--- get missing transverse momentum
  //    measured by calorimeters/reconstructed by particle-flow algorithm
  edm::Handle<reco::CaloMETCollection> caloMEtCollection;
  readEventData(evt,
                caloMEtSource_,
                caloMEtCollection,
                numWarningsCaloMEt_,
                maxNumWarnings_,
                readError,
                "Failed to access calo. MET collection");
  if (readError)
    return;

  const reco::CaloMET& caloMEt = caloMEtCollection->at(0);

  edm::Handle<reco::PFMETCollection> pfMEtCollection;
  readEventData(evt,
                pfMEtSource_,
                pfMEtCollection,
                numWarningsPFMEt_,
                maxNumWarnings_,
                readError,
                "Failed to access pf. MET collection");
  if (readError)
    return;

  const reco::PFMET& pfMEt = pfMEtCollection->at(0);

  if (!(theElectron && theTauJet && theTauJetIndex != -1))
    return;

  //-----------------------------------------------------------------------------
  // compute EWK tau analysis specific quantities
  //-----------------------------------------------------------------------------

  double dPhiElecTau = calcDeltaPhi(theElectron->phi(), theTauJet->phi());

  double mElecTau = (theElectron->p4() + theTauJet->p4()).M();

  // double mtElecCaloMEt = calcMt(theElectron->px(), theElectron->py(),
  // caloMEt.px(), caloMEt.py());
  double mtElecPFMEt = calcMt(theElectron->px(), theElectron->py(), pfMEt.px(), pfMEt.py());

  // double pZetaCaloMEt = calcPzeta(theElectron->p4(), theTauJet->p4(),
  // caloMEt.px(), caloMEt.py());
  // double pZetaPFMEt = calcPzeta(theElectron->p4(), theTauJet->p4(),
  // pfMEt.px(), pfMEt.py());

  //-----------------------------------------------------------------------------
  // apply selection criteria; fill histograms
  //-----------------------------------------------------------------------------

  ++numEventsAnalyzed_;

  bool isSelected = false;
  bool fullSelect = false;
  int cutFlowStatus = -1;

  if (mElecTau > visMassCut_) {
    cutFlowStatus = kPassedPreselection;
  }
  if (cutFlowStatus == kPassedPreselection && (isTriggered || hltPaths_.empty())) {
    cutFlowStatus = kPassedTrigger;
  }
  if (cutFlowStatus == kPassedTrigger && passesElectronId(*theElectron)) {
    cutFlowStatus = kPassedElectronId;
    hElectronTrackIsoPt_->Fill(theElectronTrackIsoPt);
  }
  if (cutFlowStatus == kPassedElectronId && theElectronTrackIsoPt < electronTrackIsoCut_) {
    cutFlowStatus = kPassedElectronTrackIso;
    hElectronEcalIsoPt_->Fill(theElectronEcalIsoPt);
  }
  if (cutFlowStatus == kPassedElectronTrackIso && theElectronEcalIsoPt < electronEcalIsoCut_) {
    cutFlowStatus = kPassedElectronEcalIso;
  }
  if (cutFlowStatus == kPassedElectronEcalIso && theTauDiscrByLeadTrackFinding > 0.5) {
    cutFlowStatus = kPassedTauLeadTrack;
    // if ( theTauJet->leadTrack().isAvailable() )
    // hTauLeadTrackPt_->Fill(theTauJet->leadTrack()->pt());
  }
  if (cutFlowStatus == kPassedTauLeadTrack && theTauDiscrByLeadTrackPtCut > 0.5) {
    cutFlowStatus = kPassedTauLeadTrackPt;
    // hTauTrackIsoPt_->Fill(theTauJet->isolationPFChargedHadrCandsPtSum());
  }
  if (cutFlowStatus == kPassedTauLeadTrackPt && theTauDiscrAgainstElectrons > 0.5) {
    cutFlowStatus = kPassedTauDiscrAgainstElectrons;
    // hTauDiscrAgainstMuons_->Fill(theTauDiscrAgainstMuons);
  }
  if (cutFlowStatus == kPassedTauDiscrAgainstElectrons && theTauDiscrAgainstMuons > 0.5) {
    cutFlowStatus = kPassedTauDiscrAgainstMuons;
    isSelected = true;
  }
  if (cutFlowStatus == kPassedTauDiscrAgainstMuons && theTauDiscrByTrackIso > 0.5) {
    cutFlowStatus = kPassedTauTrackIso;
    // hTauEcalIsoPt_->Fill(theTauJet->isolationPFGammaCandsEtSum());
  }
  if (cutFlowStatus == kPassedTauTrackIso && theTauDiscrByEcalIso > 0.5) {
    cutFlowStatus = kPassedTauEcalIso;
    fullSelect = true;
    // hTauDiscrAgainstElectrons_->Fill(theTauDiscrAgainstElectrons);
  }

  for (int iCut = 1; iCut <= cutFlowStatus; ++iCut) {
    hCutFlowSummary_->Fill(iCut);
  }

  if (isSelected) {
    hElectronPt_->Fill(theElectron->pt());
    hElectronEta_->Fill(theElectron->eta());
    hElectronPhi_->Fill(theElectron->phi());

    hTauJetPt_->Fill(theTauJet->pt());
    hTauJetEta_->Fill(theTauJet->eta());
    // hTauJetPhi_->Fill(theTauJet->phi());

    // hTauJetCharge_->Fill(theTauJet->charge());
    // if ( theTauJet->signalTracks().isAvailable()    )
    // hTauJetNumSignalTracks_->Fill(theTauJet->signalTracks().size());
    // if ( theTauJet->isolationTracks().isAvailable() )
    // hTauJetNumIsoTracks_->Fill(theTauJet->isolationTracks().size());

    if (fullSelect) {
      hVisMass_->Fill(mElecTau);
    }
    // hMtElecCaloMEt_->Fill(mtElecCaloMEt);
    hMtElecPFMEt_->Fill(mtElecPFMEt);
    // hPzetaCaloMEt_->Fill(pZetaCaloMEt);
    // hPzetaPFMEt_->Fill(pZetaPFMEt);
    hElecTauAcoplanarity_->Fill(dPhiElecTau);
    hElecTauCharge_->Fill(theElectron->charge() * theTauJet->charge());

    if (theEventVertex) {
      // hVertexChi2_->Fill(theEventVertex->normalizedChi2());
      hVertexZ_->Fill(theEventVertex->z());
      // hVertexD0_->Fill(getVertexD0(*theEventVertex, *beamSpot));
    }

    hCaloMEtPt_->Fill(caloMEt.pt());
    // hCaloMEtPhi_->Fill(caloMEt.phi());

    hPFMEtPt_->Fill(pfMEt.pt());
    // hPFMEtPhi_->Fill(pfMEt.phi());
  }

  if (isSelected)
    ++numEventsSelected_;
}

void EwkElecTauHistManager::finalizeHistograms() {
  edm::LogInfo("EwkElecTauHistManager") << "Filter-Statistics Summary:" << std::endl
                                        << " Events analyzed = " << numEventsAnalyzed_ << std::endl
                                        << " Events selected = " << numEventsSelected_;
  if (numEventsAnalyzed_ > 0) {
    double eff = numEventsSelected_ / (double)numEventsAnalyzed_;
    edm::LogInfo("") << "Overall efficiency = " << std::setprecision(4) << eff * 100. << " +/- " << std::setprecision(4)
                     << TMath::Sqrt(eff * (1 - eff) / numEventsAnalyzed_) * 100. << ")%";
  }
}

//-------------------------------------------------------------------------------
// code specific to Z --> mu + tau-jet channel
//-------------------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TMath.h"

#include <iostream>
#include <iomanip>

EwkMuTauHistManager::EwkMuTauHistManager(const edm::ParameterSet& cfg)
    : dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")),
      numEventsAnalyzed_(0),
      numEventsSelected_(0),
      cfgError_(0),
      numWarningsTriggerResults_(0),
      numWarningsHLTpath_(0),
      numWarningsVertex_(0),
      numWarningsBeamSpot_(0),
      numWarningsMuon_(0),
      numWarningsTauJet_(0),
      numWarningsTauDiscrByLeadTrackFinding_(0),
      numWarningsTauDiscrByLeadTrackPtCut_(0),
      numWarningsTauDiscrByTrackIso_(0),
      numWarningsTauDiscrByEcalIso_(0),
      numWarningsTauDiscrAgainstMuons_(0),
      numWarningsCaloMEt_(0),
      numWarningsPFMEt_(0) {
  triggerResultsSource_ = cfg.getParameter<edm::InputTag>("triggerResultsSource");
  vertexSource_ = cfg.getParameter<edm::InputTag>("vertexSource");
  beamSpotSource_ = cfg.getParameter<edm::InputTag>("beamSpotSource");
  muonSource_ = cfg.getParameter<edm::InputTag>("muonSource");
  tauJetSource_ = cfg.getParameter<edm::InputTag>("tauJetSource");
  caloMEtSource_ = cfg.getParameter<edm::InputTag>("caloMEtSource");
  pfMEtSource_ = cfg.getParameter<edm::InputTag>("pfMEtSource");

  tauDiscrByLeadTrackFinding_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackFinding");
  tauDiscrByLeadTrackPtCut_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackPtCut");
  tauDiscrByTrackIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByTrackIso");
  tauDiscrByEcalIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByEcalIso");
  tauDiscrAgainstMuons_ = cfg.getParameter<edm::InputTag>("tauDiscrAgainstMuons");

  hltPaths_ = cfg.getParameter<vstring>("hltPaths");

  muonEtaCut_ = cfg.getParameter<double>("muonEtaCut");
  muonPtCut_ = cfg.getParameter<double>("muonPtCut");
  muonTrackIsoCut_ = cfg.getParameter<double>("muonTrackIsoCut");
  muonEcalIsoCut_ = cfg.getParameter<double>("muonEcalIsoCut");
  muonCombIsoCut_ = cfg.getParameter<double>("muonCombIsoCut");
  std::string muonIsoMode_string = cfg.getParameter<std::string>("muonIsoMode");
  muonIsoMode_ = getIsoMode(muonIsoMode_string, cfgError_);

  tauJetEtaCut_ = cfg.getParameter<double>("tauJetEtaCut");
  tauJetPtCut_ = cfg.getParameter<double>("tauJetPtCut");

  visMassCut_ = cfg.getParameter<double>("visMassCut");
  deltaRCut_ = cfg.getParameter<double>("deltaRCut");

  maxNumWarnings_ = cfg.exists("maxNumWarnings") ? cfg.getParameter<int>("maxNumWarnings") : 1;
}

void EwkMuTauHistManager::bookHistograms(DQMStore::IBooker& iBooker) {
  iBooker.setCurrentFolder(dqmDirectory_);

  hMuonPt_ = iBooker.book1D("MuonPt", "P_{T}^{#mu}", 20, 0., 100.);
  hMuonEta_ = iBooker.book1D("MuonEta", "#eta_{#mu}", 20, -4.0, +4.0);
  hMuonPhi_ = iBooker.book1D("MuonPhi", "#phi_{#mu}", 20, -TMath::Pi(), +TMath::Pi());
  hMuonTrackIsoPt_ = iBooker.book1D("MuonTrackIsoPt", "Muon Track Iso.", 20, -0.01, 10.);
  hMuonEcalIsoPt_ = iBooker.book1D("MuonEcalIsoPt", "Muon Ecal Iso.", 20, -0.01, 10.);
  hMuonCombIsoPt_ = iBooker.book1D("MuonCombIsoPt", "Muon Comb Iso.", 20, -0.01, 1.);

  hTauJetPt_ = iBooker.book1D("TauJetPt", "P_{T}^{#tau-Jet}", 20, 0., 100.);
  hTauJetEta_ = iBooker.book1D("TauJetEta", "#eta_{#tau-Jet}", 20, -4.0, +4.0);
  hTauJetPhi_ = iBooker.book1D("TauJetPhi", "#phi_{#tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
  hTauLeadTrackPt_ = iBooker.book1D("TauLeadTrackPt", "P_{T}^{#tau-Jetldg trk}", 20, 0., 50.);
  hTauTrackIsoPt_ = iBooker.book1D("TauTrackIsoPt", "Tau Track Iso.", 20, -0.01, 40.);
  hTauEcalIsoPt_ = iBooker.book1D("TauEcalIsoPt", "Tau Ecal Iso.", 10, -0.01, 10.);
  hTauDiscrAgainstMuons_ = iBooker.book1D("TauDiscrAgainstMuons", "Tau Discr. against Muons", 2, -0.5, +1.5);
  hTauJetNumSignalTracks_ = iBooker.book1D("TauJetNumSignalTracks", "Num. Tau signal Cone Tracks", 20, -0.5, +19.5);
  hTauJetNumIsoTracks_ = iBooker.book1D("TauJetNumIsoTracks", "Num. Tau isolation Cone Tracks", 20, -0.5, +19.5);

  hVisMass_ = iBooker.book1D("VisMass", "#mu + #tau-Jet visible Mass", 20, 0., 120.);
  hVisMassFinal_ = iBooker.book1D("VisMassFinal", "#mu + #tau-Jet visible final Mass", 20, 0., 120.);
  hMtMuPFMEt_ = iBooker.book1D("MtMuPFMEt", "#mu + E_{T}^{miss} (PF) transverse Mass", 20, 0., 120.);
  hMuTauAcoplanarity_ =
      iBooker.book1D("MuTauAcoplanarity", "#Delta #phi_{#mu #tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
  hMuTauDeltaR_ = iBooker.book1D("MuTauDeltaR", "#Delta R_{#mu #tau-Jet}", 20, 0, 5);
  hVertexZ_ = iBooker.book1D("VertexZ", "Event Vertex z-Position", 20, -25., +25.);
  hCaloMEtPt_ = iBooker.book1D("CaloMEtPt", "E_{T}^{miss} (Calo)", 20, 0., 100.);
  hPFMEtPt_ = iBooker.book1D("PFMEtPt", "E_{T}^{miss} (PF)", 20, 0., 100.);
  hCutFlowSummary_ = iBooker.book1D("CutFlowSummary", "Cut-flow Summary", 11, 0.5, 11.5);
  hCutFlowSummary_->setBinLabel(kPassedPreselection, "Preselection");
  hCutFlowSummary_->setBinLabel(kPassedTrigger, "HLT");
  hCutFlowSummary_->setBinLabel(kPassedMuonId, "#mu ID");
  hCutFlowSummary_->setBinLabel(kPassedMuonTrackIso, "#mu Trk Iso.");
  hCutFlowSummary_->setBinLabel(kPassedMuonEcalIso, "#mu Ecal Iso.");
  hCutFlowSummary_->setBinLabel(kPassedTauLeadTrack, "#tau lead. Track");
  hCutFlowSummary_->setBinLabel(kPassedTauLeadTrackPt, "#tau lead. Track P_{T}");
  hCutFlowSummary_->setBinLabel(kPassedTauTrackIso, "#tau Track Iso.");
  hCutFlowSummary_->setBinLabel(kPassedTauEcalIso, "#tau Ecal Iso.");
  hCutFlowSummary_->setBinLabel(kPassedTauDiscrAgainstMuons, "#tau anti-#mu Discr.");
  hCutFlowSummary_->setBinLabel(kPassedDeltaR, "#DeltaR(#mu,#tau) ");
}

void EwkMuTauHistManager::fillHistograms(const edm::Event& evt, const edm::EventSetup& es) {
  if (cfgError_)
    return;

  //-----------------------------------------------------------------------------
  // access event-level information
  //-----------------------------------------------------------------------------

  bool readError = false;

  //--- get decision of high-level trigger for the event
  edm::Handle<edm::TriggerResults> hltDecision;
  readEventData(evt,
                triggerResultsSource_,
                hltDecision,
                numWarningsTriggerResults_,
                maxNumWarnings_,
                readError,
                "Failed to access Trigger results");
  if (readError)
    return;

  const edm::TriggerNames& triggerNames = evt.triggerNames(*hltDecision);

  bool isTriggered = false;
  for (vstring::const_iterator hltPath = hltPaths_.begin(); hltPath != hltPaths_.end(); ++hltPath) {
    unsigned int index = triggerNames.triggerIndex(*hltPath);
    if (index < triggerNames.size()) {
      if (hltDecision->accept(index))
        isTriggered = true;
    } else {
      if (numWarningsHLTpath_ < maxNumWarnings_ || maxNumWarnings_ == -1)
        edm::LogWarning("EwkMuTauHistManager") << " Undefined HLT path = " << (*hltPath) << " !!";
      ++numWarningsHLTpath_;
      continue;
    }
  }

  //--- get reconstructed primary event vertex of the event
  //   (take as "the" primary event vertex the first entry in the collection
  //    of vertex objects, corresponding to the vertex associated to the highest
  // Pt sum of tracks)
  edm::Handle<reco::VertexCollection> vertexCollection;
  readEventData(evt,
                vertexSource_,
                vertexCollection,
                numWarningsVertex_,
                maxNumWarnings_,
                readError,
                "Failed to access Vertex collection");
  if (readError)
    return;

  const reco::Vertex* theEventVertex = (!vertexCollection->empty()) ? &(vertexCollection->at(0)) : nullptr;

  //--- get beam-spot (expected vertex position) for the event
  edm::Handle<reco::BeamSpot> beamSpot;
  readEventData(
      evt, beamSpotSource_, beamSpot, numWarningsBeamSpot_, maxNumWarnings_, readError, "Failed to access Beam-spot");
  if (readError)
    return;

  //--- get collections of reconstructed muons from the event
  edm::Handle<reco::MuonCollection> muons;
  readEventData(
      evt, muonSource_, muons, numWarningsMuon_, maxNumWarnings_, readError, "Failed to access Muon collection");
  if (readError)
    return;

  const reco::Muon* theMuon = getTheMuon(*muons, muonEtaCut_, muonPtCut_);

  double theMuonTrackIsoPt = 1.e+3;
  double theMuonEcalIsoPt = 1.e+3;
  double theMuonCombIsoPt = 1.e+3;

  if (theMuon) {
    theMuonTrackIsoPt = theMuon->isolationR05().sumPt;
    // mu.isolationR05().emEt + mu.isolationR05().hadEt +
    // mu.isolationR05().sumPt
    theMuonEcalIsoPt = theMuon->isolationR05().emEt;

    if (muonIsoMode_ == kRelativeIso && theMuon->pt() > 0.) {
      theMuonTrackIsoPt /= theMuon->pt();
      theMuonEcalIsoPt /= theMuon->pt();
      theMuonCombIsoPt = (theMuon->isolationR05().sumPt + theMuon->isolationR05().emEt) / theMuon->pt();
      // std::cout<<"Rel Iso ="<<theMuonCombIsoPt<<std::endl;
    }
  }

  //--- get collections of reconstructed tau-jets from the event
  edm::Handle<reco::PFTauCollection> tauJets;
  readEventData(evt,
                tauJetSource_,
                tauJets,
                numWarningsTauJet_,
                maxNumWarnings_,
                readError,
                "Failed to access Tau-jet collection");
  if (readError)
    return;

  //--- get collections of tau-jet discriminators for those tau-jets
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackFinding;
  readEventData(evt,
                tauDiscrByLeadTrackFinding_,
                tauDiscrByLeadTrackFinding,
                numWarningsTauDiscrByLeadTrackFinding_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "leading Track finding");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackPtCut;
  readEventData(evt,
                tauDiscrByLeadTrackPtCut_,
                tauDiscrByLeadTrackPtCut,
                numWarningsTauDiscrByLeadTrackPtCut_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "leading Track Pt cut");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByTrackIso;
  readEventData(evt,
                tauDiscrByTrackIso_,
                tauDiscrByTrackIso,
                numWarningsTauDiscrByTrackIso_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by "
                "Track isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByEcalIso;
  readEventData(evt,
                tauDiscrByTrackIso_,
                tauDiscrByEcalIso,
                numWarningsTauDiscrByEcalIso_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators by ECAL "
                "isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrAgainstMuons;
  readEventData(evt,
                tauDiscrAgainstMuons_,
                tauDiscrAgainstMuons,
                numWarningsTauDiscrAgainstMuons_,
                maxNumWarnings_,
                readError,
                "Failed to access collection of pf. Tau discriminators against Muons");
  if (readError)
    return;

  int theTauJetIndex = -1;
  const reco::PFTau* theTauJet = getTheTauJet(*tauJets, tauJetEtaCut_, tauJetPtCut_, theTauJetIndex);

  double theTauDiscrByLeadTrackFinding = -1.;
  double theTauDiscrByLeadTrackPtCut = -1.;
  double theTauDiscrByTrackIso = -1.;
  double theTauDiscrByEcalIso = -1.;
  double theTauDiscrAgainstMuons = -1.;
  if (theTauJetIndex != -1) {
    reco::PFTauRef theTauJetRef(tauJets, theTauJetIndex);
    theTauDiscrByLeadTrackFinding = (*tauDiscrByLeadTrackFinding)[theTauJetRef];
    theTauDiscrByLeadTrackPtCut = (*tauDiscrByLeadTrackPtCut)[theTauJetRef];
    theTauDiscrByTrackIso = (*tauDiscrByTrackIso)[theTauJetRef];
    theTauDiscrByEcalIso = (*tauDiscrByEcalIso)[theTauJetRef];
    theTauDiscrAgainstMuons = (*tauDiscrAgainstMuons)[theTauJetRef];
  }

  //--- get missing transverse momentum
  //    measured by calorimeters/reconstructed by particle-flow algorithm
  edm::Handle<reco::CaloMETCollection> caloMEtCollection;
  readEventData(evt,
                caloMEtSource_,
                caloMEtCollection,
                numWarningsCaloMEt_,
                maxNumWarnings_,
                readError,
                "Failed to access calo. MET collection");
  if (readError)
    return;

  const reco::CaloMET& caloMEt = caloMEtCollection->at(0);

  edm::Handle<reco::PFMETCollection> pfMEtCollection;
  readEventData(evt,
                pfMEtSource_,
                pfMEtCollection,
                numWarningsPFMEt_,
                maxNumWarnings_,
                readError,
                "Failed to access pf. MET collection");
  if (readError)
    return;

  const reco::PFMET& pfMEt = pfMEtCollection->at(0);

  if (!(theMuon && theTauJet && theTauJetIndex != -1))
    return;

  //-----------------------------------------------------------------------------
  // compute EWK tau analysis specific quantities
  //-----------------------------------------------------------------------------

  double dPhiMuTau = calcDeltaPhi(theMuon->phi(), theTauJet->phi());
  // double dRMuTau = calcDeltaR(theMuon->p4(), theTauJet->p4());
  double dRMuTau = fabs(ROOT::Math::VectorUtil::DeltaR(theMuon->p4(), theTauJet->p4()));
  double mMuTau = (theMuon->p4() + theTauJet->p4()).M();

  // double mtMuCaloMEt = calcMt(theMuon->px(), theMuon->px(), caloMEt.px(),
  // caloMEt.py());
  double mtMuPFMEt = calcMt(theMuon->px(), theMuon->px(), pfMEt.px(), pfMEt.py());

  // double pZetaCaloMEt = calcPzeta(theMuon->p4(), theTauJet->p4(),
  // caloMEt.px(), caloMEt.py());
  // double pZetaPFMEt = calcPzeta(theMuon->p4(), theTauJet->p4(), pfMEt.px(),
  // pfMEt.py());

  //-----------------------------------------------------------------------------
  // apply selection criteria; fill histograms
  //-----------------------------------------------------------------------------

  ++numEventsAnalyzed_;

  bool isSelected = false;
  int cutFlowStatus = -1;

  // if ( muonIsoMode_ == kAbsoluteIso){
  if (mMuTau > visMassCut_) {
    cutFlowStatus = kPassedPreselection;
  }
  if (cutFlowStatus == kPassedPreselection && (isTriggered || hltPaths_.empty())) {
    cutFlowStatus = kPassedTrigger;
  }
  if (cutFlowStatus == kPassedTrigger && (theMuon->isGlobalMuon() || theMuon->isTrackerMuon())) {
    cutFlowStatus = kPassedMuonId;
  }

  if (cutFlowStatus == kPassedMuonId && (theTauDiscrByLeadTrackFinding > 0.5) && (theTauJet->eta() < tauJetEtaCut_) &&
      (theTauJet->pt() > tauJetPtCut_)) {
    cutFlowStatus = kPassedTauLeadTrack;
  }
  if (cutFlowStatus == kPassedTauLeadTrack && theTauDiscrByLeadTrackPtCut > 0.5) {
    cutFlowStatus = kPassedTauLeadTrackPt;
    // hTauTrackIsoPt_->Fill(theTauJet->isolationPFChargedHadrCandsPtSum());
  }
  if (cutFlowStatus == kPassedTauLeadTrackPt && theTauDiscrAgainstMuons > 0.5) {
    cutFlowStatus = kPassedTauDiscrAgainstMuons;
    // hTauEcalIsoPt_->Fill(theTauJet->isolationPFGammaCandsEtSum());
  }
  if (cutFlowStatus == kPassedTauDiscrAgainstMuons && dRMuTau > deltaRCut_) {
    cutFlowStatus = kPassedDeltaR;
    // hTauDiscrAgainstMuons_->Fill(theTauDiscrAgainstMuons);

    hMuonPt_->Fill(theMuon->pt());
    hMuonEta_->Fill(theMuon->eta());
    hMuonPhi_->Fill(theMuon->phi());

    hTauJetPt_->Fill(theTauJet->pt());
    hTauJetEta_->Fill(theTauJet->eta());
    hTauJetPhi_->Fill(theTauJet->phi());

    // hTauJetCharge_->Fill(theTauJet->charge());
    if (theTauJet->signalTracks().isAvailable())
      hTauJetNumSignalTracks_->Fill(theTauJet->signalTracks().size());
    if (theTauJet->isolationTracks().isAvailable())
      hTauJetNumIsoTracks_->Fill(theTauJet->isolationTracks().size());

    hVisMass_->Fill(mMuTau);
    // hMtMuCaloMEt_->Fill(mtMuCaloMEt);
    hMtMuPFMEt_->Fill(mtMuPFMEt);
    // hPzetaCaloMEt_->Fill(pZetaCaloMEt);
    // hPzetaPFMEt_->Fill(pZetaPFMEt);
    hMuTauAcoplanarity_->Fill(dPhiMuTau);
    hMuTauDeltaR_->Fill(dRMuTau);
    // hMuTauCharge_->Fill(theMuon->charge() + theTauJet->charge());

    if (theEventVertex) {
      // hVertexChi2_->Fill(theEventVertex->normalizedChi2());
      hVertexZ_->Fill(theEventVertex->z());
      // hVertexD0_->Fill(getVertexD0(*theEventVertex, *beamSpot));
    }

    hCaloMEtPt_->Fill(caloMEt.pt());
    // hCaloMEtPhi_->Fill(caloMEt.phi());

    hPFMEtPt_->Fill(pfMEt.pt());
    // hPFMEtPhi_->Fill(pfMEt.phi());
    hMuonTrackIsoPt_->Fill(theMuonTrackIsoPt);
    hMuonEcalIsoPt_->Fill(theMuonEcalIsoPt);
    hMuonCombIsoPt_->Fill(theMuonCombIsoPt);
    // hMuonCombIsoPt_->Fill((theMuonTrackIsoPt+theMuonEcalIsoPt)/theMuon->pt());

    // std::cout<<"Rel Iso Hist =
    // "<<(theMuonTrackIsoPt+theMuonEcalIsoPt)/theMuon->pt()<<std::endl;
    hTauEcalIsoPt_->Fill(theTauJet->isolationPFGammaCandsEtSum());
    hTauTrackIsoPt_->Fill(theTauJet->isolationPFChargedHadrCandsPtSum());
    hTauDiscrAgainstMuons_->Fill(theTauDiscrAgainstMuons);
    if (theTauJet->leadTrack().isAvailable())
      hTauLeadTrackPt_->Fill(theTauJet->leadTrack()->pt());
  }

  if ((cutFlowStatus == kPassedDeltaR) && (((theMuonTrackIsoPt < muonTrackIsoCut_) && (muonIsoMode_ == kAbsoluteIso)) ||
                                           ((1 > 0) && (muonIsoMode_ == kRelativeIso)))) {
    cutFlowStatus = kPassedMuonTrackIso;
    // isSelected = true;
  }
  if (cutFlowStatus == kPassedMuonTrackIso &&
      (((theMuonEcalIsoPt < muonEcalIsoCut_) && (muonIsoMode_ == kAbsoluteIso)) ||
       ((theMuonCombIsoPt < muonCombIsoCut_) && (muonIsoMode_ == kRelativeIso)))) {
    cutFlowStatus = kPassedMuonEcalIso;
    // isSelected = true;
  }

  if (cutFlowStatus == kPassedMuonEcalIso && theTauDiscrByTrackIso > 0.5) {
    cutFlowStatus = kPassedTauTrackIso;
  }

  if (cutFlowStatus == kPassedTauTrackIso && theTauDiscrByEcalIso > 0.5) {
    cutFlowStatus = kPassedTauEcalIso;
    isSelected = true;
  }

  for (int iCut = 1; iCut <= cutFlowStatus; ++iCut) {
    hCutFlowSummary_->Fill(iCut);
  }

  for (int iCut = 1; iCut <= cutFlowStatus; ++iCut) {
    hCutFlowSummary_->Fill(iCut);
  }

  //     }

  if (isSelected) {
    hVisMassFinal_->Fill(mMuTau);
    ++numEventsSelected_;
  }
}

void EwkMuTauHistManager::finalizeHistograms() {
  edm::LogInfo("EwkMuTauHistManager") << "Filter-Statistics Summary:" << std::endl
                                      << " Events analyzed = " << numEventsAnalyzed_ << std::endl
                                      << " Events selected = " << numEventsSelected_;
  if (numEventsAnalyzed_ > 0) {
    double eff = numEventsSelected_ / (double)numEventsAnalyzed_;
    edm::LogInfo("") << "Overall efficiency = " << std::setprecision(4) << eff * 100. << " +/- " << std::setprecision(4)
                     << TMath::Sqrt(eff * (1 - eff) / numEventsAnalyzed_) * 100. << ")%";
  }
}

//-------------------------------------------------------------------------------
// common auxiliary functions used by different channels
//-------------------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TMath.h>

int getIsoMode(const std::string& isoMode_string, int& error) {
  int isoMode_int;
  if (isoMode_string == "absoluteIso") {
    isoMode_int = kAbsoluteIso;
  } else if (isoMode_string == "relativeIso") {
    isoMode_int = kRelativeIso;
  } else {
    edm::LogError("getIsoMode") << " Failed to decode isoMode string = " << isoMode_string << " !!";
    isoMode_int = kUndefinedIso;
    error = 1;
  }
  return isoMode_int;
}

double calcDeltaPhi(double phi1, double phi2) {
  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0.)
    deltaPhi = -deltaPhi;

  if (deltaPhi > TMath::Pi())
    deltaPhi = 2 * TMath::Pi() - deltaPhi;

  return deltaPhi;
}

double calcMt(double px1, double py1, double px2, double py2) {
  double pt1 = TMath::Sqrt(px1 * px1 + py1 * py1);
  double pt2 = TMath::Sqrt(px2 * px2 + py2 * py2);

  double p1Dotp2 = px1 * px2 + py1 * py2;
  double cosAlpha = p1Dotp2 / (pt1 * pt2);

  return TMath::Sqrt(2 * pt1 * pt2 * (1 - cosAlpha));
}

double calcPzeta(const reco::Candidate::LorentzVector& p1,
                 const reco::Candidate::LorentzVector& p2,
                 double pxMEt,
                 double pyMEt) {
  double cosPhi1 = cos(p1.phi());
  double sinPhi1 = sin(p1.phi());
  double cosPhi2 = cos(p2.phi());
  double sinPhi2 = sin(p2.phi());
  double zetaX = cosPhi1 + cosPhi2;
  double zetaY = sinPhi1 + sinPhi2;
  double zetaR = TMath::Sqrt(zetaX * zetaX + zetaY * zetaY);
  if (zetaR > 0.) {
    zetaX /= zetaR;
    zetaY /= zetaR;
  }

  double pxVis = p1.px() + p2.px();
  double pyVis = p1.py() + p2.py();
  double pZetaVis = pxVis * zetaX + pyVis * zetaY;

  double px = pxVis + pxMEt;
  double py = pyVis + pyMEt;
  double pZeta = px * zetaX + py * zetaY;

  return pZeta - 1.5 * pZetaVis;
}

bool passesElectronPreId(const reco::GsfElectron& electron) {
  if ((TMath::Abs(electron.eta()) < 1.479 || TMath::Abs(electron.eta()) > 1.566) &&  // cut ECAL barrel/endcap crack
      electron.deltaPhiSuperClusterTrackAtVtx() < 0.8 && electron.deltaEtaSuperClusterTrackAtVtx() < 0.01 &&
      electron.sigmaIetaIeta() < 0.03) {
    return true;
  } else {
    return false;
  }
}

bool passesElectronId(const reco::GsfElectron& electron) {
  if (passesElectronPreId(electron) && ((TMath::Abs(electron.eta()) > 1.566 &&  // electron reconstructed in ECAL
                                                                                // endcap
                                         electron.sigmaEtaEta() < 0.03 && electron.hcalOverEcal() < 0.05 &&
                                         TMath::Abs(electron.deltaEtaSuperClusterTrackAtVtx()) < 0.009 &&
                                         TMath::Abs(electron.deltaPhiSuperClusterTrackAtVtx()) < 0.7) ||
                                        (TMath::Abs(electron.eta()) < 1.479 &&  // electron reconstructed in ECAL
                                                                                // barrel
                                         electron.sigmaEtaEta() < 0.01 && electron.hcalOverEcal() < 0.12 &&
                                         TMath::Abs(electron.deltaEtaSuperClusterTrackAtVtx()) < 0.007 &&
                                         TMath::Abs(electron.deltaPhiSuperClusterTrackAtVtx()) < 0.8))) {
    return true;
  } else {
    return false;
  }
}

const reco::GsfElectron* getTheElectron(const reco::GsfElectronCollection& electrons,
                                        double electronEtaCut,
                                        double electronPtCut) {
  const reco::GsfElectron* theElectron = nullptr;

  for (reco::GsfElectronCollection::const_iterator electron = electrons.begin(); electron != electrons.end();
       ++electron) {
    if (TMath::Abs(electron->eta()) < electronEtaCut && electron->pt() > electronPtCut &&
        passesElectronPreId(*electron)) {
      if (theElectron == nullptr || electron->pt() > theElectron->pt())
        theElectron = &(*electron);
    }
  }

  return theElectron;
}

const reco::Muon* getTheMuon(const reco::MuonCollection& muons, double muonEtaCut, double muonPtCut) {
  const reco::Muon* theMuon = nullptr;

  for (reco::MuonCollection::const_iterator muon = muons.begin(); muon != muons.end(); ++muon) {
    if (TMath::Abs(muon->eta()) < muonEtaCut && muon->pt() > muonPtCut) {
      if (theMuon == nullptr || muon->pt() > theMuon->pt())
        theMuon = &(*muon);
    }
  }

  return theMuon;
}

const reco::PFTau* getTheTauJet(const reco::PFTauCollection& tauJets,
                                double tauJetEtaCut,
                                double tauJetPtCut,
                                int& theTauJetIndex) {
  const reco::PFTau* theTauJet = nullptr;
  theTauJetIndex = -1;

  int numTauJets = tauJets.size();
  for (int iTauJet = 0; iTauJet < numTauJets; ++iTauJet) {
    const reco::PFTau& tauJet = tauJets.at(iTauJet);

    if (fabs(tauJet.eta()) < tauJetEtaCut && tauJet.pt() > tauJetPtCut) {
      if (theTauJet == nullptr || tauJet.pt() > theTauJet->pt()) {
        theTauJet = &tauJet;
        theTauJetIndex = iTauJet;
      }
    }
  }

  return theTauJet;
}

double getVertexD0(const reco::Vertex& vertex, const reco::BeamSpot& beamSpot) {
  double dX = vertex.x() - beamSpot.x0();
  double dY = vertex.y() - beamSpot.y0();
  return TMath::Sqrt(dX * dX + dY * dY);
}
