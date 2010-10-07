#include "DQMOffline/PFTau/plugins/PFTauDQM.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <TMath.h>

#include <iostream>
#include <iomanip>

PFTauDQM::PFTauDQM(const edm::ParameterSet& cfg)
  : dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")),
    dqmError_(0),
    numWarningsTriggerResults_(0),
    numWarningsHLTpath_(0),
    numWarningsVertex_(0),
    numWarningsTauJet_(0),
    numWarningsTauDiscrByLeadTrackFinding_(0),
    numWarningsTauDiscrByLeadTrackPtCut_(0),
    numWarningsTauDiscrByTrackIso_(0),
    numWarningsTauDiscrByEcalIso_(0),
    numWarningsTauDiscrAgainstElectrons_(0),
    numWarningsTauDiscrAgainstMuons_(0),
    numWarningsTauDiscrTaNC_(0),
    numWarningsTauDiscrTaNCWorkingPoint_(0),
    numWarningsTauDiscrHPSWorkingPoint_(0),
    hTauDiscrTaNC_(0),
    //TODO add HPS here
    hJetPt_(0),
    hJetEta_(0),
    hJetPhi_(0),
    hTauJetDiscrPassedPt_(0), hTauJetDiscrPassedEta_(0), hTauJetDiscrPassedPhi_(0),
    hTauTaNCDiscrPassedPt_(0), hTauTaNCDiscrPassedEta_(0), hTauTaNCDiscrPassedPhi_(0),
    hTauHPSDiscrPassedPt_(0), hTauHPSDiscrPassedEta_(0), hTauHPSDiscrPassedPhi_(0),
    numEventsAnalyzed_(0)
{
  if ( !edm::Service<DQMStore>().isAvailable() ) {
    edm::LogError ("PFTauDQM") << " Failed to access dqmStore --> histograms will NEITHER be booked NOR filled !!";
    dqmError_ = 1;
    return;
  }

  dqmStore_ = &(*edm::Service<DQMStore>());

  maxNumWarnings_ = cfg.exists("maxNumWarnings") ? cfg.getParameter<int>("maxNumWarnings") : 1;

  triggerResultsSource_ = cfg.getParameter<edm::InputTag>("triggerResultsSource");
  vertexSource_ = cfg.getParameter<edm::InputTag>("vertexSource");
  tauJetSource_ = cfg.getParameter<edm::InputTag>("tauJetSource");
  hpsTauJetSource_ = cfg.getParameter<edm::InputTag>("hpsTauJetSource");

  tauDiscrByLeadTrackFinding_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackFinding");
  tauDiscrByLeadTrackPtCut_ = cfg.getParameter<edm::InputTag>("tauDiscrByLeadTrackPtCut");
  tauDiscrByTrackIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByTrackIso");
  tauDiscrByEcalIso_ = cfg.getParameter<edm::InputTag>("tauDiscrByEcalIso");
  tauDiscrAgainstElectrons_ = cfg.getParameter<edm::InputTag>("tauDiscrAgainstElectrons");
  tauDiscrAgainstMuons_ = cfg.getParameter<edm::InputTag>("tauDiscrAgainstMuons");
  tauDiscrTaNC_ = cfg.getParameter<edm::InputTag>("tauDiscrTaNC");
  tauDiscrTaNCWorkingPoint_ = cfg.getParameter<edm::InputTag>("tauDiscrTaNCWorkingPoint");
  tauDiscrHPSWorkingPoint_ = cfg.getParameter<edm::InputTag>("tauDiscrHPSWorkingPoint");
  hltPaths_ = cfg.getParameter<vstring>("hltPaths");

  tauJetPtCut_ = cfg.getParameter<double>("tauJetPtCut");
  tauJetEtaCut_ = cfg.getParameter<double>("tauJetEtaCut");
  tauJetLeadTrkDxyCut_ = cfg.getParameter<double>("tauJetLeadTrkDxyCut");
  tauJetLeadTrkDzCut_ = cfg.getParameter<double>("tauJetLeadTrkDzCut");
}

PFTauDQM::~PFTauDQM()
{
  delete hJetPt_;
  delete hJetEta_;
  delete hJetPhi_;

  delete hTauJetDiscrPassedPt_;
  delete hTauJetDiscrPassedEta_;
  delete hTauJetDiscrPassedPhi_;

  delete hTauTaNCDiscrPassedPt_;
  delete hTauTaNCDiscrPassedEta_;
  delete hTauTaNCDiscrPassedPhi_;

  delete hTauHPSDiscrPassedPt_;
  delete hTauHPSDiscrPassedEta_;
  delete hTauHPSDiscrPassedPhi_;

}  

void PFTauDQM::beginJob()
{
  dqmStore_->setCurrentFolder(dqmDirectory_);

  hNumTauJets_ = dqmStore_->book1D("NumTauJets" , "Num. #tau-Jets", 20, -0.5, 19.5);

  hJetPt_ = new TH1D("JetPt", "JetPt", 20, 0., 100.);
  hJetEta_ = new TH1D("JetEta", "JetEta", 20, -4.0, +4.0);
  hJetPhi_ = new TH1D("JetPhi", "JetPhi", 20, -TMath::Pi(), +TMath::Pi());
  
  hTauJetPt_ = dqmStore_->book1D("TauJetPt" , "P_{T}^{#tau-Jet}", 20, 0., 100.);
  hTauJetEta_ = dqmStore_->book1D("TauJetEta" , "#eta_{#tau-Jet}", 20, -4.0, +4.0);
  hTauJetPhi_ = dqmStore_->book1D("TauJetPhi" , "#phi_{#tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());

  hTauJetDiscrPassedPt_ = new TH1D("TauJetDiscrPassedPt", "TauJetDiscrPassedPt", 20, 0., 100.);
  hTauJetDiscrPassedEta_ = new TH1D("TauJetDiscrPassedEta", "TauJetDiscrPassedEta", 20, -4.0, +4.0);
  hTauJetDiscrPassedPhi_ = new TH1D("TauJetDiscrPassedPhi", "TauJetDiscrPassedPhi", 20, -TMath::Pi(), +TMath::Pi());
  
  hTauTaNCDiscrPassedPt_ = new TH1D("TauTaNCDiscrPassedPt", "TauTaNCDiscrPassedPt", 20, 0., 100.);
  hTauTaNCDiscrPassedEta_ = new TH1D("TauTaNCDiscrPassedEta", "TauTaNCDiscrPassedEta", 20, -4.0, +4.0);
  hTauTaNCDiscrPassedPhi_ = new TH1D("TauTaNCDiscrPassedPhi", "TauTaNCDiscrPassedPhi", 20, -TMath::Pi(), +TMath::Pi());

  hTauHPSDiscrPassedPt_ = new TH1D("TauHPSDiscrPassedPt", "TauHPSDiscrPassedPt", 20, 0., 100.);
  hTauHPSDiscrPassedEta_ = new TH1D("TauHPSDiscrPassedEta", "TauHPSDiscrPassedEta", 20, -4.0, +4.0);
  hTauHPSDiscrPassedPhi_ = new TH1D("TauHPSDiscrPassedPhi", "TauHPSDiscrPassedPhi", 20, -TMath::Pi(), +TMath::Pi());

  hTauJetCharge_ = dqmStore_->book1D("TauJetCharge" , "Q_{#tau-Jet}", 11, -5.5, +5.5);

  hTauLeadTrackPt_ = dqmStore_->book1D("TauLeadTrackPt" , "P_{T}^{lead. Track}", 20, 0., 50.);

  hTauJetNumSignalTracks_ = dqmStore_->book1D("TauJetNumSignalTracks" , "Num. Tau signal Cone Tracks", 20, -0.5, +19.5);
  hTauJetNumIsoTracks_ = dqmStore_->book1D("TauJetNumIsoTracks" , "Num. Tau isolation Cone Tracks", 20, -0.5, +19.5);

  hTauTrackIsoPt_ = dqmStore_->book1D("TauTrackIsoPt" , "Tau Track Iso.", 20, -0.01, 40.);
  hTauEcalIsoPt_ = dqmStore_->book1D("TauEcalIsoPt" , "Tau Ecal Iso.", 10, -0.01, 10.);

  hTauDiscrByLeadTrackFinding_ = dqmStore_->book1D("TauDiscr" , "Tau Discr. by lead. Track Finding", 2, -0.5, +1.5);
  hTauDiscrByLeadTrackPtCut_ = dqmStore_->book1D("TauDiscr" , "Tau Discr. by lead. Track P_{T} Cut", 2, -0.5, +1.5);
  hTauDiscrByTrackIso_ = dqmStore_->book1D("TauDiscr" , "Tau Discr. by Track iso.", 2, -0.5, +1.5);
  hTauDiscrByEcalIso_ = dqmStore_->book1D("TauDiscr" , "Tau Discr. by ECAL iso.", 2, -0.5, +1.5);
  hTauDiscrAgainstElectrons_ = dqmStore_->book1D("TauDiscrAgainstElectrons" , "Tau Discr. against Electrons", 2, -0.5, +1.5);
  hTauDiscrAgainstMuons_ = dqmStore_->book1D("TauDiscrAgainstMuons" , "Tau Discr. against Muons", 2, -0.5, +1.5);
  hTauDiscrTaNC_ = dqmStore_->book1D("TauDiscrTaNC" , "Combined TaNC Discr.", 20, 0., 1.);
  hTauDiscrHPS_ = dqmStore_->book1D("TauDiscrHPS" , "Tau Discr. by HPS", 2, -0.5, +1.5);
  //TODO add better HPS variable here
  hFakeRatePt_ = dqmStore_->book1D("FakeRatePt", "Tau fake-rate as function of P_{T}^{#tau-Jet}", 20, 0., 100.);
  hFakeRateEta_ = dqmStore_->book1D("FakeRateEta", "Tau fake-rate as function of #eta_{#tau-Jet}", 20, -4.0, +4.0);
  hFakeRatePhi_ = dqmStore_->book1D("FakeRatePhi", "Tau fake-rate as function of #phi_{#tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
  hTaNCFakeRatePt_ = dqmStore_->book1D("TaNCFakeRatePt",   "TaNC Tau fake-rate as function of P_{T}^{#tau-Jet}", 20, 0., 100.);
  hTaNCFakeRateEta_ = dqmStore_->book1D("TaNCFakeRateEta", "TaNC Tau fake-rate as function of #eta_{#tau-Jet}", 20, -4.0, +4.0);
  hTaNCFakeRatePhi_ = dqmStore_->book1D("TaNCFakeRatePhi", "TaNC Tau fake-rate as function of #phi_{#tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
  hHPSFakeRatePt_ = dqmStore_->book1D("HPSFakeRatePt",   "HPS Tau fake-rate as function of P_{T}^{#tau-Jet}", 20, 0., 100.);
  hHPSFakeRateEta_ = dqmStore_->book1D("HPSFakeRateEta", "HPS Tau fake-rate as function of #eta_{#tau-Jet}", 20, -4.0, +4.0);
  hHPSFakeRatePhi_ = dqmStore_->book1D("HPSFakeRatePhi", "HPS Tau fake-rate as function of #phi_{#tau-Jet}", 20, -TMath::Pi(), +TMath::Pi());
}

template<typename T>
void readEventData(const edm::Event& evt, const edm::InputTag& src, edm::Handle<T>& handle, long& numWarnings, int maxNumWarnings, 
		   bool& error, const char* errorMessage)
{
  if ( !evt.getByLabel(src, handle) ) {
    if ( numWarnings < maxNumWarnings || maxNumWarnings == -1 )
      edm::LogWarning ("readEventData") << errorMessage << " !!";
    ++numWarnings;
    error = true;
  }
}

void PFTauDQM::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( dqmError_ ) return;

  //-----------------------------------------------------------------------------
  // access event-level information
  //-----------------------------------------------------------------------------

  bool readError = false;

//--- get decision of high-level trigger for the event
  edm::Handle<edm::TriggerResults> hltDecision;
  readEventData(evt, triggerResultsSource_, hltDecision, numWarningsTriggerResults_, maxNumWarnings_, 
		readError, "Failed to access Trigger results");
  if ( readError ) return;
  
  const edm::TriggerNames & triggerNames = evt.triggerNames(*hltDecision);
   
  bool isTriggered = false;
  for ( vstring::const_iterator hltPath = hltPaths_.begin();
	hltPath != hltPaths_.end(); ++hltPath ) {
    unsigned int index = triggerNames.triggerIndex(*hltPath);
    if ( index < triggerNames.size() ) {
      if ( hltDecision->accept(index) ) isTriggered = true;
    } else {
      if ( numWarningsHLTpath_ < maxNumWarnings_ || maxNumWarnings_ == -1 ) 
	edm::LogWarning ("EwkElecTauHistManager") << " Undefined HLT path = " << (*hltPath) << " !!";
      ++numWarningsHLTpath_;
      continue;
    }
  }
  
//--- get reconstructed primary event vertex of the event
//   (take as "the" primary event vertex the first entry in the collection
//    of vertex objects, corresponding to the vertex associated to the highest Pt sum of tracks)
  edm::Handle<reco::VertexCollection> vertexCollection;
  readEventData(evt, vertexSource_, vertexCollection, numWarningsVertex_, maxNumWarnings_,
		readError, "Failed to access Vertex collection");
  if ( readError ) return;

  const reco::Vertex* theEventVertex = ( vertexCollection->size() > 0 ) ? &(vertexCollection->at(0)) : 0;

//--- get collections of reconstructed tau-jets from the event
  edm::Handle<reco::PFTauCollection> tauJets;
  readEventData(evt, tauJetSource_, tauJets, numWarningsTauJet_, maxNumWarnings_,
		readError, "Failed to access Tau-jet collection");
  if ( readError ) return;

  edm::Handle<reco::PFTauCollection> hpsTauJets;
  readEventData(evt, hpsTauJetSource_, hpsTauJets, numWarningsHPSTauJet_, maxNumWarnings_,
		readError, "Failed to access HPS Tau-jet collection");
  if ( readError ) return;


//--- get collections of tau-jet discriminators for those tau-jets
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackFinding;
  readEventData(evt, tauDiscrByLeadTrackFinding_, tauDiscrByLeadTrackFinding, numWarningsTauDiscrByLeadTrackFinding_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators by leading Track finding");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByLeadTrackPtCut;
  readEventData(evt, tauDiscrByLeadTrackPtCut_, tauDiscrByLeadTrackPtCut, numWarningsTauDiscrByLeadTrackPtCut_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators by leading Track Pt cut");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByTrackIso;
  readEventData(evt, tauDiscrByTrackIso_, tauDiscrByTrackIso, numWarningsTauDiscrByTrackIso_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators by Track isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrByEcalIso;
  readEventData(evt, tauDiscrByTrackIso_, tauDiscrByEcalIso, numWarningsTauDiscrByEcalIso_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators by ECAL isolation");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrAgainstElectrons;
  readEventData(evt, tauDiscrAgainstElectrons_, tauDiscrAgainstElectrons, numWarningsTauDiscrAgainstElectrons_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators against Electrons");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrAgainstMuons;
  readEventData(evt, tauDiscrAgainstMuons_, tauDiscrAgainstMuons, numWarningsTauDiscrAgainstMuons_, maxNumWarnings_,
		readError, "Failed to access collection of pf. Tau discriminators against Muons");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrTaNC;
  readEventData(evt, tauDiscrTaNC_, tauDiscrTaNC, numWarningsTauDiscrTaNC_, maxNumWarnings_,
		readError, "Failed to access collection of TaNC combined discriminator");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrTaNCWorkingPoint;
  readEventData(evt, tauDiscrTaNCWorkingPoint_, tauDiscrTaNCWorkingPoint, numWarningsTauDiscrTaNCWorkingPoint_, maxNumWarnings_,
		readError, "Failed to access collection of TaNC Working Point discriminator");
  edm::Handle<reco::PFTauDiscriminator> tauDiscrHPSWorkingPoint;
  readEventData(evt, tauDiscrHPSWorkingPoint_, tauDiscrHPSWorkingPoint, numWarningsTauDiscrHPSWorkingPoint_, maxNumWarnings_,
		readError, "Failed to access collection of HPS Working Point discriminator");


  if ( readError ) return;

  ++numEventsAnalyzed_;

  if ( hltPaths_.size() > 0 && !isTriggered) return;

  unsigned numTauJets = tauJets->size();
  unsigned numTauJets_selected = 0;
  for ( unsigned iTauJet = 0; iTauJet < numTauJets; ++iTauJet ) {
    reco::PFTauRef tauJet(tauJets, iTauJet);

    if ( !(           tauJet->pt()   > tauJetPtCut_ ) ) continue;
    if ( !(TMath::Abs(tauJet->eta()) < tauJetEtaCut_) ) continue;

    hJetPt_->Fill(tauJet->pt());
    hJetEta_->Fill(tauJet->eta());
    hJetPhi_->Fill(tauJet->phi());

    if ( tauJetLeadTrkDxyCut_ != -1. || tauJetLeadTrkDzCut_ != -1. ) {
      const reco::Track* tauJetLeadTrk = ( tauJet->leadPFChargedHadrCand().isAvailable() ) ?
	tauJet->leadPFChargedHadrCand()->trackRef().get() : 0;

      if ( !(theEventVertex && tauJetLeadTrk) ) continue;

      double dXY = tauJetLeadTrk->dxy(theEventVertex->position());
      if ( tauJetLeadTrkDxyCut_ != -1. && !(TMath::Abs(dXY) < tauJetLeadTrkDxyCut_) ) continue;

      double dZ = tauJetLeadTrk->dz(theEventVertex->position());
      if ( tauJetLeadTrkDzCut_  != -1. && !(TMath::Abs(dZ)  < tauJetLeadTrkDzCut_ ) ) continue;
    }

    ++numTauJets_selected;
   
    hTauJetPt_->Fill(tauJet->pt());
    hTauJetEta_->Fill(tauJet->eta());
    hTauJetPhi_->Fill(tauJet->phi());
    
    bool allTauIdDiscrPassed = 
      ( (*tauDiscrByLeadTrackFinding)[tauJet] > 0.5 && 
	(*tauDiscrByLeadTrackPtCut)[tauJet]   > 0.5 && 
	(*tauDiscrByTrackIso)[tauJet]         > 0.5 && 
	(*tauDiscrByEcalIso)[tauJet]          > 0.5 && 
	(*tauDiscrAgainstElectrons)[tauJet]   > 0.5 &&
	(*tauDiscrAgainstMuons)[tauJet]       > 0.5 );
    if ( allTauIdDiscrPassed ) {
      hTauJetDiscrPassedPt_->Fill(tauJet->pt());
      hTauJetDiscrPassedEta_->Fill(tauJet->eta());
      hTauJetDiscrPassedPhi_->Fill(tauJet->phi());
    }
    if ( (*tauDiscrTaNCWorkingPoint)[tauJet]   > 0.5 &&
 	 (*tauDiscrAgainstElectrons)[tauJet]   > 0.5 &&
	 (*tauDiscrAgainstMuons)[tauJet]       > 0.5    ) {
      hTauTaNCDiscrPassedPt_->Fill(tauJet->pt());
      hTauTaNCDiscrPassedEta_->Fill(tauJet->eta());
      hTauTaNCDiscrPassedPhi_->Fill(tauJet->phi());
    }
    hTauJetCharge_->Fill(tauJet->charge());

    if ( tauJet->leadPFChargedHadrCand().isAvailable() ) hTauLeadTrackPt_->Fill(tauJet->leadPFChargedHadrCand()->pt());

    if ( tauJet->signalTracks().isAvailable()    ) hTauJetNumSignalTracks_->Fill(tauJet->signalTracks().size());
    if ( tauJet->isolationTracks().isAvailable() ) hTauJetNumIsoTracks_->Fill(tauJet->isolationTracks().size());

    hTauTrackIsoPt_->Fill(tauJet->isolationPFChargedHadrCandsPtSum());
    hTauEcalIsoPt_->Fill(tauJet->isolationPFGammaCandsEtSum());

    hTauDiscrByLeadTrackFinding_->Fill((*tauDiscrByLeadTrackFinding)[tauJet]);
    hTauDiscrByLeadTrackPtCut_->Fill((*tauDiscrByLeadTrackPtCut)[tauJet]);
    hTauDiscrByTrackIso_->Fill((*tauDiscrByTrackIso)[tauJet]);
    hTauDiscrByEcalIso_->Fill((*tauDiscrByEcalIso)[tauJet]);
    hTauDiscrAgainstElectrons_->Fill((*tauDiscrAgainstElectrons)[tauJet]);
    hTauDiscrAgainstMuons_->Fill((*tauDiscrAgainstMuons)[tauJet]);
    hTauDiscrTaNC_->Fill((*tauDiscrTaNC)[tauJet]);
  }
  hNumTauJets_->Fill(numTauJets_selected);

  // as soon as HPS and others share the same tau jet collection this code dublication will be cleaned up.
  // Since this will (hopefully) soon this done in the meantime.
  // also it is assumed that SC, TaNC and HPS share the same underlying jet collection (i.e. same denominator histograms)
  unsigned numHPSTauJets = hpsTauJets->size();
  for ( unsigned iTauJet = 0; iTauJet < numHPSTauJets; ++iTauJet ) {
    reco::PFTauRef tauJet(hpsTauJets, iTauJet);

    if ( !(           tauJet->pt()   > tauJetPtCut_ ) ) continue;
    if ( !(TMath::Abs(tauJet->eta()) < tauJetEtaCut_) ) continue;

    if ( tauJetLeadTrkDxyCut_ != -1. || tauJetLeadTrkDzCut_ != -1. ) {
      const reco::Track* tauJetLeadTrk = ( tauJet->leadPFChargedHadrCand().isAvailable() ) ?
	tauJet->leadPFChargedHadrCand()->trackRef().get() : 0;

      if ( !(theEventVertex && tauJetLeadTrk) ) continue;

      double dXY = tauJetLeadTrk->dxy(theEventVertex->position());
      if ( tauJetLeadTrkDxyCut_ != -1. && !(TMath::Abs(dXY) < tauJetLeadTrkDxyCut_) ) continue;

      double dZ = tauJetLeadTrk->dz(theEventVertex->position());
      if ( tauJetLeadTrkDzCut_  != -1. && !(TMath::Abs(dZ)  < tauJetLeadTrkDzCut_ ) ) continue;
    }
    if ( (*tauDiscrHPSWorkingPoint)[tauJet]    > 0.5 ) {
      hTauHPSDiscrPassedPt_->Fill(tauJet->pt());
      hTauHPSDiscrPassedEta_->Fill(tauJet->eta());
      hTauHPSDiscrPassedPhi_->Fill(tauJet->phi());
    }
    hTauDiscrHPS_->Fill((*tauDiscrHPSWorkingPoint)[tauJet]);
  }
}

void compFakeRate(MonitorElement* fakerate, TH1* numerator, TH1* denominator)
{
  if ( !  numerator->GetSumw2() )   numerator->Sumw2();
  if ( !denominator->GetSumw2() ) denominator->Sumw2();
  fakerate->getTH1()->Divide(numerator, denominator);
}

void PFTauDQM::endJob()
{
  compFakeRate(hFakeRatePt_,  hTauJetDiscrPassedPt_,  hJetPt_ );
  compFakeRate(hFakeRateEta_, hTauJetDiscrPassedEta_, hJetEta_);
  compFakeRate(hFakeRatePhi_, hTauJetDiscrPassedPhi_, hJetPhi_);

  compFakeRate(hTaNCFakeRatePt_,  hTauTaNCDiscrPassedPt_,  hJetPt_ );
  compFakeRate(hTaNCFakeRateEta_, hTauTaNCDiscrPassedEta_, hJetEta_);
  compFakeRate(hTaNCFakeRatePhi_, hTauTaNCDiscrPassedPhi_, hJetPhi_);

  compFakeRate(hHPSFakeRatePt_,  hTauHPSDiscrPassedPt_,  hJetPt_ );
  compFakeRate(hHPSFakeRateEta_, hTauHPSDiscrPassedEta_, hJetEta_);
  compFakeRate(hHPSFakeRatePhi_, hTauHPSDiscrPassedPhi_, hJetPhi_);
    
  edm::LogInfo ("PFTauDQM") 
    << "Filter-Statistics Summary:" << std::endl
    << " Events analyzed = " << numEventsAnalyzed_ << std::endl;
}
/*
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFTauDQM);
 */
