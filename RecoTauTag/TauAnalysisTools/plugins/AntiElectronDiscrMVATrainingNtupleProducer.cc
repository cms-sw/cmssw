#include "RecoTauTag/TauAnalysisTools/plugins/AntiElectronDiscrMVATrainingNtupleProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

AntiElectronDiscrMVATrainingNtupleProducer::AntiElectronDiscrMVATrainingNtupleProducer(const edm::ParameterSet& cfg)
{ 
  srcPFTaus_ = cfg.getParameter<edm::InputTag>("srcPFTaus");
  srcGsfElectrons_ = cfg.getParameter<edm::InputTag>("srcGsfElectrons");
  srcPrimaryVertex_ = cfg.getParameter<edm::InputTag>("srcPrimaryVertex");
  srcGenElectrons_ = cfg.getParameter<edm::InputTag>("srcGenElectrons");
  srcGenTaus_ = cfg.getParameter<edm::InputTag>("srcGenTaus");

  edm::ParameterSet tauIdDiscriminators = cfg.getParameter<edm::ParameterSet>("tauIdDiscriminators");
  typedef std::vector<std::string> vstring;
  vstring tauIdDiscriminatorNames = tauIdDiscriminators.getParameterNamesForType<edm::InputTag>();
  for ( vstring::const_iterator name = tauIdDiscriminatorNames.begin();
	name != tauIdDiscriminatorNames.end(); ++name ) {
    edm::InputTag src = tauIdDiscriminators.getParameter<edm::InputTag>(*name);
    tauIdDiscrEntries_.push_back(tauIdDiscrEntryType(*name, src));
  }

  srcWeights_ = cfg.getParameter<vInputTag>("srcWeights");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
} 

void AntiElectronDiscrMVATrainingNtupleProducer::beginJob()
{ 
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("tree", "AntiEMVA tree");

  // counters
  tree_->Branch("run", &run_, "run/l");
  tree_->Branch("event", &event_, "event/l");
  tree_->Branch("lumi", &lumi_, "lumi/l");
  tree_->Branch("NumPV", &NumPV_, "NumPV/I");
  tree_->Branch("NumGsfEle", &NumGsfEle_, "NumGsfEle/I");
  tree_->Branch("NumGenEle", &NumGenEle_, "NumGenEle/I");
  tree_->Branch("NumPFTaus", &NumPFTaus_, "NumPFTaus/I");
  tree_->Branch("NumGenHad", &NumGenHad_, "NumGenHad/I");

  // GsfElectron variables
  tree_->Branch("Elec_GenEleMatch", &Elec_GenEleMatch_, "Elec_GenEleMatch/I");
  tree_->Branch("Elec_GenHadMatch", &Elec_GenHadMatch_, "Elec_GenHadMatch/I");
  tree_->Branch("Elec_AbsEta", &Elec_AbsEta_, "Elec_AbsEta/F");
  tree_->Branch("Elec_Pt", &Elec_Pt_, "Elec_Pt/F");
  tree_->Branch("Elec_HasSC", &Elec_HasSC_, "Elec_HasSC/I");
  tree_->Branch("Elec_PFMvaOutput", &Elec_PFMvaOutput_, "Elec_PFMvaOutput/F");
  tree_->Branch("Elec_Ee", &Elec_Ee_, "Elec_Ee/F");
  tree_->Branch("Elec_Egamma", &Elec_Egamma_, "Elec_Egamma/F");
  tree_->Branch("Elec_Pin", &Elec_Pin_, "Elec_Pin/F");
  tree_->Branch("Elec_Pout", &Elec_Pout_, "Elec_Pout/F");
  tree_->Branch("Elec_EtotOverPin", &Elec_EtotOverPin_, "Elec_EtotOverPin/F");
  tree_->Branch("Elec_EeOverPout", &Elec_EeOverPout_, "Elec_EeOverPout/F");
  tree_->Branch("Elec_EgammaOverPdif", &Elec_EgammaOverPdif_, "Elec_EgammaOverPdif/F");
  tree_->Branch("Elec_EarlyBrem", &Elec_EarlyBrem_, "Elec_EarlyBrem/I");
  tree_->Branch("Elec_LateBrem", &Elec_LateBrem_, "Elec_LateBrem/I");
  tree_->Branch("Elec_Logsihih", &Elec_Logsihih_, "Elec_Logsihih/F");
  tree_->Branch("Elec_DeltaEta", &Elec_DeltaEta_, "Elec_DeltaEta/F");
  tree_->Branch("Elec_HoHplusE", &Elec_HoHplusE_, "Elec_HoHplusE/F");
  tree_->Branch("Elec_Fbrem", &Elec_Fbrem_, "Elec_Fbrem/F");
  tree_->Branch("Elec_HasKF", &Elec_HasKF_, "Elec_HasKF/I");
  tree_->Branch("Elec_Chi2KF",&Elec_Chi2KF_,"Elec_Chi2KF/F");
  tree_->Branch("Elec_KFNumHits", &Elec_KFNumHits_, "Elec_KFNumHits/I");
  tree_->Branch("Elec_KFNumPixelHits", &Elec_KFNumPixelHits_, "Elec_KFNumPixelHits/I");
  tree_->Branch("Elec_KFNumStripHits", &Elec_KFNumStripHits_, "Elec_KFNumStripHits/I");
  tree_->Branch("Elec_KFTrackResol", &Elec_KFTrackResol_, "Elec_KFTrackResol/F");
  tree_->Branch("Elec_KFTracklnPt", &Elec_KFTracklnPt_, "Elec_KFTracklnPt/F");
  tree_->Branch("Elec_KFTrackEta", &Elec_KFTrackEta_, "Elec_KFTrackEta/F");
  tree_->Branch("Elec_HasGSF", &Elec_HasGSF_, "Elec_HasGSF/I");
  tree_->Branch("Elec_Chi2GSF", &Elec_Chi2GSF_, "Elec_Chi2GSF/F");
  tree_->Branch("Elec_GSFNumHits", &Elec_GSFNumHits_, "Elec_GSFNumHits/I");
  tree_->Branch("Elec_GSFNumPixelHits", &Elec_GSFNumPixelHits_, "Elec_GSFNumPixelHits/I");
  tree_->Branch("Elec_GSFNumStripHits", &Elec_GSFNumStripHits_, "Elec_GSFNumStripHits/I");
  tree_->Branch("Elec_GSFTrackResol", &Elec_GSFTrackResol_, "Elec_GSFTrackResol/F");
  tree_->Branch("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_, "Elec_GSFTracklnPt/F");
  tree_->Branch("Elec_GSFTrackEta", &Elec_GSFTrackEta_, "Elec_GSFTrackEta/F");
 
  tree_->Branch("ElecVeto_N", &ElecVeto_N_, "ElecVeto_N/I");
  tree_->Branch("ElecVeto_Pt", &ElecVeto_Pt_, "ElecVeto_Pt/F");
  tree_->Branch("ElecVeto_Eta", &ElecVeto_Eta_, "ElecVeto_Eta/F");
  tree_->Branch("ElecVeto_Phi", &ElecVeto_Phi_, "ElecVeto_Phi/F");

  // PFTau variables
  tree_->Branch("Tau_GsfEleMatch", &Tau_GsfEleMatch_, "Tau_GsfEleMatch/I");
  tree_->Branch("Tau_GenEleMatch", &Tau_GenEleMatch_, "Tau_GenEleMatch/I");
  tree_->Branch("Tau_GenHadMatch", &Tau_GenHadMatch_, "Tau_GenHadMatch/I");
  tree_->Branch("Tau_Eta", &Tau_Eta_, "Tau_Eta/F");
  tree_->Branch("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_, "Tau_EtaAtEcalEntrance/F");
  tree_->Branch("Tau_PhiAtEcalEntrance", &Tau_PhiAtEcalEntrance_, "Tau_PhiAtEcalEntrance/F");
  tree_->Branch("Tau_EtaAtEcalEntranceEcalEnWeighted", &Tau_EtaAtEcalEntranceEcalEnWeighted_, "Tau_EtaAtEcalEntranceEcalEnWeighted/F");
  tree_->Branch("Tau_PhiAtEcalEntranceEcalEnWeighted", &Tau_PhiAtEcalEntranceEcalEnWeighted_, "Tau_PhiAtEcalEntranceEcalEnWeighted/F");
  tree_->Branch("Tau_LeadNeutralPFCandEtaAtEcalEntrance", &Tau_LeadNeutralPFCandEtaAtEcalEntrance_, "Tau_LeadNeutralPFCandEtaAtEcalEntrance/F");
  tree_->Branch("Tau_LeadNeutralPFCandPhiAtEcalEntrance", &Tau_LeadNeutralPFCandPhiAtEcalEntrance_, "Tau_LeadNeutralPFCandPhiAtEcalEntrance/F");
  tree_->Branch("Tau_LeadNeutralPFCandPt", &Tau_LeadNeutralPFCandPt_, "Tau_LeadNeutralPFCandPt/F");
  tree_->Branch("Tau_LeadChargedPFCandEtaAtEcalEntrance", &Tau_LeadChargedPFCandEtaAtEcalEntrance_, "Tau_LeadChargedPFCandEtaAtEcalEntrance/F");
  tree_->Branch("Tau_LeadChargedPFCandPhiAtEcalEntrance", &Tau_LeadChargedPFCandPhiAtEcalEntrance_, "Tau_LeadChargedPFCandPhiAtEcalEntrance/F");
  tree_->Branch("Tau_LeadChargedPFCandPt", &Tau_LeadChargedPFCandPt_, "Tau_LeadChargedPFCandPt/F");
  tree_->Branch("Tau_Phi", &Tau_Phi_, "Tau_Phi/F");
  tree_->Branch("Tau_Pt", &Tau_Pt_, "Tau_Pt/F");
  tree_->Branch("Tau_LeadHadronPt", &Tau_LeadHadronPt_, "Tau_LeadHadronPt/F");
  tree_->Branch("Tau_HasGsf", &Tau_HasGsf_, "Tau_HasGsf/I");
  tree_->Branch("Tau_GSFChi2", &Tau_GSFChi2_, "Tau_GSFChi2/F");
  tree_->Branch("Tau_GSFNumHits", &Tau_GSFNumHits_, "Tau_GSFNumHits/I");
  tree_->Branch("Tau_GSFNumPixelHits", &Tau_GSFNumPixelHits_, "Tau_GSFNumPixelHits/I");
  tree_->Branch("Tau_GSFNumStripHits", &Tau_GSFNumStripHits_, "Tau_GSFNumStripHits/I");
  tree_->Branch("Tau_GSFTrackResol", &Tau_GSFTrackResol_, "Tau_GSFTrackResol/F");
  tree_->Branch("Tau_GSFTracklnPt", &Tau_GSFTracklnPt_, "Tau_GSFTracklnPt/F");
  tree_->Branch("Tau_GSFTrackEta", &Tau_GSFTrackEta_, "Tau_GSFTrackEta/F");
  tree_->Branch("Tau_HasKF", &Tau_HasKF_, "Tau_HasKF/I");
  tree_->Branch("Tau_KFChi2", &Tau_KFChi2_, "Tau_KFChi2/F");
  tree_->Branch("Tau_KFNumHits", &Tau_KFNumHits_, "Tau_KFNumHits/I");
  tree_->Branch("Tau_KFNumPixelHits", &Tau_KFNumPixelHits_, "Tau_KFNumPixelHits/I");
  tree_->Branch("Tau_KFNumStripHits", &Tau_KFNumStripHits_, "Tau_KFNumStripHits/I");
  tree_->Branch("Tau_KFTrackResol", &Tau_KFTrackResol_, "Tau_KFTrackResol/F");
  tree_->Branch("Tau_KFTracklnPt", &Tau_KFTracklnPt_, "Tau_KFTracklnPt/F");
  tree_->Branch("Tau_KFTrackEta", &Tau_KFTrackEta_, "Tau_KFTrackEta/F");
  tree_->Branch("Tau_EmFraction", &Tau_EmFraction_, "Tau_EmFraction/F");
  tree_->Branch("Tau_NumChargedCands", &Tau_NumChargedCands_, "Tau_NumChargedCands/I");
  tree_->Branch("Tau_NumGammaCands", &Tau_NumGammaCands_, "Tau_NumGammaCands/I");
  tree_->Branch("Tau_HadrHoP", &Tau_HadrHoP_, "Tau_HadrHoP/F");
  tree_->Branch("Tau_HadrEoP", &Tau_HadrEoP_, "Tau_HadrEoP/F");
  tree_->Branch("Tau_VisMass", &Tau_VisMass_, "Tau_VisMass/F");
  tree_->Branch("Tau_GammaEtaMom", &Tau_GammaEtaMom_, "Tau_GammaEtaMom/F");
  tree_->Branch("Tau_GammaPhiMom", &Tau_GammaPhiMom_, "Tau_GammaPhiMom/F");
  tree_->Branch("Tau_GammaEnFrac", &Tau_GammaEnFrac_, "Tau_GammaEnFrac/F");
  tree_->Branch("Tau_HadrMva", &Tau_HadrMva_, "Tau_HadrMva/F");
  for ( std::vector<tauIdDiscrEntryType>::iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
    tree_->Branch(Form("Tau_%s", tauIdDiscriminator->branchName_.data()), &tauIdDiscriminator->value_, Form("Tau_%s/F", tauIdDiscriminator->branchName_.data()));
  }
  tree_->Branch("Tau_DecayMode", &Tau_DecayMode_, "Tau_DecayMode/I");
  tree_->Branch("Tau_MatchElePassVeto", &Tau_MatchElePassVeto_, "Tau_MatchElePassVeto/I");
  tree_->Branch("Tau_VtxZ", &Tau_VtxZ_, "Tau_VtxZ/F");
  tree_->Branch("Tau_zImpact", &Tau_zImpact_, "Tau_zImpact/F");

  tree_->Branch("evtWeight", &evtWeight_, "evtWeight/F");
}

void AntiElectronDiscrMVATrainingNtupleProducer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  run_ = evt.run();
  event_ = (evt.eventAuxiliary()).event();
  lumi_ = evt.luminosityBlock();
  
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  evt.getByLabel(srcGsfElectrons_, gsfElectrons);

  edm::Handle<reco::PFTauCollection> pfTaus;
  evt.getByLabel(srcPFTaus_, pfTaus);

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> genElectrons;
  evt.getByLabel(srcGenElectrons_, genElectrons);

  edm::Handle<CandidateView> genTaus;
  evt.getByLabel(srcGenTaus_, genTaus);

  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcPrimaryVertex_, vertices);
  NumPV_ = vertices->size();

  evtWeight_ = 1.0;
  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    evtWeight_ *= (*weight);
  }

  //-----------------------------------------------------------------------------
  // PFTau variables
  //-----------------------------------------------------------------------------

  NumPFTaus_ = 0;
  NumGsfEle_ = 0;

  size_t numPFTaus = pfTaus->size();
  for ( size_t idxPFTau = 0; idxPFTau < numPFTaus; ++idxPFTau ) {
    reco::PFTauRef pfTau(pfTaus, idxPFTau);

    Tau_GsfEleMatch_ = 0;
    const reco::GsfElectron* matchedElectron = 0;
    for ( reco::GsfElectronCollection::const_iterator gsfElectron = gsfElectrons->begin();
	  gsfElectron != gsfElectrons->end(); ++gsfElectron ) {
      if ( deltaR(pfTau->p4(), gsfElectron->p4()) < 0.3 && gsfElectron->pt() > 10. && (matchedElectron == 0 || gsfElectron->pt() > matchedElectron->pt()) ) {
	matchedElectron = &(*gsfElectron);
	Tau_GsfEleMatch_ = 1;
      }
      ++NumGsfEle_;
    } 

    Tau_EtaAtEcalEntrance_ = -99;
    Tau_PhiAtEcalEntrance_ = -99;
    Tau_EtaAtEcalEntranceEcalEnWeighted_ = -99;
    Tau_PhiAtEcalEntranceEcalEnWeighted_ = -99;
    Tau_LeadNeutralPFCandEtaAtEcalEntrance_ = -99;
    Tau_LeadNeutralPFCandPhiAtEcalEntrance_ = -99;
    Tau_LeadNeutralPFCandPt_ = -99;
    Tau_LeadChargedPFCandEtaAtEcalEntrance_ = -99;
    Tau_LeadChargedPFCandPhiAtEcalEntrance_ = -99;
    Tau_LeadChargedPFCandPt_ = -99;
    Tau_Eta_ = -99;
    Tau_Phi_ = -99;
    Tau_Pt_ = -99;
    Tau_LeadHadronPt_ = -99;
    Tau_HasGsf_ = -99;
    Tau_GSFChi2_ = -99;
    Tau_GSFNumHits_ = -99;
    Tau_GSFNumPixelHits_ = -99;
    Tau_GSFNumStripHits_ = -99;
    Tau_GSFTrackResol_ = -99;
    Tau_GSFTracklnPt_ = -99;
    Tau_GSFTrackEta_ = -99;
    Tau_HasKF_ = -99;
    Tau_KFChi2_ = -99;
    Tau_KFNumHits_ = -99;
    Tau_KFNumPixelHits_ = -99;
    Tau_KFNumStripHits_ = -99;
    Tau_KFTrackResol_ = -99;
    Tau_KFTracklnPt_ = -99;
    Tau_KFTrackEta_ = -99;
    Tau_EmFraction_ = -99;
    Tau_NumChargedCands_ = -99;
    Tau_NumGammaCands_ = -99;
    Tau_HadrHoP_ = -99;
    Tau_HadrEoP_ = -99;
    Tau_VisMass_ = -99;
    Tau_GammaEtaMom_ = -99;
    Tau_GammaPhiMom_ = -99;
    Tau_GammaEnFrac_ = -99;
    Tau_HadrMva_ = -99;
    Tau_MatchElePassVeto_ = -99;
    Tau_DecayMode_ = -99;
    Tau_MatchElePassVeto_ = -99;
    Tau_VtxZ_ = -99;
    Tau_zImpact_ = -99;

    // Matchings
    Tau_GenEleMatch_ = 0;
    Tau_GenHadMatch_ = 0;

    NumGenEle_ = 0;
    NumGenHad_ = 0;
    
    Elec_AbsEta_ = -99;
    Elec_Pt_ = -99;
    Elec_PFMvaOutput_ = -99;
    Elec_EarlyBrem_ =  -99;
    Elec_LateBrem_=  -99;
    Elec_Logsihih_ =  -99;
    Elec_DeltaEta_ = -99;
    Elec_Fbrem_ =  -99;
    // Variables related to the SC
    Elec_HasSC_ = -99;
    Elec_Ee_ = -99;
    Elec_Egamma_ = -99;
    Elec_Pin_ = -99;
    Elec_Pout_ = -99;
    Elec_EtotOverPin_ = -99;
    Elec_EeOverPout_ = -99;
    Elec_EgammaOverPdif_ = -99;
    Elec_HoHplusE_ = -99;
    Elec_HasKF_ = -99;
    Elec_Chi2KF_ = -99;
    Elec_KFNumHits_ = -99;
    Elec_KFNumPixelHits_ = -99;
    Elec_KFNumStripHits_ = -99;
    Elec_KFTrackResol_ = -99;
    Elec_KFTracklnPt_ = -99;
    Elec_KFTrackEta_ = -99;
    Elec_HasGSF_ = -99;
    Elec_Chi2GSF_ = -99;
    Elec_GSFNumHits_ = -99;
    Elec_GSFNumPixelHits_ = -99;
    Elec_GSFNumStripHits_ = -99;
    Elec_GSFTrackResol_ = -99;
    Elec_GSFTracklnPt_ = -99;
    Elec_GSFTrackEta_ = -99;

    ElecVeto_N_ = 0;
    ElecVeto_Pt_ = -99;
    ElecVeto_Eta_ = -99;
    ElecVeto_Phi_ = -99;

    // Matchings
    Elec_GenEleMatch_ = 0;
    Elec_GenHadMatch_ = 0; 
    
    //-----------------------------------------------------------------------------
    // GSF electron variables
    //-----------------------------------------------------------------------------

    if ( matchedElectron ) {
      for ( reco::CandidateView::const_iterator genElectron = genElectrons->begin();
	    genElectron != genElectrons->end(); ++genElectron ) {
	++NumGenEle_;
	if ( deltaR(matchedElectron->eta(), matchedElectron->phi(), genElectron->eta(), genElectron->phi()) < 0.3 ) Elec_GenEleMatch_ = 1;
      }
      for ( reco::CandidateView::const_iterator genTau = genTaus->begin();
	    genTau != genTaus->end(); ++genTau ) {
	++NumGenHad_;
	if ( deltaR(matchedElectron->eta(), matchedElectron->phi(), genTau->eta(), genTau->phi()) < 0.3 ) Elec_GenHadMatch_ = 1;
      }

      // Matchings
      Elec_AbsEta_ = TMath::Abs(matchedElectron->eta());
      Elec_Pt_ = matchedElectron->pt();
      Elec_PFMvaOutput_ = TMath::Max(matchedElectron->mvaOutput().mva, float(-1.0));
      Elec_EarlyBrem_ = matchedElectron->mvaInput().earlyBrem;
      Elec_LateBrem_= matchedElectron->mvaInput().lateBrem;
      Elec_Logsihih_ = log(matchedElectron->mvaInput().sigmaEtaEta);
      Elec_DeltaEta_ = matchedElectron->mvaInput().deltaEta;
      Elec_Fbrem_ = matchedElectron->fbrem();

      // Variables related to the SC
      Elec_HasSC_ = 0;
      reco::SuperClusterRef pfSuperCluster = matchedElectron->pflowSuperCluster();
      if ( pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable() ) {
	Elec_HasSC_ = 1;
	Elec_Ee_ = 0.;
	Elec_Egamma_ = 0.;
	for ( reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
	      pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
	  float pfClusterEn = (*pfCluster)->energy();
	  if ( pfCluster == pfSuperCluster->clustersBegin() ) Elec_Ee_ += pfClusterEn;
	  else Elec_Egamma_ += pfClusterEn;
	}
	Elec_Pin_ = TMath::Sqrt(matchedElectron->trackMomentumAtVtx().Mag2());
	Elec_Pout_ = TMath::Sqrt(matchedElectron->trackMomentumOut().Mag2()); 
	Elec_EtotOverPin_ = (Elec_Ee_ + Elec_Egamma_)/Elec_Pin_;
	Elec_EeOverPout_ = Elec_Ee_/Elec_Pout_;
	Elec_EgammaOverPdif_ = Elec_Egamma_/(Elec_Pin_ - Elec_Pout_);
	Elec_HoHplusE_ = matchedElectron->mvaInput().hadEnergy/(matchedElectron->mvaInput().hadEnergy+Elec_Ee_) ;
      }

      // Variables related to the CtfTrack
      Elec_HasKF_ = 0;
      if ( matchedElectron->closestCtfTrackRef().isNonnull() ) {
	Elec_HasKF_ = 1;
	Elec_Chi2KF_ = matchedElectron->closestCtfTrackRef()->normalizedChi2();
	Elec_KFNumHits_ = matchedElectron->closestCtfTrackRef()->numberOfValidHits();
	Elec_KFNumPixelHits_ = matchedElectron->closestCtfTrackRef()->hitPattern().numberOfValidPixelHits();
	Elec_KFNumStripHits_ = matchedElectron->closestCtfTrackRef()->hitPattern().numberOfValidStripHits();
    	Elec_KFTrackResol_ = matchedElectron->closestCtfTrackRef()->ptError()/matchedElectron->closestCtfTrackRef()->pt();
	Elec_KFTracklnPt_ = log(matchedElectron->closestCtfTrackRef()->pt())*TMath::Ln10();
	Elec_KFTrackEta_ = matchedElectron->closestCtfTrackRef()->eta();
      }

      // Variables related to the GsfTrack
      Elec_HasGSF_ = 0;
      if ( matchedElectron->gsfTrack().isNonnull() ) {
	Elec_HasGSF_ = 1;
	Elec_Chi2GSF_ = matchedElectron->gsfTrack()->normalizedChi2();
	Elec_GSFNumHits_ = matchedElectron->gsfTrack()->numberOfValidHits();
	Elec_GSFNumPixelHits_ = matchedElectron->gsfTrack()->hitPattern().numberOfValidPixelHits();
	Elec_GSFNumStripHits_ = matchedElectron->gsfTrack()->hitPattern().numberOfValidStripHits();
    	Elec_GSFTrackResol_ = matchedElectron->gsfTrack()->ptError()/matchedElectron->gsfTrack()->pt();
	Elec_GSFTracklnPt_ = log(matchedElectron->gsfTrack()->pt())*TMath::Ln10();
	Elec_GSFTrackEta_ = matchedElectron->gsfTrack()->eta();
      }

      if ( verbosity_ ) {
	std::cout << "Elec_AbsEta: " << Elec_AbsEta_ << std::endl;
	std::cout << "Elec_Pt: " << Elec_Pt_ << std::endl;
	std::cout << "Elec_HasSC: " << Elec_HasSC_ << std::endl;
	std::cout << "Elec_Ee: " << Elec_Ee_ << ", Elec_Egamma: " << Elec_Egamma_ << std::endl;
	std::cout << "Elec_Pin: " << Elec_Pin_ << ", Elec_Pout: " << Elec_Pout_ << std::endl;
	std::cout << "Elec_HasKF: " << Elec_HasKF_ << std::endl;
	std::cout << "Elec_HasGSF: " << Elec_HasGSF_ << std::endl;
	std::cout << "Elec_PFMvaOutput: " << Elec_PFMvaOutput_ << std::endl;
	std::cout << "Elec_EtotOverPin: " << Elec_EtotOverPin_ << std::endl;
	std::cout << "Elec_EeOverPout: " << Elec_EeOverPout_ << std::endl;
	std::cout << "Elec_EgammaOverPdif: " << Elec_EgammaOverPdif_ << std::endl;
	std::cout << "Elec_EarlyBrem: " << Elec_EarlyBrem_ << std::endl;
	std::cout << "Elec_LateBrem: " << Elec_LateBrem_ << std::endl;
	std::cout << "Elec_Logsihih: " << Elec_Logsihih_ << std::endl;
	std::cout << "Elec_DeltaEta: " << Elec_DeltaEta_ << std::endl;
	std::cout << "Elec_HoHplusE: " << Elec_HoHplusE_ << std::endl;
	std::cout << "Elec_FBrem: " << Elec_Fbrem_ << std::endl;
	std::cout << "Elec_Chi2KF: " << Elec_Chi2KF_ << std::endl;
	std::cout << "Elec_Chi2GSF: " << Elec_Chi2GSF_ << std::endl;
	std::cout << "Elec_GSFNumHits: " << Elec_GSFNumHits_ << std::endl;
	std::cout << "Elec_GSFNumPixelHits: " << Elec_GSFNumPixelHits_ << std::endl;
	std::cout << "Elec_GSFNumStripHits: " << Elec_GSFNumStripHits_ << std::endl;
	std::cout << "Elec_GSFTrackResol: " << Elec_GSFTrackResol_ << std::endl;
	std::cout << "Elec_GSFTracklnPt: " << Elec_GSFTracklnPt_ << std::endl;
	std::cout << "Elec_GSFTrackEta: " << Elec_GSFTrackEta_ << std::endl;
	std::cout << std::endl;
      }

    } // end if matchedElectron

    //-----------------------------------------------------------------------------
    // PFTau variables
    //-----------------------------------------------------------------------------

    for ( reco::CandidateView::const_iterator genElectron = genElectrons->begin();
	  genElectron != genElectrons->end(); ++genElectron ) {
      if ( deltaR(pfTau->eta(), pfTau->phi(), genElectron->eta(), genElectron->phi()) < 0.3 ) Tau_GenEleMatch_ = 1;
    }
    for ( reco::CandidateView::const_iterator genTau = genTaus->begin();
	  genTau != genTaus->end(); ++genTau ) {
      if ( deltaR(pfTau->eta(), pfTau->phi(), genTau->eta(), genTau->phi()) < 0.3 ) Tau_GenHadMatch_ = 1;
    }

    // Matchings
    float sumEtaTimesEnergy = 0.;
    float sumPhiTimesEnergy = 0.;
    float sumEnergy = 0.;
    float sumEtaTimesEnergyEcalEnWeighted = 0.;
    float sumPhiTimesEnergyEcalEnWeighted = 0.;
    float sumEnergyEcalEnWeighted = 0.;
    const std::vector<reco::PFCandidatePtr>& signalPFCands = pfTau->signalPFCands();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	  pfCandidate != signalPFCands.end(); ++pfCandidate ) {
      sumEtaTimesEnergy += ((*pfCandidate)->positionAtECALEntrance().eta()*(*pfCandidate)->energy());
      sumPhiTimesEnergy += ((*pfCandidate)->positionAtECALEntrance().phi()*(*pfCandidate)->energy());
      sumEnergy += (*pfCandidate)->energy();
      sumEtaTimesEnergyEcalEnWeighted += ((*pfCandidate)->positionAtECALEntrance().eta()*(*pfCandidate)->ecalEnergy());
      sumPhiTimesEnergyEcalEnWeighted += ((*pfCandidate)->positionAtECALEntrance().phi()*(*pfCandidate)->ecalEnergy());
      sumEnergyEcalEnWeighted += (*pfCandidate)->ecalEnergy();
    }
    if ( sumEnergy > 0. ) {
      Tau_EtaAtEcalEntrance_ = sumEtaTimesEnergy/sumEnergy;
      Tau_PhiAtEcalEntrance_ = sumPhiTimesEnergy/sumEnergy;
    }
    if ( sumEnergyEcalEnWeighted > 0. ) {
      Tau_EtaAtEcalEntranceEcalEnWeighted_ = sumEtaTimesEnergyEcalEnWeighted/sumEnergyEcalEnWeighted;
      Tau_PhiAtEcalEntranceEcalEnWeighted_ = sumPhiTimesEnergyEcalEnWeighted/sumEnergyEcalEnWeighted;
    }
    for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	  pfCandidate != signalPFCands.end(); ++pfCandidate ) {
      const reco::Track* track = 0;
      if ( (*pfCandidate)->trackRef().isNonnull() ) track = (*pfCandidate)->trackRef().get();
      else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->innerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->innerTrack().get();
      else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->globalTrack().isNonnull() ) track = (*pfCandidate)->muonRef()->globalTrack().get();
      else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->outerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->outerTrack().get();
      else if ( (*pfCandidate)->gsfTrackRef().isNonnull() ) track = (*pfCandidate)->gsfTrackRef().get();
      if ( track ) {
	if ( track->pt() > Tau_LeadChargedPFCandPt_ ) {
	  Tau_LeadChargedPFCandEtaAtEcalEntrance_ = (*pfCandidate)->positionAtECALEntrance().eta();
	  Tau_LeadChargedPFCandPhiAtEcalEntrance_ = (*pfCandidate)->positionAtECALEntrance().phi();
	  Tau_LeadChargedPFCandPt_ = track->pt();
	}
      } else {
	if ( (*pfCandidate)->pt() > Tau_LeadNeutralPFCandPt_ ) {
	  Tau_LeadNeutralPFCandEtaAtEcalEntrance_ = (*pfCandidate)->positionAtECALEntrance().eta();
	  Tau_LeadNeutralPFCandPhiAtEcalEntrance_ = (*pfCandidate)->positionAtECALEntrance().phi();
	  Tau_LeadNeutralPFCandPt_ = (*pfCandidate)->pt();
	}
      }
    }
    Tau_Eta_ = pfTau->eta();
    Tau_Pt_ = pfTau->pt();
    Tau_Phi_ = pfTau->phi();
    Tau_EmFraction_ = TMath::Max(pfTau->emFraction(), float(0.));
    Tau_NumChargedCands_ = pfTau->signalPFChargedHadrCands().size();
    Tau_NumGammaCands_  = pfTau->signalPFGammaCands().size();

    if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
      Tau_LeadHadronPt_ = pfTau->leadPFChargedHadrCand()->pt();
      Tau_HasGsf_ = (pfTau->leadPFChargedHadrCand()->gsfTrackRef()).isNonnull();
    }
    if ( (pfTau->leadPFChargedHadrCand()->gsfTrackRef()).isNonnull() ) {
      Tau_GSFChi2_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->normalizedChi2();
      Tau_GSFNumHits_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->numberOfValidHits();
      Tau_GSFNumPixelHits_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->hitPattern().numberOfValidPixelHits();
      Tau_GSFNumStripHits_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->hitPattern().numberOfValidStripHits();
      Tau_GSFTrackResol_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->ptError()/pfTau->leadPFChargedHadrCand()->gsfTrackRef()->pt();
      Tau_GSFTracklnPt_ = log(pfTau->leadPFChargedHadrCand()->gsfTrackRef()->pt())*TMath::Ln10();
      Tau_GSFTrackEta_ = pfTau->leadPFChargedHadrCand()->gsfTrackRef()->eta();
    }

    if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
      Tau_HasKF_ = (pfTau->leadPFChargedHadrCand()->trackRef()).isNonnull();
    }
    if ( (pfTau->leadPFChargedHadrCand()->trackRef()).isNonnull() ) {
      Tau_KFChi2_ = pfTau->leadPFChargedHadrCand()->trackRef()->normalizedChi2();
      Tau_KFNumHits_ = pfTau->leadPFChargedHadrCand()->trackRef()->numberOfValidHits();
      Tau_KFNumPixelHits_ = pfTau->leadPFChargedHadrCand()->trackRef()->hitPattern().numberOfValidPixelHits();
      Tau_KFNumStripHits_ = pfTau->leadPFChargedHadrCand()->trackRef()->hitPattern().numberOfValidStripHits();
      Tau_KFTrackResol_ = pfTau->leadPFChargedHadrCand()->trackRef()->ptError()/pfTau->leadPFChargedHadrCand()->trackRef()->pt();
      Tau_KFTracklnPt_ = log(pfTau->leadPFChargedHadrCand()->trackRef()->pt())*TMath::Ln10();
      Tau_KFTrackEta_ = pfTau->leadPFChargedHadrCand()->trackRef()->eta();
    }

    if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
      Tau_HadrHoP_ = pfTau->leadPFChargedHadrCand()->hcalEnergy()/pfTau->leadPFChargedHadrCand()->p();
      Tau_HadrEoP_ = pfTau->leadPFChargedHadrCand()->ecalEnergy()/pfTau->leadPFChargedHadrCand()->p();
    }

    GammasdEta_.clear();
    GammasdPhi_.clear();
    GammasPt_.clear();
    const std::vector<reco::PFCandidatePtr>& signalPFGammaCands = pfTau->signalPFGammaCands();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator pfGamma = signalPFGammaCands.begin();
	  pfGamma != signalPFGammaCands.end(); ++pfGamma ) {
      if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
	GammasdEta_.push_back((*pfGamma)->eta() - pfTau->leadPFChargedHadrCand()->eta());
	GammasdPhi_.push_back((*pfGamma)->phi() - pfTau->leadPFChargedHadrCand()->phi());
      } else {
	GammasdEta_.push_back((*pfGamma)->eta() - pfTau->eta());
	GammasdPhi_.push_back((*pfGamma)->phi() - pfTau->phi());
      }
      GammasPt_.push_back((*pfGamma)->pt());
    }

    float sumPt  = 0.;
    float dEta   = 0.;
    float dEta2  = 0.;
    float dPhi   = 0.;
    float dPhi2  = 0.;
    float sumPt2 = 0.;
    size_t numPFGammas = GammasPt_.size();
    assert(GammasdEta_.size() == numPFGammas);
    assert(GammasdPhi_.size() == numPFGammas);
    for ( size_t idxPFGamma = 0; idxPFGamma < numPFGammas; ++idxPFGamma ) {
      float pt  = GammasPt_[idxPFGamma];
      float dPhi = GammasdPhi_[idxPFGamma];
      if ( dPhi > TMath::Pi() ) dPhi -= 2.*TMath::Pi();
      else if ( dPhi < -TMath::Pi() ) dPhi += 2.*TMath::Pi();
      float dEta = GammasdEta_[idxPFGamma];
      sumPt  +=  pt;
      sumPt2 += (pt*pt);
      dEta   += (pt*dEta);
      dEta2  += (pt*dEta*dEta);
      dPhi   += (pt*dPhi);
      dPhi2  += (pt*dPhi*dPhi);  
    }
    
    float gammadPt = sumPt/pfTau->pt();
	    
    if ( sumPt > 0. ) {
      dEta  /= sumPt;
      dPhi  /= sumPt;
      dEta2 /= sumPt;
      dPhi2 /= sumPt;
    }

    Tau_GammaEtaMom_ = TMath::Sqrt(dEta2)*TMath::Sqrt(gammadPt)*pfTau->pt();
    Tau_GammaPhiMom_ = TMath::Sqrt(dPhi2)*TMath::Sqrt(gammadPt)*pfTau->pt();  
    Tau_GammaEnFrac_ = gammadPt;
    Tau_VisMass_ = pfTau->mass();
    Tau_HadrMva_ = TMath::Max(pfTau->electronPreIDOutput(), float(-1.));

    if ( verbosity_ ) {
      std::cout << "GammaEtaMom: " << Tau_GammaEtaMom_ << std::endl;
      std::cout << "GammaPhiMom: " << Tau_GammaPhiMom_ << std::endl;
      std::cout << "GammaPt: " << sumPt << std::endl;
      std::cout << "TauPt: " << pfTau->pt() << std::endl;
      std::cout << "TauEta: " << pfTau->eta() << std::endl;
      std::cout << "TauEtaAtEcalEntrance: " << Tau_EtaAtEcalEntrance_ << std::endl;
      std::cout << "sumEtaTimesEnergy: " << sumEtaTimesEnergy << std::endl;
      std::cout << "sumEnergy: " << sumEnergy << std::endl;
      std::cout << "GammaEnFrac: " << Tau_GammaEnFrac_ << std::endl;
      std::cout << std::endl;
    }

    for ( std::vector<tauIdDiscrEntryType>::iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	  tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
      edm::Handle<reco::PFTauDiscriminator> discriminator;
      evt.getByLabel(tauIdDiscriminator->src_, discriminator);
      tauIdDiscriminator->value_ = (*discriminator)[pfTau];
    }

    Tau_DecayMode_ = pfTau->decayMode();

    Tau_VtxZ_ = pfTau->vertex().z();
    if ( TMath::Abs(TMath::Tan(pfTau->theta())) > 1.e-3 ) {
      const double rECAL = 130; // approx. ECAL radius in cm
      Tau_zImpact_ = pfTau->vertex().z() + rECAL/TMath::Tan(pfTau->theta());
    }
    
    // Tau is matched to a gsfElectron passing selection criteria for SecondElectronVeto
    bool matchElectronCutsVeto = false;
    int idxGSFElectron = 0;
    for ( reco::GsfElectronCollection::const_iterator gsfElectron = gsfElectrons->begin();
	  gsfElectron != gsfElectrons->end(); ++gsfElectron ) {
      const reco::Track* track = (const reco::Track*)((gsfElectron)->gsfTrack().get());  
      assert(track);
      const reco::HitPattern& p_inner = track->trackerExpectedHitsInner(); 
      float nHits = p_inner.numberOfHits();
      float dPhi  = fabs(gsfElectron->deltaPhiSuperClusterTrackAtVtx());
      float dEta  = fabs(gsfElectron->deltaEtaSuperClusterTrackAtVtx());
      float sihih = gsfElectron->sigmaIetaIeta();
      float HoE   = gsfElectron->hadronicOverEm();
      if ( verbosity_ ) {
	std::cout << "gsfElectron #: " << idxGSFElectron << std::endl;
	std::cout << "gsfElectron nHits: " << nHits << std::endl;
	std::cout << "gsfElectron pt: " << gsfElectron->pt() << std::endl;
	std::cout << "gsfElectron eta: " << gsfElectron->eta() << std::endl;
	std::cout << "gsfElectron dPhi: " << dPhi << std::endl;
	std::cout << "gsfElectron dEta: " << dEta << std::endl;
	std::cout << "gsfElectron sihih: " << sihih << std::endl;
	std::cout << "gsfElectron HoE: " << HoE << std::endl;
	std::cout << std::endl;
      }
      bool ElectronPassCutsVeto = false;
      if ( nHits <= 999 &&
	   ((fabs(gsfElectron->eta()) < 1.5 &&
	     sihih < 0.010 &&
	     dPhi < 0.80 &&
	     dEta < 0.007 &&
	     HoE < 0.15) ||
	    (fabs(gsfElectron->eta()) > 1.5 && fabs(gsfElectron->eta()) < 2.3 &&
	     sihih < 0.030 &&
	     dPhi < 0.70 &&
	     dEta < 0.010 &&
	     HoE < 999))
	   ) ElectronPassCutsVeto = true;
      if ( verbosity_ ) {
	if ( ElectronPassCutsVeto ) std::cout << "gsfElectron passes 2nd electron selection." << std::endl;
	else std::cout << "gsfElectron fails 2nd electron selection." << std::endl;
      }
      double ElePt = gsfElectron->pt();
      if ( (deltaR(pfTau->eta(), pfTau->phi(), gsfElectron->eta(), gsfElectron->phi()) < 0.3) && ElectronPassCutsVeto && ElePt > ElecVeto_Pt_ ) {
	matchElectronCutsVeto = true;
	ElecVeto_Pt_ = ElePt;
	ElecVeto_Eta_ = gsfElectron->eta();
	ElecVeto_Phi_ = gsfElectron->phi();
	++ElecVeto_N_;
      } 
      ++idxGSFElectron;
    } // end loop over gsfElectrons
    Tau_MatchElePassVeto_ = matchElectronCutsVeto;

    if ( verbosity_ ) {
      std::cout << "Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << std::endl;
      std::cout << "matchElectronCutsVeto = " << matchElectronCutsVeto << std::endl;
    }

    ++NumPFTaus_;

    tree_->Fill();

  } // end loop over PFTaus   
}

AntiElectronDiscrMVATrainingNtupleProducer::~AntiElectronDiscrMVATrainingNtupleProducer()
{
// nothing to be done yet...
}

void AntiElectronDiscrMVATrainingNtupleProducer::endJob()
{
// nothing to be done yet...
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AntiElectronDiscrMVATrainingNtupleProducer);



