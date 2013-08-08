#include "RecoTauTag/TauAnalysisTools/plugins/AntiMuonDiscrMVATrainingNtupleProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <fstream>

AntiMuonDiscrMVATrainingNtupleProducer::AntiMuonDiscrMVATrainingNtupleProducer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    ntuple_(0)
{
  srcRecTaus_ = cfg.getParameter<edm::InputTag>("srcRecTaus");

  srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
  dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
  
  srcGenTauJets_ = cfg.getParameter<edm::InputTag>("srcGenTauJets");
  srcGenParticles_ = cfg.getParameter<edm::InputTag>("srcGenParticles");
  minGenVisPt_ = cfg.getParameter<double>("minGenVisPt");
  dRgenParticleMatch_ = cfg.getParameter<double>("dRgenParticleMatch");
  
  pdgIdsGenTau_.push_back(-15);
  pdgIdsGenTau_.push_back(+15);

  pdgIdsGenMuon_.push_back(-13);
  pdgIdsGenMuon_.push_back(+13);
  
  edm::ParameterSet tauIdDiscriminators = cfg.getParameter<edm::ParameterSet>("tauIdDiscriminators");
  typedef std::vector<std::string> vstring;
  vstring tauIdDiscriminatorNames = tauIdDiscriminators.getParameterNamesForType<edm::InputTag>();
  for ( vstring::const_iterator name = tauIdDiscriminatorNames.begin();
	name != tauIdDiscriminatorNames.end(); ++name ) {
    edm::InputTag src = tauIdDiscriminators.getParameter<edm::InputTag>(*name);
    tauIdDiscrEntries_.push_back(tauIdDiscrEntryType(*name, src));
  }

  edm::ParameterSet vertexCollections = cfg.getParameter<edm::ParameterSet>("vertexCollections");
  vstring vertexCollectionNames = vertexCollections.getParameterNamesForType<edm::InputTag>();
  for ( vstring::const_iterator name = vertexCollectionNames.begin();
	name != vertexCollectionNames.end(); ++name ) {
    edm::InputTag src = vertexCollections.getParameter<edm::InputTag>(*name);
    vertexCollectionEntries_.push_back(vertexCollectionEntryType(*name, src));
  }

  edm::ParameterSet cfgPFJetIdAlgo;
  cfgPFJetIdAlgo.addParameter<std::string>("version", "FIRSTDATA");
  cfgPFJetIdAlgo.addParameter<std::string>("quality", "LOOSE");
  loosePFJetIdAlgo_ = new PFJetIDSelectionFunctor(cfgPFJetIdAlgo);

  isMC_ = cfg.getParameter<bool>("isMC");

  srcWeights_ = cfg.getParameter<vInputTag>("srcWeights");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

AntiMuonDiscrMVATrainingNtupleProducer::~AntiMuonDiscrMVATrainingNtupleProducer()
{
// nothing to be done yet...
}

void AntiMuonDiscrMVATrainingNtupleProducer::beginJob()
{
//--- create TTree
  edm::Service<TFileService> fs;
  ntuple_ = fs->make<TTree>("antiMuonDiscrMVATrainingNtuple", "antiMuonDiscrMVATrainingNtuple");

//--- add branches 
  addBranchI("run");
  addBranchI("event");
  addBranchI("lumi");

  addBranch_EnPxPyPz("recTau");
  addBranch_EnPxPyPz("recTauAlternate");
  addBranchI("recTauDecayMode");
  addBranchF("recTauVtxZ");
  addBranchF("recTauCaloEnECAL");
  addBranchF("recTauCaloEnHCAL");
  addBranchF("recTauCaloEnHO");
  addBranch_EnPxPyPz("recJet");
  addBranchI("recJetLooseId");
  addBranch_EnPxPyPz("leadPFChargedHadrCand");
  addBranchF("leadPFChargedHadrCandCaloEnECAL");
  addBranchF("leadPFChargedHadrCandCaloEnHCAL");
  addBranchF("leadPFChargedHadrCandCaloEnHO");
  addBranch_EnPxPyPz("leadTrack");  
  addBranchI("numMatches");
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    addBranchI(Form("numHitsDT%i", iStation + 1));
    addBranchI(Form("numHitsCSC%i", iStation + 1));
    addBranchI(Form("numHitsRPC%i", iStation + 1));
  }
  for ( std::vector<tauIdDiscrEntryType>::const_iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
    addBranchF(tauIdDiscriminator->branchName_);
  }
  for ( std::vector<vertexCollectionEntryType>::const_iterator vertexCollection = vertexCollectionEntries_.begin();
	vertexCollection != vertexCollectionEntries_.end(); ++vertexCollection ) {
    addBranchI(vertexCollection->branchName_multiplicity_);
    addBranch_XYZ(vertexCollection->branchName_position_);
  }
  addBranch_EnPxPyPz("genTau");
  addBranchF("genTauDeltaR");
  addBranch_EnPxPyPz("genVisTau");
  addBranchF("genVisTauDeltaR");
  addBranchI("genTauDecayMode");
  addBranchI("genTauMatch");
  addBranch_EnPxPyPz("genMuon");
  addBranchI("genMuonMatch");
  addBranchF("genMuonDeltaR");
  addBranchI("genMuonPdgId");
  addBranchF("evtWeight");
}

namespace
{
  double square(double x)
  {
    return (x*x);
  }

  void countHits(const reco::Muon& muon, std::vector<int>& numHitsDT, std::vector<int>& numHitsCSC, std::vector<int>& numHitsRPC)
  {
    if ( muon.outerTrack().isNonnull() ) {
      const reco::HitPattern& muonHitPattern = muon.outerTrack()->hitPattern();
      for ( int iHit = 0; iHit < muonHitPattern.numberOfHits(); ++iHit ) {
	uint32_t hit = muonHitPattern.getHitPattern(iHit);
	if ( hit == 0 ) break;	    
	if ( muonHitPattern.muonHitFilter(hit) && (muonHitPattern.getHitType(hit) == TrackingRecHit::valid || muonHitPattern.getHitType(hit) == TrackingRecHit::bad) ) {
	  int muonStation = muonHitPattern.getMuonStation(hit) - 1; // CV: map into range 0..3
	  if ( muonStation >= 0 && muonStation < 4 ) {
	    if      ( muonHitPattern.muonDTHitFilter(hit)  ) ++numHitsDT[muonStation];
	    else if ( muonHitPattern.muonCSCHitFilter(hit) ) ++numHitsCSC[muonStation];
	    else if ( muonHitPattern.muonRPCHitFilter(hit) ) ++numHitsRPC[muonStation];
	  }
	}
      }
    }
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::setRecTauValues(const reco::PFTauRef& recTau, const edm::Event& evt)
{
  setValue_EnPxPyPz("recTau", recTau->p4());
  setValue_EnPxPyPz("recTauAlternate", recTau->alternatLorentzVect());
  setValueI("recTauDecayMode", recTau->decayMode());
  setValueF("recTauVtxZ", recTau->vertex().z());
  int recTauNumMatches = 0;
  double recTauCaloEnECAL = 0.;
  double recTauCaloEnHCAL = 0.;
  double recTauCaloEnHO   = 0.;
  const std::vector<reco::PFCandidatePtr>& recTauSignalPFCands = recTau->signalPFCands();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator recTauSignalPFCand = recTauSignalPFCands.begin();
	recTauSignalPFCand != recTauSignalPFCands.end(); ++recTauSignalPFCand ) {
    if ( (*recTauSignalPFCand)->muonRef().isNonnull() ) {
      recTauNumMatches += (*recTauSignalPFCand)->muonRef()->numberOfMatches(reco::Muon::NoArbitration);
    }
    recTauCaloEnECAL += (*recTauSignalPFCand)->ecalEnergy();
    recTauCaloEnHCAL += (*recTauSignalPFCand)->hcalEnergy();
    recTauCaloEnHO   += (*recTauSignalPFCand)->hoEnergy();
  }
  setValueF("recTauCaloEnECAL", recTauCaloEnECAL);
  setValueF("recTauCaloEnHCAL", recTauCaloEnHCAL);
  setValueF("recTauCaloEnHO",   recTauCaloEnHO);  
  setValue_EnPxPyPz("recJet", recTau->jetRef()->p4());
  int recJetLooseId = ( (*loosePFJetIdAlgo_)(*recTau->jetRef()) ) ? 1 : 0;
  setValueI("recJetLooseId", recJetLooseId);
  const reco::Track* leadTrack = 0;
  if ( recTau->leadPFChargedHadrCand().isNonnull() ) {
    setValue_EnPxPyPz("leadPFChargedHadrCand", recTau->leadPFChargedHadrCand()->p4());
    setValueF("leadPFChargedHadrCandCaloEnECAL", recTau->leadPFChargedHadrCand()->ecalEnergy());
    setValueF("leadPFChargedHadrCandCaloEnHCAL", recTau->leadPFChargedHadrCand()->hcalEnergy());
    setValueF("leadPFChargedHadrCandCaloEnHO", recTau->leadPFChargedHadrCand()->hoEnergy());
    if ( recTau->leadPFChargedHadrCand()->trackRef().isNonnull() ) leadTrack = recTau->leadPFChargedHadrCand()->trackRef().get();
    else if ( recTau->leadPFChargedHadrCand()->gsfTrackRef().isNonnull() ) leadTrack = recTau->leadPFChargedHadrCand()->gsfTrackRef().get();
  } else {
    setValue_EnPxPyPz("leadPFChargedHadrCand", reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueF("leadPFChargedHadrCandCaloEnECAL", 0.);
    setValueF("leadPFChargedHadrCandCaloEnHCAL", 0.);
    setValueF("leadPFChargedHadrCandCaloEnHO", 0.);
  }
  if ( leadTrack ) {
    const double chargedPionMass = 0.13957; // GeV
    double leadTrackEn = TMath::Sqrt(square(leadTrack->p()) + chargedPionMass*chargedPionMass);
    reco::Candidate::LorentzVector leadTrackP4(leadTrack->px(), leadTrack->py(), leadTrack->pz(), leadTrackEn);
    setValue_EnPxPyPz("leadTrack", leadTrackP4);
  } else {
    setValue_EnPxPyPz("leadTrack", reco::Candidate::LorentzVector(0.,0.,0.,0.));
  }
  int numMatches = 0;
  std::vector<int> numHitsDT(4);
  std::vector<int> numHitsCSC(4);
  std::vector<int> numHitsRPC(4);
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    numHitsDT[iStation]  = 0;
    numHitsCSC[iStation] = 0;
    numHitsRPC[iStation] = 0;
  }
  if ( recTau->leadPFChargedHadrCand().isNonnull() ) {
    reco::MuonRef muonRef = recTau->leadPFChargedHadrCand()->muonRef();      
    if ( muonRef.isNonnull() ) {
      numMatches = muonRef->numberOfMatches(reco::Muon::NoArbitration);
      countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
    }
  }
  size_t numMuons = muons_->size();
  for ( size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon ) {
    reco::MuonRef muon(muons_, idxMuon);
    if ( recTau->leadPFChargedHadrCand().isNonnull() && recTau->leadPFChargedHadrCand()->muonRef().isNonnull() && muon == recTau->leadPFChargedHadrCand()->muonRef() ) {	
      continue;
    }
    double dR = deltaR(muon->p4(), recTau->p4());
    if ( dR < dRmuonMatch_ ) {
      numMatches += muon->numberOfMatches(reco::Muon::NoArbitration);
      countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
    }
  }
  setValueI("numMatches", numMatches);
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    setValueI(Form("numHitsDT%i", iStation + 1), numHitsDT[iStation]);
    setValueI(Form("numHitsCSC%i", iStation + 1), numHitsCSC[iStation]);
    setValueI(Form("numHitsRPC%i", iStation + 1), numHitsRPC[iStation]);
  }
  for ( std::vector<tauIdDiscrEntryType>::const_iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
    edm::Handle<reco::PFTauDiscriminator> discriminator;
    evt.getByLabel(tauIdDiscriminator->src_, discriminator);
    setValueF(tauIdDiscriminator->branchName_, (*discriminator)[recTau]);
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::setGenTauMatchValues(
       const reco::Candidate::LorentzVector& recTauP4, const reco::GenParticle* genTau, const reco::Candidate::LorentzVector& genVisTauP4, int genTauDecayMode)
{
  if ( genTau ) {
    setValue_EnPxPyPz("genTau", genTau->p4());
    setValueF("genTauDeltaR", deltaR(genTau->p4(), recTauP4));
    setValue_EnPxPyPz("genVisTau", genVisTauP4);
    setValueF("genVisTauDeltaR", deltaR(genVisTauP4, recTauP4));
    setValueI("genTauDecayMode", genTauDecayMode);
    setValueI("genTauMatch", 1);
  } else {
    setValue_EnPxPyPz("genTau", reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueF("genTauDeltaR", 1.e+3);
    setValue_EnPxPyPz("genVisTau", reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueF("genVisTauDeltaR", 1.e+3);
    setValueI("genTauDecayMode", -1);
    setValueI("genTauMatch", 0);
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::setGenParticleMatchValues(const std::string& branchName, const reco::Candidate::LorentzVector& recTauP4, const reco::GenParticle* genParticle)
{
  if ( genParticle ) {
    setValue_EnPxPyPz(branchName, genParticle->p4());
    setValueI(std::string(branchName).append("Match"), 1);    
    setValueF(std::string(branchName).append("DeltaR"), deltaR(genParticle->p4(), recTauP4));
    setValueI(std::string(branchName).append("PdgId"), genParticle->pdgId());
  } else {
    setValue_EnPxPyPz(branchName, reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueI(std::string(branchName).append("Match"), 0);    
    setValueF(std::string(branchName).append("DeltaR"), 1.e+3);
    setValueI(std::string(branchName).append("PdgId"), 0);
  }
}

namespace
{
  void findDaughters(const reco::GenParticle* mother, std::vector<const reco::GenParticle*>& daughters, int status)
  {
    unsigned numDaughters = mother->numberOfDaughters();
    for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
      const reco::GenParticle* daughter = mother->daughterRef(iDaughter).get();      
      if ( status == -1 || daughter->status() == status ) daughters.push_back(daughter);
      findDaughters(daughter, daughters, status);
    }
  }
  
  bool isNeutrino(const reco::GenParticle* daughter)
  {
    return ( TMath::Abs(daughter->pdgId()) == 12 || TMath::Abs(daughter->pdgId()) == 14 || TMath::Abs(daughter->pdgId()) == 16 );
  }
  
  reco::Candidate::LorentzVector getVisMomentum(const std::vector<const reco::GenParticle*>& daughters, int status)
  {
    reco::Candidate::LorentzVector p4Vis(0,0,0,0);    
    for ( std::vector<const reco::GenParticle*>::const_iterator daughter = daughters.begin();
	  daughter != daughters.end(); ++daughter ) {
      if ( (status == -1 || (*daughter)->status() == status) && !isNeutrino(*daughter) ) {
	p4Vis += (*daughter)->p4();
      }
    }
    return p4Vis;
  }

  reco::Candidate::LorentzVector getVisMomentum(const reco::GenParticle* genTau)
  {
    std::vector<const reco::GenParticle*> stableDaughters;
    findDaughters(genTau, stableDaughters, 1);
    reco::Candidate::LorentzVector genVisTauP4 = getVisMomentum(stableDaughters, 1);    
    return genVisTauP4;
  }

  void countDecayProducts(const reco::GenParticle* genParticle,
			  int& numElectrons, int& numElecNeutrinos, int& numMuons, int& numMuNeutrinos, 
			  int& numChargedHadrons, int& numPi0s, int& numOtherNeutralHadrons, int& numPhotons)
  {
    int absPdgId = TMath::Abs(genParticle->pdgId());
    int status   = genParticle->status();
    int charge   = genParticle->charge();
    
    if      ( absPdgId == 111 ) ++numPi0s;
    else if ( status   ==   1 ) {
      if      ( absPdgId == 11 ) ++numElectrons;
      else if ( absPdgId == 12 ) ++numElecNeutrinos;
      else if ( absPdgId == 13 ) ++numMuons;
      else if ( absPdgId == 14 ) ++numMuNeutrinos;
      else if ( absPdgId == 15 ) { 
	edm::LogError ("countDecayProducts")
	  << "Found tau lepton with status code 1 !!";
	return; 
      }
      else if ( absPdgId == 16 ) return; // no need to count tau neutrinos
      else if ( absPdgId == 22 ) ++numPhotons;
      else if ( charge   !=  0 ) ++numChargedHadrons;
      else                       ++numOtherNeutralHadrons;
    } else {
      unsigned numDaughters = genParticle->numberOfDaughters();
      for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
	const reco::GenParticle* daughter = genParticle->daughterRef(iDaughter).get();
	
	countDecayProducts(daughter, 
			   numElectrons, numElecNeutrinos, numMuons, numMuNeutrinos,
			   numChargedHadrons, numPi0s, numOtherNeutralHadrons, numPhotons);
      }
    }
  }
  
  std::string getGenTauDecayMode(const reco::GenParticle* genTau) 
  {
//--- determine generator level tau decay mode
//
//    NOTE: 
//        (1) function implements logic defined in PhysicsTools/JetMCUtils/src/JetMCTag::genTauDecayMode
//            for different type of argument 
//        (2) this implementation should be more robust to handle cases of tau --> tau + gamma radiation
//
    int numElectrons           = 0;
    int numElecNeutrinos       = 0;
    int numMuons               = 0;
    int numMuNeutrinos         = 0; 
    int numChargedHadrons      = 0;
    int numPi0s                = 0; 
    int numOtherNeutralHadrons = 0;
    int numPhotons             = 0;
    
    countDecayProducts(genTau,
		       numElectrons, numElecNeutrinos, numMuons, numMuNeutrinos,
		       numChargedHadrons, numPi0s, numOtherNeutralHadrons, numPhotons);
    
    if      ( numElectrons == 1 && numElecNeutrinos == 1 ) return std::string("electron");
    else if ( numMuons     == 1 && numMuNeutrinos   == 1 ) return std::string("muon");
    
    switch ( numChargedHadrons ) {
    case 1 : 
      if ( numOtherNeutralHadrons != 0 ) return std::string("oneProngOther");
      switch ( numPi0s ) {
      case 0:
	return std::string("oneProng0Pi0");
      case 1:
	return std::string("oneProng1Pi0");
      case 2:
	return std::string("oneProng2Pi0");
      default:
	return std::string("oneProngOther");
      }
    case 3 : 
      if ( numOtherNeutralHadrons != 0 ) return std::string("threeProngOther");
      switch ( numPi0s ) {
      case 0:
	return std::string("threeProng0Pi0");
      case 1:
	return std::string("threeProng1Pi0");
      default:
	return std::string("threeProngOther");
      }
    default:
      return std::string("rare");
    }
  }
 
  const reco::GenParticle* findMatchingGenParticle(const reco::Candidate::LorentzVector& recTauP4, 
						   const reco::GenParticleCollection& genParticles, double minGenVisPt, const std::vector<int>& pdgIds, double dRmatch)
  {
    const reco::GenParticle* genParticle_matched = 0;
    double dRmin = dRmatch;
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( !(genParticle->pt() > minGenVisPt) ) continue;
      double dR = deltaR(genParticle->p4(), recTauP4);
      if ( dR < dRmin ) {
	bool matchedPdgId = false;
	for ( std::vector<int>::const_iterator pdgId = pdgIds.begin();
	      pdgId != pdgIds.end(); ++pdgId ) {
	  if ( genParticle->pdgId() == (*pdgId) ) {
	    matchedPdgId = true;
	    break;
	  }
	}
	if ( matchedPdgId ) {
	  genParticle_matched = &(*genParticle);
	  dRmin = dR;
	}
      }
    }
    return genParticle_matched;
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  assert(ntuple_);

  edm::Handle<reco::PFTauCollection> recTaus;
  evt.getByLabel(srcRecTaus_, recTaus);

  evt.getByLabel(srcMuons_, muons_);

  edm::Handle<reco::GenParticleCollection> genParticles;
  if ( isMC_ ) {  
    evt.getByLabel(srcGenParticles_, genParticles);
  }

  double evtWeight = 1.0;
  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    evtWeight *= (*weight);
  }

  size_t numRecTaus = recTaus->size();
  //std::cout << "numRecTaus = " << numRecTaus << std::endl;
  for ( size_t iRecTau = 0; iRecTau < numRecTaus; ++iRecTau ) {
    reco::PFTauRef recTau(recTaus, iRecTau);
    setRecTauValues(recTau, evt);

    const reco::GenParticle* genTau_matched = 0;
    reco::Candidate::LorentzVector genVisTauP4_matched(0.,0.,0.,0.);
    int genTauDecayMode_matched = -1;
    if ( isMC_ ) {      
      double dRmin = dRgenParticleMatch_;
      for ( reco::GenParticleCollection::const_iterator genParticle = genParticles->begin();
	    genParticle != genParticles->end(); ++genParticle ) {
	if ( !(genParticle->status() == 2) ) continue;
	bool matchedPdgId = false;
	for ( std::vector<int>::const_iterator pdgId = pdgIdsGenTau_.begin();
	      pdgId != pdgIdsGenTau_.end(); ++pdgId ) {
	  if ( genParticle->pdgId() == (*pdgId) ) {
	    matchedPdgId = true;
	    break;
	  }
	}
	if ( !matchedPdgId ) continue;
	reco::Candidate::LorentzVector genVisTauP4 = getVisMomentum(&(*genParticle));
	if ( !(genVisTauP4.pt() > minGenVisPt_) ) continue;
	std::string genTauDecayMode_string = getGenTauDecayMode(&(*genParticle));
	int genTauDecayMode = -1;
	if      ( genTauDecayMode_string == "oneProng0Pi0"    ) genTauDecayMode = reco::PFTau::kOneProng0PiZero;
	else if ( genTauDecayMode_string == "oneProng1Pi0"    ) genTauDecayMode = reco::PFTau::kOneProng1PiZero;
	else if ( genTauDecayMode_string == "oneProng2Pi0"    ) genTauDecayMode = reco::PFTau::kOneProng2PiZero;
	else if ( genTauDecayMode_string == "threeProng0Pi0"  ) genTauDecayMode = reco::PFTau::kThreeProng0PiZero;
	else if ( genTauDecayMode_string == "threeProng1Pi0"  ) genTauDecayMode = reco::PFTau::kThreeProng1PiZero;
	else if ( genTauDecayMode_string == "oneProngOther"   ||
		  genTauDecayMode_string == "threeProngOther" ||
		  genTauDecayMode_string == "rare"            ) genTauDecayMode = reco::PFTau::kRareDecayMode;
	if ( genTauDecayMode == -1 ) continue; // skip leptonic tau decays
	double dR = deltaR(genParticle->p4(), recTau->p4());
	if ( dR < dRmin ) {
	  genTau_matched = &(*genParticle);
	  genVisTauP4_matched = genVisTauP4;
	  genTauDecayMode_matched = genTauDecayMode;
	}
      }
      setGenTauMatchValues(recTau->p4(), genTau_matched, genVisTauP4_matched, genTauDecayMode_matched);

      const reco::GenParticle* genMuon_matched = findMatchingGenParticle(recTau->p4(), *genParticles, minGenVisPt_, pdgIdsGenMuon_, dRgenParticleMatch_);
      setGenParticleMatchValues("genMuon", recTau->p4(), genMuon_matched);

      int numHypotheses = 0;
      if ( genTau_matched          ) ++numHypotheses;
      if ( genMuon_matched         ) ++numHypotheses;
      if ( numHypotheses > 1 ) 
	edm::LogWarning("AntiMuonDiscrMVATrainingNtupleProducer::analyze")
	  << " Matching between reconstructed PFTau and generator level tau-jets and muons is ambiguous !!";

      setValueI("run" ,evt.run());
      setValueI("event", (evt.eventAuxiliary()).event());
      setValueI("lumi", evt.luminosityBlock());

      for ( std::vector<vertexCollectionEntryType>::const_iterator vertexCollection = vertexCollectionEntries_.begin();
	    vertexCollection != vertexCollectionEntries_.end(); ++vertexCollection ) {
	edm::Handle<reco::VertexCollection> vertices;
	evt.getByLabel(vertexCollection->src_, vertices);
	setValueI(vertexCollection->branchName_multiplicity_, vertices->size());
	if ( vertices->size() >= 1 ) {
	  setValue_XYZ(vertexCollection->branchName_position_, vertices->front().position()); // CV: first entry is vertex with highest sum(trackPt), take as "the" event vertex
	} else {
	  setValue_XYZ(vertexCollection->branchName_position_, reco::Candidate::Point(0.,0.,0.));
	}
      }

      setValueF("evtWeight", evtWeight);

//--- fill all computed quantities into TTree
      assert(ntuple_);
      ntuple_->Fill();
    }
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::addBranchF(const std::string& name) 
{
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/F";
  ntuple_->Branch(name.c_str(), &branches_[name].valueF_, name_and_format.c_str());
}

void AntiMuonDiscrMVATrainingNtupleProducer::addBranchI(const std::string& name) 
{
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/I";
  ntuple_->Branch(name.c_str(), &branches_[name].valueI_, name_and_format.c_str());
}

void AntiMuonDiscrMVATrainingNtupleProducer::printBranches(std::ostream& stream)
{
  stream << "<AntiMuonDiscrMVATrainingNtupleProducer::printBranches>:" << std::endl;
  stream << " registered branches for module = " << moduleLabel_ << std::endl;
  for ( branchMap::const_iterator branch = branches_.begin();
	branch != branches_.end(); ++branch ) {
    stream << " " << branch->first << std::endl;
  }
  stream << std::endl;
}

void AntiMuonDiscrMVATrainingNtupleProducer::setValueF(const std::string& name, double value) 
{
  if ( verbosity_ ) std::cout << "branch = " << name << ": value = " << value << std::endl;
  branchMap::iterator branch = branches_.find(name);
  if ( branch != branches_.end() ) {
    branch->second.valueF_ = value;
  } else {
    throw cms::Exception("InvalidParameter") 
      << "No branch with name = " << name << " defined !!\n";
  }
}

void AntiMuonDiscrMVATrainingNtupleProducer::setValueI(const std::string& name, int value) 
{
  if ( verbosity_ ) std::cout << "branch = " << name << ": value = " << value << std::endl;
  branchMap::iterator branch = branches_.find(name);
  if ( branch != branches_.end() ) {
    branch->second.valueI_ = value;
  } else {
    throw cms::Exception("InvalidParameter") 
      << "No branch with name = " << name << " defined !!\n";
  }
}

//
//-------------------------------------------------------------------------------
//

void AntiMuonDiscrMVATrainingNtupleProducer::addBranch_EnPxPyPz(const std::string& name) 
{
  addBranchF(std::string(name).append("En"));
  addBranchF(std::string(name).append("P"));
  addBranchF(std::string(name).append("Px"));
  addBranchF(std::string(name).append("Py"));
  addBranchF(std::string(name).append("Pz"));
  addBranchF(std::string(name).append("M"));
  addBranchF(std::string(name).append("Eta"));
  addBranchF(std::string(name).append("Phi"));
  addBranchF(std::string(name).append("Pt"));
}

void AntiMuonDiscrMVATrainingNtupleProducer::addBranch_XYZ(const std::string& name)
{
  addBranchF(std::string(name).append("X"));
  addBranchF(std::string(name).append("Y"));
  addBranchF(std::string(name).append("Z"));
  addBranchF(std::string(name).append("R"));
  addBranchF(std::string(name).append("Mag"));
}

//
//-------------------------------------------------------------------------------
//

void AntiMuonDiscrMVATrainingNtupleProducer::setValue_EnPxPyPz(const std::string& name, const reco::Candidate::LorentzVector& p4)
{
  setValueF(std::string(name).append("En"), p4.E());
  setValueF(std::string(name).append("P"), p4.P());
  setValueF(std::string(name).append("Px"), p4.px());
  setValueF(std::string(name).append("Py"), p4.py());
  setValueF(std::string(name).append("Pz"), p4.pz());
  setValueF(std::string(name).append("M"), p4.M());
  setValueF(std::string(name).append("Eta"), p4.eta());
  setValueF(std::string(name).append("Phi"), p4.phi());
  setValueF(std::string(name).append("Pt"), p4.pt());
}

template <typename T>
void AntiMuonDiscrMVATrainingNtupleProducer::setValue_XYZ(const std::string& name, const T& pos)
{
  double x = pos.x();
  double y = pos.y();
  double z = pos.z();
  double r = TMath::Sqrt(x*x + y*y);
  double mag = TMath::Sqrt(r*r + z*z);
  setValueF(std::string(name).append("X"), x);
  setValueF(std::string(name).append("Y"), y);
  setValueF(std::string(name).append("Z"), z);
  setValueF(std::string(name).append("R"), r);
  setValueF(std::string(name).append("Mag"), mag);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AntiMuonDiscrMVATrainingNtupleProducer);
