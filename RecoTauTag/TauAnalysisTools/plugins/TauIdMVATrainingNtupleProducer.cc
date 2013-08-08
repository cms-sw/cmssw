#include "RecoTauTag/TauAnalysisTools/plugins/TauIdMVATrainingNtupleProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <TPRegexp.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TString.h>
#include <TMath.h>

#include <iostream>
#include <fstream>

TauIdMVATrainingNtupleProducer::TauIdMVATrainingNtupleProducer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    maxChargedHadrons_(3),
    maxPiZeros_(2),
    loosePFJetIdAlgo_(0),
    maxWarnings_(3),
    ntuple_(0)
{
  srcRecTaus_ = cfg.getParameter<edm::InputTag>("srcRecTaus");
  srcRecTauTransverseImpactParameters_ = cfg.getParameter<edm::InputTag>("srcRecTauTransverseImpactParameters");
  
  srcGenTauJets_ = cfg.getParameter<edm::InputTag>("srcGenTauJets");
  srcGenParticles_ = cfg.getParameter<edm::InputTag>("srcGenParticles");
  minGenVisPt_ = cfg.getParameter<double>("minGenVisPt");
  dRmatch_ = cfg.getParameter<double>("dRmatch");
  
  pdgIdsGenTau_.push_back(-15);
  pdgIdsGenTau_.push_back(+15);

  pdgIdsGenElectron_.push_back(-11);
  pdgIdsGenElectron_.push_back(+11);

  pdgIdsGenMuon_.push_back(-13);
  pdgIdsGenMuon_.push_back(+13);

  pdgIdsGenQuarkOrGluon_.push_back(-6);
  pdgIdsGenQuarkOrGluon_.push_back(-5);
  pdgIdsGenQuarkOrGluon_.push_back(-4);
  pdgIdsGenQuarkOrGluon_.push_back(-3);
  pdgIdsGenQuarkOrGluon_.push_back(-2);
  pdgIdsGenQuarkOrGluon_.push_back(-1);
  pdgIdsGenQuarkOrGluon_.push_back(+1);
  pdgIdsGenQuarkOrGluon_.push_back(+2);
  pdgIdsGenQuarkOrGluon_.push_back(+3);
  pdgIdsGenQuarkOrGluon_.push_back(+4);
  pdgIdsGenQuarkOrGluon_.push_back(+5);
  pdgIdsGenQuarkOrGluon_.push_back(+6);
  pdgIdsGenQuarkOrGluon_.push_back(+21);
  
  edm::ParameterSet tauIdDiscriminators = cfg.getParameter<edm::ParameterSet>("tauIdDiscriminators");
  typedef std::vector<std::string> vstring;
  vstring tauIdDiscriminatorNames = tauIdDiscriminators.getParameterNamesForType<edm::InputTag>();
  for ( vstring::const_iterator name = tauIdDiscriminatorNames.begin();
	name != tauIdDiscriminatorNames.end(); ++name ) {
    edm::InputTag src = tauIdDiscriminators.getParameter<edm::InputTag>(*name);
    tauIdDiscrEntries_.push_back(tauIdDiscrEntryType(*name, src));
  }
  
  edm::ParameterSet isolationPtSums = cfg.getParameter<edm::ParameterSet>("isolationPtSums");
  vstring isolationPtSumNames = isolationPtSums.getParameterNamesForType<edm::ParameterSet>();
  for ( vstring::const_iterator name = isolationPtSumNames.begin();
	name != isolationPtSumNames.end(); ++name ) {
    edm::ParameterSet cfgIsolationPtSum = isolationPtSums.getParameter<edm::ParameterSet>(*name);
    tauIsolationEntries_.push_back(tauIsolationEntryType(*name, cfgIsolationPtSum));
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
  if ( isMC_ ) {
    srcGenPileUpSummary_ = cfg.getParameter<edm::InputTag>("srcGenPileUpSummary");
  } else {
    edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileNameLumiCalc");
    if ( !inputFileName.isLocal()) 
      throw cms::Exception("UnclEnCalibrationNtupleProducer") 
	<< " Failed to find File = " << inputFileName << " !!\n";
    ifstream inputFile(inputFileName.fullPath().data());
    std::string header_pattern = std::string(
      "\\|\\s*Run:Fill\\s*\\|\\s*LS\\s*\\|\\s*UTCTime\\s*\\|\\s*Beam Status\\s*\\|\\s*E\\(GeV\\)\\s*\\|\\s*Del\\(/nb\\)\\s*\\|\\s*Rec\\(/nb\\)\\s*\\|\\s*avgPU\\s*\\|\\s*");
    TPRegexp header_regexp(header_pattern.data());
    std::string pileUpInfo_pattern = std::string(
      "\\|\\s*([0-9]+):[0-9]+\\s*\\|\\s*([0-9]+):[0-9]+\\s*\\|\\s*[0-9/: ]+\\s*\\|\\s*[a-zA-Z0-9 ]+\\s*\\|\\s*[0-9.]+\\s*\\|\\s*[0-9.]+\\s*\\|\\s*[0-9.]+\\s*\\|\\s*([0-9.]+)\\s*\\|\\s*");
    TPRegexp pileUpInfo_regexp(pileUpInfo_pattern.data());
    int iLine = 0;
    bool foundHeader = false;
    while ( !(inputFile.eof() || inputFile.bad()) ) {
      std::string line;
      getline(inputFile, line);
      ++iLine;
      TString line_tstring = line.data();
      if ( header_regexp.Match(line_tstring) == 1 ) foundHeader = true;
      if ( !foundHeader ) continue;
      TObjArray* subStrings = pileUpInfo_regexp.MatchS(line_tstring);
      if ( subStrings->GetEntries() == 4 ) {
	edm::RunNumber_t run = ((TObjString*)subStrings->At(1))->GetString().Atoll();
	edm::LuminosityBlockNumber_t ls = ((TObjString*)subStrings->At(2))->GetString().Atoll();
	float numPileUp_mean = ((TObjString*)subStrings->At(3))->GetString().Atof();
	//std::cout << "run = " << run << ", ls = " << ls << ": numPileUp_mean = " << numPileUp_mean << std::endl;
	pileUpByLumiCalc_[run][ls] = numPileUp_mean;
      }
    }
    if ( !foundHeader ) 
      throw cms::Exception("UnclEnCalibrationNtupleProducer") 
	<< " Failed to find header in File = " << inputFileName.fullPath().data() << " !!\n";
  }

  srcWeights_ = cfg.getParameter<vInputTag>("srcWeights");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

TauIdMVATrainingNtupleProducer::~TauIdMVATrainingNtupleProducer()
{
// nothing to be done yet...
}

void TauIdMVATrainingNtupleProducer::beginJob()
{
//--- create TTree
  edm::Service<TFileService> fs;
  ntuple_ = fs->make<TTree>("tauIdMVATrainingNtuple", "tauIdMVATrainingNtuple");

//--- add branches 
  addBranchI("run");
  addBranchI("event");
  addBranchI("lumi");

  addBranch_EnPxPyPz("recTau");
  addBranch_EnPxPyPz("recTauAlternate");
  addBranchI("recTauDecayMode");
  addBranchF("recTauVtxZ");
  addBranch_EnPxPyPz("recJet");
  addBranchI("recJetLooseId");
  addBranch_EnPxPyPz("leadPFCand");
  addBranch_EnPxPyPz("leadPFChargedHadrCand");  
  for ( unsigned idx = 0; idx < maxChargedHadrons_; ++idx ) {
    addBranch_chargedHadron(Form("chargedHadron%i", idx + 1));
  }
  for ( unsigned idx = 0; idx < maxPiZeros_; ++idx ) {
    addBranch_piZero(Form("piZero%i", idx + 1));
  }
  for ( std::vector<tauIdDiscrEntryType>::const_iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
    addBranchF(tauIdDiscriminator->branchName_);
  }
  for ( std::vector<tauIsolationEntryType>::const_iterator tauIsolation = tauIsolationEntries_.begin();
	tauIsolation != tauIsolationEntries_.end(); ++tauIsolation ) {
    addBranchF(tauIsolation->branchNameChargedIsoPtSum_);
    addBranchF(tauIsolation->branchNameNeutralIsoPtSum_);
    addBranchF(tauIsolation->branchNamePUcorrPtSum_);    
  }
  addBranch_XYZ("recImpactParamPCA");
  addBranchF("recImpactParam");
  addBranchF("recImpactParamSign");
  addBranchI("hasRecDecayVertex");
  addBranch_XYZ("recDecayVertex");
  addBranch_Cov("recDecayVertexCov");
  addBranch_XYZ("recDecayDist");
  addBranch_Cov("recDecayDistCov");
  addBranchF("recDecayDistSign");
  addBranch_XYZ("recEvtVertex");
  addBranch_Cov("recEvtVertexCov");
  for ( std::vector<vertexCollectionEntryType>::const_iterator vertexCollection = vertexCollectionEntries_.begin();
	vertexCollection != vertexCollectionEntries_.end(); ++vertexCollection ) {
    addBranchI(vertexCollection->branchName_multiplicity_);
    addBranch_XYZ(vertexCollection->branchName_position_);
  }
  addBranchF("numPileUp");
  addBranch_EnPxPyPz("genTau");
  addBranchF("genTauDeltaR");
  addBranch_EnPxPyPz("genVisTau");
  addBranchF("genVisTauDeltaR");
  addBranchI("genTauDecayMode");
  addBranchI("genTauMatch");
  addBranchF("genImpactParam");
  addBranch_XYZ("genDecayVertex");
  addBranch_XYZ("genEvtVertex");
  addBranch_EnPxPyPz("genElectron");
  addBranchI("genElectronMatch");
  addBranchF("genElectronDeltaR");
  addBranchI("genElectronPdgId");
  addBranch_EnPxPyPz("genMuon");
  addBranchI("genMuonMatch");
  addBranchF("genMuonDeltaR");
  addBranchI("genMuonPdgId");
  addBranch_EnPxPyPz("genQuarkOrGluon");
  addBranchI("genQuarkOrGluonMatch");
  addBranchF("genQuarkOrGluonDeltaR");
  addBranchI("genQuarkOrGluonPdgId");
  addBranchF("evtWeight");
}

void TauIdMVATrainingNtupleProducer::setRecTauValues(const reco::PFTauRef& recTau, const edm::Event& evt)
{
  setValue_EnPxPyPz("recTau", recTau->p4());
  setValue_EnPxPyPz("recTauAlternate", recTau->alternatLorentzVect());
  setValueI("recTauDecayMode", recTau->decayMode());
  setValueF("recTauVtxZ", recTau->vertex().z());
  setValue_EnPxPyPz("recJet", recTau->jetRef()->p4());
  int recJetLooseId = ( (*loosePFJetIdAlgo_)(*recTau->jetRef()) ) ? 1 : 0;
  setValueI("recJetLooseId", recJetLooseId);
  if ( recTau->leadPFCand().isNonnull() ) setValue_EnPxPyPz("leadPFCand", recTau->leadPFCand()->p4());
  else setValue_EnPxPyPz("leadPFCand", reco::Candidate::LorentzVector(0.,0.,0.,0.));
  if ( recTau->leadPFChargedHadrCand().isNonnull() ) setValue_EnPxPyPz("leadPFChargedHadrCand", recTau->leadPFChargedHadrCand()->p4());
  else setValue_EnPxPyPz("leadPFChargedHadrCand", reco::Candidate::LorentzVector(0.,0.,0.,0.));
  for ( unsigned idx = 0; idx < maxChargedHadrons_; ++idx ) {
    std::string branchName = Form("chargedHadron%i", idx + 1);
    if ( recTau->signalTauChargedHadronCandidates().size() > idx ) setValue_chargedHadron(branchName, &recTau->signalTauChargedHadronCandidates().at(idx));
    else setValue_chargedHadron(branchName, 0);
  }
  for ( unsigned idx = 0; idx < maxPiZeros_; ++idx ) {
    std::string branchName = Form("piZero%i", idx + 1);
    if ( recTau->signalPiZeroCandidates().size() > idx ) setValue_piZero(branchName, &recTau->signalPiZeroCandidates().at(idx));
    else setValue_piZero(branchName, 0);
  }
  typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > PFTauTIPAssociationByRef;
  edm::Handle<PFTauTIPAssociationByRef> recTauLifetimeInfos;
  evt.getByLabel(srcRecTauTransverseImpactParameters_, recTauLifetimeInfos);
  const reco::PFTauTransverseImpactParameter& recTauLifetimeInfo = *(*recTauLifetimeInfos)[recTau];
  setValue_XYZ("recImpactParamPCA", recTauLifetimeInfo.dxy_PCA());
  setValueF("recImpactParam", recTauLifetimeInfo.dxy());
  setValueF("recImpactParamSign", recTauLifetimeInfo.dxy_Sig());
  setValueI("hasRecDecayVertex", recTauLifetimeInfo.hasSecondaryVertex());
  setValue_XYZ("recDecayVertex", recTauLifetimeInfo.secondaryVertexPos());
  setValue_Cov("recDecayVertexCov", recTauLifetimeInfo.secondaryVertexCov());
  setValue_XYZ("recDecayDist", recTauLifetimeInfo.flightLength());
  setValue_Cov("recDecayDistCov", recTauLifetimeInfo.flightLengthCov());
  setValueF("recDecayDistSign", recTauLifetimeInfo.flightLengthSig());
  setValue_XYZ("recEvtVertex", recTauLifetimeInfo.primaryVertexPos());
  setValue_Cov("recEvtVertexCov", recTauLifetimeInfo.primaryVertexCov());
  for ( std::vector<tauIdDiscrEntryType>::const_iterator tauIdDiscriminator = tauIdDiscrEntries_.begin();
	tauIdDiscriminator != tauIdDiscrEntries_.end(); ++tauIdDiscriminator ) {
    edm::Handle<reco::PFTauDiscriminator> discriminator;
    evt.getByLabel(tauIdDiscriminator->src_, discriminator);
    setValueF(tauIdDiscriminator->branchName_, (*discriminator)[recTau]);
  }
  for ( std::vector<tauIsolationEntryType>::const_iterator tauIsolation = tauIsolationEntries_.begin();
	tauIsolation != tauIsolationEntries_.end(); ++tauIsolation ) {
    edm::Handle<reco::PFTauDiscriminator> chargedIsoPtSum;
    evt.getByLabel(tauIsolation->srcChargedIsoPtSum_, chargedIsoPtSum);
    setValueF(tauIsolation->branchNameChargedIsoPtSum_, (*chargedIsoPtSum)[recTau]);
    edm::Handle<reco::PFTauDiscriminator> neutralIsoPtSum;
    evt.getByLabel(tauIsolation->srcNeutralIsoPtSum_, neutralIsoPtSum);
    setValueF(tauIsolation->branchNameNeutralIsoPtSum_, (*neutralIsoPtSum)[recTau]);
    edm::Handle<reco::PFTauDiscriminator> puCorrPtSum;
    evt.getByLabel(tauIsolation->srcPUcorrPtSum_, puCorrPtSum);
    setValueF(tauIsolation->branchNamePUcorrPtSum_, (*puCorrPtSum)[recTau]);
  }
}

void TauIdMVATrainingNtupleProducer::setGenTauMatchValues(
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

void TauIdMVATrainingNtupleProducer::setGenParticleMatchValues(const std::string& branchName, const reco::Candidate::LorentzVector& recTauP4, const reco::GenParticle* genParticle)
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

  const reco::GenParticle* getGenLeadChargedDecayProduct(const reco::GenParticle* genTau) 
  {    
    std::vector<const reco::GenParticle*> genTauDecayProducts;
    findDaughters(genTau, genTauDecayProducts, 1);
    const reco::GenParticle* genLeadChargedDecayProduct = 0;
    double genLeadChargedDecayProductPt = -1.;
    for ( std::vector<const reco::GenParticle*>::const_iterator genTauDecayProduct = genTauDecayProducts.begin();
	  genTauDecayProduct != genTauDecayProducts.end(); ++genTauDecayProduct ) {
      if ( TMath::Abs((*genTauDecayProduct)->charge()) > 0.5 && (*genTauDecayProduct)->pt() > genLeadChargedDecayProductPt ) {
	genLeadChargedDecayProduct = (*genTauDecayProduct);
	genLeadChargedDecayProductPt = (*genTauDecayProduct)->pt();
      }
    }    
    return genLeadChargedDecayProduct;
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

void TauIdMVATrainingNtupleProducer::setNumPileUpValue(const edm::Event& evt)
{
  double numPileUp_mean = -1.;
  if ( isMC_ ) {
    typedef std::vector<PileupSummaryInfo> PileupSummaryInfoCollection;
    edm::Handle<PileupSummaryInfoCollection> genPileUpInfos;
    evt.getByLabel(srcGenPileUpSummary_, genPileUpInfos);
    for ( PileupSummaryInfoCollection::const_iterator genPileUpInfo = genPileUpInfos->begin();
	  genPileUpInfo != genPileUpInfos->end(); ++genPileUpInfo ) {
      // CV: in-time PU is stored in getBunchCrossing = 0, 
      //    cf. https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupInformation
      int bx = genPileUpInfo->getBunchCrossing();
      if ( bx == 0 ) {
	numPileUp_mean = genPileUpInfo->getTrueNumInteractions();
      } 
    }
  } else {
    edm::RunNumber_t run = evt.id().run();
    edm::LuminosityBlockNumber_t ls = evt.luminosityBlock();
    if ( pileUpByLumiCalc_.find(run) == pileUpByLumiCalc_.end() || pileUpByLumiCalc_[run].find(ls) == pileUpByLumiCalc_[run].end() ) {
      if ( numWarnings_[run][ls] < maxWarnings_ ) {
	edm::LogWarning("TauIdMVATrainingNtupleProducer") 
	  << "No inst. Luminosity information available for run = " << run << ", ls = " << ls << " --> skipping !!" << std::endl;
      }
      ++numWarnings_[run][ls];
      return;
    }
    numPileUp_mean = pileUpByLumiCalc_[run][ls];
  }
  setValueF("numPileUp", numPileUp_mean);
}

void TauIdMVATrainingNtupleProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  assert(ntuple_);

  edm::Handle<reco::PFTauCollection> recTaus;
  evt.getByLabel(srcRecTaus_, recTaus);

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
  for ( size_t iRecTau = 0; iRecTau < numRecTaus; ++iRecTau ) {
    reco::PFTauRef recTau(recTaus, iRecTau);
    setRecTauValues(recTau, evt);

    const reco::GenParticle* genTau_matched = 0;
    reco::Candidate::LorentzVector genVisTauP4_matched(0.,0.,0.,0.);
    int genTauDecayMode_matched = -1;
    if ( isMC_ ) {      
      double dRmin = dRmatch_;
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

      if ( genTau_matched ) {
	reco::Candidate::Point genEvtVertex = genTau_matched->vertex();
	const reco::GenParticle* genLeadChargedHadron = getGenLeadChargedDecayProduct(genTau_matched);
	assert(genLeadChargedHadron);
	reco::Candidate::Point genDecayVertex = genLeadChargedHadron->vertex();
	double flightPathPx = genDecayVertex.x() - genEvtVertex.x();
	double flightPathPy = genDecayVertex.y() - genEvtVertex.y();
	double genImpactParam = TMath::Abs(flightPathPx*genLeadChargedHadron->py() - flightPathPy*genLeadChargedHadron->px())/genLeadChargedHadron->pt();
	setValueF("genImpactParam", genImpactParam);
	setValue_XYZ("genDecayVertex", genDecayVertex);
	setValue_XYZ("genEvtVertex", genDecayVertex);
      } else {
	setValueF("genImpactParam", -1.);
	setValue_XYZ("genDecayVertex", reco::Candidate::Point(0.,0.,0.));
	setValue_XYZ("genEvtVertex", reco::Candidate::Point(0.,0.,0.));
      }

      const reco::GenParticle* genElectron_matched = findMatchingGenParticle(recTau->p4(), *genParticles, minGenVisPt_, pdgIdsGenElectron_, dRmatch_);
      setGenParticleMatchValues("genElectron", recTau->p4(), genElectron_matched);

      const reco::GenParticle* genMuon_matched = findMatchingGenParticle(recTau->p4(), *genParticles, minGenVisPt_, pdgIdsGenMuon_, dRmatch_);
      setGenParticleMatchValues("genMuon", recTau->p4(), genMuon_matched);

      const reco::GenParticle* genQuarkOrGluon_matched = findMatchingGenParticle(recTau->p4(), *genParticles, minGenVisPt_, pdgIdsGenQuarkOrGluon_, dRmatch_);
      setGenParticleMatchValues("genQuarkOrGluon", recTau->p4(), genQuarkOrGluon_matched);

      int numHypotheses = 0;
      if ( genTau_matched          ) ++numHypotheses;
      if ( genElectron_matched     ) ++numHypotheses;
      if ( genMuon_matched         ) ++numHypotheses;
      if ( genQuarkOrGluon_matched ) ++numHypotheses;
      if ( numHypotheses > 1 ) 
	edm::LogWarning("TauIdMVATrainingNtupleProducer::analyze")
	  << " Matching between reconstructed PFTau and generator level tau-jets, electrons, muons and quark/gluon jets is ambiguous !!";

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

      setNumPileUpValue(evt);

      setValueF("evtWeight", evtWeight);

//--- fill all computed quantities into TTree
      assert(ntuple_);
      ntuple_->Fill();
    }
  }
}

void TauIdMVATrainingNtupleProducer::addBranchF(const std::string& name) 
{
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/F";
  ntuple_->Branch(name.c_str(), &branches_[name].valueF_, name_and_format.c_str());
}

void TauIdMVATrainingNtupleProducer::addBranchI(const std::string& name) 
{
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/I";
  ntuple_->Branch(name.c_str(), &branches_[name].valueI_, name_and_format.c_str());
}

void TauIdMVATrainingNtupleProducer::printBranches(std::ostream& stream)
{
  stream << "<TauIdMVATrainingNtupleProducer::printBranches>:" << std::endl;
  stream << " registered branches for module = " << moduleLabel_ << std::endl;
  for ( branchMap::const_iterator branch = branches_.begin();
	branch != branches_.end(); ++branch ) {
    stream << " " << branch->first << std::endl;
  }
  stream << std::endl;
}

void TauIdMVATrainingNtupleProducer::setValueF(const std::string& name, double value) 
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

void TauIdMVATrainingNtupleProducer::setValueI(const std::string& name, int value) 
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

void TauIdMVATrainingNtupleProducer::addBranch_EnPxPyPz(const std::string& name) 
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

void TauIdMVATrainingNtupleProducer::addBranch_XYZ(const std::string& name)
{
  addBranchF(std::string(name).append("X"));
  addBranchF(std::string(name).append("Y"));
  addBranchF(std::string(name).append("Z"));
  addBranchF(std::string(name).append("R"));
  addBranchF(std::string(name).append("Mag"));
}

void TauIdMVATrainingNtupleProducer::addBranch_Cov(const std::string& name)
{
  addBranchF(std::string(name).append("Cxx"));
  addBranchF(std::string(name).append("Cxy"));
  addBranchF(std::string(name).append("Cxz"));
  addBranchF(std::string(name).append("Cyy"));
  addBranchF(std::string(name).append("Cyz"));
  addBranchF(std::string(name).append("Czz"));
}

void TauIdMVATrainingNtupleProducer::addBranch_chargedHadron(const std::string& name)
{
  addBranch_EnPxPyPz(name);
  addBranchI(std::string(name).append("Algo"));
}

void TauIdMVATrainingNtupleProducer::addBranch_piZero(const std::string& name)
{
  addBranch_EnPxPyPz(name);
  addBranchI(std::string(name).append("NumPFGammas"));
  addBranchI(std::string(name).append("NumPFElectrons"));
  addBranchF(std::string(name).append("MaxDeltaEta"));
  addBranchF(std::string(name).append("MaxDeltaPhi"));
}

//
//-------------------------------------------------------------------------------
//

void TauIdMVATrainingNtupleProducer::setValue_EnPxPyPz(const std::string& name, const reco::Candidate::LorentzVector& p4)
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
void TauIdMVATrainingNtupleProducer::setValue_XYZ(const std::string& name, const T& pos)
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

void TauIdMVATrainingNtupleProducer::setValue_Cov(const std::string& name, const reco::PFTauTransverseImpactParameter::CovMatrix& cov)
{
  setValueF(std::string(name).append("Cxx"), cov(0,0));
  setValueF(std::string(name).append("Cxy"), cov(0,1));
  setValueF(std::string(name).append("Cxz"), cov(0,2));
  setValueF(std::string(name).append("Cyy"), cov(1,1));
  setValueF(std::string(name).append("Cyz"), cov(1,2));
  setValueF(std::string(name).append("Czz"), cov(2,2));
}

void TauIdMVATrainingNtupleProducer::setValue_chargedHadron(const std::string& name, const reco::PFRecoTauChargedHadron* chargedHadron)
{
  if ( chargedHadron ) {
    setValue_EnPxPyPz(name, chargedHadron->p4());
    setValueI(std::string(name).append("Algo"), chargedHadron->algo());
  } else {
    setValue_EnPxPyPz(name, reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueI(std::string(name).append("Algo"), -1);
  }
}

void TauIdMVATrainingNtupleProducer::setValue_piZero(const std::string& name, const reco::RecoTauPiZero* piZero)
{
  if ( piZero ) {
    setValue_EnPxPyPz(name, piZero->p4());
    setValueI(std::string(name).append("NumPFGammas"), piZero->numberOfGammas());
    setValueI(std::string(name).append("NumPFElectrons"), piZero->numberOfElectrons());
    setValueF(std::string(name).append("MaxDeltaEta"), piZero->maxDeltaEta());
    setValueF(std::string(name).append("MaxDeltaPhi"), piZero->maxDeltaPhi());
  } else {
    setValue_EnPxPyPz(name, reco::Candidate::LorentzVector(0.,0.,0.,0.));
    setValueI(std::string(name).append("NumPFGammas"), 0);
    setValueI(std::string(name).append("NumPFElectrons"), 0);
    setValueF(std::string(name).append("MaxDeltaEta"), 0.);
    setValueF(std::string(name).append("MaxDeltaPhi"), 0.);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauIdMVATrainingNtupleProducer);
