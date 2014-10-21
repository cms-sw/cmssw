//
//

#include "PhysicsTools/PatAlgos/plugins/PATTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"


#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <memory>

using namespace pat;

PATTauProducer::PATTauProducer(const edm::ParameterSet & iConfig):
  isolator_(iConfig.exists("userIsolation") ? iConfig.getParameter<edm::ParameterSet>("userIsolation") : edm::ParameterSet(), consumesCollector(), false) ,
  useUserData_(iConfig.exists("userData"))
{
  // initialize the configurables
  baseTauToken_ = consumes<edm::View<reco::BaseTau> >(iConfig.getParameter<edm::InputTag>( "tauSource" ));
  tauTransverseImpactParameterSrc_ = iConfig.getParameter<edm::InputTag>( "tauTransverseImpactParameterSource" );
  tauTransverseImpactParameterToken_ = consumes<PFTauTIPAssociationByRef>( tauTransverseImpactParameterSrc_);
  pfTauToken_ = consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>( "tauSource" ));
  caloTauToken_ = mayConsume<reco::CaloTauCollection>(iConfig.getParameter<edm::InputTag>( "tauSource" ));
  embedIsolationTracks_ = iConfig.getParameter<bool>( "embedIsolationTracks" );
  embedLeadTrack_ = iConfig.getParameter<bool>( "embedLeadTrack" );
  embedSignalTracks_ = iConfig.getParameter<bool>( "embedSignalTracks" );
  embedLeadPFCand_ = iConfig.getParameter<bool>( "embedLeadPFCand" );
  embedLeadPFChargedHadrCand_ = iConfig.getParameter<bool>( "embedLeadPFChargedHadrCand" );
  embedLeadPFNeutralCand_ = iConfig.getParameter<bool>( "embedLeadPFNeutralCand" );
  embedSignalPFCands_ = iConfig.getParameter<bool>( "embedSignalPFCands" );
  embedSignalPFChargedHadrCands_ = iConfig.getParameter<bool>( "embedSignalPFChargedHadrCands" );
  embedSignalPFNeutralHadrCands_ = iConfig.getParameter<bool>( "embedSignalPFNeutralHadrCands" );
  embedSignalPFGammaCands_ = iConfig.getParameter<bool>( "embedSignalPFGammaCands" );
  embedIsolationPFCands_ = iConfig.getParameter<bool>( "embedIsolationPFCands" );
  embedIsolationPFChargedHadrCands_ = iConfig.getParameter<bool>( "embedIsolationPFChargedHadrCands" );
  embedIsolationPFNeutralHadrCands_ = iConfig.getParameter<bool>( "embedIsolationPFNeutralHadrCands" );
  embedIsolationPFGammaCands_ = iConfig.getParameter<bool>( "embedIsolationPFGammaCands" );
  addGenMatch_ = iConfig.getParameter<bool>( "addGenMatch" );
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>( "embedGenMatch" );
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection> >(iConfig.getParameter<edm::InputTag>( "genParticleMatch" )));
    }
    else {
      genMatchTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" ), [this](edm::InputTag const & tag){return consumes<edm::Association<reco::GenParticleCollection> >(tag);});
    }
  }
  addGenJetMatch_ = iConfig.getParameter<bool>( "addGenJetMatch" );
  if(addGenJetMatch_) {
    embedGenJetMatch_ = iConfig.getParameter<bool>( "embedGenJetMatch" );
    genJetMatchToken_ = consumes<edm::Association<reco::GenJetCollection> >(iConfig.getParameter<edm::InputTag>( "genJetMatch" ));
  }
  addTauJetCorrFactors_ = iConfig.getParameter<bool>( "addTauJetCorrFactors" );
  tauJetCorrFactorsTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "tauJetCorrFactorsSource" ), [this](edm::InputTag const & tag){return mayConsume<edm::ValueMap<TauJetCorrFactors> >(tag);});
  // tau ID configurables
  addTauID_ = iConfig.getParameter<bool>( "addTauID" );
  if ( addTauID_ ) {
    // it might be a single tau ID
    if (iConfig.existsAs<edm::InputTag>("tauIDSource")) {
      tauIDSrcs_.push_back(NameTag("", iConfig.getParameter<edm::InputTag>("tauIDSource")));
    }
    // or there might be many of them
    if (iConfig.existsAs<edm::ParameterSet>("tauIDSources")) {
      // please don't configure me twice
      if (!tauIDSrcs_.empty()){
	throw cms::Exception("Configuration") << "PATTauProducer: you can't specify both 'tauIDSource' and 'tauIDSources'\n";
      }
      // read the different tau ID names
      edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("tauIDSources");
      std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
      for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
	tauIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
      }
    }
    // but in any case at least once
    if (tauIDSrcs_.empty()) throw cms::Exception("Configuration") <<
      "PATTauProducer: id addTauID is true, you must specify either:\n" <<
      "\tInputTag tauIDSource = <someTag>\n" << "or\n" <<
      "\tPSet tauIDSources = { \n" <<
      "\t\tInputTag <someName> = <someTag>   // as many as you want \n " <<
      "\t}\n";
  }
  caloTauIDTokens_ = edm::vector_transform(tauIDSrcs_, [this](NameTag const & tag){return mayConsume<reco::CaloTauDiscriminator>(tag.second);});
  pfTauIDTokens_   = edm::vector_transform(tauIDSrcs_, [this](NameTag const & tag){return mayConsume<reco::PFTauDiscriminator>(tag.second);});
  // IsoDeposit configurables
  if (iConfig.exists("isoDeposits")) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
    if ( depconf.exists("tracker")         ) isoDepositLabels_.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
    if ( depconf.exists("ecal")            ) isoDepositLabels_.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if ( depconf.exists("hcal")            ) isoDepositLabels_.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if ( depconf.exists("pfAllParticles")  ) isoDepositLabels_.push_back(std::make_pair(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    if ( depconf.exists("pfChargedHadron") ) isoDepositLabels_.push_back(std::make_pair(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadron")));
    if ( depconf.exists("pfNeutralHadron") ) isoDepositLabels_.push_back(std::make_pair(pat::PfNeutralHadronIso,depconf.getParameter<edm::InputTag>("pfNeutralHadron")));
    if ( depconf.exists("pfGamma")         ) isoDepositLabels_.push_back(std::make_pair(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfGamma")));

    if ( depconf.exists("user") ) {
      std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
      std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
      int key = UserBaseIso;
      for ( ; it != ed; ++it, ++key) {
       isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
      }
    }
  }
  isoDepositTokens_ = edm::vector_transform(isoDepositLabels_, [this](std::pair<IsolationKeys,edm::InputTag> const & label){return consumes<edm::ValueMap<IsoDeposit> >(label.second);});
  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }
  // Resolution configurables
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
     resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }
  // Check to see if the user wants to add user data
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Tau>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // produces vector of taus
  produces<std::vector<Tau> >();
}

PATTauProducer::~PATTauProducer()
{
}

void PATTauProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()){
    addGenMatch_    = false;
    embedGenMatch_  = false;
    addGenJetMatch_ = false;
  }

  // Get the collection of taus from the event
  edm::Handle<edm::View<reco::BaseTau> > anyTaus;
  try {
    iEvent.getByToken(baseTauToken_, anyTaus);
  } catch (const edm::Exception &e) {
    edm::LogWarning("DataSource") << "WARNING! No Tau collection found. This missing input will not block the job. Instead, an empty tau collection is being be produced.";
    std::auto_ptr<std::vector<Tau> > patTaus(new std::vector<Tau>());
    iEvent.put(patTaus);
    return;
  }

  if (isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositTokens_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByToken(isoDepositTokens_[j], deposits[j]);
  }

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > >genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

  edm::Handle<edm::Association<reco::GenJetCollection> > genJetMatch;
  if (addGenJetMatch_) iEvent.getByToken(genJetMatchToken_, genJetMatch);

  // read in the jet correction factors ValueMap
  std::vector<edm::ValueMap<TauJetCorrFactors> > tauJetCorrs;
  if (addTauJetCorrFactors_) {
    for ( size_t i = 0; i < tauJetCorrFactorsTokens_.size(); ++i ) {
      edm::Handle<edm::ValueMap<TauJetCorrFactors> > tauJetCorr;
      iEvent.getByToken(tauJetCorrFactorsTokens_[i], tauJetCorr);
      tauJetCorrs.push_back( *tauJetCorr );
    }
  }

  std::auto_ptr<std::vector<Tau> > patTaus(new std::vector<Tau>());

  bool first=true; // this is introduced to issue warnings only for the first tau-jet
  for (size_t idx = 0, ntaus = anyTaus->size(); idx < ntaus; ++idx) {
    edm::RefToBase<reco::BaseTau> tausRef = anyTaus->refAt(idx);
    edm::Ptr<reco::BaseTau> tausPtr = anyTaus->ptrAt(idx);

    Tau aTau(tausRef);
    if (embedLeadTrack_)       aTau.embedLeadTrack();
    if (embedSignalTracks_)    aTau.embedSignalTracks();
    if (embedIsolationTracks_) aTau.embedIsolationTracks();
    if (embedLeadPFCand_) {
      if (aTau.isPFTau() )
	aTau.embedLeadPFCand();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedLeadPFChargedHadrCand_) {
      if (aTau.isPFTau() )
	aTau.embedLeadPFChargedHadrCand();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedLeadPFNeutralCand_) {
      if (aTau.isPFTau() )
	aTau.embedLeadPFNeutralCand();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFCands_) {
      if (aTau.isPFTau() )
	aTau.embedSignalPFCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFChargedHadrCands_) {
      if (aTau.isPFTau() )
	aTau.embedSignalPFChargedHadrCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFNeutralHadrCands_) {
      if (aTau.isPFTau() )
	aTau.embedSignalPFNeutralHadrCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFGammaCands_) {
      if (aTau.isPFTau() )
	aTau.embedSignalPFGammaCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFCands_) {
      if (aTau.isPFTau() )
	aTau.embedIsolationPFCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFChargedHadrCands_) {
      if (aTau.isPFTau() )
	aTau.embedIsolationPFChargedHadrCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFNeutralHadrCands_) {
      if (aTau.isPFTau() )
	aTau.embedIsolationPFNeutralHadrCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFGammaCands_) {
      if (aTau.isPFTau() )
	aTau.embedIsolationPFGammaCands();
      else
	edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made from a reco::PFTau is impossible.\n";
    }

    if (addTauJetCorrFactors_) {
      // add additional JetCorrs to the jet
      for ( unsigned int i=0; i<tauJetCorrs.size(); ++i ) {
	const TauJetCorrFactors& tauJetCorr = tauJetCorrs[i][tausRef];
	// uncomment for debugging
	// tauJetCorr.print();
	aTau.addJECFactors(tauJetCorr);
      }
      std::vector<std::string> levels = tauJetCorrs[0][tausRef].correctionLabels();
      if(std::find(levels.begin(), levels.end(), "L2L3Residual")!=levels.end()){
	aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("L2L3Residual"));
      }
      else if(std::find(levels.begin(), levels.end(), "L3Absolute")!=levels.end()){
	aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("L3Absolute"));
      }
      else{
	aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("Uncorrected"));
	if(first){
	  edm::LogWarning("L3Absolute not found")
	    << "L2L3Residual and L3Absolute are not part of the correction applied jetCorrFactors \n"
	    << "of module " <<  tauJetCorrs[0][tausRef].jecSet() << " jets will remain"
	    << " uncorrected.";
	  first=false;
	}
      }
    }

    // store the match to the generated final state muons
    if (addGenMatch_) {
      for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
          reco::GenParticleRef genTau = (*genMatches[i])[tausRef];
          aTau.addGenParticleRef(genTau);
      }
      if (embedGenMatch_) aTau.embedGenParticle();
    }

    // store the match to the visible part of the generated tau
    if (addGenJetMatch_) {
      reco::GenJetRef genJetTau = (*genJetMatch)[tausRef];
      if (genJetTau.isNonnull() && genJetTau.isAvailable() ) {
        aTau.setGenJet( genJetTau );
      } // leave empty if no match found
    }

    // prepare ID extraction
    if ( addTauID_ ) {
      std::vector<pat::Tau::IdPair> ids(tauIDSrcs_.size());
      for ( size_t i = 0; i < tauIDSrcs_.size(); ++i ) {
	if ( typeid(*tausRef) == typeid(reco::PFTau) ) {
	  //std::cout << "filling PFTauDiscriminator '" << tauIDSrcs_[i].first << "' into pat::Tau object..." << std::endl;
	  edm::Handle<reco::PFTauCollection> pfTauCollection;
	  iEvent.getByToken(pfTauToken_, pfTauCollection);

	  edm::Handle<reco::PFTauDiscriminator> pfTauIdDiscr;
	  iEvent.getByToken(pfTauIDTokens_[i], pfTauIdDiscr);

	  ids[i].first = tauIDSrcs_[i].first;
	  ids[i].second = getTauIdDiscriminator(pfTauCollection, idx, pfTauIdDiscr);
	} else if ( typeid(*tausRef) == typeid(reco::CaloTau) ) {
	  //std::cout << "filling CaloTauDiscriminator '" << tauIDSrcs_[i].first << "' into pat::Tau object..." << std::endl;
	  edm::Handle<reco::CaloTauCollection> caloTauCollection;
	  iEvent.getByToken(caloTauToken_, caloTauCollection);

	  edm::Handle<reco::CaloTauDiscriminator> caloTauIdDiscr;
	  iEvent.getByToken(caloTauIDTokens_[i], caloTauIdDiscr);

	  ids[i].first = tauIDSrcs_[i].first;
	  ids[i].second = getTauIdDiscriminator(caloTauCollection, idx, caloTauIdDiscr);
	} else {
	  throw cms::Exception("Type Mismatch") <<
	    "PATTauProducer: unsupported datatype '" << typeid(*tausRef).name() << "' for tauSource\n";
	}
      }

      aTau.setTauIDs(ids);
    }

    // extraction of reconstructed tau decay mode
    // (only available for PFTaus)
    if ( aTau.isPFTau() ) {
      edm::Handle<reco::PFTauCollection> pfTaus;
      iEvent.getByToken(pfTauToken_, pfTaus);
      reco::PFTauRef pfTauRef(pfTaus, idx);

      aTau.setDecayMode(pfTauRef->decayMode());
    }

    // extraction of tau lifetime information
    // (only available for PFTaus)
    if ( aTau.isPFTau() && tauTransverseImpactParameterSrc_.label() != "" ) {
      edm::Handle<reco::PFTauCollection> pfTaus;
      iEvent.getByToken(pfTauToken_, pfTaus);
      reco::PFTauRef pfTauRef(pfTaus, idx);
      edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos;
      iEvent.getByToken(tauTransverseImpactParameterToken_, tauLifetimeInfos);
      const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos)[pfTauRef];
      pat::tau::TauPFEssential& aTauPFEssential = aTau.pfEssential_[0];
      aTauPFEssential.dxy_PCA_ = tauLifetimeInfo.dxy_PCA();
      aTauPFEssential.dxy_ = tauLifetimeInfo.dxy();
      aTauPFEssential.dxy_error_ = tauLifetimeInfo.dxy_error();
      //      aTauPFEssential.pv_ = tauLifetimeInfo.primaryVertex();
      // aTauPFEssential.pvPos_ = tauLifetimeInfo.primaryVertexPos();
      // aTauPFEssential.pvCov_ = tauLifetimeInfo.primaryVertexCov();
      aTauPFEssential.hasSV_ = tauLifetimeInfo.hasSecondaryVertex();
      if(tauLifetimeInfo.hasSecondaryVertex()){
	aTauPFEssential.flightLength_ = tauLifetimeInfo.flightLength();
	aTauPFEssential.flightLengthSig_ = tauLifetimeInfo.flightLengthSig();
	//      aTauPFEssential.sv_ = tauLifetimeInfo.secondaryVertex();
	// aTauPFEssential.svPos_ = tauLifetimeInfo.secondaryVertexPos();
	// aTauPFEssential.svCov_ = tauLifetimeInfo.secondaryVertexCov();
      }
    }

    // Isolation
    if (isolator_.enabled()) {
      isolator_.fill(*anyTaus, idx, isolatorTmpStorage_);
      typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
      // better to loop backwards, so the vector is resized less times
      for ( IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(),
	      ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
	aTau.setIsolation(it->first, it->second);
      }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
      aTau.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[tausRef]);
    }

    if (efficiencyLoader_.enabled()) {
      efficiencyLoader_.setEfficiencies( aTau, tausRef );
    }

    if (resolutionLoader_.enabled()) {
      resolutionLoader_.setResolutions(aTau);
    }

    if ( useUserData_ ) {
      userDataHelper_.add( aTau, iEvent, iSetup );
    }

    patTaus->push_back(aTau);
  }

  // sort taus in pT
  std::sort(patTaus->begin(), patTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  iEvent.put(patTaus);

  // clean up
  if (isolator_.enabled()) isolator_.endEvent();
}

template <typename TauCollectionType, typename TauDiscrType>
float PATTauProducer::getTauIdDiscriminator(const edm::Handle<TauCollectionType>& tauCollection, size_t tauIdx, const edm::Handle<TauDiscrType>& tauIdDiscr)
{
  edm::Ref<TauCollectionType> tauRef(tauCollection, tauIdx);
  return (*tauIdDiscr)[tauRef];
}

// ParameterSet description for module
void PATTauProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT tau producer module");

  // input source
  iDesc.add<edm::InputTag>("tauSource", edm::InputTag())->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedIsolationTracks", false)->setComment("embed external isolation tracks");
  iDesc.add<bool>("embedLeadTrack", false)->setComment("embed external leading track");
  iDesc.add<bool>("embedLeadTracks", false)->setComment("embed external signal tracks");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
                 edm::ParameterDescription<std::vector<edm::InputTag> >("genParticleMatch", emptySourceVector, true)
		 )->setComment("input with MC match information");

  // MC jet matching variables
  iDesc.add<bool>("addGenJetMatch", true)->setComment("add MC jet matching");
  iDesc.add<bool>("embedGenJetMatch", false)->setComment("embed MC jet matched jet information");
  iDesc.add<edm::InputTag>("genJetMatch", edm::InputTag("tauGenJetMatch"));


  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // tau ID configurables
  iDesc.add<bool>("addTauID", true)->setComment("add tau ID variables");
  edm::ParameterSetDescription tauIDSourcesPSet;
  tauIDSourcesPSet.setAllowAnything();
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("tauIDSource", edm::InputTag(), true) xor
                 edm::ParameterDescription<edm::ParameterSetDescription>("tauIDSources", tauIDSourcesPSet, true)
               )->setComment("input with electron ID variables");

  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker");
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("pfAllParticles");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedHadron");
  isoDepositsPSet.addOptional<edm::InputTag>("pfNeutralHadron");
  isoDepositsPSet.addOptional<edm::InputTag>("pfGamma");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Tau>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauProducer);


