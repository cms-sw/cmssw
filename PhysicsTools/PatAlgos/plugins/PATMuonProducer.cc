//
// $Id: PATMuonProducer.cc,v 1.49.2.2 2012/09/21 15:49:27 vadler Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"

#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;
using namespace std;


PATMuonProducer::PATMuonProducer(const edm::ParameterSet & iConfig) : useUserData_(iConfig.exists("userData")), 
  isolator_(iConfig.exists("userIsolation") ? iConfig.getParameter<edm::ParameterSet>("userIsolation") : edm::ParameterSet(), false)
{
  // input source
  muonSrc_ = iConfig.getParameter<edm::InputTag>( "muonSource" );
  // embedding of tracks
  embedBestTrack_ = iConfig.getParameter<bool>( "embedMuonBestTrack" );
  embedImprovedBestTrack_ = iConfig.getParameter<bool>( "embedImprovedMuonBestTrack" );
  embedTrack_ = iConfig.getParameter<bool>( "embedTrack" );
  embedCombinedMuon_ = iConfig.getParameter<bool>( "embedCombinedMuon"   );
  embedStandAloneMuon_ = iConfig.getParameter<bool>( "embedStandAloneMuon" );
  // embedding of muon MET correction information
  embedCaloMETMuonCorrs_ = iConfig.getParameter<bool>("embedCaloMETMuonCorrs" );
  embedTcMETMuonCorrs_ = iConfig.getParameter<bool>("embedTcMETMuonCorrs"   );
  caloMETMuonCorrs_ = iConfig.getParameter<edm::InputTag>("caloMETMuonCorrs" );
  tcMETMuonCorrs_ = iConfig.getParameter<edm::InputTag>("tcMETMuonCorrs"   );
  // pflow specific configurables
  useParticleFlow_ = iConfig.getParameter<bool>( "useParticleFlow" );
  embedPFCandidate_ = iConfig.getParameter<bool>( "embedPFCandidate" );
  pfMuonSrc_ = iConfig.getParameter<edm::InputTag>( "pfMuonSource" );
  // embedding of tracks from TeV refit
  embedPickyMuon_ = iConfig.getParameter<bool>( "embedPickyMuon" );
  embedTpfmsMuon_ = iConfig.getParameter<bool>( "embedTpfmsMuon" );
  embedDytMuon_ = iConfig.getParameter<bool>( "embedDytMuon" );
  // Monte Carlo matching
  addGenMatch_ = iConfig.getParameter<bool>( "addGenMatch" );
  if(addGenMatch_){
    embedGenMatch_ = iConfig.getParameter<bool>( "embedGenMatch" );
    if(iConfig.existsAs<edm::InputTag>("genParticleMatch")){
      genMatchSrc_.push_back(iConfig.getParameter<edm::InputTag>( "genParticleMatch" ));
    } else {
      genMatchSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" );
    }
  }
  // efficiencies
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if(addEfficiencies_){
    efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }
  // resolutions
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
    resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }
  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_);
  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_);
  // check to see if the user wants to add user data
  if( useUserData_ ){
    userDataHelper_ = PATUserDataHelper<Muon>(iConfig.getParameter<edm::ParameterSet>("userData"));
  }
  // embed high level selection variables
  usePV_ = true;
  embedHighLevelSelection_ = iConfig.getParameter<bool>("embedHighLevelSelection");
  if ( embedHighLevelSelection_ ) {
    beamLineSrc_ = iConfig.getParameter<edm::InputTag>("beamLineSrc");
    usePV_ = iConfig.getParameter<bool>("usePV");
    pvSrc_ = iConfig.getParameter<edm::InputTag>("pvSrc");
  }
  // produces vector of muons
  produces<std::vector<Muon> >();
}


PATMuonProducer::~PATMuonProducer() 
{
}

void PATMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) 
{ 
  // switch off embedding (in unschedules mode) 
  if (iEvent.isRealData()){
    addGenMatch_   = false;
    embedGenMatch_ = false;
  }

  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByLabel(muonSrc_, muons);

  // get the ESHandle for the transient track builder,
  // if needed for high level selection embedding
  edm::ESHandle<TransientTrackBuilder> trackBuilder;

  if(isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);
  if(efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if(resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  IsoDepositMaps deposits(isoDepositLabels_.size());
  for (size_t j = 0; j<isoDepositLabels_.size(); ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  IsolationValueMaps isolationValues(isolationValueLabels_.size());
  for (size_t j = 0; j<isolationValueLabels_.size(); ++j) {
    iEvent.getByLabel(isolationValueLabels_[j].second, isolationValues[j]);
  }  

  // prepare the MC matching
  GenAssociations  genMatches(genMatchSrc_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchSrc_.size(); j < nd; ++j) {
      iEvent.getByLabel(genMatchSrc_[j], genMatches[j]);
    }
  }

  // prepare the high level selection: needs beamline
  // OR primary vertex, depending on user selection
  reco::TrackBase::Point beamPoint(0,0,0);
  reco::Vertex primaryVertex;
  reco::BeamSpot beamSpot;
  bool beamSpotIsValid = false;
  bool primaryVertexIsValid = false;
  if ( embedHighLevelSelection_ ) {
    // get the beamspot
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    iEvent.getByLabel(beamLineSrc_, beamSpotHandle);

    // get the primary vertex
    edm::Handle< std::vector<reco::Vertex> > pvHandle;
    iEvent.getByLabel( pvSrc_, pvHandle );

    if( beamSpotHandle.isValid() ){
      beamSpot = *beamSpotHandle;
      beamSpotIsValid = true;
    } else{
      edm::LogError("DataNotAvailable")
	<< "No beam spot available from EventSetup, not adding high level selection \n";
    }
    beamPoint = reco::TrackBase::Point ( beamSpot.x0(), beamSpot.y0(), beamSpot.z0() );
    if( pvHandle.isValid() && !pvHandle->empty() ) {
      primaryVertex = pvHandle->at(0);
      primaryVertexIsValid = true;
    } else {
      edm::LogError("DataNotAvailable")
	<< "No primary vertex available from EventSetup, not adding high level selection \n";
    }
    // this is needed by the IPTools methods from the tracking group
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilder);
  }

  // this will be the new object collection
  std::vector<Muon> * patMuons = new std::vector<Muon>();

  if( useParticleFlow_ ){
    // get the PFCandidates of type muons 
    edm::Handle< reco::PFCandidateCollection >  pfMuons;
    iEvent.getByLabel(pfMuonSrc_, pfMuons);

    unsigned index=0;
    for( reco::PFCandidateConstIterator i = pfMuons->begin(); i != pfMuons->end(); ++i, ++index) {
      const reco::PFCandidate& pfmu = *i;
      //const reco::IsolaPFCandidate& pfmu = *i;
      const reco::MuonRef& muonRef = pfmu.muonRef();
      assert( muonRef.isNonnull() );

      MuonBaseRef muonBaseRef(muonRef);
      Muon aMuon(muonBaseRef);

      if ( useUserData_ ) {
	userDataHelper_.add( aMuon, iEvent, iSetup );
      }

      // embed high level selection
      if ( embedHighLevelSelection_ ) {
	// get the tracks
	reco::TrackRef innerTrack = muonBaseRef->innerTrack();
	reco::TrackRef globalTrack= muonBaseRef->globalTrack();
	reco::TrackRef bestTrack  = muon::muonBestTrack(*muonBaseRef, reco::defaultTuneP).first;

	// Make sure the collection it points to is there
	if ( bestTrack.isNonnull() && bestTrack.isAvailable() ) {
	  unsigned int nhits = bestTrack->numberOfValidHits(); // ????
	  aMuon.setNumberOfValidHits( nhits );

	  reco::TransientTrack tt = trackBuilder->build(bestTrack);
	  embedHighLevel( aMuon, 
			  bestTrack,
			  tt,
			  primaryVertex,
			  primaryVertexIsValid,
			  beamSpot,
			  beamSpotIsValid );

	  // Correct to PV, or beam spot
	  if ( !usePV_ ) {
	    double corr_d0 = -1.0 * bestTrack->dxy( beamPoint );
	    aMuon.setDB( corr_d0, -1.0 );
	  } else {
	    std::pair<bool,Measurement1D> result = IPTools::absoluteTransverseImpactParameter(tt, primaryVertex);
	    double d0_corr = result.second.value();
	    double d0_err = result.second.error();
	    aMuon.setDB( d0_corr, d0_err );
	  }
	}

	if ( globalTrack.isNonnull() && globalTrack.isAvailable() ) {
	  double norm_chi2 = globalTrack->chi2() / globalTrack->ndof();
	  aMuon.setNormChi2( norm_chi2 );
	}
      }
      reco::PFCandidateRef pfRef(pfMuons,index);
      //reco::PFCandidatePtr ptrToMother(pfMuons,index);
      reco::CandidateBaseRef pfBaseRef( pfRef ); 

      aMuon.setPFCandidateRef( pfRef  );     
      if( embedPFCandidate_ ) aMuon.embedPFCandidate();
      fillMuon( aMuon, muonBaseRef, pfBaseRef, genMatches, deposits, isolationValues );
      patMuons->push_back(aMuon); 
    }
  }
  else {
    edm::Handle<edm::View<reco::Muon> > muons;
    iEvent.getByLabel(muonSrc_, muons);

    // embedding of muon MET corrections
    edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > caloMETMuonCorrs;
    //edm::ValueMap<reco::MuonMETCorrectionData> caloMETmuCorValueMap;
    if(embedCaloMETMuonCorrs_){
      iEvent.getByLabel(caloMETMuonCorrs_, caloMETMuonCorrs);
      //caloMETmuCorValueMap  = *caloMETmuCorValueMap_h;
    }
    edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > tcMETMuonCorrs;
    //edm::ValueMap<reco::MuonMETCorrectionData> tcMETmuCorValueMap;
    if(embedTcMETMuonCorrs_) {
      iEvent.getByLabel(tcMETMuonCorrs_, tcMETMuonCorrs);
      //tcMETmuCorValueMap  = *tcMETmuCorValueMap_h;
    }
    for (edm::View<reco::Muon>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon) {
      // construct the Muon from the ref -> save ref to original object
      unsigned int idx = itMuon - muons->begin();
      MuonBaseRef muonRef = muons->refAt(idx);
      reco::CandidateBaseRef muonBaseRef( muonRef ); 
      
      Muon aMuon(muonRef);
      fillMuon( aMuon, muonRef, muonBaseRef, genMatches, deposits, isolationValues);

      // Isolation
      if (isolator_.enabled()) {
	//reco::CandidatePtr mother =  ptrToMother->sourceCandidatePtr(0);
	isolator_.fill(*muons, idx, isolatorTmpStorage_);
	typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
	// better to loop backwards, so the vector is resized less times
	for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
	  aMuon.setIsolation(it->first, it->second);
	}
      }
 
      //       for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
      // 	aMuon.setIsoDeposit(isoDepositLabels_[j].first, 
      // 			    (*deposits[j])[muonRef]);
      //       }

      // add sel to selected
      edm::Ptr<reco::Muon> muonsPtr = muons->ptrAt(idx);
      if ( useUserData_ ) {
	userDataHelper_.add( aMuon, iEvent, iSetup );
      }

      // embed high level selection
      if ( embedHighLevelSelection_ ) {
	// get the tracks
	reco::TrackRef innerTrack = itMuon->innerTrack();
	reco::TrackRef globalTrack= itMuon->globalTrack();
	reco::TrackRef bestTrack  = muon::muonBestTrack(*itMuon, reco::defaultTuneP).first;

	// Make sure the collection it points to is there
	if ( bestTrack.isNonnull() && bestTrack.isAvailable() ) {
	  unsigned int nhits = bestTrack->numberOfValidHits(); // ????
	  aMuon.setNumberOfValidHits( nhits );

	  reco::TransientTrack tt = trackBuilder->build(bestTrack);
	  embedHighLevel( aMuon, 
			  bestTrack,
			  tt,
			  primaryVertex,
			  primaryVertexIsValid,
			  beamSpot,
			  beamSpotIsValid );

	  // Correct to PV, or beam spot
	  if ( !usePV_ ) {
	    double corr_d0 = -1.0 * bestTrack->dxy( beamPoint );
	    aMuon.setDB( corr_d0, -1.0 );
	  } else {
	    std::pair<bool,Measurement1D> result = IPTools::absoluteTransverseImpactParameter(tt, primaryVertex);
	    double d0_corr = result.second.value();
	    double d0_err = result.second.error();
	    aMuon.setDB( d0_corr, d0_err );
	  }
	}

	if ( globalTrack.isNonnull() && globalTrack.isAvailable() ) {
	  double norm_chi2 = globalTrack->chi2() / globalTrack->ndof();
	  aMuon.setNormChi2( norm_chi2 );
	}
      }

      // embed MET muon corrections
      if( embedCaloMETMuonCorrs_ ) aMuon.embedCaloMETMuonCorrs((*caloMETMuonCorrs)[muonRef]);
      if( embedTcMETMuonCorrs_ ) aMuon.embedTcMETMuonCorrs((*tcMETMuonCorrs  )[muonRef]);      

      patMuons->push_back(aMuon);
    }
  }

  // sort muons in pt
  std::sort(patMuons->begin(), patMuons->end(), pTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<Muon> > ptr(patMuons);
  iEvent.put(ptr);

  if (isolator_.enabled()) isolator_.endEvent();
}


void PATMuonProducer::fillMuon( Muon& aMuon, const MuonBaseRef& muonRef, const reco::CandidateBaseRef& baseRef, const GenAssociations& genMatches, const IsoDepositMaps& deposits, const IsolationValueMaps& isolationValues ) const 
{
  // in the particle flow algorithm, 
  // the muon momentum is recomputed. 
  // the new value is stored as the momentum of the 
  // resulting PFCandidate of type Muon, and choosen 
  // as the pat::Muon momentum
  if (useParticleFlow_) 
    aMuon.setP4( aMuon.pfCandidateRef()->p4() );
  if (embedBestTrack_)         aMuon.embedMuonBestTrack();
  if (embedImprovedBestTrack_) aMuon.embedImprovedMuonBestTrack();
  if (embedTrack_)          aMuon.embedTrack();
  if (embedStandAloneMuon_) aMuon.embedStandAloneMuon();
  if (embedCombinedMuon_)   aMuon.embedCombinedMuon();

  // embed the TeV refit track refs (only available for globalMuons)
  if (aMuon.isGlobalMuon()) {
    if (embedPickyMuon_ && aMuon.isAValidMuonTrack(reco::Muon::Picky))
      aMuon.embedPickyMuon();
    if (embedTpfmsMuon_ && aMuon.isAValidMuonTrack(reco::Muon::TPFMS))
      aMuon.embedTpfmsMuon();
    if (embedDytMuon_ && aMuon.isAValidMuonTrack(reco::Muon::DYT))
      aMuon.embedDytMuon();
  }

  // store the match to the generated final state muons
  if (addGenMatch_) {
    for(size_t i = 0, n = genMatches.size(); i < n; ++i) {      
      reco::GenParticleRef genMuon = (*genMatches[i])[baseRef];
      aMuon.addGenParticleRef(genMuon);
    }
    if (embedGenMatch_) aMuon.embedGenParticle();
  }
  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies( aMuon, muonRef );
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if(useParticleFlow_) {
      if (deposits[j]->contains(baseRef.id())) {
	aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[baseRef]);
      } else if (deposits[j]->contains(muonRef.id())){
	aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[muonRef]);
      } else {  
	reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0); 
	aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[source]);
      }
    }
    else{
      aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[muonRef]);
    }
  }
  
  for (size_t j = 0; j<isolationValues.size(); ++j) {
    if(useParticleFlow_) {
      if (isolationValues[j]->contains(baseRef.id())) {
	aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[baseRef]);
      } else if (isolationValues[j]->contains(muonRef.id())) {
	aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[muonRef]);	
      } else {
	reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0);      
	aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[source]);
      }
    }
    else{
      aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[muonRef]);
    }
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(aMuon);
  }
}

// ParameterSet description for module
void PATMuonProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT muon producer module");

  // input source 
  iDesc.add<edm::InputTag>("muonSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedMuonBestTrack", true)->setComment("embed muon best track");
  iDesc.add<bool>("embedImprovedMuonBestTrack", true)->setComment("embed muon best track, new tuneP (only 53X)");
  iDesc.add<bool>("embedTrack", true)->setComment("embed external track");
  iDesc.add<bool>("embedStandAloneMuon", true)->setComment("embed external stand-alone muon");
  iDesc.add<bool>("embedCombinedMuon", false)->setComment("embed external combined muon");
  iDesc.add<bool>("embedPickyMuon", false)->setComment("embed external picky track");
  iDesc.add<bool>("embedTpfmsMuon", false)->setComment("embed external tpfms track");
  iDesc.add<bool>("embedDytMuon", false)->setComment("embed external dyt track ");

  // embedding of MET muon corrections
  iDesc.add<bool>("embedCaloMETMuonCorrs", true)->setComment("whether to add MET muon correction for caloMET or not");
  iDesc.add<edm::InputTag>("caloMETMuonCorrs", edm::InputTag("muonMETValueMapProducer"  , "muCorrData"))->setComment("source of MET muon corrections for caloMET");
  iDesc.add<bool>("embedTcMETMuonCorrs", true)->setComment("whether to add MET muon correction for tcMET or not");
  iDesc.add<edm::InputTag>("tcMETMuonCorrs", edm::InputTag("muonTCMETValueMapProducer"  , "muCorrData"))->setComment("source of MET muon corrections for tcMET");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfMuonSource", edm::InputTag("pfMuons"))->setComment("particle flow input collection");
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor 
                 edm::ParameterDescription<std::vector<edm::InputTag> >("genParticleMatch", emptySourceVector, true)
		 )->setComment("input with MC match information");

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker"); 
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("particle");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedAll");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPhotons");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // isolation values configurables
  edm::ParameterSetDescription isolationValuesPSet;
  isolationValuesPSet.addOptional<edm::InputTag>("tracker"); 
  isolationValuesPSet.addOptional<edm::InputTag>("ecal");
  isolationValuesPSet.addOptional<edm::InputTag>("hcal");
  isolationValuesPSet.addOptional<edm::InputTag>("particle");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPhotons");
  iDesc.addOptional("isolationValues", isolationValuesPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Muon>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

  iDesc.add<bool>("embedHighLevelSelection", true)->setComment("embed high level selection");
  edm::ParameterSetDescription highLevelPSet;
  highLevelPSet.setAllowAnything();
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("beamLineSrc", edm::InputTag(), true) 
                 )->setComment("input with high level selection");
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("pvSrc", edm::InputTag(), true) 
                 )->setComment("input with high level selection");
  iDesc.addNode( edm::ParameterDescription<bool>("usePV", bool(), true) 
                 )->setComment("input with high level selection, use primary vertex (true) or beam line (false)");

  //descriptions.add("PATMuonProducer", iDesc);
}


void PATMuonProducer::readIsolationLabels( const edm::ParameterSet & iConfig, const char* psetName, IsolationLabels& labels) 
{
  labels.clear();
  
  if (iConfig.exists( psetName )) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>(psetName);

    if (depconf.exists("tracker")) labels.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
    if (depconf.exists("ecal"))    labels.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if (depconf.exists("hcal"))    labels.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if (depconf.exists("pfAllParticles"))  {
      labels.push_back(std::make_pair(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    }
    if (depconf.exists("pfChargedHadrons"))  {
      labels.push_back(std::make_pair(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadrons")));
    }
    if (depconf.exists("pfChargedAll"))  {
      labels.push_back(std::make_pair(pat::PfChargedAllIso, depconf.getParameter<edm::InputTag>("pfChargedAll")));
    } 
    if (depconf.exists("pfPUChargedHadrons"))  {
      labels.push_back(std::make_pair(pat::PfPUChargedHadronIso, depconf.getParameter<edm::InputTag>("pfPUChargedHadrons")));
    }
    if (depconf.exists("pfNeutralHadrons"))  {
      labels.push_back(std::make_pair(pat::PfNeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadrons")));
    }
    if (depconf.exists("pfPhotons")) {
      labels.push_back(std::make_pair(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfPhotons")));
    }
    if (depconf.exists("user")) {
      std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
      std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
      int key = UserBaseIso;
      for ( ; it != ed; ++it, ++key) {
	labels.push_back(std::make_pair(IsolationKeys(key), *it));
      }
    }
  }  
}



// embed various impact parameters with errors
// embed high level selection
void PATMuonProducer::embedHighLevel( pat::Muon & aMuon, 
				      reco::TrackRef track,
				      reco::TransientTrack & tt,
				      reco::Vertex & primaryVertex,
				      bool primaryVertexIsValid,
				      reco::BeamSpot & beamspot,
				      bool beamspotIsValid
				      )
{
  // Correct to PV

  // PV2D
  std::pair<bool,Measurement1D> result =
    IPTools::signedTransverseImpactParameter(tt,
					     GlobalVector(track->px(),
							  track->py(),
							  track->pz()),
					     primaryVertex); 
  double d0_corr = result.second.value();
  double d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  aMuon.setDB( d0_corr, d0_err, pat::Muon::PV2D);


  // PV3D
  result =
    IPTools::signedImpactParameter3D(tt,
				     GlobalVector(track->px(),
						  track->py(),
						  track->pz()),
				     primaryVertex);
  d0_corr = result.second.value();
  d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  aMuon.setDB( d0_corr, d0_err, pat::Muon::PV3D);
  

  // Correct to beam spot
  // make a fake vertex out of beam spot
  reco::Vertex vBeamspot(beamspot.position(), beamspot.rotatedCovariance3D());
  
  // BS2D
  result =
    IPTools::signedTransverseImpactParameter(tt,
					     GlobalVector(track->px(),
							  track->py(),
							  track->pz()),
					     vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  aMuon.setDB( d0_corr, d0_err, pat::Muon::BS2D);
  
    // BS3D
  result =
    IPTools::signedImpactParameter3D(tt,
				     GlobalVector(track->px(),
						  track->py(),
						    track->pz()),
				     vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  aMuon.setDB( d0_corr, d0_err, pat::Muon::BS3D);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMuonProducer);
