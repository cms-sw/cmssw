//
// $Id: PATMuonProducer.cc,v 1.28 2009/06/30 22:00:54 cbern Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;
using namespace std;


PATMuonProducer::PATMuonProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false),
  useUserData_(iConfig.exists("userData"))
{

  
  // general configurables
  muonSrc_             = iConfig.getParameter<edm::InputTag>( "muonSource" );
  
  

  embedTrack_          = iConfig.getParameter<bool>         ( "embedTrack" );
  embedStandAloneMuon_ = iConfig.getParameter<bool>         ( "embedStandAloneMuon" );
  embedCombinedMuon_   = iConfig.getParameter<bool>         ( "embedCombinedMuon" );

  embedPickyMuon_      = iConfig.getParameter<bool>         ( "embedPickyMuon" );
  embedTpfmsMuon_      = iConfig.getParameter<bool>         ( "embedTpfmsMuon" );

  
  //pflow specific
  pfMuonSrc_           = iConfig.getParameter<edm::InputTag>( "pfMuonSource" );
  useParticleFlow_        = iConfig.getParameter<bool>( "useParticleFlow" );

  embedPFCandidate_   = iConfig.getParameter<bool>( "embedPFCandidate" );


  // TeV refit names
  addTeVRefits_ = iConfig.getParameter<bool>("addTeVRefits");
  if (addTeVRefits_) {
    pickySrc_ = iConfig.getParameter<edm::InputTag>("pickySrc");
    tpfmsSrc_ = iConfig.getParameter<edm::InputTag>("tpfmsSrc");
  }


  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>         ( "addGenMatch" );
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>         ( "embedGenMatch" );
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchSrc_.push_back(iConfig.getParameter<edm::InputTag>( "genParticleMatch" ));
    } else {
      genMatchSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" );
    }
  }
  
  
  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
    efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }

  // Resolution configurables
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
    resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }

  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_);

  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_);
  
  // Check to see if the user wants to add user data
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Muon>(iConfig.getParameter<edm::ParameterSet>("userData"));
  }

  // produces vector of muons
  produces<std::vector<Muon> >();

}


PATMuonProducer::~PATMuonProducer() {
}

void PATMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  
  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByLabel(muonSrc_, muons);

  if (isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

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

  std::vector<Muon> * patMuons = new std::vector<Muon>();


  if( useParticleFlow_ ) {

    // get the PFCandidates of type muons 
    edm::Handle< reco::PFCandidateCollection >  pfMuons;
    iEvent.getByLabel(pfMuonSrc_, pfMuons);

    unsigned index=0;
    for( reco::PFCandidateConstIterator i = pfMuons->begin(); 
	 i != pfMuons->end(); ++i, ++index) {
      

      const reco::PFCandidate& pfmu = *i;

      //const reco::IsolaPFCandidate& pfmu = *i;

      // std::cout<<pfmu<<std::endl;
      const reco::MuonRef& muonRef = pfmu.muonRef();
      assert( muonRef.isNonnull() );

      MuonBaseRef muonBaseRef(muonRef);
      Muon aMuon(muonBaseRef);

      reco::PFCandidateRef pfRef(pfMuons,index);
      //reco::PFCandidatePtr ptrToMother(pfMuons,index);
      reco::CandidateBaseRef pfBaseRef( pfRef ); 

      aMuon.setPFCandidateRef( pfRef  );     
      if( embedPFCandidate_ ) aMuon.embedPFCandidate();

      fillMuon( aMuon, muonBaseRef, pfBaseRef, 
		genMatches, deposits, isolationValues );
     
      patMuons->push_back(aMuon); 
    } 
  }
  else {

    edm::Handle<edm::View<reco::Muon> > muons;
    iEvent.getByLabel(muonSrc_, muons);

    // prepare the TeV refit track retrieval
    edm::Handle<reco::TrackToTrackMap> pickyMap, tpfmsMap;
    if (addTeVRefits_) {
      iEvent.getByLabel(pickySrc_, pickyMap);
      iEvent.getByLabel(tpfmsSrc_, tpfmsMap);
    }
    
    for (edm::View<reco::Muon>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon) {
      
      // construct the Muon from the ref -> save ref to original object
      unsigned int idx = itMuon - muons->begin();
      MuonBaseRef muonRef = muons->refAt(idx);
      reco::CandidateBaseRef muonBaseRef( muonRef ); 
      
      Muon aMuon(muonRef);
      
      fillMuon( aMuon, muonRef, muonBaseRef, 
		genMatches, deposits, isolationValues);

      // store the TeV refit track refs (only available for globalMuons)
      if (addTeVRefits_ && itMuon->isGlobalMuon()) {
	reco::TrackToTrackMap::const_iterator it;
	const reco::TrackRef& globalTrack = itMuon->globalTrack();
	
	// If the getByLabel calls failed above (i.e. if the TeV refit
	// maps/collections were not in the event), then the TrackRefs
	// in the Muon object will remain null.
	if (!pickyMap.failedToGet()) {
	  it = pickyMap->find(globalTrack);
	  if (it != pickyMap->end()) aMuon.setPickyMuon(it->val);
	  if (embedPickyMuon_) aMuon.embedPickyMuon();
	}
 
	if (!tpfmsMap.failedToGet()) {
	  it = tpfmsMap->find(globalTrack);
	  if (it != tpfmsMap->end()) aMuon.setTpfmsMuon(it->val);
	  if (embedTpfmsMuon_) aMuon.embedTpfmsMuon();
	}
      }
      
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

void PATMuonProducer::fillMuon( Muon& aMuon, 
				const MuonBaseRef& muonRef,
				const reco::CandidateBaseRef& baseRef,
				const GenAssociations& genMatches, 
				const IsoDepositMaps& deposits,
				const IsolationValueMaps& isolationValues ) const {

  // in the particle flow algorithm, 
  // the muon momentum is recomputed. 
  // the new value is stored as the momentum of the 
  // resulting PFCandidate of type Muon, and choosen 
  // as the pat::Muon momentum
  if (useParticleFlow_) 
    aMuon.setP4( aMuon.pfCandidateRef()->p4() );

  if (embedTrack_) aMuon.embedTrack();
  if (embedStandAloneMuon_) aMuon.embedStandAloneMuon();
  if (embedCombinedMuon_) aMuon.embedCombinedMuon();
  
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
      reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0);      
      aMuon.setIsoDeposit(isoDepositLabels_[j].first, 
			  (*deposits[j])[source]);
    }
    else
      aMuon.setIsoDeposit(isoDepositLabels_[j].first,
                          (*deposits[j])[muonRef]);
  }

  for (size_t j = 0; j<isolationValues.size(); ++j) {
    if(useParticleFlow_) {
      reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0);      
      aMuon.setIsolation(isolationValueLabels_[j].first, 
			 (*isolationValues[j])[source]);
    }
    else
      aMuon.setIsolation(isolationValueLabels_[j].first,
                          (*isolationValues[j])[muonRef]);
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
  iDesc.add<bool>("embedTrack", true)->setComment("embed external track");
  iDesc.add<bool>("embedStandAloneMuon", true)->setComment("embed external stand-alone muon");
  iDesc.add<bool>("embedCombinedMuon", false)->setComment("embed external combined muon");
  iDesc.add<bool>("embedPickyMuon", false)->setComment("embed external picky muon");
  iDesc.add<bool>("embedTpfmsMuon", false)->setComment("embed external tpfms muon");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfMuonSource", edm::InputTag("pfMuons"))->setComment("particle flow input collection");
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");

  // TeV refit 
  iDesc.ifValue( edm::ParameterDescription<bool>("addTeVRefits", true, true),
		 true >> (edm::ParameterDescription<edm::InputTag>("pickySrc", edm::InputTag(), true) and
			  edm::ParameterDescription<edm::InputTag>("tpfmsSrc", edm::InputTag(), true)) 
		 )->setComment("If TeV refits are added, their sources need to be specified");

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
  iDesc.add("isolation", isolationPSet);

  descriptions.add("PATMuonProducer", iDesc);

}


void PATMuonProducer::readIsolationLabels( const edm::ParameterSet & iConfig,
					   const char* psetName, 
					   IsolationLabels& labels) {
  
  labels.clear();
  
  if (iConfig.exists( psetName )) {
    edm::ParameterSet depconf 
      = iConfig.getParameter<edm::ParameterSet>(psetName);

    if (depconf.exists("tracker")) labels.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
    if (depconf.exists("ecal"))    labels.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if (depconf.exists("hcal"))    labels.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if (depconf.exists("pfAllParticles"))  {
      labels.push_back(std::make_pair(ChargedHadronIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    }
    if (depconf.exists("pfChargedHadrons"))  {
      labels.push_back(std::make_pair(ChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadrons")));
    }
    if (depconf.exists("pfNeutralHadrons"))  {
      labels.push_back(std::make_pair(NeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadrons")));
    }
    if (depconf.exists("pfPhotons")) {
      labels.push_back(std::make_pair(PhotonIso, depconf.getParameter<edm::InputTag>("pfPhotons")));
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


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMuonProducer);
