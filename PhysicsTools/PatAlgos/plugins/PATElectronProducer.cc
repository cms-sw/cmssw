//
// $Id: PATElectronProducer.cc,v 1.22 2009/03/26 05:02:41 hegner Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"

#include <vector>
#include <memory>


using namespace pat;


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) ,
  userDataHelper_ ( iConfig.getParameter<edm::ParameterSet>("userData") )
{

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );
  embedGsfTrack_    = iConfig.getParameter<bool>         ( "embedGsfTrack" );
  embedSuperCluster_= iConfig.getParameter<bool>         ( "embedSuperCluster" );
  embedTrack_       = iConfig.getParameter<bool>         ( "embedTrack" );

  // pflow specific
  pfElecSrc_           = iConfig.getParameter<edm::InputTag>( "pfElectronSource" );
  useParticleFlow_        = iConfig.getParameter<bool>( "useParticleFlow" );
  embedPFCandidate_   = iConfig.getParameter<bool>( "embedPFCandidate" );

  
  // MC matching configurables
  addGenMatch_      = iConfig.getParameter<bool>          ( "addGenMatch" );
  if (addGenMatch_) {
      embedGenMatch_ = iConfig.getParameter<bool>         ( "embedGenMatch" );
      if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
          genMatchSrc_.push_back(iConfig.getParameter<edm::InputTag>( "genParticleMatch" ));
      } else {
          genMatchSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" );
      }
  }
  
  // trigger matching configurables
  addTrigMatch_     = iConfig.getParameter<bool>         ( "addTrigMatch" );
  trigMatchSrc_     = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );

  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );

  // electron ID configurables
  addElecID_        = iConfig.getParameter<bool>         ( "addElectronID" );
  if (addElecID_) {
      // it might be a single electron ID
      if (iConfig.existsAs<edm::InputTag>("electronIDSource")) {
          elecIDSrcs_.push_back(NameTag("", iConfig.getParameter<edm::InputTag>("electronIDSource")));
      }
      // or there might be many of them
      if (iConfig.existsAs<edm::ParameterSet>("electronIDSources")) {
          // please don't configure me twice
          if (!elecIDSrcs_.empty()) throw cms::Exception("Configuration") << 
                "PATElectronProducer: you can't specify both 'electronIDSource' and 'electronIDSources'\n";
          // read the different electron ID names
          edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("electronIDSources");
          std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
          for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
              elecIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
          }
      }
      // but in any case at least once
      if (elecIDSrcs_.empty()) throw cms::Exception("Configuration") <<
            "PATElectronProducer: id addElectronID is true, you must specify either:\n" <<
            "\tInputTag electronIDSource = <someTag>\n" << "or\n" <<
            "\tPSet electronIDSources = { \n" <<
            "\t\tInputTag <someName> = <someTag>   // as many as you want \n " <<
            "\t}\n";
  }
  
  // construct resolution calculator

  // IsoDeposit configurables
  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("particle"))           isoDepositLabels_.push_back(std::make_pair(ParticleIso, depconf.getParameter<edm::InputTag>("particle")));
     if (depconf.exists("chargedparticle"))    isoDepositLabels_.push_back(std::make_pair(ChargedParticleIso, depconf.getParameter<edm::InputTag>("chargedparticle")));
     if (depconf.exists("neutralparticle")) isoDepositLabels_.push_back(std::make_pair(NeutralParticleIso,depconf.getParameter<edm::InputTag>("neutralparticle")));
     if (depconf.exists("gammaparticle"))    isoDepositLabels_.push_back(std::make_pair(GammaParticleIso, depconf.getParameter<edm::InputTag>("gammaparticle")));


     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
  }

  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }

  // Check to see if the user wants to add user data
  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }

  // electron ID configurables
  addElecShapes_        = iConfig.getParameter<bool>("addElectronShapes" );
  reducedBarrelRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection") ;
  reducedEndcapRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection") ;
   
  // produces vector of muons
  produces<std::vector<Electron> >();

}


PATElectronProducer::~PATElectronProducer() {
}


void PATElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of electrons from the event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);

  if (isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // prepare the MC matching
  GenAssociations  genMatches(genMatchSrc_.size());
  if (addGenMatch_) {
        for (size_t j = 0, nd = genMatchSrc_.size(); j < nd; ++j) {
            iEvent.getByLabel(genMatchSrc_[j], genMatches[j]);
        }
  }


  // prepare the trigger matching
  TrigAssociations  trigMatches(trigMatchSrc_.size());
  if ( addTrigMatch_ ) {
    for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
      iEvent.getByLabel(trigMatchSrc_[i], trigMatches[i]);
    }
  }

  // prepare ID extraction 
  std::vector<edm::Handle<edm::ValueMap<float> > > idhandles;
  std::vector<pat::Electron::IdPair>               ids;
  if (addElecID_) {
     idhandles.resize(elecIDSrcs_.size());
     ids.resize(elecIDSrcs_.size());
     for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
        iEvent.getByLabel(elecIDSrcs_[i].second, idhandles[i]);
        ids[i].first = elecIDSrcs_[i].first;
     }
  }


  if (addElecShapes_) {
    lazyTools_ .reset(new EcalClusterLazyTools( iEvent , iSetup , reducedBarrelRecHitCollection_ , reducedEndcapRecHitCollection_ ));  
  }

  std::vector<Electron> * patElectrons = new std::vector<Electron>();

  if( useParticleFlow_ ) {
    edm::Handle< reco::PFCandidateCollection >  pfElectrons;
    iEvent.getByLabel(pfElecSrc_, pfElectrons);
    unsigned index=0;
  
    for( reco::PFCandidateConstIterator i = pfElectrons->begin(); 
	 i != pfElectrons->end(); ++i, ++index) {

      reco::PFCandidateRef pfRef(pfElectrons,index);
      reco::PFCandidatePtr ptrToMother(pfElectrons,index);
      reco::CandidateBaseRef pfBaseRef( pfRef ); 
      
      reco::GsfTrackRef PfTk= i->gsfTrackRef(); 

      bool Matched=false;
      for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
	unsigned int idx = itElectron - electrons->begin();
	if (Matched) continue;
	reco::GsfTrackRef EgTk= itElectron->gsfTrack();
	if (itElectron->gsfTrack()==i->gsfTrackRef()){
	  const edm::RefToBase<reco::GsfElectron> elecsRef = electrons->refAt(idx);
	  Electron anElectron(elecsRef);
	  FillElectron(anElectron,elecsRef,pfBaseRef, genMatches, trigMatches);
	  Matched=true;
	  anElectron.setPFCandidateRef( pfRef  );
	  if( embedPFCandidate_ ) anElectron.embedPFCandidate();

	  if (isolator_.enabled()){
	    reco::CandidatePtr mother =  ptrToMother->sourceCandidatePtr(0);
	    isolator_.fill(mother, isolatorTmpStorage_);
	    typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
	    for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), 
		   ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
	      anElectron.setIsolation(it->first, it->second);
	    }
	    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
	      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[mother]);
	    }
	  }
	  //Electron Id
	  if (addElecID_) {
	    //STANDARD EL ID 
	    for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
	      ids[i].second = (*idhandles[i])[elecsRef];    
	    }
	    //SPECIFIC PF ID
	    ids.push_back(std::make_pair("pf_evspi",pfRef->mva_e_pi()));
	    ids.push_back(std::make_pair("pf_evsmu",pfRef->mva_e_mu()));
	    anElectron.setElectronIDs(ids);
	  }

	  patElectrons->push_back(anElectron);

	}

	
      }
 
      
    }
  }

  else{
    for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
    // construct the Electron from the ref -> save ref to original object
    unsigned int idx = itElectron - electrons->begin();
    edm::RefToBase<reco::GsfElectron> elecsRef = electrons->refAt(idx);
    reco::CandidateBaseRef elecBaseRef(elecsRef);
    Electron anElectron(elecsRef);

    FillElectron(anElectron,elecsRef,elecBaseRef, genMatches, trigMatches);
    

    // add resolution info
    
    // Isolation
    if (isolator_.enabled()) {
        isolator_.fill(*electrons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            anElectron.setIsolation(it->first, it->second);
        }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[elecsRef]);
    }

    // add electron ID info
    if (addElecID_) {
        for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
            ids[i].second = (*idhandles[i])[elecsRef];    
        }
        anElectron.setElectronIDs(ids);
    }
    

    if ( useUserData_ ) {
      userDataHelper_.add( anElectron, iEvent, iSetup );
    }
    
    
    // add sel to selected
    patElectrons->push_back(anElectron);
  }
  }
  
  // sort electrons in pt
  std::sort(patElectrons->begin(), patElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<Electron> > ptr(patElectrons);
  iEvent.put(ptr);

  // clean up
  if (isolator_.enabled()) isolator_.endEvent();

}

void PATElectronProducer::FillElectron(Electron& anElectron,
				       const edm::RefToBase<reco::GsfElectron>& elecsRef,
				       const reco::CandidateBaseRef& baseRef,
				       const GenAssociations& genMatches,
				       const TrigAssociations& trigMatches)const {
  if (embedGsfTrack_) anElectron.embedGsfTrack();
  if (embedSuperCluster_) anElectron.embedSuperCluster();
  if (embedTrack_) anElectron.embedTrack();

  // store the match to the generated final state muons
  if (addGenMatch_) {
    for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
      reco::GenParticleRef genElectron = (*genMatches[i])[elecsRef];
      anElectron.addGenParticleRef(genElectron);
    }
    if (embedGenMatch_) anElectron.embedGenParticle();
  }
    // matches to trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
        TriggerPrimitiveRef trigPrim = (*trigMatches[i])[elecsRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          anElectron.addTriggerMatch(*trigPrim);
        }
      }
    }
    
    if (efficiencyLoader_.enabled()) {
      efficiencyLoader_.setEfficiencies( anElectron, elecsRef );
    }
    //  add electron shapes info
    if (addElecShapes_) {
	std::vector<float> covariances = lazyTools_->covariances(*(elecsRef->superCluster()->seed())) ;
	std::vector<float> localCovariances = lazyTools_->localCovariances(*(elecsRef->superCluster()->seed())) ;
	float scSigmaEtaEta = sqrt(covariances[0]) ;
	float scSigmaIEtaIEta = sqrt(localCovariances[0]) ;
	float scE1x5 = lazyTools_->e1x5(*(elecsRef->superCluster()->seed()))  ;
	float scE2x5Max = lazyTools_->e2x5Max(*(elecsRef->superCluster()->seed()))  ;
	float scE5x5 = lazyTools_->e5x5(*(elecsRef->superCluster()->seed())) ;
	anElectron.setClusterShapes(scSigmaEtaEta,scSigmaIEtaIEta,scE1x5,scE2x5Max,scE5x5) ;
    }
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
