//
// $Id: PATElectronProducer.cc,v 1.29 2009/07/02 12:31:58 cbern Exp $
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

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <memory>


using namespace pat;


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) ,
  useUserData_(iConfig.exists("userData"))
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

  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  if (addResolutions_) {
     resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }


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
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Electron>(iConfig.getParameter<edm::ParameterSet>("userData"));
  }

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
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

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
	  FillElectron(anElectron,elecsRef,pfBaseRef, genMatches);
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

    FillElectron(anElectron,elecsRef,elecBaseRef, genMatches);
    

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
				       const GenAssociations& genMatches)const {
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

    if (efficiencyLoader_.enabled()) {
      efficiencyLoader_.setEfficiencies( anElectron, elecsRef );
    }
    
    if (resolutionLoader_.enabled()) {
      resolutionLoader_.setResolutions(anElectron);
    }

}


// ParameterSet description for module
void PATElectronProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT electron producer module");

  // input source 
  iDesc.add<edm::InputTag>("electronSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedGsfTrack", true)->setComment("embed external gsf track");
  iDesc.add<bool>("embedSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedTrack", false)->setComment("embed external track");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfElectronSource", edm::InputTag("pfElectrons"))->setComment("particle flow input collection");
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor 
                 edm::ParameterDescription<std::vector<edm::InputTag> >("genParticleMatch", emptySourceVector, true)
		 )->setComment("input with MC match information");

  // electron ID configurables
  iDesc.add<bool>("addElectronID",true)->setComment("add electron ID variables");
  edm::ParameterSetDescription electronIDSourcesPSet;
  electronIDSourcesPSet.setAllowAnything(); 
  iDesc.addNode( edm::ParameterDescription<edm::InputTag>("electronIDSource", edm::InputTag(), true) xor
                 edm::ParameterDescription<edm::ParameterSetDescription>("electronIDSources", electronIDSourcesPSet, true)
                 )->setComment("input with electron ID variables");


  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker"); 
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag> >("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Electron>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  // electron shapes
  iDesc.add<bool>("addElectronShapes", true);
  iDesc.add<edm::InputTag>("reducedBarrelRecHitCollection", edm::InputTag("reducedEcalRecHitsEB"));
  iDesc.add<edm::InputTag>("reducedEndcapRecHitCollection", edm::InputTag("reducedEcalRecHitsEE"));

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("isolation", isolationPSet);

  // Resolution configurables
  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  descriptions.add("PATElectronProducer", iDesc);

}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
