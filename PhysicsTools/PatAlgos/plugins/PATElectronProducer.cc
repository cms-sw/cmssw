//
// $Id: PATElectronProducer.cc,v 1.12 2008/07/10 12:21:18 fronga Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"
#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"

#include <vector>
#include <memory>


using namespace pat;


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );
  embedGsfTrack_    = iConfig.getParameter<bool>         ( "embedGsfTrack" );
  embedSuperCluster_= iConfig.getParameter<bool>         ( "embedSuperCluster" );
  embedTrack_       = iConfig.getParameter<bool>         ( "embedTrack" );
  
  // MC matching configurables
  addGenMatch_      = iConfig.getParameter<bool>          ( "addGenMatch" );
  embedGenMatch_    = iConfig.getParameter<bool>          ( "embedGenMatch" );
  genMatchSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  
  // trigger matching configurables
  addTrigMatch_     = iConfig.getParameter<bool>         ( "addTrigMatch" );
  trigMatchSrc_     = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );

  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_        = iConfig.getParameter<bool>         ( "useNNResolutions" );
  electronResoFile_ = iConfig.getParameter<std::string>  ( "electronResoFile" );

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
#ifdef PAT_patElectron_Default_eID  /// ==== If we allow a default ID =====================================================
          // get default algo and check is really among the algos
          std::string            defname = idps.getParameter<std::string>("defaultID");
          if (std::find(names.begin(), names.end(), defname) == names.end()) throw cms::Exception("Configuration") << 
                "PATElectronProducer: the name of the 'default' id must correspond to one InputTag parameter\n";
          // first put the default
          elecIDSrcs_.push_back(NameTag(defname, idps.getParameter<edm::InputTag>(defname)));
          // then all the others
          for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
              if (*it  != defname) elecIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
          }
#else /// ==========  That is, no default ID==============================================================================
          for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
              elecIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
          }
#endif /// ================================================================================================================
      }
      // but in any case at least once
      if (elecIDSrcs_.empty()) throw cms::Exception("Configuration") <<
            "PATElectronProducer: id addElectronID is true, you must specify either:\n" <<
            "\tInputTag electronIDSource = <someTag>\n" << "or\n" <<
            "\tPSet electronIDSources = { \n" <<
            "\t\tInputTag <someName> = <someTag>   // as many as you want \n " <<
#ifdef PAT_patElectron_Default_eID  /// ==== If we allow a default ID =====================================================
            "\t\tstring   defaultID  = <someName>  // one of the names above\n" <<
#endif /// ================================================================================================================
            "\t}\n";
  }
  
  // construct resolution calculator
  if(addResolutions_){
    theResoCalc_= new ObjectResolutionCalc(edm::FileInPath(electronResoFile_).fullPath(), useNNReso_);
  }

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

  // produces vector of muons
  produces<std::vector<Electron> >();

}


PATElectronProducer::~PATElectronProducer() {
  if(addResolutions_) delete theResoCalc_;
}


void PATElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of electrons from the event
  edm::Handle<edm::View<ElectronType> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) {
    iEvent.getByLabel(genMatchSrc_, genMatch);
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
  for (edm::View<ElectronType>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
    // construct the Electron from the ref -> save ref to original object
    unsigned int idx = itElectron - electrons->begin();
    edm::RefToBase<ElectronType> elecsRef = electrons->refAt(idx);
    Electron anElectron(elecsRef);
    if (embedGsfTrack_) anElectron.embedGsfTrack();
    if (embedSuperCluster_) anElectron.embedSuperCluster();
    if (embedTrack_) anElectron.embedTrack();

    // store the match to the generated final state electrons
    if (addGenMatch_) {
      reco::GenParticleRef genElectron = (*genMatch)[elecsRef];
      if (genElectron.isNonnull() && genElectron.isAvailable() ) {
        anElectron.setGenLepton(genElectron, embedGenMatch_);
      } // leave empty if no match found
    }
    
    // matches to trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigMatchSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[elecsRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          anElectron.addTriggerMatch(*trigPrim);
        }
      }
    }

    // add resolution info
    if(addResolutions_){
      (*theResoCalc_)(anElectron);
    }
    
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

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( anElectron, elecsRef );
    }

    // add electron ID info
    if (addElecID_) {
        for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
            ids[i].second = (*idhandles[i])[elecsRef];    
        }
        anElectron.setLeptonIDs(ids);
    }
    
    // add sel to selected
    patElectrons->push_back(anElectron);
  }

  
  // sort electrons in pt
  std::sort(patElectrons->begin(), patElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<Electron> > ptr(patElectrons);
  iEvent.put(ptr);

  // clean up
  if (isolator_.enabled()) isolator_.endEvent();

}


double PATElectronProducer::electronID(const edm::Handle<edm::View<ElectronType> > & electrons,
                                       const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs,
	                               unsigned int idx) {
  //find elecID for elec with index idx
  edm::Ref<std::vector<ElectronType> > elecsRef = electrons->refAt(idx).castTo<edm::Ref<std::vector<ElectronType> > >();
  reco::ElectronIDAssociationCollection::const_iterator elecID = elecIDs->find( elecsRef );

  //return corresponding elecID (only 
  //cut based available at the moment)
  const reco::ElectronIDRef& id = elecID->val;
  return id->cutBasedDecision();
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
