//
// $Id: PATJetProducer.cc,v 1.26.2.2.2.1 2009/01/16 22:10:33 srappocc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
//#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
//#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
//#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "FWCore/Framework/interface/Selector.h"

#include <vector>
#include <memory>


using namespace pat;


PATJetProducer::PATJetProducer(const edm::ParameterSet& iConfig)  :
  userDataHelper_ ( iConfig.getParameter<edm::ParameterSet>("userData") )
{
  // initialize the configurables
  jetsSrc_                 = iConfig.getParameter<edm::InputTag>	      ( "jetSource" );
  embedCaloTowers_         = iConfig.getParameter<bool>                       ( "embedCaloTowers" );
  getJetMCFlavour_         = iConfig.getParameter<bool> 		      ( "getJetMCFlavour" );
  jetPartonMapSource_      = iConfig.getParameter<edm::InputTag>	      ( "JetPartonMapSource" );
  addGenPartonMatch_       = iConfig.getParameter<bool> 		      ( "addGenPartonMatch" );
  embedGenPartonMatch_     = iConfig.getParameter<bool> 		      ( "embedGenPartonMatch" );
  genPartonSrc_            = iConfig.getParameter<edm::InputTag>	      ( "genPartonMatch" );
  addGenJetMatch_          = iConfig.getParameter<bool> 		      ( "addGenJetMatch" );
  genJetSrc_               = iConfig.getParameter<edm::InputTag>	      ( "genJetMatch" );
  addPartonJetMatch_       = iConfig.getParameter<bool> 		      ( "addPartonJetMatch" );
  partonJetSrc_            = iConfig.getParameter<edm::InputTag>	      ( "partonJetSource" );
  addJetCorrFactors_       = iConfig.getParameter<bool>                       ( "addJetCorrFactors" );
  jetCorrFactorsSrc_       = iConfig.getParameter<edm::InputTag>              ( "jetCorrFactorsSource" );
  addTrigMatch_            = iConfig.getParameter<bool>                       ( "addTrigMatch" );
  trigMatchSrc_            = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );
  addResolutions_          = iConfig.getParameter<bool> 		      ( "addResolutions" );
  addBTagInfo_             = iConfig.getParameter<bool> 		      ( "addBTagInfo" );
  addDiscriminators_       = iConfig.getParameter<bool> 		      ( "addDiscriminators" );
  discriminatorModule_     = iConfig.getParameter<edm::InputTag>              ( "discriminatorModule" );
  addTagInfoRefs_          = iConfig.getParameter<bool> 		      ( "addTagInfoRefs" );
  tagInfoModule_           = iConfig.getParameter<edm::InputTag>              ( "tagInfoModule" );
  addAssociatedTracks_     = iConfig.getParameter<bool> 		      ( "addAssociatedTracks" ); 
  trackAssociation_        = iConfig.getParameter<edm::InputTag>	      ( "trackAssociationSource" );
  addJetCharge_            = iConfig.getParameter<bool> 		      ( "addJetCharge" ); 
  jetCharge_               = iConfig.getParameter<edm::InputTag>	      ( "jetChargeSource" );

  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }

  std::vector<std::string> discriminatorNames = iConfig.getParameter<std::vector<std::string> >("discriminatorNames");
  if (discriminatorNames.empty()) { 
    addDiscriminators_ = false; // there's no point to read all discriminators and save none ;-)
  } else if (std::find(discriminatorNames.begin(), discriminatorNames.end(), "*") == discriminatorNames.end()) {
    discriminatorNames_.insert(discriminatorNames.begin(), discriminatorNames.end());
  } else {
    // we just leave the set empty, and catch everything we can
  }

  std::vector<std::string> tagInfoNames = iConfig.getParameter<std::vector<std::string> >("tagInfoNames");
  if (tagInfoNames.empty()) { 
    addTagInfoRefs_ = false; // there's no point to read all tagInfos and save none ;-)
  } else if (std::find(tagInfoNames.begin(), tagInfoNames.end(), "*") == tagInfoNames.end()) {
    tagInfoNames_.insert(tagInfoNames.begin(), tagInfoNames.end());
  } else {
    // we just leave the set empty, and catch everything we can
  }

  if (!addBTagInfo_) { addDiscriminators_ = false; addTagInfoRefs_ = false; }

  // Check to see if the user wants to add user data
  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }

  // produces vector of jets
  produces<std::vector<Jet> >();
}


PATJetProducer::~PATJetProducer() 
{
}


void 
PATJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) 
{
  // Get the vector of jets
  edm::Handle<edm::View<JetType> > jets;
  iEvent.getByLabel(jetsSrc_, jets);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);

  // For jet flavour
  edm::Handle<reco::JetFlavourMatchingCollection> jetFlavMatch;
  if (getJetMCFlavour_) iEvent.getByLabel (jetPartonMapSource_, jetFlavMatch);

  // Get the vector of generated particles from the event if needed
  edm::Handle<edm::Association<reco::GenParticleCollection> > partonMatch;
  if (addGenPartonMatch_) iEvent.getByLabel(genPartonSrc_,  partonMatch);
  // Get the vector of GenJets from the event if needed
  edm::Handle<edm::Association<reco::GenJetCollection> > genJetMatch;
  if (addGenJetMatch_) iEvent.getByLabel(genJetSrc_, genJetMatch);
/* TO BE IMPLEMENTED FOR >= 1_5_X
  // Get the vector of PartonJets from the event if needed
  edm::Handle<edm::View<reco::SomePartonJetType> > partonJets;
  if (addPartonJetMatch_) iEvent.getByLabel(partonJetSrc_, partonJets);
*/

  // read in the jet correction factors ValueMap
  edm::Handle<edm::ValueMap<JetCorrFactors> > jetCorrs;
  if (addJetCorrFactors_) iEvent.getByLabel(jetCorrFactorsSrc_, jetCorrs);

  // Get the vector of jet tags with b-tagging info
  std::vector<edm::Handle<edm::ValueMap<float> > > jetDiscriminators;
  if (addBTagInfo_ && addDiscriminators_) {
    if (discriminatorModule_.process().empty()) {
        edm::ModuleLabelSelector selector(discriminatorModule_.label());
        iEvent.getMany(selector, jetDiscriminators);
    } else {
        edm::Selector selector(edm::ModuleLabelSelector(discriminatorModule_.label()) && edm::ProcessNameSelector(discriminatorModule_.process()));
        iEvent.getMany(selector, jetDiscriminators);
    }
  }
  std::vector<edm::Handle<edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > > > jetTagInfos;
  if (addBTagInfo_ && addTagInfoRefs_) {
    if (tagInfoModule_.process().empty()) {
        edm::ModuleLabelSelector selector(tagInfoModule_.label());
        iEvent.getMany(selector, jetTagInfos);
    } else {
        edm::Selector selector(edm::ModuleLabelSelector(tagInfoModule_.label()) && edm::ProcessNameSelector(tagInfoModule_.process()));
        iEvent.getMany(selector, jetTagInfos);
    }
  }
 
  // tracks Jet Track Association
  edm::Handle<edm::ValueMap<reco::TrackRefVector> > hTrackAss;
  if (addAssociatedTracks_) iEvent.getByLabel(trackAssociation_, hTrackAss);
  edm::Handle<edm::ValueMap<float> > hJetChargeAss;
  if (addJetCharge_) iEvent.getByLabel(jetCharge_, hJetChargeAss);

  // loop over jets
  std::vector<Jet> * patJets = new std::vector<Jet>(); 
  for (edm::View<JetType>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {

    // construct the Jet from the ref -> save ref to original object
    unsigned int idx = itJet - jets->begin();
    edm::RefToBase<JetType> jetRef = jets->refAt(idx);
    edm::Ptr<JetType> jetPtr = jets->ptrAt(idx); 
    Jet ajet(jetRef);

    // ensure the internal storage of the jet constituents
    if (ajet.isCaloJet() && embedCaloTowers_) {
        const reco::CaloJet *cj = dynamic_cast<const reco::CaloJet *>(jetRef.get());
        ajet.setCaloTowers( cj->getCaloConstituents() );
    }

    if (addJetCorrFactors_) {
      // calculate the energy correction factors
      const JetCorrFactors & jcf = (*jetCorrs)[jetRef];
      ajet.setJetCorrFactors(jcf);
      // set current default which is JetCorrFactors::L3
      ajet.setJetCorrStep(JetCorrFactors::L3);
      ajet.setP4(jcf.correction(JetCorrFactors::L3) * itJet->p4());
      // print JetCorrFactors for debugging
      // jcf.print();
    }

    // get the MC flavour information for this jet
    if (getJetMCFlavour_) {
        ajet.setPartonFlavour( (*jetFlavMatch)[edm::RefToBase<reco::Jet>(jetRef)].getFlavour() );
    }
    // store the match to the generated partons
    if (addGenPartonMatch_) {
      reco::GenParticleRef parton = (*partonMatch)[jetRef];
      if (parton.isNonnull() && parton.isAvailable()) {
          ajet.setGenParton(parton, embedGenPartonMatch_);
      } // leave empty if no match found
    }
    // store the match to the GenJets
    if (addGenJetMatch_) {
      reco::GenJetRef genjet = (*genJetMatch)[jetRef];
      if (genjet.isNonnull() && genjet.isAvailable()) {
          ajet.setGenJet(*genjet);
      } // leave empty if no match found
    }

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( ajet, jetRef );
    }

    // TO BE IMPLEMENTED FOR >=1_5_X: do the PartonJet matching
    if (addPartonJetMatch_) {
    }
    
    // matches to trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigMatchSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[jetRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          ajet.addTriggerMatch(*trigPrim);
        }
      }
    }

    // add b-tag info if available & required
    if (addBTagInfo_) {
        if (addDiscriminators_) {
            for (size_t k=0; k<jetDiscriminators.size(); ++k) {
                const edm::Handle<edm::ValueMap<float> > & jetDiscHandle = jetDiscriminators[k];
                std::pair<std::string,float> pairDiscri(jetDiscHandle.provenance()->productInstanceName(), -1000.0);
                std::string::size_type pos = pairDiscri.first.find("JetTags"); 
                if ((pos !=  std::string::npos) && (pos != pairDiscri.first.length() - 7)) {
                    pairDiscri.first.erase(pos+7); // trim a tail after "JetTags"
                }
                if ( discriminatorNames_.empty() || 
                        (discriminatorNames_.find(pairDiscri.first) != discriminatorNames_.end()) ) {
                    if (jetDiscHandle->contains(jetRef.id())) {
                        pairDiscri.second = (*jetDiscHandle)[jetRef];
                        ajet.addBDiscriminatorPair(pairDiscri);
                    }
                }
            }
        }    
        if (addTagInfoRefs_) {
            for (size_t k=0; k<jetTagInfos.size(); ++k) {
                const edm::Handle<edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > > & jetTagInfoHandle = jetTagInfos[k];
                std::string label = jetTagInfoHandle.provenance()->productInstanceName();
                std::string::size_type pos = label.find("TagInfos");
                if ((pos !=  std::string::npos) && (pos != label.length() - 8)) {
                    label.erase(pos+8); // trim a tail after "TagInfos"
                }
                if ( tagInfoNames_.empty() || 
                        (tagInfoNames_.find(label) != tagInfoNames_.end()) ) {
                    if (jetTagInfoHandle->contains(jetRef.id())) {
                        ajet.addTagInfo(label, (*jetTagInfoHandle)[jetRef]);
                    }
                }
            }
        }    
    }
    
    if (addAssociatedTracks_) ajet.setAssociatedTracks( (*hTrackAss)[jetRef] );

    if (addJetCharge_)        ajet.setJetCharge( (*hJetChargeAss)[jetRef] );


    if ( useUserData_ ) {
      userDataHelper_.add( ajet, iEvent, iSetup );
    }
    

    patJets->push_back(ajet);
  }

  // sort jets in Et
  std::sort(patJets->begin(), patJets->end(), pTComparator_);

  // put genEvt  in Event
  std::auto_ptr<std::vector<Jet> > myJets(patJets);
  iEvent.put(myJets);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATJetProducer);

