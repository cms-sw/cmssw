//
// $Id: PATJetProducer.cc,v 1.39 2009/07/18 08:00:27 srappocc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/Selector.h"

#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

#include <vector>
#include <memory>


using namespace pat;


PATJetProducer::PATJetProducer(const edm::ParameterSet& iConfig)  :
  useUserData_(iConfig.exists("userData"))
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
  jetCorrFactorsSrc_       = iConfig.getParameter<std::vector<edm::InputTag> >( "jetCorrFactorsSource" );
  addBTagInfo_             = iConfig.getParameter<bool> 		      ( "addBTagInfo" );
  addDiscriminators_       = iConfig.getParameter<bool> 		      ( "addDiscriminators" );
  discriminatorTags_       = iConfig.getParameter<std::vector<edm::InputTag> >( "discriminatorSources" );
  addTagInfos_             = iConfig.getParameter<bool> 		      ( "addTagInfos" );
  tagInfoTags_             = iConfig.getParameter<std::vector<edm::InputTag> >( "tagInfoSources" );
  addAssociatedTracks_     = iConfig.getParameter<bool> 		      ( "addAssociatedTracks" ); 
  trackAssociation_        = iConfig.getParameter<edm::InputTag>	      ( "trackAssociationSource" );
  addJetCharge_            = iConfig.getParameter<bool> 		      ( "addJetCharge" ); 
  jetCharge_               = iConfig.getParameter<edm::InputTag>	      ( "jetChargeSource" );
  addJetID_                = iConfig.getParameter<bool>                       ( "addJetID");

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


  if (discriminatorTags_.empty()) { 
    addDiscriminators_ = false; 
  } else {
    for (std::vector<edm::InputTag>::const_iterator it = discriminatorTags_.begin(), ed = discriminatorTags_.end(); it != ed; ++it) {
        std::string label = it->label();
        std::string::size_type pos = label.find("JetTags");
        if ((pos !=  std::string::npos) && (pos != label.length() - 7)) {
            label.erase(pos+7); // trim a tail after "JetTags"
        }
        discriminatorLabels_.push_back(label);
    }
  }
  if (tagInfoTags_.empty()) { 
    addTagInfos_ = false; 
  } else {
    for (std::vector<edm::InputTag>::const_iterator it = tagInfoTags_.begin(), ed = tagInfoTags_.end(); it != ed; ++it) {
        std::string label = it->label();
        std::string::size_type pos = label.find("TagInfos");
        if ((pos !=  std::string::npos) && (pos != label.length() - 8)) {
            label.erase(pos+8); // trim a tail after "TagInfos"
        }
        tagInfoLabels_.push_back(label);
    }
  }

  if (!addBTagInfo_) { addDiscriminators_ = false; addTagInfos_ = false; }

  // Check to see if the user wants to add user data
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Jet>(iConfig.getParameter<edm::ParameterSet>("userData"));
  }

  if ( addJetID_ ) {
    jetIDHelper_ = reco::helper::JetIDHelper( iConfig.getParameter<edm::ParameterSet>("jetID") );
  }

  // produces vector of jets
  produces<std::vector<Jet> >();
}


PATJetProducer::~PATJetProducer() {

}


void PATJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the vector of jets
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(jetsSrc_, jets);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  // for jet flavour
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
  std::vector<edm::ValueMap<JetCorrFactors> > jetCorrs;
  if (addJetCorrFactors_) {
    for ( size_t i = 0; i < jetCorrFactorsSrc_.size(); ++i ) {
      edm::Handle<edm::ValueMap<JetCorrFactors> > jetCorr;
      iEvent.getByLabel(jetCorrFactorsSrc_[i], jetCorr);
      jetCorrs.push_back( *jetCorr );
    }
  }  

  // Get the vector of jet tags with b-tagging info
  std::vector<edm::Handle<reco::JetFloatAssociation::Container> > jetDiscriminators;
  if (addBTagInfo_ && addDiscriminators_) {
    jetDiscriminators.resize(discriminatorTags_.size());
    for (size_t i = 0; i < discriminatorTags_.size(); ++i) {
        iEvent.getByLabel(discriminatorTags_[i], jetDiscriminators[i]);
    }
  }
  std::vector<edm::Handle<edm::View<reco::BaseTagInfo> > > jetTagInfos;
  if (addBTagInfo_ && addTagInfos_) {
    jetTagInfos.resize(tagInfoTags_.size());
    for (size_t i = 0; i < tagInfoTags_.size(); ++i) {
        iEvent.getByLabel(tagInfoTags_[i], jetTagInfos[i]);
    }
  }
 
  // tracks Jet Track Association
  edm::Handle<reco::JetTracksAssociation::Container > hTrackAss;
  if (addAssociatedTracks_) iEvent.getByLabel(trackAssociation_, hTrackAss);
  edm::Handle<reco::JetFloatAssociation::Container > hJetChargeAss;
  if (addJetCharge_) iEvent.getByLabel(jetCharge_, hJetChargeAss);

  // loop over jets
  std::vector<Jet> * patJets = new std::vector<Jet>(); 
  for (edm::View<reco::Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {

    // construct the Jet from the ref -> save ref to original object
    unsigned int idx = itJet - jets->begin();
    edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);
    edm::Ptr<reco::Jet> jetPtr = jets->ptrAt(idx); 
    Jet ajet(jetRef);

    // ensure the internal storage of the jet constituents
    if (ajet.isCaloJet() && embedCaloTowers_) {
        const reco::CaloJet *cj = dynamic_cast<const reco::CaloJet *>(jetRef.get());
        ajet.setCaloTowers( cj->getCaloConstituents() );
    }

    // Add Jet Energy Scale Corrections
    if (addJetCorrFactors_) {
      // in case only one set of jet correction factors is used, clear the string
      // that contains the name of the jcf-module, to save storage per jet:
      if (jetCorrFactorsSrc_.size()<=1)
        jetCorrs.front()[jetRef].clearLabel();
      // the default jet correction is the first in the vector
      const JetCorrFactors & jcf = jetCorrs.front()[jetRef];
      // uncomment for debugging
      // jcf.print();
      //attach first (default) jet correction factors set to the jet
      ajet.setCorrFactors(jcf);
      // set current default which is JetCorrFactors::L3, change P4 of ajet 
      ajet.setCorrStep(JetCorrFactors::L3);
      
      // add additional JetCorrs for syst. studies, if present
      for ( size_t i = 1; i < jetCorrFactorsSrc_.size(); ++i ) {
	const JetCorrFactors & jcf = jetCorrs[i][jetRef];
	ajet.addCorrFactors(jcf);
      }
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

    // IMPORTANT: DO THIS AFTER JES CORRECTIONS
    if (resolutionLoader_.enabled()) {
        resolutionLoader_.setResolutions(ajet);
    }

    // TO BE IMPLEMENTED FOR >=1_5_X: do the PartonJet matching
    if (addPartonJetMatch_) {
    }

    // add b-tag info if available & required
    if (addBTagInfo_) {
        if (addDiscriminators_) {
            for (size_t k=0; k<jetDiscriminators.size(); ++k) {
                float value = (*jetDiscriminators[k])[jetRef];
                ajet.addBDiscriminatorPair(std::make_pair(discriminatorLabels_[k], value));
            }
        }    
        if (addTagInfos_) {
            for (size_t k=0; k<jetTagInfos.size(); ++k) {
                const edm::View<reco::BaseTagInfo> & taginfos = *jetTagInfos[k];
                // This is not associative, so we have to search the jet
                const reco::BaseTagInfo * match = 0;
                // Try first by 'same index'
                if ((idx < taginfos.size()) && (taginfos[idx].jet() == jetRef)) {
                    match = &taginfos[idx];
                } else {
                    // otherwise fail back to a simple search
                    for (edm::View<reco::BaseTagInfo>::const_iterator itTI = taginfos.begin(), edTI = taginfos.end(); itTI != edTI; ++itTI) {
                        if (itTI->jet() == jetRef) { match = &*itTI; break; }
                    }
                }
		//TODO !!!                if (match != 0) ajet.addTagInfo(tagInfoLabels_[k], *match);
            }
        }    
    }
    
    if (addAssociatedTracks_) ajet.setAssociatedTracks( (*hTrackAss)[jetRef] );

    if (addJetCharge_)        ajet.setJetCharge( (*hJetChargeAss)[jetRef] );

    // add jet ID for calo jets
    if (addJetID_ && ajet.isCaloJet() ) {
      jetIDHelper_.calculate( iEvent, dynamic_cast<reco::CaloJet const &>(*itJet) );
      ajet.setFHPD         ( jetIDHelper_.fHPD()            );
      ajet.setFRBX         ( jetIDHelper_.fRBX()            );
      ajet.setN90Hits      ( jetIDHelper_.n90Hits()         );
      ajet.setFSubDetector1( jetIDHelper_.fSubDetector1()   );
      ajet.setFSubDetector2( jetIDHelper_.fSubDetector2()   );
      ajet.setFSubDetector3( jetIDHelper_.fSubDetector3()   );
      ajet.setFSubDetector4( jetIDHelper_.fSubDetector4()   );
      ajet.setRestrictedEMF( jetIDHelper_.restrictedEMF()   );
      ajet.setNHCALTowers  ( jetIDHelper_.nHCALTowers()     );
      ajet.setNECALTowers  ( jetIDHelper_.nECALTowers()     );
    }

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

// ParameterSet description for module
void PATJetProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT jet producer module");

  // input source 
  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedCaloTowers", true)->setComment("embed external calo towers");

  // MC matching configurables
  iDesc.add<bool>("addGenPartonMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenPartonMatch", false)->setComment("embed MC matched MC information");
  iDesc.add<edm::InputTag>("genPartonMatch", edm::InputTag())->setComment("input with MC match information");

  iDesc.add<bool>("addGenJetMatch", true)->setComment("add MC matching");
  iDesc.add<edm::InputTag>("genJetMatch", edm::InputTag())->setComment("input with MC match information");

  iDesc.add<bool>("addJetCharge", true);
  iDesc.add<edm::InputTag>("jetChargeSource", edm::InputTag("patJetCharge"));
  
  // jet id
  iDesc.add<bool>("addJetID", true)->setComment("Add jet ID information");
  edm::ParameterSetDescription jetIDPSet;
  jetIDPSet.setAllowAnything();
  iDesc.addOptional("jetID", jetIDPSet);


  iDesc.add<bool>("addPartonJetMatch", false);
  iDesc.add<edm::InputTag>("partonJetSource", edm::InputTag("NOT IMPLEMENTED"));

  // track association
  iDesc.add<bool>("addAssociatedTracks", true);
  iDesc.add<edm::InputTag>("trackAssociationSource", edm::InputTag("ic5JetTracksAssociatorAtVertex"));

  // tag info
  iDesc.add<bool>("addTagInfos", true);
  std::vector<edm::InputTag> emptyVInputTags;
  iDesc.add<std::vector<edm::InputTag> >("tagInfoSources", emptyVInputTags);

  // jet energy corrections
  iDesc.add<bool>("addJetCorrFactors", true);
  iDesc.add<std::vector<edm::InputTag> >("jetCorrFactorsSource", emptyVInputTags);

  // btag discriminator tags
  iDesc.add<bool>("addBTagInfo",true);
  iDesc.add<bool>("addDiscriminators", true);
  iDesc.add<std::vector<edm::InputTag> >("discriminatorSources", emptyVInputTags);

  // jet flavour idetification configurables
  iDesc.add<bool>("getJetMCFlavour", true);
  iDesc.add<edm::InputTag>("JetPartonMapSource", edm::InputTag("jetFlavourAssociation"));

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Jet>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  descriptions.add("PATJetProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATJetProducer);

