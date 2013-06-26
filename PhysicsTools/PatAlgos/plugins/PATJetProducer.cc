//
// $Id: PATJetProducer.cc,v 1.58 2012/11/28 21:53:13 vadler Exp $


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

#include <vector>
#include <memory>
#include <algorithm>


using namespace pat;


PATJetProducer::PATJetProducer(const edm::ParameterSet& iConfig)  :
  useUserData_(iConfig.exists("userData"))
{
  // initialize configurables
  jetsSrc_ = iConfig.getParameter<edm::InputTag>( "jetSource" );
  embedCaloTowers_ = false; // parameter is optional
  if ( iConfig.exists("embedCaloTowers") ) {
    embedCaloTowers_ = iConfig.getParameter<bool>( "embedCaloTowers" );
  }
  embedPFCandidates_ = iConfig.getParameter<bool>( "embedPFCandidates" );
  getJetMCFlavour_ = iConfig.getParameter<bool>( "getJetMCFlavour" );
  jetPartonMapSource_ = iConfig.getParameter<edm::InputTag>( "JetPartonMapSource" );
  addGenPartonMatch_ = iConfig.getParameter<bool>( "addGenPartonMatch" );
  embedGenPartonMatch_ = iConfig.getParameter<bool>( "embedGenPartonMatch" );
  genPartonSrc_ = iConfig.getParameter<edm::InputTag>( "genPartonMatch" );
  addGenJetMatch_ = iConfig.getParameter<bool>( "addGenJetMatch" );
  embedGenJetMatch_ = iConfig.getParameter<bool>( "embedGenJetMatch" );
  genJetSrc_ = iConfig.getParameter<edm::InputTag>( "genJetMatch" );
  addPartonJetMatch_ = iConfig.getParameter<bool>( "addPartonJetMatch" );
  partonJetSrc_ = iConfig.getParameter<edm::InputTag>( "partonJetSource" );
  addJetCorrFactors_ = iConfig.getParameter<bool>( "addJetCorrFactors" );
  jetCorrFactorsSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "jetCorrFactorsSource" );
  addBTagInfo_ = iConfig.getParameter<bool>( "addBTagInfo" );
  addDiscriminators_ = iConfig.getParameter<bool>( "addDiscriminators" );
  discriminatorTags_ = iConfig.getParameter<std::vector<edm::InputTag> >( "discriminatorSources" );
  addTagInfos_ = iConfig.getParameter<bool>( "addTagInfos" );
  tagInfoTags_ = iConfig.getParameter<std::vector<edm::InputTag> >( "tagInfoSources" );
  addAssociatedTracks_ = iConfig.getParameter<bool>( "addAssociatedTracks" );
  trackAssociation_ = iConfig.getParameter<edm::InputTag>( "trackAssociationSource" );
  addJetCharge_ = iConfig.getParameter<bool>( "addJetCharge" );
  jetCharge_ = iConfig.getParameter<edm::InputTag>( "jetChargeSource" );
  addJetID_ = iConfig.getParameter<bool>( "addJetID");
  jetIDMapLabel_ = iConfig.getParameter<edm::InputTag>( "jetIDMap");
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
  // produces vector of jets
  produces<std::vector<Jet> >();
  produces<reco::GenJetCollection> ("genJets");
  produces<std::vector<CaloTower>  > ("caloTowers");
  produces<reco::PFCandidateCollection > ("pfCandidates");
  produces<edm::OwnVector<reco::BaseTagInfo> > ("tagInfos");
}


PATJetProducer::~PATJetProducer() {

}


void PATJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  // check whether dealing with MC or real data
  if (iEvent.isRealData()){
    getJetMCFlavour_   = false;
    addGenPartonMatch_ = false;
    addGenJetMatch_    = false;
    addPartonJetMatch_ = false;
  }

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

  // jet ID handle
  edm::Handle<reco::JetIDValueMap> hJetIDMap;
  if ( addJetID_ ) iEvent.getByLabel( jetIDMapLabel_, hJetIDMap );

  // loop over jets
  std::auto_ptr< std::vector<Jet> > patJets ( new std::vector<Jet>() );

  std::auto_ptr<reco::GenJetCollection > genJetsOut ( new reco::GenJetCollection() );
  std::auto_ptr<std::vector<CaloTower>  >  caloTowersOut( new std::vector<CaloTower> () );
  std::auto_ptr<reco::PFCandidateCollection > pfCandidatesOut( new reco::PFCandidateCollection() );
  std::auto_ptr<edm::OwnVector<reco::BaseTagInfo> > tagInfosOut ( new edm::OwnVector<reco::BaseTagInfo>() );


  edm::RefProd<reco::GenJetCollection > h_genJetsOut = iEvent.getRefBeforePut<reco::GenJetCollection >( "genJets" );
  edm::RefProd<std::vector<CaloTower>  >  h_caloTowersOut = iEvent.getRefBeforePut<std::vector<CaloTower>  > ( "caloTowers" );
  edm::RefProd<reco::PFCandidateCollection > h_pfCandidatesOut = iEvent.getRefBeforePut<reco::PFCandidateCollection > ( "pfCandidates" );
  edm::RefProd<edm::OwnVector<reco::BaseTagInfo> > h_tagInfosOut = iEvent.getRefBeforePut<edm::OwnVector<reco::BaseTagInfo> > ( "tagInfos" );

  bool first=true; // this is introduced to issue warnings only for the first jet
  for (edm::View<reco::Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {

    // construct the Jet from the ref -> save ref to original object
    unsigned int idx = itJet - jets->begin();
    edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);
    edm::Ptr<reco::Jet> jetPtr = jets->ptrAt(idx);
    Jet ajet(jetRef);

    // add the FwdPtrs to the CaloTowers
    if ( (ajet.isCaloJet() || ajet.isJPTJet() ) && embedCaloTowers_) {
      const reco::CaloJet *cj = 0;
      const reco::JPTJet * jptj = 0;
      if ( ajet.isCaloJet()) cj = dynamic_cast<const reco::CaloJet *>(jetRef.get());
      else {
	jptj = dynamic_cast<const reco::JPTJet *>(jetRef.get() );
	cj = dynamic_cast<const reco::CaloJet *>(jptj->getCaloJetRef().get() );
      }
      pat::CaloTowerFwdPtrCollection itowersRef;
      std::vector< CaloTowerPtr > itowers = cj->getCaloConstituents();
      for ( std::vector<CaloTowerPtr>::const_iterator towBegin = itowers.begin(), towEnd = itowers.end(), itow = towBegin; itow != towEnd; ++itow ) {
	if( itow->isAvailable() && itow->isNonnull() ){
	  caloTowersOut->push_back( **itow );
	  // set the "forward" ref to the thinned collection
	  edm::Ref<std::vector<CaloTower> > caloTowerRef( h_caloTowersOut, caloTowersOut->size() - 1);
	  edm::Ptr<CaloTower> caloForwardRef ( h_caloTowersOut.id(), caloTowerRef.key(), h_caloTowersOut.productGetter() );
	  // set the "backward" ref to the original collection for association
	  edm::Ptr<CaloTower> caloBackRef ( *itow );
	  // add to the list of FwdPtr's
	  itowersRef.push_back( pat::CaloTowerFwdPtrCollection::value_type ( caloForwardRef, caloBackRef ) );
	}
      }
      ajet.setCaloTowers( itowersRef );
    }

    // add the FwdPtrs to the PFCandidates
    if (ajet.isPFJet() && embedPFCandidates_) {
      const reco::PFJet *cj = dynamic_cast<const reco::PFJet *>(jetRef.get());
      pat::PFCandidateFwdPtrCollection iparticlesRef;
      std::vector< reco::PFCandidatePtr > iparticles = cj->getPFConstituents();
      for ( std::vector<reco::PFCandidatePtr>::const_iterator partBegin = iparticles.begin(),
	      partEnd = iparticles.end(), ipart = partBegin;
	    ipart != partEnd; ++ipart ) {
	pfCandidatesOut->push_back( **ipart );
	// set the "forward" ref to the thinned collection
	edm::Ref<reco::PFCandidateCollection> pfCollectionRef( h_pfCandidatesOut, pfCandidatesOut->size() - 1);
	edm::Ptr<reco::PFCandidate> pfForwardRef ( h_pfCandidatesOut.id(), pfCollectionRef.key(),  h_pfCandidatesOut.productGetter() );
	// set the "backward" ref to the original collection for association
	edm::Ptr<reco::PFCandidate> pfBackRef ( *ipart );
	// add to the list of FwdPtr's
	iparticlesRef.push_back( pat::PFCandidateFwdPtrCollection::value_type ( pfForwardRef, pfBackRef ) );
      }
      ajet.setPFCandidates( iparticlesRef );
    }

    if (addJetCorrFactors_) {
      // add additional JetCorrs to the jet
      for ( unsigned int i=0; i<jetCorrFactorsSrc_.size(); ++i ) {
	const JetCorrFactors& jcf = jetCorrs[i][jetRef];
	// uncomment for debugging
	// jcf.print();
	ajet.addJECFactors(jcf);
      }
      std::vector<std::string> levels = jetCorrs[0][jetRef].correctionLabels();
      if(std::find(levels.begin(), levels.end(), "L2L3Residual")!=levels.end()){
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L2L3Residual"));
      }
      else if(std::find(levels.begin(), levels.end(), "L3Absolute")!=levels.end()){
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L3Absolute"));
      }
      else{
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("Uncorrected"));
	if(first){
	  edm::LogWarning("L3Absolute not found") << "L2L3Residual and L3Absolute are not part of the correction applied jetCorrFactors \n"
						  << "of module " <<  jetCorrs[0][jetRef].jecSet() << " jets will remain"
						  << " uncorrected."; first=false;
	}
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
	genJetsOut->push_back( *genjet );
	// set the "forward" ref to the thinned collection
	edm::Ref<reco::GenJetCollection > genForwardRef ( h_genJetsOut, genJetsOut->size() - 1 );
	// set the "backward" ref to the original collection
	edm::Ref<reco::GenJetCollection > genBackRef ( genjet );
	// make the FwdPtr
	edm::FwdRef<reco::GenJetCollection > genjetFwdRef ( genForwardRef, genBackRef );
	ajet.setGenJetRef(genjetFwdRef );
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
	    edm::Ptr<reco::BaseTagInfo> match;
	    // Try first by 'same index'
	    if ((idx < taginfos.size()) && (taginfos[idx].jet() == jetRef)) {
	      match = taginfos.ptrAt(idx);
	    } else {
	      // otherwise fail back to a simple search
	      for (edm::View<reco::BaseTagInfo>::const_iterator itTI = taginfos.begin(), edTI = taginfos.end(); itTI != edTI; ++itTI) {
		if (itTI->jet() == jetRef) { match = taginfos.ptrAt( itTI - taginfos.begin() ); break; }
	      }
	    }
	    if (match.isNonnull()) {
	      tagInfosOut->push_back( match->clone() );
	      // set the "forward" ptr to the thinned collection
	      edm::Ptr<reco::BaseTagInfo> tagInfoForwardPtr ( h_tagInfosOut.id(), &tagInfosOut->back(), tagInfosOut->size() - 1 );
	      // set the "backward" ptr to the original collection for association
	      edm::Ptr<reco::BaseTagInfo> tagInfoBackPtr ( match );
	      // make FwdPtr
	      TagInfoFwdPtrCollection::value_type tagInfoFwdPtr( tagInfoForwardPtr, tagInfoBackPtr ) ;
	      ajet.addTagInfo(tagInfoLabels_[k], tagInfoFwdPtr );
	    }
	  }
        }
    }

    if (addAssociatedTracks_) ajet.setAssociatedTracks( (*hTrackAss)[jetRef] );

    if (addJetCharge_) ajet.setJetCharge( (*hJetChargeAss)[jetRef] );

    // add jet ID for calo jets
    if (addJetID_ && ajet.isCaloJet() ) {
      reco::JetID jetId = (*hJetIDMap)[ jetRef ];
      ajet.setJetID( jetId );
    }
    // add jet ID jpt jets
    else if ( addJetID_ && ajet.isJPTJet() ){
      const reco::JPTJet *jptj = dynamic_cast<const reco::JPTJet *>(jetRef.get());
      reco::JetID jetId = (*hJetIDMap)[ jptj->getCaloJetRef() ];
      ajet.setJetID( jetId );
    }
    if ( useUserData_ ) {
      userDataHelper_.add( ajet, iEvent, iSetup );
    }
    patJets->push_back(ajet);
  }

  // sort jets in pt
  std::sort(patJets->begin(), patJets->end(), pTComparator_);

  // put genEvt  in Event
  iEvent.put(patJets);

  iEvent.put( genJetsOut, "genJets" );
  iEvent.put( caloTowersOut, "caloTowers" );
  iEvent.put( pfCandidatesOut, "pfCandidates" );
  iEvent.put( tagInfosOut, "tagInfos" );


}

// ParameterSet description for module
void PATJetProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT jet producer module");

  // input source
  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.addOptional<bool>("embedCaloTowers", false)->setComment("embed external CaloTowers (not to be used on AOD input)");
  iDesc.add<bool>("embedPFCandidates", true)->setComment("embed external PFCandidates");

  // MC matching configurables
  iDesc.add<bool>("addGenPartonMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenPartonMatch", false)->setComment("embed MC matched MC information");
  iDesc.add<edm::InputTag>("genPartonMatch", edm::InputTag())->setComment("input with MC match information");

  iDesc.add<bool>("addGenJetMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenJetMatch", false)->setComment("embed MC matched MC information");
  iDesc.add<edm::InputTag>("genJetMatch", edm::InputTag())->setComment("input with MC match information");

  iDesc.add<bool>("addJetCharge", true);
  iDesc.add<edm::InputTag>("jetChargeSource", edm::InputTag("patJetCharge"));

  // jet id
  iDesc.add<bool>("addJetID", true)->setComment("Add jet ID information");
  iDesc.add<edm::InputTag>("jetIDMap", edm::InputTag())->setComment("jet id map");

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

