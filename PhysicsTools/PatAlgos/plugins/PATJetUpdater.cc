//


#include "PhysicsTools/PatAlgos/plugins/PATJetUpdater.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/transform.h"

#include <vector>
#include <memory>
#include <algorithm>


using namespace pat;


PATJetUpdater::PATJetUpdater(const edm::ParameterSet& iConfig) :
  useUserData_(iConfig.exists("userData"))
{
  // initialize configurables
  jetsToken_ = consumes<edm::View<Jet> >(iConfig.getParameter<edm::InputTag>( "jetSource" ));
  addJetCorrFactors_ = iConfig.getParameter<bool>( "addJetCorrFactors" );
  jetCorrFactorsTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "jetCorrFactorsSource" ), [this](edm::InputTag const & tag){return mayConsume<edm::ValueMap<JetCorrFactors> >(tag);});
  // Check to see if the user wants to add user data
  if ( useUserData_ ) {
    userDataHelper_ = PATUserDataHelper<Jet>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // produces vector of jets
  produces<std::vector<Jet> >();
}


PATJetUpdater::~PATJetUpdater() {

}


void PATJetUpdater::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  // Get the vector of jets
  edm::Handle<edm::View<Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);

  // read in the jet correction factors ValueMap
  std::vector<edm::ValueMap<JetCorrFactors> > jetCorrs;
  if (addJetCorrFactors_) {
    for ( size_t i = 0; i < jetCorrFactorsTokens_.size(); ++i ) {
      edm::Handle<edm::ValueMap<JetCorrFactors> > jetCorr;
      iEvent.getByToken(jetCorrFactorsTokens_[i], jetCorr);
      jetCorrs.push_back( *jetCorr );
    }
  }

  // loop over jets
  std::auto_ptr< std::vector<Jet> > patJets ( new std::vector<Jet>() );

  bool first=true; // this is introduced to issue warnings only for the first jet
  for (edm::View<Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {

    // construct the Jet from the ref -> save ref to original object
    unsigned int idx = itJet - jets->begin();
    edm::RefToBase<Jet> jetRef = jets->refAt(idx);
    edm::Ptr<Jet> jetPtr = jets->ptrAt(idx);
    Jet ajet(jetPtr);

    if (addJetCorrFactors_) {
      unsigned int setindex = ajet.availableJECSets().size();
      // Undo previous jet energy corrections
      ajet.setP4(ajet.correctedP4(0));
      // add additional JetCorrs to the jet
      for ( unsigned int i=0; i<jetCorrFactorsTokens_.size(); ++i ) {
	const JetCorrFactors& jcf = jetCorrs[i][jetRef];
	// uncomment for debugging
	// jcf.print();
	ajet.addJECFactors(jcf);
      }
      std::vector<std::string> levels = jetCorrs[0][jetRef].correctionLabels();
      if(std::find(levels.begin(), levels.end(), "L2L3Residual")!=levels.end()){
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L2L3Residual"), JetCorrFactors::NONE, setindex);
      }
      else if(std::find(levels.begin(), levels.end(), "L3Absolute")!=levels.end()){
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L3Absolute"), JetCorrFactors::NONE, setindex);
      }
      else{
	ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("Uncorrected"), JetCorrFactors::NONE, setindex);
	if(first){
	  edm::LogWarning("L3Absolute not found") << "L2L3Residual and L3Absolute are not part of the correction applied jetCorrFactors \n"
						  << "of module " <<  jetCorrs[0][jetRef].jecSet() << " jets will remain"
						  << " uncorrected."; first=false;
	}
      }
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

}

// ParameterSet description for module
void PATJetUpdater::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT jet producer module");

  // input source
  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("no default"))->setComment("input collection");

  // jet energy corrections
  iDesc.add<bool>("addJetCorrFactors", true);
  std::vector<edm::InputTag> emptyVInputTags;
  iDesc.add<std::vector<edm::InputTag> >("jetCorrFactorsSource", emptyVInputTags);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Jet>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  descriptions.add("PATJetUpdater", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATJetUpdater);
