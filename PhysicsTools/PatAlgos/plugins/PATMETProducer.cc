//
//

#include "PhysicsTools/PatAlgos/plugins/PATMETProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>


using namespace pat;


PATMETProducer::PATMETProducer(const edm::ParameterSet & iConfig):
  useUserData_(iConfig.exists("userData"))
{
  // initialize the configurables
  metSrc_         = iConfig.getParameter<edm::InputTag>("metSource");
  metToken_       = consumes<edm::View<reco::MET> >(metSrc_);
  addGenMET_      = iConfig.getParameter<bool>         ("addGenMET");
  genMETToken_    = mayConsume<edm::View<reco::GenMET> >(iConfig.getParameter<edm::InputTag>("genMETSource"));
  addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");

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
    userDataHelper_ = PATUserDataHelper<MET>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }

  // MET Significance
  calculateMETSignificance_ = iConfig.getParameter<bool>("computeMETSignificance");
  if(calculateMETSignificance_)
    {
      metSigAlgo_ = new metsig::METSignificance(iConfig);
      jetToken_ = mayConsume<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("srcJets"));
      pfCandToken_ = mayConsume<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("srcPFCands"));
      std::vector<edm::InputTag> srcLeptonsTags = iConfig.getParameter< std::vector<edm::InputTag> >("srcLeptons");
      for(std::vector<edm::InputTag>::const_iterator it=srcLeptonsTags.begin();it!=srcLeptonsTags.end();it++) {
	lepTokens_.push_back( mayConsume<edm::View<reco::Candidate> >( *it ) );
      }
    }  

  // produces vector of mets
  produces<std::vector<MET> >();
}


PATMETProducer::~PATMETProducer() {
}


void PATMETProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the vector of MET's from the event
  edm::Handle<edm::View<reco::MET> > mets;
  iEvent.getByToken(metToken_, mets);

  if (mets->size() != 1) throw cms::Exception("Corrupt Data") << "The input MET collection " << metSrc_.encode() << " has size " << mets->size() << " instead of 1 as it should.\n";
  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  // Get the vector of generated met from the event if needed
  edm::Handle<edm::View<reco::GenMET> > genMETs;
  if (addGenMET_) {
    iEvent.getByToken(genMETToken_, genMETs);
  }

  // loop over mets
  std::vector<MET> * patMETs = new std::vector<MET>();
  for (edm::View<reco::MET>::const_iterator itMET = mets->begin(); itMET != mets->end(); itMET++) {
    // construct the MET from the ref -> save ref to original object
    unsigned int idx = itMET - mets->begin();
    edm::RefToBase<reco::MET> metsRef = mets->refAt(idx);
    edm::Ptr<reco::MET> metsPtr = mets->ptrAt(idx);
    MET amet(metsRef);
    // add the generated MET
    if (addGenMET_) amet.setGenMET((*genMETs)[idx]);

    //add the MET significance
    if(calculateMETSignificance_) {
      const reco::METCovMatrix& sigcov = getMETCovMatrix(iEvent);
      amet.setSignificanceMatrix(sigcov);
      double metSig=metSigAlgo_->getSignificance(sigcov, amet);
      amet.setMETSignificance(metSig);
    }

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( amet, metsRef );
    }

    if (resolutionLoader_.enabled()) {
        resolutionLoader_.setResolutions(amet);
    }


    if ( useUserData_ ) {
      userDataHelper_.add( amet, iEvent, iSetup );
    }


    // correct for muons if demanded... never more: it's now done by JetMETCorrections
    // add the MET to the vector of METs
    patMETs->push_back(amet);
  }

  // sort MET in ET .. don't mess with this
  //  std::sort(patMETs->begin(), patMETs->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<MET> > myMETs(patMETs);
  iEvent.put(myMETs);

}

// ParameterSet description for module
void PATMETProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT MET producer module");

  // input source
  iDesc.add<edm::InputTag>("metSource", edm::InputTag("no default"))->setComment("input collection");

  // MC configurations
  iDesc.add<bool>("addGenMET", false);
  iDesc.add<edm::InputTag>("genMETSource", edm::InputTag("genMetCalo"));

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything(); // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<MET>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  // muon correction
  iDesc.add<bool>("addMuonCorrections", false);
  iDesc.add<edm::InputTag>("muonSource", edm::InputTag("muons"));

}

const reco::METCovMatrix 
PATMETProducer::getMETCovMatrix(const edm::Event& event) const {
  std::vector< edm::Handle<reco::CandidateView> > leptons;
  for ( std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator srcLeptons_i = lepTokens_.begin();
	srcLeptons_i != lepTokens_.end(); ++srcLeptons_i ) {
    edm::Handle<reco::CandidateView> leptons_i;
    event.getByToken(*srcLeptons_i, leptons_i);
    leptons.push_back( leptons_i );
  }
  // jets
  edm::Handle<edm::View<reco::Jet> > inputJets;
  event.getByToken( jetToken_, inputJets );

  //candidates
  edm::Handle<edm::View<reco::Candidate> > inputCands;
  event.getByToken( pfCandToken_, inputCands );

  //Compute the covariance matrix and fill it
  reco::METCovMatrix cov = metSigAlgo_->getCovariance( *inputJets, leptons, *inputCands);
  return cov;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMETProducer);
