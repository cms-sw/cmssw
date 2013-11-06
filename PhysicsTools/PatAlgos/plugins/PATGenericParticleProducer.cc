//
//

#include "PhysicsTools/PatAlgos/plugins/PATGenericParticleProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Utilities/interface/transform.h"
#include <memory>

using namespace pat;

PATGenericParticleProducer::PATGenericParticleProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("userIsolation") ? iConfig.getParameter<edm::ParameterSet>("userIsolation") : edm::ParameterSet(), consumesCollector(), false),
  userDataHelper_ ( iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector() )
{
  // initialize the configurables
  srcToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>( "src" ));

  // RECO embedding
  embedTrack_        = iConfig.getParameter<bool>( "embedTrack" );
  embedGsfTrack_     = iConfig.getParameter<bool>( "embedGsfTrack" );
  embedStandalone_   = iConfig.getParameter<bool>( "embedStandAloneMuon" );
  embedCombined_     = iConfig.getParameter<bool>( "embedCombinedMuon" );
  embedSuperCluster_ = iConfig.getParameter<bool>( "embedSuperCluster" );
  embedTracks_       = iConfig.getParameter<bool>( "embedMultipleTracks" );
  embedCaloTower_    = iConfig.getParameter<bool>( "embedCaloTower" );

  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>( "addGenMatch" );
  if (addGenMatch_) {
      embedGenMatch_ = iConfig.getParameter<bool>( "embedGenMatch" );
      if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
        genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection> >(iConfig.getParameter<edm::InputTag>( "genParticleMatch" )));
      } else {
        genMatchTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" ), [this](edm::InputTag const & tag){return consumes<edm::Association<reco::GenParticleCollection> >(tag);});
      }
  }

  // quality
  addQuality_ = iConfig.getParameter<bool>("addQuality");
  qualitySrcToken_ = mayConsume<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("qualitySource"));

  // produces vector of particles
  produces<std::vector<GenericParticle> >();

  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
  }
  isoDepositTokens_ = edm::vector_transform(isoDepositLabels_, [this](std::pair<IsolationKeys,edm::InputTag> const & label){return consumes<edm::ValueMap<IsoDeposit> >(label.second);});

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

  if (iConfig.exists("vertexing")) {
     vertexingHelper_ = pat::helper::VertexingHelper(iConfig.getParameter<edm::ParameterSet>("vertexing"), consumesCollector());
  }

  // Check to see if the user wants to add user data
  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }
}

PATGenericParticleProducer::~PATGenericParticleProducer() {
}

void PATGenericParticleProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // Get the vector of GenericParticle's from the event
  edm::Handle<edm::View<reco::Candidate> > cands;
  iEvent.getByToken(srcToken_, cands);

  // prepare isolation
  if (isolator_.enabled()) isolator_.beginEvent(iEvent,iSetup);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);
  if (vertexingHelper_.enabled())  vertexingHelper_.newEvent(iEvent,iSetup);

  // prepare IsoDeposits
  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositTokens_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByToken(isoDepositTokens_[j], deposits[j]);
  }

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
        for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
            iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
        }
  }

  // prepare the quality
  edm::Handle<edm::ValueMap<float> > qualities;
  if (addQuality_) iEvent.getByToken(qualitySrcToken_, qualities);

  // loop over cands
  std::vector<GenericParticle> * PATGenericParticles = new std::vector<GenericParticle>();
  for (edm::View<reco::Candidate>::const_iterator itGenericParticle = cands->begin(); itGenericParticle != cands->end(); itGenericParticle++) {
    // construct the GenericParticle from the ref -> save ref to original object
    unsigned int idx = itGenericParticle - cands->begin();
    edm::RefToBase<reco::Candidate> candRef = cands->refAt(idx);

    PATGenericParticles->push_back(GenericParticle(candRef));
    GenericParticle & aGenericParticle = PATGenericParticles->back();

    // embed RECO
    if (embedTrack_)        aGenericParticle.embedTrack();
    if (embedGsfTrack_)     aGenericParticle.embedGsfTrack();
    if (embedTracks_)       aGenericParticle.embedTracks();
    if (embedStandalone_)   aGenericParticle.embedStandalone();
    if (embedCombined_)     aGenericParticle.embedCombined();
    if (embedSuperCluster_) aGenericParticle.embedSuperCluster();
    if (embedCaloTower_)    aGenericParticle.embedCaloTower();

    // isolation
    if (isolator_.enabled()) {
        isolator_.fill(*cands, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            aGenericParticle.setIsolation(it->first, it->second);
        }
    }

    // isodeposit
    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        aGenericParticle.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[candRef]);
    }

    // store the match to the generated final state muons
    if (addGenMatch_) {
      for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
          reco::GenParticleRef genGenericParticle = (*genMatches[i])[candRef];
          aGenericParticle.addGenParticleRef(genGenericParticle);
      }
      if (embedGenMatch_) aGenericParticle.embedGenParticle();
    }

    if (addQuality_) {
      aGenericParticle.setQuality( (*qualities)[candRef] );
    }

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( aGenericParticle, candRef );
    }

    if (resolutionLoader_.enabled()) {
        resolutionLoader_.setResolutions(aGenericParticle);
    }

    if (vertexingHelper_.enabled()) {
        aGenericParticle.setVertexAssociation( vertexingHelper_(candRef) );
    }

    if ( useUserData_ ) {
        userDataHelper_.add( aGenericParticle, iEvent, iSetup );
    }

    // PATGenericParticles->push_back(aGenericParticle); // NOOOOO!!!!
    // We have already pushed_back this generic particle in the collection
    // (we first push an empty particle and then fill it, to avoid useless copies)
  }

  // sort GenericParticles in ET
  std::sort(PATGenericParticles->begin(), PATGenericParticles->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<GenericParticle> > myGenericParticles(PATGenericParticles);
  iEvent.put(myGenericParticles);
  if (isolator_.enabled()) isolator_.endEvent();

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATGenericParticleProducer);
