#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include <memory>

//
// Class decleration
//

class JPTTrackProducer : public edm::EDProducer
{
 public:
  explicit JPTTrackProducer(const edm::ParameterSet& config);
  ~JPTTrackProducer();
 private:
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
  virtual void endJob();
  
  void copyTracks(const jpt::MatchedTracks& tracks,
                         reco::TrackCollection* inVertexInCaloTracks,
			 reco::TrackCollection* outVertexInCaloTracks,
			 reco::TrackCollection* inVertexOutCaloTracks) const;
  static void copyTracks(const reco::TrackRefVector& from, reco::TrackCollection* to);
  
  const edm::InputTag zspCorrectedJetsTag_;
  const std::string jptCorrectorName_;
  const uint32_t jetIndex_;
  const bool produceInCaloInVertex_, produceOutCaloInVertex_, produceInCaloOutVertex_;
  const JetPlusTrackCorrector* jptCorrector_;
};

//
// Constants, enums and typedefs
//



//
// Static data member definitions
//



//
// Constructors and destructor
//

JPTTrackProducer::JPTTrackProducer(const edm::ParameterSet& config)
  : zspCorrectedJetsTag_(config.getParameter<edm::InputTag>("ZSPCorrectedJetsTag")),
    jptCorrectorName_(config.getParameter<std::string>("JPTCorrectorName")),
    jetIndex_(config.getParameter<uint32_t>("JetIndex")),
    produceInCaloInVertex_(config.getParameter<bool>("ProduceInCaloInVertex")),
    produceOutCaloInVertex_(config.getParameter<bool>("ProduceOutCaloInVertex")),
    produceInCaloOutVertex_(config.getParameter<bool>("ProduceInCaloOutVertex")),
    jptCorrector_(NULL)
{
  produces<reco::TrackCollection>("InVertexInCalo");
  produces<reco::TrackCollection>("InVertexOutCalo");
  produces<reco::TrackCollection>("OutVertexInCalo");
}

JPTTrackProducer::~JPTTrackProducer()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
JPTTrackProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
  std::auto_ptr<reco::TrackCollection> inVertexInCaloTracks(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> outVertexInCaloTracks(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> inVertexOutCaloTracks(new reco::TrackCollection);
  edm::Handle<reco::CaloJetCollection> unCorrectedJetsHandle;
  event.getByLabel(zspCorrectedJetsTag_,unCorrectedJetsHandle);
  const reco::CaloJetCollection& unCorrectedJets = *unCorrectedJetsHandle;
  if (unCorrectedJets.size() > jetIndex_) {
    if (jptCorrector_->canCorrect(unCorrectedJets[jetIndex_])) {
      jpt::MatchedTracks pions, muons, electrons;
      jptCorrector_->matchTracks(unCorrectedJets[jetIndex_],event,eventSetup,pions,muons,electrons);
      copyTracks(pions,inVertexInCaloTracks.get(),outVertexInCaloTracks.get(),inVertexOutCaloTracks.get());
      copyTracks(muons,inVertexInCaloTracks.get(),outVertexInCaloTracks.get(),inVertexOutCaloTracks.get());
      copyTracks(electrons,inVertexInCaloTracks.get(),outVertexInCaloTracks.get(),inVertexOutCaloTracks.get());
      event.put(inVertexInCaloTracks,"InVertexInCalo");
      event.put(inVertexOutCaloTracks,"InVertexOutCalo");
      event.put(outVertexInCaloTracks,"OutVertexInCalo");
    }
  }
}

void JPTTrackProducer::copyTracks(const jpt::MatchedTracks& tracks,
                                  reco::TrackCollection* inVertexInCaloTracks,
                                  reco::TrackCollection* outVertexInCaloTracks,
                                  reco::TrackCollection* inVertexOutCaloTracks) const
{
  if (produceInCaloInVertex_) copyTracks(tracks.inVertexInCalo_,inVertexInCaloTracks);
  if (produceInCaloOutVertex_) copyTracks(tracks.outOfVertexInCalo_,outVertexInCaloTracks);
  if (produceOutCaloInVertex_) copyTracks(tracks.inVertexOutOfCalo_,inVertexOutCaloTracks);
}

void JPTTrackProducer::copyTracks(const reco::TrackRefVector& from, reco::TrackCollection* to)
{
  for (reco::TrackRefVector::const_iterator iTrack = from.begin(); iTrack != from.end(); ++iTrack) {
    to->push_back(reco::Track(**iTrack));
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
JPTTrackProducer::beginJob(const edm::EventSetup& eventSetup)
{
  const JetCorrector* corrector = JetCorrector::getJetCorrector(jptCorrectorName_,eventSetup);
  if (!corrector) edm::LogError("JPTTrackProducer") << "Failed to get corrector with name " << jptCorrectorName_ << "from the EventSetup";
  jptCorrector_ = dynamic_cast<const JetPlusTrackCorrector*>(corrector);
  if (!jptCorrector_) edm::LogError("JPTTrackProducer") << "Corrector with name " << jptCorrectorName_ << " is not a JetPlusTrackCorrector";
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JPTTrackProducer::endJob()
{
}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JPTTrackProducer);
