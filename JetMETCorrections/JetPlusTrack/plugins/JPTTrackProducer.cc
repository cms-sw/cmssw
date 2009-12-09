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
  
  static void copyTracks(const jpt::MatchedTracks& tracks,
                         reco::TrackCollection* inVertexInCaloTracks,
			 reco::TrackCollection* outVertexInCaloTracks,
			 reco::TrackCollection* inVertexOutCaloTracks);
  static void copyTracks(const reco::TrackRefVector& from, reco::TrackCollection* to);
  
  const edm::InputTag zspCorrectedJetsTag_;
  const std::string jptCorrectorName_;
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
    jptCorrector_(NULL)
{
  produces<reco::TrackCollection>("InVertexInCalo1");
  produces<reco::TrackCollection>("OutVertexInCalo1");
  produces<reco::TrackCollection>("InVertexOutCalo1");
  produces<reco::TrackCollection>("InVertexInCalo2");
  produces<reco::TrackCollection>("OutVertexInCalo2");
  produces<reco::TrackCollection>("InVertexOutCalo2");
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
  std::auto_ptr<reco::TrackCollection> inVertexInCaloTracks1(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> outVertexInCaloTracks1(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> inVertexOutCaloTracks1(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> inVertexInCaloTracks2(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> outVertexInCaloTracks2(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection> inVertexOutCaloTracks2(new reco::TrackCollection);
  edm::Handle<reco::CaloJetCollection> unCorrectedJetsHandle;
  event.getByLabel(zspCorrectedJetsTag_,unCorrectedJetsHandle);
  const reco::CaloJetCollection& unCorrectedJets = *unCorrectedJetsHandle;
  if (unCorrectedJets.size() > 0) {
    if (jptCorrector_->canCorrect(unCorrectedJets[0])) {
      jpt::MatchedTracks pions, muons, electrons;
      jptCorrector_->matchTracks(unCorrectedJets[0],event,eventSetup,pions,muons,electrons);
      copyTracks(pions,inVertexInCaloTracks1.get(),outVertexInCaloTracks1.get(),inVertexOutCaloTracks1.get());
      copyTracks(muons,inVertexInCaloTracks1.get(),outVertexInCaloTracks1.get(),inVertexOutCaloTracks1.get());
      copyTracks(electrons,inVertexInCaloTracks1.get(),outVertexInCaloTracks1.get(),inVertexOutCaloTracks1.get());
    }
  }
  if (unCorrectedJets.size() > 1) {
    if (jptCorrector_->canCorrect(unCorrectedJets[1])) {
      jpt::MatchedTracks pions, muons, electrons;
      jptCorrector_->matchTracks(unCorrectedJets[1],event,eventSetup,pions,muons,electrons);
      copyTracks(pions,inVertexInCaloTracks2.get(),outVertexInCaloTracks2.get(),inVertexOutCaloTracks2.get());
      copyTracks(muons,inVertexInCaloTracks2.get(),outVertexInCaloTracks2.get(),inVertexOutCaloTracks2.get());
      copyTracks(electrons,inVertexInCaloTracks2.get(),outVertexInCaloTracks2.get(),inVertexOutCaloTracks2.get());
    }
  }
  event.put(inVertexInCaloTracks1,"InVertexInCalo1");
  event.put(outVertexInCaloTracks1,"OutVertexInCalo1");
  event.put(inVertexOutCaloTracks1,"InVertexOutCalo1");
  event.put(inVertexInCaloTracks2,"InVertexInCalo2");
  event.put(outVertexInCaloTracks2,"OutVertexInCalo2");
  event.put(inVertexOutCaloTracks2,"InVertexOutCalo2");

}

void JPTTrackProducer::copyTracks(const jpt::MatchedTracks& tracks,
                                  reco::TrackCollection* inVertexInCaloTracks,
                                  reco::TrackCollection* outVertexInCaloTracks,
                                  reco::TrackCollection* inVertexOutCaloTracks)
{
  copyTracks(tracks.inVertexInCalo_,inVertexInCaloTracks);
  copyTracks(tracks.outOfVertexInCalo_,outVertexInCaloTracks);
  copyTracks(tracks.inVertexOutOfCalo_,inVertexOutCaloTracks);
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
