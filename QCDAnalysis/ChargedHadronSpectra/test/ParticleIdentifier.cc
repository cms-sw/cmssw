#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "QCDAnalysis/ChargedHadronSpectra/interface/EcalShowerProperties.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

using namespace std;

/*****************************************************************************/
class ParticleIdentifier : public edm::EDAnalyzer
{
  public:
    explicit ParticleIdentifier(const edm::ParameterSet& ps);
    ~ParticleIdentifier();
    virtual void analyze(const edm::Event& ev, const edm::EventSetup& es) override;
    virtual void endJob();

  private:
    int getParticleId(const reco::Track & recTrack);
    void processEcalRecHits(const edm::Event& ev, const edm::EventSetup& es);
    void processEcalSimHits(const edm::Event& ev);

    TrackerHitAssociator * theHitAssociator;
};

/*****************************************************************************/
ParticleIdentifier::ParticleIdentifier(const edm::ParameterSet& ps)
{ 
}

/*****************************************************************************/
ParticleIdentifier::~ParticleIdentifier()
{
}

/*****************************************************************************/
void ParticleIdentifier::endJob()
{ 
}

/*****************************************************************************/
int ParticleIdentifier::getParticleId(const reco::Track & recTrack)
{  
  int pid = 0;
  double tmin = 0.;

  for(trackingRecHit_iterator recHit = recTrack.recHitsBegin();
                              recHit!= recTrack.recHitsEnd(); recHit++)
  {
    std::vector<PSimHit> simHits = theHitAssociator->associateHit(**recHit);

    for(std::vector<PSimHit>::const_iterator simHit = simHits.begin();
                                        simHit!= simHits.end(); simHit++)
     if(simHit == simHits.begin() || simHit->tof() < tmin)
      {
        pid   = simHit->particleType();
        tmin  = simHit->tof();
      }
  }

  return pid;
}

/*****************************************************************************/
void ParticleIdentifier::processEcalRecHits
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel("globalPrimTracks",  trackCollection);
  const reco::TrackCollection* recTracks = trackCollection.product();
  std::cerr << "[ParticleId] recTracks = " << recTracks->size() << std::endl;

  // EcalShowerProperties;
  EcalShowerProperties theEcalShowerProperties(ev,es); 

  // Look at all tracks
  for(reco::TrackCollection::const_iterator recTrack = recTracks->begin();
                                            recTrack!= recTracks->end();
                                            recTrack++)  
  {
    int ntime = 0;
    std::pair<double,double> result =
      theEcalShowerProperties.processTrack(*recTrack, ntime);
    double energy = result.first;
    double time   = result.second;
    int    pid    = getParticleId(*recTrack);

    std::cerr << " rec"
         << " " << recTrack->p()
         << " " << recTrack->pt()
         << " " << energy
         << " " << time
         << " " << pid
         << " " << ntime
         << std::endl;
  }
}

/*****************************************************************************/
void ParticleIdentifier::processEcalSimHits(const edm::Event& ev)
{
  edm::Handle<edm::PCaloHitContainer>      ecalSimHits;
  ev.getByLabel("g4SimHits", "EcalHitsEB", ecalSimHits);
  
  edm::Handle<edm::SimTrackContainer>  simTracks;
  ev.getByLabel<edm::SimTrackContainer>("g4SimHits", simTracks);
  
  for(edm::SimTrackContainer::const_iterator simTrack = simTracks->begin();
                                             simTrack!= simTracks->end();
                                             simTrack++)
  {
    double p  = simTrack->momentum().R();
    double pt = simTrack->momentum().rho();
    int pid   = simTrack->type();

    double energy = 0, time = 0;
    for(edm::PCaloHitContainer::const_iterator simHit = ecalSimHits->begin();
                                               simHit!= ecalSimHits->end();
                                               simHit++)
      if(simHit->geantTrackId() == int(simTrack->trackId()))
        energy += simHit->energy();

    std::cerr << " sim "
         << " " << p
         << " " << pt
         << " " << energy
         << " " << time
         << " " << pid
         << std::endl;
  }
}

/*****************************************************************************/
void ParticleIdentifier::analyze
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get associator
  theHitAssociator = new TrackerHitAssociator(ev);
  
  // Process ecal rechits
  processEcalRecHits(ev, es);

  // Process ecal simhits
//  processEcalSimHits(ev);
}

/*****************************************************************************/
DEFINE_FWK_MODULE(ParticleIdentifier);
