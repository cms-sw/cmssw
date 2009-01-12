/**
  \class    MuonsFromRefitTracks "RecoMuon/MuonIdentification/plugins/MuonsFromRefitTracks.cc"
  \brief    Replaces the kinematic information in the input muons with those of the chosen refit tracks.

  \author   Jordan Tucker
  \version  $Id: MuonsFromRefitTracksProducer.cc,v 1.4 2008/12/12 16:14:23 tucker Exp $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

using namespace std;
using namespace edm;
using namespace reco;

TrackRef tevOptimizedTMR(const Muon& muon, const TrackToTrackMap& fmsMap,
			 const double cut) {
  const TrackRef& combinedTrack = muon.globalTrack();
  const TrackRef& trackerTrack  = muon.innerTrack();

  TrackToTrackMap::const_iterator fmsTrack = fmsMap.find(combinedTrack);

  double probTK  = 0;
  double probFMS = 0;

  if (trackerTrack.isAvailable() && trackerTrack->numberOfValidHits())
    probTK = muon::trackProbability(trackerTrack);
  if (fmsTrack != fmsMap.end() && fmsTrack->val->numberOfValidHits())
    probFMS = muon::trackProbability(fmsTrack->val);

  bool TKok  = probTK > 0;
  bool FMSok = probFMS > 0;

  if (TKok && FMSok) {
    if (probFMS - probTK > cut)
      return trackerTrack;
    else
      return fmsTrack->val;
  }
  else if (FMSok)
    return fmsTrack->val;
  else if (TKok)
    return trackerTrack;
  else
    return combinedTrack;
}

class MuonsFromRefitTracksProducer : public EDProducer {
public:
  explicit MuonsFromRefitTracksProducer(const ParameterSet&);
  ~MuonsFromRefitTracksProducer() {}

private:
  virtual void beginJob(const EventSetup&) {}
  virtual void produce(Event&, const EventSetup&);
  virtual void endJob() {}

  // Store the track-to-track map(s) used when using TeV refit tracks.
  bool storeMatchMaps(const Event& event);

  // Take the muon passed in, clone it (so that we save all the muon
  // id information such as isolation, calo energy, etc.) and replace
  // its combined muon track with the passed in track.
  Muon* cloneAndSwitchTrack(const Muon& muon,
			    const TrackRef& newTrack) const;

  // The input muons -- i.e. the merged collection of reco::Muons.
  InputTag src;

  // Allow building the muon from just the tracker track. This
  // functionality should go away after understanding the difference
  // between the output of option 1 of GlobalMuonProducer and just
  // looking at the tracker tracks of these muons.
  bool fromTrackerTrack;

  // If tevMuonTracks below is not "none", use the TeV refit track as
  // the combined track of the muon.
  bool fromTeVRefit;

  // Optionally switch out the combined muon track for one of the TeV
  // muon refit tracks, specified by the input tag here
  // (e.g. "tevMuons:firstHit").
  string tevMuonTracks;

  // Whether to make a cocktail muon instead of using just the one
  // type in tevMuonTracks, where "cocktail" means use the result of
  // Piotr's tevOptimized().
  bool fromCocktail;

  // Whether to use the TMR version of the cocktail function, defined
  // above. If true, overrides fromCocktail.
  bool fromTMR;

  // The cut value for TMR, read from the config file.
  double TMRcut;

  // If we're not making cocktail muons, trackMap is the map that maps
  // global tracks to the desired TeV refit (e.g. from globalMuons to
  // tevMuons:picky).
  Handle<TrackToTrackMap> trackMap;

  // All the track maps used in making cocktail muons.
  Handle<TrackToTrackMap> trackMapDefault;
  Handle<TrackToTrackMap> trackMapFirstHit;
  Handle<TrackToTrackMap> trackMapPicky;
};

MuonsFromRefitTracksProducer::MuonsFromRefitTracksProducer(const ParameterSet& cfg)
  : src(cfg.getParameter<InputTag>("src")),
    fromTrackerTrack(cfg.getUntrackedParameter<bool>("fromTrackerTrack", false)),
    tevMuonTracks(cfg.getUntrackedParameter<string>("tevMuonTracks", "none")),
    fromCocktail(cfg.getUntrackedParameter<bool>("fromCocktail", false)),
    fromTMR(cfg.getUntrackedParameter<bool>("fromTMR", false)),
    TMRcut(cfg.getUntrackedParameter<double>("TMRcut", 3.5))
{
  fromTeVRefit = tevMuonTracks != "none";
  produces<MuonCollection>();
}

bool MuonsFromRefitTracksProducer::storeMatchMaps(const Event& event) {
  if (fromCocktail || fromTMR) {
    event.getByLabel(tevMuonTracks, "default",  trackMapDefault);
    event.getByLabel(tevMuonTracks, "firstHit", trackMapFirstHit);
    event.getByLabel(tevMuonTracks, "picky",    trackMapPicky);
    return !trackMapDefault.failedToGet() && 
      !trackMapFirstHit.failedToGet() && !trackMapPicky.failedToGet();
  }
  else {
    event.getByLabel(InputTag(tevMuonTracks), trackMap);
    return !trackMap.failedToGet();
  }
}

Muon* MuonsFromRefitTracksProducer::cloneAndSwitchTrack(const Muon& muon,
							const TrackRef& newTrack) const {
  // Muon mass to make a four-vector out of the new track.
  static const double muMass = 0.10566;

  TrackRef tkTrack  = muon.innerTrack();
  TrackRef muTrack  = muon.outerTrack();
	  
  // Make up a real Muon from the tracker track.
  Particle::Point vtx(newTrack->vx(), newTrack->vy(), newTrack->vz());
  Particle::LorentzVector p4;
  double p = newTrack->p();
  p4.SetXYZT(newTrack->px(), newTrack->py(), newTrack->pz(),
	     sqrt(p*p + muMass*muMass));

  Muon* mu = muon.clone();
  mu->setCharge(newTrack->charge());
  mu->setP4(p4);
  mu->setVertex(vtx);
  mu->setGlobalTrack(newTrack);
  mu->setInnerTrack(tkTrack);
  mu->setOuterTrack(muTrack);
  return mu;
}

void MuonsFromRefitTracksProducer::produce(Event& event, const EventSetup& eSetup) {
  // Get the global muons from the event.
  Handle<View<Muon> > muons;
  event.getByLabel(src, muons);

  // If we can't get the global muon collection, or below the
  // track-to-track maps needed, still produce an empty collection of
  // muons so consumers don't throw an exception.
  bool ok = !muons.failedToGet();

  // If we're instructed to use the TeV refit tracks in some way, we
  // need the track-to-track maps. If we're making a cocktail muon,
  // get all three track maps (the cocktail ingredients); else just
  // get the map which takes the above global tracks to the desired
  // TeV-muon refitted tracks (firstHit or picky).
  if (ok && fromTeVRefit)
    ok = storeMatchMaps(event);

  // Make the output collection.
  auto_ptr<MuonCollection> cands(new MuonCollection);

  if (ok) {
    View<Muon>::const_iterator muon;
    for (muon = muons->begin(); muon != muons->end(); muon++) {
      if (!muon->isGlobalMuon()) continue;

      if (fromTeVRefit) {
	// Start out with a null TrackRef.
	TrackRef tevTk;
      
	// If making a cocktail muon, use tevOptimized() to get the track
	// desired. Otherwise, get the refit track from the desired track
	// map.
	if (fromTMR)
	  tevTk = tevOptimizedTMR(*muon, *trackMapFirstHit, TMRcut);
	else if (fromCocktail)
	  tevTk = muon::tevOptimized(*muon, *trackMapDefault, *trackMapFirstHit,
				     *trackMapPicky);
	else {
	  TrackToTrackMap::const_iterator tevTkRef =
	    trackMap->find(muon->combinedMuon());
	  if (tevTkRef != trackMap->end())
	    tevTk = tevTkRef->val;
	}
	
	// If the TrackRef is valid, make a new MuonTrackLinks that has
	// the same tracker and stand-alone tracks, but has the refit
	// track as its global track.
	if (tevTk.isNonnull())
	  cands->push_back(*cloneAndSwitchTrack(*muon, tevTk));
      }
      else if (fromTrackerTrack)
	cands->push_back(*cloneAndSwitchTrack(*muon, muon->innerTrack()));
      else {
	cands->push_back(*muon->clone());

	// Just cloning does not work in the case of the source being
	// a pat::Muon with embedded track references -- these do not
	// get copied. Explicitly set them.
	Muon& last = cands->at(cands->size()-1);
	if (muon->globalTrack().isTransient())
	  last.setGlobalTrack(muon->globalTrack());
	if (muon->innerTrack().isTransient())
	  last.setInnerTrack(muon->innerTrack());
	if (muon->outerTrack().isTransient())
	  last.setOuterTrack(muon->outerTrack());
      }
    }
  }
  else
    edm::LogWarning("MuonsFromRefitTracksProducer")
      << "either " << src << " or the track map(s) " << tevMuonTracks
      << " not present in the event; producing empty collection";
  
  event.put(cands);
}

DEFINE_FWK_MODULE(MuonsFromRefitTracksProducer);
