/**
  \class    MuonsFromRefitTracks "RecoMuon/MuonIdentification/plugins/MuonsFromRefitTracks.cc"
  \brief    Replaces the kinematic information in the input muons with those of the chosen refit tracks.

  \author   Jordan Tucker
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

reco::Muon::MuonTrackTypePair tevOptimizedTMR(const reco::Muon& muon, const reco::TrackToTrackMap& fmsMap,
				const double cut) {
  const reco::TrackRef& combinedTrack = muon.globalTrack();
  const reco::TrackRef& trackerTrack  = muon.innerTrack();

  reco::TrackToTrackMap::const_iterator fmsTrack = fmsMap.find(combinedTrack);

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
      return make_pair(trackerTrack,reco::Muon::InnerTrack);
    else
      return make_pair(fmsTrack->val,reco::Muon::TPFMS);
  }
  else if (FMSok)
    return make_pair(fmsTrack->val,reco::Muon::TPFMS);
  else if (TKok)
    return make_pair(trackerTrack,reco::Muon::InnerTrack);

  return make_pair(combinedTrack,reco::Muon::CombinedTrack);
}

 reco::Muon::MuonTrackTypePair sigmaSwitch(const reco::Muon& muon, const double nSigma, const double ptThreshold) {
  const reco::TrackRef& combinedTrack = muon.globalTrack();
  const reco::TrackRef& trackerTrack  = muon.innerTrack();

  if (combinedTrack->pt() < ptThreshold || trackerTrack->pt() < ptThreshold)
    return make_pair(trackerTrack,reco::Muon::InnerTrack);

  double delta = fabs(trackerTrack->qoverp() - combinedTrack->qoverp());
  double threshold = nSigma * trackerTrack->qoverpError();

  return delta > threshold ? make_pair(trackerTrack,reco::Muon::InnerTrack) : make_pair(combinedTrack,reco::Muon::CombinedTrack);
}

class MuonsFromRefitTracksProducer : public edm::stream::EDProducer<> {
public:
  explicit MuonsFromRefitTracksProducer(const edm::ParameterSet&);
  ~MuonsFromRefitTracksProducer() {}

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  // Store the track-to-track map(s) used when using TeV refit tracks.
  bool storeMatchMaps(const edm::Event& event);

  // Take the muon passed in, clone it (so that we save all the muon
  // id information such as isolation, calo energy, etc.) and replace
  // its combined muon track with the passed in track.
  reco::Muon* cloneAndSwitchTrack(const reco::Muon& muon,
				  const reco::Muon::MuonTrackTypePair& newTrack) const;

  // The input muons -- i.e. the merged collection of reco::Muons.
  edm::InputTag src;
 
  // Allow building the muon from just the tracker track. This
  // functionality should go away after understanding the difference
  // between the output of option 1 of GlobalMuonProducer and just
  // looking at the tracker tracks of these muons.
  bool fromTrackerTrack;

  // Allow building the muon from just the global track. This option
  // is introduced since starting from CMSSW 3_1_0, the MuonIdProducer
  // makes the p4() of the reco::Muon object be what we call the sigma
  // switch above.
  bool fromGlobalTrack;

  // If tevMuonTracks below is not "none", use the TeV refit track as
  // the combined track of the muon.
  bool fromTeVRefit;

  // Optionally switch out the combined muon track for one of the TeV
  // muon refit tracks, specified by the input tag here
  // (e.g. "tevMuons:firstHit").
  std::string tevMuonTracks;

  // Whether to make a cocktail muon instead of using just the one
  // type in tevMuonTracks, where "cocktail" means use the result of
  // Piotr's tevOptimized().
  bool fromCocktail;

  // Whether to use the TMR version of the cocktail function, defined
  // above. If true, overrides fromCocktail.
  bool fromTMR;

  // The cut value for TMR, read from the config file.
  double TMRcut;

  // Whether to use Adam Everett's sigma-switch method, choosing
  // between the global track and the tracker track.
  bool fromSigmaSwitch;
  
  // The number of sigma to switch on in the above method.
  double nSigmaSwitch;
  
  // The pT threshold to switch at in the above method.
  double ptThreshold;

  // If we're not making cocktail muons, trackMap is the map that maps
  // global tracks to the desired TeV refit (e.g. from globalMuons to
  // tevMuons:picky).
  edm::Handle<reco::TrackToTrackMap> trackMap;

  // All the track maps used in making cocktail muons.
  edm::Handle<reco::TrackToTrackMap> trackMapDefault;
  edm::Handle<reco::TrackToTrackMap> trackMapFirstHit;
  edm::Handle<reco::TrackToTrackMap> trackMapPicky;



  // All the tokens
  edm::EDGetTokenT<edm::View<reco::Muon> > srcToken_;
  edm::EDGetTokenT<reco::TrackToTrackMap> trackMapToken_;
  edm::EDGetTokenT<reco::TrackToTrackMap> trackMapDefaultToken_;
  edm::EDGetTokenT<reco::TrackToTrackMap> trackMapFirstHitToken_;
  edm::EDGetTokenT<reco::TrackToTrackMap> trackMapPickyToken_;



};

MuonsFromRefitTracksProducer::MuonsFromRefitTracksProducer(const edm::ParameterSet& cfg)
  : src(cfg.getParameter<edm::InputTag>("src")),
    fromTrackerTrack(cfg.getParameter<bool>("fromTrackerTrack")),
    fromGlobalTrack(cfg.getParameter<bool>("fromGlobalTrack")),
    tevMuonTracks(cfg.getParameter<std::string>("tevMuonTracks")),
    fromCocktail(cfg.getParameter<bool>("fromCocktail")),
    fromTMR(cfg.getParameter<bool>("fromTMR")),
    TMRcut(cfg.getParameter<double>("TMRcut")),
    fromSigmaSwitch(cfg.getParameter<bool>("fromSigmaSwitch")),
    nSigmaSwitch(cfg.getParameter<double>("nSigmaSwitch")),
    ptThreshold(cfg.getParameter<double>("ptThreshold"))
{
  fromTeVRefit = tevMuonTracks != "none";


  srcToken_ = consumes<edm::View<reco::Muon> >(src) ;
  trackMapToken_ = consumes<reco::TrackToTrackMap> (edm::InputTag(tevMuonTracks, "default"));
  trackMapDefaultToken_ = consumes<reco::TrackToTrackMap>(edm::InputTag(tevMuonTracks)) ;
  trackMapFirstHitToken_ = consumes<reco::TrackToTrackMap>(edm::InputTag(tevMuonTracks, "firstHit"));
  trackMapPickyToken_ = consumes<reco::TrackToTrackMap> (edm::InputTag(tevMuonTracks, "picky"));




  produces<reco::MuonCollection>();
}

bool MuonsFromRefitTracksProducer::storeMatchMaps(const edm::Event& event) {
  if (fromCocktail || fromTMR) {
    event.getByToken(trackMapDefaultToken_,trackMapDefault);
    event.getByToken(trackMapFirstHitToken_, trackMapFirstHit);
    event.getByToken(trackMapPickyToken_,    trackMapPicky);
    return !trackMapDefault.failedToGet() && 
      !trackMapFirstHit.failedToGet() && !trackMapPicky.failedToGet();
  }
  else {
    event.getByToken(trackMapToken_, trackMap);
    return !trackMap.failedToGet();
  }
}

reco::Muon* MuonsFromRefitTracksProducer::cloneAndSwitchTrack(const reco::Muon& muon,
							      const  reco::Muon::MuonTrackTypePair& newTrack) const {
  // Muon mass to make a four-vector out of the new track.
  static const double muMass = 0.10566;

  reco::TrackRef tkTrack  = muon.innerTrack();
  reco::TrackRef muTrack  = muon.outerTrack();
	  
  // Make up a real Muon from the tracker track.
  reco::Particle::Point vtx(newTrack.first->vx(), newTrack.first->vy(), newTrack.first->vz());
  reco::Particle::LorentzVector p4;
  double p = newTrack.first->p();
  p4.SetXYZT(newTrack.first->px(), newTrack.first->py(), newTrack.first->pz(),
	     sqrt(p*p + muMass*muMass));

  reco::Muon* mu = muon.clone();
  mu->setCharge(newTrack.first->charge());
  mu->setP4(p4);
  mu->setVertex(vtx);
  mu->setGlobalTrack(newTrack.first);
  mu->setInnerTrack(tkTrack);
  mu->setOuterTrack(muTrack);
  mu->setBestTrack(newTrack.second);
  return mu;
}

void MuonsFromRefitTracksProducer::produce(edm::Event& event, const edm::EventSetup& eSetup) {
  // Get the global muons from the event.
  edm::Handle<edm::View<reco::Muon> > muons;
  event.getByToken(srcToken_, muons);

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
  std::auto_ptr<reco::MuonCollection> cands(new reco::MuonCollection);

  if (ok) {
    edm::View<reco::Muon>::const_iterator muon;
    for (muon = muons->begin(); muon != muons->end(); muon++) {
      // Filter out the so-called trackerMuons and stand-alone muons
      // (and caloMuons, if they were ever to get into the input muons
      // collection).
      if (!muon->isGlobalMuon()) continue;

      if (fromTeVRefit || fromSigmaSwitch) {
	// Start out with a null TrackRef.
	reco::Muon::MuonTrackTypePair tevTk;
      
	// If making a cocktail muon, use tevOptimized() to get the track
	// desired. Otherwise, get the refit track from the desired track
	// map.
	if (fromTMR)
	  tevTk = tevOptimizedTMR(*muon, *trackMapFirstHit, TMRcut);
	else if (fromCocktail)
          tevTk = muon::tevOptimized(*muon);	  
	else if (fromSigmaSwitch)
	  tevTk = sigmaSwitch(*muon, nSigmaSwitch, ptThreshold);
	else {
        reco::TrackToTrackMap::const_iterator tevTkRef =
	    trackMap->find(muon->combinedMuon());
	  if (tevTkRef != trackMap->end())
	    tevTk = make_pair(tevTkRef->val,reco::Muon::CombinedTrack);
	}
	
	// If the TrackRef is valid, make a new Muon that has the same
	// tracker and stand-alone tracks, but has the refit track as
	// its global track.
	if (tevTk.first.isNonnull())
	  cands->push_back(*cloneAndSwitchTrack(*muon, tevTk));
      }
      else if (fromTrackerTrack)
	cands->push_back(*cloneAndSwitchTrack(*muon, make_pair(muon->innerTrack(),reco::Muon::InnerTrack)));
      else if (fromGlobalTrack)
	cands->push_back(*cloneAndSwitchTrack(*muon, make_pair(muon->globalTrack(),reco::Muon::CombinedTrack)));
      else {
	cands->push_back(*muon->clone());

	// Just cloning does not work in the case of the source being
	// a pat::Muon with embedded track references -- these do not
	// get copied. Explicitly set them.
    reco::Muon& last = cands->at(cands->size()-1);
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
