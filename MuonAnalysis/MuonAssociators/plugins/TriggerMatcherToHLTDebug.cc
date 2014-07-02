// -*- C++ -*-
//
// Package:    MuonAnalysis/MuonAssociators
// Class:      TriggerMatcherToHLTDebug
//
/**\class TriggerMatcherToHLTDebug TriggerMatcherToHLTDebug.cc MuonAnalysis/MuonAssociators/plugins/TriggerMatcherToHLTDebug.cc

 Description: Matches RECO muons to Trigger ones using HLTDEBUG information.
              Muon is first matched to L1 using the PropagateToMuon tool from this same package,
              then *all* compatible L1s are examined and the corresponding L2 and L3 objects are searched
              using the references inside those objects.
*/
//
// Original Author:  Cristina Botta (Torino), Giovanni Petrucciani (UCSD)
//         Created:  Fri 30 Apr 2010
//


#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

//new for association map
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"






class TriggerMatcherToHLTDebug: public edm::EDProducer {

    public:
        // Constructor
        explicit TriggerMatcherToHLTDebug(const edm::ParameterSet &pset);

        /// Destructor
        virtual ~TriggerMatcherToHLTDebug();

        // Operations
        void produce(edm::Event & event, const edm::EventSetup& eventSetup) override;
        void beginRun(const edm::Run &run, const edm::EventSetup &eventSetup) override;

    private:
        typedef edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed> > > SeedMap;

        edm::EDGetTokenT<edm::View<reco::Muon> > tagToken_;
        edm::EDGetTokenT<l1extra::L1MuonParticleCollection> l1Token_;
        PropagateToMuon l1matcher_;

        std::string metname;

        //ForL1Assoc
        double deltaR_;

        //ForL1Quality
        int   minL1Quality_;

        //ForL2Filter
        edm::EDGetTokenT<reco::BeamSpot> beamspotToken_ ;
        int    min_N_L2;
        double max_Eta_L2;
        int    min_Nhits_L2;
        double max_Dr_L2;
        double max_Dz_L2;
        double min_Pt_L2;
        double nsigma_Pt_L2;

        //ForL3Filter
        int    min_N_L3;
        double max_Eta_L3;
        int    min_Nhits_L3;
        double max_Dr_L3;
        double max_Dz_L3;
        double min_Pt_L3;
        double nsigma_Pt_L3;

        edm::EDGetTokenT<L2MuonTrajectorySeedCollection> theL2SeedsToken_;
        edm::EDGetTokenT<reco::RecoChargedCandidateCollection> theL2MuonsToken_;
        edm::EDGetTokenT<L3MuonTrajectorySeedCollection> theL3SeedsToken_;
        edm::EDGetTokenT<reco::TrackCollection> theL3TkTracksToken_;
        edm::EDGetTokenT<reco::RecoChargedCandidateCollection> theL3MuonsToken_;
        edm::EDGetTokenT<SeedMap> seedMapToken_;

        /// Store extra information in a ValueMap
        template<typename T>
        void storeValueMap(edm::Event &iEvent,
                const edm::Handle<edm::View<reco::Muon> > & handle,
                const std::vector<T> & values,
                const std::string    & label) const ;

};

using namespace std;
using namespace edm;
using namespace l1extra;
using namespace reco;

// Constructor
TriggerMatcherToHLTDebug::TriggerMatcherToHLTDebug(const edm::ParameterSet &pset):
  tagToken_(consumes<View<reco::Muon> >(pset.getParameter<edm::InputTag>("tags"))),
  l1Token_(consumes<L1MuonParticleCollection>(pset.getParameter<edm::InputTag>("l1s"))),
  l1matcher_(pset.getParameter<edm::ParameterSet>("l1matcherConfig")),
  deltaR_(pset.getParameter<double>("deltaR")),
  minL1Quality_(pset.getParameter<int32_t>("MinL1Quality")),
  beamspotToken_(consumes<BeamSpot>(pset.getParameter<edm::InputTag>("BeamSpotTag"))),
  min_N_L2(pset.getParameter<int> ("MinN_L2")),
  max_Eta_L2(pset.getParameter<double> ("MaxEta_L2")),
  min_Nhits_L2(pset.getParameter<int> ("MinNhits_L2")),
  max_Dr_L2(pset.getParameter<double> ("MaxDr_L2")),
  max_Dz_L2(pset.getParameter<double> ("MaxDz_L2")),
  min_Pt_L2(pset.getParameter<double> ("MinPt_L2")),
  nsigma_Pt_L2(pset.getParameter<double> ("NSigmaPt_L2")),
  min_N_L3(pset.getParameter<int> ("MinN_L3")),
  max_Eta_L3(pset.getParameter<double> ("MaxEta_L3")),
  min_Nhits_L3(pset.getParameter<int> ("MinNhits_L3")),
  max_Dr_L3(pset.getParameter<double> ("MaxDr_L3")),
  max_Dz_L3(pset.getParameter<double> ("MaxDz_L3")),
  min_Pt_L3(pset.getParameter<double> ("MinPt_L3")),
  nsigma_Pt_L3(pset.getParameter<double> ("NSigmaPt_L3")),
  seedMapToken_(consumes<SeedMap>(pset.getParameter<edm::InputTag >("SeedMapTag")))
{


  theL2SeedsToken_ = consumes<L2MuonTrajectorySeedCollection>(pset.getParameter<InputTag>("L2Seeds_Collection"));
  theL2MuonsToken_ = consumes<RecoChargedCandidateCollection>(pset.getParameter<InputTag>("L2Muons_Collection"));
  theL3SeedsToken_ = consumes<L3MuonTrajectorySeedCollection>(pset.getParameter<InputTag>("L3Seeds_Collection"));
  theL3TkTracksToken_ = consumes<TrackCollection>(pset.getParameter<InputTag>("L3TkTracks_Collection"));
  theL3MuonsToken_ = consumes<RecoChargedCandidateCollection>(pset.getParameter<InputTag>("L3Muons_Collection"));


  metname = "TriggerMatcherToHLTDebug";

  produces<edm::ValueMap<int> > ("propagatesToM2");
  produces<edm::ValueMap<int> > ("hasL1Particle");
  produces<edm::ValueMap<int> > ("hasL1Filtered");
  produces<edm::ValueMap<int> > ("hasL2Seed");
  produces<edm::ValueMap<int> > ("hasL2Muon");
  produces<edm::ValueMap<int> > ("hasL2MuonFiltered");
  produces<edm::ValueMap<int> > ("hasL3Seed");
  produces<edm::ValueMap<int> > ("hasL3Track");
  produces<edm::ValueMap<int> > ("hasL3Muon");
  produces<edm::ValueMap<int> > ("hasL3MuonFiltered");

  produces<edm::ValueMap<reco::CandidatePtr> > ("l1Candidate");
  produces<edm::ValueMap<reco::CandidatePtr> > ("l2Candidate");
  produces<edm::ValueMap<reco::CandidatePtr> > ("l3Candidate");
}



// Destructor
TriggerMatcherToHLTDebug::~TriggerMatcherToHLTDebug() {}

// Analyzer
void TriggerMatcherToHLTDebug::produce(Event &event, const EventSetup &eventSetup) {

  Handle<View<reco::Muon> > muons;
  event.getByToken(tagToken_,muons);

  Handle<l1extra::L1MuonParticleCollection> L1Muons;
  event.getByToken(l1Token_,L1Muons);

  Handle<L2MuonTrajectorySeedCollection> L2Seeds;
  event.getByToken(theL2SeedsToken_,L2Seeds);

  Handle<RecoChargedCandidateCollection> L2Muons;
  event.getByToken(theL2MuonsToken_,L2Muons);

  Handle<L3MuonTrajectorySeedCollection> L3Seeds;
  event.getByToken(theL3SeedsToken_,L3Seeds);

  Handle<reco::TrackCollection> L3TkTracks;
  event.getByToken(theL3TkTracksToken_,L3TkTracks);

  Handle<RecoChargedCandidateCollection> L3Muons;
  event.getByToken(theL3MuonsToken_,L3Muons);

  //beam spot
  BeamSpot beamSpot;
  Handle<BeamSpot> recoBeamSpotHandle;
  event.getByToken(beamspotToken_,recoBeamSpotHandle);
  beamSpot = *recoBeamSpotHandle;

  //new for the MAP!!!!
  edm::Handle<SeedMap> seedMapHandle;
  event.getByToken(seedMapToken_, seedMapHandle);


  size_t nmu = muons->size();
  std::vector<int> propagatesToM2(nmu), hasL1Particle(nmu), hasL1Filtered(nmu);
  std::vector<int> hasL2Seed(nmu), hasL2Muon(nmu), hasL2MuonFiltered(nmu);
  std::vector<int> hasL3Seed(nmu), hasL3Track(nmu), hasL3TrackFiltered(nmu), hasL3Muon(nmu), hasL3MuonFiltered(nmu);
  std::vector<reco::CandidatePtr> l1ptr(nmu), l2ptr(nmu), l3ptr(nmu);

  for (size_t i = 0; i < nmu; ++i) {
    const reco::Muon &mu = (*muons)[i];

    // Propagate to muon station (using the L1 tool)
    TrajectoryStateOnSurface stateAtMB2 = l1matcher_.extrapolate(mu);
    if (!stateAtMB2.isValid())  continue;
    propagatesToM2[i] = 1;

    double etaTk = stateAtMB2.globalPosition().eta();
    double phiTk = stateAtMB2.globalPosition().phi();
    l1extra::L1MuonParticleCollection::const_iterator it;
    vector<l1extra::L1MuonParticleRef>::const_iterator itMu3;
    L2MuonTrajectorySeedCollection::const_iterator iSeed;
    L3MuonTrajectorySeedCollection::const_iterator iSeedL3;
    RecoChargedCandidateCollection::const_iterator iL2Muon;
    reco::TrackCollection::const_iterator tktrackL3;
    RecoChargedCandidateCollection::const_iterator iL3Muon;

    reco::CandidatePtr thisL1, thisL2, thisL3;
    for(it = L1Muons->begin(); it != L1Muons->end(); ++it) {

      const L1MuGMTExtendedCand muonCand = (*it).gmtMuonCand();
      unsigned int quality =  muonCand.quality();

      double L1phi =(*it).phi();
      double L1eta =(*it).eta();
      double L1pt =(*it).pt();
      double dR=deltaR(etaTk,phiTk,L1eta,L1phi);

      //CONDIZIONE-> CE NE E' UNA ASSOCIATA?
      if (dR >= deltaR_) continue;
      thisL1 = reco::CandidatePtr(L1Muons, it - L1Muons->begin());
      if (!hasL1Particle[i]) l1ptr[i] = thisL1; // if nobody reached L1 before, then we're the best L1 found up to now.
      hasL1Particle[i]++;

      if ((quality <= 3) || (L1pt<7)) continue;
      if (!hasL1Filtered[i]) l1ptr[i] = thisL1; // if nobody reached L1 before, then we're the best L1 found up to now.
      hasL1Filtered[i]++;

      if(!L2Seeds.isValid()) continue;
      //LOOP SULLA COLLEZIONE DEI SEED
      for( iSeed = L2Seeds->begin(); iSeed != L2Seeds->end(); ++iSeed) {

	l1extra::L1MuonParticleRef l1FromSeed = iSeed->l1Particle();
	if (l1FromSeed.id() != L1Muons.id()) throw cms::Exception("CorruptData") << "You're using a different L1 collection than the one used by L2 seeds.\n";
	if (l1FromSeed.key() != thisL1.key()) continue;
	if (!hasL2Seed[i]) l1ptr[i] = thisL1; // if nobody reached here before, we're the best L1
	hasL2Seed[i]++;

	if(!L2Muons.isValid()) continue;
	//LOOP SULLA COLLEZIONE L2MUON
	for( iL2Muon = L2Muons->begin(); iL2Muon != L2Muons->end(); ++iL2Muon) {


	  //MI FACCIO DARE REF E GUARDO SE E' UGUALE AL L2SEED ASSOCIATO
	  //BEFORE THE ASSOCIATION MAP!!!!!
	  //edm::Ref<L2MuonTrajectorySeedCollection> l2seedRef = iL2Muon->track()->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >();
	  //l1extra::L1MuonParticleRef l1FromL2 = l2seedRef->l1Particle();

	  //if (l1FromL2.id() != l1FromSeed.id()) throw cms::Exception("CorruptData") << "You're using L2s with a different L1 collection than the one used by L2 seeds.\n";
	  //if (l1FromL2 != l1FromSeed) continue;

	  //AFTER THE ASSOCIATION MAP
	  const edm::RefVector<L2MuonTrajectorySeedCollection>& seeds = (*seedMapHandle)[iL2Muon->track()->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >()];
	  //	  bool isTriggered = false;
	  for(size_t jjj=0; jjj<seeds.size(); jjj++){

	    if(seeds[jjj]->l1Particle()!= l1FromSeed) continue;

	  }


	  thisL2 = reco::CandidatePtr(L2Muons, iL2Muon - L2Muons->begin()) ;
	  if (!hasL2Muon[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; } // if nobody reached here before, we're the best L1 and L2)
	  hasL2Muon[i]++;

	  LogTrace(metname) <<"L2MUON TROVATO!"<<endl;
	  const reco::Track & L2Track = *iL2Muon->track();
	  double Eta_L2= L2Track.eta();
	  double Pt_L2= L2Track.pt();
	  int nValidHits_L2= L2Track.numberOfValidHits();
	  double BSPos_L2 = L2Track.dxy(beamSpot.position());
	  double dz_L2 =L2Track.dz();
	  double err0_L2 = L2Track.error(0);
	  double abspar0_L2 = fabs(L2Track.parameter(0));
	  double ptLx_L2 = Pt_L2;
	  if (abspar0_L2>0) ptLx_L2 += nsigma_Pt_L2*err0_L2/abspar0_L2*Pt_L2;

	  //GUARDO SE L2MUON ASSOCIATO AVREBBE PASSATO IL FILTRO
	  bool passFilter = (((fabs(Eta_L2))<=max_Eta_L2)&&(nValidHits_L2>=min_Nhits_L2)&&((fabs(BSPos_L2))<=max_Dr_L2)&&((fabs(dz_L2))<=max_Dz_L2)&&(ptLx_L2>=min_Pt_L2));
	  if (!passFilter) continue;
	  if (!hasL2MuonFiltered[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; } // if nobody reached here before, we're the best L1 and L2)
	  hasL2MuonFiltered[i]++;

	  const reco::TrackRef L2FilteredRef = iL2Muon->track();

	  //########L3 PART##############
	  if (!L3Seeds.isValid()) continue;
	  for (iSeedL3 = L3Seeds->begin(); iSeedL3!= L3Seeds->end(); ++iSeedL3){

	    TrackRef staTrack = iSeedL3->l2Track();
	    if (staTrack!=L2FilteredRef) continue;
	    if (!hasL3Seed[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; } // if nobody reached here before, we're the best L1 and L2)
	    hasL3Seed[i]++;

	    if (!L3TkTracks.isValid()) continue;
	    for (tktrackL3 = L3TkTracks->begin(); tktrackL3!= L3TkTracks->end(); ++tktrackL3){

	      edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tktrackL3->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
	      TrackRef staTrack2 = l3seedRef->l2Track();

	      if (staTrack2!=L2FilteredRef) continue;
	      if (!hasL3Track[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; } // if nobody reached here before, we're the best L1 and L2)
	      hasL3Track[i]++;

	      if (!L3Muons.isValid()) continue;
	      for (iL3Muon = L3Muons->begin(); iL3Muon != L3Muons->end(); ++iL3Muon) {

		edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef2 = iL3Muon->track()->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
		TrackRef staTrack3 = l3seedRef2->l2Track();

		if (staTrack3!=L2FilteredRef) continue;
		thisL3 = reco::CandidatePtr(L3Muons, iL3Muon - L3Muons->begin());

		if (!hasL3Muon[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; l3ptr[i] = thisL3; } // if nobody reached here before, we're the best L1, L2, L3
		hasL3Muon[i]++;

		const reco::Track &L3Track = *iL3Muon->track();
		double Eta_L3= L3Track.eta();
		double Pt_L3= L3Track.pt();
		int nValidHits_L3= L3Track.numberOfValidHits();
		double BSPos_L3 = L3Track.dxy(beamSpot.position());
		double dz_L3 =L3Track.dz();
		double err0_L3 = L3Track.error(0);
		double abspar0_L3 = fabs(L3Track.parameter(0));
		double ptLx_L3 = Pt_L3;

		if (abspar0_L3>0) ptLx_L3 += nsigma_Pt_L3*err0_L3/abspar0_L3*Pt_L3;

		if(((fabs(Eta_L3))<=max_Eta_L3)&&(nValidHits_L3>=min_Nhits_L3)&&((fabs(BSPos_L3))<=max_Dr_L3)&&((fabs(dz_L3))<=max_Dz_L3)&&(ptLx_L3>=min_Pt_L3)){

		  if (!hasL3MuonFiltered[i]) { l1ptr[i] = thisL1; l2ptr[i] = thisL2; l3ptr[i] = thisL3; } // if nobody reached here before, we're the best L1, L2, L3
		  hasL3MuonFiltered[i]++;

		}//L3MUON FILTERED ASSOCIATO TROVATO
	      }//L3MUON LOOP
	    }// L3 TRACKS
	  }// L3 SEEDS
	}//T L2 MUONS
      }// L2 SEEDS
    }//L1 MUONS
  } // RECO MUONS
  storeValueMap<int>(event, muons, propagatesToM2,    "propagatesToM2");
  storeValueMap<int>(event, muons, hasL1Particle,     "hasL1Particle");
  storeValueMap<int>(event, muons, hasL1Filtered,     "hasL1Filtered");
  storeValueMap<int>(event, muons, hasL2Seed,         "hasL2Seed");
  storeValueMap<int>(event, muons, hasL2Muon,         "hasL2Muon");
  storeValueMap<int>(event, muons, hasL2MuonFiltered, "hasL2MuonFiltered");
  storeValueMap<int>(event, muons, hasL3Seed,         "hasL3Seed");
  storeValueMap<int>(event, muons, hasL3Track,        "hasL3Track");
  storeValueMap<int>(event, muons, hasL3Muon,         "hasL3Muon");
  storeValueMap<int>(event, muons, hasL3MuonFiltered, "hasL3MuonFiltered");
  storeValueMap<reco::CandidatePtr>(event, muons, l1ptr, "l1Candidate");
  storeValueMap<reco::CandidatePtr>(event, muons, l2ptr, "l2Candidate");
  storeValueMap<reco::CandidatePtr>(event, muons, l3ptr, "l3Candidate");
} // METHOD

void
TriggerMatcherToHLTDebug::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  l1matcher_.init(iSetup);
}

template<typename T>
void
TriggerMatcherToHLTDebug::storeValueMap(edm::Event &iEvent,
					const edm::Handle<edm::View<reco::Muon> > & handle,
					const std::vector<T> & values,
					const std::string    & label) const {
  using namespace edm; using namespace std;
  auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(valMap, label);
}




DEFINE_FWK_MODULE(TriggerMatcherToHLTDebug);
