#include "RecoTauTag/HLTProducers/interface/TauJetSelectorForHLTTrackSeeding.h"

#include "DataFormats/Math/interface/deltaPhi.h"


TauJetSelectorForHLTTrackSeeding::TauJetSelectorForHLTTrackSeeding(const edm::ParameterSet& iConfig):
  ptMinCaloJet_(iConfig.getParameter< double > ("ptMinCaloJet")),
  etaMinCaloJet_(iConfig.getParameter< double > ("etaMinCaloJet")),
  etaMaxCaloJet_(iConfig.getParameter< double > ("etaMaxCaloJet")),
  tauConeSize_(iConfig.getParameter< double > ("tauConeSize")),
  isolationConeSize_(iConfig.getParameter< double > ("isolationConeSize")),
  fractionMinCaloInTauCone_(iConfig.getParameter< double > ("fractionMinCaloInTauCone")),
  fractionMaxChargedPUInCaloCone_(iConfig.getParameter< double > ("fractionMaxChargedPUInCaloCone")),
  ptTrkMaxInCaloCone_(iConfig.getParameter< double > ("ptTrkMaxInCaloCone")),
  nTrkMaxInCaloCone_(iConfig.getParameter< int > ("nTrkMaxInCaloCone"))
{
   //now do what ever initialization is needed
  inputTrackJetToken_ = consumes< reco::TrackJetCollection >(iConfig.getParameter< edm::InputTag > ("inputTrackJetTag"));
  inputCaloJetToken_  = consumes< reco::CaloJetCollection >(iConfig.getParameter< edm::InputTag > ("inputCaloJetTag"));
  inputTrackToken_    = consumes< reco::TrackCollection >(iConfig.getParameter< edm::InputTag > ("inputTrackTag"));

  produces<reco::TrackJetCollection>();
}


TauJetSelectorForHLTTrackSeeding::~TauJetSelectorForHLTTrackSeeding()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
void
TauJetSelectorForHLTTrackSeeding::produce(edm::StreamID iStreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   std::auto_ptr< reco::TrackJetCollection > augmentedTrackJets (new reco::TrackJetCollection);

   edm::Handle<reco::TrackJetCollection> trackjets;
   iEvent.getByToken(inputTrackJetToken_, trackjets);

   for (reco::TrackJetCollection::const_iterator trackjet = trackjets->begin();
	  trackjet != trackjets->end(); trackjet++) {
     augmentedTrackJets->push_back(*trackjet);
   }

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(inputTrackToken_,tracks);

   edm::Handle<reco::CaloJetCollection> calojets;
   iEvent.getByToken(inputCaloJetToken_,calojets);

   const double tauConeSize2       = tauConeSize_ * tauConeSize_;
   const double isolationConeSize2 = isolationConeSize_ * isolationConeSize_;

   for (reco::CaloJetCollection::const_iterator calojet = calojets->begin();
	  calojet != calojets->end(); calojet++) {

     if ( calojet->pt() < ptMinCaloJet_ ) continue;
     double etaJet = calojet->eta();
     double phiJet = calojet->phi();
     if ( etaJet < etaMinCaloJet_ ) continue;
     if ( etaJet > etaMaxCaloJet_ ) continue;

     std::vector <CaloTowerPtr> const & theTowers = calojet->getCaloConstituents();
     double ptIn = 0.;
     double ptOut = 0.;
     for ( unsigned int itwr = 0; itwr < theTowers.size(); ++itwr ) { 
       double etaTwr = theTowers[itwr]->eta() - etaJet;
       double phiTwr = deltaPhi(theTowers[itwr]->phi(), phiJet);
       double deltaR2 = etaTwr*etaTwr + phiTwr*phiTwr;
       //std::cout << "Tower eta/phi/et : " << etaTwr << " " << phiTwr << " " << theTowers[itwr]->pt() << std::endl;
       if ( deltaR2 < tauConeSize2 ) { 
	 ptIn += theTowers[itwr]->pt(); 
       } else if ( deltaR2 < isolationConeSize2 ) { 
	 ptOut += theTowers[itwr]->pt(); 
       }
     }
     double ptTot = ptIn+ptOut;
     double fracIn = ptIn/ptTot;

     // We are looking for isolated tracks
     if ( fracIn < fractionMinCaloInTauCone_) continue;

     int ntrk = 0;
     double ptTrk = 0.;

     for (reco::TrackJetCollection::const_iterator trackjet = trackjets->begin();
	  trackjet != trackjets->end(); trackjet++) {
       for (unsigned itr=0; itr<trackjet->numberOfTracks(); ++itr) { 
	 edm::Ptr<reco::Track> track = trackjet->track(itr);
	 double trackEta = track->eta() - etaJet;
	 double trackPhi = deltaPhi(track->phi(), phiJet);
	 double deltaR2 = trackEta*trackEta + trackPhi*trackPhi;
	 if ( deltaR2 < isolationConeSize2 ) { 
	   ntrk++; 
	   ptTrk += track->pt();
	 }
       }
     }
     // We are looking for calojets without signal tracks already in
     if ( ntrk > nTrkMaxInCaloCone_ ) continue;
     if ( ptTrk > ptTrkMaxInCaloCone_ ) continue;

     int ntrk2 = 0;
     double ptTrk2 = 0.;

     for (reco::TrackCollection::const_iterator track = tracks->begin();
	  track != tracks->end(); track++) {
       double trackEta = track->eta() - etaJet;
       double trackPhi = deltaPhi(track->phi(), phiJet);
       double deltaR2 = trackEta*trackEta + trackPhi*trackPhi;
       if ( deltaR2 < isolationConeSize2 ) { 
	 ntrk2++; 
	 ptTrk2 += track->pt();
       }
     }
     // We are looking for signal jets, not PU jets
     double fractionChargedPU = ptTrk2/calojet->pt(); 
     if ( fractionChargedPU > fractionMaxChargedPUInCaloCone_ ) continue;
     /*
     std::cout << "Calo Jet " << calojet->pt() << " " << calojet->eta() 
	       << " " << ptIn << " " << ptOut << " " << fracIn
	       << " " << ptTrk << " " << ntrk 
	       << " " << fractionChargedPU
	       << std::endl;
     */
     math::XYZTLorentzVector p4(calojet->p4());
     math::XYZPoint vertex(calojet->vertex());
     augmentedTrackJets->push_back(reco::TrackJet(p4,vertex));
   }

   iEvent.put(augmentedTrackJets);
}

// ------------ method called once each job just before starting event loop  ------------
void 
TauJetSelectorForHLTTrackSeeding::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void 
TauJetSelectorForHLTTrackSeeding::endJob() {}

void
TauJetSelectorForHLTTrackSeeding::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTrackJetTag",edm::InputTag("hltAntiKT5TrackJetsIter0"))->setComment("Oryginal TrackJet collection");
  desc.add<edm::InputTag>("inputCaloJetTag",edm::InputTag("hltAntiKT5CaloJetsPFEt5"))->setComment("CaloJet collection");
  desc.add<edm::InputTag>("inputTrackTag",edm::InputTag("hltPFlowTrackSelectionHighPurity"))->setComment("Track collection for track isolation");
  desc.add<double>("ptMinCaloJet",5.0)->setComment("Min. pT of CaloJet");
  desc.add<double>("etaMinCaloJet",-2.7)->setComment("Min. eta of CaloJet");
  desc.add<double>("etaMaxCaloJet",+2.7)->setComment("Max. eta of CaloJet");
  desc.add<double>("tauConeSize",0.2)->setComment("Radius of tau signal cone");
  desc.add<double>("isolationConeSize",0.5)->setComment("Radius of isolation cone");
  desc.add<double>("fractionMinCaloInTauCone",0.8)->setComment("Min. fraction of calo energy in tau signal cone");
  desc.add<double>("fractionMaxChargedPUInCaloCone",0.2)->setComment("Max. fraction of PU charged energy Sum(Pt_trks)/Pt_jet in signal cone");
  desc.add<double>("ptTrkMaxInCaloCone",1.0)->setComment("Max. sum of pT of tracks in isolation cone");
  desc.add<int>("nTrkMaxInCaloCone",0)->setComment("Max. no. of tracks in isolation cone");
  descriptions.setComment("This module produces a collection of TrackJets to seed iterative tracking.\nIt is done by enriching the input TrackJet collection with CaloJets passing isolation critera - tau candidates"); 
  descriptions.add("tauJetSelectorForHLTTrackSeeding",desc);
}
