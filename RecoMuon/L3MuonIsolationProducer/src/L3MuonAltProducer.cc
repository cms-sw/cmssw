/**  \class L3MuonAltProducer
 * 
 *   \author  J.Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonIsolationProducer/src/L3MuonAltProducer.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3MuonAltProducer::L3MuonAltProducer(const ParameterSet& parameterSet){
  LogDebug("Muon|RecoMuon|L3MuonAltProducer")<<" constructor called";

  // Collection Labels
  theMuonCollectionLabel = parameterSet.getParameter<InputTag>("MuonCollection");
  theTrackCollectionLabel = parameterSet.getParameter<InputTag>("RegionalTrackCollection");
  theMaxChi2PerDof = parameterSet.getParameter<double>("MaxChi2PerDofInMatching");

  produces<TrackCollection>();
}
  
/// destructor
L3MuonAltProducer::~L3MuonAltProducer(){
  LogDebug("Muon|RecoMuon|L3MuonAltProducer")<<" L3MuonAltProducer destructor called";
}


/// reconstruct muons
void L3MuonAltProducer::produce(Event& event, const EventSetup& eventSetup){
  const string metname = "Muon|RecoMuon|L3MuonAltProducer";
  
  // Get the transient track builder
  eventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB);
  if (!theTTB.isValid()) {
      LogDebug(metname)<<" No valid TransientTrackBuilder found!";
      return;
  }

  // Take the L3 container
  LogDebug(metname)<<" Taking the regional track collection: "<<theTrackCollectionLabel.label();
  Handle<TrackCollection> tracks; 
  event.getByLabel(theTrackCollectionLabel,tracks);

  LogDebug(metname)<<" Taking the StandAlone muon collection: "<<theMuonCollectionLabel.label();
  Handle<TrackCollection> muons; 
  event.getByLabel(theMuonCollectionLabel,muons);

  // Create the L3 muon collection
  LogDebug(metname)<<" Creating the L3 muon collection";
  auto_ptr<TrackCollection> l3muons( new TrackCollection());

  for (unsigned int i=0; i<muons->size(); i++) {
      TrackRef muref(muons,i);
      TrackRef tkref = this->muonMatch(muref, tracks);
      if (tkref.isNull()) continue;
      l3muons->push_back(*tkref);
  }
  
  event.put(l3muons);
 
  LogDebug(metname)<<" Event loaded"
		   <<"================================";
}

TrackRef L3MuonAltProducer::muonMatch(const TrackRef& mu, const Handle<TrackCollection>& trks) const {
      const string metname = "Muon|RecoMuon|L3MuonAltProducer";

      TrackRef best;

      if (!theTTB.isValid()) return best;

      TransientTrack* t_mu = (*theTTB).build(&mu);
      TrajectoryStateOnSurface tos_mu = t_mu->impactPointState();
      if (!t_mu->impactPointStateAvailable()) {
            LogDebug(metname)<<" No impactPointStateAvailable for the muon!";
            return best;
      }
      AlgebraicVector par_mu = tos_mu.globalParameters().vector();
      AlgebraicSymMatrix err_mu = tos_mu.cartesianError().matrix();

      vector<TransientTrack> t_trks = (*theTTB).build(trks);

      double chi2min = par_mu.num_row() * theMaxChi2PerDof;
      for (unsigned int it=0; it<t_trks.size(); it++) {
            TransientTrack t_tk = t_trks[it];

            TrajectoryStateOnSurface tos_tk = t_tk.impactPointState();
            if (!t_tk.impactPointStateAvailable()) {
                  LogDebug(metname)<<" No impactPointStateAvailable for track: " << it;
                  continue;
            }

            AlgebraicVector delta = par_mu - tos_tk.globalParameters().vector();
            int ifail = -1;
            AlgebraicSymMatrix covinv = err_mu + tos_tk.cartesianError().matrix();
            covinv.invertCholesky5(ifail);
            if (ifail) {
                  LogDebug(metname)<<" Covariance matrix inversion failed for track: " << it;
                  continue;
            }

            double chi2 = 0.;
            for (unsigned int i=0; i<TrackBase::dimension; i++) {
                  for (unsigned int j=0; j<TrackBase::dimension; j++) {
                        chi2 += delta[i]*covinv[i][j]*delta[j];
                  }
            }
            if (chi2<chi2min) {
                  chi2min = chi2;
                  best = t_tk.persistentTrackRef();
            }
      }

      LogDebug(metname)<<" L2 Muon has been matched to a track: ptmu= " << mu->pt() << ", pttk= " << best->pt() << ", chi2/ndof= " << chi2min/par_mu.num_row();

      return best;
}
