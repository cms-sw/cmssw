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

#include "RecoMuon/L3MuonProducer/src/L3MuonAltProducer.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"

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

      double chi2min = TrackBase::dimension * theMaxChi2PerDof;
      for (unsigned int it=0; it<trks->size(); it++) {
            TrackRef tk(trks,it);
            TrackBase::ParameterVector delta = mu->parameters() - tk->parameters();
            TrackBase::CovarianceMatrix cov = mu->covariance()+tk->covariance();
            if (!cov.Invert()) continue;
            double chi2 = 0.;
            for (unsigned int i=0; i<TrackBase::dimension; i++) {
                  for (unsigned int j=0; j<TrackBase::dimension; j++) {
                        chi2 += delta[i]*cov(i,j)*delta[j];
                  }
            }
            if (chi2<chi2min) {
                  chi2min = chi2;
                  best = tk;
            }
      }

      LogDebug(metname)<<" L2 Muon has been matched to a track: ptmu= " << mu->pt() << ", pttk= " << best->pt() << ", chi2/ndof= " << chi2min/TrackBase::dimension;

      return best;
}
