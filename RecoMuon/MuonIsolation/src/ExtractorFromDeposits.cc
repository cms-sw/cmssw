#include "RecoMuon/MuonIsolation/interface/ExtractorFromDeposits.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/Direction.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

ExtractorFromDeposits::ExtractorFromDeposits( const ParameterSet& par ) :
  theCollectionTag(par.getParameter<edm::InputTag>("IsolationCollectionTag"))
{ }

void ExtractorFromDeposits::fillVetos (const edm::Event & ev, 
    const edm::EventSetup & evSetup, const reco::TrackCollection & muons) 
{ }

MuIsoDeposit ExtractorFromDeposits::deposit(const Event & event, 
    const EventSetup & eventSetup, const Track & muon) const
{ 
  static std::string metname = "RecoMuon|ExtractorFromDeposits";
  Handle<MuIsoDepositAssociationMap> depMap;
  event.getByLabel(theCollectionTag, depMap);

  LogWarning(metname)<<"Call this method only if the original muon track collection is lost";

  // double vtx_z = muon.vz();
  Direction muonDir(muon.eta(), muon.phi());
  MuIsoDepositAssociationMap::const_iterator depPair;
  for (depPair=depMap->begin(); depPair!=depMap->end(); ++depPair) {
    // const TrackRef& tk = depPair->key;
      const MuIsoDeposit& dep = depPair->val;
       Direction depositDir(dep.eta(), dep.phi());
      if (muonDir.deltaR(depositDir) < 1.e-6) return dep;
  } 
  return MuIsoDeposit();
}

MuIsoDeposit ExtractorFromDeposits::deposit(const Event & event, 
    const EventSetup & eventSetup, const TrackRef & muon) const
{ 
  static std::string metname = "RecoMuon|ExtractorFromDeposits";
  Handle<MuIsoDepositAssociationMap> depMap;
  event.getByLabel(theCollectionTag, depMap);

  return (*depMap)[muon];
}
