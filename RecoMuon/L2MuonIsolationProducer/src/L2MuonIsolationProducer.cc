/**  \class L2MuonIsolationProducer
 * 
 *   \author  J. Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/MuonIsolation/interface/Direction.h"
#include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L2MuonIsolationProducer::L2MuonIsolationProducer(const ParameterSet& par){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer constructor called";

  theSACollectionLabel = par.getUntrackedParameter<string>("StandAloneCollectionLabel");

  etaBounds_  = par.getParameter<std::vector<double> > ("EtaBounds");
  coneCuts_  = par.getParameter<std::vector<double> > ("ConeCuts");
  edepCuts_  = par.getParameter<std::vector<double> > ("EdepCuts");

  if (etaBounds_.size()==0) etaBounds_.push_back(999.9); // whole eta range if no input

  if (coneCuts_.size()==0) coneCuts_.push_back(0.0); // no isolation if no input
  if (coneCuts_.size()<etaBounds_.size()) {
      double conelast = coneCuts_[coneCuts_.size()-1];
      int nadd = etaBounds_.size()-coneCuts_.size();
      for (int i=0; i<nadd; i++) coneCuts_.push_back(conelast);
  }  

  if (edepCuts_.size()==0) edepCuts_.push_back(0.0); // no isolation if no input
  if (edepCuts_.size()<etaBounds_.size()) {
      double edeplast = edepCuts_[edepCuts_.size()-1];
      int nadd = etaBounds_.size()-edepCuts_.size();
      for (int i=0; i<nadd; i++) edepCuts_.push_back(edeplast);
  }  

  ecalWeight_  = par.getParameter<double> ("EcalWeight");

  ParameterSet theMuIsoExtractorPSet = par.getParameter<ParameterSet>("MuIsoExtractorParameters");
  theMuIsoExtractor = muonisolation::CaloExtractor(theMuIsoExtractorPSet);
 

  produces<MuIsoDepositAssociationMap>();
  produces<MuIsoAssociationMap>();
}
  
/// destructor
L2MuonIsolationProducer::~L2MuonIsolationProducer(){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer destructor called";
}

/// build deposits
void L2MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "Muon|RecoMuon|L2MuonIsolationProducer";
  
  LogDebug(metname)<<" L2 Muon Isolation producing...";

  // Take the SA container
  LogDebug(metname)<<" Taking the StandAlone muons: "<<theSACollectionLabel;
  Handle<TrackCollection> tracks;
  event.getByLabel(theSACollectionLabel,tracks);

  // Find deposits and load into event
  LogDebug(metname)<<" Get energy around";
  std::auto_ptr<MuIsoDepositAssociationMap> depMap( new MuIsoDepositAssociationMap());
  std::auto_ptr<MuIsoAssociationMap> isoMap( new MuIsoAssociationMap());

  for (unsigned int i=0; i<tracks->size(); i++) {
      TrackRef tk(tracks,i);
      //LogDebug(metname) << " tketa: " << tk->eta();

      MuIsoDeposit depH("HCAL");

      vector<muonisolation::Direction> vetoDirections; // zero size for the time being
      vector<MuIsoDeposit> deposits = theMuIsoExtractor.deposits(event, eventSetup, *tk, vetoDirections, 0.);
      if (deposits.size()!=2) {
            // Should we write a warning here?
            deposits.clear();
            deposits.push_back(MuIsoDeposit("ECAL", tk->eta(), tk->phi()));
            deposits.push_back(MuIsoDeposit("HCAL", tk->eta(), tk->phi()));
      } 
      depMap->insert(tk, deposits[0]);
      depMap->insert(tk, deposits[1]);

      double abseta = fabs(tk->eta());
      int ieta = etaBounds_.size()-1;
      for (unsigned int i=0; i<etaBounds_.size(); i++) {
            if (abseta<etaBounds_[i]) { ieta = i; break; }
      }
      double conesize = coneCuts_[ieta];
      double dephlt = ecalWeight_*deposits[0].depositWithin(conesize)
                       + deposits[1].depositWithin(conesize);
      if (dephlt<edepCuts_[ieta]) {
            isoMap->insert(tk, true);
      } else {
            isoMap->insert(tk, false);
      }
  }
  event.put(depMap);
  event.put(isoMap);

  LogDebug(metname) <<" Event loaded"
		   <<"================================";
}
