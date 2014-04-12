/* \class dimuonHLTFilterCheck
 *
 * author: Davide Piccolo
 *
 * check HLTFilter for dimuon studies:
 *
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Math/interface/deltaR.h"

// access trigger results
#include "FWCore/Common/interface/TriggerNames.h"
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h>
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include <vector>
#include <map>
#include <string>
using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

class dimuonHLTFilterCheck : public edm::EDAnalyzer {
public:
  dimuonHLTFilterCheck(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  vector<int> nDimuonsByType(const Handle<CandidateView> d);
  vector<int> nMuonsByType(const Handle<CandidateView> d);
  virtual void endJob() override;
  EDGetTokenT<TriggerResults> triggerResultsToken;
  EDGetTokenT<CandidateView> tracksToken;
  EDGetTokenT<CandidateView> muonToken;
  EDGetTokenT<CandidateView> anyDimuonToken;


  // general histograms

  // global counters
  int counterMatrix[5][5];
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>
#include <iterator>
#include <cmath>

dimuonHLTFilterCheck::dimuonHLTFilterCheck(const ParameterSet& pset) :
  // trigger results
  triggerResultsToken(consumes<TriggerResults>(pset.getParameter<InputTag>("triggerResults"))),
  tracksToken(consumes<CandidateView>(pset.getParameter<InputTag>("tracks"))),
  muonToken(consumes<CandidateView>(pset.getParameter<InputTag>("muons"))),
  anyDimuonToken(consumes<CandidateView>(pset.getParameter<InputTag>("anyDimuon")))
{
  Service<TFileService> fs;


// general histograms

// general counters
  for (int i=0; i<5; i++) {
    for (int j=0;j<5;j++) {
      counterMatrix[i][j]=0;
    }
  }
}

void dimuonHLTFilterCheck::analyze(const Event& event, const EventSetup& setup) {

  Handle<TriggerResults> triggerResults;   // trigger response
  Handle<CandidateView> anyDimuon;         // any dimuon pair
  Handle<CandidateView> tracks;            // any track
  Handle<CandidateView> muons;             // any muon

  event.getByToken( triggerResultsToken, triggerResults );
  event.getByToken( anyDimuonToken, anyDimuon );
  event.getByToken( tracksToken, tracks );
  event.getByToken( muonToken, muons );

  const edm::TriggerNames & triggerNames = event.triggerNames(*triggerResults);

  // map of MU triggers of interest
  map<string,int> dimuonHLT_triggers;
  dimuonHLT_triggers.insert(make_pair("HLT1MuonPrescalePt3",0));
  dimuonHLT_triggers.insert(make_pair("HLT1MuonPrescalePt7x7",1));
  dimuonHLT_triggers.insert(make_pair("HLT1MuonIso",2));
  dimuonHLT_triggers.insert(make_pair("HLT1MuonNonIso15",3));
  dimuonHLT_triggers.insert(make_pair("HLT2MuonNonIso",4));
  // map of JET triggers of interest
  map<string,int> jetHLT_triggers;
  jetHLT_triggers.insert(make_pair("HLT1jetPE5",0));
  jetHLT_triggers.insert(make_pair("HLT1jetPE3",1));
  jetHLT_triggers.insert(make_pair("HLT1jetPE1",2));
  jetHLT_triggers.insert(make_pair("candHLT1jetPE7",3));

  bool trgMask[5];
  for (int i=0; i<5; i++) trgMask[i] = false;

  // table of possible dimuons
  string dimuonTableNames[10];
  dimuonTableNames[0] = "global-global";
  dimuonTableNames[1] = "global-trackerSta";
  dimuonTableNames[2] = "global-sta";
  dimuonTableNames[3] = "global-tracker";
  dimuonTableNames[4] = "trackerSta-trackerSta";
  dimuonTableNames[5] = "trackerSta-sta";
  dimuonTableNames[6] = "trackerSta-tracker";
  dimuonTableNames[7] = "sta-sta";
  dimuonTableNames[8] = "sta-tracker";
  dimuonTableNames[9] = "tracker-tracker";
  // table of possible muons
  string muonTableNames[10];
  muonTableNames[0] = "global";
  muonTableNames[1] = "trackerSta";
  muonTableNames[2] = "sta";
  muonTableNames[3] = "tracker";

  cout << "-------------------NEW event---------------------------" << endl;
  // check the dimuon reconstruction
  vector<int> dimuonTable;
  dimuonTable = nDimuonsByType(anyDimuon);
  // check the muon reconstruction
  vector<int> muonTable;
  muonTable = nMuonsByType(muons);

  if ( triggerResults.product()->wasrun() ){
      //      cout << "at least one path out of " << triggerResults.product()->size() << " ran? " << triggerResults.product()->wasrun() << endl;

    if ( triggerResults.product()->accept() )
      {
	//  cout << endl << "at least one path accepted? " << triggerResults.product()->accept() << endl;

	const unsigned int n_TriggerResults( triggerResults.product()->size() );
	for ( unsigned int itrig( 0 ); itrig < n_TriggerResults; ++itrig )
	  {
	    if ( triggerResults.product()->accept( itrig ) )
	      {
		map<string,int>::iterator iterMuHLT = dimuonHLT_triggers.find(triggerNames.triggerName( itrig ));
		if (iterMuHLT != dimuonHLT_triggers.end()) {
		  cout << "ecco la chiave Mu HLT " << (*iterMuHLT).second << endl;
		  if (triggerResults.product()->state( itrig )==1) trgMask[(*iterMuHLT).second] = true;
		  }   // end if key found
		map<string,int>::iterator iterjetHLT = jetHLT_triggers.find(triggerNames.triggerName( itrig ));
		if (iterjetHLT != jetHLT_triggers.end()) {
		  cout << "ecco la chiave jet HLT " << (*iterjetHLT).second << endl;
		  }   // end if key found

	      }  // end if trigger accepted
	  }   // end loop on triggerResults
      } // end if at least one triggerResult accepted
  }  // end if wasRun
  if ( muonTable[0]>1) {
    for(unsigned int i = 0; i < muons->size(); ++i) { //loop on candidates
      const Candidate & muCand = (*muons)[i]; //the candidate
      CandidateBaseRef muCandRef = muons->refAt(i);
      TrackRef muStaComponentRef = muCand.get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of muon
      TrackRef muTrkComponentRef = muCand.get<TrackRef>();  // track part of muon
      if (muCandRef->isGlobalMuon()) {
	cout << "muCand : " << i << "  pt " << muCandRef->pt() << "  eta " << muCandRef->eta() << endl;
	cout << "muCandStaComponent : " << i << "  pt " << muStaComponentRef->pt() << "  eta " << muStaComponentRef->eta() << endl;
      }
    }
  }

  // fill counterMatrix
  for (int i=0; i<5; i++) {
    for (int j=0; j<5; j++) {
      if (trgMask[i] && trgMask[j]) counterMatrix[i][j]++;
    }
  }
}  // end analyze

vector<int> dimuonHLTFilterCheck::nDimuonsByType(const Handle<CandidateView> d) {
  vector<int> n_;
  int nCat = 10;    // number of dimuon categories (0 = glb-glb, 1 = glb-trkSta, 2 = glb-sta, 3 = glb-trk, 4 = trkSta-trkSta, 5 = trkSta-sta, 6 = trkSta-trk, 7 = sta-sta, 8 = sta-trk, 9  trk-trk)
  // reset vector
  for (int i=0; i<nCat; i++) n_.push_back(0);
  for(unsigned int i = 0; i < d->size(); ++i) { //loop on candidates
    //      const Candidate & dCand = (*d)[i]; //the candidate
    bool mu0global = false;
    bool mu0trackerSta = false;
    bool mu0sta = false;
    bool mu0tracker = false;
    bool mu1global = false;
    bool mu1trackerSta = false;
    bool mu1sta = false;
    bool mu1tracker = false;
    CandidateBaseRef dCandRef = d->refAt(i);
    const Candidate * mu0 = dCandRef->daughter(0);
    const Candidate * mu1 = dCandRef->daughter(1);
    if (mu0->isGlobalMuon()) mu0global=true;
    if (! mu0->isGlobalMuon() && mu0->isStandAloneMuon() && mu0->isTrackerMuon()) mu0trackerSta=true;
    if (! mu0->isGlobalMuon() && mu0->isStandAloneMuon() && ! mu0->isTrackerMuon()) mu0sta=true;
    if (! mu0->isGlobalMuon() && ! mu0->isStandAloneMuon() && mu0->isTrackerMuon()) mu0tracker=true;
    if (mu1->isGlobalMuon()) mu1global=true;
    if (! mu1->isGlobalMuon() && mu1->isStandAloneMuon() && mu1->isTrackerMuon()) mu1trackerSta=true;
    if (! mu1->isGlobalMuon() && mu1->isStandAloneMuon() && ! mu1->isTrackerMuon()) mu1sta=true;
    if (! mu1->isGlobalMuon() && ! mu1->isStandAloneMuon() && mu1->isTrackerMuon()) mu1tracker=true;

    if (mu0global && mu1global) n_[0]++;
    if ( (mu0global && mu1trackerSta) || (mu1global && mu0trackerSta) ) n_[1]++;
    if ( (mu0global && mu1sta) || (mu1global && mu0sta) ) n_[2]++;
    if ( (mu0global && mu1tracker) || (mu1global && mu0tracker) ) n_[3]++;
    if (mu0trackerSta && mu1trackerSta) n_[4]++;
    if ( (mu0trackerSta && mu1sta) || (mu1trackerSta && mu0sta) ) n_[5]++;
    if ( (mu0trackerSta && mu1tracker) || (mu1trackerSta && mu0tracker) ) n_[6]++;
    if (mu0sta && mu1sta) n_[7]++;
    if ( (mu0sta && mu1tracker) || (mu1sta && mu0tracker) ) n_[8]++;
    if (mu0tracker && mu1tracker) n_[9]++;

  }
  return n_;
}

vector<int> dimuonHLTFilterCheck::nMuonsByType(const Handle<CandidateView> d) {
  vector<int> n_;
  int nCat = 4;    // number of muon categories (0 = glb, 1 = trkSta, 2 = sta, 3 = trk)
  // reset vector
  for (int i=0; i<nCat; i++) n_.push_back(0);
  for(unsigned int i = 0; i < d->size(); ++i) { //loop on candidates
    //      const Candidate & dCand = (*d)[i]; //the candidate
    bool muglobal = false;
    bool mutrackerSta = false;
    bool musta = false;
    bool mutracker = false;
    CandidateBaseRef muCandRef = d->refAt(i);
    if (muCandRef->isGlobalMuon()) muglobal=true;
    if (! muCandRef->isGlobalMuon() && muCandRef->isStandAloneMuon() && muCandRef->isTrackerMuon()) mutrackerSta=true;
    if (! muCandRef->isGlobalMuon() && muCandRef->isStandAloneMuon() && ! muCandRef->isTrackerMuon()) musta=true;
    if (! muCandRef->isGlobalMuon() && ! muCandRef->isStandAloneMuon() && muCandRef->isTrackerMuon()) mutracker=true;
    cout << "muglobal " << muglobal << " mutrackserSta " << mutrackerSta << " must " << musta << " mutracker " << mutracker << endl;
    if (muglobal) n_[0]++;
    if (mutrackerSta) n_[1]++;
    if (musta) n_[2]++;
    if (mutracker) n_[3]++;

  }
  return n_;
}

void dimuonHLTFilterCheck::endJob() {

  cout << "------------------------------------  Counters  --------------------------------" << endl;
  for (int i=0; i<5; i++) {
    cout << "trg " << i << ": ";
    for (int j=0; j<5; j++) {
      cout << counterMatrix[i][j] << " ";
    }
    cout << endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(dimuonHLTFilterCheck);

