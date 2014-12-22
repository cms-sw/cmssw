//-------------------------------------------------
//
/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   Level-2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from Level-1 trigger seeds.
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

#include "RecoMuon/L3TrackFinder/interface/HLTMuonL2SelectorForL3IO.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
//#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
//#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
//#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
//#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
//#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
//#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
//
//#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
//#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
HLTMuonL2SelectorForL3IO::HLTMuonL2SelectorForL3IO(const ParameterSet& iConfig):
		l2Src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l2Src"))),
		l3OISrc_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l3OISrc"))){

	LogTrace("Muon|RecoMuon|HLTMuonL2SelectorForL3IO")<<"constructor called"<<endl;

//  edm::ConsumesCollector  iC = consumesCollector();
  
  produces<reco::TrackCollection>();
//  produces<reco::TrackCollection>("UpdatedAtVtx");
//  produces<TrackingRecHitCollection>();
//  produces<reco::TrackExtraCollection>();
//  produces<reco::TrackToTrackMap>();
//
//  produces<std::vector<Trajectory> >();
//  produces<TrajTrackAssociationCollection>();
//
//  produces<edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed> > > >();
}
  
/// destructor
HLTMuonL2SelectorForL3IO::~HLTMuonL2SelectorForL3IO(){}

/// reconstruct muons
void HLTMuonL2SelectorForL3IO::produce(Event& iEvent, const EventSetup& iSetup){
  
 const std::string metname = "Muon|RecoMuon|HLTMuonL2SelectorForL3IO";
//    LogTrace(metname)
 	 std::cout <<"L2 Selection for L3IO Started"<<endl;
  
//  IN:
	edm::Handle<reco::TrackCollection> l2TrackCol;
	iEvent.getByToken(l2Src_, l2TrackCol);
	std::cout <<"l2TrackCol->size(): " << l2TrackCol->size() << std::endl;

	edm::Handle<reco::TrackCollection> l3TrackCol;

	try{
		iEvent.getByToken(l3OISrc_, l3TrackCol);
		std::cout <<"l3TrackCol->size(): " << l3TrackCol->size() << std::endl;
	}
	catch (...) {
		std::cout << "no L3 available" << std::endl;
	}
	 std::cout <<"L2 Selection: In collected"<< std::endl;


//	OUT:
	std::auto_ptr<std::vector<reco::Track> > result(new std::vector<reco::Track>());

	 std::cout <<"L2 Selection: Out created"<<endl;

	for (unsigned int l2TrackColIndex(0);l2TrackColIndex!=l2TrackCol->size();++l2TrackColIndex){
		std::cout << "L2 #: " << l2TrackColIndex << std::endl;
		const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);
		std::cout <<"L2 reference made"<< std::endl;
		bool l2found=false;
		try{
			for (unsigned int l3TrackColIndex(0);l3TrackColIndex!=l3TrackCol->size();++l3TrackColIndex){
				std::cout << "\tL3 #: " << l3TrackColIndex << std::endl;
				const reco::TrackRef l3(l3TrackCol, l3TrackColIndex);
				std::cout <<"L3 reference made"<< std::endl;

				std::cout << "\t(deltaR(*l3,*l2): " << deltaR(*l3,*l2) << std::endl;
				if (deltaR(*l3,*l2)<0.01) l2found=true;
			}
		}
		catch (...) {
			std::cout << "no L3 available" << std::endl;
		}
		if (!l2found) result->push_back(*l2);
	}

  iEvent.put(result);
  
  LogTrace(metname)<<"Event loaded"<<"================================"<<endl;
}

