/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   L2 muon selector for L3 IO:
 *   finds L2 muons not previous converted into L3 muons
 *
 *   \author  Benjamin Radburn-Smith - Purdue University
 */

#include "RecoMuon/L3TrackFinder/interface/HLTMuonL2SelectorForL3IO.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

/// constructor with config
HLTMuonL2SelectorForL3IO::HLTMuonL2SelectorForL3IO(const edm::ParameterSet& iConfig):
		l2Src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l2Src"))),
		l3OISrc_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l3OISrc"))),
		useOuterHitPosition_(iConfig.getParameter<bool>("useOuterHitPosition")),
		xDiffMax_(iConfig.getParameter<double>("xDiffMax")),
		yDiffMax_(iConfig.getParameter<double>("yDiffMax")),
		zDiffMax_(iConfig.getParameter<double>("zDiffMax")),
		dRDiffMax_(iConfig.getParameter<double>("dRDiffMax")){
	LogTrace("Muon|RecoMuon|HLTMuonL2SelectorForL3IO")<<"constructor called";
	produces<reco::TrackCollection>();
}
  
/// destructor
HLTMuonL2SelectorForL3IO::~HLTMuonL2SelectorForL3IO(){}

/// create collection of L2 muons not already reconstructed as L3 muons
void HLTMuonL2SelectorForL3IO::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
	const std::string metname = "Muon|RecoMuon|HLTMuonL2SelectorForL3IO";
  
//  IN:
	edm::Handle<reco::TrackCollection> l2TrackCol;
	iEvent.getByToken(l2Src_, l2TrackCol);

	edm::Handle<reco::TrackCollection> l3TrackCol;
	iEvent.getByToken(l3OISrc_, l3TrackCol);

//	OUT:
	std::auto_ptr<std::vector<reco::Track> > result(new std::vector<reco::Track>());

	auto const& l2Tracks = *l2TrackCol.product();
	auto const& l3Tracks = *l3TrackCol.product();
	for (auto&& l2 : l2Tracks){
		bool l2found=false;
		for (auto&& l3 : l3Tracks){
			if (useOuterHitPosition_){
				// If x,y are within 0.5cm of each other: L2 has been found already
				// z found to change beyond this (perhaps a refitting issue?)
				if (std::abs(l2.outerPosition().X()-l3.outerPosition().X()) < xDiffMax_ &&
					std::abs(l2.outerPosition().Y()-l3.outerPosition().Y()) < yDiffMax_ &&
					std::abs(l2.outerPosition().Z()-l3.outerPosition().Z()) < zDiffMax_){
					l2found=true;
				}
			}
			else{
				if (deltaR(l3,l2)<dRDiffMax_) l2found=true;
			}
		}
		if (!l2found) result->push_back(l2);
	}

  iEvent.put(result);
}

void HLTMuonL2SelectorForL3IO::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l2Src",edm::InputTag("hltL2Muons","UpdatedAtVtx"));
  desc.add<edm::InputTag>("l3OISrc",edm::InputTag("hltL2Muons","UpdatedAtVtx"));
  desc.add<bool>("useOuterHitPosition",true); //ToDo: Check whether untracked or tracked
  desc.add<double>("xDiffMax",0.5);
  desc.add<double>("yDiffMax",0.5);
  desc.add<double>("zDiffMax",9999.0);
  desc.add<double>("dRDiffMax",0.01);
  descriptions.add("HLTMuonL2SelectorForL3IO",desc);
}
