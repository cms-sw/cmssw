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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
//#include <string>

/// constructor with config
HLTMuonL2SelectorForL3IO::HLTMuonL2SelectorForL3IO(const edm::ParameterSet& iConfig):
		l2Src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l2Src"))),
		l3OISrc_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l3OISrc"))){
	LogTrace("Muon|RecoMuon|HLTMuonL2SelectorForL3IO")<<"constructor called";
	produces<reco::TrackCollection>();
}
  
/// destructor
HLTMuonL2SelectorForL3IO::~HLTMuonL2SelectorForL3IO(){}

/// reconstruct muons
void HLTMuonL2SelectorForL3IO::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
	const std::string metname = "Muon|RecoMuon|HLTMuonL2SelectorForL3IO";
  
//  IN:
	edm::Handle<reco::TrackCollection> l2TrackCol;
	iEvent.getByToken(l2Src_, l2TrackCol);

	edm::Handle<reco::TrackCollection> l3TrackCol;
	try{
		iEvent.getByToken(l3OISrc_, l3TrackCol);
	}
	catch (...) {
		LogTrace(metname) << "no L3 available";
	}

//	OUT:
	std::auto_ptr<std::vector<reco::Track> > result(new std::vector<reco::Track>());

	for (unsigned int l2TrackColIndex(0);l2TrackColIndex!=l2TrackCol->size();++l2TrackColIndex){
		const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);
		bool l2found=false;
		try{
			for (unsigned int l3TrackColIndex(0);l3TrackColIndex!=l3TrackCol->size();++l3TrackColIndex){
				const reco::TrackRef l3(l3TrackCol, l3TrackColIndex);
				LogTrace(metname) << "\t(deltaR(*l3,*l2): " << deltaR(*l3,*l2);
				if (deltaR(*l3,*l2)<0.01) l2found=true;
				// ToDo: try with the shared hits between the objects
			}
		}
		catch (...) {
			LogTrace(metname) << "Cannot match between L2 and L3 due to missing L3 collection";
		}
		if (!l2found) result->push_back(*l2);
	}

  iEvent.put(result);
}

