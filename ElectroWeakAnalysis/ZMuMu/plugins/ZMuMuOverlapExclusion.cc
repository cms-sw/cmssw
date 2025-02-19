#include "CommonTools/UtilAlgos/interface/OverlapExclusionSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include <iostream>

struct ZMuMuOverlap {
  ZMuMuOverlap(const edm::ParameterSet&) { }
  bool operator()(const reco::Candidate & zMuMu, const reco::Candidate & z) const {
    
    using namespace std;
    using namespace reco;
    // check if a candidate z is different from zMuMu  
    // (for example a Z can be done with two global muons, or with a global muon plus a standalone muon.
    // if the standalone muon is part of the second global muon in fact this is the same Z)
  
    unsigned int nd1 = zMuMu.numberOfDaughters();
    unsigned int nd2 = z.numberOfDaughters();
    
    assert(nd1==2 && nd2==2);
    const int maxd = 2;
    const Candidate * daughters1[maxd];
    const Candidate * daughters2[maxd];
    TrackRef trackerTrack1[maxd];
    TrackRef stAloneTrack1[maxd];
    TrackRef globalTrack1[maxd];
    TrackRef trackerTrack2[maxd];
    TrackRef stAloneTrack2[maxd];
    TrackRef globalTrack2[maxd];
    bool flag;
    unsigned int matched=0;
    
    for( unsigned int i = 0; i < nd1; ++ i ) {
      daughters1[i] = zMuMu.daughter( i );
      trackerTrack1[i] = daughters1[i]->get<TrackRef>();
      stAloneTrack1[i] = daughters1[i]->get<TrackRef,reco::StandAloneMuonTag>();
      globalTrack1[i]  = daughters1[i]->get<TrackRef,reco::CombinedMuonTag>();
      
      /*********************************************** just used for debug ********************
    if (trackerTrack1[i].isNull()) 
      cout << "in ZMuMu daughter " << i << " tracker ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " tracker ref FOUND" 
	   << " id: " << trackerTrack1[i].id() << ", index: " << trackerTrack1[i].key() 
	   << endl;
    if (stAloneTrack1[i].isNull()) 
      cout << "in ZMuMu daughter " << i << " stalone ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " stalone ref FOUND" 
	   << " id: " << stAloneTrack1[i].id() << ", index: " << stAloneTrack1[i].key() 
	   << endl;
    
    if (globalTrack1[i].isNull()) 
      cout << "in ZMuMu daughter " << i << " global ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " global ref FOUND"  
	   << " id: " << globalTrack1[i].id() << ", index: " << globalTrack1[i].key() 
	   << endl;
      */
    }
    for( unsigned int i = 0; i < nd2; ++ i ) {
      daughters2[i] = z.daughter( i );
      trackerTrack2[i] = daughters2[i]->get<TrackRef>();
      stAloneTrack2[i] = daughters2[i]->get<TrackRef,reco::StandAloneMuonTag>();
      globalTrack2[i]  = daughters2[i]->get<TrackRef,reco::CombinedMuonTag>();
      
      /******************************************** just used for debug ************
    if (trackerTrack2[i].isNull()) 
      cout << "in ZMuSta daughter " << i << " tracker ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " tracker ref FOUND"  
	   << " id: " << trackerTrack2[i].id() << ", index: " << trackerTrack2[i].key() 
	   << endl;
    if (stAloneTrack2[i].isNull()) 
      cout << "in ZMuSta daughter " << i << " standalone ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " standalone ref FOUND" 
	   << " id: " << stAloneTrack2[i].id() << ", index: " << stAloneTrack2[i].key() 
	   << endl;
    
    if (globalTrack2[i].isNull()) 
      cout << "in ZMuSta daughter " << i << " global ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " global ref FOUND" 
	   << " id: " << globalTrack2[i].id() << ", index: " << globalTrack2[i].key() 
	   << endl;
	   
      */  
    }
    for (unsigned int i = 0; i < nd1; i++) {
      flag = false;
      for (unsigned int j = 0; j < nd2; j++) {           // if the obj2 is a standalone the trackref is alwais in the trackerTRack position
	if ( ((trackerTrack2[i].id()==trackerTrack1[j].id()) && (trackerTrack2[i].key()==trackerTrack1[j].key())) ||
	     ((trackerTrack2[i].id()==stAloneTrack1[j].id()) && (trackerTrack2[i].key()==stAloneTrack1[j].key())) ) {
	  flag = true;
	}
      }
      if (flag) matched++;
    }
    if (matched==nd1) // return true if all the childrens of the ZMuMu have a children matched in ZMuXX
      return true;
    else 
      return false;
  }
};

typedef SingleObjectSelector<
  edm::View<reco::Candidate>,
  OverlapExclusionSelector<reco::CandidateView, 
			   reco::Candidate, 
			   ZMuMuOverlap>
  > ZMuMuOverlapExclusionSelector;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuOverlapExclusionSelector);


