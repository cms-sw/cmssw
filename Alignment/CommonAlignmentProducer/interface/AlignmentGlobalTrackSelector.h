#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentGlobalTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentGlobalTrackSelector_h

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//STL
#include <vector>

namespace edm {class Track;}
namespace reco {class Event;}

class AlignmentGlobalTrackSelector
{

 public:

  typedef std::vector<const reco::Track*> Tracks; 

  /// constructor
  AlignmentGlobalTrackSelector(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);

  /// destructor
  ~AlignmentGlobalTrackSelector();

  /// select tracks
  Tracks select(const Tracks& tracks, const edm::Event& iEvent, const edm::EventSetup& eSetup);
  ///returns if any of the Filters is used.
  bool useThisFilter();
 
 private:
  
  ///returns [tracks] if there are less than theMaxCount Jets with theMinJetPt and an empty set if not
  Tracks checkJetCount(const Tracks& cands,const edm::Event& iEvent)const;
  ///returns only isolated tracks in [cands]
  Tracks checkIsolation(const Tracks& cands,const edm::Event& iEvent)const;
  ///filter for Tracks that match the Track of a global Muon
  Tracks findMuons(const Tracks& tracks,const edm::Event& iEvent)const;

  /// private data members
  edm::ParameterSet theConf;

  //settings from conigfile
  bool theGMFilterSwitch;
  bool theIsoFilterSwitch;
  bool theJetCountFilterSwitch;

  //global Muon Filter
  edm::EDGetTokenT<reco::MuonCollection> theMuonToken;
  double theMaxTrackDeltaR;
  int theMinGlobalMuonCount;

  //isolation Cut
  edm::EDGetTokenT<reco::CaloJetCollection> theJetIsoToken;
  double theMaxJetPt;
  double theMinJetDeltaR;
  int theMinIsolatedCount;

  //jet count Filter
  edm::EDGetTokenT<reco::CaloJetCollection> theJetCountToken;
  double theMinJetPt;
  int theMaxJetCount;

  //helpers

  ///print Information on Track-Collection
  void printTracks(const Tracks& col) const;

  ///matches [src] with [comp] returns collection with matching Tracks coming from [src]
  Tracks matchTracks(const Tracks& src, const Tracks& comp) const;
};

#endif

