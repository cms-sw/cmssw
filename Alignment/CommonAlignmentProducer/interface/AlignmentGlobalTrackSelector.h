
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentGlobalTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentGlobalTrackSelector_h

//Framework
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//STL
#include <vector>

namespace edm {class Track;}
namespace reco {class Event;}

class AlignmentGlobalTrackSelector
{

 public:

  typedef std::vector<const reco::Track*> Tracks; 

  /// constructor
  AlignmentGlobalTrackSelector(const edm::ParameterSet & cfg);

  /// destructor
  ~AlignmentGlobalTrackSelector();

  /// select tracks
  Tracks select(const Tracks& tracks, const edm::Event& iEvent);

 bool useThisFilter();
 private:
  Tracks checkJetCount(const Tracks& cands,const edm::Event& iEvent)const;
  Tracks checkIsolation(const Tracks& cands,const edm::Event& iEvent)const;
  Tracks findMuons(const Tracks& tracks,const edm::Event& iEvent)const;

  /// private data members
  edm::ParameterSet theConf;

  //settings from conigfile
  bool theIsoFilterSwitch;
  bool theJetCountFilterSwitch;
  bool theGMFilterSwitch;
  //global Muon Filter
  edm::InputTag theMuonSource;
  int theMinGlobalMuonCount;
  //isolation Cut
  edm::InputTag theJetIsoSource;
  double theMaxJetPt;
  double theMinJetDeltaR;
  double theMaxTrackDeltaR;
  int theMinIsolatedCount;
  //jet count Filter
  edm::InputTag theJetCountSource;
  double theMinJetPt;
  int theMaxJetCount;

  //helpers
  //  double deltaR(const reco::Track* t1,const reco::Track* t2) const;
  //  double deltaR(const reco::Track* t,const reco::Particle& p) const;
  void printTracks(const Tracks& col) const;
  Tracks matchTracks(const Tracks& src, const Tracks& comp) const;

};

#endif

