
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTwoBodyDecayTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTwoBodyDecayTrackSelector_h

//Framework
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//STL
#include <vector>
// forward declaration:
namespace edm {class Track;}
namespace reco {class Event;}

class AlignmentTwoBodyDecayTrackSelector
{
 public:

  typedef std::vector<const reco::Track*> Tracks; 

  /// constructor
  AlignmentTwoBodyDecayTrackSelector(const edm::ParameterSet & cfg);

  /// destructor
  ~AlignmentTwoBodyDecayTrackSelector();

  /// select tracks
  Tracks select(const Tracks& tracks, const edm::Event& iEvent);

  bool useThisFilter();
 private:
  Tracks checkMass(const Tracks& cands)const;
  Tracks checkMETMass(const Tracks& cands,const edm::Event& iEvent)const;
  Tracks checkCharge(const Tracks& cands)const;
  Tracks checkAcoplanarity(const Tracks& cands)const;
  Tracks checkMETAcoplanarity(const Tracks& cands,const edm::Event& iEvent)const; 
  /// private data members
  edm::ParameterSet theCfg;

  //settings from conigfile
  bool theMassrangeSwitch;
  bool theChargeSwitch;
  bool theMissingETSwitch;
  bool theAcoplanarityFilterSwitch;
  //inv mass Cut
  double theMinMass;
  double theMaxMass;
  double theDaughterMass;
  //charge filter
  int theCharge;
  bool theUnsignedSwitch;
  //missing ET Filter
  edm::InputTag theMissingETSource;
  //acoplanarity Filter
  double theAcoplanarDistance;
  //helpers
  void printTracks(const Tracks& col) const;
};

#endif

