
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTwoBodyDecayTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTwoBodyDecayTrackSelector_h

//Framework
#include "FWCore/Utilities/interface/InputTag.h"
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
  ///checks if the mass of the mother is in the mass region
  Tracks checkMass(const Tracks& cands)const;
  ///checks if the mass of the mother is in the mass region adding missing E_T
  Tracks checkMETMass(const Tracks& cands,const edm::Event& iEvent)const;
  ///checks if the mother has charge = [theCharge]
  Tracks checkCharge(const Tracks& cands)const;
  ///checks if the [cands] are acoplanar (returns empty set if not)
  Tracks checkAcoplanarity(const Tracks& cands)const;
  ///checks if [cands] contains a acoplanar track w.r.t missing ET (returns empty set if not)
  Tracks checkMETAcoplanarity(const Tracks& cands,const edm::Event& iEvent)const; 
  /// private data members

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
  ///print Information on Track-Collection
  void printTracks(const Tracks& col) const;
};

#endif

