#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentMuonSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentMuonSelector_h

/** \class ALignmentMuonSelector
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, IFCA
 *
 * \version $Revision: 1.5 $
 *
 * $Id: AlignmentMuonSelector.h,v 1.5 2010/08/19 13:53:08 jfernan2 Exp $
 *
 */

#include "DataFormats/MuonReco/interface/Muon.h"
#include "CommonTools/RecoAlgos/interface/MuonSelector.h"
#include <vector>

namespace edm { class Event; }

class AlignmentMuonSelector 
{

 public:

  typedef std::vector<const reco::Muon*> Muons; 

  /// constructor
  AlignmentMuonSelector(const edm::ParameterSet & cfg);

  /// destructor
  ~AlignmentMuonSelector();

  /// select muons
  Muons select(const Muons& muons, const edm::Event& evt) const;

 private:

  /// apply basic cuts on pt,eta,phi,nhit
  Muons basicCuts(const Muons& muons) const;

  /// filter the n highest pt muons
  Muons theNHighestPtMuons(const Muons& muons) const;
  
  /// filter only those muons giving best mass pair combination
  Muons theBestMassPairCombinationMuons(const Muons& muons) const;
  
  /// compare two muons in pt (used by theNHighestPtMuons)
  struct ComparePt {
    bool operator()( const reco::Muon* t1, const reco::Muon* t2 ) const {
      return t1->pt()> t2->pt();
    }
  };
  ComparePt ptComparator;

  /// private data members
  bool applyBasicCuts,applyNHighestPt,applyMultiplicityFilter,applyMassPairFilter;
  int nHighestPt,minMultiplicity;
  double pMin,pMax,ptMin,ptMax,etaMin,etaMax,phiMin,phiMax;
  double nHitMinSA,nHitMaxSA,chi2nMaxSA;
  double nHitMinGB,nHitMaxGB,chi2nMaxGB;
  double nHitMinTO,nHitMaxTO,chi2nMaxTO;
  double minMassPair,maxMassPair;

};

#endif

