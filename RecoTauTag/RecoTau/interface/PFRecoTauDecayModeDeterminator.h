#ifndef RecoTauTag_RecoTau_PFRecoTauDecayModeDeterminator
#define RecoTauTag_RecoTau_PFRecoTauDecayModeDeterminator

/* class PFRecoTauDecayModeDeterminator
 *
 * Takes PFCandidates from PFTau and reconstructs tau decay mode.
 * Notably, merges photons (PFGammas) into pi zeros.
 * PFChargedHadrons are assumed to be charged pions.
 * Output candidate collections are owned (as shallow clones) by this object.
 * 
 * author: Evan K. Friis, UC Davis (evan.klose.friis@cern.ch) 
 */

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTauTag/TauTagTools/interface/PFCandCommonVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

#include "CLHEP/Random/RandGauss.h"

#include <memory>
#include <algorithm>

using namespace reco;
using namespace edm;
using namespace std;
typedef reco::Particle::LorentzVector LorentzVector;

class PFRecoTauDecayModeDeterminator : public EDProducer {
 public:

  typedef list<CompositeCandidate>                    compCandList;
  typedef list<CompositeCandidate>::reverse_iterator  compCandRevIter;

  void mergePiZeroes(compCandList&, compCandRevIter);

  explicit PFRecoTauDecayModeDeterminator(const ParameterSet& iConfig);
  ~PFRecoTauDecayModeDeterminator();
  virtual void produce(Event&,const EventSetup&);

 protected:
  const double chargedPionMass;
  const double neutralPionMass;

 private:
  PFCandCommonVertexFitterBase* vertexFitter_;
  InputTag              PFTauProducer_;
  AddFourMomenta        addP4;
  uint32_t              maxPhotonsToMerge_;             //number of photons allowed in a merged pi0
  double                maxPiZeroMass_;             
  bool                  mergeLowPtPhotonsFirst_;
  bool                  refitTracks_;
  bool                  filterTwoProngs_;
  bool                  filterPhotons_;  
  double                minPtFractionForSecondProng_;   //2 prongs whose second prong falls under 
  double                minPtFractionForGammas_;        //outlier unmerged gammas
  TauTagTools::sortByDescendingPt<CompositeCandidate>   candDescendingSorter;
  TauTagTools::sortByAscendingPt<CompositeCandidate>    candAscendingSorter;
};


#endif

