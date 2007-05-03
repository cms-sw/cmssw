#ifndef JetProducers_JetTrackConeAssociator_h
#define JetProducers_JetTrackConeAssociator_h

/// Associates tracks in eta-phi cone around jet direction
/// \author: F.Ratnikov, UMd
/// Apr. 20, 2007
/// $Id: JetTrackAssociator.h,v 1.7 2007/04/18 22:04:31 fedor Exp $

#include "RecoJets/JetAlgorithms/interface/JetTrackAssociator.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"


template <typename JetC>
class JetTrackConeAssociator : public JetTrackAssociator<JetC> {
 public:
  typedef typename JetTrackAssociator<JetC>::JetRef JetRef;
  typedef typename JetTrackAssociator<JetC>::TrackRef TrackRef;
  JetTrackConeAssociator (double fConeSize) : mConeSize (fConeSize) {}
  virtual ~JetTrackConeAssociator () {}

  /// virtual method to make association
  virtual bool associate (const JetRef& fJet, const TrackRef& fTrack) const {
    return deltaR (fJet->eta(), fJet->phi(), fTrack->eta(), fTrack->phi()) <= mConeSize;
  }
 private:
  double mConeSize;
};

#endif
