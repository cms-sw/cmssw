#ifndef JetProducers_CaloJetTrackTowerAssociator_h
#define JetProducers_CaloJetTrackTowerAssociator_h

/// Associates tracks in eta-phi cone around jet direction
/// \author: F.Ratnikov, UMd
/// Apr. 20, 2007
/// $Id: JetTrackAssociator.h,v 1.7 2007/04/18 22:04:31 fedor Exp $

#include "RecoJets/JetAlgorithms/interface/JetTrackAssociator.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"


class CaloJetTrackTowerAssociator : public JetTrackAssociator<CaloJetCollection> {
 public:
  CaloJetTrackTowerAssociator () {
    std::cerr << "!!!\n CaloJetTrackTowerAssociator class is not implemented \n!!!" << std::endl;
}
  virtual ~CaloJetTrackTowerAssociator () {}

  /// virtual method to make association
  virtual bool associate (const JetRef& fJet, const TrackRef fTrack&) const {
    return false;
  }
};

#endif
