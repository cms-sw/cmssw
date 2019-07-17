/*
 * BayesMuCorrelatorTrack.h
 * The objects of this class are produced by the L1Trigger/L1TMuonBayes/plugins/L1TMuonBayesMuCorrelatorTrackProducer.h
 * It is a muon track formed by matching the tracking trigger tracks to the muon stubs (DT, CSC, RPC) .
 *  Created on: Mar 15, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef DataFormats__BAYESMUCORRELATORTRACK_H_
#define DataFormats__BAYESMUCORRELATORTRACK_H_

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "boost/dynamic_bitset.hpp"

namespace l1t {
class BayesMuCorrelatorTrack;
typedef BXVector<BayesMuCorrelatorTrack> BayesMuCorrTrackBxCollection;
//typedef edm::Ref< BayesMuCorrBxCollection > BayesMuCorrRef ;
//typedef edm::RefVector< BayesMuCorrBxCollection > BayesMuCorrRefVector ;
//typedef std::vector< BayesMuCorrRef > BayesMuCorrVectorRef ;
//
//typedef ObjectRefBxCollection<BayesMuCorrelatorTrack> BayesMuCorrRefBxCollection;
//typedef ObjectRefPair<BayesMuCorrelatorTrack> BayesMuCorrRefPair;
//typedef ObjectRefPairBxCollection<BayesMuCorrelatorTrack> BayesMuCorrRefPairBxCollection;

typedef std::vector<BayesMuCorrelatorTrack> BayesMuCorrTrackCollection;

class BayesMuCorrelatorTrack: public L1Candidate {
public:
  //N.B. The same typedef is in the DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h, remove if base class is changed to L1TkMuonParticle
  typedef TTTrack< Ref_Phase2TrackerDigi_ > L1TTTrackType;

  BayesMuCorrelatorTrack();

  BayesMuCorrelatorTrack(const LorentzVector& p4);

  BayesMuCorrelatorTrack(const edm::Ptr< L1TTTrackType > ttTrackPtr);

  virtual ~BayesMuCorrelatorTrack();

  enum CandidateType {
    fastTrack, //at least 2 stubs in BX = 0, most probably muon
    slowTrack  //less then 2 stubs in BX = 0, looks like HSCP
  };

  inline void setHwSign(int sign) { this->hwSign_ = sign; };

// inline void setHwIsoSum(int isoSum) { hwIsoSum_ = isoSum; };

  int hwSign() const { return hwSign_; }

  virtual void setBeta(float beta) { this->beta = beta; }

  virtual float getBeta() const { return beta; }

  int hwBeta() const { return hwBeta_; }

  void setHwBeta(int hwBeta = 0) { this->hwBeta_ = hwBeta; }

  const boost::dynamic_bitset<>& getFiredLayerBits() const { return firedLayerBits; }

  void setFiredLayerBits(const boost::dynamic_bitset<>& firedLayerBits) { this->firedLayerBits = firedLayerBits; }

  int pdfSum() const { return pdfSum_; }

  void setPdfSum(int pdfSum = 0) { this->pdfSum_ = pdfSum; }


  const edm::Ptr<L1TTTrackType>& getTtTrackPtr() const {
    return ttTrackPtr;
  }

  const edm::Ptr<SimTrack>& getSimTrackPtr() const {
    return simTrackPtr;
  }

  void setSimTrackPtr(const edm::Ptr<SimTrack>& simTrackPtr) {
    this->simTrackPtr = simTrackPtr;
  }

  const edm::Ptr<TrackingParticle>& getTrackPartPtr() const {
    return trackPartPtr;
  }

  void setTrackPartPtr(const edm::Ptr<TrackingParticle>& trackPartPtr) {
    this->trackPartPtr = trackPartPtr;
  }

  double getEta() const {
    return eta();
  }

  //in radians
  double getPhi() const {
    return phi();
  }


  //in GeV
  double getPt() const {
    return pt();
  }

  CandidateType getCandidateType() const {
    return candidateType;
  }

  void setCandidateType(CandidateType candidateType) {
    this->candidateType = candidateType;
  }

  float getBetaLikelihood() const {
    return betaLikelihood;
  }

  void setBetaLikelihood(float betaLikelihood = 0) {
    this->betaLikelihood = betaLikelihood;
  }

//  virtual bool operator==(const l1t::BayesMuCorrelatorTrack& rhs) const;
//  virtual inline bool operator!=(const l1t::BayesMuCorrelatorTrack& rhs) const { return !(operator==(rhs)); };

private:
  int hwSign_ = 0;

  int hwBeta_ = 0;
  float beta = 0;
  float betaLikelihood = 0;

  boost::dynamic_bitset<> firedLayerBits;
  int pdfSum_ = 0;

  CandidateType candidateType = fastTrack;

  //the "pointers" to either simTrack or ttTrack or trackingParticl that was use to create this TrackingTriggerTrack, needed only for analysis
  edm::Ptr< SimTrack > simTrackPtr;
  edm::Ptr< L1TTTrackType > ttTrackPtr;
  edm::Ptr< TrackingParticle > trackPartPtr;

};

} //end of namespace l1t

#endif /* DataFormats__BAYESMUCORRELATORTRACK_H_ */
