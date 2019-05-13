/*
 * BayesMuCorrelatorTrack.h
 * The objects of this class are produced by the L1Trigger/L1TMuonBayes/plugins/L1TMuonBayesMuCorrelatorTrackProducer.h
 * It is tracked formed by matching the trackcing trigger tracks to the muon (DT, CSC, RPC) stubs.
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

class BayesMuCorrelatorTrack {
public:
  BayesMuCorrelatorTrack();
  virtual ~BayesMuCorrelatorTrack();

  enum CandidateType {
    fastTrack, //at least 2 stubs in BX = 0, most probably muon
    slowTrack  //less then 2 stubs in BX = 0, looks like HSCP
  };

  inline void setHwPt(int hwPt) { this->hwPt_ = hwPt; };
  //inline void setHwCharge(int charge) { hwCharge_ = charge; };
  //inline void setHwChargeValid(int valid) { hwChargeValid_ = valid; };
  //inline void setTfMuonIndex(int index) { tfMuonIndex_ = index; };
//  inline void setHwTag(int tag) { hwTag_ = tag; };
  inline void setHwSign(int sign) { this->hwSign_ = sign; };
  inline void setHwQual(int quality) { this->quality = quality; };


  inline void setHwEtaAtVtx(int hwEtaAtVtx) { hwEtaAtVtx_ = hwEtaAtVtx; };
  inline void setHwPhiAtVtx(int hwPhiAtVtx) { hwPhiAtVtx_ = hwPhiAtVtx; };
  //inline void setEtaAtVtx(double etaAtVtx) { etaAtVtx_ = etaAtVtx; };
  //inline void setPhiAtVtx(double phiAtVtx) { phiAtVtx_ = phiAtVtx; };

// inline void setHwIsoSum(int isoSum) { hwIsoSum_ = isoSum; };
//  inline void setHwDPhiExtra(int dPhi) { hwDPhiExtra_ = dPhi; };
//  inline void setHwDEtaExtra(int dEta) { hwDEtaExtra_ = dEta; };
//  inline void setHwRank(int rank) { hwRank_ = rank; };

//  inline void setDebug(bool debug) { debug_ = debug; };

  int hwPt() const { return hwPt_; }
  inline int hwCharge() const { return hwCharge_; };
  inline int hwChargeValid() const { return hwChargeValid_; };
  int hwSign() const { return hwSign_; }
  //inline int tfMuonIndex() const { return tfMuonIndex_; };
  //inline int hwTag() const { return hwTag_; };

  inline int hwEtaAtVtx() const { return hwEtaAtVtx_; };
  inline int hwPhiAtVtx() const { return hwPhiAtVtx_; };
  //inline double etaAtVtx() const { return etaAtVtx_; };
  //inline double phiAtVtx() const { return phiAtVtx_; }

  int hwQual() const { return quality; }

  virtual void setBeta(float beta) { this->beta = beta; }

  virtual float getBeta() const { return beta; }

  int hwBeta() const { return hwBeta_; }

  void setHwBeta(int hwBeta = 0) { this->hwBeta_ = hwBeta; }

  const boost::dynamic_bitset<>& getFiredLayerBits() const { return firedLayerBits; }

  void setFiredLayerBits(const boost::dynamic_bitset<>& firedLayerBits) { this->firedLayerBits = firedLayerBits; }

  int pdfSum() const { return pdfSum_; }

  void setPdfSum(int pdfSum = 0) { this->pdfSum_ = pdfSum; }


  const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& getTtTrackPtr() const {
    return ttTrackPtr;
  }

  void setTtTrackPtr(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& ttTrackPtr) {
    this->ttTrackPtr = ttTrackPtr;
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
    return eta;
  }

  void setEta(double eta = 0) {
    this->eta = eta;
  }

  //in radians
  double getPhi() const {
    return phi;
  }

  //in radians
  void setPhi(double phi = 0) {
    this->phi = phi;
  }

  //in GeV
  double getPt() const {
    return pt;
  }

  //in GeV
  void setPt(double pt = 0) {
    this->pt = pt;
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

//  inline int hwIsoSum() const { return hwIsoSum_; };
//  inline int hwDPhiExtra() const { return hwDPhiExtra_; };
//  inline int hwDEtaExtra() const { return hwDEtaExtra_; };
//  inline int hwRank() const { return hwRank_; };

//  inline bool debug() const { return debug_; };

//  virtual bool operator==(const l1t::BayesMuCorrelatorTrack& rhs) const;
//  virtual inline bool operator!=(const l1t::BayesMuCorrelatorTrack& rhs) const { return !(operator==(rhs)); };

private:
  int hwPt_ = 0;

  // additional hardware quantities common to L1 global jet
  int hwCharge_ = 0;
  int hwChargeValid_ = 0;
  int tfMuonIndex_ = 0;
//  int hwTag_;
  int hwSign_ = 0;
  int quality = 0;

  // additional hardware quantities only available if debug flag is set
//  bool debug_;
//  int hwIsoSum_;
//  int hwDPhiExtra_;
//  int hwDEtaExtra_;
//  int hwRank_;

  // muon coordinates at the vertex
  int hwEtaAtVtx_ = 0;
  int hwPhiAtVtx_ = 0;
  //double etaAtVtx_ = 0;
  //double phiAtVtx_ = 0;

  //original floating point coordinates and pt,
  double phi = 0;
  double eta = 0;
  double pt = 0;
  //int charge = 0;

  int hwBeta_ = 0;
  float beta = 0;
  float betaLikelihood = 0;

  boost::dynamic_bitset<> firedLayerBits;
  int pdfSum_ = 0;

  CandidateType candidateType = fastTrack;

  //the "pointers" the either sim Track ot ttTrack that was use to create this TrackingTriggerTrack, needed only for analysis
  edm::Ptr< SimTrack > simTrackPtr;
  edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > > ttTrackPtr;
  edm::Ptr< TrackingParticle > trackPartPtr;

};

} //end of namespace l1t

#endif /* DataFormats__BAYESMUCORRELATORTRACK_H_ */
