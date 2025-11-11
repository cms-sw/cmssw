/*
 * FinalMuon.h
 *
 *  Created on: Dec 17, 2024
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_FinalMuon_H
#define L1T_OmtfP1_FinalMuon_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"

class FinalMuon {
public:
  FinalMuon() {};
  FinalMuon(const AlgoMuonPtr& algoMuon) :
    algoMuon(algoMuon),
    quality(algoMuon->getQuality()),
    firedLayerCnt(algoMuon->getFiredLayerCnt()) {};

  virtual ~FinalMuon() {};

  const AlgoMuonPtr& getAlgoMuon() const { return algoMuon; }

  int getSign() const { return sign; }

  void setSign(int sign = 0) { this->sign = sign; }

  void setBx(int bx = 0) {
    this->bx = bx;
  }

  int getBx() const {
    return bx;
  }

  int getProcessor() const {
    return processor;
  }

  void setProcessor(int processor = -1) {
    this->processor = processor;
  }

  void setTrackFinderType(l1t::tftype mtfType) {
    this->mtfType = mtfType;
  }
  l1t::tftype trackFinderType() const {
    return mtfType;
  }

  int getQuality() const { return quality; }

  void setQuality(int quality = 0) { this->quality = quality; }

  float getPtGev() const {
    return ptGev;
  }

  void setPtGev(float ptGev = -1) {
    this->ptGev = ptGev;
  }

  float getPtUnconstrGev() const {
    return ptUnconstrGev;
  }
  
  void setPtUnconstrGev(float ptUnconstrGev = -1) {
    this->ptUnconstrGev = ptUnconstrGev;
  }

  float getEtaRad() const {
    return etaRad;
  }

  void setEtaRad(float etaRad = -10) {
    this->etaRad = etaRad;
  }

  float getPhiRad() const {
    return phiRad;
  }

  void setPhiRad(float phiRad = -10) {
    this->phiRad = phiRad;
  }

  int getPtGmt() const {
    return ptGmt;
  }
  void setPtGmt(int ptGmt = 0) {
    this->ptGmt = ptGmt;
  }
  int getPtUnconstrGmt() const {
    return ptUnconstrGmt;
  }
  void setPtUnconstrGmt(int ptUnconstrGmt = 0) {
    this->ptUnconstrGmt = ptUnconstrGmt;
  }
  int getPhiGmt() const {
    return phiGmt;
  }
  void setPhiGmt(int phiGmt = 0) {
    this->phiGmt = phiGmt;
  }
  int getEtaGmt() const {
    return etaGmt;
  }
  void setEtaGmt(int etaGmt = 0) {
    this->etaGmt = etaGmt;
  }

  int getFiredLayerCnt() const {
    return firedLayerCnt;
  }

  void setFiredLayerCnt(int firedLayerCnt = 0) {
    this->firedLayerCnt = firedLayerCnt;
  }

  int getFiredLayerBits() const {
    return firedLayerBits;
  }

  void setFiredLayerBits(int firedLayerBits = 0) {
    this->firedLayerBits = firedLayerBits;
  }

  friend std::ostream& operator<<(std::ostream& out, const FinalMuon& finalMuon);
private:
  AlgoMuonPtr algoMuon;

  int bx = 0;
  int processor = -1;
  l1t::tftype mtfType = l1t::omtf_pos;

  int quality = 0;

  int sign = 0;
  
  float ptGev = -1;
  float ptUnconstrGev = -1;
  float phiRad = -10;
  float etaRad = -10;

  int ptGmt = 0;
  int ptUnconstrGmt = 0;
  int phiGmt = 0;
  int etaGmt = 0;

  int firedLayerCnt = 0;

  int firedLayerBits = 0;
};

typedef std::shared_ptr<FinalMuon> FinalMuonPtr;
typedef std::vector<FinalMuonPtr> FinalMuons;

#endif /* FinalMuon */
