#ifndef DataFormatsHcalCalibObjectsHcalHBHEMuonVariables_h
#define DataFormatsHcalCalibObjectsHcalHBHEMuonVariables_h
#include <string>
#include <vector>

class HcalHBHEMuonVariables {
public:
  HcalHBHEMuonVariables() { clear(); }

  void clear() {
    muonGood_ = muonGlobal_ = muonTracker_ = muonTight_ = muonMedium_ = false;
    ptGlob_ = etaGlob_ = phiGlob_ = energyMuon_ = pMuon_ = 0;
    muonTrkKink_ = muonChi2LocalPosition_ = muonSegComp_ = 0;
    trackerLayer_ = numPixelLayers_ = tightPixelHits_;
    innerTrack_ = outerTrack_ = globalTrack_ = false;
    chiTracker_ = dxyTracker_ = dzTracker_ = 0;
    innerTrackPt_ = innerTrackEta_ = innerTrackEhi_ = 0;
    outerTrackHits_ = outerTrackRHits_ = 0;
    outerTrackPt_ = outerTrackEta_ = outerTrackPhi_ = outerTrackChi_ = 0;
    globalMuonHits_ = matchedStat_ = 0;
    globalTrackPt_ = globalTrackEta_ = globalTrackPhi_ = chiGlobal_ = 0;
    tightValidFraction_ = tightLongPara_ = tightTransImpara_ = 0;
    isolationR04_ = isolationR03_ = 0;
    ecalDetId_ = 0;
    ecalEnergy_ = ecal3x3Energy_ = 0;
    hcalDetId_ = ehcalDetId_ = 0;
    matchedId_ = hcalHot_ = false;
    hcalIeta_ = hcalIphi_ = 0;
    hcalEnergy_ = hoEnergy_ = hcal1x1Energy_ = 0;
    hcalDepthEnergy_.clear();
    hcalDepthActiveLength_.clear();
    hcalDepthEnergyHot_.clear();
    hcalDepthActiveLengthHot_.clear();
    hcalDepthChargeHot_.clear();
    hcalDepthChargeHotBG_.clear();
    hcalDepthEnergyCorr_.clear();
    hcalDepthEnergyHotCorr_.clear();
    hcalDepthMatch_.clear();
    hcalDepthMatchHot_.clear();
    hcalActiveLength_ = hcalActiveLengthHot_ = 0;
    allTriggers_.clear();
    hltResults_.clear();
  }

  bool muonGood_, muonGlobal_, muonTracker_, muonTight_, muonMedium_;
  float ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  float muonTrkKink_, muonChi2LocalPosition_, muonSegComp_;
  int trackerLayer_, numPixelLayers_, tightPixelHits_;
  bool innerTrack_, outerTrack_, globalTrack_;
  float chiTracker_, dxyTracker_, dzTracker_;
  float innerTrackPt_, innerTrackEta_, innerTrackEhi_;
  int outerTrackHits_, outerTrackRHits_;
  float outerTrackPt_, outerTrackEta_, outerTrackPhi_, outerTrackChi_;
  int globalMuonHits_, matchedStat_;
  float globalTrackPt_, globalTrackEta_, globalTrackPhi_, chiGlobal_;
  float tightValidFraction_, tightLongPara_, tightTransImpara_;
  float isolationR04_, isolationR03_;
  unsigned int ecalDetId_;
  float ecalEnergy_, ecal3x3Energy_;
  unsigned int hcalDetId_, ehcalDetId_;
  bool matchedId_, hcalHot_;
  int hcalIeta_, hcalIphi_;
  float hcalEnergy_, hoEnergy_, hcal1x1Energy_;
  std::vector<float> hcalDepthEnergy_, hcalDepthActiveLength_;
  std::vector<float> hcalDepthEnergyHot_, hcalDepthActiveLengthHot_;
  std::vector<float> hcalDepthChargeHot_, hcalDepthChargeHotBG_;
  std::vector<float> hcalDepthEnergyCorr_, hcalDepthEnergyHotCorr_;
  std::vector<bool> hcalDepthMatch_, hcalDepthMatchHot_;
  float hcalActiveLength_, hcalActiveLengthHot_;
  std::vector<std::string> allTriggers_;
  std::vector<int> hltResults_;
};

typedef std::vector<HcalHBHEMuonVariables> HcalHBHEMuonVariablesCollection;
#endif
